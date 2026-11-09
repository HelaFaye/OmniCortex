#!/usr/bin/env python3
"""
OmniCortex Remote: the agent for a computer that runs LLMs but not OmniCortex.

There are always two roles:

  HOST        this computer. It runs the models (LM Studio, Ollama, llama.cpp's
              llama-server, or any OpenAI-compatible server) and this agent.
  CONTROLLER  the computer that tells it what to do: OmniCortex Studio, or any
              MCP client (Claude Desktop, Cline, Cursor, ...).

The agent is the host's one control surface, an MCP server: list, load and
unload models, chat, receive model files and install them into LM Studio or
Ollama. Nothing else on the host is reachable through it. Controllers pair once
with a short code shown on the host; the host owner can see every call they
make and revoke them at any time.

How a controller reaches the host (the "link"):
  lan       same network. TLS with a certificate pinned at pairing; the
            controller finds the host by itself while pairing is open.
  vpn       Tailscale, Headscale, ZeroTier, WireGuard, NetBird, Nebula...: listen
            on the VPN address only.
  ssh       listen on this computer only; the controller opens an SSH tunnel.
  tunnel    listen on this computer only; a tunnel service (Cloudflare Tunnel,
            ngrok, ...) gives it a public HTTPS address.
  callhome  no listening at all: the host connects out to the controller and
            waits for instructions. For hosts behind NAT or a firewall you do not
            control. The controller must be reachable from the host.
  stdio     an MCP client on this same computer starts the agent itself.

    omnicortex-remote                       # start with the saved link; opens the console
    omnicortex-remote serve --mode lan      # choose the link
    omnicortex-remote serve --mode vpn --bind 100.101.102.103
    omnicortex-remote connect nsr1:...      # call home to a Studio (the invite from its Remote page)
    omnicortex-remote pair                  # a new pairing code
    omnicortex-remote controllers           # who may control this host; --revoke ID
    omnicortex-remote stdio                 # MCP over stdin/stdout, for a local MCP client

On a CONTROLLER that is not OmniCortex Studio (Claude Desktop, Cline...),
the same package pairs with hosts and bridges the MCP client to them:

    omnicortex-remote add-host 'nsr://192.168.1.20:47871/#c=...'   # prints the MCP client entry
    omnicortex-remote bridge gpu-box        # what that entry runs: MCP over stdio to the host

Only the Python standard library is needed. TLS certificates are made with the
`cryptography` package when present (pip install "omnicortex-remote[tls]") or
the openssl command.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import platform
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from . import __version__ as VERSION
except ImportError:                     # run as a plain script: python agent.py
    VERSION = "0.1.0"

DEFAULT_PORT = 47871          # the host's API (MCP, uploads, /v1); the console is the next port, loopback only
BEACON_PORT = 47870           # UDP: "a host is waiting to be paired" on the LAN
PROTOCOL_VERSION = "2025-06-18"
MODES = ("lan", "vpn", "ssh", "tunnel", "callhome", "loopback")
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"   # no 0/O, 1/I/L, U
CODE_LEN = 10                                      # ~49 bits
PAIR_TTL = 600
MAX_ATTEMPTS = 5
MAX_JSON = 32 * 1024 * 1024    # chat requests can carry images
MAX_CHUNK = 64 * 1024 * 1024
ALLOW_DEFAULT = {"chat": True, "load": True, "receive": True, "install": True, "delete": False, "trace": True,
                 "jobs": False, "restart": True, "busy": True}
ALLOW_TEXT = {"chat": "chat with the models",
              "trace": "check replies with OmniCortex's tracer (runs a model file once more, for a 3D view)",
              "load": "load and unload models",
              "receive": "send files to this computer (into the incoming folder)",
              "install": "install received models into LM Studio or Ollama",
              "delete": "delete files in the incoming and models folders",
              "jobs": "run OmniCortex jobs (evaluations, benchmarks, training) with the OmniCortex folder on this "
                      "computer; they run Python code, so only for controllers you trust",
              "restart": "restart this agent (it starts again by itself)",
              "busy": "mark this computer busy or free and unload its models (Studio's automation rules)"}
KNOWN_BACKENDS = (("lmstudio", "http://127.0.0.1:1234"), ("ollama", "http://127.0.0.1:11434"),
                  ("llama-server", "http://127.0.0.1:8080"), ("vllm", "http://127.0.0.1:8000"))
ROLE_LINE = "This computer is the HOST: it runs the models. Controllers send it instructions over MCP."


class ToolError(Exception):
    pass


class LinkError(Exception):
    """Pairing with or reaching another computer failed; the message says what to do."""


# ------------------------------------------------------------------ files

def _migrate_old_home(new: Path) -> None:
    """One-time move from the folder this agent used under its former name (NeuronScope Remote)."""
    old_name = "Neuron" + "ScopeRemote" if sys.platform in ("win32", "darwin") else "neuron" + "scope-remote"
    old = new.with_name(old_name)
    if old.exists() and not new.exists():
        try:
            old.rename(new)
        except OSError:
            pass


def default_home() -> Path:
    p = _default_home()
    _migrate_old_home(p)
    return p


def _default_home() -> Path:
    if os.environ.get("OCR_HOME"):
        return Path(os.environ["OCR_HOME"]).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "OmniCortexRemote"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OmniCortexRemote"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "omnicortex-remote"


class Store:
    """JSON files in the agent's folder, written atomically and readable by the owner only."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def read(self, name: str, default):
        try:
            return json.loads((self.home / name).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return default

    def write(self, name: str, obj) -> None:
        path = self.home / name
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)

    def remove(self, name: str) -> None:
        try:
            (self.home / name).unlink()
        except FileNotFoundError:
            pass


def safe_name(name: str) -> str:
    base = re.split(r"[\\/]", str(name or ""))[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip(".")[:200]
    if not base:
        raise ToolError("a file name is needed")
    return base


def _inside(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if p != root.resolve() and root.resolve() not in p.parents:
        raise ToolError(f"{rel} is outside {root}")
    return p


# ------------------------------------------------------------------ pairing

def norm_code(code: str) -> str:
    return "".join(c for c in str(code).upper() if c in CODE_ALPHABET)


def fmt_code(code: str) -> str:
    c = norm_code(code)
    return f"{c[:5]}-{c[5:]}"


def new_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def pair_proof(code: str, fingerprint: str, side: str) -> str:
    """Proof of knowing the pairing code, bound to the TLS certificate the claiming
    side sees ("" when there is no pinned certificate). The code never crosses the
    network, and a man in the middle with another certificate gets a proof that the
    real host rejects. side: "claim" from the side that types the code, "accept"
    back from the side that showed it."""
    msg = f"nsr-pair-v1|{side}|{(fingerprint or '').replace(':', '').lower()}".encode()
    return hmac.new(norm_code(code).encode(), msg, hashlib.sha256).hexdigest()


def make_invite(url: str, code: str, fingerprint: str = "", name: str = "") -> str:
    raw = json.dumps({"u": url, "c": norm_code(code), "f": fingerprint, "n": name}, separators=(",", ":"))
    return "nsr1:" + base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def parse_invite(s: str) -> dict:
    s = s.strip()
    if s.startswith("nsr1:"):
        b = s[5:] + "=" * (-len(s[5:]) % 4)
        try:
            d = json.loads(base64.urlsafe_b64decode(b))
        except ValueError:
            raise ValueError("this invite is damaged; copy it again") from None
        return {"url": d["u"], "code": d["c"], "fingerprint": d.get("f") or "", "name": d.get("n") or ""}
    if s.startswith("nsr://") or "#c=" in s:
        # a host's pairing link: nsr://address:port/#c=CODE&fp=FP&s=https&n=NAME
        u = urllib.parse.urlsplit(s)
        q = dict(urllib.parse.parse_qsl(u.fragment))
        scheme = q.get("s") or ("https" if q.get("fp") else "http")
        return {"url": f"{scheme}://{u.netloc}", "code": q.get("c", ""), "fingerprint": q.get("fp", ""),
                "name": q.get("n", "")}
    raise ValueError("not an invite or pairing link (they start with nsr1: or nsr://)")


class Pairing:
    def __init__(self, store: Store):
        self.store = store

    def open(self, ttl: float = PAIR_TTL) -> dict:
        code = new_code()
        with self.store.lock:
            self.store.write("pairing.json", {"code": code, "expires": time.time() + ttl, "attempts": 0})
        return {"code": fmt_code(code), "expires": time.time() + ttl}

    def current(self) -> dict | None:
        p = self.store.read("pairing.json", None)
        if not p or p.get("expires", 0) < time.time() or p.get("attempts", 0) >= MAX_ATTEMPTS:
            return None
        return p

    def close(self) -> None:
        self.store.remove("pairing.json")

    def consume(self, proof: str, fingerprint: str) -> str | None:
        """-> the code when `proof` matches (the code is then used up), else None.
        Five wrong proofs end the code."""
        with self.store.lock:
            p = self.current()
            if not p:
                return None
            if hmac.compare_digest(str(proof), pair_proof(p["code"], fingerprint, "claim")):
                self.close()
                return p["code"]
            p["attempts"] = p.get("attempts", 0) + 1
            self.store.write("pairing.json", p)
            return None


class Controllers:
    """Who may control this host. Only token hashes are kept."""

    def __init__(self, store: Store):
        self.store = store

    def _all(self) -> list:
        return self.store.read("controllers.json", [])

    def list(self) -> list:
        return [{k: v for k, v in c.items() if k != "token_sha256"} for c in self._all()]

    def add(self, name: str, via: str) -> tuple[dict, str]:
        token = secrets.token_urlsafe(32)
        rec = {"id": secrets.token_hex(4), "name": str(name or "controller")[:64], "via": via,
               "created": time.time(), "last_seen": None,
               "token_sha256": hashlib.sha256(token.encode()).hexdigest()}
        with self.store.lock:
            self.store.write("controllers.json", self._all() + [rec])
        return {k: v for k, v in rec.items() if k != "token_sha256"}, token

    def check(self, token: str) -> dict | None:
        if not token:
            return None
        h = hashlib.sha256(token.encode()).hexdigest()
        with self.store.lock:
            items = self._all()
            for c in items:
                if hmac.compare_digest(c["token_sha256"], h):
                    if not c.get("last_seen") or time.time() - c["last_seen"] > 60:
                        c["last_seen"] = time.time()
                        self.store.write("controllers.json", items)
                    return {k: v for k, v in c.items() if k != "token_sha256"}
        return None

    def revoke(self, cid: str) -> bool:
        with self.store.lock:
            items = self._all()
            keep = [c for c in items if c["id"] != cid]
            self.store.write("controllers.json", keep)
            return len(keep) != len(items)


# ------------------------------------------------------------------ TLS and HTTP

def ensure_cert(home: Path, names: list[str]) -> tuple[str, str] | None:
    """A self-signed certificate for this host, made once (controllers pin it)."""
    d = Path(home) / "tls"
    cert, key = d / "cert.pem", d / "key.pem"
    if cert.exists() and key.exists():
        return str(cert), str(key)
    d.mkdir(parents=True, exist_ok=True)
    try:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        k = ec.generate_private_key(ec.SECP256R1())
        sans = []
        for h in names:
            try:
                sans.append(x509.IPAddress(ipaddress.ip_address(h)))
            except ValueError:
                sans.append(x509.DNSName(h))
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "omnicortex-remote")])
        now = datetime.datetime.now(datetime.timezone.utc)
        crt = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
               .serial_number(x509.random_serial_number())
               .not_valid_before(now - datetime.timedelta(minutes=5))
               .not_valid_after(now + datetime.timedelta(days=3650))
               .add_extension(x509.SubjectAlternativeName(sans), critical=False)
               .sign(k, hashes.SHA256()))
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
        cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
        return str(cert), str(key)
    except ImportError:
        pass
    exe = shutil.which("openssl")
    if not exe:
        return None
    r = subprocess.run([exe, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
                        "-keyout", str(key), "-out", str(cert), "-days", "3650", "-subj", "/CN=omnicortex-remote"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not cert.exists():
        return None
    try:
        os.chmod(key, 0o600)
    except OSError:
        pass
    return str(cert), str(key)


def cert_fingerprint(certfile: str) -> str:
    pem = Path(certfile).read_text()
    der = ssl.PEM_cert_to_DER_cert(pem[pem.index("-----BEGIN CERTIFICATE-----"):])
    return hashlib.sha256(der).hexdigest()


def short_fp(fp: str) -> str:
    """What people compare by eye: the first 8 hex digits, grouped."""
    f = (fp or "").replace(":", "").upper()
    return f"{f[:4]}-{f[4:8]}" if f else ""


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS that trusts exactly one certificate (by SHA-256), or, with an empty
    fingerprint, any certificate while recording which one it saw (pairing: the
    proof binds what was seen, so a different certificate fails the pairing)."""

    def __init__(self, netloc: str, fingerprint: str = "", timeout: float = 30):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        super().__init__(netloc, timeout=timeout, context=ctx)
        self.fingerprint = (fingerprint or "").replace(":", "").lower()
        self.seen = ""

    def connect(self):
        super().connect()
        der = self.sock.getpeercert(binary_form=True) or b""
        self.seen = hashlib.sha256(der).hexdigest()
        if self.fingerprint and not hmac.compare_digest(self.seen, self.fingerprint):
            self.sock.close()
            raise ssl.SSLError("the other computer's certificate does not match the one pinned at pairing")


def open_conn(base: str, fingerprint: str | None, timeout: float):
    """fingerprint None: normal certificate checks (public HTTPS); "": accept and
    record any certificate; hex: pin it."""
    u = urllib.parse.urlsplit(base)
    if u.scheme == "https":
        if fingerprint is None:
            return http.client.HTTPSConnection(u.netloc, timeout=timeout, context=ssl.create_default_context())
        return PinnedHTTPSConnection(u.netloc, fingerprint, timeout)
    if u.scheme != "http":
        raise ValueError(f"{base}: use http:// or https://")
    return http.client.HTTPConnection(u.netloc, timeout=timeout)


def http_json(base: str, method: str, path: str, body=None, token: str = "", fingerprint: str | None = None,
              timeout: float = 30) -> tuple[int, dict, str]:
    """-> (status, JSON body, certificate fingerprint seen or "")."""
    conn = open_conn(base, fingerprint, timeout)
    prefix = urllib.parse.urlsplit(base).path.rstrip("/")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        conn.request(method, prefix + path, body=None if body is None else json.dumps(body).encode(),
                     headers=headers)
        r = conn.getresponse()
        data = r.read()
        seen = getattr(conn, "seen", "")
    finally:
        conn.close()
    try:
        obj = json.loads(data) if data else {}
    except ValueError:
        obj = {"error": data[:300].decode(errors="replace")}
    return r.status, obj, seen


# ------------------------------------------------------------------ backends

def lms_cli() -> str | None:
    exe = shutil.which("lms")
    if exe:
        return exe
    for d in (Path.home() / ".lmstudio" / "bin", Path.home() / ".cache" / "lm-studio" / "bin"):
        for n in ("lms.exe", "lms"):
            if (d / n).exists():
                return str(d / n)
    return None


def lmstudio_models_dir() -> Path | None:
    for s in (Path.home() / ".lmstudio" / "settings.json", Path.home() / ".cache" / "lm-studio" / "settings.json"):
        try:
            d = json.loads(s.read_text()).get("downloadsFolder")
            if d and Path(d).is_dir():
                return Path(d)
        except (OSError, ValueError, AttributeError):
            pass
    for d in (Path.home() / ".lmstudio" / "models", Path.home() / ".cache" / "lm-studio" / "models"):
        if d.is_dir():
            return d
    return None


class Backend:
    def __init__(self, kind: str, url: str, api_key: str = "", source: str = "detected"):
        self.kind, self.url, self.api_key, self.source = kind, url.rstrip("/"), api_key, source

    def describe(self) -> dict:
        return {"kind": self.kind, "url": self.url, "source": self.source}

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def request(self, method: str, path: str, body=None, timeout: float = 10):
        conn = open_conn(self.url, None, timeout)
        conn.request(method, urllib.parse.urlsplit(self.url).path + path,
                     body=None if body is None else json.dumps(body).encode(), headers=self._headers())
        return conn, conn.getresponse()

    def get_json(self, path: str, timeout: float = 10, body=None, method: str = "GET"):
        conn, r = self.request(method, path, body, timeout)
        try:
            data = r.read()
        finally:
            conn.close()
        if r.status >= 400:
            raise ToolError(f"{self.kind} at {self.url}{path}: HTTP {r.status}: {data[:300].decode(errors='replace')}")
        return json.loads(data or b"{}")

    def models(self) -> list[dict]:
        if self.kind == "ollama":
            tags = self.get_json("/api/tags").get("models") or []
            try:
                loaded = {m.get("name") for m in self.get_json("/api/ps").get("models") or []}
            except (ToolError, OSError, ValueError):
                loaded = set()
            return [{"id": m["name"], "loaded": m["name"] in loaded, "size": m.get("size"),
                     "quant": (m.get("details") or {}).get("quantization_level"),
                     "params": (m.get("details") or {}).get("parameter_size")} for m in tags if m.get("name")]
        if self.kind == "lmstudio":
            try:
                data = self.get_json("/api/v0/models").get("data") or []
                return [{"id": m["id"], "loaded": (m.get("state") == "loaded") if m.get("state") else None,
                         "type": m.get("type"), "quant": m.get("quantization"), "arch": m.get("arch"),
                         "context": m.get("max_context_length")} for m in data if m.get("id")]
            except (ToolError, ValueError):
                pass
        data = self.get_json("/v1/models").get("data") or []
        return [{"id": m["id"], "loaded": True if self.kind in ("llama-server", "vllm") else None}
                for m in data if m.get("id")]

    def load(self, model: str, context_length: int | None = None, gpu: str | None = None,
             keep_alive: str | None = None) -> dict:
        if self.kind == "ollama":
            self.get_json("/api/generate", timeout=600, method="POST",
                          body={"model": model, "keep_alive": keep_alive or "30m"})
            return {"loaded": model, "how": "ollama keep_alive " + (keep_alive or "30m")}
        if self.kind == "lmstudio":
            exe = lms_cli()
            if exe:
                cmd = [exe, "load", model, "-y"]
                if context_length:
                    cmd += ["--context-length", str(int(context_length))]
                if gpu:
                    cmd += ["--gpu", str(gpu)]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                if r.returncode != 0:
                    raise ToolError(f"lms load failed: {(r.stderr or r.stdout).strip()[-500:]}")
                return {"loaded": model, "how": "lms load"}
        if self.kind in ("llama-server", "vllm"):
            raise ToolError(f"{self.kind} serves the model it was started with; restart it to change models")
        # LM Studio without its CLI, or another OpenAI-compatible server: a one-token request
        # loads the model where just-in-time loading is on.
        self.chat({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, timeout=900)
        return {"loaded": model, "how": "first request (just-in-time loading)"}

    def unload(self, model: str) -> dict:
        if self.kind == "ollama":
            self.get_json("/api/generate", timeout=120, method="POST", body={"model": model, "keep_alive": 0})
            return {"unloaded": model}
        if self.kind == "lmstudio":
            exe = lms_cli()
            if not exe:
                raise ToolError("unloading in LM Studio needs its lms command (LM Studio → Developer → install lms)")
            r = subprocess.run([exe, "unload", model], capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise ToolError(f"lms unload failed: {(r.stderr or r.stdout).strip()[-500:]}")
            return {"unloaded": model}
        raise ToolError(f"{self.kind} cannot unload models on request")

    def chat(self, body: dict, timeout: float = 900) -> dict:
        return self.get_json("/v1/chat/completions", timeout=timeout, method="POST", body={**body, "stream": False})

    def chat_stream(self, body: dict, on_text, cancelled=lambda: False, timeout: float = 900) -> dict:
        """Chat with the reply read as it is written; on_text gets each new piece.
        Stops reading (which stops generation on llama.cpp, LM Studio and Ollama)
        when cancelled() turns true. -> a response shaped like a non-streamed one."""
        conn, r = self.request("POST", "/v1/chat/completions", {**body, "stream": True}, timeout)
        text, finish, usage, model = [], None, None, body.get("model", "")
        try:
            if r.status >= 400:
                raise ToolError(f"{self.kind} at {self.url}: HTTP {r.status}: {r.read()[:300].decode(errors='replace')}")
            for raw in r:
                if cancelled():
                    finish = "cancelled"
                    break
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                model = ev.get("model") or model
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        text.append(piece)
                        on_text(piece)
                    finish = ch.get("finish_reason") or finish
        finally:
            conn.close()
        return {"object": "chat.completion", "model": model, "usage": usage,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                             "finish_reason": finish or "stop"}]}


# ------------------------------------------------------------------ the agent

def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = MS()
            m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 2**30, 1)
        if sys.platform == "darwin":
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30, 1)
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except Exception:
        pass
    return None


def _gpus() -> list:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "vram_mb": int(float(parts[1])), "used_mb": int(float(parts[2]))})
    return rows


def _num(v):
    try:
        return float(str(v).strip().split()[0])
    except (ValueError, IndexError):
        return None


def _nvidia_telemetry() -> list:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    q = "index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw"
    try:
        out = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7:
            rows.append({"id": f"nvidia:{p[0]}", "name": p[1], "vendor": "nvidia", "temp_c": _num(p[2]),
                         "junction_c": None, "mem_c": None, "vram_used_mb": _num(p[3]), "vram_total_mb": _num(p[4]),
                         "util": _num(p[5]), "power_w": _num(p[6]), "source": "nvidia-smi"})
    return rows


def _amdgpu_sysfs() -> list:
    """amdgpu's own sensors under /sys (Linux): edge, junction and memory temperatures and VRAM use.
    Needs no ROCm tools."""
    rows = []
    for dev in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        try:
            if (dev / "vendor").read_text().strip() != "0x1002":
                continue
        except OSError:
            continue
        r = {"id": f"amd:{dev.parent.name}", "name": "AMD GPU", "vendor": "amd", "temp_c": None, "junction_c": None,
             "mem_c": None, "vram_used_mb": None, "vram_total_mb": None, "util": None, "power_w": None,
             "source": "sysfs"}
        for hw in dev.glob("hwmon/hwmon*"):
            for lab in hw.glob("temp*_label"):
                try:
                    name = lab.read_text().strip().lower()
                    val = int((hw / lab.name.replace("_label", "_input")).read_text()) / 1000
                except (OSError, ValueError):
                    continue
                key = {"edge": "temp_c", "junction": "junction_c", "mem": "mem_c"}.get(name)
                if key:
                    r[key] = val
            for pw in ("power1_average", "power1_input"):
                try:
                    r["power_w"] = int((hw / pw).read_text()) / 1e6
                    break
                except (OSError, ValueError):
                    pass
        for key, f in (("vram_used_mb", "mem_info_vram_used"), ("vram_total_mb", "mem_info_vram_total"),
                       ("util", "gpu_busy_percent")):
            try:
                v = int((dev / f).read_text())
                r[key] = v / 2**20 if key.startswith("vram") else v
            except (OSError, ValueError):
                pass
        try:
            r["name"] = (dev / "product_name").read_text().strip() or r["name"]
        except OSError:
            pass
        rows.append(r)
    return rows


def _rocm_smi() -> list:
    exe = shutil.which("rocm-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--showtemp", "--showmeminfo", "vram", "--showuse", "--showproductname", "--json"],
                             capture_output=True, text=True, timeout=15).stdout
        data = json.loads(out or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []
    rows = []
    for card, d in data.items():
        if not card.startswith("card") or not isinstance(d, dict):
            continue
        r = {"id": f"amd:{card}", "name": d.get("Card Series") or d.get("Card series") or "AMD GPU", "vendor": "amd",
             "temp_c": None, "junction_c": None, "mem_c": None, "vram_used_mb": None, "vram_total_mb": None,
             "util": None, "power_w": None, "source": "rocm-smi"}
        for k, v in d.items():
            kl = k.lower()
            if "temperature" in kl:
                key = "junction_c" if "junction" in kl else "mem_c" if "memory" in kl else "temp_c" if "edge" in kl \
                    else None
                if key:
                    r[key] = _num(v)
            elif "vram total used" in kl:
                r["vram_used_mb"] = (_num(v) or 0) / 2**20
            elif "vram total memory" in kl:
                r["vram_total_mb"] = (_num(v) or 0) / 2**20
            elif "gpu use" in kl:
                r["util"] = _num(v)
        rows.append(r)
    return rows


def gpu_telemetry() -> list:
    """Temperatures, memory and load of this computer's GPUs, from nvidia-smi, rocm-smi or
    Linux's amdgpu sensors. A value that cannot be read is None, never a guess."""
    rows = _nvidia_telemetry()
    amd = _rocm_smi() or (_amdgpu_sysfs() if sys.platform.startswith("linux") else [])
    return rows + amd


def hottest(row: dict) -> float | None:
    vals = [row.get(k) for k in ("junction_c", "temp_c", "mem_c") if row.get(k) is not None]
    return max(vals) if vals else None


BUSY_PREFIX = "HOST BUSY:"            # controllers recognise a host that is busy, not broken, by this


class PresenceGuard(threading.Thread):
    """Gives the computer back to its owner: when someone uses its keyboard, mouse, a game controller or
    VR (presence.py), unload the models and refuse new work until it has been quiet for a while."""

    def __init__(self, agent: "Agent", read=None, every: float = 5.0):
        super().__init__(daemon=True)
        if read is None:
            from . import presence
            read = presence.snapshot
        self.agent, self.read, self.every = agent, read, every
        self.stop = threading.Event()
        self.last = {}
        self.last_seen = 0.0

    def cfg(self) -> dict:
        y = self.agent.cfg.get("yield") or {}
        return {"enabled": bool(y.get("enabled", False)), "idle_under": float(y.get("idle_under", 60)),
                "gamepad": bool(y.get("gamepad", True)), "vr": bool(y.get("vr", True)),
                "cooldown": float(y.get("cooldown", 300)), "unload": bool(y.get("unload", True)),
                "stop_jobs": bool(y.get("stop_jobs", False))}

    def check(self) -> None:
        c = self.cfg()
        self.last = self.read()
        if not c["enabled"]:
            if (self.agent.busy or {}).get("source") == "owner":
                self.agent.set_busy(None)
            return
        why = []
        if c["vr"] and self.last.get("vr"):
            why.append("a VR session is running")
        if c["gamepad"] and self.last.get("gamepad"):
            why.append("a game controller is in use")
        if self.last.get("idle_s") is not None and self.last["idle_s"] < c["idle_under"]:
            why.append(f"keyboard or mouse used {int(self.last['idle_s'])} s ago")
        now = time.time()
        if why:
            self.last_seen = now
            if (self.agent.busy or {}).get("source") != "owner":
                self.agent.set_busy({"why": "the host's owner is using this computer (" + "; ".join(why) + ")",
                                     "source": "owner", "since": now}, unload=c["unload"], stop_jobs=c["stop_jobs"])
        elif (self.agent.busy or {}).get("source") == "owner" and now - self.last_seen >= c["cooldown"]:
            self.agent.set_busy(None)

    def run(self):
        while not self.stop.is_set():
            try:
                self.check()
            except Exception:
                pass
            self.stop.wait(self.every)


class ThermalGuard(threading.Thread):
    """Watches the GPUs. Above the limit, the host refuses new chats and traces and unloads
    what its LLM servers can unload; below the limit minus the margin it resumes. GPUs
    throttle themselves near their own maximum; this guard is an earlier, gentler stop."""

    def __init__(self, agent: "Agent", read=gpu_telemetry, every: float = 2.0):
        super().__init__(daemon=True)
        self.agent, self.read, self.every = agent, read, every
        self.stop = threading.Event()
        self.tripped = None           # {"gpu", "temp", "at"} while too hot
        self.last = []

    def cfg(self) -> dict:
        t = self.agent.cfg.get("thermal") or {}
        return {"enabled": bool(t.get("enabled", False)), "limit_c": float(t.get("limit_c", 95.0)),
                "margin_c": float(t.get("margin_c", 10.0)), "unload": bool(t.get("unload", True))}

    def check(self) -> None:
        c = self.cfg()
        self.last = self.read()
        if not c["enabled"]:
            self.tripped = None
            return
        hot = [(r, hottest(r)) for r in self.last if hottest(r) is not None]
        worst = max(hot, key=lambda x: x[1]) if hot else None
        if self.tripped is None and worst and worst[1] >= c["limit_c"]:
            self.tripped = {"gpu": worst[0]["name"], "temp": worst[1], "at": time.time(), "limit": c["limit_c"]}
            self.agent.note("thermal guard", f"too hot: {worst[0]['name']} at {worst[1]:.0f}°C", False,
                            "new chats and traces refused until it cools")
            if c["unload"]:
                self.unload_all()
        elif self.tripped is not None and (not worst or worst[1] <= c["limit_c"] - c["margin_c"]):
            self.agent.note("thermal guard", "cooled down", True, f"{worst[1]:.0f}°C" if worst else "")
            self.tripped = None

    def unload_all(self) -> None:
        for b in self.agent.backends():
            try:
                for m in b.models():
                    if m.get("loaded"):
                        b.unload(m["id"])
            except Exception:
                pass                  # llama-server and vLLM cannot unload; they stay up but get no work

    def run(self):
        while not self.stop.is_set():
            try:
                self.check()
            except Exception as e:
                self.last = [{"error": f"{type(e).__name__}: {e}"}]
            self.stop.wait(self.every)


def local_addresses() -> list[dict]:
    """This computer's addresses, labelled with what they probably are."""
    seen = []
    try:
        for fam, *_rest, sa in socket.getaddrinfo(socket.gethostname(), None):
            seen.append(sa[0])
    except OSError:
        pass
    for probe in ("192.0.2.1", "100.100.100.100"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 9))          # no packet is sent; this picks the outgoing interface
            seen.append(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    out = []
    for a in dict.fromkeys(seen):
        try:
            ip = ipaddress.ip_address(a.split("%")[0])
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local:
            continue
        if ip in ipaddress.ip_network("100.64.0.0/10") or ip in ipaddress.ip_network("fd7a:115c:a1e0::/48"):
            kind = "vpn (Tailscale or Headscale)"
        elif ip.is_private:
            kind = "lan or vpn"
        else:
            kind = "public"
        out.append({"address": str(ip), "kind": kind})
    return out


class Agent:
    def __init__(self, home: Path | None = None):
        self.store = Store(home or default_home())
        self.pairing = Pairing(self.store)
        self.controllers = Controllers(self.store)
        self.activity = collections.deque(maxlen=200)
        self.transfers: dict = {}
        self.lock = threading.RLock()
        self.fingerprint = ""            # of the certificate the API serves, "" without TLS
        self.link = {"mode": "stdio", "urls": []}
        self._backends = (0.0, [])
        self.callhome: dict = {}         # url -> CallHome thread
        self.guard = None                # ThermalGuard, started by the runtime
        self.presence = None             # PresenceGuard, started by the runtime
        self.busy = None                 # {why, source: owner|controller, since, until} while it takes no work
        cfg = self.store.read("config.json", {})
        self.cfg = {"name": socket.gethostname()[:40] or "host", "mode": "", "port": DEFAULT_PORT, "bind": "",
                    "backends": [], "detect": True, "models_dir": "", "incoming_dir": "", **cfg}
        self.cfg["allow"] = {**ALLOW_DEFAULT, **(cfg.get("allow") or {})}

    def save_config(self) -> None:
        self.store.write("config.json", self.cfg)

    # -- paths
    def incoming(self) -> Path:
        p = Path(self.cfg.get("incoming_dir") or self.store.home / "incoming").expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def models_dir(self) -> Path | None:
        if self.cfg.get("models_dir"):
            return Path(self.cfg["models_dir"]).expanduser()
        return lmstudio_models_dir()

    # -- backends
    def backends(self, refresh: bool = False) -> list[Backend]:
        with self.lock:
            t, cached = self._backends
            if cached and not refresh and time.time() - t < 15:
                return cached
        found = [Backend(b.get("kind", "openai"), b["url"], b.get("api_key", ""), "configured")
                 for b in self.cfg.get("backends") or [] if b.get("url")]
        if self.cfg.get("detect", True):
            have = {b.url for b in found}
            for kind, url in KNOWN_BACKENDS:
                if url in have:
                    continue
                b = Backend(kind, url)
                try:
                    b.get_json("/api/tags" if kind == "ollama" else "/v1/models", timeout=0.7)
                    found.append(b)
                except Exception:
                    pass
        with self.lock:
            self._backends = (time.time(), found)
        return found

    def backend_for(self, model: str = "", backend: str = "") -> Backend:
        bs = self.backends()
        if backend:
            bs = [b for b in bs if backend in (b.kind, b.url)]
            if not bs:
                raise ToolError(f"no backend {backend!r}; list_backends shows what this host runs")
        if not bs:
            raise ToolError("no LLM server is running on this host: start LM Studio's server (Developer → Start "
                            "server), Ollama, or llama-server, or add its address in the host console")
        if not model or len(bs) == 1:
            return bs[0]
        for b in bs:
            try:
                if any(m["id"] == model for m in b.models()):
                    return b
            except Exception:
                continue
        raise ToolError(f"no backend on this host has a model {model!r}; list_models shows them")

    def need(self, what: str) -> None:
        if not self.cfg["allow"].get(what):
            raise ToolError(f"the host does not allow controllers to {ALLOW_TEXT[what]}; the host owner can allow "
                            "it in the OmniCortex Remote console")

    def note(self, who: str, what: str, ok: bool, detail: str = "") -> None:
        with self.lock:
            last = self.activity[0] if self.activity else None
            if last and (last["who"], last["what"], last["ok"], last["detail"]) == (who, what, ok, detail[:200]):
                last["t"], last["n"] = time.time(), last.get("n", 1) + 1      # the same call again: count it
                return
            self.activity.appendleft({"t": time.time(), "who": who, "what": what, "ok": ok, "detail": detail[:200]})

    # -- MCP tools
    def t_host_info(self, a):
        mdir = self.models_dir()
        inc = self.incoming()
        free = shutil.disk_usage(inc).free
        return {"role": "host", "you_are": "controller", "name": self.cfg["name"], "agent": f"omnicortex-remote {VERSION}",
                "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
                "cpus": os.cpu_count(), "ram_gb": _ram_gb(), "gpus": _gpus(),
                "backends": [b.describe() for b in self.backends(refresh=True)],
                "incoming_dir": str(inc), "incoming_free_gb": round(free / 2**30, 1),
                "models_dir": str(mdir) if mdir else None, "allowed": self.cfg["allow"], "link": self.link["mode"],
                "jobs": bool(self.cfg["allow"].get("jobs") and self.omnicortex_dir()),
                "tracer": self.tracer()}

    # -- OmniCortex jobs on this host (off unless the host owner allows "jobs")
    def omnicortex_dir(self) -> Path | None:
        """The OmniCortex folder jobs run from: configured, or the one this agent was started from."""
        cand = [self.cfg.get("omnicortex_dir")] if self.cfg.get("omnicortex_dir") else []
        cand.append(str(Path(__file__).resolve().parents[2]))
        for c in cand:
            p = Path(c).expanduser()
            if (p / "viz" / "jobs.py").is_file() and (p / "scripts").is_dir():
                return p
        return None

    def _jobs(self):
        if not self.cfg["allow"].get("jobs"):
            raise ToolError("this host does not run jobs: its owner can allow them in the agent's console "
                            "(Allow: run OmniCortex jobs)")
        root = self.omnicortex_dir()
        if root is None:
            raise ToolError("no OmniCortex folder on this host: set one in the agent's console, or start the agent "
                            "from inside a OmniCortex folder")
        with self.lock:
            cached = getattr(self, "_jobs_cache", None)
            if cached and cached[0] == root:
                return cached[1], cached[2]
            import importlib.util
            for sub in ("scripts", "viz"):
                if str(root / sub) not in sys.path:
                    sys.path.insert(0, str(root / sub))
            spec = importlib.util.spec_from_file_location("oc_host_jobs", root / "viz" / "jobs.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            runner = mod.JobRunner(self.store.home / "jobs", int(self.cfg.get("max_jobs") or 1),
                                   cwd=self.cfg.get("jobs_cwd") or root)
            self._jobs_cache = (root, mod, runner)
            return mod, runner

    def t_jobs_info(self, a):
        root = self.omnicortex_dir()
        info = {"allowed": bool(self.cfg["allow"].get("jobs")), "omnicortex_dir": str(root) if root else None,
                "python": sys.executable}
        if info["allowed"] and root:
            mod, _ = self._jobs()
            info["kinds"] = sorted(mod.all_specs())
        return info

    def t_run_job(self, a):
        self.not_busy()
        mod, runner = self._jobs()
        try:
            return runner.start(str(a["kind"]), dict(a.get("values") or {}))
        except RuntimeError as e:
            raise ToolError(str(e))

    def t_jobs(self, a):
        _, runner = self._jobs()
        return {"jobs": runner.list(30)}

    def t_job_log(self, a):
        _, runner = self._jobs()
        return {"log": runner.log(str(a["id"]), max(1000, min(int(a.get("tail") or 20000), 200000)))}

    def t_cancel_job(self, a):
        _, runner = self._jobs()
        return runner.cancel(str(a["id"]))

    def t_restart_agent(self, a):
        if not self.cfg["allow"].get("restart", True):
            raise ToolError("the host owner does not allow restarting this agent")

        def later():
            time.sleep(0.7)                     # let the reply reach the controller
            argv = [sys.executable] + sys.argv
            if os.name == "nt":                 # Windows: a fresh process, then this one exits
                subprocess.Popen(argv, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
                os._exit(0)
            os.execv(sys.executable, argv)
        threading.Thread(target=later, daemon=True).start()
        return {"ok": True, "note": "the agent restarts now and is back in a few seconds"}

    # -- busy: the owner is using the computer, or a controller's rule (power) asked it to rest
    def set_busy(self, b: dict | None, unload: bool = False, stop_jobs: bool = False) -> None:
        was = self.busy
        self.busy = b
        if b and not was:
            self.note(b.get("source", "busy"), "busy: no new work", True, b.get("why", ""))
        elif was and not b:
            self.note(was.get("source", "busy"), "free again", True, "")
        if b and unload:
            for bk in self.backends():
                try:
                    for m in bk.models():
                        if m.get("loaded"):
                            bk.unload(m["id"])
                except Exception:
                    pass
        if b and stop_jobs and getattr(self, "_jobs_cache", None):
            runner = self._jobs_cache[2]
            for j in runner.list(30):
                if j.get("status") == "running":
                    try:
                        runner.cancel(j["id"])
                    except Exception:
                        pass

    def not_busy(self) -> None:
        b = self.busy
        if b and b.get("until") and time.time() >= b["until"]:
            self.set_busy(None)
            b = None
        if b:
            until = f" until {time.strftime('%Y-%m-%d %H:%M', time.localtime(b['until']))}" if b.get("until") else ""
            raise ToolError(f"{BUSY_PREFIX} {b.get('why') or 'busy'}{until}; use another host for now")

    def t_set_busy(self, a):
        self.need("busy")
        if a.get("busy"):
            mins = float(a.get("minutes") or 0)
            self.set_busy({"why": str(a.get("reason") or "a controller's automation rule")[:300], "source": "controller",
                           "since": time.time(), "until": time.time() + mins * 60 if mins > 0 else
                           (float(a["until"]) if a.get("until") else None)}, unload=bool(a.get("unload")))
        elif (self.busy or {}).get("source") != "owner":          # the owner's own use is not a controller's to end
            self.set_busy(None)
        return {"busy": self.busy}

    def t_list_backends(self, a):
        out = []
        for b in self.backends(refresh=True):
            d = b.describe()
            d["load_unload"] = ("yes" if b.kind == "ollama" or (b.kind == "lmstudio" and lms_cli()) else
                                "load only (just-in-time)" if b.kind in ("lmstudio", "openai") else "no")
            out.append(d)
        return {"backends": out}

    def t_list_models(self, a):
        rows = []
        for b in self.backends():
            if a.get("backend") and a["backend"] not in (b.kind, b.url):
                continue
            try:
                rows += [{**m, "backend": b.kind} for m in b.models()]
            except Exception as e:
                rows.append({"backend": b.kind, "error": f"{type(e).__name__}: {e}"[:200]})
        return {"models": rows}

    def too_hot(self) -> None:
        g = self.guard
        if g is not None and g.tripped:
            t = g.tripped
            raise ToolError(f"this host is too hot ({t['gpu']} at {t['temp']:.0f}°C, limit {t['limit']:.0f}°C): "
                            "it takes no new work until it cools down")

    def t_telemetry(self, a):
        g = self.guard
        rows = g.last if g is not None and g.last else gpu_telemetry()
        p = self.presence
        return {"gpus": rows, "thermal": {**(g.cfg() if g else {}), "tripped": g.tripped if g else None},
                "presence": (p.last if p is not None and p.last else None), "busy": self.busy,
                "yield": p.cfg() if p is not None else None}

    def t_chat(self, a):
        self.need("chat")
        self.too_hot()
        self.not_busy()
        if not a.get("messages"):
            raise ToolError("messages are needed: [{\"role\": \"user\", \"content\": \"...\"}]")
        b = self.backend_for(a.get("model", ""), a.get("backend", ""))
        body = {"model": a.get("model", ""), "messages": a["messages"]}
        for k in ("temperature", "max_tokens", "top_p", "seed", "stop", "response_format"):
            if a.get(k) is not None:
                body[k] = a[k]
        stream = a.get("__stream")
        r = b.chat_stream(body, stream.send, stream.cancelled) if stream is not None else b.chat(body)
        return {"reply": ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or "",
                "model": r.get("model", body["model"]), "backend": b.kind, "usage": r.get("usage"), "response": r}

    # -- model files and the tracer
    def _gguf_candidates(self):
        for root in (self.models_dir(), self.incoming()):
            if root and root.is_dir():
                for f in root.rglob("*.gguf"):
                    if "mmproj" not in f.name.lower():
                        yield root, f

    def find_gguf(self, model: str, backend: str = "") -> Path:
        """The GGUF file behind a model id: configured, asked of the server, or found by name."""
        if (self.cfg.get("model_files") or {}).get(model):
            return Path(self.cfg["model_files"][model]).expanduser()
        try:
            b = self.backend_for(model, backend)
        except ToolError:
            b = None
        if b is not None and b.kind == "ollama":
            try:
                mf = b.get_json("/api/show", method="POST", body={"model": model}).get("modelfile") or ""
                m = re.search(r"^FROM\s+(\S+)", mf, re.M)
                if m and Path(m.group(1)).is_file():
                    return Path(m.group(1))
            except (ToolError, OSError, ValueError):
                pass
        if b is not None and b.kind == "llama-server":
            try:
                mp = b.get_json("/props").get("model_path")
                if mp and Path(mp).is_file():
                    return Path(mp)
            except (ToolError, OSError, ValueError):
                pass
        want = model.lower().split("@")[0].strip("/")
        best = None
        for root, f in self._gguf_candidates():
            rel = f.relative_to(root).as_posix().lower()
            keys = {f.stem.lower(), f.parent.name.lower(), "/".join(rel.split("/")[:-1]), rel}
            if want in keys or any(k.endswith("/" + want) for k in keys):
                if best is None or f.stat().st_size > best.stat().st_size:
                    best = f
        if best is None:
            raise ToolError(f"cannot find the GGUF file for {model} on this host; the host owner can name it in "
                            "config.json under model_files")
        return best

    def sha256(self, f: Path) -> str:
        st = f.stat()
        cache = self.store.read("hashes.json", {})
        hit = cache.get(str(f))
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime:
            return hit[2]
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        cache[str(f)] = [st.st_size, st.st_mtime, h.hexdigest()]
        self.store.write("hashes.json", cache)
        return h.hexdigest()

    def t_model_file(self, a):
        f = self.find_gguf(a["model"], a.get("backend", ""))
        return {"model": a["model"], "file": f.name, "size": f.stat().st_size, "sha256": self.sha256(f)}

    def tracer(self) -> dict:
        """OmniCortex's tracer on this host: its scripts folder and llama-cett-dump, if present."""
        t = dict(self.cfg.get("tracer") or {})
        root = Path(t.get("omnicortex") or Path(__file__).resolve().parents[2]).expanduser()
        cett = t.get("cett") or os.environ.get("OC_CETT") or shutil.which("llama-cett-dump") or ""
        ok = (root / "scripts" / "hscore.py").is_file() and bool(cett) and Path(cett).is_file()
        return {"available": ok, "omnicortex": str(root), "cett": cett, "python": t.get("python") or sys.executable}

    def t_trace_reply(self, a):
        self.need("trace")
        self.too_hot()
        self.not_busy()
        tr = self.tracer()
        if not tr["available"]:
            raise ToolError("this host has no OmniCortex tracer: install OmniCortex here and build llama.cpp with "
                            "cett-dump, then set tracer in config.json (omnicortex folder, cett binary)")
        f = self.find_gguf(a["model"], a.get("backend", ""))
        import tempfile
        with tempfile.TemporaryDirectory(prefix="ns-trace-") as work:
            w = Path(work)
            (w / "clf.npz").write_bytes(base64.b64decode(a["classifier"]))
            (w / "msgs.json").write_text(json.dumps(a.get("messages") or []), encoding="utf-8")
            (w / "reply.txt").write_text(a["text"], encoding="utf-8")
            cmd = [tr["python"], str(Path(tr["omnicortex"]) / "scripts" / "hscore.py"), "--binary", tr["cett"],
                   "--gguf", str(f), "--classifier", str(w / "clf.npz"), "--messages", str(w / "msgs.json"),
                   "--response-file", str(w / "reply.txt"), "--trace-out", str(w / "trace.npz"),
                   "--ngl", str(int(a.get("ngl", 99)))]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise ToolError(f"the tracer failed: {(r.stderr or r.stdout).strip()[-500:]}")
            data = (w / "trace.npz").read_bytes()
        return {"model": a["model"], "file": f.name, "trace": base64.b64encode(data).decode(), "bytes": len(data)}

    def t_load_model(self, a):
        self.need("load")
        self.not_busy()
        b = self.backend_for("", a.get("backend", "")) if a.get("backend") else self.backend_for(a["model"])
        return b.load(a["model"], a.get("context_length"), a.get("gpu"), a.get("keep_alive"))

    def t_unload_model(self, a):
        self.need("load")
        b = self.backend_for("", a.get("backend", "")) if a.get("backend") else self.backend_for(a["model"])
        return b.unload(a["model"])

    def _root(self, where: str) -> Path:
        if where == "models":
            d = self.models_dir()
            if not d:
                raise ToolError("no models folder: LM Studio is not installed here, or set models_dir in the console")
            return d
        return self.incoming()

    def t_list_files(self, a):
        where = a.get("where", "incoming")
        root = self._root(where)
        rows = []
        if root.is_dir():
            pat = "*.gguf" if where == "models" else "*"
            for p in sorted(root.rglob(pat) if where == "models" else root.iterdir()):
                if p.is_file():
                    rows.append({"file": p.relative_to(root).as_posix(), "size": p.stat().st_size,
                                 "partial": p.name.endswith(".part")})
        return {"where": where, "folder": str(root), "files": rows[:2000]}

    def t_receive_file(self, a):
        self.need("receive")
        name = safe_name(a.get("name") or urllib.parse.urlsplit(a["url"]).path.split("/")[-1])
        tid = secrets.token_hex(4)
        self.transfers[tid] = {"id": tid, "file": name, "url": re.sub(r"[?#].*", "", a["url"]), "status": "running",
                               "done": 0, "size": None, "started": time.time()}
        threading.Thread(target=self._download, daemon=True,
                         args=(tid, a["url"], self.incoming() / name, a.get("sha256"), a.get("token", ""),
                               a.get("fingerprint"))).start()
        return {"transfer": tid, "file": name, "note": "runs in the background; the transfers tool shows progress"}

    def _download(self, tid, url, dest: Path, sha, token, fingerprint):
        t = self.transfers[tid]
        part = dest.with_name(dest.name + ".part")
        try:
            have = part.stat().st_size if part.exists() else 0
            u = urllib.parse.urlsplit(url)
            conn = open_conn(url, fingerprint if fingerprint is not None else (None if u.scheme == "https" else ""),
                             600)
            headers = {"Range": f"bytes={have}-"} if have else {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            conn.request("GET", u.path + (f"?{u.query}" if u.query else ""), headers=headers)
            r = conn.getresponse()
            if r.status == 416:
                r.read()
            elif r.status in (200, 206):
                if r.status == 200:
                    have = 0
                total = r.getheader("Content-Length")
                t["size"] = have + int(total) if total else None
                with open(part, "ab" if have else "wb") as f:
                    t["done"] = have
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        t["done"] += len(chunk)
            else:
                raise ToolError(f"HTTP {r.status}: {r.read()[:200].decode(errors='replace')}")
            conn.close()
            if sha:
                t["status"] = "checking"
                if file_sha256(part) != sha.lower():
                    part.unlink()
                    raise ToolError("the file arrived damaged (SHA-256 differs); it was removed, send it again")
            os.replace(part, dest)
            t.update(status="done", size=dest.stat().st_size, done=dest.stat().st_size, finished=time.time())
        except Exception as e:
            t.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])

    def t_transfers(self, a):
        return {"transfers": list(self.transfers.values())}

    def t_install_model(self, a):
        self.need("install")
        src = _inside(self.incoming(), safe_name(a["file"]))
        if not src.is_file():
            raise ToolError(f"{a['file']} is not in the incoming folder (list_files shows it)")
        into = a.get("into", "lmstudio")
        stem = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", src.name, flags=re.I)
        if into == "ollama":
            exe = shutil.which("ollama")
            if not exe:
                raise ToolError("the ollama command is not installed on this host")
            name = re.sub(r"[^a-z0-9._-]+", "-", (a.get("name") or stem).lower())
            mf = self.incoming() / f"{name}.Modelfile"
            mf.write_text(f"FROM {src}\n", encoding="utf-8")
            r = subprocess.run([exe, "create", name, "-f", str(mf)], capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise ToolError(f"ollama create failed: {(r.stderr or r.stdout).strip()[-500:]}")
            return {"installed": name, "into": "ollama", "note": "the GGUF stays in the incoming folder; Ollama copied it"}
        root = self.models_dir()
        if not root:
            raise ToolError("LM Studio's models folder was not found; set models_dir in the host console")
        pub = re.sub(r"[^A-Za-z0-9._-]+", "-", a.get("publisher") or "omnicortex")
        folder = re.sub(r"[^A-Za-z0-9._-]+", "-", a.get("name") or stem)
        dest = _inside(root, f"{pub}/{folder}/{src.name}")
        if dest.exists() and not a.get("overwrite"):
            raise ToolError(f"{dest} exists; pass overwrite to replace it")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return {"installed": str(dest), "into": "lmstudio", "model_key": f"{pub}/{folder}",
                "note": "LM Studio lists it after a refresh; with just-in-time loading on, a chat request loads it"}

    def t_delete_file(self, a):
        self.need("delete")
        root = self._root(a.get("where", "incoming"))
        p = _inside(root, a["file"])
        if not p.is_file():
            raise ToolError(f"no file {a['file']} in {root}")
        p.unlink()
        return {"deleted": str(p)}

    TOOLS = [
        ("host_info", "Who and what this host is: role, OS, RAM, GPUs, LLM servers, folders, what controllers may do.",
         {}, {"readOnlyHint": True}),
        ("list_backends", "The LLM servers running on this host (LM Studio, Ollama, llama-server, OpenAI-compatible) "
         "and whether models can be loaded and unloaded through them.", {}, {"readOnlyHint": True}),
        ("list_models", "Models the host's LLM servers offer, with loaded state where the server reports it.",
         {"backend": {"type": "string", "description": "only this backend (kind or URL)"}}, {"readOnlyHint": True}),
        ("chat", "Ask a model on this host. Returns the reply and the full OpenAI-style response.",
         {"model": {"type": "string"}, "messages": {"type": "array", "items": {"type": "object"}},
          "temperature": {"type": "number"}, "max_tokens": {"type": "integer"}, "top_p": {"type": "number"},
          "seed": {"type": "integer"}, "backend": {"type": "string"}}, {"readOnlyHint": True}, ["messages"]),
        ("telemetry", "GPU temperatures (edge, junction, memory), memory use, load and power on this host, and "
         "whether its thermal guard has paused work.", {}, {"readOnlyHint": True}),
        ("model_file", "Where a model's GGUF file is on this host, its size and SHA-256 (for finding the same "
         "file elsewhere).", {"model": {"type": "string"}, "backend": {"type": "string"}}, {"readOnlyHint": True},
         ["model"]),
        ("trace_reply", "Run a reply through the model's GGUF with OmniCortex's tracer and an H-Neuron classifier; "
         "returns the per-token activation trace (.npz, base64) for a 3D view. Needs OmniCortex and "
         "llama-cett-dump on this host.",
         {"model": {"type": "string"}, "messages": {"type": "array", "items": {"type": "object"}},
          "text": {"type": "string"}, "classifier": {"type": "string", "description": "the classifier .npz, base64"},
          "ngl": {"type": "integer"}, "backend": {"type": "string"}},
         {"readOnlyHint": True}, ["model", "text", "classifier"]),
        ("load_model", "Load a model (LM Studio through its lms command, Ollama by keep_alive).",
         {"model": {"type": "string"}, "backend": {"type": "string"}, "context_length": {"type": "integer"},
          "gpu": {"type": "string", "description": "LM Studio: max, off, or 0..1"},
          "keep_alive": {"type": "string", "description": "Ollama: how long to keep it, e.g. 30m"}},
         {"readOnlyHint": False, "destructiveHint": False}, ["model"]),
        ("unload_model", "Unload a model to free memory.", {"model": {"type": "string"}, "backend": {"type": "string"}},
         {"readOnlyHint": False, "destructiveHint": False}, ["model"]),
        ("list_files", "Files in the host's incoming folder (received files) or GGUFs in its models folder.",
         {"where": {"type": "string", "enum": ["incoming", "models"]}}, {"readOnlyHint": True}),
        ("receive_file", "Make the host download a file (a GGUF) from a URL into its incoming folder, resuming "
         "where a previous try stopped. Runs in the background; see transfers.",
         {"url": {"type": "string"}, "name": {"type": "string"}, "sha256": {"type": "string"},
          "token": {"type": "string", "description": "Bearer token for the URL"},
          "fingerprint": {"type": "string", "description": "pin the URL's TLS certificate (SHA-256 hex)"}},
         {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True}, ["url"]),
        ("transfers", "Progress of files being received.", {}, {"readOnlyHint": True}),
        ("install_model", "Move a received GGUF into LM Studio's models folder (publisher/name/file.gguf) or "
         "create an Ollama model from it.",
         {"file": {"type": "string"}, "into": {"type": "string", "enum": ["lmstudio", "ollama"]},
          "publisher": {"type": "string"}, "name": {"type": "string"}, "overwrite": {"type": "boolean"}},
         {"readOnlyHint": False, "destructiveHint": False}, ["file"]),
        ("delete_file", "Delete a file from the incoming or models folder. Off unless the host owner allows it.",
         {"file": {"type": "string"}, "where": {"type": "string", "enum": ["incoming", "models"]}},
         {"readOnlyHint": False, "destructiveHint": True}, ["file"]),
        ("jobs_info", "Whether this host runs OmniCortex jobs, from which OmniCortex folder, and the job kinds it has.",
         {}, {"readOnlyHint": True}),
        ("run_job", "Start a OmniCortex job on this host: the same job kinds and fields as Studio's Jobs page, "
         "checked by the host's own copy of OmniCortex. Paths in values are paths on this host.",
         {"kind": {"type": "string"}, "values": {"type": "object"}},
         {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True}, ["kind"]),
        ("jobs", "Jobs on this host, newest first, with progress for running ones and a reason for failed ones.",
         {}, {"readOnlyHint": True}),
        ("job_log", "The end of one job's log.", {"id": {"type": "string"}, "tail": {"type": "integer"}},
         {"readOnlyHint": True}, ["id"]),
        ("cancel_job", "Stop a running job on this host.", {"id": {"type": "string"}},
         {"readOnlyHint": False, "destructiveHint": True}, ["id"]),
        ("set_busy", "Mark this host busy (no new work, optionally unload its models) or free again. While its "
         "owner uses it, it stays busy whatever a controller says.",
         {"busy": {"type": "boolean"}, "minutes": {"type": "number"}, "until": {"type": "number"},
          "reason": {"type": "string"}, "unload": {"type": "boolean"}},
         {"readOnlyHint": False, "destructiveHint": False}, ["busy"]),
        ("restart_agent", "Restart this agent (the host comes back by itself in a few seconds).", {},
         {"readOnlyHint": False, "destructiveHint": False}),
    ]

    def tool_list(self) -> list[dict]:
        out = []
        for row in self.TOOLS:
            name, desc, props, ann = row[:4]
            schema = {"type": "object", "properties": props}
            if len(row) > 4:
                schema["required"] = row[4]
            out.append({"name": name, "description": desc, "inputSchema": schema,
                        "annotations": {"title": name.replace("_", " "), **ann}})
        return out

    def call_tool(self, name: str, args: dict, who: str, stream=None) -> dict:
        fn = getattr(self, "t_" + name, None) if any(t[0] == name for t in self.TOOLS) else None
        if fn is None:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
        try:
            a = {k: v for k, v in dict(args or {}).items() if not k.startswith("__")}
            if stream is not None and name == "chat":
                a["__stream"] = stream
            out = fn(a)
            self.note(who, name, True, str(args.get("model", "")) if name in ("chat", "trace_reply")
                      else json.dumps(args or {})[:120])
            text = (out["reply"] if name == "chat" else f"trace of {out['bytes']} bytes" if name == "trace_reply"
                    else json.dumps(out, indent=1))
            return {"content": [{"type": "text", "text": text}], "structuredContent": out, "isError": False}
        except (ToolError, KeyError, OSError, ValueError, subprocess.SubprocessError) as e:
            msg = f"missing argument {e}" if isinstance(e, KeyError) else str(e)
            self.note(who, name, False, msg)
            return {"content": [{"type": "text", "text": msg}], "isError": True}

    def instructions(self) -> str:
        return (f"You are connected to the OmniCortex Remote agent on the HOST '{self.cfg['name']}'. The host is the "
                "computer that runs the models (LM Studio, Ollama, llama.cpp); you, the MCP client, are the "
                "CONTROLLER. Your tool calls act on the host: list_models and chat use its models, load_model and "
                "unload_model change what is in its memory, receive_file and install_model put new models on it. "
                "The host owner sees every call and can revoke you.")

    def handle_rpc(self, msg, who: str, stream_for=None):
        """One JSON-RPC message -> response (None for notifications). Lists are batches.
        stream_for(progress token) -> a stream (send, cancelled) for tools that can
        report as they go, when the caller asked for progress."""
        if isinstance(msg, list):
            out = [r for r in (self.handle_rpc(m, who) for m in msg) if r is not None]
            return out or None
        if not isinstance(msg, dict):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
        mid, method, params = msg.get("id"), msg.get("method", ""), msg.get("params") or {}
        if mid is None:
            return None
        if method == "initialize":
            res = {"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "omnicortex-remote", "title": f"OmniCortex Remote host: {self.cfg['name']}",
                                  "version": VERSION},
                   "instructions": self.instructions()}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": self.tool_list()}
        elif method == "tools/call":
            token = (params.get("_meta") or {}).get("progressToken")
            stream = stream_for(token) if stream_for is not None and token is not None else None
            res = self.call_tool(params.get("name", ""), params.get("arguments") or {}, who, stream)
            if stream is not None:
                stream.flush()
        elif method in ("resources/list", "prompts/list"):
            res = {method.split("/")[0]: []}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": res}

    def state(self) -> dict:
        p = self.pairing.current()
        return {"role": "host", "name": self.cfg["name"], "version": VERSION, "link": self.link,
                "fingerprint": self.fingerprint, "short_fingerprint": short_fp(self.fingerprint),
                "pairing": {"code": fmt_code(p["code"]), "expires": p["expires"],
                            "link": self.pair_link(p["code"])} if p else None,
                "controllers": self.controllers.list(), "allow": self.cfg["allow"], "allow_text": ALLOW_TEXT,
                "omnicortex_dir": str(self.omnicortex_dir() or ""), "omnicortex_dir_set": self.cfg.get("omnicortex_dir", ""),
                "yield": self.presence.cfg() if self.presence is not None else dict(self.cfg.get("yield") or {}),
                "presence": self.presence.last if self.presence is not None else None, "busy": self.busy,
                "backends": [b.describe() for b in self.backends()], "configured_backends": self.cfg["backends"],
                "callhome": [{"url": c.url, "controller": c.entry.get("controller", ""), "status": c.status}
                             for c in self.callhome.values()],
                "addresses": local_addresses(), "activity": list(self.activity)[:50],
                "transfers": list(self.transfers.values())[-20:], "home": str(self.store.home),
                "telemetry": self.t_telemetry({}),
                "incoming_dir": str(self.incoming()), "models_dir": str(self.models_dir() or "")}

    def pair_link(self, code: str) -> str:
        urls = self.link.get("urls") or []
        if not urls:
            return ""
        u = urllib.parse.urlsplit(urls[0])
        frag = urllib.parse.urlencode({"c": fmt_code(code), **({"fp": self.fingerprint} if self.fingerprint else {}),
                                       "s": u.scheme, "n": self.cfg["name"]})
        return f"nsr://{u.netloc}/#{frag}"


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ the host's API (network)

class Throttle:
    def __init__(self, limit=10, window=60.0):
        self.limit, self.window, self.hits = limit, window, {}

    def blocked(self, ip: str) -> bool:
        now = time.time()
        self.hits[ip] = [t for t in self.hits.get(ip, []) if now - t < self.window]
        return len(self.hits[ip]) >= self.limit

    def fail(self, ip: str) -> None:
        self.hits.setdefault(ip, []).append(time.time())


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "omnicortex-remote"
    agent: Agent = None
    throttle = Throttle()

    def log_message(self, *a):
        pass

    def _send(self, code: int, obj=None, headers: dict | None = None):
        body = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(code)
        if obj is not None:
            self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self, limit=MAX_JSON) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0 or n > limit:
            raise ValueError("request too large")
        return self.rfile.read(n)

    def _controller(self):
        ip = self.client_address[0]
        if self.throttle.blocked(ip):
            self._send(429, {"error": "too many failed attempts; wait a minute"})
            return None
        auth = self.headers.get("Authorization", "")
        c = self.agent.controllers.check(auth[7:].strip() if auth.lower().startswith("bearer ") else "")
        if c is None:
            self.throttle.fail(ip)
            self._send(401, {"error": "not paired with this host: pair with the code shown in the host's "
                                      "OmniCortex Remote console"})
        return c

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/info", "/"):
            ag = self.agent
            return self._send(200, {"nsr": 1, "role": "host", "name": ag.cfg["name"], "version": VERSION,
                                    "pairing_open": ag.pairing.current() is not None, "tls": bool(ag.fingerprint),
                                    "about": ROLE_LINE})
        c = self._controller()
        if c is None:
            return
        if path == "/mcp":
            return self._send(405, {"error": "this server answers POST only (no event stream)"}, {"Allow": "POST"})
        if path == "/v1/models":
            return self._v1_models()
        if path.startswith("/upload/"):
            name = safe_name(urllib.parse.unquote(path[8:]))
            final, part = self.agent.incoming() / name, self.agent.incoming() / (name + ".part")
            return self._send(200, {"file": name, "complete": final.exists(),
                                    "size": final.stat().st_size if final.exists() else
                                    (part.stat().st_size if part.exists() else 0)})
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.split("?")[0] == "/mcp" and self._controller() is not None:
            return self._send(200, {})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/pair":
                return self._pair()
            c = self._controller()
            if c is None:
                return
            who = c["name"]
            if path == "/mcp":
                msg = json.loads(self._body() or b"null")
                out = self.agent.handle_rpc(msg, who)
                headers = {}
                if isinstance(msg, dict) and msg.get("method") == "initialize":
                    headers["Mcp-Session-Id"] = secrets.token_hex(16)
                if out is None:
                    return self._send(202, None, headers)
                return self._send(200, out, headers)
            if path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
                return self._v1(path, who)
            self._send(404, {"error": "not found"})
        except ValueError as e:
            self._send(400, {"error": str(e)})

    def _pair(self):
        ip = self.client_address[0]
        if self.throttle.blocked(ip):
            return self._send(429, {"error": "too many attempts; wait a minute"})
        req = json.loads(self._body(65536) or b"{}")
        ag = self.agent
        code = ag.pairing.consume(str(req.get("proof", "")), ag.fingerprint)
        if code is None:
            self.throttle.fail(ip)
            return self._send(403, {"error": "the pairing code is wrong or expired, or the connection went through "
                                             "another certificate. Make a new code on the host and try again."})
        rec, token = ag.controllers.add(str(req.get("controller", "controller")), ag.link["mode"])
        ag.note(rec["name"], "paired", True, f"from {ip}")
        self._send(200, {"token": token, "controller_id": rec["id"], "host": ag.cfg["name"], "role": "host",
                         "fingerprint": ag.fingerprint, "confirm": pair_proof(code, ag.fingerprint, "accept")})

    def do_PUT(self):
        path = self.path.split("?")[0]
        if not path.startswith("/upload/"):
            return self._send(404, {"error": "not found"})
        c = self._controller()
        if c is None:
            return
        try:
            self.agent.need("receive")
            name = safe_name(urllib.parse.unquote(path[8:]))
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
            inc = self.agent.incoming()
            part = inc / (name + ".part")
            have = part.stat().st_size if part.exists() else 0
            off = int(q.get("offset", 0))
            if off != have:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                return self._send(409, {"error": "offset mismatch; resume from size", "size": have})
            data = self._body(MAX_CHUNK)
            if data:
                if shutil.disk_usage(inc).free < len(data) + (256 << 20):
                    return self._send(507, {"error": "the host's disk is nearly full"})
                with open(part, "ab" if have else "wb") as f:
                    f.write(data)
            if q.get("final"):
                if q.get("sha256") and file_sha256(part) != q["sha256"].lower():
                    part.unlink()
                    return self._send(422, {"error": "SHA-256 differs; the partial file was removed, send it again"})
                os.replace(part, inc / name)
                self.agent.note(c["name"], "sent a file", True, name)
                return self._send(200, {"file": name, "complete": True, "size": (inc / name).stat().st_size})
            self._send(200, {"file": name, "complete": False, "size": have + len(data)})
        except ToolError as e:
            self._send(403, {"error": str(e)})
        except ValueError as e:
            self._send(400, {"error": str(e)})

    # the OpenAI-compatible pass-through: the data plane controllers use for chat
    def _v1_models(self):
        rows = []
        for b in self.agent.backends():
            try:
                rows += [{"id": m["id"], "object": "model", "owned_by": b.kind, "loaded": m.get("loaded")}
                         for m in b.models()]
            except Exception:
                pass
        self._send(200, {"object": "list", "data": rows})

    def _v1(self, path, who):
        try:
            self.agent.need("chat")
            raw = self._body()
            body = json.loads(raw or b"{}")
            b = self.agent.backend_for(body.get("model", ""))
        except ToolError as e:
            return self._send(403 if "allow" in str(e) else 404, {"error": {"message": str(e)}})
        conn, r = b.request("POST", path, body, timeout=900)
        try:
            self.send_response(r.status)
            self.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
            if body.get("stream"):
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    line = r.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                self.close_connection = True
            else:
                data = r.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        finally:
            conn.close()
        if path == "/v1/chat/completions":
            self.agent.note(who, "chat", r.status < 400, str(body.get("model", "")))


# ------------------------------------------------------------------ the console (this computer only)

class ConsoleHandler(BaseHTTPRequestHandler):
    agent: Agent = None
    key = ""
    port = 0
    runtime = None

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ok(self) -> bool:
        # Loopback only, the right Host (no DNS rebinding), and the per-run key that only
        # the person who started the agent has (no other web page can drive this).
        host = (self.headers.get("Host") or "").lower()
        if host not in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"):
            self._send(403, {"error": "wrong host"})
            return False
        if not hmac.compare_digest(self.headers.get("X-Console-Key", ""), self.key):
            self._send(403, {"error": "open the console from the link the agent printed"})
            return False
        return True

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            return self._send(200, CONSOLE_PAGE, "text/html; charset=utf-8")
        if self.path == "/api/state":
            if self._ok():
                self._send(200, self.agent.state())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._ok():
            return
        ag = self.agent
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/api/pair/open":
                ag.pairing.open(float(req.get("ttl") or PAIR_TTL))
            elif self.path == "/api/pair/close":
                ag.pairing.close()
            elif self.path == "/api/controllers/revoke":
                ag.controllers.revoke(str(req.get("id", "")))
                ag.note("host owner", "revoked a controller", True, str(req.get("id", "")))
            elif self.path == "/api/allow":
                for k, v in (req.get("allow") or {}).items():
                    if k in ALLOW_DEFAULT:
                        ag.cfg["allow"][k] = bool(v)
                ag.save_config()
            elif self.path == "/api/thermal":
                t = dict(ag.cfg.get("thermal") or {})
                for k in ("enabled", "unload"):
                    if k in req:
                        t[k] = bool(req[k])
                for k, lo, hi in (("limit_c", 50, 110), ("margin_c", 2, 30)):
                    if k in req:
                        t[k] = min(hi, max(lo, float(req[k])))
                ag.cfg["thermal"] = t
                ag.save_config()
                ag.note("host owner", "thermal guard settings", True, json.dumps(t))
            elif self.path == "/api/yield":
                y = dict(ag.cfg.get("yield") or {})
                for k in ("enabled", "gamepad", "vr", "unload", "stop_jobs"):
                    if k in req:
                        y[k] = bool(req[k])
                for k, lo, hi in (("idle_under", 5, 3600), ("cooldown", 0, 86400)):
                    if k in req:
                        y[k] = min(hi, max(lo, float(req[k])))
                ag.cfg["yield"] = y
                ag.save_config()
                if req.get("free_now"):
                    ag.set_busy(None)
                    if ag.presence is not None:
                        ag.presence.last_seen = 0
            elif self.path == "/api/omnicortex-dir":
                d = str(req.get("dir", "")).strip()
                if d and not (Path(d).expanduser() / "viz" / "jobs.py").is_file():
                    raise ValueError(f"{d} is not a OmniCortex folder (no viz/jobs.py in it)")
                ag.cfg["omnicortex_dir"] = d
                ag._jobs_cache = None
                ag.save_config()
            elif self.path == "/api/name":
                ag.cfg["name"] = re.sub(r"[^\w .-]+", "", str(req.get("name", "")))[:40] or ag.cfg["name"]
                ag.save_config()
            elif self.path == "/api/backends":
                url = str(req.get("url", "")).strip().rstrip("/")
                if req.get("remove"):
                    ag.cfg["backends"] = [b for b in ag.cfg["backends"] if b.get("url") != url]
                elif url:
                    if not re.match(r"https?://", url):
                        raise ValueError("an address like http://127.0.0.1:1234")
                    ag.cfg["backends"] = [b for b in ag.cfg["backends"] if b.get("url") != url] + [
                        {"kind": req.get("kind") or "openai", "url": url, "api_key": req.get("api_key", "")}]
                ag.save_config()
                ag.backends(refresh=True)
            elif self.path == "/api/link":
                if self.runtime is None:
                    raise ValueError("the link can be changed only while the agent serves")
                self.runtime.relink(req)
            elif self.path == "/api/callhome":
                entry = claim_invite(ag, str(req.get("invite", "")))
                if self.runtime is not None:
                    self.runtime.start_callhome(entry)
            elif self.path == "/api/callhome/remove":
                ag.store.write("callhome.json", [c for c in ag.store.read("callhome.json", [])
                                                 if c["url"] != req.get("url")])
                t = ag.callhome.pop(req.get("url"), None)
                if t:
                    t.stop.set()
            else:
                return self._send(404, {"error": "not found"})
            self._send(200, ag.state())
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            self._send(400, {"error": str(e)})


# ------------------------------------------------------------------ call home

def claim_invite(agent: Agent, invite: str) -> dict:
    """Join a controller that cannot reach this host: claim its invite and keep a
    connection open to it from here."""
    inv = parse_invite(invite)
    url = inv["url"].rstrip("/")
    scheme = urllib.parse.urlsplit(url).scheme
    fp = inv["fingerprint"]
    # A pinned certificate when the invite names one; public HTTPS checks otherwise.
    pin = fp if fp else (None if scheme == "https" else "")
    proof = pair_proof(inv["code"], fp, "claim")
    st, obj, _seen = http_json(url, "POST", "/api/remote/claim",
                               {"proof": proof, "name": agent.cfg["name"], "version": VERSION}, fingerprint=pin)
    if st != 200:
        raise RuntimeError(f"the controller refused: {obj.get('error', st)}")
    if not hmac.compare_digest(str(obj.get("confirm", "")), pair_proof(inv["code"], fp, "accept")):
        raise RuntimeError("the controller could not prove it made this invite; not joining")
    entry = {"url": url, "token": obj["token"], "fingerprint": fp, "controller": obj.get("controller", ""),
             "host_name": obj.get("name", agent.cfg["name"]), "added": time.time()}
    agent.store.write("callhome.json", [c for c in agent.store.read("callhome.json", []) if c["url"] != url] + [entry])
    agent.note(entry["controller"] or url, "joined (this host calls in)", True, url)
    return entry


class CallHome(threading.Thread):
    """Long-polls a controller for MCP messages and posts the answers back."""

    def __init__(self, agent: Agent, entry: dict):
        super().__init__(daemon=True)
        self.agent, self.entry, self.url = agent, entry, entry["url"]
        self.stop = threading.Event()
        self.status = "connecting"
        fp = entry.get("fingerprint") or ""
        self.pin = fp if fp else (None if urllib.parse.urlsplit(self.url).scheme == "https" else "")

    def run(self):
        delay = 2
        while not self.stop.is_set():
            try:
                st, obj, _ = http_json(self.url, "GET", "/api/remote/next?wait=25", token=self.entry["token"],
                                       fingerprint=self.pin, timeout=45)
                if st in (401, 403):
                    self.status = "removed by the controller: join again with a new invite"
                    return
                if st >= 400:
                    raise RuntimeError(f"HTTP {st}: {obj.get('error', '')}")
                self.status = "connected"
                delay = 2
                for item in obj.get("messages") or []:
                    threading.Thread(target=self._answer, args=(item,), daemon=True).start()
            except Exception as e:
                self.status = f"retrying: {type(e).__name__}: {e}"[:200]
                self.stop.wait(delay)
                delay = min(delay * 2, 60)

    def _answer(self, item):
        who = (self.entry.get("controller") or "controller") + " (calls in)"
        resp = self.agent.handle_rpc(item.get("message"), who,
                                     lambda tok: Upstream(self, str(tok)) if tok == item.get("id") else None)
        for _ in range(3):
            try:
                http_json(self.url, "POST", "/api/remote/reply", {"id": item["id"], "response": resp},
                          token=self.entry["token"], fingerprint=self.pin, timeout=60)
                return
            except Exception:
                time.sleep(2)


class Upstream:
    """Posts a reply in pieces to the controller while it is written, a few times a
    second, and learns from the answers whether the reader has gone away."""

    EVERY = 0.15

    def __init__(self, home: "CallHome", mid: str):
        self.home, self.mid = home, mid
        self.buf, self.last, self.stop, self.off = [], 0.0, False, False

    def send(self, piece: str) -> None:
        self.buf.append(piece)
        if time.time() - self.last >= self.EVERY:
            self.flush()

    def flush(self) -> None:
        self.last = time.time()
        if not self.buf or self.stop or self.off:
            return
        pieces, self.buf = self.buf, []
        try:
            st, obj, _ = http_json(self.home.url, "POST", "/api/remote/progress", {"id": self.mid, "pieces": pieces},
                                   token=self.home.entry["token"], fingerprint=self.home.pin, timeout=30)
            self.stop = bool(obj.get("cancel"))
            self.off = st >= 400      # a Studio without progress: the reply goes back whole
        except Exception:
            pass        # the whole reply still goes back at the end

    def cancelled(self) -> bool:
        return self.stop


# ------------------------------------------------------------------ the controller side

def _candidates(address: str) -> list[str]:
    a = address.strip().rstrip("/")
    if "://" in a:
        return [a]
    try:
        has_port = urllib.parse.urlsplit("//" + a).port is not None
    except ValueError:
        has_port = False
    hp = a if has_port else f"{a}:{DEFAULT_PORT}"
    return [f"https://{hp}", f"http://{hp}"]


def _probe(address: str) -> tuple[str, str | None]:
    """The host's base URL and how to trust it: None for a certificate the system
    trusts (a tunnel's), "" for the host's own certificate (pin what is seen) or
    for plain HTTP."""
    errors = []
    for base in _candidates(address):
        trusts = (None, "") if base.startswith("https://") else (None,)
        for trust in trusts:
            try:
                st, info, _ = http_json(base, "GET", "/info", fingerprint=trust, timeout=6)
            except Exception as e:
                errors.append(f"{base}: {type(e).__name__}: {e}")
                continue
            if st == 200 and info.get("nsr"):
                return base, ("" if trust is None and base.startswith("http://") else trust)
            errors.append(f"{base}: not a OmniCortex Remote host")
            break
    raise LinkError("no OmniCortex Remote host answered at " + address + ". Is the agent running there, and is "
                      "the link right? (" + "; ".join(errors[-2:])[:300] + ")")


def pair_direct(address: str, code: str, controller: str, fingerprint: str = "") -> dict:
    """Pair with a listening host. -> {url, token, fingerprint, host}. With a pinned
    `fingerprint` (from a pairing link) nothing else is trusted; otherwise the
    certificate seen now is what the proof binds and what gets pinned."""
    if fingerprint:
        base = address.strip().rstrip("/")
        if "://" not in base:
            base = "https://" + base
        trust = fingerprint
    else:
        base, trust = _probe(address)
    if trust == "" and base.startswith("https://"):
        st, _info, seen = http_json(base, "GET", "/info", fingerprint="", timeout=6)
        trust = seen
    bound = trust or ""
    st, out, _ = http_json(base, "POST", "/pair", {"proof": pair_proof(code, bound, "claim"),
                                                        "controller": controller},
                               fingerprint=trust if base.startswith("https://") else None, timeout=15)
    if st != 200:
        raise LinkError(out.get("error") or f"pairing failed (HTTP {st})")
    if not hmac.compare_digest(str(out.get("confirm", "")), pair_proof(code, bound, "accept")):
        raise LinkError("the host could not prove it showed this code; not paired")
    return {"url": base, "token": out["token"], "fingerprint": bound, "host": out.get("host", ""),
            "controller_id": out.get("controller_id")}


def controller_hosts(store: Store) -> list:
    return store.read("hosts.json", [])


def add_host(store: Store, target: str, code: str = "", name: str = "") -> dict:
    """CONTROLLER side: pair this computer with a host and keep its token here, for
    MCP clients that are not OmniCortex Studio (Studio keeps its own)."""
    fp = ""
    if target.startswith("nsr://") or "#c=" in target:
        inv = parse_invite(target)
        target, code, fp, name = inv["url"], code or inv["code"], inv["fingerprint"], name or inv["name"]
    if not norm_code(code):
        raise LinkError("the host's pairing code is needed (--code), or paste its pairing link")
    out = pair_direct(target, code, f"MCP on {socket.gethostname()}", fp)
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", name or out["host"] or "host")[:32].strip("-") or "host"
    rec = {"name": name, "url": out["url"], "token": out["token"], "fingerprint": out["fingerprint"],
           "host": out["host"], "added": time.time()}
    with store.lock:
        store.write("hosts.json", [h for h in controller_hosts(store) if h["name"] != name] + [rec])
    return rec


def bridge(store: Store, name: str) -> int:
    """CONTROLLER side: MCP over stdin/stdout, forwarded to a paired host. For MCP
    clients that only start local servers (Claude Desktop) or cannot pin a certificate."""
    rec = next((h for h in controller_hosts(store) if h["name"] == name), None)
    if rec is None:
        print(f"no host {name!r}: pair first with: omnicortex-remote add-host LINK", file=sys.stderr)
        return 2
    trust = (rec.get("fingerprint") or None) if rec["url"].startswith("https://") else None
    out_lock = threading.Lock()

    def forward(msg):
        try:
            st, resp, _ = http_json(rec["url"], "POST", "/mcp", msg, token=rec["token"], fingerprint=trust,
                                    timeout=900)
            if st == 202:
                return
            if st >= 400:
                raise LinkError(resp.get("error") or f"HTTP {st}")
        except Exception as e:
            if not isinstance(msg, dict) or msg.get("id") is None:
                return
            resp = {"jsonrpc": "2.0", "id": msg.get("id"),
                    "error": {"code": -32000, "message": f"host {name} unreachable: {e}"}}
        with out_lock:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

    pending = []
    for line in sys.stdin:
        if line.strip():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            t = threading.Thread(target=forward, args=(msg,), daemon=True)
            t.start()
            pending = [x for x in pending if x.is_alive()] + [t]
    for t in pending:                    # the client closed its end: finish what it asked
        t.join(900)
    return 0


def mcp_snippet(rec: dict) -> dict:
    here = str(Path(__file__).resolve())
    return {"mcpServers": {f"host-{rec['name']}": {"command": sys.executable, "args": [here, "bridge", rec["name"]]}}}


# ------------------------------------------------------------------ running

class Runtime:
    """The listeners, the LAN beacon and the call-home connections of one agent."""

    def __init__(self, agent: Agent, beacon_targets=None):
        self.agent = agent
        self.api = None
        self.beacon_stop = threading.Event()
        self.beacon_targets = beacon_targets or [("255.255.255.255", BEACON_PORT)]

    def listen(self, mode: str, bind: str = "", port: int = DEFAULT_PORT, tls: bool | None = None,
               allow_plaintext: bool = False, public_url: str = "") -> list[str]:
        ag = self.agent
        self.close_api()
        ag.fingerprint = ""
        if mode == "callhome":
            ag.link = {"mode": mode, "urls": []}
            return []
        if mode == "vpn" and not bind:
            vpn = [a["address"] for a in local_addresses() if a["kind"].startswith("vpn")]
            if not vpn:
                raise RuntimeError("no VPN address found: give the VPN address with --bind (Tailscale: tailscale ip -4)")
            bind = vpn[0]
        host = {"lan": "0.0.0.0", "vpn": bind, "ssh": "127.0.0.1", "tunnel": "127.0.0.1",
                "loopback": "127.0.0.1"}.get(mode)
        if host is None:
            raise RuntimeError(f"unknown link {mode}; one of {', '.join(MODES)}")
        if mode == "lan" and bind:
            host = bind
        use_tls = (mode == "lan") if tls is None else tls
        ctx = None
        if use_tls:
            names = ["localhost", "127.0.0.1"] + [a["address"] for a in local_addresses()]
            pair = ensure_cert(ag.store.home, names)
            if pair is None:
                if not allow_plaintext:
                    raise RuntimeError("TLS needs the cryptography package (pip install cryptography) or the openssl "
                                       "command. Without TLS use a VPN link, or pass --allow-plaintext on a network "
                                       "you trust.")
            else:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(*pair)
                ag.fingerprint = cert_fingerprint(pair[0])
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        Handler = type("H", (ApiHandler,), {"agent": ag, "throttle": Throttle()})
        Server = type("S", (ThreadingHTTPServer,), {"address_family": family, "daemon_threads": True})
        srv = Server((host, port), Handler)
        if ctx is not None:
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.api = srv
        scheme = "https" if ctx else "http"
        port = srv.server_address[1]
        if host in ("0.0.0.0", "::"):
            urls = [f"{scheme}://{a['address']}:{port}" for a in local_addresses() if ":" not in a["address"]]
        else:
            urls = [f"{scheme}://{'[' + host + ']' if ':' in host else host}:{port}"]
        if public_url:
            urls.insert(0, public_url.rstrip("/"))
        ag.link = {"mode": mode, "urls": urls, "port": port, "bind": host, "tls": bool(ctx)}
        if mode == "lan":
            self.beacon_stop = threading.Event()
            threading.Thread(target=self._beacon, args=(self.beacon_stop, scheme, port), daemon=True).start()
        return urls

    def close_api(self):
        self.beacon_stop.set()
        if self.api is not None:
            self.api.shutdown()
            self.api.server_close()
            self.api = None

    def relink(self, req: dict):
        ag = self.agent
        mode = req.get("mode") or ag.cfg.get("mode") or "loopback"
        urls = self.listen(mode, req.get("bind", ""), int(req.get("port") or ag.cfg.get("port") or DEFAULT_PORT),
                           public_url=req.get("public_url", ""))
        ag.cfg.update(mode=mode, bind=req.get("bind", ""), public_url=req.get("public_url", ""))
        ag.save_config()
        return urls

    def _beacon(self, stop, scheme, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not stop.is_set():
            if self.agent.pairing.current():
                msg = json.dumps({"nsr": 1, "name": self.agent.cfg["name"], "port": port, "scheme": scheme,
                                  "fp": self.agent.fingerprint}).encode()
                for t in self.beacon_targets:
                    try:
                        s.sendto(msg, t)
                    except OSError:
                        pass
            stop.wait(2)
        s.close()

    def start_callhome(self, entry: dict):
        old = self.agent.callhome.pop(entry["url"], None)
        if old:
            old.stop.set()
        t = CallHome(self.agent, entry)
        self.agent.callhome[entry["url"]] = t
        t.start()

    def console(self, port: int) -> str:
        key = secrets.token_urlsafe(18)
        Handler = type("C", (ConsoleHandler,), {"agent": self.agent, "key": key, "runtime": self})
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        Handler.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{Handler.port}/#k={key}"
        p = self.agent.store.home / "console-url.txt"
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(url + "\n")
        return url


def banner(agent: Agent, urls: list[str], console: str) -> str:
    ag = agent
    lines = ["", f"OmniCortex Remote {VERSION} on '{ag.cfg['name']}'", ROLE_LINE, ""]
    mode = ag.link["mode"]
    how = {"lan": "Controllers on this network find this host while pairing is open.",
           "vpn": "Controllers reach this host over the VPN at the address below.",
           "ssh": "Controllers reach this host through an SSH tunnel: ssh -N -L 47871:127.0.0.1:"
                  f"{ag.link.get('port', DEFAULT_PORT)} you@this-computer",
           "tunnel": "Point your tunnel (cloudflared, ngrok, ...) at the address below; give controllers its "
                     "public HTTPS address.",
           "callhome": "This host listens on nothing: it connects out to the controllers it joined.",
           "loopback": "Listening on this computer only."}[mode]
    lines.append(f"Link: {mode}. {how}")
    for u in urls:
        lines.append(f"  {u}")
    if ag.fingerprint:
        lines.append(f"Certificate: {short_fp(ag.fingerprint)}  (controllers show the same when paired)")
    p = ag.pairing.current()
    if p and mode != "callhome":
        lines += ["", f"Pairing code: {fmt_code(p['code'])}  (valid {max(1, round((p['expires'] - time.time()) / 60))} min)"]
        if ag.pair_link(p["code"]):
            lines.append(f"Pairing link: {ag.pair_link(p['code'])}")
        lines.append("Type the code (or paste the link) on the controller: OmniCortex Studio → Remote → Add a host.")
    n = len(ag.controllers.list())
    lines += ["", f"Controllers paired: {n}." + (" Revoke them in the console." if n else ""),
              f"Console (this computer only): {console}", ""]
    return "\n".join(lines)


def serve(a) -> int:
    ag = Agent(Path(a.home) if a.home else None)
    if a.name:
        ag.cfg["name"] = a.name
    mode = a.mode or ag.cfg.get("mode") or "loopback"
    for url in a.backend or []:
        kind, _, u = url.partition("=") if "=" in url else ("openai", "", url)
        ag.cfg["backends"] = [b for b in ag.cfg["backends"] if b.get("url") != u] + [{"kind": kind, "url": u}]
    if a.mode or a.name or a.backend or a.bind:
        ag.cfg.update(mode=mode, bind=a.bind or (ag.cfg.get("bind") if not a.mode else ""))
        ag.save_config()
    rt = Runtime(ag)
    ag.guard = ThermalGuard(ag)
    ag.guard.start()
    ag.presence = PresenceGuard(ag)
    ag.presence.start()
    try:
        urls = rt.listen(mode, a.bind or ag.cfg.get("bind", ""), a.port or ag.cfg.get("port") or DEFAULT_PORT,
                         tls=a.tls, allow_plaintext=a.allow_plaintext,
                         public_url=a.public_url or ag.cfg.get("public_url", ""))
    except (RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for entry in ag.store.read("callhome.json", []):
        rt.start_callhome(entry)
    if mode != "callhome" and not ag.controllers.list() and not ag.pairing.current():
        ag.pairing.open()
    console = rt.console(a.console_port if a.console_port is not None else (ag.link.get("port") or DEFAULT_PORT) + 1)
    print(banner(ag, urls, console), flush=True)
    if a.open:
        try:
            webbrowser.open(console)
        except Exception:
            pass
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("stopped")
        return 0


def stdio(a) -> int:
    """MCP over stdin/stdout for an MCP client on this computer (it starts the agent)."""
    ag = Agent(Path(a.home) if a.home else None)
    out_lock = threading.Lock()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = ag.handle_rpc(msg, "local MCP client")
        if resp is not None:
            with out_lock:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="omnicortex-remote", description=ROLE_LINE,
                                epilog="Run with no command to start with the saved settings and open the console.")
    p.add_argument("--home", help="the agent's folder (default: per-user app data; $OCR_HOME)")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the host agent")
    s.add_argument("--mode", choices=MODES, help="how controllers reach this host (saved for next time)")
    s.add_argument("--bind", default="", help="vpn: the VPN address (found automatically for Tailscale)")
    s.add_argument("--port", type=int, default=0, help=f"API port (default {DEFAULT_PORT})")
    s.add_argument("--console-port", type=int, default=None, help="console port, this computer only (default: API+1)")
    s.add_argument("--public-url", default="", help="tunnel: the public HTTPS address your tunnel gives this host")
    s.add_argument("--tls", action=argparse.BooleanOptionalAction, default=None,
                   help="serve HTTPS with a pinned self-signed certificate (default: on for lan)")
    s.add_argument("--allow-plaintext", action="store_true", help="lan without TLS (only on a network you trust)")
    s.add_argument("--backend", action="append", help="an LLM server to use: URL or KIND=URL "
                   "(kinds: lmstudio, ollama, llama-server, openai); the usual ports are found automatically")
    s.add_argument("--name", help="what controllers call this host")
    s.add_argument("--open", action="store_true", help="open the console in the browser")
    c = sub.add_parser("connect", help="join a controller that this host connects out to (call home)")
    c.add_argument("invite", help="the invite from Studio's Remote page (nsr1:...)")
    sub.add_parser("pair", help="make a new pairing code (the running agent shows it)")
    k = sub.add_parser("controllers", help="who may control this host")
    k.add_argument("--revoke", metavar="ID")
    sub.add_parser("stdio", help="MCP over stdin/stdout for an MCP client on this computer")
    sub.add_parser("console", help="open the running agent's console")
    h = sub.add_parser("add-host", help="CONTROLLER: pair this computer with a host, for an MCP client "
                                        "other than Studio (Claude Desktop, Cline, Cursor...)")
    h.add_argument("target", help="the host's pairing link (nsr://...) or its address")
    h.add_argument("--code", default="", help="the code the host shows (not needed with a pairing link)")
    h.add_argument("--name", default="", help="what to call the host")
    sub.add_parser("hosts", help="CONTROLLER: the hosts this computer controls")
    rh = sub.add_parser("remove-host", help="CONTROLLER: forget a host")
    rh.add_argument("host")
    br = sub.add_parser("bridge", help="CONTROLLER: MCP over stdin/stdout to a paired host (what an MCP client runs)")
    br.add_argument("host")
    argv = list(sys.argv[1:] if argv is None else argv)
    a = p.parse_args(argv)
    if a.cmd is None:
        a = p.parse_args(argv + ["serve", "--open"])
    if a.cmd == "serve":
        return serve(a)
    if a.cmd == "stdio":
        return stdio(a)
    ag = Agent(Path(a.home) if a.home else None)
    if a.cmd == "pair":
        r = ag.pairing.open()
        print(f"Pairing code: {r['code']} (valid {PAIR_TTL // 60} min). Type it on the controller: "
              "OmniCortex Studio → Remote → Add a host.")
        return 0
    if a.cmd == "controllers":
        if a.revoke:
            print("revoked" if ag.controllers.revoke(a.revoke) else "no such controller")
            return 0
        rows = ag.controllers.list()
        for r in rows:
            seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["last_seen"])) if r.get("last_seen") else "never"
            print(f"{r['id']}  {r['name']:<24} via {r['via']:<9} last seen {seen}")
        if not rows:
            print("No controllers. Pair one with the code the running agent shows.")
        return 0
    if a.cmd == "connect":
        try:
            entry = claim_invite(ag, a.invite)
        except (ValueError, RuntimeError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"Joined {entry['controller'] or entry['url']} as '{entry['host_name']}'. This host now connects out "
              "to it whenever the agent runs.")
        ag.cfg["mode"] = ag.cfg.get("mode") or "callhome"
        ag.save_config()
        return serve(p.parse_args(([] if not a.home else ["--home", a.home]) + ["serve"]))
    if a.cmd == "add-host":
        try:
            rec = add_host(ag.store, a.target, a.code, a.name)
        except (LinkError, ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"This computer (CONTROLLER) now controls the host '{rec['name']}' at {rec['url']}"
              + (f", certificate {short_fp(rec['fingerprint'])} pinned" if rec["fingerprint"] else "") + ".")
        print("Add this to your MCP client's configuration (Claude Desktop: claude_desktop_config.json; "
              "Cline and Cursor: their MCP settings):")
        print(json.dumps(mcp_snippet(rec), indent=2))
        return 0
    if a.cmd == "hosts":
        rows = controller_hosts(ag.store)
        for r in rows:
            print(f"{r['name']:<20} {r['url']}" + (f"  certificate {short_fp(r['fingerprint'])}" if r.get("fingerprint")
                                                    else ""))
        if not rows:
            print("This computer controls no hosts. Pair one with: omnicortex-remote add-host LINK")
        return 0
    if a.cmd == "remove-host":
        ag.store.write("hosts.json", [h for h in controller_hosts(ag.store) if h["name"] != a.host])
        return 0
    if a.cmd == "bridge":
        return bridge(ag.store, a.host)
    if a.cmd == "console":
        try:
            url = (ag.store.home / "console-url.txt").read_text().strip()
        except FileNotFoundError:
            print("the agent is not running; start it with: omnicortex-remote", file=sys.stderr)
            return 1
        webbrowser.open(url)
        print(url)
        return 0
    return 0


CONSOLE_PAGE = r"""<!DOCTYPE html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Remote</title><link rel="icon" href="data:,">
<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--ok:#2f7d4f;--no:#b23c2e;--acc:#2c5f8a;
  --soft:#eef4f9;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;--line:#2a2f37;
  --ok:#5fcf8f;--no:#ff8b7e;--acc:#6ea8e0;--soft:#1d2733;--code:#20242b;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:860px;margin:0 auto;padding:1rem}
h1{font-size:18px;margin:.2rem 0}h2{font-size:15px;margin:0 0 .5rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem;margin:.8rem 0}
.roles{display:grid;grid-template-columns:1fr auto 1fr;gap:.6rem;align-items:center}
.role{border:1px solid var(--line);border-radius:10px;padding:.6rem .8rem;background:var(--soft)}
.role.me{border-color:var(--acc);border-width:2px}.role b{font-size:15px}.arrow{color:var(--mut);font-size:20px}
.mut{color:var(--mut);font-size:13px}.ok{color:var(--ok)}.err{color:var(--no)}
code,.code{font:13px ui-monospace,monospace;background:var(--code);padding:.1rem .35rem;border-radius:4px;word-break:break-all}
.big{font:600 26px ui-monospace,monospace;letter-spacing:.08em}
button{font:inherit;padding:.3rem .7rem;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
input,select{font:inherit;padding:.3rem .45rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.3rem .4rem;border-top:1px solid var(--line);font-size:13px;vertical-align:top}
.modes{display:grid;gap:.4rem}.modes label{display:flex;gap:.5rem;align-items:flex-start;border:1px solid var(--line);border-radius:8px;padding:.45rem .6rem;cursor:pointer}
.modes label.sel{border-color:var(--acc);background:var(--soft)}
@media(max-width:640px){.roles{grid-template-columns:1fr}.arrow{transform:rotate(90deg);justify-self:center}}
</style>
<main>
<h1>OmniCortex Remote</h1>
<div class="card roles">
 <div class="role"><b>Controllers</b><div class="mut">OmniCortex Studio, Claude Desktop, Cline… on other computers. They send instructions.</div><div id="ctlCount" class="mut"></div></div>
 <div class="arrow">→</div>
 <div class="role me"><b>This computer: HOST <span id="hname"></span></b><div class="mut">Runs the models. Does what paired controllers ask, within what you allow below.</div></div>
</div>
<div id="err" class="err"></div>
<div class="card"><h2>1. How do controllers reach this computer?</h2><div id="modes" class="modes"></div>
 <div id="modeExtra" style="margin-top:.5rem"></div>
 <div style="margin-top:.5rem"><button class="pri" id="apply">Use this link</button> <span id="linkNow" class="mut"></span></div></div>
<div class="card" id="pairCard"><h2>2. Pair a controller</h2><div id="pair"></div></div>
<div class="card"><h2>Controllers allowed to control this host</h2><div id="ctls"></div></div>
<div class="card"><h2>When I use this computer</h2>
<label style="display:block"><input type="checkbox" id="yon"> Give it back to me: unload the models and take no new work while I use it</label>
<div id="ybox" style="margin:.4rem 0 0 1.4rem">
<label>Counts as in use when the keyboard or mouse was used in the last <input id="yidle" type="number" min="5" style="width:5rem"> s</label>
<label style="display:block"><input type="checkbox" id="ypad"> or a game controller is used</label>
<label style="display:block"><input type="checkbox" id="yvr"> or a VR session runs (SteamVR, Oculus, Monado…)</label>
<label>Take work again after <input id="ycool" type="number" min="0" style="width:5rem"> s of quiet</label>
<label style="display:block"><input type="checkbox" id="yjobs"> also stop OmniCortex jobs running here</label>
<div class="note" id="ystate" style="margin-top:.3rem"></div>
<button id="yfree" style="margin-top:.3rem">I am done: take work again now</button></div></div>
<div class="card"><h2>What controllers may do</h2><div id="allow"></div>
<div style="margin-top:.6rem"><label>OmniCortex folder for jobs <input id="nsdir" placeholder="found automatically when the agent runs from one" style="width:100%"></label>
<button id="nsdirsave" style="margin-top:.3rem">Save</button> <span id="nsdirmsg" class="note"></span></div></div>
<div class="card"><h2>GPUs and thermal guard</h2><div id="gpus" class="mut"></div>
<label style="display:block;margin-top:.4rem"><input type="checkbox" id="thOn"> Stop taking work when a GPU reaches <input id="thLim" type="number" min="50" max="110" style="width:4.5em"> °C (the hottest sensor: junction, edge or memory), and resume <input id="thMar" type="number" min="2" max="30" style="width:3.5em"> °C below</label>
<label style="display:block"><input type="checkbox" id="thUnl"> Unload models when it trips (LM Studio and Ollama; llama-server and vLLM stay up but get no work)</label>
<div class="mut">GPUs slow themselves down near their own limit; this is an earlier stop you choose. Values that cannot be read show as ?, and never trip it.</div></div>
<div class="card"><h2>LLM servers on this computer</h2><div id="bks"></div>
 <div style="margin-top:.5rem"><select id="bkind"><option value="openai">OpenAI-compatible</option><option value="lmstudio">LM Studio</option><option value="ollama">Ollama</option><option value="llama-server">llama-server</option><option value="vllm">vLLM</option></select>
 <input id="burl" placeholder="http://127.0.0.1:1234" size="28"> <button id="badd">Add</button></div>
 <div class="mut" style="margin-top:.3rem">LM Studio (port 1234), Ollama (11434), llama-server (8080) and vLLM (8000) are found by themselves when running.</div></div>
<div class="card"><h2>Recent activity</h2><div id="act" class="mut">nothing yet</div></div>
<div class="mut">Settings folder: <code id="home"></code></div>
</main>
<script>
const key=(location.hash.match(/k=([^&]+)/)||[])[1]||sessionStorage.getItem('nsrk')||'';
try{ if(key) sessionStorage.setItem('nsrk',key); history.replaceState(null,'',location.pathname); }catch(e){}
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const when=t=>t?new Date(t*1000).toLocaleString():'never';
const MODES=[
 ['lan','Same network (home or office)','Encrypted with this computer\'s own certificate. Controllers on the network see this host while pairing is open.'],
 ['vpn','A VPN: Tailscale, Headscale, ZeroTier, WireGuard, NetBird, Nebula…','Listens on the VPN address only; the VPN encrypts. Works across the internet.'],
 ['ssh','SSH tunnel','Listens on this computer only. The controller connects with ssh -L; needs an SSH server here.'],
 ['tunnel','Public tunnel: Cloudflare Tunnel, ngrok…','Listens on this computer only; the tunnel gives it a public HTTPS address.'],
 ['callhome','This computer calls the controller','Listens on nothing. Use when this computer is behind a router or firewall you cannot open. Needs an invite from the controller.']];
let S=null, pick=null;
async function api(p,b){ const r=await fetch(p,{method:b?'POST':'GET',headers:{'Content-Type':'application/json','X-Console-Key':key},body:b?JSON.stringify(b):undefined});
  const j=await r.json(); if(!r.ok){ $('#err').textContent=j.error||r.status; throw new Error(j.error); } $('#err').textContent=''; return j; }
const ysend=async(extra)=>{ try{ S=await api('/api/yield',{enabled:$('#yon').checked,idle_under:+$('#yidle').value||60,gamepad:$('#ypad').checked,
  vr:$('#yvr').checked,cooldown:+$('#ycool').value||0,stop_jobs:$('#yjobs').checked,...(extra||{})}); render(); }catch(e){} };
['#yon','#ypad','#yvr','#yjobs','#yidle','#ycool'].forEach(s=>$(s).onchange=()=>ysend());
$('#yfree').onclick=()=>ysend({free_now:true});
$('#nsdirsave').onclick=async()=>{ try{ S=await api('/api/omnicortex-dir',{dir:$('#nsdir').value.trim()}); render(); }catch(e){} };
function extra(){ const m=pick; let h='';
  if(m==='vpn') h=`VPN address of this computer: <input id="xbind" value="${esc(S.link.mode==='vpn'?S.link.bind:(S.addresses.find(a=>a.kind.startsWith('vpn'))||{}).address||'')}" placeholder="100.x.y.z"> <span class="mut">Tailscale: <code>tailscale ip -4</code></span>`;
  if(m==='ssh') h=`<div class="mut">On the controller: <code>ssh -N -L 47871:127.0.0.1:${S.link.port||47871} you@${esc((S.addresses[0]||{}).address||'this-computer')}</code>, then add the host at <code>http://127.0.0.1:47871</code>. OmniCortex Studio can run the tunnel for you.</div>`;
  if(m==='tunnel') h=`Public address the tunnel gives: <input id="xpub" size="34" placeholder="https://name.trycloudflare.com" value="${esc(S.link.mode==='tunnel'?(S.link.urls[0]||''):'')}"><div class="mut">For example <code>cloudflared tunnel --url http://127.0.0.1:${S.link.port||47871}</code> or <code>ngrok http ${S.link.port||47871}</code>.</div>`;
  if(m==='callhome') h=`Invite from the controller: <input id="xinv" size="44" placeholder="nsr1:…"> <button id="join">Join</button><div class="mut">In OmniCortex Studio: Remote → Add a host → "The host calls this Studio".</div>`+
    (S.callhome.length?'<table>'+S.callhome.map(c=>`<tr><td>${esc(c.controller||c.url)}</td><td>${esc(c.status)}</td><td><button data-ch="${esc(c.url)}">Leave</button></td></tr>`).join('')+'</table>':'');
  $('#modeExtra').innerHTML=h;
  if($('#join')) $('#join').onclick=async()=>{ S=await api('/api/callhome',{invite:$('#xinv').value.trim()}); render(); };
  document.querySelectorAll('[data-ch]').forEach(b=>b.onclick=async()=>{ S=await api('/api/callhome/remove',{url:b.dataset.ch}); render(); });
}
function render(){ const s=S; $('#hname').textContent='“'+s.name+'”'; $('#home').textContent=s.home;
  if(pick===null) pick=s.link.mode==='loopback'?'lan':s.link.mode;
  $('#modes').innerHTML=MODES.map(([k,t,d])=>`<label class="${k===pick?'sel':''}"><input type="radio" name="m" value="${k}" ${k===pick?'checked':''}><span><b>${t}</b>${k===s.link.mode?' <span class="ok">· in use</span>':''}<br><span class="mut">${d}</span></span></label>`).join('');
  document.querySelectorAll('[name=m]').forEach(r=>r.onchange=()=>{ pick=r.value; render(); });
  extra();
  $('#linkNow').innerHTML=s.link.urls&&s.link.urls.length?'Listening at '+s.link.urls.map(u=>'<code>'+esc(u)+'</code>').join(' ')+(s.short_fingerprint?` · certificate <b>${esc(s.short_fingerprint)}</b>`:''):(s.link.mode==='callhome'?'Not listening; calling out.':'');
  const p=s.pairing;
  $('#pairCard').style.display=s.link.mode==='callhome'?'none':'';
  $('#pair').innerHTML=p?`<div>Code: <span class="big">${esc(p.code)}</span> <span class="mut">until ${new Date(p.expires*1000).toLocaleTimeString()}, single use</span></div>
    ${p.link?`<div style="margin:.3rem 0">or paste this link on the controller: <span class="code" id="plink">${esc(p.link)}</span> <button id="cp">Copy</button></div>`:''}
    <div class="mut">On the controller: OmniCortex Studio → <b>Remote</b> → <b>Add a host</b>${s.link.mode==='lan'?' (it lists this host by itself)':''}. Other MCP clients: see the README.</div>
    <button id="pclose" style="margin-top:.4rem">Stop pairing</button>`:
    `<button class="pri" id="popen">Show a pairing code</button> <span class="mut">valid 10 minutes, for one controller</span>`;
  if($('#popen')) $('#popen').onclick=async()=>{ S=await api('/api/pair/open',{}); render(); };
  if($('#pclose')) $('#pclose').onclick=async()=>{ S=await api('/api/pair/close',{}); render(); };
  if($('#cp')) $('#cp').onclick=()=>navigator.clipboard.writeText(p.link);
  $('#ctlCount').textContent=s.controllers.length+' paired';
  $('#ctls').innerHTML=s.controllers.length?'<table><tr><th>controller</th><th>paired</th><th>last seen</th><th></th></tr>'+s.controllers.map(c=>`<tr><td>${esc(c.name)} <span class="mut">via ${esc(c.via)}</span></td><td>${when(c.created)}</td><td>${when(c.last_seen)}</td><td><button data-rv="${esc(c.id)}">Revoke</button></td></tr>`).join('')+'</table>':'<span class="mut">None yet.</span>';
  document.querySelectorAll('[data-rv]').forEach(b=>b.onclick=async()=>{ if(confirm('Revoke this controller? It loses access at once.')){ S=await api('/api/controllers/revoke',{id:b.dataset.rv}); render(); } });
  const T=s.telemetry||{gpus:[],thermal:{}}, th=T.thermal||{}, f=v=>v==null?'?':Math.round(v);
  $('#gpus').innerHTML=(th.tripped?`<div class="bad" style="color:#c0392b"><b>Paused: ${esc(th.tripped.gpu)} reached ${f(th.tripped.temp)}°C</b></div>`:'')+(T.gpus.length?T.gpus.map(g=>g.error?esc(g.error):`<div><b>${esc(g.name)}</b> · edge ${f(g.temp_c)}°C · junction ${f(g.junction_c)}°C · memory ${f(g.mem_c)}°C · VRAM ${g.vram_used_mb==null?'?':(g.vram_used_mb/1024).toFixed(1)}/${g.vram_total_mb==null?'?':(g.vram_total_mb/1024).toFixed(1)} GB · load ${f(g.util)}%${g.power_w!=null?' · '+f(g.power_w)+' W':''} <span class="mut">(${esc(g.source)})</span></div>`).join(''):'No GPU sensors found (nvidia-smi, rocm-smi or Linux amdgpu sensors).');
  if(document.activeElement?.id?.startsWith('th')!==true){ $('#thOn').checked=!!th.enabled; $('#thLim').value=th.limit_c??95; $('#thMar').value=th.margin_c??10; $('#thUnl').checked=th.unload!==false; }
  ['thOn','thLim','thMar','thUnl'].forEach(id=>$('#'+id).onchange=async()=>{ S=await api('/api/thermal',{enabled:$('#thOn').checked,limit_c:+$('#thLim').value,margin_c:+$('#thMar').value,unload:$('#thUnl').checked}); render(); });
  if(document.activeElement!==$('#nsdir')) $('#nsdir').value=s.omnicortex_dir_set||'';
  { const y=s.yield||{}; if(!['yidle','ycool'].includes(document.activeElement&&document.activeElement.id)){ $('#yidle').value=y.idle_under??60; $('#ycool').value=y.cooldown??300; }
    $('#yon').checked=!!y.enabled; $('#ypad').checked=y.gamepad!==false; $('#yvr').checked=y.vr!==false; $('#yjobs').checked=!!y.stop_jobs; $('#ybox').style.opacity=y.enabled?1:.55;
    const p=s.presence||{}; const known=p.idle_s!=null||p.gamepad!=null||p.vr!=null;
    $('#ystate').textContent=(s.busy?'Busy now: '+s.busy.why+'. ':'Taking work. ')+(known?`Last input ${p.idle_s!=null?Math.round(p.idle_s)+' s ago':'unknown'}${p.vr?', VR running':''}${p.gamepad?', controller in use':''}.`:'This computer cannot tell when it is in use (no desktop session visible to the agent).');
    $('#yfree').hidden=!(s.busy&&s.busy.source==='owner'); }
  $('#nsdirmsg').textContent=s.omnicortex_dir?('jobs run from '+s.omnicortex_dir):'no OmniCortex folder found';
  $('#allow').innerHTML=Object.keys(s.allow).map(k=>`<label style="display:block"><input type="checkbox" data-al="${k}" ${s.allow[k]?'checked':''}> ${esc(s.allow_text[k])}</label>`).join('');
  document.querySelectorAll('[data-al]').forEach(c=>c.onchange=async()=>{ S=await api('/api/allow',{allow:{[c.dataset.al]:c.checked}}); render(); });
  $('#bks').innerHTML=s.backends.length?'<table>'+s.backends.map(b=>`<tr><td>${esc(b.kind)}</td><td><code>${esc(b.url)}</code></td><td class="ok">running</td><td>${b.source==='configured'?`<button data-rb="${esc(b.url)}">Remove</button>`:''}</td></tr>`).join('')+'</table>':'<span class="err">No LLM server found. Start LM Studio\'s server (Developer → Start server), Ollama, or llama-server.</span>';
  document.querySelectorAll('[data-rb]').forEach(b=>b.onclick=async()=>{ S=await api('/api/backends',{url:b.dataset.rb,remove:true}); render(); });
  $('#act').innerHTML=s.activity.length?'<table>'+s.activity.map(x=>`<tr><td>${new Date(x.t*1000).toLocaleTimeString()}</td><td>${esc(x.who)}</td><td class="${x.ok?'':'err'}">${esc(x.what)}${x.n>1?' <span class="mut">×'+x.n+'</span>':''}</td><td>${esc(x.detail)}</td></tr>`).join('')+'</table>':'nothing yet';
}
$('#apply').onclick=async()=>{ const b={mode:pick}; if(pick==='vpn') b.bind=$('#xbind').value.trim(); if(pick==='tunnel') b.public_url=$('#xpub').value.trim();
  if(pick==='callhome'&&!S.callhome.length&&!confirm('Stop listening? Controllers then reach this host only through invites it joined.')) return;
  S=await api('/api/link',b); render(); };
$('#badd').onclick=async()=>{ S=await api('/api/backends',{url:$('#burl').value.trim(),kind:$('#bkind').value}); $('#burl').value=''; render(); };
async function tick(){ try{ S=await api('/api/state'); render(); }catch(e){} }
if(!key) $('#err').textContent='Open the console from the link the agent printed (or run: omnicortex-remote console).';
else { tick(); setInterval(()=>{ if(!document.querySelector('input:focus')) tick(); },4000); }
</script>"""


if __name__ == "__main__":
    raise SystemExit(main())
