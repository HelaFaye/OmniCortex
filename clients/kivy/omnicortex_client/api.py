"""
OmniCortex Studio from a phone or desktop app: pairing, models, streaming chat, the reply checks and the review
lookup, over Studio's own API (docs/MOBILE_CLIENTS.md). Plain Python, no Kivy, so it is tested on its own.

Pairing: Studio → Connect → Pair a device gives a link like

    https://studio.example:7870/pair#c=ABCD-EFGH-IJKL&fp=<sha256 of Studio's certificate>&a=p

`pair(link, name)` claims the code for a device token (revocable, this device only; never the owner's token).
When the link carries `fp`, every later connection checks that Studio's certificate has exactly that SHA-256
fingerprint, so a self-signed Studio is trusted without trusting any certificate authority, and a different
certificate is refused.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path


class StudioError(RuntimeError):
    pass


@dataclass
class Pairing:
    base: str                       # https://host:port
    token: str = ""
    fingerprint: str = ""           # Studio's certificate SHA-256, hex, lowercase, no colons
    name: str = "device"
    expires: float | None = None    # None: persistent until revoked
    model: str = "auto"
    extra: dict = field(default_factory=dict)


CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # Studio's pairing codes (scripts/oc_pairing.py): no 0/O, 1/I/L


def parse_pair_link(link: str) -> dict:
    """{base, code, fingerprint, access} from a Studio pairing link (or 'host:port#c=...')."""
    link = link.strip()
    if "://" not in link:
        link = "https://" + link
    u = urllib.parse.urlsplit(link)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise StudioError("not a Studio pairing link")
    frag = urllib.parse.parse_qs(u.fragment)
    code = "".join(c for c in (frag.get("c") or [""])[0].upper() if c in CODE_ALPHABET)
    if not code:
        raise StudioError("the link has no pairing code (c=...)")
    fp = (frag.get("fp") or [""])[0].replace(":", "").lower()
    if fp and not re.fullmatch(r"[0-9a-f]{64}", fp):
        raise StudioError("bad certificate fingerprint in the link")
    if u.scheme == "https" and not fp:
        raise StudioError("an https pairing link must carry Studio's certificate fingerprint (fp=)")
    return {"base": f"{u.scheme}://{u.netloc}", "code": code, "fingerprint": fp,
            "access": (frag.get("a") or ["p"])[0]}


def _norm_fp(fp: str) -> str:
    return (fp or "").replace(":", "").strip().lower()


class Studio:
    """A connection to one Studio. Pins the certificate when a fingerprint is known."""

    def __init__(self, pairing: Pairing, timeout: float = 120):
        self.p = pairing
        self.timeout = timeout
        u = urllib.parse.urlsplit(pairing.base)
        self.https = u.scheme == "https"
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if self.https else 80)
        # Plain HTTP to another computer: the device token crosses the network unencrypted. Fine inside a VPN
        # (WireGuard, Tailscale), as scripts/oc_pairing.py allows; the app shows this warning.
        self.warning = ("plain HTTP to another computer: the device token is unencrypted on the way, which is fine "
                        "only inside a VPN" if not self.https and self.host not in ("127.0.0.1", "localhost", "::1")
                        else "")

    # -------------------------------------------------------------- transport
    def _conn(self):
        if not self.https:
            return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        ctx = ssl.create_default_context()
        if self.p.fingerprint:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE          # replaced by the exact fingerprint check below
        c = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=ctx)
        c.connect()
        if self.p.fingerprint:
            der = c.sock.getpeercert(binary_form=True)
            got = hashlib.sha256(der or b"").hexdigest()
            if got != _norm_fp(self.p.fingerprint):
                c.close()
                raise StudioError("Studio's certificate does not match the one it was paired with "
                                  f"(expected {self.p.fingerprint[:16]}…, got {got[:16]}…). Pair again only if "
                                  "you know why it changed.")
        return c

    def _headers(self, body: bool) -> dict:
        h = {"Accept": "application/json"}
        if body:
            h["Content-Type"] = "application/json"
        if self.p.token:
            h["Authorization"] = f"Bearer {self.p.token}"
        return h

    def call(self, method: str, path: str, body=None):
        try:
            c = self._conn()
            c.request(method, path, body=None if body is None else json.dumps(body),
                      headers=self._headers(body is not None))
            r = c.getresponse()
            raw = r.read()
        except (OSError, socket.timeout, http.client.HTTPException) as e:
            raise StudioError(f"cannot reach Studio at {self.p.base}: {e}") from e
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {"text": raw.decode(errors="replace")}
        if r.status >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            err = err.get("message") if isinstance(err, dict) else err
            raise StudioError(f"Studio said {r.status}: {err or raw[:200].decode(errors='replace')}")
        return data

    # -------------------------------------------------------------- pairing
    @classmethod
    def pair(cls, link: str, name: str = "device") -> Pairing:
        info = parse_pair_link(link)
        p = Pairing(base=info["base"], fingerprint=info["fingerprint"], name=name)
        r = cls(p).call("POST", "/api/pair/claim", {"code": info["code"], "name": name})
        if r.get("fingerprint") and info["fingerprint"] and _norm_fp(r["fingerprint"]) != info["fingerprint"]:
            raise StudioError("Studio reported a different certificate than the pairing link")
        p.token = r["token"]
        p.expires = r.get("expires")
        p.fingerprint = info["fingerprint"] or _norm_fp(r.get("fingerprint") or "")
        return p

    # -------------------------------------------------------------- API
    def models(self) -> list[str]:
        return [m["id"] for m in self.call("GET", "/v1/models").get("data", [])]

    def chat(self, messages: list[dict], model: str | None = None, **opts) -> str:
        r = self.call("POST", "/v1/chat/completions",
                      {"model": model or self.p.model, "messages": messages, "stream": False, **opts})
        return r["choices"][0]["message"].get("content") or ""

    def chat_stream(self, messages: list[dict], model: str | None = None, **opts):
        """Yields the reply's text pieces as Studio streams them (OpenAI server-sent events)."""
        body = {"model": model or self.p.model, "messages": messages, "stream": True, **opts}
        try:
            c = self._conn()
            c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=self._headers(True))
            r = c.getresponse()
        except (OSError, http.client.HTTPException) as e:
            raise StudioError(f"cannot reach Studio at {self.p.base}: {e}") from e
        if r.status >= 400:
            raise StudioError(f"Studio said {r.status}: {r.read()[:200].decode(errors='replace')}")
        buf = b""
        while True:
            chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def check(self, messages: list[dict], text: str, model: str | None = None, threshold: float = 0.5) -> dict:
        """Per-token hallucination risk from the model's H-neuron classifier: {pieces, prob, flagged, max, mean}."""
        return self.call("POST", "/api/trace", {"model": model or self.p.model, "messages": messages,
                                                "text": text, "threshold": threshold})

    def truth(self, messages: list[dict], text: str, alternatives=(), model: str | None = None) -> dict:
        """Does the model's own activations agree with its reply? (docs/TRUTH_PROBE.md)"""
        return self.call("POST", "/api/truth", {"model": model or self.p.model, "messages": messages,
                                                "text": text, "alternatives": list(alternatives)})

    def lookup(self, model: str) -> dict:
        """The unified review: accuracy, speed per hardware, features, for a model id or Hugging Face repo."""
        return self.call("GET", "/api/lookup?" + urllib.parse.urlencode({"model": model}))


# ------------------------------------------------------------------ the saved pairing

def config_path(data_dir: str | None = None) -> Path:
    base = Path(data_dir) if data_dir else Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "omnicortex"
    return base / "client.json"


def save(p: Pairing, data_dir: str | None = None) -> Path:
    path = config_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)     # the token: this user only
    with os.fdopen(fd, "w") as f:
        json.dump(asdict(p), f, indent=1)
    os.replace(tmp, path)
    return path


def load(data_dir: str | None = None) -> Pairing | None:
    path = config_path(data_dir)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return Pairing(**{k: v for k, v in d.items() if k in Pairing.__dataclass_fields__})


def forget(data_dir: str | None = None) -> None:
    config_path(data_dir).unlink(missing_ok=True)
