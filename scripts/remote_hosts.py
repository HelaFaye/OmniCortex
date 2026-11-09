#!/usr/bin/env python3
"""
The controller side of OmniCortex Remote: Studio driving LLM hosts that do not
run OmniCortex (a Windows box with LM Studio, a server with Ollama...).

Roles: this computer (Studio) is the CONTROLLER; the computer that runs the
models and the agent (remote/omnicortex_remote) is the HOST. Everything Studio
does on a host is an MCP tool call to the agent; chat also goes through the
agent's OpenAI-compatible pass-through when the host is reachable directly.

Link methods, as Studio's Remote page offers them:
  lan       same network: Studio hears the host's beacon while its pairing is open
  vpn       Tailscale, ZeroTier, WireGuard...: the host's VPN address
  ssh       Studio keeps an SSH tunnel to the host (your SSH keys; no password prompts)
  tunnel    the host's public HTTPS address from Cloudflare Tunnel, ngrok...
  callhome  the host connects out to this Studio and waits for instructions

The pairing protocol and the pinned-certificate client are the agent's own
(remote/omnicortex_remote/agent.py), so both sides speak exactly the same thing.
"""
from __future__ import annotations

import collections
import hashlib
import hmac
import io
import json
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_PKG = ROOT / "remote"
import sys  # noqa: E402

sys.path.insert(0, str(REMOTE_PKG))
from omnicortex_remote import agent as nsr  # noqa: E402

METHODS = ("lan", "vpn", "ssh", "tunnel", "callhome")
CHUNK = 16 * 1024 * 1024


RemoteError = nsr.LinkError


# ------------------------------------------------------------------ pairing with a host that listens

pair_direct = nsr.pair_direct     # the same code the agent's own add-host command uses


def discover(seconds: float = 3.0, port: int = nsr.BEACON_PORT) -> list[dict]:
    """Hosts on this network with pairing open (their UDP beacon)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    try:
        s.bind(("", port))
    except OSError as e:
        raise RemoteError(f"cannot listen for hosts on UDP {port}: {e}") from None
    found: dict = {}
    end = time.time() + seconds
    try:
        while time.time() < end:
            s.settimeout(max(0.05, end - time.time()))
            try:
                data, (ip, _p) = s.recvfrom(4096)
            except socket.timeout:
                break
            try:
                b = json.loads(data)
            except ValueError:
                continue
            if b.get("nsr") != 1:
                continue
            url = f"{b.get('scheme', 'https')}://{ip}:{int(b.get('port', nsr.DEFAULT_PORT))}"
            found[url] = {"name": str(b.get("name", ip))[:40], "url": url, "fingerprint": b.get("fp", ""),
                          "short_fingerprint": nsr.short_fp(b.get("fp", ""))}
    finally:
        s.close()
    return list(found.values())


# ------------------------------------------------------------------ SSH tunnels

class Tunnels:
    """One `ssh -N -L` per host that is reached over SSH. Your SSH keys and config
    are used; Studio never asks for or stores a password."""

    def __init__(self):
        self.procs: dict = {}
        self.lock = threading.Lock()

    @staticmethod
    def free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    @staticmethod
    def _open(port: int) -> bool:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return True
        except OSError:
            return False

    def ensure(self, rec: dict, wait: float = 15) -> None:
        ssh = rec.get("ssh") or {}
        if not ssh.get("target"):
            return
        lp = int(ssh["local_port"])
        with self.lock:
            p = self.procs.get(rec["name"])
            if p is not None and p.poll() is None and self._open(lp):
                return
            if p is not None and p.poll() is None:
                p.terminate()
            exe = shutil.which("ssh")
            if not exe:
                raise RemoteError("the ssh command is not installed on this computer")
            cmd = [exe, "-N", "-L", f"127.0.0.1:{lp}:127.0.0.1:{int(ssh.get('remote_port') or nsr.DEFAULT_PORT)}",
                   "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes"]
            if ssh.get("port"):
                cmd += ["-p", str(int(ssh["port"]))]
            cmd.append(str(ssh["target"]))
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.procs[rec["name"]] = p
        end = time.time() + wait
        while time.time() < end:
            if self._open(lp):
                return
            if p.poll() is not None:
                err = (p.stderr.read() or b"").decode(errors="replace").strip()[-300:]
                raise RemoteError(f"SSH to {ssh['target']} failed: {err or 'exit ' + str(p.returncode)}. Studio uses "
                                  "your SSH keys (ssh-copy-id) and cannot answer password prompts.")
            time.sleep(0.2)
        raise RemoteError(f"SSH tunnel to {ssh['target']} did not open in {wait:.0f}s")

    def stop(self, name: str) -> None:
        with self.lock:
            p = self.procs.pop(name, None)
        if p is not None and p.poll() is None:
            p.terminate()

    def stop_all(self) -> None:
        for n in list(self.procs):
            self.stop(n)


# ------------------------------------------------------------------ hosts that call in

class CallHomeHub:
    """Invites, host tokens and the message queues for hosts that connect out to
    Studio (the host long-polls /api/remote/next, answers on /api/remote/reply)."""

    def __init__(self):
        self.lock = threading.Condition()
        self.invites: dict = {}       # code -> {name, fingerprint, expires}
        self.queues: dict = collections.defaultdict(collections.deque)
        self.waiting: dict = {}       # message id -> [Event, response]
        self.seen: dict = {}          # host name -> last poll time
        self.polling: dict = collections.Counter()   # host name -> long polls open now
        self.tickets: dict = {}       # file ticket -> {path, expires, host}

    def invite(self, url: str, name: str, fingerprint: str, controller: str, ttl: float = 900) -> dict:
        code = nsr.new_code()
        with self.lock:
            now = time.time()
            self.invites = {c: v for c, v in self.invites.items() if v["expires"] > now}
            self.invites[code] = {"name": name, "fingerprint": fingerprint, "expires": now + ttl,
                                  "controller": controller, "attempts": 0}
        return {"invite": nsr.make_invite(url, code, fingerprint, controller), "expires": time.time() + ttl,
                "code": nsr.fmt_code(code), "name": name}

    def claim(self, proof: str) -> tuple[dict, str] | None:
        """-> (invite, code) for a matching proof; the invite is used up."""
        with self.lock:
            now = time.time()
            for code, inv in list(self.invites.items()):
                if inv["expires"] < now:
                    self.invites.pop(code)
                    continue
                if hmac.compare_digest(str(proof), nsr.pair_proof(code, inv["fingerprint"], "claim")):
                    return self.invites.pop(code), code
        return None

    def request(self, host: str, message: dict, timeout: float = 120, on_progress=None) -> dict:
        """Queue a message for a host and wait for its answer. With on_progress, the
        message carries an MCP progress token; streamed pieces the host posts to
        /api/remote/progress are handed to on_progress as they come, and timeout
        counts from the last sign of life rather than from the start. If on_progress
        raises (the reader went away), the host is told to stop and the error is
        re-raised."""
        if not self.online(host):
            raise RemoteError(f"{host} is not connected: the agent there is not running, or cannot reach this "
                              "Studio")
        mid = secrets.token_hex(8)
        w = {"ev": threading.Event(), "resp": None, "host": host, "chunks": collections.deque(), "cancel": False,
             "touched": time.time()}
        if on_progress is not None and isinstance(message, dict):
            params = message.setdefault("params", {})
            params["_meta"] = {**(params.get("_meta") or {}), "progressToken": mid}
        with self.lock:
            self.waiting[mid] = w
            self.queues[host].append({"id": mid, "message": message})
            self.lock.notify_all()
        try:
            while not w["ev"].wait(0.05 if on_progress else timeout):
                if on_progress is not None:
                    self._drain(w, on_progress)
                if time.time() - w["touched"] > timeout:
                    raise RemoteError(f"{host} did not answer within {timeout:.0f}s")
            if on_progress is not None:
                self._drain(w, on_progress)
            return w["resp"]
        except BaseException:
            with self.lock:
                w["cancel"] = True
                self.queues[host] = collections.deque(m for m in self.queues[host] if m["id"] != mid)
            raise
        finally:
            # A cancelled entry lingers briefly so the host's next progress post learns to stop.
            threading.Timer(30 if w["cancel"] else 0, lambda: self.waiting.pop(mid, None)).start()

    def _drain(self, w: dict, on_progress) -> None:
        while True:
            with self.lock:
                if not w["chunks"]:
                    return
                piece = w["chunks"].popleft()
            on_progress(piece)

    def next(self, host: str, wait: float = 25) -> list:
        end = time.time() + max(0.0, min(wait, 55))
        with self.lock:
            self.seen[host] = time.time()
            self.polling[host] += 1
            try:
                while not self.queues[host] and time.time() < end:
                    self.lock.wait(max(0.05, end - time.time()))
                out = list(self.queues[host])
                self.queues[host].clear()
            finally:
                self.polling[host] -= 1
                self.seen[host] = time.time()
        return out

    def last_seen(self, host: str) -> float | None:
        """Seconds since the host was last heard from; 0 while it is waiting on a poll."""
        if self.polling.get(host):
            return 0.0
        t = self.seen.get(host)
        return None if t is None else time.time() - t

    def _mine(self, mid: str, host: str | None):
        w = self.waiting.get(mid)
        # A host may only answer the messages that were queued for it.
        return w if w is not None and (host is None or w["host"] == host) else None

    def reply(self, mid: str, response, host: str | None = None) -> bool:
        with self.lock:
            w = self._mine(mid, host)
            if w is None or w["ev"].is_set():
                return False
            w["resp"] = response
            w["ev"].set()
            return True

    def progress(self, mid: str, pieces: list, host: str | None = None) -> dict:
        """Streamed pieces of an answer still being made. -> {"ok", "cancel"}."""
        with self.lock:
            w = self._mine(mid, host)
            if w is None:
                return {"ok": False, "cancel": True}
            self.seen[w["host"]] = w["touched"] = time.time()
            if not w["cancel"]:
                w["chunks"].extend(p for p in pieces if isinstance(p, (str, dict)))
            return {"ok": True, "cancel": w["cancel"]}

    def online(self, host: str) -> bool:
        ago = self.last_seen(host)
        return ago is not None and ago < 70

    def ticket(self, path: str, host: str, ttl: float = 6 * 3600) -> str:
        t = secrets.token_urlsafe(24)
        with self.lock:
            self.tickets[t] = {"path": path, "host": host, "expires": time.time() + ttl}
        return t

    def ticket_path(self, t: str) -> str | None:
        with self.lock:
            v = self.tickets.get(t)
        return v["path"] if v and v["expires"] > time.time() else None


# ------------------------------------------------------------------ talking to a host

def token_hash(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


class Remote:
    """Studio's remote hosts. Records live in Studio's links file next to linked
    Studios, with kind "agent", so their models show up wherever linked models do."""

    def __init__(self, load_links, save_links):
        self.load_links, self.save_links = load_links, save_links
        self.tunnels = Tunnels()
        self.hub = CallHomeHub()
        self.sends: dict = {}
        self.ids = 0

    def hosts(self) -> list[dict]:
        return [x for x in self.load_links() if x.get("kind") == "agent"]

    def get(self, name: str) -> dict:
        h = next((x for x in self.hosts() if x["name"] == name), None)
        if h is None:
            raise RemoteError(f"no remote host {name!r}")
        return h

    def add(self, rec: dict) -> dict:
        links = [x for x in self.load_links() if x["name"] != rec["name"]]
        rec = {"kind": "agent", "added": time.time(), **rec}
        self.save_links(links + [rec])
        return rec

    def remove(self, name: str) -> None:
        self.tunnels.stop(name)
        self.save_links([x for x in self.load_links() if x["name"] != name])

    def host_for_token(self, tok: str) -> dict | None:
        h = token_hash(tok or "")
        for x in self.hosts():
            if x.get("host_token_sha256") and hmac.compare_digest(x["host_token_sha256"], h):
                return x
        return None

    # -- MCP
    def rpc(self, rec: dict, method: str, params: dict | None = None, timeout: float = 120, on_progress=None):
        self.ids += 1
        msg = {"jsonrpc": "2.0", "id": self.ids, "method": method, "params": params or {}}
        if rec.get("method") == "callhome":
            resp = self.hub.request(rec["name"], msg, timeout, on_progress=on_progress)
        else:
            self.tunnels.ensure(rec)
            st, resp, _ = nsr.http_json(rec["url"], "POST", "/mcp", msg, token=rec["token"],
                                        fingerprint=self._trust(rec), timeout=timeout)
            if st == 401:
                raise RemoteError(f"{rec['name']} no longer accepts this Studio (revoked on the host?); pair again")
            if st >= 400:
                raise RemoteError(f"{rec['name']}: HTTP {st}: {resp.get('error', resp)}")
        if not isinstance(resp, dict):
            raise RemoteError(f"{rec['name']}: no answer")
        if resp.get("error"):
            raise RemoteError(f"{rec['name']}: {resp['error'].get('message')}")
        return resp.get("result") or {}

    def call(self, name: str, tool: str, args: dict | None = None, timeout: float = 900, on_progress=None) -> dict:
        rec = self.get(name)
        res = self.rpc(rec, "tools/call", {"name": tool, "arguments": args or {}}, timeout, on_progress)
        if res.get("isError"):
            raise RemoteError(" ".join(c.get("text", "") for c in res.get("content") or []) or f"{tool} failed")
        return res.get("structuredContent") or {"text": " ".join(c.get("text", "") for c in res.get("content") or [])}

    def tools(self, name: str) -> list:
        return self.rpc(self.get(name), "tools/list").get("tools", [])

    @staticmethod
    def _trust(rec: dict):
        if not rec["url"].startswith("https://"):
            return None
        return rec.get("fingerprint") or None

    def upstream(self, rec: dict) -> dict:
        """For Studio's OpenAI pass-through (chat with a host's model)."""
        if rec.get("method") == "callhome":
            return {"callhome": rec["name"]}
        self.tunnels.ensure(rec)
        return {"base": rec["url"], "token": rec["token"], "fingerprint": rec.get("fingerprint") or "",
                "public_tls": rec["url"].startswith("https://") and not rec.get("fingerprint")}

    def models(self, rec: dict) -> list[dict]:
        return [m for m in self.call(rec["name"], "list_models", timeout=30).get("models", []) if m.get("id")]

    def status(self, rec: dict) -> dict:
        out = {k: rec.get(k) for k in ("name", "method", "url", "host", "added", "expires")}
        out["pinned"] = bool(rec.get("fingerprint"))
        out["short_fingerprint"] = nsr.short_fp(rec.get("fingerprint", ""))
        if rec.get("ssh"):
            out["ssh"] = rec["ssh"].get("target")
        if rec.get("method") == "callhome":
            out["online"] = self.hub.online(rec["name"])
            ago = self.hub.last_seen(rec["name"])
            out["last_seen"] = None if ago is None else round(ago)
            out["agent_version"] = rec.get("agent_version") or ""
        return out

    # -- files
    def send(self, name: str, path: str, public_base: str = "", install: bool = False) -> str:
        """Put a file into the host's incoming folder, in the background. -> send id."""
        rec = self.get(name)
        p = Path(path).expanduser()
        if not p.is_file():
            raise RemoteError(f"no such file: {path}")
        sid = secrets.token_hex(4)
        self.sends[sid] = {"id": sid, "host": name, "file": p.name, "size": p.stat().st_size, "done": 0,
                           "status": "hashing", "started": time.time()}
        threading.Thread(target=self._send, args=(sid, rec, p, public_base, install), daemon=True).start()
        return sid

    def _send(self, sid: str, rec: dict, p: Path, public_base: str, install: bool = False):
        s = self.sends[sid]
        try:
            sha = nsr.file_sha256(p)
            s["status"] = "sending"
            if rec.get("method") == "callhome":
                if not public_base:
                    raise RemoteError("Studio's own address is needed for a host that calls in")
                t = self.hub.ticket(str(p), rec["name"])
                r = self.call(rec["name"], "receive_file", {"url": f"{public_base}/api/remote/file/{t}", "name": p.name,
                                                            "sha256": sha, "fingerprint": rec.get("studio_fingerprint")
                                                            or None})
                s.update(status="host downloading", transfer=r.get("transfer"))
                while True:
                    time.sleep(3)
                    tr = next((x for x in self.call(rec["name"], "transfers", timeout=60).get("transfers", [])
                               if x["id"] == r.get("transfer")), None)
                    if tr is None:
                        raise RemoteError("the host lost track of the transfer")
                    s["done"] = tr.get("done") or 0
                    if tr["status"] == "done":
                        break
                    if tr["status"] == "failed":
                        raise RemoteError(tr.get("error", "failed"))
            else:
                self.tunnels.ensure(rec)
                q = urllib.parse.quote(p.name)
                st, info, _ = nsr.http_json(rec["url"], "GET", f"/upload/{q}", token=rec["token"],
                                            fingerprint=self._trust(rec), timeout=30)
                if st >= 400:
                    raise RemoteError(info.get("error") or f"HTTP {st}")
                off = 0 if info.get("complete") else int(info.get("size") or 0)
                with open(p, "rb") as f:
                    f.seek(off)
                    while True:
                        chunk = f.read(CHUNK)
                        last = f.tell() >= s["size"]
                        path = f"/upload/{q}?offset={off}" + (f"&final=1&sha256={sha}" if last else "")
                        out = self._put(rec, path, chunk)
                        if out.get("size") is not None and not out.get("complete") and out.get("error"):
                            off = int(out["size"])
                            f.seek(off)
                            continue
                        off += len(chunk)
                        s["done"] = off
                        if last:
                            break
            if install:
                s["status"] = "installing"
                s["installed"] = self.call(rec["name"], "install_model", {"file": p.name}, timeout=3600).get("installed")
            s.update(status="done", done=s["size"], finished=time.time())
        except Exception as e:
            s.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])

    def _put(self, rec: dict, path: str, data: bytes) -> dict:
        conn = nsr.open_conn(rec["url"], self._trust(rec), 600)
        try:
            conn.request("PUT", urllib.parse.urlsplit(rec["url"]).path.rstrip("/") + path, body=data,
                         headers={"Authorization": f"Bearer {rec['token']}",
                                  "Content-Type": "application/octet-stream"})
            r = conn.getresponse()
            out = json.loads(r.read() or b"{}")
        finally:
            conn.close()
        if r.status == 409:
            return out
        if r.status >= 400:
            raise RemoteError(out.get("error") or f"HTTP {r.status}")
        return out


def package_zip() -> bytes:
    """The agent as a zip to copy to a host: the package, its README and launchers."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(REMOTE_PKG.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".egg-info")) \
                    and ".egg-info" not in str(p) and "build" not in p.relative_to(REMOTE_PKG).parts:
                z.write(p, "omnicortex-remote/" + p.relative_to(REMOTE_PKG).as_posix())
    return buf.getvalue()
