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
MAX_ACTIVE_REMOTE_SENDS = 4
MAX_REMOTE_SEND_HISTORY = 100


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
    """Bounded invites, pending calls, host polls, progress buffers and tickets."""

    MAX_HOSTS = 256
    MAX_INVITES = 1024
    MAX_PENDING_CALLS = 256
    MAX_PENDING_PER_HOST = 32
    MAX_LONG_POLLS = 256
    MAX_POLLS_PER_HOST = 4
    MAX_PROGRESS_PIECES = 64
    MAX_PROGRESS_PIECE_BYTES = 64 * 1024
    MAX_PROGRESS_BYTES = 256 * 1024
    HOST_IDLE_TTL = 300.0

    def __init__(self):
        self.lock = threading.Condition()
        self.invites: dict = {}
        self.queues: dict = {}
        self.waiting: dict = {}
        self.seen: dict = {}
        self.polling: dict = collections.Counter()
        self.host_pending: dict = collections.Counter()
        self.active_polls = 0
        self.tickets: dict = {}
        self.max_tickets = 4096

    @staticmethod
    def _valid_host(host: str) -> bool:
        return isinstance(host, str) and 0 < len(host) <= 128 and "\\0" not in host

    def _prune_hosts_locked(self, now: float) -> None:
        for host, last_seen in list(self.seen.items()):
            if (now - float(last_seen) > self.HOST_IDLE_TTL
                    and not self.polling.get(host, 0)
                    and not self.host_pending.get(host, 0)
                    and not self.queues.get(host)):
                self.seen.pop(host, None)
                self.queues.pop(host, None)
                self.polling.pop(host, None)
                self.host_pending.pop(host, None)

    def _online_locked(self, host: str) -> bool:
        if self.polling.get(host, 0):
            return True
        seen = self.seen.get(host)
        return seen is not None and time.time() - seen < 70

    def invite(self, url: str, name: str, fingerprint: str,
               controller: str, ttl: float = 900) -> dict:
        code = nsr.new_code()
        with self.lock:
            now = time.time()
            self.invites = {
                c: v for c, v in self.invites.items()
                if float(v.get("expires", 0)) > now
            }
            if len(self.invites) >= self.MAX_INVITES:
                raise RemoteError("too many outstanding remote invites; wait for some to expire")
            ttl = max(1.0, min(float(ttl), 900.0))
            self.invites[code] = {
                "name": name, "fingerprint": fingerprint,
                "expires": now + ttl, "controller": controller, "attempts": 0,
            }
        return {
            "invite": nsr.make_invite(url, code, fingerprint, controller),
            "expires": time.time() + ttl,
            "code": nsr.fmt_code(code),
            "name": name,
        }

    def claim(self, proof: str) -> tuple[dict, str] | None:
        """Return and consume a matching invite proof."""
        with self.lock:
            now = time.time()
            for code, inv in list(self.invites.items()):
                if inv["expires"] < now:
                    self.invites.pop(code, None)
                    continue
                if hmac.compare_digest(
                    str(proof), nsr.pair_proof(code, inv["fingerprint"], "claim")
                ):
                    return self.invites.pop(code), code
        return None

    def request(self, host: str, message: dict, timeout: float = 120,
                on_progress=None) -> dict:
        """Queue a bounded request and wait for its answer or streamed progress."""
        if not self._valid_host(host):
            raise RemoteError("invalid call-home host name")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise RemoteError("invalid call-home request timeout") from exc
        if not 0 < timeout <= 86400:
            raise RemoteError("call-home request timeout must be between 0 and 86400 seconds")
        if not isinstance(message, dict):
            raise RemoteError("call-home request must be an object")

        mid = secrets.token_hex(8)
        w = {
            "ev": threading.Event(),
            "resp": None,
            "host": host,
            "chunks": collections.deque(),
            "chunk_sizes": collections.deque(),
            "chunks_bytes": 0,
            "cancel": False,
            "error": None,
            "touched": time.time(),
        }
        if on_progress is not None:
            params = message.setdefault("params", {})
            if not isinstance(params, dict):
                raise RemoteError("call-home request params must be an object")
            meta = params.get("_meta", {})
            if not isinstance(meta, dict):
                raise RemoteError("call-home request _meta must be an object")
            params["_meta"] = {
                **meta,
                "progressToken": mid,
            }

        with self.lock:
            now = time.time()
            self._prune_hosts_locked(now)
            if not self._online_locked(host):
                raise RemoteError(f"{host} is not connected")
            if len(self.waiting) >= self.MAX_PENDING_CALLS:
                raise RemoteError("call-home request capacity is full")
            if self.host_pending.get(host, 0) >= self.MAX_PENDING_PER_HOST:
                raise RemoteError(f"too many pending call-home requests for {host}")
            if mid in self.waiting:
                raise RemoteError("call-home request ID collision")
            self.waiting[mid] = w
            self.host_pending[host] += 1
            self.queues.setdefault(host, collections.deque()).append(
                {"id": mid, "message": message}
            )
            self.lock.notify_all()

        try:
            while not w["ev"].wait(0.05 if on_progress else timeout):
                if on_progress is not None:
                    self._drain(w, on_progress)
                if time.time() - w["touched"] > timeout:
                    raise RemoteError(f"{host} did not answer within {timeout:.0f}s")

            if w["error"] is not None:
                raise w["error"]
            if on_progress is not None:
                self._drain(w, on_progress)
            return w["resp"]

        except BaseException:
            with self.lock:
                w["cancel"] = True
            raise

        finally:
            # Removal is immediate. A late host reply/progress receives cancel=True;
            # no per-request Timer thread is needed.
            with self.lock:
                self.waiting.pop(mid, None)
                self.host_pending[host] -= 1
                if self.host_pending[host] <= 0:
                    self.host_pending.pop(host, None)

                queue = self.queues.get(host)
                if queue:
                    remaining = collections.deque(
                        item for item in queue if item.get("id") != mid
                    )
                    if remaining:
                        self.queues[host] = remaining
                    else:
                        self.queues.pop(host, None)
                self.lock.notify_all()

    def _drain(self, w: dict, on_progress) -> None:
        while True:
            with self.lock:
                if not w["chunks"]:
                    return
                piece = w["chunks"].popleft()
                size = w["chunk_sizes"].popleft()
                w["chunks_bytes"] = max(0, w["chunks_bytes"] - size)
            on_progress(piece)

    def next(self, host: str, wait: float = 25) -> list:
        if not self._valid_host(host):
            raise RemoteError("invalid call-home host name")
        end = time.monotonic() + max(0.0, min(float(wait), 55.0))

        with self.lock:
            now = time.time()
            self._prune_hosts_locked(now)
            if host not in self.seen:
                if len(self.seen) >= self.MAX_HOSTS:
                    raise RemoteError("call-home host capacity is full")
                self.seen[host] = now

            if self.polling.get(host, 0) >= self.MAX_POLLS_PER_HOST:
                raise RemoteError(f"too many simultaneous long polls for {host}")
            if self.active_polls >= self.MAX_LONG_POLLS:
                raise RemoteError("call-home long-poll capacity is full")

            self.polling[host] += 1
            self.active_polls += 1
            try:
                while not self.queues.get(host) and time.monotonic() < end:
                    self.lock.wait(max(0.05, end - time.monotonic()))
                out = list(self.queues.get(host, ()))
                self.queues.pop(host, None)
                return out
            finally:
                self.polling[host] -= 1
                self.active_polls -= 1
                if self.polling[host] <= 0:
                    self.polling.pop(host, None)
                self.seen[host] = time.time()

    def last_seen(self, host: str) -> float | None:
        with self.lock:
            if self.polling.get(host, 0):
                return 0.0
            seen = self.seen.get(host)
            return None if seen is None else time.time() - seen

    def _mine(self, mid: str, host: str | None):
        # Called while holding self.lock.
        w = self.waiting.get(mid)
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
        """Append bounded streamed pieces, or cancel an overflowing request."""
        with self.lock:
            w = self._mine(mid, host)
            if w is None:
                return {"ok": False, "cancel": True}

            now = time.time()
            self.seen[w["host"]] = w["touched"] = now
            if w["cancel"] or w["ev"].is_set():
                return {"ok": True, "cancel": True}

            for piece in pieces if isinstance(pieces, list) else ():
                if not isinstance(piece, (str, dict)):
                    continue
                try:
                    if isinstance(piece, str):
                        size = len(piece.encode("utf-8"))
                    else:
                        size = len(json.dumps(
                            piece, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8"))
                except (TypeError, ValueError, UnicodeError):
                    continue

                if (size > self.MAX_PROGRESS_PIECE_BYTES
                        or len(w["chunks"]) >= self.MAX_PROGRESS_PIECES
                        or w["chunks_bytes"] + size > self.MAX_PROGRESS_BYTES):
                    w["cancel"] = True
                    w["error"] = RemoteError("call-home progress buffer limit exceeded")
                    w["chunks"].clear()
                    w["chunk_sizes"].clear()
                    w["chunks_bytes"] = 0
                    w["ev"].set()
                    self.lock.notify_all()
                    return {"ok": True, "cancel": True}

                w["chunks"].append(piece)
                w["chunk_sizes"].append(size)
                w["chunks_bytes"] += size

            return {"ok": True, "cancel": False}

    def online(self, host: str) -> bool:
        with self.lock:
            return self._online_locked(host)

    def ticket(self, path: str, host: str, ttl: float = 6 * 3600) -> str:
        t = secrets.token_urlsafe(24)
        with self.lock:
            now = time.time()
            self._prune_tickets(now)
            if len(self.tickets) >= self.max_tickets:
                eligible = [
                    key for key, value in self.tickets.items()
                    if not value.get("in_use")
                ]
                if not eligible:
                    raise RemoteError("remote download ticket capacity is full")
                oldest = min(
                    eligible,
                    key=lambda key: self.tickets[key]["expires"],
                )
                self.tickets.pop(oldest, None)
            self.tickets[t] = {
                "path": path,
                "host": host,
                "expires": now + max(1.0, min(float(ttl), 6 * 3600)),
                "in_use": False,
            }
        return t

    def ticket_acquire(self, t: str) -> str | None:
        """Acquire an exclusive download lease without consuming the ticket."""
        with self.lock:
            now = time.time()
            self._prune_tickets(now)
            value = self.tickets.get(t)
            if value is None or value.get("expires", 0) <= now:
                return None
            if value.get("in_use"):
                return None
            value["in_use"] = True
            return value["path"]

    def ticket_release(self, t: str, *, completed: bool) -> None:
        """Consume a completed ticket or release an interrupted download for retry."""
        with self.lock:
            value = self.tickets.get(t)
            if value is None:
                return
            if completed:
                self.tickets.pop(t, None)
            else:
                value["in_use"] = False

    def ticket_revoke(self, t: str) -> None:
        """Revoke a download capability after the host finishes or fails."""
        with self.lock:
            self.tickets.pop(t, None)

    def ticket_path(self, t: str) -> str | None:
        # Preserve the existing immediate, one-shot redemption API.
        path = self.ticket_acquire(t)
        if path is not None:
            self.ticket_release(t, completed=True)
        return path

    def _prune_tickets(self, now: float) -> None:
        for key, value in list(self.tickets.items()):
            if float(value.get("expires", 0)) <= now:
                self.tickets.pop(key, None)


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
        self.send_lock = threading.RLock()
        self.send_slots = threading.BoundedSemaphore(MAX_ACTIVE_REMOTE_SENDS)
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
    def _trim_sends_locked(self) -> None:
        """Keep recent terminal records; never evict an active transfer."""
        terminal = {"done", "failed"}
        while len(self.sends) > MAX_REMOTE_SEND_HISTORY:
            oldest = next(
                (
                    sid for sid, record in self.sends.items()
                    if record.get("status") in terminal
                ),
                None,
            )
            if oldest is None:
                break
            self.sends.pop(oldest, None)

    def _send_update(self, sid: str, **fields) -> None:
        with self.send_lock:
            record = self.sends.get(sid)
            if record is not None:
                record.update(fields)
                self._trim_sends_locked()

    def send_snapshot(self, limit: int = 20) -> list[dict]:
        """Return a stable, detached view for the Studio status API."""
        limit = max(0, int(limit))
        if limit == 0:
            return []
        with self.send_lock:
            records = list(self.sends.values())[-limit:]
            return [dict(record) for record in records]

    def send(self, name: str, path: str, public_base: str = "", install: bool = False) -> str:
        """Put a file into the host's incoming folder, in the background."""
        rec = self.get(name)
        p = Path(path).expanduser()
        try:
            file_stat = p.stat()
        except OSError:
            raise RemoteError(f"no such file: {path}") from None
        if not p.is_file():
            raise RemoteError(f"not a regular file: {path}")

        if not self.send_slots.acquire(blocking=False):
            raise RemoteError(
                f"too many active remote sends (limit {MAX_ACTIVE_REMOTE_SENDS}); "
                "wait for one to finish"
            )

        sid = None
        try:
            with self.send_lock:
                sid = secrets.token_urlsafe(16)
                while sid in self.sends:
                    sid = secrets.token_urlsafe(16)

                self.sends[sid] = {
                    "id": sid,
                    "host": name,
                    "file": p.name,
                    "size": file_stat.st_size,
                    "done": 0,
                    "status": "hashing",
                    "started": time.time(),
                }
                self._trim_sends_locked()

            worker = threading.Thread(
                target=self._send,
                args=(sid, rec, p, public_base, install),
                daemon=True,
                name=f"remote-send-{sid[:8]}",
            )
            worker.start()
            return sid
        except BaseException:
            if sid is not None:
                with self.send_lock:
                    self.sends.pop(sid, None)
            self.send_slots.release()
            raise

    def _send(self, sid: str, rec: dict, p: Path, public_base: str, install: bool = False):
        ticket = None
        try:
            with self.send_lock:
                record = self.sends.get(sid)
                if record is None:
                    return
                size = int(record["size"])

            sha = nsr.file_sha256(p)
            self._send_update(sid, status="sending")

            if rec.get("method") == "callhome":
                if not public_base:
                    raise RemoteError("Studio's own address is needed for a host that calls in")

                ticket = self.hub.ticket(str(p), rec["name"])
                result = self.call(
                    rec["name"],
                    "receive_file",
                    {
                        "url": f"{public_base}/api/remote/file/{ticket}",
                        "name": p.name,
                        "sha256": sha,
                        "fingerprint": rec.get("studio_fingerprint") or None,
                    },
                )
                self._send_update(
                    sid,
                    status="host downloading",
                    transfer=result.get("transfer"),
                )

                while True:
                    time.sleep(3)
                    transfers = self.call(
                        rec["name"], "transfers", timeout=60
                    ).get("transfers", [])
                    transfer = next(
                        (
                            item for item in transfers
                            if item["id"] == result.get("transfer")
                        ),
                        None,
                    )
                    if transfer is None:
                        raise RemoteError("the host lost track of the transfer")

                    self._send_update(sid, done=transfer.get("done") or 0)
                    if transfer["status"] == "done":
                        break
                    if transfer["status"] == "failed":
                        raise RemoteError(transfer.get("error", "failed"))

            else:
                self.tunnels.ensure(rec)
                quoted_name = urllib.parse.quote(p.name)
                status, info, _ = nsr.http_json(
                    rec["url"],
                    "GET",
                    f"/upload/{quoted_name}",
                    token=rec["token"],
                    fingerprint=self._trust(rec),
                    timeout=30,
                )
                if status >= 400:
                    raise RemoteError(info.get("error") or f"HTTP {status}")

                offset = 0 if info.get("complete") else int(info.get("size") or 0)
                with open(p, "rb") as source:
                    source.seek(offset)
                    while True:
                        chunk = source.read(CHUNK)
                        last = source.tell() >= size
                        request_path = (
                            f"/upload/{quoted_name}?offset={offset}"
                            + (f"&final=1&sha256={sha}" if last else "")
                        )
                        result = self._put(rec, request_path, chunk)

                        if (
                            result.get("size") is not None
                            and not result.get("complete")
                            and result.get("error")
                        ):
                            offset = int(result["size"])
                            source.seek(offset)
                            continue

                        offset += len(chunk)
                        self._send_update(sid, done=offset)
                        if last:
                            break

            if install:
                self._send_update(sid, status="installing")
                installed = self.call(
                    rec["name"],
                    "install_model",
                    {"file": p.name},
                    timeout=3600,
                ).get("installed")
                self._send_update(sid, installed=installed)

            self._send_update(
                sid,
                status="done",
                done=size,
                finished=time.time(),
            )

        except Exception as exc:
            self._send_update(
                sid,
                status="failed",
                error=f"{type(exc).__name__}: {exc}"[:300],
                finished=time.time(),
            )
        finally:
            if ticket is not None:
                self.hub.ticket_revoke(ticket)
            self.send_slots.release()

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
