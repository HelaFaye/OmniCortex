#!/usr/bin/env python3
"""OmniCortex reliable transfer: resumable, checksummed HTTP(S) file transfer.

    # receiver (e.g. the LM Studio host)
    python scripts/ns_security.py token --out ~/.config/neuronscope/transfer.token
    python scripts/ns_transfer.py serve --host 0.0.0.0 --root ~/.lmstudio/models/neuronscope \\
        --token-file ~/.config/neuronscope/transfer.token --tls-cert cert.pem --tls-key key.pem

    # sender
    NS_TRANSFER_TOKEN=... python scripts/ns_transfer.py send --url https://host:8810 \\
        --cafile cert.pem --file model.gguf

Security model (see docs/SECURITY.md): the bearer token authenticates, TLS (or
a VPN tunnel) provides confidentiality. SHA-256 per chunk and per file detects
corruption; it is not a signature. A non-loopback bind refuses to start without
a token, and without TLS unless --allow-plaintext is given.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_security as sec  # noqa: E402

VERSION = "1.1"
CHUNK = 4 * 1024 * 1024
MAX_CHUNK = sec.MAX_CHUNK_BYTES
MAX_JSON = sec.MAX_JSON_BYTES
MAX_NAME = 240
ID_RE = re.compile(r"^[A-Za-z0-9_-]{12,64}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server bound to IPv6 only (no implicit IPv4-mapped bind)."""

    address_family = socket.AF_INET6

    def server_bind(self):
        # Make the bind semantics predictable across Linux and Windows.
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


def normalize_bind_host(host: str) -> str:
    """Normalize a bind host; bracketed literals are accepted for convenience.

    For a bind argument, ``::`` is conventional. Brackets are normally used
    in URLs, but stripping them here avoids confusing behavior when copied
    from a URL. IPv6 zone identifiers (for link-local addresses) are retained.
    """
    value = (host or "").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value or value.startswith("[") or value.endswith("]"):
        raise ValueError(f"invalid bind host: {host!r}")
    return value


def server_class_for_host(host: str, family: str = "auto"):
    """Choose a server socket family, auto-detecting literal IPv6 addresses.

    ``--family ipv6`` is available for DNS names that resolve to IPv6 but do
    not themselves contain a colon. ``auto`` preserves existing IPv4 behavior.
    """
    value = normalize_bind_host(host)
    if family not in {"auto", "ipv4", "ipv6"}:
        raise ValueError(f"unsupported address family: {family!r}")
    selected = family
    if selected == "auto":
        selected = "ipv6" if ":" in value else "ipv4"
    return IPv6ThreadingHTTPServer if selected == "ipv6" else ThreadingHTTPServer


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_name(name: str) -> str:
    name = Path(name).name.replace("\\", "_")
    name = "".join(c if c.isprintable() and c not in '\r\n\x00"' else "_" for c in name)
    if not name or name in {".", ".."}:
        name = "model.gguf"
    return name[:MAX_NAME]


def token_ok(header: str | None, expected: str) -> bool:
    return sec.bearer_ok(header, expected)


class TransferError(RuntimeError):
    pass


class Receiver:
    def __init__(self, root: Path, max_active: int = 8):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.root / ".neuronscope-transfers"
        self.meta_dir.mkdir(exist_ok=True)
        self.max_active = max_active
        self.lock = threading.RLock()

    def paths(self, tid: str) -> tuple[Path, Path]:
        return self.meta_dir / f"{tid}.json", self.meta_dir / f"{tid}.part"

    def _all(self) -> list[dict]:
        rows = []
        for mp in self.meta_dir.glob("*.json"):
            try:
                rows.append(json.loads(mp.read_text()))
            except Exception:
                continue
        return rows

    def list(self) -> list[dict]:
        with self.lock:
            return self._all()

    def init(self, payload: dict) -> dict:
        name = safe_name(str(payload.get("name", "model.gguf")))
        size = int(payload["size"])
        sha = str(payload["sha256"]).lower()
        if size < 0 or not SHA_RE.fullmatch(sha):
            raise TransferError("invalid size/sha256")
        subdir = safe_name(str(payload["subdir"])) if payload.get("subdir") else ""
        with self.lock:
            active = 0
            for old in self._all():
                if old.get("status") != "receiving":
                    continue
                active += 1
                if (old.get("name") == name and int(old.get("size", -1)) == size
                        and old.get("sha256") == sha and old.get("subdir", "") == subdir):
                    _, pp = self.paths(old["id"])
                    old["offset"] = pp.stat().st_size if pp.exists() else 0
                    old["updated"] = time.time()
                    self.save(old)
                    return old
            if active >= self.max_active:
                raise TransferError(f"too many active transfers ({active}); finish or cancel some")
            free = shutil.disk_usage(self.root).free
            if size > free:
                raise TransferError(f"insufficient space: need {size} bytes, {free} free")
            tid = secrets.token_hex(16)  # 128-bit transfer id
            now = time.time()
            meta = {"version": VERSION, "id": tid, "name": name, "size": size, "sha256": sha,
                    "created": now, "updated": now, "offset": 0, "status": "receiving",
                    "subdir": subdir}
            mp, pp = self.paths(tid)
            pp.touch()
            mp.write_text(json.dumps(meta, indent=2))
            return meta

    def get(self, tid: str) -> dict:
        if not ID_RE.match(tid):
            raise TransferError("bad id")
        mp, _ = self.paths(tid)
        if not mp.exists():
            raise FileNotFoundError(tid)
        return json.loads(mp.read_text())

    def save(self, meta: dict) -> None:
        mp = self.paths(meta["id"])[0]
        tmp = mp.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, mp)

    def chunk(self, tid: str, start: int, total: int, body: bytes, digest: str) -> dict:
        with self.lock:
            meta = self.get(tid)
            if meta["status"] != "receiving":
                raise TransferError("transfer not receiving")
            if total != meta["size"]:
                raise TransferError("total mismatch")
            if start != int(meta["offset"]):
                raise TransferError(f"offset mismatch; expected {meta['offset']}, got {start}")
            if start + len(body) > total:
                raise TransferError("chunk exceeds total")
            if not secrets.compare_digest(sha256_bytes(body), digest.lower()):
                raise TransferError("chunk sha256 mismatch")
            _, pp = self.paths(tid)
            with pp.open("ab") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            meta["offset"] = start + len(body)
            meta["updated"] = time.time()
            self.save(meta)
            return meta

    def complete(self, tid: str) -> dict:
        with self.lock:
            meta = self.get(tid)
            _, pp = self.paths(tid)
            if meta["status"] == "completed":
                return meta
            if meta["offset"] != meta["size"]:
                raise TransferError("size incomplete")
            if sha256_file(pp) != meta["sha256"]:
                raise TransferError("final sha256 mismatch")
            sub = Path(meta.get("subdir") or "")
            dest = (self.root / sub / safe_name(meta["name"])).resolve()
            if self.root not in dest.parents:
                raise TransferError("unsafe destination")
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pp, dest)  # same filesystem: atomic publish
            meta.update(status="completed", offset=meta["size"], completed_at=time.time(),
                        path=str(dest))
            self.save(meta)
            return meta

    def cancel(self, tid: str) -> dict:
        with self.lock:
            meta = self.get(tid)
            _, pp = self.paths(tid)
            meta["status"] = "cancelled"
            meta["updated"] = time.time()
            self.save(meta)
            pp.unlink(missing_ok=True)
            return meta


class Handler(BaseHTTPRequestHandler):
    receiver: Receiver = None
    token = ""
    throttle = sec.FailureThrottle()
    timeout = 120  # per-socket idle timeout; stops slow clients pinning threads
    server_version = "NeuronScopeTransfer/" + VERSION
    sys_version = ""

    def _json(self, code: int, obj) -> None:
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _auth(self) -> bool:
        addr = self.client_address[0]
        if self.throttle.blocked(addr):
            self._json(429, {"error": "too many failed attempts; try later"})
            return False
        if token_ok(self.headers.get("Authorization"), self.token):
            if self.token:
                self.throttle.succeed(addr)
            return True
        self.throttle.fail(addr)
        self._json(401, {"error": "unauthorized"})
        return False

    def _length(self, limit: int) -> int:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise TransferError("bad Content-Length")
        if n < 0 or n > limit:
            raise TransferError(f"body too large (limit {limit} bytes)")
        return n

    def _read_json(self) -> dict:
        n = self._length(MAX_JSON)
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise TransferError("JSON object expected")
        return data

    def log_message(self, *args):
        return

    def do_GET(self):
        if not self._auth():
            return
        try:
            if self.path == "/health":
                return self._json(200, {"ok": True, "version": VERSION, "max_chunk": MAX_CHUNK})
            if self.path == "/v1/transfers":
                return self._json(200, {"transfers": self.receiver.list()})
            m = re.fullmatch(r"/v1/transfers/([^/]+)", self.path)
            if m:
                return self._json(200, self.receiver.get(m.group(1)))
            m = re.fullmatch(r"/v1/transfers/([^/]+)/download", self.path)
            if m:
                meta = self.receiver.get(m.group(1))
                if meta.get("status") != "completed":
                    return self._json(409, {"error": "not completed"})
                path = Path(meta["path"])
                self.send_response(200)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Content-SHA256", meta["sha256"])
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name(meta["name"])}"')
                self.end_headers()
                with path.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile, CHUNK)
                return
            self._json(404, {"error": "not found"})
        except FileNotFoundError:
            self._json(404, {"error": "not found"})
        except TransferError as e:
            self._json(400, {"error": str(e)})
        except Exception:
            self._json(500, {"error": "internal error"})

    def do_POST(self):
        if not self._auth():
            return
        try:
            data = self._read_json()
            if self.path == "/v1/transfers/init":
                return self._json(201, self.receiver.init(data))
            m = re.fullmatch(r"/v1/transfers/([^/]+)/complete", self.path)
            if m:
                return self._json(200, self.receiver.complete(m.group(1)))
            m = re.fullmatch(r"/v1/transfers/([^/]+)/cancel", self.path)
            if m:
                return self._json(200, self.receiver.cancel(m.group(1)))
            self._json(404, {"error": "not found"})
        except FileNotFoundError:
            self._json(404, {"error": "not found"})
        except (TransferError, KeyError, ValueError) as e:
            self._json(400, {"error": str(e)})
        except Exception:
            self._json(500, {"error": "internal error"})

    def do_PUT(self):
        if not self._auth():
            return
        try:
            m = re.fullmatch(r"/v1/transfers/([^/]+)/chunk", self.path)
            if not m:
                return self._json(404, {"error": "not found"})
            mm = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", self.headers.get("Content-Range", ""))
            if not mm:
                return self._json(400, {"error": "Content-Range required"})
            start, end, total = map(int, mm.groups())
            n = end - start + 1
            if n <= 0 or n > MAX_CHUNK:
                return self._json(413, {"error": f"chunk must be 1..{MAX_CHUNK} bytes"})
            if n != self._length(MAX_CHUNK):
                return self._json(400, {"error": "length mismatch"})
            body = self.rfile.read(n)
            if len(body) != n:
                return self._json(400, {"error": "short body"})
            digest = self.headers.get("X-Chunk-SHA256", "")
            return self._json(200, self.receiver.chunk(m.group(1), start, total, body, digest))
        except FileNotFoundError:
            self._json(404, {"error": "not found"})
        except (TransferError, KeyError, ValueError) as e:
            self._json(409, {"error": str(e)})
        except Exception:
            self._json(500, {"error": "internal error"})


# ---------------------------------------------------------------- client

_CLIENT_SSL = {"cafile": "", "insecure": False}


def _ssl_for(url: str):
    if url.startswith("https://"):
        return sec.client_ssl_context(_CLIENT_SSL["cafile"], _CLIENT_SSL["insecure"])
    return None


def request(base, method, path, token="", data=None, headers=None, timeout=120, raw=False):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    if headers:
        h.update(headers)
    b = None
    if data is not None:
        if raw:
            b = data
        else:
            b = json.dumps(data).encode()
            h["Content-Type"] = "application/json"
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, data=b, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_for(url)) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raise TransferError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e


def send_one(base, path: Path, token="", subdir="", chunk=CHUNK, timeout=300, retries=4):
    path = Path(path).resolve()
    size = path.stat().st_size
    sha = sha256_file(path)
    chunk = max(1, min(chunk, MAX_CHUNK))
    meta = request(base, "POST", "/v1/transfers/init", token,
                   {"name": path.name, "size": size, "sha256": sha, "subdir": subdir}, timeout=timeout)
    tid = meta["id"]
    offset = int(meta["offset"])
    with path.open("rb") as f:
        while offset < size:
            f.seek(offset)
            data = f.read(min(chunk, size - offset))
            if not data:
                raise TransferError("unexpected EOF")
            headers = {"Content-Range": f"bytes {offset}-{offset + len(data) - 1}/{size}",
                       "X-Chunk-SHA256": sha256_bytes(data)}
            for attempt in range(retries + 1):
                try:
                    meta = request(base, "PUT", f"/v1/transfers/{tid}/chunk", token, data,
                                   headers=headers, timeout=timeout, raw=True)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == retries:
                        raise
                    time.sleep(2 ** attempt)
                    # The receiver is authoritative about how much it has.
                    meta = request(base, "GET", f"/v1/transfers/{tid}", token, timeout=timeout)
                    if int(meta["offset"]) != offset:
                        break
            offset = int(meta["offset"])
    return request(base, "POST", f"/v1/transfers/{tid}/complete", token, {}, timeout=timeout)


def pull(base, tid, token, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = request(base, "GET", f"/v1/transfers/{tid}", token)
    url = base.rstrip("/") + f"/v1/transfers/{tid}/download"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    final = out_dir / safe_name(meta["name"])
    tmp = final.with_name(final.name + ".ns-incoming")
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=300, context=_ssl_for(url)) as r, tmp.open("wb") as f:
        for b in iter(lambda: r.read(CHUNK), b""):
            h.update(b)
            f.write(b)
    if h.hexdigest() != meta["sha256"]:
        tmp.unlink(missing_ok=True)
        raise TransferError("downloaded file sha256 mismatch")
    os.replace(tmp, final)
    return meta


def _watch(a, token) -> int:
    sp = Path(a.state)
    done: dict = {}
    stable: dict = {}
    if sp.exists():
        try:
            done = dict(json.loads(sp.read_text()).get("done", {}))
        except Exception:
            pass
    while True:
        todo = []
        for pth in sorted(Path(a.dir).glob(a.glob)):
            if not pth.is_file():
                continue
            key = str(pth.resolve())
            st = pth.stat()
            sig = (st.st_size, st.st_mtime_ns)
            previous = stable.get(key)
            stable[key] = sig
            if previous != sig:  # still being written, or first sighting
                continue
            if done.get(key) == f"{st.st_size}:{st.st_mtime_ns}":
                continue
            todo.append(pth)
        if todo:
            def f(pth):
                try:
                    return str(pth.resolve()), send_one(a.url, pth, token, a.subdir)
                except Exception as e:
                    return str(pth.resolve()), {"error": str(e)}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
                for key, res in ex.map(f, todo):
                    print(json.dumps({"file": key, "result": res}, indent=2), flush=True)
                    if "error" not in res:
                        st = Path(key).stat()
                        done[key] = f"{st.st_size}:{st.st_mtime_ns}"
            sp.write_text(json.dumps({"done": done}, indent=2))
        time.sleep(max(0.5, a.interval))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ns-transfer", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the receiver")
    s.add_argument("--host", default="127.0.0.1", help="bind address; use :: for all IPv6 interfaces")
    s.add_argument("--family", choices=("auto", "ipv4", "ipv6"), default="auto",
                   help="socket family (auto selects IPv6 for a literal IPv6 bind address)")
    s.add_argument("--port", type=int, default=8810)
    s.add_argument("--root", required=True)
    s.add_argument("--max-active", type=int, default=8, help="concurrent in-progress transfers")
    sec.add_server_security_args(s)

    def client(name, help_):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--url", required=True)
        sec.add_client_security_args(c)
        c.add_argument("--insecure", action="store_true", help="skip TLS verification (testing only)")
        return c

    client("health", "check a receiver")
    c = client("send", "send one file")
    c.add_argument("--file", required=True)
    c.add_argument("--subdir", default="")
    c.add_argument("--chunk-mib", type=int, default=4)
    c = client("batch", "send every matching file in a directory")
    c.add_argument("--dir", required=True)
    c.add_argument("--glob", default="*.gguf")
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--subdir", default="")
    c = client("watch", "send new/changed files as they appear")
    c.add_argument("--dir", required=True)
    c.add_argument("--glob", default="*.gguf")
    c.add_argument("--state", default=".neuronscope-transfer-watch.json")
    c.add_argument("--interval", type=float, default=5)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--subdir", default="")
    client("status", "list transfers on the receiver")
    c = client("pull", "download a completed transfer")
    c.add_argument("--id", required=True)
    c.add_argument("--out", required=True)

    a = p.parse_args(argv)
    token = sec.resolve_token(a.token, a.token_file)

    if a.cmd == "serve":
        tls = bool(a.tls_cert and a.tls_key)
        try:
            bind_host = normalize_bind_host(a.host)
            server_class = server_class_for_host(bind_host, a.family)
            for w in sec.check_bind(bind_host, token, tls=tls, allow_plaintext=a.allow_plaintext):
                print("warning:", w, file=sys.stderr)
        except (sec.SecurityConfigError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        Handler.receiver = Receiver(Path(a.root), a.max_active)
        Handler.token = token
        srv = server_class((bind_host, a.port), Handler)
        if tls:
            ctx = sec.server_ssl_context(a.tls_cert, a.tls_key)
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
        scheme = "https" if tls else "http"
        display_host = f"[{bind_host}]" if ":" in bind_host else bind_host
        print(f"OmniCortex transfer receiver: {scheme}://{display_host}:{a.port}  root={Handler.receiver.root}"
              f"  family={server_class.address_family}"
              f"  auth={'token' if token else 'none (loopback only)'}")
        srv.serve_forever()
        return 0

    _CLIENT_SSL.update(cafile=a.cafile, insecure=a.insecure)
    sec.warn_plaintext_url(a.url, token)

    if a.cmd == "health":
        print(json.dumps(request(a.url, "GET", "/health", token), indent=2))
        return 0
    if a.cmd == "send":
        res = send_one(a.url, Path(a.file), token, a.subdir, max(1, min(a.chunk_mib, 8)) * 1024 * 1024)
        print(json.dumps(res, indent=2))
        return 0
    if a.cmd == "batch":
        files = sorted(Path(a.dir).glob(a.glob))
        if not files:
            print("No matching files.")
            return 0

        def f(pth):
            try:
                return {"file": str(pth), "ok": True, "result": send_one(a.url, pth, token, a.subdir)}
            except Exception as e:
                return {"file": str(pth), "ok": False, "error": str(e)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            rows = list(ex.map(f, files))
        print(json.dumps(rows, indent=2))
        return 0 if all(r["ok"] for r in rows) else 2
    if a.cmd == "watch":
        return _watch(a, token)
    if a.cmd == "status":
        print(json.dumps(request(a.url, "GET", "/v1/transfers", token), indent=2))
        return 0
    if a.cmd == "pull":
        print(json.dumps(pull(a.url, a.id, token, Path(a.out)), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
