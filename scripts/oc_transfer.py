#!/usr/bin/env python3
"""OmniCortex reliable transfer: resumable, checksummed HTTP(S) file transfer.

    # receiver (e.g. the LM Studio host)
    python scripts/oc_security.py token --out ~/.config/omnicortex/transfer.token
    python scripts/oc_transfer.py serve --host 0.0.0.0 --root ~/.lmstudio/models/omnicortex \\
        --token-file ~/.config/omnicortex/transfer.token --tls-cert cert.pem --tls-key key.pem

    # sender
    OC_TRANSFER_TOKEN=... python scripts/oc_transfer.py send --url https://host:8810 \\
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
import math
import os
import re
import secrets
import shutil
import socket
import stat
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oc_security as sec  # noqa: E402
import oc_http  # noqa: E402

VERSION = "1.1"
CHUNK = 4 * 1024 * 1024
MAX_CHUNK = sec.MAX_CHUNK_BYTES
MAX_JSON = sec.MAX_JSON_BYTES
MAX_NAME = 240
ID_RE = re.compile(r"[A-Za-z0-9_-]{12,64}")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INCOMPLETE_TTL = 24 * 60 * 60
HISTORY_TTL = 7 * 24 * 60 * 60
MAX_TRANSFER_HISTORY = 1000


class IPv6ThreadingHTTPServer(oc_http.BoundedThreadingHTTPServer):
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
    return IPv6ThreadingHTTPServer if selected == "ipv6" else oc_http.BoundedThreadingHTTPServer


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
        self.meta_dir = self.root / ".omnicortex-transfers"
        self.meta_dir.mkdir(mode=0o700, exist_ok=True)
        if self.meta_dir.is_symlink() or not self.meta_dir.is_dir():
            raise TransferError("transfer metadata path must be a real directory")
        os.chmod(self.meta_dir, 0o700)
        for entry in self.meta_dir.iterdir():
            if not entry.is_symlink() and entry.is_file():
                os.chmod(entry, 0o600)
        self.max_active = max(1, min(int(max_active), 256))
        self.lock = threading.RLock()

    def paths(self, tid: str) -> tuple[Path, Path]:
        return self.meta_dir / f"{tid}.json", self.meta_dir / f"{tid}.part"

    def _destination(self, meta: dict) -> Path:
        """Resolve a final destination while excluding metadata storage."""
        candidate = (
            self.root
            / Path(meta.get("subdir") or "")
            / safe_name(meta["name"])
        )
        if candidate.is_symlink():
            raise TransferError("transfer destination must not be a symlink")

        dest = candidate.resolve()
        if (
            self.root not in dest.parents
            or dest == self.meta_dir
            or self.meta_dir in dest.parents
        ):
            raise TransferError("unsafe or reserved destination")
        return dest

    @staticmethod
    def _published_matches(dest: Path, meta: dict) -> bool:
        """Check whether an already-published file is exactly this transfer."""
        try:
            return (
                dest.is_file()
                and dest.stat().st_size == meta["size"]
                and sha256_file(dest) == meta["sha256"]
            )
        except OSError:
            return False

    def _all(self) -> list[dict]:
        """Load only structurally valid metadata records."""
        rows = []
        for mp in self.meta_dir.glob("*.json"):
            try:
                if mp.is_symlink() or not mp.is_file():
                    continue

                row = json.loads(mp.read_text(encoding="utf-8"))
                if not isinstance(row, dict):
                    continue

                tid = row.get("id")
                if (
                    not isinstance(tid, str)
                    or not ID_RE.fullmatch(tid)
                    or mp.name != f"{tid}.json"
                ):
                    continue

                name = row.get("name")
                size = row.get("size")
                offset = row.get("offset")
                sha = row.get("sha256")
                status = row.get("status")
                subdir = row.get("subdir", "")

                if (
                    not isinstance(name, str)
                    or safe_name(name) != name
                    or type(size) is not int
                    or size < 0
                    or type(offset) is not int
                    or not 0 <= offset <= size
                    or not isinstance(sha, str)
                    or not SHA_RE.fullmatch(sha)
                    or not isinstance(status, str)
                    or not status
                    or not isinstance(subdir, str)
                    or (subdir and safe_name(subdir) != subdir)
                ):
                    continue

                timestamps_valid = True
                for key in ("created", "updated", "completed_at"):
                    value = row.get(key)
                    if value is not None:
                        if (
                            type(value) not in (int, float)
                            or not math.isfinite(value)
                        ):
                            timestamps_valid = False
                            break

                if timestamps_valid:
                    rows.append(row)

            except (OSError, ValueError, TypeError, OverflowError, UnicodeError):
                continue

        return rows

    def _prune_locked(self, now: float | None = None) -> list[dict]:
        """Expire abandoned uploads and bound persistent transfer history."""
        now = time.time() if now is None else now
        rows = self._all()
        kept = []
        for row in rows:
            mp, pp = self.paths(row["id"])
            status = row.get("status")
            updated = float(row.get("updated", row.get("created", now)) or now)
            if status == "receiving" and now - updated > INCOMPLETE_TTL:
                row.update(status="expired", updated=now, error="incomplete transfer expired")
                self.save(row)
                pp.unlink(missing_ok=True)
                status = "expired"
                updated = now
            age_from = float(row.get("completed_at", updated) or updated)
            if status != "receiving" and now - age_from > HISTORY_TTL:
                mp.unlink(missing_ok=True)
                pp.unlink(missing_ok=True)
                continue
            kept.append(row)
        if len(kept) > MAX_TRANSFER_HISTORY:
            active = [r for r in kept if r.get("status") == "receiving"]
            terminal = sorted(
                (r for r in kept if r.get("status") != "receiving"),
                key=lambda r: float(r.get("updated", r.get("created", 0)) or 0), reverse=True)
            kept = active + terminal[:max(0, MAX_TRANSFER_HISTORY - len(active))]
            keep_ids = {r["id"] for r in kept}
            for row in rows:
                if row["id"] not in keep_ids and row.get("status") != "receiving":
                    mp, pp = self.paths(row["id"])
                    mp.unlink(missing_ok=True)
                    pp.unlink(missing_ok=True)
        return kept

    def list(self) -> list[dict]:
        with self.lock:
            return self._prune_locked()

    def init(self, payload: dict) -> dict:
        name = safe_name(str(payload.get("name", "model.gguf")))
        raw_size = payload["size"]
        if type(raw_size) is not int:
            raise TransferError("transfer size must be an integer")
        size = raw_size
        sha = str(payload["sha256"]).lower()
        if size < 0 or not SHA_RE.fullmatch(sha):
            raise TransferError("invalid size/sha256")
        subdir = safe_name(str(payload["subdir"])) if payload.get("subdir") else ""
        reserved = self.meta_dir.name.casefold()
        if subdir and subdir.casefold() == reserved:
            raise TransferError("reserved transfer metadata directory")
        if not subdir and name.casefold() == reserved:
            raise TransferError("reserved transfer metadata destination")
        with self.lock:
            rows = self._prune_locked()
            active = 0
            reserved = 0
            now = time.time()
            requested_dest = self._destination({"name": name, "subdir": subdir})
            matching = None

            for old in rows:
                if old.get("status") != "receiving":
                    continue

                old_size = old["size"]
                _, pp = self.paths(old["id"])

                # Never follow a symlink from the private partial-file directory.
                if pp.is_symlink():
                    old.update(
                        status="failed",
                        updated=now,
                        error="partial transfer path is a symlink",
                    )
                    self.save(old)
                    pp.unlink(missing_ok=True)
                    continue

                if pp.exists():
                    if not pp.is_file():
                        raise TransferError(
                            f"partial transfer {old['id']} is not a regular file"
                        )
                    actual_size = pp.stat().st_size
                else:
                    # os.replace() may have published the final file immediately
                    # before a crash prevented the completion metadata update.
                    dest = self._destination(old)
                    if self._published_matches(dest, old):
                        old.update(
                            status="completed",
                            offset=old_size,
                            updated=now,
                            completed_at=now,
                            path=str(dest),
                        )
                        self.save(old)

                        if (
                            old.get("name") == name
                            and old_size == size
                            and old.get("sha256") == sha
                            and old.get("subdir", "") == subdir
                        ):
                            if matching is not None:
                                raise TransferError("multiple transfers reserve the same destination")
                            matching = old
                        continue

                    fd = os.open(
                        pp,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    os.close(fd)
                    actual_size = 0

                if actual_size > old_size:
                    old.update(
                        status="failed",
                        offset=0,
                        updated=now,
                        error="partial file exceeds declared transfer size",
                    )
                    self.save(old)
                    pp.unlink(missing_ok=True)
                    continue

                # The file is authoritative after an interrupted append.
                if old["offset"] != actual_size:
                    old["offset"] = actual_size
                    old["updated"] = now
                    self.save(old)

                active += 1

                old_dest = self._destination(old)
                if old_dest == requested_dest:
                    same_transfer = (
                        old.get("name") == name
                        and old_size == size
                        and old.get("sha256") == sha
                        and old.get("subdir", "") == subdir
                    )
                    if not same_transfer:
                        raise TransferError(
                            "destination is reserved by another active transfer"
                        )
                    if matching is not None:
                        raise TransferError(
                            "multiple transfers reserve the same destination"
                        )
                    matching = old

                reserved += old_size - actual_size

            if matching is not None:
                return matching

            if active >= self.max_active:
                raise TransferError(f"too many active transfers ({active}); finish or cancel some")
            free = shutil.disk_usage(self.root).free
            required = size + reserved
            if required > free:
                raise TransferError(f"insufficient space for this and active transfers: need {required} bytes, {free} free")
            tid = secrets.token_hex(16)  # 128-bit transfer id
            now = time.time()
            meta = {"version": VERSION, "id": tid, "name": name, "size": size, "sha256": sha,
                    "created": now, "updated": now, "offset": 0, "status": "receiving",
                    "subdir": subdir}
            mp, pp = self.paths(tid)
            if mp.exists():
                raise TransferError("transfer id collision")
            fd = os.open(pp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            try:
                self.save(meta)
            except Exception:
                pp.unlink(missing_ok=True)
                mp.unlink(missing_ok=True)
                raise
            return meta

    def get(self, tid: str) -> dict:
        if not isinstance(tid, str) or not ID_RE.fullmatch(tid):
            raise TransferError("bad id")
        mp, _ = self.paths(tid)
        if mp.is_symlink():
            raise TransferError("transfer metadata path is a symlink")
        if not mp.exists():
            raise FileNotFoundError(tid)
        if not mp.is_file():
            raise TransferError("transfer metadata path is not a regular file")
        try:
            row = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise TransferError("transfer metadata is unreadable or malformed") from exc
        if not isinstance(row, dict) or row.get("id") != tid:
            raise TransferError("transfer metadata identity mismatch")
        return row

    def save(self, meta: dict) -> None:
        import tempfile

        mp = self.paths(meta["id"])[0]
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{meta['id']}.", suffix=".tmp", dir=str(self.meta_dir)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), 0o600)
                json.dump(meta, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, mp)
            os.chmod(mp, 0o600)
        finally:
            tmp.unlink(missing_ok=True)

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
            if pp.is_symlink():
                raise TransferError("partial transfer path is a symlink")

            flags = (
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                fd = os.open(pp, flags)
            except OSError as exc:
                raise TransferError("partial transfer file cannot be safely opened") from exc

            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise TransferError("partial transfer path is not a regular file")
                if info.st_size != start:
                    raise TransferError(
                        f"partial file size mismatch; expected {start}, found {info.st_size}"
                    )
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)

                with os.fdopen(fd, "ab") as f:
                    fd = -1
                    f.write(body)
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)

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
            if meta["status"] != "receiving":
                raise TransferError("transfer not receiving")

            dest = self._destination(meta)

            # Recover if publication succeeded but saving completion metadata
            # did not. Do not accept a destination with different contents.
            if not pp.exists():
                if self._published_matches(dest, meta):
                    now = time.time()
                    meta.update(
                        status="completed",
                        offset=meta["size"],
                        updated=now,
                        completed_at=now,
                        path=str(dest),
                    )
                    self.save(meta)
                    return meta
                raise TransferError(
                    "partial file is missing and no matching published file exists"
                )

            if pp.is_symlink() or not pp.is_file():
                raise TransferError("partial transfer path is not a regular file")

            if meta["offset"] != meta["size"]:
                raise TransferError("size incomplete")
            if pp.stat().st_size != meta["size"]:
                raise TransferError("partial file size does not match transfer metadata")
            if sha256_file(pp) != meta["sha256"]:
                raise TransferError("final sha256 mismatch")

            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pp, dest)
            now = time.time()
            meta.update(
                status="completed",
                offset=meta["size"],
                updated=now,
                completed_at=now,
                path=str(dest),
            )
            self.save(meta)
            return meta

    def cancel(self, tid: str) -> dict:
        with self.lock:
            meta = self.get(tid)
            if meta["status"] != "receiving":
                raise TransferError("transfer not receiving")
            _, pp = self.paths(tid)
            meta["status"] = "cancelled"
            meta["updated"] = time.time()
            self.save(meta)
            pp.unlink(missing_ok=True)
            return meta


class Handler(oc_http.HeaderDeadlineMixin, BaseHTTPRequestHandler):
    receiver: Receiver = None
    token = ""
    throttle = sec.FailureThrottle()
    timeout = 120  # per-socket idle timeout; stops slow clients pinning threads
    server_version = "OmniCortexTransfer/" + VERSION
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
            self.close_connection = True
            self._json(429, {"error": "too many failed attempts; try later"})
            return False
        if token_ok(self.headers.get("Authorization"), self.token):
            if self.token:
                self.throttle.succeed(addr)
            return True
        self.close_connection = True
        self.throttle.fail(addr)
        self._json(401, {"error": "unauthorized"})
        return False

    def _length(self, limit: int) -> int:
        return oc_http.content_length(self, limit)

    def _read_json(self) -> dict:
        data = json.loads(oc_http.read_body(self, MAX_JSON, total_timeout=120) or b"{}")
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
                path = self.receiver._destination(meta)
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != meta["size"]
                ):
                    raise TransferError("completed transfer file is missing or changed")
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
                self.close_connection = True
                return self._json(404, {"error": "not found"})
            mm = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", self.headers.get("Content-Range", ""))
            if not mm:
                self.close_connection = True
                return self._json(400, {"error": "Content-Range required"})
            start, end, total = map(int, mm.groups())
            n = end - start + 1
            if n <= 0 or n > MAX_CHUNK:
                self.close_connection = True
                return self._json(413, {"error": f"chunk must be 1..{MAX_CHUNK} bytes"})
            if n != self._length(MAX_CHUNK):
                self.close_connection = True
                return self._json(400, {"error": "length mismatch"})
            body = oc_http.read_body(self, MAX_CHUNK, required=True, total_timeout=120)
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
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = request(base, "GET", f"/v1/transfers/{tid}", token)
    url = base.rstrip("/") + f"/v1/transfers/{tid}/download"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )

    final = out_dir / safe_name(meta["name"])
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final.name}.",
        suffix=".ns-incoming",
        dir=str(out_dir),
    )
    tmp = Path(tmp_name)
    digest = hashlib.sha256()
    received = 0

    try:
        with os.fdopen(fd, "wb") as f:
            with urllib.request.urlopen(
                req, timeout=300, context=_ssl_for(url)
            ) as response:
                for block in iter(lambda: response.read(CHUNK), b""):
                    received += len(block)
                    f.write(block)
                    digest.update(block)

            f.flush()
            os.fsync(f.fileno())

        expected_size = meta.get("size")
        if type(expected_size) is int and received != expected_size:
            raise TransferError(
                f"download size mismatch: expected {expected_size}, received {received}"
            )

        expected_sha = meta.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or not SHA_RE.fullmatch(expected_sha)
            or digest.hexdigest() != expected_sha
        ):
            raise TransferError("downloaded file sha256 mismatch")

        os.replace(tmp, final)

        try:
            directory_fd = os.open(
                out_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

        return meta

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


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
    s.add_argument("--max-threads", type=int, default=32, help="maximum concurrent HTTP handlers (1..256)")
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
    c.add_argument("--state", default=".omnicortex-transfer-watch.json")
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
        srv = server_class((bind_host, a.port), Handler, max_threads=a.max_threads)
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
