#!/usr/bin/env python3
"""OmniCortex remote tuning worker.

Keeps source GGUFs and generated candidates on the remote machine. The API is
metadata-only and enforces atomic output, checksums, bounded concurrency,
filesystem capacity, cancellation, and persistent job state.

Same security rules as ns_transfer.py: a non-loopback bind needs a token, and
TLS unless --allow-plaintext (VPN / reverse proxy) is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_security as sec  # noqa: E402

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def scale_tag(scale: float) -> str:
    if scale < 1:
        return f"supp{int(round(abs(scale) * 1000)):03d}"
    if scale > 1:
        return f"amp{int(round(scale * 1000)):03d}"
    return "base100"


class Store:
    def __init__(self, root, source, profile, suppressor, evaluator="", max_workers=1, token=""):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = Path(source).resolve()
        self.profile = Path(profile).resolve()
        self.suppressor = Path(suppressor).resolve()
        self.evaluator = evaluator
        self.token = token
        self.jobs: dict = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.state = self.root / "worker_state.json"
        self._load()

    def _load(self):
        try:
            self.jobs = json.loads(self.state.read_text()).get("jobs", {}) if self.state.is_file() else {}
        except Exception:
            self.jobs = {}

    def save(self):
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"jobs": self.jobs}, indent=2))
        os.replace(tmp, self.state)

    def submit(self, scale, job_id=None, timeout=3600, nonce=None):
        jid = job_id or str(uuid.uuid4())
        if not JOB_ID_RE.match(jid):
            raise ValueError("bad job_id")
        scale = float(scale)
        if not (-10.0 <= scale <= 10.0):
            raise ValueError("scale out of range")
        rec = {"job_id": jid, "scale": scale, "status": "queued", "submitted": time.time()}
        if nonce is not None:
            nonce = str(nonce)
            if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", nonce):
                raise ValueError("bad nonce")
            rec["nonce"] = nonce          # echoed inside the signed result: binds it to this request
        with self.lock:
            self.jobs[jid] = rec
            self.save()
        self.pool.submit(self.run, jid, max(1, min(int(timeout), 86400)))
        return rec

    def run(self, jid, timeout):
        with self.lock:
            self.jobs[jid]["status"] = "running"
            self.save()
            scale = float(self.jobs[jid]["scale"])
        out = self.root / f"{self.source.stem}-{scale_tag(scale)}.gguf"
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            need = self.source.stat().st_size * 2
            if shutil.disk_usage(self.root).free < need:
                raise RuntimeError(f"insufficient free space; need {need / 2**30:.2f} GiB")
            if out.is_file() and out.stat().st_size == self.source.stat().st_size:
                result = {"status": "completed", "scale": scale, "model": str(out),
                          "sha256": sha256(out), "resumed": True}
            else:
                cmd = [os.environ.get("PYTHON", sys.executable), str(self.suppressor),
                       "--gguf", str(self.source), "--h_neurons", str(self.profile),
                       "--scale", str(scale), "--out", str(tmp)]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if p.returncode != 0:
                    raise RuntimeError((p.stderr or p.stdout)[-4000:])
                if not tmp.is_file():
                    raise RuntimeError("suppressor produced no output")
                os.replace(tmp, out)
                result = {"status": "completed", "scale": scale, "model": str(out),
                          "sha256": sha256(out), "resumed": False}
            result["source_sha256"] = self.source_sha()
            result["profile_sha256"] = sha256(self.profile)
            if self.evaluator:
                result.update(self._evaluate(out, scale, timeout))
            with self.lock:
                self.jobs[jid].update(result)
                self.save()
        except Exception as e:
            tmp.unlink(missing_ok=True)
            with self.lock:
                self.jobs[jid].update({"status": "failed", "error": str(e)})
                self.save()

    _source_sha: str | None = None

    def source_sha(self) -> str:
        if self._source_sha is None:
            self._source_sha = sha256(self.source)
        return self._source_sha

    def _evaluate(self, out: Path, scale: float, timeout: int) -> dict:
        # The evaluator command comes from the operator's own command line, never
        # from the API; the model path is passed via env as well as {model}.
        env = os.environ.copy()
        env.update({"NS_MODEL": str(out), "NS_SCALE": str(scale), "NS_PROFILE": str(self.profile)})
        cmd = self.evaluator.format(model=str(out), scale=scale, profile=str(self.profile))
        ep = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        if ep.returncode != 0:
            raise RuntimeError((ep.stderr or ep.stdout)[-4000:])
        text = ep.stdout.strip()
        for i in range(len(text) - 1, -1, -1):
            if text[i] == "{":
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    pass
        raise RuntimeError("evaluator returned no JSON")

    def get(self, jid):
        with self.lock:
            j = dict(self.jobs.get(jid, {}))
        # Signed with a key derived from the worker token, so the controller can
        # tell a result the worker produced from one altered on the way (a
        # TLS-terminating proxy, a plaintext VPN hop, a cache).
        return sec.sign_result(self.token, j) if j else j


def app(store: Store, token: str):
    throttle = sec.FailureThrottle()

    class H(BaseHTTPRequestHandler):
        timeout = 120

        def auth(self):
            addr = self.client_address[0]
            if throttle.blocked(addr):
                self.sendj({"error": "too many failed attempts"}, 429)
                return False
            if not sec.bearer_ok(self.headers.get("Authorization"), token):
                throttle.fail(addr)
                self.sendj({"error": "unauthorized"}, 401)
                return False
            return True

        def sendj(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            if n < 0 or n > sec.MAX_JSON_BYTES:
                raise ValueError("body too large")
            d = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(d, dict):
                raise ValueError("JSON object expected")
            return d

        def do_GET(self):
            if not self.auth():
                return
            if self.path == "/health":
                self.sendj({"ok": True, "service": "neuronscope-tuning-worker",
                            "source": store.source.name, "profile": store.profile.name})
                return
            if self.path.startswith("/api/jobs/"):
                jid = self.path.rsplit("/", 1)[-1]
                j = store.get(jid)
                self.sendj(j or {"error": "not-found"}, 200 if j else 404)
                return
            self.sendj({"error": "not-found"}, 404)

        def safe_model_path(self, path):
            p = Path(path).resolve()
            if store.root not in p.parents or p.suffix != ".gguf" or p == store.source:
                raise ValueError("path must be a generated .gguf inside the worker root")
            return p

        def do_POST(self):
            if not self.auth():
                return
            try:
                data = self.body()
                if self.path == "/api/models/delete":
                    p = self.safe_model_path(data["path"])
                    p.unlink(missing_ok=True)
                    self.sendj({"ok": True, "path": str(p)})
                    return
                if self.path != "/api/jobs":
                    self.sendj({"error": "not-found"}, 404)
                    return
                if "scale" not in data:
                    self.sendj({"error": "scale-required"}, 400)
                    return
                rec = store.submit(data["scale"], data.get("job_id"), int(data.get("timeout", 3600)), data.get("nonce"))
                self.sendj(sec.sign_result(token, rec), 202)
            except (ValueError, KeyError, TypeError) as e:
                self.sendj({"error": str(e)}, 400)

        def log_message(self, *a):
            pass

    return H


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--suppressor", required=True)
    p.add_argument("--evaluator", default="", help="shell command printing a JSON score; {model} {scale} {profile}")
    p.add_argument("--workers", type=int, default=1)
    sec.add_server_security_args(p)
    a = p.parse_args(argv)
    token = sec.resolve_token(a.token, a.token_file)
    tls = bool(a.tls_cert and a.tls_key)
    try:
        for w in sec.check_bind(a.host, token, tls=tls, allow_plaintext=a.allow_plaintext):
            print("warning:", w, file=sys.stderr)
    except sec.SecurityConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    store = Store(a.root, a.source, a.profile, a.suppressor, a.evaluator, a.workers, token)
    srv = ThreadingHTTPServer((a.host, a.port), app(store, token))
    if tls:
        srv.socket = sec.server_ssl_context(a.tls_cert, a.tls_key).wrap_socket(
            srv.socket, server_side=True, do_handshake_on_connect=False)
    print(f"OmniCortex tuning worker: {'https' if tls else 'http'}://{a.host}:{a.port}/health")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
