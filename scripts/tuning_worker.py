#!/usr/bin/env python3
"""OmniCortex remote tuning worker.

Keeps source GGUFs and generated candidates on the remote machine. The API is
metadata-only and enforces atomic output, checksums, bounded concurrency,
filesystem capacity, cancellation, and persistent job state.

Same security rules as oc_transfer.py: a non-loopback bind needs a token, and
TLS unless --allow-plaintext (VPN / reverse proxy) is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oc_security as sec  # noqa: E402
import oc_http  # noqa: E402

JOB_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


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
        self.max_workers = max(1, min(int(max_workers), 64))
        self.pending_slots = threading.BoundedSemaphore(self.max_workers * 2)
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.closed = False
        self.state = self.root / "worker_state.json"
        self._load()

    @staticmethod
    def _valid_loaded_job(key, job):
        if not isinstance(key, str) or not JOB_ID_RE.fullmatch(key):
            return False
        if not isinstance(job, dict) or job.get("job_id") != key:
            return False

        scale = job.get("scale")
        if type(scale) not in (int, float):
            return False
        if not math.isfinite(scale) or not -10.0 <= scale <= 10.0:
            return False

        status = job.get("status")
        if not isinstance(status, str) or status not in {
            "queued", "running", "interrupted", "completed", "failed", "cancelled"
        }:
            return False

        submitted = job.get("submitted", 0)
        if type(submitted) not in (int, float) or not math.isfinite(submitted):
            return False

        nonce = job.get("nonce")
        if nonce is not None:
            if not isinstance(nonce, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{16,128}", nonce
            ):
                return False

        return True

    def _load(self):
        loaded = {}
        try:
            if not self.state.is_symlink() and self.state.is_file():
                document = json.loads(self.state.read_text(encoding="utf-8"))
                candidate = document.get("jobs", {}) if isinstance(document, dict) else {}
                if isinstance(candidate, dict):
                    loaded = candidate
        except (OSError, ValueError, TypeError, UnicodeError):
            loaded = {}

        self.jobs = {
            key: job for key, job in loaded.items()
            if self._valid_loaded_job(key, job)
        }
        changed = len(self.jobs) != len(loaded)

        for job in self.jobs.values():
            if job.get("status") in ("queued", "running"):
                job.update(
                    status="interrupted",
                    error="worker restarted before this job completed",
                    updated=time.time(),
                )
                changed = True

        before_trim = len(self.jobs)
        self._trim_jobs_locked()
        changed = changed or len(self.jobs) != before_trim

        if changed:
            self.save()

    def _trim_jobs_locked(self):
        terminal = sorted(
            ((key, job) for key, job in self.jobs.items()
             if job.get("status") not in ("queued", "running")),
            key=lambda pair: float(pair[1].get("submitted", 0) or 0))
        excess = len(self.jobs) - 1000
        for key, _job in terminal[:max(0, excess)]:
            self.jobs.pop(key, None)

    def save(self):
        import tempfile

        fd, tmp_name = tempfile.mkstemp(
            prefix=".worker-state.", suffix=".tmp", dir=str(self.root)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), 0o600)
                json.dump({"jobs": self.jobs}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(tmp, self.state)
            os.chmod(self.state, 0o600)

            # Persist the directory entry where the platform supports it.
            try:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                dir_fd = os.open(self.root, flags)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            tmp.unlink(missing_ok=True)

    def submit(self, scale, job_id=None, timeout=3600, nonce=None):
        jid = job_id or str(uuid.uuid4())
        if not isinstance(jid, str) or not JOB_ID_RE.fullmatch(jid):
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
        if not self.pending_slots.acquire(blocking=False):
            raise ValueError("worker queue is full; wait for a queued or running job to finish")
        inserted = False
        try:
            output_name = f"{self.source.stem}-{scale_tag(scale)}.gguf"
            with self.lock:
                if self.closed:
                    raise ValueError("tuning worker is shutting down")
                if jid in self.jobs:
                    raise ValueError("job_id has already been used")
                for old in self.jobs.values():
                    try:
                        old_scale = float(old.get("scale", 1))
                        old_output = f"{self.source.stem}-{scale_tag(old_scale)}.gguf"
                    except (TypeError, ValueError):
                        continue
                    if old_output != output_name:
                        continue
                    if old_scale != scale:
                        raise ValueError(
                            "scale collides with an existing output scale filename; "
                            "choose a scale that maps to a distinct tag"
                        )
                    if old.get("status") in ("queued", "running"):
                        raise ValueError(
                            "a job for this output scale is already queued or running"
                        )
                output_path = self.root / output_name
                exists, recorded_scale = self._existing_output_scale(output_path)
                if exists:
                    if recorded_scale is None:
                        raise ValueError(
                            "output filename already exists without valid scale provenance; "
                            "inspect or move the existing output and manifest"
                        )
                    if recorded_scale != scale:
                        raise ValueError(
                            "scale collides with an existing output scale filename; "
                            "choose a scale that maps to a distinct tag"
                        )

                self._trim_jobs_locked()
                self.jobs[jid] = rec
                inserted = True
                self.save()
            self.pool.submit(self.run, jid, max(1, min(int(timeout), 86400)))
            return rec
        except BaseException:
            try:
                if inserted:
                    with self.lock:
                        self.jobs.pop(jid, None)
                        try:
                            self.save()
                        except Exception as state_error:
                            print(
                                f"tuning worker: could not persist submit rollback "
                                f"for {jid}: {state_error}",
                                file=sys.stderr,
                            )
            finally:
                self.pending_slots.release()
            raise

    def _provenance_path(self, out: Path) -> Path:
        return out.with_name(out.name + ".ocmeta.json")

    def _existing_output_scale(self, out: Path) -> tuple[bool, float | None]:
        """Return whether output artifacts exist and their trusted scale, if known."""
        manifest_path = self._provenance_path(out)
        exists = (
            out.exists() or out.is_symlink()
            or manifest_path.exists() or manifest_path.is_symlink()
        )
        if not exists:
            return False, None

        try:
            if manifest_path.is_symlink() or not manifest_path.is_file():
                return True, None

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return True, None

            encoded = manifest.get("scale_hex")
            if not isinstance(encoded, str):
                return True, None

            scale = float.fromhex(encoded)
            if (
                not math.isfinite(scale)
                or not -10.0 <= scale <= 10.0
                or scale.hex() != encoded
            ):
                return True, None

            return True, scale
        except (OSError, ValueError, TypeError, OverflowError):
            return True, None

    def _input_provenance(self, scale: float) -> dict:
        return {
            "schema": 1,
            "scale_hex": float(scale).hex(),
            "source_sha256": self.source_sha(),
            "profile_sha256": sha256(self.profile),
            "suppressor_sha256": sha256(self.suppressor),
        }

    def _reusable_output_sha(self, out: Path, expected: dict) -> str | None:
        manifest_path = self._provenance_path(out)
        try:
            if (not out.is_file() or out.is_symlink()
                    or not manifest_path.is_file() or manifest_path.is_symlink()):
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return None
            if any(manifest.get(key) != value for key, value in expected.items()):
                return None
            output_digest = sha256(out)
            if manifest.get("output_sha256") != output_digest:
                return None
            return output_digest
        except (OSError, ValueError, TypeError):
            return None

    def _write_output_provenance(self, out: Path, expected: dict, output_sha: str) -> None:
        import tempfile

        target = self._provenance_path(out)
        record = {**expected, "output_sha256": output_sha}
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), 0o600)
                json.dump(record, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
            os.chmod(target, 0o600)
        finally:
            tmp.unlink(missing_ok=True)

    def run(self, jid, timeout):
        tmp = None
        try:
            with self.lock:
                job = self.jobs.get(jid)
                if job is None:
                    raise RuntimeError(f"job {jid!r} disappeared before execution")

                scale = float(job["scale"])
                out = self.root / f"{self.source.stem}-{scale_tag(scale)}.gguf"
                tmp = out.with_suffix(out.suffix + ".tmp")

                job.update(status="running", updated=time.time())
                self.save()

            provenance = self._input_provenance(scale)
            reused_sha = self._reusable_output_sha(out, provenance)

            if reused_sha is not None:
                result = {
                    "status": "completed",
                    "scale": scale,
                    "model": str(out),
                    "sha256": reused_sha,
                    "resumed": True,
                }
            else:
                need = self.source.stat().st_size * 2
                if shutil.disk_usage(self.root).free < need:
                    raise RuntimeError(
                        f"insufficient free space; need {need / 2**30:.2f} GiB"
                    )

                cmd = [
                    os.environ.get("PYTHON", sys.executable),
                    str(self.suppressor),
                    "--gguf", str(self.source),
                    "--h_neurons", str(self.profile),
                    "--scale", str(scale),
                    "--out", str(tmp),
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                if proc.returncode != 0:
                    raise RuntimeError((proc.stderr or proc.stdout)[-4000:])
                if not tmp.is_file():
                    raise RuntimeError("suppressor produced no output")
                if self._input_provenance(scale) != provenance:
                    raise RuntimeError(
                        "source, profile, or suppressor changed during generation; "
                        "output not published"
                    )

                os.replace(tmp, out)
                output_sha = sha256(out)
                self._write_output_provenance(out, provenance, output_sha)
                result = {
                    "status": "completed",
                    "scale": scale,
                    "model": str(out),
                    "sha256": output_sha,
                    "resumed": False,
                }

            result["source_sha256"] = provenance["source_sha256"]
            result["profile_sha256"] = provenance["profile_sha256"]
            result["suppressor_sha256"] = provenance["suppressor_sha256"]

            if self.evaluator:
                result.update(self._evaluate(out, scale, timeout))

            with self.lock:
                job = self.jobs.get(jid)
                if job is None:
                    raise RuntimeError(f"job {jid!r} disappeared before completion")
                job.update(result)
                job["updated"] = time.time()
                self.save()

        except Exception as exc:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

            with self.lock:
                job = self.jobs.get(jid)
                if job is not None:
                    job.update({
                        "status": "failed",
                        "error": str(exc)[:4000],
                        "updated": time.time(),
                    })
                    try:
                        self.save()
                    except Exception as state_error:
                        print(
                            f"tuning worker: could not persist failed state "
                            f"for {jid}: {state_error}",
                            file=sys.stderr,
                        )
        finally:
            self.pending_slots.release()

    def close(self):
        """Reject new jobs and wait for every accepted job to finish."""
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True)

    def source_sha(self) -> str:
        return sha256(self.source)

    def _evaluate(self, out: Path, scale: float, timeout: int) -> dict:
        # The evaluator command comes from the operator's own command line, never
        # from the API; the model path is passed via env as well as {model}.
        env = os.environ.copy()
        env.update({"OC_MODEL": str(out), "OC_SCALE": str(scale), "OC_PROFILE": str(self.profile)})
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

    class H(oc_http.HeaderDeadlineMixin, BaseHTTPRequestHandler):
        timeout = 120

        def auth(self):
            addr = self.client_address[0]
            if throttle.blocked(addr):
                self.close_connection = True
                self.sendj({"error": "too many failed attempts"}, 429)
                return False
            if not sec.bearer_ok(self.headers.get("Authorization"), token):
                self.close_connection = True
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
            d = json.loads(oc_http.read_body(self, sec.MAX_JSON_BYTES, total_timeout=120) or b"{}")
            if not isinstance(d, dict):
                raise ValueError("JSON object expected")
            return d

        def do_GET(self):
            if not self.auth():
                return
            if self.path == "/health":
                self.sendj({"ok": True, "service": "omnicortex-tuning-worker",
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
    p.add_argument("--max-threads", type=int, default=32,
                   help="maximum concurrent HTTP handlers (1..256)")
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
    srv = None
    try:
        srv = oc_http.BoundedThreadingHTTPServer(
            (a.host, a.port), app(store, token), max_threads=a.max_threads
        )
        if tls:
            srv.socket = sec.server_ssl_context(a.tls_cert, a.tls_key).wrap_socket(
                srv.socket, server_side=True, do_handshake_on_connect=False
            )
        print(
            f"OmniCortex tuning worker: "
            f"{'https' if tls else 'http'}://{a.host}:{a.port}/health"
        )
        srv.serve_forever()
    finally:
        try:
            if srv is not None:
                srv.server_close()
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
