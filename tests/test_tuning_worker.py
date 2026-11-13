"""Tuning worker results are signed, and the controller refuses anything else."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import adaptive_tuner  # noqa: E402
import oc_security as sec  # noqa: E402
import tuning_worker  # noqa: E402

TOKEN = "t" * 43


@pytest.fixture()
def worker(tmp_path):
    src = tmp_path / "model.gguf"
    src.write_bytes(b"GGUF" + b"\0" * 256)
    prof = tmp_path / "h.json"
    prof.write_text("{}")
    sup = tmp_path / "sup.py"
    sup.write_text("import sys,shutil\na=sys.argv\nshutil.copy(a[a.index('--gguf')+1], a[a.index('--out')+1])\n")
    store = tuning_worker.Store(tmp_path / "root", src, prof, sup, token=TOKEN)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), tuning_worker.app(store, TOKEN))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_signed_roundtrip(worker):
    c = adaptive_tuner.WorkerClient(worker, token=TOKEN)
    r = c.submit({"job_id": "j1", "scale": 0.5})
    assert sec.verify_result(TOKEN, r) and r["nonce"]
    done = c.wait("j1", poll=0.05)
    assert done["status"] == "completed" and done["scale"] == 0.5 and done["sha256"]


def test_sign_verify_primitives():
    s = sec.sign_result(TOKEN, {"a": 1, "b": [1, 2]})
    assert sec.verify_result(TOKEN, s)
    assert not sec.verify_result(TOKEN, {**s, "a": 2})
    assert not sec.verify_result("x" * 43, s)
    assert not sec.verify_result(TOKEN, {"a": 1})


class Tamper(BaseHTTPRequestHandler):
    """A man in the middle: forwards to the worker and edits, strips or replays results."""
    upstream = ""
    mode = ""
    saved = {}

    def _fwd(self, method, body=None):
        req = urllib.request.Request(self.upstream + self.path, data=body, method=method,
                                     headers={"Authorization": self.headers.get("Authorization", ""),
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        self._send(self._fwd("POST", self.rfile.read(int(self.headers["Content-Length"]))), 202)

    def do_GET(self):
        r = self._fwd("GET")
        if r.get("status") == "completed" and r.get("job_id") == "old":
            Tamper.saved["old"] = r
        if self.mode == "edit" and r.get("status") == "completed":
            r["sha256"] = "0" * 64                     # point the controller at a different file
        elif self.mode == "strip":
            r.pop("sig", None)
        elif self.mode == "replay" and r.get("job_id") == "new":
            r = Tamper.saved["old"]                    # a genuine signed result, for another job
        self._send(r)

    def log_message(self, *a):
        pass


@pytest.mark.parametrize("mode", ["edit", "strip", "replay"])
def test_tampered_results_are_refused(worker, mode):
    Tamper.upstream, Tamper.mode, Tamper.saved = worker, mode, {}
    mitm = ThreadingHTTPServer(("127.0.0.1", 0), Tamper)
    threading.Thread(target=mitm.serve_forever, daemon=True).start()
    try:
        c = adaptive_tuner.WorkerClient(f"http://127.0.0.1:{mitm.server_port}", token=TOKEN)
        if mode == "replay":
            Tamper.mode = ""
            c.submit({"job_id": "old", "scale": 0.3})
            c.wait("old", poll=0.05)
            Tamper.mode = "replay"
            c.submit({"job_id": "new", "scale": 0.7})
            with pytest.raises(RuntimeError, match="replayed"):
                c.get("new")
            return
        c.submit({"job_id": "j", "scale": 0.5})
        with pytest.raises(RuntimeError, match="signature"):
            c.wait("j", poll=0.05)
    finally:
        mitm.shutdown()
