"""Where recorded activations go (scripts/activation_store.py), Studio's collection by default, and its settings."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import activation_store as A  # noqa: E402
import neuron_review as nr  # noqa: E402


def vec(seed=0):
    return np.random.default_rng(seed).normal(1, 0.1, (3, 8))


def test_local_and_folder(tmp_path):
    s = A.Sink({"store": "local"}, tmp_path / "rev", tmp_path / "home")
    assert s.record("m", vec(), source="live", subjects=["code"], risk=0.3) == "local"
    rows = nr.ReviewStore(tmp_path / "rev").query("m")
    assert len(rows) == 1 and rows[0]["subjects"] == ["code"] and rows[0]["risk"] == 0.3
    f = A.Sink({"store": "folder", "folder": str(tmp_path / "nas")}, tmp_path / "rev", tmp_path / "home")
    f.record("m", vec(1))
    assert len(nr.ReviewStore(tmp_path / "nas").query("m")) == 1 and f.status()["bytes"] > 0
    with pytest.raises(ValueError):
        A.Sink({"store": "cloud"})


def test_memory_only(tmp_path):
    s = A.Sink({"store": "memory", "spool": 3}, None, tmp_path / "home")
    for i in range(5):
        assert s.record("m", vec(i)) == "memory"
    assert len(s.export_memory()) == 3 and s.stats["dropped"] == 2
    assert not (tmp_path / "home" / "activation-spool.jsonl").exists()
    v = A.decode_vec(s.export_memory()[0]["vec"])
    assert v.shape == (3, 8) and v.dtype == np.float16


def test_forward_to_another_studio_spools_and_retries(tmp_path):
    got, up = [], {"ok": False}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not up["ok"]:
                self.send_response(503); self.end_headers(); return
            assert self.headers["Authorization"] == "Bearer tok"
            got.extend(body["items"])
            out = b'{"stored": 1}'
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = {"store": "studio", "studio_url": f"http://127.0.0.1:{srv.server_port}", "studio_token": "tok"}
    try:
        s = A.Sink(cfg, None, tmp_path / "home")
        assert s.record("m", vec(), source="live") == "spool"            # the other Studio is down: kept
        assert (tmp_path / "home" / "activation-spool.jsonl").exists() and s.status()["spooled_now"] == 1
        again = A.Sink(cfg, None, tmp_path / "home")                      # survives a restart
        assert again.status()["spooled_now"] == 1
        up["ok"] = True
        assert again.retry() == 1 and len(got) == 1 and again.status()["spooled_now"] == 0
        assert again.record("m", vec(1)) == "studio" and len(got) == 2
    finally:
        srv.shutdown()
    st = nr.ReviewStore(tmp_path / "recv")
    assert A.ingest(st, got) == 2 and st.query("m")[0].get("forwarded")


def test_size_cap_prunes_oldest(tmp_path):
    s = A.Sink({"store": "local", "max_gb": 0}, tmp_path / "rev", tmp_path / "home")
    for i in range(6):
        s.record("m", vec(i), t=1000 + i)
        time.sleep(0.01)
    one = next((tmp_path / "rev").glob("*/vecs/*.npy")).stat().st_size
    s.cfg["max_gb"] = (one * 3.5) / 2**30
    assert s.prune() == 3
    left = nr.ReviewStore(tmp_path / "rev").query("m")
    assert len(left) == 3 and min(r["t"] for r in left) == 1003            # the oldest went first


def test_studio_collects_by_default(tmp_path, monkeypatch):
    import test_studio
    studio = test_studio.studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        code, a = test_studio.call(base, "/api/settings/activations")
        assert code == 200 and a["settings"]["collect"] is True and a["settings"]["store"] == "local"
        assert a["status"]["method"] is None and "llama-cett-dump" in a["status"]["why_not"]

        class Prof:
            def profile(self, messages, text):
                return {"cett": np.ones((2, 600), np.float32), "n_tokens": 3}
        monkeypatch.setattr(studio, "_scorer", lambda model, clf: Prof())
        studio.STATE["cett"] = "/bin/true"
        studio.STATE["act_sink"] = None
        m = studio.find_model("coder-gguf/coder-7b-q4_k_m")
        studio.note_reply(m, [{"role": "user", "content": "hi"}], "hello")      # no classifier: still profiled
        job = studio.SCORE_JOBS.get_nowait()
        assert job[1] is None
        studio.SCORE_JOBS.put(job)
        t = threading.Thread(target=studio.score_worker, daemon=True)
        t.start()
        for _ in range(50):
            if studio.review_store().query(m["id"]):
                break
            time.sleep(0.1)
        assert len(studio.review_store().query(m["id"])) == 1
        code, r = test_studio.call(base, "/api/settings/activations", {"store": "folder", "folder": ""})
        assert code == 400
        code, r = test_studio.call(base, "/api/settings/activations", {"collect": False, "every": 3})
        assert code == 200 and r["settings"]["collect"] is False and r["settings"]["every"] == 3
        code, r = test_studio.call(base, "/api/settings/activations", {"studio_token": "secret", "store": "studio",
                                                                        "studio_url": "https://box:7870"})
        assert r["settings"]["studio_token"] == "set" and "secret" not in json.dumps(r)
        code, r = test_studio.call(base, "/api/review/ingest", {"items": [{"model": "x", "vec": A.encode_vec(vec()),
                                                                           "meta": {"source": "live"}}]})
        assert code == 200 and r["stored"] == 1
        code, s = test_studio.call(base, "/api/setup")
        assert s["fields"]["host"]["sensitive"] and s["fields"]["idle_ttl"]["sensitive"] is None
        _, page = test_studio.call(base, "/settings")
        assert 'id="activations"' in page and "data-locked" in page and "nsLockAll" in page
    finally:
        studio.STATE["cett"] = None
        try:
            next(gen)
        except StopIteration:
            pass
