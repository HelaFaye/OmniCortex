"""Work distribution (scripts/work_log.py), activation stats (neuron_review.activation_stats) and Studio's /work."""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import work_log as W  # noqa: E402
import neuron_review as nr  # noqa: E402


def test_record_read_and_breakdown(tmp_path):
    log = W.WorkLog(tmp_path)
    now = time.time()
    log.record("chat", "big", model="big", device="GPU0", prompt_tokens=100, completion_tokens=50, seconds=2,
               gen_ms=1000, prompt_ms=200, t=now - 60)
    log.record("task", "ep:fast", device="GPU1", host="box", prompt_tokens=10, completion_tokens=30, seconds=1,
               t=now - 30)
    log.record("task", "ep:fast", device="GPU1", host="box", error="boom", seconds=3, t=now - 20)
    log.record("job", "testqa", seconds=100, ok=True, t=now - 10 * 86400)          # outside 24h
    with pytest.raises(ValueError):
        log.record("nap", "x")
    b = W.breakdown(log, "24h")
    assert b["totals"] == {"n": 3, "failed": 1, "prompt_tokens": 110, "completion_tokens": 80, "busy_s": 6.0}
    dev = {r["key"]: r for r in b["by_device"]}
    gpu0, gpu1 = dev["this computer · GPU0"], dev["box · GPU1"]
    assert gpu0["gen_tok_s"] == 50.0 and gpu0["prompt_tok_s"] == 500.0      # from the server's timings
    assert gpu1["n"] == 2 and gpu1["failed"] == 1 and gpu1["gen_tok_s"] == 7.5  # tokens over wall time
    assert gpu1["share_work"] == pytest.approx(2 / 3, abs=1e-3)
    assert gpu0["share_tokens"] == pytest.approx(150 / 190, abs=1e-3)
    assert {r["key"] for r in b["by_kind"]} == {"chat", "task"}
    assert sum(sum(s) for s in b["timeline"]["series"].values()) == 190
    assert W.breakdown(log, "all")["totals"]["n"] == 4


def test_hardware_samples(tmp_path):
    log = W.WorkLog(tmp_path)
    t = time.time() - 100
    for i in range(5):
        log.sample([{"id": "gpu0", "name": "Card", "util": 20 * i, "temp_c": 50 + i, "vram_used_mb": 1000,
                     "vram_total_mb": 8000, "power_w": 100}, {"name": "no id"}], t=t + i * 10)
    log.sample([], t=t)                                     # nothing to record
    h = W.breakdown(log, "1h")["hardware"]["this computer · Card"]
    assert h["util"] == [0, 20, 40, 60, 80] and h["summary"]["peak_temp"] == 54
    assert h["summary"]["avg_util"] == 40.0 and h["summary"]["energy_wh"] == pytest.approx(100 * 40 / 3600, abs=0.1)


def test_prune_and_since(tmp_path):
    log = W.WorkLog(tmp_path, keep_days=5)
    old = tmp_path / "work-2000-01-01.jsonl"
    tmp_path.mkdir(exist_ok=True)
    old.write_text("{}\n")
    log.record("chat", "m")
    assert not old.exists()
    assert W.parse_since("all") == 0 and abs(W.parse_since("2h") - (time.time() - 7200)) < 2
    with pytest.raises(ValueError):
        W.parse_since("soon")


def test_llama_timings():
    assert W.llama_timings({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}) == \
        {"prompt_tokens": 3, "completion_tokens": 4}
    t = W.llama_timings({"timings": {"prompt_n": 8, "prompt_ms": 16, "predicted_n": 7, "predicted_per_second": 14}})
    assert t == {"prompt_tokens": 8, "prompt_ms": 16, "completion_tokens": 7, "gen_ms": 500.0}


def test_activation_stats(tmp_path):
    store = nr.ReviewStore(tmp_path)
    rng = np.random.default_rng(0)
    for i in range(20):
        v = rng.normal(1.0, 0.05, (3, 16))
        if i % 2:
            v[1, 4] += 2.0
        store.record("m", v, source="t", kind="test", verdict="wrong" if i % 2 else "correct")
    s = nr.activation_stats(store, "m", cols=8)
    assert (s["n"], s["layers"], s["bins"]) == (20, 3, 16)
    assert len(s["heatmap"]) == 3 and len(s["heatmap"][0]) == 8
    pl = s["per_layer"]
    assert all(pl["p5"][i] <= pl["p50"][i] <= pl["p95"][i] for i in range(3))
    d = np.array(s["wrong_minus_correct"])
    assert np.unravel_index(d.argmax(), d.shape) == (1, 2) and d.max() > 0.9   # bin 4 of 16 -> column 2 of 8
    assert sum(s["histogram"]["counts"]) == 20 * 3 * 16
    only = nr.activation_stats(store, "m", verdicts=["correct"])
    assert only["n"] == 10 and "wrong_minus_correct" not in only
    assert nr.activation_stats(store, "nobody") == {"model": "nobody", "n": 0}


@pytest.fixture()
def srv(tmp_path):
    import test_studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    yield base, test_studio
    try:
        next(gen)
    except StopIteration:
        pass


def test_studio_work(srv):
    base, ts = srv
    req = urllib.request.Request(base + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"model": "coder-gguf/coder-7b-q4_k_m", "stream": True,
                                                  "messages": [{"role": "user", "content": "x"}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    code, b = ts.call(base, "/api/work?since=1h")
    assert code == 200 and b["totals"]["n"] == 1
    row = b["by_worker"][0]
    assert row["key"] == "coder-gguf/coder-7b-q4_k_m" and row["completion_tokens"] == 7 and row["gen_tok_s"] == 42.0
    assert b["by_kind"][0]["key"] == "chat"
    code, page = ts.call(base, "/work")
    assert code == 200 and "Work distribution" in page
    code, m = ts.call(base, "/api/work/neurons")
    assert code == 200 and m == {"models": []}
    code, e = ts.call(base, "/api/work?since=soon")
    assert code == 400


def test_studio_predict_measures_and_github_sharing(srv, tmp_path, monkeypatch):
    base, ts = srv
    import score_cards
    monkeypatch.setattr(score_cards, "HOME", tmp_path / "home")
    monkeypatch.delenv("OC_GITHUB_TOKEN", raising=False)
    code, p = ts.call(base, "/api/work/predict")
    assert code == 200 and p["samples"] == {"this computer": 0, "scoreboard": 0}
    assert {r["model"] for r in p["rows"]} >= {"coder-gguf/coder-7b-q4_k_m"}
    cell = next(iter(p["rows"][0]["devices"].values()))
    assert cell["low"] <= cell["tok_s"] <= cell["high"]
    code, m = ts.call(base, "/api/work/measures")
    assert code == 200 and m["weights"] == {}
    code, e = ts.call(base, "/api/settings/sharing", {"endpoint": "github:not a repo"})
    assert code == 400
    code, ok = ts.call(base, "/api/settings/sharing", {"endpoint": "github:acme/board", "github_token": "ghp_" + "x" * 36})
    assert code == 200 and ok["settings"]["endpoint"] == "github:acme/board"
    assert (tmp_path / "home" / "github.token").stat().st_mode & 0o077 == 0
    code, g = ts.call(base, "/api/settings/sharing")
    assert g["github_token_set"] and "ghp_" not in json.dumps(g)
    ts.call(base, "/api/settings/sharing", {"github_token": ""})
    assert not ts.call(base, "/api/settings/sharing")[1]["github_token_set"]
