"""Integrated options, batch 2: quantization mask, token jitter, cross-layer hijack, neuron decay, neuro-fuzzing."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import neuron_decay  # noqa: E402
import quant_mask  # noqa: E402

DAY = 86400


def test_quant_mask_plan(tmp_path):
    h = tmp_path / "h_neurons.json"
    h.write_text(json.dumps({"n_layers": 10, "by_layer": {"7": [1, 2, 3, 4], "2": [5, 6], "9": [1]}}))
    risk = quant_mask.risk_from_h(h)
    p = quant_mask.plan(risk, 10, high_fraction=0.2, low_fraction=0.3)
    assert p["high"]["layers"] == [2, 7] and len(p["low"]["layers"]) == 3 and 9 not in p["low"]["layers"]
    assert not set(p["low"]["layers"]) & {2, 7, 9}
    assert p["rules"][0].startswith("blk\\.2\\.(ffn_down|") and p["rules"][0].endswith("=q8_0")
    cmd = quant_mask.command(p, "in.gguf", "out.gguf", "Q4_K_M", "llama-quantize", tmp_path / "t.txt", "imat.dat")
    assert cmd[:3] == ["llama-quantize", "--imatrix", "imat.dat"] and cmd[-3:] == ["in.gguf", "out.gguf", "Q4_K_M"]
    assert (tmp_path / "t.txt").read_text().count("\n") == len(p["rules"])
    with pytest.raises(ValueError):
        quant_mask.plan(risk, 10, high="q9_9")
    with pytest.raises(ValueError):
        quant_mask.plan(risk, 10, high_fraction=0.8, low_fraction=0.5)
    assert quant_mask.plan({}, 4)["high"]["layers"] == []            # no risk anywhere: nothing promoted
    assert quant_mask.parse_layers("1,3-5") == [1, 3, 4, 5]


@pytest.mark.skipif(not (ROOT / "build" / "llama.cpp" / "build" / "bin" / "llama-quantize").exists(),
                    reason="no local llama-quantize build")
def test_quant_mask_rules_parse_in_llama_quantize(tmp_path):
    """llama-quantize accepts the rules file (it fails later, on the missing model, not on the rules)."""
    import subprocess
    p = quant_mask.plan({3: 1.0}, 8)
    cmd = quant_mask.command(p, str(tmp_path / "missing.gguf"), str(tmp_path / "o.gguf"), "Q4_K_M",
                             str(ROOT / "build" / "llama.cpp" / "build" / "bin" / "llama-quantize"), tmp_path / "r.txt", None)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert "tensor-type" not in (r.stdout + r.stderr).lower() or "invalid" not in (r.stdout + r.stderr).lower()


def test_decay_schedule(tmp_path):
    assert neuron_decay.live_scale(1.0, 0, 3 * DAY) == 1.0                         # within the grace period
    s1 = neuron_decay.live_scale(1.0, 0, (7 + 14) * DAY)
    assert s1 == pytest.approx(0.75)                                              # one half-life toward the 0.5 floor
    assert neuron_decay.live_scale(1.0, 0, 400 * DAY) == pytest.approx(0.5, abs=1e-3)
    assert neuron_decay.live_scale(0.0, 0, 400 * DAY) == 0.0
    cfg = {"grace_days": 7, "half_life_days": 14, "floor": 0.5}
    log = neuron_decay.Log(tmp_path / "decay.json")
    assert log.decide("m", 1.0, cfg, now=0) is None                                # clock starts
    assert log.decide("m", 1.0, cfg, now=21 * DAY) == pytest.approx(0.75)
    log.trigger("m", "risky reply", now=22 * DAY)
    assert log.decide("m", 1.0, cfg, now=22 * DAY + 60) == 1.0                     # needed again: full strength
    kinds = [e["kind"] for e in neuron_decay.Log(tmp_path / "decay.json").model("m")["events"]]
    assert kinds == ["decay", "trigger", "restore"]


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")


@pytest.fixture(scope="module")
def M(tmp_path_factory):
    import interventions as IV
    from tiny_models import make_tiny_llama
    hf = make_tiny_llama(tmp_path_factory.mktemp("tiny"), layers=3, hidden=32, ff=64)
    return IV.load(str(hf), "cpu", "float32")


def test_jitter_finds_brittle_layers_and_makes_pairs(M):
    import model_lab
    items = [{"prompt": "the cat sat on the", "answer": "mat"}, {"prompt": "the dog ran to the", "answer": "cat"}]
    quiet = model_lab.jitter(M, items, sigma=0.0, samples=2)
    assert all(r["flip_rate"] == 0 and abs(r["mean_logprob_drop"]) < 1e-4 for r in quiet["layers"]) and quiet["dpo"] == []
    loud = model_lab.jitter(M, items, sigma=5.0, samples=3)
    assert len(loud["layers"]) == 3 and any(r["flip_rate"] > 0 for r in loud["layers"])
    assert loud["brittle"] and all(0 <= L < 3 for L in loud["brittle"])
    for p in loud["dpo"]:
        assert p["chosen"][0]["content"] in ("mat", "cat") and p["rejected"][0]["content"] != p["chosen"][0]["content"]


def test_hijack_back_patching(M):
    import model_lab
    same = model_lab.hijack(M, "the cat sat on the mat", at=1, source=1, mix=1.0)
    assert same["kl"] < 1e-6 and not same["changed_top"]                          # its own state: nothing changes
    r = model_lab.hijack(M, "the cat sat on the mat", at=0, source=2, mix=1.0, positions="all")
    assert r["kl"] > 0 and len(r["applied"]) == len(r["positions"]) and r["before"] and r["after"]
    import numpy as np
    never = model_lab.hijack(M, "the cat sat", at=0, source=2, trigger=np.ones(32).tolist(), threshold=1e9)
    assert never["applied"] == [] and never["kl"] < 1e-6
    with pytest.raises(ValueError):
        model_lab.hijack(M, "x", at=0, source=9)


def test_neuro_fuzz_end_to_end(tmp_path):
    import neuro_fuzz

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            q = body["messages"][-1]["content"]
            if "You write test questions" in q:
                text = json.dumps([
                    {"kind": "qa", "prompt": "What is the capital city of Mongolia? Respond with the answer only.", "aliases": ["Ulaanbaatar"]},
                    {"kind": "reasoning", "prompt": "A cyclist rides 60 km in 3 hours. Average speed in km/h?", "answer": "20", "answer_type": "number"},
                    {"kind": "qa", "prompt": "What is the capital city of Australia? Respond with the answer only.", "aliases": ["Canberra"]},
                    {"kind": "bogus", "prompt": "x"}])
            elif "Mongolia" in q:
                text = "Ulaanbaatar"
            else:
                text = "I think it is 25."
            out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1@m"
    res = tmp_path / "tq.json"
    res.write_text(json.dumps({"raw": {"m": [{"id": "q-canberra", "verdict": "wrong"}, {"id": "q-ottawa", "verdict": "correct"}]}}))
    try:
        assert neuro_fuzz.main(["--results", str(res), "--fuzzer", url, "--verify", url, "--out", str(tmp_path / "fz")]) == 0
    finally:
        srv.shutdown()
    kept = [json.loads(l) for l in (tmp_path / "fz" / "fuzz.jsonl").read_text().splitlines()]
    unv = [json.loads(l) for l in (tmp_path / "fz" / "unverified.jsonl.txt").read_text().splitlines()]
    assert [k["aliases"] for k in kept] == [["Ulaanbaatar"]] and kept[0]["id"].startswith("fz-")
    assert kept[0]["fuzz"]["seed"] == "q-canberra"
    assert [u["answer"] for u in unv] == ["20"]                                   # the verifier disagreed
    import testqa                                                                  # the folder loads as a bank
    assert [t["id"] for t in testqa.load_tasks([str(tmp_path / "fz")], None, None, 0)] == [kept[0]["id"]]
