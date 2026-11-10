"""Truth probes (scripts/truth_probe.py): data, training on every stage, the verdicts, a looped model's passes as
stages, and the GGUF backend against cett-dump when a build is available."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import truth_probe as TP  # noqa: E402

TRUE_W = ["blue", "red", "green", "river", "ocean"]
FALSE_W = ["one", "two", "three", "four", "five"]
QS = ["what is the capital", "who wrote hamlet", "what is the city", "what is the answer", "who is the model",
      "what is the number", "what is the river", "what is the ocean"]


def synthetic(n=48):
    rows = []
    for k in range(n):
        q = QS[k % len(QS)]
        w = (TRUE_W if k % 2 == 0 else FALSE_W)[(k // 2) % 5]
        rows.append({"question": q, "answer": w, "label": k % 2 == 0})
    return rows


def test_rows_from_banks(tmp_path):
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"prompt": "Capital of France?", "aliases": ["Paris"]},
        {"prompt": "Who wrote Hamlet?", "aliases": ["Shakespeare"]},
        {"prompt": "Capital of Australia?", "answer": "Canberra", "wrong": ["Sydney", "Melbourne"]},
        {"statement": "Water boils at 50 C at sea level.", "label": False},
        {"prompt": "Unknowable?", "aliases": ["x"], "expect_abstain": True}]))
    rows = TP.build_rows([str(tmp_path)])
    assert {"question": "Capital of Australia?", "answer": "Sydney", "label": False} in rows
    assert {"question": "", "answer": "Water boils at 50 C at sea level.", "label": False} in rows
    assert not any(r["question"] == "Unknowable?" for r in rows)
    borrowed = [r for r in rows if r["question"] == "Capital of France?" and not r["label"]]
    assert borrowed and borrowed[0]["answer"] != "Paris"
    assert not any(r["question"] == "Capital of France?" and not r["label"]
                   for r in TP.build_rows([str(tmp_path)], borrow=False))


def test_bundled_bank_is_well_formed():
    rows = TP.build_rows([str(ROOT / "qa" / "truth")], borrow=False)
    assert len(rows) >= 60 and 0.3 < sum(r["label"] for r in rows) / len(rows) < 0.7


torch_needed = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("torch", "transformers", "sentencepiece")),
    reason="needs torch, transformers, sentencepiece")


@pytest.fixture(scope="module")
def feat(tmp_path_factory):
    import torch
    from tiny_models import make_tiny_llama
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = make_tiny_llama(tmp_path_factory.mktemp("tiny"), layers=3, hidden=32, ff=64)
    m = AutoModelForCausalLM.from_pretrained(str(hf), dtype=torch.float32)
    return TP.HFFeatures(m, AutoTokenizer.from_pretrained(str(hf)), model_id="tiny")


@torch_needed
def test_train_check_and_reload(feat, tmp_path):
    probe = TP.train(feat, synthetic(), pool="last", C=1.0, framing="answer", method="logreg")
    assert probe.meta["framing"] == "answer" and probe.meta["method"] == "logreg"
    assert probe.stages == ["model.layers.0", "model.layers.1", "model.layers.2"]
    assert probe.auc[probe.best] > 0.9
    probe.save(tmp_path / "p.npz")
    probe = TP.Probe.load(tmp_path / "p.npz")
    r = TP.check(feat, probe, "what is the capital", "two", ["blue", "two"])
    assert r["verdict"] == "knows_but_says_otherwise" and r["best_alternative"] == "blue"
    assert r["answer"]["p_true"] < 0.35 and r["alternatives"][0]["p_true"] > 0.65
    assert r["alt_leads_at"] and all("alternative" in t for t in r["trajectory"])
    assert TP.check(feat, probe, "what is the capital", "red")["verdict"] == "consistent"
    assert TP.check(feat, probe, "what is the capital", "four")["verdict"] == "doubts_own_answer"


@torch_needed
def test_massmean_and_judge_framing(feat, tmp_path):
    mm = TP.train(feat, synthetic(), framing="answer", method="massmean")
    assert mm.auc[mm.best] > 0.9 and mm.meta["method"] == "massmean"
    assert TP.check(feat, mm, "what is the capital", "two", ["blue"])["verdict"] == "knows_but_says_otherwise"
    j = TP.train(feat, synthetic(), framing="judge")            # the default framing: the answer sits in the prompt
    assert j.meta["framing"] == "judge" and j.auc[j.best] > 0.6
    j.save(tmp_path / "j.npz")
    assert TP.Probe.load(tmp_path / "j.npz").meta["framing"] == "judge"


def test_frames():
    msgs, reply = TP.frame("Capital of France?", " Paris ", "judge", "hf")
    assert "Question: Capital of France?" in msgs[0]["content"] and "Answer: Paris" in msgs[0]["content"]
    assert reply == " The answer is"
    assert TP.frame("Capital of France?", "Paris", "answer", "gguf") == (TP.messages_for("Capital of France?"), "Paris")
    with pytest.raises(ValueError):
        TP.frame("q", "a", "vibes", "hf")


@torch_needed
def test_old_probes_read_the_answer(feat, tmp_path):
    p = TP.train(feat, synthetic(16), C=1.0, framing="answer", method="logreg")
    del p.meta["framing"]                                        # a probe saved before framings existed
    p.save(tmp_path / "old.npz")
    r = TP.check(feat, TP.Probe.load(tmp_path / "old.npz"), "what is the capital", "blue")
    assert r["verdict"] in TP_VERDICTS


@torch_needed
def test_one_sided_data_rejected(feat):
    with pytest.raises(ValueError):
        TP.train(feat, [r for r in synthetic() if r["label"]])


@torch_needed
def test_backend_mismatch_rejected(feat):
    probe = TP.train(feat, synthetic(16), C=1.0, framing="answer")
    probe.meta["backend"] = "gguf"
    with pytest.raises(ValueError):
        TP.check(feat, probe, "q", "a")


@torch_needed
def test_repeated_passes_are_separate_stages(feat):
    """A looped model runs a block of layers more than once; each pass is its own stage."""
    layers = feat.model.model.layers
    orig = layers[1].forward

    def twice(*a, **k):
        out = orig(*a, **k)
        h = out[0] if isinstance(out, tuple) else out
        for hook in layers[1]._forward_hooks.values():      # a second pass, as a looped model would run it
            hook(layers[1], a, out)
        return out if isinstance(out, tuple) else h
    layers[1].forward = twice
    try:
        names, vecs = feat(TP.messages_for("what is the capital"), " blue")
    finally:
        layers[1].forward = orig
    assert names == ["model.layers.0", "model.layers.1", "model.layers.1@pass2", "model.layers.2"]
    assert len(vecs) == 4


LLAMA = Path(os.environ.get("OC_LLAMA", "")).expanduser()
CETT = LLAMA / "build" / "bin" / "llama-cett-dump"


@torch_needed
@pytest.mark.skipif(not CETT.exists(), reason="set OC_LLAMA to a llama.cpp build with cett-dump")
def test_gguf_backend_trains_and_checks(tmp_path):
    import subprocess
    from tiny_models import make_tiny_llama
    hf = make_tiny_llama(tmp_path / "tiny", layers=3, hidden=32, ff=64)
    gguf = tmp_path / "tiny.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(gguf),
                    "--outtype", "f32"], check=True, capture_output=True)
    feat = TP.GGUFFeatures(str(gguf), str(CETT), ngl=0)
    names, vecs = feat(TP.messages_for("what is the capital"), "blue")
    assert names == ["layer.0", "layer.1", "layer.2"] and vecs[0].shape == (64,)
    probe = TP.train(feat, synthetic(24), C=1.0)
    r = TP.check(feat, probe, "what is the capital", "two", ["blue"])
    assert r["verdict"] in ("knows_but_says_otherwise", "doubts_own_answer", "unsure")


@torch_needed
@pytest.mark.skipif(not CETT.exists(), reason="set OC_LLAMA to a llama.cpp build with cett-dump")
def test_studio_truth_reply(tmp_path, monkeypatch):
    """Studio's /api/truth path: the model's probe from its settings, the question from the last user message."""
    import subprocess
    from tiny_models import make_tiny_llama
    sys.path.insert(0, str(ROOT / "viz"))
    import studio
    hf = make_tiny_llama(tmp_path / "tiny", layers=3, hidden=32, ff=64)
    gguf = tmp_path / "tiny.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(gguf),
                    "--outtype", "f32"], check=True, capture_output=True)
    feat = TP.GGUFFeatures(str(gguf), str(CETT), ngl=0)
    TP.train(feat, synthetic(24), C=1.0).save(tmp_path / "p.npz")
    model = {"id": "tiny", "name": "tiny", "path": str(gguf)}
    monkeypatch.setitem(studio.STATE, "cett", str(CETT))
    monkeypatch.setitem(studio.STATE, "score_ngl", 0)
    monkeypatch.setattr(studio, "load_settings", lambda: {studio._key(str(gguf)): {"truth_probe": str(tmp_path / "p.npz")}})
    r = studio.truth_reply(model, [{"role": "system", "content": "x"}, {"role": "user", "content": "what is the capital"}],
                           "two", ["blue"])
    assert r["verdict"] in TP_VERDICTS and r["alternatives"][0]["text"] == "blue"
    monkeypatch.setattr(studio, "load_settings", lambda: {})
    with pytest.raises(ValueError, match="truth probe"):
        studio.truth_reply(model, [], "two")


TP_VERDICTS = {"knows_but_says_otherwise", "doubts_own_answer", "consistent", "unsure"}
