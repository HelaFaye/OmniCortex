"""train_guards.py: the collapse guard's rules, the KL term, and both guards in a real LoRA run on a tiny model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_guards as TG  # noqa: E402

VARIED = ["Canberra is the capital of Australia.", "Water boils at one hundred degrees.",
          "The Moon orbits the Earth every month.", "Paris lies on the river Seine."]


def test_measures():
    assert TG.distinct2(VARIED) == 1.0 and TG.distinct2(["a b a b a b a b"]) == pytest.approx(2 / 7)
    assert TG.abstain_rate(["I don't know.", "Canberra", ""]) == pytest.approx(1 / 3)
    assert TG.empty_rate(["", " ", "x"]) == pytest.approx(2 / 3)
    assert TG.is_anchor({"source": "replay"}) and TG.is_anchor({"source": "general"}) and not TG.is_anchor({})


def test_guard_rules():
    g = TG.CollapseGuard()
    assert g.judge(0, VARIED) is None and g.judge(5, VARIED[::-1]) is None
    assert "diversity fell" in g.judge(10, ["the the the the the the"] * 4)
    g = TG.CollapseGuard()
    g.judge(0, VARIED)
    assert "abstention rose from 0% to 75%" in g.judge(5, ["I don't know that.", "I'm not sure, sorry.",
                                                         "I cannot answer this.", VARIED[0]])
    g = TG.CollapseGuard()
    g.judge(0, VARIED)
    assert "empty replies rose" in g.judge(5, ["", "", VARIED[0], VARIED[1]])
    g = TG.CollapseGuard(max_kl=0.5)
    g.judge(0, VARIED, kl=0.0)
    assert g.judge(5, VARIED, kl=0.4) is None and "KL from the base model reached 0.900" in g.judge(9, VARIED, kl=0.9)
    assert g.report()["stopped"]["step"] == 9 and len(g.report()["history"]) == 3
    g = TG.CollapseGuard()
    g.judge(0, VARIED)
    assert "loss is nan" in g.judge(3, VARIED, loss=float("nan"))


def test_kl_term():
    torch = pytest.importorskip("torch")
    a = torch.randn(2, 5, 11)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert float(TG.kl_base_tuned(a, a, mask)) == pytest.approx(0, abs=1e-6)
    b = a + torch.randn(2, 5, 11)
    assert float(TG.kl_base_tuned(a, b, mask)) > 0
    mask[:, 2:] = False                                   # only the masked positions count
    part = TG.kl_base_tuned(a[:, :2], b[:, :2], torch.ones(2, 2, dtype=torch.bool))
    assert float(TG.kl_base_tuned(a, b, mask)) == pytest.approx(float(part), rel=1e-5)


def test_finetune_with_anchor_and_guard(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("sentencepiece")
    sys.path.insert(0, str(ROOT / "tests"))
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny")
    rows = [{"messages": [{"role": "user", "content": f"What is fact {i}?"},
                          {"role": "assistant", "content": f"Fact {i} is the number {i * 7}."}], "source": "repo"}
            for i in range(12)]
    rows += [{"messages": [{"role": "user", "content": f"Tell me about topic {i}."},
                           {"role": "assistant", "content": f"Topic {i} covers many ideas."}], "source": "replay"}
             for i in range(4)]
    (tmp_path / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    common = [sys.executable, str(ROOT / "scripts" / "finetune.py"), "--model", str(hf), "--data", str(tmp_path),
              "--method", "lora", "--lora-r", "4", "--max-steps", "4", "--batch-size", "4", "--grad-accum", "1",
              "--max-length", "128", "--logging-steps", "2", "--lr", "5e-3"]
    r = subprocess.run(common + ["--out", str(tmp_path / "a"), "--kl-anchor", "0.5", "--guard-every", "2",
                                 "--guard-max-abstain-rise", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    m = json.loads((tmp_path / "a" / "omnicortex-finetune.json").read_text())
    kls = [x["kl_anchor"] for x in m["log"]["sft"] if "kl_anchor" in x]
    assert m["kl_anchor"] == 0.5 and kls and max(kls) > 0          # the anchor term was computed and logged
    h = m["guard"]["history"]
    assert [x["step"] for x in h] == [0, 2, 4] and all("kl" in x for x in h) and m["guard"]["stopped"] is None
    r = subprocess.run(common + ["--out", str(tmp_path / "b"), "--guard-every", "2", "--guard-max-kl", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 5 and "collapse guard" in r.stdout + r.stderr
    m = json.loads((tmp_path / "b" / "omnicortex-finetune.json").read_text())
    assert m["guard"]["stopped"]["step"] == 2 and "KL from the base" in m["guard"]["stopped"]["why"]
    assert (tmp_path / "b" / "adapter_model.safetensors").exists()            # saved, but flagged
    r = subprocess.run(common + ["--out", str(tmp_path / "c"), "--method", "full", "--kl-anchor", "0.1"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--kl-anchor needs --method lora" in r.stderr
