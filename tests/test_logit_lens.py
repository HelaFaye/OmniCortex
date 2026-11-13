"""Logit lens over a reply: per-layer spread, settle and decay layers, marks, and the heatmap page."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
import logit_lens as ll  # noqa: E402


def test_spread_matches_the_chat_strip():
    # certain: one token holds everything; even: five tokens at 0.2 each
    V = 50
    certain = torch.full((V,), -1e9)
    certain[3] = 0.0
    even = torch.full((V,), -1e9)
    even[:5] = math.log(0.2)
    s, ids, p = ll.spread_top5(torch.stack([certain, even]))
    assert float(s[0]) < 1e-6 and int(ids[0, 0]) == 3
    assert float(s[1]) == pytest.approx(math.log(5) / math.log(6), abs=1e-4)


def test_lens_on_a_tiny_model(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=3)
    out, page = tmp_path / "lens.json", tmp_path / "lens.html"
    assert ll.main(["--model", str(hf), "--prompt", "Who wrote Hamlet?", "--reply", "Marlowe wrote it.",
                    "--out", str(out), "--html", str(page), "--dtype", "float32", "--late", "0", "--spread", "0"]) == 0
    r = json.loads(out.read_text())
    T = len(r["pieces"])
    assert r["layers"] == 3 and T >= 3 and "".join(r["pieces"]).strip().startswith("Marlowe")
    assert len(r["spread"]) == T and all(len(row) == 3 for row in r["spread"])
    assert all(1 <= s <= 3 for s in r["settle"])
    assert all(row[-1] == r["top1"][t][-1] for t, row in enumerate(r["top1"]))
    assert r["marked"] == list(range(T))                 # with both thresholds at 0 every token is marked
    assert "<svg" in page.read_text() and "logit lens heatmap" in page.read_text()
    # the last layer's spread is the final-layer measure, computed from the model's own logits
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)
    ids, start = ll.encode(tok, "Who wrote Hamlet?", None, "Marlowe wrote it.")
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids])).logits[0, start - 1:-1].float(), -1)
    want = ll.spread_top5(logp)[0]
    assert torch.allclose(torch.tensor(r["final_spread"]), want, atol=1e-3)
    # and it writes a reply itself when asked
    assert ll.main(["--model", str(hf), "--prompt", "Hi", "--generate", "5", "--out", str(out), "--dtype", "float32"]) == 0
    assert json.loads(out.read_text())["reply"] is not None
