"""Concept subspace editing: directions, norm-preserving biprojected edits, KL damage meter, search."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
import concept_edit as ce  # noqa: E402


def test_directions_and_biprojection():
    g = torch.Generator().manual_seed(0)
    d = 16
    base = torch.zeros(d)
    base[0] = 5.0                                      # both sets share a big mean along e0
    concept = torch.zeros(d)
    concept[1] = 2.0
    concept[0] = 1.0                                   # the concept leans partly along e0 too
    B = base + 0.1 * torch.randn(64, 3, d, generator=g)
    A = base + concept + 0.1 * torch.randn(64, 3, d, generator=g)
    bas = ce.directions(A, B, "mean", 1, biproject=True)
    v = bas[1, :, 0]
    assert abs(float(v.norm()) - 1) < 1e-5 and abs(float(v[0])) < 0.1 and float(v[1]) > 0.9   # e0 removed
    raw = ce.directions(A, B, "mean", 1, biproject=False)[1, :, 0]
    assert abs(float(raw[0])) > 0.3                     # without biprojection it keeps the shared axis
    cone = ce.directions(A, B, "pca", 3)
    q = cone[1]
    assert torch.allclose(q.T @ q, torch.eye(3), atol=1e-4)                # orthonormal cone
    assert float(ce.separation(A, B, bas)[1]) > 1.5


def test_kernel_shape():
    w = ce.kernel(10, 1.0, 4, 0.4, 3)
    assert w[4] == 1.0 and w[7] == pytest.approx(0.4) and w[8] == 0 and w[0] == 0 and w[2] == pytest.approx(0.6)


def test_edit_projects_out_and_preserves_norms(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    from transformers import AutoModelForCausalLM
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=3)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)
    ed = ce.Editor(model)
    assert [n for n, _ in ed.targets[0]] == ["self_attn.o_proj", "mlp.down_proj"]
    d = model.config.hidden_size
    r = torch.zeros(d, 1)
    r[3, 0] = 1.0
    W0 = ed.targets[1][1][1].weight.detach().clone()
    ed.apply(lambda li: r, [0.0, 1.0, 0.0], norm_preserve=True)
    W1 = ed.targets[1][1][1].weight.detach()
    assert float(W1[3].abs().max()) < 1e-6                                   # nothing written along r
    assert torch.allclose(W1.norm(dim=0), W0.norm(dim=0), atol=1e-4)        # column norms kept
    assert torch.equal(ed.targets[0][1][1].weight, ed.targets[0][1][1].weight)  # untouched layer
    ed.apply(lambda li: r, [0.0, 0.5, 0.0], norm_preserve=False)             # re-applies from the original
    assert float(ed.targets[1][1][1].weight[3].abs().max()) == pytest.approx(float(W0[3].abs().max()) * 0.5, rel=1e-4)
    ed.reset()
    assert torch.equal(ed.targets[1][1][1].weight, W0)


def test_cli_search_and_save(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=3)
    (tmp_path / "a.txt").write_text("\n".join(f"tell me a confident answer about topic {i}" for i in range(8)))
    (tmp_path / "b.jsonl").write_text("".join(json.dumps({"text": f"say you are not sure about item {i}"}) + "\n"
                                              for i in range(8)))
    (tmp_path / "n.txt").write_text("\n".join(f"the weather report for day {i}" for i in range(6)))
    out = tmp_path / "edit"
    rc = ce.main(["--model", str(hf), "--concept", str(tmp_path / "a.txt"), "--baseline", str(tmp_path / "b.jsonl"),
                  "--neutral", str(tmp_path / "n.txt"), "--out", str(out), "--search", "3", "--cpu",
                  "--marker", "sure", "--method", "pca", "--k", "2"])
    assert rc == 0
    rep = json.loads((out / "edit.json").read_text())
    assert len(rep["trials"]) == 3 and rep["chosen"]["kl"] >= 0 and "behaviour_rate" in rep["chosen"]
    assert rep["chosen"]["remaining_separation"] < 1.0 and len(rep["separation_by_layer"]) == 4
    assert (out / "model" / "config.json").exists() and (out / "directions.pt").exists()
    # analysis only edits nothing
    assert ce.main(["--model", str(hf), "--concept", str(tmp_path / "a.txt"), "--baseline",
                    str(tmp_path / "b.jsonl"), "--out", str(tmp_path / "an"), "--analyze", "--cpu"]) == 0
    assert not (tmp_path / "an" / "model").exists()


def test_edited_model_converts_to_gguf(tmp_path):
    import os
    import subprocess
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    llama = Path(os.environ.get("OC_LLAMA") or "~/llama.cpp").expanduser()
    if not (llama / "convert_hf_to_gguf.py").exists():
        pytest.skip("needs llama.cpp")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=2)
    (tmp_path / "a.txt").write_text("\n".join(f"alpha {i}" for i in range(4)))
    (tmp_path / "b.txt").write_text("\n".join(f"beta {i}" for i in range(4)))
    out = tmp_path / "e"
    assert ce.main(["--model", str(hf), "--concept", str(tmp_path / "a.txt"), "--baseline", str(tmp_path / "b.txt"),
                    "--out", str(out), "--kernel", "1,1,0.5,1", "--direction-layer", "0.5", "--cpu"]) == 0
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_export.py"), "--base", str(hf), "--adapter",
                        str(out / "model"), "--out", str(tmp_path / "g"), "--gguf", "--llama", str(llama)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((tmp_path / "g" / "omnicortex-export.json").read_text())["gguf"].endswith(".gguf")
