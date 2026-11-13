"""Random-projection steering: a spread-out signal survives the projection, and steering pulls it back."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
import rp_steer as rp  # noqa: E402


def data(n=400, d=256, signal=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(d, generator=g)
    direction /= direction.norm()                     # dense: every neuron carries a little of it
    y = (torch.rand(n, generator=g) < 0.5).long()
    h = torch.randn(n, d, generator=g) + signal * 3 * y[:, None] * direction
    return h, y, direction


def test_spread_out_signal_survives_projection():
    h, y, _ = data(signal=3.0)                        # strong and spread over every neuron
    r = rp.fit_layer(h, y, dim=16, seeds=[0, 1, 2], shrink=0.1, test_frac=0.25, seed=0)
    assert r["auc"] > 0.9 and r["auc_full"] > 0.9 and r["auc_spread"] < 0.1
    # a weak signal fades in a small projection (about sqrt(k/d) of it is kept), and the fit shows the loss
    hw, yw, _ = data(signal=1.0, seed=5)
    weak = rp.fit_layer(hw, yw, dim=16, seeds=[0], shrink=0.1, test_frac=0.25, seed=0)
    assert weak["auc_full"] - weak["auc"] > 0.15
    assert r["R"].shape == (256, 16) and torch.allclose(r["R"].T @ r["R"], torch.eye(16), atol=1e-4)
    assert r["c_wrong"] > r["c_right"] and r["u_c_wrong"] > r["u_c_right"]
    assert r["auc_ensemble"] > 0.9 and r["n_test"] == 100 and 0 < r["auc_val"] <= 1
    hn, yn, _ = data(signal=0.0, seed=3)                # no signal: chance
    assert abs(rp.fit_layer(hn, yn, 16, [0], 0.1, 0.25, 0)["auc"] - 0.5) < 0.15


class Layer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Identity()

    def forward(self, x):
        return self.mlp(x)


def test_steering_pulls_back_only_past_the_right_side(tmp_path):
    h, y, _ = data(signal=3.0)
    r = rp.fit_layer(h, y, 16, [0], 0.1, 0.25, 0)
    torch.save({"layers": {0: r}}, tmp_path / "rp.pt")
    layer = Layer(256)
    st = rp.Steerer(str(tmp_path / "rp.pt"), alpha=1.0, mode="direction")
    st.attach([layer])
    u = st.parts[0]["u"]
    wrong = h[y == 1][:20, None, :]                     # [20, 1, d]
    right = h[y == 0][:20, None, :]
    out_w, out_r = layer(wrong), layer(right)
    c = st.parts[0]["c_right"]
    assert ((out_w[:, 0] @ u) <= c + 1e-4).all()                            # pulled back to the right side
    below = (right[:, 0] @ u) <= c
    assert torch.allclose(out_r[below], right[below])                         # left alone when not drifting
    assert st.flagged() >= 0
    st.remove()
    assert torch.equal(layer(wrong), wrong)
    sub = rp.Steerer(str(tmp_path / "rp.pt"), alpha=1.0, mode="subspace")
    sub.attach([layer])
    moved = layer(wrong)
    assert float((moved[:, 0] @ u).mean()) < float((wrong[:, 0] @ u).mean())
    sub.remove()
    with pytest.raises(ValueError):
        rp.Steerer(str(tmp_path / "rp.pt"), mode="sideways")


def test_fit_and_generate_with_a_tiny_model(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    import decode_lab as dl
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=4)
    rows = []
    for i in range(12):
        rows.append({"prompt": f"What is {i} plus {i}?", "reply": f"It is {2 * i}.", "label": "right"})
        rows.append({"prompt": f"What is {i} plus {i}?", "reply": f"Certainly it is {2 * i + 7} without doubt.",
                     "label": "wrong"})
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "rp.pt"
    assert rp.main(["fit", "--model", str(hf), "--data", str(tmp_path / "pairs.jsonl"), "--layers", "1", "2",
                    "--dim", "8", "--projections", "2", "--dtype", "float32", "--out", str(out)]) == 0
    summary = json.loads(out.with_suffix(".json").read_text())
    assert set(summary["layers"]) == {"1", "2"} and summary["n"] == 24
    variants = dl.parse_variants(["plain={}", 'rp={"rp": {"file": "%s", "alpha": 1.0}}' % out])   # as the CLI does
    with pytest.raises(SystemExit):
        dl.parse_variants(['rp={"rp": {"alpha": 1.0}}'])
    lab = dl.Lab(str(hf), variants, "float32")
    r = lab.generate([{"role": "user", "content": "What is 2 plus 2?"}], "rp", max_tokens=6)
    assert len(r["rp_drift"]) == len(r["spread"]) and all(x >= 0 for x in r["rp_drift"])
    assert "rp_drift" not in lab.generate([{"role": "user", "content": "hi"}], "plain", max_tokens=3)
    # hooks come off after each request
    assert all(not m._forward_hooks for m in lab.model.modules() if m.__class__.__name__.endswith("MLP"))
    gguf = pytest.importorskip("gguf")
    assert rp.main(["export", "--file", str(out), "--out", str(tmp_path / "cv.gguf")]) == 0
    names = {t.name for t in gguf.GGUFReader(str(tmp_path / "cv.gguf")).tensors}
    assert names == {"direction.1", "direction.2"}
