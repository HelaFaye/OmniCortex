"""Looped models (docs/DORY.md): Studio's loop settings, the loop sweep's pieces, the per-loop lens, and the NeMo Gym
bridge."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import gym_bridge  # noqa: E402
import loop_sweep  # noqa: E402


def test_studio_loop_settings():
    import test_studio
    studio = test_studio.studio
    dory = {"id": "dory", "name": "dory", "path": "/m/dory-q5_k_m.gguf", "arch": "dory", "looped": True}
    cmd, _ = studio.server_cmd(dory, {**studio.DEFAULTS, "loops": "3", "loop_kv": "last_loop"}, 9999)
    assert cmd[cmd.index("--override-kv") + 1] == "dory.num_loops=int:3,dory.recurrent_kv_cache_mode=str:last_loop"
    assert "--jinja" in cmd and cmd[cmd.index("--reasoning") + 1] == "on"
    plain, _ = studio.server_cmd(dory, dict(studio.DEFAULTS), 9999)
    assert "--override-kv" not in plain and "--jinja" in plain                    # model defaults, recipe flags kept
    mine, _ = studio.server_cmd(dory, {**studio.DEFAULTS, "extra": "--reasoning off"}, 9999)
    assert mine.count("--reasoning") == 1 and mine[mine.index("--reasoning") + 1] == "off"
    for bad in ({"loops": "40"}, {"loop_kv": "every"}):
        with pytest.raises(ValueError):
            studio.server_cmd(dory, {**studio.DEFAULTS, **bad}, 9999)
    dense = {"id": "x", "name": "x", "path": "/m/x.gguf", "arch": "llama"}
    cmd, _ = studio.server_cmd(dense, {**studio.DEFAULTS, "loops": "3"}, 9999)
    assert "--override-kv" not in cmd and "--jinja" not in cmd                   # only looped models get them


def test_loop_sweep_pieces(tmp_path):
    assert [s[0] for s in loop_sweep.settings([1, 2], ["per_loop"])] == ["loops=1", "loops=2"]
    assert [s[0] for s in loop_sweep.settings([2], ["per_loop", "last_loop"])] == ["loops=2,per_loop", "loops=2,last_loop"]
    cmd = loop_sweep.command("srv", "m.gguf", "dory", 3, "last_loop", 8091, 999, 4096, ["--flash-attn", "on"])
    assert "dory.num_loops=int:3,dory.recurrent_kv_cache_mode=str:last_loop" in cmd and cmd[-2:] == ["--flash-attn", "on"]
    ref = [{"id": "a", "verdict": "correct"}, {"id": "b", "verdict": "wrong"}, {"id": "c", "verdict": "abstained"}]
    var = [{"id": "a", "verdict": "wrong"}, {"id": "b", "verdict": "correct"}, {"id": "c", "verdict": "correct"}]
    assert loop_sweep.paired(ref, var) == (["b", "c"], ["a"])
    s = loop_sweep.summary(var)
    assert s["accuracy"] == pytest.approx(2 / 3, abs=1e-3) and s["confident_wrong"] == pytest.approx(1 / 3, abs=1e-3)
    tasks = loop_sweep.load_tasks(str(ROOT / "qa" / "bank"))
    assert tasks and {t["kind"] for t in tasks} <= {"qa", "reasoning"}


def test_gym_bridge(tmp_path):
    n = gym_bridge.export([str(ROOT / "qa" / "bank")], False, tmp_path / "train.jsonl", ["factual"])
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert n == len(rows) > 5 and rows[0]["responses_create_params"]["input"][0]["role"] == "user"
    task = next(r["omnicortex_task"] for r in rows if r["uuid"] == "q-canberra")
    assert gym_bridge.reward(task, "Canberra")["reward"] == 1.0
    assert gym_bridge.reward(task, "Sydney")["reward"] == -1.0
    assert gym_bridge.reward(task, "I don't know.")["reward"] == 0.25
    assert gym_bridge.reward(task, "Canberra", risk=0.8, cfg={"risk_weight": 0.5})["reward"] == 0.6
    resp = {"output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": "Canberra"}]}]}
    assert gym_bridge.response_text(resp) == "Canberra" and gym_bridge.response_text({"output_text": "x"}) == "x"
    ref = {"kind": "reference", "reference": "The answer is 4 because two plus two is four.", "rejected": "It is 5."}
    assert gym_bridge.reward(ref, "Two plus two is four, so the answer is 4.")["verdict"] == "correct"
    assert gym_bridge.reward(ref, "It is 5.")["verdict"] == "wrong"
    ex = (ROOT / "integrations" / "nemo_gym" / "omnicortex_testqa" / "data" / "example.jsonl").read_text().splitlines()
    assert len(ex) == 5 and all("omnicortex_task" in json.loads(l) for l in ex)


def test_loop_lens_on_a_tiny_looped_model():
    torch = pytest.importorskip("torch")
    import loop_lens

    class Layer(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.up, self.mlp.down_proj = torch.nn.Linear(d, 2 * d), torch.nn.Linear(2 * d, d)

        def forward(self, x):
            return (x + self.mlp.down_proj(torch.relu(self.mlp.up(x))),)

    class Looped(torch.nn.Module):
        def __init__(self, d=16, vocab=11, loops=3):
            super().__init__()
            torch.manual_seed(0)
            self.emb, self.loops = torch.nn.Embedding(vocab, d), loops
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(d), Layer(d), Layer(d)])   # 0 prelude, 1 core, 2 coda
            self.model.norm = torch.nn.LayerNorm(d)
            self.head = torch.nn.Linear(d, vocab, bias=False)

        def get_output_embeddings(self):
            return self.head

        def forward(self, ids):
            x = self.model.layers[0](self.emb(ids))[0]
            for _ in range(self.loops):
                x = self.model.layers[1](x)[0]
            x = self.model.layers[2](x)[0]
            return type("O", (), {"logits": self.head(self.model.norm(x))})()

    class Tok:
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [ord(c) % 11 for c in text]
            return {"input_ids": torch.tensor([ids]) if return_tensors else ids}

        def decode(self, ids):
            return f"<{ids[0]}>"
    m = Looped().eval()
    out = loop_lens.lens(m, Tok(), "abc", target="d", h_neurons={"model.layers.1": [0, 1, 2]})
    assert out["loops"] == 3 and out["recurrent_layers"] == ["model.layers.1"]
    assert [r["loop"] for r in out["per_loop"]] == [1, 2, 3] and "target_rank" in out["per_loop"][0]
    assert len(out["mlp_activity"]["model.layers.1"]) == 3 and len(out["h_neuron_share"]["model.layers.1"]) == 3
    assert out["decided_at_loop"] is None or 1 <= out["decided_at_loop"] <= 3


def _looped_gguf(path, loops=2):
    """A tiny GGUF with Dory-style metadata: 4 blocks, block 1 attention-only (no ffn_down)."""
    gguf = pytest.importorskip("gguf")
    import numpy as np
    w = gguf.GGUFWriter(str(path), "dory")
    w.add_block_count(4)
    w.add_uint32("dory.num_loops", loops)
    rng = np.random.default_rng(0)
    for b in (0, 2, 3):
        w.add_tensor(f"blk.{b}.ffn_down.weight", rng.standard_normal((8, 16)).astype(np.float32))  # [n_embd, n_ff]
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_cett_dump_layout_for_looped_ggufs(tmp_path):
    """Looped GGUFs run cett-dump with --by-weight: pass k of block N is layer k*blocks+N (docs/DORY.md)."""
    import numpy as np
    from extract_activations_gguf import layer_norms, looped_args, looped_layout, weight_col_norms
    g = str(_looped_gguf(tmp_path / "d.gguf"))
    assert looped_layout(g) == (4, 2, "dory")
    assert looped_layout(g, loops=3) == (4, 3, "dory")
    assert looped_args(looped_layout(g)) == ["--by-weight", "4"]
    assert looped_args(looped_layout(g, 3), True) == ["--by-weight", "4", "--override-kv", "dory.num_loops=int:3"]
    assert looped_args(None) == []
    n = layer_norms(g, 8, looped_layout(g))
    base = weight_col_norms(g, 4, allow_missing=True)
    assert n.shape == (8, 16) and not n[1].any() and not n[5].any()          # block 1 has no MLP, in both passes
    assert np.allclose(n[0], base[0]) and np.allclose(n[4], base[0]) and np.allclose(n[7], base[3])
    import hscore
    s = hscore.HScorer("cett-dump", g, None, loops=3)
    assert s.layout() == (4, 3, "dory") and s._layers_from_gguf() == 12 and s._col_norms(12).shape == (12, 16)


def test_ordinary_ggufs_are_not_looped(tmp_path):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    from extract_activations_gguf import looped_layout
    w = gguf.GGUFWriter(str(tmp_path / "l.gguf"), "llama")
    w.add_block_count(2)
    w.add_tensor("blk.0.ffn_down.weight", np.ones((4, 8), np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    assert looped_layout(str(tmp_path / "l.gguf")) is None
