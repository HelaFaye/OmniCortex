"""CETT alternatives (scripts/neuron_attrib.py): weights-only logit attribution, quantization noise, saliency."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import neuron_attrib as NA  # noqa: E402

gguf = pytest.importorskip("gguf")
D, FF, L = 32, 64, 3
VOCAB = ["<unk>", "<s>", "</s>", "ĠParis", "ĠLondon", "ĠRome", "Ġthe", "a", "b", "c"]


def write_gguf(path, tensors: dict, qtype=None):
    """tensors: name -> float32 array in numpy order ([ne1, ne0]); qtype quantizes the 2-D ones."""
    from gguf import quants
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(VOCAB)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if qtype is not None and arr.ndim == 2:
            w.add_tensor(name, quants.quantize(arr, qtype), raw_dtype=qtype)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def model_tensors(seed=0):
    rng = np.random.default_rng(seed)
    t = {"output_norm.weight": np.ones(D, np.float32),
         "output.weight": rng.normal(0, 1, (len(VOCAB), D)).astype(np.float32)}
    for i in range(L):
        t[f"blk.{i}.ffn_down.weight"] = rng.normal(0, 0.02, (D, FF)).astype(np.float32)   # [n_embd, n_ff]
        t[f"blk.{i}.ffn_gate.weight"] = rng.normal(0, 0.02, (FF, D)).astype(np.float32)
        t[f"blk.{i}.ffn_up.weight"] = rng.normal(0, 0.02, (FF, D)).astype(np.float32)
    return t


def test_weights_dla_finds_the_neuron_that_writes_the_answer(tmp_path):
    t = model_tensors()
    paris = t["output.weight"][3]
    t["blk.1.ffn_down.weight"][:, 7] = paris / np.linalg.norm(paris)       # neuron 7 of layer 1 writes " Paris"
    t["blk.2.ffn_down.weight"][:, 9] = -paris / np.linalg.norm(paris)      # neuron 9 of layer 2 suppresses it
    g = write_gguf(tmp_path / "m.gguf", t)
    samples = [("q1", {"answer_tokens": [" Paris"]}), ("q2", {"answer_tokens": ["zzz"]})]
    meta = NA.weights_dla(g, samples, str(tmp_path / "out"), log=lambda *a: None)
    assert meta["written"] == 1 and meta["skipped_no_tokens"] == 1 and (meta["n_layers"], meta["n_neurons"]) == (L, FF)
    v = np.load(tmp_path / "out" / "answer_tokens" / "act_q1.npy").astype(np.float32)
    assert v.shape == (L, FF)
    assert np.unravel_index(v.argmax(), v.shape) == (1, 7) and np.unravel_index(v.argmin(), v.shape) == (2, 9)
    idx = json.loads((tmp_path / "out" / "neuron_index.json").read_text())
    assert idx["measure"] == "weights-dla" and not (tmp_path / "out" / ".dla-tmp.npy").exists()


def test_quant_noise_and_h_neurons(tmp_path):
    t = model_tensors(1)
    ref = write_gguf(tmp_path / "ref.gguf", t)
    q = model_tensors(1)
    q["blk.0.ffn_down.weight"][:, 4] += 0.05                               # neuron 4 of layer 0 badly damaged
    quant = write_gguf(tmp_path / "q.gguf", q, gguf.GGMLQuantizationType.Q8_0)
    h = tmp_path / "h.json"
    h.write_text(json.dumps({"by_layer": {"0": [4]}}))
    rep = NA.quant_noise(ref, quant, str(tmp_path / "qn"), str(h), log=lambda *a: None)
    assert rep["worst"][0] == {"layer": 0, "neuron": 4, "error": rep["worst"][0]["error"]}
    assert rep["overall"]["median"] < 0.02                                  # Q8_0 alone moves weights by ~0.5%
    assert rep["h_neurons"]["verdict"] == "hit harder than the rest" and rep["h_neurons"]["median_percentile"] > 99
    assert np.load(tmp_path / "qn" / "quant_noise.npy").shape == (L, FF)


def test_mismatched_and_moe_are_refused(tmp_path):
    t = model_tensors()
    a = write_gguf(tmp_path / "a.gguf", t)
    del t["blk.2.ffn_down.weight"], t["blk.2.ffn_gate.weight"], t["blk.2.ffn_up.weight"]
    b = write_gguf(tmp_path / "b.gguf", t)
    with pytest.raises(ValueError, match="layer counts"):
        NA.quant_noise(a, b, str(tmp_path / "x"), log=lambda *a: None)
    moe = write_gguf(tmp_path / "moe.gguf", {"blk.0.ffn_down_exps.weight": np.zeros((2, D, FF), np.float32)})
    with pytest.raises(ValueError, match="mixture-of-experts"):
        NA.GGUFWeights(moe)
    assert NA.main(["quant-noise", "--ref", a, "--quant", b, "--out", str(tmp_path / "y")]) == 2


def test_saliency_matches_ablation(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    from tiny_models import make_tiny_llama
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = make_tiny_llama(tmp_path / "tiny", layers=2, hidden=32, ff=64)
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.tensor(tok("the cat sat on the mat and the dog ran")["input_ids"])
    span = (len(ids) - 3, len(ids))
    hooks = NA.SaliencyHooks(model)
    att = NA.saliency_one(model, hooks, ids, span)
    assert att.shape == (2, len(ids), 64)
    assert float(att[:, span[1]:].abs().sum()) == 0.0                         # nothing after the target matters
    ig = NA.saliency_one(model, hooks, ids, span, "integrated", 4)
    assert ig.shape == att.shape and float(ig.sum()) > 0
    hooks.remove()

    def logp(zero=None):
        hs = []
        if zero is not None:
            layer, pos, n = zero
            mod = model.model.layers[layer].mlp.down_proj

            def pre(m, inp):
                x = inp[0].clone()
                x[0, pos, n] = 0
                return (x,)
            hs.append(mod.register_forward_pre_hook(pre))
        with torch.no_grad():
            lp = torch.log_softmax(model(ids.unsqueeze(0)).logits[0].float(), -1)
        for h in hs:
            h.remove()
        return float(sum(lp[t - 1, ids[t]] for t in range(*span)))
    base = logp()
    flat = att.reshape(-1)
    order = torch.argsort(flat, descending=True)
    top = [np.unravel_index(int(i), att.shape) for i in order[:5]]
    low = [np.unravel_index(int(i), att.shape) for i in order[-200:-195]]
    eff_top = np.mean([abs(logp(z) - base) for z in top])
    eff_low = np.mean([abs(logp(z) - base) for z in low])
    assert eff_top > 5 * eff_low                         # the attribution ranks real ablation effects


def test_saliency_cli_writes_extraction_format(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    from tiny_models import make_tiny_llama
    hf = make_tiny_llama(tmp_path / "tiny", layers=2, hidden=32, ff=64)
    inp = tmp_path / "a.jsonl"
    inp.write_text(json.dumps({"q1": {"question": "the cat", "response": "the dog ran", "answer_tokens": []}}) + "\n")
    out = tmp_path / "sal"
    assert NA.main(["saliency", "--model_path", str(hf), "--input_path", str(inp), "--output_root", str(out),
                    "--locations", "output", "--device", "cpu"]) == 0
    v = np.load(out / "output" / "act_q1.npy")
    idx = json.loads((out / "neuron_index.json").read_text())
    assert v.shape == (2, 64) and v.dtype == np.float16 and idx["measure"] == "saliency-grad_x_act"
    assert idx["n_layers"] == 2 and idx["n_neurons"] == 64 and idx["written"] == 1
