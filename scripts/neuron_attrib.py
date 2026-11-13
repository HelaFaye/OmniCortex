#!/usr/bin/env python3
"""
Neuron measures for computers where CETT extraction does not fit or run: older GPUs, and CPUs
(docs/PIPELINE.md#without-cett-older-gpus-and-cpus).

  saliency     Gradient x activation: how much the model's log-probability of its own answer would
               change if a neuron went quiet (a first-order estimate of patching each neuron).
               --method integrated averages the gradient along the path from zero to the actual
               activation (integrated gradients over the neuron Jacobian), which is more faithful
               when a neuron is far from linear. PyTorch with Hugging Face weights, in float32 by
               default, so it runs on a CPU and on GPUs without bfloat16.
  weights      Direct logit attribution from the weights alone, no forward pass: how strongly each
               neuron's output column writes each answer token's unembedding (after the final
               norm's scale). Reads a GGUF with numpy, one layer at a time.
  quant-noise  How far each neuron's weights moved under quantization: the relative error of its
               gate and up rows and its down column, between a reference GGUF (F16, BF16 or Q8_0)
               and a quantized one. With --neurons it says whether the H-Neurons were hit harder
               than the rest.

saliency and weights write what extraction writes, so the classifier stage reads them unchanged:
<output_root>/neuron_index.json and <output_root>/<location>/act_<qid>.npy, float16 [layers, neurons].
Like the two CETT paths, do not mix measures in one training set.

    python scripts/neuron_attrib.py saliency --model_path Qwen/Qwen3-0.6B --input_path data/answer_tokens.jsonl \\
        --ids_path data/train_qids.json --output_root data/saliency
    python scripts/neuron_attrib.py weights --gguf model.gguf --input_path data/answer_tokens.jsonl \\
        --output_root data/dla
    python scripts/neuron_attrib.py quant-noise --ref model-F16.gguf --quant model-Q4_K_M.gguf \\
        --neurons models/h_neurons.json --out runs/qnoise
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_constants import TEXT_LAYER_RE, normalise_piece  # noqa: E402

LOCATIONS = ["input", "output", "answer_tokens", "all_except_answer_tokens"]


# ================================================================ GGUF weights

class GGUFWeights:
    """Dense GGUF tensors as float32 numpy arrays, one at a time (rows sliced before dequantizing)."""

    def __init__(self, path: str):
        import gguf
        self.path = path
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        self.layers = sorted({int(m.group(1)) for n in self.t if (m := re.match(r"blk\.(\d+)\.ffn_down\.weight$", n))})
        if not self.layers:
            if any(".ffn_down_exps." in n for n in self.t):
                raise ValueError("this is a mixture-of-experts GGUF: weights and quant-noise read dense FFNs only "
                                 "for now (use saliency, or CETT with --top-experts)")
            raise ValueError("no blk.N.ffn_down.weight tensors in this GGUF")

    def has(self, name: str) -> bool:
        return name in self.t

    def get(self, name: str, rows=None) -> np.ndarray:
        """-> [ne1, ne0] float32 (ggml stores rows of ne0); rows picks some of the ne1 rows first."""
        from gguf import quants
        t = self.t[name]
        data = t.data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if rows is not None:
            data = data[np.asarray(rows)]
        out = quants.dequantize(np.ascontiguousarray(data), t.tensor_type)
        return np.asarray(out, dtype=np.float32).reshape(len(data) if rows is not None else -1, int(t.shape[0]))

    def ffn(self, layer: int, kind: str) -> np.ndarray:
        """down -> [n_embd, n_ff] (column i is neuron i's write); gate/up -> [n_ff, n_embd] (row i its read)."""
        return self.get(f"blk.{layer}.ffn_{kind}.weight")


def _vocab_ids(path: str) -> dict:
    """Decoded token piece (spaces normalised) -> its first id, from the GGUF's own vocabulary."""
    import gguf_tokenizer
    tok = gguf_tokenizer.load(path)
    out = {}
    for i in range(len(tok.tokens)):
        p = normalise_piece(tok.decode_id(i))
        out.setdefault(p, i)
    return out


def _samples(input_path: str, ids_path: str | None) -> list[tuple[str, dict]]:
    keep = None
    if ids_path:
        with open(ids_path) as f:
            ids = json.load(f)
        keep = set(ids.get("t", [])) | set(ids.get("f", []))
    out = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                qid = next(iter(s))
                if keep is None or qid in keep:
                    out.append((qid, s[qid]))
    return out


def weights_dla(gguf_path: str, samples: list[tuple[str, dict]], output_root: str, absolute: bool = False,
                log=print) -> dict:
    """Per sample: mean over its answer tokens of W_U[token] . (gamma * W_down[:, neuron]), per layer."""
    W = GGUFWeights(gguf_path)
    vocab = _vocab_ids(gguf_path)
    want, skipped = {}, 0
    for qid, d in samples:
        ids = [vocab.get(normalise_piece(t)) for t in d.get("answer_tokens") or []]
        ids = [i for i in ids if i is not None]
        if ids:
            want[qid] = ids
        else:
            skipped += 1
    if not want:
        raise ValueError("no sample's answer tokens are in this model's vocabulary")
    uniq = sorted({i for v in want.values() for i in v})
    pos = {t: k for k, t in enumerate(uniq)}
    unembed = "output.weight" if W.has("output.weight") else "token_embd.weight"
    U = W.get(unembed, uniq)                                         # [T, n_embd]
    if W.has("output_norm.weight"):
        U = U * W.get("output_norm.weight").reshape(1, -1)
    os.makedirs(os.path.join(output_root, "answer_tokens"), exist_ok=True)
    n_ff = int(W.t[f"blk.{W.layers[0]}.ffn_down.weight"].shape[0])
    tmp = os.path.join(output_root, ".dla-tmp.npy")
    S = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(W.layers), len(uniq), n_ff))
    for li, layer in enumerate(W.layers):
        S[li] = (U @ W.ffn(layer, "down")).astype(np.float16)       # [T, n_ff]
        log(f"PROGRESS {li + 1}/{len(W.layers)} layer {layer}")
    for qid, ids in want.items():
        v = S[:, [pos[i] for i in ids], :].astype(np.float32)
        v = np.abs(v).mean(axis=1) if absolute else v.mean(axis=1)
        np.save(os.path.join(output_root, "answer_tokens", f"act_{qid}.npy"), v.astype(np.float16))
    del S
    os.unlink(tmp)
    meta = {"n_layers": len(W.layers), "n_neurons": n_ff, "order": "flat = layer * n_neurons + neuron",
            "model_path": gguf_path, "measure": "weights-dla" + ("-abs" if absolute else ""),
            "written": len(want), "skipped_no_tokens": skipped}
    with open(os.path.join(output_root, "neuron_index.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _rel_err(q: np.ndarray, r: np.ndarray, axis: int) -> np.ndarray:
    return np.linalg.norm(q - r, axis=axis) / (np.linalg.norm(r, axis=axis) + 1e-12)


def quant_noise(ref_path: str, quant_path: str, out_dir: str, neurons: str | None = None, log=print) -> dict:
    R, Q = GGUFWeights(ref_path), GGUFWeights(quant_path)
    if R.layers != Q.layers:
        raise ValueError(f"different layer counts: {len(R.layers)} vs {len(Q.layers)}")
    rows = []
    for i, layer in enumerate(R.layers):
        parts = [_rel_err(Q.ffn(layer, "down"), R.ffn(layer, "down"), 0)]
        for kind in ("gate", "up"):
            if R.has(f"blk.{layer}.ffn_{kind}.weight") and Q.has(f"blk.{layer}.ffn_{kind}.weight"):
                parts.append(_rel_err(Q.ffn(layer, kind), R.ffn(layer, kind), 1))
        rows.append(np.sqrt(np.mean(np.stack(parts) ** 2, axis=0)))
        log(f"PROGRESS {i + 1}/{len(R.layers)} layer {layer}")
    E = np.stack(rows).astype(np.float32)                           # [layers, n_ff]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "quant_noise.npy"), E)
    flat = E.ravel()
    worst = np.argsort(flat)[::-1][:20]
    rep = {"ref": os.path.basename(ref_path), "quant": os.path.basename(quant_path),
           "n_layers": E.shape[0], "n_neurons": E.shape[1],
           "overall": {"mean": float(flat.mean()), "median": float(np.median(flat)),
                       "p95": float(np.percentile(flat, 95)), "max": float(flat.max())},
           "per_layer": [{"layer": R.layers[i], "mean": float(E[i].mean()), "p95": float(np.percentile(E[i], 95)),
                          "max": float(E[i].max())} for i in range(E.shape[0])],
           "worst": [{"layer": int(f // E.shape[1]), "neuron": int(f % E.shape[1]), "error": float(flat[f])}
                     for f in worst]}
    if neurons and not os.path.exists(neurons):
        rep["h_neurons"] = {"note": f"{neurons} not found: run the classifier first to compare H-Neurons"}
    elif neurons:
        with open(neurons) as f:
            h = json.load(f)
        idx = [(int(layer), int(n)) for layer, ns in (h.get("by_layer") or {}).items() for n in ns
               if int(layer) < E.shape[0] and int(n) < E.shape[1]]
        if idx:
            he = np.array([E[layer, n] for layer, n in idx])
            ranks = np.array([(flat < x).mean() for x in he])
            rep["h_neurons"] = {"n": len(idx), "mean": float(he.mean()), "ratio_to_all": float(he.mean() / flat.mean()),
                                "median_percentile": float(np.median(ranks) * 100),
                                "verdict": ("hit harder than the rest" if he.mean() > flat.mean() * 1.1 else
                                            "hit less than the rest" if he.mean() < flat.mean() * 0.9 else
                                            "about the same as the rest")}
    with open(os.path.join(out_dir, "quant_noise.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


# ================================================================ saliency (PyTorch)

class SaliencyHooks:
    """Scales each text layer's down_proj input by alpha and keeps it with its gradient."""

    def __init__(self, model):
        self.alpha, self.x, self.hooks, self.layer_ids = 1.0, {}, [], []
        for name, module in model.named_modules():
            if "down_proj" not in name:
                continue
            m = TEXT_LAYER_RE.search(name)
            if m:
                idx = int(m.group(1))
                self.layer_ids.append(idx)
                self.hooks.append(module.register_forward_pre_hook(self._hook(idx)))
        if not self.layer_ids:
            raise RuntimeError("no text decoder down_proj modules matched (preflight.py lists module names)")
        self.layer_ids.sort()
        emb = model.get_input_embeddings()
        # gradients need a graph; the weights stay frozen, so start it at the embeddings
        self.hooks.append(emb.register_forward_hook(lambda mod, i, o: o.requires_grad_(True)))

    def _hook(self, idx):
        def pre(module, inputs):
            x = inputs[0] * self.alpha if self.alpha != 1.0 else inputs[0]
            x.retain_grad()
            self.x[idx] = x
            return (x,) + tuple(inputs[1:])
        return pre

    def remove(self):
        for h in self.hooks:
            h.remove()


def saliency_one(model, hooks: SaliencyHooks, ids, target: tuple[int, int], method: str = "grad_x_act",
                 steps: int = 8):
    """-> [layers, tokens, neurons] |attribution| of log p(ids[s:e]) to each neuron at each position."""
    import torch
    s, e = target
    dev = next(model.parameters()).device
    x_in = ids.unsqueeze(0).to(dev)
    alphas = [1.0] if method == "grad_x_act" else [(k + 0.5) / steps for k in range(steps)]
    total, act = None, None
    for a in alphas:
        hooks.alpha, hooks.x = a, {}
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x_in).logits[0].float()
            logp = torch.log_softmax(logits[s - 1:e - 1], dim=-1)
            obj = logp.gather(1, x_in[0, s:e].unsqueeze(1)).sum()
            obj.backward()
        g = torch.stack([hooks.x[i].grad[0].detach().float().cpu() for i in hooks.layer_ids])
        total = g if total is None else total + g
        if act is None:                     # the actual activation: what the hook saw, unscaled
            act = torch.stack([(hooks.x[i][0].detach().float().cpu() / a) for i in hooks.layer_ids])
    return (act * total / len(alphas)).abs()


def run_saliency(a) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oc_common import build_sequence, find_regions
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype]
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=a.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=dtype, device_map={"": device},
                                                 trust_remote_code=a.trust_remote_code)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hooks = SaliencyHooks(model)
    os.makedirs(a.output_root, exist_ok=True)
    for loc in a.locations:
        os.makedirs(os.path.join(a.output_root, loc), exist_ok=True)
    samples = _samples(a.input_path, a.ids_path)
    stats = {"written": 0, "too_long": 0, "no_span": 0}
    n_neurons = None
    for k, (qid, d) in enumerate(samples):
        full, plen = build_sequence(tok, d["question"], d["response"])
        if len(full) > a.max_tokens:
            stats["too_long"] += 1
            continue
        reg = find_regions(tok, full, plen, d.get("answer_tokens", []), skip_think=not a.keep_think)
        target = reg["answer_tokens"] or reg["output"]
        if target is None or target[1] <= max(1, target[0]):
            stats["no_span"] += 1
            continue
        att = saliency_one(model, hooks, full, (max(1, target[0]), target[1]), a.method, a.steps)
        n_neurons = att.shape[2]
        wrote = False
        for loc in a.locations:
            if loc == "all_except_answer_tokens":
                if reg["answer_tokens"] is None:
                    continue
                s, e = reg["answer_tokens"]
                sel = torch.cat([att[:, reg["output"][0]:s], att[:, e:]], dim=1)
            else:
                span = reg[loc]
                if span is None or span[1] <= span[0]:
                    continue
                sel = att[:, span[0]:span[1]]
            if sel.shape[1]:
                np.save(os.path.join(a.output_root, loc, f"act_{qid}.npy"), sel.mean(dim=1).numpy().astype(np.float16))
                wrote = True
        stats["written"] += wrote
        print(f"PROGRESS {k + 1}/{len(samples)} {qid}", flush=True)
    hooks.remove()
    meta = {"n_layers": len(hooks.layer_ids), "n_neurons": n_neurons, "order": "flat = layer * n_neurons + neuron",
            "model_path": a.model_path, "measure": f"saliency-{a.method}", "dtype": a.dtype, **stats}
    with open(os.path.join(a.output_root, "neuron_index.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("saliency", help="gradient x activation, or integrated gradients (PyTorch, CPU-friendly)")
    s.add_argument("--model_path", required=True)
    s.add_argument("--input_path", required=True, help="answer_tokens.jsonl")
    s.add_argument("--ids_path", help="train_qids.json / test_qids.json (default: every sample)")
    s.add_argument("--output_root", required=True)
    s.add_argument("--locations", nargs="+", default=["answer_tokens"], choices=LOCATIONS)
    s.add_argument("--method", choices=["grad_x_act", "integrated"], default="grad_x_act")
    s.add_argument("--steps", type=int, default=8, help="integrated: points along the path (cost: this many passes)")
    s.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                   help="float32 runs everywhere; float16 halves memory on GPUs without bfloat16")
    s.add_argument("--device", default="auto", help="cpu, cuda, cuda:1, mps… (auto: a GPU when PyTorch sees one)")
    s.add_argument("--max_tokens", type=int, default=1024)
    s.add_argument("--keep_think", action="store_true")
    try:
        from oc_cli import add_trust_remote_code_argument
    except ImportError:
        from scripts.oc_cli import add_trust_remote_code_argument
    add_trust_remote_code_argument(s)
    w = sub.add_parser("weights", help="direct logit attribution from a GGUF's weights (numpy, no forward pass)")
    w.add_argument("--gguf", required=True)
    w.add_argument("--input_path", required=True, help="answer_tokens.jsonl")
    w.add_argument("--ids_path")
    w.add_argument("--output_root", required=True)
    w.add_argument("--abs", action="store_true", help="magnitude only (default keeps the sign: + promotes the answer)")
    q = sub.add_parser("quant-noise", help="per-neuron quantization error between two GGUFs")
    q.add_argument("--ref", required=True, help="F16, BF16 or Q8_0 GGUF")
    q.add_argument("--quant", required=True)
    q.add_argument("--neurons", help="h_neurons.json, to compare them with the rest")
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "saliency":
            r = run_saliency(a)
        elif a.cmd == "weights":
            r = weights_dla(a.gguf, _samples(a.input_path, a.ids_path), a.output_root, a.abs)
        else:
            r = quant_noise(a.ref, a.quant, a.out, a.neurons)
            r = {k: v for k, v in r.items() if k != "per_layer"}
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
