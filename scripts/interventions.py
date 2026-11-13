#!/usr/bin/env python3
"""
Component-level interventions on a Hugging Face model: which heads and neurons drove an output, and what
changes when they are removed or swapped (docs/INTERVENTIONS.md).

    trace       the predicted token; attention heads and MLP neurons ranked by direct logit attribution
    activations per-layer statistics of the residual stream, attention output, MLP output and neurons
    patch       zero- or mean-ablate one component (or one head, one neuron, one SAE feature) at chosen
                positions and report how the prediction and its logit move
    act-patch   activation patching with a clean and a corrupted prompt: put clean activations into the
                corrupted run, one (layer, position) or (layer, head) at a time, and report how much of the
                clean-vs-corrupted logit difference each restores
    path-patch  direct path patching: a sender head's corrupted output reaches only chosen receivers (a
                later layer's attention or MLP input); everything else keeps its clean value
    circuit     candidates ranked by attribution, then each one's ablation effect, and (with a corrupted
                prompt) its activation-patching effect
    neuronpedia a feature's description from Neuronpedia

Every result is one JSON document stamped with "schema_version" and "operation", the same from the
command line, the Jobs page and the MCP tools.

`trace`, `activations`, `patch` and `circuit` follow RudrenduPaul/NeuronScope (MIT), which wraps
TransformerLens for the same single-prompt operations. This version works on the Hugging Face models
OmniCortex already loads (Llama, Qwen, Gemma, Mistral, GPT-2, GPT-NeoX…) without TransformerLens. It
also adds what that tool leaves to others: clean/corrupted activation patching over chosen positions,
mean ablation, path patching and SAE-feature ablation.

    python scripts/interventions.py trace --model Qwen/Qwen3-0.6B --prompt "The capital of Japan is" --top-k 5
    python scripts/interventions.py act-patch --model gpt2 --clean "When Mary and John went to the store, John gave a drink to" \\
        --corrupt "When Mary and John went to the store, Mary gave a drink to" --target " Mary" --baseline " John" --by head
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCHEMA_VERSION = 1
COMPONENTS = ("resid_pre", "resid_post", "attn_out", "mlp_out", "mlp_post", "head", "neuron")
ATTN_NAMES = ("self_attn", "attn", "attention", "self_attention")
OPROJ_NAMES = ("o_proj", "c_proj", "dense", "out_proj", "wo")
FINAL_NORMS = ("model.norm", "model.language_model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
               "model.decoder.final_layer_norm", "model.final_layernorm")


# ====================================================================== model plumbing

def _get(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class Model:
    """A loaded causal LM and the handles interventions need."""

    def __init__(self, model, tok, device: str, name: str = ""):
        import torch
        from concept_edit import decoder_layers
        self.m, self.tok, self.dev, self.name = model, tok, device, name
        self.layers = decoder_layers(model)
        cfg = getattr(model.config, "text_config", model.config)
        self.n_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0))
        self.d_model = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0))
        self.head_dim = int(getattr(cfg, "head_dim", 0) or (self.d_model // max(1, self.n_heads)))
        self.final_norm = next((n for p in FINAL_NORMS if (n := self._try(p)) is not None), None)
        self.W_U = model.get_output_embeddings().weight.detach().float()            # [vocab, d]
        self.torch = torch

    def _try(self, path):
        try:
            return _get(self.m, path)
        except AttributeError:
            return None

    def attn(self, layer: int):
        blk = self.layers[layer]
        for n in ATTN_NAMES:
            if getattr(blk, n, None) is not None:
                return getattr(blk, n)
        raise ValueError("cannot find this layer's attention")

    def o_proj(self, layer: int):
        a = self.attn(layer)
        for n in OPROJ_NAMES:
            if getattr(a, n, None) is not None:
                return getattr(a, n)
        raise ValueError("cannot find the attention output projection")

    def mlp(self, layer: int):
        import features as FX
        return FX.mlp_of(self.layers[layer])

    def down(self, layer: int):
        import features as FX
        return FX.down_of(self.mlp(layer))

    @staticmethod
    def weight(lin):
        """[out, in] whatever the module type (GPT-2's Conv1D stores [in, out])."""
        w = lin.weight.detach().float()
        return w.t() if type(lin).__name__ == "Conv1D" else w

    def encode(self, text: str):
        ids = self.tok(text, return_tensors="pt")["input_ids"].to(self.dev)
        return ids

    def token_id(self, text: str) -> int:
        """A token's id: "#1234" names it directly, any other text is its first token."""
        if isinstance(text, int) or (str(text).startswith("#") and str(text)[1:].isdigit()):
            return int(str(text).lstrip("#"))
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"{text!r} is no token")
        return int(ids[0])

    def describe(self) -> dict:
        return {"requested_name": self.name, "resolved_name": getattr(self.m.config, "_name_or_path", self.name),
                "backend": "transformers", "device": str(self.dev), "n_layers": len(self.layers),
                "n_heads": self.n_heads, "d_model": self.d_model,
                "d_mlp": int(self.weight(self.down(0)).shape[1])}

    def logits(self, ids):
        with self.torch.no_grad():
            return self.m(ids).logits[0].float()

    def readout(self, resid_last):
        """Linearised final norm at the last position: a vector v -> its logits, as the model reads them."""
        torch = self.torch
        x = resid_last.float()
        n = self.final_norm
        if n is None:
            return lambda v: v.float() @ self.W_U.T
        w = getattr(n, "weight", None)
        w = torch.ones_like(x) if w is None else w.detach().float()
        if "rms" in type(n).__name__.lower():
            scale = x.pow(2).mean().add(getattr(n, "eps", getattr(n, "variance_epsilon", 1e-6))).rsqrt()
            if "gemma" in type(n).__name__.lower():
                w = 1 + w
            return lambda v: (v.float() * w * scale) @ self.W_U.T
        std = (x - x.mean()).pow(2).mean().add(getattr(n, "eps", 1e-5)).sqrt()
        return lambda v: ((v.float() - v.float().mean()) * w / std) @ self.W_U.T


def load(model_id: str, device: str = "auto", dtype: str = "float32") -> Model:
    import features as FX
    m, tok, dev = FX.load_model(model_id, dtype, device)
    return Model(m, tok, dev, model_id)


class Hooks:
    """Forward hooks that record or replace activations; removed on exit."""

    def __init__(self, M: Model):
        self.M, self.h, self.grab = M, [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.h:
            h.remove()
        self.h = []

    def _site(self, comp: str, layer: int):
        """-> (module, "out" | "in")."""
        M = self.M
        if comp == "resid_pre":
            return M.layers[layer], "in"
        if comp == "resid_post":
            return M.layers[layer], "out"
        if comp == "attn_out":
            return M.attn(layer), "out"
        if comp == "mlp_out":
            return M.mlp(layer), "out"
        if comp in ("mlp_post", "neuron"):
            return M.down(layer), "in"
        if comp == "head":
            return M.o_proj(layer), "in"
        raise ValueError(f"component: one of {', '.join(COMPONENTS)}")

    def on(self, comp: str, layer: int, fn=None, key=None):
        """Record the activation at (comp, layer) under key, and replace it with fn(activation) when given."""
        mod, kind = self._site(comp, layer)
        key = key or (comp, layer)

        def apply(x):
            self.grab[key] = x.detach()
            return fn(x) if fn is not None else None
        if kind == "out":
            def h(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                r = apply(t)
                if r is None:
                    return None
                return (r, *out[1:]) if isinstance(out, tuple) else r
            self.h.append(mod.register_forward_hook(h))
        else:
            def h(_m, args, kwargs):
                if args:
                    r = apply(args[0])
                    return None if r is None else ((r, *args[1:]), kwargs)
                hs = kwargs.get("hidden_states")
                r = apply(hs)
                return None if r is None else (args, {**kwargs, "hidden_states": r})
            self.h.append(mod.register_forward_pre_hook(h, with_kwargs=True))
        return self


def _positions(spec, n: int) -> list[int]:
    """'last', 'all', '3', '2,5,-1', '0:4' -> token positions."""
    if spec in (None, "", "last"):
        return [n - 1]
    if spec == "all":
        return list(range(n))
    out = []
    for part in str(spec).split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a or 0) % n if a else 0, (int(b) % n if b and int(b) < 0 else int(b or n))))
        else:
            out.append(int(part) % n)
    return sorted(set(out))


def _doc(op: str, M: Model, **kw) -> dict:
    return {"schema_version": SCHEMA_VERSION, "operation": op, "model": M.describe(), **kw,
            "created": int(time.time())}


def _tok(M: Model, i: int) -> str:
    return M.tok.decode([int(i)])


# ====================================================================== operations

def trace(M: Model, prompt: str, top_k: int = 10, target: str | None = None) -> dict:
    """Heads and neurons ranked by direct logit attribution to the predicted (or given) token at the last
    position; neurons also by activation, as RudrenduPaul/NeuronScope ranks them."""
    torch = M.torch
    ids = M.encode(prompt)
    H = Hooks(M)
    with H:
        for L in range(len(M.layers)):
            H.on("head", L).on("mlp_post", L)
        H.on("resid_post", len(M.layers) - 1, key="final")
        logits = M.logits(ids)
    pred = int(logits[-1].argmax())
    tid = M.token_id(target) if target else pred
    read = M.readout(H.grab["final"][0, -1])
    u = M.W_U[tid]
    heads, neurons = [], []
    for L in range(len(M.layers)):
        z = H.grab[("head", L)][0, -1].float()                                   # [n_heads * head_dim]
        Wo = M.weight(M.o_proj(L))                                               # [d, n_heads * head_dim]
        for h in range(M.n_heads):
            sl = slice(h * M.head_dim, (h + 1) * M.head_dim)
            out = Wo[:, sl] @ z[sl]
            heads.append({"layer": L, "head_index": h, "logit_attribution": float(read(out)[tid])})
        a = H.grab[("mlp_post", L)][0, -1].float()                               # [d_mlp]
        Wd = M.weight(M.down(L))                                                 # [d, d_mlp]
        # neuron i writes a_i * Wd[:, i]; its logit effect is linear in a_i
        per = read(Wd.T * a[:, None])[:, tid]                                    # [d_mlp]
        top = torch.topk(per.abs(), min(top_k, per.numel())).indices
        neurons += [{"layer": L, "neuron_index": int(i), "activation": float(a[i]),
                     "logit_attribution": float(per[i])} for i in top]
    heads.sort(key=lambda x: -abs(x["logit_attribution"]))
    by_attr = sorted(neurons, key=lambda x: -abs(x["logit_attribution"]))[:top_k]
    by_act = sorted(neurons, key=lambda x: -abs(x["activation"]))[:top_k]
    probs = torch.softmax(logits[-1], -1)
    return _doc("trace", M, prompt=prompt, predicted_token=_tok(M, pred), predicted_token_id=pred,
                target_token=_tok(M, tid), target_token_id=tid, target_prob=float(probs[tid]),
                top_heads=heads[:top_k], top_neurons=by_attr, top_neurons_by_activation=by_act,
                method="direct logit attribution through a linearised final norm, at the last position")


def activations(M: Model, prompt: str) -> dict:
    ids = M.encode(prompt)
    H = Hooks(M)
    with H:
        for L in range(len(M.layers)):
            for c in ("resid_post", "attn_out", "mlp_out", "mlp_post"):
                H.on(c, L)
        M.logits(ids)
    rows = []
    for L in range(len(M.layers)):
        row = {"layer": L}
        for c in ("resid_post", "attn_out", "mlp_out", "mlp_post"):
            x = H.grab[(c, L)][0].float()
            norms = x.norm(dim=-1)
            row[c] = {"shape": list(x.shape), "mean": float(x.mean()), "std": float(x.std()),
                      "min": float(x.min()), "max": float(x.max()),
                      "max_norm_position": int(norms.argmax()), "max_norm": float(norms.max())}
        rows.append(row)
    return _doc("activations", M, prompt=prompt, n_tokens=int(ids.shape[1]), layers=rows)


def _ablate_fn(M: Model, comp: str, positions: list[int], mode: str, index: int | None, mean_of=None):
    """fn(activation) for patch: zero or mean at the given positions, the whole component, one head's slice,
    or one neuron."""
    def fn(x):
        x = x.clone()
        if comp == "head":
            sl = slice(index * M.head_dim, (index + 1) * M.head_dim)
        elif comp == "neuron":
            sl = slice(index, index + 1)
        else:
            sl = slice(None)
        for p in positions:
            if mode == "mean":
                ref = mean_of if mean_of is not None else x[0].mean(dim=0)
                x[0, p, sl] = ref[sl] if ref.dim() == 1 else ref[p, sl]
            else:
                x[0, p, sl] = 0
        return x
    return fn


def patch(M: Model, prompt: str, layer: int, component: str, mode: str = "zero", positions="last",
          index: int | None = None, target: str | None = None, mean_prompts: list[str] | None = None,
          sae: str | None = None, feature: int | None = None, value: float = 0.0) -> dict:
    """One ablation (or SAE-feature edit) and its effect on the prediction."""
    torch = M.torch
    if not 0 <= layer < len(M.layers):
        raise ValueError(f"layer {layer}: the model has {len(M.layers)} layers (0..{len(M.layers) - 1})")
    if component in ("head", "neuron") and index is None:
        raise ValueError(f"--index: which {component}")
    ids = M.encode(prompt)
    pos = _positions(positions, ids.shape[1])
    clean = M.logits(ids)
    pred = int(clean[-1].argmax())
    tid = M.token_id(target) if target else pred
    mean_of = None
    if mode == "mean" and mean_prompts:
        acc = []
        for p in mean_prompts:
            H = Hooks(M)
            with H:
                H.on("head" if component == "head" else component, layer)
                M.logits(M.encode(p))
            acc.append(H.grab[("head" if component == "head" else component, layer)][0].float().mean(0))
        mean_of = torch.stack(acc).mean(0).to(clean.dtype)
    H = Hooks(M)
    if sae:
        fn, site = _sae_edit(M, sae, layer, feature, value, pos)
        H.on(site, layer, fn)
        what = {"sae": sae, "feature": feature, "value": value}
    else:
        H.on(component, layer, _ablate_fn(M, component, pos, mode, index, mean_of))
        what = {"component": component, "mode": mode, "index": index}
    with H:
        after = M.logits(ids)
    pa, pc = torch.softmax(after[-1], -1), torch.softmax(clean[-1], -1)
    new_pred = int(after[-1].argmax())
    return _doc("patch", M, prompt=prompt, layer=layer, positions=pos, **what,
                target_token=_tok(M, tid), clean_logit=float(clean[-1, tid]), patched_logit=float(after[-1, tid]),
                logit_delta=float(after[-1, tid] - clean[-1, tid]), clean_prob=float(pc[tid]),
                patched_prob=float(pa[tid]), clean_prediction=_tok(M, pred), patched_prediction=_tok(M, new_pred),
                prediction_changed=new_pred != pred,
                kl_divergence=float(torch.sum(pc * (torch.log(pc + 1e-12) - torch.log(pa + 1e-12)))))


def _sae_edit(M: Model, path: str, layer: int, feature: int, value: float, pos: list[int]):
    """Set one SAE feature at the given positions; the SAE's reconstruction error is kept, so only that
    feature's decoder direction changes the stream."""
    import sae as S
    import features as FX
    d, meta = S.load(path, device=str(M.dev))
    site = {"resid": "resid_post", "mlp_out": "mlp_out", "neurons": "mlp_post", "mlp_in": "resid_pre"}.get(
        meta.get("site", "resid"), "resid_post")
    if feature is None:
        raise ValueError("--feature: which SAE feature")

    def fn(x):
        x = x.clone()
        for p in pos:
            v = x[0, p].float()
            f = d.encode(v[None])[0]
            delta = (value - float(f[feature])) * d.W_dec[feature].float()
            x[0, p] = (v + delta).to(x.dtype)
        return x
    return fn, site


def _logit_diff(logits, t: int, b: int | None) -> float:
    return float(logits[-1, t] - logits[-1, b]) if b is not None else float(logits[-1, t])


def act_patch(M: Model, clean: str, corrupt: str, target: str, baseline: str | None = None,
              by: str = "resid", layers=None, positions="all") -> dict:
    """Clean activations into the corrupted run, one at a time. Score = (patched - corrupt) / (clean - corrupt)
    of the target-minus-baseline logit difference: 1 means that activation alone restores the clean answer."""
    ci, xi = M.encode(clean), M.encode(corrupt)
    if ci.shape != xi.shape:
        raise ValueError(f"the prompts tokenize to different lengths ({ci.shape[1]} and {xi.shape[1]}): "
                         "activation patching compares them position by position")
    t, b = M.token_id(target), (M.token_id(baseline) if baseline else None)
    comp = {"resid": "resid_pre", "attn": "attn_out", "mlp": "mlp_out", "head": "head"}[by]
    Ls = list(range(len(M.layers))) if layers in (None, "", "all") else [int(x) for x in str(layers).split(",")]
    H = Hooks(M)
    with H:
        for L in Ls:
            H.on(comp, L)
        clean_logits = M.logits(ci)
    cache = {k: v for k, v in H.grab.items()}
    corrupt_logits = M.logits(xi)
    ld_c, ld_x = _logit_diff(clean_logits, t, b), _logit_diff(corrupt_logits, t, b)
    span = ld_c - ld_x
    pos = _positions(positions, ci.shape[1])
    grid = []
    for L in Ls:
        src = cache[(comp, L)]
        units = range(M.n_heads) if by == "head" else [None]
        for u in units:
            row = []
            for p in (pos if by != "head" else [None]):
                def fn(x, p=p, u=u):
                    x = x.clone()
                    ps = pos if p is None else [p]
                    sl = slice(u * M.head_dim, (u + 1) * M.head_dim) if u is not None else slice(None)
                    for q in ps:
                        x[0, q, sl] = src[0, q, sl]
                    return x
                with Hooks(M).on(comp, L, fn):
                    ld = _logit_diff(M.logits(xi), t, b)
                row.append(round((ld - ld_x) / span, 4) if abs(span) > 1e-6 else None)
            grid.append({"layer": L, **({"head": u} if u is not None else {}),
                         **({"restored": row[0]} if by == "head" else {"restored_by_position": row})})
    toks = [_tok(M, i) for i in ci[0].tolist()]
    best = sorted(grid, key=lambda g: -(g.get("restored") if by == "head" else max(x or 0 for x in g["restored_by_position"])))
    return _doc("act-patch", M, clean_prompt=clean, corrupt_prompt=corrupt, target_token=target, baseline_token=baseline,
                by=by, positions=pos if by != "head" else "all", tokens=toks, clean_logit_diff=ld_c,
                corrupt_logit_diff=ld_x, grid=grid, top=best[:10],
                method="activation patching (denoising): one clean activation at a time into the corrupted run")


def path_patch(M: Model, clean: str, corrupt: str, target: str, baseline: str | None, sender_layer: int,
               sender_head: int, receivers: list[tuple[int, str]]) -> dict:
    """The corrupted output of one head, delivered only to the receivers (a later layer's "attn" or "mlp"
    input); every other component sees the clean run. Because everything in between keeps its clean value,
    the receiver's input changes by exactly (corrupt - clean) of the sender's output: this is the direct
    path from sender to receivers."""
    torch = M.torch
    ci, xi = M.encode(clean), M.encode(corrupt)
    if ci.shape != xi.shape:
        raise ValueError("the prompts tokenize to different lengths")
    t, b = M.token_id(target), (M.token_id(baseline) if baseline else None)
    outs = {}
    for name, ids in (("clean", ci), ("corrupt", xi)):
        H = Hooks(M)
        with H:
            H.on("head", sender_layer)
            outs[name + "_logits"] = M.logits(ids)
        z = H.grab[("head", sender_layer)][0].float()
        sl = slice(sender_head * M.head_dim, (sender_head + 1) * M.head_dim)
        Wo = M.weight(M.o_proj(sender_layer))
        outs[name] = (z[:, sl] @ Wo[:, sl].T)                                     # [pos, d]
    delta = (outs["corrupt"] - outs["clean"])
    ld_c, ld_x = _logit_diff(outs["clean_logits"], t, b), _logit_diff(outs["corrupt_logits"], t, b)
    results, hooks = [], []
    try:
        for L, kind in receivers:
            if L <= sender_layer:
                raise ValueError(f"receiver layer {L} must come after the sender's layer {sender_layer}")
            blk = M.layers[L]
            norm = getattr(blk, "input_layernorm" if kind == "attn" else "post_attention_layernorm", None) or \
                getattr(blk, "ln_1" if kind == "attn" else "ln_2", None)
            if norm is None:
                raise ValueError(f"cannot find layer {L}'s {kind} input norm")

            def pre(_m, args, d=delta):
                return (args[0] + d[None].to(args[0].dtype),) + tuple(args[1:])
            hooks.append(norm.register_forward_pre_hook(pre))
        ld = _logit_diff(M.logits(ci), t, b)
    finally:
        for h in hooks:
            h.remove()
    span = ld_x - ld_c
    results = {"patched_logit_diff": ld, "effect": round((ld - ld_c) / span, 4) if abs(span) > 1e-6 else None}
    return _doc("path-patch", M, clean_prompt=clean, corrupt_prompt=corrupt, target_token=target,
                baseline_token=baseline, sender={"layer": sender_layer, "head": sender_head},
                receivers=[{"layer": L, "input": k} for L, k in receivers], clean_logit_diff=ld_c,
                corrupt_logit_diff=ld_x, **results,
                method="direct path patching: the sender's corrupted output reaches only the receivers' inputs; "
                       "effect = share of the clean-to-corrupted change this path alone produces")


def circuit(M: Model, prompt: str, top_k: int = 10, corrupt: str | None = None, baseline: str | None = None) -> dict:
    """Candidates by attribution, each then ablated on its own (and, with a corrupted prompt, patched)."""
    tr = trace(M, prompt, top_k)
    rows = []
    for h in tr["top_heads"]:
        r = patch(M, prompt, h["layer"], "head", index=h["head_index"])
        rows.append({"kind": "head", "layer": h["layer"], "index": h["head_index"], "attribution": h["logit_attribution"],
                     "ablation_logit_delta": r["logit_delta"], "flips_prediction": r["prediction_changed"]})
    for n in tr["top_neurons"]:
        r = patch(M, prompt, n["layer"], "neuron", index=n["neuron_index"])
        rows.append({"kind": "neuron", "layer": n["layer"], "index": n["neuron_index"], "attribution": n["logit_attribution"],
                     "ablation_logit_delta": r["logit_delta"], "flips_prediction": r["prediction_changed"]})
    if corrupt:
        ap = act_patch(M, prompt, corrupt, tr["predicted_token"], baseline, by="head",
                       layers=",".join(str(x) for x in sorted({r["layer"] for r in rows if r["kind"] == "head"})))
        restored = {(g["layer"], g["head"]): g["restored"] for g in ap["grid"]}
        for r in rows:
            if r["kind"] == "head":
                r["patching_restored"] = restored.get((r["layer"], r["index"]))
    rows.sort(key=lambda r: -abs(r["ablation_logit_delta"]))
    return _doc("circuit", M, prompt=prompt, predicted_token=tr["predicted_token"], corrupt_prompt=corrupt,
                components=rows,
                method="candidates by direct logit attribution; effect of zero-ablating each alone"
                       + ("; and of patching each clean head into the corrupted run" if corrupt else "")
                       + ". Single components: interactions between them are not measured "
                         "(use path-patch, or feature_circuits.py graph for transcoder attribution graphs)")


def neuronpedia(model_id: str, layer: str, index: int, timeout: float = 20) -> dict:
    """A feature's descriptions from Neuronpedia, e.g. model gemma-2-2b, layer 12-gemmascope-res-16k."""
    import urllib.request
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{int(index)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OmniCortex"}), timeout=timeout) as r:
        j = json.loads(r.read())
    return {"schema_version": SCHEMA_VERSION, "operation": "neuronpedia", "url": url,
            "model": model_id, "layer": layer, "index": int(index),
            "explanations": [e.get("description") for e in j.get("explanations") or []],
            "max_activation": j.get("maxActApprox"), "frac_nonzero": j.get("frac_nonzero")}


# ====================================================================== command line

def _receivers(items) -> list[tuple[int, str]]:
    out = []
    for it in items or []:
        L, _, kind = it.partition(":")
        if kind not in ("attn", "mlp"):
            raise ValueError("--receiver LAYER:attn or LAYER:mlp")
        out.append((int(L), kind))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, prompt=True):
        s.add_argument("--model", required=True)
        s.add_argument("--device", default="auto")
        s.add_argument("--dtype", default="float32")
        s.add_argument("--out", help="write the JSON here as well")
        if prompt:
            s.add_argument("--prompt", required=True)
    s = sub.add_parser("trace"); common(s); s.add_argument("--top-k", type=int, default=10); s.add_argument("--target")
    s = sub.add_parser("activations"); common(s)
    s = sub.add_parser("patch"); common(s)
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--component", choices=COMPONENTS, default="mlp_out")
    s.add_argument("--index", type=int, help="the head or neuron, for --component head/neuron")
    s.add_argument("--mode", choices=["zero", "mean"], default="zero")
    s.add_argument("--positions", default="last", help="last, all, 3, 2,5,-1 or 0:4")
    s.add_argument("--mean-prompt", action="append", help="prompts whose average activation replaces it (mean mode)")
    s.add_argument("--target")
    s.add_argument("--sae", help="an SAE folder: edit one of its features instead of a component")
    s.add_argument("--feature", type=int)
    s.add_argument("--value", type=float, default=0.0)
    s = sub.add_parser("act-patch"); common(s, prompt=False)
    s.add_argument("--clean", required=True); s.add_argument("--corrupt", required=True)
    s.add_argument("--target", required=True); s.add_argument("--baseline")
    s.add_argument("--by", choices=["resid", "attn", "mlp", "head"], default="resid")
    s.add_argument("--layers", default="all"); s.add_argument("--positions", default="all")
    s = sub.add_parser("path-patch"); common(s, prompt=False)
    s.add_argument("--clean", required=True); s.add_argument("--corrupt", required=True)
    s.add_argument("--target", required=True); s.add_argument("--baseline")
    s.add_argument("--sender", required=True, help="LAYER.HEAD, e.g. 9.6")
    s.add_argument("--receiver", action="append", required=True, help="LAYER:attn or LAYER:mlp (repeat)")
    s = sub.add_parser("circuit"); common(s); s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--corrupt"); s.add_argument("--baseline")
    s = sub.add_parser("neuronpedia")
    s.add_argument("--np-model", required=True, help="Neuronpedia model id, e.g. gemma-2-2b")
    s.add_argument("--np-layer", required=True, help="e.g. 12-gemmascope-res-16k")
    s.add_argument("--index", type=int, required=True)
    s.add_argument("--out")
    a = p.parse_args(argv)
    try:
        if a.cmd == "neuronpedia":
            r = neuronpedia(a.np_model, a.np_layer, a.index)
        else:
            M = load(a.model, a.device, a.dtype)
            if a.cmd == "trace":
                r = trace(M, a.prompt, a.top_k, a.target)
            elif a.cmd == "activations":
                r = activations(M, a.prompt)
            elif a.cmd == "patch":
                r = patch(M, a.prompt, a.layer, a.component, a.mode, a.positions, a.index, a.target, a.mean_prompt,
                          a.sae, a.feature, a.value)
            elif a.cmd == "act-patch":
                r = act_patch(M, a.clean, a.corrupt, a.target, a.baseline, a.by, a.layers, a.positions)
            elif a.cmd == "path-patch":
                L, _, h = a.sender.partition(".")
                r = path_patch(M, a.clean, a.corrupt, a.target, a.baseline, int(L), int(h), _receivers(a.receiver))
            else:
                r = circuit(M, a.prompt, a.top_k, a.corrupt, a.baseline)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    text = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
