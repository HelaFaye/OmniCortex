#!/usr/bin/env python3
"""
Find the direction (or subspace) a model uses for a concept and edit it out of
the weights, while measuring how much else changes.

You give two prompt sets that differ in one concept, A (concept) and B
(baseline). Examples:
- confidently-answered vs declined questions, for the hallucination
  direction;
- flattering vs neutral replies, for sycophancy;
- prompts the model refuses vs ones it answers, for refusal.

You also give a neutral set for measuring damage. The script then:

1. **Collects activations**: the residual stream at the last prompt token,
   after every decoder layer.
2. **Isolates the subspace**:
   - difference of means (one direction), or PCA of the concept activations
     around the baseline mean (`--k` directions, a "concept cone");
   - **biprojection**: each direction is made orthogonal to the baseline's
     own mean direction, so the edit does not also remove what B has.
3. **Edits the weights that write into the residual stream**: attention
   output (W_O: o_proj / out_proj) and MLP output (W_out: down_proj / fc2).
   - The edit is W' = W - w·B·Bᵀ·W, which projects B out of what the layer
     writes.
   - **Norm-preserving**: every column (one input feature's write vector) is
     rescaled to its original length, so later layers see inputs of the
     usual size.
   - The per-layer strength w follows a kernel: a peak at one layer, falling
     linearly to a floor over a distance (the shape Heretic optimizes).
4. **Monitors damage**: the KL divergence between the original and edited
   models' next-token distributions on the neutral set. It also reports
   *remaining separation*, how far A still differs from B along the
   directions (1.0 = untouched, 0 = gone), and optionally a behaviour rate
   from a regex over greedy replies to the concept set.

With `--search N`, the kernel and the direction layer are optimized
(Optuna TPE when installed, random search otherwise). The objective is
remaining separation (and behaviour rate) plus `--kl-weight` × KL, under
`--max-kl`. This is the Heretic approach.

    python scripts/concept_edit.py --model org/model --concept a.txt --baseline b.txt --neutral n.txt \\
        --out runs/edit --search 40 --max-kl 0.3
    python scripts/concept_edit.py ... --kernel 1.0,20,0.4,12 --direction-layer 18 --out runs/edit
    python scripts/merge_export.py --base org/model --adapter runs/edit/model --out runs/edit-gguf --gguf Q4_K_M

Prompt files: .txt (one per line), .jsonl ({"text"} or {"prompt"}), or a
JSON list. Prompts are wrapped in the model's chat template when it has one.

What gets removed is whatever separates A from B in your data. Choose the
sets with care, check the result with TestQA and the gate (retrain_gate.py),
and keep the original. Edits that remove a model's safety behaviour are your
responsibility, under the model's licence and the law where you use it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

WRITE_NAMES = {"attn": ("o_proj", "out_proj", "dense", "c_proj", "wo"),
               "mlp": ("down_proj", "fc2", "c_proj", "w2", "dense_4h_to_h")}


# ------------------------------------------------------------------ data

def load_prompts(path: str, limit: int = 0) -> list[str]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        out = []
        for line in text.splitlines():
            if line.strip():
                r = json.loads(line)
                out.append(r.get("text") or r.get("prompt") or r.get("instruction") or "")
    elif p.suffix == ".json":
        d = json.loads(text)
        out = [x if isinstance(x, str) else (x.get("text") or x.get("prompt") or "") for x in d]
    else:
        out = [l.strip() for l in text.splitlines()]
    out = [x for x in out if x]
    if not out:
        raise SystemExit(f"{path}: no prompts")
    return out[:limit] if limit else out


def wrap(tok, prompt: str) -> str:
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            pass
    return prompt


# ------------------------------------------------------------------ model access

def decoder_layers(model):
    """The decoder blocks, in order."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "transformer.h",
                 "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise SystemExit("cannot find the decoder layers of this architecture")


def write_matrices(layer, components: tuple[str, ...]) -> list[tuple[str, object]]:
    """(name, Linear) for the matrices of one block that write into the residual stream."""
    import torch.nn as nn
    out = []
    for comp in components:
        names = WRITE_NAMES[comp]
        parent_names = ("self_attn", "attn", "attention", "self_attention") if comp == "attn" else ("mlp", "feed_forward", "ffn")
        for pn in parent_names:
            parent = getattr(layer, pn, None)
            if parent is None:
                continue
            for n in names:
                m = getattr(parent, n, None)
                if isinstance(m, nn.Linear):
                    out.append((f"{pn}.{n}", m))
                    break
            break
    return out


# ------------------------------------------------------------------ 1. activations

def collect(model, tok, prompts: list[str], batch: int = 8):
    """-> float32 tensor [n_prompts, n_layers + 1, d]: the residual stream at the
    last prompt token, before layer 0 (index 0) and after each layer (1..L)."""
    import torch
    rows = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok([wrap(tok, p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True,
                      add_special_tokens=not bool(getattr(tok, "chat_template", None))).to(dev)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
            rows.append(hs.float().cpu())
    return torch.cat(rows)


# ------------------------------------------------------------------ 2. subspace

def directions(A, B, method: str = "mean", k: int = 1, biproject: bool = True):
    """Per residual position, an orthonormal basis [L+1, d, k] of what separates A
    from B, plus the separation along it. mean: the difference of means (k=1).
    pca: the top-k directions of A around B's mean (the first is close to the
    mean difference; the others are the cone's further dimensions)."""
    import torch
    n_pos, d = A.shape[1], A.shape[2]
    k = 1 if method == "mean" else max(1, k)
    basis = torch.zeros(n_pos, d, k)
    for l in range(n_pos):
        a, b = A[:, l, :], B[:, l, :]
        mb = b.mean(0)
        if method == "mean":
            vecs = [(a.mean(0) - mb)]
        else:
            D = a - mb
            _, _, vt = torch.linalg.svd(D, full_matrices=False)
            vecs = [vt[i] * torch.sign(vt[i] @ (a.mean(0) - mb) + 1e-12) for i in range(min(k, vt.shape[0]))]
        cols = []
        mbn = mb / (mb.norm() + 1e-8)
        for v in vecs:
            if biproject:
                v = v - (v @ mbn) * mbn            # do not remove what the baseline itself has
            for c in cols:                           # Gram-Schmidt within the cone
                v = v - (v @ c) * c
            n = v.norm()
            if n > 1e-8:
                cols.append(v / n)
        for i, c in enumerate(cols):
            basis[l, :, i] = c
    return basis


def separation(A, B, basis) -> "torch.Tensor":
    """Per position: |mean(A) - mean(B)| projected on the basis (the concept's strength)."""
    import torch
    diff = (A.mean(0) - B.mean(0))                                   # [L+1, d]
    return torch.linalg.norm(torch.einsum("ld,ldk->lk", diff, basis), dim=-1)


def global_basis(basis, index: float):
    """A single basis taken at a fractional residual position (interpolated, re-orthonormalized)."""
    import torch
    lo = int(math.floor(index))
    hi = min(lo + 1, basis.shape[0] - 1)
    t = index - lo
    m = (1 - t) * basis[lo] + t * basis[hi]
    q, _ = torch.linalg.qr(m)
    return q


# ------------------------------------------------------------------ 3. the edit

def kernel(n_layers: int, max_w: float, pos: float, min_w: float, dist: float) -> list[float]:
    """Strength per layer: max_w at `pos`, falling linearly to min_w at `dist` layers away, 0 beyond."""
    out = []
    for l in range(n_layers):
        r = abs(l - pos)
        out.append(0.0 if r > dist else max_w - (max_w - min_w) * (r / dist if dist > 0 else 0.0))
    return out


class Editor:
    """Applies and undoes edits in place, keeping the original weights on the CPU."""

    def __init__(self, model, components=("attn", "mlp")):
        self.model = model
        self.layers = decoder_layers(model)
        self.targets = [write_matrices(L, components) for L in self.layers]
        if not any(self.targets):
            raise SystemExit("no residual-writing matrices found (o_proj/down_proj...) in this architecture")
        self.orig = {}

    def reset(self):
        import torch
        with torch.no_grad():
            for (li, name), w in self.orig.items():
                dict(self.targets[li])[name].weight.copy_(w)

    def apply(self, basis_for_layer, weights: list[float], norm_preserve: bool = True):
        """basis_for_layer(l) -> [d, k]. Projects the basis out of what layer l writes,
        with strength weights[l]."""
        import torch
        self.reset()
        with torch.no_grad():
            for li, mats in enumerate(self.targets):
                w = weights[li] if li < len(weights) else 0.0
                if w == 0:
                    continue
                Bm = basis_for_layer(li)
                for name, lin in mats:
                    key = (li, name)
                    if key not in self.orig:
                        self.orig[key] = lin.weight.detach().to("cpu", copy=True)
                    W = lin.weight.data.float()
                    Bd = Bm.to(W.device, W.dtype)
                    norms = W.norm(dim=0, keepdim=True)
                    Wn = W - w * Bd @ (Bd.T @ W)
                    if norm_preserve:
                        Wn = Wn * (norms / Wn.norm(dim=0, keepdim=True).clamp_min(1e-8))
                    lin.weight.data.copy_(Wn.to(lin.weight.dtype))


# ------------------------------------------------------------------ 4. damage and effect

def next_token_logprobs(model, tok, prompts: list[str], batch: int = 8):
    import torch
    tok.padding_side = "left"
    out = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok([wrap(tok, p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True,
                      add_special_tokens=not bool(getattr(tok, "chat_template", None))).to(dev)
            logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            out.append(torch.log_softmax(logits, -1).cpu())
    return torch.cat(out)


def kl(ref, new) -> float:
    """Mean KL(original || edited) over the neutral prompts' next-token distributions."""
    import torch
    return float((torch.exp(ref) * (ref - new)).sum(-1).mean())


def marker_rate(model, tok, prompts: list[str], pattern: str, max_new: int = 48, batch: int = 8) -> float:
    import torch
    rx = re.compile(pattern, re.I)
    hits = 0
    tok.padding_side = "left"
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok([wrap(tok, p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True,
                      add_special_tokens=not bool(getattr(tok, "chat_template", None))).to(dev)
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                hits += bool(rx.search(tok.decode(row, skip_special_tokens=True)))
    return hits / max(1, len(prompts))


# ------------------------------------------------------------------ running

class Run:
    def __init__(self, a):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.a = a
        dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
        dtype = torch.float32 if dev == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, trust_remote_code=False).to(dev).eval()
        self.A = load_prompts(a.concept, a.limit)
        self.B = load_prompts(a.baseline, a.limit)
        self.N = load_prompts(a.neutral, a.limit) if a.neutral else self.B[: max(8, len(self.B) // 2)]
        comps = tuple(c for c in a.components.split(",") if c)
        self.ed = Editor(self.model, comps)
        self.n_layers = len(self.ed.layers)
        t0 = time.time()
        print(f"collecting activations: {len(self.A)} concept, {len(self.B)} baseline prompts", flush=True)
        self.HA = collect(self.model, self.tok, self.A, a.batch)
        self.HB = collect(self.model, self.tok, self.B, a.batch)
        self.basis = directions(self.HA, self.HB, a.method, a.k, not a.no_biproject)
        self.sep0 = separation(self.HA, self.HB, self.basis)
        self.ref = next_token_logprobs(self.model, self.tok, self.N, a.batch)
        self.marker0 = marker_rate(self.model, self.tok, self.A, a.marker) if a.marker else None
        print(f"  done in {time.time() - t0:.0f}s; strongest separation after layer "
              f"{int(self.sep0[1:].argmax())} ({float(self.sep0[1:].max()):.3f})"
              + (f"; behaviour rate {self.marker0:.0%}" if self.marker0 is not None else ""), flush=True)

    def basis_fn(self, direction_layer):
        """direction_layer None: each layer's own direction (after that layer); a number:
        one global direction taken after that (fractional) layer."""
        if direction_layer is None:
            return lambda li: self.basis[li + 1]
        g = global_basis(self.basis, float(direction_layer) + 1)
        return lambda li: g

    def evaluate(self, params: dict) -> dict:
        w = kernel(self.n_layers, params["max_w"], params["pos"], params["min_w"], params["dist"])
        self.ed.apply(self.basis_fn(params.get("direction_layer")), w, not self.a.plain)
        HA = collect(self.model, self.tok, self.A, self.a.batch)
        HB = collect(self.model, self.tok, self.B, self.a.batch)
        sep = separation(HA, HB, self.basis)
        remaining = float((sep[1:] / self.sep0[1:].clamp_min(1e-6)).mean())
        res = {"params": params, "kl": round(kl(self.ref, next_token_logprobs(self.model, self.tok, self.N,
                                                                              self.a.batch)), 5),
               "remaining_separation": round(remaining, 4), "layer_weights": [round(x, 3) for x in w]}
        if self.a.marker:
            res["behaviour_rate"] = round(marker_rate(self.model, self.tok, self.A, self.a.marker), 4)
        res["objective"] = round(objective(res, self.a), 5)
        return res

    def search(self) -> list[dict]:
        a = self.a
        L = self.n_layers
        rng = random.Random(a.seed)
        results = []

        def suggest(get):
            mx = get("max_w", 0.4, 1.5)
            return {"max_w": round(mx, 3), "pos": round(get("pos", 0.0, L - 1.0), 2),
                    "min_w": round(get("min_w_frac", 0.0, 1.0) * mx, 3),
                    "dist": round(get("dist", 1.0, max(1.0, L / 2)), 2),
                    "direction_layer": None if a.per_layer else round(get("direction_layer", 0.2 * (L - 1),
                                                                          0.95 * (L - 1)), 2)}
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed))

            def obj(trial):
                p = suggest(lambda n, lo, hi: trial.suggest_float(n, lo, hi))
                r = self.evaluate(p)
                results.append(r)
                self._log(len(results), r)
                return r["objective"]
            study.optimize(obj, n_trials=a.search)
        except ImportError:
            for _ in range(a.search):
                r = self.evaluate(suggest(lambda n, lo, hi: rng.uniform(lo, hi)))
                results.append(r)
                self._log(len(results), r)
        return results

    def _log(self, i, r):
        print(f"[{i}] KL {r['kl']:.4f}  remaining {r['remaining_separation']:.3f}"
              + (f"  behaviour {r['behaviour_rate']:.0%}" if "behaviour_rate" in r else "")
              + f"  objective {r['objective']:.4f}  {json.dumps(r['params'])}", flush=True)


def objective(r: dict, a) -> float:
    score = r["remaining_separation"] + (r.get("behaviour_rate") or 0.0) + a.kl_weight * r["kl"]
    if a.max_kl and r["kl"] > a.max_kl:
        score += 10 + r["kl"]                       # over the damage limit: never preferred
    return score


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="Hugging Face id or folder (bf16/fp16 weights, not GGUF)")
    p.add_argument("--concept", required=True, help="prompts with the concept (A)")
    p.add_argument("--baseline", required=True, help="matched prompts without it (B)")
    p.add_argument("--neutral", help="ordinary prompts for the KL damage meter (default: part of B)")
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["mean", "pca"], default="mean")
    p.add_argument("--k", type=int, default=1, help="pca: directions in the concept subspace")
    p.add_argument("--no-biproject", action="store_true", help="keep the baseline's mean component in the direction")
    p.add_argument("--components", default="attn,mlp", help="attn (W_O), mlp (W_out), or both")
    p.add_argument("--plain", action="store_true", help="plain projection, without norm preservation")
    p.add_argument("--kernel", help="max_w,pos,min_w,dist: strength peak, its layer, floor, reach")
    p.add_argument("--strength", type=float, default=1.0, help="without --kernel: one strength for --layers")
    p.add_argument("--layers", help="without --kernel: the layers to edit, e.g. 8-20 (default all)")
    p.add_argument("--direction-layer", type=float, help="one global direction, taken after this (fractional) "
                   "layer; default: each layer's own direction")
    p.add_argument("--per-layer", action="store_true", help="--search: each layer's own direction")
    p.add_argument("--search", type=int, default=0, help="optimize the kernel and direction over N trials")
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--max-kl", type=float, default=0.0, help="reject edits above this KL")
    p.add_argument("--marker", help="regex over greedy replies to the concept set: the behaviour to reduce")
    p.add_argument("--analyze", action="store_true", help="only report where the concept lives; edit nothing")
    p.add_argument("--limit", type=int, default=0, help="at most N prompts per set")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    import torch
    run = Run(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"basis": run.basis, "separation": run.sep0, "method": a.method, "k": a.k,
                "biproject": not a.no_biproject}, out / "directions.pt")
    report = {"model": a.model, "n_layers": run.n_layers, "method": a.method, "k": a.k,
              "biproject": not a.no_biproject, "norm_preserving": not a.plain, "components": a.components,
              "separation_by_layer": [round(float(x), 4) for x in run.sep0], "behaviour_rate_before": run.marker0}
    if a.analyze:
        (out / "edit.json").write_text(json.dumps(report, indent=1))
        print(f"wrote {out / 'directions.pt'} and edit.json (analysis only)")
        return 0
    if a.search:
        trials = run.search()
        best = min(trials, key=lambda r: r["objective"])
        report["trials"] = trials
    else:
        if a.kernel:
            mx, pos, mn, dist = (float(x) for x in a.kernel.split(","))
        else:
            sel = set(range(run.n_layers))
            if a.layers:
                sel = set()
                for part in a.layers.split(","):
                    lo, _, hi = part.partition("-")
                    sel.update(range(int(lo), int(hi or lo) + 1))
            # a flat kernel over the chosen layers
            lo_l, hi_l = min(sel), max(sel)
            mx, mn, pos, dist = a.strength, a.strength, (lo_l + hi_l) / 2, (hi_l - lo_l) / 2 or 0.0
        best = run.evaluate({"max_w": mx, "pos": pos, "min_w": mn, "dist": dist,
                             "direction_layer": a.direction_layer})
    # leave the model in the chosen state and save it
    w = kernel(run.n_layers, best["params"]["max_w"], best["params"]["pos"], best["params"]["min_w"],
               best["params"]["dist"])
    run.ed.apply(run.basis_fn(best["params"].get("direction_layer")), w, not a.plain)
    run.model.save_pretrained(out / "model", safe_serialization=True)
    run.tok.save_pretrained(out / "model")
    report["chosen"] = best
    (out / "edit.json").write_text(json.dumps(report, indent=1))
    print(f"\nchosen: KL {best['kl']:.4f}, remaining separation {best['remaining_separation']:.3f}"
          + (f", behaviour {run.marker0:.0%} -> {best['behaviour_rate']:.0%}" if a.marker else "")
          + f"\nsaved {out / 'model'}; to GGUF:\n  python scripts/merge_export.py --base {a.model} "
            f"--adapter {out / 'model'} --out {out / 'gguf'} --gguf Q4_K_M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
