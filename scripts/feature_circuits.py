#!/usr/bin/env python3
"""
Features as computational intermediates: which features a model used to produce an output, and
what happens when you change them.

    # attribution graph: transcoders (features.py train --site mlp_in --target-site mlp_out) in place of
    # the MLPs, for one prompt; nodes are active features, edges their direct linear effects
    python scripts/feature_circuits.py graph --model org/m --transcoders runs/features/tc-l* \\
        --prompt "Fact: the capital of the state containing Dallas is" --out runs/graphs/dallas.json
    # which features of one residual-stream dictionary drive a completion, checked by ablation
    python scripts/feature_circuits.py attribute --model org/m --sae runs/features/qwen-l12 \\
        --prompt "..." --token " Austin" --baseline " Dallas" --ablate 20 --out runs/graphs/attr.json
    # intervene: change features and compare the next-token predictions
    python scripts/feature_circuits.py intervene --model org/m --transcoders runs/features/tc-l* \\
        --prompt "..." --set 14:3120=0 --set 18:877=x2 --out runs/graphs/swap.json

The graph follows Anthropic's circuit tracing method (Ameisen et al. 2025; Lindsey et al. 2025) as
far as per-layer transcoders allow:
  - a local replacement model: each MLP with a transcoder is replaced by it plus an error term (the
    part of the MLP output it misses); an MLP without one is an error term as a whole;
  - attention patterns and normalisation denominators are frozen at their values on the prompt, so a
    feature's pre-activation is a linear function of earlier features, errors and token embeddings;
  - edge weight A[s->t] = a_s x d(pre_t)/d(a_s) with stop-gradients on every nonlinearity; logit nodes
    are the logit minus the mean logit for the tokens that make up 95% of the probability (up to 10);
  - pruning keeps the nodes and edges with the most influence on the logits (B = A + A^2 + ...).
Cross-layer transcoders, which the papers use, are not trained here; per-layer transcoders give
graphs with larger error nodes (the paper's own comparison: replacement score 0.37 against 0.61).

`attribute` is the simpler measure from Scaling Monosemanticity: for a residual-stream dictionary,
a feature's attribution is its activation times the gradient of (logit - baseline logit) along its
decoder vector, and its ablation effect is the change when it is clamped to zero at that position.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features as FX  # noqa: E402
import sae as S  # noqa: E402


# ====================================================================== freezing nonlinearities

class Frozen:
    """While active: attention patterns and norm denominators pass no gradient (values unchanged).
    The model must use eager attention (load with attn_implementation="eager")."""

    def __init__(self, model):
        self.model, self.hooks, self.patched = model, [], []

    def __enter__(self):
        from concept_edit import decoder_layers
        mods = set()
        for layer in decoder_layers(self.model):
            for name in ("self_attn", "attn", "attention", "self_attention"):
                a = getattr(layer, name, None)
                if a is not None:
                    mods.add(sys.modules[type(a).__module__])
        for mod in mods:
            orig = getattr(mod, "eager_attention_forward", None)
            if orig is None:
                continue

            def frozen(module, query, key, value, attention_mask, *a, _orig=orig, **kw):
                out, w = _orig(module, query, key, value, attention_mask, *a, **kw)
                rep = query.shape[1] // value.shape[1]
                v = value.repeat_interleave(rep, dim=1) if rep > 1 else value
                if w is None or w.shape[-1] != v.shape[-2]:
                    return out, w
                lin = torch.matmul(w.detach().to(v.dtype), v).transpose(1, 2).contiguous()
                if lin.shape != out.shape:
                    return out, w
                return out.detach() + lin - lin.detach(), w
            mod.eager_attention_forward = frozen
            self.patched.append((mod, orig))
        for name, m in self.model.named_modules():
            cls = type(m).__name__
            if not (cls.endswith("RMSNorm") or cls.endswith("LayerNorm")):
                continue
            ln = "RMS" not in cls

            def hook(mod, inp, out, ln=ln):
                x = inp[0]
                if not torch.is_tensor(out) or out.shape != x.shape:
                    return out
                bias = getattr(mod, "bias", None) if ln else None
                xc = x - x.mean(-1, keepdim=True) if ln else x
                base = out - bias if bias is not None else out
                c = torch.where(xc.abs() > 1e-12, base / torch.where(xc.abs() > 1e-12, xc, torch.ones_like(xc)),
                                torch.zeros_like(xc)).detach()
                return xc * c + (bias if bias is not None else 0)
            self.hooks.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        for mod, orig in self.patched:
            mod.eager_attention_forward = orig
        self.hooks, self.patched = [], []


# ====================================================================== local replacement model

def load_transcoders(paths: list[str], device: str) -> dict:
    """{layer: (dictionary, info)} from folders of features.py transcoders (site mlp_in -> mlp_out)."""
    out = {}
    for spec in paths:
        for p in sorted(glob.glob(spec)) or [spec]:
            sae, info = S.load(p, device)
            if info.get("site") != "mlp_in" or info.get("target_site") != "mlp_out":
                raise SystemExit(f"{p}: not an MLP transcoder (site {info.get('site')}, target "
                                 f"{info.get('target_site')}); train one with --site mlp_in --target-site mlp_out")
            out[int(info["layer"])] = (sae, {**info, "path": str(p)})
    if not out:
        raise SystemExit("no transcoders found")
    return out


class Replacement:
    """One forward pass of the local replacement model on a prompt, keeping the leaves attribution
    differentiates with respect to: token embeddings, transcoder feature activations, error terms."""

    def __init__(self, model, tcs: dict):
        from concept_edit import decoder_layers
        self.model, self.tcs = model, tcs
        self.layers = decoder_layers(model)

    def run(self, ids: list[int]):
        model = self.model
        dev = next(model.parameters()).device
        emb = model.get_input_embeddings()(torch.tensor([ids], device=dev)).detach().float().requires_grad_()
        self.emb, self.x_in, self.acts, self.errs, self.a_val = emb, {}, {}, {}, {}
        hooks = []
        for l, layer in enumerate(self.layers):
            mlp = FX.mlp_of(layer)

            def pre(_m, args, l=l):
                self.x_in[l] = args[0]

            def post(_m, _i, out, l=l):
                y = (out[0] if isinstance(out, tuple) else out).detach().float()
                if l in self.tcs:
                    tc = self.tcs[l][0]
                    a = tc.encode(self.x_in[l].detach().float())
                    leaf = a.detach().clone().requires_grad_()
                    rec = tc.decode(leaf) / tc.out_scale
                    err = (y - rec.detach()).requires_grad_()
                    self.acts[l], self.a_val[l] = leaf, a.detach()
                    new = rec + err
                else:
                    err = y.clone().requires_grad_()
                    new = err
                self.errs[l] = err
                new = new.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
                return (new, *out[1:]) if isinstance(out, tuple) else new
            hooks.append(mlp.register_forward_pre_hook(pre))
            hooks.append(mlp.register_forward_hook(post))
        try:
            with Frozen(model):
                logits = model(inputs_embeds=emb.to(next(model.parameters()).dtype), use_cache=False).logits[0]
        finally:
            for h in hooks:
                h.remove()
        self.logits = logits.float()
        return self.logits

    def pre_of(self, layer: int, pos: int, feat: int):
        tc = self.tcs[layer][0]
        x = self.x_in[layer][0, pos].float() * tc.in_scale
        if tc.pre_bias and tc.d_in == tc.d_out:
            x = x - tc.b_dec
        return x @ tc.W_enc[:, feat] + tc.b_enc[feat]

    def leaves(self):
        ls = [self.emb]
        order = []
        for l in sorted(self.errs):
            ls.append(self.errs[l])
            order.append(("err", l))
        for l in sorted(self.acts):
            ls.append(self.acts[l])
            order.append(("feat", l))
        return ls, order


# ====================================================================== graph

def _normalize(A):
    B = A.abs()
    return B / B.sum(1, keepdim=True).clamp(min=1e-10)


def influence(A, logit_w, max_iter: int = 1000):
    """Each node's total influence on the logits: logit_w @ (A + A^2 + ...), A row-normalised |weights|
    (rows are targets, columns sources)."""
    cur = logit_w @ A
    tot = cur.clone()
    for _ in range(max_iter):
        if not bool(cur.any()):
            break
        cur = cur @ A
        tot += cur
    return tot


def _threshold(scores, frac: float):
    s = torch.sort(scores.flatten(), descending=True).values
    if s.sum() <= 0:
        return float("inf")
    c = torch.cumsum(s, 0) / s.sum()
    i = int(torch.searchsorted(c, torch.tensor(frac, dtype=c.dtype)))
    nz = int((s > 0).sum())
    return float(s[min(i, max(nz - 1, 0))])


def prune(A, logit_w, n_feat: int, n_err: int, n_tok: int, node_frac: float = 0.8, edge_frac: float = 0.98):
    """-> (node mask, edge mask, node influence). Tokens and logits are always kept; features and errors
    must keep an outgoing edge, features an incoming one."""
    An = _normalize(A)
    inf = influence(An, logit_w)
    keep = inf >= _threshold(inf, node_frac)
    n_logit = int((logit_w > 0).sum())
    keep[n_feat + n_err:] = True
    P = A.clone()
    P[~keep] = 0
    P[:, ~keep] = 0
    Pn = _normalize(P)
    score = influence(Pn, logit_w) + logit_w
    es = Pn * score[:, None]
    emask = es >= _threshold(es, edge_frac)
    emask &= P != 0
    last = None
    while last is None or not torch.equal(last, keep):
        last = keep.clone()
        emask[~keep] = False
        emask[:, ~keep] = False
        src = n_feat + n_err
        keep[:src] &= emask[:, :src].any(0)
        keep[:n_feat] &= emask[:n_feat].any(1)
    del n_logit
    return keep, emask, inf


def scores(A, logit_w, n_feat: int, n_err: int, n_tok: int) -> tuple[float, float]:
    """Replacement score (share of token-to-logit influence through features rather than errors) and
    completeness score (influence-weighted share of node inputs not from error nodes)."""
    An = _normalize(A)
    inf = influence(An, logit_w)
    e0, e1 = n_feat, n_feat + n_err
    tok, err = inf[e1:e1 + n_tok].sum(), inf[e0:e1].sum()
    repl = float(tok / (tok + err)) if float(tok + err) > 0 else 0.0
    nonerr = 1 - An[:, e0:e1].sum(1)
    w = inf + logit_w
    return repl, float((nonerr * w).sum() / w.sum().clamp(min=1e-12))


def build_graph(model, tok, tcs: dict, prompt: str, chat: bool = False, targets: list[str] | None = None,
                max_logits: int = 10, logit_prob: float = 0.95, max_features: int = 400, node_frac: float = 0.8,
                edge_frac: float = 0.98, labels: dict | None = None, log=print) -> dict:
    t0 = time.time()
    if chat and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tok(prompt)["input_ids"]
    R = Replacement(model, tcs)
    logits = R.run(ids)
    last = logits[-1]
    probs = torch.softmax(last.detach(), -1)
    if targets:
        tgt = []
        for t in targets:
            tid = tok(t, add_special_tokens=False)["input_ids"]
            if tid:
                tgt.append(tid[0])
    else:
        sp, si = probs.sort(descending=True)
        tgt, acc = [], 0.0
        for p, i in zip(sp.tolist(), si.tolist()):
            tgt.append(i)
            acc += p
            if acc >= logit_prob or len(tgt) >= max_logits:
                break
    n_pos = len(ids)
    L = len(R.layers)
    active = []                                  # (layer, pos, feat, act)
    for l in sorted(R.acts):
        a = R.a_val[l][0]
        nz = (a > 0).nonzero()
        for p, f in nz.tolist():
            active.append((l, p, f, float(a[p, f])))
    n_feat_all = len(active)
    log(f"{n_pos} tokens, {n_feat_all} active transcoder features in {len(R.acts)} layers; "
        f"{len(tgt)} output tokens ({', '.join(repr(tok.decode([t])) for t in tgt)})")
    leaves, order = R.leaves()
    err_layers = [l for kind, l in order if kind == "err"]
    err_index = {l: i for i, l in enumerate(err_layers)}
    n_err = len(err_layers) * n_pos
    feat_index = {(l, p, f): i for i, (l, p, f, _) in enumerate(active)}

    def row_for(scalar):
        g = torch.autograd.grad(scalar, leaves, retain_graph=True, allow_unused=True)
        row = torch.zeros(n_feat_all + n_err + n_pos)
        ge = g[0]
        if ge is not None:
            row[n_feat_all + n_err:] = (ge[0] * R.emb[0].detach()).sum(-1).cpu()
        k = 1
        for kind, l in order:
            gl = g[k]
            k += 1
            if gl is None:
                continue
            if kind == "err":
                v = (gl[0] * R.errs[l][0].detach()).sum(-1).cpu()
                row[n_feat_all + err_index[l] * n_pos: n_feat_all + (err_index[l] + 1) * n_pos] = v
            else:
                a = R.a_val[l][0]
                contrib = (gl[0] * a).cpu()
                for p, f in (a > 0).nonzero().tolist():
                    row[feat_index[(l, p, f)]] = contrib[p, f]
        return row

    logit_rows = []
    for t in tgt:
        logit_rows.append(row_for(last[t] - last.mean()))
    lp = torch.tensor([float(probs[t]) for t in tgt])
    rows: dict[int, torch.Tensor] = {}
    budget = min(max_features, n_feat_all)
    visited = torch.zeros(n_feat_all, dtype=torch.bool)
    done = 0
    while done < budget:
        # rank unvisited features by their influence on the logits through the rows computed so far
        n_src = n_feat_all + n_err + n_pos
        known = list(rows)
        M = torch.zeros(len(known) + len(tgt), n_src)
        for i, f in enumerate(known):
            M[i] = rows[f]
        for j, r in enumerate(logit_rows):
            M[len(known) + j] = r
        Mn = _normalize(M)
        w = torch.zeros(len(known) + len(tgt))
        w[len(known):] = lp
        inf_src = torch.zeros(n_src)
        cur = w.clone()
        for _ in range(64):
            nxt = cur @ Mn                                        # influence on sources
            inf_src += nxt
            cur = torch.zeros_like(cur)
            for i, f in enumerate(known):
                cur[i] = nxt[f]
            if not bool(cur.any()):
                break
        cand = torch.argsort(inf_src[:n_feat_all], descending=True)
        batch = [int(c) for c in cand if not visited[c]][:max(8, budget // 10)]
        if not batch:
            break
        for f in batch[: budget - done]:
            l, p, fi, _ = active[f]
            rows[f] = row_for(R.pre_of(l, p, fi))
            visited[f] = True
            done += 1
        FX.progress(done, budget, "feature nodes attributed")
    sel = sorted(rows)
    remap = {f: i for i, f in enumerate(sel)}
    n_feat = len(sel)
    N = n_feat + n_err + n_pos + len(tgt)
    A = torch.zeros(N, N)
    cols = torch.tensor(sel + list(range(n_feat_all, n_feat_all + n_err + n_pos)), dtype=torch.long)
    for f in sel:
        A[remap[f], :n_feat + n_err + n_pos] = rows[f][cols]
    for j, r in enumerate(logit_rows):
        A[n_feat + n_err + n_pos + j, :n_feat + n_err + n_pos] = r[cols]
    logit_w = torch.zeros(N)
    logit_w[n_feat + n_err + n_pos:] = lp
    keep, emask, inf = prune(A, logit_w, n_feat, n_err, n_pos, node_frac, edge_frac)
    repl, compl = scores(A, logit_w, n_feat, n_err, n_pos)
    labels = labels or {}
    nodes = []
    for i in range(N):
        if not keep[i]:
            continue
        if i < n_feat:
            l, p, f, act = active[sel[i]]
            lab = (labels.get(l) or {}).get(f, "")
            nodes.append({"i": i, "kind": "feature", "layer": l, "pos": p, "feature": f, "act": round(act, 4),
                          "influence": round(float(inf[i]), 5), "label": lab,
                          "dict": Path(tcs[l][1]["path"]).name})
        elif i < n_feat + n_err:
            k = i - n_feat
            l, p = err_layers[k // n_pos], k % n_pos
            nodes.append({"i": i, "kind": "error", "layer": l, "pos": p, "influence": round(float(inf[i]), 5)})
        elif i < n_feat + n_err + n_pos:
            p = i - n_feat - n_err
            nodes.append({"i": i, "kind": "token", "layer": -1, "pos": p, "token": tok.decode([ids[p]]),
                          "influence": round(float(inf[i]), 5)})
        else:
            j = i - n_feat - n_err - n_pos
            nodes.append({"i": i, "kind": "logit", "layer": L, "pos": n_pos - 1, "token": tok.decode([tgt[j]]),
                          "token_id": int(tgt[j]), "prob": round(float(lp[j]), 4)})
    edges = [{"s": int(s), "t": int(t), "w": round(float(A[t, s]), 5)} for t, s in emask.nonzero().tolist()]
    log(f"graph: {len(nodes)} of {N} nodes and {len(edges)} edges kept; replacement score {repl:.2f}, "
        f"completeness {compl:.2f}; {time.time() - t0:.0f} s")
    return {"prompt": prompt, "chat": chat, "tokens": [tok.decode([t]) for t in ids], "token_ids": ids, "n_layers": L,
            "targets": [{"token": tok.decode([t]), "id": int(t), "prob": round(float(probs[t]), 4)} for t in tgt],
            "nodes": nodes, "edges": edges, "replacement_score": round(repl, 4), "completeness_score": round(compl, 4),
            "transcoders": {str(l): v[1]["path"] for l, v in tcs.items()}, "active_features": n_feat_all,
            "attributed_features": n_feat, "node_threshold": node_frac, "edge_threshold": edge_frac,
            "seconds": round(time.time() - t0, 1), "method": "local replacement model, per-layer transcoders"}


# ====================================================================== interventions

def parse_set(items: list[str]) -> list[dict]:
    """LAYER:FEATURE=VALUE, LAYER:FEATURE=xM (multiply) or LAYER:FEATURE@POS=..."""
    out = []
    for it in items or []:
        left, _, val = it.partition("=")
        lf, _, pos = left.partition("@")
        l, _, f = lf.partition(":")
        if not (l.strip().lstrip("-").isdigit() and f.strip().isdigit() and val):
            raise SystemExit(f"--set {it!r}: use LAYER:FEATURE=VALUE, =xM to multiply, @POS for one position")
        mul = val.lower().startswith("x")
        out.append({"layer": int(l), "feature": int(f), "pos": int(pos) if pos.strip() else None,
                    "value": float(val[1:] if mul else val), "multiply": mul})
    return out


@torch.no_grad()
def intervene(model, tok, tcs: dict, prompt: str, changes: list[dict], chat: bool = False, top: int = 10,
              watch: list[tuple[int, int, int]] | None = None) -> dict:
    """Run the prompt as is, then with transcoder features changed (the MLP output at a changed layer is
    the transcoder's reconstruction with the change, plus its error term; every other part of the model
    runs normally). -> top next tokens before and after, and watched features before and after."""
    if chat and getattr(tok, "chat_template", None):
        ids = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    else:
        ids = tok(prompt)["input_ids"]
    dev = next(model.parameters()).device
    from concept_edit import decoder_layers
    layers = decoder_layers(model)
    seen: dict = {}

    def run(edit: bool):
        hooks, xin = [], {}
        for l in tcs:
            mlp = FX.mlp_of(layers[l])

            def pre(_m, args, l=l):
                xin[l] = args[0]

            def post(_m, _i, out, l=l):
                tc = tcs[l][0]
                y = out[0] if isinstance(out, tuple) else out
                a = tc.encode(xin[l].float())
                seen[(edit, l)] = a[0].detach().cpu()
                mine = [c for c in changes if c["layer"] == l] if edit else []
                if not mine:
                    return out
                rec = tc.decode(a) / tc.out_scale
                err = y.float() - rec
                a2 = a.clone()
                for c in mine:
                    sl = slice(None) if c["pos"] is None else slice(c["pos"], c["pos"] + 1)
                    a2[0, sl, c["feature"]] = a2[0, sl, c["feature"]] * c["value"] if c["multiply"] else c["value"]
                seen[(edit, l)] = a2[0].detach().cpu()           # watched features show the value after the change
                new = (tc.decode(a2) / tc.out_scale + err).to(y.dtype)
                return (new, *out[1:]) if isinstance(out, tuple) else new
            hooks.append(mlp.register_forward_pre_hook(pre))
            hooks.append(mlp.register_forward_hook(post))
        try:
            return model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float().cpu()
        finally:
            for h in hooks:
                h.remove()
    base, new = run(False), run(True)
    pb, pn = torch.softmax(base, -1), torch.softmax(new, -1)

    def tops(p):
        v, i = p.topk(top)
        return [{"token": tok.decode([int(t)]), "id": int(t), "p": round(float(x), 4)} for x, t in zip(v, i)]
    res = {"prompt": prompt, "changes": changes, "before": tops(pb), "after": tops(pn),
           "kl": round(float((pb * (pb.clamp(min=1e-12).log() - pn.clamp(min=1e-12).log())).sum()), 5)}
    if watch:
        res["watched"] = [{"layer": l, "pos": p, "feature": f,
                           "before": round(float(seen[(False, l)][p, f]), 4) if (False, l) in seen else None,
                           "after": round(float(seen[(True, l)][p, f]), 4) if (True, l) in seen else None}
                          for l, p, f in watch]
    return res


# ====================================================================== output attribution (one dictionary)

def output_attribution(model, tok, sae, info: dict, prompt: str, token: str, baseline: str | None = None,
                       chat: bool = False, top: int = 30, ablate: int = 0, skip_first: bool = True) -> dict:
    """Scaling Monosemanticity's attribution for one residual-stream (or MLP-output) dictionary."""
    site, layer = info.get("target_site") or info["site"], int(info["layer"])
    if chat and getattr(tok, "chat_template", None):
        ids = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    else:
        ids = tok(prompt)["input_ids"]
    t_id = tok(token, add_special_tokens=False)["input_ids"][0]
    b_id = tok(baseline, add_special_tokens=False)["input_ids"][0] if baseline else None
    dev = next(model.parameters()).device
    keep = {}

    def splice(x):
        a = sae.encode(x.detach().float())
        leaf = a.detach().clone().requires_grad_()
        rec = sae.decode(leaf) / sae.out_scale
        err = (x.detach().float() - rec.detach())
        keep["a"], keep["leaf"] = a.detach(), leaf
        return (rec + err).to(x.dtype)
    with FX.Splice(model, site, layer, splice):
        logits = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float()
    metric = logits[t_id] - (logits[b_id] if b_id is not None else logits.mean())
    g, = torch.autograd.grad(metric, keep["leaf"])
    attr = (g[0] * keep["a"][0]).detach().cpu()                              # [pos, n]
    if skip_first and attr.shape[0] > 1:
        attr[0] = 0          # the first position's activations are outliers (attention sink) dictionaries rarely see
    a = keep["a"][0].cpu()
    flat = attr.flatten()
    idx = flat.abs().argsort(descending=True)[:top]
    n = attr.shape[1]
    rows = [{"pos": int(i) // n, "feature": int(i) % n, "token": tok.decode([ids[int(i) // n]]),
             "act": round(float(a.flatten()[i]), 4), "attribution": round(float(flat[i]), 5)} for i in idx]
    base_metric = float(metric)
    if ablate:
        with torch.no_grad():
            for r in rows[:ablate]:
                def zero(x, r=r):
                    a_ = sae.encode(x.float())
                    rec = sae.decode(a_) / sae.out_scale
                    a2 = a_.clone()
                    a2[0, r["pos"], r["feature"]] = 0
                    return (sae.decode(a2) / sae.out_scale + (x.float() - rec)).to(x.dtype)
                with FX.Splice(model, site, layer, zero):
                    lg = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float()
                r["ablation"] = round(float(lg[t_id] - (lg[b_id] if b_id is not None else lg.mean())) - base_metric, 5)
        ab = [r for r in rows if "ablation" in r]
        if len(ab) > 2:
            import numpy as np
            x_ = np.array([r["attribution"] for r in ab])
            y_ = np.array([-r["ablation"] for r in ab])
            c = float(np.corrcoef(x_, y_)[0, 1]) if x_.std() > 0 and y_.std() > 0 else None
        else:
            c = None
    return {"prompt": prompt, "tokens": [tok.decode([t]) for t in ids], "token": token, "baseline": baseline,
            "metric": round(base_metric, 4), "site": site, "layer": layer, "features": rows,
            **({"attribution_ablation_corr": None if c is None else round(c, 3)} if ablate else {})}


# ====================================================================== CLI

def _labels_for(paths: dict) -> dict:
    out = {}
    for l, (_, info) in paths.items():
        f = Path(info["path"]) / "features.json"
        if f.exists():
            try:
                out[l] = {e["id"]: e["label"] for e in json.loads(f.read_text()).get("features", []) if e.get("label")}
            except (ValueError, KeyError):
                pass
        lf = Path(info["path"]) / "labels.json"
        if lf.exists():
            try:
                out.setdefault(l, {}).update({int(k): v for k, v in json.loads(lf.read_text()).items()})
            except ValueError:
                pass
    return out


def load_for_attribution(model_id: str, device: str = "auto", dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), attn_implementation="eager", trust_remote_code=False)
    dev = FX.pick_device(device)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, dev


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("graph", help="attribution graph over transcoder features for one prompt")
    g.add_argument("--model", help="HF id or folder (default: the transcoders' model)")
    g.add_argument("--transcoders", nargs="+", required=True, help="transcoder folders (globs allowed)")
    g.add_argument("--prompt", required=True)
    g.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    g.add_argument("--target", action="append", help="attribute these output tokens (default: top ones)")
    g.add_argument("--max-logits", type=int, default=10)
    g.add_argument("--logit-prob", type=float, default=0.95, help="output tokens up to this cumulative probability")
    g.add_argument("--max-features", type=int, default=400, help="feature nodes to attribute (most influential first)")
    g.add_argument("--node-threshold", type=float, default=0.8)
    g.add_argument("--edge-threshold", type=float, default=0.98)
    g.add_argument("--device", default="auto")
    g.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    g.add_argument("--out", required=True, help="graph JSON")

    at = sub.add_parser("attribute", help="features of one dictionary that drive a completion")
    at.add_argument("--model")
    at.add_argument("--sae", required=True)
    at.add_argument("--prompt", required=True)
    at.add_argument("--chat", action="store_true")
    at.add_argument("--token", required=True, help="the completion of interest, e.g. ' Austin'")
    at.add_argument("--baseline", help="a competing completion (default: the mean logit)")
    at.add_argument("--top", type=int, default=30)
    at.add_argument("--ablate", type=int, default=10, help="also clamp the top N to zero and measure the effect")
    at.add_argument("--first", action="store_true", help="include the first position (left out by default)")
    at.add_argument("--device", default="auto")
    at.add_argument("--out", required=True)

    iv = sub.add_parser("intervene", help="change transcoder features and compare the next-token predictions")
    iv.add_argument("--model")
    iv.add_argument("--transcoders", nargs="+", required=True)
    iv.add_argument("--prompt", required=True)
    iv.add_argument("--chat", action="store_true")
    iv.add_argument("--set", action="append", required=True,
                    help="LAYER:FEATURE=VALUE, =xM to multiply, LAYER:FEATURE@POS=... for one position")
    iv.add_argument("--top", type=int, default=10)
    iv.add_argument("--device", default="auto")
    iv.add_argument("--out", help="result JSON")

    a = p.parse_args(argv)
    if a.cmd == "attribute":
        sae, info = S.load(a.sae)
        model, tok, dev = load_for_attribution(a.model or info["model"], a.device)
        sae.to(dev)
        r = output_attribution(model, tok, sae, info, a.prompt, a.token, a.baseline, a.chat, a.top, a.ablate,
                               not a.first)
        for row in r["features"][:15]:
            print(f"  pos {row['pos']:3d} {row['token']!r:>12}  feature {row['feature']:6d}  act {row['act']:8.3f}  "
                  f"attribution {row['attribution']:+.4f}" + (f"  ablation {row['ablation']:+.4f}" if "ablation" in row else ""))
        if r.get("attribution_ablation_corr") is not None:
            print(f"correlation of attribution with ablation effect: {r['attribution_ablation_corr']:.2f}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
        print(f"wrote {a.out}")
        return 0
    dev_pref = a.device
    first = sorted(glob.glob(a.transcoders[0])) or [a.transcoders[0]]
    model_id = a.model or S.load(first[0])[1].get("model")
    if not model_id:
        raise SystemExit("which model? --model")
    model, tok, dev = load_for_attribution(model_id, dev_pref, getattr(a, "dtype", "float32"))
    tcs = load_transcoders(a.transcoders, dev)
    if a.cmd == "graph":
        r = build_graph(model, tok, tcs, a.prompt, a.chat, a.target, a.max_logits, a.logit_prob, a.max_features,
                        a.node_threshold, a.edge_threshold, _labels_for(tcs), log=lambda s: print(s, flush=True))
        r["model"] = model_id
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, separators=(",", ":")))
        print(f"wrote {a.out}")
        return 0
    r = intervene(model, tok, tcs, a.prompt, parse_set(a.set), a.chat, a.top)
    print("before: " + ", ".join(f"{t['token']!r} {t['p']:.3f}" for t in r["before"][:5]))
    print("after:  " + ", ".join(f"{t['token']!r} {t['p']:.3f}" for t in r["after"][:5]))
    print(f"KL(before || after) = {r['kl']:.4f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
