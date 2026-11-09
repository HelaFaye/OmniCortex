#!/usr/bin/env python3
"""
Feature dictionaries: what a model represents, as sparse features instead of neurons.

A neuron takes part in many unrelated concepts. Dictionary learning finds directions in a layer's
activations that each stand for one thing, and rewrites every activation as a few of them
(docs/FEATURES.md; the methods and their sources are in docs/CREDITS.md).

    # 1. record activations of a model over some text
    python scripts/features.py collect --model Qwen/Qwen2.5-0.5B-Instruct --corpus NeelNanda/pile-10k \\
        --layers 12 --site resid --max-tokens 400000 --out runs/features/acts-qwen
    # 2. learn a dictionary (sparse autoencoder) on one layer
    python scripts/features.py train --acts runs/features/acts-qwen --site resid --layer 12 \\
        --arch topk --k 32 --expansion 8 --out runs/features/qwen-l12
    #    or a transcoder: read the MLP's input, reconstruct its output
    python scripts/features.py train --acts ... --site mlp_in --target-site mlp_out --layer 12 --out ...
    #    or bring one trained elsewhere (SAELens, Gemma Scope, safetensors with W_enc/W_dec)
    python scripts/features.py import --path ~/saes/layer_12 --model org/m --site resid --layer 12 --out runs/features/x
    # 3. what each feature responds to: top examples, logit effects, neighbours, a map; optional labels
    python scripts/features.py interpret --sae runs/features/qwen-l12 --labeler http://127.0.0.1:1234/v1@my-model --label 200
    # 4. which neurons carry each feature (and how many of them are H-Neurons)
    python scripts/features.py neurons --sae runs/features/qwen-l12 --h-neurons runs/classifier/h_neurons.json

Attribution graphs over these features, interventions on them, and emotion vectors live in
feature_circuits.py and emotions.py; steering with them in decode_lab.py.

Sites (where in a decoder layer the activations are taken):
  resid     the residual stream after the layer
  mlp_in    the MLP's input (after the layer's second norm)
  mlp_out   the MLP's output, before it is added to the residual stream
  neurons   the MLP's hidden neurons (the input of its down projection): the neurons H-Neurons and CETT
            are measured on, so a dictionary here maps features onto those neurons directly
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SITES = ("resid", "mlp_in", "mlp_out", "neurons")
DOWN_NAMES = ("down_proj", "c_proj", "fc2", "w2", "dense_4h_to_h", "fc_out")


def progress(done: int, total: int, label: str) -> None:
    # the Jobs page turns these lines into a progress bar
    print(f"PROGRESS {done}/{total} {label}", flush=True)


# ====================================================================== model access

def pick_device(pref: str = "auto") -> str:
    import torch
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, dtype: str = "auto", device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    dev = pick_device(device)
    if dtype == "auto" and dev == "cpu":
        dtype = "float32"                    # bfloat16 matmuls are slow on most CPUs
    kw = {"dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.to(dev).eval()
    return model, tok, dev


def mlp_of(layer):
    for name in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError("cannot find this layer's MLP")


def down_of(mlp):
    for name in DOWN_NAMES:
        m = getattr(mlp, name, None)
        if m is not None:
            return m
    raise ValueError("cannot find this MLP's down projection (for the neurons site)")


def site_module(model, site: str, layer: int):
    """-> (module, "out" | "in"): the activation is that module's output, or its first input."""
    from concept_edit import decoder_layers
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer {layer}: the model has {len(layers)} layers (0..{len(layers) - 1})")
    if site == "resid":
        return layers[layer], "out"
    if site == "mlp_in":
        return mlp_of(layers[layer]), "in"
    if site == "mlp_out":
        return mlp_of(layers[layer]), "out"
    if site == "neurons":
        return down_of(mlp_of(layers[layer])), "in"
    raise ValueError(f"site: one of {', '.join(SITES)}")


class Capture:
    """Record activations at (site, layer) points during forward passes: .grab[(site, layer)]."""

    def __init__(self, model, points):
        self.model, self.points, self.grab, self.hooks = model, list(points), {}, []

    def __enter__(self):
        for site, layer in self.points:
            mod, kind = site_module(self.model, site, layer)
            key = (site, layer)
            if kind == "out":
                def h(_m, _i, out, key=key):
                    self.grab[key] = (out[0] if isinstance(out, tuple) else out).detach()
                self.hooks.append(mod.register_forward_hook(h))
            else:
                def h(_m, args, key=key):
                    self.grab[key] = args[0].detach()
                self.hooks.append(mod.register_forward_pre_hook(h))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class Splice:
    """Replace the activation at (site, layer) with fn(activation) during forward passes."""

    def __init__(self, model, site: str, layer: int, fn):
        self.mod, self.kind = site_module(model, site, layer)
        self.fn, self.h = fn, None

    def __enter__(self):
        if self.kind == "out":
            def h(_m, _i, out):
                if isinstance(out, tuple):
                    return (self.fn(out[0]), *out[1:])
                return self.fn(out)
            self.h = self.mod.register_forward_hook(h)
        else:
            def h(_m, args):
                return (self.fn(args[0]), *args[1:])
            self.h = self.mod.register_forward_pre_hook(h)
        return self

    def __exit__(self, *exc):
        if self.h is not None:
            self.h.remove()
            self.h = None


# ====================================================================== corpus

def _row_text(r: dict, field: str | None, tok=None) -> str | None:
    if field:
        v = r.get(field)
        return v if isinstance(v, str) else None
    if isinstance(r.get("text"), str):
        return r["text"]
    if isinstance(r.get("messages"), list) and tok is not None and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(r["messages"], tokenize=False)
        except Exception:
            pass
    if isinstance(r.get("messages"), list):
        return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in r["messages"] if isinstance(m, dict))
    for q, ans in (("prompt", "reply"), ("prompt", "response"), ("question", "answer"), ("prompt", "completion")):
        if isinstance(r.get(q), str):
            return r[q] + ("\n" + str(r[ans]) if r.get(ans) else "")
    for v in r.values():
        if isinstance(v, str) and len(v) > 40:
            return v
    return None


def read_corpus(spec: str, field: str | None = None, split: str = "train", config: str | None = None,
                limit: int = 0, tok=None) -> list[str]:
    """Documents from a folder of .txt/.md, a .txt (blank-line paragraphs), a .jsonl/.json, or a
    Hugging Face dataset id."""
    p = Path(spec).expanduser()
    docs: list[str] = []
    if p.is_dir():
        for f in sorted(list(p.rglob("*.txt")) + list(p.rglob("*.md"))):
            docs.append(f.read_text(errors="replace"))
    elif p.exists() and p.suffix in (".txt", ".md"):
        docs = [d.strip() for d in p.read_text(errors="replace").split("\n\n") if d.strip()]
    elif p.exists() and p.suffix in (".jsonl", ".json"):
        rows = ([json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.suffix == ".jsonl"
                else json.loads(p.read_text()))
        if isinstance(rows, dict):
            rows = rows.get("rows") or rows.get("data") or list(rows.values())
        docs = [t for t in (_row_text(r, field, tok) if isinstance(r, dict) else str(r) for r in rows) if t]
    else:
        import hf_rows
        rows = hf_rows.load(spec, split, config)
        docs = [t for t in (_row_text(r, field, tok) for r in rows) if t]
    docs = [d for d in docs if d and d.strip()]
    return docs[:limit] if limit else docs


# ====================================================================== stored activations

class Store:
    """Activations recorded by `collect` for one (site, layer), read without loading them all."""

    def __init__(self, root: str | Path, site: str, layer: int):
        self.root = Path(root)
        d = self.root / f"{site}.{layer}"
        self.shards = [np.load(f, mmap_mode="r") for f in sorted(d.glob("shard_*.npy"))]
        if not self.shards:
            have = sorted(x.name for x in self.root.iterdir() if x.is_dir()) if self.root.is_dir() else []
            raise SystemExit(f"{d}: no activations recorded for {site} layer {layer}"
                             + (f"; this folder has {', '.join(have)}" if have else ""))
        self.sizes = [s.shape[0] for s in self.shards]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.n, self.d = int(self.offsets[-1]), int(self.shards[0].shape[1])
        self.doc = np.load(self.root / "doc.npy", mmap_mode="r")
        self.site, self.layer = site, layer

    def rows(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), self.d), dtype=np.float32)
        sh = np.searchsorted(self.offsets, idx, side="right") - 1
        for s in np.unique(sh):
            m = sh == s
            out[m] = self.shards[s][idx[m] - self.offsets[s]]
        return out

    def split(self, eval_every: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Row indices for training and for evaluation; whole documents go to evaluation."""
        ev = (np.asarray(self.doc[: self.n]) % eval_every) == 0
        return np.nonzero(~ev)[0], np.nonzero(ev)[0]

    def batches(self, rows: np.ndarray, bs: int, shuffle: bool = True, seed: int = 0, paired: "Store | None" = None,
                chunk: int = 262144):
        """Endless (x, y) float32 torch batches over the given rows; y from a paired store (transcoders)."""
        import torch
        rng = np.random.default_rng(seed)
        while True:
            order = rng.permutation(rows) if shuffle else rows
            for c in range(0, len(order), chunk):
                part = np.sort(order[c:c + chunk])          # sorted reads are much faster on memmaps
                X = self.rows(part)
                Y = paired.rows(part) if paired is not None else None
                perm = rng.permutation(len(part)) if shuffle else np.arange(len(part))
                for b in range(0, len(perm) - bs + 1 if shuffle else len(perm), bs):
                    sel = perm[b:b + bs]
                    yield torch.from_numpy(X[sel]), (torch.from_numpy(Y[sel]) if Y is not None else None)
            if not shuffle:
                return


def load_meta(acts: str | Path) -> dict:
    p = Path(acts) / "meta.json"
    if not p.exists():
        raise SystemExit(f"{acts}: no meta.json; record activations with `features.py collect` first")
    return json.loads(p.read_text())


def window_tokens(acts: str | Path):
    root = Path(acts)
    return np.load(root / "win_tokens.npy", mmap_mode="r"), np.load(root / "win_ptr.npy", mmap_mode="r")


# ====================================================================== collect

def cmd_collect(a) -> int:
    import torch
    model, tok, dev = load_model(a.model, a.dtype, a.device)
    from concept_edit import decoder_layers
    n_layers = len(decoder_layers(model))
    layers = [int(x) for x in a.layers] if a.layers else [n_layers // 2]
    points = [(s, l) for s in a.site for l in layers]
    docs = read_corpus(a.corpus, a.field, a.split, a.config, a.limit_docs, tok)
    if not docs:
        raise SystemExit(f"{a.corpus}: no text found")
    if a.shuffle:
        random.Random(a.seed).shuffle(docs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, l in points:
        (out / f"{s}.{l}").mkdir(exist_ok=True)
        for f in (out / f"{s}.{l}").glob("shard_*.npy"):
            f.unlink()
    bos = tok.bos_token_id if a.bos else None
    win = a.ctx - (1 if bos is not None else 0)

    def windows():
        for d in docs:
            ids = tok(d, add_special_tokens=False)["input_ids"]
            for s in range(0, max(1, len(ids)), win):
                w = ids[s:s + win]
                if len(w) < max(4, a.skip + 2):
                    break
                yield ([bos] if bos is not None else []) + w
                if not a.chunk_long:
                    break

    bufs = {p: [] for p in points}
    shard_no, buffered = 0, 0
    doc_ids, pos_ids, tok_ids, win_tokens, win_ptr = [], [], [], [], [0]
    total, n_win, t0 = 0, 0, time.time()

    def flush():
        nonlocal shard_no, buffered
        if not buffered:
            return
        for (s, l), chunks in bufs.items():
            np.save(out / f"{s}.{l}" / f"shard_{shard_no:05d}.npy", np.concatenate(chunks).astype(np.float16))
            bufs[(s, l)] = []
        shard_no += 1
        buffered = 0

    gen = windows()
    done = False
    with torch.no_grad(), Capture(model, points) as cap:
        while not done:
            batch = []
            for w in gen:
                batch.append(w)
                if len(batch) == a.batch:
                    break
            if not batch:
                break
            L = max(len(w) for w in batch)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, w in enumerate(batch):
                ids[i, :len(w)] = torch.tensor(w)
                mask[i, :len(w)] = 1
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
            for i, w in enumerate(batch):
                take = list(range(a.skip, len(w)))
                if total + len(take) > a.max_tokens:
                    take = take[: a.max_tokens - total]
                    done = True
                if not take:
                    break
                for p in points:
                    bufs[p].append(cap.grab[p][i, take].float().cpu().numpy())
                doc_ids += [n_win] * len(take)
                pos_ids += take
                tok_ids += [w[t] for t in take]
                win_tokens += w
                win_ptr.append(len(win_tokens))
                n_win += 1
                total += len(take)
                buffered += len(take)
                if buffered >= a.shard_rows:
                    flush()
                if done:
                    break
            progress(min(total, a.max_tokens), a.max_tokens, f"tokens recorded from {n_win} windows")
    flush()
    if not total:
        raise SystemExit("no tokens recorded (texts too short?)")
    np.save(out / "doc.npy", np.asarray(doc_ids, dtype=np.int32))
    np.save(out / "pos.npy", np.asarray(pos_ids, dtype=np.int32))
    np.save(out / "tokens.npy", np.asarray(tok_ids, dtype=np.int32))
    np.save(out / "win_tokens.npy", np.asarray(win_tokens, dtype=np.int32))
    np.save(out / "win_ptr.npy", np.asarray(win_ptr, dtype=np.int64))
    dims = {f"{s}.{l}": int(np.load(next((out / f"{s}.{l}").glob("shard_*.npy")), mmap_mode="r").shape[1])
            for s, l in points}
    meta = {"model": a.model, "points": [f"{s}.{l}" for s, l in points], "dims": dims, "n_layers": n_layers,
            "tokens": total, "windows": n_win, "ctx": a.ctx, "skip": a.skip, "corpus": a.corpus,
            "field": a.field, "split": a.split, "seconds": round(time.time() - t0, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "format": "omnicortex-acts/1"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*.npy")) / 2 ** 30
    print(f"recorded {total} tokens from {n_win} windows at {', '.join(meta['points'])} "
          f"in {meta['seconds']} s ({size:.2f} GiB) -> {out}")
    return 0


# ====================================================================== train

def splice_loss(model, tok, dev, sae, site: str, layer: int, target_site: str | None, seqs: list[list[int]],
                max_seqs: int = 64) -> dict:
    """Next-token loss of the model as is, with the dictionary's reconstruction in place of the
    activation, and (for MLP sites) with that activation zeroed. Fraction of loss recovered =
    (zeroed - spliced) / (zeroed - clean)."""
    import torch
    import torch.nn.functional as F

    def ce():
        tot = n = 0.0
        with torch.no_grad():
            for s in seqs[:max_seqs]:
                ids = torch.tensor([s], device=dev)
                logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                tot += F.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
                n += len(s) - 1
        return tot / max(n, 1)

    clean = ce()
    out_site = target_site or site
    if target_site:                               # transcoder: read site's input, replace target's output
        grab = {}
        mod_in, kind_in = site_module(model, site, layer)

        def keep(_m, args):
            grab["x"] = args[0]
        h = mod_in.register_forward_pre_hook(keep) if kind_in == "in" else \
            mod_in.register_forward_hook(lambda _m, _i, o: grab.__setitem__("x", o[0] if isinstance(o, tuple) else o))
        try:
            with Splice(model, out_site, layer, lambda y: sae.reconstruct(grab["x"].float())[0].to(y.dtype)):
                spliced = ce()
        finally:
            h.remove()
    else:
        with Splice(model, site, layer, lambda x: sae.reconstruct(x.float())[0].to(x.dtype)):
            spliced = ce()
    res = {"clean": clean, "spliced": spliced, "delta": spliced - clean}
    if out_site in ("mlp_out", "neurons"):
        with Splice(model, out_site, layer, lambda x: torch.zeros_like(x)):
            zero = ce()
        res["zeroed"] = zero
        res["recovered"] = (zero - spliced) / (zero - clean) if zero > clean else None
    return res


def cmd_train(a) -> int:
    import sae as S
    import torch
    meta = load_meta(a.acts)
    store = Store(a.acts, a.site, a.layer)
    target = Store(a.acts, a.target_site, a.layer) if a.target_site else None
    tr, ev = store.split()
    if not len(tr):
        raise SystemExit("too few documents to hold some out; record more text")
    d_in, d_out = store.d, (target.d if target else store.d)
    n = a.features or int(a.expansion * d_in)
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    sae = S.SAE(d_in, n, a.arch, a.k, d_out, seed=a.seed)
    sample = np.sort(np.random.default_rng(a.seed).choice(tr, min(len(tr), 16384), replace=False))
    sae.in_scale.fill_(S.norm_scale([torch.from_numpy(store.rows(sample))], d_in))
    sae.out_scale.fill_(S.norm_scale([torch.from_numpy(target.rows(sample))], d_out) if target else float(sae.in_scale))
    steps = a.steps or max(100, math.ceil(a.epochs * len(tr) / a.batch))
    kind = "transcoder" if target else "sparse autoencoder"
    print(f"training a {a.arch} {kind}: {d_in} -> {n} features"
          + (f" -> {d_out}" if target else "") + f", {steps} steps of {a.batch} on {dev} "
          f"({len(tr)} training rows, {len(ev)} held out)", flush=True)

    def log(r):
        progress(r["step"], steps, f"fvu {r['fvu']:.3f} · L0 {r['l0']:.1f} · dead {100 * r['dead']:.1f}% "
                                   f"· {r['seconds']:.0f} s")
    hist = S.train(sae, store.batches(tr, a.batch, seed=a.seed, paired=target), steps, lr=a.lr,
                   sparsity=a.sparsity, target_l0=a.target_l0 if a.arch != "topk" else None,
                   dead_after=a.dead_after, log_every=max(1, min(50, steps // 40)), on_log=log, device=dev)
    res = S.evaluate(sae, store.batches(ev, 4096, shuffle=False, paired=target), 10 ** 9, dev) if len(ev) else {}
    print(f"held out: fvu {res.get('fvu', float('nan')):.4f} · L0 {res.get('l0', float('nan')):.1f} · "
          f"never active {100 * res.get('never_fired', 0):.1f}%", flush=True)
    info = {"model": meta["model"], "site": a.site, "layer": a.layer, "target_site": a.target_site,
            "acts": str(Path(a.acts).resolve()), "train": {"steps": steps, "batch": a.batch, "lr": a.lr,
                                                           "history": hist}, "eval": res,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "name": Path(a.out).name}
    if a.check_model and len(ev):
        model, tok, mdev = load_model(meta["model"], a.dtype, a.device)
        wt, wp = window_tokens(a.acts)
        docs = sorted(set(int(store.doc[i]) for i in ev[:: max(1, len(ev) // 512)]))[:a.check_seqs]
        seqs = [list(map(int, wt[wp[d]:wp[d + 1]])) for d in docs]
        sae.to(mdev)
        info["splice"] = splice_loss(model, tok, mdev, sae, a.site, a.layer, a.target_site, seqs, a.check_seqs)
        sp = info["splice"]
        print(f"next-token loss: clean {sp['clean']:.3f}, with the dictionary {sp['spliced']:.3f} "
              f"(+{sp['delta']:.3f})" + (f", zeroed {sp['zeroed']:.3f}, recovered {100 * sp['recovered']:.1f}%"
                                         if sp.get("recovered") is not None else ""), flush=True)
    S.save(sae.cpu(), a.out, info)
    print(f"wrote {a.out}/sae.pt and sae.json")
    return 0


def cmd_import(a) -> int:
    import sae as S
    if a.release:
        path, cat = S.from_release(a.release, a.sae_id or "")
        a.path, a.model = str(path), a.model or cat.get("model")
        print(f"downloaded {a.release} / {a.sae_id} from {cat['repo_id']}")
    elif not a.path:
        raise SystemExit("--path, or --release and --sae-id (a SAELens pretrained dictionary)")
    sae, meta = S.load(a.path)
    site = a.site or meta.get("site")
    layer = a.layer if a.layer is not None else meta.get("layer")
    if site not in SITES or layer is None:
        raise SystemExit("say where this dictionary reads from: --site resid|mlp_in|mlp_out|neurons and --layer N")
    info = {"model": a.model or meta.get("model_name") or meta.get("model"), "site": site, "layer": int(layer),
            "target_site": a.target_site, "imported_from": str(a.path), "source_cfg": meta,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "name": Path(a.out).name}
    if a.acts:
        info["acts"] = str(Path(a.acts).resolve())
    S.save(sae, a.out, info)
    print(f"imported a {sae.arch} dictionary of {sae.n} features ({sae.d_in} -> {sae.d_out}) for "
          f"{site} layer {layer} -> {a.out}")
    return 0


# ====================================================================== interpret

def _chat(spec: str, messages: list, max_tokens: int = 300, timeout: float = 300) -> str:
    """One reply from an OpenAI-compatible endpoint given as URL@model."""
    url, model = spec, None
    if "@" in spec.split("//", 1)[-1]:
        url, model = spec.rsplit("@", 1)
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if model:
        body["model"] = model
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read())["choices"][0]["message"].get("content") or ""
    if "</think>" in txt:
        txt = txt.split("</think>", 1)[1]
    return txt.strip()


def logit_effects(model, sae, meta: dict, top: int = 10, device: str = "cpu"):
    """Per feature, the tokens its decoder direction most raises and lowers through the final norm's
    scale and the unembedding (ignores the norm's centring and the layers in between)."""
    import torch
    from logit_lens import final_norm
    site = meta.get("target_site") or meta["site"]
    if site == "mlp_in":
        return None
    W = sae.W_dec.detach().float().to(device)                           # [n, d_out]
    if site == "neurons":
        down = down_of(mlp_of(__import__("concept_edit").decoder_layers(model)[meta["layer"]]))
        W = W @ down.weight.detach().float().to(device).t()                 # neuron space -> residual
    norm = final_norm(model)
    g = getattr(norm, "weight", None)
    if g is not None:
        W = W * g.detach().float().to(device)
    U = model.get_output_embeddings().weight.detach().float().to(device)  # [V, d]
    up, down_ = [], []
    for s in range(0, W.shape[0], 1024):
        L = W[s:s + 1024] @ U.t()
        up.append(L.topk(top, dim=1))
        down_.append((-L).topk(top, dim=1))
    vu = torch.cat([u.values for u in up]).cpu()
    iu = torch.cat([u.indices for u in up]).cpu()
    vd = torch.cat([-d.values for d in down_]).cpu()
    idd = torch.cat([d.indices for d in down_]).cpu()
    return vu, iu, vd, idd


def neighbours(sae, top: int = 8, device: str = "cpu"):
    import torch
    D = sae.directions().to(device)
    vals, idx = [], []
    for s in range(0, D.shape[0], 2048):
        c = D[s:s + 2048] @ D.t()
        c[torch.arange(c.shape[0]), torch.arange(s, s + c.shape[0])] = -2
        v, i = c.topk(top, dim=1)
        vals.append(v.cpu())
        idx.append(i.cpu())
    return torch.cat(vals), torch.cat(idx)


def pca2(X):
    """2-D coordinates of the rows of X by principal components (dependency-free stand-in for UMAP)."""
    import torch
    X = X - X.mean(0)
    U, S_, _ = torch.pca_lowrank(X, q=2, center=False)
    return (U[:, :2] * S_[:2]).cpu().numpy()


def _context(wt, wp, doc: int, pos: int, left: int, right: int) -> tuple[list[int], int]:
    a_, b_ = int(wp[doc]), int(wp[doc + 1])
    s, e = max(a_, a_ + pos - left), min(b_, a_ + pos + right + 1)
    return list(map(int, wt[s:e])), a_ + pos - s


def label_prompt(examples: list[tuple[list[str], list[float]]]) -> str:
    lines = []
    for i, (pieces, acts) in enumerate(examples, 1):
        mx = max(acts) or 1.0
        out = []
        for p, v in zip(pieces, acts):
            out.append(f"[[{p}]]" if v > 0.25 * mx else p)
        lines.append(f"{i}. " + "".join(out).replace("\n", "⏎"))
    return ("These are excerpts of text on which one internal feature of a language model is active. The tokens "
            "where it is most active are marked [[like this]].\n\n" + "\n".join(lines) +
            "\n\nWhat does this feature respond to? Answer with a short label of at most eight words, nothing else.")


RUBRIC = ("0 - the description has nothing to do with the excerpt.\n"
          "1 - it relates to the excerpt, but not to the marked text, or only vaguely.\n"
          "2 - it relates loosely to the marked text, or to text near it.\n"
          "3 - it cleanly identifies the marked text.")


def specificity_prompt(label: str, texts: list[str]) -> str:
    # the rubric of Scaling Monosemanticity's specificity analysis, in our words
    return (f"A feature of a language model is described as: \"{label}\". In each excerpt below, the token where the "
            "feature is measured is marked [[like this]]. Rate how well the description fits each excerpt:\n\n"
            + RUBRIC + "\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1)) +
            "\n\nReply with one line per excerpt, as 'number: score', and nothing else.")


def score_prompt(label: str, texts: list[str]) -> str:
    return (f"A feature of a language model is described as: \"{label}\".\n\nWhich of these excerpts contain what it "
            "describes?\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1)) +
            "\n\nReply with the numbers of the matching excerpts, separated by commas, or 'none'.")


def cmd_interpret(a) -> int:
    import sae as S
    import torch
    d = Path(a.sae)
    sae, info = S.load(d)
    acts = a.acts or info.get("acts")
    if not acts:
        raise SystemExit("which activations? --acts (a folder from `features.py collect`)")
    meta = load_meta(acts)
    store = Store(acts, info["site"], info["layer"])
    dev = pick_device(a.device)
    sae.to(dev)
    n, K, R = sae.n, a.top, a.random
    rows = np.arange(store.n) if not a.max_rows else np.arange(min(store.n, a.max_rows))
    freq = torch.zeros(n, dtype=torch.float64)
    ssum = torch.zeros(n, dtype=torch.float64)
    mx = torch.zeros(n)
    top_v = torch.full((K, n), -1.0)
    top_i = torch.full((K, n), -1, dtype=torch.long)
    rnd_p = torch.full((R, n), -1.0)
    rnd_i = torch.full((R, n), -1, dtype=torch.long)
    bins = torch.logspace(-3, 3, 31)
    hist = torch.zeros(n, len(bins) + 1, dtype=torch.int32)
    code_ptr, code_idx, code_val = [0], [], []
    keep = a.keep_codes
    gen = torch.Generator().manual_seed(a.seed)
    bs = a.batch
    t0 = time.time()
    for b in range(0, len(rows), bs):
        r = rows[b:b + bs]
        x = torch.from_numpy(store.rows(r)).to(dev)
        with torch.no_grad():
            f = sae.encode(x).float().cpu()
        on = f > 0
        freq += on.sum(0)
        ssum += f.sum(0)
        mx = torch.maximum(mx, f.max(0).values)
        rid = torch.from_numpy(r.astype(np.int64))
        k = min(K, f.shape[0])
        v, i = f.topk(k, dim=0)
        cv, ci = torch.cat([top_v, v]), torch.cat([top_i, rid[i]])
        top_v, sel = cv.topk(K, dim=0)
        top_i = ci.gather(0, sel)
        pr = torch.where(on, torch.rand(f.shape, generator=gen), torch.full_like(f, -1.0))
        k = min(R, f.shape[0])
        v, i = pr.topk(k, dim=0)
        cv, ci = torch.cat([rnd_p, v]), torch.cat([rnd_i, rid[i]])
        rnd_p, sel = cv.topk(R, dim=0)
        rnd_i = ci.gather(0, sel)
        nz = on.nonzero()
        if len(nz):
            hb = torch.bucketize(f[on], bins)
            hist.index_put_((nz[:, 1], hb), torch.ones(len(hb), dtype=torch.int32), accumulate=True)
        if keep:
            v, i = f.topk(min(keep, n), dim=1)
            for row_v, row_i in zip(v, i):
                m = row_v > 0
                code_idx.append(row_i[m].numpy().astype(np.int32))
                code_val.append(row_v[m].numpy().astype(np.float16))
                code_ptr.append(code_ptr[-1] + int(m.sum()))
        progress(min(b + bs, len(rows)), len(rows), f"tokens encoded · {time.time() - t0:.0f} s")
    if keep:
        np.save(d / "codes_ptr.npy", np.asarray(code_ptr, dtype=np.int64))
        np.save(d / "codes_idx.npy", np.concatenate(code_idx) if code_idx else np.zeros(0, np.int32))
        np.save(d / "codes_val.npy", np.concatenate(code_val) if code_val else np.zeros(0, np.float16))
    N = len(rows)
    wt, wp = window_tokens(acts)
    doc = store.doc
    pos = np.load(Path(acts) / "pos.npy", mmap_mode="r")

    def feature_acts(row_ids: list[int], feat_ids: list[int]) -> np.ndarray:
        """Activation of each listed feature on each listed row."""
        if not row_ids:
            return np.zeros((0, len(feat_ids)), np.float32)
        x = torch.from_numpy(store.rows(np.asarray(row_ids))).to(dev)
        with torch.no_grad():
            f = sae.encode(x)
        return f[:, torch.as_tensor(feat_ids, device=dev)].float().cpu().numpy()

    # row index of each (doc, pos) for windows: rows of a window are contiguous, in position order
    uniq, first, count = np.unique(np.asarray(doc[: store.n]), return_index=True, return_counts=True)
    first_row = dict(zip(uniq.tolist(), first.tolist()))
    rows_in = dict(zip(uniq.tolist(), count.tolist()))
    skip = int(meta.get("skip", 1))
    cp = ci = cv = None
    if keep:
        cp = np.asarray(code_ptr, dtype=np.int64)
        ci = np.concatenate(code_idx) if code_idx else np.zeros(0, np.int32)
        cv = np.concatenate(code_val).astype(np.float32) if code_val else np.zeros(0, np.float32)

    def acts_of(row_ids: list[int], feat: int) -> np.ndarray:
        """From the stored codes when they cover the rows (exact up to --keep-codes per token)."""
        if cp is None or not row_ids or max(row_ids) >= len(cp) - 1:
            return feature_acts(row_ids, [feat])[:, 0] if row_ids else np.zeros(0)
        out = np.zeros(len(row_ids), np.float32)
        for k, r_ in enumerate(row_ids):
            seg = ci[cp[r_]:cp[r_ + 1]]
            hit = np.nonzero(seg == feat)[0]
            if len(hit):
                out[k] = cv[cp[r_] + hit[0]]
        return out

    # examples in activation buckets linearly spaced between zero and the maximum (Scaling Monosemanticity)
    buckets: dict[int, list] = {}
    if keep and a.buckets > 0 and len(ci):
        row_of = np.repeat(rows[: len(cp) - 1], np.diff(cp))
        order = np.argsort(ci, kind="stable")
        ci_s, cv_s, rw_s = ci[order], cv[order], row_of[order]
        lo_i = np.searchsorted(ci_s, np.arange(n))
        hi_i = np.searchsorted(ci_s, np.arange(n), side="right")
        rng = np.random.default_rng(a.seed)
        for f in range(n):
            if hi_i[f] <= lo_i[f]:
                continue
            v, r_ = cv_s[lo_i[f]:hi_i[f]], rw_s[lo_i[f]:hi_i[f]]
            edges = np.linspace(0, float(v.max()), a.buckets + 1)
            b = np.clip(np.searchsorted(edges, v, side="left") - 1, 0, a.buckets - 1)
            out = []
            for k in range(a.buckets):
                sel = np.nonzero(b == k)[0]
                pick = rng.choice(sel, min(a.per_bucket, len(sel)), replace=False) if len(sel) else []
                out.append((float(edges[k]), float(edges[k + 1]), [int(r_[i_]) for i_ in pick]))
            buckets[f] = out

    def example(row: int, feat: int) -> dict:
        dd, pp = int(doc[row]), int(pos[row])
        ids, at = _context(wt, wp, dd, pp, a.left, a.right)
        start_pos = pp - at
        rws = [first_row[dd] + (start_pos + j - skip) if 0 <= start_pos + j - skip < rows_in[dd] else -1
               for j in range(len(ids))]
        valid = [r_ for r_ in rws if r_ >= 0]
        fa = acts_of(valid, feat) if valid else np.zeros(0)
        vals, it = [], iter(fa)
        for r_ in rws:
            vals.append(round(float(next(it)), 3) if r_ >= 0 else 0.0)
        return {"tok": ids, "act": vals, "at": at}

    alive = (freq > 0).nonzero().flatten().tolist()
    print(f"{len(alive)} of {n} features active at least once over {N} tokens", flush=True)
    logits = None
    model = tok = None
    if a.model or info.get("model"):
        try:
            model, tok, mdev = load_model(a.model or info["model"], "float32" if dev == "cpu" else "auto", a.device)
            if not a.no_logits:
                logits = logit_effects(model, sae, info, a.logit_top, mdev)
        except Exception as e:                                                     # noqa: BLE001
            print(f"skipping logit effects and token text from the model: {type(e).__name__}: {e}", flush=True)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(meta["model"])
    nb_v, nb_i = neighbours(sae, a.neighbours, dev)
    xy = pca2(sae.directions().to(dev).float())
    vocab: dict[int, str] = {}

    def piece(t: int) -> str:
        if t not in vocab:
            vocab[t] = tok.decode([t])
        return vocab[t]

    (d / "examples").mkdir(exist_ok=True)
    feats = []
    block, block_no, per = {}, 0, 256
    ex_t0 = time.time()
    for fid in range(n):
        e = {"id": fid, "freq": float(freq[fid]) / N, "mean": float(ssum[fid] / max(1.0, float(freq[fid]))),
             "max": float(mx[fid]), "hist": hist[fid].tolist(), "xy": [round(float(xy[fid][0]), 4),
                                                                      round(float(xy[fid][1]), 4)],
             "nb": [[int(i), round(float(v), 3)] for v, i in zip(nb_v[fid], nb_i[fid])]}
        if logits is not None:
            vu, iu, vd, idd = logits
            e["up"] = [[int(t), round(float(v), 3)] for v, t in zip(vu[fid], iu[fid])]
            e["down"] = [[int(t), round(float(v), 3)] for v, t in zip(vd[fid], idd[fid])]
            for t, _ in e["up"] + e["down"]:
                piece(t)
        feats.append(e)
        if freq[fid] > 0:
            tops = [int(r_) for v, r_ in zip(top_v[:, fid], top_i[:, fid]) if v > 0 and r_ >= 0]
            rnds = [int(r_) for v, r_ in zip(rnd_p[:, fid], rnd_i[:, fid]) if v >= 0 and r_ >= 0]
            exs = {"top": [example(r_, fid) for r_ in tops]}
            if fid in buckets:
                exs["buckets"] = [{"lo": round(lo, 4), "hi": round(hi, 4), "ex": [example(r_, fid) for r_ in rr]}
                                  for lo, hi, rr in buckets[fid]]
            else:
                exs["random"] = [example(r_, fid) for r_ in rnds]
            for x_ in exs["top"] + exs.get("random", []) + [y for b_ in exs.get("buckets", []) for y in b_["ex"]]:
                for t in x_["tok"]:
                    piece(t)
            block[str(fid)] = exs
        if (fid + 1) % per == 0 or fid == n - 1:
            (d / "examples" / f"{block_no:05d}.json").write_text(json.dumps(block, separators=(",", ":")))
            block, block_no = {}, block_no + 1
            progress(fid + 1, n, f"features described · {time.time() - ex_t0:.0f} s")
    labels = {}
    old = d / "features.json"
    if old.exists():
        try:
            for e in json.loads(old.read_text()).get("features", []):
                if e.get("label"):
                    labels[e["id"]] = {k: e[k] for k in ("label", "label_score", "label_baseline", "label_by",
                                                          "specificity") if k in e}
        except (ValueError, KeyError):
            pass
    if a.labeler:
        want = [int(x) for x in a.label_ids.split(",")] if a.label_ids else None
        if want is None:
            band = [f for f in alive if a.min_freq <= feats[f]["freq"] <= a.max_freq]
            band.sort(key=lambda f: -feats[f]["freq"] * feats[f]["max"])
            want = band[: a.label]
        print(f"labelling {len(want)} features with {a.labeler}", flush=True)
        for j, fid in enumerate(want):
            exs = json.loads((d / "examples" / f"{fid // per:05d}.json").read_text()).get(str(fid))
            if not exs:
                continue
            shown = [([piece(t) for t in x_["tok"]], x_["act"]) for x_ in exs["top"][: a.label_examples]]
            try:
                lab = _chat(a.labeler, [{"role": "user", "content": label_prompt(shown)}], 60).splitlines()[0]
                lab = lab.strip().strip('"').strip()[:80]
                rec = {"label": lab, "label_by": a.labeler.rsplit("@", 1)[-1]}
                if a.score == "specificity":
                    # rubric scores across activation levels, plus inactive excerpts as a control
                    pool = [(b_["lo"], x_) for b_ in exs.get("buckets", []) for x_ in b_["ex"][:2]] or \
                           [(0.0, x_) for x_ in exs.get("random", [])[: a.score_examples]] or \
                           [(0.0, x_) for x_ in exs["top"][a.label_examples:]]
                    neg_rows = [int(r_) for r_ in np.random.default_rng(fid).choice(store.n, 40, replace=False)]
                    fa = acts_of(neg_rows, fid)
                    neg = [example(r_, fid) for r_, v in zip(neg_rows, fa) if v <= 0][:3]

                    def show(x_):
                        ps = [piece(t) for t in x_["tok"]]
                        ps[x_["at"]] = f"[[{ps[x_['at']]}]]"
                        return "".join(ps).replace("\n", "⏎")
                    items = [(show(x_), lo) for lo, x_ in pool] + [(show(x_), None) for x_ in neg]
                    random.Random(fid).shuffle(items)
                    reply = _chat(a.labeler, [{"role": "user", "content": specificity_prompt(
                        lab, [t for t, _ in items])}], 12 * len(items) + 20)
                    got = {int(k): int(v) for k, v in __import__("re").findall(r"(\d+)\s*[:.)-]\s*([0-3])", reply)}
                    act_s = [got[i_] for i_, (_, lo) in enumerate(items, 1) if lo is not None and i_ in got]
                    neg_s = [got[i_] for i_, (_, lo) in enumerate(items, 1) if lo is None and i_ in got]
                    if act_s:
                        rec["label_score"] = round(sum(act_s) / (3 * len(act_s)), 3)
                        rec["label_baseline"] = round(sum(neg_s) / (3 * len(neg_s)), 3) if neg_s else None
                        by: dict = {}
                        for i_, (_, lo) in enumerate(items, 1):
                            if lo is not None and i_ in got:
                                by.setdefault(round(lo, 4), []).append(got[i_])
                        rec["specificity"] = [[k, round(sum(v) / len(v), 2)] for k, v in sorted(by.items())]
                elif a.score == "detection":
                    pos_ex = [x_ for b_ in exs.get("buckets", []) for x_ in b_["ex"]][: a.score_examples] or \
                             exs.get("random", [])[: a.score_examples] or exs["top"][a.label_examples:]
                    neg_rows = [int(r_) for r_ in np.random.default_rng(fid).choice(store.n, 60, replace=False)]
                    fa = acts_of(neg_rows, fid)
                    neg_ex = [example(r_, fid) for r_, v in zip(neg_rows, fa) if v <= 0][: len(pos_ex)]
                    items = [("".join(piece(t) for t in x_["tok"]).replace("\n", "⏎"), 1) for x_ in pos_ex] + \
                            [("".join(piece(t) for t in x_["tok"]).replace("\n", "⏎"), 0) for x_ in neg_ex]
                    random.Random(fid).shuffle(items)
                    reply = _chat(a.labeler, [{"role": "user", "content": score_prompt(lab, [t for t, _ in items])}],
                                  60)
                    picked = {int(x) for x in __import__("re").findall(r"\d+", reply)}
                    tp = sum(1 for i_, (_, y) in enumerate(items, 1) if y and i_ in picked)
                    tn = sum(1 for i_, (_, y) in enumerate(items, 1) if not y and i_ not in picked)
                    npos, nneg = sum(y for _, y in items), sum(1 - y for _, y in items)
                    rec["label_score"] = round(0.5 * (tp / max(1, npos) + tn / max(1, nneg)), 3)
                labels[fid] = rec
            except Exception as e:                                                  # noqa: BLE001
                print(f"feature {fid}: labelling failed: {type(e).__name__}: {e}", flush=True)
            progress(j + 1, len(want), f"features labelled ({fid}: {labels.get(fid, {}).get('label', '?')})")
    for fid, rec in labels.items():
        if 0 <= fid < n:
            feats[fid].update(rec)
    dens = np.histogram(np.log10(np.clip(freq.numpy() / N, 1e-9, None)), bins=np.linspace(-9, 0, 37))[0]
    summary = {"name": d.name, "model": info.get("model"), "site": info["site"], "layer": info["layer"],
               "target_site": info.get("target_site"), "n_features": n, "tokens": N, "alive": len(alive),
               "density_hist": dens.tolist(), "density_edges": [-9 + 0.25 * i for i in range(37)],
               "act_bins": [round(float(x), 5) for x in bins], "per_file": per, "features": feats,
               "codes": bool(keep), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "features.json").write_text(json.dumps(summary, separators=(",", ":")))
    (d / "vocab.json").write_text(json.dumps({str(k): v for k, v in vocab.items()}, ensure_ascii=False))
    print(f"wrote {d}/features.json, examples/ and vocab.json ({len(alive)} active features"
          + (f", {sum(1 for f in feats if f.get('label'))} labelled" if labels else "") + ")")
    return 0


# ====================================================================== neurons

def read_h_neurons(path: str | None) -> dict[int, set[int]]:
    if not path:
        return {}
    d = json.loads(Path(path).read_text())
    return {int(k): set(int(x) for x in v) for k, v in (d.get("by_layer") or {}).items()}


def cmd_neurons(a) -> int:
    """Map features onto MLP neurons: by weights, and by correlation over recorded activations."""
    import sae as S
    import torch
    from concept_edit import decoder_layers
    d = Path(a.sae)
    sae, info = S.load(d)
    model, tok, dev = load_model(a.model or info["model"], "float32", a.device)
    layers = decoder_layers(model)
    L, site, top = int(info["layer"]), info.get("target_site") or info["site"], a.top
    hn = read_h_neurons(a.h_neurons)
    out: dict = {"model": info.get("model"), "site": site, "layer": L, "method": [], "features": {}}
    D = sae.directions().float().to(dev)                                   # [n, d_out]
    n = D.shape[0]
    per_feat = [dict() for _ in range(n)]
    if site == "neurons":
        out["method"].append("decoder")
        W = sae.W_dec.detach().float().to(dev)
        E = sae.W_enc.detach().float().to(dev).t()
        v, i = W.abs().topk(min(top, W.shape[1]), dim=1)
        ve, ie = E.abs().topk(min(top, E.shape[1]), dim=1)
        for f in range(n):
            per_feat[f]["writes"] = [[L, int(j), round(float(W[f, j]), 4)] for j in i[f].tolist()]
            per_feat[f]["reads"] = [[L, int(j), round(float(E[f, j]), 4)] for j in ie[f].tolist()]
    elif site in ("resid", "mlp_out"):
        out["method"].append("weights")
        writers = range(0, L + 1) if site == "resid" else [L]
        readers = range(L + 1, len(layers)) if site == "resid" else range(L + 1, len(layers))
        cand_w = [[] for _ in range(n)]
        cand_r = [[] for _ in range(n)]
        for j, l in enumerate(writers):
            Wd = down_of(mlp_of(layers[l])).weight.detach().float().to(dev)            # [d, d_ff]
            C = D @ Wd                                                                  # [n, d_ff]
            v, i = C.abs().topk(min(top, C.shape[1]), dim=1)
            for f in range(n):
                cand_w[f] += [(float(C[f, k]), l, int(k)) for k in i[f].tolist()]
            progress(j + 1, len(writers) + len(readers), "layers compared (writers)")
        for j, l in enumerate(readers):
            mlp = mlp_of(layers[l])
            up = next((getattr(mlp, nm) for nm in ("up_proj", "gate_proj", "c_fc", "fc1", "w1", "dense_h_to_4h")
                       if getattr(mlp, nm, None) is not None), None)
            if up is None:
                continue
            Wu = up.weight.detach().float().to(dev)                                    # [d_ff, d]
            norm = getattr(layers[l], "post_attention_layernorm", None)
            g = getattr(norm, "weight", None)
            C = (D * g.detach().float().to(dev)) @ Wu.t() if g is not None else D @ Wu.t()
            v, i = C.abs().topk(min(top, C.shape[1]), dim=1)
            for f in range(n):
                cand_r[f] += [(float(C[f, k]), l, int(k)) for k in i[f].tolist()]
            progress(len(writers) + j + 1, len(writers) + len(readers), "layers compared (readers)")
        for f in range(n):
            per_feat[f]["writes"] = [[l, k, round(c, 4)] for c, l, k in sorted(cand_w[f], key=lambda t: -abs(t[0]))[:top]]
            per_feat[f]["reads"] = [[l, k, round(c, 4)] for c, l, k in sorted(cand_r[f], key=lambda t: -abs(t[0]))[:top]]
    else:
        print("mlp_in dictionaries read the MLP's input: weights map to the neurons that read them", flush=True)
        mlp = mlp_of(layers[L])
        up = next((getattr(mlp, nm) for nm in ("up_proj", "gate_proj", "c_fc", "fc1", "w1")
                   if getattr(mlp, nm, None) is not None), None)
        if up is not None:
            out["method"].append("weights")
            C = D @ up.weight.detach().float().to(dev).t()
            v, i = C.abs().topk(min(top, C.shape[1]), dim=1)
            for f in range(n):
                per_feat[f]["reads"] = [[L, int(k), round(float(C[f, k]), 4)] for k in i[f].tolist()]
    acts = a.acts or info.get("acts")
    if acts and (Path(acts) / f"neurons.{L}").is_dir():
        out["method"].append("correlation")
        st = Store(acts, info["site"], L)
        nst = Store(acts, "neurons", L)
        sae.to(dev)
        rows = np.arange(min(st.n, a.max_rows))
        sf = sf2 = None
        sn = sn2 = None
        sfn = None
        for b in range(0, len(rows), 4096):
            r = rows[b:b + 4096]
            with torch.no_grad():
                fa = sae.encode(torch.from_numpy(st.rows(r)).to(dev)).float()
            na = torch.from_numpy(nst.rows(r)).to(dev).float()
            if sf is None:
                sf, sf2 = fa.sum(0), fa.pow(2).sum(0)
                sn, sn2 = na.sum(0), na.pow(2).sum(0)
                sfn = fa.t() @ na
            else:
                sf += fa.sum(0)
                sf2 += fa.pow(2).sum(0)
                sn += na.sum(0)
                sn2 += na.pow(2).sum(0)
                sfn += fa.t() @ na
            progress(min(b + 4096, len(rows)), len(rows), "tokens correlated")
        N = len(rows)
        mf, mn = sf / N, sn / N
        cov = sfn / N - mf[:, None] * mn[None, :]
        sdf = (sf2 / N - mf ** 2).clamp(min=1e-12).sqrt()
        sdn = (sn2 / N - mn ** 2).clamp(min=1e-12).sqrt()
        R = cov / (sdf[:, None] * sdn[None, :])
        v, i = R.abs().topk(min(top, R.shape[1]), dim=1)
        best = R.abs().max(1).values.cpu()
        for f in range(n):
            per_feat[f]["corr"] = [[L, int(k), round(float(R[f, k]), 4)] for k in i[f].tolist()]
        out["max_corr_hist"] = np.histogram(best.numpy(), bins=np.linspace(0, 1, 21))[0].tolist()
        print(f"median best |correlation| of a feature with any one neuron: {float(best.median()):.2f}", flush=True)
    if hn:
        out["h_neurons"] = a.h_neurons
        shares = []
        for f in range(n):
            lst = per_feat[f].get("writes") or per_feat[f].get("corr") or per_feat[f].get("reads") or []
            tot = sum(abs(w) for _, _, w in lst) or 1.0
            hit = sum(abs(w) for l, k, w in lst if k in hn.get(int(l), ()))
            per_feat[f]["h_share"] = round(hit / tot, 3)
            per_feat[f]["h_count"] = sum(1 for l, k, _ in lst if k in hn.get(int(l), ()))
            shares.append(per_feat[f]["h_share"])
        ranked = sorted(range(n), key=lambda f: -shares[f])[:20]
        out["most_h"] = [[f, shares[f]] for f in ranked]
        print("features whose top neurons are most H-Neurons: "
              + ", ".join(f"{f} ({100 * shares[f]:.0f}%)" for f in ranked[:8]), flush=True)
    out["features"] = {str(f): v for f, v in enumerate(per_feat)}
    (d / "neurons.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {d}/neurons.json ({', '.join(out['method']) or 'no method applied'})")
    return 0


# ====================================================================== CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="record a model's activations over text")
    c.add_argument("--model", required=True, help="Hugging Face id or folder")
    c.add_argument("--corpus", required=True,
                   help="folder of .txt/.md, a .txt, a .jsonl (text / messages / prompt+reply), or a Hub dataset id")
    c.add_argument("--field", help="the text field of .jsonl rows or dataset rows (default: text, messages, ...)")
    c.add_argument("--split", default="train", help="dataset split")
    c.add_argument("--config", help="dataset config")
    c.add_argument("--site", nargs="+", default=["resid"], choices=SITES, help="where in the layer (several allowed)")
    c.add_argument("--layers", nargs="+", help="decoder layers (default: the middle one)")
    c.add_argument("--max-tokens", type=int, default=400000, help="stop after this many tokens")
    c.add_argument("--ctx", type=int, default=256, help="tokens per window")
    c.add_argument("--skip", type=int, default=1,
                   help="leave out the first N positions of each window (their norms are outliers)")
    c.add_argument("--bos", action=argparse.BooleanOptionalAction, default=True,
                   help="start each window with the tokenizer's BOS token, if it has one")
    c.add_argument("--chunk-long", action=argparse.BooleanOptionalAction, default=True,
                   help="split long documents into several windows instead of keeping only the start")
    c.add_argument("--limit-docs", type=int, default=0, help="use at most N documents (0: all)")
    c.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="shuffle documents")
    c.add_argument("--batch", type=int, default=8, help="windows per forward pass")
    c.add_argument("--shard-rows", type=int, default=65536, help="rows per file")
    c.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    c.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True, help="folder for the activations")

    t = sub.add_parser("train", help="learn a dictionary (sparse autoencoder or transcoder)")
    t.add_argument("--acts", required=True, help="folder from collect")
    t.add_argument("--site", default="resid", choices=SITES)
    t.add_argument("--layer", type=int, required=True)
    t.add_argument("--target-site", choices=SITES,
                   help="reconstruct this site instead (a transcoder, e.g. --site mlp_in --target-site mlp_out)")
    t.add_argument("--arch", default="topk", choices=["topk", "relu", "jumprelu"])
    t.add_argument("--k", type=int, default=32, help="topk: features active per token")
    t.add_argument("--features", type=int, default=0, help="dictionary size (default: expansion x width)")
    t.add_argument("--expansion", type=float, default=8.0, help="dictionary size as a multiple of the width")
    t.add_argument("--target-l0", type=float, default=32.0,
                   help="relu / jumprelu: steer the sparsity coefficient toward this many active features")
    t.add_argument("--sparsity", type=float, default=None, help="relu / jumprelu: the starting coefficient")
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch", type=int, default=4096, help="rows per step")
    t.add_argument("--epochs", type=float, default=3.0, help="passes over the training rows (if --steps is 0)")
    t.add_argument("--steps", type=int, default=0)
    t.add_argument("--dead-after", type=int, default=500000, help="tokens without firing before a feature counts as dead")
    t.add_argument("--check-model", action=argparse.BooleanOptionalAction, default=True,
                   help="measure the model's next-token loss with the dictionary spliced in")
    t.add_argument("--check-seqs", type=int, default=48, help="held-out windows for that check")
    t.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True, help="folder for the dictionary")

    m = sub.add_parser("import", help="use a dictionary trained elsewhere (SAELens, Gemma Scope, safetensors)")
    m.add_argument("--path", help="folder or file")
    m.add_argument("--release", help="a SAELens pretrained release, e.g. gemma-scope-2b-pt-res (downloads it)")
    m.add_argument("--sae-id", help="the dictionary within the release, e.g. layer_12/width_16k/average_l0_82")
    m.add_argument("--model", help="the model it was trained on (HF id or folder)")
    m.add_argument("--site", choices=SITES, help="where it reads (taken from SAELens hook names when present)")
    m.add_argument("--layer", type=int)
    m.add_argument("--target-site", choices=SITES, help="for transcoders: what it reconstructs")
    m.add_argument("--acts", help="activations to interpret it on (from collect)")
    m.add_argument("--out", required=True)

    i = sub.add_parser("interpret", help="top examples, logit effects, neighbours, map and labels per feature")
    i.add_argument("--sae", required=True, help="dictionary folder")
    i.add_argument("--acts", help="activations (default: the ones it was trained on)")
    i.add_argument("--model", help="for logit effects and token text (default: the dictionary's model)")
    i.add_argument("--top", type=int, default=12, help="top examples per feature")
    i.add_argument("--random", type=int, default=6,
                   help="random active examples per feature (when codes are not kept)")
    i.add_argument("--buckets", type=int, default=5,
                   help="activation buckets between zero and the maximum to sample examples from (0: random instead)")
    i.add_argument("--per-bucket", type=int, default=3, help="examples per bucket")
    i.add_argument("--left", type=int, default=10, help="context tokens before")
    i.add_argument("--right", type=int, default=4, help="context tokens after")
    i.add_argument("--logit-top", type=int, default=10)
    i.add_argument("--no-logits", action="store_true", help="skip logit effects")
    i.add_argument("--neighbours", type=int, default=8, help="nearest features by decoder direction")
    i.add_argument("--keep-codes", type=int, default=64,
                   help="store up to N active features per token for browsing (0: none)")
    i.add_argument("--max-rows", type=int, default=0, help="use at most N tokens (0: all)")
    i.add_argument("--batch", type=int, default=4096)
    i.add_argument("--labeler", help="URL@model of an OpenAI-compatible endpoint that writes labels")
    i.add_argument("--label", type=int, default=100, help="how many features to label")
    i.add_argument("--label-ids", help="label exactly these feature ids (comma-separated)")
    i.add_argument("--label-examples", type=int, default=8, help="examples shown to the labeller")
    i.add_argument("--score", choices=["specificity", "detection", "none"], default="specificity",
                   help="check each label: specificity (0-3 rubric across activation levels, as in Scaling "
                        "Monosemanticity) or detection (can the labeller tell active from inactive excerpts)")
    i.add_argument("--score-examples", type=int, default=5)
    i.add_argument("--min-freq", type=float, default=1e-5, help="label only features at least this frequent")
    i.add_argument("--max-freq", type=float, default=0.1, help="... and at most this frequent")
    i.add_argument("--device", default="auto")
    i.add_argument("--seed", type=int, default=0)

    n = sub.add_parser("neurons", help="which MLP neurons carry each feature")
    n.add_argument("--sae", required=True)
    n.add_argument("--model", help="default: the dictionary's model")
    n.add_argument("--acts", help="with a neurons site recorded at the same layer, also correlate over tokens")
    n.add_argument("--h-neurons", help="h_neurons.json: report how much of each feature sits on H-Neurons")
    n.add_argument("--top", type=int, default=12, help="neurons kept per feature and method")
    n.add_argument("--max-rows", type=int, default=200000, help="tokens used for correlation")
    n.add_argument("--device", default="auto")

    a = p.parse_args(argv)
    if a.cmd == "collect":
        a.site = list(dict.fromkeys(a.site))
        return cmd_collect(a)
    return {"train": cmd_train, "import": cmd_import, "interpret": cmd_interpret, "neurons": cmd_neurons}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
