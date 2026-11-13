#!/usr/bin/env python3
"""
H-Neurons for contrastive image-text models (CLIP, OpenCLIP-in-HF, SigLIP).

The text pipeline labels a generation as hallucinated when a model confidently
states something wrong. The contrastive analogue is a *confident mismatch*:
the image is matched to the wrong caption, and it stays wrong across
augmented views of the same image (so it is the model's belief, not crop noise).
Everything after labelling reuses the text pipeline unchanged:

    collect   zero-shot classify every image under K augmented views and
              keep the consistently-right (t) / consistently-wrong (f) ones
    extract   CETT per MLP neuron of the vision (or text) tower, written in
              the same act_<id>.npy + neuron_index.json layout as stage 4
    (train)   python scripts/classifier.py --acts_root ... --ans_dir image
              --train_mode 1-vs-1   -> h_neurons.json
    evaluate  sweep neuron scales and measure accuracy, confident-error
              rate, abstention and selective accuracy
    export    write an edited HF checkpoint (fc2 columns scaled); for a
              llama.cpp mmproj GGUF use scripts/suppress_mmproj.py

CETT here is computed on fc2 (the down projection of a non-gated
fc1 -> act -> fc2 MLP):

    CETT(layer, token, j) = |a_j| * ||W_fc2[:, j]|| / ||fc2 output||

Images come either from an ImageFolder tree (root/<class name>/*.jpg) or a
JSONL manifest of {"id", "image", "label", "candidates"?}.

    python scripts/clip_neurons.py collect --model openai/clip-vit-large-patch14 \\
        --images data/imagenet-val --out data/clip
    python scripts/clip_neurons.py extract --model openai/clip-vit-large-patch14 \\
        --run data/clip --ids data/clip/train_qids.json --out data/clip/acts
    python scripts/classifier.py --acts_root data/clip/acts --ans_dir image \\
        --train_mode 1-vs-1 --train_ids data/clip/train_qids.json \\
        --test_ids data/clip/test_qids.json --C 0.05 --out_dir models/clip
    python scripts/clip_neurons.py evaluate --model openai/clip-vit-large-patch14 \\
        --run data/clip --ids data/clip/test_qids.json \\
        --h_neurons models/clip/h_neurons.json --scales 1 0.75 0.5 0.25 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_style as cs  # noqa: E402

TOWER_RE = re.compile(r"(?:^|\.)(vision_model|text_model)\.encoder\.layers\.(\d+)\.mlp\.fc2$")
TOWER_PREFIX = {"vision": "vision_model", "text": "text_model"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
DEFAULT_TEMPLATES = ["a photo of a {}.", "a close-up photo of the {}.", "a blurry photo of a {}."]


# ---------------------------------------------------------------- data

def sample_id(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def load_dataset(images: str | None, manifest: str | None) -> list[dict]:
    rows: list[dict] = []
    if manifest:
        base = Path(manifest).parent
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                img = Path(r["image"])
                if not img.is_absolute():
                    img = base / img
                rows.append({"id": str(r.get("id") or sample_id(str(r["image"]))), "image": str(img),
                             "label": str(r["label"]), "candidates": r.get("candidates")})
    elif images:
        root = Path(images)
        for cls in sorted(p for p in root.iterdir() if p.is_dir()):
            label = cls.name.replace("_", " ")
            for img in sorted(cls.rglob("*")):
                if img.suffix.lower() in IMAGE_EXT:
                    rel = str(img.relative_to(root))
                    rows.append({"id": sample_id(rel), "image": str(img), "label": label, "candidates": None})
    else:
        raise SystemExit("need --images DIR or --manifest FILE")
    if not rows:
        raise SystemExit("no images found")
    return rows


def views(img, k: int):
    """K deterministic augmented views: identity, mirror, then centre/corner crops."""
    from PIL import ImageOps
    w, h = img.size
    out = [img, ImageOps.mirror(img)]
    for frac, anchor in [(0.85, "c"), (0.8, "tl"), (0.8, "br"), (0.7, "c"), (0.8, "tr"), (0.8, "bl")]:
        cw, ch = int(w * frac), int(h * frac)
        x = {"c": (w - cw) // 2, "tl": 0, "bl": 0, "tr": w - cw, "br": w - cw}[anchor]
        y = {"c": (h - ch) // 2, "tl": 0, "tr": 0, "bl": h - ch, "br": h - ch}[anchor]
        out.append(img.crop((x, y, x + cw, y + ch)))
    return out[:max(1, k)]


def open_image(path: str, style: str = "off", target: int = 224):
    """The image as the model should see it; style pixel / cartoon / auto: see clip_style.py."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return cs.prepare(img, style, target)[0] if style != "off" else img


# ---------------------------------------------------------------- model

def load(model_path: str, device: str = "cpu", dtype: str = "float32"):
    import torch
    from transformers import AutoModel, AutoProcessor
    import transformers
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModel.from_pretrained(model_path, **{key: getattr(torch, dtype)}, trust_remote_code=False).to(device).eval()
    if not (hasattr(model, "vision_model") and hasattr(model, "text_model")):
        raise SystemExit(f"{model_path} is not a dual-encoder (CLIP/SigLIP-style) model")
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    return model, proc


def _feats(out):
    return out if hasattr(out, "norm") else out.pooler_output


def text_embeddings(model, proc, labels: list[str], templates: list[str], device: str):
    """Prompt-ensembled, L2-normalised text embedding per label."""
    import torch
    embs = []
    with torch.no_grad():
        for label in labels:
            prompts = [t.format(label) for t in templates]
            tok = proc(text=prompts, return_tensors="pt", padding=True)
            tok = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
            e = _feats(model.get_text_features(**tok)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            embs.append(e / e.norm())
    return torch.stack(embs)


def image_embeddings(model, proc, imgs, device: str):
    import torch
    with torch.no_grad():
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, next(model.parameters()).dtype)
        e = _feats(model.get_image_features(pixel_values=px)).float()
    return e / e.norm(dim=-1, keepdim=True)


def zero_shot(model, img_emb, txt_emb):
    import torch
    with torch.no_grad():
        scale = model.logit_scale.exp().float() if hasattr(model, "logit_scale") else torch.tensor(100.0)
        return (scale * img_emb @ txt_emb.T).softmax(-1)


# ---------------------------------------------------------------- collect

def cmd_collect(a) -> int:
    rows = load_dataset(a.images, a.manifest)
    labels = sorted({r["label"] for r in rows})
    if a.labels:
        labels = [x.strip() for x in Path(a.labels).read_text().splitlines() if x.strip()]
    style = a.style or "off"
    templates = cs.templates(a.template or DEFAULT_TEMPLATES, style)
    model, proc = load(a.model, a.device, a.dtype)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    global_txt = text_embeddings(model, proc, labels, templates, a.device)
    cache: dict[tuple, object] = {}
    stats = {"t": 0, "f": 0, "mixed": 0, "uncertain": 0}
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            cands = r["candidates"] or labels
            if r["candidates"]:
                key = tuple(cands)
                if key not in cache:
                    cache[key] = text_embeddings(model, proc, cands, templates, a.device)
                txt = cache[key]
            else:
                txt = global_txt
            if r["label"] not in cands:
                continue
            gold = cands.index(r["label"])
            probs = zero_shot(model, image_embeddings(model, proc, views(open_image(r["image"], style, cs.target_size(proc)), a.views), a.device), txt)
            preds = probs.argmax(-1).tolist()
            conf = probs.max(-1).values.tolist()
            n_ok = sum(p == gold for p in preds)
            if n_ok == len(preds):
                verdict = "t"
            elif n_ok == 0 and len(set(preds)) == 1 and min(conf) >= a.min_conf:
                verdict = "f"  # consistently, confidently the same wrong caption
            elif n_ok == 0:
                verdict = "uncertain"
            else:
                verdict = "mixed"
            stats[verdict] += 1
            f.write(json.dumps({"id": r["id"], "image": r["image"], "label": r["label"],
                                "candidates": r["candidates"], "pred": cands[preds[0]],
                                "view_preds": [cands[p] for p in preds], "conf": conf,
                                "verdict": verdict}) + "\n")
    t_ids, f_ids = _ids_by_verdict(out / "results.jsonl")
    split = balanced_split(t_ids, f_ids, a.test_frac, a.seed)
    for name, ids in split.items():
        (out / f"{name}_qids.json").write_text(json.dumps(ids, indent=2))
    meta = {"model": a.model, "labels": labels, "templates": templates, "views": a.views, "style": style,
            "min_conf": a.min_conf, "stats": stats,
            "train": {k: len(v) for k, v in split["train"].items()},
            "test": {k: len(v) for k, v in split["test"].items()}}
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"stats": stats, "train": meta["train"], "test": meta["test"]}, indent=2))
    if min(len(t_ids), len(f_ids)) < 20:
        print("warning: fewer than 20 examples in a class; the classifier will be noise. "
              "Use more images, harder labels, or lower --min_conf.", file=sys.stderr)
    return 0


def _ids_by_verdict(path: Path):
    t, f = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        (t if r["verdict"] == "t" else f if r["verdict"] == "f" else []).append(r["id"])
    return t, f


def balanced_split(t_ids, f_ids, test_frac: float, seed: int) -> dict:
    rng = random.Random(seed)
    n = min(len(t_ids), len(f_ids))
    t = rng.sample(t_ids, n)
    f = rng.sample(f_ids, n)
    k = int(round(n * test_frac))
    return {"train": {"t": t[k:], "f": f[k:]}, "test": {"t": t[:k], "f": f[:k]}}


# ---------------------------------------------------------------- CETT

class ClipCETT:
    """Hooks every fc2 in one tower and computes per-neuron CETT."""

    def __init__(self, model, tower: str):
        import torch
        prefix = TOWER_PREFIX[tower]
        self.acts, self.out_norms, self.weight_norms, self.hooks = {}, {}, {}, []
        self.modules = {}
        for name, mod in model.named_modules():
            m = TOWER_RE.search(name)
            if not m or m.group(1) != prefix:
                continue
            i = int(m.group(2))
            self.modules[i] = mod
            self.weight_norms[i] = torch.norm(mod.weight.data.float(), dim=0).cpu()
            self.hooks.append(mod.register_forward_hook(self._hook(i)))
        self.layer_ids = sorted(self.modules)
        if not self.layer_ids:
            raise SystemExit(f"no {prefix}.encoder.layers.N.mlp.fc2 modules found")
        self.n_layers = len(self.layer_ids)
        self.n_neurons = int(self.weight_norms[self.layer_ids[0]].shape[0])
        self.template = f"{prefix}.encoder.layers.{{layer}}.mlp.fc2"

    def _hook(self, i):
        import torch

        def fn(_m, inputs, output):
            self.acts[i] = inputs[0].detach()[0].float().cpu()
            self.out_norms[i] = torch.norm(output.detach()[0].float().cpu(), dim=-1, keepdim=True)
        return fn

    def cett(self):
        import torch
        a = torch.stack([self.acts[i] for i in self.layer_ids]).abs()
        wn = torch.stack([self.weight_norms[i] for i in self.layer_ids])
        n = torch.stack([self.out_norms[i] for i in self.layer_ids])
        return (a * wn.unsqueeze(1)) / (n + 1e-8)   # [layers, tokens, neurons]

    def remove(self):
        for h in self.hooks:
            h.remove()


def read_results(run: str) -> dict:
    return {r["id"]: r for r in map(json.loads, open(Path(run) / "results.jsonl", encoding="utf-8"))}


def cmd_extract(a) -> int:
    import torch
    model, proc = load(a.model, a.device, a.dtype)
    results = read_results(a.run)
    ids = json.loads(Path(a.ids).read_text())
    wanted = list(ids.get("t", [])) + list(ids.get("f", []))
    run_meta = json.loads((Path(a.run) / "run.json").read_text())
    templates = run_meta.get("templates") or DEFAULT_TEMPLATES
    style = a.style or run_meta.get("style") or "off"
    mgr = ClipCETT(model, a.tower)
    out = Path(a.out)
    locs = ["image", "cls"] if a.tower == "vision" else ["text"]
    for loc in locs:
        (out / loc).mkdir(parents=True, exist_ok=True)
    (out / "neuron_index.json").write_text(json.dumps({
        "n_layers": mgr.n_layers, "n_neurons": mgr.n_neurons,
        "order": "flat = layer * n_neurons + neuron", "model_path": a.model,
        "arch": "clip", "tower": a.tower, "module_template": mgr.template,
        "locations": locs}, indent=2))
    print(f"hooked {mgr.n_layers} {a.tower} layers x {mgr.n_neurons} neurons")
    written = 0
    for qid in wanted:
        r = results.get(qid)
        if r is None:
            continue
        with torch.no_grad():
            if a.tower == "vision":
                px = proc(images=[open_image(r["image"], a.style or run_meta.get("style") or "off",
                                             cs.target_size(proc))], return_tensors="pt")["pixel_values"]
                model.get_image_features(pixel_values=px.to(a.device, next(model.parameters()).dtype))
                c = mgr.cett()
                np.save(out / "image" / f"act_{qid}.npy", c.mean(1).numpy().astype(np.float16))
                np.save(out / "cls" / f"act_{qid}.npy", c[:, 0].numpy().astype(np.float16))
            else:
                # The caption the model chose: that is where a confident mismatch lives.
                tok = proc(text=[templates[0].format(r["pred"])], return_tensors="pt")
                model.get_text_features(input_ids=tok["input_ids"].to(a.device),
                                        **({"attention_mask": tok["attention_mask"].to(a.device)}
                                           if "attention_mask" in tok else {}))
                np.save(out / "text" / f"act_{qid}.npy", mgr.cett().mean(1).numpy().astype(np.float16))
        written += 1
    mgr.remove()
    print(f"wrote {written} samples to {out}")
    return 0


# ---------------------------------------------------------------- intervene

def load_profile(path: str, n_neurons: int) -> dict[int, list[int]]:
    hn = json.loads(Path(path).read_text())
    if hn.get("n_neurons") not in (None, n_neurons):
        raise SystemExit(f"profile has {hn['n_neurons']} neurons/layer, model tower has {n_neurons}")
    return {int(k): sorted({int(i) for i in v}) for k, v in hn["by_layer"].items() if v}


class Scaler:
    """Forward pre-hooks that multiply selected fc2 inputs by ``scale``."""

    def __init__(self, mgr: ClipCETT, by_layer: dict[int, list[int]]):
        import torch
        self.scale = 1.0
        self.hooks = []
        for layer, idx in by_layer.items():
            mod = mgr.modules[layer]
            t = torch.tensor(idx, dtype=torch.long)

            def pre(_m, args, t=t):
                if self.scale == 1.0:
                    return None
                x = args[0].clone()
                x[..., t] *= self.scale
                return (x,) + tuple(args[1:])
            self.hooks.append(mod.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.hooks:
            h.remove()


def selective_metrics(probs, gold, thr: float) -> dict:
    probs = np.asarray(probs)
    gold = np.asarray(gold)
    pred = probs.argmax(1)
    conf = probs.max(1)
    ok = pred == gold
    answered = conf >= thr
    n = len(gold)
    out = {"n": n, "accuracy": float(ok.mean()) if n else 0.0,
           "coverage": float(answered.mean()) if n else 0.0,
           "abstain_rate": float((~answered).mean()) if n else 0.0,
           "confident_error_rate": float((answered & ~ok).mean()) if n else 0.0,
           "selective_accuracy": float(ok[answered].mean()) if answered.any() else None,
           "mean_conf_correct": float(conf[ok].mean()) if ok.any() else None,
           "mean_conf_wrong": float(conf[~ok].mean()) if (~ok).any() else None}
    try:
        from sklearn.metrics import roc_auc_score
        if 0 < ok.sum() < n:
            out["conf_auroc"] = float(roc_auc_score(ok, conf))
    except ImportError:
        pass
    return out


def cmd_evaluate(a) -> int:
    model, proc = load(a.model, a.device, a.dtype)
    results = read_results(a.run)
    ids = json.loads(Path(a.ids).read_text())
    wanted = [i for i in list(ids.get("t", [])) + list(ids.get("f", [])) if i in results]
    if a.limit:
        wanted = wanted[:a.limit]
    run_meta = json.loads((Path(a.run) / "run.json").read_text())
    labels = run_meta["labels"]
    templates = run_meta.get("templates") or DEFAULT_TEMPLATES
    style = a.style or run_meta.get("style") or "off"
    hn_meta = json.loads(Path(a.h_neurons).read_text())
    tower = a.tower or hn_meta.get("tower") or "vision"
    mgr = ClipCETT(model, tower)
    mgr.remove()  # only need module handles here
    scaler = Scaler(mgr, load_profile(a.h_neurons, mgr.n_neurons))
    report = {"model": a.model, "h_neurons": a.h_neurons, "tower": tower, "threshold": a.threshold,
              "n_selected": sum(len(v) for v in load_profile(a.h_neurons, mgr.n_neurons).values()),
              "scales": {}}
    for s in a.scales:
        scaler.scale = float(s)
        txt_cache: dict = {}
        probs, gold = [], []
        for qid in wanted:
            r = results[qid]
            cands = r["candidates"] or labels
            key = tuple(cands)
            if key not in txt_cache:
                txt_cache[key] = text_embeddings(model, proc, cands, templates, a.device)
            p = zero_shot(model, image_embeddings(model, proc, [open_image(r["image"], style, cs.target_size(proc))],
                                                  a.device), txt_cache[key])
            probs.append(p[0].cpu().numpy())
            gold.append(cands.index(r["label"]))
        # Candidate sets can differ per row; pad to a common width for the metrics.
        width = max(len(p) for p in probs)
        P = np.zeros((len(probs), width))
        for i, p in enumerate(probs):
            P[i, :len(p)] = p
        report["scales"][str(s)] = selective_metrics(P, gold, a.threshold)
    scaler.remove()
    print(f"{'scale':>6} {'acc':>6} {'cover':>6} {'conf-err':>8} {'sel-acc':>7}")
    for s, m in report["scales"].items():
        sa = m["selective_accuracy"]
        print(f"{s:>6} {m['accuracy']:6.3f} {m['coverage']:6.3f} {m['confident_error_rate']:8.3f} "
              f"{'   n/a' if sa is None else f'{sa:7.3f}'}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {a.out}")
    return 0


def cmd_export(a) -> int:
    import torch
    model, proc = load(a.model, "cpu", "float32")
    hn_meta = json.loads(Path(a.h_neurons).read_text())
    tower = a.tower or hn_meta.get("tower") or "vision"
    mgr = ClipCETT(model, tower)
    mgr.remove()
    by_layer = load_profile(a.h_neurons, mgr.n_neurons)
    with torch.no_grad():
        for layer, idx in by_layer.items():
            mgr.modules[layer].weight[:, idx] *= a.scale  # column j of fc2 == neuron j's output
    out = Path(a.out)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    (out / "omnicortex.json").write_text(json.dumps({
        "source": a.model, "h_neurons": str(Path(a.h_neurons).resolve()), "scale": a.scale,
        "tower": tower, "n_neurons": sum(len(v) for v in by_layer.values()),
        "method": "fc2 weight columns scaled"}, indent=2))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", required=True, help="HF id or local path (CLIP/SigLIP)")
        sp.add_argument("--device", default="cuda" if _cuda() else "cpu")
        sp.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                        help="float32 for extraction: CETT measures magnitudes")
        sp.add_argument("--style", choices=cs.STYLES, default=None,
                        help="pixel art / cartoon handling (clip_style.py); extract and evaluate default to "
                             "the collect run's, collect to off")

    c = sub.add_parser("collect", help="zero-shot label images as consistent right/wrong")
    common(c)
    c.add_argument("--images", help="ImageFolder root: <root>/<class name>/*.jpg")
    c.add_argument("--manifest", help="JSONL with id, image, label, optional candidates")
    c.add_argument("--labels", help="file with one candidate label per line (default: dataset labels)")
    c.add_argument("--template", action="append", help="prompt template with {} (repeatable)")
    c.add_argument("--views", type=int, default=4, help="augmented views per image for the consistency filter")
    c.add_argument("--min_conf", type=float, default=0.5, help="a wrong answer must be at least this confident to count as f")
    c.add_argument("--test_frac", type=float, default=0.2)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", required=True)

    e = sub.add_parser("extract", help="CETT activations for the classifier")
    common(e)
    e.add_argument("--run", required=True, help="collect --out directory")
    e.add_argument("--ids", required=True, help="train_qids.json or test_qids.json")
    e.add_argument("--tower", choices=["vision", "text"], default="vision")
    e.add_argument("--out", required=True)

    v = sub.add_parser("evaluate", help="sweep scales on held-out images")
    common(v)
    v.add_argument("--run", required=True)
    v.add_argument("--ids", required=True)
    v.add_argument("--h_neurons", required=True)
    v.add_argument("--tower", choices=["vision", "text"])
    v.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    v.add_argument("--threshold", type=float, default=0.5, help="abstain below this top-1 probability")
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--out")

    x = sub.add_parser("export", help="save an edited HF checkpoint")
    x.add_argument("--model", required=True)
    x.add_argument("--h_neurons", required=True)
    x.add_argument("--tower", choices=["vision", "text"])
    x.add_argument("--scale", type=float, required=True)
    x.add_argument("--out", required=True)

    a = p.parse_args(argv)
    return {"collect": cmd_collect, "extract": cmd_extract, "evaluate": cmd_evaluate, "export": cmd_export}[a.cmd](a)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
