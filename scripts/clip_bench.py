#!/usr/bin/env python3
"""
Benchmark CLIP/SigLIP models, and their H-Neuron-edited variants, with
LAION's CLIP_benchmark: ImageNet-1k, ImageNetV2, ImageNet-Sketch, ImageNet-A/R/O,
ObjectNet, VTAB (via Google's task_adaptation), CIFAR, and the rest of its
zero-shot classification suite.

    pip install clip_benchmark
    python scripts/clip_bench.py eval --model openai/clip-vit-base-patch32 \\
        --dataset imagenetv2 imagenet_sketch vtab/cifar100 --dataset-root ~/datasets \\
        --h_neurons models/clip/h_neurons.json --scales 1 0.5 0 --out runs/clip-bench.json

The model is any Hugging Face dual encoder (what clip_neurons.py uses), wrapped
to the open_clip interface CLIP_benchmark expects. Class names and prompt
templates come from CLIP_benchmark, and classification runs through its own
`zero_shot_classifier` / `run_classification`, so top-1/top-5 are comparable
with its published numbers for the same checkpoint. Each scale additionally
reports the selective metrics OmniCortex cares about (coverage, confident
error rate, selective accuracy) from the same logits.

Datasets: CLIP_benchmark downloads most automatically into --dataset-root.
ImageNet-1k, ImageNet-Sketch and ObjectNet need manual downloads; VTAB tasks
need `pip install task_adaptation tensorflow tensorflow-datasets`
(CLIP_benchmark builds them through task_adaptation). A local ImageFolder
works too: `--dataset imagefolder:/path/to/root`.

`export` writes N images per class of any of these datasets as an ImageFolder,
so `clip_neurons.py collect --images` can find H-Neurons on the same
distribution you benchmark on (use a different split or seed than the eval).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_neurons as cn  # noqa: E402
import clip_style as cs  # noqa: E402

DEFAULT_TEMPLATES = ["a photo of a {c}.", "a blurry photo of a {c}.", "a photo of the large {c}.",
                     "a photo of the small {c}.", "a sketch of a {c}.", "a rendition of a {c}."]


class HFClipAdapter:
    """open_clip-style encode_image / encode_text over a transformers model."""

    def __init__(self, model, proc, device):
        self.model, self.proc, self.device = model, proc, device

    def encode_image(self, pixel_values):
        import torch.nn.functional as F
        size = getattr(getattr(self.model.config, "vision_config", None), "image_size", None)
        if size and pixel_values.shape[-1] != size:
            # Some datasets hand over ready-made tensors and ignore the transform.
            pixel_values = F.interpolate(pixel_values.float(), size=(size, size), mode="bilinear", align_corners=False)
        return cn._feats(self.model.get_image_features(pixel_values=pixel_values.to(
            self.device, next(self.model.parameters()).dtype))).float()

    def encode_text(self, tok):
        return cn._feats(self.model.get_text_features(**tok)).float()

    def tokenizer(self, texts):
        enc = self.proc(text=list(texts), return_tensors="pt", padding=True)

        class Tok(dict):
            def to(self, device):
                return Tok({k: v.to(device) for k, v in self.items()})
        return Tok({k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")})

    style = "off"

    def transform(self, img):
        img = img.convert("RGB")
        if self.style != "off":
            img = cs.prepare(img, self.style, cs.target_size(self.proc))[0]
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


def build(name: str, root: str, transform, split: str):
    if name.startswith("imagefolder:"):
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(name.split(":", 1)[1], transform=transform)
        ds.classes = [c.replace("_", " ") for c in ds.classes]
        ds.templates = DEFAULT_TEMPLATES
        return ds
    from clip_benchmark.datasets.builder import build_dataset
    return build_dataset(name, root=root, transform=transform, split=split, download=True)


def evaluate_one(adapter, ds, scaler, scales, args) -> dict:
    import torch
    from clip_benchmark.metrics.zeroshot_classification import run_classification, zero_shot_classifier
    templates = getattr(ds, "templates", None) or DEFAULT_TEMPLATES
    if isinstance(templates, list):
        templates = cs.templates(templates, args.style)
        templates = [t.replace("{}", "{c}") for t in templates]
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False)
    amp = args.device == "cuda"
    out = {"n_classes": len(ds.classes), "n_images": len(ds), "scales": {}}
    for s in scales:
        if scaler is not None:
            scaler.scale = float(s)
        clf = zero_shot_classifier(adapter, adapter.tokenizer, ds.classes, templates, args.device, amp=amp)
        logits, target = run_classification(adapter, clf, loader, args.device, amp=amp)
        probs = logits.softmax(-1).numpy()
        y = target.numpy()
        top5 = logits.topk(min(5, logits.shape[1]), dim=-1).indices.numpy()
        per_class = [float((probs.argmax(1)[y == c] == c).mean()) for c in np.unique(y)]
        m = cn.selective_metrics(probs, y, args.threshold)
        m.update(top1=m["accuracy"], top5=float(np.mean([y[i] in top5[i] for i in range(len(y))])),
                 mean_per_class_recall=float(np.mean(per_class)))
        out["scales"][str(s)] = m
    return out


def cmd_eval(a) -> int:
    model, proc = cn.load(a.model, a.device, a.dtype)
    adapter = HFClipAdapter(model, proc, a.device)
    adapter.style = a.style
    scaler, tower = None, None
    scales = a.scales if a.h_neurons else [1.0]
    if a.h_neurons:
        tower = a.tower or json.loads(Path(a.h_neurons).read_text()).get("tower") or "vision"
        mgr = cn.ClipCETT(model, tower)
        mgr.remove()
        scaler = cn.Scaler(mgr, cn.load_profile(a.h_neurons, mgr.n_neurons))
    report = {"model": a.model, "h_neurons": a.h_neurons, "tower": tower, "threshold": a.threshold, "style": a.style,
              "benchmark": "CLIP_benchmark zero-shot classification", "datasets": {}}
    for name in a.dataset:
        ds = build(name, a.dataset_root, adapter.transform, a.split)
        if a.limit and len(ds) > a.limit:
            import torch
            idx = np.random.default_rng(0).choice(len(ds), a.limit, replace=False)
            sub = torch.utils.data.Subset(ds, sorted(idx.tolist()))
            sub.classes, sub.templates = ds.classes, getattr(ds, "templates", None)
            ds = sub
        print(f"{name}: {len(ds)} images, {len(ds.classes)} classes")
        report["datasets"][name] = evaluate_one(adapter, ds, scaler, scales, a)
        for s, m in report["datasets"][name]["scales"].items():
            print(f"  scale {s:>5}: top1 {m['top1']:.3f}  top5 {m['top5']:.3f}  "
                  f"conf-err {m['confident_error_rate']:.3f}  coverage {m['coverage']:.3f}")
    if scaler is not None:
        scaler.remove()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {a.out}")
    return 0


def cmd_export(a) -> int:
    from collections import defaultdict
    ds = build(a.dataset, a.dataset_root, None, a.split)
    out = Path(a.out)
    seen = defaultdict(int)
    rng = np.random.default_rng(a.seed)
    for i in rng.permutation(len(ds)):
        img, label = ds[int(i)]
        name = ds.classes[label].replace("/", "-").replace(" ", "_")
        if seen[name] >= a.per_class:
            continue
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(d / f"{int(i)}.png")
        seen[name] += 1
        if len(seen) == len(ds.classes) and all(v >= a.per_class for v in seen.values()):
            break
    print(f"wrote {sum(seen.values())} images in {len(seen)} classes to {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="zero-shot classification on CLIP_benchmark datasets")
    e.add_argument("--model", required=True)
    e.add_argument("--dataset", nargs="+", required=True,
                   help="CLIP_benchmark names (imagenetv2, imagenet_sketch, vtab/cifar100, ...) or imagefolder:PATH")
    e.add_argument("--dataset-root", default=str(Path.home() / "datasets" / "clip_benchmark"))
    e.add_argument("--split", default="test")
    e.add_argument("--h_neurons", help="profile from clip_neurons.py + classifier.py; enables --scales")
    e.add_argument("--tower", choices=["vision", "text"])
    e.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    e.add_argument("--threshold", type=float, default=0.5, help="abstain below this top-1 probability")
    e.add_argument("--style", choices=cs.STYLES, default="off",
                   help="pixel art / cartoon handling: nearest-neighbour upscaling, padding, style prompts")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--num-workers", type=int, default=4)
    e.add_argument("--limit", type=int, default=0, help="random subset of N images per dataset (quick runs)")
    e.add_argument("--device", default="cuda" if cn._cuda() else "cpu")
    e.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    e.add_argument("--out")
    x = sub.add_parser("export", help="dump N images per class as an ImageFolder for clip_neurons.py collect")
    x.add_argument("--dataset", required=True)
    x.add_argument("--dataset-root", default=str(Path.home() / "datasets" / "clip_benchmark"))
    x.add_argument("--split", default="train")
    x.add_argument("--per-class", type=int, default=5)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return cmd_eval(a) if a.cmd == "eval" else cmd_export(a)


if __name__ == "__main__":
    raise SystemExit(main())
