#!/usr/bin/env python3
"""
Two experimental lab tools on Hugging Face models (docs/INTEGRATED_OPTIONS.md), built on interventions.py's hooks.

jitter     Layer-wise token jitter: how brittle each layer is. For each item (a prompt and the answer it should
           give), add Gaussian noise of --sigma x the residual stream's RMS at one layer, --samples times, and
           measure how far the answer's log-probability falls and how often the first answer token flips. A layer
           where small noise flips answers is a brittle node. Items that flip become DPO pairs (the answer chosen,
           what the jittered model said rejected) for the retraining pipeline, and the brittle layers a focus list
           (finetune.py --layers, or the H-neuron pipeline's layer filter).

hijack     Cross-layer attention hijacking, the realisable form: a later layer's state cannot feed an earlier one
           in a single forward pass, so this takes two. Pass 1 records the residual stream at --source (say 28);
           pass 2 mixes it into --at (say 12) with weight --mix, at the chosen positions, optionally only where a
           trigger direction (an SAE feature or emotion vector, .npy) is active above --threshold. Reports how the
           next-token prediction changes. Known in the literature as back-patching.

    python scripts/model_lab.py jitter --model Qwen/Qwen2.5-0.5B --items items.jsonl --sigma 0.05 --out jit.json
    python scripts/model_lab.py hijack --model Qwen/Qwen2.5-0.5B --prompt "..." --at 12 --source 20 --mix 0.5

Items: JSONL of {"prompt", "answer"}, or --from-results TestQA results (its qa / reasoning items with their
reference answers), or --from-harvest (Fix This Mistake pairs: the corrected reply is the answer).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _ids(M, text):
    return M.tok(text, add_special_tokens=False)["input_ids"]


def _score(M, prompt: str, answer: str, layer: int | None = None, noise=None):
    """(log p(answer | prompt), greedy first token) with optional noise at layer's output."""
    import interventions as IV
    torch = M.torch
    p_ids, a_ids = _ids(M, prompt), _ids(M, answer)
    if not a_ids:
        raise ValueError("empty answer")
    ids = torch.tensor([p_ids + a_ids], device=M.dev)
    with IV.Hooks(M) as H:
        if layer is not None:
            H.on("resid_post", layer, noise)
        with torch.no_grad():
            lg = M.m(ids).logits[0].float().log_softmax(-1)
    n = len(p_ids)
    lp = sum(float(lg[n - 1 + i, t]) for i, t in enumerate(a_ids))
    return lp, int(lg[n - 1].argmax()), a_ids[0]


def _continue(M, prompt: str, layer: int, noise, max_new: int = 24) -> str:
    import interventions as IV
    torch = M.torch
    ids = torch.tensor([_ids(M, prompt)], device=M.dev)
    with IV.Hooks(M) as H:
        H.on("resid_post", layer, noise)
        with torch.no_grad():
            out = M.m.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=M.tok.eos_token_id or 0)
    return M.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def jitter(M, items: list[dict], sigma: float = 0.05, samples: int = 4, layers: list[int] | None = None,
           seed: int = 0, flip_pairs: bool = True) -> dict:
    torch = M.torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    layers = layers if layers is not None else list(range(len(M.layers)))
    base = [_score(M, it["prompt"], it["answer"]) for it in items]
    per_layer, flips = [], []
    for L in layers:
        drops, flipped = [], 0
        for k, it in enumerate(items):
            lp0, g0, _ = base[k]
            for _ in range(samples):
                def noise(x):
                    rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
                    eps = torch.randn(x.shape, generator=gen).to(x.device, x.dtype)
                    return x + (sigma * rms).to(x.dtype) * eps
                lp, g, _ = _score(M, it["prompt"], it["answer"], L, noise)
                drops.append(lp0 - lp)
                if g != g0:
                    flipped += 1
                    if flip_pairs and not any(f["item"] == k and f["layer"] == L for f in flips):
                        flips.append({"item": k, "layer": L, "noise": noise})
        n = max(1, len(items) * samples)
        per_layer.append({"layer": L, "mean_logprob_drop": round(sum(drops) / max(1, len(drops)), 4),
                          "flip_rate": round(flipped / n, 4)})
    ranked = sorted(per_layer, key=lambda r: (-r["flip_rate"], -r["mean_logprob_drop"]))
    cut = max(1, len(ranked) // 5)
    pairs = []
    for f in flips:
        it = items[f["item"]]
        said = _continue(M, it["prompt"], f["layer"], f["noise"])
        if said and said.strip() != it["answer"].strip():
            pairs.append({"prompt": [{"role": "user", "content": it["prompt"]}],
                          "chosen": [{"role": "assistant", "content": it["answer"]}],
                          "rejected": [{"role": "assistant", "content": said}],
                          "meta": {"source": "jitter", "layer": f["layer"], "sigma": sigma}})
    return {"sigma": sigma, "samples": samples, "items": len(items), "layers": per_layer,
            "brittle": sorted(r["layer"] for r in ranked[:cut] if r["flip_rate"] > 0 or r["mean_logprob_drop"] > 0.5),
            "baseline_logprob": [round(b[0], 4) for b in base], "dpo": pairs}


def hijack(M, prompt: str, at: int, source: int, mix: float = 0.5, positions="last", trigger=None,
           threshold: float = 0.0, top_k: int = 5) -> dict:
    import interventions as IV
    torch = M.torch
    if not (0 <= at < len(M.layers) and 0 <= source < len(M.layers)):
        raise ValueError(f"layers are 0..{len(M.layers) - 1}")
    ids = M.encode(prompt)
    n = ids.shape[1]
    pos = IV._positions(positions, n)
    with IV.Hooks(M) as H:
        H.on("resid_post", source, key="src")
        with torch.no_grad():
            base = M.m(ids).logits[0, -1].float().softmax(-1)
        src = H.grab["src"][0].float()
    gate = None
    if trigger is not None:
        v = torch.tensor(trigger, dtype=torch.float32, device=src.device)
        v = v / v.norm().clamp_min(1e-9)

    def fn(x):
        nonlocal gate
        y = x.clone()
        for p in pos:
            h = x[0, p].float()
            act = float(h @ v) if trigger is not None else 1.0
            if trigger is not None and act <= threshold:
                continue
            gate = (gate or []) + [{"position": p, "trigger": round(act, 4) if trigger is not None else None}]
            # match the source state's scale to this layer's, then mix it in
            s = src[p] * (h.norm() / src[p].norm().clamp_min(1e-9))
            y[0, p] = ((1 - mix) * h + mix * s).to(x.dtype)
        return y
    with IV.Hooks(M) as H:
        H.on("resid_post", at, fn)
        with torch.no_grad():
            new = M.m(ids).logits[0, -1].float().softmax(-1)

    def top(pr):
        v, i = pr.topk(top_k)
        return [{"token": M.tok.decode([int(t)]), "id": int(t), "p": round(float(x), 4)} for x, t in zip(v, i)]
    kl = float((base * (base.clamp_min(1e-12).log() - new.clamp_min(1e-12).log())).sum())
    return {"at": at, "source": source, "mix": mix, "positions": pos, "applied": gate or [],
            "before": top(base), "after": top(new), "kl": round(kl, 6),
            "changed_top": int(base.argmax()) != int(new.argmax())}


def load_items(a) -> list[dict]:
    items = []
    if a.items:
        items += [json.loads(l) for l in Path(a.items).read_text().splitlines() if l.strip()]
    for r in a.from_results or []:
        import testqa
        d = json.loads(Path(r).read_text())
        tasks = {t["id"]: t for t in testqa.load_tasks([str(HERE.parent / "qa" / "bank")], None, None, 0)}
        for rows in (d.get("raw") or {}).values():
            for row in rows:
                t = tasks.get(row.get("id"))
                if t and t["kind"] in ("qa", "reasoning"):
                    items.append({"prompt": t["prompt"], "answer": str(t.get("answer") or (t.get("aliases") or [""])[0])})
    if a.from_harvest:
        import harvest
        for it in harvest.items():
            items.append({"prompt": it["context"][-1]["content"], "answer": it["chosen"]})
    seen, out = set(), []
    for it in items:
        k = (it["prompt"], it["answer"])
        if it.get("answer") and k not in seen:
            seen.add(k)
            out.append(it)
    return out[:a.limit] if a.limit else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jitter")
    h = sub.add_parser("hijack")
    for s in (j, h):
        s.add_argument("--model", required=True)
        s.add_argument("--device", default="auto")
        s.add_argument("--dtype", default="float32")
        s.add_argument("--out")
    j.add_argument("--items")
    j.add_argument("--from-results", action="append")
    j.add_argument("--from-harvest", action="store_true")
    j.add_argument("--limit", type=int, default=50)
    j.add_argument("--sigma", type=float, default=0.05)
    j.add_argument("--samples", type=int, default=4)
    j.add_argument("--layers", help="e.g. 0-7,12 (default all)")
    j.add_argument("--dpo-out", help="write the flipped items as dpo.jsonl into this folder")
    h.add_argument("--prompt", required=True)
    h.add_argument("--at", type=int, required=True)
    h.add_argument("--source", type=int, required=True)
    h.add_argument("--mix", type=float, default=0.5)
    h.add_argument("--positions", default="last")
    h.add_argument("--trigger", help=".npy direction (d_model) that must be active for the hijack to apply")
    h.add_argument("--threshold", type=float, default=0.0)
    a = ap.parse_args(argv)
    import interventions as IV
    M = IV.load(a.model, a.device, a.dtype)
    if a.cmd == "jitter":
        items = load_items(a)
        if not items:
            raise SystemExit("no items: give --items, --from-results or --from-harvest")
        import quant_mask
        res = jitter(M, items, a.sigma, a.samples, quant_mask.parse_layers(a.layers) if a.layers else None)
        if a.dpo_out:
            Path(a.dpo_out).mkdir(parents=True, exist_ok=True)
            (Path(a.dpo_out) / "dpo.jsonl").write_text("".join(json.dumps(p) + "\n" for p in res["dpo"]))
        out = {**res, "dpo": len(res["dpo"])}
    else:
        trig = None
        if a.trigger:
            import numpy as np
            trig = np.load(a.trigger).astype("float32").reshape(-1).tolist()
        out = hijack(M, a.prompt, a.at, a.source, a.mix, a.positions, trig, a.threshold)
    text = json.dumps(out, indent=1)
    (Path(a.out).write_text(text) if a.out else print(text))
    if a.cmd == "jitter":
        print(f"brittle layers: {out['brittle']}; {out['dpo']} DPO pairs from flipped items", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
