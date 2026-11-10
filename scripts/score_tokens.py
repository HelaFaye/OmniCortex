#!/usr/bin/env python3
"""
Visualise H-Neuron activity token by token.

Produces a standalone HTML file: the response rendered with each token shaded by
its H-Neuron score, the reasoning block visually separated from the final
answer, plus a per-layer histogram of where the H-Neurons live.

The interesting question this answers, and that a sentence-level score cannot:
does the H-Neuron signal spike inside the reasoning trace, before the model
commits to a wrong answer? If it does, you have an early warning signal rather
than a post-hoc flag.

    # score responses you already collected
    python scripts/score_tokens.py \
        --model_path Qwen/Qwen3-8B \
        --classifier models/classifier.npz \
        --input_path data/consistency_samples.jsonl \
        --n 8 --gpu_mem 14GiB --out report.html

    # or generate fresh
    python scripts/score_tokens.py \
        --model_path Qwen/Qwen3-8B \
        --classifier models/classifier.npz \
        --question "Who wrote the novel Stoner?" \
        --gpu_mem 14GiB --out report.html
"""

import argparse
import html
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import (CETTManager, build_sequence, find_think_end,  # noqa: E402
                       load_model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--classifier", required=True, help="models/classifier.npz")
    p.add_argument("--input_path", help="jsonl of collected samples")
    p.add_argument("--question", help="Generate a fresh response instead")
    p.add_argument("--n", type=int, default=8, help="Samples to score from jsonl")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gpu_mem", default=None)
    p.add_argument("--cpu_mem", default="40GiB")
    p.add_argument("--out", default="report.html")
    p.add_argument("--no_trust_remote_code", action="store_true")
    p.add_argument("--capture", choices=CETTManager.MODES, default="auto",
                   help="how activations leave the GPU (oc_common.CETTManager): auto, stream, device or sync")
    return p.parse_args()


def score_sequence(model, tokenizer, mgr, coef, intercept, question, response,
                   entry_device):
    full_ids, prompt_len = build_sequence(tokenizer, question, response)
    mgr.clear()
    mgr.prepare(len(full_ids))
    with torch.inference_mode():
        model(full_ids.unsqueeze(0).to(entry_device))

    cett = mgr.cett()                       # [L, T, N]
    T = cett.shape[1]
    flat = cett.permute(1, 0, 2).reshape(T, -1).numpy()   # [T, L*N]
    scores = flat @ coef + intercept

    pieces = [tokenizer.decode([t]) for t in full_ids]
    think_end = find_think_end(pieces, prompt_len, T)
    return pieces, scores, prompt_len, think_end


CSS = """
body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:900px;margin:2rem auto;
padding:0 1rem;background:#fff;color:#1a1a1a;line-height:1.9}
h1{font-size:20px;font-weight:500}h2{font-size:16px;font-weight:500;margin-top:2rem}
.q{background:#f4f4f2;padding:.7rem .9rem;border-radius:8px;font-size:14px;
line-height:1.5;margin-bottom:.8rem}
.seg{border:1px solid #e2e2de;border-radius:8px;padding:.8rem;margin-bottom:1.4rem}
.lab{font-size:12px;color:#6b6b66;text-transform:none;margin-bottom:.4rem}
.tok{padding:1px 0;border-radius:2px;white-space:pre-wrap}
.bar{height:14px;background:#e8ecf4;display:inline-block;vertical-align:middle}
.row{font-size:13px;font-family:ui-monospace,monospace;color:#444}
.legend{font-size:13px;color:#6b6b66;margin:.6rem 0 1.4rem}
.sw{display:inline-block;width:14px;height:14px;vertical-align:-2px;
border-radius:3px;margin:0 .2rem 0 .6rem}
"""


def shade(z):
    """z is a standardised score. Cool for low, warm for high."""
    z = max(-2.5, min(2.5, z))
    if z <= 0:
        t = (z + 2.5) / 2.5
        r, g, b = int(230 + 25 * t), int(241 - 3 * t), int(251 - 5 * t)
    else:
        t = z / 2.5
        r, g, b = 255, int(238 - 100 * t), int(218 - 150 * t)
    return f"rgb({r},{g},{b})"


def render_tokens(pieces, scores, start, end, mu, sd):
    out = []
    for i in range(start, end):
        z = (scores[i] - mu) / (sd + 1e-8)
        txt = html.escape(pieces[i]).replace("\n", "<br>")
        out.append(f'<span class="tok" style="background:{shade(z)}" '
                   f'title="score {scores[i]:.3f} (z {z:.2f})">{txt}</span>')
    return "".join(out)


def layer_histogram(coef, n_layers, n_neurons):
    pos = np.where(coef > 0)[0]
    counts = np.zeros(n_layers, dtype=int)
    for flat in pos:
        counts[flat // n_neurons] += 1
    peak = max(counts.max(), 1)
    rows = []
    for layer, c in enumerate(counts):
        w = int(560 * c / peak)
        rows.append(
            f'<div class="row">L{layer:<3d}<span class="bar" '
            f'style="width:{w}px"></span> {c}</div>'
        )
    return "".join(rows)


def main():
    args = parse_args()
    blob = np.load(args.classifier)
    coef = blob["coef"].astype(np.float32)
    intercept = float(blob["intercept"])
    n_layers, n_neurons = int(blob["n_layers"]), int(blob["n_neurons"])

    model, tokenizer = load_model(
        args.model_path, args.gpu_mem, args.cpu_mem,
        trust_remote_code=not args.no_trust_remote_code,
    )
    mgr = CETTManager(model, capture=args.capture)
    entry_device = next(model.parameters()).device
    if mgr.n_layers * mgr.n_neurons != coef.shape[0]:
        raise SystemExit(
            f"classifier has {coef.shape[0]} weights but this model exposes "
            f"{mgr.n_layers * mgr.n_neurons}. Wrong model for this classifier."
        )

    cases = []
    if args.question:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.question}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(entry_device)
        mgr.pause()                         # nothing to capture while generating
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.6, top_p=0.95,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=False)
        cases.append({"question": args.question, "response": resp, "judge": "?"})
    elif args.input_path:
        with open(args.input_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                d = next(iter(rec.values()))
                cases.append(d)
                if len(cases) >= args.n:
                    break
    else:
        raise SystemExit("need --question or --input_path")

    blocks = []
    for case in cases:
        pieces, scores, plen, tend = score_sequence(
            model, tokenizer, mgr, coef, intercept,
            case["question"], case["response"], entry_device,
        )
        body = scores[plen:]
        mu, sd = float(body.mean()), float(body.std())

        think_html = (render_tokens(pieces, scores, plen, tend, mu, sd)
                      if tend > plen else
                      '<span style="color:#8a8a85">(no reasoning block)</span>')
        answer_html = render_tokens(pieces, scores, tend, len(pieces), mu, sd)

        label = case.get("judge", "?")
        blocks.append(
            f'<div class="seg"><div class="q">{html.escape(case["question"])}'
            f'<br><span style="color:#6b6b66">label: {label} &nbsp; '
            f'mean score {mu:.3f}</span></div>'
            f'<div class="lab">reasoning</div><div>{think_html}</div>'
            f'<div class="lab" style="margin-top:.8rem">final answer</div>'
            f'<div>{answer_html}</div></div>'
        )

    mgr.remove()

    doc = (
        f"<!DOCTYPE html><meta charset='utf-8'><title>H-Neuron trace</title>"
        f"<style>{CSS}</style>"
        f"<h1>H-Neuron activity by token</h1>"
        f'<div class="legend">Shading is the per-token classifier score, '
        f'standardised within each response. '
        f'<span class="sw" style="background:{shade(-2)}"></span>low '
        f'<span class="sw" style="background:{shade(2)}"></span>high. '
        f'Hover a token for its raw score.</div>'
        f"{''.join(blocks)}"
        f"<h2>H-Neurons per layer</h2>"
        f"{layer_histogram(coef, n_layers, n_neurons)}"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {args.out} ({len(cases)} responses)")


if __name__ == "__main__":
    main()
