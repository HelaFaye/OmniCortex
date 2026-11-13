#!/usr/bin/env python3
"""
Logit lens over a reply: what each layer would say next, and how sure it is.

    python scripts/logit_lens.py --model org/m --prompt "Who wrote Hamlet?" --reply "Christopher Marlowe." \\
        --out runs/lens.json --html runs/lens.html
    python scripts/logit_lens.py --model org/m --prompt "Who wrote Hamlet?" --generate 40 --out runs/lens.json

For every token of the reply and every layer, the residual stream at the
position before it goes through the model's final norm and output head
(nostalgebraist's logit lens). From that:

- spread: entropy of the top 5 plus the rest, over log 6 (0 = certain,
  1 = evenly spread). The same measure Studio's confidence strip uses on the
  final layer, so the two read alike.
- settle layer: the first layer from which every later layer agrees with the
  final top-1. A token the model "knew" settles early; one assembled late, as
  confident hallucinations often are, settles near the end.
- decay layer: where the top-5 spread first drops below half of its peak, the
  top-k decay the Associated Hallucinations line of work watches.

A token is marked when it settles late (--late, a fraction of the depth) and
its final spread is still high (--spread). Read the marks as places to look,
not verdicts: the lens is crude on early layers, and tuned lenses or
per-model calibration (Studio's H-Neuron classifier) do better.

Hugging Face models (transformers). GGUF models in Studio get the final-layer
part of this live: tick *confidence* under the chat.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

LOG6 = math.log(6)


def final_norm(model):
    """The norm applied before the output head, for common decoder layouts."""
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.decoder.final_layer_norm",
                 "language_model.model.norm", "model.language_model.norm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find the model's final norm; this layout is not supported yet")


def spread_top5(logp):
    """logp [..., V] log-probabilities -> (spread over log 6, top-5 ids, top-5 probs)."""
    import torch
    top, ids = logp.topk(5, dim=-1)
    p = top.exp()
    rest = (1 - p.sum(-1)).clamp(min=0)
    h = -(p * top).sum(-1) - torch.where(rest > 1e-9, rest * rest.clamp(min=1e-12).log(), torch.zeros_like(rest))
    return h / LOG6, ids, p


def encode(tok, prompt: str | None, messages: list | None, reply: str):
    """-> (ids [n], start): the reply's tokens begin at start."""
    msgs = messages or [{"role": "user", "content": prompt or ""}]
    if getattr(tok, "chat_template", None):
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        head = "\n".join(m["content"] for m in msgs) + "\n"
    h_ids = tok(head, add_special_tokens=False)["input_ids"]
    all_ids = tok(head + reply, add_special_tokens=False)["input_ids"]
    start = 0
    while start < min(len(h_ids), len(all_ids)) and h_ids[start] == all_ids[start]:
        start += 1
    if start >= len(all_ids):
        raise ValueError("empty reply")
    if tok.bos_token_id is not None and (not all_ids or all_ids[0] != tok.bos_token_id) \
            and getattr(tok, "add_bos_token", True):
        all_ids, start = [tok.bos_token_id] + all_ids, start + 1
    return all_ids, max(1, start)


def lens(model, tok, ids: list, start: int, late: float = 0.75, spread: float = 0.55) -> dict:
    import torch
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        out = model(x, output_hidden_states=True)
        hs = out.hidden_states                      # embeddings, then each layer (the last already normed)
        norm, head = final_norm(model), model.get_output_embeddings()
        L = len(hs) - 1
        pos = torch.arange(start - 1, len(ids) - 1, device=dev)      # position t predicts token t+1
        target = torch.tensor(ids[start:], device=dev)
        S, TOP1, PCH = [], [], []
        for layer in range(1, L + 1):
            h = hs[layer][0, pos]
            if layer < L:
                h = norm(h)
            logp = torch.log_softmax(head(h).float(), dim=-1)
            s, top_ids, _ = spread_top5(logp)
            S.append(s)
            TOP1.append(top_ids[:, 0])
            PCH.append(logp.gather(-1, target[:, None])[:, 0].exp())
        S, TOP1, PCH = torch.stack(S), torch.stack(TOP1), torch.stack(PCH)      # [L, T]
    T = len(ids) - start
    final = TOP1[-1]
    agree = (TOP1 == final[None])                                                 # [L, T]
    settle = []
    for t in range(T):
        s = L
        while s > 0 and bool(agree[s - 1, t]):
            s -= 1
        settle.append(s + 1)                                                      # 1-based layer
    decay = []
    for t in range(T):
        curve = S[:, t]
        peak = float(curve.max())
        idx = (curve < peak / 2).nonzero()
        decay.append(int(idx[0]) + 1 if len(idx) else None)
    pieces = [tok.decode([i]) for i in ids[start:]]
    marked = [t for t in range(T) if settle[t] / L >= late and float(S[-1, t]) >= spread]
    return {"layers": L, "pieces": pieces, "tokens": ids[start:],
            "spread": [[round(float(v), 4) for v in row] for row in S.T],             # [T][L]
            "top1": [[tok.decode([int(i)]) for i in row] for row in TOP1.T],
            "p_chosen": [round(float(v), 4) for v in PCH[-1]],
            "settle": settle, "decay": decay, "final_spread": [round(float(v), 4) for v in S[-1]],
            "marked": marked, "late": late, "spread_threshold": spread,
            "summary": {"tokens": T, "marked": len(marked),
                        "mean_settle_fraction": round(sum(settle) / (T * L), 4),
                        "mean_final_spread": round(float(S[-1].mean()), 4)}}


def to_html(r: dict, title: str = "Logit lens") -> str:
    """A self-contained heatmap: tokens across, layers down, colour = spread; settle layers outlined."""
    T, L = len(r["pieces"]), r["layers"]
    cw, ch = 34, max(6, min(14, 600 // L))
    cells = []
    for t in range(T):
        for layer in range(L):
            v = r["spread"][t][layer]
            settled = layer + 1 >= r["settle"][t]
            cells.append(f'<rect x="{t * cw}" y="{(L - 1 - layer) * ch}" width="{cw - 1}" height="{ch - 1}" '
                         f'fill="hsl({int(140 - 140 * v)},65%,{38 + int(20 * v)}%)" opacity="{0.95 if settled else 0.55}">'
                         f'<title>layer {layer + 1} · top-1 {html.escape(repr(r["top1"][t][layer]))} · spread {v:.2f}'
                         f'</title></rect>')
        y = (L - r["settle"][t]) * ch
        cells.append(f'<line x1="{t * cw}" x2="{t * cw + cw - 1}" y1="{y + ch - 1}" y2="{y + ch - 1}" '
                     f'stroke="currentColor" stroke-width="2"/>')
    labels = "".join(f'<text x="{t * cw + cw / 2}" y="{L * ch + 14}" font-size="11" text-anchor="middle" '
                     f'fill="{"var(--no)" if t in r["marked"] else "currentColor"}">'
                     f'{html.escape(r["pieces"][t].strip()[:5] or "␣")}</text>' for t in range(T))
    s = r["summary"]
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>:root{{--bg:#fafaf9;--fg:#1c1c1a;--mut:#6b6b66;--no:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--no:#ff7a6b}}}}
body{{margin:0;padding:1rem 16px;font:14px/1.5 system-ui;background:var(--bg);color:var(--fg)}}
.wrap{{overflow-x:auto}}small{{color:var(--mut)}}</style>
<h1>{html.escape(title)}</h1>
<p>{s["tokens"]} tokens · {L} layers · <b style="color:var(--no)">{s["marked"]} marked</b> (settled after
{int(r["late"] * 100)}% of the depth with final spread ≥ {r["spread_threshold"]}) · mean settle at
{int(s["mean_settle_fraction"] * 100)}% of the depth.<br><small>Tokens across, layers up the page; green = sure,
red = spread out. The line under each column is where the final answer settled; hover a cell for that layer's
top-1. Marked tokens are red.</small></p>
<div class="wrap"><svg width="{T * cw}" height="{L * ch + 20}" role="img" aria-label="logit lens heatmap">{"".join(cells)}{labels}</svg></div>
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="HF id or folder")
    p.add_argument("--prompt")
    p.add_argument("--messages", help="JSON file with the chat before the reply")
    p.add_argument("--reply", help="the reply to look into (or --reply-file, or --generate)")
    p.add_argument("--reply-file")
    p.add_argument("--generate", type=int, default=0, help="let the model write a reply of up to N tokens (greedy)")
    p.add_argument("--late", type=float, default=0.75, help="settle after this fraction of the depth counts as late")
    p.add_argument("--spread", type=float, default=0.55, help="final spread at or above this counts as unsure")
    p.add_argument("--out", required=True, help="JSON result")
    p.add_argument("--html", help="also write a heatmap page")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    a = p.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    dtype = "auto" if a.dtype == "auto" else getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, trust_remote_code=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    messages = json.loads(Path(a.messages).read_text()) if a.messages else None
    reply = Path(a.reply_file).read_text() if a.reply_file else a.reply
    if reply is None:
        if not a.generate:
            raise SystemExit("give --reply, --reply-file or --generate N")
        ids, start = encode(tok, a.prompt, messages, " ")
        head = torch.tensor([ids[:start]], device=next(model.parameters()).device)
        with torch.no_grad():
            gen = model.generate(head, max_new_tokens=a.generate, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(gen[0, start:], skip_special_tokens=True)
    ids, start = encode(tok, a.prompt, messages, reply)
    r = lens(model, tok, ids, start, a.late, a.spread)
    r.update(model=a.model, reply=reply)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r))
    if a.html:
        Path(a.html).write_text(to_html(r, f"Logit lens: {Path(a.model).name}"))
    s = r["summary"]
    print(f"{s['tokens']} tokens, {r['layers']} layers: {s['marked']} marked; mean settle at "
          f"{s['mean_settle_fraction']:.0%} of the depth, mean final spread {s['mean_final_spread']:.2f}")
    print("marked:", " | ".join(repr(r["pieces"][t]) for t in r["marked"]) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
