#!/usr/bin/env python3
"""
Neural Reset (docs/INTEGRATED_OPTIONS.md): catch a reply stuck repeating itself while it streams, and resume it
past the loop instead of making you cancel and retry.

What it really does, since llama-server exposes no way to edit activations mid-generation:

  1. detect   the reply's tail is one passage repeated (--min-repeats times, at least --min-span characters), or
              the same line keeps coming back
  2. cut      generation stops; the reply is kept up to the end of the passage's first copy
  3. resume   the same request again with the kept text as the start of the assistant's reply (llama-server
              continues a trailing assistant message), the loop's opening words pushed down with logit_bias,
              and the DRY sampler plus a repeat penalty for the rest of this reply only

Studio's chat page is told to trim what it already showed. Other API clients never see a reset (the guard is
for Studio's own chat), because a client that has received text cannot be asked to forget it.
"""
from __future__ import annotations

import re


def detect(text: str, min_unit: int = 8, max_unit: int = 400, min_repeats: int = 3, min_span: int = 60,
           window: int = 4000) -> dict | None:
    """-> {"start", "unit", "repeats"} when text ends in a repeated passage (start = where the repeats begin)."""
    tail = text[-window:]
    off = len(text) - len(tail)
    n = len(tail)
    for L in range(min_unit, min(max_unit, n // min_repeats) + 1):
        unit = tail[n - L:]
        if not unit.strip():
            continue
        r = 1
        while n - (r + 1) * L >= 0 and tail[n - (r + 1) * L:n - r * L] == unit:
            r += 1
        if r >= min_repeats and r * L >= min_span:
            return {"start": off + n - r * L, "unit": unit, "repeats": r}
    lines = [x for x in tail.split("\n")]
    if len(lines) >= 6:
        last = [x.strip() for x in lines[-13:-1] if x.strip()]       # complete lines only
        if last:
            top = max(set(last), key=last.count)
            if len(top) >= 6 and last.count(top) >= 4:
                first = text.find(top, off)
                again = text.find(top, first + len(top))
                return {"start": again if again > 0 else first, "unit": top, "repeats": last.count(top), "lines": True}
    return None


def keep_upto(text: str, hit: dict) -> str:
    """The reply up to the end of the passage's first copy."""
    if hit.get("lines"):
        return text[:hit["start"]].rstrip() + "\n"
    return text[:hit["start"] + len(hit["unit"])]


def opening(unit: str, words: int = 4) -> str:
    w = re.findall(r"\S+", unit)
    return " ".join(w[:words])


def resume_payload(payload: dict, kept: str, hit: dict, bias: float = -3.0) -> dict:
    msgs = list(payload.get("messages") or [])
    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("_neural_reset"):
        msgs = msgs[:-1]
    msgs.append({"role": "assistant", "content": kept, "_neural_reset": True})
    out = {**payload, "messages": [{k: v for k, v in m.items() if k != "_neural_reset"} for m in msgs]}
    lb = list(out.get("logit_bias") or []) if isinstance(out.get("logit_bias"), list) else []
    op = opening(hit["unit"])
    if op:
        lb.append([op, bias])
    out["logit_bias"] = lb
    out.setdefault("dry_multiplier", 0.8)
    out["repeat_penalty"] = max(float(out.get("repeat_penalty") or 1.0), 1.15)
    out["frequency_penalty"] = max(float(out.get("frequency_penalty") or 0.0), 0.3)
    return out


class Guard:
    """Feed it streamed text; it says when to reset. Checks every `every` characters to stay cheap."""

    def __init__(self, max_resets: int = 2, every: int = 24, **kw):
        self.max_resets, self.every, self.kw = max_resets, every, kw
        self.resets, self._last = 0, 0

    def check(self, text: str) -> dict | None:
        if self.resets >= self.max_resets or len(text) - self._last < self.every:
            return None
        self._last = len(text)
        hit = detect(text, **self.kw)
        if hit:
            self.resets += 1
        return hit
