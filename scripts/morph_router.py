#!/usr/bin/env python3
"""
Morphing architecture, the realisable form (docs/INTEGRATED_OPTIONS.md): send simple requests to a small model and
hard ones to the full model, decided per request from the prompt itself.

complexity() scores a request 0..1 from what makes requests hard for small models: length, code (fences,
identifiers, stack traces), maths (equations, many numbers), several constraints or questions at once, reasoning
words ("prove", "derive", "step by step", "why"), and a long conversation. Each part is shown, so a routing decision
can be explained. Below the threshold the small model answers.

Not done: skipping layers inside one model. llama.cpp runs every layer it loads, and early exit only works for
models trained for it (LayerSkip and the like); a dense model with half its layers skipped mostly produces noise.
Two real models do what the idea wanted. Keep the small one where it does not evict the large one: a second
worker, a linked host, or both in memory on a large machine. A local model swap costs a reload each time.
"""
from __future__ import annotations

import re

REASON = re.compile(r"\b(prove|derive|step by step|explain why|why does|analy[sz]e|compare|trade-?offs?|design|optimi[sz]e|"
                    r"debug|refactor|architecture|algorithm|complexity|edge cases?|counterexample)\b", re.I)
CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|=>|;\s*$|Traceback|Error:|\w+\(\)|#include|SELECT .* FROM", re.I | re.M)
MATH = re.compile(r"[=<>]\s*-?\d|\d\s*[\^*/]\s*\d|\b(integral|derivative|matrix|probability|equation|solve for)\b", re.I)


def _text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else m
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def complexity(messages: list[dict]) -> dict:
    user = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    last = _text(user[-1]) if user else ""
    words = len(last.split())
    parts = {
        "length": min(1.0, words / 250),
        "code": min(1.0, len(CODE.findall(last)) / 3),
        "math": min(1.0, len(MATH.findall(last)) / 3 + len(re.findall(r"\d+(?:\.\d+)?", last)) / 30),
        "reasoning": min(1.0, len(REASON.findall(last)) / 2),
        "asks": min(1.0, max(0, last.count("?") - 1) / 3 + len(re.findall(r"^\s*(?:\d+[.)]|[-*])\s", last, re.M)) / 6),
        "history": min(1.0, max(0, len(user) - 1) / 12),
        "images": 1.0 if any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]
                                                                       if isinstance(p, dict)) for m in user) else 0.0,
    }
    w = {"length": 0.2, "code": 0.25, "math": 0.2, "reasoning": 0.2, "asks": 0.1, "history": 0.05, "images": 0.3}
    score = min(1.0, sum(parts[k] * w[k] for k in parts) + (0.35 if parts["code"] >= 0.67 or parts["reasoning"] >= 1 else 0))
    return {"score": round(score, 3), "parts": {k: round(v, 2) for k, v in parts.items()}}


def route(model: str, messages: list[dict], cfg: dict) -> tuple[str, dict | None]:
    """-> (model to use, why) when morphing applies to this request's model."""
    if not (cfg.get("enabled") and cfg.get("full_model") and cfg.get("small_model")) or model != cfg["full_model"]:
        return model, None
    c = complexity(messages)
    small = c["score"] < float(cfg.get("threshold", 0.35))
    return (cfg["small_model"] if small else model), {"morph": "small" if small else "full", **c}
