#!/usr/bin/env python3
"""
Semantic friction (the semantic-friction add-on, docs/ADDONS.md): make a model less likely to repeat itself, at the
sampler, without an extra forward pass.

Two layers:
  within a reply   llama.cpp's DRY sampler ("don't repeat yourself"): while the model is extending a sequence it
                   has already produced, the tokens that would continue the repeat are penalised, more steeply the
                   longer the repeat. The add-on turns it on (dry_multiplier, dry_base, dry_allowed_length).
  across replies   the phrases this conversation's earlier replies keep reusing ("Certainly! Here's", "I hope this
                   helps", the same transition every time) are found here; their most distinctive word gets a small
                   negative logit_bias for the next reply, so the model reaches for another way to say it.

Only distinctive words are biased (never "the", "and" or a word the user is asking about), with a small bias, so
it nudges rather than forbids. A word the user's latest message uses is never biased.
"""
from __future__ import annotations

import re
from collections import Counter

COMMON = set("""the a an and or but if of to in on at for with by from as is are was were be been being it its this that
these those i you he she we they me my your our their his her them us not no yes do does did have has had will would
can could should may might must shall here there what which who whom when where why how all any some more most
other such only own same so than too very just also then now one two about into over after before up down out""".split())


def _text(m) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def repeated_phrases(replies: list[str], n_min: int = 3, n_max: int = 6, min_replies: int = 2) -> list[tuple[str, int]]:
    """Phrases (n_min..n_max words) found in at least min_replies different replies, longest and most frequent first."""
    seen = Counter()
    for r in replies:
        words = re.findall(r"[A-Za-z']+", r)
        grams = set()
        for n in range(n_min, n_max + 1):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i:i + n])
                if sum(w.lower() not in COMMON for w in words[i:i + n]) >= 1:
                    grams.add(g)
        seen.update(grams)
    hits = [(g, c) for g, c in seen.items() if c >= min_replies]
    hits.sort(key=lambda x: (-x[1], -len(x[0].split())))
    out = []
    for g, c in hits:                                  # drop phrases contained in a longer one already kept
        if not any(g in k for k, _ in out):
            out.append((g, c))
    return out


def bias_for(messages: list[dict], max_words: int = 8, bias: float = -1.5, min_replies: int = 2) -> list[list]:
    """logit_bias entries ([[word, bias], ...]) for the next reply in this conversation."""
    replies = [_text(m) for m in messages if m.get("role") == "assistant"][-8:]
    if len(replies) < min_replies:
        return []
    user_last = next((_text(m) for m in reversed(messages) if m.get("role") == "user"), "").lower()
    asked = set(re.findall(r"[a-z']+", user_last))
    words, out = [], []
    for phrase, _ in repeated_phrases(replies, min_replies=min_replies):
        cands = [w for w in phrase.split() if w.lower() not in COMMON and w.lower() not in asked and len(w) > 3]
        if cands:
            w = max(cands, key=len)
            if w.lower() not in {x.lower() for x in words}:
                words.append(w)
        if len(words) >= max_words:
            break
    for w in words:
        out += [[w, bias], [" " + w, bias]]
    return out


def apply(payload: dict, sampling: dict) -> dict:
    """Merge a plugin sampling block (and friction) into a chat request's payload. Leaves values the request set."""
    out = dict(payload)
    for k, v in sampling.items():
        if k != "friction" and k not in out:
            out[k] = v
    fr = sampling.get("friction")
    if fr:
        lb = bias_for(out.get("messages") or [], int(fr.get("max_words", 8)), float(fr.get("bias", -1.5)),
                      int(fr.get("min_replies", 2)))
        if lb:
            cur = out.get("logit_bias") if isinstance(out.get("logit_bias"), list) else []
            out["logit_bias"] = cur + lb
    return out
