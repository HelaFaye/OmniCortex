#!/usr/bin/env python3
"""
Semantic anchors (docs/INTEGRATED_OPTIONS.md: feature-interlocked KV compression, the realisable form).

When a chat no longer fits, Studio's "summarize" folds the oldest turns into one chronological summary. "anchor"
groups those turns by topic first: TF-IDF vectors, merged greedily by cosine similarity, so a plot point or a
module discussed in turns 3, 9 and 14 becomes one cluster. The model then writes one short anchor note per
topic, keeping every concrete detail, and the notes stand in for all the turns they cover. Repetition across
turns costs context once instead of once per mention.

Not done: compressing the KV cache itself. llama-server exposes no way to merge cache entries, so this works on
the text that goes back into the context. A transformer reading an anchor note recomputes its keys and values from
it, which is what keeps it usable.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("""a an and are as at be but by can do for from has have he her his i if in into is it its me my no not
of on or our she so than that the their them then there these they this to too was we were what when which who
will with you your yes ok okay please thanks thank would could should just also about like get got""".split())


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def _terms(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", s.lower()) if w not in STOP]


def vectors(texts: list[str]) -> list[dict[str, float]]:
    docs = [Counter(_terms(t)) for t in texts]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    out = []
    for d in docs:
        v = {w: (1 + math.log(c)) * math.log((1 + n) / (1 + df[w])) + 1e-9 for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({w: x / norm for w, x in v.items()})
    return out


def cos(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def cluster(messages: list[dict], max_clusters: int = 6, min_sim: float = 0.12) -> list[list[int]]:
    """Indices of messages grouped by topic, each group in conversation order, groups by first appearance."""
    texts = [_text(m) for m in messages]
    vec = vectors(texts)
    groups = [[i] for i in range(len(messages))]
    cents = [dict(v) for v in vec]

    def merge(a, b):
        groups[a] += groups[b]
        c = Counter(cents[a])
        for w, x in cents[b].items():
            c[w] += x
        norm = math.sqrt(sum(x * x for x in c.values())) or 1.0
        cents[a] = {w: x / norm for w, x in c.items()}
        del groups[b], cents[b]
    while len(groups) > 1:
        best, pair = -1.0, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                s = cos(cents[i], cents[j])
                if s > best:
                    best, pair = s, (i, j)
        if best < min_sim and len(groups) <= max_clusters:
            break
        merge(*pair)
    for g in groups:
        g.sort()
    return sorted(groups, key=lambda g: g[0])


def top_terms(messages: list[dict], idx: list[int], k: int = 4) -> list[str]:
    c = Counter(w for i in idx for w in _terms(_text(messages[i])))
    return [w for w, _ in c.most_common(k)]


def anchor_prompt(folded: list[dict], previous: str | None = None, words_per_topic: int = 80) -> list[dict]:
    groups = cluster(folded)
    parts = []
    for n, g in enumerate(groups, 1):
        body = "\n".join(f"{folded[i].get('role', 'user').upper()} (message {i + 1}): {_text(folded[i])}" for i in g)
        parts.append(f"TOPIC {n} ({', '.join(top_terms(folded, g))}):\n{body}")
    convo = "\n\n".join(parts)
    if previous:
        convo = f"EARLIER NOTES:\n{previous}\n\n" + convo
    return [{"role": "system", "content": "You compress conversations so they can continue in a small context. "
                                          "Write plain notes, no preamble."},
            {"role": "user", "content": f"The earlier part of a conversation is grouped by topic below. For each topic "
                                        f"write one note of at most {words_per_topic} words, starting with its topic "
                                        "in a few words and a colon. Keep every name, number, file path, code "
                                        "identifier, decision and open question; merge what was said more than once. "
                                        f"Carry the earlier notes forward.\n\n{convo}"}]
