"""
Contextual compression: fit more of what matters into a model's context.

Two places fill a local model's context fastest:

  passages   retrieved document chunks. Most sentences in a chunk have nothing
             to do with the question. compress_passages() keeps the sentences
             that share the question's words (BM25-weighted), plus the
             sentence on either side for continuity, in their original order,
             up to a token budget. No model is needed; the result is cited
             like the full chunk.
  history    a long chat. fit_history() decides which of the oldest turns no
             longer fit; the caller either drops them ("cut") or asks the model
             for a summary of them ("summarize", summary_prompt()), which then
             stands in for those turns as one system message. The newest
             turns are always kept word for word.

Both are lossy by design: say so where the result is shown.
"""
from __future__ import annotations

import math
import re
from collections import Counter

WORD = re.compile(r"[A-Za-z0-9_]+(?:['.-][A-Za-z0-9_]+)*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=[-*•]|\d+[.)]\s)")
STOP = set("""a an the and or but if then of to in on at by for with from as is are was were be been being it its this
that these those there here what which who whom whose how why when where do does did done can could should would will
shall may might must i you he she we they me him her us them my your his our their not no yes so than too very just
about into over under again more most some any each other such only own same also""".split())


def est_tokens(text: str) -> int:
    return max(0, round(len(text or "") / 4))


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text or "") if w.lower() not in STOP]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text or "") if s and s.strip()]


def compress_passages(query: str, hits: list[dict], budget: int = 1200, count=est_tokens,
                      neighbours: int = 1, min_keep: int = 1) -> tuple[list[dict], dict]:
    """Keep the sentences of each hit that bear on the query. -> (new hits, {before, after}).

    budget is in tokens for all hits together, shared in proportion to each hit's score. A hit
    keeps at least min_keep sentences, so every cited source still says something."""
    q = Counter(words(query))
    if not hits or not q:
        n = sum(count(h.get("text", "")) for h in hits)
        return hits, {"before": n, "after": n}
    docs = [[words(s) for s in sentences(h.get("text", ""))] for h in hits]
    all_sents = [s for d in docs for s in d]
    n_s = max(1, len(all_sents))
    df = Counter(w for s in all_sents for w in set(s))
    avg = sum(len(s) for s in all_sents) / n_s or 1.0

    def bm25(s):
        tf = Counter(s)
        score = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n_s - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(s) / avg))
        return score

    before = sum(count(h.get("text", "")) for h in hits)
    weights = [max(float(h.get("score") or 0), 0.0) for h in hits]
    tot = sum(weights) or len(hits)
    weights = [w / tot if sum(weights) else 1 / len(hits) for w in weights]
    out = []
    for h, d, wgt in zip(hits, docs, weights):
        sents = sentences(h.get("text", ""))
        if not sents:
            out.append(h)
            continue
        scores = [bm25(s) for s in d]
        order = sorted(range(len(sents)), key=lambda i: -scores[i])
        keep, used, cap = set(), 0, max(40, int(budget * wgt))
        for i in order:
            if scores[i] <= 0 and len(keep) >= min_keep:
                break
            group = [j for j in range(i - neighbours, i + neighbours + 1) if 0 <= j < len(sents) and j not in keep]
            cost = sum(count(sents[j]) for j in group)
            if keep and used + cost > cap:
                continue
            keep.update(group)
            used += cost
        idx = sorted(keep)
        text = ""
        for k, j in enumerate(idx):
            text += ("" if k == 0 else (" " if j == idx[k - 1] + 1 else " … ")) + sents[j]
        if idx and idx[-1] < len(sents) - 1:
            text += " …"
        if idx and idx[0] > 0:
            text = "… " + text
        out.append({**h, "text": text, "compressed": len(idx) < len(sents),
                    "kept": f"{len(idx)}/{len(sents)} sentences"})
    return out, {"before": before, "after": sum(count(h.get("text", "")) for h in out)}


def message_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict) and p.get("type") == "text")


def fit_history(msgs: list[dict], budget: int, count=est_tokens, keep_last: int = 4) -> int:
    """How many of the oldest non-system messages must go for the rest to fit in budget tokens.

    Leading system messages always stay; so do the last keep_last messages (the newest exchange
    and the question), even if they alone do not fit. Cuts land on a user message, so the kept
    part starts with a question, not a dangling answer. -> number of messages after the system
    messages to drop (0 = everything fits)."""
    head = 0
    while head < len(msgs) and msgs[head].get("role") == "system":
        head += 1
    body = msgs[head:]
    sizes = [count(message_text(m)) + 4 for m in body]
    fixed = sum(count(message_text(m)) + 4 for m in msgs[:head])
    if fixed + sum(sizes) <= budget:
        return 0
    limit = max(0, len(body) - keep_last)
    cut, total = 0, fixed + sum(sizes)
    while cut < limit and total > budget:
        total -= sizes[cut]
        cut += 1
    while cut < limit and body[cut].get("role") != "user":
        cut += 1
    return cut


def summary_prompt(folded: list[dict], previous: str | None = None, words_max: int = 250) -> list[dict]:
    """Messages asking a model to summarize the turns that no longer fit."""
    convo = "\n\n".join(f"{m.get('role', 'user').upper()}: {message_text(m)}" for m in folded)
    if previous:
        convo = f"EARLIER SUMMARY: {previous}\n\n" + convo
    return [{"role": "system", "content": "You compress conversations so they can continue in a small context. "
                                          "Write plain prose, no preamble."},
            {"role": "user", "content": f"Summarize this conversation in at most {words_max} words. Keep every fact, "
                                        "name, number, file path, decision and open question; drop pleasantries "
                                        f"and repetition.\n\n{convo}"}]


def summary_message(text: str, n: int) -> dict:
    return {"role": "system", "content": f"Summary of the first {n} messages of this conversation, which no longer "
                                         f"fit in the context window:\n{text}"}
