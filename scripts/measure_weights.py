#!/usr/bin/env python3
"""
How much to trust each measurement system when ranking models (docs/REVIEW.md#weighting-measurements).

A model can be measured several ways:
- TestQA by subject;
- LiveBench and SWE-bench results;
- published reference scores;
- how often its replies are flagged by H-Neuron activation;
- what people marked right or wrong in use.

No one system is best everywhere, so each one is weighted by how well it agrees with what matters:

- **With a target** (by default the verdicts people gave in use, when at least five models have
  them), a system's weight is its rank correlation with the target across models.
- **Without one**, it is the system's correlation with the mean of all the others (leave-one-out
  consensus). A system that disagrees with everyone else counts less.

A correlation from few models is shrunk toward an equal share, n / (n + 5). Negative correlations count
as zero. The combined score is the weighted mean of each system's z-score, over the systems a model has.
Every fit is appended to a history, so the weights can be watched as results arrive.

    python scripts/measure_weights.py table.json      # {model: {system: score}} -> weights and ranking
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

SHRINK = 5.0
MIN_MODELS = 3
TARGET_MIN = 5
TARGET = "user_verdicts"


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < MIN_MODELS:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def _z(table: dict, system: str) -> dict:
    vals = {m: s[system] for m, s in table.items() if s.get(system) is not None}
    if len(vals) < 2:
        return {m: 0.0 for m in vals}
    mu = sum(vals.values()) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals.values()) / (len(vals) - 1)) or 1.0
    return {m: (v - mu) / sd for m, v in vals.items()}


def fit(table: dict[str, dict[str, float]], target: str | None = TARGET) -> dict:
    """table: {model: {system: score, higher is better}}. -> {"weights", "detail", "ranking", "mode"}."""
    systems = sorted({k for s in table.values() for k, v in s.items() if v is not None})
    has_target = target in systems and sum(1 for s in table.values() if s.get(target) is not None) >= TARGET_MIN
    detail = {}
    z = {k: _z(table, k) for k in systems}
    for k in systems:
        if has_target and k == target:
            detail[k] = {"r": 1.0, "n": sum(1 for s in table.values() if s.get(k) is not None), "against": "itself"}
            continue
        pairs = []
        for m, s in table.items():
            if s.get(k) is None:
                continue
            if has_target:
                if s.get(target) is not None:
                    pairs.append((s[k], s[target]))
            else:
                others = [z[o][m] for o in systems if o != k and m in z[o]]
                if others:
                    pairs.append((s[k], sum(others) / len(others)))
        r = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
        detail[k] = {"r": None if r is None else round(r, 3), "n": len(pairs),
                     "against": target if has_target else "the other systems"}
    raw = {}
    for k, d in detail.items():
        n, r = d["n"], d["r"]
        trust = n / (n + SHRINK)
        raw[k] = (1 - trust) * 1.0 + trust * max(0.0, r if r is not None else 1.0)
    tot = sum(raw.values()) or 1.0
    weights = {k: round(v / tot, 4) for k, v in raw.items()}
    ranking = []
    for m in table:
        num = sum(weights[k] * z[k][m] for k in systems if m in z[k])
        den = sum(weights[k] for k in systems if m in z[k])
        if den:
            ranking.append({"model": m, "score": round(num / den, 3), "systems": sorted(k for k in systems if m in z[k])})
    ranking.sort(key=lambda r: -r["score"])
    return {"weights": weights, "detail": detail, "ranking": ranking, "models": len(table),
            "mode": f"against {target}" if has_target else "consensus", "updated": time.time()}


def table_from_stats(store, model_ids: list[str], min_n: int = 20, review=None) -> dict:
    """{model: {system: score}} from Studio's stats (model_stats.StatsStore): graded accuracy per source,
    1 - flagged rate from activation, and published reference scores; with review (neuron_review.ReviewStore),
    the replies people labelled right or wrong as user_verdicts."""
    out = {}
    for mid in model_ids:
        recs = store._records(mid, None)
        row, by_src = {}, {}
        for r in recs:
            if r.get("kind") == "graded" and r.get("verdict") in ("correct", "wrong", "abstained"):
                by_src.setdefault(str(r.get("source") or "testqa"), []).append(r["verdict"] == "correct")
        if review is not None:
            by_src[TARGET] = [r["verdict"] == "correct" for r in review.query(mid, verdicts=["correct", "wrong"])
                              if r.get("labelled")]
        for k, v in by_src.items():
            if len(v) >= (min_n if k != TARGET else 5):
                row[k] = sum(v) / len(v)
        s = store.summary(mid)
        if (s.get("activation") or {}).get("n", 0) >= min_n:
            row["activation"] = 1 - s["activation"]["flagged_rate"]
        for b, ref in (s.get("reference") or {}).items():
            if isinstance(ref.get("score"), (int, float)):
                row["ref:" + b] = float(ref["score"])
        if row:
            out[mid] = row
    return out


def from_board(rows: list[dict]) -> dict:
    """A scoreboard has one system per subject group; its weights say which subjects separate models."""
    table = {}
    for r in rows:
        key = f"{r.get('file')} [{r.get('variant_id')}]"
        table[key] = {f"testqa:{s}": v["accuracy"] for s, v in (r.get("per_subject") or {}).items()
                      if v.get("accuracy") is not None and v.get("n", 0) >= 10}
    out = fit({k: v for k, v in table.items() if v}, target=None)
    out.pop("ranking", None)
    return out


def remember(result: dict, path: str | Path) -> list[dict]:
    """Append to the history of fits (newest last, at most 200)."""
    p = Path(path).expanduser()
    try:
        hist = json.loads(p.read_text())
    except (OSError, ValueError):
        hist = []
    hist.append({"t": result["updated"], "weights": result["weights"], "models": result["models"], "mode": result["mode"]})
    hist = hist[-200:]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist))
    return hist


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", help="JSON {model: {system: score}}")
    p.add_argument("--target", default=TARGET)
    a = p.parse_args(argv)
    print(json.dumps(fit(json.loads(Path(a.table).read_text()), a.target), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
