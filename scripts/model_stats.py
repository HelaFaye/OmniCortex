#!/usr/bin/env python3
"""
Rolling per-model performance statistics, and routing that uses them.

Three kinds of observation are recorded per model file:

    graded       a TestQA item with a verdict: correct, wrong (a hallucination:
                 answered and incorrect) or abstained. These are the only
                 observations that say how good a model is.
    activation   an H-Neuron classifier score for a real reply (hscore.py):
                 how strongly the hallucination-associated neurons fired.
    live         an ungraded reply from real traffic: whether it declined.

Everything is append-only JSONL under ~/.omnicortex/stats, one file per model
id. Summaries are *rolling*: only the most recent --window observations of each
kind count (and nothing older than --max-age-days), so a model that was edited,
re-quantized or simply re-evaluated is judged on recent evidence.

Stats are tied to a model *file*: each record carries the file size, and a
summary for a file of a different size ignores them. A re-downloaded or
re-quantized file under the same name starts again with no stats.

Routing (`rank`) only ever considers models with at least `min_graded` graded
observations. A model with no graded stats is never auto-selected; it can still
be chosen explicitly.

    python scripts/model_stats.py show
    python scripts/model_stats.py show --model qwen3-8b-gguf/qwen3-8b-q6_k
    python scripts/model_stats.py rank "Write a function that parses dates"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("OC_STATS_DIR", Path.home() / ".omnicortex" / "stats"))
GRADED = {"correct", "wrong", "abstained"}
# testqa verdicts that map onto the three graded outcomes
VERDICT_MAP = {"correct": "correct", "answered": None, "refused": None, "wrong": "wrong",
               "unparsable": "wrong", "timeout": "wrong", "abstained": "abstained"}
ABSTAIN_MARKERS = ["i don't know", "i do not know", "i'm not sure", "i am not sure", "cannot determine",
                   "can't determine", "unable to answer", "no information", "i don't have", "unsure",
                   "i cannot answer", "i can't answer"]


def model_key(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id.strip().lower())[:180] or "unknown"


def looks_abstained(text: str) -> bool:
    low = (text or "").strip().lower()
    return not low or any(m in low for m in ABSTAIN_MARKERS)


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class StatsStore:
    def __init__(self, root: str | Path | None = None, window: int = 500, max_age_days: float = 180):
        self.root = Path(root or DEFAULT_DIR).expanduser()
        self.window = window
        self.max_age = max_age_days * 86400
        self.lock = threading.Lock()

    # ------------------------------------------------------------- writing
    def _path(self, model_id: str) -> Path:
        return self.root / f"{model_key(model_id)}.jsonl"

    def record(self, model_id: str, kind: str, size: int | None = None, **fields) -> dict:
        if kind not in ("graded", "activation", "live", "reference"):
            raise ValueError(f"unknown stats kind {kind!r}")
        rec = {"t": time.time(), "model": model_id, "kind": kind, **fields}
        if size is not None:
            rec["size"] = int(size)
        if kind == "graded":
            v = VERDICT_MAP.get(rec.get("verdict"), rec.get("verdict"))
            if v not in GRADED:
                return {}
            rec["verdict"] = v
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock, open(self._path(model_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # ------------------------------------------------------------- reading
    def models(self) -> list[str]:
        out = []
        if self.root.is_dir():
            for p in sorted(self.root.glob("*.jsonl")):
                try:
                    with open(p, encoding="utf-8") as f:
                        first = f.readline()
                    out.append(json.loads(first)["model"])
                except Exception:
                    continue
        return out

    def _records(self, model_ids, size: int | None) -> list[dict]:
        ids = [model_ids] if isinstance(model_ids, str) else list(model_ids)
        lines = []
        for mid in dict.fromkeys(model_key(i) for i in ids):
            p = self.root / f"{mid}.jsonl"
            if p.exists():
                lines += p.read_text(encoding="utf-8").splitlines()
        cutoff = time.time() - self.max_age
        rows = []
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("t", 0) < cutoff:
                continue
            if size is not None and r.get("size") not in (None, size):
                continue          # observations of a different file
            rows.append(r)
        rows.sort(key=lambda r: r.get("t", 0))
        return rows

    def summary(self, model_id, size: int | None = None) -> dict:
        """Rolling summary. `model_id` may be a list of ids that all name the
        same file (its id, served alias, file name...)."""
        rows = self._records(model_id, size)
        model_id = model_id if isinstance(model_id, str) else list(model_id)[0]
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r)
        graded = by_kind["graded"][-self.window:]
        act = by_kind["activation"][-self.window:]
        live = by_kind["live"][-self.window:]

        def block(rs):
            c = Counter(r["verdict"] for r in rs)
            n = sum(c.values())
            out = {"n": n, "correct": c["correct"], "wrong": c["wrong"], "abstained": c["abstained"]}
            if n:
                out.update(accuracy=c["correct"] / n, hallucination_rate=c["wrong"] / n,
                           abstention_rate=c["abstained"] / n,
                           accuracy_ci=wilson(c["correct"], n), hallucination_ci=wilson(c["wrong"], n))
            return out

        subjects = defaultdict(list)
        for r in graded:
            subjects[r.get("subject") or "unknown"].append(r)
        out = {"model": model_id, "graded": block(graded),
               "subjects": {s: block(rs) for s, rs in sorted(subjects.items())},
               "first": rows[0]["t"] if rows else None, "last": rows[-1]["t"] if rows else None,
               "window": self.window}
        scores = [float(r["h_score"]) for r in act if r.get("h_score") is not None]
        if scores:
            s = sorted(scores)
            thr = act[-1].get("threshold", 0.0)
            out["activation"] = {"n": len(s), "mean": sum(s) / len(s), "p95": s[min(len(s) - 1, int(0.95 * len(s)))],
                                 "flagged_rate": sum(x > thr for x in s) / len(s), "threshold": thr}
        else:
            out["activation"] = {"n": 0}
        ref = {}
        for r in by_kind["reference"]:          # latest published score per benchmark
            ref[r.get("benchmark", "?")] = {"score": r.get("score"), "subject": r.get("subject"),
                                            "source": r.get("source")}
        out["reference"] = ref
        sources = Counter(r.get("source", "testqa") for r in graded)
        out["graded"]["sources"] = dict(sources)
        if live:
            ab = sum(1 for r in live if r.get("abstained"))
            out["live"] = {"n": len(live), "abstention_rate": ab / len(live)}
        else:
            out["live"] = {"n": 0}
        return out


# ------------------------------------------------------------------ routing

def borrowed(block: dict, n_eff: int) -> dict:
    """A model's overall rates standing in for a subject it was never measured
    on, with confidence bounds as wide as `n_eff` observations. Transfer
    across subjects is unverified, so it must not compete at full confidence
    with a model that has real evidence for the subject."""
    acc, wrong = block["accuracy"], block["hallucination_rate"]
    return {"n": n_eff, "accuracy_ci": wilson(acc * n_eff, n_eff),
            "hallucination_ci": wilson(wrong * n_eff, n_eff)}


def subject_utility(block: dict, hallucination_cost: float) -> float | None:
    """Pessimistic expected value of answering: lower CI of accuracy minus the
    cost-weighted upper CI of the hallucination rate. Abstaining scores 0, so a
    model that declines when unsure beats one that guesses wrong."""
    if not block.get("n"):
        return None
    return block["accuracy_ci"][0] - hallucination_cost * block["hallucination_ci"][1]


def rank(proba: dict[str, float], summaries: dict[str, dict], min_graded: int = 20,
         min_subject: int = 5, hallucination_cost: float = 1.0) -> dict:
    """Pick a model for a prompt whose subject distribution is `proba`.

    Only models with >= min_graded graded observations are eligible. Per
    subject, a model's own numbers are used when it has >= min_subject items
    in that subject; otherwise its overall numbers stand in with the
    uncertainty of only `min_subject` observations (and the pick says so)."""
    eligible, excluded = {}, {}
    for mid, s in summaries.items():
        n = s.get("graded", {}).get("n", 0)
        if n >= min_graded:
            eligible[mid] = s
        else:
            excluded[mid] = f"{n} graded observations (< {min_graded})"
    if not eligible:
        return {"model": None, "reason": "no model has enough performance stats", "excluded": excluded,
                "candidates": []}
    rows = []
    for mid, s in eligible.items():
        total, fallback = 0.0, []
        overall = subject_utility(borrowed(s["graded"], min_subject), hallucination_cost)
        if not proba:
            # Subject unknown: rank on overall numbers rather than on a guess.
            total = overall
        for subj, p in proba.items():
            b = s["subjects"].get(subj, {})
            u = subject_utility(b, hallucination_cost) if b.get("n", 0) >= min_subject else None
            if u is None:
                u = overall
                if p >= 0.2:
                    fallback.append(subj)
            total += p * u
        rows.append({"model": mid, "expected": round(total, 4), "n": s["graded"]["n"],
                     "accuracy": round(s["graded"]["accuracy"], 3),
                     "hallucination_rate": round(s["graded"]["hallucination_rate"], 3),
                     "subject_fallback": fallback})
    rows.sort(key=lambda r: -r["expected"])
    top = next(iter(proba), "unknown")
    best = rows[0]
    if not proba:
        return {"model": best["model"], "subject": "unknown", "candidates": rows, "excluded": excluded,
                "reason": f"subject unknown -> best overall model (expected {best['expected']:+.2f})"}
    why = f"subject {'+'.join(proba)} ({proba.get(top, 0):.0%} {top}) -> expected {best['expected']:+.2f}"
    if best["subject_fallback"]:
        why += f"; no {', '.join(best['subject_fallback'])} stats, used overall (discounted)"
    return {"model": best["model"], "subject": top, "reason": why, "candidates": rows, "excluded": excluded}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", default=str(DEFAULT_DIR))
    p.add_argument("--window", type=int, default=500)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--model")
    r = sub.add_parser("rank")
    r.add_argument("text")
    r.add_argument("--min-graded", type=int, default=20)
    r.add_argument("--hallucination-cost", type=float, default=1.0)
    a = p.parse_args(argv)
    store = StatsStore(a.dir, a.window)
    if a.cmd == "show":
        for m in ([a.model] if a.model else store.models()):
            print(json.dumps(store.summary(m), indent=2))
    else:
        from subject_classifier import default_classifier
        sums = {m: store.summary(m) for m in store.models()}
        print(json.dumps(rank(default_classifier().route_weights(a.text), sums, a.min_graded,
                              hallucination_cost=a.hallucination_cost), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
