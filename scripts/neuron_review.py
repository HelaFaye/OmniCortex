#!/usr/bin/env python3
"""
Neuron review: how each neuron behaves across everything a model has done,
filtered by where the data came from and what it was about, and how that
changes over time.

Every observation is one reply's per-neuron activation profile (CETT averaged
over the reply, binned like the traces) with:

  model      which model file wrote it
  source     the benchmark or test run it came from ("testqa", "livebench", …)
             or "chat-check" for replies checked in Studio's chat
  kind       "test" (TestQA), "benchmark" (anything else graded),
             or "observed" (chat replies)
  subjects   the subject classifier's labels for the prompt (code, math,
             graphics, …) or the benchmark's own subject
  verdict    right / wrong / abstained when graded, a person's label for a chat
             reply, else unknown
  risk       the hallucination classifier's probability, when there is one
  t          when the reply was written

Per neuron, for any filter of those fields, three statistics:

  association  how much more the neuron fires on wrong answers than on right
               ones: Cohen's d. Positive (red-orange) means it fires more on
               hallucinations, negative (blue) more on right answers.
  risk         its correlation with the classifier's risk, for replies nobody
               graded (chat replies)
  firing       how often it is among the most active (above the 97th
               percentile of everything selected)

Grouped by day, week, month or every N observations, these are frames: the 3D
view plays them like a trace, with the error rate over time in the strip.

    python scripts/neuron_review.py ingest-testqa --results runs/testqa.json --endpoint mine \\
        --gguf model.gguf --binary llama-cett-dump
    python scripts/neuron_review.py ingest-items --items graded.jsonl --source livebench \\
        --gguf model.gguf --binary llama-cett-dump
    python scripts/neuron_review.py summary --model acme/my-model-q4_k_m --source testqa --subject math
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_ROOT = Path(os.environ.get("OC_REVIEW_DIR") or Path.home() / ".omnicortex" / "review")
BINS = 512
VERDICTS = ("correct", "wrong", "abstained", "unknown")
KINDS = ("test", "benchmark", "observed")
STATS = ("association", "risk", "firing")
BUCKETS = ("day", "week", "month", "all")
D_SHOW = 0.2          # |d| below this is never coloured
R_SHOW = 0.15         # nor |r| below this
FIRE_PCT = 97.0


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120] or "model"


def studio_id(gguf: str) -> str:
    """The id Studio gives a model file (viz/studio.py model_id), so ingested
    runs and chat checks land under the same model."""
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", os.path.basename(gguf)[:-5] if gguf.endswith(".gguf")
                  else os.path.basename(gguf))
    return f"{os.path.basename(os.path.dirname(os.path.abspath(gguf)))}/{stem}".lower()


def bin_profile(cett: np.ndarray, bins: int = BINS) -> np.ndarray:
    from trace_sample import bin_axis
    return bin_axis(np.asarray(cett, dtype=np.float32), bins).astype(np.float16)


# ================================================================ store

class ReviewStore:
    """One folder per model: index.jsonl (metadata, one line per observation)
    and vecs/<id>.npy (float16 [layers, bins])."""

    def __init__(self, root=None):
        self.root = Path(root or DEFAULT_ROOT).expanduser()
        self.lock = threading.Lock()

    def _dir(self, model: str) -> Path:
        return self.root / _safe(model)

    def models(self) -> list[dict]:
        out = []
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                idx = d / "index.jsonl"
                if idx.exists():
                    rows = self._rows(idx)
                    if rows:
                        out.append({"model": rows[-1]["model"], "observations": len(rows),
                                    "last": max(r["t"] for r in rows)})
        return out

    @staticmethod
    def _rows(idx: Path) -> list[dict]:
        rows, labels = [], {}
        for line in idx.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("op") == "label":                  # later labels override the verdict
                labels[r["id"]] = r["verdict"]
            else:
                rows.append(r)
        for r in rows:
            if r["id"] in labels:
                r["verdict"], r["labelled"] = labels[r["id"]], True
        return rows

    def record(self, model: str, vec, *, source: str, kind: str, subjects=None, verdict: str = "unknown",
               risk=None, t=None, item=None, extra=None) -> str:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        vec = np.asarray(vec, dtype=np.float16)
        if vec.ndim != 2:
            raise ValueError("an observation is [layers, bins]")
        d = self._dir(model)
        (d / "vecs").mkdir(parents=True, exist_ok=True)
        oid = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:10]
        np.save(d / "vecs" / f"{oid}.npy", vec)
        row = {"id": oid, "model": model, "t": float(t if t is not None else time.time()), "source": str(source),
               "kind": kind, "subjects": sorted(set(subjects or [])) or ["unknown"], "verdict": verdict,
               "risk": None if risk is None else round(float(risk), 4), "item": item, **(extra or {})}
        with self.lock, open(d / "index.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return oid

    def label(self, model: str, oid: str, verdict: str) -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        idx = self._dir(model) / "index.jsonl"
        if not idx.exists() or not any(r["id"] == oid for r in self._rows(idx)):
            raise KeyError(f"no observation {oid} for {model}")
        with self.lock, open(idx, "a") as f:
            f.write(json.dumps({"op": "label", "id": oid, "verdict": verdict, "t": time.time()}) + "\n")

    def query(self, model: str, sources=None, kinds=None, subjects=None, verdicts=None,
              since=None, until=None) -> list[dict]:
        idx = self._dir(model) / "index.jsonl"
        if not idx.exists():
            return []
        out = []
        for r in self._rows(idx):
            if sources and r["source"] not in sources:
                continue
            if kinds and r["kind"] not in kinds:
                continue
            if subjects and not set(r["subjects"]) & set(subjects):
                continue
            if verdicts and r["verdict"] not in verdicts:
                continue
            if since and r["t"] < since:
                continue
            if until and r["t"] > until:
                continue
            out.append(r)
        out.sort(key=lambda r: r["t"])
        return out

    def vecs(self, model: str, rows: list[dict]) -> np.ndarray:
        d = self._dir(model) / "vecs"
        arrs = [np.load(d / f"{r['id']}.npy").astype(np.float32) for r in rows]
        shapes = {a.shape for a in arrs}
        if len(shapes) > 1:
            raise ValueError(f"observations of {model} have different shapes {shapes}: mixed model files?")
        return np.stack(arrs) if arrs else np.zeros((0, 0, 0), np.float32)

    def facets(self, model: str) -> dict:
        """What the filters can offer for this model, with counts."""
        rows = self.query(model)
        src, subj, verd = {}, {}, {}
        for r in rows:
            s = src.setdefault(r["source"], {"kind": r["kind"], "n": 0})
            s["n"] += 1
            for x in r["subjects"]:
                subj[x] = subj.get(x, 0) + 1
            verd[r["verdict"]] = verd.get(r["verdict"], 0) + 1
        return {"sources": src, "subjects": dict(sorted(subj.items(), key=lambda kv: -kv[1])), "verdicts": verd,
                "n": len(rows), "first": rows[0]["t"] if rows else None, "last": rows[-1]["t"] if rows else None}


# ================================================================ statistics

def _z(cells: int, df: int, alpha: float = 0.05) -> float:
    """Two-sided critical value over `cells` tests (Bonferroni), from a t
    distribution with `df` degrees of freedom (first Cornish-Fisher term):
    small buckets have heavier tails than the normal."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / (2 * max(1, cells)))
    return z * (1 + (z * z + 1) / (4 * max(1, df)))


def neuron_stats(X: np.ndarray, rows: list[dict], stat: str = "association") -> dict:
    """X [n, L, B] -> {value [L, B], floor, n_wrong, n_right, n_risk, note}.

    floor: the smallest |value| worth showing for this many replies and this
    many neurons (and never below D_SHOW / R_SHOW): z standard errors, with z
    from a Bonferroni correction over every cell (about 4.5 for 28 x 512), so
    that across the whole grid, chance lights up a cell in fewer than 1 in 20
    reviews. With 10 right and 10 wrong answers that takes |d| above 2."""
    n = len(rows)
    out = {"n": n, "n_wrong": 0, "n_right": 0, "n_risk": 0, "floor": 0.0}
    if n == 0:
        out.update(value=None, note="no observations match")
        return out
    if stat == "association":
        w = np.array([r["verdict"] == "wrong" for r in rows])
        c = np.array([r["verdict"] == "correct" for r in rows])
        out.update(n_wrong=int(w.sum()), n_right=int(c.sum()))
        if w.sum() < 2 or c.sum() < 2:
            out.update(value=np.zeros(X.shape[1:], np.float32),
                       note=f"needs at least 2 right and 2 wrong answers (have {int(c.sum())} and {int(w.sum())})")
            return out
        a, b = X[w], X[c]
        pooled = np.sqrt(((a.var(0, ddof=1) * (len(a) - 1)) + (b.var(0, ddof=1) * (len(b) - 1))) /
                         (len(a) + len(b) - 2))
        d = (a.mean(0) - b.mean(0)) / np.maximum(pooled, 1e-6)
        # A neuron that never moves has no effect, whatever the arithmetic says.
        d[pooled < 1e-5] = 0.0
        out.update(value=d.astype(np.float32), note="",
                   floor=round(max(D_SHOW, _z(d.size, len(a) + len(b) - 2) * float(np.sqrt(1 / len(a) + 1 / len(b)))), 3))
    elif stat == "risk":
        idx = [i for i, r in enumerate(rows) if r.get("risk") is not None]
        out["n_risk"] = len(idx)
        if len(idx) < 3:
            out.update(value=np.zeros(X.shape[1:], np.float32),
                       note=f"needs at least 3 replies with a risk score (have {len(idx)})")
            return out
        Y = np.array([rows[i]["risk"] for i in idx], np.float32)
        Z = X[idx]
        zc = Z - Z.mean(0)
        yc = (Y - Y.mean())[:, None, None]
        den = np.sqrt((zc ** 2).sum(0) * (yc ** 2).sum())
        r = (zc * yc).sum(0) / np.maximum(den, 1e-9)
        r[den < 1e-9] = 0.0
        out.update(value=r.astype(np.float32), note="",
                   floor=round(min(0.95, max(R_SHOW, _z(r.size, len(idx) - 2) / np.sqrt(len(idx)))), 3))
    elif stat == "firing":
        thr = float(np.percentile(X, FIRE_PCT)) if X.size else 0.0
        out.update(value=(X > thr).mean(0).astype(np.float32), note=f"active above {thr:.3g}", floor=0.05)
    else:
        raise ValueError(f"stat must be one of {STATS}")
    return out


def bucket_key(t: float, how: str) -> str:
    lt = time.localtime(t)
    if how == "day":
        return time.strftime("%Y-%m-%d", lt)
    if how == "week":
        return time.strftime("%G-W%V", lt)
    if how == "month":
        return time.strftime("%Y-%m", lt)
    return "all"


def buckets(rows: list[dict], how: str = "week", every: int = 0) -> list[tuple[str, list[int]]]:
    """-> [(label, row indices)] in time order. every > 0: every N observations."""
    if every and every > 0:
        return [(f"#{i + 1}–{min(i + every, len(rows))}", list(range(i, min(i + every, len(rows)))))
                for i in range(0, len(rows), every)]
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(bucket_key(r["t"], how), []).append(i)
    return list(groups.items())


def error_rate(rows: list[dict]) -> float | None:
    g = [r for r in rows if r["verdict"] in ("correct", "wrong", "abstained")]
    return (sum(r["verdict"] == "wrong" for r in g) / len(g)) if g else None


def summary(store: ReviewStore, model: str, stat: str = "association", how: str = "week", every: int = 0,
            top: int = 25, **filters) -> dict:
    rows = store.query(model, **filters)
    X = store.vecs(model, rows) if rows else None
    overall = neuron_stats(X, rows, stat) if rows else neuron_stats(np.zeros((0, 1, 1)), rows, stat)
    v = overall["value"]
    top_rows = []
    if v is not None and v.size:
        # Both directions: the strongest hallucination neurons and the strongest
        # right-answer neurons, so one cluster cannot crowd out the other.
        flat, B = v.ravel(), v.shape[1]
        keep = np.abs(flat) >= max(overall["floor"], 1e-9)
        pos = [k for k in np.argsort(-flat) if keep[k] and flat[k] > 0][:top]
        neg = [k for k in np.argsort(flat) if keep[k] and flat[k] < 0][:top]
        if stat == "firing":
            pos, neg = pos[:top], []
        else:
            half = max(top // 2, top - len(neg))
            pos, neg = pos[:half], neg[:top - min(len(pos), half)]
        for k in sorted(pos + neg, key=lambda k: -abs(flat[k])):
            l, b = divmod(int(k), B)
            top_rows.append({"layer": l, "bin": b, "value": round(float(v[l, b]), 3),
                             "mean": round(float(X[:, l, b].mean()), 4)})
    timeline = []
    for label, idx in buckets(rows, how, every):
        sub = [rows[i] for i in idx]
        timeline.append({"label": label, "n": len(sub), "error_rate": error_rate(sub),
                         "mean_risk": (round(float(np.mean([r["risk"] for r in sub if r.get("risk") is not None])), 3)
                                       if any(r.get("risk") is not None for r in sub) else None)})
    return {"model": model, "stat": stat, "n": len(rows), "note": overall["note"], "floor": overall["floor"],
            "n_wrong": overall["n_wrong"], "n_right": overall["n_right"], "n_risk": overall["n_risk"],
            "grid": None if v is None else np.round(v, 3).tolist(), "top": top_rows, "timeline": timeline,
            "error_rate": error_rate(rows)}


def subject_map(store: ReviewStore, model: str, top: int = 400, min_obs: int = 3) -> dict:
    """Which neurons go with which subject: per subject, the cells (layer, bin) whose
    activation is higher on that subject's observations than on the others' by more
    than the review's significance floor (Cohen's d, Bonferroni over the grid).
    Each cell belongs to the subject it separates most. -> {subjects, cells:
    {subject: [[layer, bin, d], ...]}, bins, layers, counts}."""
    rows = store.query(model)
    if not rows:
        return {"subjects": [], "cells": {}, "counts": {}}
    X = store.vecs(model, rows)                                    # [n, L, B]
    n, L, B = X.shape
    best_d = np.zeros((L, B), np.float32)
    best_s = np.full((L, B), -1, np.int32)
    counts, names = {}, []
    for s in sorted({x for r in rows for x in r["subjects"] if x != "unknown"}):
        idx = np.array([s in r["subjects"] for r in rows])
        if idx.sum() < min_obs or (~idx).sum() < min_obs:
            continue
        a, b = X[idx], X[~idx]
        pooled = np.sqrt(((a.var(0, ddof=1) * (len(a) - 1)) + (b.var(0, ddof=1) * (len(b) - 1))) / (len(a) + len(b) - 2))
        d = (a.mean(0) - b.mean(0)) / np.maximum(pooled, 1e-6)
        d[pooled < 1e-5] = 0.0
        floor = max(D_SHOW, _z(d.size, len(a) + len(b) - 2) * float(np.sqrt(1 / len(a) + 1 / len(b))))
        d = np.where(d >= floor, d, 0.0)
        if top and (d > 0).sum() > top:
            cut = np.sort(d.ravel())[-top]
            d = np.where(d >= cut, d, 0.0)
        k = len(names)
        names.append(s)
        counts[s] = int(idx.sum())
        better = d > best_d
        best_d[better], best_s[better] = d[better], k
    cells = {s: [] for s in names}
    for l, bn in zip(*np.nonzero(best_s >= 0)):
        cells[names[best_s[l, bn]]].append([int(l), int(bn), round(float(best_d[l, bn]), 3)])
    return {"subjects": [s for s in names if cells[s]], "cells": {s: c for s, c in cells.items() if c},
            "bins": B, "layers": L, "counts": counts}


def deficit_layers(store: ReviewStore, model: str, subjects=None, top: int = 8, **filters) -> dict:
    """The layers where wrong answers differ most from right ones, for focusing a
    fine-tune there (finetune.py --layers plan).

    Per layer, the evidence is the association (Cohen's d, wrong minus right)
    above the review's significance floor, summed over the layer's bins:
    only cells that clear the Bonferroni-corrected floor count, so with few
    observations nothing is selected rather than noise. Layers are those of
    the model the review recorded; the Hugging Face checkpoint being trained
    has the same layer count when it is the same architecture."""
    rows = store.query(model, subjects=subjects or None, **filters)
    X = store.vecs(model, rows) if rows else None
    st = neuron_stats(X, rows, "association") if rows else neuron_stats(np.zeros((0, 1, 1)), rows, "association")
    out = {"model": model, "subjects": list(subjects or []), "n_wrong": st["n_wrong"], "n_right": st["n_right"],
           "floor": st["floor"], "layers": [], "evidence": [], "note": st["note"]}
    v = st["value"]
    if v is None or not v.size or st["note"]:
        out["note"] = out["note"] or "no observations"
        return out
    mass = np.clip(v - st["floor"], 0, None).sum(axis=1)
    cells = (v >= st["floor"]).sum(axis=1)
    order = [int(i) for i in np.argsort(-mass) if mass[i] > 0][:max(1, top)]
    out["layers"] = sorted(order)
    out["evidence"] = [{"layer": i, "mass": round(float(mass[i]), 3), "cells": int(cells[i])} for i in order]
    out["n_layers"] = int(v.shape[0])
    if not order:
        out["note"] = (f"no layer has bins above the significance floor ({st['floor']}) with "
                       f"{st['n_wrong']} wrong and {st['n_right']} right answers; record more checked replies")
    return out


def view_payload(store: ReviewStore, model: str, stat: str = "association", how: str = "week", every: int = 0,
                 max_cells: int = 40000, **filters) -> tuple[bytes, dict]:
    """The 3D views' payload (viz/bloom.py format): one frame per time bucket,
    every neuron that shows an effect in any bucket. State 2 (red-orange):
    fires more on hallucinations (or with risk); state 1 (blue): fires more on
    right answers (or against risk), or simply often for 'firing'."""
    rows = store.query(model, **filters)
    if not rows:
        raise ValueError("no observations match these filters")
    X = store.vecs(model, rows)
    groups = buckets(rows, how, every)
    frames, floors, labels, err = [], [], [], []
    for label, idx in groups:
        st = neuron_stats(X[idx], [rows[i] for i in idx], stat)
        frames.append(st["value"])
        floors.append(st["floor"] if st["note"] == "" or stat == "firing" else np.inf)
        labels.append(f"{label} · {len(idx)}   ")
        err.append(error_rate([rows[i] for i in idx]))
    F = np.stack(frames)                                  # [T, L, B]
    T, L, B = F.shape
    show = np.abs(F) >= np.array(floors, np.float32)[:, None, None]
    scale = {"association": 1.5, "risk": 0.6, "firing": 1.0}[stat]   # full brightness
    ever = show.any(axis=0)
    ly, nx = np.nonzero(ever)
    if len(ly) > max_cells:
        keep = np.argsort(-np.abs(F[:, ly, nx]).max(axis=0))[:max_cells]
        ly, nx = ly[keep], nx[keep]
    vals = F[:, ly, nx]
    inten = np.clip(np.abs(vals) / scale, 0, 1).astype(np.float32)
    state = np.zeros(vals.shape, np.uint8)
    on = show[:, ly, nx]
    if stat == "firing":
        state[on] = 1
    else:
        state[on & (vals > 0)] = 2
        state[on & (vals < 0)] = 1
    blob = bytearray(struct.pack("<iii", T, len(ly), L))
    blob += ly.astype("<i4").tobytes() + nx.astype("<i4").tobytes() + inten.tobytes() + state.tobytes()
    known = [e for e in err if e is not None]
    meta = {"frames": T, "cells": int(len(ly)), "layers": L, "neurons": B, "labels": labels,
            "prob": [e if e is not None else 0.0 for e in err] if known else None,
            "threshold": round(float(np.mean(known)), 3) if known else None,
            "flagged": [], "mode": "review", "relative_z": None, "order": "index", "h_band": 0,
            "z": [0.0] * T, "model": model, "verdict": None,
            "question": f"{len(rows)} replies · {stat} by {'every ' + str(every) if every else how}"
                        + ("" if stat == "firing" else " · shown above the noise floor for each bucket's size"),
            "cells_note": {"association": "Cohen's d, wrong vs right answers",
                           "risk": "correlation with the classifier's risk",
                           "firing": f"share of replies above the {FIRE_PCT:.0f}th percentile"}[stat],
            "legend": ({"active": "fires more on right answers", "halluc": "fires more on hallucinations"}
                       if stat == "association" else
                       {"active": "fires against risk", "halluc": "fires with risk"} if stat == "risk" else
                       {"active": "fires often", "halluc": ""}),
            "strip": "error rate over time (dashed: average)" if known else "no graded replies in range"}
    return bytes(blob), meta


# ================================================================ ingest

def activation_stats(store: ReviewStore, model: str, max_obs: int = 500, cols: int = 128, **filters) -> dict:
    """The distribution of recorded activations (CETT profiles) for the stat plots on /work: per layer a
    five-number summary and mean over every observation and bin, a layer x bin heatmap of the mean
    (cols wide), wrong minus correct where both exist, and a histogram of every value."""
    rows = store.query(model, **filters)[-max_obs:]
    if not rows:
        return {"model": model, "n": 0}
    X = store.vecs(model, rows)                                  # [n, layers, bins]
    n, L, B = X.shape
    flat = X.transpose(1, 0, 2).reshape(L, -1)
    q = np.percentile(flat, [5, 25, 50, 75, 95], axis=1)
    cols = max(1, min(cols, B))

    def squeeze(M):                                              # [L, B] -> [L, cols]
        return M[:, : B - B % cols].reshape(L, cols, -1).mean(axis=2)

    out = {"model": model, "n": n, "layers": L, "bins": B, "first": rows[0]["t"], "last": rows[-1]["t"],
           "per_layer": {"p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
                         "mean": flat.mean(axis=1), "max": flat.max(axis=1)},
           "heatmap": squeeze(X.mean(axis=0))}
    verdict = np.array([r["verdict"] for r in rows])
    if (verdict == "wrong").any() and (verdict == "correct").any():
        out["wrong_minus_correct"] = squeeze(X[verdict == "wrong"].mean(axis=0) - X[verdict == "correct"].mean(axis=0))
        out["n_wrong"], out["n_correct"] = int((verdict == "wrong").sum()), int((verdict == "correct").sum())
    lo, hi = float(np.percentile(X, 0.5)), float(np.percentile(X, 99.5))
    counts, edges = np.histogram(np.clip(X, lo, hi), bins=40, range=(lo, hi + 1e-9))
    out["histogram"] = {"counts": counts, "edges": edges}
    def plain(v):
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, np.ndarray):
            return np.round(v.astype(np.float64), 5).tolist()
        return v
    return plain(out)


def _subjects(text: str, given=None) -> list[str]:
    if given:
        return [given] if isinstance(given, str) else list(given)
    try:
        import subject_classifier as sc
        a = sc.default_classifier().analyze(text)
        return a["labels"] or ["unknown"]
    except Exception:
        return ["unknown"]


def _scorer(a):
    from hscore import HScorer
    binary = a.binary or os.environ.get("OC_CETT")
    gguf = a.gguf or os.environ.get("OC_GGUF")
    if not binary or not gguf:
        raise SystemExit("need --binary (llama-cett-dump) and --gguf, or $OC_CETT and $OC_GGUF")
    return HScorer(binary, gguf, a.classifier, ngl=a.ngl)


def _ingest(store, sc, model, items, source, kind, t, log=print):
    ok = bad = 0
    for it in items:
        try:
            p = sc.profile([{"role": "user", "content": it["prompt"]}], it["response"])
        except Exception as e:
            bad += 1
            log(f"  skipped {it.get('id', '?')}: {e}")
            continue
        store.record(model, bin_profile(p["cett"]), source=source, kind=kind,
                     subjects=_subjects(it["prompt"], it.get("subject")), verdict=it["verdict"],
                     risk=p.get("prob"), t=it.get("t") or t, item=it.get("id"))
        ok += 1
        if ok % 10 == 0:
            log(f"  {ok} done")
    log(f"recorded {ok} observation(s) for {model} from {source}" + (f"; {bad} skipped" if bad else ""))
    return ok


VERDICT_MAP = {"correct": "correct", "right": "correct", "pass": "correct", "passed": "correct", "true": "correct",
               "wrong": "wrong", "incorrect": "wrong", "hallucinated": "wrong", "fail": "wrong", "failed": "wrong",
               "false": "wrong", "abstained": "abstained", "abstain": "abstained"}


def testqa_items(results_path: str, endpoint: str | None = None, bank=None) -> tuple[list[dict], str]:
    """TestQA results -> graded text items (prompt, response, verdict, subject)."""
    import testqa as tq
    d = json.loads(Path(results_path).read_text())
    raw = d.get("raw") or {}
    if not raw:
        raise SystemExit(f"{results_path} has no per-item results (raw)")
    label = endpoint or next(iter(raw))
    if label not in raw:
        raise SystemExit(f"no endpoint {label!r} in {results_path}; it has {', '.join(raw)}")
    tasks = {t["id"]: t for t in tq.load_tasks(bank or [str(HERE.parent / "qa" / "bank")], None, None, 0)}
    out = []
    for r in raw[label]:
        v = VERDICT_MAP.get(str(r.get("verdict")).lower())
        t = tasks.get(r.get("id"))
        if not v or t is None or t.get("image") or not r.get("text"):
            continue           # canaries, errors, vision items (need the image), unknown ids
        out.append({"id": r["id"], "prompt": tq.prompt_for(t), "response": r["text"], "verdict": v,
                    "subject": r.get("subject") or t.get("subject")})
    model = (d.get("endpoints") or {}).get(label, {}).get("model") or label
    return out, model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(DEFAULT_ROOT), help="review store (default ~/.omnicortex/review)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest-testqa", "ingest-items"):
        q = sub.add_parser(name, parents=[common])
        q.add_argument("--gguf", help="the model that wrote the replies (default $OC_GGUF)")
        q.add_argument("--binary", help="llama-cett-dump (default $OC_CETT)")
        q.add_argument("--classifier", help="classifier.npz, to record each reply's risk too")
        q.add_argument("--model-id", help="the model's id in Studio (default: Studio's id for --gguf)")
        q.add_argument("--ngl", type=int, default=99)
        q.add_argument("--source", help="name for this run, shown in the filters")
        q.add_argument("--kind", choices=KINDS, help="test, benchmark or observed")
        q.add_argument("--limit", type=int, default=0)
    sub.choices["ingest-testqa"].add_argument("--results", required=True, help="testqa.py --out file")
    sub.choices["ingest-testqa"].add_argument("--endpoint", help="which endpoint's replies (default: the first)")
    sub.choices["ingest-items"].add_argument("--items", required=True,
                                             help="JSONL: prompt, response, verdict (right/wrong/abstained), "
                                                  "optional subject, id, t")
    s = sub.add_parser("summary", help="the strongest neurons for a filter", parents=[common])
    s.add_argument("--model", required=True)
    s.add_argument("--source", action="append")
    s.add_argument("--kind", action="append", choices=KINDS)
    s.add_argument("--subject", action="append")
    s.add_argument("--stat", choices=STATS, default="association")
    s.add_argument("--by", choices=BUCKETS, default="week")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--json", action="store_true")
    sub.add_parser("models", help="models with observations", parents=[common])
    a = p.parse_args(argv)
    store = ReviewStore(a.root)

    if a.cmd == "models":
        for m in store.models():
            print(f"{m['model']:<40} {m['observations']:>6} observations, last {time.strftime('%Y-%m-%d', time.localtime(m['last']))}")
        return 0
    if a.cmd == "summary":
        r = summary(store, a.model, a.stat, a.by, top=a.top, sources=a.source, kinds=a.kind, subjects=a.subject)
        if a.json:
            r.pop("grid")
            print(json.dumps(r, indent=1))
            return 0
        print(f"{a.model}: {r['n']} replies ({r['n_right']} right, {r['n_wrong']} wrong) · {a.stat}")
        if r["note"]:
            print("  " + r["note"])
        for t in r["top"]:
            print(f"  layer {t['layer']:>3} bin {t['bin']:>4}  {t['value']:+.3f}")
        for b in r["timeline"]:
            er = "" if b["error_rate"] is None else f"  error rate {b['error_rate']:.0%}"
            print(f"  {b['label']:<12} {b['n']:>5} replies{er}")
        return 0
    sc = _scorer(a)
    if a.cmd == "ingest-testqa":
        items, _ = testqa_items(a.results, a.endpoint)
        t = Path(a.results).stat().st_mtime
        source, kind = a.source or "testqa", a.kind or "test"
    else:
        items = []
        for line in Path(a.items).read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                v = VERDICT_MAP.get(str(it.get("verdict")).lower())
                if v and it.get("prompt") and it.get("response"):
                    items.append({**it, "verdict": v})
        t = Path(a.items).stat().st_mtime
        source, kind = a.source or Path(a.items).stem, a.kind or "benchmark"
    model = a.model_id or studio_id(sc.gguf)
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} graded item(s) -> {model} [{source}, {kind}]")
    _ingest(store, sc, model, items, source, kind, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
