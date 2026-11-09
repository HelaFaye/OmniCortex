#!/usr/bin/env python3
"""
Where the work went: every chat reply, project task and job, with the worker, model, device and computer
that did it, its tokens and its time, plus hardware samples (load, memory, temperature, power) over time
(docs/WORKERS.md#work-distribution).

    work-YYYY-MM-DD.jsonl  one line per unit of work:
        {t, kind: chat|task|job, worker, model, device, host, prompt_tokens, completion_tokens,
         seconds, prompt_ms, gen_ms, ok, error, source}
    hw-YYYY-MM-DD.jsonl    one line per sample: {t, host, devices: [{id, name, util, vram_used_mb,
                                                                       vram_total_mb, temp_c, power_w}]}

Kept for keep_days (90), then pruned. breakdown() turns a window of it into shares per device, worker,
model and kind, tokens per second (prompt and generation), busy time and utilisation, a token timeline,
and hardware curves.

    python scripts/work_log.py --since 24h         # the breakdown as JSON
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import threading
import time
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("OC_WORK_LOG") or Path.home() / ".omnicortex" / "worklog")
KINDS = ("chat", "task", "job")
_LOCK = threading.Lock()


def parse_since(s: str | float | None) -> float:
    """'24h', '7d', '30m', 'all' or seconds -> a start time."""
    if s in (None, "", "all"):
        return 0.0
    if isinstance(s, (int, float)):
        return time.time() - float(s)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhdw]?)", str(s).strip())
    if not m:
        raise ValueError("since: like 30m, 24h, 7d or all")
    mult = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[m.group(2)]
    return time.time() - float(m.group(1)) * mult


class WorkLog:
    def __init__(self, root: str | Path | None = None, keep_days: int = 90):
        self.root = Path(root or DEFAULT_ROOT).expanduser()
        self.keep_days = keep_days
        self._pruned = 0.0

    def _file(self, stream: str, t: float) -> Path:
        return self.root / f"{stream}-{time.strftime('%Y-%m-%d', time.localtime(t))}.jsonl"

    def _append(self, stream: str, row: dict) -> None:
        with _LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self._file(stream, row["t"]), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if time.time() - self._pruned > 3600:
                self._prune()

    def _prune(self) -> None:
        self._pruned = time.time()
        cut = time.strftime("%Y-%m-%d", time.localtime(time.time() - self.keep_days * 86400))
        for p in self.root.glob("*-????-??-??.jsonl"):
            if p.stem[-10:] < cut:
                p.unlink(missing_ok=True)

    def record(self, kind: str, worker: str, **kw) -> dict:
        """One unit of work. Unknown values are left out rather than guessed."""
        if kind not in KINDS:
            raise ValueError(f"kind: one of {KINDS}")
        row = {"t": float(kw.pop("t", time.time())), "kind": kind, "worker": str(worker or "?")[:120]}
        for k in ("model", "device", "host", "source", "error"):
            if kw.get(k):
                row[k] = str(kw[k])[:200]
        for k in ("prompt_tokens", "completion_tokens"):
            if kw.get(k) is not None:
                row[k] = int(kw[k])
        for k in ("seconds", "prompt_ms", "gen_ms"):
            if kw.get(k) is not None:
                row[k] = round(float(kw[k]), 3)
        row["ok"] = bool(kw.get("ok", not kw.get("error")))
        self._append("work", row)
        return row

    def sample(self, devices: list, host: str = "this computer", t: float | None = None) -> None:
        keep = ("id", "name", "util", "vram_used_mb", "vram_total_mb", "temp_c", "junction_c", "power_w")
        rows = [{k: round(d[k], 1) if isinstance(d[k], float) else d[k] for k in keep if d.get(k) is not None}
                for d in devices or [] if d.get("id")]
        if rows:
            self._append("hw", {"t": t or time.time(), "host": host, "devices": rows})

    def read(self, stream: str, since: float = 0.0) -> list[dict]:
        out = []
        first = time.strftime("%Y-%m-%d", time.localtime(since)) if since else ""
        for p in sorted(self.root.glob(f"{stream}-????-??-??.jsonl")) if self.root.is_dir() else []:
            if p.stem[-10:] < first:
                continue
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("t", 0) >= since:
                    out.append(r)
        return out


def _group(rows: list[dict], key, span: float) -> list[dict]:
    g = collections.defaultdict(lambda: {"n": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                         "seconds": 0.0, "prompt_ms": 0.0, "gen_ms": 0.0, "timed_prompt": 0,
                                         "timed_gen": 0})
    for r in rows:
        a = g[key(r)]
        a["n"] += 1
        a["failed"] += 0 if r.get("ok", True) else 1
        a["prompt_tokens"] += r.get("prompt_tokens") or 0
        a["completion_tokens"] += r.get("completion_tokens") or 0
        a["seconds"] += r.get("seconds") or 0.0
        if r.get("prompt_ms") and r.get("prompt_tokens"):
            a["prompt_ms"] += r["prompt_ms"]
            a["timed_prompt"] += r["prompt_tokens"]
        if r.get("gen_ms") and r.get("completion_tokens"):
            a["gen_ms"] += r["gen_ms"]
            a["timed_gen"] += r["completion_tokens"]
    total_n = sum(a["n"] for a in g.values()) or 1
    total_tok = sum(a["prompt_tokens"] + a["completion_tokens"] for a in g.values()) or 1
    out = []
    for k, a in g.items():
        tok = a["prompt_tokens"] + a["completion_tokens"]
        gen_s = a["gen_ms"] / 1000 if a["gen_ms"] else None
        out.append({"key": k, "n": a["n"], "failed": a["failed"], "share_work": round(a["n"] / total_n, 4),
                    "share_tokens": round(tok / total_tok, 4), "prompt_tokens": a["prompt_tokens"],
                    "completion_tokens": a["completion_tokens"], "busy_s": round(a["seconds"], 1),
                    "utilisation": round(min(1.0, a["seconds"] / span), 4) if span else None,
                    "avg_s": round(a["seconds"] / a["n"], 2) if a["n"] else None,
                    # measured by the server where it reports timings, else tokens over wall time
                    "gen_tok_s": round(a["timed_gen"] / gen_s, 1) if gen_s else (
                        round(a["completion_tokens"] / a["seconds"], 1) if a["seconds"] and a["completion_tokens"] else None),
                    "prompt_tok_s": round(a["timed_prompt"] / (a["prompt_ms"] / 1000), 1) if a["prompt_ms"] else None})
    return sorted(out, key=lambda x: (-x["n"], str(x["key"])))


def breakdown(log: WorkLog, since: str | float | None = "24h", buckets: int = 48) -> dict:
    t0 = parse_since(since)
    rows = log.read("work", t0)
    hw = log.read("hw", t0)
    now = time.time()
    start = t0 or min([r["t"] for r in rows] + [h["t"] for h in hw] + [now - 3600])
    span = max(60.0, now - start)
    dev = lambda r: f"{r.get('host') or 'this computer'} · {r.get('device') or 'unknown device'}"  # noqa: E731
    out = {"since": start, "until": now, "span_s": round(span),
           "totals": {"n": len(rows), "failed": sum(1 for r in rows if not r.get("ok", True)),
                      "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
                      "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
                      "busy_s": round(sum(r.get("seconds") or 0 for r in rows), 1)},
           "by_device": _group(rows, dev, span), "by_worker": _group(rows, lambda r: r["worker"], span),
           "by_model": _group(rows, lambda r: r.get("model") or r["worker"], span),
           "by_kind": _group(rows, lambda r: r["kind"], span),
           "by_source": _group(rows, lambda r: r.get("source") or r["kind"], span)}
    # tokens per time bucket, one series per device
    w = span / buckets
    series = collections.defaultdict(lambda: [0] * buckets)
    for r in rows:
        i = min(buckets - 1, max(0, int((r["t"] - start) / w)))
        series[dev(r)][i] += (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
    out["timeline"] = {"bucket_s": round(w, 1), "start": start, "series": dict(series)}
    # hardware: per device, its samples (thinned to at most 400 points)
    curves = collections.defaultdict(lambda: collections.defaultdict(list))
    for h in hw:
        for d in h.get("devices") or []:
            c = curves[f"{h.get('host', 'this computer')} · {d.get('name') or d['id']}"]
            c["t"].append(round(h["t"], 1))
            for k in ("util", "vram_used_mb", "vram_total_mb", "temp_c", "power_w"):
                c[k].append(d.get(k))
    hwout = {}
    for k, c in curves.items():
        step = max(1, len(c["t"]) // 400)
        vals = {x: v[::step] for x, v in c.items()}
        util = [u for u in c["util"] if u is not None]
        pw = [p for p in c["power_w"] if p is not None]
        dt = (c["t"][-1] - c["t"][0]) if len(c["t"]) > 1 else 0
        top = lambda xs: round(max(xs), 1) if xs else None  # noqa: E731
        vals["summary"] = {"avg_util": round(sum(util) / len(util), 1) if util else None,
                           "peak_temp": top([x for x in c["temp_c"] if x is not None]),
                           "peak_vram_mb": top([x for x in c["vram_used_mb"] if x is not None]),
                           "avg_power_w": round(sum(pw) / len(pw), 1) if pw else None,
                           "energy_wh": round(sum(pw) / len(pw) * dt / 3600, 1) if pw and dt else None}
        hwout[k] = vals
    out["hardware"] = hwout
    return out


def llama_timings(d: dict) -> dict:
    """prompt/completion tokens and times from an OpenAI-style reply or stream chunk (llama-server adds
    "timings"; others give "usage" only)."""
    out = {}
    u = d.get("usage") or {}
    if u.get("prompt_tokens") is not None:
        out["prompt_tokens"] = u["prompt_tokens"]
    if u.get("completion_tokens") is not None:
        out["completion_tokens"] = u["completion_tokens"]
    tm = d.get("timings") or {}
    if tm.get("prompt_n") is not None:
        out.setdefault("prompt_tokens", tm["prompt_n"])
        out["prompt_ms"] = tm.get("prompt_ms")
    if tm.get("predicted_n") is not None:
        out.setdefault("completion_tokens", tm["predicted_n"])
        out["gen_ms"] = tm.get("predicted_ms") or (
            tm["predicted_n"] / tm["predicted_per_second"] * 1000 if tm.get("predicted_per_second") else None)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None)
    p.add_argument("--since", default="24h")
    a = p.parse_args(argv)
    print(json.dumps(breakdown(WorkLog(a.root), a.since), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
