#!/usr/bin/env python3
"""
Score one tuning candidate in LM Studio, for the adaptive tuner.

The adaptive tuner (scripts/adaptive_tuner.py) writes a GGUF per scale,
calls an evaluator for each, and narrows the scale range around the best
score. When the model is meant for LM Studio on another machine, the
candidate goes there with the transfer watcher (oc_transfer.py watch, or the
transfer MCP's start_watch), and this script is the evaluator:

  1. waits until LM Studio on the destination lists the candidate (the transfer
     can take minutes for a large GGUF);
  2. asks it the same TestQA items every time (a fixed, seeded, balanced sample),
     so candidates are compared on equal terms;
  3. prints one JSON line the tuner reads: {"score", "correct", "total", …}.

Nothing is deleted here; the tuner deletes underperformers only with
--auto-delete, and only once you trust these scores.

    # by hand, on one candidate:
    python scripts/eval_lmstudio.py --lmstudio http://windows-box:1234 --model models/cands/m-supp350.gguf

    # as the tuner's evaluator ({model} is the candidate the tuner just wrote):
    python scripts/adaptive_tuner.py tune --state runs/tune.json --worker local ... \\
        --evaluator "python scripts/eval_lmstudio.py --lmstudio http://windows-box:1234 --model {model}"

LM Studio needs "Just-in-time model loading" on (Developer → Server settings)
so a request for a new model loads it, and preferably "auto unload unused JIT
models" so candidates do not pile up in memory. The watcher's destination must
be inside LM Studio's models folder, two levels deep (models/<publisher>/<name>/
file.gguf), or LM Studio will not list the file.

Metrics (--metric):
  accuracy          right / graded                        (default)
  not_hallucinated  1 - wrong / graded: abstaining is fine, inventing is not
  utility           (right - cost * wrong) / graded, rescaled to 0..1 (--cost, default 1)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRADED = ("correct", "wrong", "abstained")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def stem_of(path: str) -> str:
    name = Path(path).name
    name = re.sub(r"\.gguf$", "", name, flags=re.I)
    return re.sub(r"-\d{5}-of-\d{5}$", "", name)


def match_id(ids: list[str], stem: str) -> str | None:
    """The LM Studio id for a candidate file: exact, then by normalised name.
    Ids look like "publisher/model-name" or "model-name@q4_k_m"; the file stem
    is what both are made from."""
    want = _norm(stem)
    exact = [i for i in ids if i == stem or i.split("/")[-1] == stem]
    if exact:
        return exact[0]
    hits = [i for i in ids if want and (want == _norm(i.split("/")[-1].split("@")[0]) or want in _norm(i))]
    # The shortest id containing the stem: "m-supp350" must not pick "m-supp3500".
    hits.sort(key=lambda i: (want != _norm(i.split("/")[-1].split("@")[0]), len(i)))
    return hits[0] if hits else None


def list_models(base: str, api_key: str | None, timeout: float = 15) -> list[str]:
    req = urllib.request.Request(base.rstrip("/") + "/v1/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return [m.get("id") for m in json.loads(r.read()).get("data", []) if m.get("id")]


def wait_for(base: str, stem: str, api_key: str | None, wait: float, poll: float, log=print) -> str:
    end = time.time() + wait
    last = None
    while True:
        try:
            ids = list_models(base, api_key)
            mid = match_id(ids, stem)
            if mid:
                return mid
            last = f"{len(ids)} models listed, none matching {stem!r}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = f"cannot reach LM Studio at {base}: {e}"
        if time.time() >= end:
            raise TimeoutError(f"{stem} did not appear in LM Studio within {wait:.0f}s ({last}). Is the transfer "
                               "watcher running, and is its destination inside LM Studio's models folder?")
        log(f"waiting for {stem}: {last}", file=sys.stderr)
        time.sleep(poll)


def score(rows: list[dict], metric: str, cost: float = 1.0) -> dict:
    n = {v: sum(r["verdict"] == v for r in rows) for v in GRADED}
    graded = sum(n.values())
    errors = sum(r["verdict"] == "error" for r in rows)
    if graded == 0:
        raise ValueError(f"no graded answers ({errors} request errors): the model did not answer")
    out = {"graded": graded, "right": n["correct"], "wrong": n["wrong"], "abstained": n["abstained"],
           "request_errors": errors, "metric": metric}
    if metric == "accuracy":
        out.update(score=n["correct"] / graded, correct=n["correct"], total=graded)
    elif metric == "not_hallucinated":
        ok = graded - n["wrong"]
        out.update(score=ok / graded, correct=ok, total=graded)
    elif metric == "utility":
        u = (n["correct"] - cost * n["wrong"]) / graded          # in [-cost, 1]
        out.update(score=(u + cost) / (1 + cost), cost=cost)
    else:
        raise ValueError(f"unknown metric {metric}")
    out["score"] = round(out["score"], 6)
    return out


def run_testqa(base: str, model_id: str, a, out: Path) -> list[dict]:
    cmd = [sys.executable, str(HERE / "testqa.py"), "--endpoint", f"cand={base.rstrip('/')}/v1@{model_id}",
           "--per-subject", str(a.per_subject), "--seed", str(a.seed), "--temperature", "0",
           "--request-timeout", str(a.request_timeout), "--out", str(out)]
    for s in a.subject or []:
        cmd += ["--subject", s]
    if a.tasks:
        cmd += ["--tasks", *a.tasks]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.allow_exec:
        cmd += ["--allow-exec", "--sandbox", a.sandbox]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"TestQA failed ({r.returncode}): {(r.stderr or r.stdout)[-1500:]}")
    return json.loads(out.read_text())["raw"]["cand"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lmstudio", default=os.environ.get("OC_LMSTUDIO_URL", "http://127.0.0.1:1234"),
                   help="LM Studio's server (default $OC_LMSTUDIO_URL or http://127.0.0.1:1234)")
    p.add_argument("--model", required=True, help="the candidate GGUF the tuner wrote (its file name is matched)")
    p.add_argument("--model-id", help="LM Studio's id for it, if the name does not match")
    p.add_argument("--api-key", default=os.environ.get("OC_LMSTUDIO_KEY"))
    p.add_argument("--wait", type=float, default=3600, help="seconds to wait for the transfer (default 1 h)")
    p.add_argument("--poll", type=float, default=15)
    p.add_argument("--per-subject", type=int, default=10, help="TestQA items per subject (default 10)")
    p.add_argument("--subject", action="append", help="only these subjects (repeatable)")
    p.add_argument("--tasks", nargs="+", help="your own test set (TestQA JSONL files or folders) instead of the bank")
    p.add_argument("--seed", type=int, default=0, help="the sample; keep it fixed across candidates")
    p.add_argument("--metric", choices=("accuracy", "not_hallucinated", "utility"), default="accuracy")
    p.add_argument("--cost", type=float, default=1.0, help="utility: how much a wrong answer costs")
    p.add_argument("--allow-exec", action="store_true", help="grade code tasks by running them")
    p.add_argument("--sandbox", choices=("docker", "podman", "none"), default="docker")
    p.add_argument("--request-timeout", type=float, default=300, help="per request; the first one loads the model")
    p.add_argument("--timeout", type=float, default=7200, help="the whole TestQA run")
    p.add_argument("--keep", default=os.environ.get("OC_EVAL_DIR", "runs/lmstudio-eval"),
                   help="folder for each candidate's TestQA results (default runs/lmstudio-eval)")
    a = p.parse_args(argv)
    stem = stem_of(a.model)
    t0 = time.time()
    try:
        mid = a.model_id or wait_for(a.lmstudio, stem, a.api_key, a.wait, a.poll)
        keep = Path(a.keep)
        keep.mkdir(parents=True, exist_ok=True)
        out = keep / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)}.json"
        rows = run_testqa(a.lmstudio, mid, a, out)
        res = score(rows, a.metric, a.cost)
    except (TimeoutError, RuntimeError, ValueError) as e:
        print(json.dumps({"error": str(e), "model": a.model}), flush=True)
        print(f"error: {e}", file=sys.stderr)
        return 1
    res.update(model=a.model, lmstudio_id=mid, endpoint=a.lmstudio, results=str(out), seed=a.seed,
               per_subject=a.per_subject, seconds=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
