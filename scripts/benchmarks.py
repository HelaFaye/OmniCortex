#!/usr/bin/env python3
"""
External benchmarks for local models: LiveBench, SWE-bench, and published
leaderboard scores (e.g. an export from BenchLM.ai). Results can be recorded
as rolling model stats, so Studio's `auto` routing and the retraining
pipeline see them next to TestQA.

LiveBench (https://github.com/LiveBench/LiveBench). Contamination-limited,
objectively graded questions in six categories, refreshed over time.

    git clone https://github.com/LiveBench/LiveBench && pip install -e LiveBench
    python scripts/benchmarks.py livebench run --livebench LiveBench \\
        --endpoint http://127.0.0.1:7870/v1@my-model --bench live_bench/math live_bench/coding
    python scripts/benchmarks.py livebench import --livebench LiveBench --model my-model \\
        --publish-stats http://127.0.0.1:7870

SWE-bench (https://github.com/SWE-bench/SWE-bench). Resolve real GitHub issues.
Predictions are generated here (single-shot retrieval baseline: the model
reads the issue plus retrieved files and writes a patch); scoring runs in
SWE-bench's own Docker harness.

    python scripts/benchmarks.py swebench predict --endpoint http://127.0.0.1:7870/v1@my-model \\
        --dataset princeton-nlp/SWE-bench_Lite_bm25_13K --limit 50 --out runs/swe/preds.jsonl
    python scripts/benchmarks.py swebench evaluate --predictions runs/swe/preds.jsonl \\
        --dataset princeton-nlp/SWE-bench_Lite --run-id my-model-1
    python scripts/benchmarks.py swebench import --report my-model.my-model-1.json --model my-model \\
        --publish-stats http://127.0.0.1:7870

Retraining on SWE-bench deficits uses the *train* split (different
repositories from the test sets, with gold patches), never test instances:

    python scripts/benchmarks.py swebench predict --endpoint ...@my-model \\
        --dataset princeton-nlp/SWE-bench_bm25_13K --split train --limit 2000 --out runs/swe/train-preds.jsonl
    python scripts/benchmarks.py swebench train-data --dataset princeton-nlp/SWE-bench_bm25_13K \\
        --predictions runs/swe/train-preds.jsonl --report my-model.my-model-1.json \\
        --anchors runs/retrain/sft.jsonl --out runs/swe-retrain
    python scripts/finetune.py --model org/base-model --data runs/swe-retrain --method qlora --dpo \\
        --max-length 16384 --out runs/swe-adapter

Published scores (BenchLM.ai and similar leaderboards): CSV or JSON rows of
model, benchmark, score, recorded as *reference* stats. They describe the
vendor's model, not your local quant or edit, so they are shown but never
used for auto routing.

    python scripts/benchmarks.py leaderboard import --file benchlm.csv --map my-model=Qwen3-8B
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testqa as tq  # noqa: E402

LIVEBENCH_SUBJECT = {"coding": "code", "agentic_coding": "code", "agentic_coding_v2": "code", "math": "math",
                     "data_analysis": "math", "reasoning": "logic", "language": "writing",
                     "instruction_following": "writing"}
# Common leaderboard benchmark names -> OmniCortex subjects (case-insensitive substring match).
BENCHMARK_SUBJECT = [
    ("swe-bench", "code"), ("swebench", "code"), ("humaneval", "code"), ("mbpp", "code"), ("livecodebench", "code"),
    ("terminal-bench", "code"), ("aider", "code"), ("codeforces", "code"),
    ("aime", "math"), ("math", "math"), ("gsm8k", "math"), ("hmmt", "math"),
    ("gpqa", "science"), ("scicode", "science"), ("chemistry", "science"), ("physics", "science"),
    ("mmlu", "factual"), ("simpleqa", "factual"), ("triviaqa", "factual"), ("humanity", "factual"),
    ("arc", "logic"), ("bbh", "logic"), ("big-bench", "logic"), ("zebra", "logic"), ("reasoning", "logic"),
    ("ifeval", "writing"), ("arena", "writing"), ("writing", "writing"), ("instruction", "writing"),
    ("mmmu", "vision"), ("mathvista", "vision"), ("chartqa", "vision"), ("docvqa", "vision"), ("vision", "vision"),
]


def subject_for_benchmark(name: str) -> str | None:
    low = name.lower()
    for key, subj in BENCHMARK_SUBJECT:
        if key in low:
            return subj
    return None


def record(model_id: str, rows: list[dict], a, source: str) -> int:
    """Rows shaped like TestQA results (id, kind, subject, verdict)."""
    if a.record_stats is None and not a.publish_stats:
        return 0
    args = SimpleNamespace(record_stats=a.record_stats, publish_stats=a.publish_stats, api_key=a.api_key)
    for r in rows:
        r.setdefault("source", source)
    return tq.record_stats(model_id, rows, args, source=source)


def add_stats_args(p):
    p.add_argument("--record-stats", nargs="?", const="", metavar="DIR", help="append to local rolling stats")
    p.add_argument("--publish-stats", metavar="STUDIO_URL", help="send to a running Studio")
    p.add_argument("--api-key", default="")


# ---------------------------------------------------------------- LiveBench

def livebench_run(a) -> int:
    root = Path(a.livebench).resolve()
    script = root / "livebench" / "run_livebench.py"
    if not script.exists():
        raise SystemExit(f"{script} not found; git clone https://github.com/LiveBench/LiveBench and pip install -e it")
    _, url, model = tq.parse_endpoint("m=" + a.endpoint)
    model = model or a.model
    if not model:
        raise SystemExit("give the endpoint as URL@model")
    cmd = [sys.executable, str(script), "--model", model, "--api-base", url, "--bench-name", *a.bench,
           "--model-display-name", a.display_name or model, "--parallel-requests", str(a.parallel)]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.max_tokens:
        cmd += ["--max-tokens", str(a.max_tokens)]
    if a.release:
        cmd += ["--livebench-release-option", a.release]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=root / "livebench")
    if rc == 0 and (a.record_stats is not None or a.publish_stats):
        a.model = a.display_name or model
        return livebench_import(a)
    return rc


def livebench_judgments(root: Path, model: str) -> list[dict]:
    out = []
    for f in sorted((root / "livebench" / "data").rglob("model_judgment/ground_truth_judgment.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model", "").lower() == model.lower() or r.get("model", "").lower().endswith("/" + model.lower()):
                out.append(r)
    return out


def livebench_import(a) -> int:
    root = Path(a.livebench).resolve()
    js = livebench_judgments(root, a.model)
    if not js:
        raise SystemExit(f"no LiveBench judgments for {a.model!r} under {root}/livebench/data")
    by_cat = defaultdict(list)
    rows = []
    for r in js:
        cat = r.get("category") or r.get("task", "unknown")
        score = float(r.get("score", 0) or 0)
        by_cat[cat].append(score)
        subj = LIVEBENCH_SUBJECT.get(cat)
        if subj is None:
            continue
        # LiveBench scores are 0..1, some with partial credit; it has no abstention
        # notion. Full marks count as correct, anything less as wrong.
        rows.append({"id": f"livebench:{r['question_id']}", "kind": "livebench", "subject": subj,
                     "verdict": "correct" if score >= a.full_marks else "wrong"})
    print(f"LiveBench, {a.model}: {len(js)} judged questions")
    for cat, s in sorted(by_cat.items()):
        print(f"  {cat:<24} n {len(s):>4}   mean {sum(s) / len(s):.3f}   -> {LIVEBENCH_SUBJECT.get(cat, '(unmapped)')}")
    n = record(a.stats_model or a.model, rows, a, "livebench")
    if n:
        print(f"recorded {n} results as stats for {a.stats_model or a.model}")
    return 0


# ---------------------------------------------------------------- SWE-bench

PATCH_RE = [re.compile(r"<patch>\s*(.*?)\s*</patch>", re.S), re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.S)]
SWE_INSTRUCTIONS = ("You will be given a GitHub issue and parts of the repository. Write a patch that resolves "
                    "the issue. Reply with a single unified diff (as produced by `git diff`) inside <patch></patch> "
                    "tags and nothing else.\n\n")


def extract_patch(text: str) -> str:
    body = tq.strip_think(text)
    for rx in PATCH_RE:
        m = rx.search(body)
        if m:
            body = m.group(1)
            break
    body = body.strip("\n")
    if not re.search(r"^(diff --git |--- )", body, re.M):
        return ""
    return body + "\n"


def load_rows(spec: str, split: str) -> list[dict]:
    """A local file or a Hub dataset; works without the `datasets` package (hf_rows.py)."""
    import hf_rows
    return hf_rows.load(spec, split)


def swebench_predict(a) -> int:
    label, url, model = tq.parse_endpoint("m=" + a.endpoint)
    rows = load_rows(a.dataset, a.split)
    if a.instance_ids:
        rows = [r for r in rows if r["instance_id"] in set(a.instance_ids)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["instance_id"] for l in out.read_text().splitlines() if l.strip()}
    ask_args = SimpleNamespace(temperature=0.0, max_tokens=a.max_tokens, api_key=a.api_key,
                               request_timeout=a.request_timeout)
    name = a.model_name or model or label
    n_patch = 0
    todo = [r for r in rows if r["instance_id"] not in done]
    print(f"{len(rows)} instances, {len(rows) - len(todo)} already answered in {out}; asking {name} for {len(todo)}",
          flush=True)
    for i, r in enumerate(todo):
        print(f"PROGRESS {i}/{len(todo)} {r['instance_id']}", flush=True)
        # *_bm25_* and *_oracle variants carry a ready prompt with retrieved code in "text";
        # the plain datasets only have the issue, which rarely suffices.
        prompt = r.get("text") or (SWE_INSTRUCTIONS + "Repository: " + r["repo"] + "\n\nIssue:\n" + r["problem_statement"])
        try:
            reply = tq.ask(url, model, prompt, ask_args)
        except Exception as e:
            print(f"{r['instance_id']}: request failed: {e}", file=sys.stderr)
            continue
        patch = extract_patch(reply)
        n_patch += bool(patch)
        with out.open("a") as f:
            f.write(json.dumps({"instance_id": r["instance_id"], "model_name_or_path": name,
                                "model_patch": patch}) + "\n")
        print(f"{r['instance_id']}: {'patch' if patch else 'no patch'} ({len(reply)} characters of reply)", flush=True)
    print(f"PROGRESS {len(todo)}/{len(todo)} done", flush=True)
    print(f"wrote {out}; {n_patch} patches this run")
    return 0


def swebench_evaluate(a) -> int:
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", a.dataset,
           "--predictions_path", a.predictions, "--max_workers", str(a.max_workers), "--run_id", a.run_id]
    if a.split:
        cmd += ["--split", a.split]
    print("running:", " ".join(cmd), "\n(needs Docker; see the SWE-bench README for disk and memory requirements)")
    return subprocess.call(cmd)


def swebench_import(a) -> int:
    rep = json.loads(Path(a.report).read_text())
    resolved = set(rep.get("resolved_ids", []))
    empty = set(rep.get("empty_patch_ids", []))
    failed = set(rep.get("unresolved_ids", [])) | set(rep.get("error_ids", []))
    rows = ([{"id": f"swebench:{i}", "kind": "swebench", "subject": "code", "verdict": "correct"} for i in resolved]
            + [{"id": f"swebench:{i}", "kind": "swebench", "subject": "code", "verdict": "abstained"} for i in empty]
            + [{"id": f"swebench:{i}", "kind": "swebench", "subject": "code", "verdict": "wrong"}
               for i in failed - resolved - empty])
    total = len(rows)
    print(f"SWE-bench, {a.model}: {len(resolved)}/{total} resolved ({len(resolved) / max(1, total):.1%}), "
          f"{len(empty)} without a patch")
    n = record(a.stats_model or a.model, rows, a, "swebench")
    if n:
        print(f"recorded {n} results as stats for {a.stats_model or a.model}")
    return 0


HUNK_RE = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", re.M)


def patch_ok(patch: str) -> bool:
    """Structurally a unified diff: file headers and at least one hunk."""
    return bool(patch) and bool(re.search(r"^(diff --git |--- )", patch, re.M)) and bool(HUNK_RE.search(patch))


def patch_files(patch: str) -> set[str]:
    return set(re.findall(r"^\+\+\+ b/(\S+)", patch, re.M))


def failure_mode(model_patch: str, gold: str) -> str:
    """Why a model's patch is wrong, as far as can be told without running it."""
    if not model_patch.strip():
        return "no_patch"            # no diff in the reply: a format deficit
    if not patch_ok(model_patch):
        return "malformed_patch"     # diff-like but not applicable
    if not patch_files(model_patch) & patch_files(gold):
        return "wrong_files"         # localisation: edited none of the files the fix touches
    return "wrong_fix"


def swebench_train_data(a) -> int:
    """SWE-bench train split -> sft.jsonl / dpo.jsonl in deficits.py's layout.

    Targets are the gold patches, kept only when they are well-formed diffs.
    A prompt is the dataset's retrieval text when present (bm25 / oracle
    variants), else the issue. With --predictions (the model's own patches on
    the train split), each instance where the model's patch differs from gold
    gives a DPO pair, and the failure mode is counted. With --report (a test
    run's harness report) the test failure modes steer which training
    instances are taken first: if the model mostly fails to produce a diff,
    the instances it also failed to produce one for come first, and so on.
    Instances whose model patch equals the gold patch are anchors, not
    deficits."""
    import random
    import deficits as dfx
    rng = random.Random(a.seed)
    rows = load_rows(a.dataset, a.split)
    test_ids = set()
    for spec in a.exclude or []:
        test_ids |= {r["instance_id"] for r in load_rows(spec, "test")}
    preds = {}
    if a.predictions:
        for line in Path(a.predictions).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                preds[r["instance_id"]] = r.get("model_patch") or ""
    # failure modes on the *test* run: what to prioritise
    priority = {}
    if a.report:
        rep = json.loads(Path(a.report).read_text())
        n_empty = len(rep.get("empty_patch_ids", []))
        n_fail = len(set(rep.get("unresolved_ids", [])) | set(rep.get("error_ids", [])))
        priority = {"no_patch": n_empty, "malformed_patch": len(rep.get("error_ids", [])),
                    "wrong_files": n_fail, "wrong_fix": n_fail}
    skipped, modes, items, anchors = defaultdict(int), defaultdict(int), [], []
    for r in rows:
        iid = r["instance_id"]
        if iid in test_ids:
            skipped["in an excluded (test) set"] += 1
            continue
        gold = (r.get("patch") or "").strip("\n") + "\n"
        if not patch_ok(gold):
            skipped["gold patch is not a well-formed diff"] += 1
            continue
        prompt = r.get("text") or (SWE_INSTRUCTIONS + "Repository: " + r["repo"] + "\n\nIssue:\n" + r["problem_statement"])
        if a.max_chars and len(prompt) + len(gold) > a.max_chars:
            skipped[f"longer than --max-chars {a.max_chars}"] += 1
            continue
        target = f"<patch>\n{gold}</patch>"
        meta = {"task": f"swebench:{iid}", "subject": "code", "repo": r.get("repo")}
        if iid in preds:
            mine = extract_patch(preds[iid]) if "<patch>" in preds[iid] or "```" in preds[iid] else preds[iid]
            if mine.strip() == gold.strip():
                anchors.append({"messages": dfx.chat(prompt, target), "source": "anchor", **meta})
                continue
            mode = failure_mode(mine, gold)
            modes[mode] += 1
            item = {"messages": dfx.chat(prompt, target), "source": "correction", "category": "swebench_" + mode,
                    "target": "gold patch", **meta}
            dpo = {"prompt": [{"role": "user", "content": prompt}],
                   "chosen": [{"role": "assistant", "content": target}],
                   "rejected": [{"role": "assistant", "content": preds[iid] or "(no patch)"}],
                   "source": "correction", "category": "swebench_" + mode, **meta}
            items.append((priority.get(mode, 0), item, dpo))
        else:
            items.append((-1, {"messages": dfx.chat(prompt, target), "source": "gold", "category": "swebench",
                               "target": "gold patch", **meta}, None))
    rng.shuffle(items)
    items.sort(key=lambda x: -x[0])          # stable: shuffled within each priority
    if a.limit:
        items = items[:a.limit]
    deficits = [i for _, i, _ in items]
    dpo = [d for _, _, d in items if d]
    pool = list(anchors)
    if a.anchors:
        for line in Path(a.anchors).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if isinstance(rec.get("messages"), list) and isinstance(rec["messages"][0].get("content"), str):
                    pool.append({"messages": rec["messages"], "source": rec.get("source", "anchor")})
    sft, replay = dfx.mix(deficits, pool, a.deficit_fraction, rng)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sft))
    (out / "dpo.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dpo))
    plan = {"source": a.dataset, "split": a.split, "sft_examples": len(sft), "deficit_examples": len(deficits),
            "replay_examples": len(replay), "dpo_pairs": len(dpo), "train_failure_modes": dict(modes),
            "test_failure_priority": priority, "skipped": dict(skipped),
            "method": "LoRA / QLoRA, SFT then DPO; set finetune.py --max-length to fit the prompts"}
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))
    if not pool:
        print("warning: no anchor data; pass --anchors (e.g. deficits.py's sft.jsonl) to avoid forgetting",
              file=sys.stderr)
    return 0


# ---------------------------------------------------------------- leaderboards

def leaderboard_import(a) -> int:
    from model_stats import StatsStore
    p = Path(a.file)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
    alias = dict(m.split("=", 1) for m in a.map)          # local id -> leaderboard model name
    wanted = {v.lower(): k for k, v in alias.items()}
    store = StatsStore(a.record_stats or None)
    n = 0
    for r in rows:
        name = str(r.get(a.model_col, "")).strip()
        local = wanted.get(name.lower())
        if local is None:
            continue
        bench = str(r.get(a.benchmark_col, "")).strip()
        try:
            score = float(str(r.get(a.score_col, "")).rstrip("%"))
        except ValueError:
            continue
        if score > 1.0:
            score /= 100.0
        store.record(local, "reference", benchmark=bench, subject=subject_for_benchmark(bench), score=score,
                     source=a.source, leaderboard_model=name)
        n += 1
    print(f"recorded {n} published scores as reference stats ({a.source})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    top = p.add_subparsers(dest="bench", required=True)

    lb = top.add_parser("livebench").add_subparsers(dest="cmd", required=True)
    r = lb.add_parser("run")
    r.add_argument("--livebench", required=True, help="LiveBench checkout")
    r.add_argument("--endpoint", required=True, help="URL@model (OpenAI-compatible)")
    r.add_argument("--bench", nargs="+", default=["live_bench"], help="e.g. live_bench/math live_bench/coding")
    r.add_argument("--display-name")
    r.add_argument("--model")
    r.add_argument("--release", help="LiveBench release, e.g. 2025-11-25")
    r.add_argument("--parallel", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=0)
    r.add_argument("--stats-model")
    r.add_argument("--full-marks", type=float, default=1.0)
    add_stats_args(r)
    i = lb.add_parser("import")
    i.add_argument("--livebench", required=True)
    i.add_argument("--model", required=True, help="model name as LiveBench recorded it")
    i.add_argument("--stats-model", help="record under this id instead (e.g. Studio's model id)")
    i.add_argument("--full-marks", type=float, default=1.0, help="score that counts as correct")
    add_stats_args(i)

    sw = top.add_parser("swebench").add_subparsers(dest="cmd", required=True)
    pr = sw.add_parser("predict")
    pr.add_argument("--endpoint", required=True)
    pr.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite_bm25_13K",
                    help="HF dataset or local .jsonl/.json/.parquet; *_bm25_* / *_oracle variants include code context")
    pr.add_argument("--split", default="test")
    pr.add_argument("--instance-ids", nargs="*")
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--model-name")
    pr.add_argument("--max-tokens", type=int, default=4096)
    pr.add_argument("--request-timeout", type=float, default=1800)
    pr.add_argument("--api-key", default="")
    pr.add_argument("--out", required=True)
    ev = sw.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ev.add_argument("--split", default="test")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--max-workers", type=int, default=2)
    im = sw.add_parser("import")
    im.add_argument("--report", required=True, help="<model>.<run_id>.json written by the harness")
    im.add_argument("--model", required=True)
    im.add_argument("--stats-model")
    add_stats_args(im)

    td = sw.add_parser("train-data", help="retraining data from the train split's gold patches")
    td.add_argument("--dataset", default="princeton-nlp/SWE-bench_bm25_13K",
                    help="dataset with a train split and gold `patch` (bm25/oracle variants carry code context)")
    td.add_argument("--split", default="train")
    td.add_argument("--predictions", help="the model's own predictions on the same split (enables DPO)")
    td.add_argument("--report", help="harness report from a test run; its failure modes set priorities")
    td.add_argument("--exclude", nargs="*", default=[],
                    help="datasets whose instance ids must never be trained on (e.g. SWE-bench_Lite)")
    td.add_argument("--anchors", help="replay data: JSONL with `messages` (e.g. deficits.py sft.jsonl)")
    td.add_argument("--deficit-fraction", type=float, default=0.25)
    td.add_argument("--max-chars", type=int, default=60000, help="drop prompt+patch longer than this (0: keep all)")
    td.add_argument("--limit", type=int, default=0, help="at most this many training instances")
    td.add_argument("--seed", type=int, default=0)
    td.add_argument("--out", required=True)

    ld = top.add_parser("leaderboard").add_subparsers(dest="cmd", required=True)
    li = ld.add_parser("import", help="published scores (CSV/JSON) as reference stats")
    li.add_argument("--file", required=True)
    li.add_argument("--map", action="append", default=[], required=True, metavar="LOCAL_ID=LEADERBOARD_NAME")
    li.add_argument("--model-col", default="model")
    li.add_argument("--benchmark-col", default="benchmark")
    li.add_argument("--score-col", default="score")
    li.add_argument("--source", default="benchlm")
    li.add_argument("--record-stats", nargs="?", const="", metavar="DIR")
    a = p.parse_args(argv)

    if a.bench == "livebench":
        return livebench_run(a) if a.cmd == "run" else livebench_import(a)
    if a.bench == "swebench":
        return {"predict": swebench_predict, "evaluate": swebench_evaluate, "import": swebench_import,
                "train-data": swebench_train_data}[a.cmd](a)
    return leaderboard_import(a)


if __name__ == "__main__":
    raise SystemExit(main())
