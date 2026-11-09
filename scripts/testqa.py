#!/usr/bin/env python3
"""
TestQA: one bank of graded prompts, any number of OpenAI-compatible endpoints,
results per kind and per subject, paired comparison against a reference.

Task kinds (qa/bank/*.jsonl, mixable, one JSON object per line):

    reasoning  {"prompt", "answer", "answer_type": number|choice|text}
               the model is asked to end with "Answer: <value>"; numbers
               compare with tolerance and accept fractions like 3/28
    code_exec  {"prompt", "entry_point", "tests": ["assert ...", ...]}
               the INTERPRETER: the reply's code block runs with the tests in
               a separate, time- and memory-limited Python process.
               Off unless --allow-exec (it runs model-written code)
    code       {"prompt", "modules": [...]}  every referenced symbol must exist
    qa         {"prompt", "aliases": [...], "expect_abstain"?: bool}
    constraints {"prompt", "checks": {...}}  instruction following, checked
               mechanically: lines, sentences, bullets, numbered, min/max_words,
               max_chars, must_include, must_not_include, forbid_words, forbid_chars,
               paragraphs, regex,
               starts_with, ends_with, json_keys, acrostic, title_case
    canary     {"prompt"}  ungraded; answered vs refused

Any task may carry "image": a spec rendered by scripts/qa_images.py, or a
file path; it is sent as an image_url part, so vision endpoints can be graded
with the same verdicts.

The bank is organised by subject (qa/bank/<subject>.jsonl). --per-subject N
draws a balanced sample; --list prints what the bank covers.

Every task may carry a "subject"; tasks without one are labelled by the
subject classifier (scripts/subject_classifier.py), which is the same model a
router would use to pick an endpoint for a prompt.

    python scripts/testqa.py --endpoint base=http://127.0.0.1:1234/v1@my-model \\
        --endpoint supp=http://127.0.0.1:1234/v1@my-model-supp050 \\
        --allow-exec --out runs/testqa.json

Endpoints are LABEL=URL[@model]; URL may or may not end in /v1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_eval import (ABSTAIN, _code_blocks, grade_canary, grade_code, mcnemar, norm,  # noqa: E402
                        strip_think)
from subject_classifier import default_classifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REASONING_SUFFIX = "\n\nThink it through, then finish with a final line of the form 'Answer: <value>'."
GOOD = {"correct", "answered"}


# ---------------------------------------------------------------- graders

ANSWER_RE = re.compile(r"answer\s*(?:is)?\s*[:：]\s*(.+)", re.I)
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?|-?\.\d+")


def final_answer(text: str) -> str:
    body = strip_think(text).strip()
    m = ANSWER_RE.findall(body)
    if m:
        return m[-1].strip().strip("*").strip()
    lines = [l for l in body.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def parse_number(s: str) -> float | None:
    s = s.replace("\\frac{", "").replace("}{", "/").replace("}", "")
    m = NUM_RE.findall(s)
    if not m:
        return None
    tok = m[0].replace(",", "").replace(" ", "")
    try:
        return float(Fraction(tok)) if "/" in tok else float(tok)
    except (ValueError, ZeroDivisionError):
        return None


def grade_reasoning(text: str, task: dict) -> str:
    body = strip_think(text).strip()
    if not body:
        return "abstained"
    ans = final_answer(body)
    kind = task.get("answer_type", "text")
    gold = str(task["answer"])
    if kind == "number":
        # "9 - 4 = 5": the result is what follows the last "=".
        got, want = parse_number(ans.rsplit("=", 1)[-1]), parse_number(gold)
        if got is None:
            # Fall back to the last number anywhere in the reply.
            nums = NUM_RE.findall(body)
            got = parse_number(nums[-1]) if nums else None
        if got is None:
            return "abstained" if any(m in body.lower() for m in ABSTAIN) else "wrong"
        return "correct" if abs(got - want) <= 1e-6 * max(1.0, abs(want)) + 1e-9 else "wrong"
    if kind == "choice":
        m = re.search(r"\b([A-E])\b", ans.upper())
        return "correct" if m and m.group(1) == gold.upper() else "wrong"
    a, g = norm(ans), norm(gold)
    if not a:
        return "wrong"
    return "correct" if a == g or re.search(rf"\b{re.escape(g)}\b", a) else "wrong"


def grade_qa(text: str, task: dict) -> str:
    body = strip_think(text).strip()
    low = body.lower()
    abstained = not body or any(m in low for m in ABSTAIN)
    if task.get("expect_abstain"):
        return "correct" if abstained or any(norm(a) in norm(body) for a in task["aliases"]) else "wrong"
    if abstained:
        return "abstained"
    n = norm(body)
    return "correct" if any(norm(a) and norm(a) in n for a in task["aliases"]) else "wrong"


SENT_RE = re.compile(r"[^.!?]+[.!?]+(?=\s|$)")


def _clean(text: str) -> str:
    body = strip_think(text).strip()
    if body.startswith("```") and body.endswith("```"):
        body = body.strip("`").split("\n", 1)[-1].strip()
    return body.strip().strip('"').strip("\u201c\u201d").strip()


def check_constraints(text: str, checks: dict) -> list[str]:
    """Failed checks (empty list = all satisfied)."""
    body = _clean(text)
    lines = [l for l in body.splitlines() if l.strip()]
    words = re.findall(r"[A-Za-z0-9\u00C0-\u024F']+", body)
    low = body.lower()
    fail = []
    c = checks
    if "lines" in c and len(lines) != c["lines"]:
        fail.append(f"{len(lines)} lines, want {c['lines']}")
    if "sentences" in c:
        n = len(SENT_RE.findall(body)) or (1 if body and body[-1] not in ".!?" else 0)
        if n != c["sentences"]:
            fail.append(f"{n} sentences, want {c['sentences']}")
    if "bullets" in c:
        n = sum(1 for l in lines if re.match(r"\s*[-*\u2022]\s+", l))
        if n != c["bullets"]:
            fail.append(f"{n} bullets, want {c['bullets']}")
    if "numbered" in c:
        nums = [int(m.group(1)) for l in lines if (m := re.match(r"\s*(\d+)[.)]\s", l))]
        if nums != list(range(1, c["numbered"] + 1)):
            fail.append(f"numbering {nums}")
    if "min_words" in c and len(words) < c["min_words"]:
        fail.append(f"{len(words)} words < {c['min_words']}")
    if "max_words" in c and len(words) > c["max_words"]:
        fail.append(f"{len(words)} words > {c['max_words']}")
    if "max_chars" in c and len(body) > c["max_chars"]:
        fail.append(f"{len(body)} chars > {c['max_chars']}")
    for w in c.get("must_include", []):
        if w.lower() not in low:
            fail.append(f"missing {w!r}")
    for w in c.get("must_not_include", []):
        if w.lower() in low:
            fail.append(f"contains {w!r}")
    if "paragraphs" in c:
        n = len([b for b in re.split(r"\n\s*\n", body) if b.strip()])
        if n != c["paragraphs"]:
            fail.append(f"{n} paragraphs, want {c['paragraphs']}")
    for w in c.get("forbid_words", []):
        if re.search(rf"\b{re.escape(w)}\b", body, re.I):
            fail.append(f"uses the word {w!r}")
    if c.get("forbid_chars") and any(ch in body for ch in c["forbid_chars"]):
        fail.append("uses a forbidden character")
    for rx in c.get("regex", []):
        if not re.search(rx, body, re.M):
            fail.append(f"does not match {rx}")
    if c.get("starts_with") and not body.startswith(c["starts_with"]):
        fail.append(f"does not start with {c['starts_with']!r}")
    if c.get("ends_with") and not body.rstrip().endswith(c["ends_with"]):
        fail.append(f"does not end with {c['ends_with']!r}")
    if c.get("json_keys"):
        try:
            obj = json.loads(body)
            missing = [k for k in c["json_keys"] if not isinstance(obj, dict) or k not in obj]
            if missing:
                fail.append(f"JSON missing {missing}")
        except json.JSONDecodeError:
            fail.append("not valid JSON")
    if c.get("acrostic"):
        initials = "".join(l.strip().lstrip("*-_#> ")[:1].upper() for l in lines)
        if initials != c["acrostic"].upper():
            fail.append(f"initials {initials!r}")
    if c.get("title_case"):
        small = {"a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "at", "by", "with"}
        bad = [w for i, w in enumerate(words) if (i == 0 or w.lower() not in small) and not w[:1].isupper()]
        if bad:
            fail.append(f"not title case: {bad[:3]}")
    return fail


def grade_constraints(text: str, task: dict) -> tuple[str, str]:
    body = _clean(text)
    if not body:
        return "abstained", ""
    if any(m in body.lower()[:80] for m in ("i can't", "i cannot", "i won't", "i'm unable", "i am unable")):
        return "abstained", ""
    fail = check_constraints(text, task["checks"])
    return ("wrong", "; ".join(fail)[:160]) if fail else ("correct", "")


# ---------------------------------------------------------------- interpreter

RUNNER = r'''
import resource, sys
def _limit(res, v):
    try:
        resource.setrlimit(res, (v, v))
    except (ValueError, OSError):
        pass
_limit(resource.RLIMIT_CPU, {cpu})
_limit(resource.RLIMIT_AS, {mem})
_limit(resource.RLIMIT_FSIZE, 1 << 20)
_limit(resource.RLIMIT_CORE, 0)
'''


# Container sandbox for model-written code: set by --sandbox (or by callers via
# configure_sandbox). None means the rlimited local interpreter.
SANDBOX = {"engine": None, "image": "python:3.12-slim"}


def configure_sandbox(engine: str | None, image: str | None = None) -> None:
    if engine in (None, "", "none"):
        SANDBOX["engine"] = None
        return
    if engine not in ("docker", "podman"):
        raise ValueError("sandbox must be none, docker or podman")
    if not shutil.which(engine):
        raise SystemExit(f"--sandbox {engine}: {engine} is not installed. Install it, use the other engine "
                         "(--sandbox docker|podman), or leave --sandbox off to run model code with local "
                         "CPU/memory limits only, which is not a security boundary")
    SANDBOX["engine"] = engine
    if image:
        SANDBOX["image"] = image


def _sandbox_cmd(engine: str, image: str, workdir: str, name: str, timeout: float, mem_mb: int) -> list[str]:
    return [engine, "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--memory", f"{mem_mb}m", "--memory-swap", f"{mem_mb}m",
            "--cpus", "1", "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--ulimit", f"cpu={int(timeout) + 1}", "--ulimit", "fsize=1048576",
            "-v", f"{workdir}:/work:ro", "-w", "/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0",
            image, "python", "-I", "/work/candidate.py"]


def run_code(code: str, tests: list[str], timeout: float = 10.0, mem_mb: int = 1024) -> tuple[str, str]:
    """Run candidate code plus tests in a fresh interpreter.

    Default isolation is best effort: separate process, empty temp cwd, -I (no
    user site, no env vars), stripped environment, CPU/memory/file-size
    rlimits on POSIX, wall-clock timeout. That is NOT a security sandbox.
    With --sandbox docker|podman the code runs in a throwaway container with no
    network, a read-only root, all capabilities dropped, no-new-privileges, an
    unprivileged user, and memory/CPU/pid limits. Returns (verdict, detail)."""
    engine = SANDBOX["engine"]
    with tempfile.TemporaryDirectory(prefix="testqa-") as d:
        prelude = RUNNER.format(cpu=int(timeout) + 1, mem=mem_mb * 1024 * 1024) if os.name == "posix" or engine else ""
        body = [prelude, code, "", "# --- tests ---"]
        for i, t in enumerate(tests):
            body.append(f"def __t{i}():")
            body.extend("    " + line for line in t.splitlines())
            body.append(f"__t{i}()")
        body.append('print("__TESTQA_PASS__")')
        path = Path(d) / "candidate.py"
        path.write_text("\n".join(body), encoding="utf-8")
        if engine:
            os.chmod(d, 0o755)
            os.chmod(path, 0o644)
            name = "testqa-" + os.urandom(8).hex()
            cmd = _sandbox_cmd(engine, SANDBOX["image"], d, name, timeout, mem_mb)
            env, cwd = None, None
            wall = timeout + 30                   # container start-up is not the candidate's time
        else:
            cmd = [sys.executable, "-I", str(path)]
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
            cwd, wall = d, timeout
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=wall,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            if engine:
                subprocess.run([engine, "kill", name], capture_output=True)
            return "timeout", f"exceeded {timeout}s"
        if engine and p.returncode == 125:
            raise RuntimeError(f"{engine} could not start the sandbox: {(p.stderr or '').strip()[-300:]}")
        if engine and p.returncode in (137, 152) and "__TESTQA_PASS__" not in p.stdout:
            return "timeout", f"killed by the sandbox's CPU or memory limit (exit {p.returncode})"
        if p.returncode == 0 and "__TESTQA_PASS__" in p.stdout:
            return "correct", ""
        err = (p.stderr or p.stdout).strip().splitlines()
        return "wrong", (err[-1] if err else f"exit {p.returncode}")[:200]


def exec_permitted(model: str, a) -> tuple[bool, str]:
    """May this model's code run here? The permission policy for job testqa (scripts/permissions.py),
    for the program that runs it: the sandbox engine, or Python. A batch run has no one to ask, so
    only allow runs it."""
    import permissions as P
    prog = SANDBOX["engine"] or sys.executable
    d = P.Policy().check("exec", prog, model=model, job="testqa", profile=getattr(a, "permissions", "") or "")
    return d["effect"] == "allow", f"exec {prog}: {d['effect']} ({d['why']})"


def grade_code_exec(text: str, task: dict, allow_exec: bool, timeout: float) -> tuple[str, str]:
    code = _code_blocks(strip_think(text))
    if not code.strip():
        return "abstained", ""
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        return "unparsable", str(e)[:120]
    if f"def {task['entry_point']}" not in code:
        return "wrong", f"no def {task['entry_point']}"
    if not allow_exec:
        return "skipped", "pass --allow-exec to run code"
    return run_code(code, task["tests"], timeout)


# ---------------------------------------------------------------- tasks

def kind_of(task: dict) -> str:
    if task.get("kind"):
        return task["kind"]
    if task.get("tests"):
        return "code_exec"
    if task.get("checks"):
        return "constraints"
    if "answer" in task:
        return "reasoning"
    if task.get("aliases"):
        return "qa"
    if task.get("modules"):
        return "code"
    return "canary"


def load_tasks(paths: list[str], kinds: set[str] | None, subjects: set[str] | None, limit: int) -> list[dict]:
    files: list[Path] = []
    for p in paths:
        q = Path(p)
        files += sorted(q.glob("*.jsonl")) if q.is_dir() else [q]
    tasks, seen = [], set()
    clf = None
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t["id"] in seen:
                raise SystemExit(f"duplicate task id {t['id']} in {f}")
            seen.add(t["id"])
            t["kind"] = kind_of(t)
            t["_base"] = str(f.parent)
            if not t.get("subject"):
                clf = clf or default_classifier()
                t["subject"] = clf.predict(t["prompt"])
                t["subject_inferred"] = True
            if kinds and t["kind"] not in kinds:
                continue
            if subjects and t["subject"] not in subjects:
                continue
            tasks.append(t)
    return tasks[:limit] if limit else tasks


def per_subject_sample(tasks: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to n graded tasks per subject (balanced across kinds), canaries kept."""
    import random
    rng = random.Random(seed)
    by = defaultdict(list)
    out = [t for t in tasks if t["kind"] == "canary"]
    for t in tasks:
        if t["kind"] != "canary":
            by[t["subject"]].append(t)
    for subj, ts in sorted(by.items()):
        kinds = defaultdict(list)
        for t in ts:
            kinds[t["kind"]].append(t)
        for v in kinds.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < n and any(kinds.values()):     # round-robin over kinds
            for k in sorted(kinds):
                if kinds[k] and len(picked) < n:
                    picked.append(kinds[k].pop())
        out += picked
    return out


def bank_listing(tasks: list[dict]) -> str:
    by = defaultdict(Counter)
    for t in tasks:
        by[t["subject"] if t["kind"] != "canary" else "(canary)"][t["kind"]] += 1
    lines = [f"{'subject':<10} {'graded':>6}  kinds"]
    for s, c in sorted(by.items()):
        graded = sum(v for k, v in c.items() if k != "canary")
        lines.append(f"{s:<10} {graded:>6}  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    return "\n".join(lines)


def prompt_for(task: dict) -> str:
    return task["prompt"] + (REASONING_SUFFIX if task["kind"] == "reasoning" else "")


# ---------------------------------------------------------------- endpoints

def parse_endpoint(spec: str) -> tuple[str, str, str | None]:
    if "=" not in spec:
        raise SystemExit(f"--endpoint needs LABEL=URL[@model], got {spec!r}")
    label, rest = spec.split("=", 1)
    url, model = rest, None
    if "@" in rest.split("//", 1)[-1]:
        url, model = rest.rsplit("@", 1)
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return label, url, model


def parse_concurrency(v) -> int | str:
    if str(v).strip().lower() == "auto":
        return "auto"
    try:
        n = int(v)
    except ValueError:
        raise SystemExit(f"concurrency: a number or auto, got {v!r}")
    if n < 1:
        raise SystemExit("concurrency: at least 1")
    return n


def message_for(task: dict) -> dict:
    text = prompt_for(task)
    if not task.get("image"):
        return {"role": "user", "content": text}
    from qa_images import data_url
    url = data_url(task["image"], Path(task.get("_base", ".")))
    return {"role": "user", "content": [{"type": "text", "text": text},
                                        {"type": "image_url", "image_url": {"url": url}}]}


def ask(url: str, model: str | None, message, a, usage: dict | None = None) -> str:
    import requests
    if isinstance(message, str):
        message = {"role": "user", "content": message}
    body = {"messages": [message], "temperature": a.temperature,
            "max_tokens": a.max_tokens}
    if model:
        body["model"] = model
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    r = requests.post(f"{url}/chat/completions", json=body, headers=headers, timeout=a.request_timeout)
    r.raise_for_status()
    d = r.json()
    if usage is not None:
        usage.update(d.get("usage") or {})
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or ""
    rc = msg.get("reasoning_content") or msg.get("reasoning")
    if rc and "</think>" not in txt:
        txt = f"<think>{rc}</think>{txt}"
    return txt


def grade(task: dict, text: str, a) -> tuple[str, str]:
    k = task["kind"]
    if k == "reasoning":
        return grade_reasoning(text, task), final_answer(text)[:80]
    if k == "code_exec":
        return grade_code_exec(text, task, a.allow_exec, a.exec_timeout)
    if k == "qa":
        return grade_qa(text, task), ""
    if k == "constraints":
        return grade_constraints(text, task)
    if k == "code":
        return grade_code(text, task), ""
    return grade_canary(text, task), ""


def run_endpoint(label: str, members: list[dict], tasks: list[dict], a, cache_dir: Path | None,
                 progress=None) -> list[dict]:
    """Every task through this label's pool of servers (usually one): each server takes the next task
    whenever it has a free slot (scripts/endpoint_pool.py)."""
    import endpoint_pool
    cache: dict = {}
    cpath = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}.jsonl" if cache_dir else None
    if cpath and cpath.exists():
        for line in cpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cache[r["id"]] = r["text"]
    clock = threading.Lock()
    pooled = len(members) > 1

    def one(m, t):
        t0 = time.time()
        text, usage = cache.get(t["id"]), {}
        if text is None:
            try:
                text = ask(m["url"], m["model"], message_for(t), m["args"], usage)
            except Exception as e:
                r = {"id": t["id"], "kind": t["kind"], "subject": t["subject"], "verdict": "error",
                     "note": str(e)[:160]}
                if pooled:
                    r["served_by"] = m["name"]
                return r, False, None
            if cpath:
                with clock, open(cpath, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": t["id"], "text": text}) + "\n")
        verdict, note = grade(t, text, m["args"])
        r = {"id": t["id"], "kind": t["kind"], "subject": t["subject"], "verdict": verdict, "note": note,
             "seconds": round(time.time() - t0, 2), "text": strip_think(text).strip()[:600]}
        if pooled:
            r["served_by"] = m["name"]
        return r, True, usage.get("completion_tokens")

    lock, n = threading.Lock(), [0]

    def done(t, r, m):
        with lock:
            n[0] += 1
            if progress:
                progress(label, t, r)
            else:
                print(f"PROGRESS {n[0]}/{len(tasks)} {label}: {t['id']} {r['verdict']}", flush=True)
    return endpoint_pool.run_pool(members, tasks, one, done)


# ---------------------------------------------------------------- report

def compare(ref: list[dict], var: list[dict], tasks: list[dict]):
    """Per-item movement between two runs -> gained, lost, canary gained, canary lost."""
    by_id = {t["id"]: t for t in tasks}
    r = {x["id"]: x for x in ref}
    v = {x["id"]: x for x in var}
    gained, lost, canary_gained, canary_lost = [], [], [], []
    for tid, x in r.items():
        y = v.get(tid)
        if y is None:
            continue
        a, b = x["verdict"], y["verdict"]
        if a == b or {"error", "skipped"} & {a, b}:
            continue
        kind = by_id[tid]["kind"]
        rec = {"id": tid, "kind": kind, "subject": by_id[tid]["subject"], "from": a, "to": b,
               "prompt": by_id[tid]["prompt"][:110], "answer": y.get("text", "")[:160]}
        if kind == "canary":
            (canary_lost if b == "refused" else canary_gained).append(rec)
        elif a not in GOOD and b in GOOD:
            gained.append(rec)
        elif a in GOOD and b not in GOOD:
            lost.append(rec)
    return gained, lost, canary_gained, canary_lost


def summarise(rows: list[dict]) -> dict:
    by = {"kind": defaultdict(Counter), "subject": defaultdict(Counter)}
    for r in rows:
        by["kind"][r["kind"]][r["verdict"]] += 1
        by["subject"][r["subject"]][r["verdict"]] += 1

    def score(c: Counter) -> dict:
        graded = sum(v for k, v in c.items() if k not in ("error", "skipped"))
        good = c["correct"] + c["answered"]
        out = {"n": sum(c.values()), "graded": graded, "score": round(good / graded, 3) if graded else None,
               **dict(c)}
        strict = c["correct"] + c["wrong"] + c["abstained"] + c["unparsable"] + c["timeout"]
        if strict:
            from model_stats import wilson
            wrong = c["wrong"] + c["unparsable"] + c["timeout"]
            out.update(accuracy=round(c["correct"] / strict, 3), hallucination_rate=round(wrong / strict, 3),
                       abstention_rate=round(c["abstained"] / strict, 3),
                       accuracy_ci=[round(x, 3) for x in wilson(c["correct"], strict)])
        return out
    return {g: {k: score(c) for k, c in sorted(d.items())} for g, d in by.items()}


def print_table(title: str, results: dict, group: str) -> None:
    keys = sorted({k for s in results.values() for k in s[group]})
    labels = list(results)
    print(f"\n{title}")
    print(f"  {'':<12}" + "".join(f"{l[:12]:>14}" for l in labels))
    for k in keys:
        cells = []
        for l in labels:
            s = results[l][group].get(k)
            cells.append("-" if not s or s["score"] is None else f"{s['score']:.2f} ({s['graded']})")
        print(f"  {k:<12}" + "".join(f"{c:>14}" for c in cells))


def print_subjects(summ: dict, min_n: int) -> None:
    print(f"\nper subject (right / hallucinated / abstained, 95% CI on right; '*' = fewer than {min_n} graded)")
    for label, sm in summ.items():
        print(f"  {label}")
        for subj, v in sm["subject"].items():
            if "accuracy" not in v:
                continue
            # Outcome counts list only outcomes that happened: a subject with no right answers has no "correct".
            n = sum(v.get(k, 0) for k in ("correct", "wrong", "abstained", "unparsable", "timeout"))
            lo, hi = v["accuracy_ci"]
            flag = "*" if n < min_n else " "
            print(f"    {subj:<9}{flag} n {n:>3}   right {v['accuracy']:5.0%} [{lo:4.0%}-{hi:4.0%}]   "
                  f"halluc {v['hallucination_rate']:5.0%}   abstain {v['abstention_rate']:5.0%}")


def skills_from(summary: dict) -> dict:
    """Per-subject scores in the shape subject_classifier.py route expects."""
    return {s: v["score"] for s, v in summary["subject"].items() if v["score"] is not None}


def stats_rows(rows: list[dict], source: str = "testqa") -> list[dict]:
    """Graded outcomes only: canaries, skipped code and request errors say
    nothing about accuracy."""
    out = []
    for r in rows:
        if r["kind"] == "canary" or r["verdict"] in ("error", "skipped"):
            continue
        out.append({"subject": r["subject"], "task_kind": r["kind"], "verdict": r["verdict"],
                    "task": r["id"], "source": r.get("source", source)})
    return out


def record_stats(model_id: str, rows: list[dict], a, source: str = "testqa") -> int:
    graded = stats_rows(rows, source)
    if a.publish_stats:
        import requests
        headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
        r = requests.post(a.publish_stats.rstrip("/") + "/api/stats/ingest", headers=headers,
                          json={"model": model_id, "records": graded}, timeout=60)
        r.raise_for_status()
        return int(r.json().get("recorded", 0))
    from model_stats import StatsStore
    store = StatsStore(a.record_stats or None)
    return sum(1 for g in graded if store.record(model_id, "graded", **g))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--endpoint", action="append", default=[], metavar="LABEL=URL[@model]")
    p.add_argument("--tasks", nargs="+", default=[str(ROOT / "qa" / "bank")], help="JSONL files or directories")
    p.add_argument("--kind", nargs="*", help="only these kinds")
    p.add_argument("--with", dest="with_packs", nargs="*", default=[], metavar="PACK",
                   help="add optional task packs from qa/bank/optional/ (e.g. word-bans); "
                        f"available: {', '.join(sorted(x.stem for x in (ROOT / 'qa' / 'bank' / 'optional').glob('*.jsonl')))}")
    p.add_argument("--subject", nargs="*", help="only these subjects")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only-ids", metavar="FILE",
                   help="JSON with an 'ids' list (e.g. deficits.py holdout_ids.json): run only these tasks")
    p.add_argument("--per-subject", type=int, default=0, metavar="N",
                   help="balanced sample of up to N graded tasks per subject")
    p.add_argument("--seed", type=int, default=0, help="sampling seed for --per-subject")
    p.add_argument("--min-subject-n", type=int, default=30, help="flag subjects graded on fewer items")
    p.add_argument("--list", action="store_true", help="show bank coverage by subject and kind, then exit")
    p.add_argument("--reference", help="label compared against (default: first endpoint)")
    p.add_argument("--allow-exec", action="store_true", help="run code_exec tasks (executes model-written code)")
    p.add_argument("--permissions", default="", help="permission profile for running code (default: the policy for job testqa; docs/PERMISSIONS.md)")
    p.add_argument("--exec-timeout", type=float, default=10.0)
    p.add_argument("--sandbox", choices=["none", "docker", "podman"], default="none",
                   help="run --allow-exec code in a locked-down container (no network, read-only, unprivileged)")
    p.add_argument("--sandbox-image", default="python:3.12-slim", help="container image with python")
    p.add_argument("--concurrency", default="auto",
                   help="requests in flight per server: a number, or auto (start at one and add while "
                        "throughput rises, up to --max-concurrency and the server's slots)")
    p.add_argument("--max-concurrency", type=int, default=8, help="the most auto goes to")
    p.add_argument("--parallel", action="append", default=[], metavar="LABEL=N|auto",
                   help="this endpoint's own concurrency, e.g. a slow one at 1")
    p.add_argument("--pool", action="append", default=[], metavar="LABEL=URL[@model]",
                   help="another server for an endpoint's tasks (same model): the servers share them, "
                        "each taking the next whenever it has a free slot")
    p.add_argument("--slow", type=float, default=120.0,
                   help="auto keeps a server at one request when a single request takes longer (seconds)")
    p.add_argument("--sequential", action="store_true",
                   help="one endpoint after another (default: every endpoint at once)")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--cache", help="directory for per-endpoint reply caches (resume / regrade without re-asking)")
    p.add_argument("--show", type=int, default=8, help="regressions to print")
    p.add_argument("--out")
    p.add_argument("--record-stats", nargs="?", const="", metavar="DIR",
                   help="append graded results to the rolling model stats (default ~/.omnicortex/stats) "
                        "that Studio's model \"auto\" routes on")
    p.add_argument("--publish-stats", metavar="STUDIO_URL",
                   help="send graded results to a running Studio instead (it ties them to the exact model file)")
    p.add_argument("--stats-model", action="append", default=[], metavar="LABEL=MODEL_ID",
                   help="model id to record an endpoint's stats under (default: its @model, else its label)")
    a = p.parse_args(argv)

    for pack in a.with_packs:
        f = ROOT / "qa" / "bank" / "optional" / f"{pack}.jsonl"
        if not f.exists():
            raise SystemExit(f"no optional pack {pack!r} in qa/bank/optional/")
        a.tasks.append(str(f))
    tasks = load_tasks(a.tasks, set(a.kind or []), set(a.subject or []), a.limit)
    if a.only_ids:
        keep = set(json.loads(Path(a.only_ids).read_text())["ids"])
        tasks = [t for t in tasks if t["id"] in keep or t["kind"] == "canary"]
    if a.per_subject:
        tasks = per_subject_sample(tasks, a.per_subject, a.seed)
    if a.list:
        print(bank_listing(tasks))
        return 0
    if not a.endpoint:
        raise SystemExit("--endpoint is required (or use --list)")
    if not tasks:
        raise SystemExit("no tasks selected")
    kinds = Counter(t["kind"] for t in tasks)
    print(f"{len(tasks)} tasks: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    configure_sandbox(a.sandbox, a.sandbox_image)
    if kinds.get("code_exec") and not a.allow_exec:
        print("  code_exec tasks will be checked for syntax only; --allow-exec runs them "
              "(model-written code; add --sandbox docker to contain it)")
    elif kinds.get("code_exec") and a.sandbox == "none":
        print("  running model-written code with rlimits only; --sandbox docker isolates it in a container")
    cache_dir = Path(a.cache) if a.cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    eps = [parse_endpoint(s) for s in a.endpoint]
    import endpoint_pool
    conc = parse_concurrency(a.concurrency)
    per = {k: parse_concurrency(v) for k, v in (x.split("=", 1) for x in a.parallel)}
    servers = endpoint_pool.Servers(conc, a.max_concurrency, a.slow)
    extra: dict = {}
    for spec in a.pool:
        lab, u, m = parse_endpoint(spec)
        extra.setdefault(lab, []).append((u, m))
    unknown = (set(extra) | set(per)) - {e[0] for e in eps}
    if unknown:
        raise SystemExit(f"--pool/--parallel name no --endpoint: {', '.join(sorted(unknown))}")
    pools = {}
    for label, url, model in eps:
        a_ep = a
        if kinds.get("code_exec") and a.allow_exec:
            ok, why = exec_permitted(model or label, a)
            if not ok:
                print(f"  not running {label}'s code: {why} (docs/PERMISSIONS.md)")
                a_ep = argparse.Namespace(**{**vars(a), "allow_exec": False})
        members = [(url, model)] + extra.get(label, [])
        pools[label] = [{"name": u + (f" @ {m}" if m else ""), "url": u, "model": m or model, "args": a_ep,
                         "limiter": servers.get(u, per.get(label))} for u, m in members]
        print(f"querying {label}: " + ", ".join(f"{x['name']} (" + (f"{x['limiter'].limit} at a time" if
                                                  not x["limiter"].auto else f"auto, up to {x['limiter'].cap}") + ")"
                                                  for x in pools[label]))
    total, n = len(tasks) * len(eps), [0]
    plock = threading.Lock()

    def progress(label, t, r):
        with plock:
            n[0] += 1
            # Studio's Jobs page reads this as a progress bar
            print(f"PROGRESS {n[0]}/{total} {label}: {t['id']} {r['verdict']}", flush=True)

    raw, summ = {}, {}
    if a.sequential or len(eps) == 1:
        for label, _, _ in eps:
            raw[label] = run_endpoint(label, pools[label], tasks, a, cache_dir, progress)
    else:
        with ThreadPoolExecutor(max_workers=len(eps)) as ex:
            futs = {label: ex.submit(run_endpoint, label, pools[label], tasks, a, cache_dir, progress)
                    for label, _, _ in eps}
            raw = {label: f.result() for label, f in futs.items()}
    sched = servers.summary()
    print("\nscheduling: " + "; ".join(f"{x['server']}: {x['limit']} at a time ({x['why']}), {x['done']} done"
                                       + (f", {x['avg_s']} s each" if x["avg_s"] else "")
                                       + (f", {x['gen_tok_s']} tok/s" if x["gen_tok_s"] else "") for x in sched))
    for label, _, model in eps:
        if len(pools[label]) > 1:
            share = Counter(r.get("served_by") for r in raw[label])
            print(f"  {label}: " + ", ".join(f"{k} took {v}" for k, v in share.most_common()))
        summ[label] = summarise(raw[label])
        if a.record_stats is not None or a.publish_stats:
            mid = dict(x.split("=", 1) for x in a.stats_model).get(label) or model or label
            n = record_stats(mid, raw[label], a)
            print(f"  recorded {n} graded results as stats for {mid}")
        errs = sum(1 for r in raw[label] if r["verdict"] == "error")
        if errs:
            first = next(r for r in raw[label] if r["verdict"] == "error")
            print(f"  {errs} request errors, e.g. {first['note']}")

    print_table("score by kind (fraction correct/answered, graded n)", summ, "kind")
    print_table("score by subject", summ, "subject")
    print_subjects(summ, a.min_subject_n)

    ref = a.reference or eps[0][0]
    comparisons = []
    for label, _, _ in eps:
        if label == ref:
            continue
        g, lost, cg, cl = compare(raw[ref], raw[label], tasks)
        chi2, disc = mcnemar(len(g), len(lost))
        verdict = ("significant" if chi2 > 3.84 else "not significant") if disc >= 10 else "too few changes to call"
        print(f"\n{label} vs {ref}: gained {len(g)}, regressed {len(lost)}, net {len(g) - len(lost):+d}, "
              f"chi2={chi2:.1f} ({verdict}); canary newly refused {len(cl)}")
        for r in lost[:a.show]:
            print(f"    [{r['kind']}] {r['id']}: {r['from']} -> {r['to']}")
        comparisons.append({"endpoint": label, "gained": g, "regressed": lost, "canary_newly_refused": cl,
                            "canary_newly_answered": cg, "chi2": chi2, "discordant": disc})

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "tasks": len(tasks), "reference": ref,
            "endpoints": {l: {"url": u, "model": m, **({"pool": [x["name"] for x in pools[l]]} if len(pools[l]) > 1 else {})}
                          for l, u, m in eps},
            "scheduling": sched,
            "summary": summ, "skills": {l: skills_from(s) for l, s in summ.items()},
            "comparisons": comparisons, "raw": raw}, indent=2))
        print(f"\nwrote {a.out} (\"skills\" can seed a routing table for subject_classifier.py route)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
