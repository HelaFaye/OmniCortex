#!/usr/bin/env python3
"""
Adapt a model to one repository ("parametric context distillation"): training data grounded in the repository's
own source, with a record of where every pair came from and a check that says when the adapter is out of date.

    python scripts/repo_adapt.py run   --repo ~/src/app --teacher https://api.example.com/v1@model \\
                                       --replay general.jsonl --out runs/adapt-app
    python scripts/repo_adapt.py check --manifest runs/adapt-app/manifest.json --repo ~/src/app

Steps (`run` does them all, and resumes: each step's output is reused when it is already there):
1. **scan**: the files, the public interfaces (functions, classes, methods, exported names, with their signatures),
   the build files and the tests; the git commit; and a fingerprint of the core interfaces.
2. **qa**: for each public symbol and each document, the teacher writes questions and answers about the source
   it is shown, each citing the lines it rests on.
3. **judge**: a pair is kept only if
   - its cited lines exist, inside what the teacher was shown;
   - it names no identifier that the repository lacks (a made-up function is the commonest teacher error);
   - the judge model (the teacher unless `--judge` names another) answers that the source supports it.
4. **build**: `sft.jsonl` for `finetune.py`: 85% repository pairs, 15% general replay so the model keeps what it
   knew (`--replay`, chat-format JSONL); `holdout.jsonl`, whole symbols never trained on; `manifest.json`.

**Keys:** the teacher's API key is read from an environment variable (`--key-env`, default `OC_TEACHER_KEY`),
never from the command line, and it is never written to any output.

**Staleness:** the manifest keeps the commit and the core interfaces. `check` recomputes them: commits that leave
the interfaces alone keep the adapter fresh; a changed, added or removed interface makes it stale (exit 1), and
the changes are listed. `--core` limits which paths count as core (globs, such as `src/api/**`).

**Size:** `check --adapter FILE --max-adapter-mb N` also refuses an adapter larger than its budget.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag  # noqa: E402

BUILD_FILES = {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json", "Cargo.toml",
               "go.mod", "CMakeLists.txt", "Makefile", "meson.build", "build.gradle", "pom.xml", "Dockerfile",
               "Containerfile", "BUILD", "WORKSPACE", "build.zig"}
DOC_EXT = {".md", ".markdown", ".rst", ".txt"}
TEST_RE = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(spec|test)\.[jt]sx?$")
IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_./]*)(?:\(\))?`")
MAX_UNIT_LINES = 120

# exported symbols in other languages: (extensions, pattern, kind); group "name" is the name, "sig" the signature
OTHER = [
    ({".js", ".jsx", ".ts", ".tsx", ".mjs"},
     re.compile(r"^export\s+(?:default\s+)?(?:async\s+)?(?P<kind>function|class|const|let|interface|type)\s+"
                r"(?P<name>[A-Za-z_$][\w$]*)(?P<sig>[^{=]*)"), None),
    ({".rs"}, re.compile(r"^\s*pub\s+(?:async\s+)?(?P<kind>fn|struct|enum|trait|type|const)\s+(?P<name>\w+)"
                         r"(?P<sig>[^{;]*)"), None),
    ({".go"}, re.compile(r"^func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Z]\w*)(?P<sig>\([^{]*)"), "func"),
    ({".go"}, re.compile(r"^type\s+(?P<name>[A-Z]\w*)\s+(?P<sig>struct|interface|\w+)"), "type"),
    ({".h", ".hpp", ".hh"}, re.compile(r"^(?!\s*(?:#|//|return\b|typedef\b))[A-Za-z_][\w\s\*&:<>,]*?\b"
                                       r"(?P<name>[A-Za-z_]\w*)\s*(?P<sig>\([^;{]*\))\s*;"), "func"),
    ({".java", ".kt", ".cs"}, re.compile(r"^\s*public\s+(?:static\s+|final\s+|abstract\s+)*(?:class\s+|interface\s+)?"
                                         r"[\w<>\[\], ]*?\b(?P<name>\w+)\s*(?P<sig>\([^)]*\))?\s*[{;]"), None),
]


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def git(root: Path, *args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


# ---------------------------------------------------------------- 1. scan

def _py_symbols(path: str, text: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out = []

    def sig(n):
        if isinstance(n, ast.ClassDef):
            return "(" + ", ".join(ast.unparse(b) for b in n.bases) + ")" if n.bases else ""
        s = "(" + ast.unparse(n.args) + ")"
        return s + (" -> " + ast.unparse(n.returns) if n.returns else "")

    def add(n, qual):
        kind = "class" if isinstance(n, ast.ClassDef) else "def"
        out.append({"path": path, "name": qual, "kind": kind, "sig": sig(n), "start": n.lineno,
                    "end": getattr(n, "end_lineno", n.lineno), "doc": (ast.get_docstring(n) or "")[:300]})

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
            add(n, n.name)
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                            (not m.name.startswith("_") or m.name == "__init__"):
                        add(m, f"{n.name}.{m.name}")
    return out


def _other_symbols(path: str, text: str) -> list[dict]:
    ext, lines, out = Path(path).suffix.lower(), text.splitlines(), []
    for exts, pat, kind in OTHER:
        if ext not in exts:
            continue
        for i, ln in enumerate(lines):
            m = pat.match(ln)
            if not m:
                continue
            k = kind or (m.groupdict().get("kind") or "symbol")
            end = i
            if ln.rstrip().endswith("{") or "{" in ln:            # to the matching brace, roughly
                depth = 0
                for j in range(i, min(len(lines), i + MAX_UNIT_LINES)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    end = j
                    if depth <= 0 and j > i:
                        break
            out.append({"path": path, "name": m.group("name"), "kind": k,
                        "sig": " ".join((m.groupdict().get("sig") or "").split())[:200],
                        "start": i + 1, "end": end + 1, "doc": ""})
    return out


def symbols(path: str, text: str) -> list[dict]:
    return _py_symbols(path, text) if path.endswith(".py") else _other_symbols(path, text)


def interface_key(s: dict) -> str:
    return f"{s['path']}::{s['name']}{s['sig']}"


def is_core(path: str, core: list[str]) -> bool:
    return not core or any(fnmatch.fnmatch(path, g) for g in core)


def scan(root: str | Path, core: list[str] | None = None) -> dict:
    root, core = Path(root).expanduser().resolve(), list(core or [])
    files, syms = [], []
    for p in rag.folder_files(root):
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() in (".pdf", ".docx"):
            continue
        data = p.read_bytes()
        text = data.decode("utf-8", errors="replace")
        kind = ("test" if TEST_RE.search(rel) else "build" if p.name in BUILD_FILES
                else "doc" if p.suffix.lower() in DOC_EXT else "source")
        files.append({"path": rel, "kind": kind, "lines": text.count("\n") + 1, "sha": sha(data)})
        if kind == "source":
            syms += symbols(rel, text)
    interfaces = sorted({interface_key(s) for s in syms if is_core(s["path"], core)})
    head = git(root, "rev-parse", "HEAD")
    return {"repo": root.name, "commit": head, "dirty": bool(git(root, "status", "--porcelain")) if head else None,
            "core": core, "fingerprint": sha("\n".join(interfaces)), "interfaces": interfaces,
            "files": files, "symbols": syms, "scanned": time.strftime("%Y-%m-%dT%H:%M:%S")}


def units(sc: dict, root: str | Path, doc_lines: int = 60) -> list[dict]:
    """What the teacher is shown: each public symbol's source, and the documents in pieces."""
    root, out, cache = Path(root).expanduser().resolve(), [], {}

    def lines_of(path):
        if path not in cache:
            cache[path] = (root / path).read_text(errors="replace").splitlines()
        return cache[path]

    for s in sc["symbols"]:
        if "." in s["name"] and s["name"].split(".")[-1] == "__init__":
            continue
        start, end = s["start"], min(s["end"], s["start"] + MAX_UNIT_LINES - 1)
        out.append({"id": f"{s['path']}::{s['name']}", "path": s["path"], "start": start, "end": end,
                    "kind": s["kind"], "text": "\n".join(lines_of(s["path"])[start - 1:end])})
    for f in sc["files"]:
        if f["kind"] != "doc":
            continue
        ls = lines_of(f["path"])
        for a in range(0, len(ls), doc_lines):
            piece = "\n".join(ls[a:a + doc_lines])
            if piece.strip():
                out.append({"id": f"{f['path']}#{a + 1}", "path": f["path"], "start": a + 1,
                            "end": min(len(ls), a + doc_lines), "kind": "doc", "text": piece})
    return out


# ---------------------------------------------------------------- the teacher

class Teacher:
    """An OpenAI-compatible model. The key comes from the environment only, and is never stored."""

    def __init__(self, spec: str, key_env: str = "OC_TEACHER_KEY", max_tokens: int = 2048, timeout: float = 300):
        import testqa as tq
        _, self.url, self.model = tq.parse_endpoint("teacher=" + spec)
        self._tq = tq
        self._a = SimpleNamespace(temperature=0.2, max_tokens=max_tokens, api_key=os.environ.get(key_env, ""),
                                  request_timeout=timeout)

    @property
    def name(self) -> str:
        """What the manifest records: the model id and host, no path, query or key."""
        from urllib.parse import urlsplit
        return f"{self.model or '(default)'} at {urlsplit(self.url).hostname}"

    def __call__(self, prompt: str) -> str:
        return self._tq.strip_think(self._tq.ask(self.url, self.model, prompt, self._a)).strip()


def numbered(u: dict) -> str:
    return "\n".join(f"{u['start'] + i:5d}  {ln}" for i, ln in enumerate(u["text"].splitlines()))


QA_PROMPT = """You are writing training questions about the repository "{repo}". Below is {what} from `{path}`, \
lines {start} to {end}, with line numbers.

```
{code}
```

Write {n} questions a developer working on this repository might ask, each answered only from the lines above: \
what it does, its parameters and return value, how to call it, its edge cases and errors, how it relates to the \
rest of the file. Put names of functions, classes and variables in backticks. Don't ask about anything the lines \
don't show.

Reply with only a JSON array: [{{"question": "...", "answer": "...", "lines": [first, last]}}], where lines are \
the line numbers the answer rests on."""

JUDGE_PROMPT = """Source: `{path}`, lines {start} to {end}:

```
{code}
```

Question: {question}
Proposed answer: {answer}

Is every claim in the proposed answer supported by the source above? Reply SUPPORTED, or UNSUPPORTED followed \
by the claim that is not supported."""


def parse_pairs(reply: str) -> list[dict]:
    m = re.search(r"\[.*\]", reply, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        q, a, ln = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip(), d.get("lines")
        if q and a:
            try:
                lines = [int(ln[0]), int(ln[-1])] if isinstance(ln, list) and ln else None
            except (TypeError, ValueError):
                lines = None
            out.append({"question": q, "answer": a, "lines": lines})
    return out


def generate(sc: dict, us: list[dict], teacher, n: int = 3, limit: int | None = None,
             done: dict | None = None, on_pair=None) -> list[dict]:
    """Pairs for each unit; units already in `done` (unit id -> pairs) are not asked again."""
    done, out = dict(done or {}), []
    for u in us[:limit] if limit else us:
        if u["id"] in done:
            out += done[u["id"]]
            continue
        what = "documentation" if u["kind"] == "doc" else f"the {u['kind']} `{u['id'].split('::')[-1]}`"
        try:
            reply = teacher(QA_PROMPT.format(repo=sc["repo"], what=what, path=u["path"], start=u["start"],
                                             end=u["end"], code=numbered(u), n=n))
        except Exception as e:                                   # one failed request doesn't end the run
            print(f"  {u['id']}: teacher failed ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            continue
        pairs = [{**p, "unit": u["id"], "path": u["path"], "unit_lines": [u["start"], u["end"]],
                  "id": sha(u["id"] + "\0" + p["question"])[:12]} for p in parse_pairs(reply)[:n]]
        done[u["id"]] = pairs
        out += pairs
        if on_pair:
            on_pair(u["id"], pairs)
    return out


# ---------------------------------------------------------------- 3. judge

def repo_identifiers(sc: dict, root: str | Path) -> set[str]:
    """Every identifier in the repository's text, for catching names the teacher made up."""
    root, names = Path(root).expanduser().resolve(), set()
    for f in sc["files"]:
        names |= {f["path"], Path(f["path"]).name}                # `README.md` and `src/app.py` are real names too
        try:
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", (root / f["path"]).read_text(errors="replace")))
        except OSError:
            pass
    return names


def mechanical(pair: dict, idents: set[str]) -> str | None:
    """Why a pair fails the cheap checks, or None."""
    lo, hi = pair["unit_lines"]
    ln = pair.get("lines")
    if not ln or not (lo <= ln[0] <= ln[1] <= hi):
        return f"cited lines {ln} are not inside the lines shown ({lo}-{hi})"
    made_up = sorted({p for name in IDENT_RE.findall(pair["question"] + " " + pair["answer"]) if name not in idents
                      for p in name.split(".") if p and p not in idents})
    if made_up:
        return "names not in the repository: " + ", ".join(made_up[:5])
    return None


def judge(pairs: list[dict], us: list[dict], sc: dict, root, model=None, done: dict | None = None,
          on_verdict=None) -> list[dict]:
    """Each pair with `verdict` (kept / rejected) and `why`."""
    by_unit, idents, done, out = {u["id"]: u for u in us}, repo_identifiers(sc, root), dict(done or {}), []
    for p in pairs:
        if p["id"] in done:
            out.append({**p, **done[p["id"]]})
            continue
        why = mechanical(p, idents)
        if why is None and model is not None:
            u = by_unit[p["unit"]]
            try:
                r = model(JUDGE_PROMPT.format(path=u["path"], start=u["start"], end=u["end"], code=numbered(u),
                                              question=p["question"], answer=p["answer"]))
            except Exception as e:
                r = f"UNSUPPORTED (the judge failed: {type(e).__name__})"
            if not r.strip().upper().startswith("SUPPORTED"):
                why = "judge: " + " ".join(r.split())[:200]
        v = {"verdict": "rejected" if why else "kept", "why": why or ""}
        done[p["id"]] = v
        out.append({**p, **v})
        if on_verdict:
            on_verdict(p["id"], v)
    return out


# ---------------------------------------------------------------- 4. build

def user_prompt(repo: str, question: str) -> str:
    return f"In the {repo} repository: {question}"


def build(sc: dict, judged: list[dict], out: str | Path, replay: list[str] | None = None,
          replay_share: float = 0.15, holdout: float = 0.1, seed: int = 0, teacher: str = "",
          judge_name: str = "") -> dict:
    out, rng = Path(out), random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    kept = [p for p in judged if p["verdict"] == "kept"]
    unit_ids = sorted({p["unit"] for p in kept})
    rng.shuffle(unit_ids)
    n_hold = max(1, round(len(unit_ids) * holdout)) if len(unit_ids) >= 5 and holdout > 0 else 0
    held = set(unit_ids[:n_hold])                                 # whole symbols: never half in training
    train = [p for p in kept if p["unit"] not in held]
    test = [p for p in kept if p["unit"] in held]

    pool = []
    for f in replay or []:
        for ln in Path(f).read_text().splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if isinstance(rec.get("messages"), list):
                pool.append(rec["messages"])
    want = round(len(train) * replay_share / (1 - replay_share)) if replay_share > 0 else 0
    rng.shuffle(pool)
    rep = pool[:want]
    warn = None
    if want and len(rep) < want:
        warn = (f"only {len(rep)} replay rows for the {want} that a {replay_share:.0%} share needs; add more with "
                "--replay, or the model may forget more of what it knew")

    sft = [{"messages": [{"role": "user", "content": user_prompt(sc["repo"], p["question"])},
                         {"role": "assistant", "content": p["answer"]}], "source": "repo", "pair": p["id"]}
           for p in train] + [{"messages": m, "source": "replay"} for m in rep]
    rng.shuffle(sft)
    (out / "sft.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sft))
    (out / "holdout.jsonl").write_text("".join(json.dumps(
        {"id": p["id"], "prompt": user_prompt(sc["repo"], p["question"]), "answer": p["answer"],
         "path": p["path"], "lines": p["lines"]}, ensure_ascii=False) + "\n" for p in test))
    sha_of = {f["path"]: f["sha"] for f in sc["files"]}
    manifest = {
        "kind": "omnicortex.repo_adapt", "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo": sc["repo"], "commit": sc["commit"], "dirty": sc["dirty"], "core": sc["core"],
        "fingerprint": sc["fingerprint"], "interfaces": sc["interfaces"],
        "teacher": teacher, "judge": judge_name,
        "counts": {"units": len({p["unit"] for p in judged}), "pairs": len(judged), "kept": len(kept),
                   "rejected": len(judged) - len(kept), "train": len(train), "holdout": len(test),
                   "replay": len(rep), "replay_wanted": want},
        "mix": {"repo": len(train), "replay": len(rep),
                "replay_share": round(len(rep) / max(1, len(train) + len(rep)), 4)},
        "warnings": [warn] if warn else [],
        "pairs": [{"id": p["id"], "unit": p["unit"], "path": p["path"], "lines": p["lines"],
                   "file_sha": sha_of.get(p["path"]), "verdict": p["verdict"], "why": p["why"],
                   "split": "holdout" if p["unit"] in held else ("train" if p["verdict"] == "kept" else None)}
                  for p in judged],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


# ---------------------------------------------------------------- staleness and size

def check(manifest: dict, root: str | Path, adapter: str | None = None, max_adapter_mb: float | None = None) -> dict:
    now = scan(root, manifest.get("core"))
    old, new = set(manifest["interfaces"]), set(now["interfaces"])
    by_name = lambda keys: {k.split("(")[0]: k for k in keys}           # noqa: E731
    o, n = by_name(old - new), by_name(new - old)
    rep = {"fresh": now["fingerprint"] == manifest["fingerprint"],
           "commit_then": manifest.get("commit"), "commit_now": now["commit"],
           "changed": sorted(k for k in o if k in n), "removed": sorted(k for k in o if k not in n),
           "added": sorted(k for k in n if k not in o)}
    if adapter:
        mb = Path(adapter).stat().st_size / 2 ** 20
        rep["adapter_mb"] = round(mb, 2)
        rep["adapter_ok"] = max_adapter_mb is None or mb <= max_adapter_mb
    return rep


# ---------------------------------------------------------------- the command line

def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc_path = out / "scan.json"
    sc = scan(a.repo, a.core)
    old = _load(sc_path, None)
    if old and old["fingerprint"] != sc["fingerprint"]:
        print("the repository's interfaces changed since the last run: starting the pairs again", file=sys.stderr)
        for f in ("pairs.json", "verdicts.json"):
            (out / f).unlink(missing_ok=True)
    sc_path.write_text(json.dumps(sc, indent=1))
    us = units(sc, a.repo)
    print(f"scan: {len(sc['files'])} files, {len(sc['symbols'])} public symbols, {len(us)} units, "
          f"commit {(sc['commit'] or 'none')[:12]}, fingerprint {sc['fingerprint'][:12]}", flush=True)
    if a.scan_only:
        return sc

    teacher = Teacher(a.teacher, a.key_env, a.max_tokens)
    judge_model = Teacher(a.judge, a.key_env, 512) if a.judge else (None if a.no_judge_model else teacher)
    pairs_path, verdicts_path = out / "pairs.json", out / "verdicts.json"
    done = _load(pairs_path, {})

    def save_pairs(uid, ps):
        done[uid] = ps
        pairs_path.write_text(json.dumps(done))
        print(f"  qa {len(done)}/{min(len(us), a.limit or len(us))}: {uid} ({len(ps)} pairs)", flush=True)
    pairs = generate(sc, us, teacher, a.per_unit, a.limit, done, save_pairs)

    verdicts = _load(verdicts_path, {})

    def save_verdict(pid, v):
        verdicts[pid] = v
        if len(verdicts) % 10 == 0:
            verdicts_path.write_text(json.dumps(verdicts))
    judged = judge(pairs, us, sc, a.repo, judge_model, verdicts, save_verdict)
    verdicts_path.write_text(json.dumps(verdicts))
    m = build(sc, judged, out, a.replay, a.replay_share, a.holdout, a.seed, teacher.name,
              judge_model.name if judge_model else "checks only")
    c = m["counts"]
    print(f"pairs: {c['kept']} kept, {c['rejected']} rejected; train {c['train']} + replay {c['replay']}, "
          f"holdout {c['holdout']}", flush=True)
    for w in m["warnings"]:
        print("warning: " + w, flush=True)
    print(f"next: python scripts/finetune.py --model <base> --data {out} --method lora --lora-r {a.lora_r}; then "
          f"merge_export.py --lora-gguf, and repo_adapt.py check --manifest {out / 'manifest.json'}", flush=True)
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="scan, ask, judge and build (resumes)")
    r.add_argument("--repo", required=True, help="the repository's folder")
    r.add_argument("--out", required=True, help="output folder (an existing one resumes)")
    r.add_argument("--teacher", help="URL@model of the teacher (OpenAI-compatible)")
    r.add_argument("--judge", help="URL@model of a separate judge (default: the teacher)")
    r.add_argument("--no-judge-model", action="store_true", help="only the cheap checks, no judge model")
    r.add_argument("--key-env", default="OC_TEACHER_KEY", help="environment variable holding the API key")
    r.add_argument("--core", action="append", default=[], help="glob of core paths (repeatable; default: all)")
    r.add_argument("--per-unit", type=int, default=3, help="pairs per symbol or document piece")
    r.add_argument("--limit", type=int, help="ask about at most this many units (a trial run)")
    r.add_argument("--replay", action="append", default=[], help="general chat-format JSONL to mix in")
    r.add_argument("--replay-share", type=float, default=0.15)
    r.add_argument("--holdout", type=float, default=0.1, help="share of symbols held out")
    r.add_argument("--lora-r", type=int, default=16, help="the LoRA rank suggested for training (16 or 32)")
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--scan-only", action="store_true", help="scan and stop")
    c = sub.add_parser("check", help="is an adapter still current, and within its size budget?")
    c.add_argument("--manifest", required=True)
    c.add_argument("--repo", required=True)
    c.add_argument("--adapter", help="the adapter file, for the size check")
    c.add_argument("--max-adapter-mb", type=float)
    a = p.parse_args(argv)
    if a.cmd == "run":
        if not Path(a.repo).expanduser().is_dir():
            p.error(f"not a folder: {a.repo}")
        if not a.scan_only and not a.teacher:
            p.error("run needs --teacher (or --scan-only)")
        run(a)
        return 0
    rep = check(json.loads(Path(a.manifest).read_text()), a.repo, a.adapter, a.max_adapter_mb)
    print(json.dumps(rep, indent=1))
    if not rep["fresh"]:
        print("stale: the repository's core interfaces changed since the adapter was trained", file=sys.stderr)
    if rep.get("adapter_ok") is False:
        print(f"the adapter is {rep['adapter_mb']} MB, over its {a.max_adapter_mb} MB budget", file=sys.stderr)
    return 0 if rep["fresh"] and rep.get("adapter_ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
