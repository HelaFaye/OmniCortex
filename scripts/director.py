#!/usr/bin/env python3
"""
Project director: split a project into skill-labelled tasks, keep a plan,
hand tasks to worker models, and keep a person in charge of every decision
that matters.

The pieces, in the order a project goes through them:

  analyze    A description (markdown, a list of skills, an issue) becomes
             candidate tasks, each labelled by subject_classifier (several
             labels when a task spans skills, "unknown" when the text is too
             thin to tell). A local checkout can be surveyed too: file types,
             build files and imports say which skills the code needs, and
             skills the code needs but no task mentions are pointed out.
  plan       Tasks, dependencies, acceptance criteria and an assigned model
             each, in a versioned JSON file. While the plan is a draft anyone
             may edit it. Once a person approves it, the director sticks to
             it: it runs only approved tasks, in dependency order, and any
             change it wants (a follow-up task, a different model after
             repeated rejections) is a *proposal* that a person accepts or
             rejects. A person's own edits apply at once. Every change bumps
             the version and lands in the history with who made it and why.
  assign     Each task gets the model whose measured, per-subject results
             (model_stats) best fit the task's labels; with no stats, the
             policy's default model, else the largest model that fits.
             A person can pin any task to any model.
  run        A worker gets the task, the project goal, the accepted output of
             the tasks it depends on and any reviewer feedback, and ends its
             reply with a small JSON report (done or blocked, follow-ups,
             questions). Workers produce text: patches, analyses, drafts.
             Nothing here runs a command or writes into a repository; applying
             a result is a person's step.
  review     By policy, every result waits for a person (default), only
             results the activation check flagged or that report a problem,
             or none. A rejected result goes back with the feedback; after
             max_attempts rejections the task is blocked and the director
             proposes the next-best model.

Studio runs this (/projects, /api/projects) and starts the worker models. The
module itself has no HTTP and starts nothing, so it is testable on its own:

    python scripts/director.py analyze notes.md [--repo PATH]
    python scripts/director.py show ~/.omnicortex/projects/<id>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import oc_requirements as nrq
import subject_classifier as sc

PLAN_STATES = ("draft", "approved", "running", "paused", "done")
TASK_STATES = ("todo", "running", "review", "rework", "done", "blocked", "dropped")
REVIEW_POLICIES = ("all", "flagged", "none")
EDITABLE = ("title", "detail", "labels", "depends_on", "acceptance", "model", "permissions")
MAX_RESULT_CHARS = 100_000
RETRY_DELAY = 15.0
DEFAULT_POLICY = {"review": "all", "max_parallel": 2, "max_attempts": 3, "default_model": None,
                  "director_model": None, "check": True,
                  # Hardware for this project: device ids it may use ([] = any enabled device),
                  # and llama-server settings per device on top of the device's own.
                  "devices": [], "device_settings": {},
                  # A node graph (graph_engine.py) that screens each result before review: inputs task and
                  # result; output "accepted" passes it, "feedback" fails it. verify_auto sends a failed
                  # result back to the worker with that feedback instead of waiting for a person.
                  "verify_graph": None, "verify_auto": False}
DEVICE_SETTINGS = ("ngl", "ctx", "batch", "threads", "flash_attn", "cache_type", "parallel")


class PlanError(ValueError):
    pass


# ================================================================ analysis

_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
_LEADIN = re.compile(r"^\s*(?:\*\*)?([A-Za-z][\w /&+.'-]{1,48}?)(?:\*\*)?\s*:\s+(\S.*)$")


def _clean(s: str) -> str:
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def split_units(text: str) -> list[dict]:
    """Markdown -> candidate work items: list items, "Name: description"
    lines, table rows and plain paragraphs, with the heading they sit under.
    Code blocks are skipped. A unit's continuation lines are joined to it."""
    units, section, cur, in_code, header_row = [], "", None, False, None

    def flush():
        nonlocal cur
        if cur and len(cur["text"].split()) >= 4:
            units.append(cur)
        cur = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            flush()
            continue
        if in_code:
            continue
        if not line.strip():
            flush()
            header_row = None
            continue
        h = _HEADING.match(line)
        if h:
            flush()
            section = _clean(h.group(2))
            continue
        if line.lstrip().startswith("|"):
            flush()
            cells = [_clean(c) for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c or "--") for c in cells):
                continue
            if header_row is None:
                header_row = cells
                continue
            units.append({"text": " — ".join(c for c in cells if c), "section": section, "kind": "row"})
            continue
        b = _BULLET.match(line)
        if b:
            flush()
            cur = {"text": _clean(b.group(1)), "section": section, "kind": "item"}
            continue
        if cur is not None and (raw.startswith((" ", "\t")) or cur["kind"] == "para"):
            cur["text"] += " " + _clean(line)
            continue
        flush()
        cur = {"text": _clean(line), "section": section, "kind": "para"}
    flush()
    return units


def _title(text: str) -> tuple[str, str]:
    m = _LEADIN.match(text)
    if m and len(m.group(1).split()) <= 6:
        return m.group(1).strip(), m.group(2).strip()
    words = text.split()
    t = " ".join(words[:8]).rstrip(".,;:")
    return (t + ("…" if len(words) > 8 else "")), text


def analyze_text(text: str, clf: sc.SubjectClassifier | None = None) -> list[dict]:
    """-> candidate tasks [{title, detail, labels, proba, why, section}], in
    document order. Labels come from the subject classifier on the whole unit
    (title and detail), so "Graphics: shaders for the viewer" counts the word
    graphics too."""
    clf = clf or sc.default_classifier()
    out, seen = [], set()
    units = split_units(text)
    if any(u["kind"] != "para" for u in units):
        # A document with lists or tables keeps its work items there; its
        # prose paragraphs are context, not tasks.
        units = [u for u in units if u["kind"] != "para" or _LEADIN.match(u["text"])]
    for u in units:
        key = u["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        title, detail = _title(u["text"])
        a = clf.analyze(u["text"])
        out.append({"title": title, "detail": detail, "labels": a["labels"],
                    "proba": {k: v for k, v in a["proba"].items() if v >= 0.05},
                    "why": a["why"], "section": u["section"], "source": u["kind"]})
    return out


def skill_breakdown(tasks: list[dict]) -> dict:
    """Subject -> task ids (or titles), the split of a project by skill."""
    out: dict[str, list] = {}
    for t in tasks:
        if t.get("status") == "dropped":
            continue
        for lb in (t.get("labels") or [sc.UNKNOWN]):
            out.setdefault(lb, []).append(t.get("id") or t["title"])
    return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


# ---------------------------------------------------------------- repo survey

EXT_SUBJECT = {
    "graphics": {".glsl", ".vert", ".frag", ".comp", ".geom", ".tesc", ".tese", ".hlsl", ".wgsl", ".metal",
                 ".spv", ".shader", ".gltf", ".glb", ".fbx", ".mtl"},
    "reverse-engineering": {".s", ".asm", ".ld", ".sha1", ".idb", ".i64", ".gpr", ".sym"},
    "code": {".py", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".rs", ".go", ".js", ".mjs", ".ts", ".tsx",
             ".java", ".kt", ".cs", ".rb", ".swift", ".lua", ".zig", ".gd"},
    "writing": {".md", ".rst", ".adoc"},
}
BUILD_FILES = {"cmakelists.txt", "makefile", "gnumakefile", "meson.build", "build", "build.bazel", "workspace",
               "xmake.lua", "premake5.lua", "sconstruct", "dockerfile", "docker-compose.yml", "compose.yaml",
               "configure.ac", "cargo.toml", "go.mod", "package.json", "pyproject.toml", "setup.py",
               "build.gradle", "build.gradle.kts", "pom.xml", "justfile", "flake.nix", "vcpkg.json",
               "conanfile.txt", "conanfile.py", "build.zig", "project.godot"}
RE_DIRS = {"asm", "disasm", "decomp", "decompiled", "ghidra"}
IMPORT_SUBJECT = [
    (re.compile(r"^\s*(?:import|from)\s+(PIL|cv2|skimage|imageio)\b", re.M), "vision"),
    (re.compile(r"^\s*(?:import|from)\s+(numpy|scipy|sympy)\b", re.M), "math"),
    (re.compile(r"^\s*(?:import|from)\s+(OpenGL|moderngl|pygfx|wgpu|vulkan|trimesh|open3d|pyrender)\b", re.M),
     "graphics"),
    (re.compile(r"(?:from\s+['\"]three['\"]|THREE\.|getContext\(\s*['\"]webgl2?['\"])"), "graphics"),
    (re.compile(r"#include\s*[<\"](?:GL/|vulkan/|SDL3?/SDL_gpu|d3d1[12]|Metal/)"), "graphics"),
]
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", "target", ".cache",
             ".idea", ".vscode", "out"}


def analyze_repo(path: str, max_files: int = 30000) -> dict:
    """Which skills does this checkout need? File types, build files and
    imports, counted. Reads names and the first 4 KB of source files only."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise PlanError(f"not a directory: {path}")
    counts: dict[str, int] = {}
    examples: dict[str, list] = {}
    build, n = [], 0

    def note(subject, rel):
        counts[subject] = counts.get(subject, 0) + 1
        ex = examples.setdefault(subject, [])
        if len(ex) < 5:
            ex.append(rel)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(root)
        in_re = any(part.lower() in RE_DIRS for part in rel_dir.parts)
        for fn in sorted(filenames):
            n += 1
            if n > max_files:
                break
            rel = str(rel_dir / fn)
            low, ext = fn.lower(), os.path.splitext(fn.lower())[1]
            if low in BUILD_FILES or low.endswith((".cmake", ".mk")) or \
                    (rel_dir.parts[:2] == (".github", "workflows")):
                build.append(rel)
                note("systems", rel)
            if in_re:
                note("reverse-engineering", rel)
            for subject, exts in EXT_SUBJECT.items():
                if ext in exts and not (subject == "code" and in_re):
                    note(subject, rel)
            if ext in (".py", ".js", ".mjs", ".ts", ".html", ".c", ".cpp", ".h", ".hpp"):
                try:
                    with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="ignore") as f:
                        head = f.read(4096)
                except OSError:
                    continue
                for rx, subject in IMPORT_SUBJECT:
                    if rx.search(head):
                        note(subject, rel)
        if n > max_files:
            break
    subjects = {s: {"files": counts[s], "examples": examples[s]}
                for s in sorted(counts, key=lambda k: -counts[k])}
    return {"path": str(root), "files": min(n, max_files), "truncated": n > max_files,
            "subjects": subjects, "build": build[:20]}


def coverage_gaps(tasks: list[dict], census: dict | None, min_files: int = 3) -> list[str]:
    """Skills the code needs that no task mentions."""
    if not census:
        return []
    covered = {lb for t in tasks if t.get("status") != "dropped" for lb in t.get("labels") or []}
    return [s for s, v in census.get("subjects", {}).items()
            if v["files"] >= min_files and s not in covered and s != "writing"]


# ================================================================ the plan

def _now() -> float:
    return round(time.time(), 3)


def new_plan(title: str, goal: str, tasks: list[dict], source: str = "", census: dict | None = None) -> dict:
    pid = time.strftime("%Y%m%d-") + uuid.uuid4().hex[:8]
    plan = {"id": pid, "title": title.strip() or "Untitled project", "goal": goal.strip(), "status": "draft",
            "version": 1, "created": _now(), "updated": _now(), "policy": dict(DEFAULT_POLICY),
            "source": source[:200_000], "census": census, "tasks": [], "proposals": [], "history": [],
            "next_task": 1, "next_proposal": 1,
            "requirements": {"root": None, "venv": None, "items": [], "files": [], "last": None}}
    if census and census.get("path"):
        try:
            scan_requirements(plan, census["path"], log=False)
        except (ValueError, OSError):
            pass
    for t in tasks:
        _add_task(plan, t)
    _log(plan, "director", "created", f"{len(plan['tasks'])} task(s) from the description", [])
    return plan


def _task_id(plan) -> str:
    tid = f"T{plan['next_task']}"
    plan["next_task"] += 1
    return tid


def _add_task(plan: dict, t: dict) -> dict:
    task = {"id": _task_id(plan), "title": str(t.get("title") or "").strip()[:200] or "untitled",
            "detail": str(t.get("detail") or "").strip()[:20_000],
            "labels": [x for x in (t.get("labels") or []) if x in sc.SUBJECTS],
            "proba": t.get("proba") or {}, "why": t.get("why", ""),
            "depends_on": list(t.get("depends_on") or []), "acceptance": str(t.get("acceptance") or "")[:4000],
            "model": t.get("model") or None, "assignee": None, "status": "todo", "attempts": [],
            "questions": [], "section": t.get("section", "")}
    plan["tasks"].append(task)
    return task


def task(plan: dict, tid: str) -> dict:
    for t in plan["tasks"]:
        if t["id"] == tid:
            return t
    raise PlanError(f"no task {tid}")


def _log(plan, by, action, reason, changes):
    plan["history"].append({"version": plan["version"], "at": _now(), "by": by, "action": action,
                            "reason": reason, "changes": changes})


def _check_graph(plan: dict) -> None:
    ids = {t["id"] for t in plan["tasks"]}
    deps = {t["id"]: [d for d in t["depends_on"]] for t in plan["tasks"]}
    for tid, ds in deps.items():
        for d in ds:
            if d not in ids:
                raise PlanError(f"{tid} depends on unknown task {d}")
            if d == tid:
                raise PlanError(f"{tid} depends on itself")
    state = {}

    def visit(n, path):
        if state.get(n) == 1:
            raise PlanError("dependency cycle: " + " -> ".join(path + [n]))
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in deps[n]:
            visit(d, path + [n])
        state[n] = 2

    for n in deps:
        visit(n, [])


def _apply(plan: dict, changes: list[dict], by: str) -> list[dict]:
    """Validate change ops on a copy, then apply them to the plan in place
    (task dicts keep their identity, so a worker holding one stays current).
    Raises PlanError and leaves the plan untouched when any op is invalid."""
    trial = copy.deepcopy(plan)
    _ops(trial, changes, by)
    _check_graph(trial)
    return _ops(plan, changes, by)


def _ops(work: dict, changes: list[dict], by: str) -> list[dict]:
    done = []
    for ch in changes:
        op = ch.get("op")
        if op == "add":
            t = _add_task(work, ch.get("task") or {})
            done.append({"op": "add", "id": t["id"], "title": t["title"]})
        elif op == "update":
            t = task(work, ch.get("id"))
            if t["status"] == "running":
                raise PlanError(f"{t['id']} is running; change it after the attempt finishes")
            fields = {k: v for k, v in (ch.get("fields") or {}).items() if k in EDITABLE}
            if not fields:
                raise PlanError(f"nothing editable in the update to {t['id']}")
            if "labels" in fields:
                bad = [x for x in fields["labels"] if x not in sc.SUBJECTS]
                if bad:
                    raise PlanError(f"unknown subject(s) {bad}; known: {sc.SUBJECTS}")
            for k, v in fields.items():
                t[k] = list(v) if k in ("labels", "depends_on") else (v or None if k == "model" else str(v))
            if "model" in fields:
                t["assignee"] = ({"model": fields["model"], "reason": f"pinned by {by}"}
                                 if fields["model"] else None)
            done.append({"op": "update", "id": t["id"], "fields": sorted(fields)})
        elif op == "drop":
            t = task(work, ch.get("id"))
            if t["status"] == "running":
                raise PlanError(f"{t['id']} is running")
            t["status"] = "dropped"
            done.append({"op": "drop", "id": t["id"]})
        elif op == "reopen":
            t = task(work, ch.get("id"))
            if t["status"] not in ("done", "dropped", "blocked"):
                raise PlanError(f"{t['id']} is {t['status']}, nothing to reopen")
            t["status"] = "todo"
            done.append({"op": "reopen", "id": t["id"]})
        elif op == "assign":
            t = task(work, ch.get("id"))
            t["assignee"] = {"model": ch.get("model"), "reason": ch.get("reason") or f"set by {by}"}
            done.append({"op": "assign", "id": t["id"], "model": ch.get("model")})
        else:
            raise PlanError(f"unknown change op {op!r}")
    return done


def edit(plan: dict, changes: list[dict], by: str = "human", reason: str = "") -> dict:
    """Change the plan. A person's edits apply at once in any state. The
    director's own edits apply directly only while the plan is a draft; after
    approval they become a pending proposal. -> {"applied": [...]} or
    {"proposal": {...}}."""
    if not changes:
        raise PlanError("no changes")
    if by != "human" and plan["status"] != "draft":
        return {"proposal": propose(plan, changes, by, reason)}
    applied = _apply(plan, changes, by)
    plan["version"] += 1
    plan["updated"] = _now()
    _log(plan, by, "edit", reason, applied)
    _maybe_done(plan)
    return {"applied": applied}


def propose(plan: dict, changes: list[dict], by: str = "director", reason: str = "") -> dict:
    trial = copy.deepcopy(plan)
    _apply(trial, changes, by)             # validate now, so a bad proposal never waits for a person
    p = {"id": f"P{plan['next_proposal']}", "at": _now(), "by": by, "reason": reason, "changes": changes,
         "status": "pending", "base_version": plan["version"]}
    plan["next_proposal"] += 1
    plan["proposals"].append(p)
    plan["updated"] = _now()
    return p


def decide(plan: dict, pid: str, accept: bool, by: str = "human", note: str = "") -> dict:
    p = next((x for x in plan["proposals"] if x["id"] == pid), None)
    if p is None:
        raise PlanError(f"no proposal {pid}")
    if p["status"] != "pending":
        raise PlanError(f"{pid} is already {p['status']}")
    if accept:
        applied = _apply(plan, p["changes"], p["by"])
        plan["version"] += 1
        _log(plan, by, "accepted " + pid, p["reason"] + (f" ({note})" if note else ""), applied)
    else:
        _log(plan, by, "rejected " + pid, note or p["reason"], [])
    p["status"] = "accepted" if accept else "rejected"
    p["decided_by"], p["decided_at"] = by, _now()
    plan["updated"] = _now()
    _maybe_done(plan)
    return p


def set_policy(plan: dict, values: dict, by: str = "human") -> dict:
    pol = plan["policy"]
    changed = {}
    for k, v in values.items():
        if k not in DEFAULT_POLICY:
            raise PlanError(f"unknown policy {k!r}")
        if k == "review" and v not in REVIEW_POLICIES:
            raise PlanError(f"review must be one of {REVIEW_POLICIES}")
        if k in ("max_parallel", "max_attempts"):
            v = max(1, min(16, int(v)))
        if k in ("check", "verify_auto"):
            v = bool(v)
        if k == "verify_graph":
            v = str(v).strip() or None if v else None
            if v and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,59}|template:[a-z0-9_-]+", v):
                raise PlanError("verify_graph: the name of a saved graph")
        if k == "devices":
            if not isinstance(v, list) or not all(isinstance(x, str) and re.fullmatch(r"[a-z]+:\d+", x) for x in v):
                raise PlanError("devices must be a list like [\"rocm:0\", \"vulkan:1\"]")
        if k == "device_settings":
            if not isinstance(v, dict):
                raise PlanError("device_settings must map a device id to settings")
            clean = {}
            for dev, st in v.items():
                if not re.fullmatch(r"[a-z]+:\d+", str(dev)) or not isinstance(st, dict):
                    raise PlanError(f"bad device settings for {dev!r}")
                bad = [x for x in st if x not in DEVICE_SETTINGS]
                if bad:
                    raise PlanError(f"{dev}: unknown setting(s) {bad}; allowed: {DEVICE_SETTINGS}")
                clean[dev] = st
            v = clean
        if pol.get(k) != v:
            pol[k] = v
            changed[k] = v
    if changed:
        _log(plan, by, "policy", "", [{"op": "policy", **changed}])
        plan["updated"] = _now()
    return pol


# ---------------------------------------------------------------- requirements

def _reqs(plan: dict) -> dict:
    return plan.setdefault("requirements", {"root": None, "venv": None, "items": [], "files": [], "last": None})


def scan_requirements(plan: dict, root: str, by: str = "director", log: bool = True) -> dict:
    """Infer the toolchain, libraries and packages from a checkout's build files.
    Items a person added or edited are kept; earlier inferred ones are replaced."""
    found = nrq.infer_project(root)
    r = _reqs(plan)
    human = [i for i in r["items"] if i.get("source") == "human"]
    names = {(i["kind"], i["name"].lower()) for i in human}
    r.update(root=found["root"], venv=found["venv"], files=found["files"], last=None,
             items=human + [i for i in found["items"] if (i["kind"], i["name"].lower()) not in names])
    if log:
        _log(plan, by, "requirements", f"scanned {found['root']}: {len(found['items'])} item(s) "
                                       f"from {len(found['files'])} file(s)", [])
        plan["updated"] = _now()
    return r


def set_requirements(plan: dict, items: list, by: str = "human", reason: str = "") -> dict:
    """Replace the requirement list (a person's edit: every item becomes theirs
    unless it was inferred and left as it was)."""
    try:
        clean = nrq.normalize_items(items)
    except ValueError as e:
        raise PlanError(str(e))
    r = _reqs(plan)
    before = {(i["kind"], i["name"].lower()): i for i in r["items"]}
    for i in clean:
        old = before.get((i["kind"], i["name"].lower()))
        same = old and all(old.get(k) == i.get(k) for k in ("need", "optional"))
        i["source"] = "inferred" if same and old.get("source") == "inferred" else "human"
        if same and not i["why"]:
            i["why"] = old.get("why", "")
    added = sorted({k for k in (x["kind"] + ":" + x["name"] for x in clean)} -
                   {k for k in (x["kind"] + ":" + x["name"] for x in r["items"])})
    removed = sorted({k for k in (x["kind"] + ":" + x["name"] for x in r["items"])} -
                     {k for k in (x["kind"] + ":" + x["name"] for x in clean)})
    r["items"], r["last"] = clean, None
    _log(plan, by, "requirements", reason, [{"op": "requirements", "added": added, "removed": removed}])
    plan["updated"] = _now()
    return r


def check_requirements(plan: dict, **kw) -> dict:
    r = _reqs(plan)
    r["last"] = nrq.check_project(r, **kw)
    return r["last"]


# ---------------------------------------------------------------- lifecycle

def approve(plan: dict, assign_fn=None, by: str = "human") -> dict:
    """Freeze the draft as the plan to follow. `assign_fn(task) -> {model,
    reason}` fills in models for tasks nobody pinned."""
    if plan["status"] != "draft":
        raise PlanError(f"plan is {plan['status']}, not a draft")
    live = [t for t in plan["tasks"] if t["status"] != "dropped"]
    if not live:
        raise PlanError("the plan has no tasks")
    _check_graph(plan)
    missing, why = [], []
    for t in live:
        if t.get("model"):
            t["assignee"] = {"model": t["model"], "reason": "pinned"}
        elif assign_fn is not None:
            t["assignee"] = assign_fn(t)
        if not (t.get("assignee") or {}).get("model"):
            missing.append(t["id"])
            r = (t.get("assignee") or {}).get("reason")
            if r and r not in why:
                why.append(r)
    if missing:
        raise PlanError(f"no model for {', '.join(missing)}" + (f" ({'; '.join(why)})" if why else "")
                        + ": pin one on the task, set a default model in the plan's policy, pick one under "
                          "Settings -> Models per subject, or add models (Studio -> Models)")
    plan["status"] = "approved"
    plan["version"] += 1
    plan["approved_version"] = plan["version"]
    _log(plan, by, "approved", f"{len(live)} task(s)", [])
    return plan


def set_running(plan: dict, running: bool, by: str = "human") -> dict:
    if running and plan["status"] not in ("approved", "paused"):
        raise PlanError(f"cannot start a plan that is {plan['status']}")
    if not running and plan["status"] != "running":
        raise PlanError("plan is not running")
    plan["status"] = "running" if running else "paused"
    _log(plan, by, "started" if running else "paused", "", [])
    plan["updated"] = _now()
    return plan


def ready_tasks(plan: dict) -> list[dict]:
    """Tasks the director may start now: todo or sent back for rework, with
    every dependency done, in plan order."""
    by_id = {t["id"]: t for t in plan["tasks"]}
    now = time.time()
    return [t for t in plan["tasks"] if t["status"] in ("todo", "rework")
            and (t.get("not_before") or 0) <= now
            and all(by_id[d]["status"] == "done" for d in t["depends_on"])]


def _maybe_done(plan: dict) -> None:
    live = [t for t in plan["tasks"] if t["status"] != "dropped"]
    if plan["status"] in ("running", "paused", "approved") and live and all(t["status"] == "done" for t in live):
        plan["status"] = "done"
        _log(plan, "director", "finished", f"all {len(live)} task(s) done", [])
    elif plan["status"] == "done" and any(t["status"] != "done" for t in live):
        plan["status"] = "paused"           # a reopened task: wait for a person to restart


# ---------------------------------------------------------------- workers

WORKER_SYSTEM = (
    "You are a worker on a project run by a director and supervised by a person. Do only the task you are "
    "given; other tasks belong to other workers. Use the inputs from earlier tasks where they help. If you "
    "cannot do the task with what you have, say exactly what is missing instead of guessing. Do not invent "
    "file contents, APIs or results you have not seen.\n\n"
    "End your reply with a JSON block in ```json fences: {\"status\": \"done\" or \"blocked\", "
    "\"summary\": one sentence, \"followups\": [tasks this one revealed, each one sentence], "
    "\"questions\": [what you need from the person, if blocked]}.")


def worker_messages(plan: dict, t: dict, max_input_chars: int = 12_000) -> list[dict]:
    by_id = {x["id"]: x for x in plan["tasks"]}
    lines = [f"Project: {plan['title']}"]
    if plan.get("goal"):
        lines.append(f"Goal: {plan['goal']}")
    lines.append("Plan:")
    for x in plan["tasks"]:
        if x["status"] != "dropped":
            mark = "  <- your task" if x["id"] == t["id"] else ""
            lines.append(f"  {x['id']} [{x['status']}] {x['title']}{mark}")
    env = nrq.environment_brief((plan.get("requirements") or {}).get("last"))
    if env:
        lines += ["", env]
    lines += ["", f"Your task ({t['id']}): {t['title']}", t["detail"]]
    if t.get("acceptance"):
        lines += ["", f"Done means: {t['acceptance']}"]
    # what the worker may do (scripts/permissions.py): job project_task, the task's own profile if set
    try:
        import permissions as perm
        model = (t.get("assignee") or {}).get("model") or ""
        lines += ["", perm.prompt_text(perm.Policy(), model=model, job="project_task",
                                       profile=t.get("permissions") or "",
                                       ctx=perm.context(workdir=plan.get("workdir") or plan.get("checkout")))]
    except (ImportError, ValueError):
        pass
    budget = max_input_chars
    for d in t["depends_on"]:
        dep = by_id[d]
        out = _accepted(dep)
        if out:
            piece = out[: max(500, budget // max(1, len(t["depends_on"])))]
            budget -= len(piece)
            lines += ["", f"Input from {d} ({dep['title']}), accepted by the reviewer:", piece]
    feedback = [a for a in t["attempts"] if a.get("review") and not a["review"].get("accepted")]
    for a in feedback[-2:]:
        lines += ["", f"Your attempt {a['n']} was sent back. Reviewer: {a['review'].get('feedback') or '(no note)'}"]
    answers = [q for q in t.get("questions", []) if q.get("answer")]
    for q in answers[-4:]:
        lines += ["", f"You asked: {q['q']}", f"Answer: {q['answer']}"]
    return [{"role": "system", "content": WORKER_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def _accepted(t: dict) -> str:
    for a in reversed(t["attempts"]):
        if (a.get("review") or {}).get("accepted"):
            return a["text"]
    return ""


def parse_report(text: str) -> dict:
    """The worker's closing JSON block -> {status, summary, followups,
    questions}. A missing or broken block reads as status "unreported", which
    the reviewer sees, rather than as success."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = blocks[-1] if blocks else None
    if cand is None:
        i = text.rfind('{"status"')
        cand = text[i:text.rfind("}") + 1] if i >= 0 else None
    rep = {"status": "unreported", "summary": "", "followups": [], "questions": []}
    if cand:
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                st = str(d.get("status", "")).lower()
                rep["status"] = st if st in ("done", "blocked") else "unreported"
                rep["summary"] = str(d.get("summary") or "")[:500]
                rep["followups"] = [str(x)[:500] for x in (d.get("followups") or []) if str(x).strip()][:8]
                rep["questions"] = [str(x)[:500] for x in (d.get("questions") or []) if str(x).strip()][:8]
        except (ValueError, TypeError):
            pass
    return rep


def start_attempt(plan: dict, t: dict) -> dict:
    if t["status"] not in ("todo", "rework"):
        raise PlanError(f"{t['id']} is {t['status']}")
    model = (t.get("assignee") or {}).get("model")
    a = {"n": len(t["attempts"]) + 1, "model": model, "started": _now(), "finished": None, "text": "",
         "report": None, "check": None, "error": None, "review": None}
    t["attempts"].append(a)
    t["status"] = "running"
    plan["updated"] = _now()
    return a


def yield_attempt(plan: dict, t: dict, reason: str) -> dict:
    """The computer the attempt ran on was taken back (its owner came back, a power limit): the task goes
    back to the queue at once, for another worker, and the attempt does not count as a failure."""
    a = t["attempts"][-1]
    a["finished"] = _now()
    a["yielded"] = str(reason)[:300]
    t["status"] = "todo"
    t["not_before"] = 0
    t["avoid"] = sorted(set(t.get("avoid") or []) | ({a["model"]} if a.get("model") else set()))
    plan["updated"] = _now()
    return a


def finish_attempt(plan: dict, t: dict, text: str | None, check: dict | None = None,
                   error: str | None = None, clf: sc.SubjectClassifier | None = None, fatal: bool = False,
                   screen: dict | None = None) -> dict:
    """Record a worker's answer and decide where it goes: review, done (when
    the policy allows), blocked (worker said so), or back to todo on error.
    screen: the verification graph's verdict ({"status": passed|failed|person|error, "feedback", "run"}):
    a failed result never skips review, and with verify_auto goes straight back to the worker.
    Follow-ups become a proposal for a person to accept."""
    a = t["attempts"][-1]
    a["finished"] = _now()
    pol = plan["policy"]
    if error:
        a["error"] = str(error)[:2000]
        errors = sum(1 for x in t["attempts"] if x.get("error"))
        if fatal or errors >= pol["max_attempts"]:
            t["status"] = "blocked"
            t["questions"].append({"q": f"Attempt {a['n']} failed and retrying will not help: {a['error'][:600]}",
                                   "answer": None, "attempt": a["n"]})
        else:
            t["status"] = "todo"
            t["not_before"] = _now() + RETRY_DELAY * 2 ** (errors - 1)     # back off: 15 s, 30 s, 60 s…
        plan["updated"] = _now()
        return a
    a["text"] = (text or "")[:MAX_RESULT_CHARS]
    a["check"] = check
    if screen:
        a["screen"] = screen
    rep = parse_report(a["text"])
    a["report"] = rep
    flagged = bool(check and check.get("flagged")) or bool(screen and screen.get("status") != "passed")
    if rep["status"] != "blocked" and screen and screen.get("status") == "failed" and pol.get("verify_auto"):
        t["status"] = "review"
        review(plan, t["id"], False, "The verification graph sent this back: " + str(screen.get("feedback") or "")[:3000],
               by="graph")
    elif rep["status"] == "blocked":
        t["status"] = "blocked"
        t["questions"] += [{"q": q, "answer": None, "attempt": a["n"]} for q in rep["questions"] or
                           ["The worker reported it is blocked; see its reply."]]
    elif pol["review"] == "none" or (pol["review"] == "flagged" and not flagged and rep["status"] == "done"):
        a["review"] = {"accepted": True, "by": "policy", "at": _now(), "feedback": ""}
        t["status"] = "done"
    else:
        t["status"] = "review"
    if rep["followups"]:
        clf = clf or sc.default_classifier()
        adds = []
        for f in rep["followups"]:
            title, detail = _title(f)
            adds.append({"op": "add", "task": {"title": title, "detail": detail, "labels": clf.analyze(f)["labels"],
                                               "depends_on": [t["id"]]}})
        try:
            propose(plan, adds, "director", f"follow-ups reported by the worker on {t['id']}")
        except PlanError:
            pass
    plan["updated"] = _now()
    _maybe_done(plan)
    return a


def review(plan: dict, tid: str, accept: bool, feedback: str = "", by: str = "human",
           next_model_fn=None) -> dict:
    """A person's verdict on the latest attempt. Rejections carry feedback to
    the next attempt; after max_attempts the task is blocked and, when
    `next_model_fn(task) -> {model, reason}` finds another model, the director
    proposes it."""
    t = task(plan, tid)
    if t["status"] not in ("review", "blocked") or not t["attempts"]:
        raise PlanError(f"{tid} has no result waiting for review")
    a = t["attempts"][-1]
    a["review"] = {"accepted": bool(accept), "by": by, "at": _now(), "feedback": feedback[:4000]}
    if accept:
        t["status"] = "done"
    else:
        rejected = sum(1 for x in t["attempts"] if x.get("review") and not x["review"]["accepted"])
        if rejected >= plan["policy"]["max_attempts"]:
            t["status"] = "blocked"
            alt = next_model_fn(t) if next_model_fn else None
            if alt and alt.get("model") and alt["model"] != (t.get("assignee") or {}).get("model"):
                propose(plan, [{"op": "assign", "id": tid, "model": alt["model"], "reason": alt.get("reason", "")},
                               {"op": "reopen", "id": tid}],
                        "director", f"{tid} was rejected {rejected} times with "
                                    f"{(t.get('assignee') or {}).get('model')}; try {alt['model']}")
        else:
            t["status"] = "rework"
    _log(plan, by, ("accepted " if accept else "sent back ") + tid, feedback[:300], [])
    plan["updated"] = _now()
    _maybe_done(plan)
    return a


def answer(plan: dict, tid: str, answers: list[str], by: str = "human") -> dict:
    """Answer a blocked task's questions and send it back to the queue."""
    t = task(plan, tid)
    open_q = [q for q in t["questions"] if not q.get("answer")]
    for q, ans in zip(open_q, answers):
        q["answer"] = str(ans)[:4000]
    if t["status"] == "blocked":
        t["status"] = "rework"
        t["not_before"] = 0
    _log(plan, by, "answered " + tid, "", [])
    plan["updated"] = _now()
    return t


# ================================================================ assignment

def context_need(t: dict, output_tokens: int = 2048) -> int:
    """Rough context a worker needs for this task: the instructions and plan
    (~1.5k tokens), the task text, inputs from dependencies (budgeted at up to
    12k characters), and room to answer. About 3.5 characters per token."""
    text = len(t.get("detail") or "") + len(t.get("acceptance") or "")
    deps = min(12_000, 4_000 * len(t.get("depends_on") or []))
    return max(4096, int(1500 + (text + deps) / 3.5 + output_tokens))


def assign(t: dict, candidates: list[dict], summaries: dict, policy: dict, min_graded: int = 20,
           min_subject: int = 5, hallucination_cost: float = 1.0, exclude: tuple = ()) -> dict:
    """-> {model, reason}. candidates: [{id, size, fits}] of models that may
    serve. Measured per-subject results decide when any candidate has them;
    otherwise the policy's default model; otherwise the largest that fits."""
    import model_stats
    need = context_need(t)
    short = [c["id"] for c in candidates if c.get("context") and c["context"] < need]
    pool = [c for c in candidates if c.get("fits") is not False and c["id"] not in exclude
            and c["id"] not in short]
    if not pool:
        why = "no model fits this machine"
        if short:
            why += f" with the {need}-token context this task needs ({', '.join(short[:3])} too short)"
        return {"model": None, "reason": why}
    proba = t.get("proba") or {}
    labels = t.get("labels") or []
    tot = sum(proba.get(k, 0) for k in labels)
    weights = {k: proba.get(k, 0) / tot for k in labels} if tot else {k: 1 / len(labels) for k in labels}
    sums = {c["id"]: summaries[c["id"]] for c in pool if c["id"] in summaries}
    if sums:
        pick = model_stats.rank(weights, sums, min_graded, min_subject, hallucination_cost)
        if pick.get("model"):
            return {"model": pick["model"], "reason": "measured: " + pick["reason"]}
    dm = policy.get("default_model")
    if dm and any(c["id"] == dm for c in pool):
        return {"model": dm, "reason": "default model (no model has enough graded results)"}
    big = max(pool, key=lambda c: c.get("size") or 0)
    return {"model": big["id"], "reason": "largest model that fits (no graded results; run testqa.py to measure)"}


# ================================================================ planning help

def refine_messages(plan: dict) -> list[dict]:
    """Ask a director model to turn the draft into dependencies and acceptance
    criteria. It may merge, split or drop candidates; labels are re-checked."""
    live = [t for t in plan["tasks"] if t["status"] != "dropped"]
    draft = "\n".join(f"{t['id']}: {t['title']} — {t['detail'][:400]}" for t in live)
    census = plan.get("census") or {}
    skills = ", ".join(f"{k} ({v['files']} files)" for k, v in census.get("subjects", {}).items())
    sys_p = ("You plan software and research projects for a team of worker models. Turn the draft task list "
             "into a plan: keep tasks that are real work, merge duplicates, split a task that mixes unrelated "
             "work, drop lines that only describe history or status, give each task acceptance criteria a "
             "reviewer can check, and say which tasks must finish before which. Reply with JSON only: "
             '{"tasks": [{"title": "...", "detail": "...", "acceptance": "...", "depends_on": ["<title of an '
             'earlier task>"]}]}. Order tasks so dependencies come first.')
    user = f"Project: {plan['title']}\nGoal: {plan.get('goal') or '(not stated)'}\n"
    if skills:
        user += f"The code base needs: {skills}\n"
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user + "\nDraft tasks:\n" + draft}]


def parse_refined(text: str, clf: sc.SubjectClassifier | None = None) -> list[dict]:
    """The director model's JSON -> tasks with depends_on as indices. Raises
    PlanError on anything malformed, so a bad answer leaves the draft as is."""
    body = text
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        body = m.group(1)
    elif "{" in text:
        body = text[text.index("{"):text.rindex("}") + 1]
    try:
        d = json.loads(body)
    except ValueError as e:
        raise PlanError(f"the director model's plan is not JSON: {e}")
    items = d.get("tasks") if isinstance(d, dict) else None
    if not isinstance(items, list) or not items:
        raise PlanError("the director model returned no tasks")
    clf = clf or sc.default_classifier()
    titles = {}
    out = []
    for i, it in enumerate(items[:60]):
        if not isinstance(it, dict) or not str(it.get("title") or "").strip():
            raise PlanError(f"task {i + 1} has no title")
        title = str(it["title"]).strip()[:200]
        detail = str(it.get("detail") or "").strip()
        a = clf.analyze(f"{title}: {detail}")
        deps = []
        for dref in it.get("depends_on") or []:
            j = titles.get(str(dref).strip().lower())
            if j is None:
                raise PlanError(f"{title!r} depends on {dref!r}, which is not an earlier task")
            deps.append(j)
        titles[title.lower()] = i
        out.append({"title": title, "detail": detail, "acceptance": str(it.get("acceptance") or ""),
                    "labels": a["labels"], "proba": {k: v for k, v in a["proba"].items() if v >= 0.05},
                    "why": a["why"], "dep_index": deps})
    return out


def replace_tasks(plan: dict, refined: list[dict], by: str, reason: str) -> dict:
    """Swap a draft's tasks for a refined list (draft only)."""
    if plan["status"] != "draft":
        raise PlanError("only a draft can be re-planned; propose changes instead")
    for t in plan["tasks"]:
        t["status"] = "dropped"
    new_ids = []
    for r in refined:
        t = _add_task(plan, {**r, "depends_on": []})
        new_ids.append(t["id"])
    for r, tid in zip(refined, new_ids):
        task(plan, tid)["depends_on"] = [new_ids[j] for j in r.get("dep_index", [])]
    plan["tasks"] = [t for t in plan["tasks"] if t["status"] != "dropped" or t["id"] in new_ids]
    _check_graph(plan)
    plan["version"] += 1
    _log(plan, by, "re-planned", reason, [{"op": "replace", "tasks": len(new_ids)}])
    plan["updated"] = _now()
    return plan


# ================================================================ storage

class ProjectStore:
    """One directory per project: plan.json, written atomically. Callers hold
    lock(pid) around load-modify-save."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def lock(self, pid: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(pid, threading.RLock())

    def _path(self, pid: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", pid or ""):
            raise PlanError("bad project id")
        return self.root / pid / "plan.json"

    def save(self, plan: dict) -> None:
        p = self._path(plan["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(plan, indent=1))
        os.replace(tmp, p)

    def load(self, pid: str) -> dict:
        p = self._path(pid)
        if not p.exists():
            raise PlanError(f"no project {pid}")
        return json.loads(p.read_text())

    def list(self) -> list[dict]:
        out = []
        if self.root.is_dir():
            for d in sorted(self.root.iterdir(), reverse=True):
                try:
                    pl = json.loads((d / "plan.json").read_text())
                except (OSError, ValueError):
                    continue
                c = {}
                for t in pl["tasks"]:
                    c[t["status"]] = c.get(t["status"], 0) + 1
                out.append({"id": pl["id"], "title": pl["title"], "status": pl["status"], "version": pl["version"],
                            "updated": pl["updated"], "tasks": c,
                            "pending": sum(1 for p in pl["proposals"] if p["status"] == "pending")})
        return out


# ================================================================ CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("analyze", help="split a description into skill-labelled tasks")
    a1.add_argument("file", help="markdown or text file ('-' for stdin)")
    a1.add_argument("--repo", help="also survey this checkout")
    a1.add_argument("--json", action="store_true")
    a2 = sub.add_parser("show", help="print a saved plan")
    a2.add_argument("dir")
    a = p.parse_args(argv)
    if a.cmd == "analyze":
        import sys
        text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text(encoding="utf-8")
        tasks = analyze_text(text)
        census = analyze_repo(a.repo) if a.repo else None
        if a.json:
            print(json.dumps({"tasks": tasks, "skills": skill_breakdown(tasks), "census": census,
                              "gaps": coverage_gaps(tasks, census)}, indent=1))
            return 0
        for i, t in enumerate(tasks, 1):
            print(f"{i:>3}. [{', '.join(t['labels']) or 'unknown'}] {t['title']}  ({t['why']})")
        print("\nby skill: " + "; ".join(f"{k} {len(v)}" for k, v in skill_breakdown(tasks).items()))
        if census:
            print("code base: " + ", ".join(f"{k} {v['files']}" for k, v in census["subjects"].items()))
            gaps = coverage_gaps(tasks, census)
            if gaps:
                print("no task covers: " + ", ".join(gaps))
    else:
        d = Path(a.dir)
        plan = json.loads((d / "plan.json" if d.is_dir() else d).read_text())
        print(f"{plan['title']}  [{plan['status']} v{plan['version']}]")
        for t in plan["tasks"]:
            who = (t.get("assignee") or {}).get("model") or "-"
            deps = f" after {','.join(t['depends_on'])}" if t["depends_on"] else ""
            print(f"  {t['id']:<4} {t['status']:<8} {who:<28} {t['title']}{deps}")
        pend = [x for x in plan["proposals"] if x["status"] == "pending"]
        if pend:
            print(f"{len(pend)} proposal(s) waiting for a decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
