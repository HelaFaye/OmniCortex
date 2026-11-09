#!/usr/bin/env python3
"""
Hardware support profiles (docs/HARDWARE_PROFILES.md): for a family of hardware, legacy or current, what to
install and how to configure it to get the most out of it, kept separately for each purpose.

    general      one setup for everything (a profile with a single purpose uses this label)
    inference    serving and chatting: the llama.cpp backend, its build flags, engine defaults
    retraining   fine-tuning and the PyTorch side: the wheel index, training precision, method

A profile is a small JSON file: the built-in catalogue lives in profiles/hardware/, your own in
~/.omnicortex/hw-profiles/, and accepted community ones come from a board's profiles/ folder.

    python scripts/hw_profiles.py match                       # which profiles fit this computer, and why
    python scripts/hw_profiles.py plan --id nvidia-legacy-sm50-70 --purpose inference
    python scripts/hw_profiles.py apply --id … --purpose …    # run those steps (a Studio job does the same)
    python scripts/hw_profiles.py check FILE                  # is a profile valid before you share it
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
SCHEMA = "omnicortex-hw-profile/1"
PURPOSES = ("general", "inference", "retraining")
BACKENDS = ("cpu", "cuda", "hip", "vulkan", "metal", "sycl")
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,63}")
USER_DIR = Path(os.environ.get("OC_HOME") or Path.home() / ".omnicortex").expanduser() / "hw-profiles"


class ProfileError(ValueError):
    pass


# ================================================================ loading and checking

def validate(p: dict) -> dict:
    """Raise ProfileError on anything a profile must not be, return it otherwise."""
    if not isinstance(p, dict) or p.get("schema") != SCHEMA:
        raise ProfileError(f"schema must be {SCHEMA}")
    if not ID_RE.fullmatch(str(p.get("id", ""))):
        raise ProfileError("id: lowercase letters, digits, . _ - (2-64)")
    if not str(p.get("title", "")).strip():
        raise ProfileError("title is required")
    pur = p.get("purposes")
    if not isinstance(pur, dict) or not pur:
        raise ProfileError("purposes: at least one of general, inference, retraining")
    bad = [k for k in pur if k not in PURPOSES]
    if bad:
        raise ProfileError(f"purposes: unknown {', '.join(bad)} (use {', '.join(PURPOSES)})")
    if len(pur) == 1 and "general" not in pur:
        raise ProfileError("a profile with a single purpose labels it 'general'")
    for name, spec in pur.items():
        if not isinstance(spec, dict) or not str(spec.get("summary", "")).strip():
            raise ProfileError(f"{name}: a summary of what this setup is for is required")
        ll = spec.get("llama") or {}
        if ll and ll.get("backend") not in BACKENDS:
            raise ProfileError(f"{name}: llama.backend must be one of {', '.join(BACKENDS)}")
        if ll.get("cmake") and not re.fullmatch(r"(-D[A-Za-z0-9_]+=[\w.;:/+-]*\s*)*", ll["cmake"]):
            raise ProfileError(f"{name}: llama.cmake may only hold -DNAME=value flags")
        if ll.get("cuda_arch") and not re.fullmatch(r"[0-9]+(-real|-virtual)?(;[0-9]+(-real|-virtual)?)*", ll["cuda_arch"]):
            raise ProfileError(f"{name}: llama.cuda_arch looks like 61-real;70-real")
        t = spec.get("torch") or {}
        if t.get("index") and not str(t["index"]).startswith("https://download.pytorch.org/whl/"):
            raise ProfileError(f"{name}: torch.index must be a download.pytorch.org wheel index")
        if t.get("spec") and not re.fullmatch(r"torch([<>=!~]=?[\w.*+]+)?(,[<>=!~]=?[\w.*+]+)*", t["spec"]):
            raise ProfileError(f"{name}: torch.spec must be a version spec of torch")
        for k in (spec.get("env") or {}):
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", k):
                raise ProfileError(f"{name}: env names are UPPER_CASE")
    if not isinstance(p.get("match", {}), dict):
        raise ProfileError("match must be an object")
    return p


def _load_dir(d: Path, status: str) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            p = validate(json.loads(f.read_text()))
        except (ValueError, OSError) as e:
            out.append({"id": f.stem, "invalid": str(e), "file": str(f)})
            continue
        out.append({**p, "status": p.get("status") if status == "built-in" else status, "file": str(f)})
    return out


def catalogue(extra_dirs=()) -> list[dict]:
    """Built-in, then the owner's own, then community ones; a later one with the same id replaces an earlier."""
    seen = {}
    for d, st in [(ROOT / "profiles" / "hardware", "built-in"), (USER_DIR, "yours")] + [(Path(x), "community") for x in extra_dirs]:
        for p in _load_dir(Path(d).expanduser(), st):
            seen[p["id"]] = p
    return list(seen.values())


def get(pid: str, extra_dirs=()) -> dict:
    p = next((x for x in catalogue(extra_dirs) if x["id"] == pid), None)
    if p is None or p.get("invalid"):
        raise ProfileError(f"no valid profile {pid!r}")
    return p


# ================================================================ matching

def facts(devices=None, smi=None) -> list[dict]:
    """This computer's devices with what matching needs: vendor, backend, unified, gfx, compute capability."""
    if devices is None:
        import accelerators
        devices = accelerators.detect()
    if smi is None:
        try:
            import cuda_info
            smi = cuda_info.query_gpus()
        except Exception:
            smi = []
    cc = {g["index"]: g["cc"] / 10 for g in smi or [] if g.get("cc")}
    out = []
    for d in devices:
        f = {k: d.get(k) for k in ("id", "name", "backend", "vendor", "unified", "gfx", "memory_total")}
        if d.get("backend") == "cuda":
            f["compute"] = cc.get(d.get("index"))
        if sys.platform == "darwin" and d.get("backend") == "metal":
            f["vendor"] = "apple"
        out.append(f)
    return out


def _fits(m: dict, f: dict, machine: str) -> tuple[bool, str]:
    if m.get("vendor") and f.get("vendor") != m["vendor"]:
        return False, ""
    if m.get("vendor_any") and f.get("vendor") not in m["vendor_any"]:
        return False, ""
    if "unified" in m and bool(f.get("unified")) != bool(m["unified"]):
        return False, ""
    if m.get("backend_only") and f.get("backend") != m["backend_only"]:
        return False, ""
    if m.get("machine") and machine not in m["machine"]:
        return False, ""
    c = f.get("compute")
    if "compute_min" in m or "compute_max" in m:
        if c is None:
            return False, ""
        if c < m.get("compute_min", 0) or c > m.get("compute_max", 99):
            return False, ""
    g = str(f.get("gfx") or "")
    if m.get("gfx_prefix"):
        if not g or not any(g.startswith(x) for x in m["gfx_prefix"]) or any(g.startswith(x) for x in m.get("gfx_exclude", [])):
            return False, ""
    why = f"{f.get('name') or f.get('id')}" + (f", compute {c}" if c else "") + (f", {g}" if g else "")
    return True, why


def match(profiles: list[dict], devs: list[dict], machine: str | None = None) -> list[dict]:
    """Profiles that fit a device here, most specific first: [{profile, devices: [why…]}]."""
    machine = machine or platform.machine()
    out = []
    for p in profiles:
        if p.get("invalid"):
            continue
        hits = [w for f in devs for ok, w in [_fits(p.get("match") or {}, f, machine)] if ok]
        if hits:
            spec = len(p.get("match") or {})
            out.append({"profile": p, "devices": hits, "specificity": spec})
    order = {"cuda": 0, "rocm": 0, "metal": 0, "vulkan": 1, "cpu": 2}
    return sorted(out, key=lambda x: (-x["specificity"], x["profile"]["id"]))


# ================================================================ plans and applying

def purpose_of(p: dict, purpose: str | None) -> tuple[str, dict]:
    pur = p["purposes"]
    if purpose and purpose in pur:
        return purpose, pur[purpose]
    if "general" in pur:
        return "general", pur["general"]
    if purpose:
        raise ProfileError(f"{p['id']} has no '{purpose}' setup; it has {', '.join(pur)}")
    return next(iter(pur.items()))


def plan(p: dict, purpose: str | None = None, python: str | None = None, llama_dir: str | None = None) -> dict:
    """The steps a profile's purpose takes, as commands, and the settings it recommends."""
    name, spec = purpose_of(p, purpose)
    steps = []
    ll = spec.get("llama") or {}
    if ll:
        cmd = ["bash", str(HERE / "build_llama_tools.sh"), "--backend", ll["backend"]]
        if llama_dir:
            cmd += ["--dir", llama_dir]
        if ll.get("cuda_arch"):
            cmd += ["--cuda-arch", ll["cuda_arch"]]
        if not ll.get("native", False):
            cmd += ["--portable"]
        if ll.get("cmake"):
            cmd += ["--cmake", ll["cmake"]]
        steps.append({"id": "llama", "title": f"Build llama.cpp for {ll['backend']}" +
                      (f" ({ll['toolkit']})" if ll.get("toolkit") else ""), "cmd": cmd})
    t = spec.get("torch") or {}
    if t:
        py = python or sys.executable
        cmd = [py, "-m", "pip", "install", "--upgrade", t.get("spec") or "torch"] + (["--index-url", t["index"]] if t.get("index") else [])
        steps.append({"id": "torch", "title": "Install PyTorch" + (f" from {t['index'].rsplit('/', 1)[-1]}" if t.get("index") else ""),
                      "cmd": cmd})
    return {"profile": p["id"], "title": p["title"], "purpose": name, "summary": spec["summary"], "steps": steps,
            "engine": spec.get("engine") or {}, "training": spec.get("training") or {}, "env": spec.get("env") or {},
            "notes": [x for x in (spec.get("notes"), p.get("notes")) if x]}


def apply(pl: dict, dry_run: bool = False, log=print) -> int:
    for i, s in enumerate(pl["steps"], 1):
        log(f"PROGRESS {i - 1}/{len(pl['steps'])} {s['title']}")
        log("$ " + " ".join(shlex.quote(x) for x in s["cmd"]))
        if dry_run:
            continue
        r = subprocess.run(s["cmd"])
        if r.returncode:
            log(f"step '{s['title']}' failed (exit {r.returncode})")
            return r.returncode
    log(f"PROGRESS {len(pl['steps'])}/{len(pl['steps'])} done")
    if pl["engine"]:
        log("recommended engine settings for models on this hardware: " + json.dumps(pl["engine"]))
    if pl["training"]:
        log("recommended training settings: " + json.dumps(pl["training"]))
    return 0


def save_user(p: dict) -> Path:
    validate(p)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    f = USER_DIR / f"{p['id']}.json"
    clean = {k: v for k, v in p.items() if k not in ("file", "status", "invalid")}
    f.write_text(json.dumps({**clean, "status": "yours"}, indent=1) + "\n")
    return f


# ================================================================ sharing and reports

def submission(p: dict) -> dict:
    """What is sent when a profile is shared: the profile itself (no paths, no names)."""
    validate(p)
    return {k: v for k, v in p.items() if k not in ("file", "status")}


def report(pid: str, purpose: str, worked: bool, devs: list[dict], note: str = "", tok_s: float | None = None) -> dict:
    """A "this profile worked / did not work for me" report, with coarse hardware only."""
    if purpose not in PURPOSES:
        raise ProfileError(f"purpose: one of {', '.join(PURPOSES)}")
    return {"schema": "omnicortex-hw-report/1", "profile": pid, "purpose": purpose, "worked": bool(worked),
            "hardware": [{"name": d.get("name"), "backend": d.get("backend"), "compute": d.get("compute"),
                          "gfx": d.get("gfx")} for d in devs if d.get("backend") != "cpu"][:8]
            or [{"name": platform.processor() or platform.machine(), "backend": "cpu"}],
            "os": platform.system(), "machine": platform.machine(), "note": str(note)[:500],
            **({"gen_tok_s": float(tok_s)} if tok_s else {})}


def aggregate_reports(reports: list[dict]) -> dict:
    """{profile id: {purpose: {reports, worked, failed, hardware: [...], median_tok_s}}} for a board."""
    out: dict = {}
    for r in reports:
        a = out.setdefault(r.get("profile"), {}).setdefault(r.get("purpose", "general"),
                                                             {"reports": 0, "worked": 0, "failed": 0, "hardware": [], "_t": []})
        a["reports"] += 1
        a["worked" if r.get("worked") else "failed"] += 1
        for h in r.get("hardware") or []:
            if h.get("name") and h["name"] not in a["hardware"]:
                a["hardware"].append(h["name"])
        if r.get("gen_tok_s"):
            a["_t"].append(r["gen_tok_s"])
    for pur in out.values():
        for a in pur.values():
            t = sorted(a.pop("_t"))
            a["median_tok_s"] = t[len(t) // 2] if t else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("match")
    for n in ("plan", "apply"):
        s = sub.add_parser(n)
        s.add_argument("--id", required=True)
        s.add_argument("--purpose", choices=PURPOSES)
        s.add_argument("--llama-dir")
        s.add_argument("--python", help="the Python whose environment gets PyTorch (default: this one)")
        if n == "apply":
            s.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("file")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "list":
            for p in catalogue():
                print(f"{p['id']:28} {p.get('status', ''):10} {', '.join(p.get('purposes') or {}):30} {p.get('title') or p.get('invalid')}")
        elif a.cmd == "match":
            for m in match(catalogue(), facts()):
                print(f"{m['profile']['id']}: {m['profile']['title']}  [{'; '.join(m['devices'])}]")
        elif a.cmd == "check":
            validate(json.loads(Path(a.file).read_text()))
            print("valid")
        else:
            pl = plan(get(a.id), a.purpose, a.python, a.llama_dir)
            if a.cmd == "plan":
                print(json.dumps(pl, indent=1))
            else:
                return apply(pl, a.dry_run)
    except (ProfileError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
