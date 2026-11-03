#!/usr/bin/env python3
"""
Dependency snapshots (docs/INTEGRATED_OPTIONS.md): what a computer has that OmniCortex depends on, how two
computers differ, and a sandbox that fills the gap for one job without changing the main environment.

    snapshot   Python and OS, every installed package and version, the PyTorch build (CUDA / ROCm / CPU) read
               from package metadata (no import), the llama.cpp tools found and their versions, the NS_*
               settings in the environment, and the requirements files' hashes
    diff       A against B: packages missing or at another version, tools missing, a different PyTorch build
    sandbox    a venv that sees the main environment's packages (--system-site-packages) and adds only the
               missing or differing ones pinned to the source's versions; its python runs the job

Studio serves its snapshot at /api/env/snapshot, so a linked host's can be compared with this one before a
model, an edited GGUF or a graph is sent there (Connect → a host → Compare environments).

    python scripts/dep_snapshot.py snapshot --out here.json
    python scripts/dep_snapshot.py diff here.json there.json
    python scripts/dep_snapshot.py sandbox --source here.json --dir ~/.neuronscope/sandboxes/job1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("llama-server", "llama-cett-dump", "llama-quantize", "llama-cli", "rpc-server")
REQS = ("requirements.txt", "requirements-core.txt", "requirements-model-transfer.txt", "requirements-transfer-full.txt")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def packages() -> dict[str, str]:
    from importlib import metadata
    out = {}
    for d in metadata.distributions():
        n = d.metadata.get("Name")
        if n:
            out[norm(n)] = d.version
    return dict(sorted(out.items()))


def torch_build(pkgs: dict) -> str | None:
    v = pkgs.get("torch")
    if not v:
        return None
    m = re.search(r"\+(cu\d+|rocm[\d.]+|cpu|xpu)", v)
    if m:
        return m.group(1)
    if any(k.startswith("nvidia-cuda-runtime") for k in pkgs):
        return "cuda (pip wheels)"
    return "default"


def tool_versions(dirs=None) -> dict:
    out = {}
    search = [Path(d) for d in (dirs or [])] + [ROOT / "build" / "llama.cpp" / "build" / "bin",
                                                 Path.home() / "llama.cpp" / "build" / "bin"]
    for t in TOOLS:
        path = shutil.which(t) or next((str(d / t) for d in search if (d / t).exists()), None)
        if not path:
            continue
        ver = None
        try:
            p = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
            m = re.search(r"version:\s*(\S+)(?:\s*\((\w+)\))?", p.stdout + p.stderr)
            ver = m.group(0) if m else (p.stdout + p.stderr).strip().splitlines()[0][:80] if (p.stdout + p.stderr).strip() else None
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
        out[t] = {"path": path, "version": ver}
    return out


def snapshot(tool_dirs=None) -> dict:
    pk = packages()
    return {"schema": "omnicortex-env/1", "t": int(time.time()), "python": platform.python_version(),
            "os": platform.system(), "machine": platform.machine(), "packages": pk, "torch": torch_build(pk),
            "tools": tool_versions(tool_dirs),
            "env": {k: v for k, v in os.environ.items() if k.startswith("NS_") and "TOKEN" not in k and "KEY" not in k},
            "requirements": {r: hashlib.sha256((ROOT / r).read_bytes()).hexdigest()[:16]
                             for r in REQS if (ROOT / r).exists()}}


def diff(a: dict, b: dict) -> dict:
    """What b lacks or has differently, compared with a."""
    pa, pb = a.get("packages") or {}, b.get("packages") or {}
    out = {"missing": {k: v for k, v in pa.items() if k not in pb},
           "different": {k: {"here": v, "there": pb[k]} for k, v in pa.items() if k in pb and pb[k] != v},
           "extra": sorted(k for k in pb if k not in pa),
           "tools_missing": sorted(k for k in (a.get("tools") or {}) if k not in (b.get("tools") or {})),
           "tools_different": {k: {"here": v.get("version"), "there": (b.get("tools") or {})[k].get("version")}
                               for k, v in (a.get("tools") or {}).items()
                               if k in (b.get("tools") or {}) and v.get("version") != (b.get("tools") or {})[k].get("version")},
           "python": None if a.get("python", "")[:4] == b.get("python", "")[:4] else {"here": a.get("python"), "there": b.get("python")},
           "torch": None if a.get("torch") == b.get("torch") else {"here": a.get("torch"), "there": b.get("torch")},
           "requirements": {k: v for k, v in (a.get("requirements") or {}).items() if (b.get("requirements") or {}).get(k) != v}}
    out["same"] = not any(out[k] for k in ("missing", "different", "tools_missing", "python", "torch"))
    return out


def plan(d: dict, only: list[str] | None = None) -> list[str]:
    """pip requirement lines to bring the other computer up to this one (for the packages given, or all)."""
    want = {**d["missing"], **{k: v["here"] for k, v in d["different"].items()}}
    if only:
        want = {k: v for k, v in want.items() if k in {norm(x) for x in only}}
    lines = []
    for k, v in sorted(want.items()):
        if k == "torch" or "+" in v:
            continue                    # a PyTorch build comes from its own index: hw_profiles.py plans that
        lines.append(f"{k}=={v}")
    return lines


def sandbox(dirpath: Path, requirements: list[str], python: str | None = None) -> Path:
    """A venv over the main environment with these pins added. -> its python."""
    py = python or sys.executable
    dirpath.mkdir(parents=True, exist_ok=True)
    subprocess.run([py, "-m", "venv", "--system-site-packages", str(dirpath)], check=True)
    vpy = dirpath / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if requirements:
        subprocess.run([str(vpy), "-m", "pip", "install", "--no-deps", *requirements], check=True)
    (dirpath / "omnicortex-sandbox.json").write_text(json.dumps({"created": time.time(), "pins": requirements}))
    return vpy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out")
    s.add_argument("--tools", action="append", default=[])
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    b = sub.add_parser("sandbox")
    b.add_argument("--source", required=True, help="the snapshot to match")
    b.add_argument("--dir", required=True)
    b.add_argument("--only", nargs="*", help="only these packages")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        snap = json.dumps(snapshot(a.tools), indent=1)
        (Path(a.out).write_text(snap) if a.out else print(snap))
        return 0
    if a.cmd == "diff":
        r = diff(json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text()))
        print(json.dumps({**r, "plan": plan(r)}, indent=1))
        return 0
    src = json.loads(Path(a.source).read_text())
    r = diff(src, snapshot())
    print(sandbox(Path(a.dir).expanduser(), plan(r, a.only)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
