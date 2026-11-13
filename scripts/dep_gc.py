#!/usr/bin/env python3
"""
Dependency garbage collector (docs/INTEGRATED_OPTIONS.md): find what builds, downloads and installs left behind,
and remove it when you say so. It lists first; nothing is deleted without --apply.

    partial      interrupted downloads: *.part, *.incomplete, *.download in the model folders and the Hugging
                 Face cache
    hf-orphans   Hugging Face cache blobs no snapshot points to any more (left by re-downloads and revisions)
    failed-build llama.cpp build folders whose bin/ has no llama-server or llama-cli (a build that failed)
    pycache      __pycache__ folders in OmniCortex's checkout
    pip-cache    pip's download cache
    addon-cache  add-on packages downloaded for installation that are no longer installed

Reported, never deleted:
    bad-gguf     .gguf files that do not start with the GGUF signature (truncated or not a model)
    dup-gguf     the same model file in two places (same size and SHA-256)

Never touched: models (other than partial downloads), workspaces, runs, settings, the venv itself.

    python scripts/dep_gc.py [--models DIR ...] [--apply] [--kinds partial,hf-orphans,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("partial", "hf-orphans", "failed-build", "pycache", "pip-cache", "addon-cache")
REPORT_ONLY = ("bad-gguf", "dup-gguf")
PARTIAL = (".part", ".incomplete", ".download", ".crdownload")


def size(p: Path) -> int:
    if p.is_symlink():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def hf_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")


def find(models: list[Path], kinds=KINDS, llama_dirs=None, home=None) -> list[dict]:
    out = []

    def add(kind, path, why, delete=True):
        out.append({"kind": kind, "path": str(path), "bytes": size(Path(path)), "why": why, "delete": delete})
    hub = hf_cache()
    if "partial" in kinds:
        for d in [*models, hub]:
            if d.is_dir():
                for dp, _, fs in os.walk(d):
                    for f in fs:
                        if f.endswith(PARTIAL):
                            add("partial", Path(dp) / f, "an interrupted download")
    if "hf-orphans" in kinds and hub.is_dir():
        for repo in hub.glob("models--*"):
            blobs = repo / "blobs"
            if not blobs.is_dir():
                continue
            used = set()
            for link in (repo / "snapshots").rglob("*") if (repo / "snapshots").is_dir() else []:
                if link.is_symlink():
                    used.add(Path(os.path.realpath(link)).name)
            for b in blobs.iterdir():
                if b.is_file() and b.name not in used and not b.name.endswith(PARTIAL):
                    add("hf-orphans", b, f"no snapshot of {repo.name[8:].replace('--', '/')} uses it")
    if "failed-build" in kinds:
        cands = [Path(x) for x in (llama_dirs or [])] + [ROOT / "build" / "llama.cpp", Path.home() / "llama.cpp"]
        for c in cands:
            for b in [c / "build", *c.glob("build-*")] if c.is_dir() else []:
                if b.is_dir() and not any((b / "bin" / n).exists() or (b / "bin" / f"{n}.exe").exists()
                                          for n in ("llama-server", "llama-cli")):
                    add("failed-build", b, "a llama.cpp build folder with no llama-server: the build failed or stopped")
    if "pycache" in kinds:
        for p in ROOT.rglob("__pycache__"):
            if "venv" not in p.parts and ".git" not in p.parts:
                add("pycache", p, "compiled Python, rebuilt on demand")
    if "pip-cache" in kinds:
        try:
            d = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"], capture_output=True, text=True, timeout=30).stdout.strip()
            if d and Path(d).is_dir():
                add("pip-cache", Path(d), "pip's download cache (pip cache purge)")
        except (OSError, subprocess.SubprocessError):
            pass
    if "addon-cache" in kinds:
        h = Path(home or os.environ.get("OC_HOME") or Path.home() / ".omnicortex")
        cache, inst = h / "addons" / "cache", h / "plugins"
        installed = {p.name for p in inst.iterdir()} if inst.is_dir() else set()
        for z in cache.glob("*.zip") if cache.is_dir() else []:
            if z.stem.split("-")[0] not in installed:
                add("addon-cache", z, "a downloaded add-on package that is not installed")
    seen = {}
    for d in models:
        for f in d.rglob("*.gguf") if d.is_dir() else []:
            try:
                with open(f, "rb") as fh:
                    if fh.read(4) != b"GGUF":
                        add("bad-gguf", f, "does not start with the GGUF signature: truncated or not a model", delete=False)
                        continue
                st = f.stat()
                seen.setdefault(st.st_size, []).append(f)
            except OSError:
                continue
    for sz, fs in seen.items():
        if len(fs) > 1 and sz > 50 * 2**20:
            by = {}
            for f in fs:
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
                by.setdefault(h.hexdigest(), []).append(f)
            for group in by.values():
                for f in group[1:]:
                    add("dup-gguf", f, f"same file as {group[0]}", delete=False)
    return out


def remove(found: list[dict]) -> int:
    freed = 0
    for x in found:
        if not x["delete"]:
            continue
        p = Path(x["path"])
        try:
            if x["kind"] == "pip-cache":
                subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True, timeout=300)
            elif p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
            freed += x["bytes"]
        except OSError as e:
            x["error"] = str(e)
    return freed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="append", default=[], help="a model folder (repeatable)")
    ap.add_argument("--llama", action="append", default=[], help="a llama.cpp checkout")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--apply", action="store_true", help="delete what is listed (report-only kinds are kept)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    found = find([Path(m).expanduser() for m in a.models], tuple(a.kinds.split(",")), a.llama)
    if a.json:
        print(json.dumps(found, indent=1))
    else:
        for x in found:
            print(f"{x['kind']:13} {x['bytes'] / 2**20:10.1f} MB  {x['path']}\n{'':13} {x['why']}"
                  + ("" if x["delete"] else "  (report only)"))
        print(f"{sum(x['bytes'] for x in found if x['delete']) / 2**30:.2f} GB can be freed")
    if a.apply:
        print(f"freed {remove(found) / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
