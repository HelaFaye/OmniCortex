#!/usr/bin/env python3
"""
Core shunt (docs/INTEGRATED_OPTIONS.md): run background jobs (TestQA, fine-tuning, extraction, builds) at low
priority, on the CPU's efficiency cores where it has them, so the computer stays smooth while they run.

    priority   low (default): nice 10 and idle-class disk I/O on Linux; background QoS on macOS (which also
               moves the work onto E-cores); BELOW_NORMAL on Windows. normal: leave it alone
    cores      auto (default): on a CPU with efficiency cores (Intel hybrid, ARM big.LITTLE), pin jobs to them;
               all: no pinning; or a list like 0-3,8

Not covered: GPU scheduling priority. No vendor offers a portable per-process GPU priority (NVIDIA none,
AMD/Intel only through compositor-specific paths), so a GPU-bound job still competes with your desktop for the
GPU. Lower its batch size or give it another GPU.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

DEFAULTS = {"priority": "low", "cores": "auto"}


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def efficiency_cores() -> list[int]:
    """The E-cores on Linux: Intel hybrid lists them in cpu_atom; ARM marks them with a lower cpu_capacity."""
    atom = Path("/sys/devices/cpu_atom/cpus")
    if atom.exists():
        try:
            return parse_cpus(atom.read_text().strip())
        except (OSError, ValueError):
            pass
    caps = {}
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpu_capacity"):
        try:
            caps[int(p.parent.name[3:])] = int(p.read_text())
        except (OSError, ValueError):
            continue
    if caps and len(set(caps.values())) > 1:
        low = min(caps.values())
        return sorted(c for c, v in caps.items() if v == low)
    return []


def describe(cfg: dict | None = None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    sysname = platform.system()
    e = efficiency_cores() if sysname == "Linux" else []
    return {"settings": c, "system": sysname, "efficiency_cores": e,
            "will_pin": (parse_cpus(c["cores"]) if c["cores"] not in ("auto", "all") else e if c["cores"] == "auto" else []),
            "macos_background_qos": sysname == "Darwin" and c["priority"] == "low"}


def popen_kwargs(cfg: dict | None = None) -> dict:
    """Extra subprocess.Popen arguments (Windows priority class)."""
    c = {**DEFAULTS, **(cfg or {})}
    if platform.system() == "Windows" and c["priority"] == "low":
        return {"creationflags": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)}
    return {}


def apply(pid: int, cfg: dict | None = None) -> list[str]:
    """Lower a started process's priority and pin it. -> what was done (best effort; never raises)."""
    c = {**DEFAULTS, **(cfg or {})}
    done = []
    sysname = platform.system()
    if c["priority"] == "low":
        if sysname in ("Linux", "Darwin", "FreeBSD"):
            try:
                os.setpriority(os.PRIO_PROCESS, pid, max(os.getpriority(os.PRIO_PROCESS, pid), 10))
                done.append("nice 10")
            except (OSError, AttributeError):
                pass
        if sysname == "Linux":
            try:
                subprocess.run(["ionice", "-c", "3", "-p", str(pid)], capture_output=True, timeout=5, check=True)
                done.append("idle I/O")
            except (OSError, subprocess.SubprocessError):
                pass
        if sysname == "Darwin":
            try:
                subprocess.run(["taskpolicy", "-b", "-p", str(pid)], capture_output=True, timeout=5, check=True)
                done.append("background QoS (E-cores)")
            except (OSError, subprocess.SubprocessError):
                pass
    cores = [] if c["cores"] == "all" else efficiency_cores() if c["cores"] == "auto" else parse_cpus(c["cores"])
    if cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(pid, cores)
            done.append(f"cores {cores[0]}-{cores[-1]}" if cores == list(range(cores[0], cores[-1] + 1)) else f"cores {cores}")
        except OSError:
            pass
    return done
