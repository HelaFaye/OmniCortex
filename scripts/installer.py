#!/usr/bin/env python3
"""
OmniCortex's installer, the same on Windows, macOS and Linux. Standard
library only, so it runs before anything is installed.

    python scripts/installer.py              # the wizard, in your browser
    python scripts/installer.py --yes        # install with the recommended choices, in the terminal
    python scripts/installer.py --check      # what it found and would do, nothing installed
    python scripts/installer.py --yes --torch none          # without PyTorch
    python scripts/installer.py --yes --torch rocm --rocm 7.0

The launchers next to it (Install OmniCortex.bat, Install OmniCortex.command,
install-gui.sh) find a suitable Python first, and say how to get one if there
is none.

What it does: a virtual environment in ./venv, the core packages, a PyTorch
build that matches this computer's GPU (or none), the rest of the packages,
a GPU smoke test, and a shortcut (Start menu, Applications, app menu). Then
Studio opens on its Setup page, which builds llama.cpp and finds models.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PY = (3, 10)
SYSTEM = platform.system()            # Windows, Darwin, Linux


# ---------------------------------------------------------------- detection

def _run(cmd, timeout=20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if SYSTEM == "Windows" else "bin/python")


def _gpus() -> dict:
    out = {"nvidia": [], "amd": [], "apple": False, "other": [], "gfx": None, "kfd": False}
    if shutil.which("nvidia-smi"):
        out["nvidia"] = [ln.split(":", 1)[1].split("(UUID")[0].strip()
                         for ln in _run(["nvidia-smi", "-L"]).splitlines() if ln.startswith("GPU ")]
    if SYSTEM == "Darwin" and platform.machine() == "arm64":
        out["apple"] = True
    if SYSTEM == "Linux":
        out["kfd"] = os.path.exists("/dev/kfd")
        for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
            try:
                if (dev / "vendor").read_text().strip() == "0x1002":
                    name = (dev / "product_name").read_text().strip() if (dev / "product_name").exists() else "AMD GPU"
                    out["amd"].append(name or "AMD GPU")
            except OSError:
                pass
        if shutil.which("rocminfo"):
            env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}
            try:
                txt = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20, env=env).stdout
            except (OSError, subprocess.SubprocessError):
                txt = ""
            gfx = sorted({g for g in re.findall(r"gfx[0-9a-f]{3,}", txt) if g != "gfx000"})
            out["gfx"] = gfx[0] if gfx else None
    if SYSTEM == "Windows":
        txt = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        for name in (x.strip() for x in txt.splitlines() if x.strip()):
            if re.search(r"AMD|Radeon", name, re.I):
                out["amd"].append(name)
            elif not re.search(r"NVIDIA", name, re.I):
                out["other"].append(name)
    return out


def _store_python(exe: str) -> bool:
    return SYSTEM == "Windows" and "WindowsApps" in exe


def detect(venv: Path | None = None) -> dict:
    venv = venv or ROOT / "venv"
    free = shutil.disk_usage(ROOT).free
    d = {"system": SYSTEM, "release": platform.release(), "machine": platform.machine(),
         "python": sys.executable, "python_version": ".".join(map(str, sys.version_info[:3])),
         "python_ok": sys.version_info[:2] >= MIN_PY, "store_python": _store_python(sys.executable),
         "venv": str(venv), "venv_exists": venv_python(venv).exists(), "free_gb": round(free / 2**30, 1),
         "root": str(ROOT), "gpus": _gpus(), "notes": []}
    try:
        import venv as _v  # noqa: F401
        import ensurepip  # noqa: F401
        d["venv_ok"] = True
    except ImportError:
        d["venv_ok"] = False
    if SYSTEM == "Linux" and os.path.exists("/etc/os-release"):
        rel = dict(re.findall(r'^(\w+)="?([^"\n]*)"?', open("/etc/os-release").read(), re.M))
        d["distro"] = rel.get("PRETTY_NAME") or rel.get("ID")
        d["distro_like"] = f"{rel.get('ID', '')} {rel.get('ID_LIKE', '')}"
    d["notes"] = platform_notes(d)
    return d


def _pkg_hint(d: dict, debian: str, fedora: str, arch: str) -> str:
    like = d.get("distro_like", "")
    if re.search(r"debian|ubuntu|mint|pop", like):
        return f"sudo apt install {debian}"
    if re.search(r"fedora|rhel|centos", like):
        return f"sudo dnf install {fedora}"
    if re.search(r"arch|manjaro|endeavour", like):
        return f"sudo pacman -S {arch}"
    return f"install your distribution's package for: {debian}"


def platform_notes(d: dict) -> list[dict]:
    """What is peculiar about this computer, as {level: ok|info|warn|stop, text}."""
    n, g = [], d["gpus"]
    if not d["python_ok"]:
        n.append({"level": "stop", "text": f"Python {d['python_version']} is too old; OmniCortex needs 3.10 or newer."})
    if not d.get("venv_ok", True):
        n.append({"level": "stop", "text": "Python's venv module is missing: "
                  + _pkg_hint(d, "python3-venv", "python3", "python")})
    if d["free_gb"] < 10:
        n.append({"level": "warn", "text": f"{d['free_gb']} GB free here. PyTorch alone takes 2-8 GB, and models "
                                           "are several GB each."})
    if SYSTEM == "Windows":
        if d["store_python"]:
            n.append({"level": "warn", "text": "This is the Microsoft Store's Python. It works, but keeps files in a "
                                               "hidden folder; the python.org installer is easier to manage."})
        if len(d["root"]) > 120:
            n.append({"level": "warn", "text": "The folder path is long; Windows limits paths to 260 characters "
                      "unless long paths are turned on. Move OmniCortex closer to the drive's root (e.g. "
                      "C:\\neuronscope) if installing fails with 'path too long'."})
        if g["amd"] and not g["nvidia"]:
            n.append({"level": "info", "text": "AMD GPU on Windows: PyTorch has no ROCm build for Windows, so it "
                      "installs for the CPU. Serving models still uses the GPU, through llama.cpp's Vulkan build "
                      "(Studio's Setup page)."})
        n.append({"level": "info", "text": "Building llama.cpp on Windows needs Visual Studio Build Tools and CMake; "
                  "Studio's Setup page can also use a prebuilt llama.cpp release instead."})
    if SYSTEM == "Darwin":
        if d["machine"] == "x86_64":
            n.append({"level": "warn", "text": "Intel Mac: PyTorch stopped publishing Intel macOS builds after 2.2, "
                      "so the installer pins torch<2.3. Studio and the llama.cpp path do not need PyTorch."})
        else:
            n.append({"level": "ok", "text": "Apple Silicon: PyTorch uses the GPU through Metal (MPS); llama.cpp "
                      "builds with Metal."})
        n.append({"level": "info", "text": "Building llama.cpp needs the Xcode command-line tools "
                  "(xcode-select --install) and CMake (brew install cmake)."})
    if SYSTEM == "Linux":
        if g["amd"] and not g["kfd"]:
            n.append({"level": "info", "text": "AMD GPU without /dev/kfd: no ROCm compute here, so PyTorch installs "
                      "for the CPU. llama.cpp's Vulkan build still serves models on the GPU."})
        if g["kfd"]:
            groups = _run(["id", "-nG"]).split()
            missing = [x for x in ("video", "render") if x not in groups]
            if missing:
                n.append({"level": "warn", "text": f"To use the AMD GPU you need the {' and '.join(missing)} "
                          f"group(s): sudo usermod -aG {','.join(missing)} $USER, then log out and in."})
        if re.search(r"arch|manjaro|endeavour", d.get("distro_like", "")):
            n.append({"level": "info", "text": "Arch: ROCm is not officially supported, but the PyTorch ROCm wheels "
                      "bundle what they need and usually work. NVIDIA: legacy GPUs (Maxwell, Pascal, Volta) "
                      "need the 580xx driver branch or older, and CUDA 12 for building."})
    if g["nvidia"]:
        n.append({"level": "ok", "text": "NVIDIA: " + ", ".join(g["nvidia"]) + ". The PyTorch build is chosen per "
                  "GPU generation (older cards need the CUDA 12.6 wheels)."})
    if g["gfx"]:
        n.append({"level": "ok", "text": f"AMD ROCm target {g['gfx']}." + (
            " RDNA4 needs ROCm 7; do not set HSA_OVERRIDE_GFX_VERSION for it." if g["gfx"].startswith("gfx120")
            else "")})
    return n


ROCM_FOR = {"gfx1200": "7.0", "gfx1201": "7.0", "gfx1100": "6.4", "gfx1101": "6.4", "gfx1102": "6.4",
            "gfx1030": "6.4", "gfx90a": "6.4", "gfx942": "6.4", "gfx908": "6.4"}


def torch_choice(d: dict, mode: str = "auto", rocm: str | None = None) -> dict:
    """-> {mode, spec, index, why}: which PyTorch to install (mode none: skip it)."""
    g = d["gpus"]
    if mode == "auto":
        if SYSTEM == "Darwin":
            mode = "default"
        elif g["nvidia"]:
            mode = "cuda"
        elif g["kfd"] and SYSTEM == "Linux":
            mode = "rocm"
        else:
            mode = "cpu"
    if mode == "none":
        return {"mode": "none", "spec": None, "index": None, "why": "skipped: Studio, TestQA, Projects and the "
                "llama.cpp path work without PyTorch; add it later by running the installer again"}
    if mode == "default":
        spec = "torch<2.3" if SYSTEM == "Darwin" and d["machine"] == "x86_64" else "torch"
        return {"mode": mode, "spec": spec, "index": None, "why": "macOS: the default build (Metal on Apple Silicon)"}
    if mode == "cpu":
        return {"mode": mode, "spec": "torch", "index": "https://download.pytorch.org/whl/cpu",
                "why": "no GPU PyTorch can use here; CPU build"}
    if mode == "cuda":
        spec, index = "torch", None
        out = _run([sys.executable, str(ROOT / "scripts" / "cuda_info.py"), "--torch-pip"], timeout=30).split()
        if out:
            spec = out[0]
            if len(out) > 2 and out[1] == "--index-url":
                index = out[2]
        return {"mode": mode, "spec": spec, "index": index, "why": "NVIDIA: the build with kernels for every GPU here"}
    if mode == "rocm":
        ver = rocm or ROCM_FOR.get(g.get("gfx") or "", "6.4")
        return {"mode": mode, "spec": "torch", "index": f"https://download.pytorch.org/whl/rocm{ver}",
                "why": f"AMD ROCm {ver}" + (f" for {g['gfx']}" if g.get("gfx") else "")}
    raise ValueError(f"unknown torch mode {mode!r}")


# ---------------------------------------------------------------- the plan

def plan(d: dict, torch: dict, extras: bool = True, shortcut: bool = True) -> list[dict]:
    venv = Path(d["venv"])
    py = str(venv_python(venv))
    pip = [py, "-m", "pip", "install", "--disable-pip-version-check"]
    steps = [{"id": "venv", "title": "Create the Python environment" + (" (reusing it)" if d["venv_exists"] else ""),
              "cmd": [sys.executable, "-m", "venv", str(venv)]},
             {"id": "pip", "title": "Update pip", "cmd": pip + ["--upgrade", "pip", "wheel"]},
             {"id": "core", "title": "Install the core packages", "cmd": pip + ["-r", str(ROOT / "requirements-core.txt")]}]
    if torch["mode"] != "none":
        steps.append({"id": "torch", "title": f"Install PyTorch ({torch['why']})",
                      "cmd": pip + [torch["spec"]] + (["--index-url", torch["index"]] if torch["index"] else [])})
        steps.append({"id": "rest", "title": "Install the rest", "cmd": pip + ["-r", str(ROOT / "requirements.txt")]})
    if extras and (ROOT / "requirements-transfer-full.txt").exists():
        steps.append({"id": "extras", "title": "Install the extras (transfer GUI, MCP client, keyring)",
                      "cmd": pip + ["-r", str(ROOT / "requirements-transfer-full.txt")], "optional": True})
    if torch["mode"] != "none":
        steps.append({"id": "smoke", "title": "Check that PyTorch works on this GPU",
                      "cmd": [py, str(ROOT / "scripts" / "torch_smoke.py")], "optional": True})
    if shortcut:
        steps.append({"id": "shortcut", "title": "Add OmniCortex to the " + {"Windows": "Start menu",
                      "Darwin": "Applications folder"}.get(SYSTEM, "app menu"), "fn": "shortcut"})
    return steps


def make_shortcut(d: dict) -> str:
    root = Path(d["root"])
    if SYSTEM == "Windows":
        start = Path(os.environ.get("APPDATA", str(Path.home()))) / "Microsoft/Windows/Start Menu/Programs"
        start.mkdir(parents=True, exist_ok=True)
        lnk = start / "OmniCortex.lnk"
        ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
              f"$s.TargetPath='{root / 'OmniCortex.bat'}';$s.WorkingDirectory='{root}';"
              "$s.WindowStyle=7;$s.Description='OmniCortex Studio';$s.Save()")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-300:] or "could not create the shortcut")
        return str(lnk)
    if SYSTEM == "Darwin":
        app = Path.home() / "Applications" / "OmniCortex.app"
        (app / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
        (app / "Contents" / "Info.plist").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>'
            "<key>CFBundleName</key><string>OmniCortex</string>"
            "<key>CFBundleIdentifier</key><string>org.neuronscope.studio</string>"
            "<key>CFBundleExecutable</key><string>OmniCortex</string>"
            "<key>CFBundlePackageType</key><string>APPL</string></dict></plist>\n")
        exe = app / "Contents" / "MacOS" / "OmniCortex"
        exe.write_text(f'#!/bin/bash\nexec "{root}/neuronscope"\n')
        exe.chmod(0o755)
        return str(app)
    apps = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    f = apps / "neuronscope.desktop"
    f.write_text("[Desktop Entry]\nType=Application\nName=OmniCortex\nComment=Run, test and look inside local "
                 f"models\nExec={root / 'neuronscope'}\nTerminal=false\nCategories=Development;Science;\n")
    return str(f)


class Run:
    """Runs a plan in a thread; lines and step changes go to subscribers."""

    def __init__(self, d: dict, steps: list[dict]):
        self.d, self.steps = d, steps
        for s in self.steps:
            s["status"] = "waiting"
        self.lines: list[str] = []
        self.done = threading.Event()
        self.ok = None
        self.subs: list[queue.Queue] = []
        self.lock = threading.Lock()

    def emit(self, kind: str, **data) -> None:
        ev = {"kind": kind, **data}
        with self.lock:
            if kind == "line":
                self.lines.append(data["text"])
            for q in self.subs:
                q.put(ev)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            for t in self.lines[-400:]:
                q.put({"kind": "line", "text": t})
            for s in self.steps:
                q.put({"kind": "step", "id": s["id"], "status": s["status"]})
            if self.done.is_set():
                q.put({"kind": "end", "ok": self.ok})
            self.subs.append(q)
        return q

    def run(self) -> bool:
        ok = True
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PIP_NO_INPUT": "1"}
        for s in self.steps:
            s["status"] = "running"
            self.emit("step", id=s["id"], status="running")
            self.emit("line", text=f"== {s['title']}")
            try:
                if s.get("fn") == "shortcut":
                    self.emit("line", text="created " + make_shortcut(self.d))
                    rc = 0
                else:
                    self.emit("line", text="$ " + " ".join(s["cmd"]))
                    p = subprocess.Popen(s["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         env=env, cwd=str(ROOT), bufsize=1)
                    for line in p.stdout:
                        self.emit("line", text=line.rstrip())
                    rc = p.wait()
            except Exception as e:
                self.emit("line", text=f"error: {type(e).__name__}: {e}")
                rc = 1
            s["status"] = "done" if rc == 0 else ("skipped" if s.get("optional") else "failed")
            self.emit("step", id=s["id"], status=s["status"])
            if rc != 0 and not s.get("optional"):
                ok = False
                self.emit("line", text=f"!! {s['title']} failed. " + advice(s["id"], self.d))
                break
            if rc != 0:
                self.emit("line", text=f"(optional; carrying on) " + advice(s["id"], self.d))
        self.ok = ok
        self.done.set()
        self.emit("end", ok=ok)
        return ok


def advice(step: str, d: dict) -> str:
    if step == "venv":
        return "Python could not create the environment. " + (
            "Install the venv package: " + _pkg_hint(d, "python3-venv", "python3", "python") if SYSTEM == "Linux"
            else "Reinstall Python from python.org.")
    if step == "torch":
        return ("The PyTorch download failed. Check the network, or install without PyTorch for now (Studio, "
                "TestQA, Projects and the llama.cpp path work without it) and add it later.")
    if step == "smoke":
        return ("PyTorch installed but could not use the GPU as expected; scripts/cuda_info.py (NVIDIA) or "
                "docs/HARDWARE.md (AMD) explain what this GPU needs. The CPU still works.")
    if step == "extras":
        return "These are optional (WebRTC transfer GUI, MCP client, keyring); everything else works without them."
    return "The output above says why."


# ---------------------------------------------------------------- the wizard

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Install OmniCortex</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:820px;margin:0 auto;padding:1.2rem 16px 3rem}h1{font-size:22px;margin:.2rem 0 .2rem}
.steps{display:flex;gap:.4rem;margin:.6rem 0 1rem;flex-wrap:wrap}.steps span{font-size:12px;padding:.15rem .6rem;border-radius:10px;border:1px solid var(--line);color:var(--mut)}
.steps span.on{border-color:var(--acc);color:var(--acc)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem;margin-bottom:1rem}
h2{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:0 0 .6rem}
.note{color:var(--mut);font-size:13px}table{border-collapse:collapse;width:100%}td{padding:.3rem .4rem;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}
td:first-child{color:var(--mut);width:9rem}
.n{padding:.4rem .6rem;border-radius:6px;margin:.3rem 0;font-size:14px;border-left:3px solid var(--line)}
.n.ok{border-color:var(--ok)}.n.warn{border-color:var(--warn)}.n.stop{border-color:var(--no)}.n.info{border-color:var(--acc)}
label.opt{display:flex;gap:.6rem;align-items:flex-start;padding:.5rem;border:1px solid var(--line);border-radius:8px;margin:.35rem 0;cursor:pointer}
label.opt:has(input:checked){border-color:var(--acc)}label.opt b{display:block}
button{font:600 14px ui-sans-serif,system-ui;padding:.55rem 1rem;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);border-color:var(--acc);color:var(--accfg)}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
pre{font:12px/1.4 ui-monospace,monospace;background:var(--code);padding:.6rem;border-radius:6px;max-height:45vh;overflow:auto;white-space:pre-wrap;word-break:break-all}
.st{display:inline-block;width:5.5rem;font-size:12px}.st.done{color:var(--ok)}.st.failed{color:var(--no)}.st.running{color:var(--acc)}.st.skipped{color:var(--warn)}
input[type=text]{font:13px ui-monospace,monospace;padding:.35rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:100%}
</style>
<main>
<h1>Install OmniCortex</h1><div class="note">Everything happens on this computer. Nothing is sent anywhere except package downloads (pypi.org, download.pytorch.org).</div>
<div class="steps"><span data-s="1">1 · This computer</span><span data-s="2">2 · Choices</span><span data-s="3">3 · Installing</span><span data-s="4">4 · Done</span></div>
<section id="s1" class="card"><h2>What the installer found</h2><table id="facts"></table><div id="notes" style="margin-top:.6rem"></div>
 <div class="row" style="margin-top:.8rem"><button class="pri" id="next1">Next</button></div></section>
<section id="s2" class="card" hidden><h2>Choices</h2>
 <div class="note">PyTorch is needed for the PyTorch analysis path, CLIP/SigLIP and fine-tuning. Studio, TestQA, Projects and everything through llama.cpp work without it.</div>
 <div id="torch"></div>
 <label class="opt"><input type="checkbox" id="extras" checked><span><b>Extras</b><span class="note">The WebRTC transfer GUI, the MCP client for tools in Studio's chat, and keyring for stored keys.</span></span></label>
 <label class="opt"><input type="checkbox" id="shortcut" checked><span><b>Shortcut</b><span class="note" id="scnote"></span></span></label>
 <label class="note" style="display:block;margin-top:.5rem">Install into<input type="text" id="venv"></label>
 <div class="row" style="margin-top:.8rem"><button id="back2">Back</button><button class="pri" id="go">Install</button><span class="note">About 1-10 minutes, depending on the PyTorch download.</span></div></section>
<section id="s3" class="card" hidden><h2>Installing</h2><div id="plist"></div><details open style="margin-top:.5rem"><summary class="note">Output</summary><pre id="log"></pre></details></section>
<section id="s4" class="card" hidden><h2 id="endh">Done</h2><div id="endmsg"></div>
 <div class="row" style="margin-top:.8rem"><button class="pri" id="open">Open OmniCortex</button><button id="quit">Close the installer</button></div></section>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const T=location.pathname.split('/')[1];
const api=async(p,b)=>{ const r=await fetch('/'+T+p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.status); return j; };
let D=null;
function show(n){ for(let i=1;i<=4;i++){ $('#s'+i).hidden=i!==n; document.querySelector(`[data-s="${i}"]`).classList.toggle('on',i===n); } }
async function load(){ D=await api('/api/detect'); const g=D.gpus;
  const gp=[...g.nvidia.map(x=>'NVIDIA '+x),...g.amd.map(x=>x+(g.gfx?` (${g.gfx})`:'')),...(g.apple?['Apple Silicon GPU']:[]),...g.other].join('<br>')||'none found';
  $('#facts').innerHTML=[['System',`${esc(D.system)} ${esc(D.release)} · ${esc(D.machine)}${D.distro?' · '+esc(D.distro):''}`],['Python',`${esc(D.python_version)} <span class="note">${esc(D.python)}</span>`],['GPUs',gp],['Free space',D.free_gb+' GB'],['Folder',`<span class="note">${esc(D.root)}</span>`],['Existing install',D.venv_exists?'yes: it will be updated, not replaced':'no']].map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  $('#notes').innerHTML=D.notes.map(n=>`<div class="n ${n.level}">${esc(n.text)}</div>`).join('');
  $('#next1').disabled=D.notes.some(n=>n.level==='stop');
  const opts=[['auto','Recommended for this computer',D.torch.auto.why+` (${D.torch.auto.spec}${D.torch.auto.index?' from '+D.torch.auto.index:''})`]];
  if(D.system!=='Darwin'){ opts.push(['cuda','NVIDIA (CUDA)','for NVIDIA GPUs; the build is picked per GPU generation']); if(D.system==='Linux') opts.push(['rocm','AMD (ROCm)','Linux with an AMD GPU that ROCm supports']); opts.push(['cpu','CPU only','smaller; works everywhere']); }
  opts.push(['none','No PyTorch','smallest; Studio, TestQA, Projects and the llama.cpp path still work']);
  $('#torch').innerHTML=opts.map(([v,t,d],i)=>`<label class="opt"><input type="radio" name="torch" value="${v}" ${i===0?'checked':''}><span><b>${esc(t)}</b><span class="note">${esc(d)}</span></span></label>`).join('');
  $('#scnote').textContent={Windows:'OmniCortex in the Start menu.',Darwin:'OmniCortex in your Applications folder.'}[D.system]||'OmniCortex in the app menu.';
  $('#venv').value=D.venv; show(1); }
$('#next1').onclick=()=>show(2); $('#back2').onclick=()=>show(1);
$('#go').onclick=async()=>{ const torch=document.querySelector('[name=torch]:checked').value;
  try{ const r=await api('/api/install',{torch,extras:$('#extras').checked,shortcut:$('#shortcut').checked,venv:$('#venv').value}); steps(r.steps); show(3); follow(); }catch(e){ alert(e.message); } };
function steps(st){ $('#plist').innerHTML=st.map(s=>`<div><span class="st" id="st-${s.id}">waiting</span>${esc(s.title)}</div>`).join(''); }
function follow(){ const es=new EventSource('/'+T+'/api/events'); const log=$('#log');
  es.onmessage=e=>{ const ev=JSON.parse(e.data);
    if(ev.kind==='line'){ log.textContent+=ev.text+'\n'; log.scrollTop=1e9; }
    if(ev.kind==='step'){ const el=$('#st-'+ev.id); if(el){ el.textContent=ev.status; el.className='st '+ev.status; } }
    if(ev.kind==='end'){ es.close(); show(4); $('#endh').textContent=ev.ok?'Done':'Not finished';
      $('#endmsg').innerHTML=ev.ok?'<p>OmniCortex is installed. It opens in your browser on its Setup page, which builds llama.cpp (or uses one you have), finds your models and checks everything. The first time, it offers a short tour (optional).</p>':'<p>A step failed; the output says why, and what to try. Fix it and press Install again: finished steps are quick the second time.</p>';
      $('#open').hidden=!ev.ok; } }; }
$('#open').onclick=async()=>{ await api('/api/launch',{}); $('#endmsg').innerHTML+='<p class="note">Starting OmniCortex… it opens in a new tab.</p>'; };
$('#quit').onclick=async()=>{ try{ await api('/api/quit',{}); }catch{} document.body.innerHTML='<main><h1>You can close this tab.</h1></main>'; };
load().catch(e=>{ document.body.innerHTML='<main><h1>Installer error</h1><pre>'+esc(e.message)+'</pre></main>'; });
</script>"""


def serve(port: int = 0, open_browser: bool = True) -> None:
    token = secrets.token_urlsafe(12)
    state = {"run": None, "det": detect()}
    state["det"]["torch"] = {"auto": torch_choice(state["det"], "auto")}
    stop = threading.Event()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _path(self):
            p = self.path.split("?")[0]
            pre = "/" + token
            return p[len(pre):] if p.startswith(pre) else None

        def do_GET(self):
            p = self._path()
            if p is None:
                return self._send(404, {"error": "not found"})
            if p in ("", "/"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/detect":
                return self._send(200, state["det"])
            if p == "/api/events" and state["run"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                q = state["run"].subscribe()
                try:
                    while True:
                        ev = q.get()
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                        if ev["kind"] == "end":
                            return
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self._path()
            if p is None or self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"):
                return self._send(403, {"error": "forbidden"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if p == "/api/install":
                if state["run"] and not state["run"].done.is_set():
                    return self._send(409, {"error": "already installing"})
                d = dict(state["det"])
                d["venv"] = str(Path(req.get("venv") or d["venv"]).expanduser())
                d["venv_exists"] = venv_python(Path(d["venv"])).exists()
                try:
                    t = torch_choice(d, req.get("torch") or "auto")
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                st = plan(d, t, bool(req.get("extras", True)), bool(req.get("shortcut", True)))
                state["run"] = Run(d, st)
                threading.Thread(target=state["run"].run, daemon=True).start()
                return self._send(200, {"steps": [{"id": s["id"], "title": s["title"]} for s in st]})
            if p == "/api/launch":
                launch(Path(state["run"].d["venv"]) if state["run"] else ROOT / "venv")
                return self._send(200, {"ok": True})
            if p == "/api/quit":
                stop.set()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{srv.server_port}/{token}/"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"OmniCortex installer: {url}\n(open it in a browser if it did not open; Ctrl+C to quit)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    srv.shutdown()


def launch(venv: Path) -> None:
    py = venv_python(venv)
    py = str(py) if py.exists() else sys.executable
    kw = {"creationflags": 0x00000008} if SYSTEM == "Windows" else {"start_new_session": True}   # detached
    subprocess.Popen([py, str(ROOT / "scripts" / "launch.py")], cwd=str(ROOT), stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **kw)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--yes", action="store_true", help="install in the terminal with the recommended choices")
    p.add_argument("--check", action="store_true", help="show what was found and the plan; install nothing")
    p.add_argument("--torch", default="auto", choices=["auto", "cuda", "rocm", "cpu", "default", "none"])
    p.add_argument("--rocm", help="ROCm wheel version, e.g. 7.0")
    p.add_argument("--venv", default=str(ROOT / "venv"))
    p.add_argument("--no-extras", action="store_true")
    p.add_argument("--no-shortcut", action="store_true")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    if not (a.yes or a.check):
        serve(a.port, not a.no_browser)
        return 0
    d = detect(Path(a.venv).expanduser())
    t = torch_choice(d, a.torch, a.rocm)
    steps = plan(d, t, not a.no_extras, not a.no_shortcut)
    print(f"{d['system']} {d['release']} {d['machine']} · Python {d['python_version']} · {d['free_gb']} GB free")
    for n in d["notes"]:
        print(f"  [{n['level']}] {n['text']}")
    print(f"PyTorch: {t['why']}" + (f" -> {t['spec']}" + (f" from {t['index']}" if t["index"] else "") if t["spec"] else ""))
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s['title']}")
    if a.check:
        return 0
    if any(n["level"] == "stop" for n in d["notes"]):
        print("Cannot install here; see the notes above.", file=sys.stderr)
        return 2
    r = Run(d, steps)
    r.subs.append(_Printer())
    ok = r.run()
    print("\nDone. Start it from the shortcut, or: " + ("OmniCortex.bat" if SYSTEM == "Windows" else "./neuronscope")
          if ok else "\nNot finished; the output above says why.")
    return 0 if ok else 1


class _Printer:
    def put(self, ev):
        if ev["kind"] == "line":
            print(ev["text"], flush=True)
        elif ev["kind"] == "step" and ev["status"] in ("done", "failed", "skipped"):
            print(f"   -> {ev['status']}", flush=True)


if __name__ == "__main__":
    if sys.version_info[:2] < MIN_PY:
        sys.exit(f"OmniCortex needs Python {MIN_PY[0]}.{MIN_PY[1]} or newer; this is {platform.python_version()}. "
                 "Get it from https://www.python.org/downloads/")
    sys.exit(main())
