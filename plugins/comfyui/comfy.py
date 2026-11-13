"""ComfyUI for OmniCortex: install it in its own virtual environment, keep it updated, run it.

ComfyUI (https://github.com/comfy-org/ComfyUI) is GPL-3.0 and is downloaded from its own repository at
install time; nothing of it is shipped here. Where it went and how it runs is remembered in
~/.omnicortex/comfyui.json, outside this add-on's folder (changing files here would turn the add-on off
until it is reviewed again).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/comfy-org/ComfyUI"
MANAGER = "https://github.com/Comfy-Org/ComfyUI-Manager"
LLM_PACKS = {
    "llm_party": ("https://github.com/heshengtao/comfyui_LLM_party", "comfyui_LLM_party"),
    "llms_toolkit": ("https://github.com/ComfyUI-Kelin/ComfyUI-LLMs-Toolkit", "ComfyUI-LLMs-Toolkit"),
    "llms": ("https://github.com/leoleelxh/ComfyUI-LLMs", "ComfyUI-LLMs"),
}
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm6.4",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
STATE = Path.home() / ".omnicortex" / "comfyui.json"
HERE = Path(__file__).resolve().parent


def say(*a):
    print(*a, flush=True)


def run(cmd, cwd=None, check=True):
    say("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if check and r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r.returncode


def state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    s = {**state(), **kw}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def folder(arg: str) -> Path:
    return Path(os.path.expanduser(arg or "~/.omnicortex/comfyui")).resolve()


def venv_python(root: Path) -> Path:
    v = root / ".venv"
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def detect_torch() -> str:
    if platform.system() == "Darwin":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocminfo") or shutil.which("rocm-smi") or Path("/opt/rocm").exists():
        return "rocm"
    if os.name == "nt":
        return "directml"
    return "cpu"


def need_git():
    if not shutil.which("git"):
        raise SystemExit("git is not installed: Setup -> Requirements shows how to get it")


def clone_or_pull(url: str, dest: Path):
    if (dest / ".git").exists():
        run(["git", "-C", dest, "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", "--depth", "1", url, dest])


def pip(root: Path, *args):
    run([venv_python(root), "-m", "pip", "install", "--upgrade", *args])


def requirements(root: Path, d: Path):
    req = d / "requirements.txt"
    if req.exists():
        pip(root, "-r", req)


def install_torch(root: Path, kind: str):
    say(f"PyTorch for {kind}")
    if kind == "mps":
        pip(root, "torch", "torchvision", "torchaudio")
    elif kind == "directml":
        pip(root, "torch-directml")
    else:
        pip(root, "torch", "torchvision", "torchaudio", "--index-url", TORCH_INDEX[kind])


def omnicortex_nodes(root: Path):
    src = HERE / "comfy_nodes" / "omnicortex_nodes"
    dst = root / "custom_nodes" / "omnicortex_nodes"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    say(f"OmniCortex nodes -> {dst}")
