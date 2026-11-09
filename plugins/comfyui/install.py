"""ComfyUI: install (or repair) in its own virtual environment."""
import argparse
import sys
import venv

import comfy as C

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--folder", default="~/.omnicortex/comfyui")
p.add_argument("--torch", default="auto", choices=["auto", "cuda", "rocm", "xpu", "cpu", "mps", "directml"])
p.add_argument("--manager", action="store_true")
p.add_argument("--omnicortex-nodes", action="store_true")
p.add_argument("--llm-nodes", nargs="*", default=[], choices=sorted(C.LLM_PACKS))
a = p.parse_args()
C.need_git()
root = C.folder(a.folder)
C.say(f"ComfyUI in {root}")
C.clone_or_pull(C.REPO, root)
if not C.venv_python(root).exists():
    C.say("making its virtual environment (.venv)")
    venv.create(root / ".venv", with_pip=True)
C.pip(root, "pip", "wheel")
kind = C.detect_torch() if a.torch == "auto" else a.torch
C.install_torch(root, kind)
C.requirements(root, root)
if a.manager:
    C.clone_or_pull(C.MANAGER, root / "custom_nodes" / "comfyui-manager")
    C.requirements(root, root / "custom_nodes" / "comfyui-manager")
for k in a.llm_nodes:
    url, name = C.LLM_PACKS[k]
    C.clone_or_pull(url, root / "custom_nodes" / name)
    C.requirements(root, root / "custom_nodes" / name)
if a.omnicortex_nodes:
    C.omnicortex_nodes(root)
C.save_state(folder=str(root), torch=kind)
C.say("done. Start it with the job 'ComfyUI: run', then open the ComfyUI tab.")
