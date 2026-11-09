"""ComfyUI: run until this job is cancelled."""
import argparse
import os
import shlex
import sys

import comfy as C

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--folder", default="~/.omnicortex/comfyui")
p.add_argument("--port", type=int, default=8188)
p.add_argument("--listen", default="127.0.0.1")
p.add_argument("--vram", default="normal", choices=["normal", "low", "high", "cpu"])
p.add_argument("--args", default="")
a = p.parse_args()
root = C.folder(a.folder)
py = C.venv_python(root)
if not (root / "main.py").exists() or not py.exists():
    raise SystemExit(f"ComfyUI is not installed in {root}: run the job 'ComfyUI: install' first")
cmd = [str(py), "main.py", "--listen", a.listen, "--port", str(a.port),
       *{"low": ["--lowvram"], "high": ["--highvram"], "cpu": ["--cpu"]}.get(a.vram, []), *shlex.split(a.args)]
C.save_state(folder=str(root), port=a.port, listen=a.listen)
C.say(f"ComfyUI on http://{a.listen}:{a.port}  (the ComfyUI tab in Studio)")
C.say("$ " + " ".join(cmd))
os.chdir(root)
os.execv(cmd[0], cmd)
