"""Godot client: open it on this computer, connected to a Studio."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--studio", default="http://127.0.0.1:7870")
p.add_argument("--tab", default="")
p.add_argument("--godot", default="")
a = p.parse_args()
here = Path(__file__).resolve().parent / "client"
mine = Path.home() / ".omnicortex" / "godot" / ("godot.exe" if os.name == "nt" else "godot")
g = a.godot or os.environ.get("OC_GODOT") or (str(mine) if mine.exists() else None) or shutil.which("godot4") \
    or shutil.which("godot")
if not g:
    raise SystemExit("Godot 4 not found: run the job 'Godot client: get Godot' first")
env = dict(os.environ, OC_STUDIO=a.studio, **({"OC_TAB": a.tab} if a.tab else {}))
print(f"$ {g} --path {here}  (Studio {a.studio})", flush=True)
sys.exit(subprocess.call([g, "--path", str(here)], env=env))
