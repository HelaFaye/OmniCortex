"""ComfyUI: add a custom node from a git repository."""
import argparse
import re

import comfy as C

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--folder", default="~/.omnicortex/comfyui")
p.add_argument("--repo", required=True)
a = p.parse_args()
m = re.fullmatch(r"https://(github\.com|gitlab\.com|codeberg\.org)/[\w.-]+/([\w.-]+?)(\.git)?/?", a.repo.strip())
if not m:
    raise SystemExit("an https URL of a repository on GitHub, GitLab or Codeberg")
C.need_git()
root = C.folder(a.folder)
if not (root / "main.py").exists():
    raise SystemExit(f"ComfyUI is not installed in {root}")
dest = root / "custom_nodes" / m.group(2)
C.clone_or_pull(a.repo.strip(), dest)
C.requirements(root, dest)
C.say(f"added {dest.name}. Restart the 'ComfyUI: run' job to load it.")
