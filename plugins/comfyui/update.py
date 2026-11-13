"""ComfyUI: update it and its custom nodes."""
import argparse

import comfy as C

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--folder", default="~/.omnicortex/comfyui")
a = p.parse_args()
C.need_git()
root = C.folder(a.folder)
C.clone_or_pull(C.REPO, root)
C.requirements(root, root)
nodes = root / "custom_nodes"
for d in sorted(nodes.iterdir()) if nodes.is_dir() else []:
    if (d / ".git").exists():
        C.run(["git", "-C", d, "pull", "--ff-only"], check=False)
        C.requirements(root, d)
if (nodes / "omnicortex_nodes").exists():
    C.omnicortex_nodes(root)
C.say("updated. Restart the 'ComfyUI: run' job to use it.")
