"""One-time move of OmniCortex's data from its former name's folders (it was called NeuronScope).

    ~/.neuronscope          -> ~/.omnicortex
    ~/.config/neuronscope   -> ~/.config/omnicortex      (tokens and secrets)

Runs at start (launcher, Studio, the CLI, the MCP server). Moves a folder only when the new one does not exist yet,
so it never overwrites anything; when both exist, it leaves the old one alone and says so once.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

OLD = "neuron" + "scope"                  # spelled out so a repository-wide rename leaves it alone


def pairs() -> list[tuple[Path, Path]]:
    home = Path.home()
    cfg = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return [(home / f".{OLD}", home / ".omnicortex"), (cfg / OLD, cfg / "omnicortex")]


def migrate(quiet: bool = False) -> list[str]:
    done = []
    for old, new in pairs():
        if not old.exists() or old.is_symlink():
            continue
        if new.exists():
            if not quiet and not (new / ".migrated-note").exists():
                print(f"[omnicortex] both {old} and {new} exist; using {new}. Move anything you still need from {old}.",
                      file=sys.stderr)
                try:
                    (new / ".migrated-note").write_text(str(old))
                except OSError:
                    pass
            continue
        try:
            shutil.move(str(old), str(new))
            done.append(f"{old} -> {new}")
        except OSError as e:
            if not quiet:
                print(f"[omnicortex] could not move {old} to {new}: {e}", file=sys.stderr)
    if done and not quiet:
        print("[omnicortex] moved " + "; ".join(done), file=sys.stderr)
    return done
