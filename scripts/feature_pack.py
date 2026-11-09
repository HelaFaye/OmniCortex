#!/usr/bin/env python3
"""
P2P feature exchange (docs/INTEGRATED_OPTIONS.md): share what you found inside a model without sharing the model.

A feature pack is a small zip: H-neuron sets (h_neurons.json), emotion vectors and steering directions (.npy),
SAE feature picks (JSON lists of feature ids with labels), llama.cpp control vectors (.gguf, small), steering and
graph presets (.json), plus a manifest naming the base model it was found on:

    {"schema": "omnicortex-feature-pack/1", "id", "title", "description", "kind",
     "base": {"hf_repo", "arch", "n_layers", "d_model", "gguf_sha256"?}, "files": [{"path", "sha256", "bytes"}]}

Sharing: send the zip over Connect's encrypted transfer like any file, or submit it to a review board
(github_board.py: an issue marked omnicortex-feature-pack, the pack inline up to ~45 KB; the board files it under
features/pending/ for a maintainer). Installing checks every file's hash and that the pack fits the model you
install it for (same layer count and width; the same repo when both name one). A pack made on another base model
is refused: neuron indices and directions mean nothing in a different network.

    python scripts/feature_pack.py pack --id code-hneurons --title "..." --kind h_neurons --hf-repo org/model \\
        --n-layers 32 --d-model 4096 runs/h/h_neurons.json --out code-hneurons.zip
    python scripts/feature_pack.py inspect code-hneurons.zip
    python scripts/feature_pack.py install code-hneurons.zip --n-layers 32 --d-model 4096 [--hf-repo org/model]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

SCHEMA = "omnicortex-feature-pack/1"
KINDS = ("h_neurons", "emotion_vectors", "sae_features", "control_vector", "steering", "graph", "mixed")
EXT = {".json", ".npy", ".gguf", ".txt", ".md"}
MAX_BYTES = 64 * 2**20
ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class PackError(ValueError):
    pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pack(files: list[str], meta: dict) -> bytes:
    if not ID.match(meta.get("id", "")):
        raise PackError("id: lower-case letters, digits, . _ -")
    if meta.get("kind") not in KINDS:
        raise PackError(f"kind: one of {', '.join(KINDS)}")
    base = meta.get("base") or {}
    if not (base.get("n_layers") and base.get("d_model")):
        raise PackError("the base model's layer count and width are required (they decide where a pack fits)")
    buf = io.BytesIO()
    entries = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            p = Path(f)
            if p.suffix.lower() not in EXT:
                raise PackError(f"{p.name}: only {', '.join(sorted(EXT))} files")
            b = p.read_bytes()
            name = p.name
            if any(e["path"] == name for e in entries):
                raise PackError(f"two files named {name}")
            z.writestr(name, b)
            entries.append({"path": name, "sha256": sha(b), "bytes": len(b)})
        man = {"schema": SCHEMA, "id": meta["id"], "title": str(meta.get("title") or meta["id"])[:120],
               "description": str(meta.get("description") or "")[:2000], "kind": meta["kind"],
               "base": {k: base[k] for k in ("hf_repo", "arch", "n_layers", "d_model", "gguf_sha256") if base.get(k)},
               "author": str(meta.get("author") or "")[:80], "files": entries}
        z.writestr("manifest.json", json.dumps(man, indent=1))
    out = buf.getvalue()
    if len(out) > MAX_BYTES:
        raise PackError("a feature pack is at most 64 MB")
    return out


def read(data: bytes) -> tuple[dict, dict[str, bytes]]:
    if len(data) > MAX_BYTES:
        raise PackError("larger than 64 MB")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise PackError("not a zip")
    names = z.namelist()
    if "manifest.json" not in names:
        raise PackError("no manifest.json")
    man = json.loads(z.read("manifest.json"))
    if man.get("schema") != SCHEMA or not ID.match(str(man.get("id", ""))) or man.get("kind") not in KINDS:
        raise PackError("not a version 1 feature pack")
    files = {}
    listed = {e["path"] for e in man.get("files") or []}
    for n in names:
        if n == "manifest.json":
            continue
        if "/" in n or "\\" in n or n.startswith(".") or n not in listed:
            raise PackError(f"unexpected file {n!r}")
    for e in man["files"]:
        if Path(e["path"]).suffix.lower() not in EXT or e["path"] not in names:
            raise PackError(f"bad entry {e['path']!r}")
        b = z.read(e["path"])
        if sha(b) != e["sha256"]:
            raise PackError(f"{e['path']}: hash mismatch (damaged or altered)")
        files[e["path"]] = b
    return man, files


def fits(man: dict, n_layers: int, d_model: int, hf_repo: str | None = None) -> tuple[bool, str]:
    b = man.get("base") or {}
    if int(b.get("n_layers") or -1) != int(n_layers) or int(b.get("d_model") or -1) != int(d_model):
        return False, (f"made on a model with {b.get('n_layers')} layers of width {b.get('d_model')}; "
                       f"this one has {n_layers} of {d_model}")
    if hf_repo and b.get("hf_repo") and b["hf_repo"].lower() != hf_repo.lower():
        return True, f"same shape, but made on {b['hf_repo']}, not {hf_repo}: likely a fine-tune of the same base; check it"
    return True, "fits"


def home() -> Path:
    return Path(os.environ.get("OC_HOME") or Path.home() / ".omnicortex") / "features" / "packs"


def install(data: bytes, n_layers: int, d_model: int, hf_repo: str | None = None, root: Path | None = None) -> dict:
    man, files = read(data)
    ok, why = fits(man, n_layers, d_model, hf_repo)
    if not ok:
        raise PackError(why)
    d = (root or home()) / man["id"]
    d.mkdir(parents=True, exist_ok=True)
    for name, b in files.items():
        (d / name).write_bytes(b)
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    return {"id": man["id"], "dir": str(d), "files": sorted(files), "note": why}


def installed(root: Path | None = None) -> list[dict]:
    r = root or home()
    out = []
    for m in sorted(r.glob("*/manifest.json")) if r.is_dir() else []:
        try:
            j = json.loads(m.read_text())
            out.append({k: j.get(k) for k in ("id", "title", "kind", "base")} | {"dir": str(m.parent)})
        except ValueError:
            continue
    return out


def board_doc(data: bytes) -> dict:
    """The issue body's JSON for a review board (the pack inline, base64)."""
    man, _ = read(data)
    if len(data) > 45000:
        raise PackError("too large to submit inline (45 KB); share it over Connect's transfer instead")
    return {"manifest": man, "zip_b64": base64.b64encode(data).decode()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("files", nargs="+")
    for k in ("id", "title", "kind", "out"):
        p.add_argument("--" + k, required=k in ("id", "kind", "out"))
    p.add_argument("--description", default="")
    p.add_argument("--author", default="")
    p.add_argument("--hf-repo")
    p.add_argument("--arch")
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--d-model", type=int, required=True)
    i = sub.add_parser("inspect")
    i.add_argument("zip")
    n = sub.add_parser("install")
    n.add_argument("zip")
    n.add_argument("--n-layers", type=int, required=True)
    n.add_argument("--d-model", type=int, required=True)
    n.add_argument("--hf-repo")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "pack":
            data = pack(a.files, {"id": a.id, "title": a.title, "kind": a.kind, "description": a.description,
                                  "author": a.author, "base": {"hf_repo": a.hf_repo, "arch": a.arch,
                                                               "n_layers": a.n_layers, "d_model": a.d_model}})
            Path(a.out).write_bytes(data)
            print(f"{a.out}: {len(data)} bytes")
        elif a.cmd == "inspect":
            print(json.dumps(read(Path(a.zip).read_bytes())[0], indent=1))
        elif a.cmd == "install":
            print(json.dumps(install(Path(a.zip).read_bytes(), a.n_layers, a.d_model, a.hf_repo), indent=1))
        else:
            print(json.dumps(installed(), indent=1))
    except PackError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
