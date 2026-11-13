#!/usr/bin/env python3
"""
Pick a draft model for speculative decoding, and a device for it (docs/ENGINE.md#a-draft-on-another-device).

A draft must share the big model's vocabulary, or llama-server refuses it:
- the same tokenizer type;
- vocab sizes within 128 of each other;
- the same BOS and EOS ids;
- the same token text at the ids compared (the first 1000, then every 97th).

Among the drafts that pass, the best fit is about a tenth of the big model's size; anything over 40% saves
little. The draft goes on a device the big model does not use, from `llama-server --list-devices`: the
GPU with the most free memory that holds it (for example the iGPU while the big model has the dGPU), or
the CPU when none does. The big model then keeps its GPU to itself and verifies the drafted tokens in one
pass.

    python scripts/draft_pick.py big.gguf --models-dir ~/models [--server llama-server] [--device Vulkan0]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_utils  # noqa: E402

VOCAB_SLACK = 128
BEST_RATIO = 0.1
MAX_RATIO = 0.4
_CACHE: dict = {}


def vocab_info(path: str) -> dict | None:
    """The parts of a GGUF's vocabulary that decide draft compatibility, or None without one."""
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key in _CACHE:
        return _CACHE[key]
    import gguf
    r = gguf.GGUFReader(path)
    f = r.fields.get("tokenizer.ggml.tokens")
    if f is None:
        _CACHE[key] = None
        return None
    n = len(f.data)
    ids = sorted(set(range(min(n, 1000))) | set(range(1000, n, 97)))

    def tok(i):
        return bytes(f.parts[f.data[i]]).decode("utf-8", "replace")
    info = {"model": gguf_utils.read_kv(r, "tokenizer.ggml.model"), "n": n,
            "bos": gguf_utils.read_kv(r, "tokenizer.ggml.bos_token_id"),
            "eos": gguf_utils.read_kv(r, "tokenizer.ggml.eos_token_id"),
            "arch": gguf_utils.read_kv(r, "general.architecture"),
            "tokens": {i: tok(i) for i in ids}}
    _CACHE[key] = info
    return info


def compatible(main: dict | None, cand: dict | None) -> tuple[bool, str]:
    if not main or not cand:
        return False, "no vocabulary in the file"
    if main["model"] != cand["model"]:
        return False, f"tokenizer {cand['model']} vs {main['model']}"
    if abs(main["n"] - cand["n"]) > VOCAB_SLACK:
        return False, f"vocab {cand['n']} vs {main['n']}"
    if (main["bos"], main["eos"]) != (cand["bos"], cand["eos"]):
        return False, "different BOS/EOS tokens"
    shared = set(main["tokens"]) & set(cand["tokens"])
    bad = next((i for i in sorted(shared) if main["tokens"][i] != cand["tokens"][i]), None)
    if bad is not None:
        return False, f"token {bad} differs"
    return True, "same vocabulary"


def candidates(main_path: str, paths: list[str], info=vocab_info) -> list[dict]:
    """Compatible drafts for main_path, best first."""
    size = os.path.getsize(main_path)
    try:
        mi = info(main_path)
    except Exception as e:
        raise ValueError(f"cannot read {os.path.basename(main_path)}: {e}")
    out = []
    for p in paths:
        if os.path.realpath(p) == os.path.realpath(main_path) or re.search(r"mmproj", os.path.basename(p), re.I):
            continue
        ratio = os.path.getsize(p) / size
        if ratio > MAX_RATIO:
            continue
        try:
            ok, why = compatible(mi, info(p))
        except Exception as e:
            ok, why = False, str(e)[:80]
        if ok:
            ci = info(p)
            score = abs(math.log(ratio / BEST_RATIO)) - (0.3 if ci.get("arch") == mi.get("arch") else 0)
            out.append({"path": p, "name": os.path.basename(p), "bytes": os.path.getsize(p),
                        "ratio": round(ratio, 3), "why": why, "same_arch": ci.get("arch") == mi.get("arch"),
                        "score": round(score, 3)})
    return sorted(out, key=lambda c: c["score"])


DEV_LINE = re.compile(r"^\s*([A-Za-z]+\d+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)")


def list_devices(server: str, timeout: float = 30) -> list[dict]:
    """llama-server --list-devices -> [{name, desc, total_mb, free_mb}] (empty when it cannot say)."""
    try:
        r = subprocess.run([server, "--list-devices"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_devices(r.stdout + "\n" + r.stderr)


def parse_devices(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = DEV_LINE.match(line)
        if m:
            out.append({"name": m.group(1), "desc": m.group(2), "total_mb": int(m.group(3)), "free_mb": int(m.group(4))})
    return out


def place(devices: list[dict], main_devices: list[str], draft_bytes: int, main_bytes: int | None = None) -> dict:
    """Where the draft goes: {"draft_device", "draft_ngl", "device" (for the big model, when it should be
    pinned to leave room), "why"}."""
    need_mb = draft_bytes * 1.15 / 2**20 + 512
    gpus = [d for d in devices if not d["name"].upper().startswith("CPU")]
    main = [d for d in gpus if d["name"] in main_devices]
    pin = None
    if not main and len(gpus) > 1:
        # unpinned, the big model spreads over every GPU: give it the biggest and the draft another
        big = max(gpus, key=lambda d: (d["free_mb"], d["total_mb"]))
        if main_bytes is None or big["free_mb"] > main_bytes / 2**20 * 1.1:
            main, pin = [big], big["name"]
    if not main:
        main = gpus                     # unpinned, the big model uses every GPU
    others = [d for d in gpus if d not in main and d["free_mb"] >= need_mb]
    if others:
        d = max(others, key=lambda x: x["free_mb"])
        return {"draft_device": d["name"], "draft_ngl": 99, "device": pin,
                "why": f"{d['name']} ({d['desc']}, {d['free_mb']} MiB free) is not used by the big model"
                       + (f"; the big model is pinned to {pin}" if pin else "")}
    return {"draft_device": None, "draft_ngl": 0, "device": None,
            "why": "no other GPU with room: the draft runs on the CPU while the GPU verifies"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--models-dir", action="append", default=[])
    p.add_argument("--server", help="llama-server, to ask which devices it sees")
    p.add_argument("--device", default="", help="the big model's devices, e.g. Vulkan0")
    a = p.parse_args(argv)
    paths = [x for d in a.models_dir for x in glob.glob(os.path.join(os.path.expanduser(d), "**", "*.gguf"),
                                                           recursive=True)]
    cands = candidates(a.model, paths)
    out = {"candidates": cands}
    if cands and a.server:
        out["placement"] = place(list_devices(a.server), [x for x in a.device.split(",") if x], cands[0]["bytes"],
                                 os.path.getsize(a.model))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
