"""
The staging pool (docs/POSE.md): pending/ items from the sieve, approved/ and discarded/ by review, and the
export of approved items as a dataset finetune.py trains on.

Export writes, into --out:
    with frame images  dpo_vision.jsonl, sft_vision.jsonl, images/   (finetune.py --vision --dpo)
    without            dpo.jsonl, sft.jsonl                           (finetune.py --dpo)
each record: the standard prompt, the chosen (corrected) pose as the assistant's JSON reply, the rejected
(as tracked) one, and meta (failed joint, error type, frame, track).

Also the backend of the add-on's review page: as a Studio plugin check it reads one command as `text`:
    {"op": "list", "dir": "runs/pose/staging"}     counts and the pending items' summaries
    {"op": "get", "dir": ..., "id": ...}           one item, with its frame image (base64)
    {"op": "approve" | "discard" | "restore", "dir": ..., "id": ...}
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

from sieve import pose_text

STATES = ("pending", "approved", "discarded")
MAX_IMAGE = 3 * 2**20


def items(staging: Path, state: str) -> list[dict]:
    d = staging / state
    out = []
    for fp in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.is_dir() else []:
        try:
            out.append(json.loads(fp.read_text()))
        except ValueError:
            continue
    return out


def find(staging: Path, iid: str):
    if not iid or not all(c in "0123456789abcdef" for c in iid):
        raise ValueError("bad id")
    for s in STATES:
        p = staging / s / f"{iid}.json"
        if p.exists():
            return s, p
    raise KeyError(f"no item {iid}")


def move(staging: Path, iid: str, to: str) -> dict:
    s, p = find(staging, iid)
    it = json.loads(p.read_text())
    it["status"] = to
    (staging / to).mkdir(parents=True, exist_ok=True)
    (staging / to / p.name).write_text(json.dumps(it, indent=1))
    if s != to:
        p.unlink()
    return {"id": iid, "status": to}


def counts(staging: Path) -> dict:
    return {s: len(list((staging / s).glob("*.json"))) if (staging / s).is_dir() else 0 for s in STATES}


def export(staging: Path, out: Path) -> dict:
    ok = items(staging, "approved")
    out.mkdir(parents=True, exist_ok=True)
    vision = bool(ok) and all(it.get("image") and Path(it["image"]).exists() for it in ok)
    dpo, sft = [], []
    if vision:
        (out / "images").mkdir(exist_ok=True)
    for it in ok:
        meta = {"source": "pose-sieve", "frame": it["frame"], "track": it["track"], "failures": it["failures"]}
        if vision:
            src = Path(it["image"])
            rel = f"images/{it['id']}{src.suffix}"
            shutil.copyfile(src, out / rel)
            user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["prompt"]}]}
            reply = lambda t: [{"role": "assistant", "content": [{"type": "text", "text": t}]}]  # noqa: E731
            dpo.append({"images": [rel], "prompt": [user], "chosen": reply(pose_text(it["chosen"])),
                        "rejected": reply(pose_text(it["rejected"])), "meta": meta})
            sft.append({"images": [rel], "messages": [user] + reply(pose_text(it["chosen"])), "meta": meta})
        else:
            user = {"role": "user", "content": it["prompt"]}
            dpo.append({"prompt": [user], "chosen": [{"role": "assistant", "content": pose_text(it["chosen"])}],
                        "rejected": [{"role": "assistant", "content": pose_text(it["rejected"])}], "meta": meta})
            sft.append({"messages": [user, {"role": "assistant", "content": pose_text(it["chosen"])}], "meta": meta})
    names = ("dpo_vision.jsonl", "sft_vision.jsonl") if vision else ("dpo.jsonl", "sft.jsonl")
    for name, rows in zip(names, (dpo, sft)):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"pairs": len(dpo), "vision": vision, "files": list(names), "out": str(out)}


def command(cmd: dict) -> dict:
    staging = Path(str(cmd.get("dir") or "runs/pose/staging")).expanduser()
    op = cmd.get("op")
    if op == "list":
        pend = items(staging, "pending")
        return {"label": f"{len(pend)} to review", "counts": counts(staging), "dir": str(staging.resolve()),
                "items": [{"id": i["id"], "frame": i["frame"], "track": i["track"], "image": bool(i.get("image")),
                           "failures": i["failures"]} for i in pend[:500]]}
    if op == "get":
        s, p = find(staging, str(cmd.get("id") or ""))
        it = json.loads(p.read_text())
        img = it.get("image")
        if img and Path(img).exists() and Path(img).stat().st_size <= MAX_IMAGE:
            mime = "image/png" if img.endswith(".png") else "image/webp" if img.endswith(".webp") else "image/jpeg"
            it["image_data"] = f"data:{mime};base64," + base64.b64encode(Path(img).read_bytes()).decode()
        return {"label": it["id"], "item": it, "state": s}
    if op in ("approve", "discard", "restore"):
        to = {"approve": "approved", "discard": "discarded", "restore": "pending"}[op]
        r = move(staging, str(cmd.get("id") or ""), to)
        return {"label": f"{r['id']} {to}", **r, "counts": counts(staging)}
    raise ValueError("op: list, get, approve, discard or restore")


def check_main() -> int:
    """Studio plugin check: the command arrives as the payload's text."""
    payload = json.loads(sys.stdin.read() or "{}")
    try:
        out = command(json.loads(payload.get("text") or "{}"))
    except (ValueError, KeyError, OSError) as e:
        out = {"label": "error", "flag": True, "error": str(e).strip("'\"")}
    print(json.dumps(out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--export", help="write the approved items as a training dataset here")
    ap.add_argument("--approve-all", action="store_true", help="approve every pending item (no review)")
    a = ap.parse_args(argv)
    st = Path(a.staging)
    if a.approve_all:
        for it in items(st, "pending"):
            move(st, it["id"], "approved")
    print(json.dumps(counts(st)))
    if a.export:
        print(json.dumps(export(st, Path(a.export))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
