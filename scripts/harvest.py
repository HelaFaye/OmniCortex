#!/usr/bin/env python3
"""
Continuous deficit harvesting, "Fix This Mistake" (docs/INTEGRATED_OPTIONS.md).

Under any reply in Studio's chat, Fix this mistake takes your correction. It keeps:
  - the conversation up to that reply (the context), the reply (rejected) and your correction (chosen);
  - the reply's activation profile, recorded in the neuron review as a verdict "wrong" observation, so the
    neurons active on confident mistakes are counted there;
  - whether the H-neuron classifier flagged it, if the model has one.

Pairs gather in ~/.omnicortex/harvest/. When enough have gathered and the computer is idle, Studio can run a
short micro-tune on them (finetune.py --dpo, a few dozen steps, on the base model you name). The result is an
adapter in runs/harvest/; nothing is swapped in until you choose to (Retrain shows it like any other run).

    python scripts/harvest.py list
    python scripts/harvest.py export --out runs/harvest/data         # sft.jsonl + dpo.jsonl for finetune.py
    python scripts/harvest.py train --model org/base --out runs/harvest/run1 [--min 20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULTS = {"enabled": True, "train_model": "", "min_pairs": 20, "idle_minutes": 20, "steps": 40, "auto_train": False}


def root(home=None) -> Path:
    import os
    return Path(home or os.environ.get("OC_HOME") or Path.home() / ".omnicortex") / "harvest"


def _text(c) -> str:
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def add(model: str, messages: list[dict], reply: str, correction: str, note: str = "", risk=None,
        home=None) -> dict:
    if not correction.strip():
        raise ValueError("the correction is empty")
    if correction.strip() == reply.strip():
        raise ValueError("the correction is the same as the reply")
    ctx = [{"role": m.get("role"), "content": _text(m.get("content"))} for m in messages
           if m.get("role") in ("system", "user", "assistant")][-12:]
    if not ctx or ctx[-1]["role"] != "user":
        raise ValueError("the conversation must end with the question the reply answered")
    iid = hashlib.sha1(json.dumps([model, ctx, reply]).encode()).hexdigest()[:12]
    item = {"id": iid, "t": time.time(), "model": model, "context": ctx, "rejected": reply, "chosen": correction,
            "note": note[:500], "risk": risk, "used_in": []}
    d = root(home) / "pairs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{iid}.json").write_text(json.dumps(item, indent=1))
    return item


def items(home=None) -> list[dict]:
    d = root(home) / "pairs"
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return sorted(out, key=lambda x: x["t"])


def remove(iid: str, home=None) -> bool:
    if not iid.isalnum():
        return False
    p = root(home) / "pairs" / f"{iid}.json"
    if p.exists():
        p.unlink()
        return True
    return False


def unused(home=None) -> list[dict]:
    return [i for i in items(home) if not i.get("used_in")]


def export(out: Path, home=None, only_unused: bool = False) -> dict:
    rows = unused(home) if only_unused else items(home)
    out.mkdir(parents=True, exist_ok=True)
    sft, dpo = [], []
    for it in rows:
        sft.append({"messages": it["context"] + [{"role": "assistant", "content": it["chosen"]}],
                    "meta": {"source": "harvest", "id": it["id"], "model": it["model"]}})
        dpo.append({"prompt": it["context"], "chosen": [{"role": "assistant", "content": it["chosen"]}],
                    "rejected": [{"role": "assistant", "content": it["rejected"]}],
                    "meta": {"source": "harvest", "id": it["id"], "model": it["model"]}})
    (out / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sft))
    (out / "dpo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dpo))
    return {"pairs": len(rows), "out": str(out), "ids": [r["id"] for r in rows]}


def mark_used(ids: list[str], run: str, home=None) -> None:
    for iid in ids:
        p = root(home) / "pairs" / f"{iid}.json"
        if p.exists():
            it = json.loads(p.read_text())
            it["used_in"] = sorted(set(it.get("used_in") or []) | {run})
            p.write_text(json.dumps(it, indent=1))


def train(model: str, out: Path, steps: int = 40, min_pairs: int = 1, home=None, method: str = "qlora") -> int:
    data = out / "data"
    ex = export(data, home, only_unused=True)
    if ex["pairs"] < min_pairs:
        print(f"{ex['pairs']} new pairs; {min_pairs} needed")
        return 0
    cmd = [sys.executable, str(HERE / "finetune.py"), "--model", model, "--data", str(data), "--out", str(out / "adapter"),
           "--method", method, "--dpo", "--max-steps", str(max(1, steps // 2)), "--dpo-max-steps", str(steps),
           "--logging-steps", "1"]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc == 0:
        mark_used(ex["ids"], str(out), home)
        print(f"micro-tune done on {ex['pairs']} pairs -> {out / 'adapter'}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--unused", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=40)
    t.add_argument("--min", type=int, default=1)
    t.add_argument("--method", default="qlora", choices=["qlora", "lora"])
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for it in items():
            print(f"{it['id']}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(it['t']))}  {it['model'][:30]:30}  "
                  f"{'used' if it.get('used_in') else 'new':4}  {it['context'][-1]['content'][:60]!r}")
        return 0
    if a.cmd == "export":
        print(json.dumps(export(Path(a.out), only_unused=a.unused)))
        return 0
    return train(a.model, Path(a.out), a.steps, a.min, method=a.method)


if __name__ == "__main__":
    sys.exit(main())
