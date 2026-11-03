#!/usr/bin/env python3
"""
Neuron-granular expert sharding, the realisable form (docs/INTEGRATED_OPTIONS.md): one suppressed variant of a model
per subject, each with only that subject's H-neurons scaled down, and Studio's per-subject routing sending each
request to its subject's variant.

    python scripts/expert_shards.py --gguf model-Q4_K_M.gguf --scale 0.3 \\
        --subject code=runs/h-code/h_neurons.json --subject math=runs/h-math/h_neurons.json --out models/shards

builds model-Q4_K_M.code.gguf, model-Q4_K_M.math.gguf with suppress_gguf.py and writes shards.json, the
subject -> model mapping. --apply URL sets it as Studio's models per subject (Settings → Models per subject), so the
subject classifier routes each prompt to its shard. Train a classifier per subject on that subject's TestQA items
(classifier.py on activations from those items) to get each subject's H-neurons.

What this is not: a speed-up, or one model with a router inside. Each shard is a whole model, the same size as the
original; a request touches one shard, and switching subjects on one GPU reloads the model. It keeps one subject's
correction from bleeding into another's answers, which a single suppression over every subject's neurons does.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(gguf: Path, subjects: dict[str, str], scale: float, out: Path, run=subprocess.call) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for subj, hn in sorted(subjects.items()):
        dst = out / f"{gguf.stem}.{subj}.gguf"
        rc = run([sys.executable, str(HERE / "suppress_gguf.py"), "--gguf", str(gguf), "--h_neurons", hn,
                  "--scale", f"{scale:g}", "--out", str(dst)])
        if rc != 0:
            raise RuntimeError(f"suppress_gguf.py failed for {subj} ({rc})")
        mapping[subj] = str(dst)
    (out / "shards.json").write_text(json.dumps({"base": str(gguf), "scale": scale, "subjects": mapping}, indent=1))
    return mapping


def apply(studio: str, token: str | None, mapping: dict[str, str], models: list[dict]) -> dict:
    """Map each shard's path to Studio's model id, then set Models per subject."""
    by_path = {m["path"]: m["id"] for m in models}
    sm = {s: by_path[p] for s, p in mapping.items() if p in by_path}
    missing = sorted(s for s, p in mapping.items() if p not in by_path)
    if missing:
        raise RuntimeError(f"Studio does not list these shards yet (put them in a models folder and refresh): {missing}")
    req = urllib.request.Request(studio.rstrip("/") + "/api/settings/subjects", method="POST",
                                 data=json.dumps({"subject_models": sm}).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--subject", action="append", default=[], help="subject=h_neurons.json (repeat)")
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", help="Studio URL: set the shards as its models per subject")
    ap.add_argument("--token")
    a = ap.parse_args(argv)
    subjects = dict(x.split("=", 1) for x in a.subject)
    if not subjects:
        raise SystemExit("give --subject subject=h_neurons.json at least once")
    mapping = build(Path(a.gguf), subjects, a.scale, Path(a.out))
    print(json.dumps(mapping, indent=1))
    if a.apply:
        with urllib.request.urlopen(urllib.request.Request(a.apply.rstrip("/") + "/api/models", headers=
                                    {"Authorization": f"Bearer {a.token}"} if a.token else {}), timeout=30) as r:
            models = json.loads(r.read())
        print(json.dumps(apply(a.apply, a.token, mapping, models), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
