#!/usr/bin/env python3
"""
OmniCortex <-> NVIDIA NeMo Gym (docs/DORY.md, "Training with NeMo RL and Gym").

NeMo Gym serves RL environments (a dataset, an agent, a verifier that returns a reward) that NeMo RL trains on with
GRPO. This bridge makes OmniCortex's evaluation an environment:

    export   TestQA bank items, Fix This Mistake pairs or neuro-fuzzing banks as a Gym dataset (JSONL: each row's
             responses_create_params.input is the chat; the task rides along in omnicortex_task)
    reward   TestQA's grader turned into a reward that does not pay for confident guessing:
                 correct +1, declined `abstain` (default 0.25), confidently wrong `wrong` (default -1)
             A model trained on accuracy alone learns to guess; this is the calibration OmniCortex measures, as a
             training signal. Optionally shaped by the H-neuron classifier: a correct answer given while the
             model's own hallucination risk was high earns a little less (risk_weight x risk), so the policy is
             pushed toward answers it actually knows.

The Gym side is integrations/nemo_gym/omnicortex_testqa (copy it into Gym's resources_servers/).

    python scripts/gym_bridge.py export --bank qa/bank --out data/train.jsonl [--fuzz qa/fuzz/round1] [--harvest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

GRADABLE = ("qa", "reasoning")
DEFAULTS = {"abstain": 0.25, "wrong": -1.0, "risk_weight": 0.0}


def row(task: dict) -> dict:
    t = {k: v for k, v in task.items() if not k.startswith("_")}
    return {"uuid": str(t["id"]), "responses_create_params": {"input": [{"role": "user", "content": t["prompt"]}]},
            "omnicortex_task": t}


def export(bank: list[str], harvest: bool, out: Path, subjects=None, limit: int = 0) -> int:
    import testqa
    rows = [row(t) for t in testqa.load_tasks(bank, set(GRADABLE), set(subjects) if subjects else None, limit)]
    if harvest:
        import harvest as H
        for it in H.items():
            rows.append({"uuid": "harvest-" + it["id"],
                         "responses_create_params": {"input": [{"role": m["role"], "content": m["content"]} for m in it["context"]]},
                         "omnicortex_task": {"id": "harvest-" + it["id"], "kind": "reference",
                                             "reference": it["chosen"], "rejected": it["rejected"]}})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def response_text(response) -> str:
    """The final text of an OpenAI Responses object (dict or pydantic), reasoning left out."""
    if response is None:
        return ""
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if isinstance(response, str):
        return response
    if response.get("output_text"):
        return str(response["output_text"])
    parts = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text") or "")
    return "".join(parts)


def overlap(a: str, b: str) -> float:
    import re
    x, y = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(x & y) / max(1, len(x | y))


def reward(task: dict, text: str, risk: float | None = None, cfg: dict | None = None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    if task.get("kind") == "reference":            # a Fix This Mistake pair: closer to the correction than the mistake
        a, b = overlap(text, task["reference"]), overlap(text, task.get("rejected", ""))
        verdict = "correct" if a > b and a >= 0.5 else "wrong"
    else:
        import testqa
        verdict = testqa.grade(task, text, argparse.Namespace(allow_exec=False, exec_timeout=5))[0]
        verdict = "correct" if verdict in ("correct", "answered") else verdict
    r = {"correct": 1.0, "abstained": c["abstain"]}.get(verdict, c["wrong"])
    if verdict == "correct" and risk is not None and c["risk_weight"]:
        r -= c["risk_weight"] * max(0.0, min(1.0, float(risk)))
    return {"reward": round(r, 4), "verdict": verdict, **({"risk": risk} if risk is not None else {})}


def risk_of(studio: str, model: str, messages: list[dict], text: str) -> float | None:
    """The model's own hallucination risk for this reply (the mean per-token risk), from a Studio where the model
    has a classifier: the same check as **check** under a reply (/api/trace)."""
    import os
    import urllib.request
    req = urllib.request.Request(studio.rstrip("/") + "/api/trace", method="POST",
                                 data=json.dumps({"model": model, "messages": messages, "text": text}).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + os.environ["OC_STUDIO_TOKEN"]}
                                             if os.environ.get("OC_STUDIO_TOKEN") else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read())
        return float(j["mean"]) if isinstance(j.get("mean"), (int, float)) else None
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--bank", action="append", default=[], help="TestQA bank folder or file (default qa/bank)")
    e.add_argument("--fuzz", action="append", default=[], help="a neuro-fuzzing bank folder")
    e.add_argument("--harvest", action="store_true", help="also the Fix This Mistake pairs")
    e.add_argument("--subject", action="append")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n = export((a.bank or [str(ROOT / "qa" / "bank")]) + a.fuzz, a.harvest, Path(a.out), a.subject, a.limit)
    print(f"{n} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
