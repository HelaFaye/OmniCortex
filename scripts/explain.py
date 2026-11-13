#!/usr/bin/env python3
"""
One answer to "how do I explain this reply?" (docs/EXPLAIN.md).

OmniCortex has several ways to look inside a reply, built at different times:
- the H-Neuron risk check (CETT through llama.cpp, or PyTorch);
- feature dictionaries and emotion vectors;
- component interventions (interventions.py);
- attribution graphs over transcoders (feature_circuits.py);
- CPU-friendly saliency and weights-only attribution (neuron_attrib.py).

Each answers a different question and needs different things. plan() takes what this model and this computer
have and returns every method, in the order to try it: what it answers, whether it can run here, why not, what
it costs, and how to start it (a button on the reply, a page, or a prefilled job).
"""
from __future__ import annotations

import importlib.util
import json
import urllib.parse


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def capabilities(model: dict, settings: dict, cett: str | None, inventory: dict) -> dict:
    """What plan() needs to know, from Studio's view of one model. inventory: features_api.list_all()."""
    hf = settings.get("hf_model") or settings.get("tokenizer") or model.get("hf_model")

    def mine(rows):
        return [d for d in rows or [] if hf and d.get("model") and
                (d["model"] == hf or str(d["model"]).rstrip("/").split("/")[-1] == hf.rstrip("/").split("/")[-1])]
    dicts = mine(inventory.get("dicts"))
    return {"model": model.get("id"), "gguf": model.get("path") if str(model.get("path", "")).endswith(".gguf") else None,
            "hf": hf, "cett": bool(cett), "classifier": bool(settings.get("classifier")),
            "torch": _have("torch"), "transformers": _have("transformers"),
            "saes": sum(1 for d in dicts if not d.get("target_site")),
            "transcoders": sum(1 for d in dicts if d.get("target_site")),
            "emotions": len(mine(inventory.get("emotions")))}


def _job(kind: str, **vals) -> str:
    return "/jobs#" + urllib.parse.urlencode({"kind": kind, **{k: v for k, v in vals.items() if v}})


def plan(c: dict, prompt: str = "", reply: str = "") -> list[dict]:
    torch_ok = c["torch"] and c["transformers"]
    no_hf = "needs the model's Hugging Face weights: set its HF model under its settings (Tokenizer / HF model)"
    no_torch = "needs PyTorch and transformers (Setup → Requirements → Features)"
    rows = [
        {"id": "risk", "question": "Is this a confident guess?",
         "method": "H-Neuron risk per token, from the model's own neurons (CETT and its classifier)",
         "available": c["cett"] and c["classifier"],
         "why": None if c["cett"] and c["classifier"] else
         ("needs a classifier for this model: run the H-Neuron pipeline (docs/PIPELINE.md), then pick it in the "
          "model's settings" if c["cett"] else "needs llama-cett-dump: Setup → Build llama.cpp"),
         "cost": "one pass over the reply", "action": {"type": "check"}},
        {"id": "read", "question": "Which concepts and emotions run through it?",
         "method": "feature dictionaries and emotion vectors, token by token",
         "available": bool(c["saes"] or c["emotions"]) and torch_ok,
         "why": None if (c["saes"] or c["emotions"]) and torch_ok else
         (no_torch if not torch_ok else "needs a feature dictionary or emotion vectors for this model "
          "(Jobs → Interpret → Features, or import one from SAELens)"),
         "cost": "one pass with hooks", "action": {"type": "link", "href": "/features#read"}},
        {"id": "trace", "question": "Which heads and neurons chose its next word?",
         "method": "direct logit attribution, then ablate the top ones (interventions.py trace / circuit)",
         "available": torch_ok and bool(c["hf"]), "why": None if torch_ok and c["hf"] else (no_torch if not torch_ok else no_hf),
         "cost": "a few passes", "action": {"type": "link", "href": _job("iv_circuit", model=c["hf"], prompt=(prompt + reply)[-2000:])}},
        {"id": "contrast", "question": "What makes it answer this instead of that?",
         "method": "activation patching between this prompt and a changed one (interventions.py act-patch)",
         "available": torch_ok and bool(c["hf"]), "why": None if torch_ok and c["hf"] else (no_torch if not torch_ok else no_hf),
         "cost": "one pass per layer and position", "action": {"type": "link", "href": _job("iv_act_patch", model=c["hf"], clean=prompt[-2000:])}},
        {"id": "graph", "question": "What chain of features led there?",
         "method": "an attribution graph over transcoders (circuit tracing)",
         "available": torch_ok and c["transcoders"] > 0,
         "why": None if torch_ok and c["transcoders"] else (no_torch if not torch_ok else
                                                           "needs transcoders for this model (Jobs → Features 2 with a target site)"),
         "cost": "heavy: gradients for every active feature", "action": {"type": "link", "href": _job("feat_graph", model=c["hf"], prompt=prompt[-1000:])}},
        {"id": "saliency", "question": "Which neurons carried its answer? (on a CPU or an older GPU)",
         "method": "gradient x activation in float32 (neuron_attrib.py saliency)",
         "available": torch_ok and bool(c["hf"]), "why": None if torch_ok and c["hf"] else (no_torch if not torch_ok else no_hf),
         "cost": "one forward and backward pass", "action": {"type": "link", "href": _job("attrib_saliency", model_path=c["hf"])}},
        {"id": "weights", "question": "Which neurons can write these words at all?",
         "method": "weights-only direct logit attribution from the GGUF (no forward pass, no PyTorch)",
         "available": bool(c["gguf"]), "why": None if c["gguf"] else "needs a GGUF file",
         "cost": "seconds per layer", "action": {"type": "link", "href": _job("attrib_weights", gguf=c["gguf"])}},
    ]
    best = next((r["id"] for r in rows if r["available"]), None)
    for r in rows:
        r["recommended"] = r["id"] == best
    return rows


if __name__ == "__main__":
    import sys
    print(json.dumps(plan(json.loads(sys.argv[1]) if len(sys.argv) > 1 else
                          {"model": "m", "gguf": "m.gguf", "hf": None, "cett": False, "classifier": False, "torch": False,
                           "transformers": False, "saes": 0, "transcoders": 0, "emotions": 0}), indent=1))
