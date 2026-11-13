#!/usr/bin/env python3
"""
Speculative quantization mask rescaling (docs/INTEGRATED_OPTIONS.md): quantize a GGUF with more precision where the
model makes confident mistakes and less where it is safe, instead of one type everywhere.

How "where" is decided, from what you have:
    --h-neurons h_neurons.json   classifier.py's H-neurons: a layer's share of them is its risk
    --review summary.json        a neuron review summary (Review → Summary, or /api/review/summary): per-layer
                                 association with wrong answers
    --risk-layers 12,13,20-24    by hand
Layers are ranked by risk. The top --high-fraction get --high (default q8_0) for their MLP and attention weights,
the bottom --low-fraction get --low (default q3_k), the rest the base type you quantize to.

And per neuron, as far as GGUF allows: quant types are per tensor, not per neuron, so the per-neuron part is the
importance matrix. --imatrix-from writes the mistake contexts (TestQA failures, Fix This Mistake pairs) to a text
file; llama-imatrix on it weighs exactly the neurons those contexts use, and llama-quantize --imatrix then spends
precision on them inside every tensor.

    python scripts/quant_mask.py plan --gguf model-f16.gguf --h-neurons runs/h/h_neurons.json --base Q4_K_M \\
        --out model-masked.gguf [--imatrix imatrix.dat] [--run]
    python scripts/quant_mask.py contexts --results runs/tq.json --harvest --out mistakes.txt

Check the result against a uniform quant of the same size with TestQA (Retrain → compare), as with any quant.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TENSORS = ("ffn_down", "ffn_up", "ffn_gate", "attn_q", "attn_k", "attn_v", "attn_output")
TYPES = {"f32", "f16", "bf16", "q8_0", "q6_k", "q5_k", "q5_0", "q5_1", "q4_k", "q4_0", "q4_1", "q3_k", "q2_k",
         "iq4_xs", "iq4_nl", "iq3_s", "iq3_xxs", "iq2_s", "iq2_xs", "iq2_xxs", "iq1_s", "iq1_m", "tq1_0", "tq2_0"}


def parse_layers(s: str) -> list[int]:
    out = []
    for part in str(s).replace(" ", "").split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def risk_from_h(path) -> dict[int, float]:
    d = json.loads(Path(path).read_text())
    by = {int(k): len(v) for k, v in (d.get("by_layer") or {}).items()}
    tot = sum(by.values()) or 1
    n = int(d.get("n_layers") or (max(by) + 1 if by else 0))
    return {i: by.get(i, 0) / tot for i in range(n)}


def risk_from_review(path) -> dict[int, float]:
    d = json.loads(Path(path).read_text())
    rows = d.get("layers") or d.get("per_layer") or []
    out = {}
    for r in rows:
        if isinstance(r, dict) and "layer" in r:
            out[int(r["layer"])] = float(r.get("value", r.get("association", r.get("score", 0))) or 0)
    if not out and isinstance(d.get("top"), list):
        for r in d["top"]:
            out[int(r["layer"])] = out.get(int(r["layer"]), 0) + abs(float(r.get("value", 0)))
    return out


def plan(risk: dict[int, float], n_layers: int, high="q8_0", low="q3_k", high_fraction=0.2, low_fraction=0.3,
         tensors=TENSORS) -> dict:
    for t in (high, low):
        if t not in TYPES:
            raise ValueError(f"unknown ggml type {t!r}")
    if not 0 <= high_fraction <= 1 or not 0 <= low_fraction <= 1 or high_fraction + low_fraction > 1:
        raise ValueError("fractions between 0 and 1, adding up to at most 1")
    layers = list(range(n_layers))
    ranked = sorted(layers, key=lambda i: (-risk.get(i, 0.0), i))
    n_hi = round(n_layers * high_fraction)
    hi = sorted(ranked[:n_hi]) if any(risk.get(i, 0) > 0 for i in ranked[:n_hi]) else []
    rest = [i for i in ranked if i not in hi]
    n_lo = round(n_layers * low_fraction)
    lo = sorted(rest[-n_lo:]) if n_lo else []
    rules = [f"blk\\.{i}\\.({'|'.join(tensors)})={high}" for i in hi] + \
            [f"blk\\.{i}\\.({'|'.join(tensors)})={low}" for i in lo]
    return {"high": {"type": high, "layers": hi}, "low": {"type": low, "layers": lo},
            "base_layers": sorted(i for i in layers if i not in hi and i not in lo), "rules": rules,
            "risk": {str(i): round(risk.get(i, 0.0), 5) for i in layers}}


def command(p: dict, gguf: str, out: str, base: str, quantize: str, rules_file: Path, imatrix: str | None) -> list[str]:
    rules_file.write_text("".join(r + "\n" for r in p["rules"]))
    cmd = [quantize]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    if p["rules"]:
        cmd += ["--tensor-type-file", str(rules_file)]
    return cmd + [gguf, out, base]


def n_layers_of(gguf: str) -> int | None:
    try:
        import gguf
        r = gguf.GGUFReader(gguf)
        for f in r.fields.values():
            if f.name.endswith(".block_count"):
                return int(f.parts[f.data[0]][0])
    except Exception:
        return None
    return None


def contexts(results: list[str], use_harvest: bool, out: Path) -> int:
    """Mistake contexts as plain text for llama-imatrix -f."""
    texts = []
    for r in results:
        d = json.loads(Path(r).read_text())
        for label, rows in (d.get("raw") or {}).items():
            for row in rows:
                if row.get("verdict") == "wrong" and row.get("prompt"):
                    texts.append(str(row["prompt"]) + "\n" + str(row.get("text", "")))
    if use_harvest:
        import harvest
        for it in harvest.items():
            texts.append("\n".join(m["content"] for m in it["context"]) + "\n" + it["chosen"])
    out.write_text("\n\n".join(texts))
    return len(texts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--gguf", required=True, help="the F16/BF16 (or Q8_0) GGUF to quantize from")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Q4_K_M", help="the quant for everything else")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h-neurons")
    src.add_argument("--review")
    src.add_argument("--risk-layers")
    p.add_argument("--layers", type=int, help="layer count (read from the GGUF when omitted)")
    p.add_argument("--high", default="q8_0")
    p.add_argument("--low", default="q3_k")
    p.add_argument("--high-fraction", type=float, default=0.2)
    p.add_argument("--low-fraction", type=float, default=0.3)
    p.add_argument("--imatrix", help="importance matrix from llama-imatrix (e.g. on the contexts below)")
    p.add_argument("--quantize", default=shutil.which("llama-quantize") or "llama-quantize")
    p.add_argument("--run", action="store_true", help="run llama-quantize (otherwise print the plan and command)")
    c = sub.add_parser("contexts", help="mistake contexts for llama-imatrix -f")
    c.add_argument("--results", action="append", default=[], help="TestQA results JSON")
    c.add_argument("--harvest", action="store_true", help="also the Fix This Mistake pairs")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE))
    if a.cmd == "contexts":
        print(f"{contexts(a.results, a.harvest, Path(a.out))} contexts -> {a.out}  "
              f"(next: llama-imatrix -m MODEL.gguf -f {a.out} -o imatrix.dat)")
        return 0
    n = a.layers or n_layers_of(a.gguf)
    if a.h_neurons:
        risk = risk_from_h(a.h_neurons)
        n = n or len(risk)
    elif a.review:
        risk = risk_from_review(a.review)
    else:
        risk = {i: 1.0 for i in parse_layers(a.risk_layers)}
    if not n:
        raise SystemExit("cannot tell the layer count: pass --layers (or pip install gguf)")
    pl = plan(risk, n, a.high, a.low, a.high_fraction, a.low_fraction)
    rules = Path(a.out).with_suffix(".tensor-types.txt")
    cmd = command(pl, a.gguf, a.out, a.base, a.quantize, rules, a.imatrix)
    print(json.dumps({k: pl[k] for k in ("high", "low")}, indent=1))
    print(f"{len(pl['base_layers'])} layers at {a.base}; rules in {rules}")
    print("$ " + " ".join(cmd))
    if a.run:
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
