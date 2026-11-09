#!/usr/bin/env python3
"""
Preflight check for the H-Neurons pipeline.

Loads only the config and a meta-device skeleton (no weights, no VRAM), then
reports the numbers that decide whether the run is feasible:

  - which down_proj modules exist, and whether a vision tower pollutes them
  - feature count per sample, storage per sample, total storage
  - peak RAM for the sklearn classifier (this is what usually kills the run)

Run this before downloading 19GB of weights.

    python preflight.py --model_path Qwen/Qwen3-8B --n_pairs 400
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import TEXT_LAYER_RE, text_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--n_pairs", type=int, default=400,
                   help="Balanced pairs per class you intend to train on")
    p.add_argument("--locations", type=int, default=4,
                   help="Number of activation locations you will extract")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p.parse_args()






def main():
    args = parse_args()

    cfg = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    tcfg = text_config(cfg)

    n_layers = tcfg.num_hidden_layers
    d_inter = tcfg.intermediate_size
    d_hidden = tcfg.hidden_size

    print(f"model type       : {cfg.model_type}")
    print(f"nested text_config: {hasattr(cfg, 'text_config')}")
    print(f"layers           : {n_layers}")
    print(f"hidden_size      : {d_hidden}")
    print(f"intermediate_size: {d_inter}")

    # Instantiate on meta so no memory is allocated, just to enumerate modules.
    print("\nEnumerating down_proj modules (meta device, no weights loaded)...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=args.trust_remote_code
        )

    matched, unmatched = [], []
    for name, mod in model.named_modules():
        if "down_proj" not in name:
            continue
        if TEXT_LAYER_RE.search(name):
            matched.append(name)
        else:
            unmatched.append(name)

    print(f"  text decoder down_proj : {len(matched)}")
    print(f"  OTHER down_proj        : {len(unmatched)}")
    if unmatched:
        print("\n  !! These would be hooked by the repo's naive `'down_proj' in name`")
        print("     check and would corrupt the flat-index -> (layer, neuron) map:")
        for n in unmatched[:8]:
            print(f"       {n}")
        if len(unmatched) > 8:
            print(f"       ... and {len(unmatched) - 8} more")

    if len(matched) != n_layers:
        print(f"\n  !! Expected {n_layers} matched modules, found {len(matched)}.")
        print("     The layer regex needs adjusting for this architecture.")

    # Verify the intermediate dim on a real module rather than trusting config.
    if matched:
        probe = dict(model.named_modules())[matched[0]]
        in_features = probe.in_features
        if in_features != d_inter:
            print(f"\n  !! {matched[0]}.in_features = {in_features}, "
                  f"config says {d_inter}. Trust the module.")
            d_inter = in_features

    n_feat = n_layers * d_inter
    print(f"\nfeatures per sample : {n_feat:,}")

    # Storage. float16 halves the repo's float32 without losing anything that
    # matters for a linear classifier on normalised contributions.
    per_sample_f32 = n_feat * 4 / 1e9
    per_sample_f16 = n_feat * 2 / 1e9
    n_samples = args.n_pairs * 2
    total_f16 = per_sample_f16 * n_samples * args.locations
    print(f"per sample (fp32)   : {per_sample_f32 * 1000:.1f} MB")
    print(f"per sample (fp16)   : {per_sample_f16 * 1000:.1f} MB")
    print(f"total, {args.locations} locations : {total_f16:.1f} GB (fp16)")

    # Classifier RAM. 3-vs-1 mode loads answer-token rows for both classes plus
    # other-token rows for both classes, so 2x the sample count.
    rows_1v1 = n_samples
    rows_3v1 = n_samples * 2
    print("\nsklearn peak RAM (dense X):")
    for label, rows in (("1-vs-1", rows_1v1), ("3-vs-1", rows_3v1)):
        gb32 = rows * n_feat * 4 / 1e9
        gb64 = rows * n_feat * 8 / 1e9
        print(f"  {label}: {rows} rows -> {gb32:.1f} GB fp32, "
              f"{gb64:.1f} GB after liblinear promotes to fp64")
    print("\n  liblinear copies to float64. If the fp64 column exceeds your RAM,")
    print("  use solver='saga' with float32, or drop to 1-vs-1, or cut n_pairs.")

    # Weights footprint.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nparameters        : {n_params / 1e9:.2f} B")
    print(f"bf16 weights      : {n_params * 2 / 1e9:.1f} GB")
    print("  On 16GB VRAM this needs device_map='auto' with max_memory set,")
    print("  or full CPU. Do NOT quantize: CETT measures activation magnitude.")

    out = {
        "model_type": cfg.model_type,
        "n_layers": n_layers,
        "intermediate_size": d_inter,
        "hidden_size": d_hidden,
        "n_features": n_feat,
        "down_proj_modules": matched,
    }
    with open("preflight.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote preflight.json")


if __name__ == "__main__":
    main()
