#!/usr/bin/env python3
"""
Suppress H-Neurons by scaling down_proj input columns, then save the edited
model so it can be converted to GGUF and served from LM Studio.

The edit is static, which is the whole reason this reaches LM Studio at all:
llama.cpp control vectors can only add to the residual stream, they cannot
scale individual MLP neurons. Baking the scale into the weights sidesteps that.

Read this before running it. Suppressing H-Neurons does not make the model more
correct. The paper links these neurons to over-compliance, so the effect is that
the model becomes more willing to decline. You are trading hallucinations for
abstentions. On short factual QA that is usually a good trade; on a coding
workload it may simply make the model less useful. Always run --eval_prompts
before and after, and check that the model can still do its actual job.

    python scripts/intervene_model.py \
        --model_path Qwen/Qwen3-8B \
        --h_neurons models/h_neurons.json \
        --scale 0.1 \
        --output_path models/model-suppressed

Then:
    python llama.cpp/convert_hf_to_gguf.py models/model-suppressed \
        --outfile model-suppressed-f16.gguf --outtype f16
    ./llama.cpp/build/bin/llama-quantize \
        model-suppressed-f16.gguf model-suppressed-Q8_0.gguf Q8_0
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import TEXT_LAYER_RE, load_model  # noqa: E402
from profiles import validate_selection  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--h_neurons", help="models/h_neurons.json")
    p.add_argument("--profile", help="A tuned profile from tune_scale.py. Supplies both the neuron set and the scale.")
    p.add_argument("--scale", type=float, default=None,
                   help="0.0 ablates entirely. 0.1-0.5 is the usual sweep.")
    p.add_argument("--output_path", help="Where to save. Omit for a dry run.")
    p.add_argument("--max_layer_fraction", type=float, default=0.05,
                   help="Abort if any single layer has more than this fraction "
                        "of its neurons selected. Guards against a bad C value "
                        "wrecking general capability.")
    try:
        from oc_cli import add_trust_remote_code_argument
    except ImportError:
        from scripts.oc_cli import add_trust_remote_code_argument
    add_trust_remote_code_argument(p)
    return p.parse_args()


def main():
    args = parse_args()

    if not (args.h_neurons or args.profile):
        raise SystemExit("need --h_neurons or --profile")
    src = args.profile or args.h_neurons
    with open(src) as f:
        spec = json.load(f)
    by_layer = {int(k): v for k, v in spec["by_layer"].items()}
    n_neurons = spec["n_neurons"]
    if args.scale is None:
        if args.profile:
            args.scale = spec["scale"]
            print(f"using tuned scale {args.scale} from the profile")
        else:
            args.scale = 0.1
            print("no --scale given, defaulting to 0.1; "
                  "tune_scale.py picks this properly")

    by_layer = validate_selection(
        by_layer, spec["n_layers"], n_neurons, label=os.path.basename(src)
    )
    total = sum(len(v) for v in by_layer.values())
    if total == 0:
        raise SystemExit(f"{src}: contains no selected neurons")
    print(f"{total} H-Neurons across {len(by_layer)} layers "
          f"({total / (spec['n_layers'] * n_neurons) * 100:.4f}% of all neurons)")

    worst = max(((k, len(v)) for k, v in by_layer.items()), key=lambda kv: kv[1])
    frac = worst[1] / n_neurons
    print(f"densest layer: L{worst[0]} with {worst[1]} ({frac * 100:.2f}%)")
    if frac > args.max_layer_fraction:
        raise SystemExit(
            f"Layer {worst[0]} has {frac * 100:.2f}% of its neurons selected, "
            f"above --max_layer_fraction {args.max_layer_fraction * 100:.2f}%.\n"
            "Retrain the classifier with a lower --C. Scaling this many neurons "
            "will damage general capability, not just hallucination."
        )

    if not args.output_path:
        print("\nDry run. Pass --output_path to write the edited model.")
        return

    # Load on CPU: this is a weight edit, no forward passes needed, and it
    # avoids any offload bookkeeping when saving.
    model, tokenizer = load_model(
        args.model_path, gpu_mem=None,
        trust_remote_code=args.trust_remote_code,
    )

    touched = 0
    seen_layers = set()
    for name, module in model.named_modules():
        if "down_proj" not in name or not isinstance(module, torch.nn.Linear):
            continue
        m = TEXT_LAYER_RE.search(name)
        if not m:
            continue  # never touch the vision tower
        layer = int(m.group(1))
        if layer not in by_layer:
            continue
        cols = by_layer[layer]
        if module.in_features != n_neurons:
            raise SystemExit(
                f"{name}.in_features={module.in_features} but h_neurons.json "
                f"says {n_neurons}. Mismatched model."
            )
        with torch.no_grad():
            module.weight.data[:, cols] *= args.scale
        touched += len(cols)
        seen_layers.add(layer)

    missing = set(by_layer) - seen_layers
    if missing:
        raise SystemExit(f"layers in h_neurons.json never matched a module: {missing}")
    if touched != total:
        raise SystemExit(f"expected to scale {total} columns, scaled {touched}")

    print(f"scaled {touched} down_proj columns by {args.scale}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    with open(os.path.join(args.output_path, "h_neuron_edit.json"), "w") as f:
        json.dump({
            "source_model": args.model_path,
            "scale": args.scale,
            "neurons_scaled": touched,
            "source": os.path.abspath(src),
        }, f, indent=2)

    print(f"saved to {args.output_path}")
    print("\nNext: convert to GGUF at f16, then quantize. Verify the edit "
          "survived quantization by comparing abstention rates before and after.")


if __name__ == "__main__":
    main()
