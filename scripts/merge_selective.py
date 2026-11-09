#!/usr/bin/env python3
"""
Selectively merge weights between two models, at neuron granularity.

THE TRAP. A "neuron" in a gated MLP is not one weight vector. Neuron j is
defined jointly by three:

    gate_proj[j, :]   row j      how strongly it activates
    up_proj[j, :]     row j      what it reads
    down_proj[:, j]   column j   what it writes back

Splicing only `down_proj[:, j]` from model B while leaving B's gate and up
behind pairs A's activation pattern with B's output projection. The result is
not "neuron j from B" -- it is a neuron that exists in neither model, and it
will produce fluent output while meaning nothing. That failure is invisible
until you evaluate. This tool always moves the complete triple.

WHAT MERGES ARE VALID. The same tiering as viz/compare.py. Merging
requires shared lineage: two models post-trained from the same base still have
neuron correspondence, because post-training does not permute the intermediate
dimension. Two independently trained models do not, and merging them produces
noise regardless of matching shapes. Geometry equality is necessary and nowhere
near sufficient, so cross-model merges need --assert-lineage.

    # inspect first, always
    python scripts/merge_selective.py --base A --donor B \\
        --h-neurons models/h_neurons.json --dry-run

    # transplant the H-Neuron triples from the donor
    python scripts/merge_selective.py --base A --donor B \\
        --h-neurons models/h_neurons.json --assert-lineage \\
        --alpha 1.0 --output_path models/merged

    # blend a layer range instead
    python scripts/merge_selective.py --base A --donor B \\
        --layers 8:24 --alpha 0.5 --assert-lineage --output_path models/merged
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import TEXT_LAYER_RE, load_model, text_config  # noqa: E402
from profiles import validate_selection  # noqa: E402
from profiles import fingerprint  # noqa: E402

TRIPLE = ("gate_proj", "up_proj", "down_proj")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="model to merge into")
    p.add_argument("--donor", required=True, help="model to take weights from")
    p.add_argument("--h-neurons", help="restrict to these neurons")
    p.add_argument("--layers", help="range like 8:24, or 'all'")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="1.0 replaces with the donor, 0.5 blends evenly")
    p.add_argument("--assert-lineage", action="store_true",
                   help="Assert the two share a base so neuron indices "
                        "correspond. Required for different models.")
    p.add_argument("--output_path")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no_trust_remote_code", action="store_true")
    return p.parse_args()


def parse_layers(spec, n_layers):
    if not spec or spec == "all":
        return list(range(n_layers))
    m = re.fullmatch(r"(\d*):(\d*)", spec)
    if not m:
        raise SystemExit(f"bad --layers {spec!r}; use 8:24 or 'all'")
    a = int(m.group(1) or 0)
    b = int(m.group(2) or n_layers)
    if not (0 <= a < b <= n_layers):
        raise SystemExit(f"--layers {spec} outside 0:{n_layers}")
    return list(range(a, b))


def mlp_modules(model):
    """-> {layer: {name: module}} for text decoder MLPs only."""
    out = {}
    for name, mod in model.named_modules():
        m = TEXT_LAYER_RE.search(name)
        if not m or not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TRIPLE:
            out.setdefault(int(m.group(1)), {})[leaf] = mod
    return out


def check_lineage(base, donor, asserted):
    fb, gb = fingerprint(base.config)
    fd, gd = fingerprint(donor.config)
    if gb != gd:
        raise SystemExit(
            f"geometry differs, merge is impossible:\n  base  {gb}\n  donor {gd}")
    if fb == fd:
        return "IDENTICAL", "same architecture fingerprint"
    if not asserted:
        raise SystemExit(
            "same geometry but different fingerprints. Neuron indices only "
            "correspond if\nboth were post-trained from the same base. Pass "
            "--assert-lineage if that is true;\nif it is not, this merge "
            "produces noise that will still generate fluent text.")
    return "LINEAGE", "asserted by the user, unverified"


def main():
    args = parse_args()
    trc = not args.no_trust_remote_code

    from hostcheck import guard
    try:
        sz = sum(os.path.getsize(os.path.join(args.base, f))
                 for f in os.listdir(args.base)) if os.path.isdir(args.base) else 0
    except OSError:
        sz = 0
    if sz:
        guard("merge", model_bytes=sz, path=args.output_path or ".")

    print("loading base...")
    base, tokenizer = load_model(args.base, gpu_mem=None, trust_remote_code=trc)
    print("loading donor...")
    donor, _ = load_model(args.donor, gpu_mem=None, trust_remote_code=trc)

    tier, why = check_lineage(base, donor, args.assert_lineage)
    print(f"tier: {tier} -- {why}")

    tcfg = text_config(base.config)
    n_layers, n_ff = tcfg.num_hidden_layers, tcfg.intermediate_size

    base_mlp, donor_mlp = mlp_modules(base), mlp_modules(donor)
    missing = set(base_mlp) ^ set(donor_mlp)
    if missing:
        raise SystemExit(f"layer sets differ: {sorted(missing)[:8]}")

    layers = parse_layers(args.layers, n_layers)

    selection = None
    if args.h_neurons:
        with open(args.h_neurons) as f:
            h = json.load(f)
        if h.get("n_neurons") != n_ff or h.get("n_layers") != n_layers:
            raise SystemExit(
                f"h_neurons.json is {h.get('n_layers')}x{h.get('n_neurons')}, "
                f"models are {n_layers}x{n_ff}")
        validate_selection(
            h["by_layer"], n_layers, n_ff, label=os.path.basename(args.h_neurons)
        )
        selection = {int(k): list(v) for k, v in h["by_layer"].items()
                     if int(k) in layers}
        if not selection:
            raise SystemExit("no H-Neurons fall inside --layers")
    else:
        selection = {l: list(range(n_ff)) for l in layers}

    total = sum(len(v) for v in selection.values())
    frac = total / (n_layers * n_ff)
    print(f"\nselection: {total} neurons across {len(selection)} layers "
          f"({frac * 100:.3f}% of all)")
    print(f"alpha {args.alpha}: "
          f"{'full replacement' if args.alpha == 1.0 else 'blend'}")

    # Report divergence before touching anything. If the donor's selected
    # weights are nearly identical to the base's, the merge is a no-op and you
    # want to know that before spending an evaluation on it.
    print("\nper-layer divergence on the selected neurons (relative L2):")
    shown = 0
    for layer in sorted(selection):
        cols = torch.tensor(selection[layer], dtype=torch.long)
        b = base_mlp[layer]["down_proj"].weight.data[:, cols]
        d = donor_mlp[layer]["down_proj"].weight.data[:, cols]
        rel = (d - b).norm() / (b.norm() + 1e-8)
        if shown < 8:
            print(f"  L{layer:<3} n={len(cols):<6} {rel:.4f}")
            shown += 1
    if len(selection) > 8:
        print(f"  ... {len(selection) - 8} more layers")

    if args.dry_run or not args.output_path:
        print("\ndry run; pass --output_path to write the merged model")
        return

    changed = 0
    with torch.no_grad():
        for layer, neurons in selection.items():
            cols = torch.tensor(neurons, dtype=torch.long)
            for leaf in TRIPLE:
                b = base_mlp[layer][leaf].weight.data
                d = donor_mlp[layer][leaf].weight.data
                if b.shape != d.shape:
                    raise SystemExit(
                        f"L{layer} {leaf}: {tuple(b.shape)} vs {tuple(d.shape)}")
                # gate/up are [intermediate, hidden] -> neuron j is row j.
                # down is [hidden, intermediate]     -> neuron j is column j.
                if leaf == "down_proj":
                    b[:, cols] = (1 - args.alpha) * b[:, cols] + args.alpha * d[:, cols]
                else:
                    b[cols, :] = (1 - args.alpha) * b[cols, :] + args.alpha * d[cols, :]
            changed += len(neurons)

    print(f"\nmerged {changed} complete neuron triples "
          f"({changed * 3} weight vectors)")
    os.makedirs(args.output_path, exist_ok=True)
    base.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)
    with open(os.path.join(args.output_path, "merge_info.json"), "w") as f:
        json.dump({"base": args.base, "donor": args.donor, "tier": tier,
                   "alpha": args.alpha, "neurons_merged": changed,
                   "layers": sorted(selection),
                   "h_neurons": args.h_neurons,
                   "note": "gate_proj, up_proj and down_proj moved together; "
                           "a partial triple would create a neuron present in "
                           "neither source model"}, f, indent=2)
    print(f"saved to {args.output_path}")
    print("\nEvaluate before trusting this. A merge across an asserted lineage "
          "that was\nactually wrong still produces fluent text; only the "
          "benchmarks will show it.")


if __name__ == "__main__":
    main()
