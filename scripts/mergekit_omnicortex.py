#!/usr/bin/env python3
"""
OmniCortex as a mergekit merge method.

mergekit already does general model merging well -- TIES, DARE, SLERP, task
arithmetic, model stock, SCE. Reimplementing any of that would be a waste. What
OmniCortex contributes is a different *selection criterion*.

TIES and DARE decide which parameters to keep by magnitude: large deltas
survive, small ones are pruned or randomly dropped. That is a statistical
heuristic with no reference to what the parameters do. OmniCortex selects by a
classifier trained on measured behaviour -- these specific neurons carry the
signal that separated hallucinated answers from grounded ones. Semantic
selection instead of magnitude selection.

So this registers `omnicortex_select`, which merges the donor's contribution only
at neurons the classifier identified, and leaves everything else at the base.

    pip install -e /path/to/mergekit
    export OMNICORTEX_NEURONS=/path/to/models/h_neurons.json
    mergekit-yaml merge.yaml ./out --allow-crimes

    # merge.yaml
    merge_method: omnicortex_select
    base_model: Qwen/Qwen3-8B
    models:
      - model: Qwen/Qwen3-8B
        parameters:
          weight: 1.0
    parameters:
      invert: false        # true merges everything EXCEPT the H-Neurons
    dtype: bfloat16

A neuron is three weight vectors -- gate_proj row, up_proj row, down_proj
column -- and this handles all three, because mergekit calls the method once
per tensor and the mask is applied on the correct axis for each. See
scripts/merge_selective.py for why a partial triple is silently wrong.

LINEAGE STILL APPLIES. mergekit will happily merge any two models with matching
shapes. Neuron indices only correspond when both were post-trained from the
same base; otherwise this selects meaningless coordinates. mergekit cannot
check that and neither can this plugin -- it is on you.
"""

import json
import os
import re
from typing import List

import torch

try:
    from mergekit.architecture import WeightInfo
    from mergekit.merge_methods.easy_define import merge_method
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "mergekit is not importable. Install it first:\n"
        "  pip install -e /path/to/mergekit"
    ) from e

# Matches the decoder MLP tensors, capturing the layer index and which member
# of the triple this is.
_MLP_RE = re.compile(
    r"(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)

_CACHE = {}


def _load_masks(path):
    """-> ({layer: LongTensor of neuron ids}, n_layers, n_neurons), cached."""
    if path in _CACHE:
        return _CACHE[path]
    with open(path) as f:
        h = json.load(f)
    n_layers, n_neurons = int(h["n_layers"]), int(h["n_neurons"])
    if n_layers <= 0 or n_neurons <= 0:
        raise SystemExit(f"{os.path.basename(path)} has invalid geometry "
                         f"{n_layers}x{n_neurons}")
    masks = {}
    for raw_layer, raw_cols in h["by_layer"].items():
        layer = int(raw_layer)
        if layer < 0 or layer >= n_layers:
            raise SystemExit(
                f"{os.path.basename(path)}: layer {layer} outside 0..{n_layers - 1}")
        seen = set()
        cols = []
        for raw_col in raw_cols:
            col = int(raw_col)
            if col < 0 or col >= n_neurons:
                raise SystemExit(
                    f"{os.path.basename(path)}: layer {layer} references neuron "
                    f"{col}, valid range is 0..{n_neurons - 1}")
            if col in seen:
                raise SystemExit(
                    f"{os.path.basename(path)}: layer {layer} contains duplicate "
                    f"neuron {col}")
            seen.add(col)
            cols.append(col)
        if cols:
            masks[layer] = torch.tensor(sorted(cols), dtype=torch.long)
    _CACHE[path] = (masks, n_layers, n_neurons)
    return _CACHE[path]


def _spec_path():
    p = os.environ.get("OMNICORTEX_NEURONS")
    if not p:
        raise SystemExit(
            "set OMNICORTEX_NEURONS to your h_neurons.json:\n"
            "  export OMNICORTEX_NEURONS=models/h_neurons.json")
    if not os.path.exists(p):
        raise SystemExit(f"OMNICORTEX_NEURONS points at a missing file: {p}")
    return p


@merge_method(
    name="omnicortex_select",
    pretty_name="OmniCortex Neuron Selection",
    reference_url="https://arxiv.org/abs/2512.01797",
)
def omnicortex_select_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    weight: List[float],
    invert: bool = False,
) -> torch.Tensor:
    """Blend donors into the base only at classifier-selected neurons.

    Non-MLP tensors (attention, norms, embeddings) pass through as the base
    unchanged: the H-Neuron index describes the MLP intermediate dimension and
    means nothing anywhere else. Silently blending them would defeat the point
    of a targeted merge.
    """
    if not tensors:
        return base_tensor

    m = _MLP_RE.search(output_weight.name)
    if m is None:
        return base_tensor

    layer, leaf = int(m.group(1)), m.group(2)
    masks, n_layers, n_neurons = _load_masks(_spec_path())

    # The intermediate dimension differs per tensor: gate/up are
    # [intermediate, hidden] so neurons are rows; down is [hidden,
    # intermediate] so neurons are columns.
    axis = 1 if leaf == "down_proj" else 0
    if base_tensor.shape[axis] != n_neurons:
        raise SystemExit(
            f"{output_weight.name} has {base_tensor.shape[axis]} on the "
            f"intermediate axis but h_neurons.json says {n_neurons}. "
            "Wrong model for this neuron set.")
    if layer >= n_layers:
        raise SystemExit(
            f"{output_weight.name} is layer {layer} but h_neurons.json covers "
            f"{n_layers} layers.")

    idx = masks.get(layer)
    if invert:
        keep = torch.ones(n_neurons, dtype=torch.bool)
        if idx is not None:
            keep[idx] = False
        idx = keep.nonzero(as_tuple=True)[0]
    if idx is None or idx.numel() == 0:
        return base_tensor

    ws = list(weight) if weight else [1.0] * len(tensors)
    if len(ws) != len(tensors):
        raise SystemExit(f"{len(ws)} weights for {len(tensors)} models")
    total = sum(ws)
    if total <= 0:
        return base_tensor

    out = base_tensor.clone()
    idx = idx.to(out.device)

    # Weighted blend of the donors at the selected neurons, with the base
    # holding whatever weight is left over.
    donor = torch.zeros_like(
        out.index_select(axis, idx), dtype=torch.float32)
    for t, w in zip(tensors, ws):
        if t.shape != base_tensor.shape:
            raise SystemExit(
                f"{output_weight.name}: donor is {tuple(t.shape)}, base is "
                f"{tuple(base_tensor.shape)}")
        donor += t.index_select(axis, idx).to(torch.float32) * w

    base_slice = out.index_select(axis, idx).to(torch.float32)
    keep_base = max(0.0, 1.0 - total)
    blended = (donor + base_slice * keep_base).to(out.dtype)

    if axis == 0:
        out[idx, :] = blended
    else:
        out[:, idx] = blended
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Check an h_neurons.json and print a mergekit config")
    p.add_argument("--h-neurons", default=os.environ.get("OMNICORTEX_NEURONS"))
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--donor", default="Qwen/Qwen3-8B")
    a = p.parse_args()
    if not a.h_neurons:
        raise SystemExit("pass --h-neurons or set OMNICORTEX_NEURONS")

    masks, nl, nn = _load_masks(a.h_neurons)
    total = sum(int(v.numel()) for v in masks.values())
    print(f"{total} neurons across {len(masks)} of {nl} layers "
          f"({total / (nl * nn) * 100:.3f}% of {nl * nn})")
    print(f"\n# {total * 3} weight vectors will be touched "
          f"(gate/up rows + down columns)")
    print(f"""
merge_method: omnicortex_select
base_model: {a.base}
models:
  - model: {a.donor}
    parameters:
      weight: 1.0
parameters:
  invert: false
dtype: bfloat16
""")
    print("Lineage is not checked by mergekit or by this plugin. Neuron indices")
    print("correspond only if both models were post-trained from the same base.")
