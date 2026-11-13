#!/usr/bin/env python3
"""
OmniCortex explorer: 2D and 3D activation maps over a recorded session.

    python viz/explore.py runs/model-q6
    python viz/explore.py runs/model-q6 --h-neurons models/h_neurons.json
    python viz/explore.py runs/model-q6 --split-by verdict

Four panels:

  mean map        layers x neurons, averaged over samples
  contrast        mean(hallucinated) - mean(correct), if verdicts were recorded.
                  This is the panel worth looking at: the H-Neurons should show
                  up here as bright columns, and if they do not, the classifier
                  found something the eye cannot corroborate.
  volume          samples x layers x neurons, MIP projection, neuron axis binned
  profile         per-layer totals, with H-Neuron counts overlaid if given

Click any 2D panel to print the layer, neuron and value under the cursor.

All array preparation lives in module-level functions with no fastplotlib
import, so it is testable headless and reusable by any other frontend.

Memory. A 9B session at 32x14336 is 1.8MB per sample in float32; 500 samples is
~900MB before the volume is built. --max-samples caps the load and --bins
reduces the neuron axis for the 3D view only -- 2D panels stay full resolution.
On a 12GB shared-memory iGPU the defaults are chosen to fit.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import Session  # noqa: E402


# ---------------------------------------------------------------- data prep

def robust_limits(a, lo=1.0, hi=99.5):
    """Percentile limits. CETT is heavily right-skewed, so min/max scaling
    puts everything in the bottom of the colormap and shows nothing."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def bin_neurons(stack, bins):
    """Reduce the neuron axis by max-pooling into `bins` columns.

    Max rather than mean: H-Neurons are sparse (<0.1% of neurons), and mean
    pooling over 14336 columns would average a strong signal into nothing.
    """
    *lead, n = stack.shape
    if bins is None or bins >= n:
        return stack
    edges = np.linspace(0, n, bins + 1).astype(int)
    out = np.empty((*lead, bins), dtype=stack.dtype)
    for i in range(bins):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        out[..., i] = stack[..., a:b].max(axis=-1)
    return out


def group_by_field(session, ids, field):
    """-> {value: [ids]} from the per-sample fields, skipping unlabelled."""
    groups = {}
    for qid in ids:
        try:
            d = session.get(qid)
        except Exception:
            continue
        v = (d.get("fields") or {}).get(field)
        if v is not None:
            groups.setdefault(str(v), []).append(qid)
    return groups


def contrast_map(stack, ids, groups, positive, negative):
    """mean(positive) - mean(negative), or None if either group is empty."""
    idx = {q: i for i, q in enumerate(ids)}
    pi = [idx[q] for q in groups.get(positive, []) if q in idx]
    ni = [idx[q] for q in groups.get(negative, []) if q in idx]
    if not pi or not ni:
        return None, len(pi), len(ni)
    return stack[pi].mean(0) - stack[ni].mean(0), len(pi), len(ni)


def load_h_neurons(path, n_layers, n_neurons):
    """-> boolean mask [n_layers, n_neurons], or None."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        h = json.load(f)
    if h.get("n_layers") != n_layers or h.get("n_neurons") != n_neurons:
        raise SystemExit(
            f"h_neurons.json is {h.get('n_layers')}x{h.get('n_neurons')} but "
            f"this session is {n_layers}x{n_neurons}. Different model or quant."
        )
    mask = np.zeros((n_layers, n_neurons), dtype=bool)
    for k, v in h["by_layer"].items():
        mask[int(k), np.asarray(v, dtype=int)] = True
    return mask


def layer_profile(mean_map, mask=None):
    """Per-layer mean activation, and H-Neuron counts if a mask is given."""
    prof = mean_map.mean(axis=1)
    counts = mask.sum(axis=1) if mask is not None else None
    return prof, counts


# -------------------------------------------------------------------- render

def check_gpu():
    import fastplotlib as fpl
    adapters = fpl.enumerate_adapters()
    if not adapters:
        raise SystemExit(
            "WGPU found no adapters, so nothing can render.\n"
            "  On a Vega iGPU install the Vulkan driver (vulkan-radeon on Arch)\n"
            "  and confirm with: vulkaninfo --summary\n"
            "  Headless or over SSH, install a software renderer (lavapipe).")
    return adapters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--h-neurons", help="models/h_neurons.json, to overlay")
    p.add_argument("--split-by", default="verdict",
                   help="per-sample field to contrast on")
    p.add_argument("--positive", default="wrong")
    p.add_argument("--negative", default="correct")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--bins", type=int, default=512,
                   help="neuron bins for the 3D volume only")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--dump", metavar="NPZ",
                   help="Write the prepared arrays and exit. No GPU needed.")
    a = p.parse_args()

    sess = Session(a.session)
    stack, ids = sess.stack(max_samples=a.max_samples)
    if stack.size == 0:
        raise SystemExit(f"no samples with an 'agg' array in {a.session}")
    n_samples, n_layers, n_neurons = stack.shape
    print(f"{n_samples} samples, {n_layers} layers x {n_neurons} neurons "
          f"({stack.nbytes / 1e6:.0f} MB)")
    print(f"model: {sess.meta.get('model')}  quant: {sess.meta.get('quant')}")

    mean_map = stack.mean(0)
    mask = load_h_neurons(a.h_neurons, n_layers, n_neurons)
    groups = group_by_field(sess, ids, a.split_by)
    contrast, n_pos, n_neg = contrast_map(stack, ids, groups,
                                          a.positive, a.negative)
    if contrast is None:
        print(f"no contrast panel: field '{a.split_by}' gave "
              f"{n_pos} '{a.positive}' and {n_neg} '{a.negative}'")
    else:
        print(f"contrast: {n_pos} {a.positive} vs {n_neg} {a.negative}")
    volume = bin_neurons(stack, a.bins)
    prof, counts = layer_profile(mean_map, mask)

    if a.dump:
        np.savez_compressed(
            a.dump, mean_map=mean_map, volume=volume, profile=prof,
            **({"contrast": contrast} if contrast is not None else {}),
            **({"h_mask": mask, "h_counts": counts} if mask is not None else {}))
        print(f"wrote {a.dump}")
        return

    check_gpu()
    import fastplotlib as fpl

    titles = ["mean", "contrast" if contrast is not None else "mean (log)",
              "volume", "per-layer"]
    fig = fpl.Figure(shape=(2, 2), size=(1100, 780),
                     cameras=[["2d", "2d"], ["3d", "2d"]], names=titles)

    vmin, vmax = robust_limits(mean_map)
    fig[0, 0].add_image(mean_map, cmap=a.cmap, vmin=vmin, vmax=vmax,
                        name="mean")

    if contrast is not None:
        lim = float(np.percentile(np.abs(contrast), 99.5)) or 1.0
        fig[0, 1].add_image(contrast, cmap="bwr", vmin=-lim, vmax=lim,
                            name="contrast")
    else:
        logm = np.log1p(np.clip(mean_map, 0, None))
        lv = robust_limits(logm)
        fig[0, 1].add_image(logm, cmap=a.cmap, vmin=lv[0], vmax=lv[1],
                            name="mean_log")

    vv = robust_limits(volume)
    fig[1, 0].add_image_volume(volume, mode="mip", cmap=a.cmap,
                               vmin=vv[0], vmax=vv[1], name="volume")

    xs = np.arange(n_layers, dtype=np.float32)
    fig[1, 1].add_line(np.column_stack([xs, prof / (prof.max() or 1)]),
                       colors="cyan", thickness=2.0, name="mean per layer")
    if counts is not None:
        fig[1, 1].add_line(
            np.column_stack([xs, counts / (counts.max() or 1)]),
            colors="magenta", thickness=2.0, name="H-Neurons per layer")
        print("per-layer H-Neuron counts:",
              ", ".join(f"L{i}={int(c)}" for i, c in enumerate(counts) if c))

    # Click-to-inspect. Image data is [row, col] = [layer, neuron]; the world
    # coordinates come back as x=column, y=row.
    def inspector(panel_name, arr):
        def handler(ev):
            pos = getattr(ev, "pick_info", {}).get("index")
            if pos is None:
                return
            try:
                col, row = int(pos[0]), int(pos[1])
            except (TypeError, ValueError, IndexError):
                return
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                flag = ""
                if mask is not None and mask[row, col]:
                    flag = "  <- H-Neuron"
                print(f"{panel_name}: layer {row}, neuron {col} = "
                      f"{arr[row, col]:.5f}{flag}")
        return handler

    fig[0, 0]["mean"].add_event_handler(inspector("mean", mean_map),
                                        "pointer_down")
    if contrast is not None:
        fig[0, 1]["contrast"].add_event_handler(
            inspector("contrast", contrast), "pointer_down")

    print("\nclick a 2D panel to inspect a cell; drag to pan, scroll to zoom")
    fig.show()
    if __name__ == "__main__":
        fpl.loop.run()


if __name__ == "__main__":
    main()
