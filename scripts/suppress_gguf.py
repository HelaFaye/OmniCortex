#!/usr/bin/env python3
"""
Write a copy of a GGUF with the H-Neurons scaled down. Loads anywhere a GGUF
loads: llama-server, LM Studio, our studio.

How it works without re-quantizing. In a gated MLP, neuron j's activation is

    a_j = silu(gate_j . x) * (up_j . x)

which is linear in row j of ffn_up. Scaling that row by s scales a_j by s
exactly, the same intervention as scaling column j of ffn_down. A row of
ffn_up is stored as whole quantization blocks (n_embd is a multiple of the
block size), and every block carries its own fp16 scale `d` (plus `dmin` or
`m` for the offset formats). Multiplying those by s rescales the block's
values exactly. So the edit touches a few kilobytes, keeps the original quant
and file size, and adds no quantization error beyond fp16 rounding of `d`.

Supported: Q8_0 Q6_K Q5_K Q4_K Q3_K Q2_K Q4_0 Q5_0 Q4_1 Q5_1 F16 BF16 F32.
I-quants are refused rather than guessed at.

This makes the model more willing to decline, not more knowledgeable. Whether
it reduces wrong answers on this model is exactly what compare_models.py
measures. Do not assume it from the classifier's AUROC.

    python scripts/suppress_gguf.py --h_neurons models_1v1_fixed/h_neurons.json \\
        --scale 0.25 --out ~/models/my-model-supp25.Q6_K.gguf
"""
import argparse
import json
import os
import shutil

import numpy as np

# type name -> (byte offsets of fp16 multipliers within one block). Every
# dequantized value in these formats is (one of these) * (integer stuff), or
# a difference of two such terms, so scaling all of them scales the block.
_SCALE_FIELDS = {
    "Q8_0": [0],          # d, qs[32]
    "Q4_0": [0],          # d, qs[16]
    "Q5_0": [0],          # d, qh[4], qs[16]
    "Q4_1": [0, 2],       # d, m, qs[16]          x = d*q + m
    "Q5_1": [0, 2],       # d, m, qh[4], qs[16]
    "Q2_K": [80, 82],     # scales[16], qs[64], d, dmin
    "Q3_K": [108],        # hmask[32], qs[64], scales[12], d
    "Q4_K": [0, 2],       # d, dmin, scales[12], qs[128]
    "Q5_K": [0, 2],       # d, dmin, scales[12], qh[32], qs[128]
    "Q6_K": [208],        # ql[128], qh[64], scales[16], d
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default=os.environ.get("OC_GGUF"),
                   help="source model (default $OC_GGUF)")
    p.add_argument("--h_neurons", required=True)
    p.add_argument("--scale", type=float, required=True,
                   help="multiplier for H-Neuron activations: 1 = unchanged, "
                        "0 = silenced. Try 0.5 / 0.25 / 0 and compare")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true",
                   help="allow an h_neurons.json made from a different file")
    return p.parse_args()


def scale_row(row_bytes, ttype, s):
    """Scale one row in place. row_bytes is a writable uint8 view."""
    name = ttype.name
    if name == "F32":
        v = row_bytes.view(np.float32)
        v *= np.float32(s)
    elif name == "F16":
        v = row_bytes.view(np.float16)
        v[:] = (v.astype(np.float32) * s).astype(np.float16)
    elif name == "BF16":
        v = row_bytes.view(np.uint16)
        f = (v.astype(np.uint32) << 16).view(np.float32) * np.float32(s)
        v[:] = (f.view(np.uint32) >> 16).astype(np.uint16)
    elif name in _SCALE_FIELDS:
        import gguf
        _, bsize = gguf.GGML_QUANT_SIZES[ttype]
        blocks = row_bytes.reshape(-1, bsize)
        for off in _SCALE_FIELDS[name]:
            d = blocks[:, off:off + 2].copy().view(np.float16).reshape(-1)
            new = (d.astype(np.float32) * s).astype(np.float16)
            blocks[:, off:off + 2] = new.view(np.uint8).reshape(-1, 2)
    else:
        raise SystemExit(f"{name} is not supported for in-place scaling; "
                         "use a K-quant, Q8_0 or float GGUF")


def main():
    args = parse_args()
    try:
        import gguf
    except ImportError:
        raise SystemExit("needs the gguf package: pip install gguf")
    if not args.gguf or not os.path.isfile(args.gguf):
        raise SystemExit(f"--gguf not found: {args.gguf!r} (source env.sh?)")
    if not 0.0 <= args.scale <= 1.0:
        print(f"note: scale {args.scale} is outside [0, 1]; >1 amplifies")

    with open(args.h_neurons) as f:
        hn = json.load(f)
    if hn.get("n_experts", 1) != 1:
        raise SystemExit("MoE profiles are not supported by this tool yet")
    src_name = os.path.basename(hn.get("model_path", ""))
    if src_name and src_name != os.path.basename(args.gguf) and not args.force:
        raise SystemExit(
            f"{args.h_neurons} was found on {src_name}, not "
            f"{os.path.basename(args.gguf)}. Neuron indices belong to specific "
            "weights; pass --force only if these are the same model.")
    by_layer = {int(k): sorted(set(int(i) for i in v))
                for k, v in hn["by_layer"].items() if v}
    total = sum(len(v) for v in by_layer.values())
    if not total:
        raise SystemExit("no H-Neurons in that file")

    # Validate everything before the 7 GB copy.
    reader = gguf.GGUFReader(args.gguf)
    tensors = {t.name: t for t in reader.tensors}
    plan = []
    for layer, idxs in sorted(by_layer.items()):
        name = f"blk.{layer}.ffn_up.weight"
        t = tensors.get(name)
        if t is None:
            raise SystemExit(f"{name} not in the GGUF (gated MLP expected)")
        ne0, ne1 = int(t.shape[0]), int(t.shape[1])   # ne0 contiguous = n_embd
        n_ff = hn.get("n_neurons", ne1)
        if ne1 != n_ff:
            raise SystemExit(f"{name} has {ne1} rows, profile says {n_ff}")
        if max(idxs) >= ne1:
            raise SystemExit(f"neuron {max(idxs)} out of range in layer {layer}")
        bel, bsize = gguf.GGML_QUANT_SIZES[t.tensor_type]
        if ne0 % bel:
            raise SystemExit(f"{name}: row of {ne0} is not whole blocks")
        row_bytes = ne0 // bel * bsize
        if row_bytes * ne1 != int(t.n_bytes):
            raise SystemExit(f"{name}: unexpected size {t.n_bytes}")
        if t.tensor_type.name not in _SCALE_FIELDS and \
                t.tensor_type.name not in ("F32", "F16", "BF16"):
            raise SystemExit(f"{name} is {t.tensor_type.name}; not supported")
        plan.append((layer, name, int(t.data_offset), row_bytes,
                     t.tensor_type, idxs))
    types = sorted({p[4].name for p in plan})
    print(f"{total} H-Neurons in {len(plan)} layers, ffn_up as {', '.join(types)}")
    del reader, tensors

    out = os.path.expanduser(args.out)
    if os.path.abspath(out) == os.path.abspath(args.gguf):
        raise SystemExit("--out must differ from the source")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"copying {os.path.getsize(args.gguf) / 1e9:.1f} GB -> {out}")
    shutil.copyfile(args.gguf, out)

    mm = np.memmap(out, dtype=np.uint8, mode="r+")
    for layer, name, off, rb, ttype, idxs in plan:
        for j in idxs:
            scale_row(mm[off + j * rb: off + (j + 1) * rb], ttype, args.scale)
    mm.flush()
    del mm

    # Re-read and check one edited row against the source.
    import gguf.quants as q
    a = {t.name: t for t in gguf.GGUFReader(args.gguf).tensors}
    b = {t.name: t for t in gguf.GGUFReader(out).tensors}
    layer, name, _, rb, ttype, idxs = plan[0]
    ra = q.dequantize(np.asarray(a[name].data)[idxs[0]:idxs[0] + 1], ttype)
    rb_ = q.dequantize(np.asarray(b[name].data)[idxs[0]:idxs[0] + 1], ttype)
    want = ra.astype(np.float64) * args.scale
    err = np.abs(rb_ - want).max() / (np.abs(want).max() + 1e-12)
    untouched = next((k for k in range(int(a[name].shape[1])) if k not in idxs), 0)
    same = np.array_equal(np.asarray(a[name].data)[untouched],
                          np.asarray(b[name].data)[untouched])
    print(f"check: layer {layer} neuron {idxs[0]} scaled within {err:.1e} "
          f"relative; neighbour row unchanged: {same}")
    if err > 1e-1 or not same:
        raise SystemExit("verification failed; do not use this file")

    side = os.path.splitext(out)[0] + ".suppression.json"
    with open(side, "w") as f:
        json.dump({"source": os.path.abspath(args.gguf),
                   "h_neurons": os.path.abspath(args.h_neurons),
                   "scale": args.scale, "n_neurons": total,
                   "by_layer": {str(k): v for k, v in by_layer.items()},
                   "method": "ffn_up row scale via block multipliers"}, f,
                  indent=2)
    print(f"wrote {out}\n      {side}")


if __name__ == "__main__":
    main()
