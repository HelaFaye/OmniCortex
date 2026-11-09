#!/usr/bin/env python3
"""
Export a suppression profile as a LoRA adapter.

Why this works. Scaling the H-Neuron columns of down_proj by s is

    W' = W @ diag(s)   =>   dW = W @ (diag(s) - I)

and diag(s) - I is zero everywhere except the k selected columns, so

    dW = W[:, S] @ diag(s - 1) @ E_S^T

which is rank k, with k the number of H-Neurons in that layer -- typically a
few dozen. So the intervention is exactly a small LoRA:

    lora_B = W[:, S] * (s - 1)      [hidden, k]
    lora_A = E_S^T                  [k, intermediate]   one-hot row selectors

Two consequences worth the trouble:

  * llama.cpp loads LoRA adapters and can rescale them on a running server via
    --lora-scaled and the /lora-adapters endpoint, so suppression strength
    becomes a runtime dial instead of a rebuild.
  * Because dW is linear in (s - 1), the adapter's runtime scale a gives an
    effective neuron scale of 1 + a*(s - 1). a=0 is the untouched model, a=1 is
    the profile as tuned, and anything between interpolates.

On precision. llama.cpp applies the delta against the *quantized* base, so with
B built from bf16 weights you get Q(W)[:, S] + W_bf16[:, S]*(s-1), not
s*Q(W)[:, S]: the base's quantization error rides along unscaled. Pass --gguf
pointing at the exact file you will serve and B is built from Q(W) instead,
giving Q(W)[:, S] + Q(W)[:, S]*(s-1) = s*Q(W)[:, S] exactly. If you are serving
a fixed quant, always use --gguf.

    python scripts/export_lora.py \
        --model_path Qwen/Qwen3-8B \
        --profile profiles/<fp>/trivia-q8.json \
        --output_dir adapters/model-suppress

    python llama.cpp/convert_lora_to_gguf.py adapters/model-suppress \
        --base Qwen/Qwen3-8B --outfile model-suppress-lora.gguf
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import TEXT_LAYER_RE, load_model, text_config  # noqa: E402
from profiles import Profile  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scale", type=float,
                   help="Override the profile's scale for this export")
    p.add_argument("--gguf",
                   help="Build the adapter from this GGUF's dequantized "
                        "ffn_down tensors instead of the bf16 safetensors. Use "
                        "the exact file you will serve: it makes the adapter "
                        "mathematically exact rather than approximate.")
    p.add_argument("--no_trust_remote_code", action="store_true")
    return p.parse_args()



GGUF_FFN_DOWN = "blk.{layer}.ffn_down.weight"


def load_gguf_down_proj(path, n_layers, hidden, inter):
    """Dequantized ffn_down tensors, keyed by layer index.

    Only the down_proj weights are read, so this costs a fraction of a full
    model load. GGUF stores dimensions in reverse of numpy's convention and
    exporters vary, so the orientation is checked rather than assumed.
    """
    try:
        import gguf
    except ImportError:
        raise SystemExit("--gguf needs the gguf package: pip install gguf")

    reader = gguf.GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}
    out = {}
    for layer in range(n_layers):
        name = GGUF_FFN_DOWN.format(layer=layer)
        t = by_name.get(name)
        if t is None:
            raise SystemExit(
                f"{name} not found in {path}. Tensor names present look like: "
                + ", ".join(sorted(by_name)[:5])
            )
        arr = gguf.quants.dequantize(t.data, t.tensor_type).astype("float32")
        arr = arr.reshape(-1)
        if arr.size != hidden * inter:
            raise SystemExit(
                f"{name} has {arr.size} elements, expected {hidden * inter}"
            )
        # Try [hidden, inter]; fall back to the transpose.
        cand = arr.reshape(hidden, inter)
        if hidden != inter:
            try:
                shp = tuple(int(x) for x in t.shape)
            except Exception:
                shp = ()
            if shp and shp[0] == hidden and shp[-1] == inter:
                cand = arr.reshape(inter, hidden).T
        out[layer] = torch.from_numpy(cand.copy())
    print(f"read {len(out)} dequantized ffn_down tensors from {os.path.basename(path)}")
    return out


def main():
    args = parse_args()
    profile = Profile.load(args.profile)
    scale = args.scale if args.scale is not None else profile.data["scale"]
    if scale == 1.0:
        raise SystemExit("scale is 1.0; the adapter would be all zeros")

    model, tokenizer = load_model(
        args.model_path, gpu_mem=None,
        trust_remote_code=not args.no_trust_remote_code,
    )
    profile.check(model)
    by_layer = profile.by_layer

    gguf_w = None
    if args.gguf:
        t = text_config(model.config)
        gguf_w = load_gguf_down_proj(
            args.gguf, t.num_hidden_layers, t.hidden_size, t.intermediate_size
        )

    # PEFT's adapter_config carries a single rank. Pad every layer to the
    # largest so the adapter has a uniform r, which every converter handles.
    r = max(len(v) for v in by_layer.values())
    print(f"scale {scale}, uniform rank {r} "
          f"(max {r} neurons in one layer, padded with zeros)")

    tensors = {}
    prefix = "base_model.model"
    matched = 0

    for name, module in model.named_modules():
        if "down_proj" not in name or not isinstance(module, torch.nn.Linear):
            continue
        m = TEXT_LAYER_RE.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        if layer not in by_layer:
            continue
        cols = by_layer[layer]
        k = len(cols)
        W = (gguf_w[layer] if gguf_w is not None
             else module.weight.data).float()   # [hidden, intermediate]
        out_dim, in_dim = W.shape
        if gguf_w is not None and W.shape != module.weight.shape:
            raise SystemExit(
                f"layer {layer}: GGUF tensor is {tuple(W.shape)} but the model "
                f"has {tuple(module.weight.shape)}"
            )

        # A selects the H-Neuron inputs; B carries the scaled columns.
        A = torch.zeros(r, in_dim, dtype=torch.float32)
        for i, c in enumerate(cols):
            A[i, c] = 1.0
        B = torch.zeros(out_dim, r, dtype=torch.float32)
        B[:, :k] = W[:, cols] * (scale - 1.0)

        key = f"{prefix}.{name}"
        tensors[f"{key}.lora_A.weight"] = A.to(torch.float16)
        tensors[f"{key}.lora_B.weight"] = B.to(torch.float16)
        matched += 1

    if matched != len(by_layer):
        raise SystemExit(f"matched {matched} modules, profile has {len(by_layer)}")

    # Verify the factorisation on one layer before writing anything.
    probe_layer = next(iter(by_layer))
    for name, module in model.named_modules():
        if not (TEXT_LAYER_RE.search(name) and "down_proj" in name):
            continue
        if int(TEXT_LAYER_RE.search(name).group(1)) != probe_layer:
            continue
        W = (gguf_w[probe_layer] if gguf_w is not None
             else module.weight.data).float()
        A = tensors[f"{prefix}.{name}.lora_A.weight"].float()
        B = tensors[f"{prefix}.{name}.lora_B.weight"].float()
        want = W.clone()
        want[:, by_layer[probe_layer]] *= scale
        got = W + B @ A
        err = (want - got).abs().max().item()
        scaleref = W.abs().max().item()
        print(f"factorisation check on layer {probe_layer}: "
              f"max abs error {err:.3e} (weights up to {scaleref:.3f})")
        if err > 1e-2 * max(scaleref, 1.0):
            raise SystemExit("factorisation does not reproduce the edit")
        break

    os.makedirs(args.output_dir, exist_ok=True)
    save_file(tensors, os.path.join(args.output_dir, "adapter_model.safetensors"))

    with open(os.path.join(args.output_dir, "adapter_config.json"), "w") as f:
        json.dump({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": args.model_path,
            "r": r,
            "lora_alpha": r,          # alpha/r == 1, so no implicit rescale
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": False,
            "target_modules": ["down_proj"],
            "inference_mode": True,
        }, f, indent=2)

    with open(os.path.join(args.output_dir, "h_neuron_profile.json"), "w") as f:
        json.dump(profile.data, f, indent=2)

    size = sum(t.numel() * 2 for t in tensors.values()) / 1e6
    print(f"wrote {args.output_dir} ({size:.1f} MB, {matched} layers)")
    print(f"""
Convert and serve:

  python llama.cpp/convert_lora_to_gguf.py {args.output_dir} \\
      --base {args.model_path} --outfile suppress-lora.gguf

  ./llama.cpp/build/bin/llama-server -m base-Q8_0.gguf \\
      --lora-scaled suppress-lora.gguf 1.0

Effective neuron scale is 1 + a*({scale} - 1) for adapter scale a, so a=0
disables suppression entirely. Change it live:

  curl -X POST http://127.0.0.1:8080/lora-adapters \\
      -H 'Content-Type: application/json' -d '[{{"id":0,"scale":0.5}}]'

Vulkan carries LoRA the same as any backend (it is applied in the ggml graph,
not in backend code), but adapters have had backend-specific bugs, so run a
short generation with adapter scale 0 and confirm it matches the plain base.

LM Studio's LoRA support is less clearly documented than llama.cpp's. If it
will not load the adapter, merge it into a standalone GGUF instead:

  ./llama.cpp/build/bin/llama-export-lora -m base-Q8_0.gguf \\
      --lora suppress-lora.gguf -o base-suppressed-Q8_0.gguf
""")


if __name__ == "__main__":
    main()
