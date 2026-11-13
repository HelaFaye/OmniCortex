#!/usr/bin/env python3
"""
Per-loop lens for looped (latent-recurrent) models such as NVIDIA's Dory (docs/DORY.md), on Hugging Face weights.

A looped model applies one block of layers several times per token. Every pass through that block is a natural
checkpoint, so this asks, for a prompt and an optional expected answer:

    after loop k, what would the model say?   the state after each pass, read through the final norm and the
                                              output head (a logit lens with loops in place of layers): top tokens,
                                              the expected answer's probability and rank, entropy
    which loop decides?                       the first loop whose top token is the final top token and stays there
    where does the work happen?               each recurrent layer's MLP activity per loop (mean |input to the down
                                              projection|), and with --h-neurons the H-neurons' share of it, per loop

For a confident mistake this shows whether the wrong answer is there from the first loop (more loops will not fix
it) or forms in a later one (fewer loops might avoid it), and which loop's H-neurons light up.

    python scripts/loop_lens.py --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --loops 4 \\
        --prompt "The capital of Australia is" --target " Canberra" --out runs/dory-lens.json

The recurrent block is found by watching which layers run more than once in a forward pass, so this works for any
looped model whose loop count is a config value (--loops sets it: n_recurrent_loops, num_loops or
num_recurrent_loops, whichever the config has).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

LOOP_KEYS = ("n_recurrent_loops", "num_loops", "num_recurrent_loops", "n_loops")
NORMS = ("model.norm", "model.final_norm", "model.final_layernorm", "transformer.ln_f", "model.decoder.norm", "norm")


def _get(obj, path):
    for p in path.split("."):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def load(model_id: str, loops: int | None, device: str = "auto", dtype: str = "bfloat16", trust: bool = False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    key = next((k for k in LOOP_KEYS if hasattr(cfg, k)), None)
    if loops is not None:
        if key is None:
            raise SystemExit(f"{model_id}'s config has no loop count ({', '.join(LOOP_KEYS)}): is it a looped model?")
        setattr(cfg, key, int(loops))
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    m = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, torch_dtype=dt, trust_remote_code=trust,
                                             device_map=device if device != "cpu" else None)
    if device == "cpu":
        m = m.to("cpu")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    return m.eval(), tok, (getattr(cfg, key) if key else None)


def layer_lists(model):
    """(name, ModuleList) pairs holding transformer layers."""
    import torch
    return [(n, mod) for n, mod in model.named_modules() if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0
            and n.split(".")[-1] in ("layers", "h", "blocks", "recurrent_layers", "input_layers", "output_layers",
                                     "prelude", "core", "coda", "recurrent", "loop_layers")]


def down_proj(layer):
    for path in ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "feed_forward.w2", "mlp.dense_4h_to_h", "mlp.out_proj"):
        mod = _get(layer, path)
        if mod is not None:
            return mod
    return None


def readout(model):
    head = model.get_output_embeddings()
    norm = next((m for p in NORMS if (m := _get(model, p)) is not None and not hasattr(m, "__len__")), None)

    def dtype_of(m, default):
        return next((p.dtype for p in m.parameters()), default)

    def f(h):
        x = h if norm is None else norm(h.to(dtype_of(norm, h.dtype)))
        return head(x.to(head.weight.dtype)).float()
    return f


def lens(model, tok, prompt: str, target: str | None = None, top_k: int = 5, h_neurons: dict | None = None) -> dict:
    import torch
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
    calls = defaultdict(list)           # layer name -> [output hidden state at the last position, per call]
    acts = defaultdict(list)            # layer name -> [mean |down_proj input| over positions, per call]
    hn = defaultdict(list)              # layer name -> [H-neurons' share of the activity, per call]
    hooks = []
    named = {}
    for lname, ml in layer_lists(model):
        for i, layer in enumerate(ml):
            name = f"{lname}.{i}"
            named[name] = layer

            def out_hook(_m, _inp, out, name=name):
                t = out[0] if isinstance(out, tuple) else out
                calls[name].append(t[0, -1].detach())
            hooks.append(layer.register_forward_hook(out_hook))
            dp = down_proj(layer)
            if dp is not None:
                def act_hook(_m, inp, name=name):
                    a = inp[0][0].detach().float().abs()          # [positions, d_mlp]
                    acts[name].append(float(a.mean()))
                    sel = (h_neurons or {}).get(name)
                    if sel:
                        idx = torch.tensor([j for j in sel if j < a.shape[-1]], device=a.device)
                        hn[name].append(float(a[:, idx].sum() / a.sum().clamp_min(1e-9)) if len(idx) else 0.0)
                hooks.append(dp.register_forward_pre_hook(act_hook))
    try:
        with torch.no_grad():
            final = model(ids).logits[0, -1].float()
    finally:
        for h in hooks:
            h.remove()
    looped = {n: c for n, c in calls.items() if len(c) > 1}
    if not looped:
        raise SystemExit("no layer ran more than once: this does not look like a looped model (or loops = 1)")
    n_loops = max(len(c) for c in looped.values())
    rec_layers = [n for n in calls if len(calls[n]) == n_loops]
    last = rec_layers[-1]                                 # the recurrent block's last layer: the state after each loop
    rd = readout(model)
    t_id = None
    if target:
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        t_id = t_ids[0] if t_ids else None
    final_top = int(final.argmax())
    loops = []
    for k, h in enumerate(calls[last]):
        lp = rd(h).log_softmax(-1).detach()
        p = lp.exp()
        v, i = p.topk(top_k)
        row = {"loop": k + 1, "top": [{"token": tok.decode([int(t)]), "p": round(float(x), 4)} for x, t in zip(v, i)],
               "entropy": round(float(-(p * lp).sum()), 4), "top_is_final": int(i[0]) == final_top}
        if t_id is not None:
            row["target_p"] = round(float(p[t_id]), 5)
            row["target_rank"] = int((p > p[t_id]).sum()) + 1
        loops.append(row)
    decided = next((r["loop"] for k, r in enumerate(loops) if all(x["top_is_final"] for x in loops[k:])), None)
    fp = final.softmax(-1)
    out = {"prompt": prompt, "target": target, "loops": n_loops, "recurrent_layers": rec_layers,
           "per_loop": loops, "decided_at_loop": decided,
           "final": {"token": tok.decode([final_top]), "p": round(float(fp[final_top]), 4),
                     **({"target_p": round(float(fp[t_id]), 5)} if t_id is not None else {})},
           "mlp_activity": {n: [round(x, 5) for x in acts[n]] for n in rec_layers if acts.get(n)}}
    if h_neurons:
        out["h_neuron_share"] = {n: [round(x, 5) for x in hn[n]] for n in rec_layers if hn.get(n)}
    return out


def h_neurons_by_layer(path: str, model) -> dict:
    """classifier.py's h_neurons.json (by_layer: absolute layer index -> neurons) mapped onto layer names, in the
    order the model's layer lists run."""
    d = json.loads(Path(path).read_text())
    order = [f"{ln}.{i}" for ln, ml in layer_lists(model) for i in range(len(ml))]
    return {order[int(k)]: v for k, v in (d.get("by_layer") or {}).items() if int(k) < len(order)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--target", help="the expected answer's first token, e.g. ' Canberra'")
    ap.add_argument("--loops", type=int, help="set the loop count (default: the model's)")
    ap.add_argument("--h-neurons", help="h_neurons.json from classifier.py on this model")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    model, tok, n = load(a.model, a.loops, a.device, a.dtype, a.trust_remote_code)
    hn = h_neurons_by_layer(a.h_neurons, model) if a.h_neurons else None
    res = lens(model, tok, a.prompt, a.target, a.top_k, hn)
    text = json.dumps(res, indent=1)
    (Path(a.out).write_text(text) if a.out else print(text))
    print(f"{res['loops']} loops; decided at loop {res['decided_at_loop']}; final '{res['final']['token']}' "
          f"p={res['final']['p']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
