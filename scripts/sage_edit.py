#!/usr/bin/env python3
"""
SAGE: Safety-Aware Gated Editing (Nonta & Rambhatla, ICLR 2026 TTU workshop,
"Refusal-Orthogonal Gated Editing for Safer Localized LLM Adaptation").

Adapt a frozen model to a task with small, gated activation edits, while
keeping a *protected* subspace of its representations untouched.

The paper protects refusal so that task fine-tuning cannot erode safety.
In OmniCortex the same method can protect, for example:
- the "I don't know" direction, so retraining on deficits does not make the
  model guess more;
- any concept found with concept_edit.py.

**Stage 1: the protected subspace R.** It is spanned by the directions that
separate a concept set from a baseline set (`--protect-concept` and
`--protect-baseline`, difference of means or a PCA cone), or loaded from
`concept_edit.py --analyze`'s directions.pt.

**Stage 2: refusal-orthogonal gated editing.**
- Every attention head i in layer l contributes z (its slice of the
  attention output, in residual space).
- It gets an additive edit a, a multiplicative edit m and two Hard-Concrete
  gates g_a, g_m in [0, 1] with an L0 penalty, as in JoLA, so only a few
  heads end up edited.
- The edit touches only the part of z outside R:

      z' = P_R z + P_R⊥ ( (1 + g_m·m) ⊙ P_R⊥ z + g_a·a )

  a and m are projected onto R⊥ after every step. What z says along R is
  exactly what it said before, which the script checks after training.
- The base weights never change. Training data is a task set: deficits.py's
  sft.jsonl, or {"prompt", "completion"} lines.

**Inference.**
- Here: `sage_edit.py chat`, with the edits.
- In llama.cpp (Studio, LM Studio): `--export-cvector` writes the additive
  part as a control vector GGUF (one vector per layer: the gated additive
  edits summed over heads).
- That export is approximate, for three reasons:
  - llama.cpp adds control vectors at the end of a layer rather than after
    attention;
  - it has no vector for layer 0;
  - multiplicative edits have no control-vector form, so they are dropped.

  For the exact method, use the edits through transformers.

    python scripts/concept_edit.py --model org/m --concept dk.txt --baseline guess.txt --out runs/dk --analyze
    python scripts/sage_edit.py train --model org/m --protect runs/dk/directions.pt --data runs/retrain \\
        --out runs/sage --epochs 2
    python scripts/sage_edit.py chat --edits runs/sage/edits.pt "Who won the 1987 Tour de France?"
    python scripts/sage_edit.py export --edits runs/sage/edits.pt --cvector runs/sage/sage.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1          # Hard Concrete (Louizos et al., 2018)


def _torch():
    import torch
    return torch


# ------------------------------------------------------------------ the edit module

class SageEdits:
    """Edit parameters for every (layer, head), with the protected basis per layer."""

    def __init__(self, n_layers: int, n_heads: int, d: int, R, layers=None, init_log_alpha: float = 0.0):
        torch = _torch()
        self.n_layers, self.n_heads, self.d = n_layers, n_heads, d
        self.layers = list(layers if layers is not None else range(n_layers))
        self.R = R                                              # [n_layers, d, r] orthonormal columns
        self.a = torch.zeros(n_layers, n_heads, d, requires_grad=True)
        self.m = torch.zeros(n_layers, n_heads, d, requires_grad=True)
        self.la = torch.full((n_layers, n_heads), init_log_alpha, requires_grad=True)   # additive gate
        self.lm = torch.full((n_layers, n_heads), init_log_alpha, requires_grad=True)   # multiplicative gate
        self.training = True

    def params(self):
        return [self.a, self.m, self.la, self.lm]

    def gate(self, log_alpha):
        torch = _torch()
        if self.training:
            u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / BETA)
        else:
            s = torch.sigmoid(log_alpha)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def l0(self):
        torch = _torch()
        shift = BETA * math.log(-GAMMA / ZETA)
        return torch.sigmoid(self.la - shift).sum() + torch.sigmoid(self.lm - shift).sum()

    def project_(self):
        """Keep a and m in R⊥ (after every optimizer step)."""
        torch = _torch()
        with torch.no_grad():
            for l in self.layers:
                R = self.R[l]
                self.a[l] -= (self.a[l] @ R) @ R.T
                self.m[l] -= (self.m[l] @ R) @ R.T

    def edit(self, l: int, z):
        """z [..., H, d] per-head contributions -> edited contributions."""
        R = self.R[l].to(z.device, z.dtype)
        par = (z @ R) @ R.T                                     # P_R z
        perp = z - par
        ga = self.gate(self.la[l]).to(z.device, z.dtype)[:, None]
        gm = self.gate(self.lm[l]).to(z.device, z.dtype)[:, None]
        e = (1 + gm * self.m[l].to(z.device, z.dtype)) * perp + ga * self.a[l].to(z.device, z.dtype)
        e = e - (e @ R) @ R.T                                   # re-project: nothing leaks into R
        return par + e

    def state(self) -> dict:
        return {"a": self.a.detach().cpu(), "m": self.m.detach().cpu(), "la": self.la.detach().cpu(),
                "lm": self.lm.detach().cpu(), "R": self.R.cpu(), "layers": self.layers,
                "n_layers": self.n_layers, "n_heads": self.n_heads, "d": self.d}

    @classmethod
    def from_state(cls, st: dict) -> "SageEdits":
        e = cls(st["n_layers"], st["n_heads"], st["d"], st["R"], st["layers"])
        with _torch().no_grad():
            e.a.copy_(st["a"])
            e.m.copy_(st["m"])
            e.la.copy_(st["la"])
            e.lm.copy_(st["lm"])
        e.training = False
        return e


def attn_out(layer):
    """The attention output projection (o_proj) of a decoder block."""
    for pn in ("self_attn", "attn", "attention"):
        p = getattr(layer, pn, None)
        if p is not None:
            for n in ("o_proj", "out_proj", "dense"):
                if hasattr(p, n):
                    return getattr(p, n)
    raise SystemExit("cannot find the attention output projection of this architecture")


class Hooks:
    """Replaces each edited layer's o_proj output with the sum of edited per-head contributions."""

    def __init__(self, model, edits: SageEdits, record=None):
        import concept_edit as ce
        self.handles = []
        layers = ce.decoder_layers(model)
        for l in edits.layers:
            proj = attn_out(layers[l])
            H = edits.n_heads

            def hook(mod, inputs, output, l=l, H=H):
                x = inputs[0]
                hd = x.shape[-1] // H
                W = mod.weight.view(mod.weight.shape[0], H, hd)            # [d, H, hd]
                z = _torch().einsum("...hk,dhk->...hd", x.view(*x.shape[:-1], H, hd), W)
                z2 = edits.edit(l, z)
                if record is not None:
                    R = edits.R[l].to(z.device, z.dtype)
                    record.append(float(((z2 - z) @ R).abs().max()))
                out = z2.sum(-2)
                return out + mod.bias if mod.bias is not None else out
            self.handles.append(proj.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------ data

def load_task(path: Path) -> list[tuple[list, str]]:
    """-> [(chat messages before the answer, answer)]."""
    files = [path / "sft.jsonl"] if path.is_dir() else [path]
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r.get("messages"), list) and r["messages"] and r["messages"][-1]["role"] == "assistant":
                rows.append((r["messages"][:-1], str(r["messages"][-1]["content"])))
            elif "prompt" in r and "completion" in r:
                rows.append(([{"role": "user", "content": str(r["prompt"])}], str(r["completion"])))
    if not rows:
        raise SystemExit(f"{path}: no training examples (messages or prompt/completion)")
    return rows


def encode(tok, msgs: list, answer: str, max_len: int):
    """-> (input_ids, labels) with the loss on the answer only."""
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(m["content"] if isinstance(m["content"], str) else str(m["content"]) for m in msgs) + "\n"
    p = tok(prompt, add_special_tokens=not bool(getattr(tok, "chat_template", None)))["input_ids"]
    a = tok(answer + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return ids, labels


# ------------------------------------------------------------------ stage 1

def protected_basis(a, model, tok, n_layers: int, d: int):
    """[n_layers, d, r]: for edits in layer l, the protected directions in the residual after layer l."""
    torch = _torch()
    import concept_edit as ce
    if a.protect:
        st = torch.load(a.protect, map_location="cpu")
        basis = st["basis"]
    else:
        if not (a.protect_concept and a.protect_baseline):
            raise SystemExit("give --protect (concept_edit.py directions.pt) or --protect-concept and --protect-baseline")
        A = ce.collect(model, tok, ce.load_prompts(a.protect_concept, a.limit), a.batch)
        B = ce.collect(model, tok, ce.load_prompts(a.protect_baseline, a.limit), a.batch)
        basis = ce.directions(A, B, a.method, a.k, True)
    if basis.shape[1] != d or basis.shape[0] != n_layers + 1:
        raise SystemExit(f"protected directions are for a different model ({tuple(basis.shape)} vs {n_layers + 1} "
                         f"positions x {d})")
    if a.protect_global is not None:
        g = ce.global_basis(basis, a.protect_global + 1)
        return g.unsqueeze(0).repeat(n_layers, 1, 1)
    return basis[1:].clone()


# ------------------------------------------------------------------ commands

def load(model_id: str, cpu: bool):
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32 if dev == "cpu" else torch.bfloat16, trust_remote_code=False)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def dims(model) -> tuple[int, int, int]:
    cfg = getattr(model.config, "text_config", model.config)
    return cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size


def task_loss(model, tok, rows, edits, max_len) -> float:
    torch = _torch()
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    with torch.no_grad():
        for msgs, ans in rows:
            ids, lab = encode(tok, msgs, ans, max_len)
            out = model(input_ids=torch.tensor([ids], device=dev), labels=torch.tensor([lab], device=dev))
            tot += float(out.loss)
            n += 1
    return tot / max(1, n)


def train(a) -> int:
    torch = _torch()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, tok = load(a.model, a.cpu)
    L, H, d = dims(model)
    R = protected_basis(a, model, tok, L, d)
    layers = list(range(L))
    if a.layers:
        layers = sorted({int(x) for part in a.layers.split(",") for x in
                         range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)})
    edits = SageEdits(L, H, d, R, layers, a.init_log_alpha)
    rows = load_task(Path(a.data))
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * a.val_frac)) if len(rows) > 4 else 0
    val, tr = rows[:n_val], rows[n_val:]
    edits.training = False
    hooks = Hooks(model, edits)
    before = task_loss(model, tok, val, edits, a.max_length) if val else None
    edits.training = True
    opt = torch.optim.Adam(edits.params(), lr=a.lr)
    dev = next(model.parameters()).device
    step = 0
    for ep in range(int(a.epochs)):
        random.shuffle(tr)
        for msgs, ans in tr:
            ids, lab = encode(tok, msgs, ans, a.max_length)
            out = model(input_ids=torch.tensor([ids], device=dev), labels=torch.tensor([lab], device=dev))
            loss = out.loss.float() + a.l0 * edits.l0() / (L * H * 2)
            loss.backward()
            if (step + 1) % a.grad_accum == 0:
                opt.step()
                opt.zero_grad()
                edits.project_()
            step += 1
            if step % a.log_every == 0:
                print(f"epoch {ep} step {step}: loss {float(out.loss):.4f}", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        if a.max_steps and step >= a.max_steps:
            break
    opt.step()
    opt.zero_grad()
    edits.project_()
    edits.training = False
    after = task_loss(model, tok, val, edits, a.max_length) if val else None
    # check: the protected component is unchanged
    hooks.remove()
    leak = []
    hooks = Hooks(model, edits, record=leak)
    task_loss(model, tok, (val or tr)[:4], edits, a.max_length)
    hooks.remove()
    ga, gm = edits.gate(edits.la), edits.gate(edits.lm)
    active = int(((ga > 0.5) | (gm > 0.5)).sum())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = edits.state()
    st["base_model"] = a.model
    torch.save(st, out / "edits.pt")
    rep = {"base_model": a.model, "layers": layers, "heads_per_layer": H, "protected_dims": int(R.shape[-1]),
           "steps": step, "val_loss_before": before, "val_loss_after": after,
           "active_components": active, "components": L * H,
           "max_change_along_protected": max(leak) if leak else 0.0}
    (out / "sage.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def chat(a) -> int:
    torch = _torch()
    st = torch.load(a.edits, map_location="cpu")
    model, tok = load(a.model or st["base_model"], a.cpu)
    edits = SageEdits.from_state(st)
    if not a.no_edits:
        Hooks(model, edits)
    msgs = [{"role": "user", "content": a.prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) \
        if getattr(tok, "chat_template", None) else a.prompt
    enc = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    print(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return 0


def export(a) -> int:
    """The additive part as a llama.cpp control vector (direction.<layer>, layer >= 1)."""
    torch = _torch()
    import numpy as np
    st = torch.load(a.edits, map_location="cpu")
    edits = SageEdits.from_state(st)
    ga = edits.gate(edits.la).detach()                          # [L, H]
    vec = (ga[:, :, None] * edits.a.detach()).sum(1)            # [L, d]: summed over heads
    try:
        import gguf
    except ImportError:
        llama = Path(a.llama or "~/llama.cpp").expanduser() / "gguf-py"
        sys.path.insert(0, str(llama))
        import gguf
    w = gguf.GGUFWriter(a.cvector, "controlvector")
    w.add_string("controlvector.model_hint", str(st.get("base_model", "")))
    w.add_uint32("controlvector.layer_count", int(vec.shape[0]))
    dropped = []
    for l in range(vec.shape[0]):
        if l == 0:
            if float(vec[0].abs().max()) > 0:
                dropped.append("layer 0 (llama.cpp has no control vector there)")
            continue
        w.add_tensor(f"direction.{l}", vec[l].numpy().astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    gm = edits.gate(edits.lm).detach()
    if float((gm[:, :, None] * edits.m.detach()).abs().max()) > 0:
        dropped.append("multiplicative edits (no control-vector form)")
    print(f"wrote {a.cvector}" + (f"; not included: {', '.join(dropped)}" if dropped else ""))
    print(f"use: llama-server ... --control-vector {a.cvector}  (or the model's Extra flags in Studio)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="stage 1 (protected subspace) and stage 2 (gated edits)")
    t.add_argument("--model", required=True)
    t.add_argument("--data", required=True, help="deficits.py output folder (sft.jsonl) or a JSONL file")
    t.add_argument("--out", required=True)
    t.add_argument("--protect", help="directions.pt from concept_edit.py")
    t.add_argument("--protect-concept", help="or: prompts with the behaviour to protect")
    t.add_argument("--protect-baseline", help="and matched prompts without it")
    t.add_argument("--protect-global", type=float, help="one protected subspace for all layers, taken after this layer")
    t.add_argument("--method", choices=["mean", "pca"], default="mean")
    t.add_argument("--k", type=int, default=1, help="protected dimensions (pca)")
    t.add_argument("--layers", help="edit only these layers, e.g. 8-20")
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--lr", type=float, default=5e-3)
    t.add_argument("--l0", type=float, default=1e-2, help="sparsity pressure on the gates")
    t.add_argument("--init-log-alpha", type=float, default=0.0)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--max-length", type=int, default=1024)
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("chat", help="generate with the edits applied")
    c.add_argument("--edits", required=True)
    c.add_argument("--model", help="default: the model the edits were trained on")
    c.add_argument("--no-edits", action="store_true", help="the base model, for comparison")
    c.add_argument("--max-new-tokens", type=int, default=200)
    c.add_argument("--cpu", action="store_true")
    c.add_argument("prompt")
    e = sub.add_parser("export", help="the additive edits as a llama.cpp control vector GGUF")
    e.add_argument("--edits", required=True)
    e.add_argument("--cvector", required=True)
    e.add_argument("--llama", help="llama.cpp checkout, for its gguf-py (if gguf is not installed)")
    a = p.parse_args(argv)
    return {"train": train, "chat": chat, "export": export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
