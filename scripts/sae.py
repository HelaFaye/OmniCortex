"""
Sparse autoencoders and transcoders: dictionary learning over a model's activations.

A dictionary rewrites one activation vector x (the residual stream after a layer, an MLP's input or
output, or an MLP's neurons) as a sparse sum of learned directions:

    f = act(W_enc x + b_enc)          feature activations, mostly zero
    x_hat = W_dec f + b_dec           the reconstruction

A transcoder is the same with a different target: it reads one site (an MLP's input) and
reconstructs another (that MLP's output), so its features stand in for the MLP's computation.

Three activation functions, each from published work (docs/CREDITS.md):

  relu      ReLU, with an L1 penalty weighted by each feature's decoder norm, the decoder left
            unconstrained and the input scaled so that its average norm is sqrt(d)
            (Anthropic, Scaling Monosemanticity and its April 2024 training update).
  topk      the k largest pre-activations are kept (Gao et al. 2024), with the auxiliary loss on
            dead features (AuxK); unit-norm decoder rows.
  jumprelu  each feature has a learned threshold below which it is zero, trained with an L0
            penalty through straight-through estimates (Rajamanoharan et al. 2024; the form
            Gemma Scope's dictionaries use).

relu and jumprelu take a sparsity coefficient. Rather than hand-tune it, give target_l0 (how many
features should be active per token on average) and training steers the coefficient toward it.

Dictionaries trained elsewhere load too: SAELens folders (cfg.json + sae_weights.safetensors),
Gemma Scope's params.npz, or any .safetensors / .pt / .npz with W_enc, W_dec, b_enc, b_dec
(and threshold). See load().
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("relu", "topk", "jumprelu")
SITES = ("resid", "mlp_in", "mlp_out", "neurons")


def _rect(x):
    return ((x > -0.5) & (x < 0.5)).to(x)


class _JumpReLU(torch.autograd.Function):
    """x * (x > t). The threshold's gradient is a kernel estimate around the jump."""

    @staticmethod
    def forward(ctx, x, t, bw):
        ctx.save_for_backward(x, t)
        ctx.bw = bw
        return x * (x > t)

    @staticmethod
    def backward(ctx, g):
        x, t = ctx.saved_tensors
        bw = ctx.bw
        gx = (x > t) * g
        gt = (-(t / bw) * _rect((x - t) / bw) * g).reshape(-1, x.shape[-1]).sum(0)
        return gx, gt, None


class _Step(torch.autograd.Function):
    """(x > t), for counting active features (L0) with a gradient on the threshold."""

    @staticmethod
    def forward(ctx, x, t, bw):
        ctx.save_for_backward(x, t)
        ctx.bw = bw
        return (x > t).to(x)

    @staticmethod
    def backward(ctx, g):
        x, t = ctx.saved_tensors
        bw = ctx.bw
        gt = (-(1.0 / bw) * _rect((x - t) / bw) * g).reshape(-1, x.shape[-1]).sum(0)
        return torch.zeros_like(x), gt, None


class SAE(nn.Module):
    """W_enc [d_in, n], W_dec [n, d_out]. in_scale / out_scale map model units to the normalised
    units the dictionary was trained in (x * in_scale; reconstruction / out_scale)."""

    def __init__(self, d_in: int, n_features: int, arch: str = "topk", k: int = 32, d_out: int | None = None,
                 bandwidth: float = 1e-3, init_threshold: float = 1e-3, dec_init_norm: float = 0.1,
                 pre_bias: bool = False, seed: int = 0):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch: one of {', '.join(ARCHS)}")
        d_out = d_out or d_in
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n_features, d_out, generator=g)
        w = w / w.norm(dim=1, keepdim=True) * (dec_init_norm if arch == "relu" else 1.0)
        self.W_dec = nn.Parameter(w)
        if d_out == d_in:
            enc = w.t().clone()
        else:
            enc = torch.randn(d_in, n_features, generator=g) / math.sqrt(d_in)
        self.W_enc = nn.Parameter(enc)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        if arch == "jumprelu":
            self.log_threshold = nn.Parameter(torch.full((n_features,), math.log(init_threshold)))
        self.register_buffer("in_scale", torch.tensor(1.0))
        self.register_buffer("out_scale", torch.tensor(1.0))
        self.arch, self.k, self.bandwidth, self.pre_bias = arch, int(k), float(bandwidth), bool(pre_bias)
        self.d_in, self.d_out, self.n = d_in, d_out, n_features

    # ---------------------------------------------------------------- forward
    @property
    def threshold(self):
        return self.log_threshold.exp() if self.arch == "jumprelu" else None

    def pre(self, x):
        x = x * self.in_scale
        if self.pre_bias and self.d_in == self.d_out:
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def act(self, pre):
        if self.arch == "relu":
            return F.relu(pre)
        if self.arch == "topk":
            k = min(self.k, pre.shape[-1])
            vals, idx = pre.topk(k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
        return _JumpReLU.apply(pre, self.threshold, self.bandwidth)

    def encode(self, x):
        """Feature activations for x in model units."""
        return self.act(self.pre(x))

    def decode(self, f):
        """Reconstruction in the dictionary's normalised units."""
        return f @ self.W_dec + self.b_dec

    def reconstruct(self, x):
        """x (model units) -> reconstruction (model units) and the features."""
        f = self.encode(x)
        return self.decode(f) / self.out_scale, f

    def directions(self):
        """Unit decoder directions [n, d_out] in model space."""
        return F.normalize(self.W_dec.detach(), dim=1)

    @torch.no_grad()
    def init_thresholds(self, x, target_l0: float):
        """JumpReLU: start each threshold at the pre-activation quantile that leaves about twice the
        wanted number of features active, and the kernel bandwidth at a twentieth of a typical
        threshold, so short training runs need not first climb from a near-zero threshold."""
        pre = self.pre(x.float())
        q = 1.0 - min(0.5, 2.0 * float(target_l0) / self.n)
        t = torch.quantile(pre, q, dim=0).clamp(min=1e-4)
        self.log_threshold.copy_(t.log())
        self.bandwidth = max(1e-4, 0.05 * float(t.median()))

    @torch.no_grad()
    def fold_decoder_norms(self):
        """relu: the same predictions with unit-norm decoder rows, so a feature's activation reads as
        f_i ||W_d,i|| (the April 2024 update's last step)."""
        nrm = self.W_dec.norm(dim=1).clamp(min=1e-8)
        self.W_enc.mul_(nrm[None, :])
        self.b_enc.mul_(nrm)
        self.W_dec.div_(nrm[:, None])

    def config(self) -> dict:
        return {"d_in": self.d_in, "d_out": self.d_out, "n_features": self.n, "arch": self.arch, "k": self.k,
                "bandwidth": self.bandwidth, "pre_bias": self.pre_bias,
                "in_scale": float(self.in_scale), "out_scale": float(self.out_scale)}


# -------------------------------------------------------------------- training

def norm_scale(xs, d: int) -> float:
    """The factor that makes the mean norm sqrt(d)."""
    m = torch.cat([x.float().norm(dim=-1) for x in xs]).mean().item()
    return math.sqrt(d) / max(m, 1e-8)


def _lr_at(step: int, steps: int, lr: float, warmup: float = 0.01, decay: float = 0.2) -> float:
    w, dstart = max(1, int(steps * warmup)), int(steps * (1 - decay))
    if step < w:
        return lr * (step + 1) / w
    if step >= dstart and steps > dstart:
        return lr * max(0.0, (steps - step) / (steps - dstart))
    return lr


def train(sae: SAE, batches, steps: int, lr: float = 2e-4, sparsity: float | None = None,
          target_l0: float | None = None, sparsity_warmup: float = 0.05, aux_alpha: float = 1 / 32,
          dead_after: int = 1_000_000, log_every: int = 50, on_log=None, device: str = "cpu",
          clip: float = 1.0) -> list[dict]:
    """batches: an iterator of (x, y) in model units (y None for a plain autoencoder) on any device.
    sparsity: the L1 (relu) or L0 (jumprelu) coefficient; with target_l0 it is only the starting
    point and is adjusted every few steps. Returns the logged history."""
    sae.to(device).train()
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    if sparsity is None:
        # relu: the Scaling Monosemanticity update's coefficient. jumprelu: the JumpReLU paper's scale
        # once our input normalisation (mean norm sqrt(d), not 1) is accounted for.
        sparsity = {"relu": 5.0, "jumprelu": 1e-3 * sae.d_out, "topk": 0.0}[sae.arch]
    lam = float(sparsity)
    since = torch.zeros(sae.n, device=device)
    hist, t0, acc = [], time.time(), {"mse": 0.0, "fvu_num": 0.0, "fvu_den": 0.0, "l0": 0.0, "n": 0}
    for step in range(steps):
        x, y = next(batches)
        x = x.to(device, torch.float32)
        y = x if y is None else y.to(device, torch.float32)
        if step == 0 and sae.arch == "jumprelu":
            sae.init_thresholds(x, target_l0 or 32)
        yn = y * sae.out_scale
        pre = sae.pre(x)
        f = sae.act(pre)
        y_hat = sae.decode(f)
        err = y_hat - yn
        mse = err.pow(2).sum(-1).mean()
        loss = mse
        warm = min(1.0, (step + 1) / max(1, int(steps * sparsity_warmup)))
        if sae.arch == "relu":
            loss = loss + lam * warm * (f * sae.W_dec.norm(dim=1)).sum(-1).mean()
        elif sae.arch == "jumprelu":
            loss = loss + lam * warm * _Step.apply(pre, sae.threshold, sae.bandwidth).sum(-1).mean()
        fired = (f > 0).any(0)
        since += x.shape[0]
        since[fired] = 0
        dead = since > dead_after
        if sae.arch == "topk" and aux_alpha > 0 and bool(dead.any()):
            k_aux = min(sae.d_out // 2, int(dead.sum()))
            masked = pre.masked_fill(~dead, float("-inf"))
            vals, idx = masked.topk(k_aux, dim=-1)
            f_aux = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
            e_hat = f_aux @ sae.W_dec
            aux = (e_hat - (yn - y_hat).detach()).pow(2).sum(-1).mean()
            loss = loss + aux_alpha * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(sae.parameters(), clip)
        for gr in opt.param_groups:
            gr["lr"] = _lr_at(step, steps, lr)
        opt.step()
        if sae.arch in ("topk", "jumprelu"):
            with torch.no_grad():
                sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
        with torch.no_grad():
            l0 = (f > 0).float().sum(-1).mean().item()
            acc["mse"] += mse.item()
            acc["fvu_num"] += err.pow(2).sum().item()
            acc["fvu_den"] += (yn - yn.mean(0)).pow(2).sum().item()
            acc["l0"] += l0
            acc["n"] += 1
        if target_l0 and sae.arch in ("relu", "jumprelu") and step % 10 == 9 and step > steps * sparsity_warmup:
            # multiplicative steering toward the wanted number of active features
            lam *= math.exp(0.05 * max(-1.0, min(1.0, math.log(max(l0, 1e-3) / target_l0))))
        if (step + 1) % log_every == 0 or step == steps - 1:
            n = max(1, acc["n"])
            rec = {"step": step + 1, "mse": acc["mse"] / n, "fvu": acc["fvu_num"] / max(acc["fvu_den"], 1e-12),
                   "l0": acc["l0"] / n, "dead": float(dead.float().mean()), "sparsity": lam,
                   "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
            hist.append(rec)
            if on_log:
                on_log(rec)
            acc = {"mse": 0.0, "fvu_num": 0.0, "fvu_den": 0.0, "l0": 0.0, "n": 0}
    if sae.arch == "relu":
        sae.fold_decoder_norms()
    sae.eval()
    return hist


@torch.no_grad()
def evaluate(sae: SAE, batches, n_batches: int, device: str = "cpu") -> dict:
    """Held-out fraction of variance unexplained, L0, and features never active in these batches."""
    sae.to(device).eval()
    num = den = l0 = 0.0
    rows = 0
    fired = torch.zeros(sae.n, dtype=torch.bool, device=device)
    for _ in range(n_batches):
        try:
            x, y = next(batches)
        except StopIteration:
            break
        x = x.to(device, torch.float32)
        y = x if y is None else y.to(device, torch.float32)
        yn = y * sae.out_scale
        f = sae.encode(x)
        err = sae.decode(f) - yn
        num += err.pow(2).sum().item()
        den += (yn - yn.mean(0)).pow(2).sum().item()
        l0 += (f > 0).float().sum().item()
        rows += x.shape[0]
        fired |= (f > 0).any(0)
    return {"fvu": num / max(den, 1e-12), "l0": l0 / max(rows, 1), "never_fired": float((~fired).float().mean()),
            "rows": rows}


# -------------------------------------------------------------------- save / load

def save(sae: SAE, out: str | Path, meta: dict | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": sae.state_dict(), "cfg": sae.config()}, out / "sae.pt")
    info = {**(meta or {}), **sae.config(), "format": "neuronscope-sae/1"}
    (out / "sae.json").write_text(json.dumps(info, indent=1))
    return out


def _from_tensors(t: dict, cfg: dict) -> SAE:
    def get(*names):
        for n in names:
            if n in t:
                return torch.as_tensor(t[n]).float()
        return None
    W_enc, W_dec = get("W_enc", "encoder.weight", "w_enc"), get("W_dec", "decoder.weight", "w_dec")
    if W_enc is None or W_dec is None:
        raise ValueError(f"no W_enc / W_dec among {', '.join(sorted(t))[:300]}")
    n = int(cfg.get("d_sae") or cfg.get("n_features") or 0) or min(W_dec.shape)
    if W_enc.shape[0] == n and W_enc.shape[1] != n:          # stored [n, d_in] (nn.Linear layout)
        W_enc = W_enc.t()
    if W_dec.shape[1] == n and W_dec.shape[0] != n:          # stored [d_out, n]
        W_dec = W_dec.t()
    th = get("threshold", "log_threshold")
    arch = cfg.get("arch") or ("jumprelu" if th is not None else
                              "topk" if (cfg.get("activation_fn_str") == "topk" or cfg.get("architecture") == "topk")
                              else "relu")
    k = int(cfg.get("k") or (cfg.get("activation_fn_kwargs") or {}).get("k") or 32)
    sae = SAE(W_enc.shape[0], W_enc.shape[1], arch, k, W_dec.shape[1], pre_bias=bool(cfg.get("apply_b_dec_to_input")))
    with torch.no_grad():
        sae.W_enc.copy_(W_enc)
        sae.W_dec.copy_(W_dec)
        b = get("b_enc", "encoder.bias")
        if b is not None:
            sae.b_enc.copy_(b)
        b = get("b_dec", "decoder.bias")
        if b is not None:
            sae.b_dec.copy_(b)
        if arch == "jumprelu" and th is not None:
            sae.log_threshold.copy_(th if "log_threshold" in t and "threshold" not in t else th.clamp(min=1e-12).log())
        for key in ("in_scale", "out_scale"):
            if cfg.get(key):
                getattr(sae, key).fill_(float(cfg[key]))
    return sae


HOOK_RX = re.compile(r"blocks\.(\d+)\.hook_(resid_post|resid_pre|mlp_out|resid_mid|mlp\.hook_post|mlp_in)")


def site_from_hook(hook: str) -> tuple[str | None, int | None]:
    """TransformerLens hook name -> (site, layer) in OmniCortex's terms."""
    m = HOOK_RX.search(hook or "")
    if not m:
        return None, None
    layer, kind = int(m.group(1)), m.group(2)
    return {"resid_post": ("resid", layer), "resid_pre": ("resid", layer - 1), "mlp_out": ("mlp_out", layer),
            "mlp.hook_post": ("neurons", layer), "mlp_in": ("mlp_in", layer),
            "resid_mid": ("mlp_in", layer)}[kind]


SAELENS_REGISTRY = "https://raw.githubusercontent.com/decoderesearch/SAELens/main/sae_lens/pretrained_saes.yaml"


def saelens_registry(cache: str | Path | None = None, url: str = SAELENS_REGISTRY, max_age: float = 7 * 86400) -> dict:
    """SAELens's catalogue of pretrained dictionaries ({release: {repo_id, model, saes: [{id, path, ...}]}}),
    fetched from its repository and cached for a week."""
    import os
    import time
    import urllib.request
    import yaml
    p = Path(cache or Path(os.environ.get("NS_HOME") or Path.home() / ".neuronscope") / "saelens" / "pretrained_saes.yaml")
    if not p.exists() or time.time() - p.stat().st_mtime > max_age:
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OmniCortex"}), timeout=60) as r:
            p.write_bytes(r.read())
    return yaml.safe_load(p.read_text()) or {}


def from_release(release: str, sae_id: str, registry: dict | None = None, download=None) -> tuple[Path, dict]:
    """Download one SAELens release's dictionary (e.g. gemma-scope-2b-pt-res, layer_12/width_16k/average_l0_82)
    from Hugging Face. -> (local folder to load(), what the catalogue says about it)."""
    reg = registry if registry is not None else saelens_registry()
    rel = reg.get(release)
    if rel is None:
        close = sorted(k for k in reg if release.split("-")[0] in k)[:8]
        raise ValueError(f"no SAELens release {release!r}" + (f"; similar: {', '.join(close)}" if close else ""))
    entry = next((x for x in rel.get("saes") or [] if x.get("id") == sae_id), None)
    if entry is None:
        ids = [x.get("id") for x in rel.get("saes") or []]
        raise ValueError(f"{release} has no {sae_id!r}; it has {len(ids)}, e.g. {', '.join(map(str, ids[:5]))}")
    if download is None:
        from huggingface_hub import snapshot_download as download
    path = entry.get("path") or sae_id
    root = download(repo_id=rel["repo_id"], allow_patterns=[f"{path}/*", f"{path}*"])
    folder = Path(root) / path
    info = {"release": release, "sae_id": sae_id, "repo_id": rel["repo_id"], "model": rel.get("model"),
            "conversion_func": rel.get("conversion_func"), **{k: v for k, v in entry.items() if k not in ("id", "path")}}
    return (folder if folder.is_dir() else folder.parent), info


def load(path: str | Path, device: str = "cpu") -> tuple[SAE, dict]:
    """-> (dictionary, its metadata). A OmniCortex folder or sae.pt, an SAELens folder, a Gemma Scope
    params.npz, or a file with W_enc / W_dec / b_enc / b_dec."""
    p = Path(path).expanduser()
    meta: dict = {}
    if p.is_dir():
        if (p / "sae.pt").exists():
            meta = json.loads((p / "sae.json").read_text()) if (p / "sae.json").exists() else {}
            p = p / "sae.pt"
        elif (p / "sae_weights.safetensors").exists():
            meta = json.loads((p / "cfg.json").read_text()) if (p / "cfg.json").exists() else {}
            p = p / "sae_weights.safetensors"
        elif (p / "params.npz").exists():
            p = p / "params.npz"
        else:
            cands = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.npz")) + sorted(p.glob("*.pt"))
            if not cands:
                raise FileNotFoundError(f"{p}: no dictionary weights found")
            p = cands[0]
    if p.suffix == ".pt":
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state" in obj and "cfg" in obj:
            c = obj["cfg"]
            sae = SAE(c["d_in"], c["n_features"], c["arch"], c.get("k", 32), c.get("d_out"),
                      c.get("bandwidth", 1e-3), pre_bias=c.get("pre_bias", False))
            sae.load_state_dict(obj["state"])
            meta = {**meta, **c}
        else:
            sae = _from_tensors(obj if isinstance(obj, dict) else {}, meta)
    elif p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sae = _from_tensors(load_file(str(p)), meta)
    elif p.suffix == ".npz":
        import numpy as np
        z = np.load(p)
        sae = _from_tensors({k: z[k] for k in z.files}, meta)
    else:
        raise ValueError(f"{p}: expected .pt, .safetensors or .npz")
    if meta.get("hook_name") and not meta.get("site"):
        meta["site"], meta["layer"] = site_from_hook(meta["hook_name"])
    meta.setdefault("arch", sae.arch)
    meta.setdefault("n_features", sae.n)
    return sae.to(device).eval(), meta
