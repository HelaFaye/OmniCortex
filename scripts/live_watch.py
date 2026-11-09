"""
Watch a Hugging Face model while it generates: for every new token, its CETT (each MLP neuron's share of
its layer's output), the emotion vectors' activations and the top features of feature dictionaries.

The decoding lab (decode_lab.py) uses this to drive the realtime views without a patched llama-server:
frames go to the live viewer (viz/stream.py, POST /api/push) as tokens are generated, and each reply can
be saved as a trace session that the 3D replay (viz/bloom.py), the timeline and the Godot client scrub,
with emotion and feature tracks alongside the neurons.

    CETT(layer, neuron) = |a| x ||W_down[:, neuron]|| / ||down_proj output||   (oc_common.CETTManager)

Only the newest position is read at each step, so the cost is a few vector operations per layer.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import torch

import features as FX


def _bin(v: torch.Tensor, bins: int) -> torch.Tensor:
    """Max-pool the last axis to `bins` (max, not mean: the cells worth watching are few)."""
    n = v.shape[-1]
    if not bins or bins >= n:
        return v
    step = -(-n // bins)
    pad = step * bins - n
    if pad:
        v = torch.nn.functional.pad(v, (0, pad), value=0.0)
    return v.reshape(*v.shape[:-1], bins, step).amax(-1)


class Watcher:
    """Context manager around generation. Call step(piece) after each forward pass."""

    def __init__(self, model, tok, cett: bool = True, bins: int = 256, emotions=None, emo_top: int = 3,
                 dicts: list | None = None, feat_top: int = 5, hide_dense: bool = True):
        from concept_edit import decoder_layers
        self.model, self.tok = model, tok
        self.layers = decoder_layers(model)
        self.cett_on, self.bins = cett, bins
        self.vec, self.emo_top = emotions, emo_top
        self.dicts = dicts or []                    # (sae, info, maxes, labels, dense, name)
        self.feat_top, self.hide_dense = feat_top, hide_dense
        self.hooks, self.cur = [], {}
        self.frames: list[dict] = []
        self.cett_rows: list[torch.Tensor] = []
        self.emo_rows: list[list[float]] = []
        self.wn = {}
        if cett:
            for l, layer in enumerate(self.layers):
                d = FX.down_of(FX.mlp_of(layer))
                self.wn[l] = d.weight.detach().float().norm(dim=0)          # [d_ff]

    # ------------------------------------------------------------ hooks
    def __enter__(self):
        if self.cett_on:
            for l, layer in enumerate(self.layers):
                d = FX.down_of(FX.mlp_of(layer))

                def hook(_m, inp, out, l=l):
                    a = inp[0][0, -1].detach().float()
                    o = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().float()
                    self.cur[("cett", l)] = a.abs() * self.wn[l].to(a.device) / (o.norm() + 1e-8)
                self.hooks.append(d.register_forward_hook(hook))
        if self.vec is not None:
            l = self.vec.default_layer

            def ehook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                self.cur["emo"] = h[0, -1].detach().float()
            self.hooks.append(self.layers[l].register_forward_hook(ehook))
        for k, (sae, info, *_rest) in enumerate(self.dicts):
            site = info.get("site")
            mod, kind = FX.site_module(self.model, site, int(info["layer"]))
            if kind == "out":
                def fhook(_m, _i, out, k=k):
                    x = out[0] if isinstance(out, tuple) else out
                    self.cur[("feat", k)] = x[0, -1].detach().float()
                self.hooks.append(mod.register_forward_hook(fhook))
            else:
                def fpre(_m, args, k=k):
                    self.cur[("feat", k)] = args[0][0, -1].detach().float()
                self.hooks.append(mod.register_forward_pre_hook(fpre))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ------------------------------------------------------------ per token
    @torch.no_grad()
    def step(self, piece: str, spread: float | None = None) -> dict:
        i = len(self.frames)
        fr: dict = {"i": i, "tok": piece}
        if spread is not None:
            fr["spread"] = round(float(spread), 4)
        if self.cett_on and ("cett", 0) in self.cur:
            rows = torch.stack([self.cur[("cett", l)] for l in range(len(self.layers))])     # [L, d_ff]
            binned = _bin(rows, self.bins).cpu()
            self.cett_rows.append(binned.half())
            fr["cett"] = binned
        if self.vec is not None and "emo" in self.cur:
            j = self.vec.j(None)
            h = self.cur["emo"]
            z = ((h - self.vec.center[j].to(h.device)) @ self.vec.U[j].to(h.device).t()).cpu()
            z = (z - self.vec.mu[j]) / self.vec.sd[j]
            self.emo_rows.append([round(float(v), 3) for v in z])
            top = z.topk(min(self.emo_top, len(z)))
            fr["emo"] = [[self.vec.names[int(k)], round(float(v), 2)] for v, k in zip(top.values, top.indices)]
            # every emotion clearly away from its story mean, so lanes stay continuous (absent = near 0)
            strong = (z.abs() >= 0.5).nonzero().flatten().tolist()
            strong = sorted(strong, key=lambda k: -abs(float(z[k])))[:24]
            fr["ez"] = {self.vec.names[k]: round(float(z[k]), 1) for k in strong}
        feats = []
        for k, (sae, info, maxes, labels, dense, name) in enumerate(self.dicts):
            x = self.cur.get(("feat", k))
            if x is None:
                continue
            f = sae.encode(x[None])[0]
            if self.hide_dense and dense is not None:
                f = f.masked_fill(dense.to(f.device), 0)
            rel = f / maxes.to(f.device).clamp(min=1e-6)
            v, idx = rel.topk(min(self.feat_top, rel.shape[0]))
            for vv, ii in zip(v, idx):
                if vv > 0:
                    feats.append([name, int(ii), round(float(vv), 3), labels.get(int(ii), "")])
        if feats:
            fr["feat"] = feats
        self.cur.clear()
        self.frames.append(fr)
        return fr

    # ------------------------------------------------------------ outputs
    def live_frame(self, fr: dict, tier: str = "sparse", top_k: int = 48) -> dict:
        """A frame for viz/stream.py: its tiers for the neuron field, plus emotion and feature tracks."""
        out = {"i": fr["i"], "tok": fr["tok"], "scored": False, "src": "lab"}
        if "cett" in fr:
            c = fr["cett"]
            out["s"] = round(float(c.max()), 4)
            if tier == "binned":
                out.update({"t": "binned", "v": [[round(float(x), 4) for x in row] for row in c]})
            else:
                vals, idx = c.flatten().topk(min(top_k, c.numel()))
                B = c.shape[1]
                out.update({"t": "sparse", "l": c.shape[0],
                            "c": [[int(k) // B, int(k) % B, round(float(v), 4)] for v, k in zip(vals, idx)]})
        else:
            out.update({"t": "none", "s": 0.0})
        for k in ("emo", "ez", "feat", "spread"):
            if k in fr:
                out[k] = fr[k]
        return out

    def summary(self) -> dict:
        """What the reply returns: per-token tracks without the neuron field."""
        out = {"tokens": [f["tok"] for f in self.frames]}
        if self.emo_rows:
            out["emotions"] = {"names": self.vec.names, "layer": self.vec.default_layer, "top": [f.get("emo") for f in self.frames]}
        if any("feat" in f for f in self.frames):
            out["features"] = [f.get("feat", []) for f in self.frames]
        return out

    def save(self, root: str | Path, prompt: str, model_id: str, extra: dict | None = None) -> str | None:
        """The reply as a trace session (records.py), readable by the 3D replay, the timeline and Godot."""
        if not self.frames:
            return None
        import sys
        viz = str(Path(__file__).resolve().parent.parent / "viz")
        if viz not in sys.path:
            sys.path.insert(0, viz)
        from records import Recorder
        T, L = len(self.frames), len(self.layers)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = Path(root).expanduser() / f"lab-{stamp}-{int(time.time() * 1000) % 1000:03d}"
        if self.cett_rows:
            agg = torch.cat(self.cett_rows).float().numpy()                   # [T * L, B]
            n_neurons = agg.shape[1]
        else:
            import numpy as np
            agg, n_neurons = np.zeros((T * L, 1), dtype="float32"), 1
        fields = {"kind": "trace", "n_frames": T, "n_layers": L, "stride": 1,
                  "pieces": [f["tok"] for f in self.frames], "question": prompt[:2000], "source": "decode_lab",
                  "binned": bool(self.cett_rows), **(extra or {})}
        if self.emo_rows:
            fields["emotions"] = {"names": self.vec.names, "layer": self.vec.default_layer, "z": self.emo_rows}
        if any("feat" in f for f in self.frames):
            fields["features"] = [f.get("feat", []) for f in self.frames]
        with Recorder(str(path), {"model": model_id, "n_layers": L, "n_neurons": n_neurons, "kind": "trace",
                                  "source": "decode_lab", "fingerprint": f"{model_id}:bins{n_neurons}"}) as r:
            r.add("reply", agg=agg, **fields)
        return str(path)


class Publisher:
    """POST frames to a live viewer (viz/stream.py /api/push). Failures are reported once, not raised."""

    def __init__(self, url: str, token: str | None = None, tier: str = "sparse"):
        self.url = url.rstrip("/") + "/api/push"
        self.token, self.tier, self.warned = token, tier, False

    def send(self, frame: dict) -> None:
        req = urllib.request.Request(self.url, data=json.dumps(frame, separators=(",", ":")).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except Exception as e:                                                # noqa: BLE001
            if not self.warned:
                print(f"live viewer at {self.url} not reachable ({type(e).__name__}: {e}); "
                      "frames are dropped until it answers", flush=True)
                self.warned = True
