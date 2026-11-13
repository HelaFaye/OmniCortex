"""
Feature dictionaries and emotion vectors inside the decoding lab (decode_lab.py): steering while it
generates, and the analysis endpoints Studio's Features page calls.

Variant options (also accepted per request under "omnicortex": {...}):

feature  clamp or add dictionary features at every token, as in Scaling Monosemanticity: the
         activation is split into the dictionary's reconstruction and its error; the reconstruction
         is rebuilt with the chosen features set, and the error is left as it was.
         {"sae": folder, "set": {"123": 5, "877": -2}, "mode": "clamp" | "add"}
         Values are multiples of the feature's maximum activation (from features.json; 1 if it has not
         been interpreted). "add" adds value x max x the feature's decoder direction instead.
emotion  add an emotion mix: sum_e s_e x |h| x v_e after the given layers (s as a fraction of the
         residual stream's norm at each token, the emotions paper's unit).
         {"file": folder from emotions.py build, "mix": {"calm": 0.05, "desperate": -0.03}, "layers": [l]}

Endpoints (POST JSON, loopback only like the lab itself):
  /ns/info        the model
  /ns/inspect     per token: top features of each dictionary, emotion activations, logit-lens spread
  /ns/compare     a reply with and without the given steering
  /ns/graph       attribution graph over transcoders (feature_circuits.build_graph)
  /ns/intervene   change transcoder features, compare next-token predictions
  /ns/attribute   which features of one dictionary drive a completion
  /ns/sweep       emotion activation before the reply as a quantity in a template changes
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import torch

_cache: dict = {}
_lock = threading.Lock()


def _check_dir(p: str) -> Path:
    q = Path(str(p)).expanduser()
    if not q.exists():
        raise ValueError(f"{p}: not found")
    return q


def load_dict(path: str, device):
    """-> (dictionary, info, max activation per feature, labels) cached by path and modification time."""
    import sae as S
    p = _check_dir(path)
    key = ("sae", str(p.resolve()), (p / "sae.pt").stat().st_mtime if (p / "sae.pt").exists() else 0,
           (p / "labels.json").stat().st_mtime if (p / "labels.json").exists() else 0)
    with _lock:
        if key in _cache:
            return _cache[key]
    sae, info = S.load(p, device)
    maxes = torch.ones(sae.n)
    dense = torch.zeros(sae.n, dtype=torch.bool)
    labels: dict[int, str] = {}
    fj = p / "features.json"
    if fj.exists():
        for e in json.loads(fj.read_text()).get("features", []):
            if e.get("max"):
                maxes[e["id"]] = float(e["max"])
            if (e.get("freq") or 0) > 0.2:
                dense[e["id"]] = True
            if e.get("label"):
                labels[e["id"]] = e["label"]
    lj = p / "labels.json"
    if lj.exists():
        labels.update({int(k): v for k, v in json.loads(lj.read_text()).items() if v})
    val = (sae, info, maxes.to(device), labels, dense.to(device))
    with _lock:
        _cache[key] = val
    return val


def load_emotions(path: str, device):
    import emotions as EM
    p = _check_dir(path)
    f = p / "emotions.pt" if p.is_dir() else p
    key = ("emo", str(f.resolve()), f.stat().st_mtime)
    with _lock:
        if key not in _cache:
            _cache[key] = EM.Vectors(f, device)
        return _cache[key]


def validate(opt: dict, name: str = "variant") -> None:
    f = opt.get("feature")
    if f is not None and not (isinstance(f, dict) and f.get("sae") and isinstance(f.get("set"), dict)):
        raise ValueError(f"{name}: feature needs sae and set ({{\"feature id\": value}})")
    e = opt.get("emotion")
    if e is not None and not (isinstance(e, dict) and e.get("file") and isinstance(e.get("mix"), dict)):
        raise ValueError(f"{name}: emotion needs file and mix ({{\"emotion\": strength}})")


class Steering:
    """Context manager: install the feature and emotion steering of one request."""

    def __init__(self, model, opt: dict):
        self.model, self.opt, self.ctx = model, opt or {}, []

    def __enter__(self):
        import emotions as EM
        import features as FX
        dev = next(self.model.parameters()).device
        f = self.opt.get("feature")
        if f:
            sae, info, maxes, _, _ = load_dict(f["sae"], dev)
            setv = {int(k): float(v) for k, v in f["set"].items()}
            ids = torch.tensor(list(setv), device=dev)
            vals = torch.tensor([setv[i] for i in setv], device=dev) * maxes[ids]
            site = info.get("target_site") or info["site"]
            if f.get("mode", "clamp") == "add":
                dirs = (sae.W_dec.detach()[ids] / sae.out_scale)                       # [k, d]
                delta = (vals[:, None] * dirs).sum(0)

                def fn(x):
                    return x + delta.to(x.dtype)
            else:
                def fn(x):
                    xf = x.float()
                    a = sae.encode(xf) if not info.get("target_site") else None
                    if a is None:                                  # transcoders: clamp needs the MLP input
                        return x
                    rec = sae.decode(a) / sae.out_scale
                    a2 = a.clone()
                    a2[..., ids] = vals
                    return (sae.decode(a2) / sae.out_scale + (xf - rec)).to(x.dtype)
            if f.get("mode", "clamp") != "add" and info.get("target_site"):
                raise ValueError("clamping a transcoder's features needs feature_circuits.py intervene; "
                                 "use mode 'add' here")
            s = FX.Splice(self.model, site, int(info["layer"]), fn)
            s.__enter__()
            self.ctx.append(s)
        e = self.opt.get("emotion")
        if e:
            vec = load_emotions(e["file"], dev)
            st = EM.Steer(self.model, vec, {k: float(v) for k, v in e["mix"].items()},
                          [int(x) for x in e["layers"]] if e.get("layers") else None)
            st.__enter__()
            self.ctx.append(st)
        return self

    def __exit__(self, *exc):
        for c in reversed(self.ctx):
            c.__exit__(*exc)
        self.ctx = []


# ====================================================================== endpoints

def _ids(lab, req: dict) -> tuple[list[int], int]:
    tok = lab.tok
    if req.get("reply") is not None or req.get("messages"):
        from logit_lens import encode
        return encode(tok, None, req.get("messages") or [], req.get("reply") or "")
    if req.get("prompt") and req.get("chat", True) and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": req["prompt"]}], tokenize=False,
                                       add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"], 0
    text = req.get("text") or req.get("prompt") or ""
    if not text:
        raise ValueError("give text, prompt, or messages with a reply")
    return tok(text)["input_ids"], 0


def inspect(lab, req: dict) -> dict:
    import features as FX
    model, tok = lab.model, lab.tok
    ids, start = _ids(lab, req)
    ids = ids[: int(req.get("max_tokens", 1024))]
    dev = lab.dev
    top = int(req.get("top", 5))
    dicts = [load_dict(p, dev) + (p,) for p in (req.get("sae") or [])]
    points = [((d[1].get("target_site") or d[1]["site"]) if not d[1].get("target_site") else d[1]["site"],
               int(d[1]["layer"])) for d in dicts]
    out: dict = {"tokens": [tok.decode([t]) for t in ids], "start": start, "dicts": []}
    with torch.no_grad(), FX.Capture(model, points) as cap:
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    for (sae, info, maxes, labels, dense, path), pt in zip(dicts, points):
        f = sae.encode(cap.grab[pt][0].float())                                   # [T, n]
        if not req.get("dense", False):
            f[:, dense] = 0      # active on over a fifth of all tokens: they rarely mean one thing
        if req.get("skip_first", True) and f.shape[0] > 1:
            f[0] = 0             # the first position is an outlier the dictionary was not trained on
        rel = f / maxes.clamp(min=1e-6)
        v, i = rel.topk(min(top, f.shape[1]), dim=1)
        rows = []
        for t in range(f.shape[0]):
            rows.append([[int(ii), round(float(f[t, ii]), 3), round(float(vv), 3), labels.get(int(ii), "")]
                         for vv, ii in zip(v[t], i[t]) if vv > 0])
        tot = (f > 0).float().sum(0)
        common = tot.topk(min(20, f.shape[1]))
        out["dicts"].append({"path": path, "name": Path(path).name, "site": info["site"], "layer": info["layer"],
                             "per_token": rows,
                             "most_active": [[int(ii), int(c), labels.get(int(ii), "")]
                                             for c, ii in zip(common.values, common.indices) if c > 0]})
    if req.get("emotions"):
        import emotions as EM
        vec = load_emotions(req["emotions"], dev)
        out["emotions"] = EM.trace(model, tok, vec, ids, req.get("emotion_layer"), int(req.get("emotion_top", 3)))
    if req.get("lens", False):
        from logit_lens import lens
        out["lens"] = lens(model, tok, ids, max(1, start))
    return out


def compare(lab, req: dict) -> dict:
    msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt") or ""}]
    steer = req.get("steer") or {}
    validate(steer, "steer")
    mt = int(req.get("max_tokens") or 120)
    base = lab.generate(msgs, req.get("variant") or next(iter(lab.variants)), mt)
    steered = lab.generate(msgs, req.get("variant") or next(iter(lab.variants)), mt, extra=steer)
    return {"plain": base["text"], "steered": steered["text"], "steer": steer}


class _Eager:
    """Temporarily run attention eagerly (the graph freezes its patterns)."""

    def __init__(self, model):
        self.model, self.old = model, None

    def __enter__(self):
        cfg = self.model.config
        self.old = getattr(cfg, "_attn_implementation", None)
        for c in {cfg, getattr(cfg, "text_config", cfg)}:
            c._attn_implementation = "eager"

    def __exit__(self, *exc):
        cfg = self.model.config
        for c in {cfg, getattr(cfg, "text_config", cfg)}:
            c._attn_implementation = self.old


def graph(lab, req: dict) -> dict:
    import feature_circuits as FC
    tcs = FC.load_transcoders(req.get("transcoders") or [], lab.dev)
    with _Eager(lab.model), torch.enable_grad():
        r = FC.build_graph(lab.model, lab.tok, tcs, req["prompt"], bool(req.get("chat")), req.get("targets"),
                           int(req.get("max_logits", 10)), float(req.get("logit_prob", 0.95)),
                           int(req.get("max_features", 300)), float(req.get("node_threshold", 0.8)),
                           float(req.get("edge_threshold", 0.98)), FC._labels_for(tcs), log=lambda s: None)
    r["model"] = lab.model_id
    if req.get("save"):
        p = Path(req["save"]).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, separators=(",", ":")))
    return r


def intervene(lab, req: dict) -> dict:
    import feature_circuits as FC
    tcs = FC.load_transcoders(req.get("transcoders") or [], lab.dev)
    changes = req.get("changes") or FC.parse_set(req.get("set") or [])
    watch = [tuple(int(x) for x in w) for w in req.get("watch") or []]
    return FC.intervene(lab.model, lab.tok, tcs, req["prompt"], changes, bool(req.get("chat")),
                        int(req.get("top", 10)), watch or None)


def attribute(lab, req: dict) -> dict:
    import feature_circuits as FC
    sae, info, _, labels, _ = load_dict(req["sae"], lab.dev)
    with torch.enable_grad():
        r = FC.output_attribution(lab.model, lab.tok, sae, info, req["prompt"], req["token"], req.get("baseline"),
                                  bool(req.get("chat")), int(req.get("top", 30)), int(req.get("ablate", 10)),
                                  bool(req.get("skip_first", True)))
    for row in r["features"]:
        row["label"] = labels.get(row["feature"], "")
    return r


def sweep(lab, req: dict) -> dict:
    vec = load_emotions(req["emotions"], lab.dev)
    tok, model = lab.tok, lab.model
    j = vec.j(req.get("layer"))
    l = vec.layers[j]
    names = req.get("names") or vec.names
    rows = []
    for v in req.get("values") or []:
        text = str(req["template"]).replace("{x}", str(v))
        if getattr(tok, "chat_template", None) and req.get("chat", True):
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
        else:
            ids = tok(text)["input_ids"]
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids], device=lab.dev), output_hidden_states=True,
                       use_cache=False).hidden_states
        h = hs[l + 1][0, -1].float()
        z = ((h - vec.center[j].to(h.device)) @ vec.U[j].to(h.device).t()).cpu()
        z = (z - vec.mu[j]) / vec.sd[j]
        rows.append({"value": v, "z": {e: round(float(z[vec.names.index(e)]), 3) for e in names if e in vec.names}})
    return {"layer": l, "template": req["template"], "rows": rows}


def info(lab) -> dict:
    from concept_edit import decoder_layers
    p = next(lab.model.parameters())
    return {"model": lab.model_id, "layers": len(decoder_layers(lab.model)), "device": str(p.device),
            "dtype": str(p.dtype).replace("torch.", ""), "variants": list(lab.variants)}


ROUTES = {"inspect": inspect, "compare": compare, "graph": graph, "intervene": intervene, "attribute": attribute,
          "sweep": sweep}


def handle(lab, path: str, req: dict) -> tuple[int, dict]:
    name = path.strip("/").split("/", 1)[-1]
    if name == "info":
        return 200, info(lab)
    fn = ROUTES.get(name)
    if fn is None:
        return 404, {"error": f"no endpoint /ns/{name}; have info, {', '.join(ROUTES)}"}
    try:
        if name == "compare":
            return 200, fn(lab, req)                    # takes the lab lock itself, per generation
        with lab.lock:
            return 200, fn(lab, req)
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return 400, {"error": f"{type(e).__name__}: {e}"}
    except SystemExit as e:
        return 400, {"error": str(e)}
