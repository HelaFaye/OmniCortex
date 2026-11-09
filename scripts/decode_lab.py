#!/usr/bin/env python3
"""
Decoding lab: one Hugging Face model, several decoding variants, behind an
OpenAI-compatible API, so TestQA and the retrain gate can compare them.

    python scripts/decode_lab.py --model org/m --port 8090 \\
        --variant plain='{}' \\
        --variant sled='{"sled": {"alpha": 2, "k": 10}}' \\
        --variant steer='{"steer": {"vectors": "runs/dk/directions.pt", "layer": 14, "alpha": 4, "adapt": "spread"}}'
    python scripts/testqa.py --endpoint plain=http://127.0.0.1:8090/v1@plain \\
        --endpoint sled=http://127.0.0.1:8090/v1@sled --out runs/sled.json
    python scripts/retrain_gate.py --results runs/sled.json --before plain --after sled

The model stays in memory once; the request's "model" names the variant.
Options a variant may set (all off unless set):

sled     Self Logits Evolution Decoding (Zhang et al., NeurIPS 2024). Each step,
         the earlier layers' logit-lens logits are compared with the final
         ones over the final top-k tokens:
           m_i(n) = max(0, cos(logits_n - logits_N, p_N - e_i))^2
         gives each layer's estimate of the "latent" distribution
         (m normalised over i), the layers are averaged weighted by their
         total m, and the final logits take one step toward it:
           logits_N[i] -= alpha * (p_N[i] - P_latent[i])   for i in the top k;
         tokens outside the top k are dropped.
         {"alpha": evolution rate (2), "k": evolution scale (10),
          "layers": "all" | "even" | [indices] (default "even": every second
          layer below the last), "temperature": for p_N (1)}
         This follows the paper's description; its reference code differs in
         details, so measure it rather than assume its published gains.
steer    Add alpha * v to the residual stream after one layer, at every step.
         v comes from concept_edit.py's directions.pt (or a .npy / .pt vector).
         {"vectors": path, "layer": l, "alpha": strength (negative removes),
          "direction": which column of a multi-direction basis (0),
          "adapt": "fixed" | "spread" | "attention",
          "target": the uncertainty at which the full alpha applies (0.5),
          "min": 0, "max": 2: the bounds on the scale}
         With adapt, the strength follows the model's uncertainty on the
         previous token: alpha_t = alpha * clip(u / target, min, max), where u
         is the output spread (entropy of the top 5 plus the rest, over log 6)
         or the mean entropy of the last layer's attention at the current
         position over log(context length). Steering then acts when the model
         hesitates and stays out of the way when it is sure.

rp       Steering through a low-dimensional random projection, fitted by
         rp_steer.py fit: on each fitted layer the MLP output is pulled back
         toward the right answers' side when it drifts past it.
         {"file": rp.pt, "alpha": 1.0, "mode": "direction" | "subspace",
          "layers": "all" | [indices]}
         The reply reports the drift per step ("rp_drift"), so you can see
         where the model leaned toward its wrong-answer pattern.

feature  clamp or add feature-dictionary features (features.py) at every token
emotion  add an emotion-vector mix (emotions.py)
         See feature_lab.py. Both can also be given per request under
         "omnicortex": {...}, and feature_lab's /ns/* endpoints (inspect,
         compare, graph, intervene, attribute, sweep) serve Studio's Features page.

Watching (any variant; scripts/live_watch.py):
  --watch-cett, --watch-emotions DIR, --watch-sae DIR (repeatable) record, for
  every generated token, each MLP neuron's CETT, the emotion activations and the
  top features. --publish URL sends each token's frame to the live viewer
  (viz/stream.py) as it is generated; --record DIR saves each reply as a trace
  session for the 3D replay, the timeline and the Godot client. The reply's
  "decode_lab" field carries the per-token emotions and features. A request can
  turn either off with "watch": false or "record": false.

Greedy unless the request sets temperature > 0. One request at a time.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOG6 = math.log(6)


def spread_of(logits):
    from logit_lens import spread_top5
    import torch
    return float(spread_top5(torch.log_softmax(logits.float(), -1))[0])


def sled_logits(final, early: list, alpha: float = 2.0, k: int = 10, temperature: float = 1.0):
    """final [V] logits, early: list of [V] logit-lens logits -> evolved logits [V]
    (-inf outside the final top k)."""
    import torch
    top = final.topk(k).indices
    zN = final[top].float()
    pN = torch.softmax(zN / temperature, -1)
    eye = torch.eye(k, dtype=zN.dtype, device=zN.device)
    grads = pN[None, :] - eye                                   # row i: p_N - e_i
    num, wsum = torch.zeros(k, dtype=zN.dtype, device=zN.device), 0.0
    for z in early:
        d = z[top].float() - zN                                 # logits_n - logits_N
        cos = torch.nn.functional.cosine_similarity(d[None, :].expand(k, -1), grads, dim=-1)
        m = cos.clamp(min=0) ** 2
        s = float(m.sum())
        if s > 0:
            num += m          # = s * (m / s): the layer's estimate weighted by its total
            wsum += s
    out = torch.full_like(final, float("-inf"), dtype=torch.float32)
    if wsum == 0:
        out[top] = zN
        return out
    latent = num / wsum
    out[top] = zN - alpha * (pN - latent)
    return out


def load_vector(path: str, direction: int = 0, layer: int | None = None):
    import torch
    p = Path(path)
    if p.suffix == ".npy":
        import numpy as np
        v = torch.tensor(np.load(p)).float()
    else:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("basis", obj.get("directions", next(iter(obj.values()))))
        v = torch.as_tensor(obj).float()
    if v.dim() == 3:                       # concept_edit: [L+1, d, k], index l+1 = after layer l
        v = v[layer + 1 if layer is not None else v.shape[0] // 2]
    if v.dim() == 2:
        v = v[:, direction] if v.shape[0] > v.shape[1] else v[direction]
    return v / v.norm().clamp(min=1e-8)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Lab:
    def __init__(self, model_id: str, variants: dict, dtype: str = "auto", attn_eager: bool = False,
                 watch: dict | None = None, publisher=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from logit_lens import final_norm
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
        if attn_eager or any((v.get("steer") or {}).get("adapt") == "attention" for v in variants.values()):
            kw["attn_implementation"] = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.dev = next(self.model.parameters()).device
        self.norm, self.head = final_norm(self.model), self.model.get_output_embeddings()
        from concept_edit import decoder_layers
        self.layers = decoder_layers(self.model)
        self.variants = variants
        self.vectors = {}
        for name, v in variants.items():
            st = v.get("steer")
            if st:
                self.vectors[name] = load_vector(st["vectors"], int(st.get("direction", 0)),
                                                 int(st["layer"])).to(self.dev)
        self.rp = {}
        for name, v in variants.items():
            r = v.get("rp")
            if r:
                from rp_steer import Steerer
                self.rp[name] = Steerer(r["file"], float(r.get("alpha", 1.0)), r.get("mode", "direction"),
                                        r.get("layers"), self.dev)
        self.lock = threading.Lock()
        self.model_id = model_id
        self.watch = dict(watch or {})
        self.publisher = publisher

    def _early(self, hidden_states, which) -> list:
        L = len(hidden_states) - 1
        if which == "all":
            idx = list(range(1, L))
        elif which in (None, "even"):
            idx = list(range(2, L, 2))
        else:
            idx = [int(i) for i in which if 0 < int(i) < L]
        return [self.head(self.norm(hidden_states[i][0, -1])) for i in idx]

    def generate(self, messages: list, variant: str, max_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, stop: list | None = None, on_text=None, extra: dict | None = None) -> dict:
        import torch
        import feature_lab
        if variant not in self.variants:
            raise KeyError(f"no variant {variant!r}; have {', '.join(self.variants)}")
        opt = self.variants[variant]
        steer_opt = {k: v for k, v in {**opt, **(extra or {})}.items() if k in ("feature", "emotion")}
        sled, st = opt.get("sled"), opt.get("steer")
        tok = self.tok
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        ids = tok(text, return_tensors="pt", add_special_tokens=not getattr(tok, "chat_template", None))["input_ids"]
        ids = ids.to(self.dev)
        scale = {"a": 0.0}
        hook = None
        if st:
            v, alpha = self.vectors[variant], float(st.get("alpha", 1.0))
            scale["a"] = alpha if st.get("adapt", "fixed") == "fixed" else alpha * float(st.get("min", 0.0))

            def add(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + scale["a"] * v.to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hook = self.layers[int(st["layer"])].register_forward_hook(add)
        rp = self.rp.get(variant)
        if rp is not None:
            rp.attach(self.layers)
        drift = []
        want_att = bool(st and st.get("adapt") == "attention")
        out_ids, past, cur, alphas, spreads = [], None, ids, [], []
        eos = {tok.eos_token_id} | set(getattr(self.model.generation_config, "eos_token_id", None) or []
                                       if isinstance(getattr(self.model.generation_config, "eos_token_id", None), list)
                                       else [])
        text_out, t0 = "", time.time()
        steering = feature_lab.Steering(self.model, steer_opt)
        watch_opt = (extra or {}).get("watch", True)
        watcher = self.watcher() if watch_opt is not False else None
        try:
            with torch.no_grad(), steering, (watcher if watcher is not None else _Null()):
                for _ in range(max_tokens):
                    o = self.model(input_ids=cur, past_key_values=past, use_cache=True,
                                   output_hidden_states=bool(sled), output_attentions=want_att)
                    past = o.past_key_values
                    logits = o.logits[0, -1].float()
                    if sled:
                        logits = sled_logits(logits, self._early(o.hidden_states, sled.get("layers")),
                                             float(sled.get("alpha", 2.0)), int(sled.get("k", 10)),
                                             float(sled.get("temperature", 1.0)))
                    u = spread_of(torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e9)))
                    spreads.append(round(u, 4))
                    if st and st.get("adapt") in ("spread", "attention"):
                        if st["adapt"] == "attention":
                            a = o.attentions[-1][0, :, -1].float().clamp(min=1e-12)       # heads x context
                            n = a.shape[-1]
                            u_ctx = float((-(a * a.log()).sum(-1)).mean() / math.log(max(2, n)))
                        else:
                            u_ctx = u
                        k = min(float(st.get("max", 2.0)), max(float(st.get("min", 0.0)),
                                                                u_ctx / float(st.get("target", 0.5))))
                        scale["a"] = float(st.get("alpha", 1.0)) * k
                    alphas.append(round(scale["a"], 4))
                    if rp is not None:
                        drift.append(round(rp.flagged(), 4))
                    if temperature and temperature > 0:
                        probs = torch.softmax(logits / temperature, -1)
                        if top_p < 1:
                            sp, si = probs.sort(descending=True)
                            keep = sp.cumsum(-1) - sp <= top_p
                            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
                        nxt = int(torch.multinomial(probs / probs.sum(), 1))
                    else:
                        nxt = int(logits.argmax())
                    if nxt in eos:
                        break
                    out_ids.append(nxt)
                    new = tok.decode(out_ids, skip_special_tokens=True)
                    piece, text_out = new[len(text_out):], new
                    if watcher is not None:                      # the model's state when it chose this token
                        fr = watcher.step(tok.decode([nxt]), u)
                        if self.publisher is not None:
                            self.publisher.send(watcher.live_frame(fr, self.publisher.tier))
                    if on_text and piece:
                        on_text(piece)
                    if stop and any(s and s in text_out for s in stop):
                        text_out = text_out[:min(text_out.index(s) for s in stop if s and s in text_out)]
                        break
                    cur = torch.tensor([[nxt]], device=self.dev)
        finally:
            if hook is not None:
                hook.remove()
            if rp is not None:
                rp.remove()
        out = {"text": text_out, "n_prompt": int(ids.shape[1]), "n_out": len(out_ids), "seconds": time.time() - t0,
               "spread": spreads, "alpha": alphas}
        if rp is not None:
            out["rp_drift"] = drift
        if watcher is not None:
            out["watch"] = watcher.summary()
            # only the lab's own --record folder is written to; a request can skip it with "record": false
            if self.watch.get("record") and (extra or {}).get("record", True) is not False:
                prompt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
                out["session"] = watcher.save(self.watch["record"], str(prompt), self.model_id, {"variant": variant})
        return out

    def watcher(self):
        """A live_watch.Watcher for this reply, when watching is configured (--watch-*)."""
        w = self.watch
        if not (w.get("cett") or w.get("emotions") or w.get("sae")):
            return None
        import feature_lab
        import live_watch
        if "_loaded" not in w:                           # vectors and dictionaries are read once, not per reply
            vec = feature_lab.load_emotions(w["emotions"], self.dev) if w.get("emotions") else None
            dicts = []
            for p in w.get("sae") or []:
                sae, info, maxes, labels, dense = feature_lab.load_dict(p, self.dev)
                dicts.append((sae, info, maxes, labels, dense, Path(p).name))
            w["_loaded"] = (vec, dicts)
        vec, dicts = w["_loaded"]
        return live_watch.Watcher(self.model, self.tok, cett=bool(w.get("cett")), bins=int(w.get("bins", 256)),
                                  emotions=vec, dicts=dicts)


def make_handler(lab: Lab):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "decode_lab",
                                                                     "options": v} for n, v in lab.variants.items()]})
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/ns/info":
                import feature_lab
                return self._json(*feature_lab.handle(lab, self.path, {}))
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/ns/"):
                import feature_lab
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except ValueError:
                    return self._json(400, {"error": "body is not JSON"})
                return self._json(*feature_lab.handle(lab, self.path, req))
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                variant = req.get("model") or next(iter(lab.variants))
                if variant not in lab.variants:
                    return self._json(404, {"error": {"message": f"no variant {variant!r}"}})
                msgs = [{"role": m.get("role", "user"), "content": m["content"] if isinstance(m.get("content"), str)
                         else " ".join(p.get("text", "") for p in m.get("content") or [] if isinstance(p, dict))}
                        for m in req.get("messages") or []]
                stop = req.get("stop")
                stop = [stop] if isinstance(stop, str) else (stop or None)
                extra = req.get("omnicortex") or None
                if extra:
                    import feature_lab
                    feature_lab.validate(extra, "omnicortex")
                args = dict(max_tokens=int(req.get("max_tokens") or 256), temperature=float(req.get("temperature") or 0),
                            top_p=float(req.get("top_p") or 1), stop=stop, extra=extra)
                cid = "decodelab-" + uuid.uuid4().hex[:10]
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()

                    def send(piece):
                        ch = {"id": cid, "object": "chat.completion.chunk", "model": variant,
                              "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                        self.wfile.flush()
                    with lab.lock:
                        r = lab.generate(msgs, variant, on_text=send, **args)
                    end = {"id": cid, "object": "chat.completion.chunk", "model": variant,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": r["n_prompt"], "completion_tokens": r["n_out"]}}
                    self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
                    return
                with lab.lock:
                    r = lab.generate(msgs, variant, **args)
                return self._json(200, {
                    "id": cid, "object": "chat.completion", "created": int(time.time()), "model": variant,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": r["n_prompt"], "completion_tokens": r["n_out"],
                              "total_tokens": r["n_prompt"] + r["n_out"]},
                    "decode_lab": {"spread": r["spread"], "alpha": r["alpha"], **({"rp_drift": r["rp_drift"]}
                                                                                    if "rp_drift" in r else {}),
                                   **({"watch": r["watch"]} if "watch" in r else {}),
                                   **({"session": r["session"]} if r.get("session") else {}),
                                   "tokens_per_second": round(r["n_out"] / max(r["seconds"], 1e-6), 2)}})
            except (KeyError, ValueError, TypeError) as e:
                return self._json(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
    return H


def parse_variants(items: list) -> dict:
    out = {}
    for it in items or []:
        name, _, spec = it.partition("=")
        if not name or not spec:
            raise SystemExit(f"--variant {it!r}: use name='{{json}}'")
        opt = json.loads(spec)
        if set(opt) - {"sled", "steer", "rp", "feature", "emotion"}:
            raise SystemExit(f"--variant {name}: options are sled, steer, rp, feature and emotion")
        try:
            import feature_lab
            feature_lab.validate(opt, f"--variant {name}")
        except ValueError as e:
            raise SystemExit(str(e))
        if opt.get("steer") and not ("vectors" in opt["steer"] and "layer" in opt["steer"]):
            raise SystemExit(f"--variant {name}: steer needs vectors and layer")
        if opt.get("rp") and "file" not in opt["rp"]:
            raise SystemExit(f"--variant {name}: rp needs file (from rp_steer.py fit)")
        out[name] = opt
    return out or {"plain": {}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="HF id or folder")
    p.add_argument("--variant", action="append", default=[],
                   help="name='{json options}' (repeatable); default one variant 'plain' with nothing on")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--prompt", help="answer this once with every variant and exit (no server)")
    g = p.add_argument_group("watching (realtime views; docs/FEATURES.md)")
    g.add_argument("--watch-cett", action="store_true", help="CETT of every MLP neuron for each generated token")
    g.add_argument("--watch-bins", type=int, default=256, help="max-pool each layer's neurons to this many bins")
    g.add_argument("--watch-emotions", help="emotion vectors folder (emotions.py build): activations per token")
    g.add_argument("--watch-sae", action="append", help="feature dictionary folder: top features per token (repeatable)")
    g.add_argument("--publish", help="live viewer to send frames to as tokens are generated, e.g. http://127.0.0.1:7890")
    g.add_argument("--publish-token-file", help="the viewer's access token")
    g.add_argument("--publish-tier", choices=["sparse", "binned"], default="sparse")
    g.add_argument("--record", help="save every reply as a trace session in this folder (3D replay, timeline, Godot)")
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args(argv)
    variants = parse_variants(a.variant)
    watch = {"cett": a.watch_cett, "bins": a.watch_bins, "emotions": a.watch_emotions, "sae": a.watch_sae or [],
             "record": a.record}
    publisher = None
    if a.publish:
        import live_watch
        tok = Path(a.publish_token_file).read_text().strip() if a.publish_token_file else None
        publisher = live_watch.Publisher(a.publish, tok, a.publish_tier)
        if not (a.watch_cett or a.watch_emotions or a.watch_sae):
            watch["cett"] = True                       # publishing without anything to show would be empty
    lab = Lab(a.model, variants, a.dtype, watch=watch, publisher=publisher)
    if a.prompt:
        for name in variants:
            r = lab.generate([{"role": "user", "content": a.prompt}], name, a.max_tokens)
            al = f" · alpha {min(r['alpha']):.2f}..{max(r['alpha']):.2f}" if r["alpha"] and any(r["alpha"]) else ""
            dr = f" · rp drift on {sum(1 for x in r['rp_drift'] if x > 0)} of {len(r['rp_drift'])} steps" \
                if r.get("rp_drift") else ""
            print(f"[{name}] {r['text']!r}  ({r['n_out']} tokens, mean spread "
                  f"{sum(r['spread']) / max(1, len(r['spread'])):.2f}{al}{dr})")
        return 0
    import oc_security as sec
    sec.loopback_only(a.host, "decode_lab")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(lab))
    print(f"decode_lab: {a.model} as {', '.join(variants)} at http://{a.host}:{a.port}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
