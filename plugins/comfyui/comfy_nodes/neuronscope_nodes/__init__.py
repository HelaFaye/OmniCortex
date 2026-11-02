"""OmniCortex nodes for ComfyUI: chat through Studio (with its model routing), check a reply for
hallucination from the model's own neurons, and run a OmniCortex graph. Standard library only.

Studio's address comes from each node (default http://127.0.0.1:7870); its token from NS_STUDIO_TOKEN,
or ~/.neuronscope/studio.token if that exists. Installed by OmniCortex's ComfyUI add-on."""
import json
import os
import time
import urllib.request
from pathlib import Path


def _token():
    t = os.environ.get("NS_STUDIO_TOKEN")
    if t:
        return t
    p = Path.home() / ".neuronscope" / "studio.token"
    return p.read_text().strip() if p.exists() else None


def _call(base, method, path, body=None, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + _token()} if _token() else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except Exception:
            msg = str(e)
        raise RuntimeError(f"OmniCortex Studio: {msg}")


STUDIO = ("STRING", {"default": "http://127.0.0.1:7870"})


class NeuronScopeChat:
    """Ask one of your models through Studio. model 'auto' lets Studio pick by the prompt's subject."""
    CATEGORY = "OmniCortex"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning", "model")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"multiline": True}), "model": ("STRING", {"default": "auto"})},
                "optional": {"system": ("STRING", {"multiline": True, "default": ""}),
                             "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                             "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}), "studio": STUDIO}}

    def run(self, prompt, model, system="", temperature=0.7, max_tokens=1024, studio="http://127.0.0.1:7870"):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = _call(studio, "POST", "/v1/chat/completions", {"model": model, "messages": msgs, "stream": False,
                                                           "temperature": temperature, "max_tokens": max_tokens})
        m = (r.get("choices") or [{}])[0].get("message") or {}
        return (m.get("content") or "", m.get("reasoning_content") or "", r.get("model") or model)


class NeuronScopeCheck:
    """Each token's hallucination risk from the model's neurons (needs its H-Neuron classifier)."""
    CATEGORY = "OmniCortex"
    RETURN_TYPES = ("FLOAT", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("peak_risk", "flagged_text", "flagged", "view_3d")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reply": ("STRING", {"multiline": True}), "model": ("STRING", {"default": ""})},
                "optional": {"prompt": ("STRING", {"multiline": True, "default": ""}), "studio": STUDIO}}

    def run(self, reply, model, prompt="", studio="http://127.0.0.1:7870"):
        r = _call(studio, "POST", "/api/trace", {"model": model, "messages": [{"role": "user", "content": prompt}],
                                                 "text": reply})
        flagged = [r["pieces"][i] for i in r.get("flagged", [])]
        return (float(r.get("max") or 0.0), "".join(flagged), bool(flagged), studio.rstrip("/") + "/" + r.get("url", ""))


class NeuronScopeGraph:
    """Run a OmniCortex graph (Studio -> Graphs) and return one of its outputs."""
    CATEGORY = "OmniCortex"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "outputs_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"graph": ("STRING", {"default": "template:screen-and-verify"}),
                             "text": ("STRING", {"multiline": True})},
                "optional": {"input_name": ("STRING", {"default": "question"}),
                             "output_name": ("STRING", {"default": "result"}), "studio": STUDIO}}

    def run(self, graph, text, input_name="question", output_name="result", studio="http://127.0.0.1:7870"):
        r = _call(studio, "POST", "/api/graphs/run", {"name": graph, "inputs": {input_name: text}})
        for _ in range(1800):
            run = _call(studio, "GET", f"/api/graphs/runs/{r['id']}")
            if run.get("status") not in ("running", "queued"):
                break
            time.sleep(1)
        if run.get("status") == "waiting":
            raise RuntimeError(f"the graph waits for a person: answer it in Studio -> Graphs (run {r['id']})")
        if run.get("status") != "done":
            raise RuntimeError(f"the graph {run.get('status')}: {run.get('error')}")
        out = run.get("outputs") or {}
        v = out.get(output_name, "")
        return (v if isinstance(v, str) else json.dumps(v), json.dumps(out))


NODE_CLASS_MAPPINGS = {"NeuronScopeChat": NeuronScopeChat, "NeuronScopeCheck": NeuronScopeCheck,
                       "NeuronScopeGraph": NeuronScopeGraph}
NODE_DISPLAY_NAME_MAPPINGS = {"NeuronScopeChat": "OmniCortex: chat", "NeuronScopeCheck": "OmniCortex: hallucination check",
                              "NeuronScopeGraph": "OmniCortex: run a graph"}
