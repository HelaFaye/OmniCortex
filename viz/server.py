#!/usr/bin/env python3
"""
OmniCortex: watch the OmniCortex pipeline.

Deliberately not Qt6 or Electron. You want to watch a sweep running on another
machine from this one -- that is a web page. Standard library HTTP server plus
one HTML file: no PySide6 to install per machine, no bundled browser per app, no
build step, and it works over the LAN for free.

    python viz/server.py --root . --port 7860
    # then open http://localhost:7860, or http://<this-box>:7860 from elsewhere

Three panels:

  Pipeline     which stages have produced output, and what is missing. Reads
               the filesystem, so it is honest about where you actually are.
  Layer map    where the H-Neurons sit, from h_neurons.json or classifier.npz.
  Alpha sweep  drives a llama-server /lora-adapters sweep and streams results.

Binds to localhost by default. --host 0.0.0.0 exposes it to your LAN, which is
the point when the GPU is on another machine, but there is no authentication --
do not put it on an untrusted network.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ns_security as sec  # noqa: E402
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "."


# ----------------------------------------------------------------- inspection

def _count(path, suffix=".npy"):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith(suffix))


def pipeline_state():
    """Report from the filesystem rather than a status file, so it cannot drift
    out of sync with what actually exists on disk."""
    d = lambda *p: os.path.join(ROOT, *p)
    acts = d("data", "activations")
    stages = [
        {"n": 1, "name": "Collect responses",
         "path": d("data", "consistency_samples.jsonl"),
         "detail": "sample_num generations per question, consistency filtered"},
        {"n": 2, "name": "Tag answer tokens",
         "path": d("data", "answer_tokens.jsonl"), "detail": ""},
        {"n": 3, "name": "Balanced split",
         "path": d("data", "train_qids.json"), "detail": ""},
        {"n": 4, "name": "Extract activations",
         "path": os.path.join(acts, "neuron_index.json"),
         "detail": "llama-cett-dump, two phases"},
        {"n": 5, "name": "Train classifier",
         "path": d("models", "h_neurons.json"), "detail": ""},
        {"n": 7, "name": "Tuned profile",
         "path": d("profiles"), "detail": "tune_scale_server.py"},
        {"n": 9, "name": "LoRA adapter",
         "path": d("adapters"), "detail": "export_lora.py --gguf"},
    ]
    out = []
    for s in stages:
        exists = os.path.exists(s["path"])
        extra = ""
        if s["n"] == 1 and exists:
            try:
                with open(s["path"], encoding="utf-8") as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                t = sum(1 for r in rows
                        if next(iter(r.values())).get("judge") == "true")
                fa = len(rows) - t
                extra = f"{len(rows)} kept, {t} correct / {fa} hallucinated"
                if min(t, fa) < 200:
                    extra += f" -- {min(t, fa)} balanced pairs, thin"
            except Exception as e:
                extra = f"unreadable: {e}"
        elif s["n"] == 4 and exists:
            n = _count(os.path.join(acts, "answer_tokens"))
            extra = f"{n} samples extracted"
        elif s["n"] == 5 and exists:
            try:
                with open(s["path"]) as f:
                    h = json.load(f)
                pct = h["total"] / (h["n_layers"] * h["n_neurons"]) * 100
                extra = f"{h['total']} neurons ({pct:.3f}% of all)"
                if pct > 1.0:
                    extra += " -- high for L1, consider lowering C"
            except Exception:
                pass
        elif s["n"] in (7, 9) and exists:
            n = sum(len(fs) for _, _, fs in os.walk(s["path"]))
            exists = n > 0
            extra = f"{n} file(s)"
        out.append({"n": s["n"], "name": s["name"], "done": bool(exists),
                    "detail": s["detail"], "extra": extra})
    return out


def layer_map():
    p = os.path.join(ROOT, "models", "h_neurons.json")
    if os.path.exists(p):
        with open(p) as f:
            h = json.load(f)
        counts = [0] * h["n_layers"]
        for k, v in h["by_layer"].items():
            counts[int(k)] = len(v)
        return {"ok": True, "counts": counts, "n_neurons": h["n_neurons"],
                "total": h["total"]}
    return {"ok": False, "reason": "no models/h_neurons.json yet -- "
                                   "run stages 1 through 5 first"}


def sweep_once(base_url, adapter_id, alpha, questions, max_tokens):
    """One alpha point. Returns per-question verdicts."""
    if alpha is not None:
        req = urllib.request.Request(
            f"{base_url}/lora-adapters", method="POST",
            data=json.dumps([{"id": adapter_id, "scale": alpha}]).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()

    verdicts = []
    for q in questions:
        body = {"messages": [{"role": "user", "content": q["question"]}],
                "temperature": 0.0, "max_tokens": max_tokens}
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions", method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            txt = r["choices"][0]["message"].get("content") or ""
        except Exception:
            verdicts.append("error")
            continue
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1]
        low = txt.strip().lower()
        if not low or any(m in low for m in
                          ("i don't know", "i'm not sure", "not certain",
                           "unable to answer", "i do not know")):
            verdicts.append("abstained")
            continue
        norm = " ".join(re.sub(r"[^a-z0-9\s]", " ", low).split())
        hit = any(" ".join(re.sub(r"[^a-z0-9\s]", " ", a.lower()).split()) in norm
                  for a in q["aliases"] if a)
        verdicts.append("correct" if hit else "wrong")
    return verdicts


def load_questions(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = next(iter(json.loads(line).values()))
            if d.get("aliases"):
                out.append({"question": d["question"], "aliases": d["aliases"]})
            if len(out) >= n:
                break
    return out


# --------------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/pipeline":
            return self._send(200, json.dumps(pipeline_state()))
        if self.path == "/api/layers":
            return self._send(200, json.dumps(layer_map()))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/sweep":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        eval_path = os.path.join(ROOT, "data", "consistency_samples.jsonl")
        if not os.path.exists(eval_path):
            return self._send(400, json.dumps(
                {"error": "no data/consistency_samples.jsonl -- run stage 1"}))
        qs = load_questions(eval_path, int(req.get("n_eval", 25)))
        if not qs:
            return self._send(400, json.dumps(
                {"error": "no gold aliases in the eval file"}))
        try:
            v = sweep_once(req["base_url"].rstrip("/"),
                           int(req.get("adapter_id", 0)),
                           req.get("alpha"), qs, int(req.get("max_tokens", 512)))
        except urllib.error.URLError as e:
            return self._send(502, json.dumps(
                {"error": f"cannot reach the server: {e.reason}. "
                          "llama-server with --lora-scaled is required."}))
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}))
        counts = {k: v.count(k) for k in
                  ("correct", "abstained", "wrong", "error")}
        self._send(200, json.dumps({"alpha": req.get("alpha"),
                                    "n": len(v), "counts": counts,
                                    "verdicts": v}))


PAGE = r"""<!DOCTYPE html><meta charset="utf-8"><title>OmniCortex</title>
<style>
:root{--bg:#faf9f7;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--ok:#2f7d4f;--no:#b23c2e;--warn:#a8701c}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:1.2rem 1.6rem;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:17px;font-weight:600}
.sub{color:var(--mut);font-size:13px;margin-top:.2rem}
main{max-width:1000px;margin:0 auto;padding:1.4rem 1.6rem}
section{margin-bottom:2.2rem}
h2{font-size:14px;font-weight:600;margin:0 0 .7rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.stage{display:flex;gap:.7rem;padding:.5rem .7rem;border:1px solid var(--line);border-radius:7px;margin-bottom:.4rem;background:#fff;align-items:baseline}
.dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;margin-top:.45rem}
.done .dot{background:var(--ok)}.todo .dot{background:var(--line);border:1px solid #cfcec8}
.nm{font-weight:500;min-width:170px}
.dt{color:var(--mut);font-size:13px}
.ex{margin-left:auto;font-size:13px;color:var(--mut);font-family:ui-monospace,monospace}
.warnx{color:var(--warn)}
.bars{font-family:ui-monospace,monospace;font-size:12px;line-height:1.35}
.bar{display:inline-block;height:10px;background:#c7d6ea;vertical-align:middle;border-radius:2px}
label{display:block;font-size:13px;color:var(--mut);margin:.5rem 0 .15rem}
input{font:13px ui-monospace,monospace;padding:.4rem .5rem;border:1px solid var(--line);border-radius:6px;width:100%;background:#fff}
.row{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:.7rem}
button{margin-top:.9rem;font:500 13px ui-sans-serif,system-ui;padding:.5rem 1rem;border:1px solid var(--line);background:#fff;border-radius:7px;cursor:pointer}
button:hover{background:#f2f1ec}button:disabled{opacity:.5;cursor:default}
table{border-collapse:collapse;width:100%;margin-top:1rem;font-size:13px}
th,td{text-align:right;padding:.4rem .6rem;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;font-family:ui-monospace,monospace}
.msg{margin-top:.8rem;padding:.6rem .8rem;border-radius:7px;font-size:13px}
.err{background:#fdf0ee;color:var(--no)}
.note{color:var(--mut);font-size:13px;margin-top:.5rem}
</style>
<header><h1>OmniCortex</h1>
<div class="sub">OmniCortex pipeline state, layer map, and live suppression sweeps</div></header>
<main>
<section><h2>Pipeline</h2><div id="pipe">loading…</div></section>
<section><h2>H-Neurons per layer</h2><div id="layers" class="bars">loading…</div></section>
<section><h2>Alpha sweep</h2>
<div class="row">
<div><label>llama-server</label><input id="url" value="http://127.0.0.1:8080"></div>
<div><label>alphas</label><input id="alphas" value="0, 0.5, 1.0"></div>
<div><label>questions</label><input id="n" value="25"></div>
<div><label>adapter id</label><input id="aid" value="0"></div>
</div>
<button id="go">Run sweep</button>
<div id="out"></div>
<div class="note">Needs llama-server started with <code>--lora-scaled</code>;
LM Studio does not expose <code>/lora-adapters</code>. Effective neuron scale is
1 + &alpha;(s&minus;1), so &alpha;=0 is the unmodified model. This is a live
readout, not a measurement &mdash; 25 questions is for watching, not reporting.</div>
</section>
</main>
<script>
const $=s=>document.querySelector(s);
async function pipeline(){
  const d=await (await fetch('/api/pipeline')).json();
  $('#pipe').innerHTML=d.map(s=>`<div class="stage ${s.done?'done':'todo'}">
    <span class="dot"></span><span class="nm">${s.n}. ${s.name}</span>
    <span class="dt">${s.detail||''}</span>
    <span class="ex ${/thin|high/.test(s.extra)?'warnx':''}">${s.extra||(s.done?'':'not started')}</span>
  </div>`).join('');
}
async function layers(){
  const d=await (await fetch('/api/layers')).json();
  if(!d.ok){$('#layers').innerHTML=`<span style="color:var(--mut)">${d.reason}</span>`;return;}
  const peak=Math.max(...d.counts,1);
  $('#layers').innerHTML=d.counts.map((c,i)=>
    `L${String(i).padEnd(3)} <span class="bar" style="width:${Math.round(560*c/peak)}px"></span> ${c}`
  ).join('<br>')+`<div class="note">${d.total} of ${d.counts.length*d.n_neurons} neurons</div>`;
}
$('#go').onclick=async()=>{
  const btn=$('#go'); btn.disabled=true;
  const alphas=$('#alphas').value.split(',').map(s=>parseFloat(s.trim())).filter(x=>!isNaN(x));
  const rows=[];
  $('#out').innerHTML='<div class="msg">running…</div>';
  for(const a of alphas){
    let r;
    try{
      r=await (await fetch('/api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({base_url:$('#url').value,alpha:a,n_eval:+$('#n').value,adapter_id:+$('#aid').value})})).json();
    }catch(e){ r={error:String(e)}; }
    if(r.error){$('#out').innerHTML=`<div class="msg err">${r.error}</div>`;btn.disabled=false;return;}
    rows.push(r); render(rows);
  }
  btn.disabled=false;
};
function render(rows){
  const pc=(v,n)=>n?((100*v/n).toFixed(1)+'%'):'—';
  $('#out').innerHTML=`<table><tr><th>alpha</th><th>correct</th><th>abstained</th><th>wrong</th><th>error</th></tr>`+
   rows.map(r=>`<tr><td>${r.alpha}</td><td>${pc(r.counts.correct,r.n)}</td>
   <td>${pc(r.counts.abstained,r.n)}</td><td>${pc(r.counts.wrong,r.n)}</td>
   <td>${r.counts.error||0}</td></tr>`).join('')+`</table>`;
}
pipeline();layers();setInterval(pipeline,5000);
</script>"""


def main():
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="repo root to inspect")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--allow-unauthenticated", action="store_true", help="permit a non-loopback bind (no auth)")
    a = p.parse_args()
    sec.loopback_only(a.host, "the dashboard", a.allow_unauthenticated)
    ROOT = os.path.abspath(a.root)
    print(f"OmniCortex on http://{a.host}:{a.port}  (root {ROOT})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
