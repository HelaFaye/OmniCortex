#!/usr/bin/env python3
"""Browser GUI for adaptive scale tuning against a remote tuning worker.

    python viz/adaptive_tuning_lab.py --port 8800

The worker (scripts/tuning_worker.py) builds and evaluates candidates on the
machine that holds the model; this page only drives the coarse-to-fine search
and plots score against scale. See docs/LABS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_tuner import AdaptiveTuner, WorkerClient  # noqa: E402
import ns_security as sec  # noqa: E402

STATE: dict = {}
MAX_BODY = 64 * 1024

HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive Tuning</title><link rel="icon" href="data:,">
<style>
:root{--bg:#f6f7f9;--panel:#fff;--line:#d9dee6;--fg:#141a22;--mut:#5d6878;--acc:#2563eb;--ok:#0f7a4f;--bad:#c0262d;--grid:#e8ebf0;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--panel:#171b21;--line:#2a313b;--fg:#e6e9ee;--mut:#97a1ae;--acc:#6ea8ff;--ok:#5fcf8f;--bad:#ff8b7e;--grid:#222831;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
label{display:block;font-size:12px;color:var(--mut)}input{width:100%;margin-top:3px;padding:7px 8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--fg);font:13px ui-monospace,monospace}
input[type=checkbox]{width:auto}.wide{grid-column:span 2}@media(max-width:500px){.wide{grid-column:span 1}}
button{padding:8px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:#fff;font:600 14px system-ui;cursor:pointer}button:disabled{opacity:.5}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;border:1px solid var(--line);font-size:12px}
.pill.ok{color:var(--ok)}.pill.bad{color:var(--bad)}
svg{width:100%;height:auto;display:block}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500}
td.num{font-variant-numeric:tabular-nums}details summary{cursor:pointer;color:var(--mut)}pre{white-space:pre-wrap;font-size:12px;max-height:300px;overflow:auto}
</style>
<main>
<h1>Adaptive tuning</h1>
<div class="sub">Coarse-to-fine search for the suppression scale. Candidates are built and scored on the worker; only results come back here.</div>
<section class="card">
 <div class="grid">
  <label class="wide">Worker URL<input id="worker" placeholder="https://gpu-host:8799"></label>
  <label>Worker token<input id="token" type="password" autocomplete="off"></label>
  <label>CA file (self-signed TLS)<input id="cafile" placeholder="optional"></label>
  <label class="wide">State file (resume by reusing it)<input id="state" value="runs/adaptive-gui-state.json"></label>
  <label>Minimum scale<input id="min" type="number" step="0.01" value="0.1"></label>
  <label>Maximum scale<input id="max" type="number" step="0.01" value="0.9"></label>
  <label>Initial step<input id="initial_step" type="number" step="0.01" value="0.1"></label>
  <label>Stop at resolution<input id="resolution" type="number" step="0.001" value="0.01"></label>
  <label>Models per batch<input id="batch_size" type="number" min="1" value="2"></label>
  <label>Margin of error<input id="margin_of_error" type="number" step="0.005" value="0.03"></label>
  <label>Baseline score<input id="baseline_score" type="number" step="0.001" placeholder="optional"></label>
  <label>Max batches<input id="max_batches" type="number" min="1" value="40"></label>
 </div>
 <div class="row">
  <label style="display:flex;gap:6px;align-items:center"><input id="auto_delete" type="checkbox"> delete candidates clearly below baseline</label>
  <button id="start">Start</button><span id="status" class="pill">idle</span>
 </div>
</section>
<section class="card">
 <svg id="plot" viewBox="0 0 1000 320" role="img" aria-label="score by scale"></svg>
 <div id="best" class="sub" style="margin:8px 0 0"></div>
</section>
<section class="card">
 <table><thead><tr><th>scale</th><th>score</th><th>95% CI</th><th>status</th><th>model</th></tr></thead><tbody id="rows"><tr><td colspan="5" class="sub">No results yet.</td></tr></tbody></table>
 <details style="margin-top:10px"><summary>raw state</summary><pre id="raw"></pre></details>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const FIELDS=['worker','token','cafile','state','min','max','initial_step','resolution','batch_size','margin_of_error','baseline_score','max_batches'];
try{const saved=JSON.parse(localStorage.getItem('ns-adaptive')||'{}'); for(const k of FIELDS) if(k!=='token'&&saved[k]!==undefined) $(k).value=saved[k];}catch{}
$('start').onclick=async()=>{
  const b={}; for(const k of FIELDS) b[k]=$(k).value; b.auto_delete=$('auto_delete').checked;
  try{const keep={...b}; delete keep.token; localStorage.setItem('ns-adaptive',JSON.stringify(keep));}catch{}
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const d=await r.json(); if(!r.ok){ setStatus(d.error||'failed','bad'); return; }
  $('start').disabled=true; poll();
};
function setStatus(t,cls=''){ $('status').textContent=t; $('status').className='pill '+cls; }
async function poll(){
  const s=await (await fetch('/api/state')).json(); render(s);
  const live=['starting','running'].includes(s.status);
  $('start').disabled=live; if(live) setTimeout(poll,1000);
}
function render(s){
  setStatus(s.status+(s.error?': '+s.error:''), s.status==='failed'||s.status==='batch_failed'?'bad':(['margin_reached','resolution_reached'].includes(s.status)?'ok':''));
  $('raw').textContent=JSON.stringify(s,null,2);
  const rs=(s.results||[]).filter(r=>r.scale!==undefined).sort((a,b)=>a.scale-b.scale);
  $('rows').innerHTML=rs.length?rs.map(r=>`<tr><td class="num">${(+r.scale).toFixed(3)}</td><td class="num">${r.score!=null?(+r.score).toFixed(3):'–'}</td>`+
    `<td class="num">${r.lower_ci!=null?`${(+r.lower_ci).toFixed(3)}–${(+r.upper_ci).toFixed(3)}`:'–'}</td><td>${esc(r.status)}${r.error?' · '+esc(r.error).slice(0,80):''}</td>`+
    `<td>${esc((r.model||'').split('/').pop())}</td></tr>`).join(''):'<tr><td colspan="5" class="sub">No results yet.</td></tr>';
  const b=s.best; $('best').innerHTML=b?`best so far: scale <b>${(+b.scale).toFixed(3)}</b>, score <b>${(+b.score).toFixed(3)}</b>`+(s.next_interval?` · next interval ${s.next_interval.map(x=>(+x).toFixed(3)).join('–')}`:''):(s.next_interval?`next interval ${s.next_interval.map(x=>(+x).toFixed(3)).join('–')}`:'');
  plot(rs.filter(r=>r.score!=null), s);
}
function plot(rs,s){
  const W=1000,H=320,L=56,R=16,T=16,B=40,svg=$('plot'); const css=getComputedStyle(document.documentElement);
  const fg=css.getPropertyValue('--mut'),acc=css.getPropertyValue('--acc'),grid=css.getPropertyValue('--grid'),ok=css.getPropertyValue('--ok');
  const lo=+($('min').value||0), hi=+($('max').value||1);
  const xs=rs.map(r=>+r.scale), x0=Math.min(lo,...xs), x1=Math.max(hi,...xs);
  const ys=rs.flatMap(r=>[+r.score, r.lower_ci??+r.score, r.upper_ci??+r.score]);
  const base=$('baseline_score').value===''?null:+$('baseline_score').value; if(base!=null) ys.push(base);
  let y0=Math.min(...ys,1), y1=Math.max(...ys,0); if(!rs.length){y0=0;y1=1} const pad=(y1-y0)*0.1||0.05; y0-=pad; y1+=pad;
  const X=v=>L+(v-x0)/Math.max(1e-9,x1-x0)*(W-L-R), Y=v=>T+(1-(v-y0)/Math.max(1e-9,y1-y0))*(H-T-B);
  let g='';
  for(let i=0;i<=5;i++){ const v=y0+(y1-y0)*i/5; g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${grid}"/><text x="${L-8}" y="${Y(v)+4}" text-anchor="end" font-size="12" fill="${fg}">${v.toFixed(2)}</text>`; }
  for(let i=0;i<=5;i++){ const v=x0+(x1-x0)*i/5; g+=`<text x="${X(v)}" y="${H-B+20}" text-anchor="middle" font-size="12" fill="${fg}">${v.toFixed(2)}</text>`; }
  g+=`<text x="${(L+W-R)/2}" y="${H-4}" text-anchor="middle" font-size="12" fill="${fg}">scale (1 = unchanged)</text>`;
  if(base!=null) g+=`<line x1="${L}" x2="${W-R}" y1="${Y(base)}" y2="${Y(base)}" stroke="${fg}" stroke-dasharray="6 5"/><text x="${W-R}" y="${Y(base)-6}" text-anchor="end" font-size="12" fill="${fg}">baseline</text>`;
  if(s.next_interval){ const [a,b]=s.next_interval; g+=`<rect x="${X(a)}" y="${T}" width="${Math.max(1,X(b)-X(a))}" height="${H-T-B}" fill="${acc}" opacity="0.08"/>`; }
  for(const r of rs){ if(r.lower_ci!=null) g+=`<line x1="${X(r.scale)}" x2="${X(r.scale)}" y1="${Y(r.lower_ci)}" y2="${Y(r.upper_ci)}" stroke="${acc}" stroke-width="2" opacity="0.6"/>`; }
  if(rs.length>1) g+=`<polyline fill="none" stroke="${acc}" stroke-width="2" points="${rs.map(r=>X(r.scale)+','+Y(r.score)).join(' ')}"/>`;
  for(const r of rs){ const best=s.best&&+s.best.scale===+r.scale; g+=`<circle cx="${X(r.scale)}" cy="${Y(r.score)}" r="${best?7:5}" fill="${best?ok:acc}"><title>scale ${(+r.scale).toFixed(3)}: ${(+r.score).toFixed(3)}</title></circle>`; }
  if(!rs.length) g+=`<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="${fg}">results appear here as candidates finish</text>`;
  svg.innerHTML=g;
}
poll();
</script>"""


class H(BaseHTTPRequestHandler):
    def sendj(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path == "/api/state":
            self.sendj(STATE.get("state", {"status": "idle", "results": []}))
            return
        self.sendj({"error": "not-found"}, 404)

    def do_POST(self):
        if self.path != "/api/start":
            self.sendj({"error": "not-found"}, 404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        if n < 0 or n > MAX_BODY:
            self.sendj({"error": "body too large"}, 413)
            return
        try:
            d = json.loads(self.rfile.read(n))
            if not str(d.get("worker", "")).startswith(("http://", "https://")):
                raise ValueError("worker URL must start with http:// or https://")
            kw = dict(minimum=float(d["min"]), maximum=float(d["max"]),
                      initial_step=float(d["initial_step"]), autotune_resolution=float(d["resolution"]),
                      batch_size=int(d["batch_size"]), margin_of_error=float(d["margin_of_error"]),
                      baseline_score=None if d.get("baseline_score") in (None, "") else float(d["baseline_score"]),
                      max_batches=int(d.get("max_batches") or 40), auto_delete=bool(d.get("auto_delete")))
            state_path = Path(d.get("state") or "runs/adaptive-gui-state.json")
            state_path.parent.mkdir(parents=True, exist_ok=True)
        except (ValueError, KeyError, TypeError) as e:
            self.sendj({"error": str(e)}, 400)
            return
        if STATE.get("thread") and STATE["thread"].is_alive():
            self.sendj({"error": "already running"}, 409)
            return
        STATE["state"] = {"status": "starting", "results": []}

        def run():
            try:
                tuner = AdaptiveTuner(str(state_path))
                client = WorkerClient(d["worker"], token=d.get("token", ""), cafile=d.get("cafile", ""))
                STATE["tuner"] = tuner
                STATE["state"] = tuner.run(client=client, **kw)
            except Exception as e:
                STATE["state"] = {**STATE.get("state", {}), "status": "failed", "error": str(e)}

        th = threading.Thread(target=run, daemon=True)
        STATE["thread"] = th
        th.start()
        self.sendj({"ok": True})

    def log_message(self, *a):
        pass


def live_state():
    """While a run is in progress, expose the tuner's persisted state."""
    while True:
        t = STATE.get("tuner")
        if t is not None and STATE.get("thread") and STATE["thread"].is_alive():
            STATE["state"] = {**t.state, "status": t.state.get("status", "running")}
        threading.Event().wait(1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--allow-unauthenticated", action="store_true",
                   help="permit a non-loopback bind (this page has no auth)")
    a = p.parse_args(argv)
    sec.loopback_only(a.host, "the Adaptive Tuning lab", a.allow_unauthenticated)
    threading.Thread(target=live_state, daemon=True).start()
    s = ThreadingHTTPServer((a.host, a.port), H)
    print(f"OmniCortex Adaptive Tuning: http://{a.host}:{a.port}/")
    s.serve_forever()


if __name__ == "__main__":
    main()
