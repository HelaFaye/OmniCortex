#!/usr/bin/env python3
"""OmniCortex Quantization Lab - local browser GUI.

Stdlib-only web server; quantization work is delegated to
scripts/quantization_lab.py. Default bind is loopback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
sys.path.insert(0, SCRIPTS)

from quantization_lab import (  # noqa: E402
    QuantizationLabError,
    build_model,
    build_plan,
    parse_weighted_specs,
    read_gguf_info,
    scratch_status,
)

STATE = {
    "models_dir": os.path.expanduser("~/.models"),
    "profiles_dir": str(ROOT / "profiles"),
    "jobs": {},
    "lock": threading.Lock(),
}

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Quantization Lab</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color-scheme:dark}
body{margin:0;background:#111;color:#eee}main{max-width:1280px;margin:auto;padding:22px}
h1{margin:.1rem 0 .3rem}p,small{color:#aaa}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:16px}.wide{grid-column:1/-1}
label{display:block;margin:.55rem 0 .25rem;color:#bbb}input,select,textarea,button{font:inherit;border-radius:8px;border:1px solid #444;background:#101010;color:#eee;padding:9px}
input,select,textarea{width:100%;box-sizing:border-box}textarea{min-height:90px;font-family:ui-monospace,monospace}.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}.row button{flex:0 0 auto}
button{cursor:pointer;background:#252525}button.primary{background:#3a4d70;border-color:#6f86b4}.status{padding:10px;border-radius:8px;background:#151515;margin-top:10px}.ok{color:#8fd18f}.warn{color:#f0c674}.err{color:#f07178}
pre{white-space:pre-wrap;word-break:break-word;background:#0c0c0c;padding:12px;border-radius:8px;max-height:420px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:.92rem}th,td{text-align:left;padding:7px;border-bottom:1px solid #303030}code{font-family:ui-monospace,monospace}
.badge{display:inline-block;padding:3px 7px;border-radius:999px;background:#292929;font-size:.8rem}.layerbar{height:7px;background:#333;border-radius:5px;overflow:hidden}.layerbar>i{display:block;height:100%;background:#8b9dc3}
@media(max-width:900px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head>
<body><main>
<h1>OmniCortex Quantization Lab</h1>
<p>Capability-aware GGUF quantization using weighted calibration + H-Neuron layer protection. Builds in RAM scratch when it fits, then copies atomically to the destination.</p>
<div id="global" class="status">Loading…</div>
<div class="grid">
<section class="card">
<h2>Source / target</h2>
<label>Model directory</label><input id="modelsDir" value="">
<div class="row"><button onclick="scanModels()">Scan models</button><select id="model" onchange="modelChanged()"></select></div>
<div id="modelMeta" class="status"></div>
<label>Output GGUF</label><input id="output" placeholder="/path/to/model-Q4-preserved.gguf">
<div class="row"><div><label>Base quant</label><select id="baseQuant"><option>Q4_K_M</option><option>Q5_K_M</option><option>Q6_K</option><option>Q8_0</option></select></div>
<div><label>Protected quant</label><select id="protectedQuant"><option>Q6_K</option><option>Q8_0</option><option>F16</option></select></div></div>
<label>Protection budget: <span id="budgetValue">10</span>% of decoder layers</label><input id="budget" type="range" min="0" max="50" value="10" oninput="budgetValue.textContent=this.value">
<label><input id="sharedExpert" type="checkbox" style="width:auto"> include MoE shared-expert down tensors</label>
</section>
<section class="card">
<h2>Capability profiles</h2>
<label>Profile directory</label><input id="profilesDir" value="">
<p>One <code>PROFILE.json::weight</code> per line. Higher weight = higher preservation priority.</p>
<textarea id="profiles" placeholder="profiles/&lt;fingerprint&gt;/math.json::1.0&#10;profiles/&lt;fingerprint&gt;/coding.json::1.2"></textarea>
<div class="row"><button onclick="scanProfiles()">Show profiles</button><button onclick="preview()" class="primary">Preview plan</button></div>
<pre id="profileList">No profiles scanned.</pre>
</section>
<section class="card">
<h2>Calibration corpus</h2>
<p>One text file per line as <code>/path/file.txt::weight</code>. The lab deterministically samples and interleaves them.</p>
<textarea id="calibration" placeholder="/data/math.txt::1.0&#10;/data/general.txt::0.5&#10;/data/reasoning.txt::1.0"></textarea>
<label>Maximum calibration lines</label><input id="maxLines" type="number" value="20000" min="1">
</section>
<section class="card">
<h2>Scratch / toolchain</h2>
<label>Scratch</label><select id="scratch"><option value="auto">AUTO (RAM if it fits)</option><option value="ram">RAM only</option><option value="disk">Disk temp</option><option value="none">No preferred scratch</option></select>
<div class="row"><div><label>llama-imatrix</label><input id="imatrixBin" value="llama-imatrix"></div><div><label>llama-quantize</label><input id="quantizeBin" value="llama-quantize"></div><div><label>Threads</label><input id="threads" type="number" value="0" min="0"></div></div>
<label><input id="allowRequantize" type="checkbox" style="width:auto"> allow quantized source requantization</label>
<p><small>Recommended: keep the canonical F16/BF16 source and build deployment quants from it.</small></p>
</section>
<section class="card wide"><h2>Protection plan</h2><div id="plan"></div></section>
<section class="card wide"><h2>Build</h2><div class="row"><button onclick="build()" class="primary">Build preserved GGUF</button><button onclick="refreshJob()">Refresh job</button></div><div id="buildStatus" class="status">No build running.</div><pre id="log"></pre></section>
</div></main>
<script>
let lastJob=null, models=[];
const $=id=>document.getElementById(id);
async function j(url,opt){const r=await fetch(url,opt);const t=await r.text();let x;try{x=JSON.parse(t)}catch{x={error:t}}if(!r.ok)throw new Error(x.error||t);return x}
async function scanModels(){const dir=$('modelsDir').value;const d=await j('/api/models?dir='+encodeURIComponent(dir));models=d.models;const s=$('model');s.innerHTML='';models.forEach((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=m.path+' ['+(m.quant||'?')+']';s.appendChild(o)});if(models.length)modelChanged()}
async function scanProfiles(){const d=await j('/api/profiles?dir='+encodeURIComponent($('profilesDir').value));$('profileList').textContent=(d.profiles||[]).map(p=>p.path+'  '+(p.config_name||'default')+'  '+(p.total_neurons||0)+' neurons').join('\n')||'No profiles found.'}
function modelChanged(){const m=models[$('model').value];if(!m)return;$('output').value=m.path.replace(/\.gguf$/,'')+'-preserved-'+$('baseQuant').value+'.gguf';$('modelMeta').innerHTML='<b>Quant:</b> '+(m.quant||'?')+' &nbsp; <b>Size:</b> '+m.size_gib.toFixed(2)+' GiB &nbsp; <b>Arch:</b> '+(m.arch||'?')}
function specs(id){return $(id).value.split(/\n/).map(x=>x.trim()).filter(Boolean)}
function showPlan(p){
 let h='<p><b>Protected layers:</b> '+JSON.stringify(p.selected_layers)+' &nbsp; <b>Rule:</b> <code>'+escapeHtml(p.tensor_regex||'none')+'</code></p>';
 if(p.warnings.length)h+='<div class="status warn">'+p.warnings.map(escapeHtml).join('<br>')+'</div>';
 h+='<table><thead><tr><th>Layer</th><th>Score</th><th>Selected neurons</th><th>Contribution</th></tr></thead><tbody>';
 p.layer_scores.slice(0,Math.min(p.layer_scores.length,40)).forEach(r=>{const selected=p.selected_layers.includes(r.layer);h+='<tr><td><span class="badge">'+r.layer+'</span></td><td>'+r.score.toFixed(6)+'</td><td>'+r.selected_neurons+'</td><td>'+escapeHtml(JSON.stringify(r.contributions))+'</td></tr>'});
 h+='</tbody></table>';if(p.estimate.estimated_delta_gib!=null)h+='<p><small>Estimated additional output size for protected tensors: '+p.estimate.estimated_delta_gib.toFixed(3)+' GiB (approximate).</small></p>';$('plan').innerHTML=h
}
async function preview(){try{const p=await j('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});showPlan(p)}catch(e){$('plan').innerHTML='<div class="status err">'+escapeHtml(e.message)+'</div>'}}
function payload(){return{model:models[$('model').value]?.path||$('model').value,output:$('output').value,profiles:specs('profiles'),calibration:specs('calibration'),base_quant:$('baseQuant').value,protected_quant:$('protectedQuant').value,budget_percent:+$('budget').value,scratch:$('scratch').value,imatrix_bin:$('imatrixBin').value,quantize_bin:$('quantizeBin').value,threads:+$('threads').value,max_calibration_lines:+$('maxLines').value,include_shared_expert:$('sharedExpert').checked,allow_requantize_source:$('allowRequantize').checked,force:false}}
async function build(){try{const r=await j('/api/build',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});lastJob=r.id;refreshJob()}catch(e){$('buildStatus').innerHTML='<span class="err">'+escapeHtml(e.message)+'</span>'}}
async function refreshJob(){if(!lastJob)return;try{const r=await j('/api/jobs/'+lastJob);$('buildStatus').innerHTML='<b>'+r.state+'</b>'+(r.error?'<span class="err"> '+escapeHtml(r.error)+'</span>':'')+'<br>'+escapeHtml(r.started||'');$('log').textContent=(r.log||[]).join('\n');if(r.state==='running')setTimeout(refreshJob,1000)}catch(e){$('buildStatus').textContent=e.message}}
async function global(){try{const d=await j('/api/health');$('global').innerHTML='<span class="ok">Server ready.</span> Scratch /dev/shm: '+d.ram_free_gib.toFixed(2)+' GiB free'+(d.ram_tmpfs?' (tmpfs)':'') }catch(e){$('global').textContent=e.message}}
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
$('modelsDir').value=__MODELS__;$('profilesDir').value=__PROFILES__;scanModels();global();
</script></body></html>'''


def _json(handler, value, code=200):
    data = json.dumps(value).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler):
    n = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(n) or b"{}")


def _scan_models(root):
    root = os.path.expanduser(root)
    out = []
    if not os.path.isdir(root):
        return out
    for p in sorted(Path(root).rglob("*.gguf")):
        try:
            st = p.stat()
            info = read_gguf_info(str(p))
            out.append({"path": str(p), "size": st.st_size, "size_gib": st.st_size / 2**30,
                        "quant": info.get("quant"), "arch": info.get("arch"),
                        "n_layers": info.get("n_layers")})
        except OSError:
            continue
    return out


def _scan_profiles(root):
    root = os.path.expanduser(root)
    out = []
    if not os.path.isdir(root):
        return out
    for p in sorted(Path(root).rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "by_layer" not in d or "fingerprint" not in d:
            continue
        total = sum(len(v) for v in (d.get("by_layer") or {}).values())
        out.append({"path": str(p), "config_name": d.get("config_name"), "fingerprint": d.get("fingerprint"),
                    "n_layers": d.get("n_layers"), "n_neurons": d.get("n_neurons"), "total_neurons": total})
    return out


def _run_job(job_id, payload):
    with STATE["lock"]:
        job = STATE["jobs"][job_id]
        job["state"] = "running"
    logs = job["log"]
    try:
        calibration = parse_weighted_specs(payload.get("calibration") or [])
        profiles = parse_weighted_specs(payload.get("profiles") or [])
        manifest = build_model(
            model=payload["model"], output=payload["output"], calibration=calibration, profiles=profiles,
            base_quant=payload.get("base_quant", "Q4_K_M"), protected_quant=payload.get("protected_quant", "Q6_K"),
            budget_percent=float(payload.get("budget_percent", 10)), scratch=payload.get("scratch", "auto"),
            imatrix_bin=payload.get("imatrix_bin", "llama-imatrix"), quantize_bin=payload.get("quantize_bin", "llama-quantize"),
            threads=int(payload.get("threads", 0)), max_calibration_lines=int(payload.get("max_calibration_lines", 20000)),
            include_shared_expert=bool(payload.get("include_shared_expert")),
            allow_requantize_source=bool(payload.get("allow_requantize_source")), force=bool(payload.get("force")), log=logs)
        with STATE["lock"]:
            job["state"] = "done"; job["manifest"] = manifest; job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    except Exception as e:
        logs.append(f"ERROR: {e}")
        with STATE["lock"]:
            job["state"] = "error"; job["error"] = str(e); job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            # JSON-encode: the values land inside a <script>, so a raw path
            # would be a syntax error (and an injection point).
            data = (HTML.replace("__MODELS__", json.dumps(STATE["models_dir"]).replace("<", "\\u003c"))
                        .replace("__PROFILES__", json.dumps(STATE["profiles_dir"]).replace("<", "\\u003c"))).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if u.path == "/api/health":
            st = scratch_status("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
            _json(self, {"ok": True, "ram_free_gib": st["free_gib"], "ram_tmpfs": st["is_tmpfs"]}); return
        if u.path == "/api/models":
            qs = parse_qs(u.query); root = qs.get("dir", [STATE["models_dir"]])[0]; _json(self, {"models": _scan_models(root)}); return
        if u.path == "/api/profiles":
            qs = parse_qs(u.query); root = qs.get("dir", [STATE["profiles_dir"]])[0]; _json(self, {"profiles": _scan_profiles(root)}); return
        if u.path.startswith("/api/jobs/"):
            job_id = u.path.rsplit("/", 1)[-1]
            with STATE["lock"]:
                job = STATE["jobs"].get(job_id)
            if not job: _json(self, {"error": "job not found"}, 404); return
            _json(self, job); return
        _json(self, {"error": "not found"}, 404)
    def do_POST(self):
        u = urlparse(self.path)
        try:
            payload = _read_json(self)
            if u.path == "/api/plan":
                p = build_plan(payload["model"], payload.get("base_quant", "Q4_K_M"), payload.get("protected_quant", "Q6_K"),
                                float(payload.get("budget_percent", 10)), parse_weighted_specs(payload.get("profiles") or []),
                                bool(payload.get("include_shared_expert")))
                _json(self, p); return
            if u.path == "/api/build":
                job_id = uuid.uuid4().hex[:12]
                job = {"id": job_id, "state": "queued", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": [], "error": None}
                with STATE["lock"]: STATE["jobs"][job_id] = job
                threading.Thread(target=_run_job, args=(job_id, payload), daemon=True).start()
                _json(self, {"id": job_id}); return
            _json(self, {"error": "not found"}, 404)
        except QuantizationLabError as e:
            _json(self, {"error": str(e)}, 400)
        except Exception as e:
            _json(self, {"error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--models-dir", action="append", default=[])
    ap.add_argument("--profiles-dir", default=str(ROOT / "profiles"))
    ap.add_argument("--allow-unauthenticated", action="store_true", help="permit a non-loopback bind (no auth)")
    a = ap.parse_args()
    import ns_security as sec
    sec.loopback_only(a.host, "the Quantization Lab", a.allow_unauthenticated)
    STATE["models_dir"] = os.path.expanduser(a.models_dir[0]) if a.models_dir else os.path.expanduser("~/.models")
    STATE["profiles_dir"] = os.path.expanduser(a.profiles_dir)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"OmniCortex Quantization Lab: http://{a.host}:{a.port}")
    print(f"models: {STATE['models_dir']}")
    print(f"profiles: {STATE['profiles_dir']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
