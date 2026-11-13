#!/usr/bin/env python3
"""OmniCortex Scale Sweep Lab browser GUI."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scale_sweep_lab import SweepEngine, SweepError, activation_profile_data, parse_scales, scratch_status  # noqa: E402

STATE = {"jobs": {}, "lock": threading.Lock()}

HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Scale Sweep Lab</title>
<style>
body{margin:0;background:#101216;color:#e9ecf1;font:14px system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:20px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.wide{grid-column:1/-1}.card{background:#181b21;border:1px solid #30343c;border-radius:12px;padding:14px}h1{margin:0 0 6px}h2{margin:0 0 10px;font-size:18px}small,p{color:#aeb5c0}label{display:block;color:#aeb5c0;margin:7px 0 3px}input,select,textarea,button{font:inherit;background:#0e1116;color:#eef;border:1px solid #3a404a;border-radius:7px;padding:8px;box-sizing:border-box}input,select,textarea{width:100%}textarea{min-height:90px;font-family:ui-monospace,monospace}.row{display:flex;gap:8px}.row>*{flex:1}button{cursor:pointer}.primary{background:#2d4770;border-color:#6382b6}.danger{border-color:#724047}.status{padding:9px;background:#12151a;border-radius:8px;margin-top:8px;white-space:pre-wrap}.ok{color:#8bd39b}.warn{color:#e8c476}.err{color:#f17c86}.muted{color:#8e97a5}
.bar{height:10px;background:#2a2f37;border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:#7a92bd}.chart{width:100%;height:260px;display:block;background:#0e1116;border-radius:8px}.heat{display:grid;grid-template-columns:repeat(32,1fr);gap:1px}.cell{height:10px;background:#242933}.cell.on{background:#c5d7f8}.scale-dot{stroke:#111}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px;border-bottom:1px solid #2c3139}.pill{display:inline-block;padding:2px 7px;border-radius:999px;background:#282e38;font-size:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.row{flex-direction:column}}
</style></head><body><main>
<h1>OmniCortex Scale Sweep Lab</h1><p>Generate static GGUF suppression/amplification variants from an H-Neuron profile, visualize the activation footprint, batch builds through scratch/RAM disk, and optionally evaluate/prune candidates.</p>
<div id="top" class="status">Ready.</div>
<div class="grid">
<section class="card"><h2>Model + profile</h2>
<label>Base GGUF</label><input id="model" placeholder="/path/to/model.Q6_K.gguf">
<label>H-Neuron profile JSON</label><input id="profile" placeholder="/path/to/h_neurons.json" onblur="loadProfile()">
<label>Output directory</label><input id="outdir" placeholder="/path/to/sweep-output">
<div class="row"><button onclick="loadProfile()">Load activation profile</button><button onclick="preview()" class="primary">Preview sweep</button></div></section>
<section class="card"><h2>Scale tuning</h2>
<div class="row"><div><label>Minimum scale</label><input id="min" type="number" step="0.01" value="0.20"></div><div><label>Maximum scale</label><input id="max" type="number" step="0.01" value="0.40"></div><div><label>Resolution / step</label><input id="step" type="number" step="0.01" value="0.05"></div></div>
<p>Below 1.0 = suppression. 1.0 = baseline. Above 1.0 = amplification.</p><div id="scalePreview"></div></section>
<section class="card"><h2>Batch + scratch</h2>
<label>Models per batch</label><input id="batch" type="number" min="1" value="1">
<label>Scratch strategy</label><select id="scratch"><option value="auto">AUTO — RAM if it fits</option><option value="ram">RAM only</option><option value="disk">Disk temp</option><option value="none">Direct output</option></select>
<label>Scratch root (optional)</label><input id="scratchRoot" placeholder="/dev/shm/omnicortex">
<div id="scratchInfo" class="status">Scratch not checked.</div></section>
<section class="card"><h2>Evaluation + pruning</h2>
<label>Baseline score (0–1)</label><input id="baseline" type="number" min="0" max="1" step="0.001" placeholder="0.80">
<label>Margin of error / tolerance</label><input id="moe" type="number" min="0" max="1" step="0.01" value="0.05">
<label>Evaluator command</label><textarea id="eval" placeholder="python scripts/my_evaluator.py --model \"{model}\" --scale {scale} --json"></textarea>
<p>Evaluator must print JSON such as <code>{&quot;correct&quot;:82,&quot;total&quot;:100}</code> or <code>{&quot;score&quot;:0.82}</code>. Auto-delete only runs when an evaluator and baseline are supplied.</p>
<label><input id="autodel" type="checkbox" style="width:auto"> automatically delete underperforming candidates</label></section>
<section class="card wide"><h2>Activation / H-Neuron footprint</h2><div id="activationMeta"></div><svg id="layerChart" class="chart" viewBox="0 0 1000 260"></svg><div id="heatmap"></div></section>
<section class="card wide"><h2>Suppression / amplification response</h2><svg id="scaleChart" class="chart" viewBox="0 0 1000 260"></svg><div id="planTable"></div></section>
<section class="card wide"><h2>Run</h2><div class="row"><button onclick="start()" class="primary">Start sweep</button><button onclick="stop()" class="danger">Stop after current batch</button><button onclick="refresh()">Refresh</button></div><div id="runStatus" class="status">No run started.</div><pre id="log" class="status"></pre></section>
</div></main>
<script>
const $=id=>document.getElementById(id);let job=null,act=null;
async function api(url,opt){const r=await fetch(url,opt);const t=await r.text();let d;try{d=JSON.parse(t)}catch{d={error:t}}if(!r.ok)throw Error(d.error||t);return d}
function payload(){return {model:$('model').value,profile:$('profile').value,output_dir:$('outdir').value,min_scale:+$('min').value,max_scale:+$('max').value,step:+$('step').value,batch_size:+$('batch').value,scratch:$('scratch').value,scratch_root:$('scratchRoot').value,baseline_score:$('baseline').value===''?null:+$('baseline').value,margin_of_error:+$('moe').value,evaluator_cmd:$('eval').value,auto_delete:$('autodel').checked}}
function escape(s){return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
async function loadProfile(){try{act=await api('/api/profile?path='+encodeURIComponent($('profile').value));renderActivation();}catch(e){$('top').innerHTML='<span class="err">'+escape(e.message)+'</span>'}}
function renderActivation(){if(!act)return;$('activationMeta').innerHTML='<span class="pill">layers '+act.n_layers+'</span> <span class="pill">neurons '+act.n_neurons+'</span>';let pts=act.layers;let max=Math.max(...pts.map(x=>x.density),1e-9);let svg=$('layerChart');let w=1000,h=240,left=40,bottom=25,gw=w-left-15,gh=h-bottom;svg.innerHTML='<line x1="40" y1="240" x2="985" y2="240" stroke="#454b56"/>' + pts.map((p,i)=>{let bw=gw/Math.max(1,pts.length)-1;let x=left+i*gw/pts.length;let bh=gh*(p.density/max);return `<rect x="${x.toFixed(1)}" y="${(h-bottom-bh).toFixed(1)}" width="${Math.max(1,bw).toFixed(1)}" height="${bh.toFixed(1)}" fill="#7a92bd"><title>Layer ${p.layer}: ${p.selected} selected (${(p.density*100).toFixed(2)}%)</title></rect>`}).join('');let rows=act.layers.map(p=>p.neurons);$('heatmap').innerHTML=rows.map((ns,i)=>{let set=new Set(ns);let n=Math.min(act.n_neurons,256);let cells=Array.from({length:Math.ceil(n/Math.max(1,Math.ceil(n/32)))},()=>0);let width=Math.ceil(n/32);let html='<div class="muted">L'+i+'</div><div class="heat">';for(let j=0;j<n;j++){html+=`<span class="cell ${set.has(j)?'on':''}" title="Layer ${i} neuron ${j}"></span>`}return html+'</div>'}).join('')}
async function updateScratch(){try{const root=$('scratchRoot').value;const d=await api('/api/scratch?path='+encodeURIComponent(root));const need=(+$('batch').value||1)*1.03;$('scratchInfo').textContent='Free: '+d.free_gib.toFixed(2)+' GiB at '+d.path+(d.is_tmpfs_hint?' (tmpfs)':'')+'; batch capacity is estimated against '+(need)+'× model size.'}catch(e){$('scratchInfo').textContent='Scratch check failed: '+e.message}}
async function preview(){try{const d=await api('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});$('scalePreview').innerHTML=d.scales.map(s=>`<span class="pill">${s.toFixed(3)}</span>`).join(' ');renderScaleChart(d);$('planTable').innerHTML='<table><tr><th>Scale</th><th>Mode</th><th>Expected</th></tr>'+d.scales.map(s=>`<tr><td>${s.toFixed(3)}</td><td>${s<1?'Suppression':s>1?'Amplification':'Baseline'}</td><td>${s<1?'reduce selected contribution':s>1?'increase selected contribution':'unchanged'}</td></tr>`).join('')+'</table>'}catch(e){$('scalePreview').innerHTML='<span class="err">'+escape(e.message)+'</span>'}}
function renderScaleChart(d){let svg=$('scaleChart'),w=1000,h=260,l=50,r=15,t=20,b=35,x0=d.min,x1=d.max||1,xr=Math.max(1e-9,x1-x0);let pts=d.scales.map(s=>{let x=l+(s-x0)/xr*(w-l-r);let y=130-(s-1)*120/Math.max(Math.abs(x0-1),Math.abs(x1-1),0.01);return [x,Math.max(20,Math.min(220,y))]}),path=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');let baseX=l+(1-x0)/xr*(w-l-r);svg.innerHTML=`<line x1="${baseX}" y1="20" x2="${baseX}" y2="225" stroke="#6c7380" stroke-dasharray="6 5"/><path d="${path}" fill="none" stroke="#91a8d0" stroke-width="3"/>`+pts.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="5" fill="#b9cbed"><title>${d.scales[i].toFixed(3)}</title></circle>`).join('')+'<text x="55" y="18" fill="#9ba4b2">suppression ← 1.0 baseline → amplification</text>'}
async function start(){try{await preview();job=await api('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});$('runStatus').textContent='Job '+job.id+' started';poll()}catch(e){$('runStatus').innerHTML='<span class="err">'+escape(e.message)+'</span>'}}
async function stop(){if(job)try{await api('/api/stop?id='+job.id,{method:'POST'})}catch(e){$('runStatus').textContent=e.message}}
async function refresh(){if(job)poll();else $('runStatus').textContent='No active job.'}
async function poll(){if(!job)return;try{const d=await api('/api/job?id='+job.id);$('runStatus').textContent=d.status+'  '+d.completed+'/'+d.total+' candidates';$('log').textContent=JSON.stringify(d.history,null,2);if(d.done){$('top').innerHTML=d.error?'<span class="err">'+escape(d.error)+'</span>':'<span class="ok">Sweep complete</span>';job=null;return}setTimeout(poll,1000)}catch(e){$('runStatus').textContent=e.message}}
if($('profile').value)loadProfile();preview();updateScratch();setInterval(updateScratch,5000);
</script></body></html>'''


def json_response(h: BaseHTTPRequestHandler, obj: dict, code=200):
    data=json.dumps(obj).encode()
    h.send_response(code);h.send_header('Content-Type','application/json');h.send_header('Content-Length',str(len(data)));h.end_headers();h.wfile.write(data)


def body(h):
    n=int(h.headers.get('Content-Length','0'));return json.loads(h.rfile.read(n) or b'{}')


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        u=urlparse(self.path);q=parse_qs(u.query)
        try:
            if u.path=='/':
                d=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(d)));self.end_headers();self.wfile.write(d);return
            if u.path=='/api/profile':
                path=q.get('path',[''])[0];json_response(self,activation_profile_data(os.path.expanduser(path)));return
            if u.path=='/api/scratch':
                json_response(self,scratch_status(q.get('path',[''])[0] or None));return
            if u.path=='/api/job':
                jid=q.get('id',[''])[0]
                with STATE['lock']: j=STATE['jobs'].get(jid)
                if not j: json_response(self,{'error':'unknown job'},404);return
                json_response(self,j['public']);return
            json_response(self,{'error':'not found'},404)
        except Exception as e: json_response(self,{'error':str(e)},400)
    def do_POST(self):
        u=urlparse(self.path);q=parse_qs(u.query)
        try:
            if u.path=='/api/preview':
                a=body(self);sc=parse_scales(float(a['min_scale']),float(a['max_scale']),float(a['step']));json_response(self,{'scales':sc,'min':float(a['min_scale']),'max':float(a['max_scale'])});return
            if u.path=='/api/start':
                a=body(self);sc=parse_scales(float(a['min_scale']),float(a['max_scale']),float(a['step']));jid=uuid.uuid4().hex[:12]
                job={'public':{'id':jid,'status':'starting','completed':0,'total':len(sc),'history':[]},'engine':SweepEngine(Path(a['output_dir'])/'sweep_state.json'),'args':a,'scales':sc}
                with STATE['lock']: STATE['jobs'][jid]=job
                def worker():
                    try:
                        job['public']['status']='running'
                        res=job['engine'].run(model=a['model'],profile=a['profile'],output_dir=a['output_dir'],scales=sc,batch_size=int(a['batch_size']),scratch_policy=a['scratch'],scratch_root=a.get('scratch_root',''),baseline_score=a.get('baseline_score'),margin_of_error=float(a.get('margin_of_error',0.05)),evaluator_cmd=a.get('evaluator_cmd',''),auto_delete=bool(a.get('auto_delete',False)))
                        job['public']['history']=res['history'];job['public']['completed']=len(res['history']);job['public']['status']='complete';job['public']['done']=True
                    except Exception as e:
                        job['public']['status']='error';job['public']['error']=str(e);job['public']['done']=True
                threading.Thread(target=worker,daemon=True).start();json_response(self,job['public']);return
            if u.path=='/api/stop':
                jid=q.get('id',[''])[0]
                with STATE['lock']: j=STATE['jobs'].get(jid)
                if not j: json_response(self,{'error':'unknown job'},404);return
                j['engine'].stop();j['public']['status']='stopping';json_response(self,{'ok':True});return
            json_response(self,{'error':'not found'},404)
        except Exception as e: json_response(self,{'error':str(e)},400)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8797);ap.add_argument('--allow-unauthenticated',action='store_true',help='permit a non-loopback bind (no auth)');a=ap.parse_args()
    import oc_security as sec; sec.loopback_only(a.host,'the Scale Sweep Lab',a.allow_unauthenticated)
    s=ThreadingHTTPServer((a.host,a.port),Handler);print(f'OmniCortex Scale Sweep Lab: http://{a.host}:{a.port}')
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()

if __name__=='__main__':main()
