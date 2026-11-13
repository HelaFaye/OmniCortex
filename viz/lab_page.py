"""Studio's /lab page: start, open and stop the labs, the live and replay
views and the pipeline dashboard (viz/hub.py's service manager), plus links to
the desktop viewers, which run as jobs."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Lab</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
a{color:var(--acc)}
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{max-width:1100px;margin:0 auto;padding:1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.85rem 1rem;display:flex;flex-direction:column;gap:.4rem}
h2{font-size:15px;margin:0}h3{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:1.2rem 0 .5rem}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no);font-size:12px;white-space:pre-wrap}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin-top:auto}
button,a.btn{font:500 13px ui-sans-serif,system-ui;padding:.35rem .75rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer;text-decoration:none}
button.pri,a.btn.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button:disabled{opacity:.5}
select,input{font:12.5px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg);width:100%}
.up{color:var(--ok);font-size:12px}
</style>
<header><h1>Studio · Lab</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a></header>
<main>
<h3>Live views and labs</h3><div class="grid" id="svcs"></div>
<h3>Desktop viewers</h3>
<div class="note">These open a window on this computer (pygfx or fastplotlib), so they run as jobs: <a href="/jobs#kind=view_timeline">timeline</a> · <a href="/jobs#kind=view_explore">2D/3D explorer</a> · <a href="/jobs#kind=view_weights">weights</a> · <a href="/jobs#kind=compare">compare two traces</a>.</div>
<div class="grid" style="margin-top:.6rem" id="views"></div>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const TITLES={live:'Live activations',replay:'Replay a trace (3D)',pipeline:'Pipeline dashboard',quant_lab:'Quantization lab',
  scale_sweep:'Scale sweep lab',adaptive_tuning:'Adaptive tuning lab',transfer:'Model transfer'};
let D=null;
function url(port){ return `${location.protocol}//${location.hostname}:${port}/` }
function render(){
  $('#svcs').innerHTML=Object.entries(D.services).map(([n,s])=>{
    let extra='';
    if(n==='replay') extra=`<label class="note">trace</label><select data-cfg="trace">${D.traces.map(t=>`<option value="${esc(t.path)}">${esc((t.model||'')+' · '+(t.created||'')+' · '+({studio:'checked reply',decode_lab:'decoding lab'}[t.source]||'trace run'))}</option>`).join('')||'<option value="">no traces yet: check a reply in chat, or run “Trace one sample”</option>'}</select>`;
    if(n==='live') extra=`<label class="note">patched llama-server URL (empty = simulated until the decoding lab or autotrace pushes frames)</label><input data-cfg="live_source" placeholder="http://127.0.0.1:8080">`;
    return `<div class="card" data-svc="${esc(n)}"><h2>${esc(TITLES[n]||n)}</h2><div class="note">${esc(s.desc)}</div>${extra}
      <div class="row">${s.running?`<a class="btn pri" target="_blank" rel="noopener" href="${url(s.port)}">Open ↗</a><button data-a="stop">Stop</button><span class="up">running on port ${s.port}</span>`
        :`<button class="pri" data-a="start" ${s.available?'':'disabled'}>Start</button>`}</div><div class="err"></div></div>`}).join('');
  document.querySelectorAll('[data-svc]').forEach(card=>card.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{
    const n=card.dataset.svc, cfg={}; card.querySelectorAll('[data-cfg]').forEach(e=>{ if(e.value) cfg[e.dataset.cfg]=e.value });
    b.disabled=true; b.textContent=b.dataset.a==='start'?'starting…':'stopping…';
    try{ const r=await api('api/lab/'+b.dataset.a,{name:n,cfg}); if(b.dataset.a==='start') window.open(url(r.port),'_blank'); }
    catch(e){ card.querySelector('.err').textContent=e.message }
    await load(); }));
  $('#views').innerHTML=[['Timeline','Exact per-token view of a trace, no glow: for measuring.'],['Explorer','Mean and contrast maps and a 3D volume of an activations run; click a cell to inspect.'],
    ['Weights','Weight magnitude, quantization error and where the H-Neurons sit in a GGUF.'],['Compare','Two or more runs side by side: depth profiles, overlap (a report).']]
    .map(([t,d])=>`<div class="card"><h2>${t}</h2><div class="note">${d}</div><div class="row"><a class="btn" href="/jobs">Open in Jobs</a></div></div>`).join('');
}
async function load(){ D=await api('api/lab'); render(); }
load().catch(e=>$('#svcs').innerHTML=`<div class="err">${esc(e.message)}</div>`);
setInterval(()=>load().catch(()=>{}),5000);
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
