"""Studio's /workers page: where the director's tasks run, and control over it.

Workers are this computer's models (started on demand), OpenAI-compatible
endpoints you run yourself (here or elsewhere), and models on remote hosts and
linked Studios. The page shows each one's state, load and results, the GPUs'
temperatures, which subjects go where, and the queue; it can pause, drain,
limit and test workers, route subjects, and move a task to another worker.
Served by viz/studio.py to the owner only; everything shown is escaped.
"""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Workers</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
a{color:var(--acc)}
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0;--chip:#eef1fb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--chip:#22283a}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--chip:#22283a}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:1rem;padding:1rem;max-width:1600px;margin:0 auto}
main>*{min-width:0}.wide{grid-column:1/-1}
@media (max-width:1000px){main{grid-template-columns:1fr;padding:.6rem}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem;margin-bottom:1rem}
h2{font-size:12px;margin:.1rem 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.55rem .7rem}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{font-size:12px;color:var(--mut)}
label{display:block;font-size:12px;color:var(--mut);margin:.45rem 0 .1rem}
input,select{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto;margin-right:.3rem}
button{font:500 12px ui-sans-serif,system-ui;padding:.28rem .6rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}
.note{font-size:12px;color:var(--mut)}.err{color:var(--no);font-size:12px;white-space:pre-wrap}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.grow{flex:1}
.tscroll{overflow-x:auto}
table{width:100%;border-collapse:collapse}td,th{padding:.4rem .35rem;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;font-size:13px}
th{font-size:11px;color:var(--mut);font-weight:500;text-transform:uppercase}
.dot{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;margin-right:.35rem;vertical-align:middle;background:var(--mut)}
.dot.up,.dot.running,.dot.loaded{background:var(--ok)}.dot.down,.dot.hot{background:var(--no)}.dot.loading,.dot.checking{background:var(--warn)}
.kind{font-size:11px;padding:0 .45rem;border-radius:9px;background:var(--chip);white-space:nowrap}
.chip{display:inline-block;font-size:11px;padding:0 .45rem;border-radius:9px;background:var(--chip);margin:0 .2rem .2rem 0}
.bar{height:.45rem;background:var(--code);border-radius:4px;overflow:hidden;min-width:5rem}.bar>i{display:block;height:100%}
.ctl{display:flex;gap:.3rem;flex-wrap:wrap;align-items:center}.ctl input{width:3.6rem}
details summary{cursor:pointer}
</style>
<header><h1>Studio · Workers</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/remote">Remote</a><a href="/jobs">Jobs</a><span class="note" id="upd"></span></header>
<main>
<section class="wide"><div class="tiles" id="tiles"></div></section>
<section>
 <div class="card"><div class="row"><h2 class="grow">Workers</h2><label class="note" style="margin:0"><input type="checkbox" id="allLocal"> show every local model</label></div>
  <div class="note">This Studio is the <b>controller</b>: it plans the work and sends each task to a worker. A worker is where the model runs and the work happens: <b>this computer</b> (Studio starts the model when a task needs it), an <b>endpoint</b> (a server you started yourself, here or on another computer), or a <b>remote host</b> (a computer running OmniCortex Remote, or a linked Studio).</div>
  <div class="tscroll"><table id="workers"></table></div><div id="werr" class="err"></div></div>
 <div class="card"><h2>Queue</h2><div class="note">Unfinished tasks of approved, running and paused projects. Pick another worker to move a task; it takes effect when the task next starts.</div>
  <div class="tscroll"><table id="queue"></table></div></div>
</section>
<section>
 <div class="card"><h2>GPUs</h2><div id="gpus" class="note">reading…</div></div>
 <div class="card"><h2>Routing</h2><div class="note">For each subject, the workers to try first, in order. A task goes to the first one that is not paused, draining or too hot. Subjects with no route use the director's choice: the model with the best measured results, else the project's default, else the largest that fits.</div>
  <div id="routes"></div><div class="row" style="margin-top:.5rem"><button class="pri" id="saveRoutes">Save routes</button><span id="rmsg" class="note"></span></div></div>
 <div class="card"><h2>Add an endpoint</h2><div class="note">An OpenAI-compatible server you run yourself: llama-server, vLLM, LM Studio, Ollama (<code>/v1</code>). This Studio sends it tasks; it does the work.</div>
  <div class="row"><input id="eName" placeholder="name, e.g. heavy" style="width:9rem"><input id="eUrl" placeholder="http://192.168.1.20:8082/v1" class="grow"></div>
  <div class="row" style="margin-top:.35rem"><input id="eModel" placeholder="model id (blank: the server's own)" class="grow"><input id="eTok" type="password" placeholder="API key, if any" style="width:10rem"><input id="ePar" type="number" min="1" value="1" style="width:4rem" title="tasks at once"></div>
  <div class="row" style="margin-top:.35rem"><input id="eNote" placeholder="note: what it is good for" class="grow"><button class="pri" id="eAdd">Save</button></div><div id="emsg" class="note"></div></div>
 <div class="card"><h2>Activity</h2><div class="tscroll"><table id="events"></table></div></div>
</section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
let D=null, editing=false;
const KIND={local:'this computer',endpoint:'endpoint',remote:'remote host'};
const ago=t=>{ if(!t) return ''; const s=Math.round(Date.now()/1000-t); return s<60?s+'s':s<3600?Math.round(s/60)+'m':Math.round(s/3600)+'h'; };
const f0=v=>v==null?'?':Math.round(v);
function shown(w){ if(w.kind!=='local'||$('#allLocal').checked) return true;
  const c=w.control, st=w.stats; return w.status!=='stopped'||st.inflight||st.done||st.failed||c.paused||c.draining||c.max_parallel||c.takes.length||Object.values(D.routes).some(ids=>ids.includes(w.id)); }
function statusDot(w){ const s=w.status==='too hot'?'hot':w.status; return `<span class="dot ${esc(s)}" title="${esc(w.status)}"></span>`; }
function renderWorkers(){ const ws=D.workers.filter(shown);
  $('#workers').innerHTML='<tr><th>worker</th><th>where</th><th>tasks</th><th>results</th><th>controls</th></tr>'+(ws.length?ws.map(w=>{ const c=w.control, s=w.stats;
    const state=c.paused?'<span class="chip">paused</span>':c.draining?'<span class="chip">draining</span>':'';
    return `<tr data-w="${esc(w.id)}"><td>${statusDot(w)}<b>${esc(w.name)}</b> <span class="kind">${KIND[w.kind]}</span> ${state}
      ${w.model?`<div class="note">model ${esc(w.model)}</div>`:''}${w.note?`<div class="note">${esc(w.note)}</div>`:''}${w.error?`<div class="err">${esc(w.error)}</div>`:''}${c.takes.length?`<div>${c.takes.map(t=>`<span class="chip">${esc(t)}</span>`).join('')}</div>`:''}</td>
     <td class="note">${esc(w.where)}${w.port?` · port ${w.port}`:''}</td>
     <td>${s.inflight}${w.limit?' / '+w.limit:''} running</td>
     <td class="note">${s.done} done · ${s.failed} failed${s.avg_s!=null?`<br>${s.avg_s}s each`:''}${s.tok_s!=null?` · ${s.tok_s} tok/s`:''}${s.last_error&&s.last_error!==w.error?`<div class="err" title="${esc(s.last_error)}">${esc(s.last_error.slice(0,80))}</div>`:''}</td>
     <td><div class="ctl">${c.paused||c.draining?`<button data-a="resume">Resume</button>`:`<button data-a="pause" title="take no new tasks">Pause</button><button data-a="drain" title="finish what it has, then pause">Drain</button>`}
      <input type="number" min="1" data-a="par" value="${c.max_parallel??''}" placeholder="${w.limit??'∞'}" title="tasks at once (blank: no limit of its own)">
      ${w.kind!=='local'?'<button data-a="test">Test</button>':''}${w.kind==='local'&&w.status!=='stopped'?'<button data-a="stop" class="bad">Stop</button>':''}${w.kind==='endpoint'?'<button data-a="remove" class="bad">Remove</button>':''}
      <details><summary class="note">subjects</summary>${D.subjects.map(sj=>`<label style="display:inline-block;margin-right:.5rem"><input type="checkbox" data-take="${esc(sj)}" ${c.takes.includes(sj)?'checked':''}>${esc(sj)}</label>`).join('')}<div class="note">None ticked: any subject.</div></details></div>
      <div class="note" data-out></div></td></tr>`; }).join(''):'<tr><td colspan="5" class="note">No workers yet: add a model folder (Setup), an endpoint (right), or a remote host (Remote).</td></tr>');
  document.querySelectorAll('#workers tr[data-w]').forEach(tr=>{ const id=tr.dataset.w, out=tr.querySelector('[data-out]');
    const ctl=async b=>{ try{ await api('/api/workers/control',{id,...b}); load(); }catch(e){ out.innerHTML=`<span class="err">${esc(e.message)}</span>`; } };
    tr.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{ const a=b.dataset.a;
      if(a==='pause') ctl({paused:true}); if(a==='resume') ctl({paused:false,draining:false}); if(a==='drain') ctl({draining:true});
      if(a==='stop'&&confirm('Stop this model now? A task running on it fails and is retried.')) ctl({stop:true});
      if(a==='remove'&&confirm('Remove this endpoint?')){ await api('/api/workers/endpoint',{remove:id.slice(3)}); load(); }
      if(a==='test'){ out.textContent='asking…'; try{ const r=await api('/api/workers/test',{id}); out.textContent=`answered in ${r.seconds}s: ${r.reply.slice(0,80)}`; }catch(e){ out.innerHTML=`<span class="err">${esc(e.message)}</span>`; } } });
    const par=tr.querySelector('[data-a=par]'); par.onfocus=()=>editing=true; par.onblur=()=>editing=false;
    par.onchange=()=>ctl({max_parallel:par.value?+par.value:null});
    tr.querySelectorAll('[data-take]').forEach(x=>x.onchange=()=>ctl({takes:[...tr.querySelectorAll('[data-take]:checked')].map(y=>y.dataset.take)})); }); }
function tempColor(t){ return t==null?'var(--mut)':t>=90?'var(--no)':t>=80?'var(--warn)':'var(--ok)'; }
function gpuRow(g,where){ if(g.error) return `<div class="err">${esc(where)}: ${esc(g.error)}</div>`;
  const hot=Math.max(...[g.junction_c,g.temp_c,g.mem_c].filter(x=>x!=null),-1), mem=g.vram_total_mb?g.vram_used_mb/g.vram_total_mb:null;
  return `<div style="margin:.3rem 0"><div class="row"><b>${esc(g.name)}</b><span class="note">${esc(where)}</span><span class="grow"></span><span style="color:${tempColor(hot<0?null:hot)}">${hot<0?'? °C':Math.round(hot)+' °C'}</span></div>
   <div class="row note"><span>edge ${f0(g.temp_c)}° · junction ${f0(g.junction_c)}° · memory ${f0(g.mem_c)}°</span><span>load ${f0(g.util)}%</span>${g.power_w!=null?`<span>${f0(g.power_w)} W</span>`:''}</div>
   <div class="row"><span class="note">VRAM</span><div class="bar grow"><i style="width:${mem==null?0:Math.round(mem*100)}%;background:${mem>.92?'var(--no)':'var(--acc)'}"></i></div><span class="note">${g.vram_used_mb==null?'?':(g.vram_used_mb/1024).toFixed(1)} / ${g.vram_total_mb==null?'?':(g.vram_total_mb/1024).toFixed(1)} GB</span></div></div>`; }
function renderGpus(){ const T=D.telemetry; let h=(T.local||[]).map(g=>gpuRow(g,'this computer')).join('');
  for(const [host,t] of Object.entries(T.hosts||{})){ if(t.error){ h+=`<div class="err">${esc(host)}: ${esc(t.error)}</div>`; continue; }
    const th=t.thermal||{}; if(th.tripped) h+=`<div class="err"><b>${esc(host)} paused: ${esc(th.tripped.gpu)} at ${f0(th.tripped.temp)}°C</b></div>`;
    h+=(t.gpus||[]).map(g=>gpuRow(g,host+(th.enabled?` · guard at ${th.limit_c}°C`:''))).join('')||`<div class="note">${esc(host)}: no GPU sensors</div>`; }
  $('#gpus').innerHTML=h||'No GPU sensors found yet. NVIDIA needs nvidia-smi; AMD needs rocm-smi, or Linux’s amdgpu driver. Remote hosts report theirs once they run this version of OmniCortex Remote.';
  $('#upd').textContent=T.t?'updated '+ago(T.t)+' ago':''; }
function renderRoutes(){ if(editing) return; const opts=w=>`<option value="">—</option>`+D.workers.filter(x=>x.kind!=='local'||shown(x)||Object.values(D.routes).some(ids=>ids.includes(x.id))).map(x=>`<option value="${esc(x.id)}" ${x.id===w?'selected':''}>${esc(x.name)} (${KIND[x.kind]})</option>`).join('');
  $('#routes').innerHTML='<table style="table-layout:fixed"><colgroup><col style="width:6.5rem"><col><col><col></colgroup>'+D.subjects.map(s=>{ const r=D.routes[s]||[]; return `<tr><td>${esc(s)}</td>${[0,1,2].map(i=>`<td><select style="width:100%" data-route="${esc(s)}" data-i="${i}">${opts(r[i]||'')}</select></td>`).join('')}</tr>`; }).join('')+'</table>';
  document.querySelectorAll('[data-route]').forEach(x=>{ x.onfocus=()=>editing=true; x.onblur=()=>editing=false; }); }
$('#saveRoutes').onclick=async()=>{ const routes={}; document.querySelectorAll('[data-route]').forEach(x=>{ if(x.value) (routes[x.dataset.route]??=[]).push(x.value); });
  try{ await api('/api/workers/routes',{routes}); $('#rmsg').textContent='saved; used for tasks assigned from now on (approve a plan again to re-route it)'; editing=false; load(); }catch(e){ $('#rmsg').innerHTML=`<span class="err">${esc(e.message)}</span>`; } };
function renderQueue(){ if(editing) return; const q=D.queue;
  $('#queue').innerHTML=q.length?'<tr><th>task</th><th>worker</th><th>state</th></tr>'+q.map((t,i)=>`<tr><td><a href="/projects#${esc(t.project)}">${esc(t.project_title)}</a> · ${esc(t.task)}<div>${esc(t.title)}</div>${t.labels.map(l=>`<span class="chip">${esc(l)}</span>`).join('')}</td>
    <td>${t.status==='running'?esc(t.worker):`<select data-q="${i}">${D.workers.filter(w=>w.kind!=='local'||shown(w)||w.id===t.worker).map(w=>`<option value="${esc(w.id)}" ${w.id===t.worker?'selected':''}>${esc(w.name)} (${KIND[w.kind]})</option>`).join('')}</select>`}<div class="note">${esc(t.reason||'')}</div></td>
    <td class="note">${esc(t.why)}</td></tr>`).join(''):'<tr><td class="note">Nothing queued. Projects are planned and started on the <a href="/projects">Projects</a> page.</td></tr>';
  document.querySelectorAll('[data-q]').forEach(x=>{ x.onfocus=()=>editing=true; x.onblur=()=>editing=false;
    x.onchange=async()=>{ const t=D.queue[+x.dataset.q]; try{ await api(`/api/projects/${encodeURIComponent(t.project)}/edit`,{changes:[{op:'assign',id:t.task,model:x.value}],reason:'moved on the Workers page'}); }catch(e){ alert(e.message); } editing=false; load(); }; }); }
function renderEvents(){ $('#events').innerHTML=D.events.length?D.events.map(e=>`<tr><td class="note">${ago(e.t)}</td><td>${esc(e.worker)}</td><td>${esc(e.what)}</td><td class="note">${esc(e.detail)}</td></tr>`).join(''):'<tr><td class="note">Nothing yet.</td></tr>'; }
function renderTiles(){ const ws=D.workers, up=ws.filter(w=>['up','running','loaded','available'].includes(w.status)).length;
  const run=ws.reduce((a,w)=>a+w.stats.inflight,0), done=ws.reduce((a,w)=>a+w.stats.done,0), fail=ws.reduce((a,w)=>a+w.stats.failed,0);
  const waiting=D.queue.filter(t=>t.status!=='running').length, paused=ws.filter(w=>w.control.paused||w.control.draining).length;
  $('#tiles').innerHTML=[[run,'tasks running'],[waiting,'tasks waiting'],[done,'done since start'],[fail,'failed'],[`${up}`,'workers ready'],[paused,'paused or draining']].map(([v,l])=>`<div class="tile"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join(''); }
async function load(){ try{ D=await api('/api/workers/overview'); $('#werr').textContent=''; }catch(e){ $('#werr').textContent=e.message; return; }
  if(!editing){ renderWorkers(); renderRoutes(); renderQueue(); } renderGpus(); renderEvents(); renderTiles(); }
$('#allLocal').onchange=()=>{ renderWorkers(); renderRoutes(); renderQueue(); };
$('#eAdd').onclick=async()=>{ try{ await api('/api/workers/endpoint',{name:$('#eName').value,url:$('#eUrl').value,model:$('#eModel').value,token:$('#eTok').value||undefined,max_parallel:+$('#ePar').value||1,note:$('#eNote').value});
  $('#emsg').textContent='saved; Test it on the left'; ['eName','eUrl','eModel','eTok','eNote'].forEach(i=>$('#'+i).value=''); load(); }catch(e){ $('#emsg').innerHTML=`<span class="err">${esc(e.message)}</span>`; } };
load(); setInterval(()=>{ if(!document.hidden) load(); },3000);
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
