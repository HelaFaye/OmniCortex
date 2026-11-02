"""Studio's /projects page: the director's plan, proposals and reviews.

Served by viz/studio.py to the owner only. Everything a worker wrote is shown
as text (escaped), never as markup.
"""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Projects</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
a{color:var(--acc)}
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0;--chip:#eef1fb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--chip:#22283a}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--chip:#22283a}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{display:grid;grid-template-columns:minmax(240px,320px) 1fr;gap:1rem;padding:1rem;max-width:1500px;margin:0 auto}
main>*{min-width:0}
@media (max-width:860px){main{grid-template-columns:1fr;padding:.6rem}#view{order:-1}header{flex-wrap:wrap;gap:.5rem}}
.tscroll{overflow-x:auto}
@media (max-width:600px){.hide-s{display:none}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem;margin-bottom:1rem}
h2{font-size:12px;margin:.1rem 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
label{display:block;font-size:12px;color:var(--mut);margin:.45rem 0 .1rem}
input,select,textarea{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
textarea{min-height:5rem;resize:vertical}input[type=checkbox]{width:auto;margin-right:.3rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.35rem .75rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}
button:disabled{opacity:.5;cursor:default}
.note{font-size:12px;color:var(--mut)}.err{color:var(--no);font-size:12px;white-space:pre-wrap}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.grow{flex:1}
.proj{border-bottom:1px solid var(--line);padding:.45rem .2rem;cursor:pointer}.proj:hover,.proj.on{background:var(--code)}
.st{font-size:11px;padding:0 .4rem;border-radius:4px;border:1px solid var(--line);white-space:nowrap}
.st.running,.st.review{color:var(--acc)}.st.done,.st.approved{color:var(--ok)}.st.blocked,.st.rework{color:var(--warn)}
.st.dropped{opacity:.5;text-decoration:line-through}.st.draft,.st.paused,.st.todo{color:var(--mut)}
.chips{display:flex;flex-wrap:wrap;gap:.25rem;margin-top:.15rem}
.chipb{font-size:12px;padding:.1rem .5rem;border-radius:9px;border:1px solid var(--line);background:transparent;color:var(--mut);cursor:pointer;max-width:14rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chipb.on{background:var(--chip);color:var(--fg);border-color:var(--acc,var(--line))}
.chip{display:inline-block;font-size:11px;padding:0 .45rem;border-radius:9px;background:var(--chip);margin:0 .2rem .2rem 0}
.chip.unknown{background:transparent;border:1px dashed var(--line);color:var(--mut)}
table{width:100%;border-collapse:collapse}td,th{padding:.4rem .35rem;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;font-size:13px}
th{font-size:11px;color:var(--mut);font-weight:500;text-transform:uppercase}
tr.task{cursor:pointer}tr.task:hover{background:var(--code)}tr.open{background:var(--code)}
.detail{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.6rem;margin:.2rem 0 .6rem}
pre{background:var(--code);padding:.6rem;border-radius:6px;max-height:50vh;overflow:auto;font:12px ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;margin:.4rem 0}
.attn{border-left:3px solid var(--acc);padding:.4rem .6rem;margin:.4rem 0;background:var(--bg);border-radius:0 6px 6px 0}
.attn.warn{border-left-color:var(--warn)}
.st.ok{color:var(--ok)}.st.missing{color:var(--no)}.st.warn{color:var(--warn)}.st.note{color:var(--mut)}
.src{font-size:10.5px;color:var(--mut);border:1px dashed var(--line);border-radius:4px;padding:0 .3rem}
.rq td{font-size:12.5px}.rq code{font-size:11.5px;word-break:break-word}
details summary{cursor:pointer;color:var(--mut);font-size:12px}
.risk{font-size:11px;padding:0 .4rem;border-radius:4px}.risk.hi{background:color-mix(in srgb,var(--no) 25%,transparent)}
.risk.lo{background:color-mix(in srgb,var(--ok) 20%,transparent)}
.empty{color:var(--mut);padding:2rem;text-align:center}
</style>
<header><h1>Studio · Projects</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a><span class="note" id="hint"></span></header>
<main>
<div>
 <section class="card"><h2>Projects</h2><div id="plist" class="note">none yet</div></section>
 <section class="card"><h2>New project</h2>
  <label>title</label><input id="ntitle" placeholder="Renderer rewrite">
  <label>goal (one or two sentences)</label><input id="ngoal">
  <label>description: notes, a skill list, an issue or a README. Lists and tables become candidate tasks</label>
  <textarea id="ntext" style="min-height:9rem"></textarea>
  <label>local checkout to survey (optional): which skills the code needs</label><input id="nrepo" placeholder="/home/me/src/project">
  <button class="pri" id="create" style="margin-top:.6rem;width:100%">Analyze into a draft plan</button>
  <div id="nerr" class="err"></div>
  <div class="note" style="margin-top:.4rem">Nothing runs until you approve the plan and start it.</div>
 </section>
 <section class="card"><h2>Worker models</h2><div id="workers" class="note">none running</div></section>
 <section class="card"><h2>Devices</h2><div id="devices" class="note">…</div></section>
</div>
<div id="view"><div class="card empty">Pick a project, or describe a new one.</div></div>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
let cur=null, plan=null, meta={models:[],subjects:[],checks:false}, open=new Set(), timer=null;
const chips=ls=>(ls&&ls.length?ls:['unknown']).map(l=>`<span class="chip ${l==='unknown'?'unknown':''}">${esc(l)}</span>`).join('');
const ago=t=>{const s=Math.round(Date.now()/1000-t);return s<90?s+'s ago':s<5400?Math.round(s/60)+' min ago':Math.round(s/3600)+' h ago'};

async function loadList(){
  const j=await api('/api/projects'); meta=j;
  $('#plist').innerHTML=j.projects.length?j.projects.map(p=>{const n=Object.entries(p.tasks).map(([k,v])=>`${v} ${k}`).join(', ');
    return `<div class="proj ${p.id===cur?'on':''}" data-id="${esc(p.id)}"><div class="row"><b class="grow">${esc(p.title)}</b><span class="st ${esc(p.status)}">${esc(p.status)}</span></div>
      <div class="note">${esc(n)}${p.pending?` · <b>${p.pending} proposal(s)</b>`:''} · ${ago(p.updated)}</div></div>`}).join(''):'none yet';
  document.querySelectorAll('.proj').forEach(e=>e.onclick=()=>{cur=e.dataset.id;open.clear();load()});
}
let devices=[];
const gib=b=>b==null?'?':(b/2**30).toFixed(1);
async function loadWorkers(){
  const j=await api('/api/workers'); devices=j.devices||[];
  $('#devices').innerHTML=devices.map(d=>`<div style="margin:.25rem 0;${d.enabled?'':'opacity:.55'}"><b>${esc(d.id)}</b> ${esc(d.name)}
     <div class="note">${d.memory_total==null?'memory unknown':gib(d.memory_free)+' of '+gib(d.memory_total)+' GiB free'+(d.estimated?' (estimated)':'')}${d.unified&&d.backend!=='cpu'?' · shared with system RAM':''}${d.enabled?'':' · off'}${d.twin_of?' · same GPU as '+esc(d.twin_of):''}</div>
     ${d.note?`<div class="note" style="color:var(--warn)">${esc(d.note)}</div>`:''}</div>`).join('')+
    `<div class="note">settings: ${esc(j.hardware_config||'')}</div>`;
  $('#workers').innerHTML=j.workers.length?j.workers.map(w=>`<div class="row" style="margin:.2rem 0"><span class="grow">${esc(w.model)}<br><span class="note">${esc(w.where)} · port ${w.port} · ${w.loading?'loading':w.busy?'busy':'idle '+w.idle_s+'s'}</span></span>
     <button data-m="${esc(w.model)}">Stop</button></div>`).join('')+`<div class="note">up to ${j.max_workers} at once; idle ones stop on their own</div>`
    :`none running (up to ${j.max_workers} at once)`;
  document.querySelectorAll('#workers button').forEach(b=>b.onclick=async()=>{await api('/api/workers/stop',{model:b.dataset.m});loadWorkers()});
}
$('#create').onclick=async()=>{ $('#nerr').textContent='';
  try{const j=await api('/api/projects',{title:$('#ntitle').value,goal:$('#ngoal').value,text:$('#ntext').value,repo:$('#nrepo').value.trim()||undefined});
    cur=j.id; $('#ntext').value=''; await loadList(); load();}catch(e){$('#nerr').textContent=e.message}};

// Errors show on the page (with links to where to fix them), not in an alert.
function problem(m){ let b=$('#problem'); if(!b){ b=document.createElement('div'); b.id='problem'; b.className='card attn warn'; $('#view').prepend(b); }
  b.innerHTML=m?`${linkify(esc(m))} <button style="float:right" onclick="this.parentNode.remove()">×</button>`:''; if(!m) b.remove(); else b.scrollIntoView({block:'nearest'}); }
async function act(action,body){ try{ await api(`/api/projects/${cur}/${action}`,body||{}); }catch(e){ await load(); return problem(e.message); } await load(); }

function modelOpts(sel,blank){return (blank?`<option value="">${blank}</option>`:'')+meta.models.map(m=>`<option ${m.id===sel?'selected':''} value="${esc(m.id)}">${esc(m.id)} (${(m.size/2**30).toFixed(1)} GiB)</option>`).join('')}

function summarize(ch){ if(ch.op==='add') return `add “${esc(ch.task.title)}” [${esc((ch.task.labels||[]).join(', ')||'unknown')}]${ch.task.depends_on&&ch.task.depends_on.length?' after '+esc(ch.task.depends_on.join(',')):''}`;
  if(ch.op==='assign') return `assign ${esc(ch.id)} to ${esc(ch.model)}`; if(ch.op==='update') return `change ${esc(ch.id)}: ${esc(Object.keys(ch.fields||{}).join(', '))}`;
  return `${esc(ch.op)} ${esc(ch.id||'')}`; }

function attemptHtml(t,a,last){
  const dur=a.finished?Math.round(a.finished-a.started)+'s':'running…';
  const rep=a.report||{}; const ck=a.check;
  const risk=ck&&ck.max!=null?`<a class="risk ${ck.flagged?'hi':'lo'}" href="${esc(ck.url)}" target="_blank" title="activation check: open the 3D view">${ck.flagged?ck.n_flagged+' flagged token(s)':'no flags'} · peak ${(ck.max*100).toFixed(0)}%</a>`
    :ck&&ck.error?`<span class="risk hi">check failed</span>`:(meta.checks?'<span class="note">unchecked (no classifier for this model)</span>':'');
  const sc=a.screen, scr=sc?`<a class="risk ${sc.status==='passed'?'lo':'hi'}" href="/graphs#graph=${encodeURIComponent(sc.graph||'')}" title="${esc(sc.feedback||'')}">graph ${esc(sc.graph||'')}: ${esc({passed:'passed',failed:'failed',person:'asks a person',error:'did not finish'}[sc.status]||sc.status)}</a>`:'';
  const rv=a.review?(a.review.accepted?`<span class="st done">accepted by ${esc(a.review.by)}</span>`:`<span class="st rework">sent back</span> <span class="note">${esc(a.review.feedback)}</span>`):'';
  let h=`<div style="margin:.5rem 0"><div class="row"><b>attempt ${a.n}</b><span class="note">${esc(a.model)} · ${dur}</span>
    ${rep.status?`<span class="st ${rep.status==='done'?'done':rep.status==='blocked'?'blocked':'todo'}">reported ${esc(rep.status)}</span>`:''}${risk}${scr}${rv}</div>`;
  if(sc&&sc.status!=='passed'&&sc.feedback) h+=`<div class="note">verification: ${esc(sc.feedback).slice(0,600)}</div>`;
  if(a.error) h+=`<div class="err">${esc(a.error)}</div>`;
  if(rep.summary) h+=`<div class="note">${esc(rep.summary)}</div>`;
  if(a.text) h+=`<details ${last?'open':''}><summary>worker's reply (${a.text.length} chars)</summary><pre>${esc(a.text)}</pre></details>`;
  return h+'</div>';
}

// A set picked from known values: toggle chips over a hidden input that holds "a, b, c" for the save code.
function chipPick(field,options,selected,tipFor){
  const on=new Set(selected||[]), extra=[...on].filter(x=>!options.includes(x));
  return `<input type="hidden" data-f="${field}" value="${esc([...on].join(', '))}"><div class="chips">`+
    [...options,...extra].map(o=>`<button type="button" class="chipb${on.has(o)?' on':''}" data-chip="${field}" data-v="${esc(o)}" title="${esc((tipFor&&tipFor(o))||o)}">${esc(o)}</button>`).join('')+
    (options.length||extra.length?'':'<span class="note">none</span>')+'</div>'; }
document.addEventListener('click',e=>{ const b=e.target.closest('[data-chip]'); if(!b) return;
  b.classList.toggle('on'); const box=b.parentNode, inp=box.previousElementSibling;
  inp.value=[...box.querySelectorAll('.chipb.on')].map(x=>x.dataset.v).join(', '); });

function taskDetail(t){
  const draft=plan.status==='draft';
  let h=`<div class="detail">`;
  h+=`<label>detail</label><textarea data-f="detail">${esc(t.detail)}</textarea>
   <label>done means (acceptance criteria a reviewer can check)</label><input data-f="acceptance" value="${esc(t.acceptance)}">
   <div class="row"><div class="grow"><label title="Pick the skills this task needs: they decide which model takes it">skills</label>${chipPick('labels',meta.subjects,t.labels)}</div>
   <div class="grow"><label title="Tasks that must be done before this one starts">after tasks</label>${chipPick('depends_on',plan.tasks.map(x=>x.id).filter(x=>x!==t.id),t.depends_on,id=>{const o=plan.tasks.find(x=>x.id===id);return o?o.title:''})}</div>
   <div class="grow"><label>model (pin)</label><select data-f="model">${modelOpts(t.model,'assign automatically')}</select></div></div>
   <div class="row" style="margin-top:.5rem"><button class="pri" data-a="save">Save task</button>
   ${t.status!=='dropped'&&t.status!=='running'?'<button class="bad" data-a="drop">Drop</button>':''}
   ${['done','dropped','blocked'].includes(t.status)?'<button data-a="reopen">Reopen</button>':''}
   <span class="note">${t.why?'classifier: '+esc(t.why):''}${t.assignee?' · model: '+esc(t.assignee.reason):''}${!draft?' · your edits apply at once and are logged':''}</span></div>`;
  const openQ=(t.questions||[]).filter(q=>!q.answer);
  if(t.status==='blocked'&&openQ.length){ h+=`<div class="attn warn"><b>The worker needs:</b>`+openQ.map((q,i)=>`<label>${esc(q.q)}</label><input data-q="${i}">`).join('')+
    `<button class="pri" data-a="answer" style="margin-top:.4rem">Answer and retry</button></div>`; }
  if(t.status==='review'||(t.status==='blocked'&&t.attempts.length&&!openQ.length)){ h+=`<div class="attn"><b>Review attempt ${t.attempts.length}</b>
    <label>feedback for the worker (sent with the next attempt if you send it back)</label><textarea data-f="feedback" style="min-height:3rem"></textarea>
    <div class="row" style="margin-top:.4rem"><button class="pri" data-a="accept">Accept</button><button data-a="reject">Send back</button></div></div>`; }
  t.attempts.slice().reverse().forEach((a,i)=>h+=attemptHtml(t,a,i===0));
  return h+'</div>';
}

function render(){
  const p=plan, live=p.tasks.filter(t=>t.status!=='dropped');
  const pend=p.proposals.filter(x=>x.status==='pending');
  const counts={}; live.forEach(t=>counts[t.status]=(counts[t.status]||0)+1);
  let h=`<section class="card"><div class="row"><h1 class="grow" style="font-size:18px;margin:0">${esc(p.title)}</h1>
    <span class="st ${esc(p.status)}">${esc(p.status)}</span><span class="note">v${p.version}</span></div>
    ${p.goal?`<div style="margin:.3rem 0">${esc(p.goal)}</div>`:''}
    <div class="note">${esc(Object.entries(counts).map(([k,v])=>v+' '+k).join(' · '))}</div>
    <div class="row" style="margin-top:.6rem">`;
  if(p.status==='draft') h+=`<select id="rmodel" style="width:auto">${modelOpts(p.policy.director_model||'', 'model to re-plan with…')}</select>
      <button id="refine" ${p.note&&p.note.busy?'disabled':''}>Re-plan with model</button>
      <button class="pri" id="approve" title="assigns a model to every task and freezes the plan">Approve plan</button>`;
  if(['approved','paused'].includes(p.status)) h+=`<button class="pri" id="start">Start</button>`;
  if(p.status==='running') h+=`<button id="pause">Pause</button>`;
  if(p.status!=='running') h+=`<button class="bad" id="del" style="margin-left:auto">Delete</button>`;
  h+=`</div>${p.note?`<div class="note" style="margin-top:.3rem">${esc(p.note.text)}</div>`:''}
    <div style="margin-top:.6rem">${Object.entries(p.skills).map(([k,v])=>`<span class="chip ${k==='unknown'?'unknown':''}" title="${esc(v.join(', '))}">${esc(k)} ${v.length}</span>`).join('')}</div>
    ${p.gaps&&p.gaps.length?`<div class="attn warn">The code base needs <b>${esc(p.gaps.join(', '))}</b>, but no task covers it.</div>`:''}
    ${p.census?`<div class="note">surveyed ${esc(p.census.path)}: ${esc(Object.entries(p.census.subjects).map(([k,v])=>k+' '+v.files).join(', '))}</div>`:''}
    <details style="margin-top:.5rem"><summary>policy</summary><div class="row">
      <div class="grow"><label>review</label><select data-p="review">${['all','flagged','none'].map(v=>`<option ${p.policy.review===v?'selected':''} value="${v}">${{all:'every result waits for me',flagged:'only flagged or reported problems',none:'accept without review'}[v]}</option>`).join('')}</select></div>
      <div><label>parallel</label><input data-p="max_parallel" type="number" min="1" max="16" value="${p.policy.max_parallel}" style="width:5rem"></div>
      <div><label>attempts</label><input data-p="max_attempts" type="number" min="1" max="16" value="${p.policy.max_attempts}" style="width:5rem"></div>
      <div class="grow"><label>default model (when none has graded results)</label><select data-p="default_model">${modelOpts(p.policy.default_model,'largest that fits')}</select></div>
      <div><label><input type="checkbox" data-p="check" ${p.policy.check?'checked':''}>activation check</label></div></div>
    <div class="row"><div class="grow"><label>verification graph (screens each result before review; inputs task and result, outputs accepted or feedback)</label>
      <input data-p="verify_graph" list="graphList" value="${esc(p.policy.verify_graph||'')}" placeholder="none · e.g. template:verify-delegated-result"></div>
      <div><label><input type="checkbox" data-p="verify_auto" ${p.policy.verify_auto?'checked':''}>send failures back automatically</label></div>
      <a class="note" href="/graphs#graph=template:verify-delegated-result" style="align-self:end">edit graphs</a></div>
    <datalist id="graphList"></datalist>
      <label>devices this project may use (none ticked = any enabled device; AMD first)</label>
      <div class="row">${devices.map(d=>`<label style="margin:0"><input type="checkbox" data-dev="${esc(d.id)}" ${(p.policy.devices||[]).includes(d.id)?'checked':''} ${d.enabled?'':'disabled'}>${esc(d.id)}</label>`).join('')||'<span class="note">no devices detected</span>'}</div>
      <label>llama-server settings per device for this project, JSON (ngl, ctx, batch, threads, flash_attn, cache_type, parallel)</label>
      <textarea id="devset" style="min-height:3rem" placeholder='{"vulkan:0": {"ctx": 16384, "threads": 8}}'>${esc(Object.keys(p.policy.device_settings||{}).length?JSON.stringify(p.policy.device_settings,null,1):'')}</textarea>
      <button id="savedev" style="margin-top:.3rem">Save device settings</button></details>
  </section>`;
  if(pend.length){ h+=`<section class="card"><h2>Proposed changes (${pend.length})</h2>`+pend.map(x=>`<div class="attn"><div class="row"><b class="grow">${esc(x.id)} · ${esc(x.reason)}</b>
      <button class="pri" data-pa="${esc(x.id)}">Accept</button><button data-pr="${esc(x.id)}">Reject</button></div>
      <div class="note">${x.changes.map(summarize).join('<br>')}</div></div>`).join('')+`</section>`; }
  h+=`<section class="card"><h2>Tasks</h2><div class="tscroll"><table><tr><th></th><th>task</th><th>skills</th><th class="hide-s">after</th><th class="hide-s">model</th><th>status</th></tr>`;
  for(const t of p.tasks){ const isOpen=open.has(t.id); const ready=p.ready.includes(t.id);
    const needs=t.status==='review'?' · <b>needs review</b>':t.status==='blocked'?' · <b>blocked</b>':'';
    h+=`<tr class="task ${isOpen?'open':''}" data-t="${esc(t.id)}"><td class="note">${esc(t.id)}</td><td><b>${esc(t.title)}</b><div class="note">${esc(t.detail.slice(0,140))}${t.detail.length>140?'…':''}</div></td>
      <td>${chips(t.labels)}</td><td class="note hide-s">${esc(t.depends_on.join(', '))}</td>
      <td class="note hide-s">${esc((t.assignee&&t.assignee.model)||t.model||'—')}</td>
      <td><span class="st ${esc(t.status)}">${esc(t.status)}</span>${ready?'<div class="note">ready</div>':''}<span class="note">${needs}</span></td></tr>`;
    if(isOpen) h+=`<tr><td></td><td colspan="5" data-d="${esc(t.id)}">${taskDetail(t)}</td></tr>`; }
  h+=`</table></div><details style="margin-top:.6rem"><summary>add a task</summary><label>title</label><input id="atitle"><label>detail</label><textarea id="adetail"></textarea>
     <button id="add" style="margin-top:.4rem">Add task</button></details></section>`;
  h+=reqCard(p);
  h+=`<section class="card"><details><summary>history (${p.history.length})</summary>`+p.history.slice().reverse().map(e=>`<div class="note">v${e.version} · ${ago(e.at)} · <b>${esc(e.by)}</b> ${esc(e.action)}${e.reason?': '+esc(e.reason):''}</div>`).join('')+`</details></section>`;
  $('#view').innerHTML=h; wire();
}

// ---------- per-project requirements (scripts/ns_requirements.py)
function reqCard(p){
  const r=p.requirements||{items:[]}, last=r.last, byKey={};
  if(last) for(const i of last.items) byKey[i.kind+':'+i.name.toLowerCase()]=i;
  const rows=r.items.map((i,n)=>{ const c=byKey[i.kind+':'+i.name.toLowerCase()]||{};
    const label=i.kind==='feature'?(meta.req_features[i.name]||i.name):i.kind==='hardware'?(meta.req_hardware[i.name]||i.name):i.name;
    return `<tr><td><span class="st ${esc(c.status||'')}">${esc(c.status||'unchecked')}</span></td><td class="note">${esc(i.kind)}</td>
      <td><b>${esc(label)}</b>${i.optional?' <span class="note">(optional)</span>':''} <span class="src" title="${esc(i.why||'')}">${esc(i.source)}</span>
        ${i.why?`<div class="note">${esc(i.why)}</div>`:''}</td>
      <td class="note">${esc(i.need||'')}</td><td>${c.have!=null?`<code>${esc(c.have)}</code>`:''}${c.fix?`<div class="note">fix: <code>${linkify(esc(c.fix))}</code></div>`:''}</td>
      <td><button data-rq-del="${n}" title="remove">×</button></td></tr>`; }).join('');
  const summary=last?`${last.status==='ok'?'<span class="st ok">ok</span> everything required is here':last.status==='missing'?`<span class="st missing">missing</span> ${esc(last.missing.join(', '))}`:'<span class="st warn">warn</span> required items are here; some optional ones are not'} · checked ${ago(last.checked)}${last.python_from?` · Python packages checked in ${esc(last.python_from)}`:''}`:'not checked yet';
  const kindOpts=meta.req_kinds.map(k=>`<option>${k}</option>`).join('');
  return `<section class="card"><h2>Requirements</h2>
    <div class="note">What this project needs on the machine that builds it: the toolchain, libraries and Python packages from its build files, hardware, OmniCortex features, and your notes. Workers are told what is available; Start waits until required items are here (or you start anyway).</div>
    <div class="row" style="margin:.5rem 0"><span class="grow">${summary}</span><button id="rqcheck">Check now</button></div>
    ${r.items.length?`<div class="tscroll"><table class="rq"><tr><th></th><th>kind</th><th>what</th><th>needs</th><th>this machine</th><th></th></tr>${rows}</table></div>`:'<div class="note">No requirements yet. Scan the checkout, or add them below.</div>'}
    <div class="row" style="margin-top:.6rem"><input id="rqroot" class="grow" data-pick="dir" placeholder="project checkout folder" value="${esc(r.root||(p.census&&p.census.path)||'')}"><button id="rqscan">Scan build files</button></div>
    ${r.files&&r.files.length?`<div class="note">read: ${esc(r.files.join(', '))}</div>`:''}
    <details style="margin-top:.5rem"><summary>add a requirement</summary><div class="row">
      <select id="rqkind" style="width:auto">${kindOpts}</select><span id="rqnamebox" class="grow"></span>
      <input id="rqneed" placeholder="version, e.g. >=3.20" style="width:11rem">
      <label style="margin:0"><input type="checkbox" id="rqopt">optional</label><button id="rqadd">Add</button></div>
      <input id="rqwhy" placeholder="why (optional)" style="margin-top:.3rem"></details></section>`;
}
function rqNameField(){ const k=$('#rqkind').value, box=$('#rqnamebox'); if(!box) return;
  box.innerHTML=k==='feature'?`<select id="rqname">${Object.entries(meta.req_features).map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('')}</select>`
    :k==='hardware'?`<select id="rqname">${Object.entries(meta.req_hardware).map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('')}</select>`
    :`<input id="rqname" placeholder="${{tool:'e.g. cmake, meson, ninja',library:'pkg-config name, e.g. glfw3',package:'Python package, e.g. numpy',note:'anything to check by hand'}[k]||''}">`;
  $('#rqneed').placeholder=k==='hardware'?'GiB, or amd/nvidia/intel for gpu':'version, e.g. >=3.20'; $('#rqneed').disabled=['feature','note'].includes(k); }
async function saveReqs(items,reason){ try{ await api(`/api/projects/${cur}/requirements`,{items,reason}); await api(`/api/projects/${cur}/requirements`); }catch(e){ alert(e.message); } await load(); }
async function startProject(){
  const r=await fetch(`/api/projects/${cur}/start`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); const j=await r.json().catch(()=>({}));
  if(r.status===409&&j.missing){ if(confirm(`This machine is missing what the project needs:\n\n  ${j.missing.join('\n  ')}\n\nStart anyway? Workers will be told these are not available.`)) await act('start',{force:true}); else await load(); return; }
  await load(); if(!r.ok) problem(j.error||String(r.status));
}

function wire(){
  const on=(s,f)=>{const e=$(s); if(e) e.onclick=f};
  on('#approve',()=>act('approve')); on('#start',startProject);
  if($('#rqkind')){ $('#rqkind').onchange=rqNameField; rqNameField(); }
  on('#rqcheck',async()=>{ $('#rqcheck').disabled=true; $('#rqcheck').textContent='checking…'; try{ await api(`/api/projects/${cur}/requirements`); }catch(e){ alert(e.message); } await load(); });
  on('#rqscan',async()=>{ try{ await api(`/api/projects/${cur}/requirements/scan`,{root:$('#rqroot').value}); await api(`/api/projects/${cur}/requirements`); }catch(e){ alert(e.message); } await load(); });
  on('#rqadd',()=>{ const it={kind:$('#rqkind').value,name:($('#rqname').value||'').trim(),need:$('#rqneed').disabled?'':$('#rqneed').value.trim(),optional:$('#rqopt').checked,why:$('#rqwhy').value.trim()};
    if(!it.name) return alert('name it'); saveReqs([...plan.requirements.items,it],'added by hand'); });
  document.querySelectorAll('[data-rq-del]').forEach(b=>b.onclick=()=>{ const n=+b.dataset.rqDel; const it=plan.requirements.items[n];
    saveReqs(plan.requirements.items.filter((_,k)=>k!==n),`removed ${it.kind} ${it.name}`); }); on('#pause',()=>act('pause'));
  on('#refine',()=>{const m=$('#rmodel').value; if(!m) return alert('choose a model to re-plan with'); act('refine',{model:m})});
  on('#del',async()=>{ if(confirm('Delete this project and its history?')){ await api(`/api/projects/${cur}/delete`,{}); cur=null; plan=null; $('#view').innerHTML='<div class="card empty">Deleted.</div>'; loadList(); }});
  on('#add',()=>act('edit',{changes:[{op:'add',task:{title:$('#atitle').value,detail:$('#adetail').value}}],reason:'added by hand'}));
  document.querySelectorAll('[data-p]').forEach(e=>e.onchange=()=>{const k=e.dataset.p; let v=e.type==='checkbox'?e.checked:e.value; if(e.type==='number') v=+v; if(k==='default_model'&&!v) v=null; act('policy',{policy:{[k]:v}})});
  document.querySelectorAll('[data-dev]').forEach(e=>e.onchange=()=>act('policy',{policy:{devices:[...document.querySelectorAll('[data-dev]:checked')].map(x=>x.dataset.dev)}}));
  fetch('/api/graphs').then(r=>r.json()).then(j=>{ const d=document.getElementById('graphList'); if(d) d.innerHTML=[...(j.graphs||[]).map(g=>g.name),...(j.templates||[]).map(t=>'template:'+t.name)].map(n=>`<option value="${esc(n)}">`).join(''); }).catch(()=>{});
  on('#savedev',()=>{ let v={}; const t=$('#devset').value.trim(); if(t){ try{ v=JSON.parse(t) }catch(e){ return alert('not JSON: '+e.message) } } act('policy',{policy:{device_settings:v}}) });
  document.querySelectorAll('[data-pa]').forEach(b=>b.onclick=()=>act('decide',{proposal:b.dataset.pa,accept:true}));
  document.querySelectorAll('[data-pr]').forEach(b=>b.onclick=()=>act('decide',{proposal:b.dataset.pr,accept:false}));
  document.querySelectorAll('tr.task').forEach(r=>r.onclick=()=>{const id=r.dataset.t; open.has(id)?open.delete(id):open.add(id); render()});
  document.querySelectorAll('[data-d]').forEach(box=>{ const id=box.dataset.d, f=n=>box.querySelector(`[data-f=${n}]`);
    box.querySelectorAll('button[data-a]').forEach(b=>b.onclick=()=>{ const a=b.dataset.a;
      if(a==='save'){ const split=v=>v.split(',').map(x=>x.trim()).filter(Boolean);
        act('edit',{changes:[{op:'update',id,fields:{detail:f('detail').value,acceptance:f('acceptance').value,labels:split(f('labels').value),depends_on:split(f('depends_on').value),model:f('model').value||null}}],reason:'edited by hand'}); }
      if(a==='drop') act('edit',{changes:[{op:'drop',id}],reason:'dropped by hand'});
      if(a==='reopen') act('edit',{changes:[{op:'reopen',id}],reason:'reopened by hand'});
      if(a==='accept') act('review',{task:id,accept:true,feedback:f('feedback').value});
      if(a==='reject'){ if(!f('feedback').value.trim()&&!confirm('Send back without saying what is wrong?')) return; act('review',{task:id,accept:false,feedback:f('feedback').value}); }
      if(a==='answer') act('answer',{task:id,answers:[...box.querySelectorAll('[data-q]')].map(x=>x.value)}); }); });
}

async function load(){
  if(!cur) return;
  const typing=document.activeElement&&['TEXTAREA','INPUT','SELECT'].includes(document.activeElement.tagName)&&$('#view').contains(document.activeElement);
  try{ const p=await api('/api/projects/'+cur); const same=plan&&plan.id===p.id&&plan.updated===p.updated&&JSON.stringify(plan.note)===JSON.stringify(p.note);
    plan=p; if(!(same||typing)) render(); }catch(e){ $('#view').innerHTML=`<div class="card err">${esc(e.message)}</div>`; }
  loadList(); loadWorkers();
}
loadList().catch(e=>$('#plist').textContent=e.message); loadWorkers().catch(()=>{});
setInterval(()=>{ if(cur&&plan&&(plan.status==='running'||(plan.note&&plan.note.busy))) load(); else loadWorkers().catch(()=>{}); },3000);
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
