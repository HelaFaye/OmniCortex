"""Studio's /setup page: everything that used to need a terminal after install.
Paths and behaviour (saved to ~/.neuronscope/config.json), building llama.cpp,
hardware (hardware.json), and the doctor's checks."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Setup</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
a{color:var(--acc)}
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{max-width:1000px;margin:0 auto;padding:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem;margin-bottom:1rem}
h2{font-size:15px;margin:.1rem 0 .5rem}.note{font-size:12.5px;color:var(--mut)}
label{display:block;font-size:12.5px;margin:.6rem 0 .15rem}label .note{display:block}
input,select,textarea{font:12.5px ui-monospace,monospace;padding:.35rem .45rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto;margin-right:.35rem}textarea{min-height:3.2rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button:disabled{opacity:.5}
.row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}.grow{flex:1;min-width:12rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 1rem}@media (max-width:720px){.grid{grid-template-columns:1fr}}
.st{display:inline-block;min-width:3.2rem;text-align:center;font:600 11px ui-sans-serif;padding:.05rem .4rem;border-radius:4px;border:1px solid var(--line)}
.st.ok{color:var(--ok)}.st.warn{color:var(--warn)}.st.fail{color:var(--no)}.st.skip{color:var(--mut)}
.err{color:var(--no);font-size:12.5px;white-space:pre-wrap}.okm{color:var(--ok);font-size:12.5px}
pre{background:var(--code);padding:.6rem;border-radius:6px;max-height:40vh;overflow:auto;font:12px ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}
table{width:100%;border-collapse:collapse}td,th{padding:.4rem .3rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:500}
.next{border-left:3px solid var(--acc);padding:.5rem .7rem;background:var(--bg);border-radius:0 6px 6px 0;margin-top:.5rem}
.dstage{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:.7rem 0 .2rem}
.card{scroll-margin-top:1rem}.card:target{border-color:var(--acc)}
.check{display:flex;gap:.5rem;padding:.15rem 0;font-size:13px}.check .fix{color:var(--mut);font-size:12px}
.st.missing{color:var(--no)}.st.n\/a{color:var(--mut)}
details.feat{border-bottom:1px solid var(--line);padding:.35rem 0}details.feat summary{cursor:pointer;font-size:13.5px}
.ri{display:grid;grid-template-columns:4.2rem 1fr;gap:.1rem .5rem;padding:.25rem 0 .25rem 1rem;font-size:13px}
.ri code{font-size:12px}.ri .fx{grid-column:2;color:var(--mut);font-size:12px}
.diffsec{margin:.4rem 0}.diffsec b{font-size:12px;color:var(--mut);text-transform:uppercase}
</style>
<header><h1>Studio · Setup</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a></header>
<main>
<section class="card" id="status-card"><h2>Status</h2><div id="status" class="note">…</div><div id="next"></div></section>

<section class="card" id="paths"><h2>Settings</h2><div class="note">Folders are saved to the running workspace; keep several layouts under <a href="/settings#workspaces">Settings → Workspaces</a>.</div>
<div class="note">Saved to <code id="cfgpath"></code>. Most changes apply at once; the rest after a restart.</div>
<form id="form"></form>
<div class="row" style="margin-top:.8rem"><button class="pri" id="save">Save</button><button id="restart">Restart Studio</button><span id="saved"></span></div>
</section>

<section class="card" id="build"><h2>Build llama.cpp</h2>
<div class="note">Studio runs models with llama.cpp's <code>llama-server</code>; the H-Neuron tools also need OmniCortex's <code>cett-dump</code>. This clones llama.cpp, adds cett-dump and builds both (about 5 minutes on a laptop CPU). Vulkan runs on AMD (APUs included), Intel and NVIDIA.</div>
<div class="row" style="margin-top:.5rem"><select id="backend" style="width:auto"><option>vulkan</option><option>cpu</option><option>hip</option><option>cuda</option><option>metal</option></select>
<label style="margin:0"><input type="checkbox" id="portable">portable (runs on other CPUs)</label>
<button class="pri" id="build">Build</button><span id="buildst" class="note"></span></div>
<pre id="buildlog" style="display:none"></pre></section>

<section class="card" id="hardware"><h2>Hardware for worker models</h2>
<div class="note">Where the director's worker models may run. AMD is used first when several devices have room. Saved to <code id="hwpath"></code>.</div>
<table id="devs"></table>
<label>llama-server per backend <span class="note">a build for that backend; empty uses the llama-server above</span></label>
<div class="grid" id="servers"></div>
<div class="row" style="margin-top:.8rem"><button class="pri" id="savehw">Save hardware</button><span id="hwsaved"></span></div></section>

<section class="card" id="health"><h2>Health check</h2><div class="note">Python packages, GPUs, the llama.cpp build, the model, pipeline progress. Each problem says what it blocks and how to fix it.</div>
<button id="doctor" style="margin-top:.5rem">Run checks</button><div id="checks"></div></section>

<section class="card" id="requirements"><h2>Requirements</h2>
<div class="note">What each feature needs (Python packages, programs, llama.cpp builds, drivers, device access, memory) and what this machine has. Python packages install from here into Studio’s Python, at the versions <code>requirements*.txt</code> ask for; system packages show the command for your OS. Same as <code>python scripts/ns_requirements.py</code>.</div>
<div class="row" style="margin-top:.5rem"><button id="reqcheck">Check requirements</button><span class="note" id="reqsys"></span></div>
<div id="reqs"></div>
<h2 style="margin-top:1.2rem" id="snapshots">Environment snapshots</h2>
<div class="note">Record packages, programs, llama.cpp builds and devices, then compare after an update or a driver change. <a href="api/requirements/freeze">Download exact package versions</a> (a requirements file that reproduces this setup).</div>
<div class="row" style="margin-top:.5rem"><button id="snap">Take a snapshot</button>
<select id="snapA" style="width:auto"></select><span class="note">→</span><select id="snapB" style="width:auto"></select><button id="snapdiff">Compare</button></div>
<div id="diff"></div></section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw Object.assign(new Error(j.error||r.status),{fields:j.fields});return j}
const gib=b=>b==null?'?':(b/2**30).toFixed(1);
let S=null, HW=null;

function renderStatus(s){
  const r=s.ready, items=[[r.server,'llama-server set'],[r.models,`${s.models} model(s) found`],[r.cett,'cett-dump set (activation checks, H-Neuron tools)']];
  $('#status').innerHTML=items.map(([ok,t])=>`<div><span class="st ${ok?'ok':'warn'}">${ok?'ok':'missing'}</span> ${esc(t)}</div>`).join('');
  let nx='';
  if(!r.server) nx=s.found.server?`A llama-server is already built at <code>${esc(s.found.server)}</code>: use it below.`:'Build llama.cpp below, then set its llama-server here.';
  else if(!r.models) nx='Add a folder with .gguf files below, or download a model in Studio’s Models tab.';
  else nx='Ready. <a href="/">Open Studio</a> to chat, <a href="/jobs">Jobs</a> to evaluate and train, <a href="/connect">Connect</a> to use these models from Cline.';
  $('#next').innerHTML=`<div class="next">${nx}</div>`;
}
function fieldHTML(s,k,f){
    const v=f.value; let inp;
    if(f.kind==='bool') return `<div><label style="margin-top:1.4rem"><input type="checkbox" data-k="${k}" ${v?'checked':''}>${esc(f.help)}${f.live?'':' <span class="note">(after restart)</span>'}</label><div class="err" data-err="${k}"></div></div>`;
    const val=f.kind==='dirs'?(v||[]).join('\n'):(v??'');
    const ph={duration:'e.g. 5m',durations:'e.g. 1h,8h,1d',url:'http://host:port/v1@model',path:'a file (created when needed)',readfile:'a file'}[f.kind]||'';
    const pick={file:'file',readfile:'file',path:'file',dir:'dir'}[f.kind], ext={rag_embed_gguf:'gguf',mcp_config:'json',hardware:'json',tls_cert:'pem,crt',tls_key:'pem,key'}[k]||'';
    const dl={host:['127.0.0.1','0.0.0.0'],pair_default:['persistent','1h','8h','1d','7d']}[k];
    const el=f.kind==='dirs'?`<textarea data-k="${k}" hidden>${esc(val)}</textarea><div data-list="${k}"></div>`
      :`<input data-k="${k}" value="${esc(val)}" placeholder="${ph}" ${['int','float'].includes(f.kind)?'inputmode="decimal"':''}${pick?` data-pick="${pick}"${ext?` data-ext="${ext}"`:''}`:''}${dl?` list="dl-${k}"`:''}>`+(dl?`<datalist id="dl-${k}">${dl.map(x=>`<option value="${x}">`).join('')}</datalist>`:'');
    const sug=s.found[k]&&s.found[k]!==v?`<button type="button" data-use="${k}" style="margin-top:.25rem">Use ${esc(s.found[k])}</button>`:'';
    return `<div><label>${esc(k.replace(/_/g,' '))}<span class="note">${esc(f.help)}${f.live?'':' (after restart)'}</span></label>${el}${sug}<div class="err" data-err="${k}"></div></div>`; }
function renderForm(s){
  $('#cfgpath').textContent=s.config_path;
  const groups={}; for(const [k,f] of Object.entries(s.fields)) (groups[f.group||'Other']??=[]).push([k,f]);
  const block=(g,xs)=>`<h3 style="margin:1rem 0 .2rem">${esc(g)}</h3><div class="grid">${xs.map(([k,f])=>fieldHTML(s,k,f)).join('')}</div>`;
  const basic=Object.entries(groups).filter(([g,xs])=>!xs[0][1].advanced), adv=Object.entries(groups).filter(([g,xs])=>xs[0][1].advanced);
  $('#form').innerHTML=basic.map(([g,xs])=>block(g,xs)).join('')+
    (adv.length?`<details style="margin-top:1rem" ${location.hash==='#advanced'?'open':''}><summary><b>Advanced</b> <span class="note">${adv.map(([g])=>esc(g)).join(', ')}. These apply after a restart; an address Studio would refuse to start with is refused here first.</span></summary>${adv.map(([g,xs])=>block(g,xs)).join('')}</details>`:'');
  // Folder lists: one row per folder (Browse…, reorder, remove), kept in the hidden field the save reads.
  document.querySelectorAll('[data-list]').forEach(d=>{ const k=d.dataset.list, ta=document.querySelector(`textarea[data-k=${k}]`);
    nsListEditor(d,{values:(s.fields[k].value||[]),want:'dir',addLabel:'+ Add folder',empty:'No folders yet: add the ones that hold your .gguf models.',onchange:v=>ta.value=v.join('\n')}); });
  document.querySelectorAll('[data-use]').forEach(b=>b.onclick=()=>{document.querySelector(`[data-k=${b.dataset.use}]`).value=s.found[b.dataset.use]});
}
async function load(){ S=await api('api/setup'); renderStatus(S); renderForm(S); }
$('#save').onclick=async()=>{
  const values={}; document.querySelectorAll('[data-k]').forEach(e=>{const k=e.dataset.k, f=S.fields[k];
    values[k]=f.kind==='bool'?e.checked:f.kind==='dirs'?e.value.split('\n').map(x=>x.trim()).filter(Boolean):e.value.trim();
    if(['int','float'].includes(f.kind)&&values[k]!==''&&!isNaN(+values[k])) values[k]=+values[k]; if(f.kind!=='dirs'&&values[k]==='') values[k]=null;
    // Only what changed: unchanged fields stay out of the config file (and keep following the defaults).
    const was=f.kind==='dirs'?(f.value||[]):f.value; if(k in values && JSON.stringify(values[k]??null)===JSON.stringify(was??null)) delete values[k];});
  if(!Object.keys(values).length){ $('#saved').innerHTML='<span class="note">nothing changed</span>'; return; }
  document.querySelectorAll('[data-err]').forEach(e=>e.textContent='');
  try{ const r=await api('api/setup',{values}); S=r; renderStatus(r);
    $('#saved').innerHTML=`<span class="okm">saved</span>`+(r.restart_needed.length?` <span class="note">· ${esc(r.restart_needed.join(', '))} apply after a restart</span>`:'');
  }catch(e){ if(e.fields) for(const [k,m] of Object.entries(e.fields)){const el=document.querySelector(`[data-err=${k}]`); if(el) el.textContent=m}
    $('#saved').innerHTML=`<span class="err">${esc(e.message)}</span>`; }
};
$('#restart').onclick=async()=>{ if(!confirm('Restart Studio? Loaded models and running workers stop; jobs keep running.')) return;
  await api('api/restart',{}); $('#saved').innerHTML='<span class="note">restarting…</span>';
  const t0=Date.now(); const poll=async()=>{ try{ await api('api/setup'); location.reload(); }catch{ if(Date.now()-t0<60000) setTimeout(poll,1000); } }; setTimeout(poll,1500); };

// ---- build
let buildJob=null;
$('#build').onclick=async()=>{ try{ const j=await api('api/jobs',{kind:'build_llama',values:{backend:$('#backend').value,portable:$('#portable').checked}});
  buildJob=j.id; $('#buildlog').style.display='block'; $('#build').disabled=true; pollBuild(); }catch(e){ $('#buildst').innerHTML=`<span class="err">${esc(e.message)}</span>` } };
async function pollBuild(){ if(!buildJob) return;
  const [jobs,log]=await Promise.all([api('api/jobs'),api(`api/jobs/${buildJob}/log`)]); const j=jobs.find(x=>x.id===buildJob);
  const el=$('#buildlog'); el.textContent=log.log.split('\n').filter(l=>!/^\[ *\d+%\]/.test(l)||/Built target/.test(l)).slice(-40).join('\n'); el.scrollTop=1e9;
  const pct=(log.log.match(/\[ *(\d+)%\]/g)||[]).pop(); $('#buildst').textContent=j.status==='running'?`building… ${pct||''}`:j.status;
  if(j.status==='running') setTimeout(pollBuild,2000); else { $('#build').disabled=false; buildJob=null; await load(); } }

// ---- hardware
function renderHW(h){
  HW=h; $('#hwpath').textContent=h.path; const cfg=h.config||{}; const dc=cfg.devices||{};
  $('#devs').innerHTML='<tr><th>use</th><th>device</th><th>memory</th><th>settings for this device</th></tr>'+h.devices.map(d=>{
    const c=dc[d.id]||{}; const env=Object.entries(c.env||{}).map(([k,v])=>`${k}=${v}`).join('\n');
    return `<tr><td><input type="checkbox" data-dev="${esc(d.id)}" ${d.enabled?'checked':''}></td>
      <td><b>${esc(d.id)}</b> ${esc(d.name)}${d.note?`<div class="note" style="color:var(--warn)">${esc(d.note)}</div>`:''}</td>
      <td class="note">${d.memory_total==null?'unknown':gib(d.memory_free)+' / '+gib(d.memory_total)+' GiB'+(d.estimated?' (est.)':'')}${d.unified&&d.backend!=='cpu'?'<br>shared with RAM':''}</td>
      <td><details><summary class="note">${c.server||c.rocm_path||env||c.reserve_gib!=null?'customised':'defaults'}</summary>
        <label>llama-server for this device</label><input data-x="${esc(d.id)}" data-f="server" value="${esc(c.server||'')}" data-pick="file">
        ${d.backend==='rocm'?`<label>pinned ROCm release <span class="note">e.g. /opt/rocm-6.2.4</span></label><input data-x="${esc(d.id)}" data-f="rocm_path" value="${esc(c.rocm_path||'')}">`:''}
        <label>memory to keep free (GiB)</label><input data-x="${esc(d.id)}" data-f="reserve_gib" value="${esc(c.reserve_gib??'')}">
        <label>environment <span class="note">KEY=value per line; HSA_, HIP_, ROCR_, GGML_, CUDA_, VK_… only</span></label><textarea data-x="${esc(d.id)}" data-f="env" data-lines="text" data-add="+ Add variable" placeholder="NAME=value">${esc(env)}</textarea>
      </details></td></tr>`}).join('');
  $('#servers').innerHTML=['rocm','vulkan','cuda','metal','cpu'].map(b=>`<div><label>${b}</label><input data-srv="${b}" value="${esc((cfg.servers||{})[b]||'')}"></div>`).join('');
}
$('#savehw').onclick=async()=>{
  const cfg=JSON.parse(JSON.stringify(HW.config||{})); cfg.devices=cfg.devices||{}; cfg.servers={};
  document.querySelectorAll('[data-srv]').forEach(e=>{ if(e.value.trim()) cfg.servers[e.dataset.srv]=e.value.trim() });
  document.querySelectorAll('[data-dev]').forEach(e=>{ const d=HW.devices.find(x=>x.id===e.dataset.dev); const c=cfg.devices[d.id]=cfg.devices[d.id]||{};
    if(e.checked!==d.enabled||'enabled' in c) c.enabled=e.checked; });
  document.querySelectorAll('[data-x]').forEach(e=>{ const c=cfg.devices[e.dataset.x]=cfg.devices[e.dataset.x]||{}; const f=e.dataset.f, v=e.value.trim();
    if(f==='env'){ const env={}; v.split('\n').forEach(l=>{const i=l.indexOf('='); if(i>0) env[l.slice(0,i).trim()]=l.slice(i+1).trim()}); if(Object.keys(env).length) c.env=env; else delete c.env; }
    else if(f==='reserve_gib'){ if(v) c.reserve_gib=+v; else delete c.reserve_gib; } else { if(v) c[f]=v; else delete c[f]; } });
  for(const k of Object.keys(cfg.devices)) if(!Object.keys(cfg.devices[k]).length) delete cfg.devices[k];
  try{ const r=await api('api/hardware',{config:cfg}); renderHW({...HW,config:cfg,devices:r.devices}); $('#hwsaved').innerHTML='<span class="okm">saved</span>'; }
  catch(e){ $('#hwsaved').innerHTML=`<span class="err">${esc(e.message)}</span>` } };

// ---- doctor
$('#doctor').onclick=async()=>{ $('#doctor').disabled=true; $('#checks').innerHTML='<div class="note">checking… (up to a minute)</div>';
  try{ const r=await api('api/doctor'); let st=null, h='';
    for(const c of r.checks){ if(c.stage!==st){ st=c.stage; h+=`<div class="dstage">${esc(st)}</div>`; }
      h+=`<div class="check"><span class="st ${esc(c.status)}">${esc(c.status==='skip'?'--':c.status)}</span><div><b>${esc(c.name)}</b> ${esc(c.detail)}${
        (c.status==='warn'||c.status==='fail')&&(c.fix||c.blocks)?`<div class="fix">${c.blocks?'blocks: '+esc(c.blocks)+'. ':''}${c.fix?'fix: '+linkify(esc(c.fix)):''}</div>`:''}</div></div>`; }
    $('#checks').innerHTML=h; }catch(e){ $('#checks').innerHTML=`<div class="err">${esc(e.message)}</div>` }
  $('#doctor').disabled=false; };

load().catch(e=>$('#status').innerHTML=`<span class="err">${esc(e.message)}</span>`);
api('api/hardware').then(renderHW).catch(e=>$('#devs').innerHTML=`<tr><td class="err">${esc(e.message)}</td></tr>`);

// ---------- requirements (scripts/ns_requirements.py)
function reqItem(i){
  const opt=i.optional?' <span class="note">(optional)</span>':'';
  const need=i.need&&!['installed','built','any'].includes(i.need)?` <span class="note">needs ${esc(i.need)}</span>`:'';
  const have=i.have?`<code>${esc(i.have)}</code>`:'<span class="note">not found</span>';
  const be=i.backends&&i.backends.length?` <span class="note">GPU backends: ${esc(i.backends.join(', '))}</span>`:'';
  const why=i.why?` <span class="note">· ${esc(i.why)}</span>`:'';
  const fix=i.fix?`<div class="fx">fix: <code>${linkify(esc(i.fix))}</code>${i.installable?` <button data-pkg="${esc(i.name)}">Install</button>`:''}</div>`:'';
  return `<div class="ri"><span class="st ${i.status}">${i.status==='missing'?'missing':i.status}</span><div>${esc(i.name)}${opt}: ${have}${need}${be}${why}</div>${fix}</div>`;
}
async function loadReqs(){
  $('#reqs').innerHTML='<div class="note">checking…</div>';
  let r; try{ r=await api('api/requirements'); }catch(e){ $('#reqs').innerHTML=`<div class="err">${esc(e.message)}</div>`; return; }
  const hw=r.hardware||{};
  $('#reqsys').textContent=`${r.os} · Python ${r.python} · ${r.package_manager||'unknown package manager'} · ${hw.cpu||''} · ${hw.ram_gib??'?'} GiB RAM · ${hw.disk_free_gib??'?'} GiB free`;
  $('#reqs').innerHTML=Object.entries(r.features).map(([k,f])=>{
    const bad=f.items.filter(i=>i.status!=='ok').length;
    return `<details class="feat"${f.status==='missing'&&k==='core'?' open':''}><summary><span class="st ${f.status}">${esc(f.status)}</span> ${esc(f.title)}`+
      (f.note?` <span class="note">· ${esc(f.note)}</span>`:bad?` <span class="note">· ${bad} to look at</span>`:'')+`</summary>${f.items.map(reqItem).join('')}</details>`;}).join('');
  document.querySelectorAll('[data-pkg]').forEach(b=>b.onclick=()=>installPkg(b));
}
async function installPkg(b){
  b.disabled=true; b.textContent='installing…';
  try{ const j=await api('api/jobs',{kind:'install_package',values:{package:b.dataset.pkg}});
    for(;;){ await new Promise(r=>setTimeout(r,2000)); const jobs=await api('api/jobs'); const x=jobs.find(y=>y.id===j.id);
      if(x&&!['running','queued'].includes(x.status)){ if(x.status!=='done'){ b.textContent='failed: see Jobs'; return; } break; } }
    loadReqs();
  }catch(e){ b.textContent=e.message; }
}
async function loadSnaps(){
  const r=await api('api/requirements/snapshots'); const o=r.snapshots.map(x=>`<option value="${esc(x.id)}">${esc(x.when)} · ${esc(x.host||'')}</option>`).join('');
  $('#snapA').innerHTML=o; $('#snapB').innerHTML=o; if(r.snapshots.length>1) $('#snapA').selectedIndex=1;
  $('#snapdiff').disabled=r.snapshots.length<2;
}
function diffSec(title,d){ const rows=[...Object.entries(d.changed).map(([k,v])=>`<div>${esc(k)}: <code>${esc(v[0])}</code> → <code>${esc(v[1])}</code></div>`),
  ...Object.entries(d.added).map(([k,v])=>`<div class="okm">+ ${esc(k)} <code>${esc(v)}</code></div>`),
  ...Object.entries(d.removed).map(([k,v])=>`<div class="err">− ${esc(k)} <code>${esc(v)}</code></div>`)];
  return rows.length?`<div class="diffsec"><b>${title}</b>${rows.join('')}</div>`:''; }
$('#reqcheck').onclick=loadReqs;
$('#snap').onclick=async()=>{ $('#snap').disabled=true; try{ await api('api/requirements/snapshot',{}); await loadSnaps(); }catch(e){ alert(e.message); } $('#snap').disabled=false; };
$('#snapdiff').onclick=async()=>{ try{ const d=await api('api/requirements/diff',{a:$('#snapA').value,b:$('#snapB').value});
  const sys=Object.entries(d.system).map(([k,v])=>`<div>${esc(k)}: <code>${esc(v[0])}</code> → <code>${esc(v[1])}</code></div>`).join('');
  $('#diff').innerHTML=d.unchanged?'<div class="note" style="margin-top:.5rem">Nothing changed.</div>':
    `<div style="margin-top:.5rem">${sys?`<div class="diffsec"><b>System</b>${sys}</div>`:''}${diffSec('llama.cpp',d.llama)}${diffSec('Devices',d.devices)}${diffSec('Programs',d.tools)}${diffSec('Python packages',d.packages)}</div>`;
  }catch(e){ $('#diff').innerHTML=`<div class="err">${esc(e.message)}</div>`; } };
loadReqs(); loadSnaps();

if(location.hash){ const t=document.querySelector(location.hash); if(t) setTimeout(()=>t.scrollIntoView(),300); }
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
