"""Studio's /review page: per-neuron statistics across everything a model has
done (scripts/neuron_review.py), filtered by source, subject and time, and
played over time in the 3D view."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Review</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--code:#f3f3f0;--good:#2f6fe0;--bad:#e0552f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624;--good:#5b8cff;--bad:#ff7a4d}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624;--good:#5b8cff;--bad:#ff7a4d}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--acc)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{font-size:13px}
main{max-width:1180px;margin:0 auto;padding:1rem;display:grid;grid-template-columns:290px 1fr;gap:1rem}
@media (max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem .9rem;margin-bottom:1rem;min-width:0}
h2{font-size:14px;margin:0 0 .4rem}h3{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:.8rem 0 .3rem}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no);font-size:12.5px}
select,input{font:12.5px ui-sans-serif,system-ui;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg);width:100%}
label.ck{display:flex;gap:.4rem;align-items:center;font-size:13px;margin:.1rem 0}label.ck input{width:auto}
label.ck .n{margin-left:auto;color:var(--mut);font-size:12px}
.kind{font-size:11px;color:var(--mut);margin-top:.35rem}
.row{display:flex;gap:.5rem;align-items:center}.row>*{min-width:0}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button:disabled{opacity:.5}
.stats{display:flex;gap:1.4rem;flex-wrap:wrap;font-size:13px}.stats b{font-size:18px;display:block}
canvas{display:block;width:100%;image-rendering:pixelated}
#heat{border:1px solid var(--line);border-radius:4px;cursor:crosshair}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:.25rem .4rem;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500;font-size:12px}td.num{font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin:0 .3rem 0 .6rem}
.empty{padding:2rem;text-align:center;color:var(--mut)}
code{font:12px ui-monospace,monospace;background:var(--code);padding:.05rem .3rem;border-radius:4px}
</style>
<header><h1>Studio · Review</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a></header>
<main>
<aside>
<section class="card"><h2>What to review</h2>
<h3>Model</h3><select id="model"></select>
<h3>Statistic</h3><select id="stat">
<option value="association">Hallucination association (wrong vs right answers)</option>
<option value="risk">Risk correlation (unlabelled replies)</option>
<option value="firing">Firing rate</option></select>
<div class="note" id="statNote"></div>
<h3>Over time, by</h3><div class="row"><select id="by"><option value="day">day</option><option value="week" selected>week</option><option value="month">month</option><option value="all">all at once</option><option value="every">every N replies</option></select><input id="every" type="number" min="2" value="50" style="width:5.5rem" hidden></div>
<h3>From</h3><div class="row"><input id="since" type="date"><input id="until" type="date"></div>
</section>
<section class="card"><h2>Sources</h2><div class="note">Tests (TestQA), benchmarks you ingested, and observed replies (chat checks, project work, live scoring).</div><div id="sources"></div></section>
<section class="card"><h2>Skills and subjects</h2><div class="note">None ticked means all.</div><div id="subjects"></div></section>
<section class="card"><h2>Verdicts</h2><div id="verdicts"></div></section>
</aside>
<div>
<section class="card" id="head"><div class="empty">Loading…</div></section>
<section class="card"><h2>Error rate over time</h2><canvas id="tl" style="height:110px"></canvas><div class="note" id="tlNote"></div></section>
<section class="card"><h2>Every neuron: layer (rows) × neuron bin (columns)</h2>
<div class="note" id="legend"></div><canvas id="heat"></canvas><div class="note" id="hover">&nbsp;</div></section>
<section class="card"><h2>Strongest neurons</h2><table id="top"></table></section>
<section class="card"><h2>Adding data</h2><div class="note">
<b>Observed:</b> every <i>check</i> in Studio's chat is recorded here, and so are replies scored in the background (model settings: classifier) and checked project work. Mark a checked reply <i>right</i> or <i>wrong</i> under it and it counts toward the association too.<br>
<b>Tests and benchmarks:</b> <a href="/jobs#kind=review_testqa">ingest a TestQA run</a> takes a TestQA results file (no run yet? <a href="/jobs#kind=testqa">run TestQA</a>); <a href="/jobs#kind=review_items">ingest graded replies</a> takes JSONL with <code>prompt</code>, <code>response</code>, <code>verdict</code> (right/wrong/abstained) and optionally <code>subject</code> from any other benchmark. Each replays the replies through the model once to record its activations.<br>
Without the UI: <code>python scripts/neuron_review.py --help</code>. Store: <code id="store"></code></div></section>
</div>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const KIND={test:'Tests',benchmark:'Benchmarks',observed:'Observed'};
const NOTES={association:"Cohen's d between wrong and right answers. Red-orange fires more when the model is wrong, blue when it is right. Needs graded replies.",
  risk:"Correlation with the hallucination classifier's risk, for replies nobody graded. Red-orange rises with risk.",
  firing:"How often each neuron is among the most active (top 3%) in the selected replies."};
let INFO=null, LAST=null, timer=null;
async function api(path,body){ const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.status); return j; }
function checks(el,items,name){ el.innerHTML=items.map(([v,label,n,extra])=>`${extra||''}<label class="ck"><input type="checkbox" name="${name}" value="${esc(v)}">${esc(label)}<span class="n">${n}</span></label>`).join('')||'<div class="note">none yet</div>'; }
function picked(name){ const v=[...document.querySelectorAll(`input[name=${name}]:checked`)].map(x=>x.value); return v.length?v:null; }
function day(s,end){ if(!s) return null; const d=new Date(s+'T00:00:00'); if(end) d.setDate(d.getDate()+1); return d.getTime()/1000; }
function filters(){ const by=$('#by').value; return {model:$('#model').value, stat:$('#stat').value, by:by==='every'?'all':by,
  every:by==='every'?Math.max(2,+$('#every').value||50):0, sources:picked('src'), subjects:picked('subj'), verdicts:picked('verd'),
  since:day($('#since').value), until:day($('#until').value,true), top:25}; }
function facets(){
  const f=INFO.facets[$('#model').value]; if(!f) return;
  const groups={}; for(const [s,v] of Object.entries(f.sources)) (groups[v.kind]=groups[v.kind]||[]).push([s,s,v.n]);
  const items=[]; for(const k of ['test','benchmark','observed']) (groups[k]||[]).forEach((x,i)=>items.push([...x, i?'':`<div class="kind">${KIND[k]}</div>`]));
  checks($('#sources'),items,'src');
  checks($('#subjects'),Object.entries(f.subjects).map(([s,n])=>[s,s,n]),'subj');
  checks($('#verdicts'),Object.entries(f.verdicts).map(([s,n])=>[s,s==='unknown'?'not graded':s,n]),'verd');
  document.querySelectorAll('aside input[type=checkbox]').forEach(x=>x.onchange=refresh);
}
function refresh(){ clearTimeout(timer); timer=setTimeout(load,150); }
async function load(){
  $('#statNote').textContent=NOTES[$('#stat').value]; $('#every').hidden=$('#by').value!=='every';
  let r; try{ r=await api('/api/review/summary',filters()); }catch(e){ $('#head').innerHTML=`<div class="err">${esc(e.message)}</div>`; return; }
  LAST=r;
  const er=r.error_rate==null?'—':Math.round(r.error_rate*100)+'%';
  $('#head').innerHTML=`<div class="row" style="justify-content:space-between;flex-wrap:wrap;gap:.6rem"><div class="stats">
    <div><b>${r.n}</b>replies</div><div><b>${r.n_right}</b>right</div><div><b>${r.n_wrong}</b>wrong</div><div><b>${er}</b>error rate</div></div>
    <button class="pri" id="open3d" ${r.n?'':'disabled'}>Open 3D over time ↗</button></div>`+
    (r.note?`<div class="note" style="margin-top:.4rem">${esc(r.note)}</div>`:'');
  $('#open3d').onclick=open3d;
  FLOOR=r.floor||0; timeline(r.timeline); heat(r.grid, r.stat); topTable(r.top, r.stat);
}
async function open3d(){
  const w=window.open('about:blank','_blank');
  try{ const v=await api('/api/review/view',filters()); if(w) w.location=v.url; else location.href=v.url; }
  catch(e){ if(w) w.close(); alert(e.message); }
}
function timeline(tl){
  const c=$('#tl'), W=c.clientWidth||600, H=110, dpr=devicePixelRatio||1; c.width=W*dpr; c.height=H*dpr; c.style.height=H+'px';
  const g=c.getContext('2d'); g.scale(dpr,dpr); g.clearRect(0,0,W,H);
  if(!tl.length){ $('#tlNote').textContent='no replies match'; return; }
  const bw=W/tl.length, maxN=Math.max(...tl.map(b=>b.n));
  g.font='11px ui-sans-serif,system-ui';
  tl.forEach((b,i)=>{ const x=i*bw;
    g.fillStyle=css('--line'); const nh=(b.n/maxN)*(H-28); g.fillRect(x+bw*.15,H-16-nh,bw*.7,nh);
    if(b.error_rate!=null){ g.fillStyle=css('--bad'); const eh=b.error_rate*(H-28); g.fillRect(x+bw*.3,H-16-eh,bw*.4,eh); }
    // Label every k-th bucket so labels never overlap.
    const lw=g.measureText(b.label).width+8, k=Math.max(1,Math.ceil(lw/bw));
    if(i%k===0){ g.fillStyle=css('--mut'); g.fillText(b.label,x+2,H-3); } });
  $('#tlNote').innerHTML=`<span class="sw" style="background:${css('--line')};margin-left:0"></span>replies (relative)<span class="sw" style="background:${css('--bad')}"></span>error rate (full height = 100%) among graded replies`;
}
let FLOOR=0;
function color(v,stat){
  if(stat==='firing'){ const a=Math.min(1,v*3); return [a*255,a*190,a*80]; }
  const s=stat==='association'?1.5:0.6, a=Math.min(1,Math.abs(v)/s);
  if(Math.abs(v)<FLOOR) return [40,40,46];
  return v>0?[40+a*215,40+a*45,40]:[40,40+a*70,40+a*215];
}
function heat(grid,stat){
  const c=$('#heat');
  if(!grid||!grid.length){ c.width=1; c.height=1; $('#legend').textContent='nothing to show'; return; }
  const L=grid.length, B=grid[0].length, rh=Math.max(3,Math.min(10,Math.floor(360/L)));
  c.width=B; c.height=L*rh; c.style.height=Math.min(L*rh*Math.max(1,(c.clientWidth||B)/B),520)+'px';
  const g=c.getContext('2d'), img=g.createImageData(B,L*rh);
  for(let l=0;l<L;l++) for(let b=0;b<B;b++){ const [r,gg,bb]=color(grid[l][b],stat);
    for(let k=0;k<rh;k++){ const o=((l*rh+k)*B+b)*4; img.data[o]=r; img.data[o+1]=gg; img.data[o+2]=bb; img.data[o+3]=255; } }
  g.putImageData(img,0,0);
  $('#legend').innerHTML=stat==='firing'?'brighter: fires more often':
    `<span class="sw" style="background:rgb(255,85,40);margin-left:0"></span>${stat==='association'?'fires more on hallucinations':'rises with risk'}<span class="sw" style="background:rgb(40,110,255)"></span>${stat==='association'?'fires more on right answers':'falls with risk'} · grey: within chance for this many replies (|${stat==='association'?'d':'r'}| &lt; ${FLOOR}) · layer 0 at the top`;
  c.onmousemove=e=>{ const r=c.getBoundingClientRect(); const b=Math.floor((e.clientX-r.left)/r.width*B), l=Math.floor((e.clientY-r.top)/r.height*L);
    if(grid[l]&&grid[l][b]!=null) $('#hover').textContent=`layer ${l} · bin ${b} · ${grid[l][b]}`; };
}
function topTable(rows,stat){
  const lab={association:"d (wrong − right)",risk:'r with risk',firing:'firing rate'}[stat];
  $('#top').innerHTML=rows.length?`<tr><th>layer</th><th>bin</th><th>${lab}</th><th>mean activation</th></tr>`+
    rows.map(t=>`<tr><td>${t.layer}</td><td>${t.bin}</td><td class="num" style="color:${stat==='firing'?'inherit':t.value>0?css('--bad'):css('--good')}">${t.value>0&&stat!=='firing'?'+':''}${t.value}</td><td class="num">${t.mean}</td></tr>`).join('')
    :'<tr><td class="note">none stand out for this selection</td></tr>';
}
(async()=>{
  try{ INFO=await api('/api/review'); }catch(e){ $('#head').innerHTML=`<div class="err">${esc(e.message)}</div>`; return; }
  $('#store').textContent=INFO.store;
  if(!INFO.models.length){ $('#head').innerHTML='<div class="empty">No observations yet. <a href="/#chats">Check a reply in the chat</a> (the model needs a classifier: <a href="/#models">its settings</a>), or <a href="/jobs#kind=review_testqa">ingest a TestQA run</a>; see <i>Adding data</i> below.</div>'; return; }
  $('#model').innerHTML=INFO.models.map(m=>`<option value="${esc(m.model)}">${esc(m.model)} (${m.observations})</option>`).join('');
  $('#model').onchange=()=>{ facets(); refresh(); };
  for(const id of ['#stat','#by','#every','#since','#until']) $(id).onchange=refresh;
  facets(); load();
  addEventListener('resize',()=>{ if(LAST) timeline(LAST.timeline); });
})();
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
