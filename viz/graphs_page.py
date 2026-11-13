"""Studio's /graphs page: a node editor for screening, verification and routing graphs (scripts/graph_engine.py).
Drag nodes from the palette, wire outputs to inputs, run with inputs, watch each node's result, approve
paused runs, debug one node, and run a graph over test cases."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Graphs</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0;--grid:#ececE8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--grid:#1f1f1d}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--grid:#1f1f1d}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;font:13.5px/1.45 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg);display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:.5rem;padding:.45rem 12px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0 .4rem 0 0}
input,select,textarea,button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.3rem .5rem}
textarea{width:100%;resize:vertical;font:12px ui-monospace,monospace}
button{cursor:pointer;background:var(--panel)}button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}
button:disabled{opacity:.5;cursor:default}
.note{color:var(--mut);font-size:12px}.err{color:var(--no)}.okm{color:var(--ok)}
main{flex:1;display:grid;grid-template-columns:210px 1fr 300px;min-height:0}
#pal,#insp{overflow:auto;background:var(--panel);border-right:1px solid var(--line);padding:.5rem}
#insp{border-right:none;border-left:1px solid var(--line)}
#pal details{margin-bottom:.3rem}#pal summary{font-weight:600;font-size:12px;cursor:pointer;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
.pi{padding:.25rem .45rem;margin:.15rem 0;border:1px solid var(--line);border-radius:6px;cursor:grab;background:var(--bg);font-size:12.5px}
.pi:hover{border-color:var(--acc)}
#wrap{position:relative;overflow:hidden;background:var(--bg);background-image:radial-gradient(var(--line) 1px,transparent 1px);background-size:22px 22px;cursor:grab}
#wrap.panning{cursor:grabbing}
#world{position:absolute;left:0;top:0;transform-origin:0 0}
#edges{position:absolute;left:0;top:0;overflow:visible;pointer-events:none}
#edges path{fill:none;stroke:var(--mut);stroke-width:2;pointer-events:stroke;cursor:pointer}
#edges path.sel{stroke:var(--acc);stroke-width:3}#edges path.live{stroke:var(--ok)}#edges path.dead{stroke-dasharray:4 4;opacity:.5}
.nd{position:absolute;width:190px;background:var(--panel);border:1px solid var(--line);border-radius:9px;box-shadow:0 2px 8px #0003;font-size:12px;user-select:none}
.nd.sel{border-color:var(--acc);box-shadow:0 0 0 2px var(--acc)}
.nd .h{padding:.3rem .5rem;border-bottom:1px solid var(--line);font-weight:600;cursor:move;display:flex;gap:.3rem;align-items:center;border-radius:9px 9px 0 0}
.nd .h .id{font-weight:400;color:var(--mut);margin-left:auto;font-size:11px}
.nd .b{display:flex;justify-content:space-between;padding:.25rem 0}
.nd .ins,.nd .outs{display:flex;flex-direction:column;gap:.15rem}
.nd .p{position:relative;padding:0 .55rem;white-space:nowrap;font-size:11.5px;color:var(--mut)}
.nd .p .dot{position:absolute;top:50%;width:11px;height:11px;margin-top:-5.5px;border-radius:50%;border:2px solid var(--panel);background:var(--mut);cursor:crosshair}
.nd .ins .dot{left:-6px}.nd .outs .dot{right:-6px}.nd .outs .p{text-align:right}
.nd .p.req{color:var(--fg)}.nd .p.err{color:var(--no)}.nd .p.pass .dot{background:var(--ok)}.nd .p.fail .dot,.nd .p.err .dot{background:var(--no)}
.nd .st{padding:.15rem .5rem .3rem;font-size:11px;color:var(--mut);max-height:3.2em;overflow:hidden;border-top:1px dashed var(--line)}
.nd.ok .h{background:color-mix(in srgb,var(--ok) 18%,transparent)}.nd.error .h{background:color-mix(in srgb,var(--no) 22%,transparent)}
.nd.skipped{opacity:.55}.nd.waiting .h{background:color-mix(in srgb,var(--warn) 25%,transparent)}
.g-Screening .h::before,.g-Verification .h::before,.g-Routing .h::before,.g-Models .h::before,.g-Tools .h::before,.g-Delegation .h::before,.g-Data .h::before{content:'';width:8px;height:8px;border-radius:2px;flex:none}
.g-Screening .h::before{background:#e7b65c}.g-Verification .h::before{background:#5fcf8f}.g-Routing .h::before{background:#b48ead}
.g-Models .h::before{background:#6f93ff}.g-Tools .h::before{background:#ff8b7e}.g-Delegation .h::before{background:#5dcaa5}.g-Data .h::before{background:#9a9a94}
#tmp{position:absolute;pointer-events:none}
#insp h3{margin:.2rem 0 .3rem;font-size:14px}#insp label{display:block;margin:.35rem 0 .1rem;font-size:12px;color:var(--mut)}
#insp input,#insp select,#insp textarea{width:100%}
#zoom{position:absolute;right:10px;bottom:10px;display:flex;gap:.25rem;z-index:3}#zoom button{padding:.15rem .5rem}
#bottom{border-top:1px solid var(--line);background:var(--panel);max-height:42vh;display:flex;flex-direction:column}
#btabs{display:flex;gap:.2rem;padding:.2rem 10px 0;border-bottom:1px solid var(--line)}
#btabs button{border:none;border-bottom:2px solid transparent;border-radius:0;background:none;color:var(--mut);padding:.3rem .6rem}
#btabs button.on{color:var(--fg);border-bottom-color:var(--acc)}#btabs .grow{flex:1}
#bbody{overflow:auto;padding:.5rem 12px;min-height:7rem}
.bp{display:none}.bp.on{display:block}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:.25rem .4rem;text-align:left;vertical-align:top;font-size:12.5px}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font:12px ui-monospace,monospace;max-height:14em;overflow:auto;background:var(--code);padding:.3rem .45rem;border-radius:5px}
.chip{display:inline-block;padding:0 .4rem;border-radius:9px;font-size:11px;border:1px solid var(--line)}
.chip.ok,.chip.done{color:var(--ok);border-color:var(--ok)}.chip.error,.chip.failed{color:var(--no);border-color:var(--no)}.chip.waiting,.chip.running{color:var(--warn);border-color:var(--warn)}
#inputs{display:flex;gap:.35rem;flex-wrap:wrap;align-items:center}
#inputs input{width:13rem}
@media (max-width:900px){main{grid-template-columns:1fr}#pal,#insp{display:none}main.pal #pal,main.insp #insp{display:block;position:fixed;inset:auto 0 0 0;max-height:55vh;z-index:20;border-top:1px solid var(--line)}
  #togs{display:flex!important}}
#togs{display:none;gap:.3rem}
</style>
<header><h1>Graphs</h1>
  <select id="gsel" title="your graphs and the templates"></select>
  <input id="gname" placeholder="name" style="width:11rem" title="lowercase letters, digits, - and _">
  <button id="save" title="Ctrl+S">Save</button><button id="del" class="bad">Delete</button>
  <span id="togs"><button id="tPal">Nodes</button><button id="tInsp">Inspector</button></span>
  <span style="flex:1"></span>
  <span id="inputs"></span>
  <input id="model" placeholder="model (empty: the loaded one)" list="models" style="width:12rem" title="the graph's model; nodes can pick their own">
  <select id="profile" title="permission profile for the whole run (Settings → Permissions)"><option value="">permissions: per job</option></select>
  <button class="pri" id="run" title="Ctrl+Enter">Run</button>
</header>
<main id="main">
<aside id="pal"><input id="psearch" placeholder="find a node" style="width:100%;margin-bottom:.4rem"><div id="plist"></div>
  <div class="note" style="margin-top:.6rem">Drag a node onto the canvas, or click it. Wire an output (right) to an input (left). Pass and fail ports route a value one way or the other; a node whose required input was skipped is skipped. Every node has an error port: wire it to keep going after a failure.</div></aside>
<div id="wrap"><div id="world"><svg id="edges"></svg><svg id="tmp"></svg></div>
  <div id="zoom"><button id="zin">+</button><button id="zout">−</button><button id="zfit" title="fit everything">⛶</button></div></div>
<aside id="insp"><div class="note">Select a node to edit it. Delete removes it; Ctrl+D duplicates.</div></aside>
</main>
<section id="bottom"><div id="btabs"><button data-b="trace" class="on">Run</button><button data-b="approve">Approvals <span id="nwait"></span></button><button data-b="eval">Evals</button><button data-b="runs">History</button><button data-b="json">JSON</button><span class="grow"></span><span id="status" class="note" style="padding:.3rem .6rem"></span></div>
<div id="bbody">
  <div class="bp on" data-b="trace"><div id="trace" class="note">Run the graph to see what each node did.</div></div>
  <div class="bp" data-b="approve"><div id="approve" class="note">Nothing is waiting for you.</div></div>
  <div class="bp" data-b="eval"><div class="note">Test cases, one JSON object per line: the run's inputs by name, plus the expected value (as in PySpur's test cases). A case passes when the chosen output matches.</div>
    <textarea id="cases" rows="5" placeholder='{"question": "What is the capital of France?", "answer": "Paris"}'></textarea>
    <div class="row" style="display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.3rem;align-items:center"><span class="note">expected field</span><input id="expect" value="answer" style="width:8rem"><span class="note">output</span><input id="evout" value="result" style="width:8rem">
      <select id="grade"><option>contains</option><option>exact</option><option>regex</option></select><button id="evrun">Run cases</button></div>
    <div id="evres"></div></div>
  <div class="bp" data-b="runs"><div id="runs"></div></div>
  <div class="bp" data-b="json"><textarea id="json" rows="12"></textarea><button id="jsonApply">Apply</button> <span class="note">Edit the graph as JSON (docs/GRAPHS.md).</span></div>
</div></section>
<datalist id="models"></datalist>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const status=(t,bad)=>{$('#status').innerHTML=bad?`<span class="err">${esc(t)}</span>`:esc(t)};
let CAT=null, T={}, G={name:'',description:'',nodes:[],edges:[]}, SEL=null, SELE=null, view={x:0,y:0,k:1}, RUN=null, dirty=false;
const typeOf=n=>T[n.type]||{inputs:[],outputs:[],params:[],title:n.type,group:''};

// ---------------------------------------------------------------- palette
function palette(){ const q=$('#psearch').value.toLowerCase();
  $('#plist').innerHTML=CAT.groups.map(g=>{ const items=CAT.types.filter(t=>t.group===g&&(!q||(t.title+t.name+t.help).toLowerCase().includes(q)));
    return items.length?`<details open><summary>${esc(g)}</summary>${items.map(t=>`<div class="pi" draggable="true" data-t="${t.name}" title="${esc(t.help)}">${esc(t.title)}</div>`).join('')}</details>`:''; }).join('');
  $$('.pi').forEach(el=>{ el.ondragstart=e=>e.dataTransfer.setData('text/ns-node',el.dataset.t); el.onclick=()=>{ const r=$('#wrap').getBoundingClientRect(); add(el.dataset.t,(r.width/2-95-view.x)/view.k,(r.height/2-40-view.y)/view.k); }; }); }
$('#psearch').oninput=palette;
$('#wrap').ondragover=e=>e.preventDefault();
$('#wrap').ondrop=e=>{ e.preventDefault(); const t=e.dataTransfer.getData('text/ns-node'); if(!t) return; const r=$('#wrap').getBoundingClientRect(); add(t,(e.clientX-r.left-view.x)/view.k-95,(e.clientY-r.top-view.y)/view.k-15); };
function add(type,x,y){ const t=T[type], base=type.replace(/_.*/,''); let i=1; while(G.nodes.some(n=>n.id===base+i)) i++;
  const params={}; t.params.forEach(p=>{ if(p.default!==null&&p.default!==undefined) params[p.name]=JSON.parse(JSON.stringify(p.default)); });
  const n={id:base+i,type,params,x:Math.round(x),y:Math.round(y)}; G.nodes.push(n); SEL=n.id; SELE=null; changed(); }

// ---------------------------------------------------------------- canvas
function portPos(id,port,side){ const el=document.querySelector(`.nd[data-id="${CSS.escape(id)}"] .${side} [data-p="${CSS.escape(port)}"] .dot`); const n=G.nodes.find(n=>n.id===id);
  if(!el||!n) return null; return {x:n.x+(side==='ins'?0:190), y:n.y+el.parentElement.offsetTop+el.parentElement.offsetHeight/2}; }
function curve(a,b){ const dx=Math.max(40,Math.abs(b.x-a.x)/2); return `M${a.x},${a.y} C${a.x+dx},${a.y} ${b.x-dx},${b.y} ${b.x},${b.y}`; }
function nodeState(id){ if(!RUN||!RUN.trace) return null; const t=RUN.trace.filter(r=>r.node===id); return t.length?t[t.length-1]:null; }
function draw(){
  const w=$('#world'); $$('.nd').forEach(e=>e.remove());
  for(const n of G.nodes){ const t=typeOf(n), st=nodeState(n.id), el=document.createElement('div');
    el.className=`nd g-${t.group.split(' ')[0]}${n.id===SEL?' sel':''}${st?' '+st.status:''}`; el.dataset.id=n.id; el.style.left=n.x+'px'; el.style.top=n.y+'px';
    const wiredIn=new Set(G.edges.filter(e=>e.to[0]===n.id).map(e=>e.to[1]));
    el.innerHTML=`<div class="h" title="${esc(t.help)}">${esc(t.title)}<span class="id">${esc(n.id)}</span></div><div class="b">
      <div class="ins">${t.inputs.map(p=>`<div class="p${p.required?' req':''}" data-p="${p.name}" title="${p.type}${p.required?', required':''}"><span class="dot"></span>${esc(p.name)}${wiredIn.has(p.name)?'':''}</div>`).join('')}</div>
      <div class="outs">${t.outputs.map(p=>`<div class="p ${p.name==='error'?'err':p.name}" data-p="${p.name}" title="${p.type}"><span class="dot"></span>${esc(p.name)}</div>`).join('')}</div></div>
      ${st?`<div class="st" title="${esc(JSON.stringify(st.outputs||st.error||'',null,1))}">${esc(st.status)}${st.ms!=null?' · '+st.ms+' ms':''}${st.error?' · '+esc(st.error):st.outputs?' · '+esc(Object.entries(st.outputs).filter(([k,v])=>v!=='(skipped)').map(([k,v])=>k+': '+(typeof v==='string'?v:JSON.stringify(v))).join(' · ').slice(0,160)):''}</div>`:''}`;
    w.appendChild(el); nodeEvents(el,n); }
  drawEdges(); $('#json').value=JSON.stringify(G,null,1); inputsBar(); }
function drawEdges(){ const svg=$('#edges'); let h='';
  G.edges.forEach((e,i)=>{ const a=portPos(e.from[0],e.from[1],'outs'), b=portPos(e.to[0],e.to[1],'ins'); if(!a||!b) return;
    let cls=SELE===i?'sel':''; const st=nodeState(e.from[0]);
    if(st&&st.outputs){ const v=st.outputs[e.from[1]]; cls+= v===undefined||v==='(skipped)'?' dead':' live'; }
    h+=`<path d="${curve(a,b)}" data-e="${i}" class="${cls}"/>`; });
  svg.innerHTML=h; svg.querySelectorAll('path').forEach(p=>p.onclick=ev=>{ ev.stopPropagation(); SELE=+p.dataset.e; SEL=null; drawEdges(); inspect(); }); }
function applyView(){ $('#world').style.transform=`translate(${view.x}px,${view.y}px) scale(${view.k})`; }
function nodeEvents(el,n){
  el.querySelector('.h').onpointerdown=e=>{ e.stopPropagation(); SEL=n.id; SELE=null; $$('.nd').forEach(x=>x.classList.toggle('sel',x.dataset.id===n.id)); inspect();
    const sx=e.clientX, sy=e.clientY, ox=n.x, oy=n.y;
    const mv=ev=>{ n.x=Math.round(ox+(ev.clientX-sx)/view.k); n.y=Math.round(oy+(ev.clientY-sy)/view.k); el.style.left=n.x+'px'; el.style.top=n.y+'px'; drawEdges(); };
    const up=()=>{ removeEventListener('pointermove',mv); removeEventListener('pointerup',up); if(n.x!==ox||n.y!==oy){ dirty=true; $('#json').value=JSON.stringify(G,null,1); } };
    addEventListener('pointermove',mv); addEventListener('pointerup',up); };
  el.onpointerdown=e=>{ e.stopPropagation(); if(SEL!==n.id){ SEL=n.id; SELE=null; $$('.nd').forEach(x=>x.classList.toggle('sel',x.dataset.id===n.id)); inspect(); } };
  el.querySelectorAll('.outs .dot').forEach(d=>d.onpointerdown=e=>{ e.stopPropagation(); wire(n.id,d.parentElement.dataset.p,e); });
  el.querySelectorAll('.ins .dot').forEach(d=>d.onpointerdown=e=>{ e.stopPropagation();       // pull an existing wire off an input
    const i=G.edges.findIndex(x=>x.to[0]===n.id&&x.to[1]===d.parentElement.dataset.p); if(i<0) return; const ed=G.edges.splice(i,1)[0]; changed(); wire(ed.from[0],ed.from[1],e); }); }
function wire(id,port,e){ const a=portPos(id,port,'outs'), r=$('#wrap').getBoundingClientRect(), tmp=$('#tmp');
  const mv=ev=>{ const b={x:(ev.clientX-r.left-view.x)/view.k, y:(ev.clientY-r.top-view.y)/view.k}; tmp.innerHTML=`<path d="${curve(a,b)}" style="fill:none;stroke:var(--acc);stroke-width:2;stroke-dasharray:5 4"/>`; };
  const up=ev=>{ removeEventListener('pointermove',mv); removeEventListener('pointerup',up); tmp.innerHTML='';
    const t=document.elementFromPoint(ev.clientX,ev.clientY), p=t&&t.closest('.ins .p'), nd=t&&t.closest('.nd');
    if(p&&nd&&nd.dataset.id!==id){ const to=[nd.dataset.id,p.dataset.p]; G.edges=G.edges.filter(x=>!(x.to[0]===to[0]&&x.to[1]===to[1])); G.edges.push({from:[id,port],to}); changed(); } };
  addEventListener('pointermove',mv); addEventListener('pointerup',up); mv(e); }
$('#wrap').onpointerdown=e=>{ if(e.target.closest('.nd')||e.target.closest('#zoom')) return; SEL=null; SELE=null; $$('.nd').forEach(x=>x.classList.remove('sel')); drawEdges(); inspect();
  const sx=e.clientX-view.x, sy=e.clientY-view.y; $('#wrap').classList.add('panning');
  const mv=ev=>{ view.x=ev.clientX-sx; view.y=ev.clientY-sy; applyView(); };
  const up=()=>{ removeEventListener('pointermove',mv); removeEventListener('pointerup',up); $('#wrap').classList.remove('panning'); };
  addEventListener('pointermove',mv); addEventListener('pointerup',up); };
function zoomAt(f,cx,cy){ const k=Math.min(2.5,Math.max(0.25,view.k*f)); view.x=cx-(cx-view.x)*k/view.k; view.y=cy-(cy-view.y)*k/view.k; view.k=k; applyView(); }
$('#wrap').onwheel=e=>{ e.preventDefault(); const r=$('#wrap').getBoundingClientRect(); zoomAt(e.deltaY<0?1.1:1/1.1,e.clientX-r.left,e.clientY-r.top); };
$('#zin').onclick=()=>{ const r=$('#wrap').getBoundingClientRect(); zoomAt(1.2,r.width/2,r.height/2); };
$('#zout').onclick=()=>{ const r=$('#wrap').getBoundingClientRect(); zoomAt(1/1.2,r.width/2,r.height/2); };
function fit(){ if(!G.nodes.length){ view={x:0,y:0,k:1}; applyView(); return; } const r=$('#wrap').getBoundingClientRect();
  const xs=G.nodes.map(n=>n.x), ys=G.nodes.map(n=>n.y), x0=Math.min(...xs)-30, y0=Math.min(...ys)-30, x1=Math.max(...xs)+230, y1=Math.max(...ys)+170;
  const k=Math.min(1.2,Math.max(0.25,Math.min(r.width/(x1-x0),r.height/(y1-y0)))); view={k,x:(r.width-(x1-x0)*k)/2-x0*k,y:(r.height-(y1-y0)*k)/2-y0*k}; applyView(); }
$('#zfit').onclick=fit;

// ---------------------------------------------------------------- inspector
function field(n,p){ const v=n.params[p.name], id='pp-'+p.name, h=p.help?` title="${esc(p.help)}"`:'';
  let inp;
  if(p.kind==='bool') inp=`<input type="checkbox" id="${id}"${v?' checked':''} style="width:auto">`;
  else if(p.kind==='choice') inp=`<select id="${id}">${p.choices.map(c=>`<option${c===v?' selected':''}>${esc(c)}</option>`).join('')}</select>`;
  else if(p.kind==='text') inp=`<textarea id="${id}" rows="3">${esc(v??'')}</textarea>`;
  else if(p.kind==='json') inp=`<textarea id="${id}" rows="4">${esc(v==null?'':JSON.stringify(v,null,1))}</textarea>`;
  else if(p.kind==='profile') inp=`<select id="${id}"><option value="">(the graph's)</option>${PROFILES.map(c=>`<option${c===v?' selected':''}>${esc(c)}</option>`).join('')}</select>`;
  else if(p.kind==='graph') inp=`<select id="${id}"><option value=""></option>${CAT.graphs.map(g=>`<option${g.name===v?' selected':''}>${esc(g.name)}</option>`).join('')}</select>`;
  else inp=`<input id="${id}" value="${esc(v??'')}"${p.kind==='model'?' list="models"':''}${['int','float'].includes(p.kind)?' inputmode="decimal"':''}>`;
  return `<label${h}>${esc(p.name.replace(/_/g,' '))}${p.help?' <span class="note">· '+esc(p.help)+'</span>':''}</label>${inp}`; }
function inspect(){ const box=$('#insp');
  if(SELE!=null&&G.edges[SELE]){ const e=G.edges[SELE]; box.innerHTML=`<h3>Wire</h3><div>${esc(e.from.join('.'))} → ${esc(e.to.join('.'))}</div><button class="bad" id="edel" style="margin-top:.5rem">Remove</button>`;
    $('#edel').onclick=()=>{ G.edges.splice(SELE,1); SELE=null; changed(); inspect(); }; return; }
  const n=G.nodes.find(n=>n.id===SEL); if(!n){ box.innerHTML='<div class="note">Select a node to edit it. Delete removes it; Ctrl+D duplicates.</div>'+(G.description?`<h3 style="margin-top:.8rem">${esc(G.name||'graph')}</h3><div class="note">${esc(G.description)}</div>`:''); return; }
  const t=typeOf(n);
  box.innerHTML=`<h3>${esc(t.title)}</h3><div class="note">${esc(t.help)}</div>${t.acts&&t.acts.length?`<div class="note">permissions: ${esc(t.acts.join(', '))}</div>`:''}
    <label>id</label><input id="nid" value="${esc(n.id)}">${t.params.map(p=>field(n,p)).join('')}
    <details style="margin-top:.6rem"><summary class="note">run this node alone</summary>${t.inputs.map(p=>`<label>${esc(p.name)}</label><textarea data-ti="${p.name}" rows="2"></textarea>`).join('')}
      <button id="trynode" style="margin-top:.3rem">Run node</button><div id="tryout"></div></details>
    <div style="margin-top:.6rem;display:flex;gap:.3rem"><button id="ndup">Duplicate</button><button class="bad" id="ndel">Delete</button></div>`;
  $('#nid').onchange=e=>{ const v=e.target.value.trim(); if(!/^[A-Za-z0-9_-]{1,40}$/.test(v)||G.nodes.some(x=>x.id===v)) { e.target.value=n.id; return; }
    G.edges.forEach(ed=>{ if(ed.from[0]===n.id) ed.from[0]=v; if(ed.to[0]===n.id) ed.to[0]=v; }); n.id=v; SEL=v; changed(); };
  t.params.forEach(p=>{ const el=document.getElementById('pp-'+p.name); if(!el) return; el.onchange=()=>{
    let v=p.kind==='bool'?el.checked:el.value;
    if(p.kind==='int') v=parseInt(v||'0'); if(p.kind==='float') v=parseFloat(v||'0');
    if(p.kind==='json'){ try{ v=el.value.trim()?JSON.parse(el.value):null; el.style.borderColor=''; }catch(err){ el.style.borderColor='var(--no)'; return; } }
    n.params[p.name]=v; dirty=true; draw(); }; });
  $('#ndel').onclick=()=>delNode(n.id); $('#ndup').onclick=()=>dup(n);
  $('#trynode').onclick=async()=>{ const ins={}; $$('[data-ti]').forEach(el=>{ if(el.value) { try{ ins[el.dataset.ti]=JSON.parse(el.value); }catch{ ins[el.dataset.ti]=el.value; } } });
    $('#tryout').innerHTML='<span class="note">running…</span>';
    try{ const r=await api('/api/graphs/node',{graph:G,node:n.id,inputs:ins,...runOpts()});
      $('#tryout').innerHTML=r.ok?`<span class="note">${r.ms??''} ms</span>`+Object.entries(r.outputs).map(([k,v])=>`<div><b>${esc(k)}</b><pre>${esc(typeof v==='string'?v:JSON.stringify(v,null,1))}</pre></div>`).join('')+(r.note?`<div class="note">${esc(r.note)}</div>`:'')
        :`<pre class="err">${esc(r.error)}</pre>`; }catch(e){ $('#tryout').innerHTML=`<span class="err">${esc(e.message)}</span>`; } }; }
function delNode(id){ G.nodes=G.nodes.filter(n=>n.id!==id); G.edges=G.edges.filter(e=>e.from[0]!==id&&e.to[0]!==id); SEL=null; changed(); inspect(); }
function dup(n){ const c=JSON.parse(JSON.stringify(n)); let i=1; while(G.nodes.some(x=>x.id===n.id+'_'+i)) i++; c.id=n.id+'_'+i; c.x+=30; c.y+=30; G.nodes.push(c); SEL=c.id; changed(); inspect(); }
addEventListener('keydown',e=>{ const typing=/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if((e.ctrlKey||e.metaKey)&&e.key==='s'){ e.preventDefault(); save(); }
  else if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){ e.preventDefault(); run(); }
  else if(!typing&&(e.key==='Delete'||e.key==='Backspace')){ if(SEL) delNode(SEL); else if(SELE!=null){ G.edges.splice(SELE,1); SELE=null; changed(); inspect(); } }
  else if(!typing&&(e.ctrlKey||e.metaKey)&&e.key==='d'&&SEL){ e.preventDefault(); dup(G.nodes.find(n=>n.id===SEL)); } });
function changed(){ dirty=true; draw(); }

// ---------------------------------------------------------------- inputs, running
let INAMES='';
function inputsBar(){ const names=[...new Set(G.nodes.filter(n=>n.type==='input').map(n=>n.params.name||'input'))]; const old={};
  if(names.join('\n')===INAMES) return; INAMES=names.join('\n');
  $$('#inputs input').forEach(i=>old[i.dataset.n]=i.value);
  $('#inputs').innerHTML=names.map(n=>`<input data-n="${esc(n)}" placeholder="${esc(n)}" value="${esc(old[n]??'')}" title="input ${esc(n)}">`).join(''); }
function runOpts(){ return {model:$('#model').value.trim(), profile:$('#profile').value}; }
async function run(){ const inputs={}; $$('#inputs input').forEach(i=>inputs[i.dataset.n]=i.value);
  try{ status('running…'); const r=await api('/api/graphs/run',{graph:G,inputs,...runOpts()}); poll(r.id); tab('trace'); }catch(e){ status(e.message,true); } }
async function poll(id){ for(;;){ let r; try{ r=await api('/api/graphs/runs/'+id); }catch(e){ status(e.message,true); return; }
    RUN=r; showRun(r); if(!['running','queued'].includes(r.status)) return; await new Promise(z=>setTimeout(z,700)); } }
function showRun(r){ draw();
  status(`${r.status}${r.error?': '+r.error:''}${r.steps?' · '+r.steps+' steps':''}`, r.status==='failed');
  $('#trace').innerHTML=(r.outputs&&Object.keys(r.outputs).length?'<h4 style="margin:.2rem 0">Outputs</h4>'+Object.entries(r.outputs).map(([k,v])=>`<div><b>${esc(k)}</b><pre>${esc(typeof v==='string'?v:JSON.stringify(v,null,1))}</pre></div>`).join(''):'')+
    `<table><tr><th>node</th><th>status</th><th>ms</th><th>in</th><th>out</th></tr>${(r.trace||[]).map(t=>`<tr><td>${esc(t.node)} <span class="note">${esc(t.type)}</span></td><td><span class="chip ${t.status}">${t.status}</span></td><td>${t.ms??''}</td>
      <td><pre>${esc(JSON.stringify(t.inputs||{},null,1)).slice(0,1500)}</pre></td><td>${t.error?`<pre class="err">${esc(t.error)}</pre>`:`<pre>${esc(JSON.stringify(t.outputs||{},null,1)).slice(0,1500)}</pre>`}</td></tr>`).join('')}</table>`;
  const w=r.status==='waiting'&&r.waiting; $('#nwait').textContent=w?'(1)':'';
  $('#approve').innerHTML=w?`<div><b>${esc(w.question)}</b> <span class="note">node ${esc(w.node)}</span></div><textarea id="apv" rows="5">${esc(typeof w.value==='string'?w.value:JSON.stringify(w.value,null,1))}</textarea>
    <input id="apnote" placeholder="note (sent on reject)" style="width:100%;margin:.3rem 0"><button class="pri" id="apyes">Approve</button> <button class="bad" id="apno">Reject</button>`:'<span class="note">Nothing is waiting for you.</span>';
  if(w){ tab('approve'); $('#apyes').onclick=()=>decide(r.id,w.node,true); $('#apno').onclick=()=>decide(r.id,w.node,false); } }
async function decide(id,node,approve){ try{ await api(`/api/graphs/runs/${id}/decide`,{node,approve,value:$('#apv').value,note:$('#apnote').value}); tab('trace'); poll(id); }catch(e){ status(e.message,true); } }
$('#run').onclick=run;
$('#evrun').onclick=async()=>{ try{ const r=await api('/api/graphs/eval',{graph:G,cases:$('#cases').value,expect:$('#expect').value,output:$('#evout').value,grade:$('#grade').value,...runOpts()});
  $('#evres').innerHTML='<span class="note">running the cases…</span>';
  for(;;){ const x=await api('/api/graphs/runs/'+r.id); if(x.status==='running'){ await new Promise(z=>setTimeout(z,1000)); continue; }
    $('#evres').innerHTML=x.error?`<span class="err">${esc(x.error)}</span>`:`<div><b>${x.passed}/${x.cases}</b> passed (${Math.round((x.rate||0)*100)}%)</div><table><tr><th></th><th>inputs</th><th>got</th><th>status</th></tr>${(x.rows||[]).map(w=>`<tr><td>${w.ok?'<span class="okm">✓</span>':'<span class="err">✗</span>'}</td><td><pre>${esc(JSON.stringify(w.inputs))}</pre></td><td><pre>${esc(w.got||w.error||'')}</pre></td><td>${esc(w.status)}</td></tr>`).join('')}</table>`; break; } }catch(e){ $('#evres').innerHTML=`<span class="err">${esc(e.message)}</span>`; } };

// ---------------------------------------------------------------- files
async function load(name){ try{ G=await api('/api/graphs/get?name='+encodeURIComponent(name)); }catch(e){ status(e.message,true); return; }
  if([...$('#gsel').options].some(o=>o.value===name)) $('#gsel').value=name;
  if(name.startsWith('template:')) G.name=''; G.edges=G.edges||[]; SEL=null; SELE=null; RUN=null; dirty=false; $('#gname').value=G.name||''; draw(); inspect(); requestAnimationFrame(()=>{ fit(); drawEdges(); }); }
async function save(){ G.name=$('#gname').value.trim(); if(!G.name){ status('name the graph first',true); $('#gname').focus(); return; }
  try{ await api('/api/graphs/save',{graph:G}); dirty=false; status('saved'); await refresh(G.name); }catch(e){ status(e.message,true); } }
$('#save').onclick=save;
$('#del').onclick=async()=>{ const n=$('#gsel').value; if(n.startsWith('template:')||!n||!confirm('Delete graph '+n+'?')) return; await api('/api/graphs/delete',{name:n}); await refresh(''); G={name:'',nodes:[],edges:[]}; draw(); };
$('#gsel').onchange=e=>{ if(dirty&&!confirm('Discard the changes to this graph?')){ e.target.value=G.name||''; return; } if(e.target.value==='') { G={name:'',description:'',nodes:[],edges:[]}; $('#gname').value=''; draw(); inspect(); } else load(e.target.value); };
$('#jsonApply').onclick=()=>{ try{ const g=JSON.parse($('#json').value); if(!Array.isArray(g.nodes)) throw new Error('nodes must be a list'); G=g; G.edges=G.edges||[]; changed(); status('applied'); }catch(e){ status(e.message,true); } };
async function refresh(sel){ CAT=await api('/api/graphs'); T=Object.fromEntries(CAT.types.map(t=>[t.name,t]));
  $('#gsel').innerHTML=`<option value="">(new graph)</option><optgroup label="your graphs">${CAT.graphs.map(g=>`<option value="${esc(g.name)}"${g.name===sel?' selected':''} title="${esc(g.description)}">${esc(g.name)}</option>`).join('')}</optgroup>`+
    `<optgroup label="start from a template">${CAT.templates.map(t=>`<option value="template:${esc(t.name)}" title="${esc(t.description)}">${esc(t.name)}</option>`).join('')}</optgroup>`;
  $('#runs').innerHTML=`<table><tr><th>when</th><th>graph</th><th>kind</th><th>status</th></tr>${CAT.runs.map(r=>`<tr data-run="${r.id}" style="cursor:pointer"><td>${new Date((r.started||0)*1000).toLocaleString()}</td><td>${esc(r.graph)}</td><td>${esc(r.kind||'run')}</td><td><span class="chip ${r.status}">${esc(r.status)}</span>${r.error?' <span class="err">'+esc(r.error).slice(0,120)+'</span>':''}</td></tr>`).join('')}</table>`;
  $$('[data-run]').forEach(tr=>tr.onclick=()=>{ tab('trace'); poll(tr.dataset.run); }); palette(); }
function tab(b){ $$('#btabs [data-b]').forEach(x=>x.classList.toggle('on',x.dataset.b===b)); $$('.bp').forEach(x=>x.classList.toggle('on',x.dataset.b===b)); }
$$('#btabs [data-b]').forEach(x=>x.onclick=()=>tab(x.dataset.b));
$('#tPal').onclick=()=>{ $('#main').classList.toggle('pal'); $('#main').classList.remove('insp'); };
$('#tInsp').onclick=()=>{ $('#main').classList.toggle('insp'); $('#main').classList.remove('pal'); };
addEventListener('beforeunload',e=>{ if(dirty){ e.preventDefault(); e.returnValue=''; } });
let PROFILES=[];
(async()=>{ await refresh('');
  try{ const p=await api('/api/settings/permissions'); PROFILES=Object.keys(Object.assign({},p.presets,p.policy.profiles));
    $('#profile').innerHTML='<option value="">permissions: per job</option>'+PROFILES.map(n=>`<option>${esc(n)}</option>`).join(''); }catch{}
  try{ const m=await api('/v1/models'); $('#models').innerHTML=['auto',...(m.data||[]).map(x=>x.id)].map(x=>`<option value="${esc(x)}">`).join(''); }catch{}
  const want=decodeURIComponent((location.hash.match(/graph=([^&]+)/)||[])[1]||'');
  if(want) load(want); else if(CAT.graphs.length) { $('#gsel').value=CAT.graphs[0].name; load(CAT.graphs[0].name); } else { $('#gsel').value='template:screen-and-verify'; load('template:screen-and-verify'); }
  applyView(); })();
</script>
"""

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
