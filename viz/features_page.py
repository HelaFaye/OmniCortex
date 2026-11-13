"""Studio's /features page: feature dictionaries, what a reply was made of, attribution graphs, emotion vectors,
and steering with both. The heavy work runs as jobs (scripts/features.py, feature_circuits.py, emotions.py) or in
the decoding lab (scripts/feature_lab.py), which this page reaches through Studio."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Features</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0;--pos:#d9480f;--neg:#1c7ed6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--pos:#ff8a4c;--neg:#5aa9ff}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--pos:#ff8a4c;--neg:#5aa9ff}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;align-items:center;gap:.8rem;padding:.55rem 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header .r{margin-left:auto;display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;font-size:12.5px;color:var(--mut)}
nav.tabs{display:flex;gap:.2rem;padding:.4rem 16px 0;border-bottom:1px solid var(--line);background:var(--panel);overflow-x:auto}
nav.tabs button{border:none;border-bottom:2px solid transparent;border-radius:0;background:none;padding:.45rem .75rem;color:var(--mut);white-space:nowrap}
nav.tabs button.on{color:var(--fg);border-bottom-color:var(--acc)}
@media (max-width:600px){nav.tabs{flex-wrap:wrap;overflow-x:visible}nav.tabs button{padding:.4rem .55rem}}
main{max-width:1280px;margin:0 auto;padding:1rem 16px 3rem}
section[data-tab]{display:none;gap:1rem}section[data-tab].on{display:grid}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem;min-width:0}
h2{font-size:13px;margin:0 0 .6rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
h2 .r{margin-left:auto;text-transform:none;letter-spacing:0;font-weight:400}
h3{font-size:12.5px;margin:.9rem 0 .35rem}
.note{font-size:12.5px;color:var(--mut)}a{color:var(--acc)}
button{font:500 13px ui-sans-serif,system-ui;padding:.32rem .75rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--acc)}button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button:disabled{opacity:.5;cursor:default}
input,select,textarea{font:13px ui-sans-serif,system-ui;padding:.3rem .45rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);min-width:0}
textarea{width:100%;min-height:4.5rem;resize:vertical}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.grow{flex:1;min-width:10rem}
label.f{display:flex;flex-direction:column;gap:.15rem;font-size:12px;color:var(--mut)}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:1rem;align-items:start}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.steps{display:grid;gap:.5rem;counter-reset:s}
.step{display:grid;grid-template-columns:1.6rem 1fr auto;gap:.6rem;align-items:center;padding:.5rem .6rem;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.step .n{width:1.6rem;height:1.6rem;border-radius:50%;display:grid;place-items:center;font:600 12px ui-sans-serif;border:1px solid var(--line)}
.step.done .n{background:var(--ok);color:#fff;border-color:var(--ok)}
.step .t{font-weight:600}.step .d{font-size:12.5px;color:var(--mut)}
.pill{font-size:11.5px;border:1px solid var(--line);border-radius:999px;padding:.05rem .5rem;white-space:nowrap}
.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.off{color:var(--mut)}.pill.warn{color:var(--warn);border-color:var(--warn)}
.list{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.list .it{display:grid;grid-template-columns:4.2rem 1fr 7.2rem;gap:.5rem;padding:.28rem .5rem;border-bottom:1px solid var(--line);cursor:pointer;font-size:12.5px}
.list .it:hover,.list .it.on{background:var(--code)}.list .it .lab{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mono{font:12px ui-monospace,monospace}
.ex{font:12.5px/1.75 ui-monospace,monospace;padding:.2rem .3rem;border-bottom:1px dashed var(--line);white-space:pre-wrap;word-break:break-word}
.tk{border-radius:3px;padding:0 .5px}
.chips{display:flex;gap:.3rem;flex-wrap:wrap}.chip{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:5px;padding:0 .35rem;cursor:default;white-space:pre}
.chip.up{border-color:var(--pos)}.chip.dn{border-color:var(--neg)}.chip.link{cursor:pointer}.chip.link:hover{border-color:var(--acc)}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:.6rem}.cmp pre{margin:0}
@media(max-width:700px){.cmp{grid-template-columns:1fr}}
pre{background:var(--code);padding:.6rem;border-radius:6px;font:12.5px/1.5 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;margin:.3rem 0}
table{width:100%;border-collapse:collapse}td,th{padding:.28rem .3rem;border-bottom:1px solid var(--line);text-align:left;font-size:12.5px}
th{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:500}
.bar{height:7px;border-radius:3px;background:var(--acc);display:inline-block;vertical-align:middle}
.strip{font:13px/2 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;padding:.4rem;background:var(--bg);border:1px solid var(--line);border-radius:8px;max-height:340px;overflow:auto}
.strip .tk{cursor:pointer}.strip .tk.sel{outline:2px solid var(--acc)}
svg{display:block;max-width:100%}canvas{display:block;max-width:100%;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.err{color:var(--no);font-size:12.5px}.busy{color:var(--warn);font-size:12.5px}
details>summary{cursor:pointer;font-size:12.5px;color:var(--mut)}
.i{display:inline-grid;place-items:center;width:15px;height:15px;border-radius:50%;border:1px solid var(--mut);color:var(--mut);font:600 10px ui-sans-serif;cursor:help;margin-left:.2rem}
.gwrap{overflow:auto;border:1px solid var(--line);border-radius:8px;background:var(--bg);max-height:620px}
.slider{display:flex;gap:.4rem;align-items:center}.slider input[type=range]{flex:1;min-width:6rem}
.legend{display:flex;gap:.8rem;flex-wrap:wrap;font-size:12px;color:var(--mut)}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.25rem;vertical-align:middle}
</style>
<header><h1>Studio · Features</h1><span id="labpill" class="pill off" title="The decoding lab holds the model for reading replies, graphs and steering">lab: checking…</span>
<span class="r"><label for="labUrl">Lab address<span class="i" title="The decoding lab (Jobs → Decoding lab) answers on this computer only. Studio passes your requests to it.">i</span></label>
<input id="labUrl" style="width:13rem" aria-label="Decoding lab address"><button id="labStart" title="Start the decoding lab with a model">Start lab…</button></span></header>
<nav class="tabs" id="ftabs" role="tablist">
<button data-t="start" role="tab">Start here</button><button data-t="dicts" role="tab">Dictionaries</button>
<button data-t="read" role="tab">Read a reply</button><button data-t="circuits" role="tab">Circuits</button>
<button data-t="emotions" role="tab">Emotions</button><button data-t="steer" role="tab">Steer</button></nav>
<main>
<!-- ======================================================== start -->
<section data-tab="start">
<div class="card"><h2>What this page does</h2>
<p class="note" style="margin:0">A neuron takes part in many unrelated things, so reading neurons one by one rarely explains a model. These tools find
<b>features</b>: directions in a layer's activations that each stand for one thing. They show what each feature responds to, which
neurons carry it, which features a reply was built from and how they feed each other, and they change features to see what
the model does without them. <b>Emotion vectors</b> do the same for emotion concepts. Everything here is optional; nothing runs
unless you start it. Methods and their sources: <a href="/help/FEATURES">Features guide</a> · <a href="/help/CREDITS">Credits</a>.</p></div>
<div class="cols">
<div class="card"><h2>Feature dictionaries</h2><div class="steps" id="stepsF"></div></div>
<div class="card"><h2>Emotion vectors</h2><div class="steps" id="stepsE"></div>
<h3>Already trained elsewhere?</h3><p class="note" style="margin:0">Dictionaries from SAELens or Gemma Scope load with
<a href="/jobs#kind=feat_import">Import a dictionary</a>.</p></div></div>
</section>
<!-- ======================================================== dictionaries -->
<section data-tab="dicts">
<div class="card"><div class="row"><label for="dsel" class="note">Dictionary</label><select id="dsel" class="grow" style="max-width:34rem"></select>
<span id="dstats" class="note"></span></div></div>
<div class="cols">
<div class="card"><h2>All features <span class="r note" id="dcount"></span></h2>
<div class="row"><input id="dq" class="grow" placeholder="Search labels or output tokens, or type an id" aria-label="Search features">
<select id="dsort" aria-label="Sort features" title="Order of the list. Features active on over a fifth of all tokens are marked dense: they rarely mean one thing"><option value="typ">frequent, dense last</option><option value="freq">most frequent</option><option value="max">strongest</option><option value="score">best label score</option><option value="id">id</option><option value="rare">rarest</option></select>
<label class="note"><input type="checkbox" id="dlab"> labelled</label></div>
<div class="list" id="dlist" style="margin-top:.5rem"></div>
<details style="margin-top:.6rem"><summary>Map of all features</summary>
<p class="note">Each dot is a feature, placed by its direction (two principal components); nearby dots point similar ways. Colour: how often it fires. Click a dot to open it.</p>
<canvas id="dmap" width="560" height="380"></canvas></details>
<details><summary>How often features fire</summary><svg id="ddens" viewBox="0 0 560 140"></svg>
<p class="note">Features per frequency (log10 share of tokens). A pile at the far left means many features almost never fire.</p></details>
</div>
<div class="card" id="fcard"><p class="note">Pick a feature on the left.</p></div>
</div>
</section>
<!-- ======================================================== read a reply -->
<section data-tab="read">
<div class="card"><h2>Read a reply <span class="i" title="Runs the text through the model in the lab: per token, the features of the chosen dictionaries, emotion activations and what each layer would have said next">i</span></h2>
<div class="row"><label class="note"><input type="radio" name="rmode" value="text" checked> any text</label>
<label class="note"><input type="radio" name="rmode" value="chat"> a prompt and the model's reply</label></div>
<div id="rtext"><textarea id="rT" placeholder="Paste text"></textarea></div>
<div id="rchat" style="display:none"><textarea id="rP" placeholder="The prompt"></textarea><textarea id="rR" placeholder="The reply (only the reply's tokens are read)"></textarea></div>
<div class="row" style="margin-top:.4rem"><span class="note">Dictionaries:</span><span id="rD" class="chips"></span></div>
<div class="row" style="margin-top:.3rem"><label class="note">Emotions <select id="rE"><option value="">none</option></select></label>
<label class="note"><input type="checkbox" id="rL"> logit lens</label><button class="pri" id="rGo">Read</button><span id="rMsg"></span></div></div>
<div class="cols" id="rOut" style="display:none">
<div class="card"><h2>Tokens <span class="r"><label class="note">Colour by <select id="rColor"></select></label></span></h2>
<div class="strip" id="rStrip"></div><div id="rTimeline"></div></div>
<div class="card" id="rTok"><p class="note">Click a token.</p></div>
<div class="card" style="grid-column:1/-1" id="rMost"></div>
</div>
</section>
<!-- ======================================================== circuits -->
<section data-tab="circuits">
<div class="card"><h2>Attribution graph <span class="i" title="Which transcoder features the model used to produce its next token and how they feed each other: MLPs replaced by transcoders plus error terms, attention and norms frozen, edges = direct linear effects, pruned by influence on the output (Anthropic's circuit tracing)">i</span></h2>
<div class="row"><span class="note">Transcoders:</span><span id="gT" class="chips"></span></div>
<p class="note" id="gTnone" style="display:none">No transcoders yet. Train one per layer you want in the graph:
<a id="gTlink" href="/jobs#kind=feat_train&site=mlp_in&target_site=mlp_out">Train a transcoder</a> (needs mlp_in and mlp_out recorded). Layers without one count as error nodes.</p>
<textarea id="gP" placeholder="Prompt, e.g. Fact: the capital of the state containing Dallas is"></textarea>
<div class="row"><label class="note"><input type="checkbox" id="gChat"> chat template</label>
<label class="note">feature nodes <input id="gMax" type="number" value="300" min="20" max="5000" style="width:5.5rem" title="Features to attribute, most influential first"></label>
<label class="note">keep nodes <input id="gNode" type="number" value="0.8" step="0.05" min="0.1" max="1" style="width:4.5rem" title="Keep the nodes making up this share of influence on the output (paper default 0.8)"></label>
<label class="note">keep edges <input id="gEdge" type="number" value="0.98" step="0.01" min="0.1" max="1" style="width:4.5rem" title="Keep the edges making up this share of influence (paper default 0.98)"></label>
<button class="pri" id="gGo">Build in the lab</button><a id="gJob" href="/jobs#kind=feat_graph">or run as a job</a><span id="gMsg"></span></div>
<div class="row" style="margin-top:.4rem"><label class="note">Saved graphs <select id="gSaved"><option value="">…</option></select></label></div></div>
<div class="cols" id="gOut" style="display:none"><div class="card"><h2>Graph <span class="r note" id="gScore"></span></h2>
<div class="legend"><span><span class="sw" style="background:var(--acc)"></span>feature</span><span><span class="sw" style="background:var(--mut)"></span>error (unexplained)</span>
<span><span class="sw" style="background:var(--fg)"></span>prompt token</span><span><span class="sw" style="background:var(--ok)"></span>output token</span>
<span><span class="sw" style="background:var(--pos)"></span>raises</span><span><span class="sw" style="background:var(--neg)"></span>lowers</span></div>
<div class="gwrap"><svg id="gSvg"></svg></div><p class="note">Bottom: prompt tokens. Top: output tokens. Rows are layers, columns positions. Click a node to follow its edges and to intervene.</p></div>
<div class="card" id="gNodeCard"><p class="note">Click a node.</p></div></div>
<div class="card"><h2>What drives a completion (one dictionary) <span class="i" title="Scaling Monosemanticity's attribution: activation times the gradient of (token logit minus baseline) along the feature, checked by clamping the top features to zero">i</span></h2>
<div class="row"><select id="aD" class="grow"></select><input id="aTok" placeholder="token, e.g. ' Austin'" style="width:10rem"><input id="aBase" placeholder="baseline (optional)" style="width:10rem">
<label class="note">ablate <input id="aAbl" type="number" value="10" min="0" max="100" style="width:4rem"></label><label class="note"><input type="checkbox" id="aChat"> chat</label></div>
<textarea id="aP" placeholder="Prompt"></textarea><div class="row"><button class="pri" id="aGo">Attribute</button><span id="aMsg"></span></div><div id="aOut"></div></div>
</section>
<!-- ======================================================== emotions -->
<section data-tab="emotions">
<div class="card"><div class="row"><label for="eSel" class="note">Emotion vectors</label><select id="eSel" class="grow" style="max-width:34rem"></select><span id="eStats" class="note"></span></div></div>
<div class="cols" id="eOut" style="display:none">
<div class="card"><h2>Where they are found <span class="i" title="Held-out stories: does the right emotion's vector score highest? Chance is 1 in the number of emotions">i</span></h2><svg id="eAcc" viewBox="0 0 560 170"></svg>
<h2 style="margin-top:.8rem">Map <span class="i" title="The vectors' two main directions at the default layer. The paper finds the first is valence (pleasant vs unpleasant) and the second arousal (intensity)">i</span></h2><svg id="eMap" viewBox="0 0 560 420"></svg></div>
<div class="card"><h2>Similarity <span class="r note" id="eHover"></span></h2><canvas id="eSim" width="420" height="420"></canvas>
<div id="eDetail" style="margin-top:.6rem"></div></div>
<div class="card"><h2>Trace a text</h2><textarea id="eT" placeholder="Text, or leave empty and use Read a reply's input"></textarea>
<div class="row"><label class="note">Colour by <select id="eTE"></select></label><button class="pri" id="eTGo">Trace</button><span id="eTMsg"></span></div>
<div class="strip" id="eTStrip" style="margin-top:.4rem"></div><div id="eTLine"></div></div>
<div class="card"><h2>Sweep a quantity <span class="i" title="The emotion activation just before the reply as one number in the prompt changes (the paper's Tylenol dose example)">i</span></h2>
<input id="eSwT" class="grow" style="width:100%" value="I just took {x} mg of Tylenol for my back pain. Is that okay?">
<div class="row" style="margin-top:.3rem"><input id="eSwV" class="grow" value="500 1000 2000 4000 8000 16000" aria-label="Values"><input id="eSwE" class="grow" value="afraid calm" aria-label="Emotions"><button class="pri" id="eSwGo">Sweep</button><span id="eSwMsg"></span></div>
<svg id="eSwSvg" viewBox="0 0 560 200" style="margin-top:.4rem"></svg></div>
</div>
</section>
<!-- ======================================================== steer -->
<section data-tab="steer">
<div class="card"><h2>Steer <span class="i" title="Feature clamps keep the dictionary's error term and set the features in its reconstruction (Scaling Monosemanticity); values are multiples of the feature's largest observed activation. Emotion strengths are fractions of the residual stream's norm at each token (the emotions paper)">i</span></h2>
<div id="sProf"></div>
<h3>Features</h3><div id="sF"></div><button id="sFAdd">+ feature</button>
<h3>Emotions</h3><div id="sE"></div><button id="sEAdd">+ emotion</button>
<h3>Prompt</h3><textarea id="sP">Tell me about your day.</textarea>
<div class="row"><label class="note">max tokens <input id="sMax" type="number" value="120" min="1" max="2000" style="width:5rem"></label>
<button class="pri" id="sGo">Compare with and without</button><span id="sMsg"></span></div>
<div class="cmp" id="sOut" style="margin-top:.5rem"></div>
<details style="margin-top:.6rem"><summary>Use it elsewhere</summary>
<p class="note">As a decoding-lab variant (TestQA and the retrain gate can then measure it against plain):</p><pre id="sVar"></pre>
<div class="row"><button id="sCopy">Copy</button><a id="sLab" href="/jobs#kind=decode_lab">Open the decoding lab job</a><a id="sCvec" href="/jobs#kind=emo_cvec">Emotion mix as a llama.cpp control vector</a></div></details></div>
</section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,d=3)=>x==null||isNaN(x)?'–':(+x).toFixed(d);
const pct=x=>x==null?'–':!x?'0%':(100*x).toFixed(x<0.001?3:x<0.01?2:1)+'%';
function ls(k,v){ try{ if(v===undefined) return localStorage.getItem(k); localStorage.setItem(k,v); }catch(e){ return null; } }
async function api(path,body){ const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json().catch(()=>({error:'HTTP '+r.status})); if(!r.ok) throw new Error(j.error||('HTTP '+r.status)); return j; }
const labUrl=()=>($('#labUrl').value||'http://127.0.0.1:8090').trim();
const labCall=(op,body)=>api('/api/features/lab/'+op,{...body,lab:labUrl()});
function busy(el,t){ el.className='busy'; el.textContent=t; } function fail(el,e){ el.className='err'; el.textContent=e.message||e; } function done(el,t){ el.className='note'; el.textContent=t||''; }
let LIST={dicts:[],acts:[],emotions:[],stories:[],graphs:[]}, LAB=null;
// ---------------------------------------------------------------- tabs
function tab(t){ $$('nav.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t)); $$('section[data-tab]').forEach(s=>s.classList.toggle('on',s.dataset.tab===t)); ls('ns-feat-tab',t);
  if(location.hash.slice(1).split('?')[0]!==t) history.replaceState(null,'','#'+t); }
$$('nav.tabs button').forEach(b=>b.onclick=()=>tab(b.dataset.t));
// ---------------------------------------------------------------- lab
$('#labUrl').value=ls('ns-feat-lab')||'http://127.0.0.1:8090';
$('#labUrl').onchange=()=>{ ls('ns-feat-lab',labUrl()); checkLab(); };
async function checkLab(){ const p=$('#labpill');
  try{ LAB=await api('/api/features/lab/info?lab='+encodeURIComponent(labUrl())); if(LAB.running===false) throw new Error(LAB.error||'not running'); p.className='pill ok'; p.textContent=`lab: ${LAB.model.split('/').pop()} · ${LAB.device} ${LAB.dtype}`; p.title=`${LAB.model}, ${LAB.layers} layers`; }
  catch(e){ LAB=null; p.className='pill off'; p.textContent='lab: not running'; p.title=e.message; } }
$('#labStart').onclick=async()=>{ const models=[...new Set([...LIST.dicts,...LIST.acts,...LIST.emotions].map(d=>d.model).filter(Boolean))];
  const m=prompt('Model for the decoding lab (Hugging Face id or folder)'+(models.length?'\nUsed here before: '+models.join(', '):''),models[0]||''); if(!m) return;
  const port=(labUrl().match(/:(\d+)/)||[])[1]||'8090';
  try{ await api('/api/jobs',{kind:'decode_lab',values:{model:m,port:+port,dtype:'auto'}}); const p=$('#labpill'); p.className='pill warn'; p.textContent='lab: loading the model…';
    for(let i=0;i<120;i++){ await new Promise(r=>setTimeout(r,5000)); await checkLab(); if(LAB) return; p.className='pill warn'; p.textContent='lab: loading the model… (Jobs shows its log)'; } }
  catch(e){ alert(e.message); } };
// ---------------------------------------------------------------- start
const jobLink=(kind,vals)=>'/jobs#'+new URLSearchParams({kind,...Object.fromEntries(Object.entries(vals||{}).filter(([k,v])=>v!==undefined&&v!==''))}).toString();
function step(n,ok,title,desc,href,btn){ return `<div class="step ${ok?'done':''}"><span class="n">${ok?'✓':n}</span><div><div class="t">${esc(title)}</div><div class="d">${desc}</div></div><a href="${esc(href)}"><button>${esc(btn)}</button></a></div>`; }
function drawStart(){ const a=LIST.acts[0], d=LIST.dicts.find(x=>!x.target_site), di=LIST.dicts.find(x=>x.interpreted), tc=LIST.dicts.filter(x=>x.target_site==='mlp_out');
  const model=(a||d||{}).model||''; const slug=(model.split('/').pop()||'model').toLowerCase().replace(/[^a-z0-9.-]+/g,'-');
  $('#stepsF').innerHTML=
    step(1,LIST.acts.length,'Record activations',LIST.acts.length?`${LIST.acts.length} recorded · latest ${esc(a.name)}: ${(a.tokens||0).toLocaleString()} tokens at ${esc((a.points||[]).join(', '))}`:'Run the model over text and keep what a few layers produce.',jobLink('feat_collect',{model,out:`runs/features/acts-${slug}`,site:'resid'}),'Record')+
    step(2,LIST.dicts.length,'Train a dictionary',LIST.dicts.length?`${LIST.dicts.length} dictionaries`:'A sparse autoencoder learns the features of one layer.',jobLink('feat_train',{acts:a?.path,site:'resid',out:`runs/features/${slug}-resid`}),'Train')+
    step(3,!!di,'Describe the features',di?`${esc(di.name)} is described`:'Top examples, output effects, a map, and labels written by a model you choose.',jobLink('feat_interpret',{sae:(d||{}).path}),'Describe')+
    step(4,LIST.dicts.some(x=>x.neurons),'Map onto neurons','Which neurons carry each feature, and how many are H-Neurons.',jobLink('feat_neurons',{sae:(di||d||{}).path}),'Map')+
    step(5,LIST.graphs.length,'Trace a reply','Transcoders (one per layer) let the Circuits tab draw which features produced an output.',tc.length?'#circuits':jobLink('feat_train',{acts:a?.path,site:'mlp_in',target_site:'mlp_out',out:`runs/features/${slug}-tc`}),tc.length?'Circuits':'Train a transcoder');
  const s=LIST.stories[0], e=LIST.emotions[0];
  $('#stepsE').innerHTML=
    step(1,LIST.stories.length,'Write stories',s?`${s.stories} stories in ${esc(s.name)}`:'A model you choose writes short stories where a character feels each emotion without naming it.',jobLink('emo_stories',{out:'runs/emotions/stories',preset:'core'}),'Write')+
    step(2,LIST.emotions.length,'Build the vectors',e?`${esc(e.name)}: ${e.n} emotions, best held-out accuracy ${pct(e.best_accuracy)}`:'The model under study reads them; each emotion becomes a direction.',jobLink('emo_build',{stories:s?.path,model,out:`runs/emotions/${slug}`}),'Build')+
    step(3,false,'Explore and steer','Where each emotion is active in a reply, how they relate, and replies with an emotion added or removed.',LIST.emotions.length?'#emotions':'#start','Open'); }
// ---------------------------------------------------------------- dictionaries
let DICT=null, CUR=null, ORDER=[];
function dictOpts(sel,filter,keep){ const v=keep??sel.value; sel.innerHTML=LIST.dicts.filter(filter||(()=>true)).map(d=>`<option value="${esc(d.path)}">${esc(d.name)} · ${esc(d.site)}${d.target_site?'→'+esc(d.target_site):''} L${d.layer} · ${d.n_features} features${d.interpreted?'':' · not described yet'}</option>`).join('')||'<option value="">none yet</option>'; if(v) sel.value=v; }
async function openDict(path,fid){ if(!path) return; ls('ns-feat-dict',path);
  try{ DICT=await api('/api/features/dict?path='+encodeURIComponent(path)); }catch(e){ $('#dstats').innerHTML=`<span class="err">${esc(e.message)}</span>`; return; }
  const i=DICT.info||{}; const sp=i.splice||{}, ev=i.eval||{};
  $('#dstats').innerHTML=`${esc(i.model||'')} · ${esc(i.arch||'')} · held-out FVU ${fmt(ev.fvu)} · L0 ${fmt(ev.l0,1)}${sp.delta!=null?` · loss +${fmt(sp.delta)} with it spliced in`:''}`+
    (i.fvu!=null||ev.fvu>0.3?` <span class="i" title="FVU: share of the activations' variance the dictionary misses (lower is better). L0: features active per token. A large loss increase means the dictionary misses what the model needs; train on more tokens or a larger dictionary">i</span>`:'');
  if(!DICT.interpreted){ $('#dlist').innerHTML=`<p class="note" style="padding:.6rem">Not described yet. <a href="${jobLink('feat_interpret',{sae:path})}">Describe its features</a> to browse them.</p>`; $('#fcard').innerHTML=''; $('#dcount').textContent=''; return; }
  drawDens(); drawMap(); filterList(); if(fid!=null) openFeature(+fid); }
function filterList(){ if(!DICT||!DICT.interpreted) return; const q=$('#dq').value.trim().toLowerCase(), lab=$('#dlab').checked, s=$('#dsort').value; const n=DICT.id.length;
  let idx=[...Array(n).keys()].filter(i=>DICT.freq[i]>0);
  if(lab) idx=idx.filter(i=>DICT.label[i]);
  if(q){ if(/^\d+$/.test(q)) idx=idx.filter(i=>String(DICT.id[i]).startsWith(q)); else idx=idx.filter(i=>(DICT.label[i]||'').toLowerCase().includes(q)); }
  const key={typ:i=>(DICT.freq[i]>0.2?1:0)-DICT.freq[i],freq:i=>-DICT.freq[i],rare:i=>DICT.freq[i],max:i=>-DICT.max[i],score:i=>-(DICT.score[i]??-1),id:i=>i}[s];
  idx.sort((a,b)=>key(a)-key(b)); ORDER=idx;
  $('#dcount').textContent=`${idx.length.toLocaleString()} of ${n.toLocaleString()} (${(DICT.alive||0).toLocaleString()} ever active)`;
  const show=idx.slice(0,400);
  $('#dlist').innerHTML=show.map(i=>`<div class="it ${i===CUR?'on':''}" data-i="${i}"><span class="mono">#${i}</span><span class="lab" title="${esc(DICT.label[i]||'')}">${esc(DICT.label[i]||'')||'<span class="note">no label</span>'}</span><span class="note" style="text-align:right">${DICT.freq[i]>0.2?'<span class="pill warn" title="Active on over a fifth of all tokens">dense</span> ':''}${pct(DICT.freq[i])}</span></div>`).join('')+(idx.length>400?`<p class="note" style="padding:.4rem">${idx.length-400} more: search to narrow</p>`:'');
  if(q && !/^\d+$/.test(q) && DICT.path) api('/api/features/search?path='+encodeURIComponent(DICT.path)+'&q='+encodeURIComponent(q)).then(j=>{ const extra=j.hits.filter(h=>!idx.includes(h.id)); if(extra.length) $('#dlist').insertAdjacentHTML('beforeend',`<p class="note" style="padding:.3rem .5rem">Also matching output tokens:</p>`+extra.map(h=>`<div class="it" data-i="${h.id}"><span class="mono">#${h.id}</span><span class="lab">${esc(h.label)||'<span class="note">no label</span>'}</span><span class="note" style="text-align:right">${pct(h.freq)}</span></div>`).join('')); }).catch(()=>{}); }
$('#dlist').onclick=e=>{ const it=e.target.closest('.it'); if(it) openFeature(+it.dataset.i); };
['#dq','#dsort','#dlab'].forEach(s=>$(s).addEventListener(s==='#dq'?'input':'change',filterList));
$('#dsel').onchange=()=>openDict($('#dsel').value);
function drawDens(){ const h=DICT.density_hist||[], m=Math.max(1,...h), W=560,H=140,bw=(W-40)/Math.max(1,h.length);
  $('#ddens').innerHTML=h.map((v,i)=>`<rect x="${30+i*bw}" y="${H-20-(H-30)*v/m}" width="${bw-1}" height="${(H-30)*v/m}" fill="var(--acc)"><title>10^${(DICT.density_edges||[])[i]}: ${v} features</title></rect>`).join('')+
    [-8,-6,-4,-2,0].map(x=>`<text x="${30+(x+9)/9*(W-40)}" y="${H-5}" font-size="10" fill="var(--mut)" text-anchor="middle">1e${x}</text>`).join(''); }
let MAPPTS=[];
function drawMap(){ const c=$('#dmap'), g=c.getContext('2d'), W=c.width, H=c.height; g.clearRect(0,0,W,H);
  const xy=DICT.xy.map(p=>p||[0,0]); const xs=xy.map(p=>p[0]), ys=xy.map(p=>p[1]); const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const st=getComputedStyle(document.documentElement); const acc=st.getPropertyValue('--acc').trim()||'#6f93ff';
  MAPPTS=xy.map((p,i)=>[10+(p[0]-x0)/(x1-x0||1)*(W-20),10+(p[1]-y0)/(y1-y0||1)*(H-20),i]);
  for(const [x,y,i] of MAPPTS){ const f=DICT.freq[i]; if(!f) continue; const a=Math.min(1,0.25+(Math.log10(f)+6)/6); g.globalAlpha=a; g.fillStyle=DICT.label[i]?st.getPropertyValue('--pos').trim():acc; g.fillRect(x-1.5,y-1.5,3,3); }
  g.globalAlpha=1; if(CUR!=null){ const p=MAPPTS[CUR]; if(p){ g.strokeStyle=st.getPropertyValue('--fg').trim(); g.lineWidth=2; g.strokeRect(p[0]-5,p[1]-5,10,10); } } }
$('#dmap').onclick=e=>{ const r=e.target.getBoundingClientRect(), sx=e.target.width/r.width; const x=(e.clientX-r.left)*sx, y=(e.clientY-r.top)*sx;
  let best=null,bd=1e9; for(const [px,py,i] of MAPPTS){ if(!DICT.freq[i]) continue; const d=(px-x)**2+(py-y)**2; if(d<bd){bd=d;best=i;} } if(best!=null&&bd<200) openFeature(best); };
function tokSpan(t,a,mx,at){ const al=Math.max(0,Math.min(1,a/(mx||1))); return `<span class="tk" style="background:color-mix(in srgb,var(--pos) ${Math.round(al*85)}%,transparent)${at?';text-decoration:underline':''}" title="${fmt(a,2)}">${esc(t).replace(/\n/g,'⏎')}</span>`; }
function exLine(x,mx){ return `<div class="ex">${x.tok.map((t,i)=>tokSpan(t,x.act[i],mx,i===x.at)).join('')}</div>`; }
async function openFeature(i){ CUR=i; $$('#dlist .it').forEach(e=>e.classList.toggle('on',+e.dataset.i===i)); drawMap(); const c=$('#fcard'); c.innerHTML='<p class="busy">loading…</p>';
  history.replaceState(null,'','#dicts?path='+encodeURIComponent(DICT.path)+'&id='+i);
  let f; try{ f=await api(`/api/features/feature?path=${encodeURIComponent(DICT.path)}&id=${i}`); }catch(e){ c.innerHTML=`<p class="err">${esc(e.message)}</p>`; return; }
  const mx=f.max||1, ex=f.examples||{};
  const hist=f.hist||[], hm=Math.max(1,...hist);
  const nrn=f.neurons, nrows=(k,t)=>nrn&&nrn[k]&&nrn[k].length?`<tr><th colspan="3">${t}</th></tr>`+nrn[k].slice(0,6).map(([l,n,w])=>`<tr><td class="mono">L${l} #${n}</td><td><span class="bar" style="width:${Math.min(100,Math.abs(w)/Math.max(...nrn[k].map(r=>Math.abs(r[2])))*100)}px;background:${w<0?'var(--neg)':'var(--pos)'}"></span></td><td class="mono">${fmt(w,3)}</td></tr>`).join(''):'';
  c.innerHTML=`<h2>Feature #${i} <span class="r note">fires on ${pct(f.freq)} of tokens · strongest ${fmt(f.max,2)}</span></h2>
  <div class="row"><input id="fLab" class="grow" value="${esc(f.label||'')}" placeholder="Name this feature" aria-label="Feature label"><button id="fLabSave">Save name</button></div>
  <p class="note" style="margin:.25rem 0 0">${f.label?`label by ${esc(f.label_by||'?')}`:'No label yet: name it yourself, or let a model label features (Describe step).'}${f.label_score!=null?` · specificity ${fmt(f.label_score,2)}${f.label_baseline!=null?` (unrelated text: ${fmt(f.label_baseline,2)})`:''} <span class="i" title="A labeller rated, on Scaling Monosemanticity's 0-3 rubric, how well the label fits text where the feature fires, scaled to 0-1; unrelated text should score near 0">i</span>`:''}</p>
  <h3>Strongest examples</h3>${(ex.top||[]).slice(0,8).map(x=>exLine(x,mx)).join('')||'<p class="note">none</p>'}
  ${ex.buckets?`<details><summary>Across activation levels</summary>${ex.buckets.map(b=>`<p class="note" style="margin:.4rem 0 .1rem">${fmt(b.lo,2)} – ${fmt(b.hi,2)}</p>${b.ex.map(x=>exLine(x,mx)).join('')||'<p class="note">none</p>'}`).join('')}</details>`:''}
  ${ex.random?`<details><summary>Random examples</summary>${ex.random.map(x=>exLine(x,mx)).join('')}</details>`:''}
  ${f.up?`<h3>Raises the chance of <span class="i" title="Through the final norm and the output layer alone (logit lens); later layers can change this">i</span></h3><div class="chips">${f.up.map(([t,v])=>`<span class="chip up" title="${fmt(v,3)}">${esc(t)}</span>`).join('')}</div>
  <h3>Lowers</h3><div class="chips">${f.down.map(([t,v])=>`<span class="chip dn" title="${fmt(v,3)}">${esc(t)}</span>`).join('')}</div>`:''}
  <h3>Activation strengths</h3><svg viewBox="0 0 300 60" style="width:300px">${hist.map((v,k)=>`<rect x="${k*9.5}" y="${55-50*v/hm}" width="8.5" height="${50*v/hm}" fill="var(--acc)"><title>${(f.act_bins||[])[k-1]??0}–${(f.act_bins||[])[k]??'∞'}: ${v}</title></rect>`).join('')}</svg>
  <h3>Nearest features</h3><div class="chips">${(f.nb||[]).map(([j,cv,l])=>`<span class="chip link" data-f="${j}" title="cosine ${cv}">#${j}${l?' '+esc(l):''}</span>`).join('')}</div>
  ${nrn?`<h3>Neurons <span class="note">(${esc((f.neuron_method||[]).join(', '))})</span> ${nrn.h_share!=null?`<span class="pill ${nrn.h_share>0.2?'warn':''}" title="Share of its top neurons' weight on H-Neurons">${pct(nrn.h_share)} on H-Neurons</span>`:''}</h3><table>${nrows('writes','writes it')}${nrows('reads','reads it')}${nrows('corr','fires with it')}</table>`:`<p class="note">Neurons: <a href="${jobLink('feat_neurons',{sae:DICT.path})}">map this dictionary onto neurons</a>.</p>`}
  <h3>Steer with it</h3><div class="slider"><span class="note">value</span><input type="range" id="fV" min="-10" max="10" step="0.5" value="2"><span id="fVv" class="mono">2×</span>
  <select id="fMode" title="clamp: rebuild the reconstruction with the feature set to this value at every token (error term kept); add: add value × strongest activation × its direction"><option value="clamp">clamp</option><option value="add">add</option></select></div>
  <textarea id="fP" placeholder="Prompt">Tell me about your day.</textarea><div class="row"><button class="pri" id="fGo">Compare with and without</button><button id="fToSteer" title="Add it to the Steer tab with other features and emotions">To Steer tab</button><span id="fMsg"></span></div><div class="cmp" id="fOut"></div>`;
  c.querySelectorAll('.chip.link').forEach(e=>e.onclick=()=>openFeature(+e.dataset.f));
  $('#fV').oninput=()=>$('#fVv').textContent=$('#fV').value+'×';
  $('#fLabSave').onclick=async()=>{ try{ await api('/api/features/label',{path:DICT.path,id:i,label:$('#fLab').value}); DICT.label[i]=$('#fLab').value; filterList(); $('#fLabSave').textContent='Saved'; setTimeout(()=>$('#fLabSave').textContent='Save name',1500); }catch(e){ alert(e.message); } };
  $('#fGo').onclick=()=>compare({feature:{sae:DICT.path,set:{[i]:+$('#fV').value},mode:$('#fMode').value}},$('#fP').value,120,$('#fOut'),$('#fMsg'));
  $('#fToSteer').onclick=()=>{ addFeatRow({sae:DICT.path,id:i,v:+$('#fV').value,mode:$('#fMode').value}); tab('steer'); }; }
async function compare(steer,prompt,max,out,msg){ busy(msg,'generating twice in the lab…'); out.innerHTML='';
  try{ const r=await labCall('compare',{prompt,max_tokens:max,steer}); done(msg,'');
    out.innerHTML=`<div><h3>As is</h3><pre>${esc(r.plain)}</pre></div><div><h3>Steered</h3><pre>${esc(r.steered)}</pre></div>`; }
  catch(e){ fail(msg,e); } }
// ---------------------------------------------------------------- read a reply
let READ=null, RSEL=null;
$$('input[name=rmode]').forEach(r=>r.onchange=()=>{ const c=$('input[name=rmode]:checked').value==='chat'; $('#rtext').style.display=c?'none':''; $('#rchat').style.display=c?'':'none'; });
function drawReadPickers(){ $('#rD').innerHTML=LIST.dicts.filter(d=>!d.target_site).map(d=>`<label class="chip"><input type="checkbox" value="${esc(d.path)}" ${(ls('ns-feat-read')||'').includes(d.path)?'checked':''}> ${esc(d.name)} L${d.layer}</label>`).join('')||'<span class="note">none yet</span>';
  const eo=LIST.emotions.map(e=>`<option value="${esc(e.path)}">${esc(e.name)} (${e.n})</option>`).join(''); $('#rE').innerHTML='<option value="">none</option>'+eo; }
$('#rGo').onclick=async()=>{ const m=$('#rMsg'); const sae=$$('#rD input:checked').map(x=>x.value); ls('ns-feat-read',sae.join('|'));
  const chat=$('input[name=rmode]:checked').value==='chat'; const body={sae,top:5,lens:$('#rL').checked};
  if(chat){ body.messages=[{role:'user',content:$('#rP').value}]; body.reply=$('#rR').value; } else body.text=$('#rT').value;
  if($('#rE').value) body.emotions=$('#rE').value;
  if(!sae.length&&!body.emotions&&!body.lens) return fail(m,{message:'choose a dictionary, emotions or the logit lens'});
  busy(m,'reading in the lab…');
  try{ READ=await labCall('inspect',body); done(m,`${READ.tokens.length} tokens`); drawRead(); }catch(e){ fail(m,e); } };
function drawRead(){ $('#rOut').style.display=''; const R=READ;
  const opts=['<option value="">nothing</option>']; if(R.emotions) R.emotions.names.forEach((n,k)=>opts.push(`<option value="e:${k}">emotion: ${esc(n)}</option>`));
  R.dicts.forEach((d,k)=>d.most_active.slice(0,15).forEach(([f,c,l])=>opts.push(`<option value="f:${k}:${f}">${esc(d.name)} #${f}${l?' '+esc(l):''}</option>`)));
  if(R.lens) opts.push('<option value="lens">logit lens: final spread</option>');
  $('#rColor').innerHTML=opts.join(''); $('#rColor').onchange=paintRead; paintRead();
  $('#rMost').innerHTML=R.dicts.map((d,k)=>`<h2>${esc(d.name)} · most active here</h2><div class="chips">${d.most_active.map(([f,c,l])=>`<span class="chip link" data-d="${esc(d.path)}" data-f="${f}" title="active on ${c} tokens">#${f}${l?' '+esc(l):''} ·${c}</span>`).join('')}</div>`).join('')||'';
  $('#rMost').querySelectorAll('.chip.link').forEach(e=>e.onclick=()=>{ $('#dsel').value=e.dataset.d; tab('dicts'); openDict(e.dataset.d,+e.dataset.f); }); }
function paintRead(){ const R=READ, v=$('#rColor').value; const st=R.start||0;
  const val=t=>{ if(!v) return 0; if(v.startsWith('e:')){ const z=R.emotions.z[t]?.[+v.slice(2)]; return z==null?0:Math.max(-1,Math.min(1,z/3)); }
    if(v.startsWith('f:')){ const [,k,f]=v.split(':'); const row=R.dicts[+k].per_token[t]||[]; const h=row.find(r=>r[0]===+f); return h?Math.min(1,h[2]):0; }
    if(v==='lens'){ const i=t-st; return i>=0&&R.lens.final_spread[i]!=null?R.lens.final_spread[i]:0; } return 0; };
  $('#rStrip').innerHTML=R.tokens.map((t,k)=>{ const a=val(k); const col=a>=0?'var(--pos)':'var(--neg)'; return `<span class="tk ${k===RSEL?'sel':''}" data-k="${k}" style="background:color-mix(in srgb,${col} ${Math.round(Math.abs(a)*80)}%,transparent);${k<st?'opacity:.55':''}">${esc(t).replace(/\n/g,'⏎\n')}</span>`; }).join('');
  $('#rStrip').onclick=e=>{ const s=e.target.closest('.tk'); if(s){ RSEL=+s.dataset.k; paintRead(); tokDetail(RSEL); } }; }
function tokDetail(k){ const R=READ; const st=R.start||0; let h=`<h2>Token ${k}: <span class="mono">${esc(JSON.stringify(R.tokens[k]))}</span></h2>`;
  R.dicts.forEach(d=>{ const row=d.per_token[k]||[]; h+=`<h3>${esc(d.name)} L${d.layer}</h3>`+(row.length?`<table>${row.map(([f,a,rel,l])=>`<tr><td class="mono"><a href="#" data-d="${esc(d.path)}" data-f="${f}">#${f}</a></td><td>${esc(l)||'<span class="note">no label</span>'}</td><td><span class="bar" style="width:${Math.min(100,rel*100)}px"></span> ${fmt(rel,2)}</td></tr>`).join('')}</table>`:'<p class="note">nothing active (first position is left out)</p>'); });
  if(R.emotions){ const tp=R.emotions.top[k]||[]; h+=`<h3>Emotions (z)</h3><div class="chips">${tp.map(([n,z])=>`<span class="chip ${z>0?'up':'dn'}">${esc(n)} ${z>0?'+':''}${fmt(z,1)}</span>`).join('')}</div>`; }
  if(R.lens&&k>=st&&k-st<R.lens.top1.length){ const i=k-st; h+=`<h3>What each layer would have said here <span class="i" title="Logit lens: each layer's residual stream through the final norm and output layer. Settles at layer ${R.lens.settle[i]} of ${R.lens.layers}">i</span></h3><div class="chips">${R.lens.top1[i].map((t,l)=>`<span class="chip" title="layer ${l+1}, spread ${fmt(R.lens.spread[i][l],2)}">${esc(t)}</span>`).join('')}</div><p class="note">final spread ${fmt(R.lens.final_spread[i],2)} · settles at layer ${R.lens.settle[i]} · p(chosen) ${fmt(R.lens.p_chosen[i],3)}</p>`; }
  $('#rTok').innerHTML=h; $('#rTok').querySelectorAll('a[data-f]').forEach(a=>a.onclick=e=>{ e.preventDefault(); $('#dsel').value=a.dataset.d; tab('dicts'); openDict(a.dataset.d,+a.dataset.f); }); }
// ---------------------------------------------------------------- circuits
let GRAPH=null, GSEL=null;
function drawCircuitPickers(){ const tcs=LIST.dicts.filter(d=>d.target_site==='mlp_out'); $('#gTnone').style.display=tcs.length?'none':'';
  $('#gT').innerHTML=tcs.map(d=>`<label class="chip"><input type="checkbox" value="${esc(d.path)}" checked> L${d.layer} ${esc(d.name)}</label>`).join('');
  $('#gSaved').innerHTML='<option value="">…</option>'+LIST.graphs.map(g=>`<option value="${esc(g.path)}">${esc(g.name)}${g.prompt?' — '+esc(g.prompt.slice(0,60)):''}</option>`).join('');
  dictOpts($('#aD'),d=>!d.target_site); }
$('#gGo').onclick=async()=>{ const m=$('#gMsg'); const t=$$('#gT input:checked').map(x=>x.value); if(!t.length) return fail(m,{message:'choose transcoders'});
  const p=$('#gP').value.trim(); if(!p) return fail(m,{message:'write a prompt'});
  const slug=p.toLowerCase().replace(/[^a-z0-9]+/g,'-').slice(0,40)||'graph';
  busy(m,'building the graph in the lab (a backward pass per feature node; minutes on a CPU)…');
  try{ GRAPH=await labCall('graph',{transcoders:t,prompt:p,chat:$('#gChat').checked,max_features:+$('#gMax').value,node_threshold:+$('#gNode').value,edge_threshold:+$('#gEdge').value,save:LIST.root+'/graphs/'+slug+'.json'});
    done(m,`saved as runs/graphs/${slug}.json`); drawGraph(); refresh(); }catch(e){ fail(m,e); } };
$('#gP').oninput=()=>{ $('#gJob').href=jobLink('feat_graph',{prompt:$('#gP').value,transcoders:$$('#gT input:checked').map(x=>x.value).join(' '),out:'runs/graphs/graph.json'}); };
$('#gSaved').onchange=async()=>{ const p=$('#gSaved').value; if(!p) return; try{ GRAPH=await api('/api/features/graph?path='+encodeURIComponent(p)); drawGraph(); }catch(e){ fail($('#gMsg'),e); } };
function drawGraph(){ const G=GRAPH; $('#gOut').style.display=''; GSEL=null;
  $('#gScore').innerHTML=`replacement ${fmt(G.replacement_score,2)} · completeness ${fmt(G.completeness_score,2)} <span class="i" title="Replacement: share of the path from prompt to output that goes through features rather than error nodes. Completeness: how much of each node's input features explain. Low values mean the transcoders miss much of the computation">i</span> · ${G.nodes.length} nodes, ${G.edges.length} edges`;
  const T=G.tokens.length, L=G.n_layers, cw=Math.max(54,Math.min(110,900/T)), rh=26, W=60+cw*T, H=40+rh*(L+2);
  const X=n=>60+cw*n.pos+cw/2, Y=n=>H-20-rh*(n.layer+1);
  const by={}; G.nodes.forEach(n=>{ const k=n.layer+':'+n.pos; (by[k]=by[k]||[]).push(n); });
  const P={}; Object.values(by).forEach(arr=>arr.forEach((n,j)=>{ const off=(j-(arr.length-1)/2)*Math.min(12,(cw-10)/Math.max(1,arr.length)); P[n.i]=[X(n)+off,Y(n)]; }));
  const maxw=Math.max(1e-9,...G.edges.map(e=>Math.abs(e.w))); const top=new Set([...G.edges].sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).slice(0,250));
  const col={feature:'var(--acc)',error:'var(--mut)',token:'var(--fg)',logit:'var(--ok)'};
  let s=`<g class="axis">${G.tokens.map((t,k)=>`<text x="${60+cw*k+cw/2}" y="${H-4}" font-size="10" text-anchor="middle" fill="var(--mut)">${esc(t.length>9?t.slice(0,8)+'…':t)}</text>`).join('')}${[...Array(L).keys()].filter(l=>l%Math.max(1,Math.ceil(L/12))===0).map(l=>`<text x="4" y="${H-20-rh*(l+1)+3}" font-size="10" fill="var(--mut)">L${l}</text>`).join('')}</g>`;
  s+=`<g id="gE">${G.edges.map((e,k)=>{ if(!P[e.s]||!P[e.t]) return ''; const a=P[e.s],b=P[e.t]; return `<line data-k="${k}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${e.w>0?'var(--pos)':'var(--neg)'}" stroke-width="${0.5+2.5*Math.abs(e.w)/maxw}" opacity="${top.has(e)?0.18+0.6*Math.abs(e.w)/maxw:0}"/>`; }).join('')}</g>`;
  const mi=Math.max(1e-9,...G.nodes.map(n=>n.influence||0)); const topOut=new Set(G.nodes.filter(n=>n.kind==='logit').sort((a,b)=>b.prob-a.prob).slice(0,3).map(n=>n.i));
  s+=G.nodes.map(n=>{ const [x,y]=P[n.i]; const r=n.kind==='feature'?3+6*Math.sqrt((n.influence||0)/mi):n.kind==='error'?4:5; const lab=n.kind==='logit'?n.token:n.kind==='feature'?(n.label||'#'+n.feature):'';
    return `<g class="gn" data-i="${n.i}" style="cursor:pointer">${n.kind==='error'?`<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" fill="${col.error}" opacity=".7"/>`:`<circle cx="${x}" cy="${y}" r="${r}" fill="${col[n.kind]}"/>`}<title>${esc(n.kind==='feature'?`L${n.layer} #${n.feature} ${n.label||''} act ${n.act}`:n.kind==='error'?`error L${n.layer} pos ${n.pos}`:n.kind==='logit'?`output ${n.token} p=${n.prob}`:`token ${n.token}`)}</title>${lab&&n.kind==='logit'&&topOut.has(n.i)?`<text x="${x}" y="${y-8}" font-size="10" text-anchor="middle" fill="var(--fg)">${esc(lab)}</text>`:''}</g>`; }).join('');
  const svg=$('#gSvg'); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width',W); svg.setAttribute('height',H); svg.innerHTML=s;
  svg.querySelectorAll('.gn').forEach(g=>g.onclick=()=>selectNode(+g.dataset.i)); }
function selectNode(i){ GSEL=i; const G=GRAPH, n=G.nodes.find(x=>x.i===i); const name=x=>{ const m=G.nodes.find(y=>y.i===x); return !m?'?':m.kind==='feature'?`L${m.layer} #${m.feature}${m.label?' '+m.label:''} @${m.pos}`:m.kind==='error'?`error L${m.layer} @${m.pos}`:m.kind==='token'?`token ${JSON.stringify(m.token)}`:`output ${JSON.stringify(m.token)}`; };
  const maxw=Math.max(1e-9,...G.edges.map(e=>Math.abs(e.w)));
  $$('#gE line').forEach(l=>{ const e=G.edges[+l.dataset.k]; const on=e.s===i||e.t===i; l.setAttribute('opacity',on?0.25+0.7*Math.abs(e.w)/maxw:0.03); });
  const inc=G.edges.filter(e=>e.t===i).sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).slice(0,10), out=G.edges.filter(e=>e.s===i).sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).slice(0,10);
  const rows=(arr,k)=>arr.map(e=>`<tr><td><a href="#" data-n="${e[k]}">${esc(name(e[k]))}</a></td><td class="mono" style="color:${e.w>0?'var(--pos)':'var(--neg)'}">${e.w>0?'+':''}${fmt(e.w,3)}</td></tr>`).join('');
  let h=`<h2>${esc(name(i))}</h2>`;
  if(n.kind==='feature'){ const tc=Object.entries(G.transcoders||{}).find(([l])=>+l===n.layer); h+=`<p class="note">activation ${fmt(n.act,3)} · influence on the output ${fmt(n.influence,4)}${tc?` · <a href="#" id="gOpen" data-d="${esc(tc[1])}">open the feature</a>`:''}</p>
    <div class="row"><span class="note">Intervene at this position:</span><button data-m="0">turn off</button><button data-m="x2">× 2</button><button data-m="x-1">× −1</button><button data-m="x5">× 5</button></div><div id="gIv"></div>`; }
  h+=`<h3>Fed by</h3><table>${rows(inc,'s')||'<tr><td class="note">nothing kept</td></tr>'}</table><h3>Feeds</h3><table>${rows(out,'t')||'<tr><td class="note">nothing kept</td></tr>'}</table>`;
  $('#gNodeCard').innerHTML=h;
  $('#gNodeCard').querySelectorAll('a[data-n]').forEach(a=>a.onclick=e=>{ e.preventDefault(); selectNode(+a.dataset.n); });
  const op=$('#gOpen'); if(op) op.onclick=e=>{ e.preventDefault(); $('#dsel').value=op.dataset.d; tab('dicts'); openDict(op.dataset.d,n.feature); };
  $('#gNodeCard').querySelectorAll('button[data-m]').forEach(b=>b.onclick=()=>intervene(n,b.dataset.m)); }
async function intervene(n,m){ const box=$('#gIv'); box.innerHTML='<p class="busy">running the prompt with and without the change…</p>';
  const watch=GRAPH.edges.filter(e=>e.s===n.i).map(e=>GRAPH.nodes.find(x=>x.i===e.t)).filter(x=>x&&x.kind==='feature').slice(0,12).map(x=>[x.layer,x.pos,x.feature]);
  const set=`${n.layer}:${n.feature}@${n.pos}=${m}`;
  try{ const r=await labCall('intervene',{transcoders:Object.values(GRAPH.transcoders||{}),prompt:GRAPH.prompt,chat:!!GRAPH.chat,set:[set],watch});
    const toks=arr=>arr.slice(0,6).map(t=>`<span class="chip">${esc(t.token)} ${fmt(t.p,3)}</span>`).join('');
    box.innerHTML=`<p class="note">${esc(set)} · KL ${fmt(r.kl,4)}</p><div class="row"><span class="note">before</span><div class="chips">${toks(r.before)}</div></div><div class="row"><span class="note">after</span><div class="chips">${toks(r.after)}</div></div>`+
      (r.watched&&r.watched.length?`<table><tr><th>downstream feature</th><th>before</th><th>after</th></tr>${r.watched.map(w=>`<tr><td class="mono">L${w.layer} #${w.feature} @${w.pos}</td><td>${fmt(w.before,3)}</td><td>${fmt(w.after,3)}</td></tr>`).join('')}</table>`:''); }
  catch(e){ box.innerHTML=`<p class="err">${esc(e.message)}</p>`; } }
$('#aGo').onclick=async()=>{ const m=$('#aMsg'); busy(m,'attributing in the lab…');
  try{ const r=await labCall('attribute',{sae:$('#aD').value,prompt:$('#aP').value,token:$('#aTok').value,baseline:$('#aBase').value||null,ablate:+$('#aAbl').value,chat:$('#aChat').checked});
    done(m,r.attribution_ablation_corr!=null?`attribution vs ablation: r = ${fmt(r.attribution_ablation_corr,2)}`:'');
    const mx=Math.max(1e-9,...r.features.map(f=>Math.abs(f.attribution)));
    $('#aOut').innerHTML=`<table><tr><th>where</th><th>feature</th><th>attribution</th><th>ablation</th></tr>${r.features.slice(0,25).map(f=>`<tr><td class="mono">${f.pos} ${esc(JSON.stringify(f.token))}</td><td><a href="#" data-f="${f.feature}">#${f.feature}</a> ${esc(f.label||'')}</td><td><span class="bar" style="width:${80*Math.abs(f.attribution)/mx}px;background:${f.attribution>0?'var(--pos)':'var(--neg)'}"></span> ${fmt(f.attribution,3)}</td><td class="mono">${f.ablation!=null?fmt(f.ablation,3):''}</td></tr>`).join('')}</table>`;
    $('#aOut').querySelectorAll('a[data-f]').forEach(a=>a.onclick=e=>{ e.preventDefault(); const d=$('#aD').value; $('#dsel').value=d; tab('dicts'); openDict(d,+a.dataset.f); }); }
  catch(e){ fail(m,e); } };
// ---------------------------------------------------------------- emotions
let EMO=null;
async function openEmo(path){ if(!path) return; ls('ns-feat-emo',path);
  try{ EMO=await api('/api/features/emotions?path='+encodeURIComponent(path)); }catch(e){ $('#eStats').innerHTML=`<span class="err">${esc(e.message)}</span>`; return; }
  $('#eOut').style.display=''; const acc=EMO.accuracy||{}; const best=EMO.best_layer;
  $('#eStats').textContent=`${EMO.model} · ${EMO.names.length} emotions · default layer ${EMO.default_layer} · best held-out accuracy ${pct(acc[best])} at layer ${best} (chance ${pct(EMO.chance)})`;
  $('#eTE').innerHTML=EMO.names.map(n=>`<option>${esc(n)}</option>`).join(''); drawAcc(); drawEmap(); drawSim(); }
$('#eSel').onchange=()=>openEmo($('#eSel').value);
function drawAcc(){ const acc=EMO.accuracy||{}, raw=EMO.accuracy_raw||{}, L=EMO.layers, W=560,H=170; if(!L.length) return; const x=l=>30+(W-40)*(L.indexOf(l)/Math.max(1,L.length-1)), y=v=>H-20-(H-35)*v;
  const pts=L.filter(l=>acc[l]!=null).map(l=>`${x(l)},${y(acc[l])}`).join(' ');
  $('#eAcc').innerHTML=`<line x1="30" x2="${W-10}" y1="${y(EMO.chance)}" y2="${y(EMO.chance)}" stroke="var(--mut)" stroke-dasharray="3 3"/><text x="${W-10}" y="${y(EMO.chance)-3}" font-size="10" text-anchor="end" fill="var(--mut)">chance</text>
   <polyline points="${pts}" fill="none" stroke="var(--acc)" stroke-width="2"/>${raw&&Object.keys(raw).length?`<polyline points="${L.filter(l=>raw[l]!=null).map(l=>`${x(l)},${y(raw[l])}`).join(' ')}" fill="none" stroke="var(--mut)" stroke-width="1.5" stroke-dasharray="4 3"><title>without denoising</title></polyline><text x="${W-10}" y="14" font-size="10" text-anchor="end" fill="var(--mut)">dashed: without denoising</text>`:''}${L.filter(l=>acc[l]!=null).map(l=>`<circle cx="${x(l)}" cy="${y(acc[l])}" r="${l===EMO.default_layer?4.5:2.5}" fill="${l===EMO.default_layer?'var(--pos)':'var(--acc)'}"><title>layer ${l}: ${pct(acc[l])}</title></circle>`).join('')}
   ${[0,0.5,1].map(v=>`<text x="2" y="${y(v)+3}" font-size="10" fill="var(--mut)">${v*100}%</text>`).join('')}<text x="${W/2}" y="${H-3}" font-size="10" text-anchor="middle" fill="var(--mut)">layer (orange: default, two thirds of the depth)</text>`; }
function drawEmap(){ const P=EMO.pca, xy=P.xy, W=560,H=420; const xs=xy.map(p=>p[0]), ys=xy.map(p=>p[1]); const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const va=EMO.valence_arousal||{}; const X=v=>30+(W-130)*(v-x0)/(x1-x0||1), Y=v=>H-30-(H-60)*(v-y0)/(y1-y0||1);
  $('#eMap').innerHTML=`<text x="${W/2}" y="${H-6}" font-size="10" text-anchor="middle" fill="var(--mut)">PC1 (${pct(P.explained[0])} of variance)${EMO.pc_corr?` · r with valence ${fmt(EMO.pc_corr.pc1_valence,2)}`:''}</text><text x="10" y="14" font-size="10" fill="var(--mut)">PC2 (${pct(P.explained[1])})${EMO.pc_corr?` · r with arousal ${fmt(EMO.pc_corr.pc2_arousal,2)}`:''}</text>`+
    EMO.names.map((n,k)=>{ const v=va[n]?va[n][0]:null; const c=v==null?'var(--acc)':v>0?'var(--pos)':'var(--neg)'; return `<g style="cursor:pointer" data-k="${k}"><circle cx="${X(xy[k][0])}" cy="${Y(xy[k][1])}" r="3" fill="${c}"/><text x="${X(xy[k][0])+4}" y="${Y(xy[k][1])+3}" font-size="${EMO.names.length>60?8:10}" fill="var(--fg)">${esc(n)}</text></g>`; }).join('');
  $('#eMap').querySelectorAll('g[data-k]').forEach(g=>g.onclick=()=>emoDetail(+g.dataset.k)); }
function drawSim(){ const c=$('#eSim'), g=c.getContext('2d'), E=EMO.names.length; const ord=[...Array(E).keys()].sort((a,b)=>EMO.pca.xy[b][0]-EMO.pca.xy[a][0]); const s=c.width/E;
  g.clearRect(0,0,c.width,c.height);
  for(let a=0;a<E;a++) for(let b=0;b<E;b++){ const v=EMO.cosine[ord[a]][ord[b]]; g.fillStyle=v>=0?`rgba(217,72,15,${Math.min(1,v)})`:`rgba(28,126,214,${Math.min(1,-v)})`; g.fillRect(b*s,a*s,Math.ceil(s),Math.ceil(s)); }
  if(E<=24){ g.font='11px ui-sans-serif,system-ui'; g.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--fg').trim()||'#eee'; ord.forEach((k,a)=>g.fillText(EMO.names[k],4,a*s+s/2+4)); }
  c.onmousemove=e=>{ const r=c.getBoundingClientRect(), k=c.width/r.width; const a=Math.floor((e.clientY-r.top)*k/s), b=Math.floor((e.clientX-r.left)*k/s); if(a<E&&b<E) $('#eHover').textContent=`${EMO.names[ord[a]]} · ${EMO.names[ord[b]]}: ${fmt(EMO.cosine[ord[a]][ord[b]],2)}`; };
  c.onclick=e=>{ const r=c.getBoundingClientRect(), k=c.width/r.width; const a=Math.floor((e.clientY-r.top)*k/s); if(a<E) emoDetail(ord[a]); }; }
function emoDetail(k){ const n=EMO.names[k], lg=(EMO.logits||{})[n]; const sim=EMO.cosine[k].map((v,j)=>[v,j]).filter(([,j])=>j!==k).sort((a,b)=>b[0]-a[0]); const nc=Math.min(5,Math.ceil(sim.length/2)), nf=Math.min(3,sim.length-nc);
  $('#eDetail').innerHTML=`<h3>${esc(n)} <span class="note">${(EMO.counts||{})[n]?`${EMO.counts[n][0]} training / ${EMO.counts[n][1]} held-out stories`:''}</span></h3>
   <p class="note" style="margin:.1rem 0">closest: ${sim.slice(0,nc).map(([v,j])=>esc(EMO.names[j])+' '+fmt(v,2)).join(', ')}${nf>0?` · furthest: ${sim.slice(-nf).reverse().map(([v,j])=>esc(EMO.names[j])+' '+fmt(v,2)).join(', ')}`:''}</p>
   ${lg?`<div class="row"><span class="note">raises</span><div class="chips">${lg.up.map(t=>`<span class="chip up">${esc(t)}</span>`).join('')}</div></div><div class="row"><span class="note">lowers</span><div class="chips">${lg.down.map(t=>`<span class="chip dn">${esc(t)}</span>`).join('')}</div></div>`:''}
   <div class="row" style="margin-top:.3rem"><button id="eToSteer">Steer with ${esc(n)}</button><a href="${jobLink('emo_scan',{emotions:EMO.path,corpus:'NeelNanda/pile-10k',out:`runs/emotions/scan-${EMO.name}.json`})}">Find the passages it fires on most</a></div>`;
  $('#eToSteer').onclick=()=>{ addEmoRow({file:EMO.path,name:n,s:0.05}); tab('steer'); }; }
$('#eTGo').onclick=async()=>{ const m=$('#eTMsg'); const text=$('#eT').value||$('#rT').value; if(!text) return fail(m,{message:'write a text'}); busy(m,'tracing in the lab…');
  try{ const r=await labCall('inspect',{text,emotions:EMO.path,sae:[]}); done(m,`layer ${r.emotions.layer}`); const E=r.emotions; const sel=()=>E.names.indexOf($('#eTE').value);
    const paint=()=>{ const k=sel(); $('#eTStrip').innerHTML=E.tokens.map((t,i)=>{ const z=E.z[i][k]; const a=Math.max(-1,Math.min(1,z/3)); return `<span class="tk" title="${esc(E.top[i].map(x=>x[0]+' '+x[1]).join(', '))} · ${esc($('#eTE').value)} z ${fmt(z,2)}" style="background:color-mix(in srgb,${a>=0?'var(--pos)':'var(--neg)'} ${Math.round(Math.abs(a)*80)}%,transparent)">${esc(t)}</span>`; }).join('');
      const W=560,H=120, n=E.tokens.length; const zs=E.z.map(r=>r[k]); const mx=Math.max(3,...zs.map(Math.abs)); const x=i=>10+(W-20)*i/Math.max(1,n-1), y=v=>H/2-(H/2-8)*v/mx;
      $('#eTLine').innerHTML=`<svg viewBox="0 0 ${W} ${H}"><line x1="10" x2="${W-10}" y1="${H/2}" y2="${H/2}" stroke="var(--line)"/><polyline fill="none" stroke="var(--pos)" stroke-width="1.6" points="${zs.map((v,i)=>x(i)+','+y(v)).join(' ')}"/></svg><p class="note">${esc($('#eTE').value)} along the text (z-score against the stories; red above, blue below).</p>`; };
    $('#eTE').onchange=paint; paint(); }
  catch(e){ fail(m,e); } };
$('#eSwGo').onclick=async()=>{ const m=$('#eSwMsg'); busy(m,'sweeping in the lab…'); const names=$('#eSwE').value.split(/[\s,]+/).filter(Boolean);
  try{ const r=await labCall('sweep',{emotions:EMO.path,template:$('#eSwT').value,values:$('#eSwV').value.split(/[\s,]+/).filter(Boolean),names}); done(m,`layer ${r.layer}`);
    const W=560,H=200, rows=r.rows, n=rows.length; const all=rows.flatMap(x=>Object.values(x.z)); const lo=Math.min(-1,...all), hi=Math.max(1,...all);
    const x=i=>44+(W-130)*i/Math.max(1,n-1), y=v=>H-25-(H-40)*(v-lo)/(hi-lo||1); const cols=['var(--pos)','var(--neg)','var(--ok)','var(--warn)','var(--acc)','var(--fg)'];
    $('#eSwSvg').innerHTML=`<line x1="44" x2="${x(n-1)}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line)"/>`+[lo,0,hi].map(v=>`<text x="38" y="${y(v)+3}" font-size="10" text-anchor="end" fill="var(--mut)">${v>0?'+':''}${fmt(v,1)}</text>`).join('')+`<text x="4" y="12" font-size="10" fill="var(--mut)">z</text>`+
      names.map((nm,k)=>`<polyline fill="none" stroke="${cols[k%6]}" stroke-width="2" points="${rows.map((rw,i)=>x(i)+','+y(rw.z[nm]??0)).join(' ')}"/><text x="${x(n-1)+6}" y="${y(rows[n-1].z[nm]??0)+3}" font-size="10" fill="${cols[k%6]}">${esc(nm)}</text>`).join('')+rows.map((rw,i)=>`<text x="${x(i)}" y="${H-8}" font-size="10" text-anchor="middle" fill="var(--mut)">${esc(rw.value)}</text>`).join(''); }
  catch(e){ fail(m,e); } };
// ---------------------------------------------------------------- steer
function addFeatRow(v){ v=v||{}; const d=document.createElement('div'); d.className='row'; d.style.margin='.2rem 0';
  d.innerHTML=`<select class="sd grow" style="max-width:22rem" aria-label="Dictionary"></select><input class="sid" type="number" min="0" placeholder="feature id" style="width:6.5rem" value="${v.id??''}" aria-label="Feature id">
   <div class="slider grow"><input class="sv" type="range" min="-10" max="10" step="0.5" value="${v.v??2}" aria-label="Value"><span class="mono svv">${v.v??2}×</span></div>
   <select class="sm" aria-label="Mode" title="clamp: set the feature in the reconstruction, keep the error; add: add along its direction"><option value="clamp">clamp</option><option value="add">add</option></select><button class="sx" title="Remove">×</button>`;
  $('#sF').appendChild(d); dictOpts(d.querySelector('.sd'),x=>!x.target_site,v.sae); if(v.mode) d.querySelector('.sm').value=v.mode;
  d.querySelector('.sv').oninput=e=>{ d.querySelector('.svv').textContent=e.target.value+'×'; drawVariant(); }; d.querySelector('.sx').onclick=()=>{ d.remove(); drawVariant(); }; d.oninput=drawVariant; drawVariant(); }
function addEmoRow(v){ v=v||{}; const d=document.createElement('div'); d.className='row'; d.style.margin='.2rem 0';
  d.innerHTML=`<select class="ef grow" style="max-width:16rem" aria-label="Emotion vectors">${LIST.emotions.map(e=>`<option value="${esc(e.path)}">${esc(e.name)}</option>`).join('')}</select>
   <input class="en" list="emoNames" placeholder="emotion" style="width:9rem" value="${esc(v.name||'')}" aria-label="Emotion">
   <div class="slider grow"><input class="es" type="range" min="-0.15" max="0.15" step="0.005" value="${v.s??0.05}" aria-label="Strength"><span class="mono esv">${v.s??0.05}</span></div>
   <input class="el" placeholder="layers" style="width:6rem" title="Layers to add it after (default: the vectors' default layer)" value="${esc(v.layers||'')}"><button class="ex2" title="Remove">×</button>`;
  $('#sE').appendChild(d); if(v.file) d.querySelector('.ef').value=v.file;
  d.querySelector('.es').oninput=e=>{ d.querySelector('.esv').textContent=(+e.target.value).toFixed(3); drawVariant(); }; d.querySelector('.ex2').onclick=()=>{ d.remove(); drawVariant(); }; d.oninput=drawVariant; drawVariant(); }
$('#sFAdd').onclick=()=>addFeatRow(); $('#sEAdd').onclick=()=>addEmoRow();
function steerSpec(){ const st={}; const fr=$$('#sF .row').filter(r=>r.querySelector('.sid').value!=='');
  if(fr.length){ const sae=fr[0].querySelector('.sd').value; st.feature={sae,set:{},mode:fr[0].querySelector('.sm').value}; fr.filter(r=>r.querySelector('.sd').value===sae).forEach(r=>st.feature.set[r.querySelector('.sid').value]=+r.querySelector('.sv').value); }
  const er=$$('#sE .row').filter(r=>r.querySelector('.en').value.trim());
  if(er.length){ const file=er[0].querySelector('.ef').value; st.emotion={file,mix:{}}; const ly=er[0].querySelector('.el').value.trim(); if(ly) st.emotion.layers=ly.split(/[\s,]+/).map(Number);
    er.filter(r=>r.querySelector('.ef').value===file).forEach(r=>st.emotion.mix[r.querySelector('.en').value.trim()]=+r.querySelector('.es').value); }
  return st; }
function drawVariant(){ const st=steerSpec(); const v=`steered=${JSON.stringify(st)}`; $('#sVar').textContent=`--variant 'plain={}' --variant '${v}'`;
  $('#sLab').href=jobLink('decode_lab',{model:LAB?LAB.model:'',variant:`plain={} ${v}`});
  if(st.emotion) $('#sCvec').href=jobLink('emo_cvec',{emotions:st.emotion.file,mix:Object.entries(st.emotion.mix).map(([k,s])=>k+'='+s).join(' '),out:'runs/emotions/mix.gguf'}); }
$('#sCopy').onclick=()=>navigator.clipboard?.writeText($('#sVar').textContent);
$('#sGo').onclick=()=>{ const st=steerSpec(); if(!st.feature&&!st.emotion) return fail($('#sMsg'),{message:'add a feature or an emotion'}); compare(st,$('#sP').value,+$('#sMax').value,$('#sOut'),$('#sMsg')); };
function steerGet(){ return {rowsF:$$('#sF .row').map(r=>({sae:r.querySelector('.sd').value,id:r.querySelector('.sid').value,v:+r.querySelector('.sv').value,mode:r.querySelector('.sm').value})),
  rowsE:$$('#sE .row').map(r=>({file:r.querySelector('.ef').value,name:r.querySelector('.en').value,s:+r.querySelector('.es').value,layers:r.querySelector('.el').value})),prompt:$('#sP').value,max:+$('#sMax').value}; }
function steerSet(v){ $('#sF').innerHTML=''; $('#sE').innerHTML=''; (v.rowsF||[]).forEach(addFeatRow); (v.rowsE||[]).forEach(addEmoRow); if(v.prompt!=null) $('#sP').value=v.prompt; if(v.max) $('#sMax').value=v.max; drawVariant(); }
// ---------------------------------------------------------------- data
async function refresh(){ try{ LIST=await api('/api/features/list'); }catch(e){ LIST={dicts:[],acts:[],emotions:[],stories:[],graphs:[]}; }
  drawStart(); dictOpts($('#dsel'),null,ls('ns-feat-dict')||undefined); drawReadPickers(); drawCircuitPickers();
  const es=$('#eSel'), ev=es.value||ls('ns-feat-emo'); es.innerHTML=LIST.emotions.map(e=>`<option value="${esc(e.path)}">${esc(e.name)} · ${esc(e.model||'')} · ${e.n} emotions</option>`).join('')||'<option value="">none yet</option>'; if(ev) es.value=ev;
  let dl=$('#emoNames'); if(!dl){ dl=document.createElement('datalist'); dl.id='emoNames'; document.body.appendChild(dl); }
  return LIST; }
(async()=>{ await refresh(); checkLab();
  const h=location.hash.slice(1); const [t,qs]=h.split('?'); const q=new URLSearchParams(qs||'');
  tab(['start','dicts','read','circuits','emotions','steer'].includes(t)?t:(ls('ns-feat-tab')||(LIST.dicts.length||LIST.emotions.length?'dicts':'start')));
  const dp=q.get('path')||$('#dsel').value; if(dp) openDict(dp,q.get('id')); if($('#eSel').value) openEmo($('#eSel').value).then(()=>{ if(EMO) document.getElementById('emoNames').innerHTML=EMO.names.map(n=>`<option value="${esc(n)}">`).join(''); });
  const pr=nsProfiles($('#sProf'),{scope:'features:steer',get:steerGet,set:steerSet,system:()=>steerSet({rowsF:[],rowsE:[],prompt:'Tell me about your day.',max:120})});
  const dv=await pr.ready; if(dv) steerSet(dv); else drawVariant();
  setInterval(checkLab,30000);
  window.addEventListener('hashchange',()=>{ const [t,qs]=location.hash.slice(1).split('?'); if(!['start','dicts','read','circuits','emotions','steer'].includes(t)) return;
    tab(t); const q=new URLSearchParams(qs||''); if(t==='dicts'&&q.get('path')&&(!DICT||DICT.path!==q.get('path'))) openDict(q.get('path'),q.get('id')); else if(t==='dicts'&&q.get('id')) openFeature(+q.get('id')); }); })();
</script>
"""

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
