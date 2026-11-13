"""Studio's /work page: where the work went (scripts/work_log.py). Shares of work and tokens per device,
worker, model, kind and source; token and hardware curves over time; and stat plots of the neuron
activations the review store has recorded (scripts/neuron_review.py activation_stats)."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Work</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:13.5px/1.45 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;align-items:center;gap:.5rem;padding:.5rem 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap;position:sticky;top:0;z-index:2}
header h1{font-size:15px;margin:0 .4rem 0 0}
input,select,button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.25rem .5rem;max-width:100%}
button{cursor:pointer;background:var(--panel)}
main{max-width:1200px;margin:0 auto;padding:12px 16px 40px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin:0 0 12px}
section h2{font-size:14px;margin:0 0 .45rem;display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
section h2 .grow{flex:1}
.note{color:var(--mut);font-size:12px}.err{color:var(--no)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px}
.tile{border:1px solid var(--line);border-radius:8px;padding:.45rem .6rem;background:var(--bg)}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.tile span{color:var(--mut);font-size:12px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:.28rem .45rem;text-align:right;white-space:nowrap;font-size:12.5px}
th{color:var(--mut);font-weight:500;cursor:help}td:first-child,th:first-child{text-align:left}
td.name{max-width:280px;overflow:hidden;text-overflow:ellipsis}
.bar{display:inline-block;width:90px;height:8px;border-radius:4px;background:var(--line);vertical-align:middle;margin-right:.35rem;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.35rem;vertical-align:-1px}
canvas{width:100%;display:block}
.legend{display:flex;flex-wrap:wrap;gap:.2rem .8rem;font-size:12px;color:var(--mut);margin-top:.3rem}
.legend span{max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hw{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}
.hw>div{border:1px solid var(--line);border-radius:8px;padding:.45rem .55rem;background:var(--bg)}
.hw h3{font-size:13px;margin:0 0 .2rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px}
.plots>div{min-width:0}.plots h3{font-size:12.5px;margin:.2rem 0;color:var(--mut);font-weight:500}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.25rem .45rem;font-size:12px;display:none;z-index:9;box-shadow:0 2px 8px #0004}
@media (max-width:640px){.plots{grid-template-columns:1fr}.hw{grid-template-columns:1fr}}
</style>
<header><h1>Work distribution</h1>
<label class="note" title="Which stretch of the work log to show">window
<select id="since"><option value="1h">last hour</option><option value="24h" selected>last 24 hours</option><option value="7d">last 7 days</option><option value="30d">last 30 days</option><option value="all">everything kept</option></select></label>
<label class="note" title="Reload every 15 seconds"><input type="checkbox" id="auto" checked> live</label>
<button onclick="load()">Refresh</button><span class="note" id="status"></span></header>
<main>
<section><div class="tiles" id="tiles"></div>
<p class="note" style="margin:.5rem 0 0">Every chat reply, project task and job is logged with the worker, model, device and computer that did it. Studio also samples each GPU every 30 seconds. Nothing leaves this computer. Turn it off with <code>--no-work-log</code>.</p></section>

<section><h2>Who did the work <span class="grow"></span>
<label class="note" title="How to group the rows">by <select id="by"><option value="by_device">device</option><option value="by_worker">worker</option><option value="by_model">model</option><option value="by_kind">kind of work</option><option value="by_source">source</option></select></label></h2>
<div class="scroll"><table id="tbl"></table></div></section>

<section><h2>Tokens over time <span class="note">prompt + generated, per device</span></h2>
<canvas id="tl" height="200"></canvas><div class="legend" id="tlleg"></div></section>

<section><h2>Hardware <span class="note">load, memory, temperature and power from the samples</span></h2>
<div class="hw" id="hw"></div></section>

<section id="predict"><h2>Predicted speed <span class="note">generation tokens/s, each model on each device</span><span class="grow"></span><button onclick="predict()" title="Fit again on the latest measurements">Refit</button></h2>
<div id="pmeta" class="note"></div><div class="scroll"><table id="ptbl"></table></div>
<details style="margin-top:.4rem"><summary class="note">What the prediction has learned</summary><div id="plearn" class="note"></div></details></section>

<section id="measures"><h2>How measurements are weighted <span class="note">when ranking models</span></h2>
<div id="mmeta" class="note"></div><div class="scroll"><table id="mtbl"></table></div>
<details style="margin-top:.4rem"><summary class="note">Combined ranking</summary><div class="scroll"><table id="mrank"></table></div></details></section>

<section id="neurons"><h2>Neuron activations <span class="grow"></span>
<label class="note" title="Models with recorded activation profiles (Neuron review)">model <select id="model"></select></label>
<label class="note" title="Only observations with this verdict">verdict <select id="verdict"><option value="">all</option><option>correct</option><option>wrong</option><option>abstained</option><option>unknown</option></select></label></h2>
<div id="nmeta" class="note"></div>
<div class="plots">
<div><h3 title="Per layer: the box spans the 25th to 75th percentile, the whiskers the 5th to 95th, the line is the median and the dot the mean, over every observation and position">Distribution per layer</h3><canvas id="box" height="230"></canvas></div>
<div><h3 title="All recorded activation values (0.5th to 99.5th percentile)">All values</h3><canvas id="hist" height="230"></canvas></div>
<div><h3 title="Mean activation, layer (rows) by position across the layer's neurons (columns)">Mean, layer × neuron position</h3><canvas id="heat" height="230"></canvas></div>
<div><h3 title="Mean over wrong answers minus mean over correct ones: red is higher when wrong">Wrong minus correct</h3><canvas id="diff" height="230"></canvas></div>
</div></section>
</main><div id="tip"></div>
<script>
const $=id=>document.getElementById(id);
const PAL=['#6f93ff','#5fcf8f','#e7b65c','#ff7a6b','#b48ead','#5dcaa5','#f29e4c','#9a9a94','#d16ba5','#86a8e7'];
const css=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num=n=>n==null?'–':n>=1e9?(n/1e9).toFixed(1)+'G':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e4?(n/1e3).toFixed(1)+'k':String(Math.round(n*10)/10);
const dur=s=>s==null?'–':s<90?Math.round(s)+' s':s<5400?Math.round(s/60)+' min':(s/3600).toFixed(1)+' h';
const pct=x=>x==null?'–':(x*100).toFixed(x<0.1?1:0)+'%';
let DATA=null,NDATA=null,colors={};
function color(k){if(!(k in colors))colors[k]=PAL[Object.keys(colors).length%PAL.length];return colors[k]}
function canvas(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=c.getAttribute('height')*d;const g=c.getContext('2d');g.scale(d,d);return [g,r.width,+c.getAttribute('height')]}
function tip(e,html){const t=$('tip');if(!html){t.style.display='none';return}t.innerHTML=html;t.style.display='block';t.style.left=Math.min(innerWidth-t.offsetWidth-8,e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px'}

async function load(){
  $('status').textContent='loading…';
  try{const r=await fetch('/api/work?since='+$('since').value);DATA=await r.json();if(DATA.error)throw new Error(DATA.error)}
  catch(e){$('status').innerHTML='<span class="err">'+esc(e.message)+'</span>';return}
  $('status').textContent='updated '+new Date().toLocaleTimeString();
  tiles();table();timeline();hardware();
}
function tiles(){
  const t=DATA.totals,gen=DATA.by_device.filter(x=>x.gen_tok_s);
  const best=gen.sort((a,b)=>b.gen_tok_s-a.gen_tok_s)[0];
  $('tiles').innerHTML=[[num(t.n),'units of work'],[num(t.failed),'failed'],[num(t.prompt_tokens),'prompt tokens'],[num(t.completion_tokens),'generated tokens'],
    [dur(t.busy_s),'busy time (summed)'],[best?num(best.gen_tok_s)+' tok/s':'–',best?'fastest: '+esc(best.key):'fastest device']]
    .map(([v,l])=>`<div class="tile" title="${l}"><b>${v}</b><span>${l}</span></div>`).join('');
}
const COLS=[['n','work','Replies, tasks and jobs'],['share_work','share of work','Share of all units of work in the window'],['share_tokens','share of tokens','Share of all prompt and generated tokens'],
  ['prompt_tokens','prompt tok','Prompt tokens read'],['completion_tokens','generated tok','Tokens generated'],['gen_tok_s','gen tok/s','Generation speed: from the server\'s own timings where it reports them, else tokens over wall time'],
  ['prompt_tok_s','prompt tok/s','Prompt processing speed (servers that report timings)'],['busy_s','busy','Summed time spent on this work'],['utilisation','utilisation','Busy time over the window (can pass 100% with parallel work, shown capped)'],
  ['avg_s','avg','Average time per unit'],['failed','failed','Units that ended in an error']];
function table(){
  const rows=DATA[$('by').value]||[];
  if(!rows.length){$('tbl').innerHTML='<tr><td class="note">Nothing logged in this window yet. Chat with a model, run a project or a job, and it shows here.</td></tr>';return}
  const isDev=$('by').value==='by_device';
  $('tbl').innerHTML='<tr><th>'+esc($('by').selectedOptions[0].text)+'</th>'+COLS.map(c=>`<th title="${esc(c[2])}">${c[1]}</th>`).join('')+'</tr>'+
    rows.map(r=>'<tr><td class="name" title="'+esc(r.key)+'">'+(isDev?`<span class="sw" style="background:${color(r.key)}"></span>`:'')+esc(r.key)+'</td>'+COLS.map(([k])=>{
      const v=r[k];
      if(k.startsWith('share'))return `<td><span class="bar"><i style="width:${(v||0)*100}%"></i></span>${pct(v)}</td>`;
      if(k==='utilisation')return `<td>${pct(v)}</td>`;
      if(k==='busy_s'||k==='avg_s')return `<td>${dur(v)}</td>`;
      if(k==='failed')return `<td${v?' class="err"':''}>${v}</td>`;
      return `<td>${num(v)}</td>`}).join('')+'</tr>').join('');
}
function timeline(){
  const c=$('tl'),[g,W,H]=canvas(c),tl=DATA.timeline,keys=Object.keys(tl.series),n=keys.length?tl.series[keys[0]].length:0;
  g.clearRect(0,0,W,H);g.font='11px ui-sans-serif,system-ui';g.fillStyle=css('--mut');
  if(!n){g.fillText('No tokens in this window.',10,20);$('tlleg').innerHTML='';return}
  const tot=Array.from({length:n},(_,i)=>keys.reduce((s,k)=>s+tl.series[k][i],0)),mx=Math.max(1,...tot);
  const L=46,B=18,bw=(W-L-4)/n;
  g.strokeStyle=css('--line');g.beginPath();for(let q=0;q<=4;q++){const y=H-B-(H-B-6)*q/4;g.moveTo(L,y);g.lineTo(W,y);g.fillText(num(mx*q/4),2,y+4)}g.stroke();
  for(let i=0;i<n;i++){let y=H-B;for(const k of keys){const h=(H-B-6)*tl.series[k][i]/mx;if(h>0){g.fillStyle=color(k);g.fillRect(L+i*bw+1,y-h,Math.max(1,bw-2),h);y-=h}}}
  g.fillStyle=css('--mut');for(const [i,al] of [[0,'left'],[Math.floor(n/2),'center'],[n-1,'right']]){const t=new Date((tl.start+i*tl.bucket_s)*1000);g.textAlign=al;g.fillText(tl.bucket_s>=3600*12?t.toLocaleDateString():t.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),al==='right'?W:L+i*bw+(al==='center'?bw/2:0),H-4)}g.textAlign='left';
  c.onmousemove=e=>{const r=c.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-L)/bw);if(i<0||i>=n)return tip(e);
    tip(e,'<b>'+new Date((tl.start+i*tl.bucket_s)*1000).toLocaleString()+'</b><br>'+keys.filter(k=>tl.series[k][i]).map(k=>`<span class="sw" style="background:${color(k)}"></span>${esc(k)}: ${num(tl.series[k][i])}`).join('<br>')||'none')};
  c.onmouseleave=e=>tip(e);
  $('tlleg').innerHTML=keys.map(k=>`<span title="${esc(k)}"><span class="sw" style="background:${color(k)}"></span>${esc(k)}</span>`).join('');
}
function spark(c,t,ys,max,col,unit){
  const [g,W,H]=canvas(c);g.clearRect(0,0,W,H);const pts=t.map((x,i)=>[x,ys[i]]).filter(p=>p[1]!=null);
  g.strokeStyle=css('--line');g.strokeRect(.5,.5,W-1,H-1);if(pts.length<1)return;
  const t0=t[0],t1=t[t.length-1]||t0+1,mx=max||Math.max(1,...pts.map(p=>p[1]));
  g.strokeStyle=col;g.lineWidth=1.5;g.beginPath();pts.forEach(([x,y],i)=>{const X=(x-t0)/Math.max(1,t1-t0)*(W-2)+1,Y=H-2-(H-4)*y/mx;i?g.lineTo(X,Y):g.moveTo(X,Y)});g.stroke();
  const lab=num(pts[pts.length-1][1])+unit;g.font='10px ui-sans-serif,system-ui';g.fillStyle=css('--panel');g.fillRect(2,2,g.measureText(lab).width+6,12);g.fillStyle=css('--mut');g.fillText(lab,5,11);
}
function hardware(){
  const hw=DATA.hardware,keys=Object.keys(hw);
  if(!keys.length){$('hw').innerHTML='<p class="note">No hardware samples yet: Studio samples GPUs every 30 seconds while it runs (none were found, or the window is too short).</p>';return}
  $('hw').innerHTML=keys.map((k,i)=>{const s=hw[k].summary;return `<div><h3 title="${esc(k)}">${esc(k)}</h3>
    <div class="note">avg load ${s.avg_util==null?'–':s.avg_util+'%'} · peak ${s.peak_temp==null?'–':s.peak_temp+' °C'} · peak VRAM ${s.peak_vram_mb==null?'–':num(s.peak_vram_mb)+' MB'} · avg ${s.avg_power_w==null?'–':s.avg_power_w+' W'}${s.energy_wh?' · '+s.energy_wh+' Wh':''}</div>
    ${['util','vram','temp_c','power_w'].map(m=>`<div class="note" style="margin-top:.2rem">${{util:'load %',vram:'VRAM %',temp_c:'temperature °C',power_w:'power W'}[m]}</div><canvas id="hw${i}${m}" height="44"></canvas>`).join('')}</div>`}).join('');
  keys.forEach((k,i)=>{const h=hw[k];
    spark($('hw'+i+'util'),h.t,h.util,100,PAL[0],'%');
    spark($('hw'+i+'vram'),h.t,h.vram_used_mb.map((v,j)=>v!=null&&h.vram_total_mb[j]?100*v/h.vram_total_mb[j]:null),100,PAL[4],'%');
    spark($('hw'+i+'temp_c'),h.t,h.temp_c,null,PAL[3],' °C');spark($('hw'+i+'power_w'),h.t,h.power_w,null,PAL[2],' W')});
}

async function models(){
  try{const r=await (await fetch('/api/work/neurons')).json();const ms=r.models||[];
    $('model').innerHTML=ms.length?ms.map(m=>`<option value="${esc(m.model)}">${esc(m.model)} (${m.observations})</option>`).join(''):'<option value="">none recorded</option>';
    if(ms.length)neurons();else $('nmeta').textContent='No activation profiles recorded yet. Neuron review records them from checked replies, TestQA and benchmarks.';
  }catch(e){$('nmeta').innerHTML='<span class="err">'+esc(e.message)+'</span>'}
}
async function neurons(){
  const m=$('model').value;if(!m)return;
  $('nmeta').textContent='loading…';
  const r=await (await fetch('/api/work/neurons?model='+encodeURIComponent(m)+($('verdict').value?'&verdict='+$('verdict').value:''))).json();
  NDATA=r;if(r.error){$('nmeta').innerHTML='<span class="err">'+esc(r.error)+'</span>';return}
  drawNeurons();
}
function drawNeurons(){
  const r=NDATA;
  if(!r.n){$('nmeta').textContent='No observations match.';['box','hist','heat','diff'].forEach(id=>{const [g,W,H]=canvas($(id));g.clearRect(0,0,W,H)});return}
  $('nmeta').textContent=`${r.n} observations (the latest ${r.n}), ${r.layers} layers × ${r.bins} positions, ${new Date(r.first*1000).toLocaleDateString()} to ${new Date(r.last*1000).toLocaleDateString()}`+(r.n_wrong?` · ${r.n_wrong} wrong, ${r.n_correct} correct`:'');
  boxplot();hist();heat($('heat'),r.heatmap,false);
  if(r.wrong_minus_correct)heat($('diff'),r.wrong_minus_correct,true);
  else{const [g,W,H]=canvas($('diff'));g.clearRect(0,0,W,H);g.fillStyle=css('--mut');g.font='12px ui-sans-serif,system-ui';g.fillText('Needs both correct and wrong observations.',10,20)}
}
function boxplot(){
  const c=$('box'),[g,W,H]=canvas(c),p=NDATA.per_layer,L=p.p50.length;g.clearRect(0,0,W,H);
  const lo=Math.min(...p.p5),hi=Math.max(...p.p95),Y=v=>H-16-(H-24)*(v-lo)/Math.max(1e-9,hi-lo),X0=40,bw=(W-X0-4)/L;
  g.font='10px ui-sans-serif,system-ui';g.fillStyle=css('--mut');g.strokeStyle=css('--line');
  g.beginPath();for(let q=0;q<=4;q++){const v=lo+(hi-lo)*q/4;g.moveTo(X0,Y(v));g.lineTo(W,Y(v));g.fillText(v.toPrecision(2),2,Y(v)+3)}g.stroke();
  for(let i=0;i<L;i++){const x=X0+i*bw+bw/2,w=Math.max(1,bw*.6);
    g.strokeStyle=PAL[0];g.beginPath();g.moveTo(x,Y(p.p5[i]));g.lineTo(x,Y(p.p95[i]));g.stroke();
    g.fillStyle=PAL[0]+'66';g.fillRect(x-w/2,Y(p.p75[i]),w,Math.max(1,Y(p.p25[i])-Y(p.p75[i])));
    g.strokeStyle=css('--fg');g.beginPath();g.moveTo(x-w/2,Y(p.p50[i]));g.lineTo(x+w/2,Y(p.p50[i]));g.stroke();
    g.fillStyle=PAL[2];g.fillRect(x-1.5,Y(p.mean[i])-1.5,3,3)}
  g.fillStyle=css('--mut');for(const i of [0,Math.floor(L/2),L-1])g.fillText('L'+i,X0+i*bw,H-3);
  c.onmousemove=e=>{const r=c.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-X0)/bw);if(i<0||i>=L)return tip(e);
    tip(e,`<b>layer ${i}</b><br>p5 ${p.p5[i].toPrecision(3)} · p25 ${p.p25[i].toPrecision(3)}<br>median ${p.p50[i].toPrecision(3)} · mean ${p.mean[i].toPrecision(3)}<br>p75 ${p.p75[i].toPrecision(3)} · p95 ${p.p95[i].toPrecision(3)}<br>max ${p.max[i].toPrecision(3)}`)};
  c.onmouseleave=e=>tip(e);
}
function hist(){
  const c=$('hist'),[g,W,H]=canvas(c),h=NDATA.histogram,n=h.counts.length,mx=Math.max(1,...h.counts),bw=(W-8)/n;g.clearRect(0,0,W,H);
  for(let i=0;i<n;i++){const hh=(H-22)*Math.sqrt(h.counts[i]/mx);g.fillStyle=PAL[1];g.fillRect(4+i*bw,H-16-hh,Math.max(1,bw-1),hh)}
  g.fillStyle=css('--mut');g.font='10px ui-sans-serif,system-ui';g.fillText(h.edges[0].toPrecision(2),4,H-3);g.fillText(h.edges[n].toPrecision(2),W-40,H-3);g.fillText('(height: √count)',W-90,10);
  c.onmousemove=e=>{const r=c.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-4)/bw);if(i<0||i>=n)return tip(e);tip(e,`${h.edges[i].toPrecision(3)} to ${h.edges[i+1].toPrecision(3)}: ${h.counts[i]}`)};
  c.onmouseleave=e=>tip(e);
}
function heat(c,M,div){
  const [g,W,H]=canvas(c),L=M.length,B=M[0].length,cw=W/B,ch=H/L;g.clearRect(0,0,W,H);
  const flat=M.flat().sort((a,b)=>a-b),q=k=>flat[Math.floor(k*(flat.length-1))];
  const lo=div?-Math.max(Math.abs(q(.02)),Math.abs(q(.98)),1e-9):q(.02),hi=div?-lo:q(.98);
  for(let i=0;i<L;i++)for(let j=0;j<B;j++){let v=(M[i][j]-lo)/Math.max(1e-9,hi-lo);v=Math.max(0,Math.min(1,v));
    g.fillStyle=div?(v<.5?`rgba(111,147,255,${(.5-v)*2})`:`rgba(255,122,107,${(v-.5)*2})`):`hsl(${250-v*200},70%,${18+v*45}%)`;
    g.fillRect(j*cw,i*ch,Math.ceil(cw),Math.ceil(ch))}
  c.onmousemove=e=>{const r=c.getBoundingClientRect(),j=Math.floor((e.clientX-r.left)/cw),i=Math.floor((e.clientY-r.top)/ch);if(i<0||i>=L||j<0||j>=B)return tip(e);
    tip(e,`layer ${i}, position ${j}/${B}: ${M[i][j].toPrecision(3)}`)};
  c.onmouseleave=e=>tip(e);
}
async function predict(){
  $('pmeta').textContent='fitting…';
  let r;try{r=await (await fetch('/api/work/predict')).json()}catch(e){$('pmeta').innerHTML='<span class="err">'+esc(e.message)+'</span>';return}
  if(r.error){$('pmeta').innerHTML='<span class="err">'+esc(r.error)+'</span>';return}
  const m=r.model,pc=x=>x==null?'–':Math.round(x*100)+'%';
  $('pmeta').innerHTML=!m.n?'No speed measurements yet, so these come from memory bandwidth alone. Each reply Studio serves adds one (with llama-server\'s timings), and so do scoreboard cards when sharing is on.'+(r.board_error?` <span class="err">Scoreboard: ${esc(r.board_error)}</span>`:''):`Fitted on ${m.n} measurements (${r.samples['this computer']} from this computer, ${r.samples.scoreboard} from the scoreboard). Typical error: ${pc(m.fit_error)} on measured hardware`+(m.cv_error!=null?`, ${pc(m.cv_error)} on hardware it has not seen`:'')+'. Ranges are 80% intervals; the more measurements a device has, the narrower.'+(r.board_error?` <span class="err">Scoreboard: ${esc(r.board_error)}</span>`:'');
  if(!r.rows.length){$('ptbl').innerHTML='<tr><td class="note">No models found.</td></tr>';return}
  $('ptbl').innerHTML='<tr><th>model</th><th title="File size">GB</th>'+r.devices.map(d=>`<th class="name" title="${esc(d.id)} · ${esc(d.backend)} · ${d.memory_gb} GB">${esc(d.name||d.id)}</th>`).join('')+'</tr>'+
    r.rows.map(x=>`<tr><td class="name" title="${esc(x.model)}${x.quant?' · '+esc(x.quant):''}">${esc(x.model)}</td><td>${x.gb}</td>`+r.devices.map(d=>{const c=x.devices[d.id];
      const tip=`${c.low}–${c.high} tok/s predicted from ${c.bandwidth_gbs} GB/s (${c.specs_from})`+(c.measured!=null?`; measured ${c.measured}`:'')+(c.fits?'':'; does not fit in its memory');
      return `<td title="${esc(tip)}"${c.fits?'':' style="opacity:.45"'}>${num(c.tok_s)}${c.measured!=null?` <span class="note">(${num(c.measured)})</span>`:''}</td>`}).join('')+'</tr>').join('');
  const ex=m.explain||[];
  $('plearn').innerHTML=(ex.length?'<table>'+ex.map(e=>`<tr><td>${esc(e.feature)}</td><td>× ${e.factor}</td><td class="note">${esc(e.unit)}</td></tr>`).join('')+'</table>':'Nothing yet beyond the bandwidth estimate: it needs measurements.')+
    `<p>Per-device corrections: ${Object.entries(m.device||{}).map(([k,v])=>`${esc(k)} × ${Math.exp(v.offset).toFixed(2)} (${v.n})`).join(', ')||'none yet'}.</p>`;
}
async function measures(){
  let r;try{r=await (await fetch('/api/work/measures')).json()}catch(e){$('mmeta').innerHTML='<span class="err">'+esc(e.message)+'</span>';return}
  const ks=Object.keys(r.weights||{});
  if(!ks.length){$('mmeta').textContent='No model has scores from a measurement yet. TestQA, LiveBench, SWE-bench, activation checks and your right/wrong labels all count.';$('mtbl').innerHTML='';return}
  $('mmeta').textContent=`${r.models} models, weighted ${r.mode==='consensus'?'by agreement with the other measurements':'by agreement with '+r.mode.replace('against ','')}. Weights move toward an equal share while few models have results.`;
  $('mtbl').innerHTML='<tr><th>measurement</th><th title="Share of the combined score">weight</th><th title="Rank correlation across models">agreement</th><th title="Models compared">models</th><th title="Weight in earlier fits, oldest first">history</th></tr>'+
    ks.sort((a,b)=>r.weights[b]-r.weights[a]).map(k=>{const d=r.detail[k],h=(r.history||[]).map(x=>x.weights[k]).filter(x=>x!=null);
      return `<tr><td class="name" title="${esc(k)}">${esc(k)}</td><td><span class="bar"><i style="width:${r.weights[k]*100}%"></i></span>${pct(r.weights[k])}</td><td>${d.r==null?'–':d.r}</td><td>${d.n}</td><td>${h.slice(-12).map(x=>pct(x)).join(' → ')||'–'}</td></tr>`}).join('');
  $('mrank').innerHTML='<tr><th>model</th><th title="Weighted mean of z-scores">score</th><th>measured by</th></tr>'+(r.ranking||[]).map(x=>`<tr><td class="name" title="${esc(x.model)}">${esc(x.model)}</td><td>${x.score}</td><td class="name" title="${esc(x.systems.join(', '))}">${esc(x.systems.join(', '))}</td></tr>`).join('');
}
$('since').onchange=load;$('by').onchange=table;$('model').onchange=neurons;$('verdict').onchange=neurons;
onresize=()=>{if(DATA){timeline();hardware()}if(NDATA&&!NDATA.error)drawNeurons()};
setInterval(()=>{if($('auto').checked&&!document.hidden)load()},15000);
load();models();predict();measures();
</script>"""

from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
