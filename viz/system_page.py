"""Studio's /system page: what is running and how to restart any part of it, the hardware right now and
over time, and automation rules that react to the host's use and power."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio System</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;align-items:center;gap:.8rem;padding:.6rem 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}main{max-width:1100px;margin:0 auto;padding:1rem 16px 3rem;display:grid;gap:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}
h2{font-size:13px;margin:0 0 .6rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;display:flex;gap:.6rem;align-items:center}
h2 .r{margin-left:auto;text-transform:none;letter-spacing:0;font-weight:400}
.note{font-size:12.5px;color:var(--mut)}a{color:var(--acc)}
button{font:500 13px ui-sans-serif,system-ui;padding:.32rem .75rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--acc)}button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button:disabled{opacity:.5;cursor:default}
.svc{display:grid;grid-template-columns:minmax(10rem,14rem) minmax(7rem,9rem) 1fr auto;gap:.3rem .9rem;align-items:center;padding:.5rem 0;border-bottom:1px solid var(--line)}
.svc:last-child{border-bottom:none}.svc .nm{font-weight:600}.svc .det{font-size:12.5px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.svc .acts{display:flex;gap:.35rem;justify-content:flex-end}
.st{font-size:12.5px}.st.on{color:var(--ok)}.st.off{color:var(--mut)}.st.warn{color:var(--warn)}
@media(max-width:700px){.svc{grid-template-columns:1fr auto}.svc .det{grid-column:1/-1}}
#msg{min-height:1.2rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.6rem}
.tile{border:1px solid var(--line);border-radius:8px;padding:.5rem .6rem;background:var(--bg)}
.tile .k{font-size:11.5px;color:var(--mut);display:flex;justify-content:space-between;gap:.4rem}.tile .v{font:600 18px ui-sans-serif,system-ui;margin:.1rem 0}
.tile svg{display:block;width:100%;height:34px}.tile .sub{font-size:11.5px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
h3{font-size:12.5px;margin:.8rem 0 .4rem}
table{width:100%;border-collapse:collapse}td,th{padding:.35rem .3rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:500}
input,select{font:12.5px ui-sans-serif,system-ui;padding:.28rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.4rem .8rem}
label.f{display:flex;flex-direction:column;gap:.15rem;font-size:12px;color:var(--mut)}
.pill{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:0 .45rem;white-space:nowrap}.pill.on{color:var(--warn);border-color:var(--warn)}
pre{background:var(--code);padding:.6rem;border-radius:6px;font:12px ui-monospace,monospace;white-space:pre-wrap;word-break:break-all}
</style>
<header><h1>Studio · System</h1><span class="note" id="where"></span></header>
<main>
<section class="card" id="services"><h2>Running parts <span class="r"><button id="restartAll" class="pri" title="Stop every model this Studio started and start Studio again with the same settings">Restart Studio</button></span></h2>
<div class="note" style="margin-bottom:.4rem">Each part Studio runs on this computer. Restarting Studio stops all of them first, so nothing is left holding a port or GPU memory. Jobs keep running in the background only until Studio stops.</div>
<div id="svcList">…</div><div id="msg" class="note"></div></section>
<section class="card" id="monitor"><h2>Monitor <span class="r note" id="monnote"></span></h2>
<div class="note" style="margin-bottom:.5rem">This computer every 2 s and each remote host every 10 s while this page is open (or while automation rules are on); the last hour is kept.</div>
<div id="mon">…</div></section>

<section class="card" id="automation"><h2>Automation</h2>
<div class="note">Rules that react to how a computer is used and how much power it draws: unload its models, give it no new work, move its running tasks to another computer, or run one task at a time there. A host can also do the first part on its own: its agent's console has <i>When I use this computer</i>.</div>
<h3>Busy now</h3><div id="busy" class="note">nothing is busy</div>
<h3>Do it now</h3>
<div class="row"><select id="atarget" aria-label="Computer"></select>
<select id="aaction" aria-label="Action" title="busy: no new work · unload: unload its models and no new work · throttle: one task at a time · free: undo"><option value="busy">Give it no new work</option><option value="unload">Unload its models</option><option value="throttle">One task at a time</option><option value="free">Free it again</option></select>
<label class="row note">for <input id="amins" type="number" min="0" value="0" style="width:5rem"> min <span title="0 = until freed">(0 = until freed)</span></label>
<button id="ago" class="pri">Apply</button><span id="amsg" class="note"></span></div>
<h3>Rules</h3><table id="rules"></table>
<div class="row" style="margin-top:.5rem"><button id="radd">+ Add a rule</button><span id="rmsg" class="note"></span></div>
<div id="rform" hidden style="border:1px solid var(--line);border-radius:8px;padding:.7rem;margin-top:.5rem">
 <div class="grid2">
  <label class="f">Name<input id="rname" placeholder="e.g. Give the rig back"></label>
  <label class="f">Computer<select id="rtarget"></select></label>
  <label class="f">When<select id="rkind"><option value="activity">its owner uses it</option><option value="power">power use is over a limit (Home Assistant)</option><option value="manual">switched on by hand or the API</option></select></label>
 </div>
 <div id="rk-activity" class="grid2" style="margin-top:.4rem">
  <label class="f" title="keyboard or mouse input this recent counts as in use">In use if input in the last (s)<input id="ridle" type="number" value="60" min="5"></label>
  <label class="f" title="stay yielded this long after the last input">Take work again after quiet (s)<input id="rcool" type="number" value="300" min="0"></label>
  <label class="row note"><input type="checkbox" id="rpad" checked> game controller</label><label class="row note"><input type="checkbox" id="rvr" checked> VR session</label>
 </div>
 <div id="rk-power" class="grid2" hidden style="margin-top:.4rem">
  <label class="f" title="a Home Assistant sensor: power in W, or a cumulative energy meter in kWh">Sensor<input id="rent" placeholder="sensor.rig_power"></label>
  <label class="f">Measure<select id="rmeas"><option value="power">average power (W)</option><option value="energy">energy used (kWh)</option></select></label>
  <label class="f">Per<select id="rper"></select></label>
  <label class="f">Limit<input id="rlim" type="number" step="any" placeholder="e.g. 350"></label>
  <label class="f" title="for month and year: the day the billing period starts">Billing day<input id="rday" type="number" min="1" max="31" value="1"></label>
  <label class="f">Until<select id="runtil"><option value="period">the period ends</option><option value="clear">it is under the limit again</option></select></label>
 </div>
 <div class="row" style="margin-top:.5rem"><span class="note">Then:</span>
  <label class="row note"><input type="checkbox" data-act="unload" checked> unload its models</label>
  <label class="row note"><input type="checkbox" data-act="busy" checked> give it no new work</label>
  <label class="row note"><input type="checkbox" data-act="requeue" checked> move its running tasks elsewhere</label>
  <label class="row note"><input type="checkbox" data-act="throttle"> one task at a time</label></div>
 <div class="row" style="margin-top:.6rem"><button id="rsave" class="pri">Save rule</button><button id="rcancel">Cancel</button></div>
</div>
<h3>Home Assistant</h3>
<div class="grid2"><label class="f">Address<input id="haurl" placeholder="http://homeassistant.local:8123"></label>
<label class="f" title="Profile → Security → Long-lived access tokens. Stored in automation.json, readable only by you, and never sent back to this page.">Long-lived token<input id="hatok" type="password" placeholder="(kept)" autocomplete="off"></label>
<label class="f" title="or a file holding only the token">or token file<input id="hafile" data-pick="file"></label>
<label class="f">Test a sensor<input id="hatest" placeholder="sensor.rig_power"></label></div>
<div class="row" style="margin-top:.4rem"><button id="hasave">Save and test</button><span id="hamsg" class="note"></span></div>
<details style="margin-top:.6rem"><summary class="note" style="cursor:pointer">Automation API: let Home Assistant or a script act on a computer</summary>
<div class="note" style="margin:.4rem 0">POST to <code>/api/automation/action</code> with Studio's owner token. <code>target</code> is <code>local</code> or <code>host:NAME</code>; <code>action</code> is busy, unload, throttle or free; <code>minutes</code> 0 means until freed. A Home Assistant <code>rest_command</code>:</div><pre id="apiex"></pre></details>
<h3>Recent actions</h3><div id="alog" class="note">none yet</div>
</section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const LABEL={restart:'Restart',reload:'Reload',stop:'Stop',reconnect:'Reconnect'};
const TIP={'studio:restart':'Stop every model Studio started, then start Studio again with the same settings',
  'chat:reload':'Stop the chat model server and load the same model again with its settings','chat:stop':'Unload the chat model and free its memory',
  'workers:stop':'Stop every worker model (project tasks wait until one starts again)','embed:restart':'Stop the embedding server; the next document search starts it fresh',
  'mcp:reconnect':'Disconnect and reconnect every MCP server, re-reading mcp.json','labs:stop':'Stop every running lab'};
function draw(list){ $('#svcList').innerHTML=list.map(s=>{ const on=/running|connected/.test(s.state)&&!/^0 of/.test(s.state);
  return `<div class="svc" data-id="${esc(s.id)}"><div class="nm">${esc(s.name)}</div>
    <div class="st ${s.warn?'warn':on?'on':'off'}">${s.warn?'⚠ ':on?'● ':'○ '}${esc(s.state)}</div>
    <div class="det" title="${esc(s.detail)}">${esc(s.detail||'')}</div>
    <div class="acts">${(s.actions||[]).map(a=>`<button data-a="${a}" title="${esc(TIP[s.id+':'+a]||'')}">${LABEL[a]||a}</button>`).join('')}</div></div>`; }).join('');
  document.querySelectorAll('.svc [data-a]').forEach(b=>b.onclick=async()=>{ const id=b.closest('.svc').dataset.id, a=b.dataset.a;
    if(id==='studio') return nsRestart();
    b.disabled=true; const was=b.textContent; b.textContent='…';
    try{ const r=await api('/api/services',{id,action:a}); $('#msg').innerHTML=`<span style="color:var(--ok)">${esc(r.note)}</span>`; draw(r.services); }
    catch(e){ $('#msg').innerHTML=`<span style="color:var(--no)">${esc(e.message)}</span>`; b.disabled=false; b.textContent=was; } }); }
async function load(){ try{ const j=await api('/api/services'); draw(j.services); }catch(e){ $('#svcList').innerHTML=`<div class="note">${esc(e.message)}</div>`; } }
$('#restartAll').onclick=()=>nsRestart();
fetch('/api/status').then(r=>r.json()).then(j=>$('#where').textContent='on '+(j.computer||location.hostname)).catch(()=>{});
load(); setInterval(load,5000);
// ---- monitor
const fmt=(v,d=0)=>v==null?'–':(+v).toFixed(d);
function spark(vals,max){ const v=vals.filter(x=>x!=null); if(v.length<2) return '<svg></svg>'; const m=max||Math.max(...v,1);
  const n=vals.length, pts=vals.map((x,i)=>x==null?null:`${(i/(n-1)*100).toFixed(1)},${(34-30*x/m-2).toFixed(1)}`).filter(Boolean).join(' ');
  return `<svg viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="var(--acc)" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`; }
function tile(k,v,unit,series,max,sub,warn){ return `<div class="tile"><div class="k"><span>${esc(k)}</span>${warn?`<span style="color:var(--warn)">${esc(warn)}</span>`:''}</div><div class="v">${v}<span class="note" style="font-weight:400"> ${unit}</span></div>${spark(series,max)}<div class="sub" title="${esc(sub||'')}">${esc(sub||'')}</div></div>`; }
function gpuTiles(name,rows){ const last=rows[rows.length-1]||{}; const ids=(last.gpus||[]).map(g=>g.id);
  return ids.map(id=>{ const ser=k=>rows.map(r=>((r.gpus||[]).find(g=>g.id===id)||{})[k]); const g=(last.gpus||[]).find(x=>x.id===id)||{};
    const temp=g.junction_c??g.temp_c; return tile(`${name} · ${g.name||id}`, fmt(g.util), '% busy', ser('util'), 100,
      `${fmt((g.vram_used_mb||0)/1024,1)} / ${fmt((g.vram_total_mb||0)/1024,1)} GB · ${fmt(temp)}°C${g.power_w!=null?' · '+fmt(g.power_w)+' W':''}`, temp>=85?'hot':''); }).join(''); }
function presenceText(p){ if(!p) return 'in use: not known yet'; const k=p.idle_s!=null||p.gamepad!=null||p.vr!=null; if(!k) return 'cannot tell whether it is in use';
  const parts=[p.idle_s!=null?`last keyboard or mouse input ${Math.round(p.idle_s)} s ago`:'keyboard and mouse: cannot tell',
    p.gamepad?'a controller is in use':null, p.vr?'VR is running':p.vr===false?'no VR':null].filter(Boolean);
  return parts.join(', '); }
async function mon(){ let j; try{ j=await api('/api/system/monitor'); }catch(e){ $('#mon').innerHTML=`<div class="note">${esc(e.message)}</div>`; return; }
  const L=j.local||[], last=L[L.length-1]||{};
  let h=`<h3 style="margin-top:0">This computer (${esc(j.computer)}) <span class="note">· ${esc(presenceText((j.presence||{}).local))}${j.busy&&j.busy.local?' · <span style="color:var(--warn)">busy: '+esc(j.busy.local.why)+'</span>':''}</span></h3><div class="tiles">`
    +tile('CPU',fmt(last.cpu),'%',L.map(r=>r.cpu),100,'')
    +tile('Memory',fmt(last.ram_used_gb,1),`of ${fmt(last.ram_total_gb,1)} GB`,L.map(r=>r.ram_used_gb),last.ram_total_gb,'')
    +gpuTiles('GPU',L)+'</div>';
  for(const [name,rows] of Object.entries(j.hosts||{})){ const lr=rows[rows.length-1]||{}; const b=(j.busy||{})['host:'+name];
    h+=`<h3>${esc(name)} <span class="note">· remote host · ${lr.error?'<span style="color:var(--no)">'+esc(lr.error)+'</span>':esc(presenceText((j.presence||{})['host:'+name]))}${b?' · <span style="color:var(--warn)">busy: '+esc(b.why)+'</span>':''}</span></h3>`
      +`<div class="tiles">${gpuTiles(name,rows)||'<div class="note">no GPU readings</div>'}</div>`; }
  $('#mon').innerHTML=h; $('#monnote').textContent=L.length?`${L.length} samples`:'starting…'; }
mon(); setInterval(mon,2000);
// ---- automation
let A=null;
const KIND_TXT=r=>{ const w=r.when; if(w.kind==='activity') return `its owner uses it (input in the last ${w.idle_under} s${w.gamepad?', controller':''}${w.vr?', VR':''}; free after ${w.cooldown} s quiet)`;
  if(w.kind==='power') return `${w.measure==='energy'?'energy':'average power'} per ${w.period} over ${w.limit} ${w.measure==='energy'?'kWh':'W'} (${esc(w.entity)})${['month','year'].includes(w.period)?`, billing day ${w.reset_day}`:''}`;
  return 'switched on by hand or the API'; };
const tname=t=>t==='local'?'this computer':t.slice(5);
async function auto(){ try{ A=await api('/api/automation'); }catch(e){ return; }
  const tg=A.targets.map(t=>`<option value="${esc(t)}">${esc(tname(t))}</option>`).join('');
  for(const id of ['atarget','rtarget']){ const el=$('#'+id), v=el.value; el.innerHTML=tg; if(v) el.value=v; }
  if(!$('#rper').options.length) $('#rper').innerHTML=A.periods.map(p=>`<option ${p==='month'?'selected':''}>${p}</option>`).join('');
  const busy=Object.entries(A.busy||{});
  $('#busy').innerHTML=busy.length?busy.map(([t,b])=>`<div class="row"><span class="pill on">busy</span><b>${esc(tname(t))}</b> <span>${esc(b.why||'')}</span>${b.until?`<span class="note">until ${new Date(b.until*1000).toLocaleString()}</span>`:''}<span class="note">(${esc(b.source)})</span>${b.source!=='host'?`<button data-free="${esc(t)}">Free</button>`:''}</div>`).join(''):'nothing is busy';
  document.querySelectorAll('[data-free]').forEach(b=>b.onclick=async()=>{ await api('/api/automation/action',{target:b.dataset.free,action:'free'}); auto(); });
  $('#rules').innerHTML=A.rules.length?'<tr><th>on</th><th>rule</th><th>computer</th><th>when</th><th>then</th><th>now</th><th></th></tr>'+A.rules.map((r,i)=>{ const res=(A.results||{})[r.id]||{};
    return `<tr><td><input type="checkbox" data-on="${i}" ${r.enabled?'checked':''} aria-label="rule on"></td><td>${esc(r.name)}</td><td>${esc(tname(r.target))}</td><td>${KIND_TXT(r)}</td><td>${r.then.map(esc).join(', ')}</td>
      <td>${res.active?`<span class="pill on" title="${esc(res.why)}">active</span>`:r.enabled?`<span class="note" title="${esc(res.why||'')}">${res.why?esc(res.why):'waiting'}</span>`:'<span class="note">off</span>'}${r.when.kind==='manual'?` <button data-man="${esc(r.id)}">${(A.manual||{})[r.id]?'Switch off':'Switch on'}</button>`:''}</td>
      <td><button data-del="${i}" aria-label="Delete rule">Delete</button></td></tr>`; }).join(''):'<tr><td class="note">No rules yet. <b>+ Add a rule</b> to give a computer back when its owner uses it, or to cap its power use.</td></tr>';
  document.querySelectorAll('[data-on]').forEach(c=>c.onchange=()=>{ A.rules[+c.dataset.on].enabled=c.checked; saveRules(); });
  document.querySelectorAll('[data-del]').forEach(b=>b.onclick=()=>{ if(confirm('Delete this rule?')){ A.rules.splice(+b.dataset.del,1); saveRules(); } });
  document.querySelectorAll('[data-man]').forEach(b=>b.onclick=async()=>{ await api('/api/automation/manual',{rule:b.dataset.man,on:!(A.manual||{})[b.dataset.man]}); auto(); });
  if(document.activeElement!==$('#haurl')) $('#haurl').value=A.ha.url||''; $('#hatok').placeholder=A.ha.has_token?'(kept)':'paste a long-lived token'; if(document.activeElement!==$('#hafile')) $('#hafile').value=A.ha.token_file||'';
  $('#apiex').textContent=`rest_command:\n  rig_busy:\n    url: "${location.origin}/api/automation/action"\n    method: post\n    headers:\n      Authorization: "Bearer YOUR_STUDIO_TOKEN"\n    content_type: "application/json"\n    payload: '{"target": "host:rig", "action": "unload", "minutes": 0, "reason": "gaming"}'`;
  $('#alog').innerHTML=(A.log||[]).length?A.log.map(l=>`<div>${new Date(l.t*1000).toLocaleTimeString()} · <b>${esc(tname(l.target))}</b> ${esc(l.what)} <span class="note">${esc(l.detail||'')}</span></div>`).join(''):'none yet'; }
async function saveRules(){ try{ const r=await api('/api/automation/rules',{rules:A.rules}); $('#rmsg').textContent='saved'; }catch(e){ $('#rmsg').textContent=e.message; } auto(); }
$('#rkind').onchange=()=>{ $('#rk-activity').hidden=$('#rkind').value!=='activity'; $('#rk-power').hidden=$('#rkind').value!=='power'; };
$('#radd').onclick=()=>{ $('#rform').hidden=false; $('#rname').focus(); };
$('#rcancel').onclick=()=>{ $('#rform').hidden=true; };
$('#rsave').onclick=async()=>{ const k=$('#rkind').value; const when={kind:k};
  if(k==='activity') Object.assign(when,{idle_under:+$('#ridle').value,cooldown:+$('#rcool').value,gamepad:$('#rpad').checked,vr:$('#rvr').checked});
  if(k==='power') Object.assign(when,{entity:$('#rent').value.trim(),measure:$('#rmeas').value,period:$('#rper').value,limit:+$('#rlim').value,reset_day:+$('#rday').value});
  const r={name:$('#rname').value.trim()||($('#rkind').selectedOptions[0].text),target:$('#rtarget').value,enabled:true,when,
    then:[...document.querySelectorAll('[data-act]:checked')].map(c=>c.dataset.act),until:k==='power'?$('#runtil').value:'clear'};
  try{ await api('/api/automation/rules',{rules:[...A.rules,r]}); $('#rform').hidden=true; $('#rmsg').textContent='rule added'; auto(); }catch(e){ $('#rmsg').textContent=e.message; } };
$('#ago').onclick=async()=>{ try{ await api('/api/automation/action',{target:$('#atarget').value,action:$('#aaction').value,minutes:+$('#amins').value||0}); $('#amsg').textContent='done'; auto(); }catch(e){ $('#amsg').textContent=e.message; } };
$('#hasave').onclick=async()=>{ try{ const r=await api('/api/automation/ha',{url:$('#haurl').value.trim(),token:$('#hatok').value.trim()||undefined,token_file:$('#hafile').value.trim(),test_entity:$('#hatest').value.trim()||undefined});
  $('#hatok').value=''; $('#hamsg').innerHTML=r.reading?`<span style="color:var(--ok)">${esc(r.reading.name)}: ${esc(r.reading.value)} ${esc(r.reading.unit||'')}</span>`:'saved'; auto(); }catch(e){ $('#hamsg').innerHTML=`<span style="color:var(--no)">${esc(e.message)}</span>`; } };
auto(); setInterval(auto,10000);
</script>
"""

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
