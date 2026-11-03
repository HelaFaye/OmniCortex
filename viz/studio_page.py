"""Studio's own pages, kept apart from the server code in studio.py: the main page (chat, models, discover,
documents, tools), device pairing and linking, and the login form. studio.py fills in the model defaults and the
shared navigation script (ui_links) when it imports them."""

_PAIR_STYLE = """<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--acc:#2c5f8a;--accfg:#fff;--ok:#2f7d4f;--no:#b23c2e;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--panel:#1e1e1c;--fg:#ecebe6;--mut:#9b9a93;--line:#34332f;--acc:#7aa7d6;--accfg:#0f0f0e;--ok:#5fb27f;--no:#e0705f;--code:#262522;color-scheme:dark}}
:root[data-theme=dark]{--bg:#161615;--panel:#1e1e1c;--fg:#ecebe6;--mut:#9b9a93;--line:#34332f;--acc:#7aa7d6;--accfg:#0f0f0e;--ok:#5fb27f;--no:#e0705f;--code:#262522;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}a{color:var(--acc)}
main{max-width:900px;margin:0 auto;padding:1rem;display:grid;gap:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem}
h2{font-size:13px;margin:0 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
input{font:13px ui-monospace,monospace;padding:.35rem .45rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
select{font:13px ui-sans-serif,system-ui;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
#access{width:100%}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no)}.ok{color:var(--ok)}
code,.code{font:12.5px ui-monospace,monospace;background:var(--code);padding:.15rem .35rem;border-radius:4px;word-break:break-all}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:.3rem .25rem;border-bottom:1px solid var(--line)}
.row{display:flex;gap:.5rem;align-items:flex-end;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
label{display:block;font-size:12px;color:var(--mut);margin:.3rem 0 .1rem}
</style><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>"""

PAIR_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pair device</title>""" + _PAIR_STYLE + r"""
<header><h1>OmniCortex Studio · Pair this device</h1></header>
<main><div class="card"><h2>Pair</h2><div id="msg" class="note">Reading the pairing code…</div>
<div id="form" style="display:none"><label>Name for this device</label><input id="name">
<button class="pri" id="go" style="margin-top:.6rem">Pair and open Studio</button></div></div></main>
<script>
const p=new URLSearchParams(location.hash.slice(1)), code=p.get('c'), acc=p.get('a'), $=s=>document.querySelector(s);
const accText=acc==='p'?'Access lasts until the owner revokes it.':acc?`Access is temporary: it ends ${acc} after pairing.`:'';
history.replaceState(null,'',location.pathname);         // keep the code out of history
if(!code){ $('#msg').innerHTML='<span class="err">This link has no pairing code. Ask the owner for a new one.</span>'; }
else { $('#msg').textContent='Pairing code '+code+'. This browser will get its own access, which the owner can revoke. '+accText;
  $('#name').value=(navigator.userAgentData?.platform||navigator.platform||'browser')+' browser'; $('#form').style.display=''; }
$('#go').onclick=async()=>{ const r=await fetch('/api/pair/claim',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,name:$('#name').value,cookie:true})}); const j=await r.json().catch(()=>({}));
  if(r.ok){ if(j.expires) alert('Paired. Access ends '+new Date(j.expires*1000).toLocaleString()+'.'); location.href='/'; } else { $('#msg').innerHTML='<span class="err">'+(j.error||r.status)+'</span>'; } };
</script>"""

LINK_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Devices</title>""" + _PAIR_STYLE + r"""
<header><h1>Studio · Devices</h1><a href="/">← Studio</a></header>
<main>
<div class="card"><h2>Pair a device with this Studio</h2>
 <div class="note">Creates a one-time link (<span id="codettl">5 minutes</span>). Open it on a phone or laptop, or paste it into another Studio below. Each device gets its own token, revocable here; the master token is never shared. The link carries this server's certificate fingerprint, so the other side pins it instead of trusting any certificate.</div>
 <div class="row" style="margin-top:.5rem"><div style="max-width:260px"><label>Access</label><select id="access"></select></div>
 <div id="customBox" style="max-width:160px;display:none"><label>Duration (e.g. 12h, 3d)</label><input id="custom"></div>
 <button class="pri" id="start">Create pairing link</button></div>
 <div id="policy" class="note" style="margin-top:.3rem"></div>
 <div id="pairout" style="margin-top:.6rem"></div></div>
<div class="card"><h2>Paired devices</h2><div id="devs" class="note">loading…</div></div>
<div class="card"><h2>Use another machine's models here</h2>
 <div class="note">On the other machine's Studio, open Connect → Devices → Create pairing link, and paste it here. Its models then appear in this Studio's model picker and its <code>/v1</code> API as <code>name:model</code>; requests are served by that machine.</div>
 <div class="row"><div><label>Pairing link from the other Studio</label><input id="lnk" placeholder="https://host:7870/pair#c=…&fp=…"></div>
 <div style="max-width:200px"><label>Name here</label><input id="lname" placeholder="gpu-box"></div><button class="pri" id="add">Link</button></div>
 <div id="addmsg" class="note" style="margin-top:.4rem"></div>
 <div id="links" style="margin-top:.6rem"></div></div>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const when=t=>t?new Date(t*1000).toLocaleString():'never';
let policy=null;
const left=t=>{ const s=t-Date.now()/1000; if(s<=0) return 'expired';
  return s>86400?`${Math.floor(s/86400)}d ${Math.floor(s%86400/3600)}h left`:s>3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m left`:`${Math.ceil(s/60)}m left`; };
const accessCell=d=>d.persistent?'persistent':d.expired?`<span class="err">expired ${when(d.expires)}</span>`:`until ${when(d.expires)} <span class="note">(${left(d.expires)})</span>`;
function accessChoice(){ const v=$('#access').value; if(v==='persistent') return {persistent:true};
  if(v==='custom') return {persistent:false, ttl:$('#custom').value.trim()}; return {persistent:false, ttl:v}; }
function renderPolicy(pol){ policy=pol; const keep=$('#access').value;
  $('#access').innerHTML=(pol.allow_persistent?'<option value="persistent">Persistent (until revoked)</option>':'')+
    pol.presets.map(p=>`<option value="${p}">Temporary: ${p}</option>`).join('')+`<option value="custom">Temporary: custom…</option>`;
  $('#access').value=keep||(pol.default==='persistent'?'persistent':pol.default);
  if(!$('#access').value) $('#access').selectedIndex=0;
  $('#codettl').textContent=pol.code_ttl>=120?`${Math.round(pol.code_ttl/60)} minutes`:`${pol.code_ttl} seconds`;
  $('#policy').textContent=`Host policy: temporary access up to ${pol.max_ttl}`+(pol.allow_persistent?'; persistent allowed.':'; persistent pairing is disabled on this host.'); }
$('#access').onchange=()=>{ $('#customBox').style.display=$('#access').value==='custom'?'':'none'; };
async function load(){ const r=await fetch('/api/devices'); const j=await r.json();
  if(!r.ok){ $('#devs').innerHTML='<span class="err">'+esc(j.error)+'</span>'; $('#start').disabled=true; $('#add').disabled=true; return; }
  renderPolicy(j.policy);
  $('#devs').innerHTML=j.devices.length?'<table><tr><th>device</th><th>paired</th><th>last seen</th><th>access</th><th></th></tr>'+j.devices.map(d=>`<tr><td>${esc(d.name)}</td><td>${when(d.created)}</td><td>${when(d.last_seen)}</td><td>${accessCell(d)}</td><td style="white-space:nowrap"><select data-u="${esc(d.id)}"><option value="">change…</option>${policy.allow_persistent&&!d.persistent?'<option value="persistent">make persistent</option>':''}${policy.presets.map(p=>`<option value="${p}">${d.persistent?'expire in':d.expired?'renew for':'reset to'} ${p}</option>`).join('')}</select> <button data-r="${esc(d.id)}">Revoke</button></td></tr>`).join('')+'</table>':'No paired devices.';
  document.querySelectorAll('[data-r]').forEach(b=>b.onclick=async()=>{ if(confirm('Revoke this device? It loses access immediately.')){ await post('/api/devices/revoke',{id:b.dataset.r}); load(); } });
  document.querySelectorAll('[data-u]').forEach(sel=>sel.onchange=async()=>{ const v=sel.value; if(!v) return;
    const r=await post('/api/devices/update', v==='persistent'?{id:sel.dataset.u,persistent:true}:{id:sel.dataset.u,persistent:false,ttl:v});
    if(!r.ok) alert((await r.json()).error); load(); });
  $('#links').innerHTML=j.links.length?'<table><tr><th>name</th><th>host</th><th>access</th><th>models</th><th></th></tr>'+j.links.map(l=>`<tr><td>${esc(l.name)}</td><td><code>${esc(l.url)}</code> ${l.pinned?'🔒 pinned':''}</td><td>${l.expires?accessCell({expires:l.expires,expired:l.expires<Date.now()/1000}):'persistent'}</td><td>${l.error?'<span class="err">'+esc(l.error)+'</span>':l.models}</td><td><button data-l="${esc(l.name)}">Unlink</button></td></tr>`).join('')+'</table>':'';
  document.querySelectorAll('[data-l]').forEach(b=>b.onclick=async()=>{ await post('/api/links/remove',{name:b.dataset.l}); load(); }); }
$('#start').onclick=async()=>{ const r=await post('/api/pair/start',accessChoice()); const j=await r.json();
  const grants=j.persistent?'persistent access (until revoked)':`temporary access for ${Math.round(j.access_seconds/3600*10)/10} h after pairing`;
  $('#pairout').innerHTML=r.ok?`<div><span class="code" id="plink">${esc(j.link)}</span> <button id="cp">Copy</button></div><div class="note">Grants ${grants}. Code <b>${esc(j.code)}</b>, claimable until ${new Date(j.expires*1000).toLocaleTimeString()}, single use.${j.fingerprint?'':' <span class="err">No TLS on this Studio: use the link only inside a VPN.</span>'}</div>`:'<span class="err">'+esc(j.error)+'</span>';
  if(r.ok) $('#cp').onclick=()=>navigator.clipboard.writeText(j.link); };
$('#add').onclick=async()=>{ $('#addmsg').textContent='pairing…'; const r=await post('/api/links/add',{link:$('#lnk').value.trim(),name:$('#lname').value.trim()||'remote',device_name:location.host});
  const j=await r.json(); $('#addmsg').innerHTML=r.ok?`<span class="ok">Linked ${esc(j.name)}: ${j.models.length} models.</span>`:'<span class="err">'+esc(j.error)+'</span>'; if(r.ok){ $('#lnk').value=''; } load(); };
load();
</script>"""

PAGE = r"""<!DOCTYPE html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Studio</title><link rel="icon" href="data:,"><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--ok:#2f7d4f;--no:#b23c2e;--warn:#a8701c;
  --acc:#2c5f8a;--accfg:#fff;--soft:#f0f5fa;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;
  --line:#2a2f37;--ok:#5fcf8f;--no:#ff8b7e;--warn:#e7b65c;--acc:#6ea8e0;--accfg:#0d1117;--soft:#1d2733;--code:#20242b;color-scheme:dark}}
:root[data-theme=dark]{--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;--line:#2a2f37;--ok:#5fcf8f;--no:#ff8b7e;
  --warn:#e7b65c;--acc:#6ea8e0;--accfg:#0d1117;--soft:#1d2733;--code:#20242b;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{padding:.55rem 1rem;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:.5rem;background:var(--panel);flex-wrap:wrap}
h1{margin:0;font-size:15px;font-weight:600}
.status{font-size:13px;color:var(--mut);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--line);margin-right:.35rem}.dot.on{background:var(--ok)}
.chip{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;color:var(--mut);cursor:pointer;background:var(--bg)}
main{flex:1;display:grid;grid-template-columns:330px 1fr;min-height:0}
@media(max-width:800px){main{grid-template-columns:1fr;grid-template-rows:auto 1fr}aside{max-height:30vh;border-right:none;border-bottom:1px solid var(--line)}
  .dials{gap:.45rem .7rem;padding:.4rem .8rem}.dials input[type=range]{width:90px}#export{display:none}
  form{padding:.5rem .8rem}#log{padding:.7rem .8rem}header{gap:.5rem}#api{display:none}}
aside{border-right:1px solid var(--line);overflow-y:auto;padding:.7rem;background:var(--panel)}
.tabs{display:flex;gap:.25rem;margin-bottom:.6rem;flex-wrap:wrap}.tabs button{flex:1 1 auto;min-width:0;padding:.35rem .45rem;font-size:12.5px}
section.chat{display:flex;flex-direction:column;min-height:0;min-width:0}
.m{border:1px solid var(--line);border-radius:8px;padding:.5rem .65rem;margin-bottom:.4rem;cursor:pointer;background:var(--panel);position:relative}
.m:hover{background:var(--soft)}.m.sel{border-color:var(--acc);background:var(--soft)}
.mn{font-weight:500;font-size:13px;word-break:break-all}.mm{font-size:12px;color:var(--mut);margin-top:.1rem}
.x{position:absolute;right:.4rem;top:.3rem;border:none;background:none;color:var(--mut);cursor:pointer;padding:0 .3rem}
.tag{display:inline-block;font-size:11px;padding:0 .35rem;border:1px solid var(--line);border-radius:4px;margin-right:.25rem}
.bad{color:var(--no)}.warn{color:var(--warn)}.okc{color:var(--ok)}
details.eo{margin-top:.5rem;border:1px solid var(--line);border-radius:6px;padding:.3rem .5rem}details.eo>summary{cursor:pointer;font-size:12.5px}
.eoG{margin:.3rem 0}.eoG>summary{cursor:pointer;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
.eoRow{margin:.25rem 0}.eoRow label{margin:.2rem 0 .1rem}.eoRow code{font-size:10.5px;color:var(--mut)}.eoRow .note{font-size:11px}
label{display:block;font-size:12px;color:var(--mut);margin:.4rem 0 .1rem}
input,select,textarea{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{background:var(--soft)}button:disabled{opacity:.45;cursor:default}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
#log{flex:1;overflow-y:auto;padding:1rem 1.2rem}
.msg{max-width:800px;margin:0 auto 1.1rem}
.who{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.2rem;display:flex;gap:.6rem;align-items:center}
.who button{font-size:11px;padding:0 .4rem;border:none;background:none;color:var(--mut)}
.body{white-space:pre-wrap;word-wrap:break-word}
.body pre{background:var(--code);padding:.6rem .7rem;border-radius:6px;overflow-x:auto;white-space:pre;font:12.5px ui-monospace,monospace}
.body code{background:var(--code);padding:0 .25rem;border-radius:3px;font:12.5px ui-monospace,monospace}
details.think{color:var(--mut);font-size:13px;border-left:2px solid var(--line);padding-left:.7rem;margin-bottom:.5rem}
details.think summary{cursor:pointer}
.stats{font-size:11px;color:var(--mut);margin-top:.3rem}
.risk{font-size:12px;margin-top:.35rem}.risk:empty{display:none}
.risk .sum{color:var(--mut);margin-bottom:.25rem}.risk .sum b{color:var(--fg)}
.risk .rt{white-space:pre-wrap;word-wrap:break-word;line-height:1.7;font-size:13px}
.pchk{font-size:12px;margin-top:.35rem}.pchk:empty{display:none}.pchk .sum{color:var(--mut);margin-bottom:.25rem}.pchk .sum b{color:var(--fg)}.pchk .sum b.bad{color:var(--no)}
.pchk .rt{white-space:pre-wrap;word-wrap:break-word;line-height:1.7;font-size:13px;color:var(--fg)}.pchk mark{background:color-mix(in srgb,var(--acc) 30%,transparent);color:inherit;border-radius:3px;padding:0 1px}
.risk .rt span{border-radius:2px}.risk .rt .fl{text-decoration:underline 2px var(--no);text-underline-offset:3px}
.imgs{display:flex;gap:.4rem;flex-wrap:wrap;margin:.3rem 0}.imgs img{max-height:120px;max-width:200px;border-radius:6px;border:1px solid var(--line)}
form{border-top:1px solid var(--line);padding:.6rem 1.2rem;display:flex;gap:.5rem;background:var(--panel);align-items:flex-end;flex-wrap:wrap}
#in{flex:1;min-width:200px;font:14px ui-sans-serif,system-ui;padding:.5rem .6rem;resize:none}
#pending{width:100%}
.dials{border-top:1px solid var(--line);padding:.5rem 1.2rem;font-size:12px;color:var(--mut);display:flex;gap:1rem;align-items:center;background:var(--panel);flex-wrap:wrap}
.dials select{width:auto}.dials input[type=range]{width:140px}
.note{font-size:12px;color:var(--mut);margin:.4rem 0}
.empty{max-width:560px;margin:15vh auto;text-align:center;color:var(--mut)}
ol.gs{list-style:none;padding:0;margin:.8rem 0 0;display:grid;gap:.55rem}ol.gs li{display:flex;gap:.6rem;align-items:flex-start;color:var(--fg)}
ol.gs .ck{flex:none;width:20px;height:20px;border-radius:50%;border:1.5px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:12px;margin-top:1px}
ol.gs li.done .ck{background:var(--ok);border-color:var(--ok);color:var(--accfg)}ol.gs li.done b{color:var(--mut)}
.starters{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.35rem}.starters button{text-align:left;font-weight:400;font-size:12.5px;line-height:1.35}
.tasks{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.35rem}
.task{display:flex;flex-direction:column;gap:.15rem;padding:.5rem .6rem;border:1px solid var(--line);border-radius:8px;text-decoration:none;color:inherit;font-size:12.5px;line-height:1.35}
.task:hover{border-color:var(--acc)}.task b{color:var(--fg)}.task span{color:var(--mut);font-size:12px}
@media(max-width:760px){.tasks{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.starters{grid-template-columns:1fr}.tasks{grid-template-columns:1fr}.empty{margin:4vh auto}}
.warnico{color:#e8890c;cursor:help;font-weight:700;margin-left:.3rem}
.stat{font-size:11px;color:var(--mut);cursor:help}
.route{font-size:11px;color:var(--mut);margin-top:.2rem}
table.st{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:.4rem}
table.st th,table.st td{text-align:left;padding:.15rem .3rem;border-bottom:1px solid var(--line)}
#modelPick{width:auto;max-width:260px}
.tool{font-size:12px;border:1px solid var(--line);border-radius:6px;padding:.35rem .5rem;margin:.3rem 0;background:var(--code)}
.tool .will{margin:.3rem 0}.tool .wl{display:flex;gap:.4rem;align-items:baseline;margin:.15rem 0}.tool .wl code{flex:1;white-space:pre-wrap;word-break:break-all;background:var(--bg);padding:.15rem .35rem;border-radius:4px}
.tool .wl.deny code{outline:1px solid var(--no)}.tool .wl.ask code{outline:1px solid var(--warn,#e7b65c)}
.tool .qp{display:flex;gap:.35rem;flex-wrap:wrap;align-items:center;margin-top:.3rem}
.tool .tn{font:600 12px ui-monospace,monospace}.tool pre{white-space:pre-wrap;margin:.2rem 0;font:12px ui-monospace,monospace;max-height:220px;overflow:auto}
.tool button{font-size:11px;padding:.1rem .5rem;margin-right:.3rem}
.src{font-size:12px;color:var(--mut);margin-top:.3rem}.src details{margin:.1rem 0}.src summary{cursor:pointer}
.src pre{white-space:pre-wrap;font:12px ui-monospace,monospace;background:var(--code);padding:.4rem;border-radius:5px;margin:.2rem 0}
/* the context strip over the chat, and what it opens: environment, context usage, outputs */
section.chat{position:relative}
#ctxStrip{display:flex;align-items:center;gap:.6rem;padding:.3rem 1.2rem;border-bottom:1px solid var(--line);background:var(--panel);font-size:12.5px;color:var(--mut);cursor:pointer;user-select:none;min-width:0}
#ctxStrip:hover{background:var(--soft)}#ctxStrip:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
#ctxStrip .sm{color:var(--fg);font-weight:500;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:32%}
#ctxStrip .cbar{flex:1;max-width:340px;height:7px;margin:0}#ctxStrip .chev{margin-left:auto;transition:transform .15s}
#ctxStrip[aria-expanded=true] .chev{transform:rotate(180deg)}
#side{position:absolute;left:0;right:0;top:var(--strip-h,31px);z-index:50;background:var(--panel);border-bottom:1px solid var(--line);box-shadow:0 10px 24px rgba(0,0,0,.22);
  padding:.4rem 1.2rem .7rem;font-size:13px;display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:1.2rem;max-height:60vh;overflow-y:auto}
#side[hidden]{display:none}
#side h3{margin:.3rem 0 .45rem;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);display:flex;gap:.4rem}#side h3 .r{margin-left:auto;text-transform:none;letter-spacing:0}
@media(max-width:900px){#side{grid-template-columns:1fr;gap:.3rem}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.15rem .7rem;font-size:12.5px;margin:0}.kv dt{color:var(--mut)}.kv dd{margin:0;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cbar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--code);margin:.2rem 0 .5rem;border:1px solid var(--line)}
.cbar span{display:block;height:100%;min-width:0}.cbar .rsv{background:repeating-linear-gradient(45deg,var(--line) 0 3px,transparent 3px 6px)}
.cleg{display:grid;grid-template-columns:10px 1fr auto;gap:.2rem .45rem;align-items:center;font-size:12.5px}.cleg i{width:10px;height:10px;border-radius:3px;display:block}
.cleg .n{color:var(--mut);font-variant-numeric:tabular-nums}
.outs a,.outs div.o{display:flex;gap:.45rem;align-items:baseline;padding:.25rem .3rem;border-radius:5px;color:var(--fg);text-decoration:none;font-size:12.5px}
.outs a:hover{background:var(--soft)}.outs .k{color:var(--mut);font-size:11px;margin-left:auto;white-space:nowrap}
</style>
<header>
  <h1>OmniCortex Studio</h1>
  <div class="status"><span id="dot" class="dot"></span><span id="st">no model loaded</span></div>
  <span class="chip" id="api" title="OpenAI-compatible endpoint; click to copy"></span>
  <span style="margin-left:auto"></span>
  <button id="theme" title="toggle theme">◐</button>
  <button id="unload" disabled>Unload</button>
</header>
<main>
<aside>
  <div class="tabs">
    <button data-tab="chats" class="pri">Chats</button>
    <button data-tab="local">Models</button>
    <button data-tab="hub">Discover</button>
    <button data-tab="docs">Docs</button>
    <button data-tab="tools">Tools</button>
  </div>
  <div id="tab-tools" style="display:none">
    <div class="note">MCP servers whose tools the model may call when <b>tools</b> is on under the chat. Every call asks first unless the server's <code>autoApprove</code> allows it.</div>
    <div id="mcpcfg" class="note"></div>
    <button id="mcpreload" style="width:100%;margin:.4rem 0">Reconnect / reload config</button>
    <div id="mcplist"></div>
  </div>
  <div id="tab-docs" style="display:none">
    <div class="note">Document collections for chat. Pick one under the chat ("docs") and replies cite the passages they used.</div>
    <div style="display:flex;gap:.4rem;margin:.4rem 0"><input id="newcoll" placeholder="new collection name"><button id="mkcoll">Create</button></div>
    <div id="colls"></div>
    <div id="collpanel" style="display:none">
      <hr style="border:none;border-top:1px solid var(--line);margin:.6rem 0">
      <b id="collname"></b> <span id="colldense" class="note"></span>
      <input type="file" id="docfile" multiple hidden>
      <button id="adddocs" style="width:100%;margin:.4rem 0">+ Add files (txt, md, code, html, pdf, docx)</button>
      <div id="docmsg" class="note"></div>
      <div id="docs"></div>
      <label>Try a search</label><input id="ragq" placeholder="query, Enter to search"><div id="raghits" class="note"></div>
      <button id="dropcoll" style="margin-top:.6rem;width:100%;color:var(--no)">Delete collection</button>
    </div>
  </div>
  <div id="tab-chats">
    <button id="newchat" style="width:100%;margin-bottom:.5rem">+ New chat</button>
    <div id="chats"></div>
  </div>
  <div id="tab-hub" style="display:none">
    <input id="q" placeholder="search GGUF models on Hugging Face">
    <button id="go" style="width:100%;margin-top:.4rem">Search</button>
    <div id="hits" class="note">Search by model name, e.g. <a href="#" data-qs="qwen2.5 instruct">qwen2.5 instruct</a>, <a href="#" data-qs="llama 3.1 instruct">llama 3.1 instruct</a>, <a href="#" data-qs="gemma">gemma</a>, <a href="#" data-qs="phi">phi</a>, <a href="#" data-qs="mistral instruct">mistral instruct</a>. Each file then shows whether it fits this computer, and ★ marks the one to start with.</div>
    <div id="jobs"></div>
  </div>
  <div id="tab-local" style="display:none">
  <input id="filter" placeholder="filter models" style="margin-bottom:.5rem">
  <div id="list">scanning…</div>
  <div id="panel" style="display:none">
    <hr style="border:none;border-top:1px solid var(--line);margin:.7rem 0">
    <div id="mprof" style="margin-bottom:.4rem"></div>
    <label>Engine</label><select id="engine"><option value="llama">llama.cpp (llama-server)</option><option value="vllm">vLLM</option></select>
    <div class="row2">
      <div><label>GPU layers</label><input id="ngl" value="99"></div>
      <div><label>Context</label><input id="ctx" value="8192"></div>
    </div>
    <div class="row2">
      <div><label>Batch</label><input id="batch" value="2048"></div>
      <div><label>Threads (0=auto)</label><input id="threads" value="0"></div>
    </div>
    <div id="gpubox" style="display:none">
      <label>GPUs <span id="gpuhint" class="note"></span></label>
      <div class="row2">
        <div><label>Visible GPUs</label><input id="gpus" placeholder="all, or 0,1"></div>
        <div><label>Split</label><select id="split_mode"><option value="">default (layer)</option><option>layer</option><option>row</option><option>none</option></select></div>
      </div>
      <div class="row2">
        <div><label>Tensor split</label><input id="tensor_split" placeholder="1,1,1,1"></div>
        <div><label>Main GPU (-1 = auto)</label><input id="main_gpu" value="-1"></div>
      </div>
    </div>
    <div id="moebox" style="display:none">
      <label>Active experts <span id="moehint" class="note"></span></label>
      <input id="experts" value="0">
    </div>
    <label>Served name (API model id)</label><input id="served_name" placeholder="defaults to the model id">
    <label>LoRA adapter (optional)</label><input id="lora" placeholder="a LoRA .gguf file" data-pick="file" data-ext="gguf">
    <details id="eoBox" class="eo"><summary>Engine options <span id="eoCount" class="note"></span></summary>
      <div id="eoForm" class="note">loading…</div></details>
    <label>Extra flags</label><input id="extra" placeholder="--override-tensor '\.ffn_.*_exps\.=CPU'">
    <label id="visionrow" style="display:none"><input type="checkbox" id="vision" checked> load vision projector <span id="mmname"></span></label>
    <label>H-Neuron classifier (optional, enables activation stats)</label><input id="classifier" placeholder="models/classifier.npz for this model">
    <button id="load" class="pri" style="margin-top:.7rem;width:100%">Load</button>
    <div id="radarBox" style="display:none"><div class="row2" style="align-items:end"><div><label>Compare with (2P)</label><select id="vsPick"></select></div><div class="note" id="radarNote"></div></div><div id="radar"></div></div>
    <div id="statsbox"></div>
    <div id="quantbox" class="note"></div>
    <div id="loadmsg" class="note"></div>
  </div>
  </div>
</aside>
<section class="chat">
  <div id="ctxStrip" role="button" tabindex="0" aria-controls="side" aria-expanded="false" title="Environment, context usage and outputs of this chat">
    <span class="sm" id="stripModel">no model loaded</span><div class="cbar" id="stripBar" aria-hidden="true"></div><span id="stripPct"></span><span id="stripOut"></span><span class="chev" aria-hidden="true">▾</span></div>
  <div id="side" hidden aria-label="Environment, context and outputs">
    <section><h3>Environment <span class="r" id="envState"></span></h3><dl class="kv" id="envKv"></dl></section>
    <section><h3>Context usage <span class="r" id="ctxPct"></span></h3>
      <div class="cbar" id="ctxBar" role="img" aria-label="context usage"></div><div class="cleg" id="ctxLeg"></div><div class="note" id="ctxNote"></div>
      <label style="margin-top:.5rem"><input type="checkbox" id="cmpPass"> Trim document passages to the sentences that match the question</label>
      <label>When the chat no longer fits <select id="cmpHist" style="width:auto;margin-left:.3rem"><option value="off">do nothing</option><option value="summarize">summarize the oldest messages</option><option value="cut">drop the oldest messages</option></select></label></section>
    <section><h3>Outputs <span class="r" id="outN"></span></h3><div class="outs" id="outList"></div>
      <button id="outExport" style="margin-top:.4rem;font-size:12px;padding:.25rem .6rem">Export chat as .md</button></section>
  </div>
  <div id="log"></div>
  <div class="dials">
    <span>preset</span><select id="preset"></select>
    <span title="retrieve passages from a document collection for each message">docs</span><select id="ragPick"><option value="">none</option></select>
    <label id="toolsBox" style="display:none;margin:0" title="let the model call MCP tools (each call asks first)"><input type="checkbox" id="toolsOn"> tools</label>
    <label style="margin:0" title="after each reply, score every token with the model's hallucination classifier (needs studio --cett and a classifier in the model's settings)"><input type="checkbox" id="autoCheck"> check replies</label>
    <label style="margin:0" title="ask the engine for each token's top-5 probabilities and mark where the model was unsure (llama-server and vLLM; costs almost nothing)"><input type="checkbox" id="confOn"> confidence</label>
    <span>suppression α</span>
    <input type="range" id="alpha" min="0" max="1" step="0.05" value="1" disabled>
    <span id="av">1.00</span>
    <span id="adesc">no adapter loaded</span>
    <button id="export" style="margin-left:auto">Export .md</button>
  </div>
  <form id="f">
    <div id="pending" class="imgs"></div>
    <input type="file" id="file" accept="image/*" multiple hidden>
    <select id="modelPick" title="which model answers"></select><span id="pickWarn" class="warnico" style="display:none">⚠</span>
    <button type="button" id="attach" title="attach image (vision models)" disabled>📎</button>
    <textarea id="in" rows="2" placeholder="Message… (Enter to send, Shift+Enter for a new line)"></textarea>
    <button class="pri" id="send">Send</button>
  </form>
</section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let models=[], sel=null, busy=false, ctrl=null, status_={}, pendingImgs=[];
let chat={id:null,title:'New chat',messages:[]};
const F=["ngl","ctx","batch","threads","experts","served_name","lora","extra","classifier","gpus","split_mode","tensor_split","main_gpu"];
const TEXT_FIELDS=["served_name","lora","extra","classifier","gpus","split_mode","tensor_split"];
// ---------- engine options (scripts/engine_options.py: one table for the form, the flags and the optimizer)
let EO=null;
async function loadEO(engine, values){
  const r=await fetch('/api/engine/options?engine='+engine); EO=await r.json(); renderEO(values||{});
}
let EO_DEVS=[];
fetch('/api/engine/devices').then(r=>r.json()).then(j=>{EO_DEVS=j.devices||[]; if(EO_DEVS.length&&EO&&$('#eoForm').innerHTML) renderEO({...(sel&&sel.settings||{}),...readEO()})}).catch(()=>{});
const EO_DEV_KEYS=['device','draft_device'];
function eoInput(o,v){ const id='eo_'+o.key, dis=o.available?'':' disabled';
  if(EO_DEV_KEYS.includes(o.key)&&EO_DEVS.length) return `<div class="eoRow" title="${esc(o.help)}"><label>${esc(o.label)} <code>${esc(o.flag)}</code></label><input id="${id}" value="${esc(v??'')}" list="eoDevs_${o.key}" placeholder="engine default"${dis}><datalist id="eoDevs_${o.key}">${EO_DEVS.map(d=>`<option value="${esc(d.name)}">${esc(d.desc)} · ${d.free_mb} of ${d.total_mb} MiB free</option>`).join('')}</datalist>${o.help?`<div class="note">${esc(o.help)}</div>`:''}</div>`;
  if(o.kind==='bool') return `<label class="eoRow" title="${esc(o.help)}"><input type="checkbox" id="${id}"${v?' checked':''}${dis}> ${esc(o.label)} <code>${esc(o.flag)}</code></label>`;
  const ctl = o.kind==='choice' ? `<select id="${id}"${dis}><option value="">${o.default?esc(o.default)+' (default)':'engine default'}</option>${o.choices.map(c=>`<option${String(v)===c?' selected':''}>${esc(c)}</option>`).join('')}</select>`
    : `<input id="${id}" value="${esc(v??'')}" placeholder="${o.kind==='text'?'':'engine default'}"${dis}>`;
  return `<div class="eoRow" title="${esc(o.help)}"><label>${esc(o.label)} <code>${esc(o.flag)}</code></label>${ctl}${o.help?`<div class="note">${esc(o.help)}</div>`:''}</div>`; }
function renderEO(s){
  if(!EO) return; let n=0, html='';
  for(const [g,opts] of Object.entries(EO.groups)){
    const have=opts.filter(o=>o.available), lack=opts.filter(o=>!o.available);
    n+=opts.filter(o=>s[o.key]!==undefined&&s[o.key]!==''&&s[o.key]!==null&&s[o.key]!==false&&!(o.key==='flash_attn'&&s[o.key]==='auto')&&!(o.key==='parallel'&&+s[o.key]===1)).length;
    html+=`<details class="eoG"${opts.some(o=>s[o.key])?' open':''}><summary>${esc(g)}</summary>${have.map(o=>eoInput(o,s[o.key])).join('')}`+
      (g==='Speculative decoding'&&EO.engine==='llama'?`<div class="eoRow"><button type="button" onclick="suggestDraft()" title="Find models in your folders that share this model's vocabulary, and a device the big model does not use">Suggest a draft model and device</button><div id="draftOut" class="note"></div></div>`:'')+
      (lack.length?`<details><summary class="note">${lack.length} not in this ${EO.engine==='llama'?'llama-server':'vLLM'} build</summary>${lack.map(o=>eoInput(o,s[o.key])).join('')}<div class="note">Update the engine (Setup → Build llama.cpp) to use these.</div></details>`:'')+`</details>`;
  }
  const env=s.env||{};
  html+=`<details class="eoG"${Object.keys(env).length||s.wrap?' open':''}><summary>Environment and launch</summary>
    <div class="note">Only these documented variables can be set. ${Object.entries(EO.env).map(([k,h])=>`<div class="eoRow"><label>${esc(k)}</label><input data-env="${esc(k)}" value="${esc(env[k]??'')}" placeholder="unset"><div class="note">${esc(h)}</div></div>`).join('')}</div>
    <div class="eoRow"><label>Launch wrapper</label><select id="eo_wrap"><option value="">none</option>${Object.entries(EO.wrappers).map(([k,h])=>`<option value="${k}"${s.wrap===k?' selected':''}>${esc(h)}</option>`).join('')}</select></div>
    <div class="row2"><div><label>NUMA node</label><input id="eo_numa_node" value="${esc(s.numa_node??0)}"></div><div><label>taskset cores</label><input id="eo_taskset_cores" value="${esc(s.taskset_cores??'')}" placeholder="0,2,4,6"></div></div></details>`;
  if(EO.engine==='vllm') html=(EO.vllm_installed?'':'<div class="bad">vLLM is not installed here (pip install vllm), or set its command with studio --vllm.</div>')+
    `<div class="eoRow"><label>Tokenizer (GGUF only)</label><input id="eo_tokenizer" value="${esc(s.tokenizer??'')}" placeholder="the original HF model id"></div>`+html;
  $('#eoForm').innerHTML=html; $('#eoCount').textContent=n?`(${n} set)`:'';
}
async function suggestDraft(){
  const out=$('#draftOut'); if(!sel){out.textContent='Pick a model first.';return}
  out.textContent='Reading vocabularies…';
  const r=await (await fetch('/api/models/draft?model='+encodeURIComponent(sel.id))).json();
  if(r.error){out.innerHTML=`<span class="bad">${esc(r.error)}</span>`;return}
  if(!r.candidates.length){out.textContent='No model in your folders shares this one\'s vocabulary and is small enough (under 40% of its size). Download a smaller model of the same family, or try an n-gram method, which needs no draft.';return}
  const pl=r.placement||{};
  out.innerHTML=(pl.why?`<div>${esc(pl.why)}.</div>`:'')+r.candidates.map((c,i)=>`<div class="eoRow" style="display:flex;gap:.4rem;align-items:center"><button type="button" data-i="${i}">Use</button><span title="${esc(c.path)}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(c.id)}</span><span class="note">${(c.ratio*100).toFixed(0)}% of the size${c.same_arch?', same family':''}</span></div>`).join('');
  out.querySelectorAll('button[data-i]').forEach(b=>b.onclick=()=>{const c=r.candidates[+b.dataset.i],set=(k,v)=>{const el=$('#eo_'+k);if(el&&v!=null&&v!=='')el.value=v};
    set('draft_model',c.path);set('spec_type',$('#eo_spec_type')&&[...$('#eo_spec_type').options].some(o=>o.value==='draft-simple')?'draft-simple':'');
    set('draft_device',pl.draft_device);set('draft_ngl',pl.draft_ngl);if(pl.device)set('device',pl.device);
    b.textContent='✓ set'; out.insertAdjacentHTML('beforeend','<div>Filled in: save the settings and load the model to use it.</div>')});
}
function readEO(){ const s={};
  for(const opts of Object.values(EO?EO.groups:{})) for(const o of opts){ const el=$('#eo_'+o.key); if(!el) continue;
    if(o.kind==='bool'){ if(el.checked) s[o.key]=true; continue; }
    const v=el.value.trim(); if(v==='') continue; s[o.key]= o.kind==='int'?parseInt(v): o.kind==='float'?parseFloat(v): v; }
  s.env={}; $$('[data-env]').forEach(el=>{ if(el.value.trim()) s.env[el.dataset.env]=el.value.trim(); });
  s.wrap=$('#eo_wrap')?$('#eo_wrap').value:''; if(s.wrap){ s.numa_node=$('#eo_numa_node').value.trim(); s.taskset_cores=$('#eo_taskset_cores').value.trim(); }
  if($('#eo_tokenizer')&&$('#eo_tokenizer').value.trim()) s.tokenizer=$('#eo_tokenizer').value.trim();
  return s; }
$('#engine').onchange=()=>loadEO($('#engine').value, {...(sel&&sel.settings||{}), ...readEO()});
const pct=x=>x==null?'–':(100*x).toFixed(0)+'%';
const NOSTATS_HELP=' Auto routing skips it: it only picks models whose answers have been graded. You can still load it or pick it by name. To add stats, grade it with TestQA (Jobs → Evaluate, or the link under its settings).';
function statBadge(m){ const s=m.stats; if(!s) return '';
  if(!s.eligible) return `<span class="warnico" title="${esc(s.why_not+NOSTATS_HELP)}">⚠</span>`;
  return ''; }
function statTitle(s){ const g=s.graded; let t=`Rolling stats over the last ${g.n} graded answers:\n`+
  `correct ${pct(g.accuracy)}, hallucinated (answered wrong) ${pct(g.hallucination_rate)}, abstained ${pct(g.abstention_rate)}\n`;
  for(const [k,v] of Object.entries(s.subjects||{})) t+=`  ${k}: ${pct(v.accuracy)} right, ${pct(v.hallucination_rate)} wrong, ${pct(v.abstention_rate)} abstained (n ${v.n})\n`;
  if(s.activation&&s.activation.n) t+=`H-Neuron activation over ${s.activation.n} replies: mean ${s.activation.mean.toFixed(2)}, flagged ${pct(s.activation.flagged_rate)}\n`;
  if(s.live&&s.live.n) t+=`Live traffic: ${s.live.n} replies, ${pct(s.live.abstention_rate)} declined\n`;
  if(g.sources) t+=`Graded by: ${Object.entries(g.sources).map(([k,v])=>k+' '+v).join(', ')}\n`;
  const refs=Object.entries(s.reference||{}); if(refs.length) t+=`Published scores (reference only, not used by auto): `+refs.map(([k,v])=>`${k} ${pct(v.score)}`).join(', ');
  return t; }
function statLine(m){ const s=m.stats; if(!s||!s.graded.n) return '';
  const g=s.graded; return `<div class="stat" title="${esc(statTitle(s))}">✓ ${pct(g.accuracy)} · halluc ${pct(g.hallucination_rate)} · abst ${pct(g.abstention_rate)} · n ${g.n}</div>`; }
// A TestQA job for this model, filled in (Jobs page).
function gradeLink(m){ if(!m) return ''; const ep=`m=${location.origin}/v1@${m.id}`;
  const out='runs/testqa-'+m.id.replace(/[^A-Za-z0-9_.-]+/g,'_')+'.json';
  return `<a href="/jobs#kind=testqa&endpoint=${encodeURIComponent(ep)}&publish_stats=${encodeURIComponent(location.origin)}&out=${encodeURIComponent(out)}">Grade it with TestQA →</a>`; }
function statsTable(s){ if(!s) return '';
  if(!s.graded.n && !(s.activation&&s.activation.n) && !(s.live&&s.live.n)) return `<div class="note"><span class="warnico">⚠</span> ${esc(s.why_not+NOSTATS_HELP)} ${gradeLink(sel)}</div>`;
  let h='<table class="st"><tr><th>subject</th><th>n</th><th>right</th><th>halluc</th><th>abstain</th></tr>';
  const rows=[['all',s.graded],...Object.entries(s.subjects||{})];
  for(const [k,v] of rows) if(v.n) h+=`<tr><td>${esc(k)}</td><td>${v.n}</td><td>${pct(v.accuracy)}</td><td>${pct(v.hallucination_rate)}</td><td>${pct(v.abstention_rate)}</td></tr>`;
  h+='</table>';
  if(s.activation&&s.activation.n) h+=`<div class="note">H-Neuron activation: ${s.activation.n} replies, mean score ${s.activation.mean.toFixed(2)}, p95 ${s.activation.p95.toFixed(2)}, flagged ${pct(s.activation.flagged_rate)}</div>`;
  if(s.live&&s.live.n) h+=`<div class="note">Live: ${s.live.n} replies, ${pct(s.live.abstention_rate)} declined</div>`;
  if(!s.eligible) h+=`<div class="note"><span class="warnico">⚠</span> ${esc(s.why_not)}</div>`;
  return h; }
// ---------- radar (DDR-style): per-subject accuracy and honesty, 1P = this model, 2P = a comparison
function radarAxes(stats){ const out={}; if(!stats) return out;
  for(const [k,v] of Object.entries(stats.subjects||{})) if(v.n>=3) out[k]=v.accuracy;
  if(stats.graded&&stats.graded.n>=3) out['honesty']=1-(stats.graded.hallucination_rate||0);
  return out; }
function renderRadar(){ if(!sel) return; const a=radarAxes(sel.stats);
  const others=models.filter(m=>m!==sel&&Object.keys(radarAxes(m.stats)).length>=3);
  const keep=$('#vsPick').value;
  $('#vsPick').innerHTML='<option value="">nobody</option>'+others.map(m=>`<option value="${esc(m.path)}">${esc(m.id)}</option>`).join('');
  if(keep&&others.some(m=>m.path===keep)) $('#vsPick').value=keep;
  const vs=others.find(m=>m.path===$('#vsPick').value);
  const b=vs?radarAxes(vs.stats):{};
  const axes=[...new Set([...Object.keys(a),...Object.keys(b)])];
  $('#radarBox').style.display=Object.keys(a).length?'':'none';
  if(axes.length<3){ $('#radar').innerHTML=''; $('#radarNote').textContent='The radar needs results in at least 3 subjects (3+ graded items each).'; return; }
  $('#radarNote').textContent='accuracy per subject; honesty = 1 − hallucination rate';
  $('#radar').innerHTML=radarSVG(axes,[{name:sel.id,v:a},...(vs?[{name:vs.id,v:b}]:[])]); }
function radarSVG(axes,series){ const C=nsChart(), W=380, H=280, cx=190, cy=142, R=92, n=axes.length;
  const pt=(i,r)=>{ const t=-Math.PI/2+2*Math.PI*i/n; return [cx+r*Math.cos(t), cy+r*Math.sin(t)]; };
  const poly=r=>axes.map((_,i)=>pt(i,r).map(x=>x.toFixed(1)).join(',')).join(' ');
  let g=''; for(const f of [1,.75,.5,.25]) g+=`<polygon points="${poly(R*f)}" fill="${f===1?'var(--soft,rgba(127,127,127,.08))':'none'}" stroke="var(--line)" stroke-width="${f===1?2:1}"/>`;
  axes.forEach((_,i)=>{ const [x,y]=pt(i,R); g+=`<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--line)"/>`; });
  series.forEach((s,k)=>{ const c=C[k%C.length]; const pts=axes.map((ax,i)=>pt(i,R*Math.max(0,Math.min(1,s.v[ax]??0))).map(x=>x.toFixed(1)).join(',')).join(' ');
    g+=`<polygon points="${pts}" fill="${c}" fill-opacity="${k?0.22:0.32}" stroke="${c}" stroke-width="2.5" stroke-linejoin="round"/>`; });
  axes.forEach((ax,i)=>{ const [x,y]=pt(i,R+22); const anchor=Math.abs(x-cx)<8?'middle':(x>cx?'start':'end');
    const vals=series.map((s,k)=>s.v[ax]==null?'–':Math.round(s.v[ax]*100)).join(' / ');
    g+=`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" font-size="11" font-weight="700" fill="var(--fg)" style="text-transform:uppercase;letter-spacing:.06em">${esc(ax)}</text>`+
       `<text x="${x.toFixed(1)}" y="${(y+12).toFixed(1)}" text-anchor="${anchor}" font-size="10" fill="var(--mut)">${vals}</text>`; });
  const legend=series.map((s,k)=>{ const c=C[k%C.length]; const vals=Object.values(s.v); const avg=vals.length?vals.reduce((x,y)=>x+y,0)/vals.length:0;
    const seg=Array.from({length:10},(_,i)=>`<span style="display:inline-block;width:9px;height:12px;margin-right:2px;border-radius:2px;background:${i<Math.round(avg*10)?c:'var(--line)'}"></span>`).join('');
    return `<div style="display:flex;align-items:center;gap:.4rem;margin-top:.25rem"><span style="font:700 11px ui-monospace,monospace;color:${c};border:2px solid ${c};border-radius:5px;padding:0 .3rem">${k+1}P</span><span class="note" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(s.name)}</span>${seg}</div>`; }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:380px;display:block;margin:.3rem auto 0">${g}</svg>${legend}`; }
$('#vsPick').onchange=renderRadar;
window.addEventListener('ns-theme',renderRadar);
function human(b){const u=["B","KB","MB","GB","TB"];let i=0;while(b>=1024&&i<4){b/=1024;i++}return b.toFixed(1)+' '+u[i]}
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b??{})});

// theme
// Dark by default (set in <head>); the toggle remembers the choice.
$('#theme').onclick=async()=>{ const nx=await nsCycleTheme(); $('#theme').title='theme: '+nx+' (click for the next; all of them under Settings → Appearance)'; };
$('#theme').title='theme: '+nsThemeName()+' (click for the next; all of them under Settings → Appearance)';
$('#api').textContent=location.origin+'/v1';
$('#api').onclick=async()=>{try{await navigator.clipboard.writeText(location.origin+'/v1'); $('#api').textContent='copied';
  setTimeout(()=>$('#api').textContent=location.origin+'/v1',1200)}catch{}};

// tabs
function showTab(name){ $$('.tabs button').forEach(x=>x.classList.toggle('pri',x.dataset.tab===name));
  // The sidebar shows which tab is open.
  window.NS_TAB=name; document.dispatchEvent(new CustomEvent('ns-tab',{detail:name}));
  ['chats','local','hub','docs','tools'].forEach(t=>$('#tab-'+t).style.display=t===name?'block':'none'); }
$$('.tabs button').forEach(b=>b.onclick=()=>{ showTab(b.dataset.tab); history.replaceState(null,'','#'+(b.dataset.tab==='local'?'models':b.dataset.tab)); });
// Links from other pages: /#tools, /#models, /#hub, /#docs, /#chats, or
// /#model=<id>&field=classifier to open a model's settings at that field, /#hub=<org/name> to list a repo's files.
const TAB_ALIAS={models:'local',local:'local',chats:'chats',hub:'hub',docs:'docs',tools:'tools'};
function applyHash(){ const h=location.hash.slice(1); if(!h) return;
  if(TAB_ALIAS[h]) return showTab(TAB_ALIAS[h]);
  const q=new URLSearchParams(h), id=q.get('model');
  if(q.get('hub')){ showTab('hub'); $('#q').value=q.get('hub'); return files(q.get('hub')); }
  if(id){ showTab('local'); const i=models.findIndex(m=>m.id===id); if(i<0) return;
    pick(i); const f=q.get('field'); const el=f&&document.getElementById(f);
    if(el){ el.scrollIntoView({block:'center'}); el.focus(); el.style.outline='2px solid var(--acc)'; setTimeout(()=>el.style.outline='',2500); } } }
addEventListener('hashchange',applyHash);

// ---------- models
let linked=[];
async function refresh(){
  models=await (await fetch('/api/models')).json(); modelsLoaded=true;
  try{ linked=await (await fetch('/api/links/models')).json(); }catch{ linked=[]; }
  renderModels(); renderPicker();
  if(chat&&!chat.messages.length) emptyState();
}
function renderPicker(){
  const cur=$('#modelPick').value; const anyOk=models.some(m=>m.stats&&m.stats.eligible);
  let o=`<option value="">Loaded model</option><option value="auto"${anyOk?'':' disabled'}>Auto (by performance stats)${anyOk?'':' - no models have stats'}</option>`;
  o+=models.map(m=>`<option value="${esc(m.id)}">${m.stats&&!m.stats.eligible?'⚠ ':''}${esc(m.id)}${m.stats&&!m.stats.eligible?' (no stats)':''}</option>`).join('');
  if(linked.length) o+=`<optgroup label="linked hosts">`+linked.map(m=>`<option value="${esc(m.id)}">⇢ ${esc(m.id)}</option>`).join('')+`</optgroup>`;
  $('#modelPick').innerHTML=o; if([...$('#modelPick').options].some(x=>x.value===cur&&!x.disabled)) $('#modelPick').value=cur;
  pickChanged();
}
function pickChanged(){
  const m=models.find(x=>x.id===$('#modelPick').value); const w=$('#pickWarn');
  if(m&&m.stats&&!m.stats.eligible){ w.style.display='inline'; w.title=m.stats.why_not+' You picked it manually, so it will be used.'+NOSTATS_HELP; }
  else w.style.display='none';
  $('#attach').disabled=!( status_.vision || (m&&m.mmproj) || $('#modelPick').value==='auto');
}
$('#modelPick').onchange=pickChanged;
function renderModels(){
  const q=$('#filter').value.toLowerCase();
  const rows=models.map((m,i)=>[m,i]).filter(([m])=>!q||m.name.toLowerCase().includes(q)||(m.arch||'').includes(q));
  $('#list').innerHTML = rows.length ? rows.map(([m,i])=>{
    const f=m.fit||{}; const cls=f.ok===false?'bad':(/tight/.test(f.note||'')||f.level==='partial'?'warn':'');
    return `<div class="m${sel&&sel.path===m.path?' sel':''}" data-i="${i}">
      <div class="mn">${esc(m.name)}${statBadge(m)}</div>
      <div class="mm">${m.quant?`<span class="tag">${esc(m.quant)}</span>`:''}${m.arch?`<span class="tag">${esc(m.arch)}</span>`:''}
        ${m.n_experts?`<span class="tag">MoE ${m.n_experts}</span>`:''}${m.mmproj?'<span class="tag">vision</span>':''}
        <span class="tag">${human(m.size)}</span>${fitBadge(f)}</div>
      <div class="mm ${cls}">${esc(f.note||'')}</div>${statLine(m)}</div>`;
  }).join('') : '<div class="note">No .gguf files found under the configured directories.</div>';
  $$('#list .m').forEach(el=>el.onclick=()=>pick(+el.dataset.i));
}
$('#filter').oninput=renderModels;
// LM Studio-style: will it run here, and how.
function fitBadge(f){ const t={gpu:['okc','Full GPU offload'],partial:['warn','Partial GPU offload'],cpu:['','Runs on the CPU'],no:['bad','Likely too large']}[f&&f.level];
  return t?`<span class="tag ${t[0]}" title="${esc(f.note||'')}">${t[1]}</span>`:''; }
function pick(i){
  sel=models[i]; renderModels(); $('#panel').style.display='block'; modelProfiles();
  const s=sel.settings||{};
  F.forEach(k=>{ if($('#'+k)) $('#'+k).value = s[k] ?? ''; });
  $('#moebox').style.display = sel.n_experts?'block':'none';
  if(sel.n_experts) $('#moehint').textContent = `0 = model default (of ${sel.n_experts})`;
  $('#visionrow').style.display = sel.mmproj?'block':'none';
  $('#statsbox').innerHTML = statsTable(sel.stats); renderRadar(); quantBox();
  $('#vision').checked = s.vision!==false; $('#mmname').textContent = sel.mmproj?'('+sel.mmproj.split('/').pop()+')':'';
  const eng = s.engine || (sel.format==='hf'?'vllm':'llama'); $('#engine').value=eng;
  const fa = s.flash_attn===true?'on':s.flash_attn===false?'off':s.flash_attn;
  loadEO(eng, {...s, flash_attn: fa});
}
// Other quantizations of the same model, and which one was measured best on this computer.
async function quantBox(){ const m=sel, box=$('#quantbox'); box.innerHTML=''; if(!m||!m.family) return;
  const sib=models.filter(x=>x.family===m.family).sort((a,b)=>b.size-a.size);
  let ch=null; try{ ch=await (await fetch('/api/quant/choice?family='+encodeURIComponent(m.family))).json(); }catch{}
  if(sel!==m) return;
  const best=ch&&ch.here;
  const chips=sib.map(x=>`<button type="button" data-q="${esc(x.id)}" style="font-size:11.5px;padding:.1rem .45rem;margin:.1rem .15rem 0 0;${x.id===m.id?'border-color:var(--acc)':''}" title="${esc(x.path)}">${esc(x.quant||'?')} · ${human(x.size)}${best&&best.model===x.path?' ★':''}</button>`).join('');
  const jobs='/jobs#kind=quant_select&model='+encodeURIComponent(sib.map(x=>x.path).join(' '))+'&out='+encodeURIComponent('runs/qsel-'+m.family);
  box.innerHTML=`<b>Quantizations</b> ${sib.length>1?chips:''}<div>${best?`★ <b>${esc(best.quant)}</b> was measured the best fit for this computer (${best.speed&&best.speed.gen_tps?best.speed.gen_tps.toFixed(1)+' tok/s':''}${best.fidelity&&best.fidelity.kld!=null?', KLD '+best.fidelity.kld.toFixed(3):best.fidelity&&best.fidelity.agreement!=null?', '+Math.round(best.fidelity.agreement*100)+'% agreement':''}).`:''}
    ${sib.length>1?`<a href="${jobs}">Measure which suits this computer</a>`:`Only ${esc(m.quant||'this file')} is here. <a href="/jobs#kind=quant_select&from=${encodeURIComponent(m.path)}&make=Q8_0,Q6_K,Q4_K_M&out=${encodeURIComponent('runs/qsel-'+m.family)}">Make others and measure</a> (from a high-precision file works best).`}</div>`;
  box.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>{ const i=models.findIndex(x=>x.id===b.dataset.q); if(i>=0) pick(i); }); }
// The model settings form as values, and back: what Load sends, and what setting profiles keep.
function readModelForm(){ const s={}; F.forEach(k=>{ const el=$('#'+k); if(!el) return; s[k] = TEXT_FIELDS.includes(k) ? el.value : (parseFloat(el.value)||0); });
  if($('#main_gpu').value.trim()==='' || isNaN(parseFloat($('#main_gpu').value))) s.main_gpu=-1; else s.main_gpu=parseInt($('#main_gpu').value);
  s.lora_scale=1.0; s.vision=$('#vision').checked; s.engine=$('#engine').value; Object.assign(s, readEO()); return s; }
function fillModelForm(v){ F.forEach(k=>{ if($('#'+k)&&k in v) $('#'+k).value=v[k]??''; }); if('vision' in v) $('#vision').checked=v.vision!==false;
  const eng=v.engine||$('#engine').value; $('#engine').value=eng; const fa=v.flash_attn===true?'on':v.flash_attn===false?'off':v.flash_attn; loadEO(eng,{...v,flash_attn:fa}); }
const MODEL_DEFAULTS=__MODEL_DEFAULTS__;   // DEFAULTS in studio.py: what a model gets before anyone changes it
let MPROF=null;
function modelProfiles(){ if(MPROF) return; MPROF=nsProfiles($('#mprof'),{scope:'model-settings',get:readModelForm,set:fillModelForm,
  system:()=>fillModelForm({...MODEL_DEFAULTS})}); }
$('#load').onclick=async()=>{
  if(!sel) return;
  const s=readModelForm();
  $('#load').disabled=true; $('#loadmsg').textContent=s.engine==='vllm'?'starting vLLM…':'starting llama-server…';
  const r=await (await post('/api/load',{path:sel.path,settings:s})).json();
  $('#load').disabled=false; sel.settings=s;
  $('#loadmsg').innerHTML = (r.ok?'<span class="okc">loaded</span>':'<span class="bad">'+linkify(esc(r.message||r.error||'failed'))+'</span>')+
    (r.notes&&r.notes.length?'<div class="warn">'+r.notes.map(esc).join('<br>')+'</div>':'')+
    (r.args?`<details><summary class="note">command</summary><code style="word-break:break-all">${esc(r.args.join(' '))}</code></details>`:'');
  status();
};
$('#unload').onclick=async()=>{ await post('/api/unload'); status(); };
async function status(){
  const s=status_=await (await fetch('/api/status')).json();
  $('#dot').className='dot'+(s.running?' on':'');
  $('#st').textContent = s.running ? `${s.id||s.model}${s.vision?' · vision':''} · up ${s.uptime}s${s.idle_ttl?` · TTL ${s.idle_ttl}s`:''}` : 'no model loaded';
  $('#unload').disabled=!s.running; pickChanged(); sideEnv();
  if(s.running!==status.was){ status.was=s.running; if(chat&&!chat.messages.length) emptyState(); }
  const hasLora=!!s.lora; $('#alpha').disabled=!hasLora;
  $('#adesc').textContent = hasLora ? s.lora.split('/').pop() : 'no adapter loaded';
  if(hasLora && document.activeElement!==$('#alpha')){ $('#alpha').value=s.lora_scale; $('#av').textContent=(+s.lora_scale).toFixed(2); }
}
$('#alpha').oninput=async e=>{ const v=parseFloat(e.target.value); $('#av').textContent=v.toFixed(2); await post('/api/lora',{scale:v}); };

// ---------- chats
async function loadChats(){
  const cs=await (await fetch('/api/chats')).json();
  $('#chats').innerHTML = cs.length ? cs.map(c=>`<div class="m${c.id===chat.id?' sel':''}" data-id="${esc(c.id)}">
    <button class="x" data-del="${esc(c.id)}" title="delete">×</button>
    <div class="mn">${esc(c.title||'Untitled')}</div>
    <div class="mm">${c.n} messages · ${new Date(c.updated*1000).toLocaleString()}</div></div>`).join('')
    : '<div class="note">Conversations are saved here automatically.</div>';
  $$('#chats .m').forEach(el=>el.onclick=async e=>{
    if(e.target.dataset.del){ e.stopPropagation(); if(confirm('Delete this chat?')){ await post('/api/chats/delete',{id:e.target.dataset.del});
      if(chat.id===e.target.dataset.del) newChat(); loadChats(); } return; }
    chat=await (await fetch('/api/chats/'+el.dataset.id)).json(); renderChat(); loadChats(); });
}
function newChat(){ chat={id:null,title:'New chat',messages:[]}; renderChat(); loadChats(); }
$('#newchat').onclick=newChat;
async function persist(){
  if(!chat.messages.length) return;
  if(chat.title==='New chat'){ const first=chat.messages.find(m=>m.role==='user'); chat.title=textOf(first?.content).slice(0,60)||'New chat'; }
  chat.model=status_.id||null; chat.preset=$('#preset').value;
  const r=await (await post('/api/chats',chat)).json(); chat.id=r.id; loadChats();
}
const textOf=c=>Array.isArray(c)?c.filter(x=>x.type==='text').map(x=>x.text).join(' '):String(c||'');
const imagesOf=c=>Array.isArray(c)?c.filter(x=>x.type==='image_url').map(x=>x.image_url.url):[];

// minimal, safe markdown: fenced code blocks and inline code only
function render(text){
  const parts=String(text).split(/```/); let out='';
  parts.forEach((p,i)=>{ if(i%2){ const nl=p.indexOf('\n'); out+='<pre>'+esc(nl>=0?p.slice(nl+1):p)+'</pre>'; }
    else out+=esc(p).replace(/`([^`\n]+)`/g,'<code>$1</code>'); });
  return out;
}
// ---------- explain: one entry point to every way of looking inside a reply (scripts/explain.py)
async function explainReply(d){
  let box=d.querySelector('.explain'); if(box){ box.remove(); return; }
  box=document.createElement('div'); box.className='explain note'; box.textContent='finding what can run here…'; d.appendChild(box);
  const prompt=_last_user_prompt(d), reply=(d.querySelector('.body')||{}).textContent||'';
  const model=(status_&&(status_.id||status_.model))||'';
  try{ const r=await (await fetch('/api/explain/plan?'+new URLSearchParams({model,prompt:prompt.slice(-1500),reply:reply.slice(0,500)}))).json();
    if(r.error){ box.innerHTML=`<span class="bad">${esc(r.error)}</span>`; return; }
    box.innerHTML=`<div style="margin:.3rem 0"><b>Explain this reply</b> · ${esc(r.model)}</div>`+r.plan.map(p=>`<div class="exrow" style="display:flex;gap:.5rem;align-items:baseline;margin:.15rem 0;${p.available?'':'opacity:.55'}">
      ${p.available?(p.action.type==='check'?`<button data-ex="check">${esc(p.question)}</button>`:`<a href="${esc(p.action.href)}" target="_blank" rel="noopener">${esc(p.question)}</a>`):`<span>${esc(p.question)}</span>`}
      ${p.recommended?'<b class="okc" title="the first that can run here">start here</b>':''}
      <span title="${esc(p.method+' · cost: '+p.cost)}">${esc(p.available?p.method:p.why)}</span></div>`).join('');
    const b=box.querySelector('[data-ex=check]'); if(b) b.onclick=()=>checkReply(d);
  }catch(e){ box.innerHTML=`<span class="bad">${esc(String(e))}</span>`; } }
function _last_user_prompt(d){ const all=[...document.querySelectorAll('#log .msg')]; const i=all.indexOf(d);
  for(let j=i-1;j>=0;j--){ if(((all[j].querySelector('.who')||{}).textContent||'').startsWith('you')) return (all[j].querySelector('.body')||all[j]).textContent||''; } return ''; }
function bubble(role,content,extra={}){
  if(!$('#log .msg')) $('#log').innerHTML='';
  const d=document.createElement('div'); d.className='msg';
  d.innerHTML=`<div class="who">${role==='user'?'you':'assistant'}<span class="acts"></span></div>`+
    (extra.think!==undefined?`<details class="think"><summary>reasoning</summary><div></div></details>`:'')+
    `<div class="imgs"></div><div class="tools"></div><div class="body"></div><div class="src"></div><div class="stats"></div><div class="risk"></div><div class="pchk"></div>`;
  d.querySelector('.imgs').innerHTML=imagesOf(content).map(u=>`<img src="${esc(u)}" alt="attached image">`).join('');
  d.querySelector('.body').innerHTML=render(textOf(content));
  if(extra.think) d.querySelector('.think div').textContent=extra.think;
  if(extra.stats) d.querySelector('.stats').textContent=extra.stats;
  if(extra.sources) showSources(d.querySelector('.src'),extra.sources);
  for(const t of extra.tools||[]) toolCard(d.querySelector('.tools'),t);
  if(role==='assistant'){ const a=d.querySelector('.acts');
    a.innerHTML='<button data-a="copy">copy</button><button data-a="regen">regenerate</button><button data-a="check" title="score each token of this reply for hallucination risk">check</button><button data-a="explain" title="every way to look inside this reply: which can run here, and what each answers">explain</button><button data-a="fix" title="Fix this mistake: write what it should have said; the pair is kept for training (Settings → Integrated options)">fix</button>';
    a.querySelector('[data-a=copy]').onclick=()=>navigator.clipboard.writeText(d.querySelector('.body').textContent);
    a.querySelector('[data-a=regen]').onclick=regenerate;
    a.querySelector('[data-a=check]').onclick=()=>checkReply(d);
    a.querySelector('[data-a=explain]').onclick=()=>explainReply(d);
    a.querySelector('[data-a=fix]').onclick=()=>fixMistake(d);
    if(PLUGIN_CHECKS.length){ const b=document.createElement('button'); b.textContent='plugins'; b.title='run the checks from your plugins: '+PLUGIN_CHECKS.map(c=>c.title).join(', '); b.onclick=()=>pluginChecks(d,false); a.appendChild(b); }
    if(extra.check) showRisk(d.querySelector('.risk'),extra.check);
    if(extra.plugin_checks) showPluginChecks(d,extra.plugin_checks);
    if(extra.confidence) showConfidence(d,extra.confidence); }
  $('#log').appendChild(d); $('#log').scrollTop=1e9;
  return {root:d, think:d.querySelector('.think div'), thinkBox:d.querySelector('.think'), body:d.querySelector('.body'), stats:d.querySelector('.stats'), src:d.querySelector('.src'), tools:d.querySelector('.tools')};
}
function toolCard(box,t){
  let el=box.querySelector(`[data-k="${t.key}"]`);
  if(!el){ el=document.createElement('div'); el.className='tool'; el.dataset.k=t.key; box.appendChild(el); }
  const args=typeof t.arguments==='string'?t.arguments:JSON.stringify(t.arguments,null,1);
  const state=t.result?(t.result.ok?'<span class="okc">done</span>':'<span class="bad">failed</span>'):(t.needs_approval&&!t.decided?'<span class="warn">waiting for you</span>':'<span>running…</span>');
  // what the call would do (Settings → Permissions): the command line, paths, programs, hosts, each with its answer
  const acts=(t.checks||[]).filter(c=>c.action!=='tool');
  const will=acts.length?`<div class="will">${acts.map(c=>`<div class="wl ${c.effect}"><span class="tag">${esc(c.action)}</span><code>${esc(c.target)}</code><span class="note" title="${esc(c.why)}">${esc(c.effect)}</span></div>`).join('')}</div>`:'';
  const waiting=t.needs_approval&&!t.decided&&!t.result;
  el.innerHTML=`<span class="tn">🔧 ${esc(t.name)}</span> ${state}${waiting?will:''}<details ${waiting&&!acts.length?'open':''}><summary>${!waiting&&acts.length?'what it does, ':''}arguments</summary>${waiting?'':will}<pre>${esc(args)}</pre></details>`+
    (waiting?`<div class="qp"><button class="pri" data-a="1" title="just this time">Allow once</button><button data-q="always" title="this exact call stops asking for this model (Settings → Permissions, Tool approval)">Always allow</button><button class="bad" data-a="0">Deny</button>
      <select data-r title="a broader standing rule (Settings → Tool approval)"><option value="">more…</option>
      <option value="tool">always allow this tool, any arguments</option><option value="args">always allow these exact arguments</option>
      <option value="deny">always deny this tool</option></select></div>`:'')+
    (!t.needs_approval&&t.decision&&t.decision.by!=='default'?`<div class="note">${esc(t.decision.action==='deny'?'blocked':'allowed')} by ${esc(t.decision.reason)} · <a href="/settings#tools">rules</a></div>`:'')+
    (t.result?`<details><summary>result</summary><pre>${esc(t.result.text)}</pre></details>`:'');
  el.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{ t.decided=true; toolCard(box,t); await post('/api/tools/approve',{key:t.key,allow:b.dataset.a==='1'}); });
  el.querySelectorAll('button[data-q]').forEach(b=>b.onclick=async()=>{ t.decided=true; toolCard(box,t);
    const r=await post('/api/tools/approve',{key:t.key,allow:true,remember:'always'}); if(!r.ok){ try{ alert((await r.json()).error) }catch{} } });
  const rs=el.querySelector('select[data-r]'); if(rs) rs.onchange=async()=>{ const v=rs.value; if(!v) return; t.decided=true; toolCard(box,t);
    const r=await post('/api/tools/approve',{key:t.key,allow:v!=='deny',remember:v}); if(!r.ok){ try{ alert((await r.json()).error) }catch{} } };
}
function showSources(el,src){ el.innerHTML='sources: '+src.map((h,i)=>`<details><summary>[${i+1}] ${esc(h.source)}${h.page?' p.'+h.page:''}</summary><pre>${esc(h.text)}</pre></details>`).join(''); }
// What an empty chat shows: where you are in getting started, and something to ask.
let modelsLoaded=false;
const STARTERS=['Explain how a transformer decides the next word, in plain language.','Write a Python function that checks whether a string is a palindrome.',
  'What were the main causes of the French Revolution?','Plan a three-day trip to Lisbon on a small budget.'];
async function emptyState(){ const el=$('#log');
  el.innerHTML='<div class="empty"><h2>Start a conversation</h2></div>';
  let ready=null; try{ const r=await fetch('api/setup'); if(r.ok) ready=(await r.json()).ready; }catch{}
  if(chat.messages.length) return;
  const nModels=models.length, running=!!(status_&&status_.running);
  const step=(done,title,body)=>`<li class="${done?'done':''}"><span class="ck" aria-hidden="true">${done?'✓':''}</span><div><b>${title}</b><div class="note" style="margin:.1rem 0 0">${body}</div></div></li>`;
  const steps=[
    ready?step(ready.server,'The engine (llama.cpp)',ready.server?'Ready.':'Not found yet. <a href="/setup#build">Setup → Build llama.cpp</a> makes it with one button, or point to an existing llama-server.'):'',
    !modelsLoaded&&!nModels?step(false,'A model','Looking in your model folders…'):step(nModels>0,'A model',nModels?`${nModels} model${nModels>1?'s':''} in <a href="#models">My Models</a>.`:'<a href="#hub">Discover</a> finds and downloads one (each file says whether it fits this computer), or <a href="/setup#paths">add a folder</a> that already has .gguf files.'),
    step(running,'Load it',running?`${esc(status_.id||status_.model)} is loaded.`:'Pick one in <a href="#models">My Models</a>, or just send a message: with the model picker on <i>auto</i>, Studio loads the best one for the question.'),
  ].join('');
  const allSet=ready&&ready.server&&nModels>0&&running;
  const pair=models.slice(0,2).map(m=>`${m.id.split('/').pop().replace(/[^\w.-]/g,'-').slice(0,24)}=${location.origin}/v1@${m.id}`);
  // the most common things people come to do, the ones this person uses most first
  const TASKS=[
    ['Find a model that fits this computer','Each file says whether it fits in memory, and how fast it should run','#hub'],
    ['Compare models on your own questions','A graded TestQA run: accuracy, confident mistakes and declines per subject','/jobs#'+new URLSearchParams({kind:'testqa',endpoint:pair.join(' '),out:'runs/compare.json'}).toString()],
    ['Use these models in another app','Claude Desktop, Cline, or any OpenAI client, pointed at this Studio','/connect'],
    ['See what a reply is made of','Features, emotions and the logit lens over any reply','/features#read'],
    ['Split a project across models','Tasks planned, assigned to the right models, checked and reviewed','/projects'],
    ['Where the work went','Tokens, speed and hardware per device, and predicted speed for every model','/work']];
  const use=typeof nsUsage==='function'?nsUsage():{};
  const used=h=>use[h.split('?')[0].split('#')[0]+(h.startsWith('#')?h:'')]||use[h.split('#')[0]]||0;
  const tasks=TASKS.map((t,i)=>[t,i]).sort((a,b)=>used(b[0][2])-used(a[0][2])||a[1]-b[1]).map(x=>x[0]);
  const tasksHtml=`<div class="note" style="margin-top:1rem">Or:</div><div class="tasks">${tasks.map(([t,d,h])=>`<a class="task" href="${esc(h)}" title="${esc(d)}"><b>${esc(t)}</b><span>${esc(d)}</span></a>`).join('')}</div>`;
  el.innerHTML=`<div class="empty" style="text-align:left;max-width:${allSet?640:560}px"><h2 style="text-align:center">Start a conversation</h2>
    ${allSet?`<div class="note" style="text-align:center">✓ engine · ✓ ${nModels} model${nModels>1?'s':''} · ✓ ${esc(status_.id||status_.model)} loaded</div>`:`<ol class="gs">${steps}</ol>`}
    <div class="note" style="margin-top:1rem">Try one:</div><div class="starters">${STARTERS.map((q,i)=>`<button type="button" data-st="${i}">${esc(q)}</button>`).join('')}</div>
    ${allSet?tasksHtml:`<div class="note" style="margin-top:1rem">Other apps can use these models too: point any OpenAI client at <code>${esc(location.origin)}/v1</code>, or see <a href="/connect">Connect</a>. New here? The <b>?</b> button has a two-minute tour.</div>`}</div>`;
  el.querySelectorAll('[data-st]').forEach(b=>b.onclick=()=>{ $('#in').value=STARTERS[+b.dataset.st]; $('#in').focus(); sideSoon(); }); }
function renderChat(){
  sideSoon(); $('#log').innerHTML='';
  if(!chat.messages.length){ emptyState(); return; }
  for(const m of chat.messages){ const b=bubble(m.role,m.content,{think:m.think,stats:m.stats,sources:m.sources,tools:m.tools,check:m.check,plugin_checks:m.plugin_checks,confidence:m.confidence}); b.root._msg=m; }
}
// ---------- per-reply check: token risk from the activation classifier
function showRisk(el,c){
  if(c.error){ el.innerHTML=`<div class="sum bad">check failed: ${linkify(esc(c.error))}${c.links&&c.links.length?' · '+linkList(c.links):''}</div>`; return; }
  const n=c.flagged.length, T=c.prob.length;
  const verdict=n?`<b class="bad">${n} of ${T} tokens flagged</b>`:`<b class="okc">no tokens flagged</b>`;
  el.innerHTML=`<div class="sum">risk: ${verdict} · peak ${(c.max*100).toFixed(0)}% · mean ${(c.mean*100).toFixed(0)}% `+
    `(flag at ${(c.threshold*100).toFixed(0)}%) · <a href="${esc(c.url)}" target="_blank" rel="noopener">open 3D view ↗</a>`+
    (c.traced_on?` · <span title="where the activations were computed">traced on ${esc(c.traced_on)}</span>`:'')+
    ` · <a href="#" data-a="tog">${n?'hide':'show'} tokens</a>`+
    (c.review?` · <span class="lbl">was it right? <a href="#" data-v="correct">right</a> / <a href="#" data-v="wrong">wrong</a>${c.label?` <b>(${esc(c.label==='correct'?'right':c.label)})</b>`:''}</span>`:'')+
    `</div><div class="rt"${n?'':' hidden'}></div>`;
  // A person's verdict makes the reply count in the Review page's right-vs-wrong statistics.
  el.querySelectorAll('[data-v]').forEach(a=>a.onclick=async e=>{ e.preventDefault();
    const r=await post('/api/review/label',{model:c.review.model,id:c.review.id,verdict:a.dataset.v});
    if(r.ok){ c.label=a.dataset.v; const m=el.closest('.msg')?._msg; if(m&&m.check){ m.check.label=c.label; await persist(); } showRisk(el,c); }
    else alert((await r.json()).error||r.statusText); });
  // Shade each token by its risk; underline the ones over the threshold.
  // Shading starts at half the threshold, so tokens the classifier calls clean stay unshaded.
  el.querySelector('.rt').innerHTML=c.pieces.map((p,i)=>{ const r=c.prob[i]||0, a=Math.max(0,Math.min(1,(r-c.threshold/2)/(c.threshold/2)));
    return `<span class="${r>=c.threshold?'fl':''}" title="risk ${(r*100).toFixed(0)}%" style="background:color-mix(in srgb,var(--no) ${Math.round(a*(r>=c.threshold?50:30))}%,transparent)">${esc(p)}</span>`; }).join('');
  el.querySelector('[data-a=tog]').onclick=e=>{ e.preventDefault(); const rt=el.querySelector('.rt'); rt.hidden=!rt.hidden; e.target.textContent=(rt.hidden?'show':'hide')+' tokens'; };
}
// ---------- plugin checks (Settings -> Plugins)
let PLUGIN_CHECKS=[];
fetch('/api/plugins').then(r=>r.ok?r.json():{checks:[]}).then(j=>{ PLUGIN_CHECKS=j.checks||[]; }).catch(()=>{});
function showPluginChecks(d,res){ const el=d.querySelector('.pchk');
  el.innerHTML=res.map((c,i)=>`<div class="sum">${esc(c.title)}: ${c.error?'<span class="bad">'+esc(c.error)+'</span>':`<b class="${c.flag?'bad':''}">${esc(c.label)}</b>${c.note?' · '+esc(c.note):''}${c.spans&&c.spans.length?` · <a href="#" data-pc="${i}">show</a><div class="rt" hidden></div>`:''}`}</div>`).join('');
  el.querySelectorAll('[data-pc]').forEach(a=>a.onclick=e=>{ e.preventDefault(); const c=res[+a.dataset.pc], box=a.nextElementSibling, text=d._msg?textOf(d._msg.content):d.querySelector('.body').textContent;
    if(box.hidden&&!box.innerHTML){ let h='',i=0; for(const [x,y,n] of c.spans){ if(x<i) continue; h+=esc(text.slice(i,x))+`<mark title="${esc(n)}">${esc(text.slice(x,y))}</mark>`; i=y; } box.innerHTML=h+esc(text.slice(i)); }
    box.hidden=!box.hidden; a.textContent=box.hidden?'show':'hide'; }); }
async function pluginChecks(d,autoOnly){
  const m=d._msg; if(!m) return; const list=PLUGIN_CHECKS.filter(c=>!autoOnly||c.auto); if(!list.length) return;
  const idx=chat.messages.indexOf(m), el=d.querySelector('.pchk'); el.innerHTML='<div class="sum">running plugin checks…</div>';
  const body={model:m.model,messages:chat.messages.slice(0,idx).map(x=>({role:x.role,content:textOf(x.content)})),text:textOf(m.content)};
  const res=await Promise.all(list.map(async c=>{ try{ const r=await post('/api/plugins/check',{...body,plugin:c.plugin,check:c.name}); const j=await r.json(); return r.ok?j:{title:c.title,error:j.error}; }catch(e){ return {title:c.title,error:String(e)}; } }));
  showPluginChecks(d,res); m.plugin_checks=res.filter(x=>!x.error); await persist(); }
// ---------- confidence from the engine's top-5 probabilities per token
// u = entropy of the top 5 plus the rest, over log(6): 0 = certain, 1 = spread evenly.
// A token is uncertain when u >= CONF_U or the chosen token had p < CONF_P.
const CONF_U=0.55, CONF_P=0.35;
function confidenceOf(lp){
  const pieces=[], u=[], p=[];
  for(const t of lp){ const tops=(t.top_logprobs||[]).map(x=>Math.exp(x.logprob)).filter(x=>isFinite(x));
    let sum=0,h=0; for(const q of tops){ sum+=q; if(q>0) h-=q*Math.log(q); }
    const rest=Math.max(0,1-sum); if(rest>1e-9) h-=rest*Math.log(rest);
    pieces.push(t.token??''); u.push(+(h/Math.log(6)).toFixed(3)); p.push(+Math.exp(t.logprob??0).toFixed(3)); }
  const flagged=u.map((x,i)=>x>=CONF_U||p[i]<CONF_P?i:-1).filter(i=>i>=0);
  return {pieces,u,p,flagged,mean:+(u.reduce((a,b)=>a+b,0)/Math.max(1,u.length)).toFixed(3)}; }
function showConfidence(d,c){ let el=d.querySelector('.conf'); if(!el){ el=document.createElement('div'); el.className='conf risk'; d.querySelector('.risk').before(el); }
  const n=c.flagged.length, T=c.u.length, low=Math.min(...c.p);
  el.innerHTML=`<div class="sum">confidence: ${n?`<b class="bad">${n} of ${T} tokens uncertain</b>`:'<b class="okc">no uncertain tokens</b>'} · mean spread ${(c.mean*100).toFixed(0)}% · lowest chosen-token p ${(low*100).toFixed(0)}% · <a href="#" data-a="ctog">${n?'show':'show'} tokens</a></div><div class="rt" hidden></div>`;
  const set=new Set(c.flagged);
  el.querySelector('[data-a=ctog]').onclick=e=>{ e.preventDefault(); const rt=el.querySelector('.rt');
    if(!rt.innerHTML) rt.innerHTML=c.pieces.map((t,i)=>`<span class="${set.has(i)?'fl':''}" title="spread ${(c.u[i]*100).toFixed(0)}% · p ${(c.p[i]*100).toFixed(0)}%" style="background:color-mix(in srgb,var(--warn,#a8701c) ${Math.round(Math.min(1,c.u[i]/CONF_U)*(set.has(i)?45:20))}%,transparent)">${esc(t)}</span>`).join('');
    rt.hidden=!rt.hidden; e.target.textContent=(rt.hidden?'show':'hide')+' tokens'; }; }
// Fix This Mistake: the correction and the reply become a training pair (scripts/harvest.py)
function fixMistake(d){
  const box=d.querySelector('.pchk'); if(box.querySelector('.fixbox')) return box.querySelector('.fixbox textarea').focus();
  const msg=d._msg; const reply=msg?textOf(msg.content):d.querySelector('.body').textContent;
  const f=document.createElement('div'); f.className='fixbox note';
  f.innerHTML=`<b>Fix this mistake</b> · write the reply it should have given. The pair (this reply rejected, yours chosen) is kept for training.
    <textarea rows="6" style="width:100%;margin:.3rem 0"></textarea><input placeholder="what was wrong (optional)" style="width:100%">
    <div style="margin-top:.3rem"><button data-f="save">Save the pair</button> <button data-f="cancel">Cancel</button> <span class="k"></span></div>`;
  f.querySelector('textarea').value=reply; box.appendChild(f); f.querySelector('textarea').focus();
  f.querySelector('[data-f=cancel]').onclick=()=>f.remove();
  f.querySelector('[data-f=save]').onclick=async()=>{
    const i=chat.messages.indexOf(msg); const ctx=(i>=0?chat.messages.slice(0,i):chat.messages).map(m=>({role:m.role,content:textOf(m.content)}));
    const r=await fetch('/api/harvest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:(msg&&msg.model)||$('#model')?.value||'',
      messages:ctx,reply,correction:f.querySelector('textarea').value,note:f.querySelector('input').value})});
    const j=await r.json(); const k=f.querySelector('.k');
    if(!r.ok){ k.textContent=j.error||'not saved'; k.className='k bad'; return; }
    f.innerHTML=`✓ kept for training (${j.pending} pair${j.pending===1?'':'s'} waiting${j.auto_train?`; a micro-tune runs after ${j.min_pairs}, when idle`:''}). <a href="/settings#options">Settings</a>`; };
}
async function checkReply(d){
  const m=d._msg; if(!m) return;
  const idx=chat.messages.indexOf(m), el=d.querySelector('.risk');
  el.innerHTML='<div class="sum">checking… (one extra pass over the reply)</div>';
  let c;
  try{ const r=await post('/api/trace',{model:m.model||$('#modelPick').value||undefined,messages:chat.messages.slice(0,idx).map(x=>({role:x.role,content:textOf(x.content)})),text:textOf(m.content)});
    c=await r.json(); if(!r.ok) c={error:c.error||r.statusText,links:c.links}; }catch(e){ c={error:String(e)}; }
  showRisk(el,c);
  if(!c.error){ m.check={id:c.id,pieces:c.pieces,prob:c.prob,flagged:c.flagged,threshold:c.threshold,max:c.max,mean:c.mean,url:c.url,review:c.review,traced_on:c.traced_on}; await persist(); }
}
try{ $('#autoCheck').checked=localStorage.getItem('ns_autocheck')==='1'; }catch{}
$('#autoCheck').onchange=()=>{ try{ localStorage.setItem('ns_autocheck',$('#autoCheck').checked?'1':'0'); }catch{} };
try{ $('#confOn').checked=localStorage.getItem('ns_conf')!=='0'; }catch{ $('#confOn').checked=true; }
$('#confOn').onchange=()=>{ try{ localStorage.setItem('ns_conf',$('#confOn').checked?'1':'0'); }catch{} };

// ---------- images
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=async()=>{ for(const f of $('#file').files){ if(f.size>20e6){ alert(f.name+' is larger than 20 MB'); continue; }
    pendingImgs.push(await new Promise(r=>{const fr=new FileReader(); fr.onload=()=>r(fr.result); fr.readAsDataURL(f);})); }
  $('#file').value=''; drawPending(); };
function drawPending(){ $('#pending').innerHTML=pendingImgs.map((u,i)=>`<img src="${u}" title="click to remove" data-i="${i}">`).join('');
  $$('#pending img').forEach(im=>im.onclick=()=>{pendingImgs.splice(+im.dataset.i,1); drawPending();}); }

// ---------- send / stream
$('#in').onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); $('#f').requestSubmit(); } };
$('#f').onsubmit=async e=>{
  e.preventDefault();
  if(busy){ ctrl?.abort(); return; }
  const text=$('#in').value.trim(); if(!text&&!pendingImgs.length) return;
  const content = pendingImgs.length ? [{type:'text',text}, ...pendingImgs.map(u=>({type:'image_url',image_url:{url:u}}))] : text;
  $('#in').value=''; pendingImgs=[]; drawPending();
  chat.messages.push({role:'user',content}); bubble('user',content);
  await stream();
};
async function regenerate(){ if(busy) return; while(chat.messages.length && chat.messages.at(-1).role==='assistant') chat.messages.pop(); renderChat(); await stream(); }
async function stream(){
  busy=true; $('#send').textContent='Stop'; ctrl=new AbortController();
  const out=bubble('assistant','',{think:''}); const lp=[]; let acc='', resets=0, think='', t0=performance.now(), tFirst=0, usage=null, timings=null, routeNote='', sources=null, tools=[], servedBy=null, compressed=null;
  try{
    const r=await fetch('/api/chat',{method:'POST',signal:ctrl.signal,headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:chat.messages.map(m=>({role:m.role,content:m.content})),preset:$('#preset').value,model:$('#modelPick').value,
        rag:$('#ragPick').value?{collection:$('#ragPick').value,k:4}:undefined, tools:$('#toolsOn').checked||undefined,
        confidence:$('#confOn').checked||undefined, ...cmpReq()})});
    if(!r.ok){ let e=await r.text(); try{e=JSON.parse(e).error||e}catch{} e=typeof e==='string'?e:(e.message||JSON.stringify(e)); out.body.innerHTML='error: '+linkify(esc(e)); throw 0; }
    try{ const rt=JSON.parse(r.headers.get('X-NeuronScope-Route')||'null');
      if(rt) servedBy=rt.model;
      if(rt){ routeNote = rt.mode==='auto' ? `auto → ${rt.model}: ${rt.reason}` : `model: ${rt.model}${rt.mode==='manual'?' (chosen manually)':''}`;
        const m=models.find(x=>x.id===rt.model); if(m&&m.stats&&!m.stats.eligible) routeNote+=' ⚠ no performance stats'; } }catch{}
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let idx;
      while((idx=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        const d=line.slice(5).trim(); if(d==='[DONE]') continue;
        let j; try{ j=JSON.parse(d) }catch{ continue }
        if(j.error){ out.body.innerHTML='error: '+linkify(esc(j.error.message||j.error)); continue; }
        if(j.sources){ sources=j.sources; showSources(out.src,sources); continue; }
        if(j.compressed){ compressed=j.compressed; const h=j.compressed.history;
          if(h&&h.history==='summarized') chat.summary={upto:h.upto,text:h.summary};
          out.stats.textContent=cmpNote(j.compressed); continue; }
        if(j.neural_reset){ acc=acc.slice(0,Math.max(0,acc.length-j.neural_reset.drop)); resets++;
          out.body.innerHTML=render(acc); out.stats.textContent=`↺ Neural Reset: a ${j.neural_reset.repeats}× loop was cut and the reply resumed`; continue; }
        if(j.tool_call){ tools.push(j.tool_call); toolCard(out.tools,j.tool_call); $('#log').scrollTop=1e9; continue; }
        if(j.tool_result){ const t=tools.find(x=>x.key===j.tool_result.key); if(t){ t.result=j.tool_result; t.decided=true; toolCard(out.tools,t); } continue; }
        if(j.usage) usage=j.usage; if(j.timings) timings=j.timings;
        const delta=j.choices?.[0]?.delta||{};
        if((delta.reasoning_content||delta.content) && !tFirst) tFirst=performance.now();
        if(delta.reasoning_content){ think+=delta.reasoning_content; out.think.textContent=think; }
        if(delta.content){ acc+=delta.content; out.body.innerHTML=render(acc); }
        for(const t of j.choices?.[0]?.logprobs?.content||[]) lp.push(t);
        $('#log').scrollTop=1e9;
      }
    }
  }catch(err){ if(err && err.name==='AbortError') out.body.innerHTML=render(acc+'\n[stopped]'); else if(err) out.body.textContent='error: '+err; }
  if(!think) out.thinkBox.remove();
  const secs=(performance.now()-(tFirst||t0))/1000;
  const ntok=usage?.completion_tokens ?? timings?.predicted_n;
  const tps=timings?.predicted_per_second ?? (ntok?ntok/secs:null);
  const stats=[resets?`↺ ${resets} loop${resets>1?'s':''} cut (Neural Reset)`:null, ntok?`${ntok} tokens`:null, tps?`${tps.toFixed(1)} tok/s`:null, tFirst?`first token ${((tFirst-t0)/1000).toFixed(2)}s`:null].filter(Boolean).join(' · ');
  const fullStats=[routeNote,stats,compressed?cmpNote(compressed):''].filter(Boolean).join(' · ');
  out.stats.textContent=fullStats;
  let msg=null;
  const conf=lp.length?confidenceOf(lp):null; if(conf) showConfidence(out.root,conf);
  if(acc||think){ msg={role:'assistant',content:acc,model:servedBy||undefined,think:think||undefined,stats:fullStats,sources:sources||undefined,tools:tools.length?tools:undefined,confidence:conf||undefined,usage:usage||undefined,compressed:compressed||undefined};
    chat.messages.push(msg); out.root._msg=msg; await persist(); }
  status(); refresh();
  busy=false; $('#send').textContent='Send';
  if(msg&&acc&&$('#autoCheck').checked) checkReply(out.root);
  if(msg&&acc) pluginChecks(out.root,true);
  sideSoon();
}
// ---------- side panel: environment, context usage, outputs
const CTX_COL={system:'var(--acc)',documents:'var(--ok)',tools:'var(--warn)',messages:'color-mix(in srgb,var(--fg) 55%,transparent)'};
let sideT=null, ctxLast=null;
function sideShown(){ return !$('#side').hidden; }
function sideSet(open){ $('#side').hidden=!open; $('#ctxStrip').setAttribute('aria-expanded',open);
  $('section.chat').style.setProperty('--strip-h',$('#ctxStrip').offsetHeight+'px'); try{ localStorage.setItem('ns-side',open?'1':'0'); }catch{} }
$('#ctxStrip').onclick=()=>sideSet(!sideShown());
$('#ctxStrip').onkeydown=e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); sideSet(!sideShown()); } };
document.addEventListener('keydown',e=>{ if(e.key==='Escape'&&sideShown()) sideSet(false); });
function sideSoon(){ clearTimeout(sideT); sideT=setTimeout(()=>{ ctxRefresh(); sideOutputs(); },400); }
function fmtN(n){ return n==null?'?':n>=10000?(n/1000).toFixed(1)+'k':String(n); }
function sideEnv(){ const s=status_||{};
  $('#stripModel').textContent=s.running?(s.id||s.model):'no model loaded'; const m=models.find(x=>x.id===s.id); const st=m?.settings||{};
  $('#envState').innerHTML=s.running?'<span class="okc">● running</span>':'not loaded';
  const row=(k,v)=>v==null||v===''?'':`<dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd>`;
  $('#envKv').innerHTML=row('Model',s.id||s.model||'none')+row('Context',ctxLast?.n_ctx?ctxLast.n_ctx.toLocaleString()+' tokens':(st.ctx?st.ctx.toLocaleString()+' (setting)':null))
    +row('Trained for',ctxLast?.trained_ctx?ctxLast.trained_ctx.toLocaleString():null)+row('GPU layers',s.running?st.ngl:null)+row('Vision',s.running?(s.vision?'yes':'no'):null)
    +row('LoRA',s.lora?s.lora.split(/[\\/]/).pop():null)+row('Workspace',s.workspace)+row('Computer',s.computer)
    +(s.id?`<dt></dt><dd><a href="#model=${encodeURIComponent(s.id)}&field=ctx">Model settings</a></dd>`:''); }
async function ctxRefresh(){
  const draft=$('#in').value.trim(); const msgs=chat.messages.map(m=>({role:m.role,content:m.content})); if(draft) msgs.push({role:'user',content:draft});
  const last=[...chat.messages].reverse().find(m=>m.role==='assistant'&&m.sources);
  const documents=$('#ragPick').value&&last?last.sources.map(h=>h.text).join('\n'):'';
  let c; try{ c=await post('/api/context',{messages:msgs,preset:$('#preset').value,tools:$('#toolsOn').checked||undefined,documents,...cmpReq()}).then(r=>r.json()); }catch{ return; }
  ctxLast=c; const cap=c.n_ctx||(models.find(x=>x.id===(status_||{}).id)?.settings?.ctx)||null;
  const used=c.total, rsv=c.reply_reserve||0, denom=Math.max(cap||0,used+rsv,1);
  $('#ctxBar').innerHTML=c.parts.map(p=>p.tokens?`<span style="width:${100*p.tokens/denom}%;background:${CTX_COL[p.key]}" title="${esc(p.label)}: ${p.tokens}"></span>`:'').join('')
    +(rsv?`<span class="rsv" style="width:${100*Math.min(rsv,Math.max(0,denom-used))/denom}%" title="room kept for the reply: ${rsv}"></span>`:'');
  const pct=cap?Math.round(100*used/cap):null;
  $('#stripBar').innerHTML=$('#ctxBar').innerHTML;
  $('#stripPct').innerHTML=cap?`<span class="${pct>=90?'bad':pct>=75?'warn':''}">${fmtN(used)} / ${fmtN(cap)} tokens · ${pct}%</span>`:`${fmtN(used)} tokens`;
  $('#ctxPct').innerHTML=cap?`<span class="${pct>=90?'bad':pct>=75?'warn':''}">${fmtN(used)} / ${fmtN(cap)} · ${pct}%</span>`:fmtN(used)+' tokens';
  $('#ctxLeg').innerHTML=c.parts.map(p=>`<i style="background:${CTX_COL[p.key]}"></i><span>${esc(p.label)}</span><span class="n">${p.tokens.toLocaleString()}</span>`).join('')
    +(rsv?`<i class="rsv" style="background:repeating-linear-gradient(45deg,var(--line) 0 3px,transparent 3px 6px)"></i><span>Room for the reply</span><span class="n">${rsv.toLocaleString()}</span>`:'');
  const lastU=[...chat.messages].reverse().find(m=>m.usage)?.usage;
  const notes=[c.method==='estimate'?'Estimated at about 4 characters a token; load the model for exact counts.':'Counted with the model\'s tokenizer; the chat template adds a little.',
    lastU?`The last reply used ${lastU.prompt_tokens?.toLocaleString?.()??'?'} prompt + ${lastU.completion_tokens?.toLocaleString?.()??'?'} reply tokens.`:'',
    c.images?`${c.images} image(s) also take context (how much depends on the vision projector).`:'',
    c.summarized?`The first ${c.summarized} messages count as their summary.`:'',
    pct!=null&&used+rsv>cap?(cmpGet().history&&cmpGet().history!=='off'?`<span class="warn">Over the window: the oldest messages will be ${cmpGet().history==='cut'?'left out':'summarized'} on the next send.</span>`:'<b class="bad">The next request will not fit:</b> start a new chat, raise the context in Model settings, or choose what happens below.'):pct>=75?'<span class="warn">Getting full.</span>':''];
  $('#ctxNote').innerHTML=notes.filter(Boolean).join(' '); sideEnv(); }
function sideOutputs(){ const out=[]; let n=0;
  chat.messages.forEach((m,i)=>{ if(m.role!=='assistant') return; n++;
    if(m.check&&m.check.url) out.push(`<a href="${esc(m.check.url)}" target="_blank" rel="noopener">🧠 3D view of reply ${n}<span class="k">${m.check.flagged?m.check.flagged+' flagged':'checked'}</span></a>`);
    if(m.check&&m.check.review) out.push(`<a href="${esc(m.check.review)}">▦ Neuron review, reply ${n}</a>`);
    for(const t of m.tools||[]) out.push(`<div class="o">🔧 ${esc(t.name||t.server+'.'+t.tool||'tool call')}<span class="k">reply ${n}${t.result?.error?' · failed':''}</span></div>`);
    if(m.sources&&m.sources.length) out.push(`<div class="o">📄 ${m.sources.length} source(s): ${esc([...new Set(m.sources.map(h=>h.source))].slice(0,3).join(', '))}<span class="k">reply ${n}</span></div>`);
    if(m.compressed&&m.compressed.history&&m.compressed.history.summary) out.push(`<details class="o" style="display:block"><summary>🗜 Summary of messages 1–${m.compressed.history.upto}<span class="k">reply ${n}</span></summary><div class="note" style="white-space:pre-wrap">${esc(m.compressed.history.summary)}</div></details>`);
    for(const pc of m.plugin_checks||[]) out.push(`<div class="o">🧩 ${esc(pc.title||pc.plugin||'plugin check')}<span class="k">reply ${n}</span></div>`); });
  $('#outN').textContent=out.length||''; $('#stripOut').textContent=out.length?`· ${out.length} output${out.length>1?'s':''}`:'';
  $('#outList').innerHTML=out.length?out.join(''):'<div class="note">Nothing yet. 3D views of checked replies, tool calls, cited sources and plugin checks from this chat collect here.</div>'; }
$('#outExport').onclick=()=>$('#export').click();
// contextual compression (scripts/context_compress.py): passages trimmed to matching sentences; old turns cut or summarized
function cmpGet(){ try{ return JSON.parse(localStorage.getItem('ns-compress')||'{}'); }catch{ return {}; } }
function cmpReq(){ const c=cmpGet(), o={};
  if(c.passages||(c.history&&c.history!=='off')) o.compress={passages:!!c.passages,history:c.history||'off'};
  if(c.history==='summarize'&&chat.summary) o.summary=chat.summary; return o; }
function cmpNote(c){ const p=c.passages, h=c.history, out=[];
  if(p&&p.before) out.push(`passages ${p.before}→${p.after} tokens`);
  if(h&&h.history==='summarized') out.push(`summarized the first ${h.upto} messages`);
  if(h&&h.history==='summary') out.push(`first ${h.upto} messages as a summary`);
  if(h&&h.history==='cut') out.push(`left out the first ${h.dropped} messages`);
  if(h&&h.history==='failed') out.push('compression failed: '+h.error);
  return out.join(' · '); }
{ const c=cmpGet(); $('#cmpPass').checked=!!c.passages; $('#cmpHist').value=c.history||'off'; }
['#cmpPass','#cmpHist'].forEach(s=>$(s).addEventListener('change',()=>{ try{ localStorage.setItem('ns-compress',JSON.stringify({passages:$('#cmpPass').checked,history:$('#cmpHist').value})); }catch{} sideSoon(); }));
$('#in').addEventListener('input',sideSoon); ['#preset','#ragPick','#toolsOn'].forEach(s=>$(s).addEventListener('change',sideSoon));
try{ if(localStorage.getItem('ns-side')==='1') setTimeout(()=>sideSet(true),0); }catch{}
sideSoon();
$('#export').onclick=()=>{
  const md=`# ${chat.title}\n\n`+chat.messages.map(m=>`**${m.role}**\n\n${textOf(m.content)}${imagesOf(m.content).length?`\n\n_[${imagesOf(m.content).length} image(s)]_`:''}\n`).join('\n');
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([md],{type:'text/markdown'}));
  a.download=(chat.title||'chat').replace(/[^\w -]+/g,'_').slice(0,50)+'.md'; a.click(); URL.revokeObjectURL(a.href);
};

// ---------- hub
$('#q').onkeydown=e=>{ if(e.key==='Enter'){ e.preventDefault(); $('#go').click(); } };
$$('[data-qs]').forEach(a=>a.onclick=e=>{ e.preventDefault(); $('#q').value=a.dataset.qs; $('#go').click(); });
$('#go').onclick=async()=>{
  const q=$('#q').value.trim(); if(!q) return;
  $('#hits').textContent='searching…';
  const r=await (await fetch('/api/search?q='+encodeURIComponent(q))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#hits').innerHTML=r.map(m=>`<div class="m" data-repo="${esc(m.repo)}"><div class="mn">${esc(m.repo)}</div>
    <div class="mm">${m.downloads.toLocaleString()} downloads ${m.gated?'<span class="tag warn">gated</span>':''}</div></div>`).join('');
  $$('#hits .m').forEach(el=>el.onclick=()=>files(el.dataset.repo));
};
async function files(repo){
  $('#hits').innerHTML='<div class="note">loading files…</div>';
  const r=await (await fetch('/api/files?repo='+encodeURIComponent(repo))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#hits').innerHTML=`<div class="note"><b>${esc(repo)}</b> · click a file to download</div>`+r.map((g,i)=>
    `<div class="m" data-i="${i}"><div class="mn">${esc(g.name)}</div>
     <div class="mm">${g.recommended?'<span class="tag okc" title="the largest file that runs fully on this computer\'s GPU (or in RAM)">★ Recommended</span>':''}${g.quant?`<span class="tag">${esc(g.quant)}</span>`:''}${g.size?human(g.size):'size unknown'}
     ${g.parts.length>1?`<span class="tag">${g.parts.length} parts</span>`:''}${fitBadge(g.fit)}</div>${g.fit&&g.fit.note?`<div class="mm ${g.fit.ok===false?'bad':g.fit.level==='partial'?'warn':''}">${esc(g.fit.note)}</div>`:''}</div>`).join('');
  $$('#hits .m[data-i]').forEach(el=>el.onclick=async()=>{ await post('/api/download',{repo,group:r[+el.dataset.i]}); jobs(); });
}
async function jobs(){
  const js=await (await fetch('/api/downloads')).json();
  $('#jobs').innerHTML = js.map(j=>{ const pct=j.total?Math.min(100,100*j.done/j.total):0;
    return `<div class="m"><div class="mn">${esc(j.name)}</div>
      <div class="mm">${esc(j.state)}${j.error?': <span class="bad">'+esc(j.error)+'</span>':''}${j.total?` · ${pct.toFixed(1)}% of ${human(j.total)}`:''}</div>
      <div style="height:4px;background:var(--line);border-radius:2px;margin-top:.3rem"><div style="height:4px;width:${pct}%;background:var(--acc);border-radius:2px"></div></div>
      ${j.state==='running'?`<button data-c="${esc(j.id)}" style="margin-top:.35rem;font-size:11px;padding:.15rem .5rem">Cancel</button>`:''}</div>`;
  }).join('');
  $$('#jobs button[data-c]').forEach(b=>b.onclick=async()=>{ await post('/api/download/cancel',{id:b.dataset.c}); jobs(); });
  if(js.some(j=>j.state==='running')) setTimeout(jobs,1200); else if(js.some(j=>j.state==='done')) refresh();
}
async function presets(){
  const p=await (await fetch('/api/presets')).json();
  $('#preset').innerHTML=Object.keys(p).map(k=>`<option>${esc(k)}</option>`).join('');
}
// ---------- docs (RAG)
let curColl=null;
async function colls(){
  const r=await (await fetch('/api/rag')).json(), cs=r.collections||[];
  const keep=$('#ragPick').value;
  $('#ragPick').innerHTML='<option value="">none</option>'+cs.map(c=>`<option ${c.name===keep?'selected':''}>${esc(c.name)}</option>`).join('');
  $('#colls').innerHTML=cs.length?cs.map(c=>`<div class="m" data-c="${esc(c.name)}"><div class="mn">${esc(c.name)}</div><div class="mm">${c.docs} files · ${c.chunks} passages${c.dense?' · <span class="tag">dense</span>':''}</div></div>`).join(''):'<div class="note">No collections yet.</div>';
  $$('#colls .m').forEach(el=>el.onclick=()=>openColl(el.dataset.c));
  return r;
}
async function openColl(name){ curColl=name; $('#collpanel').style.display=''; $('#collname').textContent=name; $('#raghits').textContent='';
  const ds=await (await fetch('/api/rag/docs?c='+encodeURIComponent(name))).json();
  $('#docs').innerHTML=(ds.length?ds:[]).map(d=>`<div class="m"><div class="mn">${esc(d.source)}</div><div class="mm">${d.chunks} passages</div><button class="x" data-d="${esc(d.doc)}" title="remove">×</button></div>`).join('')||'<div class="note">empty</div>';
  $$('#docs button[data-d]').forEach(b=>b.onclick=async()=>{ await post('/api/rag/delete',{collection:name,doc:b.dataset.d}); openColl(name); colls(); }); }
$('#mkcoll').onclick=async()=>{ const n=$('#newcoll').value.trim(); if(!n) return;
  const r=await post('/api/rag/create',{collection:n}); if(!r.ok){ alert((await r.json()).error); return; } $('#newcoll').value=''; await colls(); openColl(n); };
$('#adddocs').onclick=()=>$('#docfile').click();
$('#docfile').onchange=async()=>{ const fs=[...$('#docfile').files]; $('#docfile').value='';
  for(const f of fs){ if(f.size>32e6){ $('#docmsg').textContent=f.name+' is larger than 32 MB'; continue; }
    $('#docmsg').textContent='indexing '+f.name+'…';
    const data=await new Promise(r=>{const fr=new FileReader(); fr.onload=()=>r(fr.result); fr.readAsDataURL(f);});
    const r=await post('/api/rag/upload',{collection:curColl,filename:f.name,data}); const j=await r.json();
    $('#docmsg').textContent=r.ok?`${f.name}: ${j.chunks} passages`:`${f.name}: ${j.error}`; }
  openColl(curColl); colls(); };
$('#ragq').onkeydown=async e=>{ if(e.key!=='Enter') return; const r=await post('/api/rag/search',{collection:curColl,query:$('#ragq').value,k:4}); const j=await r.json();
  $('#raghits').innerHTML=r.ok?(j.map((h,i)=>`<details><summary>[${i+1}] ${esc(h.source)}${h.page?' p.'+h.page:''}</summary><pre style="white-space:pre-wrap">${esc(h.text)}</pre></details>`).join('')||'no match'):esc(j.error); };
$('#dropcoll').onclick=async()=>{ if(!confirm('Delete collection '+curColl+'?')) return; await post('/api/rag/delete',{collection:curColl}); $('#collpanel').style.display='none'; colls(); };

// ---------- GPUs
async function gpus(){ try{ const r=await (await fetch('/api/gpus')).json(); if(!r.nvidia||!r.count) return;
  $('#gpubox').style.display='';
  $('#gpuhint').textContent=`${r.count} × ${r.gpus[0].name}, ${(r.total_vram/2**30).toFixed(0)} GiB`+(r.multi_gpu?` · suggested: ${r.multi_gpu.llama_server}`:'');
  $('#gpuhint').title=(r.multi_gpu?r.multi_gpu.why+'\n\n':'')+(r.notes||[]).join('\n');
}catch{} }

// ---------- MCP tools
async function mcp(reload){
  const r=reload?await (await post('/api/mcp/reload',{})).json():await (await fetch('/api/mcp')).json();
  if(r.error){ $('#mcplist').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#mcpcfg').innerHTML=(r.config?'config: <code>'+esc(r.config)+'</code> · ':'No connectors yet. ')+'<a href="/settings#connectors">Manage connectors</a>';
  const sv=r.servers||[]; const n=sv.reduce((a,x)=>a+(x.connected?x.tools.length:0),0);
  $('#toolsBox').style.display=n?'':'none';
  $('#mcplist').innerHTML=sv.map(x=>`<div class="m"><div class="mn">${esc(x.name)} <span class="tag">${esc(x.transport)}</span>${x.disabled?'<span class="tag">disabled</span>':x.connected?'<span class="tag okc">connected</span>':'<span class="tag bad">down</span>'}${x.auto_approve===true?'<span class="tag warn">auto-approve all</span>':''}${x.disabled?'':` <button class="x" style="width:auto;padding:0 .4rem" data-mtest="${esc(x.name)}" title="list its tools again, and call one read-only tool that needs no arguments">Test</button>`}</div>
    <div class="mm" data-mres="${esc(x.name)}"></div><div class="mm">${x.error?'<span class="bad">'+esc(x.error)+'</span>':x.tools.map(t=>`<div title="${esc(t.description)}">${esc(t.name)}${Array.isArray(x.auto_approve)&&x.auto_approve.includes(t.name)?' <span class="tag">auto</span>':''}</div>`).join('')}</div></div>`).join('');
  document.querySelectorAll('[data-mtest]').forEach(b=>b.onclick=async()=>{ const out=document.querySelector(`[data-mres="${b.dataset.mtest}"]`); out.textContent='testing…';
    try{ const j=await (await post('/api/mcp/test',{server:b.dataset.mtest})).json();
      out.innerHTML=j.ok?`<span class="okc">answered</span> in ${j.seconds}s · ${j.tools} tools`+(j.called?` · ${esc(j.called)}: ${j.call_ok?'<span class="okc">ok</span>':'<span class="bad">failed</span>'} <code>${esc((j.result||'').slice(0,160))}</code>`:' · no read-only tool without arguments to call')
        :`<span class="bad">${esc(j.error||'failed')}</span>`; }catch(e){ out.innerHTML=`<span class="bad">${esc(String(e))}</span>`; } });
}
$('#mcpreload').onclick=()=>mcp(true);

refresh(); if(TAB_ALIAS[location.hash.slice(1)]) applyHash(); (function hashWhenReady(n){ if(models.length||n>40) applyHash(); else setTimeout(()=>hashWhenReady(n+1),150); })(0); status(); presets(); jobs(); loadChats(); renderChat(); colls(); mcp(); gpus(); setInterval(status,4000);
// First run: point at Setup while llama-server or models are missing (owner only; devices get 403).
fetch('/api/setup').then(r=>r.ok?r.json():null).then(s=>{ if(!s||(s.ready.server&&s.ready.models)) return;
  const miss=[!s.ready.server&&'llama-server',!s.ready.models&&'models'].filter(Boolean).join(' and ');
  const d=document.createElement('div'); d.id='setupBanner';
  d.style.cssText='padding:.55rem 1rem;background:color-mix(in srgb,var(--acc) 18%,var(--panel));border-bottom:1px solid var(--line);font-size:13px';
  d.innerHTML=`Almost there: Studio needs ${miss}. <a href="/setup" style="color:var(--acc);font-weight:600">Open Setup →</a>`;
  document.querySelector('header').after(d); }).catch(()=>{});
</script>"""


LOGIN = """<!DOCTYPE html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Studio</title>
<style>:root{color-scheme:dark}body{font:14px ui-sans-serif,system-ui,sans-serif;background:#111316;color:#e6e8eb;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#181b20;border:1px solid #2a2f37;border-radius:10px;padding:1.6rem;width:320px}
h1{font-size:15px;margin:0 0 1rem}input{width:100%;padding:.5rem;border:1px solid #2a2f37;background:#20242b;color:#e6e8eb;
border-radius:6px;font:13px ui-monospace,monospace}
button{width:100%;margin-top:.7rem;padding:.5rem;border:1px solid #6ea8e0;background:#6ea8e0;
color:#0d1117;border-radius:6px;cursor:pointer;font:500 13px ui-sans-serif,system-ui}
.e{color:#ff8b7e;font-size:13px;margin-top:.5rem;min-height:1em}</style>
<form id="f"><h1>OmniCortex Studio</h1>
<input id="t" type="password" placeholder="access token" autofocus>
<button>Unlock</button><div class="e" id="e"></div></form>
<script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({token:document.getElementById('t').value})});
if(r.ok) location.reload(); else document.getElementById('e').textContent='Incorrect token.';};
</script>"""
