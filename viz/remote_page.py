"""Studio's /remote page: the computers this Studio controls that run models but
not OmniCortex (LM Studio, Ollama, llama.cpp with the OmniCortex Remote agent).
This Studio is the controller; those computers are hosts."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Remote Hosts</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0;--soft:#eef2fc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--soft:#1f2433}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624;--soft:#1f2433}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--acc)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{font-size:13px}
main{max-width:1050px;margin:0 auto;padding:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem;margin-bottom:1rem;scroll-margin-top:1rem}
h2{font-size:15px;margin:.1rem 0 .5rem}h3{font-size:13px;margin:.8rem 0 .3rem}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no);font-size:12.5px;white-space:pre-wrap}.okm{color:var(--ok);font-size:12.5px}.warn{color:var(--warn);font-size:12.5px}
input,select{font:12.5px ui-monospace,monospace;padding:.35rem .45rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg);min-width:0}
button{font:500 13px ui-sans-serif,system-ui;padding:.38rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}button:disabled{opacity:.5}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.grow{flex:1}
table{width:100%;border-collapse:collapse}td,th{padding:.35rem .3rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:500}
code,.code{font:12px ui-monospace,monospace;background:var(--code);padding:.05rem .3rem;border-radius:4px;word-break:break-all}
.roles{display:grid;grid-template-columns:1fr auto 1fr;gap:.7rem;align-items:center}
.role{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;background:var(--bg)}.role.me{border:2px solid var(--acc);background:var(--soft)}
.role b{font-size:14px}.arrow{font-size:20px;color:var(--mut)}
@media(max-width:700px){.roles{grid-template-columns:1fr}.arrow{justify-self:center;transform:rotate(90deg)}}
.steps{counter-reset:s}.step{border-left:3px solid var(--line);padding:.2rem 0 .6rem .9rem;margin-left:.3rem;position:relative}
.step.on{border-color:var(--acc)}.step h3{margin-top:0}
.methods{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:.5rem}
.method{border:1px solid var(--line);border-radius:8px;padding:.55rem .65rem;cursor:pointer;background:var(--bg)}
.method:hover{border-color:var(--acc)}.method.sel{border:2px solid var(--acc);background:var(--soft)}
.method b{display:block;font-size:13.5px}.method span{font-size:12px;color:var(--mut)}
.pill{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 .35rem;white-space:nowrap}
.on-line{color:var(--ok)}.off-line{color:var(--no)}
.host{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;margin:.5rem 0;background:var(--bg)}
.host .detail{margin-top:.5rem;display:none}.host.open .detail{display:block}
.os{display:flex;gap:.3rem;margin:.3rem 0}.os button{padding:.2rem .6rem;font-size:12px}.os button.on{border-color:var(--acc);color:var(--acc)}
pre{font:12px ui-monospace,monospace;background:var(--code);padding:.5rem .6rem;border-radius:6px;white-space:pre-wrap;word-break:break-all;margin:.3rem 0}
</style>
<header><h1>Studio · Remote hosts</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/remote">Remote</a><a href="/link">Link (Studio to Studio)</a><a href="/connect">Connect</a><a href="/jobs">Jobs</a></header>
<main>
<section class="card"><div class="roles">
 <div class="role me"><b>This computer: CONTROLLER</b><div class="note">OmniCortex Studio. It decides what runs where, and sends the instructions.</div><div class="note" id="meAddr"></div></div>
 <div class="arrow">→</div>
 <div class="role"><b>Hosts: <span id="hostCount">0</span></b><div class="note">Computers that run the models (LM Studio, Ollama, llama.cpp) with the small OmniCortex Remote agent. They do what this Studio asks, within what their owner allows, and can revoke it at any time.</div></div>
</div>
<div class="note" style="margin-top:.5rem">The other computer runs OmniCortex Studio itself? Use <a href="/link">Link</a> instead. Any MCP client (Claude Desktop, Cline…) can control a host too: see the agent's README.</div></section>

<section class="card" id="hosts"><div class="row"><h2 class="grow">Your hosts</h2><button id="refresh">Refresh</button></div><div id="hostList" class="note">loading…</div><div id="sends"></div></section>

<section class="card" id="add"><h2>Add a host</h2>
<div class="row" style="margin-bottom:.6rem"><input id="quick" class="grow" placeholder="Shortcut: paste the host's pairing link (nsr://…) here"><button class="pri" id="quickGo">Pair</button></div>
<div class="steps">
<div class="step on"><h3>1. Put the agent on the host</h3>
 <div class="row"><a href="/remote/download"><button class="pri">Download OmniCortex Remote (.zip)</button></a><span class="note">Python 3.9 or newer on the host; nothing else to install.</span></div>
 <div class="os" id="os"><button data-os="win" class="on">Windows</button><button data-os="mac">macOS</button><button data-os="linux">Linux</button><button data-os="pip">pip</button></div>
 <pre id="osCmd"></pre>
 <div class="note">It starts with a console in the browser on the host, where its owner chooses how controllers reach it, sees a pairing code, and allows or revokes what controllers may do.</div></div>
<div class="step on"><h3>2. How can this Studio reach the host?</h3>
 <div class="methods" id="methods"></div>
 <div id="methodForm" style="margin-top:.6rem"></div></div>
<div class="step on"><h3>3. Result</h3><div id="result" class="note">Pick a method above.</div></div>
</div></section>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const post=async(u,b)=>{ const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}); const j=await r.json().catch(()=>({error:'HTTP '+r.status})); if(!r.ok) throw new Error(j.error||('HTTP '+r.status)); return j; };
const gb=n=>n==null?'':(n/2**30).toFixed(n>2**30*10?0:1)+' GB';
let S=null, method=null, open=new Set();
const OS={win:'Unzip neuronscope-remote.zip, then double-click\n  neuronscope-remote\\OmniCortex Remote.bat\n(Python from python.org or the Microsoft Store; tick "Add to PATH")',
 mac:'unzip neuronscope-remote.zip && cd neuronscope-remote\n./neuronscope-remote.sh',
 linux:'unzip neuronscope-remote.zip && cd neuronscope-remote\n./neuronscope-remote.sh',
 pip:'pip install ./neuronscope-remote          # or: pipx install ./neuronscope-remote\nneuronscope-remote                          # add [tls] for certificates without openssl'};
document.querySelectorAll('#os button').forEach(b=>b.onclick=()=>{ document.querySelectorAll('#os button').forEach(x=>x.classList.toggle('on',x===b)); $('#osCmd').textContent=OS[b.dataset.os]; });
$('#osCmd').textContent=OS.win;
const METHODS=[
 ['lan','Same network','Home or office network. Studio finds the host while its pairing code is shown.','lan'],
 ['vpn','VPN','Tailscale, Headscale, ZeroTier, WireGuard, NetBird, Nebula… Works across the internet.','vpn'],
 ['ssh','SSH tunnel','The host has an SSH server and this computer your SSH key. Nothing else exposed.','ssh'],
 ['tunnel','Public tunnel','Cloudflare Tunnel, ngrok… give the host a public HTTPS address.','tunnel'],
 ['callhome','The host calls this Studio','The host is behind a router or firewall you cannot open. It connects out to here.','callhome']];
function hostMode(m){ return {lan:'Same network (home or office)',vpn:'A VPN',ssh:'SSH tunnel',tunnel:'Public tunnel',callhome:'This computer calls the controller'}[m]; }
function renderMethods(){ $('#methods').innerHTML=METHODS.map(([k,t,d])=>`<div class="method ${k===method?'sel':''}" data-m="${k}"><b>${t}</b><span>${d}</span></div>`).join('');
  document.querySelectorAll('[data-m]').forEach(el=>el.onclick=()=>{ method=el.dataset.m; renderMethods(); form(); }); }
const codeInput=`<input id="fcode" placeholder="code, like ABCDE-FGH23" size="16" autocomplete="off">`;
const nameInput=`<input id="fname" placeholder="name (optional)" size="14">`;
function form(){ const m=method, onHost=`<div class="note">On the host, in the agent's console: <b>1.</b> choose <i>${hostMode(m)}</i> → <b>Use this link</b>, <b>2.</b> <b>Show a pairing code</b>.</div>`; let h='';
  if(m==='lan') h=onHost+`<div class="row" style="margin-top:.4rem"><button class="pri" id="scan">Search the network</button><span class="note" id="scanMsg"></span></div><div id="found"></div>
    <div class="note" style="margin-top:.4rem">Not found? Hosts on another subnet or Wi-Fi with client isolation are not heard: type its address instead.</div>
    <div class="row" style="margin-top:.3rem"><input id="faddr" placeholder="192.168.1.20" size="22">${codeInput}${nameInput}<button class="pri" id="pairBtn">Pair</button></div>`;
  if(m==='vpn') h=onHost+`<div class="row" style="margin-top:.4rem"><input id="faddr" placeholder="100.101.102.103 or host.tailnet.ts.net" size="34">${codeInput}${nameInput}<button class="pri" id="pairBtn">Pair</button></div>
    <div class="note">The address the host's console shows under “Listening at”. Both computers must be in the same VPN network.</div>`;
  if(m==='ssh') h=`<div class="note">On the host: choose <i>SSH tunnel</i> in the agent's console, then <b>Show a pairing code</b>. This Studio runs <code>ssh -N -L</code> itself with your SSH keys and config (no password prompts: set up a key with <code>ssh-copy-id</code> first), and keeps the tunnel open while it uses the host.</div>
    <div class="row" style="margin-top:.4rem"><input id="fssh" placeholder="you@host or an ~/.ssh/config name" size="28"><input id="fsshp" placeholder="SSH port (22)" size="10">${codeInput}${nameInput}<button class="pri" id="pairBtn">Pair</button></div>`;
  if(m==='tunnel') h=`<div class="note">On the host: choose <i>Public tunnel</i>, start the tunnel (for example <code>cloudflared tunnel --url http://127.0.0.1:47871</code>), enter the address it prints in the console, then <b>Show a pairing code</b>. The tunnel's own HTTPS certificate is checked as usual; the pairing code and the host's token still guard the host.</div>
    <div class="row" style="margin-top:.4rem"><input id="faddr" placeholder="https://something.trycloudflare.com" size="36">${codeInput}${nameInput}<button class="pri" id="pairBtn">Pair</button></div>`;
  if(m==='callhome'){ const st=S&&S.studio; h=`<div class="note">The host connects out to this Studio, so this Studio must be reachable from the host: on the network with a token and TLS (<a href="/setup">Setup</a>, docs/SECURITY.md), over a VPN, or behind a tunnel. The host needs no open port.</div>
    ${st&&!st.network?'<div class="warn" style="margin:.4rem 0">This Studio listens on this computer only right now; the host cannot reach it. Restart it with <code>--host 0.0.0.0</code> (plus a token and TLS), or give a tunnel address below.</div>':''}
    <div class="row" style="margin-top:.4rem"><input id="furl" size="34" value="${esc(st?st.url:location.origin)}" title="Studio's address as the host sees it">${nameInput}<button class="pri" id="invBtn">Make an invite</button></div><div id="inv"></div>`; }
  $('#methodForm').innerHTML=h;
  if($('#scan')) $('#scan').onclick=scan;
  if($('#pairBtn')) $('#pairBtn').onclick=()=>pair({address:($('#faddr')||{}).value, code:$('#fcode').value, name:$('#fname').value, ssh_target:($('#fssh')||{}).value, ssh_port:($('#fsshp')||{}).value||null});
  if($('#invBtn')) $('#invBtn').onclick=invite;
}
async function scan(){ $('#scanMsg').textContent='listening for 4 s…'; $('#found').innerHTML='';
  try{ const j=await post('/api/remote/discover',{seconds:4}); $('#scanMsg').textContent=j.hosts.length?'':'No host with pairing open was heard.';
    $('#found').innerHTML=j.hosts.map((h,i)=>`<div class="host"><div class="row"><b>${esc(h.name)}</b><code>${esc(h.url)}</code>${h.short_fingerprint?`<span class="pill">certificate ${esc(h.short_fingerprint)}</span>`:''}<span class="grow"></span><input id="dc${i}" placeholder="its code" size="14"><button class="pri" data-d="${i}">Pair</button></div></div>`).join('');
    document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>{ const h=j.hosts[+b.dataset.d]; pair({address:h.url, code:$('#dc'+b.dataset.d).value, name:h.name}); });
  }catch(e){ $('#scanMsg').innerHTML='<span class="err">'+esc(e.message)+'</span>'; } }
async function pair(b){ $('#result').innerHTML='pairing…';
  try{ const j=await post('/api/remote/pair',{method:method||'lan',...b});
    $('#result').innerHTML=`<span class="okm">Paired with <b>${esc(j.name)}</b>${j.host&&j.host!==j.name?' ('+esc(j.host)+')':''}: ${j.models.length} model(s).</span>`+
      (j.short_fingerprint?` <span class="note">Certificate <b>${esc(j.short_fingerprint)}</b>: the host's console shows the same.</span>`:'')+
      (j.error?`<div class="err">${esc(j.error)}</div>`:'')+`<div class="note">Its models are now in Studio's model list as <code>${esc(j.name)}:MODEL</code>, for chat, TestQA and Jobs.</div>`;
    $('#quick').value=''; open.add(j.name); load(); }
  catch(e){ $('#result').innerHTML='<span class="err">'+esc(e.message)+'</span>'; } }
async function invite(){ try{ const j=await post('/api/remote/invite',{url:$('#furl').value.trim(),name:$('#fname').value.trim()});
  $('#inv').innerHTML=(j.warning?`<div class="warn">${esc(j.warning)}</div>`:'')+`<div style="margin-top:.4rem">On the host, paste the invite into the agent's console (<i>This computer calls the controller</i> → Join), or run:</div><pre>${esc(j.command)}</pre><div class="row"><button id="cpInv">Copy the invite</button><span class="note">Single use, valid until ${new Date(j.expires*1000).toLocaleTimeString()}. The host appears below when it joins.</span></div>`;
  $('#cpInv').onclick=()=>navigator.clipboard.writeText(j.invite); $('#result').textContent='Waiting for the host to join…';
  const before=S?S.hosts.length:0; let n=0; const iv=setInterval(async()=>{ await load(); if(S.hosts.length>before||++n>180){ clearInterval(iv); if(S.hosts.length>before) $('#result').innerHTML='<span class="okm">The host joined.</span>'; } },5000);
 }catch(e){ $('#inv').innerHTML='<span class="err">'+esc(e.message)+'</span>'; } }
$('#quickGo').onclick=()=>{ const v=$('#quick').value.trim(); if(!v) return; method=v.startsWith('nsr://')&&/s=https/.test(v)?'lan':'vpn'; pair({link:v}); };
function linkCell(h){ const how={lan:'same network',vpn:'VPN',ssh:'SSH '+(h.ssh||''),tunnel:'public tunnel',callhome:'calls in'}[h.method]||h.method;
  return `<span class="pill">${esc(how)}</span> ${h.method==='callhome'?(h.online?'<span class="on-line">● connected</span>':'<span class="off-line">● not connected</span>')+(h.last_seen!=null?` <span style="color:var(--mut)" title="the host checks in at least every 25 s">seen ${h.last_seen<60?h.last_seen+' s':Math.round(h.last_seen/60)+' min'} ago</span>`:'')+(h.agent_version?` <span style="color:var(--mut)">agent ${esc(h.agent_version)}</span>`:''):'<code>'+esc(h.url)+'</code>'}${h.pinned?` <span class="pill" title="certificate pinned at pairing">🔒 ${esc(h.short_fingerprint)}</span>`:''}`; }
function render(){ $('#hostCount').textContent=S.hosts.length; const st=S.studio;
  $('#meAddr').innerHTML=`${esc(st.url)} · ${st.network?'reachable from the network':'this computer only'}${st.tls?' · TLS':''}`;
  $('#hostList').innerHTML=S.hosts.length?S.hosts.map((h,i)=>`<div class="host ${open.has(h.name)?'open':''}" data-h="${esc(h.name)}">
   <div class="row"><b>${esc(h.name)}</b>${h.host&&h.host!==h.name?`<span class="note">(${esc(h.host)})</span>`:''} ${linkCell(h)}<span class="grow"></span>
   ${h.error?`<span class="err">${esc(h.error)}</span>`:`<span class="note">${h.models.length} model(s)</span>`}
   <button data-tog="${i}">${open.has(h.name)?'Close':'Open'}</button></div>
   <div class="detail" id="d${i}"></div></div>`).join(''):'<span class="note">None yet. Add one below.</span>';
  document.querySelectorAll('[data-tog]').forEach(b=>b.onclick=()=>{ const h=S.hosts[+b.dataset.tog]; open.has(h.name)?open.delete(h.name):open.add(h.name); render(); });
  S.hosts.forEach((h,i)=>{ if(open.has(h.name)) detail(h,i); });
  $('#sends').innerHTML=S.sends.length?'<h3>Sending</h3><table>'+S.sends.map(x=>`<tr><td>${esc(x.file)} → ${esc(x.host)}</td><td>${x.status==='failed'?'<span class="err">'+esc(x.error)+'</span>':esc(x.status)}</td><td>${gb(x.done)} / ${gb(x.size)}</td></tr>`).join('')+'</table>':''; }
async function detail(h,i){ const el=$('#d'+i); if(!el) return;
  el.innerHTML=`<div class="row"><button data-act="info">Host info</button><button data-act="models">Models</button><button class="bad" data-act="remove">Remove</button><span class="note grow">Removing forgets the host here; its owner can also revoke this Studio there.</span></div>
   <div class="row" style="margin-top:.4rem"><select id="snd${i}" style="max-width:24rem">${S.local_models.map(m=>`<option value="${esc(m.path)}">${esc(m.id)} (${gb(m.size)})</option>`).join('')}</select><label class="note"><input type="checkbox" id="ins${i}" checked> then install into LM Studio</label><button data-act="send">Send model</button></div>
   <div class="row" style="margin-top:.4rem"><input id="q${i}" class="grow" placeholder="Try a model: ask something"><select id="qm${i}">${h.models.map(m=>`<option>${esc(m)}</option>`).join('')}</select><button data-act="ask">Ask</button></div>
   <div id="o${i}" class="note" style="margin-top:.4rem"></div>`;
  const out=$('#o'+i), call=(tool,args)=>post('/api/remote/call',{host:h.name,tool,args});
  el.querySelectorAll('[data-act]').forEach(b=>b.onclick=async()=>{ const a=b.dataset.act; out.textContent='…';
   try{
    if(a==='info'){ const r=await call('host_info',{}); out.innerHTML=`<table><tr><td>name</td><td>${esc(r.name)} <span class="pill">role: ${esc(r.role)}</span> <span class="pill">this Studio: ${esc(r.you_are)}</span></td></tr><tr><td>system</td><td>${esc(r.os)} · ${esc(r.cpus)} CPUs · ${esc(r.ram_gb)} GB RAM</td></tr><tr><td>GPUs</td><td>${r.gpus.length?r.gpus.map(g=>esc(g.name)+' '+Math.round(g.vram_mb/1024)+' GB').join(', '):'none reported'}</td></tr><tr><td>LLM servers</td><td>${r.backends.map(b=>esc(b.kind)+' <code>'+esc(b.url)+'</code>').join('<br>')||'<span class="err">none running</span>'}</td></tr><tr><td>models folder</td><td><code>${esc(r.models_dir||'-')}</code></td></tr><tr><td>incoming</td><td><code>${esc(r.incoming_dir)}</code> (${esc(r.incoming_free_gb)} GB free)</td></tr><tr><td>allowed</td><td>${Object.entries(r.allowed).map(([k,v])=>`<span class="pill" style="color:var(--${v?'ok':'no'})">${esc(k)}</span>`).join(' ')}</td></tr><tr><td>agent</td><td>${esc(r.agent)} · link ${esc(r.link)}</td></tr><tr><td>tracer</td><td>${r.tracer&&r.tracer.available?'OmniCortex tracer ready: 3D checks of its models can run there':'<span class="note">none: 3D checks need the same model file on this computer, or OmniCortex and llama-cett-dump on the host</span>'}</td></tr></table>`; }
    if(a==='models'){ const r=await call('list_models',{}); out.innerHTML='<table><tr><th>model</th><th>server</th><th>state</th><th></th></tr>'+r.models.map(m=>m.error?`<tr><td colspan=4 class="err">${esc(m.backend)}: ${esc(m.error)}</td></tr>`:`<tr><td>${esc(m.id)}</td><td>${esc(m.backend)}</td><td>${m.loaded===true?'loaded':m.loaded===false?'not loaded':''}</td><td><button data-l="${esc(m.id)}">Load</button> <button data-u="${esc(m.id)}">Unload</button> <button data-c="${esc(m.id)}" title="the H-Neuron classifier used when you check this model's replies (3D view)">Classifier…</button></td></tr>`).join('')+'</table>';
      out.querySelectorAll('[data-c]').forEach(x=>x.onclick=async()=>{ const p=prompt('Classifier (.npz) on this computer for '+x.dataset.c+' (empty to clear). Without one, a local copy of the same file lends its classifier.'); if(p===null) return; try{ await post('/api/remote/classifier',{host:h.name,model:x.dataset.c,classifier:p.trim()}); x.textContent=p.trim()?'Classifier set':'Classifier…'; }catch(e){ x.insertAdjacentHTML('afterend',' <span class="err">'+esc(e.message)+'</span>'); } });
      out.querySelectorAll('[data-l],[data-u]').forEach(x=>x.onclick=async()=>{ x.disabled=true; try{ await call(x.dataset.l?'load_model':'unload_model',{model:x.dataset.l||x.dataset.u}); x.textContent='done'; }catch(e){ x.insertAdjacentHTML('afterend',' <span class="err">'+esc(e.message)+'</span>'); } }); }
    if(a==='ask'){ const r=await call('chat',{model:$('#qm'+i).value,messages:[{role:'user',content:$('#q'+i).value}],max_tokens:512}); out.innerHTML=`<pre>${esc(r.reply)}</pre><span class="note">${esc(r.model)} on ${esc(r.backend)}</span>`; }
    if(a==='send'){ const r=await post('/api/remote/send',{host:h.name,path:$('#snd'+i).value,install:$('#ins'+i).checked}); out.textContent='Sending in the background; progress below.'; load(); }
    if(a==='remove'){ if(!confirm('Forget '+h.name+'?')) { out.textContent=''; return; } await post('/api/remote/remove',{host:h.name}); open.delete(h.name); load(); }
   }catch(e){ out.innerHTML='<span class="err">'+esc(e.message)+'</span>'; } }); }
async function load(refresh){ const r=await fetch('/api/remote'+(refresh?'?refresh=1':'')); S=await r.json(); if(!r.ok){ $('#hostList').innerHTML='<span class="err">'+esc(S.error)+'</span>'; return; } render(); if(method===null){ renderMethods(); } }
$('#refresh').onclick=()=>load(true);
renderMethods(); load(true);
setInterval(()=>{ if(S&&S.sends.some(x=>!['done','failed'].includes(x.status))) load(); },3000);
</script>"""

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
