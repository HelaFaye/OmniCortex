"""Studio's /connect page: settings for Cline, Claude Desktop and OpenAI
clients, built from scripts/oc_connect.py so the page and the CLI agree."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Connect Apps</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.55 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{max-width:980px;margin:0 auto;padding:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem;margin-bottom:1rem}
h2{font-size:15px;margin:.1rem 0 .4rem}h3{font-size:13px;margin:.8rem 0 .3rem;color:var(--mut);font-weight:600}
.note{font-size:12.5px;color:var(--mut)}a{color:var(--acc)}
table{border-collapse:collapse;width:100%}td{padding:.3rem .4rem;border-bottom:1px solid var(--line);vertical-align:top;font-size:13px}
td:first-child{color:var(--mut);white-space:nowrap;width:9rem}
code,pre{font:12.5px ui-monospace,monospace;background:var(--code);border-radius:5px}
code{padding:.05rem .3rem}pre{padding:.7rem;overflow:auto;margin:.3rem 0;white-space:pre}
.cp{font:500 12px ui-sans-serif,system-ui;padding:.2rem .55rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:5px;cursor:pointer;margin-left:.4rem}
button.pri{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--acc);background:var(--acc);color:var(--accfg);border-radius:6px;cursor:pointer}
ol{margin:.3rem 0 .3rem 1.2rem;padding:0}
.roles{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:.6rem;align-items:center}
.role{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;background:var(--bg)}.role.me{border:2px solid var(--acc)}
.arrow{font-size:20px;color:var(--mut)}
@media(max-width:800px){.roles{grid-template-columns:1fr}.arrow{justify-self:center;transform:rotate(90deg)}}li{margin:.2rem 0}
.ctabs{display:flex;gap:.3rem;margin:.5rem 0;flex-wrap:wrap}.ctabs button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:999px;padding:.25rem .8rem;cursor:pointer;font:500 13px ui-sans-serif,system-ui}
.ctabs button.on{border-color:var(--acc);color:var(--acc)}#ccHttp,#ccStdio{white-space:pre-wrap;word-break:break-all}ol.steps{margin:.4rem 0 .4rem 1.1rem;padding:0}ol.steps li{margin:.2rem 0}
.tok{background:color-mix(in srgb,var(--ok) 15%,transparent);border:1px solid var(--ok);border-radius:6px;padding:.6rem;margin-top:.5rem;word-break:break-all}
</style>
<header><h1>Studio · Connect apps</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a></header>
<main>
<section class="card"><h2>Who controls whom</h2>
<div class="roles"><div class="role"><b>Your agent app: CONTROLLER</b><div class="note">Cline, Claude Desktop, Cursor, a script. It decides what to ask, and calls Studio (as the MCP client, or through the OpenAI API).</div></div>
<div class="arrow">→</div>
<div class="role me"><b>This Studio (<span id="me"></span>): does the work</b><div class="note">The tools run here and the models answer from here. Studio is the MCP <i>server</i> for your app, on port <span class="port"></span>: <code>/mcp</code> for tools, <code>/v1</code> for chat.</div></div>
<div class="arrow">→</div>
<div class="role"><b>Optional: workers and hosts</b><div class="note">Studio can pass project tasks on to models on other computers (<a href="/workers">Workers</a>, <a href="/remote">Remote</a>). For those, Studio is the controller.</div></div></div>
<div class="note" style="margin-top:.4rem">An LLM server's port (llama-server 8080, LM Studio 1234, Ollama 11434) is a chat API, not MCP. MCP lives at Studio's <code>/mcp</code> and on OmniCortex Remote hosts. <a href="/help/ROLES">Roles and ports</a>.</div></section>

<section class="card"><h2>What Studio offers your app</h2>
<div class="note">Set up either or both:</div>
<ol>
<li><b>Your local models as the app's AI</b>, through Studio's OpenAI-compatible API. In Cline this is the <i>API Provider</i>.</li>
<li><b>OmniCortex's harness and lab as tools</b>, through MCP: the agent can list and route models, check a reply for hallucination risk, run TestQA and retraining jobs, and plan and review projects. Tools that only read are pre-approved; anything that starts work asks first.</li>
</ol><div id="auth" class="note"></div></section>

<section class="card"><h2>Cline (VS Code, Cursor, VSCodium)</h2>
<h3>1 · Models: Cline settings → API Provider</h3><table id="prov"></table>
<h3>2 · Tools: Cline → MCP Servers → Configure (opens cline_mcp_settings.json), paste inside "mcpServers"</h3>
<div class="note">Studio serves MCP itself; Studio must be running.</div><pre id="mcphttp"></pre>
<details><summary class="note">If your Cline cannot reach Studio over HTTP: let Cline start a local process instead (stdio)</summary><pre id="mcpstdio"></pre></details>
<div class="note">Or have OmniCortex write it for you: <code>python scripts/oc_connect.py cline --write</code> (add <code>--editor cursor</code>, <code>codium</code>…). It keeps a backup and touches only the omnicortex entry.</div>
</section>

<section class="card" id="claude"><h2>Claude</h2>
<div class="note">Give Claude OmniCortex's tools. Claude is the controller; the tools run on this Studio. Which Claude you use decides how:</div>
<div class="ctabs" role="tablist"><button role="tab" data-ct="desktop" class="on">Claude Desktop</button><button role="tab" data-ct="code">Claude Code</button><button role="tab" data-ct="web">claude.ai (web, mobile)</button></div>
<div data-cp="desktop">
<ol class="steps"><li>In Claude Desktop open <b>Settings → Developer → Edit Config</b>. That opens <code>claude_desktop_config.json</code>.</li>
<li>Paste this inside <code>"mcpServers"</code> (or have OmniCortex write it: <code>python scripts/oc_connect.py claude-desktop --write</code>, which keeps a backup).</li></ol>
<pre id="claudeDesk"></pre>
<ol class="steps" start="3"><li>Quit Claude Desktop completely and start it again. OmniCortex's tools appear under the tools (slider) button in a chat.</li>
<li>Claude asks before it uses a tool; Studio's own <a href="/settings#tools">Tool approval</a> applies on top.</li></ol>
<div class="note" id="deskTok"></div></div>
<div data-cp="code" hidden>
<div class="note">Run one of these in a terminal. <code>--scope user</code> makes it available in every project; leave it out for this project only. Check with <code>claude mcp list</code>, or <code>/mcp</code> inside Claude Code.</div>
<h3>Over HTTP (Studio must be running)</h3><pre id="ccHttp"></pre>
<h3>Or let Claude Code start the bridge itself</h3><pre id="ccStdio"></pre></div>
<div data-cp="web" hidden>
<div class="note"><b>Not with this Studio as it is, and that is deliberate.</b> claude.ai custom connectors (Settings → Connectors → <i>Add custom connector</i>) are reached from Anthropic's servers, not from your computer. So the address must be a public HTTPS URL that anyone on the internet can reach, and sign-in has to work through OAuth. Studio signs apps in with its own token, which that dialog has no place for. Putting Studio on the public internet without a sign-in would hand your models and tools to anyone who finds the address.</div>
<div class="note">Use Claude Desktop or Claude Code on a computer that can reach this Studio (this one, or one on your LAN or VPN). Claude Desktop's own connectors list then shows OmniCortex too.</div></div>
</section>

<section class="card"><h2>Any OpenAI-compatible app</h2><table id="oa"></table>
<div class="note">Open WebUI, Continue, LibreChat, scripts with the <code>openai</code> package… Model ids are in Studio's model list; <code>auto</code> picks the model with the best measured record for each prompt.</div></section>
</main>
<script>
const $=s=>document.querySelector(s);
fetch('/api/status').then(r=>r.json()).then(j=>{ document.getElementById('me').textContent=j.computer||location.hostname; }).catch(()=>{ document.getElementById('me').textContent=location.hostname; });
document.querySelectorAll('.port').forEach(e=>e.textContent=location.port||(location.protocol==='https:'?'443':'80'));
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function copyBtn(text){const b=document.createElement('button');b.className='cp';b.textContent='copy';b.onclick=async()=>{try{await navigator.clipboard.writeText(text);b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}catch{}};return b}
function table(el,obj){el.innerHTML='';for(const [k,v] of Object.entries(obj)){const tr=document.createElement('tr');tr.innerHTML=`<td>${esc(k)}</td><td><code>${esc(v)}</code></td>`;tr.lastChild.appendChild(copyBtn(String(v)));el.appendChild(tr)}}
// The Model ID row: pick one of this Studio's models; copy copies the pick.
function modelRow(tbl,models){const tr=[...tbl.rows].find(r=>r.cells[0].textContent==='Model ID'); if(!tr||!models) return;
  const td=tr.cells[1]; td.innerHTML=''; const sel=document.createElement('select');
  sel.style.cssText='font:12.5px ui-monospace,monospace;background:var(--code);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:.15rem';
  sel.innerHTML=models.map(m=>`<option>${esc(m)}</option>`).join(''); if(models.length>1) sel.selectedIndex=1;
  const b=copyBtn(''); b.onclick=async()=>{try{await navigator.clipboard.writeText(sel.value);b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}catch{}};
  td.append(sel,b); td.insertAdjacentHTML('beforeend','<div class="note">auto picks per prompt from measured results</div>')}
function block(el,obj){const inner=obj.mcpServers?JSON.stringify(obj.mcpServers,null,2).replace(/^\{\n|\n\}$/g,''):JSON.stringify(obj,null,2);el.textContent=JSON.stringify(obj,null,2);el.before(copyBtn(inner))}
function line(el,text){el.textContent=text;el.before(copyBtn(text))}
document.querySelectorAll('[data-ct]').forEach(b=>b.onclick=()=>{ document.querySelectorAll('[data-ct]').forEach(x=>{ x.classList.toggle('on',x===b); x.setAttribute('aria-selected',x===b); });
  document.querySelectorAll('[data-cp]').forEach(p=>p.hidden=p.dataset.cp!==b.dataset.ct); });
(async()=>{
  const r=await fetch('api/connect'); const j=await r.json();
  if(!r.ok){document.querySelector('main').innerHTML='<div class="card">'+esc(j.error||r.status)+'</div>';return}
  table($('#prov'),j.cline_provider); modelRow($('#prov'),j.models); block($('#mcphttp'),j.cline_mcp_http); block($('#mcpstdio'),j.cline_mcp_stdio);
  block($('#claudeDesk'),j.claude_desktop); line($('#ccHttp'),j.claude_code.http); line($('#ccStdio'),j.claude_code.stdio);
  $('#deskTok').innerHTML=j.token_required?'This Studio needs a token: add <code>"--token-file", "/path/to/a/file/with/the/token"</code> to <code>args</code> (<b>Create a token for an app</b> above makes one; save it in a file only you can read).':''; table($('#oa'),{'Base URL':j.openai.base_url,'API key':j.openai.api_key,'Model':j.openai.model});
  $('#auth').innerHTML=j.token_required
    ?`This Studio requires a token. Use the owner token for full access (jobs, projects), or create a separate one for an app: it can chat, check replies and read status, and you can revoke it under Link. <button class="pri" id="mk">Create a token for an app</button><div id="tokout"></div>`
    :`This Studio is on ${esc(location.host)} with no token: apps on this computer need no key (Cline still wants some text in API Key; any will do).`;
  const mk=$('#mk'); if(mk) mk.onclick=async()=>{const name=prompt('Name for this app token','Cline'); if(!name) return;
    const t=await (await fetch('api/connect/token',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
    $('#tokout').innerHTML=t.token?`<div class="tok">Shown once, copy it now:<br><code>${esc(t.token)}</code></div>`:`<div class="note">${esc(t.error||'failed')}</div>`;
    if(t.token) $('#tokout .tok').appendChild(copyBtn(t.token));};
})();
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
