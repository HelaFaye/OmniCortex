"""Studio's /settings page: workspaces (where models, llama.cpp, pipeline data
and Studio's state live), the model to use per subject, and tool approval
rules for MCP tools in chat. Same as scripts/ns_workspace.py and
scripts/tool_rules.py on the command line."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Settings</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--acc)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{font-size:13px}
nav.toc{display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;font-size:13px}
main{max-width:1050px;margin:0 auto;padding:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem;margin-bottom:1rem;scroll-margin-top:1rem}
.card:target{border-color:var(--acc)}
h2{font-size:15px;margin:.1rem 0 .4rem}h3{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:1rem 0 .4rem}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no);font-size:12.5px;white-space:pre-wrap}.okm{color:var(--ok);font-size:12.5px}
label{display:block;font-size:12.5px;margin:.55rem 0 .15rem}label .note{display:block}
input,select,textarea{font:12.5px ui-monospace,monospace;padding:.3rem .45rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto}textarea{min-height:3rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.38rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}button:disabled{opacity:.5}
.row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}.row>*{min-width:0}.grow{flex:1}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 1rem}@media (max-width:760px){.grid{grid-template-columns:1fr}}
.tabs{display:flex;gap:.3rem;flex-wrap:wrap;margin:.4rem 0}.tab{padding:.25rem .7rem;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px;background:var(--bg)}
.tab.on{border-color:var(--acc);color:var(--acc)}.tab .act{font-size:10.5px;color:var(--ok);margin-left:.3rem}
table{width:100%;border-collapse:collapse}td,th{padding:.35rem .3rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:500}
.miss{color:var(--warn);font-size:11.5px}.dflt{color:var(--mut);font-size:11.5px}
.pill{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 .35rem;white-space:nowrap}
.pill.allow{color:var(--ok)}.pill.deny{color:var(--no)}.pill.ask{color:var(--warn)}
.rule{border:1px solid var(--line);border-radius:6px;padding:.5rem .6rem;margin:.4rem 0;background:var(--bg)}
.rule .row label{margin:0}.cond{display:grid;grid-template-columns:9rem 1fr 6rem 2rem;gap:.3rem;margin-top:.3rem}
code{font:12px ui-monospace,monospace;background:var(--code);padding:.05rem .3rem;border-radius:4px}
.themes{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.6rem;margin-top:.6rem}
.th{border:1px solid var(--line);border-radius:8px;padding:.5rem;cursor:pointer}.th.on{outline:2px solid var(--acc)}
.th .sw{display:flex;height:34px;border-radius:5px;overflow:hidden;margin:.35rem 0}.th .sw span{flex:1}
.cn{display:flex;gap:.7rem;align-items:center;padding:.6rem .2rem;border-bottom:1px solid var(--line)}.cn:last-child{border-bottom:none}
.cn .ic{width:34px;height:34px;flex:none;border-radius:8px;border:1px solid var(--line);display:flex;align-items:center;justify-content:center;color:var(--mut);background:var(--bg)}
.cn .nm{font-weight:600}.cn .sub{font:11.5px ui-monospace,monospace;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cn .st{font-size:12px;white-space:nowrap}.cn .st.on{color:var(--ok)}.cn .st.err{color:var(--no)}.cn .st.off{color:var(--mut)}
.cn .grow{min-width:0}.cn button{font-size:12px;padding:.25rem .6rem}
.sw2{position:relative;width:36px;height:20px;flex:none}.sw2 input{opacity:0;width:0;height:0;position:absolute}
.sw2 span{position:absolute;inset:0;border-radius:999px;background:var(--line);transition:.15s;cursor:pointer}
.sw2 span::after{content:'';position:absolute;left:3px;top:3px;width:14px;height:14px;border-radius:50%;background:var(--panel);transition:.15s}
.sw2 input:checked+span{background:var(--acc)}.sw2 input:checked+span::after{transform:translateX(16px)}.sw2 input:focus-visible+span{outline:2px solid var(--acc);outline-offset:2px}
dialog.dlg{border:1px solid var(--line);border-radius:12px;background:var(--panel);color:var(--fg);padding:0;width:min(560px,94vw);box-shadow:0 12px 40px rgba(0,0,0,.4)}
dialog.dlg::backdrop{background:rgba(0,0,0,.5)}dialog.dlg form{padding:1rem 1.1rem}dialog.dlg h3{margin:0 0 .6rem;font-size:17px;text-transform:none;letter-spacing:0;color:var(--fg)}
.seg{display:flex;border:1px solid var(--line);border-radius:7px;overflow:hidden;margin:.2rem 0 .3rem}.seg label{flex:1;margin:0;text-align:center;padding:.35rem;cursor:pointer;font-size:12.5px;color:var(--mut)}
.seg input{display:none}.seg label:has(input:checked){background:var(--soft,var(--code));color:var(--fg);font-weight:600}
.cnres{font-size:12px;margin:.2rem 0 0 2.8rem}.cnres.ok{color:var(--ok)}.cnres.bad{color:var(--no)}
</style>
<header><h1>Studio · Settings</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a></header>
<main>
<nav class="toc"><a href="#appearance">Appearance</a><a href="#workspaces">Workspaces</a><a href="#subjects">Models per subject</a><a href="#connectors">Connectors</a><a href="#tools">Tool approval</a><a href="#permissions">Permissions</a><a href="#sharing">Sharing results</a><a href="#plugins">Plugins</a><span class="note">Behaviour (unloading, routing thresholds, workers) and building llama.cpp are on <a href="/setup">Setup</a>.</span></nav>

<section class="card" id="appearance"><h2>Appearance</h2>
<div class="note">A theme colours Studio's pages, its charts (the model radar) and the 3D views of activations and hallucination risk. Your choice is kept in this browser; <b>Make default</b> sets it for browsers that have not chosen one, and for the Godot client's 3D view.</div>
<div id="themeGrid" class="themes"></div><div id="themeMsg" class="note"></div>
<h3 id="widgets" style="margin:1rem 0 .3rem">Widgets</h3>
<div class="note">The emotion jewel, the emotion graph and the 3D navigation, in the 3D replay, the live viewer and the Godot client. These are the defaults for every viewer; a viewer's own picks (right-click the jewel for its next style) stay in that browser or client.</div>
<div class="row" style="align-items:flex-start;gap:1rem;margin-top:.5rem"><canvas id="wdPrev" style="width:120px;height:120px;flex:none;background:#05070d;border-radius:10px" title="a preview with made-up emotions"></canvas>
<div id="wdForm" style="flex:1;min-width:16rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(18rem,1fr));gap:.5rem"></div></div>
<div class="row" style="margin-top:.5rem"><button class="pri" id="wdSave">Save widgets</button><button id="wdReset">Back to the theme's</button><span id="wdMsg" class="note"></span></div></section>

<section class="card" id="workspaces"><h2>Workspaces</h2>
<div class="note">Where things are, by name: the models, the llama.cpp build, the folder that holds the pipeline's data (<code>data/</code>, <code>models/</code>, <code>runs/</code>…; jobs run there), Studio's own state, and environment for the GPU stack. Keep one per disk layout or machine, and switch instead of passing paths. Command line: <code>python scripts/ns_workspace.py</code>.</div>
<div class="tabs" id="wstabs"></div>
<div id="wsform"></div>
<div class="row" style="margin-top:.7rem"><button class="pri" id="wssave">Save</button><button id="wsuse">Use this workspace</button>
  <span class="grow"></span><input id="wsnew" placeholder="new workspace name" style="width:13rem"><button id="wsadd">Add</button><button id="wsdup">Duplicate</button><button class="bad" id="wsdel">Delete</button></div>
<div id="wsmsg"></div></section>

<section class="card" id="subjects"><h2>Models per subject</h2>
<div class="note">When a request asks for model <code>auto</code>, Studio sends it to the model measured best for the prompt's subject (from TestQA and benchmark results). Pick a model for a subject to make it the default for that subject instead; leave it on <i>measured best</i> to follow the numbers. Your pick wins whenever that subject is at least 30% of the prompt.</div>
<table id="subjtbl"></table>
<div class="row" style="margin-top:.6rem"><button class="pri" id="subjsave">Save</button><span id="subjmsg"></span></div>
<div class="note" id="subjnote" style="margin-top:.4rem"></div></section>

<section class="card" id="connectors"><h2>Connectors</h2>
<div class="note">MCP servers that give your models tools in chat (tick <b>tools</b> under the chat). Here Studio is the <i>client</i>: a tool runs wherever its server runs. Every call asks first unless <a href="#tools">Tool approval</a> says otherwise. To use OmniCortex's own tools from Claude, Cline or another app, see <a href="/connect#claude">Connect → Claude</a>.</div>
<div class="row" style="margin:.5rem 0"><input id="cnq" placeholder="Search connectors" style="width:16rem"><span class="grow"></span><button class="pri" id="cnadd">+ Add connector</button></div>
<div id="cnlist"></div><div id="cnmsg" class="note"></div><div id="cnpath" class="note"></div>
<dialog class="dlg" id="cndlg" aria-labelledby="cntitle"><form method="dialog" id="cnform">
  <h3 id="cntitle">Add connector</h3>
  <label>Name<input id="cnname" placeholder="e.g. files, github, transfer" autocomplete="off"></label>
  <label>How Studio reaches it</label>
  <div class="seg" role="radiogroup"><label><input type="radio" name="cnkind" value="url" checked> Server URL (HTTP)</label><label><input type="radio" name="cnkind" value="cmd"> Program on this computer</label></div>
  <div id="cnurlbox">
    <label>MCP server URL<input id="cnurl" placeholder="https://host:port/mcp" autocomplete="off"></label>
    <label>Sign-in</label><select id="cnauth"><option value="none">None</option><option value="file">Bearer token from a file</option><option value="env">Bearer token from an environment variable</option><option value="headers">Custom headers</option></select>
    <div id="cnauthfile" hidden><label>Token file<input id="cntokfile" data-pick="file" placeholder="a file holding only the token"></label></div>
    <div id="cnauthenv" hidden><label>Environment variable<input id="cntokenv" placeholder="MY_SERVER_TOKEN"></label></div>
    <div id="cnauthhdr" hidden><label>Headers <span class="note">Name: value, one per row. ${file:/path} and ${env:NAME} are read when Studio connects.</span></label><div id="cnhdr"></div></div>
  </div>
  <div id="cncmdbox" hidden>
    <label>Program<input id="cncmd" data-pick="file" placeholder="npx, uvx, python, or a path"></label>
    <label>Arguments <span class="note">one per row</span></label><div id="cnargs"></div>
    <label>Environment <span class="note">NAME=value, one per row</span></label><div id="cnenv"></div>
    <label>Working folder <span class="note">optional</span><input id="cncwd" data-pick="dir"></label>
  </div>
  <label style="margin-top:.6rem"><input type="checkbox" id="cnauto"> Run this server's tools without asking <span class="note">(only for servers you trust completely)</span></label>
  <div id="cnerr" class="err"></div>
  <div class="row" style="margin-top:.8rem;justify-content:flex-end"><button value="cancel" formnovalidate>Cancel</button><button class="pri" id="cnsave" value="default">Add</button></div>
</form></dialog></section>

<section class="card" id="permissions"><h2>Permissions</h2>
<div class="note">What a model may <b>read</b>, <b>write</b>, <b>run</b> (command lines), <b>exec</b> (programs), reach on the <b>net</b>, touch in the <b>system</b> (GPU, Docker, packages, power…) and which <b>tool</b>s it may call. A <b>profile</b> is a set of rules (allow = whitelist, deny = blacklist; exact, prefix, glob or regex) and a default per action. A check stacks the base profile, the profiles bound to the model, the job's profile and any profile asked for. Every matching rule counts and the strictest wins, so a job can open what it needs but never lift a deny. Patterns may use {workdir}, {repo}, {home}, {tmp}. Chat tool calls, TestQA's code runs, project tasks and graph nodes all check here (docs/PERMISSIONS.md).</div>
<div class="row" style="margin:.6rem 0 .3rem;gap:.5rem;flex-wrap:wrap"><span class="note">profile</span><select id="pmProf"></select><button id="pmCopy" title="A built-in profile cannot be changed; copy it to edit">Copy as…</button><button class="bad" id="pmDel">Delete</button><span class="grow"></span><span class="note">base for every check</span><select id="pmBase"></select></div>
<div id="pmDesc" class="note"></div>
<div id="pmDefaults" class="row" style="flex-wrap:wrap;gap:.6rem;margin:.4rem 0"></div>
<div style="overflow-x:auto"><table id="pmRules"></table></div>
<div class="row" style="margin:.3rem 0"><button id="pmAdd">+ Rule</button></div>
<h3 style="margin:.8rem 0 .2rem">Models</h3><div class="note">Profiles for models whose id matches a glob, e.g. <code>*coder*</code>.</div>
<div style="overflow-x:auto"><table id="pmBind"></table></div><div class="row"><button id="pmBindAdd">+ Model</button></div>
<h3 style="margin:.8rem 0 .2rem">Jobs</h3><div class="note">Each job kind starts with the preset that fits it (TriviaQA and other QA runs: no actions; TestQA code: a sandbox; SWE-bench evaluation: git, Python and Docker in the checkout). Change one here.</div>
<details><summary class="note">job profiles</summary><div style="overflow-x:auto"><table id="pmJobs"></table></div></details>
<h3 style="margin:.8rem 0 .2rem">Try it</h3>
<div class="row" style="flex-wrap:wrap;gap:.4rem"><select id="ptAct"></select><input id="ptTarget" placeholder="target: a path, command line, program, URL, component or server/tool" style="flex:1;min-width:14rem"><input id="ptModel" placeholder="model" list="pmModels" style="width:10rem"><input id="ptJob" placeholder="job kind" list="pmJobList" style="width:9rem"><input id="ptRepo" placeholder="{repo} / {workdir}" style="width:10rem"><button id="ptGo">Check</button></div>
<div id="ptOut" class="note"></div>
<div class="row" style="margin-top:.6rem"><button class="pri" id="pmSave">Save permissions</button><span id="pmMsg"></span></div>
<datalist id="pmModels"></datalist><datalist id="pmJobList"></datalist></section>

<section class="card" id="tools"><h2>Tool approval</h2>
<div class="note">When a model calls an MCP tool in chat (Studio → Tools), each call is decided in order: <b>1</b> your rules below, first match wins; <b>2</b> a server's <code>autoApprove</code> in mcp.json; <b>3</b> the quick setting for the tool's kind of operation (from its MCP annotations); <b>4</b> the default. <i>Ask</i> shows Allow/Deny in the chat, where <i>always…</i> adds a rule here. Command line: <code>python scripts/tool_rules.py</code>.</div>
<h3>Quick settings</h3>
<div class="grid" id="cats"></div>
<h3>Rules (first match wins)</h3>
<div id="rules"></div>
<div class="row"><button id="radd">Add a rule</button><button id="rpreset">Add common rules…</button><span class="grow"></span><button class="pri" id="rsave">Save rules</button><span id="rmsg"></span></div>
<h3>Try a call</h3>
<div class="row"><input id="tserver" placeholder="server" style="width:9rem"><input id="ttool" placeholder="tool" style="width:11rem"><input id="targs" class="grow" placeholder='arguments JSON, e.g. {"path": "/home/me/notes/a.md"}'>
<select id="tann" style="width:auto"><option value="{}">no annotations</option><option value='{"readOnlyHint":true}'>read-only</option><option value='{"readOnlyHint":false,"destructiveHint":false}'>writes</option><option value='{"readOnlyHint":false,"destructiveHint":true}'>destructive</option><option value='{"readOnlyHint":true,"openWorldHint":true}'>read-only, open world</option></select><button id="ttest">Test</button></div>
<div id="tout" class="note"></div>
<h3>Connected tools</h3><div id="tlist" class="note"></div></section>
<section class="card" id="plugins"><h2>Plugins</h2><div class="note">Install packages from a file or a repository, and remove parts of Studio you do not use, under <a href="/addons">Add-ons</a>.</div>
<div class="note">A plugin is a folder with a <code>plugin.json</code>. It can add jobs, checks under chat replies, a page of its own and MCP tools. <b>Plugins run code on this computer</b>, so none is used until you turn it on. If a plugin's files change after that, it turns itself off until you look at it again. See docs/PLUGINS.md to write one.</div>
<div id="plDirs" class="note"></div><div style="overflow-x:auto"><table id="plList"></table></div><div id="plMsg" class="note"></div></section>

<section class="card" id="sharing"><h2>Sharing results</h2>
<div class="note"><b>Off unless you turn it on.</b> When on, TestQA results can be sent to a scoreboard you name, so models can be compared across machines and setups. OmniCortex runs no scoreboard of its own; anyone can host <code>scripts/scoreboard_server.py</code>, or keep one in a GitHub repository (<code>scripts/github_board.py init</code>).</div>
<details open style="margin:.5rem 0"><summary><b>What is sent, and what it is for</b></summary>
<ul class="note">
<li><b>A card, which may be public</b>:
  <ul>
  <li>the model file's SHA-256, size, name (not its path), quantization, and its Hugging Face repo if the file is verified to come from one;</li>
  <li>what makes this setup different, so results are never mixed: the hash of the H-Neuron set and the suppression scale, the LoRA's hash and scale, fine-tune / merge / concept-edit lineage, the engine and its version, the engine options that can change answers (context, KV cache, speculative decoding, flash attention), and the sampling;</li>
  <li>coarse hardware (OS, CPU model, cores, RAM and GPU memory in whole GB, GPU names);</li>
  <li>the test bank's fingerprint and the right / wrong / declined counts per subject.</li>
  </ul></li>
<li><b>A verification log, private</b>: per test item, its id, the verdict and the model's reply. The scoreboard uses it only to check the card by grading the replies again; it must not publish it.</li>
<li><b>Never sent</b>: your chats, prompts you typed, file paths, user or computer names, tokens, other models.</li>
<li><b>Public only for models anyone can find</b>: the file's SHA-256 matches a file in its Hugging Face repo, or you list an unlisted model below. Everything else is private, or not sent.</li>
<li>Every submission is kept on this computer exactly as sent (<code>~/.neuronscope/submissions/</code>), and listed below.</li>
</ul></details>
<label><input type="checkbox" id="shOn"> Share results</label>
<label title="A scoreboard server's address, or github:owner/repo for a scoreboard kept in a GitHub repository">Scoreboard<input id="shEp" placeholder="https://scores.example.org or github:owner/repo"></label>
<div id="shGh" style="display:none"><label title="Stored on this computer only (0600). Needs permission to open issues on the board's repository; a fine-grained token limited to that repository is enough.">GitHub token, to open the submission issue <input id="shTok" type="password" autocomplete="off"></label>
<div class="note">On a GitHub board everything accepted is public, so only cards for public models are sent. The verification log is sealed to the board's key: only the board's own workflow can read it, and it is never committed. <span id="shTokSt"></span></div></div>
<label><input type="checkbox" id="shAuto"> Submit automatically when a TestQA run finishes, for models found on Hugging Face</label>
<details style="margin:.4rem 0"><summary class="note">Advanced</summary>
<label><input type="checkbox" id="shUnl"> Also submit results for unlisted models automatically <span class="note">(sent privately: only the scoreboard's operator can see them, unless you list the model below)</span></label></details>
<div class="row" style="margin-top:.4rem"><button class="pri" id="shSave">Save</button><span id="shMsg" class="note"></span></div>
<h3>Models</h3><div class="note">Found on Hugging Face = this exact file is in the repo it came from (checking reads the whole file once).</div>
<table id="shModels"></table>
<h3>Submit a result by hand</h3>
<div class="grid"><label>Results (TestQA runs from Jobs)<select id="shRes"></select></label><label>Model<select id="shModel"></select></label></div>
<div class="row" style="margin-top:.4rem"><button id="shPrev">Preview exactly what is sent</button><button id="shSend">Submit</button></div>
<pre id="shOut" class="note" style="max-height:22rem;overflow:auto;white-space:pre-wrap"></pre>
<h3>Sent so far</h3><table id="shHist"></table>
</section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const msg=(el,t,bad)=>{$(el).innerHTML=t?`<span class="${bad?'err':'okm'}">${t}</span>`:''};
// ---------- connectors (MCP servers Studio's chat can use)
let CN=null, cnEdit=null, cnRes={}, cnLists={};
const CN_IC={http:'<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3.5 3.5 3.5 14.5 0 18M12 3c-3.5 3.5-3.5 14.5 0 18"/></svg>',
  stdio:'<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M4 5h16v14H4zM7 10l3 2-3 2M12 15h5"/></svg>'};
async function loadCN(){ try{ CN=await api('/api/connectors'); drawCN(); }catch(e){ $('#cnlist').innerHTML=`<div class="err">${esc(e.message)}</div>`; } }
function cnSub(c){ const s=c.spec||{}; return s.url||[s.command,...(s.args||[])].filter(Boolean).join(' '); }
function cnState(c){ if(c.disabled) return '<span class="st off">off</span>'; if(c.error) return `<span class="st err" title="${esc(c.error)}">● not connected</span>`;
  return c.connected?`<span class="st on">● ${c.tools.length} tool${c.tools.length===1?'':'s'}</span>`:'<span class="st off">connecting…</span>'; }
function drawCN(){ const q=$('#cnq').value.toLowerCase(); const hit=c=>!q||c.name.toLowerCase().includes(q)||cnSub(c).toLowerCase().includes(q);
  const rows=CN.servers.filter(hit).map(c=>`<div class="cn" data-cn="${esc(c.name)}"><div class="ic">${CN_IC[c.transport]}</div>
    <div class="grow" style="flex:1"><div class="nm">${esc(c.name)}</div><div class="sub" title="${esc(cnSub(c))}">${esc(cnSub(c))}</div></div>
    ${cnState(c)}<button data-a="test" ${c.disabled?'disabled':''}>Test</button><button data-a="edit">Edit</button><button data-a="rm" class="bad" aria-label="Remove ${esc(c.name)}">Remove</button>
    <label class="sw2" title="${c.disabled?'Turn on':'Turn off'}"><input type="checkbox" data-a="on" ${c.disabled?'':'checked'} aria-label="${esc(c.name)} on"><span></span></label></div>
    ${cnRes[c.name]?`<div class="cnres ${cnRes[c.name].ok?'ok':'bad'}">${esc(cnRes[c.name].text)}</div>`:''}${c.error&&!c.disabled?`<div class="cnres bad">${esc(c.error)}</div>`:''}`).join('');
  const plug=CN.plugins.filter(hit).map(c=>`<div class="cn"><div class="ic">${CN_IC[c.transport]}</div><div class="grow" style="flex:1"><div class="nm">${esc(c.name)}</div><div class="sub">from a plugin · <a href="#plugins">Plugins</a></div></div>${cnState(c)}</div>`).join('');
  $('#cnlist').innerHTML=rows+plug||(CN.servers.length?'<div class="note">No connector matches.</div>':'<div class="note" style="padding:.6rem 0">No connectors yet. <b>+ Add connector</b> connects a server by its URL, or starts a program on this computer (most MCP servers you find are started with <code>npx</code> or <code>uvx</code>).</div>');
  $('#cnpath').innerHTML=`Saved in <code>${esc(CN.config)}</code>${CN.exists?'':' (created when you add one)'}.`+(CN.error?` <span class="err">${esc(CN.error)}</span>`:'');
  document.querySelectorAll('#cnlist [data-cn]').forEach(row=>{ const name=row.dataset.cn, c=CN.servers.find(x=>x.name===name);
    row.querySelector('[data-a=edit]').onclick=()=>cnOpen(c);
    row.querySelector('[data-a=rm]').onclick=async()=>{ if(!confirm(`Remove ${name}? Its tools stop being offered in chat.`)) return; try{ CN=await api('/api/connectors/remove',{name}); delete cnRes[name]; drawCN(); }catch(e){ msg('#cnmsg',esc(e.message),true); } };
    row.querySelector('[data-a=on]').onchange=async e=>{ try{ CN=await api('/api/connectors/toggle',{name,disabled:!e.target.checked}); drawCN(); }catch(err){ msg('#cnmsg',esc(err.message),true); } };
    row.querySelector('[data-a=test]').onclick=()=>cnTest(name); }); }
async function cnTest(name){ cnRes[name]={ok:true,text:'testing…'}; drawCN();
  try{ const r=await api('/api/mcp/test',{server:name});
    cnRes[name]=r.ok?{ok:true,text:`Works: ${r.tools} tool${r.tools===1?'':'s'} in ${r.seconds}s`+(r.called?`; called ${r.called}: ${r.call_ok?'ok':'failed'}`:'')}:{ok:false,text:r.error||'failed'}; }
  catch(e){ cnRes[name]={ok:false,text:e.message}; } drawCN(); }
$('#cnq').oninput=()=>CN&&drawCN();
const cnKind=()=>document.querySelector('[name=cnkind]:checked').value;
function cnShow(){ const k=cnKind(); $('#cnurlbox').hidden=k!=='url'; $('#cncmdbox').hidden=k!=='cmd'; const a=$('#cnauth').value;
  $('#cnauthfile').hidden=a!=='file'; $('#cnauthenv').hidden=a!=='env'; $('#cnauthhdr').hidden=a!=='headers'; }
document.querySelectorAll('[name=cnkind]').forEach(r=>r.onchange=cnShow); $('#cnauth').onchange=cnShow;
function cnOpen(c){ cnEdit=c||null; const s=(c&&c.spec)||{}; $('#cntitle').textContent=c?`Edit ${c.name}`:'Add connector'; $('#cnsave').textContent=c?'Save':'Add'; $('#cnerr').textContent='';
  $('#cnname').value=c?c.name:''; document.querySelector(`[name=cnkind][value=${s.command?'cmd':'url'}]`).checked=true;
  $('#cnurl').value=s.url||''; $('#cntokfile').value=s.token_file||''; $('#cntokenv').value=s.token_env||'';
  $('#cnauth').value=s.token_file?'file':s.token_env?'env':s.headers?'headers':'none';
  $('#cncmd').value=s.command||''; $('#cncwd').value=s.cwd||''; $('#cnauto').checked=!!s.autoApprove;
  cnLists.hdr=nsListEditor($('#cnhdr'),{values:Object.entries(s.headers||{}).map(([k,v])=>`${k}: ${v}`),browse:false,addLabel:'+ Add header',placeholder:'Authorization: Bearer ${file:/path/to/token}'});
  cnLists.args=nsListEditor($('#cnargs'),{values:s.args||[],browse:false,addLabel:'+ Add argument',placeholder:'-y'});
  cnLists.env=nsListEditor($('#cnenv'),{values:Object.entries(s.env||{}).map(([k,v])=>`${k}=${v}`),browse:false,addLabel:'+ Add variable',placeholder:'NAME=value'});
  cnShow(); $('#cndlg').showModal(); $('#cnname').focus(); }
$('#cnadd').onclick=()=>cnOpen(null);
const kvs=(rows,sep)=>{ const o={}; for(const r of rows){ const i=r.indexOf(sep); if(i<1) throw new Error(`"${r}": expected NAME${sep}value`); o[r.slice(0,i).trim()]=r.slice(i+1).trim(); } return o; };
$('#cnsave').onclick=async e=>{ e.preventDefault(); $('#cnerr').textContent='';
  try{ const spec={autoApprove:$('#cnauto').checked,disabled:cnEdit?cnEdit.disabled:false};
    if(cnKind()==='url'){ spec.url=$('#cnurl').value.trim(); const a=$('#cnauth').value;
      if(a==='file') spec.token_file=$('#cntokfile').value.trim(); if(a==='env') spec.token_env=$('#cntokenv').value.trim(); if(a==='headers') spec.headers=kvs(cnLists.hdr.get(),':'); }
    else { spec.command=$('#cncmd').value.trim(); spec.args=cnLists.args.get(); spec.env=kvs(cnLists.env.get(),'='); const cwd=$('#cncwd').value.trim(); if(cwd) spec.cwd=cwd; }
    const name=$('#cnname').value.trim(); $('#cnsave').disabled=true;
    CN=await api('/api/connectors/save',{name,old_name:cnEdit?cnEdit.name:undefined,spec}); $('#cndlg').close(); drawCN(); if(!spec.disabled) cnTest(name); }
  catch(err){ $('#cnerr').textContent=err.message; } finally{ $('#cnsave').disabled=false; } };
async function restartBtn(el){ $(el).insertAdjacentHTML('beforeend',' <button id="rs">Restart Studio now</button>');
  $('#rs').onclick=async()=>{ await api('/api/restart',{}).catch(()=>{}); msg(el,'restarting…'); setTimeout(()=>location.reload(),4000); }; }

// ---------- workspaces
let WS=null, cur=null;
const LABEL={root:'Pipeline data root',models_dirs:'Models folders',download_dir:'Downloads folder',llama_dir:'llama.cpp checkout',
  server:'llama-server',cett:'llama-cett-dump',gguf:'Model for the H-Neuron pipeline (.gguf)',state_dir:'Studio state folder',env:'Environment variables',note:'Note'};
function wsTabs(){ $('#wstabs').innerHTML=WS.workspaces.map(w=>`<span class="tab ${w.name===cur?'on':''}" data-w="${esc(w.name)}">${esc(w.name)}${w.active?'<span class="act">active</span>':''}${w.name===WS.running&&!w.active?'<span class="act" style="color:var(--warn)">running</span>':''}</span>`).join('');
  document.querySelectorAll('[data-w]').forEach(t=>t.onclick=()=>{cur=t.dataset.w;wsTabs();wsForm();}); }
function wsForm(){
  const w=WS.workspaces.find(x=>x.name===cur), v=w.values, r=w.resolved, ex=w.exists;
  const field=k=>{ const f=WS.fields[k]; let val=v[k];
    if(k==='models_dirs') val=(val||[]).join('\n'); if(k==='env') val=Object.entries(val||{}).map(([a,b])=>a+'='+b).join('\n');
    const dflt=k==='models_dirs'?r.models_dirs.join(', '):k==='env'?'':r[k]||'';
    const missing=k==='models_dirs'?(ex.models_dirs.every(x=>!x)?'none of these folders exists':''):(k in ex&&r[k]&&!ex[k]?'does not exist':'');
    const big=['models_dirs','env'].includes(k), pick={root:'dir',download_dir:'dir',llama_dir:'dir',state_dir:'dir',gguf:'file',server:'file',cett:'file'}[k];
    return `<label>${esc(LABEL[k]||k)}<span class="note">${esc(f.help)}</span></label>`+
      (big?`<textarea data-f="${k}" hidden>${esc(val||'')}</textarea><div data-wl="${k}" data-dflt="${esc(dflt)}"></div>`:`<input data-f="${k}" value="${esc(val||'')}" placeholder="${esc(dflt?'default: '+dflt:'')}"${pick?` data-pick="${pick}"`:''}${k==='gguf'?' data-ext="gguf"':''}>`)+
      `${missing?`<div class="miss">${esc(missing)}${k==='server'||k==='cett'?' · <a href="/setup#build">build llama.cpp</a>':''}</div>`:''}<div data-err="${k}" class="err"></div>`; };
  $('#wsform').innerHTML=`<div class="grid"><div>${['root','models_dirs','download_dir','gguf','note'].map(field).join('')}</div><div>${['llama_dir','server','cett','state_dir','env'].map(field).join('')}</div></div>`+
    `<div class="note" style="margin-top:.5rem">Saved in <code>${esc(WS.config)}</code>. ${WS.env_override?`Studio was started with workspace ${esc(WS.env_override)} (--workspace or NS_WORKSPACE); “active” applies when it starts without one.`:''} Empty fields use the default shown in grey.</div>`;
  // Per-entry lists, synced into the hidden fields wsValues reads.
  document.querySelectorAll('[data-wl]').forEach(d=>{ const k=d.dataset.wl, ta=document.querySelector(`#wsform textarea[data-f=${k}]`), dirs=k==='models_dirs';
    nsListEditor(d,{values:ta.value.split('\n').map(x=>x.trim()).filter(Boolean),want:'dir',browse:dirs,addLabel:dirs?'+ Add folder':'+ Add variable',
      placeholder:dirs?'':'NAME=value',empty:dirs?(d.dataset.dflt?'None set: the default is '+d.dataset.dflt:'None set.'):'None set.',onchange:v=>ta.value=v.join('\n')}); });
  $('#wsuse').disabled=w.active; $('#wsdel').disabled=w.name==='default'||w.name===WS.running;
}
function wsValues(){ const out={}; document.querySelectorAll('#wsform [data-f]').forEach(e=>{ const k=e.dataset.f; let v=e.value.trim();
    if(k==='models_dirs') v=v?v.split('\n').map(x=>x.trim()).filter(Boolean):null;
    else if(k==='env'){ const o={}; for(const line of v.split('\n')){ const t=line.trim(); if(!t) continue; const i=t.indexOf('='); if(i<1) throw new Error('environment lines are VAR=value'); o[t.slice(0,i).trim()]=t.slice(i+1).trim(); } v=Object.keys(o).length?o:null; }
    else v=v||null; out[k]=v; }); return out; }
async function loadWS(){ WS=await api('/api/workspaces'); if(!cur||!WS.workspaces.some(w=>w.name===cur)) cur=WS.active; wsTabs(); wsForm(); }
$('#wssave').onclick=async()=>{ document.querySelectorAll('[data-err]').forEach(e=>e.textContent=''); try{ const r=await api('/api/workspaces/save',{name:cur,values:wsValues(),replace:true}); WS=r; wsTabs(); wsForm();
  msg('#wsmsg','saved'+(r.restart_needed?' · Studio uses the new paths after a restart':'')); if(r.restart_needed) restartBtn('#wsmsg'); }
  catch(e){ msg('#wsmsg',esc(e.message),true); for(const part of e.message.split('; ')){ const [k,...m]=part.split(': '); const el=document.querySelector(`[data-err="${k}"]`); if(el) el.textContent=m.join(': '); } } };
$('#wsuse').onclick=async()=>{ try{ const r=await api('/api/workspaces/use',{name:cur}); WS=r; wsTabs(); wsForm(); msg('#wsmsg',`${esc(cur)} is now the active workspace`+(r.restart_needed?'; restart Studio to switch to it':'')); if(r.restart_needed) restartBtn('#wsmsg'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };
$('#wsadd').onclick=async()=>{ const n=$('#wsnew').value.trim(); if(!n) return msg('#wsmsg','name the new workspace first',true);
  try{ await api('/api/workspaces/save',{name:n,values:{}}); cur=n; $('#wsnew').value=''; await loadWS(); msg('#wsmsg','added; fill in its folders and Save'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };
$('#wsdup').onclick=async()=>{ const n=$('#wsnew').value.trim(); if(!n) return msg('#wsmsg','type a name for the copy',true);
  try{ const w=WS.workspaces.find(x=>x.name===cur); await api('/api/workspaces/save',{name:n,values:w.values}); cur=n; $('#wsnew').value=''; await loadWS(); msg('#wsmsg','copied'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };
$('#wsdel').onclick=async()=>{ if(!confirm(`Delete workspace ${cur}? Its files stay; only the setting goes.`)) return; try{ await api('/api/workspaces/delete',{name:cur}); cur=null; await loadWS(); msg('#wsmsg','deleted'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };

// ---------- subjects and tools
let ST=null, RULES=null;
function subjTable(){
  const s=ST.subjects, opts=sel=>`<option value="">measured best</option>`+ST.models.map(m=>`<option ${m===sel?'selected':''}>${esc(m)}</option>`).join('');
  $('#subjtbl').innerHTML=`<tr><th>subject</th><th>measured best</th><th>default for this subject</th></tr>`+Object.entries(s).map(([k,v])=>
    `<tr><td><b>${esc(k)}</b></td><td>${v.best?`${esc(v.best)}<div class="dflt">${v.subject_n?`${v.subject_n} graded ${esc(k)} items`:'no '+esc(k)+' items yet: overall numbers'}</div>`:`<span class="dflt">no model has ${ST.min_graded} graded results yet · <a href="/jobs#kind=testqa">run TestQA</a></span>`}</td>
      <td><select data-s="${esc(k)}">${opts(v.chosen)}</select>${v.chosen_missing?'<div class="miss">this model is not in the models folders any more</div>':''}</td></tr>`).join('');
  $('#subjnote').innerHTML=`Measured best needs ${ST.min_graded} graded results per model (Setup → <a href="/setup#paths">routing thresholds</a>). Per-model results: <a href="/">Studio → Models</a>.`;
}
$('#subjsave').onclick=async()=>{ const v={}; document.querySelectorAll('[data-s]').forEach(e=>v[e.dataset.s]=e.value||null);
  try{ const r=await api('/api/settings/subjects',{subject_models:v}); ST.subjects=r.subjects; subjTable(); msg('#subjmsg','saved; applies to the next request'); }catch(e){ msg('#subjmsg',esc(e.message),true); } };

const KLABEL={read_only:'Read-only tools',write:'Tools that write (not destructive)',destructive:'Destructive tools (or not marked otherwise)',open_world:'Tools that reach the outside world',unknown:'Tools with no annotations'};
function cats(){ $('#cats').innerHTML=`<div><label>When nothing else decides</label><select id="dflt">${['ask','allow','deny'].map(a=>`<option ${RULES.default===a?'selected':''}>${a}</option>`).join('')}</select></div>`+
  ST.tool_kinds.map(k=>`<div><label>${esc(KLABEL[k])}</label><select data-c="${k}">${['default','ask','allow','deny'].map(a=>`<option value="${a}" ${RULES.categories[k]===a?'selected':''}>${a==='default'?'same as the default':a}</option>`).join('')}</select></div>`).join(''); }
function ruleHtml(r,i){
  const c=(x,j)=>`<div class="cond" data-c="${j}"><input data-k="path" value="${esc(x.path)}" placeholder="argument path, * = any"><input data-k="regex" value="${esc(x.regex)}" placeholder="regex"><label><input type="checkbox" data-k="negate" ${x.negate?'checked':''}> not</label><button data-cdel="${j}" title="remove">×</button></div>`;
  const exp=r.expires?new Date(r.expires*1000).toISOString().slice(0,16):'';
  return `<div class="rule" data-r="${i}"><div class="row"><b>${i+1}.</b><select data-k="action" style="width:auto">${['allow','ask','deny'].map(a=>`<option ${r.action===a?'selected':''}>${a}</option>`).join('')}</select>
    <input data-k="server" value="${esc(r.server)}" placeholder="server glob" style="width:9rem"><input data-k="tool" value="${esc(r.tool)}" placeholder="tool glob" style="width:11rem">
    <select data-k="kind" style="width:auto">${['any',...ST.tool_kinds].map(k=>`<option value="${k}" ${r.kind===k?'selected':''}>${k==='any'?'any kind':k.replace('_',' ')}</option>`).join('')}</select>
    <label><input type="checkbox" data-k="enabled" ${r.enabled?'checked':''}> on</label>
    <span class="grow"></span><span class="note">${r.hits||0} hits</span><button data-up="${i}" title="earlier">↑</button><button data-down="${i}" title="later">↓</button><button class="bad" data-del="${i}">Delete</button></div>
    ${(r.args||[]).map(c).join('')}
    <div class="row" style="margin-top:.3rem"><button data-cadd="${i}">+ argument condition</button><input data-k="max_per_hour" type="number" min="1" value="${r.max_per_hour||''}" placeholder="max calls / hour" style="width:9rem" title="allow rules only: after N calls in an hour it asks">
      <input data-k="expires" type="datetime-local" value="${esc(exp)}" style="width:13rem" title="stops applying after this"><input data-k="note" value="${esc(r.note||'')}" placeholder="note" class="grow"></div></div>`; }
function readRules(){ const out={default:$('#dflt').value,categories:{},rules:[]};
  document.querySelectorAll('[data-c]').forEach(e=>{ if(e.tagName==='SELECT') out.categories[e.dataset.c]=e.value; });
  document.querySelectorAll('.rule').forEach((el,i)=>{ const g=k=>el.querySelector(`:scope > .row [data-k="${k}"], :scope > .row > [data-k="${k}"]`);
    const old=RULES.rules[+el.dataset.r]||{};
    const r={id:old.id,hits:old.hits,recent:old.recent,action:g('action').value,server:g('server').value||'*',tool:g('tool').value||'*',kind:g('kind').value,enabled:g('enabled').checked,
      note:el.querySelector('[data-k="note"]').value,args:[]};
    const mph=el.querySelector('[data-k="max_per_hour"]').value; if(mph) r.max_per_hour=+mph;
    const ex=el.querySelector('[data-k="expires"]').value; if(ex) r.expires=new Date(ex).getTime()/1000;
    el.querySelectorAll('.cond').forEach(c=>r.args.push({path:c.querySelector('[data-k="path"]').value||'*',regex:c.querySelector('[data-k="regex"]').value,negate:c.querySelector('[data-k="negate"]').checked}));
    out.rules.push(r); }); return out; }
function drawRules(){ $('#rules').innerHTML=RULES.rules.map(ruleHtml).join('')||'<div class="note">No rules yet: every call follows the quick settings and the default.</div>';
  const redo=f=>b=>b.onclick=()=>{ RULES=readRules(); f(+(b.dataset.del??b.dataset.up??b.dataset.down??b.dataset.cadd)); drawRules(); };
  document.querySelectorAll('[data-del]').forEach(redo(i=>RULES.rules.splice(i,1)));
  document.querySelectorAll('[data-up]').forEach(redo(i=>{ if(i>0) [RULES.rules[i-1],RULES.rules[i]]=[RULES.rules[i],RULES.rules[i-1]]; }));
  document.querySelectorAll('[data-down]').forEach(redo(i=>{ if(i<RULES.rules.length-1) [RULES.rules[i+1],RULES.rules[i]]=[RULES.rules[i],RULES.rules[i+1]]; }));
  document.querySelectorAll('[data-cadd]').forEach(redo(i=>RULES.rules[i].args.push({path:'*',regex:'',negate:false})));
  document.querySelectorAll('[data-cdel]').forEach(b=>b.onclick=()=>{ RULES=readRules(); const i=+b.closest('.rule').dataset.r; RULES.rules[i].args.splice(+b.dataset.cdel,1); drawRules(); }); }
const PRESETS=[
  {action:'deny',server:'*',tool:'*',args:[{path:'*',regex:'(?i)\\brm\\s+-rf\\b|mkfs|dd\\s+if=|:\\(\\)\\s*\\{',negate:false}],note:'destructive shell commands, anywhere in the arguments'},
  {action:'deny',server:'*',tool:'*',args:[{path:'*',regex:'(?i)(/\\.ssh/|/\\.gnupg/|id_rsa|\\.env\\b|/etc/shadow)',negate:false}],note:'secrets and keys'},
  {action:'allow',server:'*',tool:'*',kind:'read_only',args:[],max_per_hour:200,note:'read-only tools, up to 200 an hour'},
  {action:'allow',server:'filesystem',tool:'read_*',args:[{path:'path',regex:'^/home/[^/]+/(notes|src)/',negate:false}],note:'example: reading notes and source'},
];
$('#radd').onclick=()=>{ RULES=readRules(); RULES.rules.push({action:'allow',server:'*',tool:'*',kind:'any',args:[],enabled:true,note:''}); drawRules(); };
$('#rpreset').onclick=()=>{ RULES=readRules(); for(const p of PRESETS) RULES.rules.push({kind:'any',enabled:true,...JSON.parse(JSON.stringify(p))}); drawRules(); msg('#rmsg','added; check them, then Save'); };
$('#rsave').onclick=async()=>{ try{ const r=await api('/api/settings/tool-rules',{rules:readRules()}); RULES=r.tool_rules; cats(); drawRules(); msg('#rmsg','saved'); loadTools(); }catch(e){ msg('#rmsg',esc(e.message),true); } };
$('#ttest').onclick=async()=>{ try{ const r=await api('/api/settings/tool-rules/test',{server:$('#tserver').value,tool:$('#ttool').value,args:$('#targs').value||'{}',annotations:JSON.parse($('#tann').value)});
  $('#tout').innerHTML=`<span class="pill ${r.action}">${r.action}</span> ${esc(r.reason)} · kinds: ${esc(r.kinds.join(', '))} <span class="note">(saved rules; Save first to test edits)</span>`; }catch(e){ $('#tout').innerHTML=`<span class="err">${esc(e.message)}</span>`; } };
async function loadTools(){ const s=await api('/api/settings'); ST.tools=s.tools;
  $('#tlist').innerHTML=!s.mcp_configured?'No MCP servers yet: add them in <a href="/#tools">Studio → Tools</a>.':!s.tools.length?'No server connected right now (<a href="/#tools">Studio → Tools</a>).':
    `<table><tr><th>server</th><th>tool</th><th>kinds</th><th>with no arguments</th></tr>`+s.tools.map(t=>`<tr><td>${esc(t.server)}</td><td>${esc(t.tool)}</td><td>${esc(t.decision.kinds.join(', '))}</td><td><span class="pill ${t.decision.action}">${t.decision.action}</span> <span class="note">${esc(t.decision.reason)}</span></td></tr>`).join('')+'</table>'; }
(async()=>{
  loadCN();
  try{ await loadWS(); }catch(e){ $('#wsform').innerHTML=`<div class="err">${esc(e.message)}</div>`; }
  try{ ST=await api('/api/settings'); RULES=ST.tool_rules; subjTable(); cats(); drawRules(); loadTools(); }catch(e){ $('#subjtbl').innerHTML=`<tr><td class="err">${esc(e.message)}</td></tr>`; }
  if(location.hash){ const t=document.querySelector(location.hash); if(t) t.scrollIntoView(); }
})();

async function themes(){ const j=await (await fetch('/api/themes')).json(); const cur=nsThemeName();
  $('#themeGrid').innerHTML=Object.entries(j.themes).map(([k,t])=>{ const u=t.ui||{}, c=t.chart||[];
    return `<div class="th${k===cur?' on':''}" data-th="${k}"><b>${esc(t.label||k)}</b>${k===j.default?' <span class="note">(default)</span>':''}
      <div class="sw">${[u.bg,u.panel,u.acc,u.ok,u.no].filter(Boolean).map(x=>`<span style="background:${x}"></span>`).join('')||'<span class="note" style="padding:.4rem">built-in '+esc(t.base)+'</span>'}</div>
      <div class="sw" style="height:12px">${c.map(x=>`<span style="background:${x}"></span>`).join('')}</div>
      <div class="note">3D view: ${esc(t.viz||'dark')}</div>
      <div class="row" style="margin-top:.3rem"><button data-def="${k}">Make default</button></div></div>`; }).join('');
  document.querySelectorAll('[data-th]').forEach(el=>el.onclick=e=>{ if(e.target.dataset.def) return; nsApplyTheme(el.dataset.th, j.themes[el.dataset.th]); themes(); });
  document.querySelectorAll('[data-def]').forEach(b=>b.onclick=async()=>{ const r=await fetch('/api/settings/theme',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({theme:b.dataset.def})});
    $('#themeMsg').textContent=r.ok?'Default theme saved.':(await r.json()).error; themes(); }); }
themes();

// ---- permissions (scripts/permissions.py)
let PM=null, pmCur='';
const opt=(vals,cur)=>vals.map(v=>`<option${v===cur?' selected':''}>${esc(v)}</option>`).join('');
function pmProfiles(){ return Object.assign({}, PM.presets, PM.policy.profiles); }
function pmDraw(){ const all=pmProfiles(), names=Object.keys(all), user=PM.policy.profiles, acts=Object.keys(PM.actions);
  if(!all[pmCur]) pmCur=PM.policy.base;
  $('#pmProf').innerHTML=names.map(n=>`<option value="${esc(n)}"${n===pmCur?' selected':''}>${esc(n)}${user[n]?'':' (built-in)'}</option>`).join('');
  $('#pmBase').innerHTML=opt(names, PM.policy.base);
  const p=all[pmCur], ro=!user[pmCur]; $('#pmDel').disabled=ro; $('#pmAdd').disabled=ro;
  $('#pmDesc').textContent=(p.description||'')+(ro?' Built in: copy it to change it.':'');
  $('#pmDefaults').innerHTML=acts.map(a=>`<label title="${esc(PM.actions[a])}">${a} <select data-def="${a}"${ro?' disabled':''}><option value="">(inherit)</option>${opt(PM.effects, p.defaults[a]||'')}</select></label>`).join('');
  $('#pmRules').innerHTML='<tr><th>action</th><th>effect</th><th>match</th><th>pattern</th><th>note</th><th></th></tr>'+p.rules.map((r,i)=>`<tr data-i="${i}">
    <td><select data-k="action"${ro?' disabled':''}>${opt(['*',...acts], r.action)}</select></td><td><select data-k="effect"${ro?' disabled':''}>${opt(PM.effects, r.effect)}</select></td>
    <td><select data-k="match"${ro?' disabled':''}>${opt(PM.matches, r.match)}</select></td><td><input data-k="pattern" value="${esc(r.pattern)}"${ro?' disabled':''} style="width:100%;min-width:12rem"></td>
    <td><input data-k="note" value="${esc(r.note||'')}"${ro?' disabled':''}></td><td>${ro?'':'<button data-rm>×</button>'}</td></tr>`).join('');
  if(!ro){ $('#pmRules').querySelectorAll('tr[data-i]').forEach(tr=>{ const r=p.rules[+tr.dataset.i];
      tr.querySelectorAll('[data-k]').forEach(el=>el.oninput=()=>{ r[el.dataset.k]=el.value; });
      tr.querySelector('[data-rm]').onclick=()=>{ p.rules.splice(+tr.dataset.i,1); pmDraw(); }; });
    $('#pmDefaults').querySelectorAll('[data-def]').forEach(el=>el.onchange=()=>{ if(el.value) p.defaults[el.dataset.def]=el.value; else delete p.defaults[el.dataset.def]; }); }
  $('#pmBind').innerHTML='<tr><th>model (glob)</th><th>profile</th><th></th></tr>'+PM.policy.bindings.map((b,i)=>`<tr data-i="${i}"><td><input data-k="model" value="${esc(b.model)}" list="pmModels"></td><td><select data-k="profile">${opt(names,b.profile)}</select></td><td><button data-rm>×</button></td></tr>`).join('');
  $('#pmBind').querySelectorAll('tr[data-i]').forEach(tr=>{ const b=PM.policy.bindings[+tr.dataset.i];
    tr.querySelectorAll('[data-k]').forEach(el=>el.oninput=()=>{ b[el.dataset.k]=el.value; });
    tr.querySelector('[data-rm]').onclick=()=>{ PM.policy.bindings.splice(+tr.dataset.i,1); pmDraw(); }; });
  const kinds=[...new Set([...Object.keys(PM.job_presets), ...PM.job_kinds])].sort();
  $('#pmJobs').innerHTML='<tr><th>job</th><th>preset</th><th>use</th></tr>'+kinds.map(k=>`<tr><td>${esc(k)}</td><td class="note">${esc(PM.job_presets[k]||'(none: the model takes no actions)')}</td><td><select data-job="${esc(k)}"><option value="">(preset)</option>${opt(names, PM.policy.jobs[k]||'')}</select></td></tr>`).join('');
  $('#pmJobs').querySelectorAll('[data-job]').forEach(el=>el.onchange=()=>{ if(el.value) PM.policy.jobs[el.dataset.job]=el.value; else delete PM.policy.jobs[el.dataset.job]; });
  $('#ptAct').innerHTML=opt(acts, $('#ptAct').value||'run');
  $('#pmModels').innerHTML=PM.models.map(m=>`<option value="${esc(m)}">`).join(''); $('#pmJobList').innerHTML=kinds.map(m=>`<option value="${esc(m)}">`).join(''); }
async function pmLoad(){ try{ PM=await api('/api/settings/permissions'); pmDraw(); }catch(e){ msg('#pmMsg',esc(e.message),true); } }
$('#pmProf').onchange=e=>{ pmCur=e.target.value; pmDraw(); };
$('#pmBase').onchange=e=>{ PM.policy.base=e.target.value; };
$('#pmCopy').onclick=()=>{ const n=(prompt('Name for the copy (lowercase, digits, - _ .)', pmCur+'-mine')||'').trim(); if(!n) return;
  PM.policy.profiles[n]=JSON.parse(JSON.stringify(pmProfiles()[pmCur])); pmCur=n; pmDraw(); };
$('#pmDel').onclick=()=>{ if(!PM.policy.profiles[pmCur]||!confirm('Delete profile '+pmCur+'?')) return; delete PM.policy.profiles[pmCur]; pmCur=PM.policy.base; pmDraw(); };
$('#pmAdd').onclick=()=>{ pmProfiles()[pmCur].rules.push({action:'run',effect:'deny',match:'regex',pattern:'',note:''}); pmDraw(); };
$('#pmBindAdd').onclick=()=>{ PM.policy.bindings.push({model:'*',profile:'read_only'}); pmDraw(); };
$('#pmSave').onclick=async()=>{ try{ PM=await api('/api/settings/permissions',{policy:PM.policy}); pmDraw(); msg('#pmMsg','Saved.'); }catch(e){ msg('#pmMsg',esc(e.message),true); } };
$('#ptGo').onclick=async()=>{ try{ const r=$('#ptRepo').value; const d=await api('/api/settings/permissions/check',{action:$('#ptAct').value,target:$('#ptTarget').value,model:$('#ptModel').value,job:$('#ptJob').value,repo:r,workdir:r});
  $('#ptOut').innerHTML=`<b class="${d.effect==='deny'?'err':d.effect==='allow'?'okm':''}">${d.effect}</b> · ${esc(d.why)} <span class="note">(layers: ${esc(d.layers.join(' + '))}; target ${esc(d.target)})</span>`; }catch(e){ msg('#ptOut',esc(e.message),true); } };
pmLoad();

// ---- widgets (viz/widgets.py): a form drawn from the spec, and a jewel preview with made-up emotions
let WD=null, prev=null;
async function wdLoad(){ const j=await (await fetch('/api/themes')).json(); WD=j.widgets; if(!WD) return;
  $('#wdForm').innerHTML=Object.entries(WD.spec).map(([w,sp])=>`<fieldset style="border:1px solid var(--line);border-radius:8px;margin:0;padding:.3rem .6rem;min-width:0"><legend class="note">${esc(w.replace('_',' '))}</legend>`+
    Object.entries(sp).map(([k,s])=>{ const v=WD.values[w][k], id=`wd-${w}-${k}`;
      const inp = s.kind==='choice' ? `<select id="${id}" style="width:auto;justify-self:start">${s.choices.map(c=>`<option${c===v?' selected':''}>${c}</option>`).join('')}</select>`
        : s.kind==='bool' ? `<input type="checkbox" id="${id}"${v?' checked':''} style="justify-self:start">`
        : `<input type="range" id="${id}" min="${s.min}" max="${s.max}" step="${s.kind==='int'?1:0.05}" value="${v}" style="width:100%"><span class="note" id="${id}-v">${v}</span>`;
      return `<label title="${esc(s.help||'')}" style="display:grid;grid-template-columns:6.5rem minmax(0,1fr) 2.5rem;gap:.5rem;align-items:center;margin:.15rem 0"><span>${k.replace('_',' ')}</span>${inp}</label>`; }).join('')+'</fieldset>').join('');
  $('#wdForm').querySelectorAll('input,select').forEach(el=>el.oninput=()=>{ const o=document.getElementById(el.id+'-v'); if(o) o.textContent=el.value; if(prev) prev.settings(wdValues().jewel); });
  if(!prev){ prev=NSWidgets.create('jewel', $('#wdPrev'), {settings: WD.values.jewel});
    const names=['happy','afraid','calm','angry','sad','desperate'], pal=['#FF4D1A','#3D8BFF','#1D9E75','#E7B65C','#B07CFF','#c9c9d2'];
    setInterval(()=>{ const s=performance.now()/1000; prev.set(names.map((n,k)=>({name:n, color:pal[k], z: 2.2*Math.sin(s*0.5+k*1.3)*Math.sin(s*0.21+k)}))); }, 100); }
  else prev.settings(WD.values.jewel); }
function wdValues(){ const out={};
  for(const [w,sp] of Object.entries(WD.spec)) for(const [k,s] of Object.entries(sp)){ const el=document.getElementById(`wd-${w}-${k}`); if(!el) continue;
    (out[w]??={})[k] = s.kind==='bool' ? el.checked : s.kind==='choice' ? el.value : s.kind==='int' ? parseInt(el.value) : parseFloat(el.value); }
  return out; }
async function wdSave(values){ const r=await fetch('/api/settings/widgets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({widgets:values})});
  const j=await r.json(); $('#wdMsg').textContent=r.ok?'Saved. Open 3D views pick it up when reloaded.':j.error; if(r.ok) wdLoad(); }
$('#wdSave').onclick=()=>wdSave(wdValues()); $('#wdReset').onclick=()=>wdSave({});
wdLoad();

let SH=null;
async function shLoad(){ const r=await fetch('/api/settings/sharing'); SH=await r.json(); const st=SH.settings;
  $('#shOn').checked=st.enabled; $('#shEp').value=st.endpoint; $('#shAuto').checked=st.auto_listed; $('#shUnl').checked=st.auto_unlisted;
  const gh=()=>{$('#shGh').style.display=$('#shEp').value.trim().startsWith('github:')?'block':'none'}; $('#shEp').oninput=gh; gh();
  $('#shTokSt').textContent=SH.github_token_set?'A token is saved (leave the field empty to keep it).':'No token saved.';
  $('#shModels').innerHTML='<tr><th>model</th><th>Hugging Face</th><th>public</th></tr>'+SH.models.map(m=>{ const c=m.check;
    const hf=c?(c.listed?`<span class="okm">found in ${esc(c.repo)}</span>`:`<span class="note">${esc(c.reason)}</span>`):'<span class="note">not checked</span>';
    const pub=c&&c.listed?'yes (listed)':`<label style="margin:0"><input type="checkbox" data-pub="${esc(m.id)}"${st.public_models.includes(m.id)?' checked':''}> list this unlisted model</label>`;
    return `<tr><td>${esc(m.id)}</td><td>${hf} <button data-chk="${esc(m.path)}">Check</button></td><td>${pub}</td></tr>`; }).join('');
  document.querySelectorAll('[data-chk]').forEach(b=>b.onclick=async()=>{ b.disabled=true; b.textContent='checking…';
    await fetch('/api/settings/sharing/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:b.dataset.chk})}); shLoad(); });
  $('#shRes').innerHTML=SH.results.length?SH.results.flatMap(r=>r.labels.map(l=>`<option value="${esc(r.file)}|${esc(l.label)}">${esc(r.file.split('/').pop())} · ${esc(l.label)}${l.model?' ('+esc(l.model)+')':''}</option>`)).join(''):'<option value="">no finished TestQA runs yet</option>';
  $('#shModel').innerHTML=SH.models.map(m=>`<option value="${esc(m.path)}">${esc(m.id)}</option>`).join('');
  $('#shHist').innerHTML=SH.history.length?'<tr><th>file</th><th>model</th><th>visibility</th><th>answer</th></tr>'+SH.history.map(h=>`<tr><td>${esc(h.file)}</td><td>${esc(h.model)}</td><td>${esc(h.visibility)}</td><td>${esc(h.answer?h.answer.status+(h.answer.shown_publicly?' · shown':''):'')}</td></tr>`).join(''):'<tr><td class="note">Nothing sent.</td></tr>';
}
$('#shSave').onclick=async()=>{ const pub=[...document.querySelectorAll('[data-pub]')].filter(x=>x.checked).map(x=>x.dataset.pub);
  const r=await fetch('/api/settings/sharing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:$('#shOn').checked,endpoint:$('#shEp').value,auto_listed:$('#shAuto').checked,auto_unlisted:$('#shUnl').checked,public_models:pub,...($('#shTok').value.trim()?{github_token:$('#shTok').value.trim()}:{})})});
  $('#shTok').value=''; $('#shMsg').textContent=r.ok?'Saved.':(await r.json()).error; if(r.ok) shLoad(); };
async function shAct(kind){ const [results,label]=($('#shRes').value||'|').split('|');
  $('#shOut').textContent=kind==='preview'?'building the card (hashing the model file the first time)…':'sending…';
  const r=await fetch('/api/settings/sharing/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({results,label,path:$('#shModel').value})});
  const j=await r.json(); $('#shOut').textContent=r.ok?JSON.stringify(j,null,1):'error: '+j.error; if(kind==='submit') shLoad(); }
$('#shPrev').onclick=()=>shAct('preview'); $('#shSend').onclick=()=>shAct('submit');
shLoad();
async function plLoad(){ let r; try{ r=await api('/api/plugins'); }catch(e){ $('#plList').innerHTML=`<tr><td class="err">${esc(e.message)}</td></tr>`; return; }
  $('#plDirs').innerHTML='Looked in: '+r.dirs.map(d=>'<code>'+esc(d)+'</code>').join(', ');
  const st={on:'<span style="color:var(--ok)">on</span>',off:'off',changed:'<span class="err">changed since you turned it on: off</span>',broken:'<span class="err">broken</span>'};
  $('#plList').innerHTML=r.plugins.length?'<tr><th>plugin</th><th>adds</th><th>status</th><th></th></tr>'+r.plugins.map(p=>{ const a=p.adds||{}, adds=[...(a.jobs||[]).map(x=>'job: '+x),...(a.checks||[]).map(x=>'check: '+x),a.page?'a page':'',a.mcp?'MCP tools':''].filter(Boolean);
    return `<tr><td><b>${esc(p.name)}</b> ${p.version?'<span class="note">'+esc(p.version)+'</span>':''}<div class="note">${esc(p.description||p.error||'')}</div><div class="note"><code>${esc(p.folder)}</code></div></td><td class="note">${adds.map(esc).join('<br>')}</td><td>${st[p.status]||esc(p.status)}</td><td>${p.status==='broken'?'':`<button data-pl="${esc(p.name)}" data-on="${p.status==='on'?0:1}">${p.status==='on'?'Turn off':(p.status==='changed'?'Review and turn on':'Turn on')}</button>`}${p.status==='on'&&a.page?` <a href="/plugins/${encodeURIComponent(p.name)}/">open page</a>`:''}</td></tr>`; }).join(''):'<tr><td class="note">No plugins found. Put plugin folders in the first folder above.</td></tr>';
  document.querySelectorAll('[data-pl]').forEach(b=>b.onclick=async()=>{ const on=b.dataset.on==='1';
    if(on&&!confirm('Turn on '+b.dataset.pl+'? It runs code on this computer with your permissions. Turn it on only if you trust where it came from.')) return;
    try{ await api('/api/plugins/enable',{name:b.dataset.pl,on}); msg('#plMsg',on?'On. Its jobs are under Jobs → Plugins.':'Off.'); }catch(e){ msg('#plMsg',esc(e.message),true); } plLoad(); }); }
plLoad();
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
from widgets import WIDGETS_JS as _WJS  # noqa: E402
PAGE = _inject_links(PAGE.replace("<script>", "<script>" + _WJS, 1))
