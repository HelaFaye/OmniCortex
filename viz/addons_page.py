"""Studio's /addons page: built-in add-ons to remove or reinstall, packaged add-ons to install from a file or a
repository, update, turn on or off, and uninstall (viz/addons.py, docs/ADDONS.md)."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Add-ons</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
main{max-width:1100px;margin:0 auto;padding:1rem 16px 3rem}h1{font-size:20px;margin:.2rem 0 .3rem}h2{font-size:16px;margin:0 0 .4rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem;margin:1rem 0}
.note{color:var(--mut);font-size:12.5px}.err{color:var(--no)}.okm{color:var(--ok)}
button,input{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:.35rem .7rem}
button{cursor:pointer;background:var(--panel)}button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.6rem}
.ad{border:1px solid var(--line);border-radius:10px;padding:.65rem .75rem;display:flex;flex-direction:column;gap:.3rem;background:var(--bg)}
.ad.off{opacity:.62}.ad b{font-size:14px}.ad .row{display:flex;gap:.35rem;align-items:center;flex-wrap:wrap;margin-top:auto}
.tag{display:inline-block;font-size:11px;padding:0 .45rem;border-radius:9px;border:1px solid var(--line);color:var(--mut)}
.tag.on{color:var(--ok);border-color:var(--ok)}.tag.changed,.tag.broken{color:var(--no);border-color:var(--no)}.tag.off{color:var(--warn);border-color:var(--warn)}
.drop{border:2px dashed var(--line);border-radius:10px;padding:1rem;text-align:center;color:var(--mut)}.drop.over{border-color:var(--acc);color:var(--fg)}
code{background:var(--code);padding:0 .3rem;border-radius:4px;font-size:12.5px}
a{color:var(--acc)}
</style>
<main>
<h1>Add-ons</h1>
<div class="note">OmniCortex's optional parts. Remove what you do not use, and its pages, jobs and sidebar entries go. Install more from a package file or a repository. A packaged add-on runs code on this computer, so it stays off until you turn it on, and goes off again whenever its files change. <a href="/help/ADDONS">How add-ons work</a></div>
<div id="msg"></div>
<section class="card"><h2>Installed packages</h2><div class="grid" id="pk"></div>
  <div class="drop" id="drop" style="margin-top:.8rem">Drop an add-on package (.zip) here, or <label style="color:var(--acc);cursor:pointer"><input type="file" id="file" accept=".zip,application/zip" hidden>choose one</label>.
  <div class="note">Make one from a plugin folder with <code>python viz/addons.py pack plugins/NAME</code>.</div></div></section>
<section class="card"><h2>From a repository</h2>
  <div class="note">A repository is an <code>index.json</code> at an https address (or a path on this computer) listing packages and their SHA-256. Each download is checked against it.</div>
  <div style="display:flex;gap:.4rem;margin:.5rem 0;flex-wrap:wrap"><input id="repo" placeholder="https://example.org/neuronscope-addons/index.json" style="flex:1;min-width:16rem"><button id="addrepo">Add repository</button></div>
  <div id="repos" class="note"></div><div class="grid" id="cat" style="margin-top:.5rem"></div></section>
<section class="card"><h2>Built into Studio</h2><div class="note">Removing one takes it out of Studio until you reinstall it here; nothing is downloaded either way and your data stays.</div>
  <div class="grid" id="bi" style="margin-top:.5rem"></div></section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const msg=(t,bad)=>{ $('#msg').innerHTML=t?`<div class="card ${bad?'err':'okm'}" style="padding:.5rem .8rem">${esc(t)}</div>`:''; if(t) scrollTo(0,0); };
let A=null;
const navReset=()=>{ try{ localStorage.removeItem('ns-addon-nav'); }catch(e){} };
async function load(){ A=await api('/api/addons');
  $('#bi').innerHTML=A.builtin.map(b=>`<div class="ad${b.installed?'':' off'}"><b>${esc(b.title)}</b><span class="note">${esc(b.desc)}</span>
    <span class="note">${[b.adds.pages.length?'pages: '+b.adds.pages.join(', '):'', b.adds.jobs.length?'jobs: '+b.adds.jobs.map(j=>j.replace('group:','')).join(', '):'',
      b.adds.services.length?'labs: '+b.adds.services.join(', '):'', b.adds.nodes.length?'graph nodes: '+b.adds.nodes.join(', '):''].filter(Boolean).map(esc).join(' · ')}</span>
    ${b.requires.length?`<span class="note">needs the ${esc(b.requires.map(r=>(A.builtin.find(x=>x.name===r)||{}).title||r).join(', '))} add-on</span>`:''}
    ${b.needs.length?`<span class="note">uses ${esc(b.needs.join(', '))}</span>`:''}
    <div class="row"><span class="tag ${b.installed?'on':'off'}">${b.installed?'installed':'removed'}</span>${b.docs?`<a class="note" href="/help/${esc(b.docs.replace('.md',''))}">docs</a>`:''}<span style="flex:1"></span>
    <button data-bi="${b.name}" data-to="${b.installed?0:1}" class="${b.installed?'bad':'pri'}">${b.installed?'Remove':'Reinstall'}</button></div></div>`).join('');
  document.querySelectorAll('[data-bi]').forEach(x=>x.onclick=async()=>{ const on=x.dataset.to==='1', name=x.dataset.bi;
    if(!on&&!confirm('Remove this part of Studio? Its pages and jobs go until you reinstall it; your data stays.')) return;
    const title=n=>(A.builtin.find(b=>b.name===n)||{}).title||n;
    try{ let r;
      try{ r=await api('/api/addons/builtin',{name,installed:on}); }
      catch(e){ if(on||!/ need/.test(e.message)||!confirm(e.message.replace(/: remove .*$/,'')+'. Remove '+(e.message.includes(' needs ')?'it':'them')+' as well?')) throw e;
        r=await api('/api/addons/builtin',{name,installed:false,with_dependents:true}); }
      navReset(); msg((on?'Reinstalled: ':'Removed: ')+r.changed.map(title).join(', ')+'.'); load(); }catch(e){ msg(e.message,true); } });
  $('#pk').innerHTML=A.packages.map(p=>`<div class="ad${p.status==='on'?'':' off'}"><b>${esc(p.name)} <span class="note">${esc(p.version||'')}</span></b>
    <span class="note">${esc(p.description)}</span>${p.error?`<span class="err">${esc(p.error)}</span>`:''}
    <span class="note">${[...(p.adds.jobs||[]).map(x=>'job: '+x),...(p.adds.nodes||[]).map(x=>'node: '+x),p.adds.page?'a page':'',p.tab?'a tab':'',p.adds.mcp?'MCP tools':''].filter(Boolean).map(esc).join(' · ')}</span>
    <span class="note" title="${esc(p.folder)}">${p.removable?'installed here':'came with OmniCortex'} · sha256 ${esc((p.sha256||'').slice(0,12))}</span>
    <div class="row"><span class="tag ${p.status}">${{on:'on',off:'off',changed:'changed: review it',broken:'broken'}[p.status]||p.status}</span><span style="flex:1"></span>
    ${p.status==='broken'?'':`<button data-en="${esc(p.name)}" data-on="${p.status==='on'?0:1}" class="${p.status==='on'?'':'pri'}">${p.status==='on'?'Turn off':'Turn on'}</button>`}
    ${p.status==='on'&&p.tab?`<a href="/plugins/${esc(p.name)}/"><button>Open</button></a>`:''}
    ${p.removable?`<button class="bad" data-un="${esc(p.name)}">Uninstall</button>`:''}</div></div>`).join('')||'<span class="note">None yet.</span>';
  document.querySelectorAll('[data-en]').forEach(x=>x.onclick=async()=>{ const on=x.dataset.on==='1';
    if(on&&!confirm(`Turn on ${x.dataset.en}? It runs code on this computer with your permissions. Turn it on only if you trust where it came from.`)) return;
    try{ await api('/api/plugins/enable',{name:x.dataset.en,on}); navReset(); msg(on?'On.':'Off.'); load(); }catch(e){ msg(e.message,true); } });
  document.querySelectorAll('[data-un]').forEach(x=>x.onclick=async()=>{ if(!confirm('Uninstall '+x.dataset.un+'? Its folder is deleted.')) return;
    try{ await api('/api/addons/uninstall',{name:x.dataset.un}); navReset(); msg('Uninstalled.'); load(); }catch(e){ msg(e.message,true); } });
  $('#repos').innerHTML=A.repos.length?A.repos.map((r,i)=>`<div>${esc(r)} <a href="#" data-rr="${i}">remove</a></div>`).join(''):'No repositories yet.';
  document.querySelectorAll('[data-rr]').forEach(x=>x.onclick=async e=>{ e.preventDefault(); const r=A.repos.filter((_,i)=>i!==+x.dataset.rr); await api('/api/addons/repos',{repos:r}); load(); });
  catalog(); }
async function catalog(){ if(!A.repos.length){ $('#cat').innerHTML=''; return; } $('#cat').innerHTML='<span class="note">reading the repositories…</span>';
  try{ const c=await api('/api/addons/catalog');
    $('#cat').innerHTML=c.addons.map(a=>a.error?`<div class="ad"><b class="err">${esc(a.repo)}</b><span class="err">${esc(a.error)}</span></div>`:
      `<div class="ad"><b>${esc(a.name)} <span class="note">${esc(a.version||'')}</span></b><span class="note">${esc(a.description||'')}</span>
      <span class="note">${esc(a.repo_name||a.repo)}${a.size?' · '+Math.round(a.size/1024)+' KB':''}${a.homepage?` · <a href="${esc(a.homepage)}" target="_blank" rel="noopener">home</a>`:''}</span>
      <div class="row">${a.installed?`<span class="tag on">installed ${esc(a.installed)}</span>`:''}<span style="flex:1"></span>
      ${!a.installed||a.update?`<button class="pri" data-in="${esc(a.name)}" data-repo="${esc(a.repo)}">${a.update?'Update':'Install'}</button>`:''}</div></div>`).join('')||'<span class="note">The repositories list no add-ons.</span>';
    document.querySelectorAll('[data-in]').forEach(x=>x.onclick=async()=>{ x.disabled=true; x.textContent='installing…';
      try{ const r=await api('/api/addons/install',{name:x.dataset.in,repo:x.dataset.repo}); msg(`Installed ${r.name}. Review it, then turn it on under Installed packages.`); load(); }catch(e){ msg(e.message,true); x.disabled=false; } }); }
  catch(e){ $('#cat').innerHTML=`<span class="err">${esc(e.message)}</span>`; } }
$('#addrepo').onclick=async()=>{ const u=$('#repo').value.trim(); if(!u) return; try{ await api('/api/addons/repos',{repos:[...A.repos,u]}); $('#repo').value=''; load(); }catch(e){ msg(e.message,true); } };
async function upload(f){ if(!f) return; msg('installing '+f.name+'…');
  try{ const r=await fetch('/api/addons/upload',{method:'POST',headers:{'Content-Type':'application/zip'},body:f}); const j=await r.json(); if(!r.ok) throw new Error(j.error);
    msg(`Installed ${j.name}. Review it, then turn it on under Installed packages.`); load(); }catch(e){ msg(e.message,true); } }
$('#file').onchange=e=>upload(e.target.files[0]);
const d=$('#drop'); d.ondragover=e=>{ e.preventDefault(); d.classList.add('over'); }; d.ondragleave=()=>d.classList.remove('over');
d.ondrop=e=>{ e.preventDefault(); d.classList.remove('over'); upload(e.dataTransfer.files[0]); };
load().catch(e=>msg(e.message,true));
</script>
"""

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
