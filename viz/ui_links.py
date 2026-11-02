"""Links between Studio's pages. When a message or a fix says "Setup -> Build
llama.cpp" or "Settings → Workspaces", the page shows it as a link to that
section and option. Every page includes the same linkify(), so a message from
the server reads as a link wherever it is shown.

Targets:
  /setup#paths  #build  #hardware  #health  #requirements  #snapshots
  /settings#workspaces  #subjects  #tools
  /jobs#kind=<kind>&<field>=<value>…   (a job form, filled in)
  /#chats  #models  #hub  #docs  #tools  /#model=<id>&field=<setting>
  /review  /lab  /projects  /connect
"""

# (regex over escaped text, href). Arrows: -> → >. Order matters: specific first.
TARGETS = [
    (r"Setup\s*(?:->|→|&gt;)\s*Build(?: llama\.cpp)?", "/setup#build"),
    (r"Setup\s*(?:->|→|&gt;)\s*Hardware", "/setup#hardware"),
    (r"Setup\s*(?:->|→|&gt;)\s*Requirements", "/setup#requirements"),
    (r"Setup\s*(?:->|→|&gt;)\s*Health check", "/setup#health"),
    (r"Setup\s*(?:->|→|&gt;)\s*(?:Paths|routing thresholds)", "/setup#paths"),
    (r"Settings\s*(?:->|→|&gt;)\s*Workspaces", "/settings#workspaces"),
    (r"Settings\s*(?:->|→|&gt;)\s*Models per subject", "/settings#subjects"),
    (r"Settings\s*(?:->|→|&gt;)\s*Tool approval", "/settings#tools"),
    (r"Jobs\s*(?:->|→|&gt;)\s*Review", "/jobs#kind=review_testqa"),
    (r"Jobs\s*(?:->|→|&gt;)\s*Evaluate", "/jobs#kind=testqa"),
    (r"Jobs\s*(?:->|→|&gt;)\s*H-Neurons", "/jobs#kind=preflight"),
    (r"Studio\s*(?:->|→|&gt;)\s*Tools", "/#tools"),
    (r"Studio\s*(?:->|→|&gt;)\s*Models", "/#models"),
    (r"Studio\s*(?:->|→|&gt;)\s*Docs", "/#docs"),
    (r"Remote\s*(?:->|→|&gt;)\s*Add a host", "/remote#add"),
    (r"Workers page", "/workers"),
]

JS = r"""
// ---- links between pages (viz/ui_links.py)
const NS_LINKS=__TARGETS__;
function linkify(html){ let out=String(html??'');
  for(const [rx,href] of NS_LINKS){ out=out.replace(new RegExp('(?<![">])('+rx+')(?![^<]*</a>)','g'),`<a href="${href}">$1</a>`); }
  return out; }
// ---- themes (viz/themes.json "_studio"): UI colours, chart colours, the 3D viewer's theme
const NS_UI_VARS=['bg','panel','fg','mut','line','acc','accfg','ok','no','warn','soft','code'];
const NS_DEFAULT_CHART=['#6ea8e0','#ff8b7e','#5fcf8f','#e7b65c','#b48ead','#5dcaa5'];
function nsApplyTheme(name,t){ const d=document.documentElement; d.dataset.theme=t.base||'dark';
  NS_UI_VARS.forEach(k=>d.style.removeProperty('--'+k));
  for(const [k,v] of Object.entries(t.ui||{})) d.style.setProperty('--'+k,v);
  try{ localStorage.setItem('ns-theme',name); localStorage.setItem('ns-theme-base',t.base||'dark');
    localStorage.setItem('ns-theme-vars',JSON.stringify(t.ui||{})); localStorage.setItem('ns-chart',JSON.stringify(t.chart||NS_DEFAULT_CHART)); }catch(e){}
  document.cookie='ns_viz='+encodeURIComponent(t.viz||'dark')+'; path=/; SameSite=Strict; max-age=31536000';
  window.dispatchEvent(new CustomEvent('ns-theme',{detail:{name,theme:t}})); }
function nsChart(){ try{ return JSON.parse(localStorage.getItem('ns-chart')||'null')||NS_DEFAULT_CHART; }catch(e){ return NS_DEFAULT_CHART; } }
function nsThemeName(){ try{ return localStorage.getItem('ns-theme')||'midnight'; }catch(e){ return 'midnight'; } }
async function nsCycleTheme(){ const j=await (await fetch('/api/themes')).json(); const names=Object.keys(j.themes);
  const cur=nsThemeName(); const nx=names[(names.indexOf(cur)+1)%names.length]; nsApplyTheme(nx,j.themes[nx]); return nx; }
// ---- the sidebar on every page: what Studio does, in the order people use it (LM Studio-style)
const NS_ICON={puzzle:'<path d="M9 4h3a2 2 0 1 1 4 0h3v4a2 2 0 1 1 0 4v4h-4a2 2 0 1 0-4 0H8v-4a2 2 0 1 1 0-4z"/>',chat:'<path d="M4 5h16v11H9l-5 4z"/>',box:'<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
  search:'<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',doc:'<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h7"/>',
  plug:'<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4"/>',play:'<path d="M7 4l13 8-13 8z"/>',
  tasks:'<path d="M10 6h10M10 12h10M10 18h10M4 6l1.2 1.2L7.5 5M4 12l1.2 1.2L7.5 11M4 18l1.2 1.2L7.5 17"/>',
  cpu:'<path d="M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4"/>',
  flask:'<path d="M9 3h6M10 3v6l-5 10a1.2 1.2 0 0 0 1 2h12a1.2 1.2 0 0 0 1-2l-5-10V3M7.5 15h9"/>',
  grid:'<path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z"/>',
  globe:'<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3.5 3.5 3.5 14.5 0 18M12 3c-3.5 3.5-3.5 14.5 0 18"/>',
  link:'<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  share:'<circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8.3 10.9l7.4-3.7M8.3 13.1l7.4 3.7"/>',
  wrench:'<path d="M14.5 4.5a4.5 4.5 0 0 0 5 6L11 19a2.1 2.1 0 0 1-3-3l8.5-8.5a4.5 4.5 0 0 1-2-3z"/>',
  gear:'<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>',
  more:'<path d="M5 12h.01M12 12h.01M19 12h.01" stroke-width="3.2"/>',
  pulse:'<path d="M3 12h4l2-6 4 12 2-6h6"/>',
  find:'<circle cx="10" cy="10" r="6"/><path d="M19 19l-4.5-4.5M7.5 10h5"/>',
  spark:'<circle cx="12" cy="12" r="2.2"/><circle cx="5" cy="6" r="1.6"/><circle cx="19" cy="6" r="1.6"/><circle cx="5" cy="18" r="1.6"/><circle cx="19" cy="18" r="1.6"/><path d="M6.3 7l3.9 3.6M17.7 7l-3.9 3.6M6.3 17l3.9-3.6M17.7 17l-3.9-3.6"/>'};
// tab: the Studio-page tab the item opens. mobile: also on the phone bar. kids: a group; its pages share a
// row of tabs at the top, and the group's icon opens the page last used in it.
const NS_NAV=[
  {href:'/#chats',tab:'chats',label:'Chat',icon:'chat',tip:'Chat with your models',mobile:1},
  {label:'Models',icon:'box',tip:'Your models, and new ones to download',mobile:1,kids:[
    {href:'/#models',tab:'local',label:'My Models',tip:'Models on this computer: load, settings, measured accuracy'},
    {href:'/#hub',tab:'hub',label:'Discover',tip:'Find and download GGUF models from Hugging Face'}]},
  {href:'/#docs',tab:'docs',label:'Docs',icon:'doc',tip:'Document collections replies can cite'},
  {href:'/#tools',tab:'tools',label:'Tools',icon:'plug',tip:'MCP servers whose tools models may call'},
  {sep:1},
  {label:'Tasks',icon:'tasks',tip:'Jobs, projects split into tasks, and the workers that run them',mobile:1,kids:[
    {href:'/jobs',label:'Jobs',tip:'Evaluation, retraining, benchmark and analysis jobs'},
    {href:'/projects',label:'Projects',tip:'Split a project into tasks for worker models'},
    {href:'/workers',label:'Workers',tip:'Where tasks run; GPU temperatures'},
    {href:'/work',label:'Distribution',tip:'Where the work went: shares, tokens, speeds and hardware per device, and neuron activation plots'}]},
  {label:'Studio',icon:'spark',tip:'Studio tools: features and emotions, graphs, neuron review, labs and 3D views',kids:[
    {href:'/features',label:'Features',tip:'Feature dictionaries, what a reply was made of, circuits, emotion vectors, steering'},
    {href:'/graphs',label:'Graphs',tip:'Node graphs: screen, verify and route model inputs and outputs'},
    {href:'/review',label:'Review',tip:'How neurons behave across tests, by subject and over time'},
    {href:'/lab',label:'Lab',tip:'Labs, live and 3D views'}]},
  {sep:1},
  {label:'Connect',icon:'share',tip:'Other apps, other computers and your devices',kids:[
    {href:'/connect',label:'Apps',tip:'Use these models and tools from Cline, Claude and other apps'},
    {href:'/remote',label:'Remote hosts',tip:'Models on other computers (OmniCortex Remote)'},
    {href:'/link',label:'Devices',tip:'Pair devices; link other Studios'}]},
  {sep:1,bottom:1},
  {href:'/addons',label:'Add-ons',icon:'puzzle',tip:'Install, remove and update add-ons; ComfyUI and other packages',bottom:1},
  {href:'/system',label:'System',icon:'pulse',tip:'What is running, restart any part, monitor the hardware, automation rules',bottom:1},
  {href:'/setup',label:'Setup',icon:'wrench',tip:'Paths, building llama.cpp, hardware, health check',bottom:1},
  {href:'/settings',label:'Settings',icon:'gear',tip:'Themes, workspaces, models per subject, tool approval, plugins',bottom:1}];
// Add-ons change the sidebar: a removed add-on's pages go, an add-on's own tab comes (viz/addons.py).
// The last answer is kept, so the sidebar draws at once, and redrawn if it changed.
const NS_NAV_BASE=JSON.parse(JSON.stringify(NS_NAV));
function nsNavApply(cfg){ NS_NAV.splice(0,NS_NAV.length,...JSON.parse(JSON.stringify(NS_NAV_BASE))); if(!cfg) return;
  const off=new Set(cfg.off||[]);
  for(let i=NS_NAV.length-1;i>=0;i--){ const it=NS_NAV[i];
    if(it.kids){ it.kids=it.kids.filter(k=>!off.has(k.href)); if(!it.kids.length) NS_NAV.splice(i,1); }
    else if(it.href&&off.has(it.href)) NS_NAV.splice(i,1); }
  if((cfg.tabs||[]).length){ const at=NS_NAV.findIndex(x=>x.sep&&x.bottom);
    NS_NAV.splice(at<0?NS_NAV.length:at,0,{label:'Add-ons',icon:'puzzle',tip:'Tabs added by add-ons',kids:cfg.tabs.map(t=>({href:t.href,label:t.label,tip:t.tip||''}))}); } }
(function(){ let c=null; try{ c=JSON.parse(localStorage.getItem('ns-addon-nav')||'null'); }catch(e){} nsNavApply(c);
  if(/^\/(pair|login)/.test(location.pathname)||window.self!==window.top) return;
  fetch('/api/addons/nav').then(r=>r.ok?r.json():null).then(j=>{ if(!j) return; const s=JSON.stringify(j);
    let old=null; try{ old=localStorage.getItem('ns-addon-nav'); localStorage.setItem('ns-addon-nav',s); }catch(e){}
    if(s!==old){ nsNavApply(j); ['ns-rail','ns-bar','ns-more','ns-fly','ns-sub'].forEach(id=>{ const e=document.getElementById(id); if(e) e.remove(); });
      if(document.body&&document.body.classList.contains('ns-railed')) nsRail(); } }).catch(()=>{}); })();
function nsSvg(n){ return `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${NS_ICON[n]||''}</svg>`; }
function nsNavActive(it){ if(it.kids) return it.kids.some(nsNavActive); const here=location.pathname.replace(/\/+$/,'')||'/';
  if(it.tab) return here==='/'&&(window.NS_TAB||'chats')===it.tab;
  return here===it.href||here.startsWith(it.href+'/'); }
// the page a group's icon opens: the one last used in it, else its first
function nsGroupHref(g){ let h=null; try{ h=localStorage.getItem('ns-nav-'+g.label); }catch(e){} return g.kids.some(k=>k.href===h)?h:g.kids[0].href; }
function nsSub(){ let bar=document.getElementById('ns-sub'); const g=NS_NAV.find(it=>it.kids&&nsNavActive(it));
  if(!g){ if(bar) bar.remove(); return; }
  const cur=g.kids.find(nsNavActive); try{ localStorage.setItem('ns-nav-'+g.label,cur.href); }catch(e){}
  if(!bar){ bar=document.createElement('nav'); bar.id='ns-sub'; document.body.prepend(bar); }
  bar.setAttribute('aria-label',g.label);
  bar.innerHTML=`<span class="g">${nsSvg(g.icon)}${nsEsc(g.label)}</span>`+g.kids.map(k=>`<a href="${k.href}" title="${nsEsc(k.tip)}" class="${k===cur?'on':''}" ${k===cur?'aria-current="page"':''}>${nsEsc(k.label)}</a>`).join('');
  const r=document.getElementById('ns-rail'); if(r) r.querySelectorAll('a.grp').forEach(a=>{ const it=NS_NAV[+a.dataset.nav]; a.href=nsGroupHref(it); }); }
function nsRailDraw(){ const r=document.getElementById('ns-rail'); if(!r) return;
  r.querySelectorAll('[data-nav]').forEach(a=>{ const it=NS_NAV[+a.dataset.nav]; const on=nsNavActive(it); a.classList.toggle('on',on); if(on) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current'); });
  const b=document.getElementById('ns-bar'); if(b) b.querySelectorAll('[data-nav]').forEach(a=>a.classList.toggle('on',nsNavActive(NS_NAV[+a.dataset.nav])));
  nsSub(); }
function nsRail(){ if(window.self!==window.top||document.getElementById('ns-rail')||/^\/(pair|login)/.test(location.pathname)) return;
  const st=document.createElement('style'); st.textContent=`
#ns-rail{position:fixed;left:0;top:0;bottom:0;width:76px;z-index:8000;background:var(--panel,#1d1d1b);border-right:1px solid var(--line,#33332f);display:flex;flex-direction:column;align-items:stretch;padding:.35rem .3rem;gap:0;overflow-y:auto;font:500 10.5px/1.15 ui-sans-serif,system-ui}
#ns-rail .brand{display:flex;justify-content:center;margin:.1rem 0 .3rem;color:var(--acc,#6f93ff);text-decoration:none}
#ns-rail a.it,#ns-bar a.it,#ns-more a.it{display:flex;flex-direction:column;align-items:center;gap:.1rem;padding:.24rem .1rem;border-radius:8px;color:var(--mut,#9a9a94);text-decoration:none;text-align:center}
#ns-rail a.it:hover,#ns-bar a.it:hover,#ns-more a.it:hover{background:var(--code,#262624);color:var(--fg,#ececea)}
#ns-rail a.it.on,#ns-bar a.it.on{background:color-mix(in srgb,var(--acc,#6f93ff) 18%,transparent);color:var(--fg,#ececea)}
#ns-rail a.it.on svg,#ns-bar a.it.on svg{color:var(--acc,#6f93ff)}
#ns-rail a.it:focus-visible,#ns-bar a.it:focus-visible{outline:2px solid var(--acc,#6f93ff);outline-offset:-2px}
#ns-rail hr{border:none;border-top:1px solid var(--line,#33332f);margin:.18rem .5rem;width:auto}
@media(max-height:780px){#ns-rail a.it svg{width:18px;height:18px}#ns-rail a.it{padding:.18rem .1rem}}
#ns-rail .brand svg{width:24px;height:24px}
#ns-rail .grow{flex:1}
body.ns-railed{padding-left:76px}
body.ns-railed>header>a{display:none}
body.ns-railed aside>.tabs{display:none}
#ns-rail a.grp{position:relative}#ns-rail a.grp::after{content:'';position:absolute;right:5px;top:50%;margin-top:-9px;border:3.5px solid transparent;border-left-color:currentColor;opacity:.55}
#ns-fly{position:fixed;left:78px;z-index:8100;min-width:12rem;background:var(--panel,#1d1d1b);border:1px solid var(--line,#33332f);border-radius:10px;padding:.35rem;box-shadow:0 8px 28px rgba(0,0,0,.35);font:13px/1.35 ui-sans-serif,system-ui}
#ns-fly[hidden]{display:none}#ns-fly .h{font:600 11px ui-sans-serif,system-ui;color:var(--mut,#9a9a94);text-transform:uppercase;letter-spacing:.04em;padding:.25rem .5rem}
#ns-fly a{display:block;padding:.38rem .55rem;border-radius:6px;color:var(--fg,#ececea);text-decoration:none}#ns-fly a small{display:block;color:var(--mut,#9a9a94);font-size:11.5px}
#ns-fly a:hover,#ns-fly a:focus-visible{background:var(--code,#262624);outline:none}#ns-fly a.on{background:color-mix(in srgb,var(--acc,#6f93ff) 18%,transparent)}
#ns-sub{display:flex;align-items:center;gap:.15rem;padding:0 12px;border-bottom:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);font:500 13px ui-sans-serif,system-ui;overflow-x:auto;flex:none;min-height:38px}
#ns-sub .g{display:flex;align-items:center;gap:.35rem;color:var(--mut,#9a9a94);margin-right:.6rem;font-size:12px;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}#ns-sub .g svg{width:16px;height:16px}
#ns-sub a{padding:.55rem .75rem;color:var(--mut,#9a9a94);text-decoration:none;border-bottom:2px solid transparent;white-space:nowrap}
#ns-sub a:hover{color:var(--fg,#ececea)}#ns-sub a.on{color:var(--fg,#ececea);border-bottom-color:var(--acc,#6f93ff)}
#ns-more .gh{grid-column:1/-1;font:600 10.5px ui-sans-serif,system-ui;color:var(--mut,#9a9a94);text-transform:uppercase;letter-spacing:.04em;padding:.3rem .3rem 0}
#ns-bar{display:none}
@media(max-width:700px){#ns-rail{display:none}body.ns-railed{padding-left:0;padding-bottom:58px}
  #ns-bar{display:flex;position:fixed;left:0;right:0;bottom:0;height:58px;z-index:8000;background:var(--panel,#1d1d1b);border-top:1px solid var(--line,#33332f);font:500 10.5px/1.1 ui-sans-serif,system-ui;padding:0 .2rem}
  #ns-bar a.it{flex:1;justify-content:center;padding:.3rem 0;margin:.25rem .1rem}
  #ns-more{position:fixed;left:8px;right:8px;bottom:66px;z-index:9100;background:var(--panel,#1d1d1b);border:1px solid var(--line,#33332f);border-radius:12px;padding:.4rem;display:grid;grid-template-columns:repeat(4,1fr);gap:.2rem;box-shadow:0 8px 30px rgba(0,0,0,.4);font:500 11px/1.15 ui-sans-serif,system-ui}
  #ns-more[hidden]{display:none}}`;
  document.head.appendChild(st);
  const item=(it,i)=>`<a class="it${it.kids?' grp':''}" href="${it.kids?nsGroupHref(it):it.href}" data-nav="${i}" title="${nsEsc(it.tip)}"${it.kids?' aria-haspopup="true"':''}>${nsSvg(it.icon)}<span>${nsEsc(it.label)}</span></a>`;
  let top='',bot='';
  NS_NAV.forEach((it,i)=>{ const h=it.sep?'<hr>':item(it,i); if(it.bottom) bot+=h; else top+=h; });
  const r=document.createElement('nav'); r.id='ns-rail'; r.setAttribute('aria-label','Studio');
  r.innerHTML=`<a class="brand" href="/#chats" title="OmniCortex Studio" aria-label="OmniCortex Studio home"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="12" cy="12" r="3"/><circle cx="4.5" cy="6" r="1.6"/><circle cx="19.5" cy="6" r="1.6"/><circle cx="4.5" cy="18" r="1.6"/><circle cx="19.5" cy="18" r="1.6"/><path d="M6 6.8l3.6 3.4M18 6.8l-3.6 3.4M6 17.2l3.6-3.4M18 17.2l-3.6-3.4"/></svg></a><a class="it" href="#" data-find="1" title="Search every page, setting and job (Ctrl+K)">${nsSvg('find')}<span>Search</span></a><hr>${top}<div class="grow"></div>${bot}`;
  const bar=document.createElement('nav'); bar.id='ns-bar'; bar.setAttribute('aria-label','Studio');
  bar.innerHTML=NS_NAV.map((it,i)=>it.mobile?item(it,i):'').join('')+`<a class="it" href="#" data-more="1" aria-haspopup="true">${nsSvg('more')}<span>More</span></a>`;
  const more=document.createElement('div'); more.id='ns-more'; more.hidden=true;
  const kid=(k,i,j)=>`<a class="it" href="${k.href}" data-nav="${i}" data-kid="${j}" title="${nsEsc(k.tip)}">${nsSvg(NS_NAV[i].icon)}<span>${nsEsc(k.label)}</span></a>`;
  // single pages first (Docs, Tools, System, Setup, Settings), then each group under its name
  more.innerHTML=NS_NAV.map((it,i)=>it.sep||it.kids||it.mobile?'':item(it,i)).join('')+
    NS_NAV.map((it,i)=>it.kids?`<div class="gh">${nsEsc(it.label)}</div>`+it.kids.map((k,j)=>kid(k,i,j)).join(''):'').join('');
  document.body.prepend(r); document.body.append(bar,more); document.body.classList.add('ns-railed');
  bar.querySelector('[data-more]').onclick=e=>{ e.preventDefault(); more.hidden=!more.hidden; };
  r.querySelector('[data-find]').onclick=e=>{ e.preventDefault(); nsFind(); };
  more.insertAdjacentHTML('afterbegin',`<a class="it" href="#" data-find="1">${nsSvg('find')}<span>Search</span></a>`);
  more.querySelector('[data-find]').onclick=e=>{ e.preventDefault(); nsFind(); };
  more.onclick=()=>more.hidden=true;
  // a group's pages in a flyout beside the rail, on hover or keyboard focus
  const fly=document.createElement('div'); fly.id='ns-fly'; fly.hidden=true; fly.setAttribute('role','menu'); document.body.append(fly);
  let shut=null; const close=()=>{ shut=setTimeout(()=>fly.hidden=true,180); };
  const open=a=>{ clearTimeout(shut); const g=NS_NAV[+a.dataset.nav]; const rc=a.getBoundingClientRect();
    fly.innerHTML=`<div class="h">${nsEsc(g.label)}</div>`+g.kids.map(k=>`<a role="menuitem" href="${k.href}" class="${nsNavActive(k)?'on':''}">${nsEsc(k.label)}<small>${nsEsc(k.tip)}</small></a>`).join('');
    fly.hidden=false; fly.style.top=Math.max(6,Math.min(rc.top-6,innerHeight-fly.offsetHeight-6))+'px'; };
  r.querySelectorAll('a.grp').forEach(a=>{ a.addEventListener('mouseenter',()=>open(a)); a.addEventListener('mouseleave',close);
    a.addEventListener('focus',()=>open(a)); a.addEventListener('blur',close);
    a.addEventListener('keydown',e=>{ if(e.key==='ArrowRight'||e.key==='ArrowDown'){ e.preventDefault(); open(a); fly.querySelector('a')?.focus(); } }); });
  fly.addEventListener('mouseenter',()=>clearTimeout(shut)); fly.addEventListener('mouseleave',close);
  fly.addEventListener('focusin',()=>clearTimeout(shut)); fly.addEventListener('focusout',close);
  fly.addEventListener('keydown',e=>{ if(e.key==='Escape'){ fly.hidden=true; } });
  addEventListener('hashchange',nsRailDraw); document.addEventListener('ns-tab',e=>{ window.NS_TAB=e.detail; nsRailDraw(); });
  nsRailDraw(); }
(function(){ if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',nsRail); else nsRail(); })();
// ---- restarting Studio: ask, then wait until it answers again and reload the page
async function nsRestart(what){ const o=document.createElement('div');
  o.style.cssText='position:fixed;inset:0;z-index:9800;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;font:15px ui-sans-serif,system-ui;color:#ececea';
  o.innerHTML='<div style="background:#1d1d1b;border:1px solid #33332f;border-radius:12px;padding:1.2rem 1.6rem;max-width:90vw;text-align:center"><b>Restarting Studio…</b><div data-r style="margin-top:.4rem;color:#9a9a94">Stopping models and starting again.</div></div>';
  document.body.appendChild(o); const msg=o.querySelector('[data-r]');
  try{ const r=await fetch('/api/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); if(!r.ok){ msg.textContent='Could not restart: '+((await r.json().catch(()=>({}))).error||r.status); return; } }catch(e){}
  const t0=Date.now(); await new Promise(r=>setTimeout(r,1500));
  while(Date.now()-t0<90000){ try{ const r=await fetch('/api/status',{cache:'no-store'}); if(r.ok){ const j=await r.json(); if(!j.started||j.started*1000>t0-2000){ msg.textContent='Back. Reloading…'; location.reload(); return; } } }catch(e){}
    msg.textContent=`Waiting for Studio to come back… ${Math.round((Date.now()-t0)/1000)} s`; await new Promise(r=>setTimeout(r,1000)); }
  msg.innerHTML='Studio has not come back after 90 s. Check the terminal or log it was started from, or start it again (<code>./neuronscope</code>).'; }
// When the code on disk is newer than the running Studio, say so once per page.
(function(){ if(window.self!==window.top||/^\/(pair|login)/.test(location.pathname)) return;
  const go=()=>fetch('/api/status',{cache:'no-store'}).then(r=>r.ok?r.json():null).then(j=>{ if(!j||!j.stale||j.role!=='owner') return;
    try{ if(sessionStorage.getItem('ns-stale-later')) return; }catch(e){}
    const b=document.createElement('div'); b.id='ns-stale'; b.setAttribute('role','status');
    b.style.cssText='position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9600;max-width:min(680px,calc(100vw - 24px));background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--warn,#e7b65c);border-radius:10px;padding:.55rem .8rem;font:13.5px/1.45 ui-sans-serif,system-ui;box-shadow:0 6px 24px rgba(0,0,0,.35);display:flex;gap:.6rem;align-items:center;flex-wrap:wrap';
    b.innerHTML='<span style="flex:1;min-width:14rem">OmniCortex was updated after this Studio started. Pages and fixes added since then are missing until it restarts.</span><button data-s="go" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.3rem .8rem;font-weight:600;cursor:pointer">Restart Studio</button><button data-s="later" style="background:none;border:1px solid var(--line,#33332f);color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer">Later</button>';
    b.querySelector('[data-s=go]').onclick=()=>{ b.remove(); nsRestart(); };
    b.querySelector('[data-s=later]').onclick=()=>{ try{ sessionStorage.setItem('ns-stale-later','1'); }catch(e){} b.remove(); };
    document.body.appendChild(b); }).catch(()=>{});
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',go); else go(); })();
// ---- search: every page, section, job and help page, and the controls on this page (Ctrl+K)
const NS_SECTIONS=[
  ['Status and next step','/setup#status-card','Setup'],['Folders, ports and Studio settings','/setup#paths','Setup'],
  ['Build llama.cpp','/setup#build','Setup'],['Hardware for worker models (GPUs, devices)','/setup#hardware','Setup'],
  ['Health check (doctor)','/setup#health','Setup'],['Requirements and Python packages','/setup#requirements','Setup'],
  ['Appearance and themes','/settings#appearance','Settings'],['Workspaces (folders per machine)','/settings#workspaces','Settings'],
  ['Models per subject (auto routing)','/settings#subjects','Settings'],['Connectors (MCP servers)','/settings#connectors','Settings'],
  ['Tool approval rules','/settings#tools','Settings'],['Plugins','/settings#plugins','Settings'],['Sharing results (scoreboard)','/settings#sharing','Settings'],
  ['Connect Claude Desktop, Claude Code, claude.ai','/connect#claude','Connect'],['Connect Cline and OpenAI-compatible apps','/connect','Connect'],
  ['Add a remote host','/remote#add','Remote'],['Running parts: restart Studio, models, connectors','/system#services','System'],
  ['Chats','/#chats','Studio'],['My Models: load, settings, accuracy','/#models','Studio'],['Discover and download models','/#hub','Studio'],
  ['Documents (RAG collections)','/#docs','Studio'],['Tools (MCP) in chat','/#tools','Studio'],['Pair a device, link Studios','/link','Devices'],
  ['Workers: where project tasks run, GPU temperatures','/workers','Workers'],['Projects','/projects','Projects'],['Lab and 3D views','/lab','Lab'],
  ['Neuron review','/review','Review'],['Jobs','/jobs','Jobs'],
  ['Work distribution: tokens, speed and hardware per device','/work','Distribution'],['Neuron activation stat plots','/work#neurons','Distribution'],
  ['Feature dictionaries (sparse autoencoders)','/features#dicts','Features'],['Read a reply: features, emotions, logit lens','/features#read','Features'],
  ['Attribution graph (circuit tracing)','/features#circuits','Features'],['Emotion vectors','/features#emotions','Features'],
  ['Steer with features and emotions','/features#steer','Features'],['Train a dictionary or transcoder','/jobs#kind=feat_train','Features']];
let NS_FIND_EXTRA=null;
// ---- what this person uses: page visits counted in this browser only, for "recent" in search and for
// ordering the common tasks on the home page
function nsUsage(){ try{ return JSON.parse(localStorage.getItem('ns-usage')||'{}'); }catch(e){ return {}; } }
function nsRecentPages(){ try{ return JSON.parse(localStorage.getItem('ns-recent')||'[]'); }catch(e){ return []; } }
function nsTrack(h,t){ try{ const u=nsUsage(); u[h]=(u[h]||0)+1; localStorage.setItem('ns-usage',JSON.stringify(u));
  const r=nsRecentPages().filter(x=>x.h!==h); r.unshift({h,t:t||h,ts:Date.now()}); localStorage.setItem('ns-recent',JSON.stringify(r.slice(0,20))); }catch(e){} }
(function(){ const where=()=>{ const sec=location.hash&&/^#[a-z][\w-]*$/i.test(location.hash)?location.hash:'';
    return location.pathname+(location.pathname==='/'||location.pathname==='/jobs'?sec:''); };
  const name=()=>{ const n=NS_NAV.flatMap(it=>[it,...(it.kids||[])]).find(k=>k.href&&k.href.split('#')[0]===location.pathname&&(!k.href.includes('#')||k.href===where()));
    return (n&&n.label)||document.title.replace(/^Studio\s*[·:]?\s*/,'')||location.pathname; };
  const go=()=>nsTrack(where(),name()); if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',go); else go();
  addEventListener('hashchange',()=>{ if(location.pathname==='/'||location.pathname==='/jobs') go(); }); })();
async function nsFindIndex(){ const out=NS_SECTIONS.map(([t,h,d])=>({t,h,d}));
  NS_NAV.forEach(it=>{ if(it.href) out.push({t:it.label,h:it.href,d:it.tip}); (it.kids||[]).forEach(k=>out.push({t:k.label,h:k.href,d:it.label+': '+k.tip})); });
  if(!NS_FIND_EXTRA){ NS_FIND_EXTRA=[];
    try{ const sp=await (await fetch('/api/jobs/specs')).json(); for(const [k,s] of Object.entries(sp)) NS_FIND_EXTRA.push({t:s.title,h:'/jobs#'+k,d:'Job · '+(s.group||'')+' · '+(s.help||'')}); }catch(e){}
    try{ const hp=await (await fetch('/api/help/index')).json(); for(const x of hp.pages||[]) NS_FIND_EXTRA.push({t:x.title,h:'/help/'+x.name,d:'Help'}); }catch(e){} }
  out.push(...NS_FIND_EXTRA);
  // controls and sections on this page
  document.querySelectorAll('main label, main h2, main h3, main button, main summary').forEach((el,i)=>{ if(el.closest('#ns-find')) return;
    const t=(el.innerText||'').trim().split('\n')[0].slice(0,90); if(t.length<3) return;
    out.push({t,h:null,el,d:'On this page'+(el.title?' · '+el.title:'')}); });
  return out; }
function nsScore(it,q){ const t=it.t.toLowerCase(), all=(t+' '+(it.d||'')).toLowerCase(); const ws=q.toLowerCase().split(/\s+/).filter(Boolean);
  if(!ws.every(w=>all.includes(w))) return -1; let s=0; if(t.startsWith(ws[0])) s+=6; ws.forEach(w=>{ if(t.includes(w)) s+=3; if(new RegExp('\\b'+w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).test(t)) s+=2; });
  if(it.el) s+=1; return s-t.length/200; }
async function nsFind(){ if(document.getElementById('ns-find')) return; const idx=await nsFindIndex();
  const o=document.createElement('div'); o.id='ns-find';
  o.style.cssText='position:fixed;inset:0;z-index:9700;background:rgba(0,0,0,.45);display:flex;justify-content:center;align-items:flex-start;padding:10vh 12px';
  o.innerHTML=`<div role="dialog" aria-label="Search" style="width:min(640px,100%);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:12px;box-shadow:0 14px 40px rgba(0,0,0,.45);overflow:hidden;font:14px ui-sans-serif,system-ui">
    <input data-f="q" placeholder="Search pages, settings, jobs, help and this page's controls" aria-label="Search" style="width:100%;border:none;border-bottom:1px solid var(--line,#33332f);padding:.8rem 1rem;font:15px ui-sans-serif,system-ui;background:transparent;color:inherit;outline:none">
    <div data-f="list" role="listbox" style="max-height:55vh;overflow:auto;padding:.3rem"></div>
    <div style="padding:.35rem .8rem;font-size:11.5px;color:var(--mut,#9a9a94);border-top:1px solid var(--line,#33332f)">↑ ↓ to move · Enter to open · Esc to close</div></div>`;
  document.body.appendChild(o); const q=o.querySelector('[data-f=q]'), list=o.querySelector('[data-f=list]'); let hits=[], sel=0;
  const close=()=>{ o.remove(); document.removeEventListener('keydown',key,true); };
  const go=it=>{ close(); if(it.el){ it.el.scrollIntoView({block:'center'}); const tgt=it.el.tagName==='LABEL'&&it.el.htmlFor?document.getElementById(it.el.htmlFor):it.el;
      (tgt||it.el).focus&&(tgt||it.el).focus(); it.el.style.outline='2px solid var(--acc,#6f93ff)'; it.el.style.outlineOffset='3px'; setTimeout(()=>{ it.el.style.outline=''; },2500); return; }
    const [path,hash]=it.h.split('#'); if((path||'/')===location.pathname&&hash!==undefined){ location.hash=hash; const el=document.getElementById(hash); if(el) el.scrollIntoView({block:'start'}); } else location.href=it.h; };
  const draw=()=>{ const v=q.value.trim(); hits=v?idx.map(it=>[nsScore(it,v),it]).filter(x=>x[0]>=0).sort((a,b)=>b[0]-a[0]).slice(0,40).map(x=>x[1])
      :[...nsRecentPages().filter(r=>r.h!==location.pathname+location.hash&&r.h!==location.pathname).slice(0,5).map(r=>({t:r.t,h:r.h,d:'Recent · '+r.h})),
        ...idx.filter(it=>!it.el).slice(0,8)]; sel=Math.min(sel,Math.max(0,hits.length-1));
    list.innerHTML=hits.length?hits.map((it,i)=>`<div role="option" data-i="${i}" aria-selected="${i===sel}" style="padding:.45rem .7rem;border-radius:7px;cursor:pointer;${i===sel?'background:color-mix(in srgb,var(--acc,#6f93ff) 18%,transparent)':''}"><div>${nsEsc(it.t)}</div><div style="font-size:12px;color:var(--mut,#9a9a94);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${nsEsc(it.d||it.h||'')}</div></div>`).join('')
      :'<div style="padding:.8rem;color:var(--mut,#9a9a94)">Nothing matches.</div>';
    list.querySelectorAll('[data-i]').forEach(el=>{ el.onclick=()=>go(hits[+el.dataset.i]); el.onmousemove=()=>{ if(sel!==+el.dataset.i){ sel=+el.dataset.i; draw(); } }; }); };
  const key=e=>{ if(e.key==='Escape'){ e.preventDefault(); close(); } else if(e.key==='ArrowDown'){ e.preventDefault(); sel=Math.min(hits.length-1,sel+1); draw(); list.children[sel]?.scrollIntoView({block:'nearest'}); }
    else if(e.key==='ArrowUp'){ e.preventDefault(); sel=Math.max(0,sel-1); draw(); list.children[sel]?.scrollIntoView({block:'nearest'}); } else if(e.key==='Enter'&&hits[sel]){ e.preventDefault(); go(hits[sel]); } };
  document.addEventListener('keydown',key,true); o.onclick=e=>{ if(e.target===o) close(); };
  q.oninput=()=>{ sel=0; draw(); }; draw(); q.focus(); }
document.addEventListener('keydown',e=>{ const typing=/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName)||document.activeElement?.isContentEditable;
  if((e.key==='k'||e.key==='K')&&(e.ctrlKey||e.metaKey)){ e.preventDefault(); nsFind(); }
  else if(e.key==='/'&&!typing&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&window.self===window.top){ e.preventDefault(); nsFind(); } });
// ---- setting profiles: any number of named settings per tool; one can be the saved default.
// nsProfiles(el, {scope, get:()=>values, set:values=>{}, system:()=>{}}) -> {ready: Promise<default values or null>}
function nsProfiles(el,o){ let P={profiles:{},default:null};
  const btn='border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border-radius:6px;padding:.22rem .55rem;cursor:pointer;font:500 12px ui-sans-serif,system-ui';
  el.classList.add('ns-prof');
  const call=async(body)=>{ const r=await fetch('/api/profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scope:o.scope,...body})});
    const j=await r.json().catch(()=>({})); if(!r.ok) throw new Error(j.error||r.status); P=j; draw(); return j; };
  const msg=t=>{ const m=el.querySelector('[data-p=m]'); if(m){ m.textContent=t; clearTimeout(m._t); m._t=setTimeout(()=>m.textContent='',3500); } };
  function draw(){ const names=Object.keys(P.profiles).sort((a,b)=>a.localeCompare(b)); const cur=el.querySelector('[data-p=sel]')?.value||'';
    el.innerHTML=`<div style="display:flex;gap:.35rem;align-items:center;flex-wrap:wrap;font-size:12px">
      <select data-p="sel" aria-label="Setting profile" title="Saved settings for this tool" style="width:auto;max-width:14rem;font:12px ui-sans-serif,system-ui"><option value="">Profile…</option>${names.map(n=>`<option ${n===cur?'selected':''} value="${nsEsc(n)}">${nsEsc(n)}${n===P.default?' ★':''}</option>`).join('')}</select>
      <button type="button" data-p="save" style="${btn}" title="Save the current settings into the selected profile, or a new one">Save</button>
      <button type="button" data-p="saveas" style="${btn}" title="Save the current settings as a new profile">Save as…</button>
      <button type="button" data-p="star" style="${btn}" title="Load the selected profile automatically when this tool opens">★ Default</button>
      <button type="button" data-p="del" style="${btn}" title="Delete the selected profile">Delete</button>
      <span style="flex:1"></span>
      <button type="button" data-p="sys" style="${btn}" title="Reset every setting to OmniCortex's own defaults">System defaults</button>
      <button type="button" data-p="def" style="${btn}" ${P.default?'':'disabled'} title="${P.default?'Load the saved default: '+nsEsc(P.default):'No saved default yet: select a profile and press ★ Default'}">Saved default</button>
      <span data-p="m" style="color:var(--mut,#9a9a94);min-width:0"></span></div>`;
    const q=k=>el.querySelector(`[data-p=${k}]`), sel=q('sel');
    sel.onchange=()=>{ if(sel.value&&P.profiles[sel.value]){ o.set(P.profiles[sel.value]); msg('loaded '+sel.value); } };
    q('save').onclick=async()=>{ let n=sel.value; if(!n){ n=prompt('Name for these settings'); if(!n) return; }
      try{ await call({action:'save',name:n,values:o.get()}); el.querySelector('[data-p=sel]').value=n; msg('saved'); }catch(e){ msg(e.message); } };
    q('saveas').onclick=async()=>{ const n=prompt('Name for these settings'); if(!n) return;
      try{ await call({action:'save',name:n,values:o.get()}); el.querySelector('[data-p=sel]').value=n; msg('saved as '+n); }catch(e){ msg(e.message); } };
    q('star').onclick=async()=>{ if(!sel.value) return msg('select a profile first');
      try{ const n=sel.value; await call({action:'default',name:n}); el.querySelector('[data-p=sel]').value=n; msg(n+' opens by default now'); }catch(e){ msg(e.message); } };
    q('del').onclick=async()=>{ if(!sel.value||!confirm(`Delete profile ${sel.value}?`)) return;
      try{ await call({action:'delete',name:sel.value}); msg('deleted'); }catch(e){ msg(e.message); } };
    q('sys').onclick=()=>{ o.system(); el.querySelector('[data-p=sel]').value=''; msg('system defaults'); };
    q('def').onclick=()=>{ if(P.default&&P.profiles[P.default]){ o.set(P.profiles[P.default]); el.querySelector('[data-p=sel]').value=P.default; msg('saved default: '+P.default); } }; }
  draw();
  const ready=fetch('/api/profiles?scope='+encodeURIComponent(o.scope)).then(r=>r.ok?r.json():null).then(j=>{ if(j){ P=j; draw();
    if(P.default&&P.profiles[P.default]){ el.querySelector('[data-p=sel]').value=P.default; return P.profiles[P.default]; } } return null; }).catch(()=>null);
  return {ready}; }
// ---- files and folders on Studio's computer: a browse dialog, path fields, per-entry lists
function nsEsc(x){ return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function nsRecent(kind){ try{ return JSON.parse(localStorage.getItem('ns-recent-'+kind)||'[]'); }catch(e){ return []; } }
function nsRemember(kind,p){ try{ const r=[p,...nsRecent(kind).filter(x=>x!==p)].slice(0,8); localStorage.setItem('ns-recent-'+kind,JSON.stringify(r)); }catch(e){} }
function nsFmtSize(b){ if(b==null) return ''; const u=['B','KB','MB','GB','TB']; let i=0; while(b>=1024&&i<4){ b/=1024; i++; } return (i?b.toFixed(1):b)+' '+u[i]; }
// opts: {want:'dir'|'file'|'any', ext:['gguf'], start:'/path', title:'Choose…'} -> Promise<path or null>
function nsPickPath(opts){ opts=opts||{}; const want=opts.want||'dir', ext=(opts.ext||[]).join(',');
  return new Promise(resolve=>{
    const back=document.createElement('div'); back.className='ns-pick';
    back.style.cssText='position:fixed;inset:0;z-index:9500;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;padding:12px';
    back.innerHTML=`<div role="dialog" aria-modal="true" aria-label="${nsEsc(opts.title||'Choose')}" style="width:min(860px,100%);height:min(620px,100%);display:flex;flex-direction:column;background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.4);font:14px/1.45 ui-sans-serif,system-ui;overflow:hidden">
     <div style="display:flex;align-items:center;gap:.5rem;padding:.7rem .9rem;border-bottom:1px solid var(--line,#33332f)"><b style="font-size:16px;flex:1">${nsEsc(opts.title||(want==='dir'?'Choose a folder':'Choose a file'))}</b><button data-p="x" aria-label="Close" style="background:none;border:none;color:inherit;font-size:20px;cursor:pointer">×</button></div>
     <div style="display:flex;flex:1;min-height:0">
      <nav data-p="places" style="width:200px;flex:none;overflow:auto;border-right:1px solid var(--line,#33332f);padding:.4rem;font-size:13px"></nav>
      <div style="flex:1;display:flex;flex-direction:column;min-width:0">
       <div style="display:flex;gap:.4rem;padding:.5rem;border-bottom:1px solid var(--line,#33332f)"><button data-p="up" title="Up one folder">↑</button><input data-p="path" aria-label="Path" style="flex:1;min-width:0;font:13px ui-monospace,monospace;padding:.3rem .45rem;border:1px solid var(--line,#33332f);border-radius:6px;background:var(--bg,#151514);color:inherit"><input data-p="filter" placeholder="Filter" aria-label="Filter" style="width:8rem;padding:.3rem .45rem;border:1px solid var(--line,#33332f);border-radius:6px;background:var(--bg,#151514);color:inherit"></div>
       <div data-p="list" role="listbox" style="flex:1;overflow:auto;padding:.25rem"></div>
      </div></div>
     <div style="display:flex;gap:.5rem;align-items:center;padding:.6rem .9rem;border-top:1px solid var(--line,#33332f)"><span data-p="sel" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:12px ui-monospace,monospace;color:var(--mut,#9a9a94)"></span>
      <label style="font-size:12px;color:var(--mut,#9a9a94)"><input type="checkbox" data-p="hidden"> hidden</label><button data-p="cancel">Cancel</button><button data-p="ok" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.35rem .9rem;font-weight:600;cursor:pointer"></button></div></div>`;
    document.body.appendChild(back);
    const q=k=>back.querySelector(`[data-p=${k}]`);
    if(innerWidth<640){ const nav=q('places'); nav.parentNode.style.flexDirection='column'; nav.style.cssText+=';width:auto;max-height:7.5rem;border-right:none;border-bottom:1px solid var(--line,#33332f);display:flex;flex-wrap:wrap;gap:.1rem .3rem;align-content:flex-start';
      q('filter').style.width='5rem'; }
    back.querySelectorAll('button').forEach(b=>{ if(!b.style.cssText) b.style.cssText='border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer'; });
    let cur=null, chosen=null, data=null;
    const done=v=>{ back.remove(); document.removeEventListener('keydown',key,true); if(v){ nsRemember(want,v); } resolve(v); };
    const setSel=p=>{ chosen=p; q('sel').textContent=p||''; q('ok').disabled=!p; q('ok').style.opacity=p?1:.5; };
    q('ok').textContent=want==='dir'?'Choose this folder':'Choose';
    const draw=()=>{ const f=q('filter').value.toLowerCase(); const rows=(data.entries||[]).filter(e=>!f||e.name.toLowerCase().includes(f));
      q('list').innerHTML=(data.error?`<div style="padding:.6rem;color:var(--no,#ff7a6b)">${nsEsc(data.error)}</div>`:'')+(rows.length?rows.map((e,i)=>`<div role="option" tabindex="-1" data-i="${i}" style="display:flex;gap:.6rem;align-items:center;padding:.3rem .5rem;border-radius:6px;cursor:pointer"><span aria-hidden="true">${e.dir?'📁':'📄'}</span><span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${nsEsc(e.name)}</span><span style="font-size:12px;color:var(--mut,#9a9a94)">${nsFmtSize(e.size)}</span></div>`).join(''):`<div style="padding:.8rem;color:var(--mut,#9a9a94)">${want==='dir'?'No folders here.':'Nothing here matches.'}</div>`)+(data.truncated?'<div style="padding:.5rem;color:var(--mut,#9a9a94)">Only the first entries are shown; filter to narrow down.</div>':'');
      q('list').querySelectorAll('[data-i]').forEach(el=>{ const e=rows[+el.dataset.i];
        el.onmouseenter=()=>el.style.background='var(--code,#262624)'; el.onmouseleave=()=>{ if(chosen!==e.path) el.style.background=''; };
        el.onclick=()=>{ q('list').querySelectorAll('[data-i]').forEach(x=>x.style.background=''); el.style.background='var(--code,#262624)';
          if(e.dir){ if(want==='file') return go(e.path); setSel(e.path); } else setSel(e.path); };
        el.ondblclick=()=>{ if(e.dir) go(e.path); else done(e.path); }; }); };
    const go=async p=>{ q('list').innerHTML='<div style="padding:.8rem;color:var(--mut,#9a9a94)">…</div>';
      try{ const r=await fetch(`/api/fs/list?want=${want}&ext=${encodeURIComponent(ext)}&hidden=${q('hidden').checked?1:0}&path=${encodeURIComponent(p||'')}`); data=await r.json(); if(!r.ok) throw new Error(data.error||r.status); }
      catch(e){ data={entries:[],error:String(e.message||e),path:p}; }
      cur=data.path; q('path').value=cur||''; q('filter').value=''; setSel(want==='file'?null:cur); draw(); };
    q('filter').oninput=draw; q('hidden').onchange=()=>go(cur);
    q('path').onkeydown=e=>{ if(e.key==='Enter'){ e.preventDefault(); go(q('path').value); } };
    q('up').onclick=()=>data&&data.parent&&go(data.parent);
    q('x').onclick=q('cancel').onclick=()=>done(null); back.onclick=e=>{ if(e.target===back) done(null); };
    q('ok').onclick=()=>chosen&&done(chosen);
    const key=e=>{ if(e.key==='Escape'){ e.stopPropagation(); done(null); } }; document.addEventListener('keydown',key,true);
    fetch('/api/fs/places').then(r=>r.json()).then(j=>{ const rec=nsRecent(want);
      const item=(l,p)=>`<a href="#" data-go="${nsEsc(p)}" title="${nsEsc(p)}" style="display:block;padding:.3rem .45rem;border-radius:6px;color:inherit;text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${nsEsc(l)}</a>`;
      q('places').innerHTML=(rec.length?'<div style="font-size:11px;color:var(--mut,#9a9a94);margin:.3rem .45rem;text-transform:uppercase">Recent</div>'+rec.map(p=>item(p.split(/[\\/]/).filter(Boolean).pop()||p,want==='file'?p.replace(/[\\/][^\\/]*$/,'')||p:p)).join(''):'')
        +'<div style="font-size:11px;color:var(--mut,#9a9a94);margin:.5rem .45rem .3rem;text-transform:uppercase">Places</div>'+(j.places||[]).map(x=>item(x.label,x.path)).join('');
      q('places').querySelectorAll('[data-go]').forEach(a=>a.onclick=e=>{ e.preventDefault(); go(a.dataset.go); });
      go(opts.start||rec[0]&&(want==='file'?rec[0].replace(/[\\/][^\\/]*$/,''):rec[0])||(j.places||[])[0]?.path||''); }).catch(()=>go(opts.start||''));
  }); }
// An input with a Browse… button (and suggestions) next to it. opts as nsPickPath, plus suggest:[paths].
function nsPathField(input,opts){ if(!input||input.dataset.nsPick==='1') return; input.dataset.nsPick='1'; opts=opts||{};
  const wrap=document.createElement('span'); wrap.style.cssText='display:flex;gap:.35rem;align-items:center;width:100%';
  input.parentNode.insertBefore(wrap,input); wrap.appendChild(input); input.style.flex='1'; input.style.minWidth='0';
  const sug=[...new Set([...(opts.suggest||[]),...nsRecent(opts.want||'dir')])];
  if(sug.length){ const id='nsdl'+Math.random().toString(36).slice(2,8); const dl=document.createElement('datalist'); dl.id=id; dl.innerHTML=sug.map(x=>`<option value="${nsEsc(x)}">`).join(''); wrap.appendChild(dl); input.setAttribute('list',id); }
  const b=document.createElement('button'); b.type='button'; b.textContent='Browse…'; b.className='ns-browse';
  b.style.cssText='flex:none;border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border-radius:6px;padding:.28rem .6rem;cursor:pointer;font:500 12px ui-sans-serif,system-ui';
  b.onclick=async()=>{ const v=input.value.trim(); const p=await nsPickPath({...opts,start:v?(opts.want==='file'?v.replace(/[\\/][^\\/]*$/,''):v):opts.start});
    if(p){ input.value=p; input.dispatchEvent(new Event('input',{bubbles:true})); input.dispatchEvent(new Event('change',{bubbles:true})); } };
  wrap.appendChild(b); return wrap; }
// A list edited one entry at a time: each row has its path, Browse…, ↑ ↓ and ×; "+ Add" adds a row.
// opts: {values, want, ext, addLabel, placeholder, onchange}. -> {get(), set(values), el}
function nsListEditor(container,opts){ opts=opts||{}; let vals=[...(opts.values||[])];
  const el=document.createElement('div'); el.className='ns-list'; container.innerHTML=''; container.appendChild(el);
  const changed=()=>opts.onchange&&opts.onchange(vals.filter(Boolean));
  const btn='border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border-radius:6px;padding:.2rem .5rem;cursor:pointer;font:500 12px ui-sans-serif,system-ui';
  const draw=()=>{ el.innerHTML=(vals.length?'':`<div style="font-size:12px;color:var(--mut,#9a9a94);padding:.2rem 0">${nsEsc(opts.empty||'None yet.')}</div>`)+vals.map((v,i)=>`<div data-r="${i}" style="display:flex;gap:.35rem;align-items:center;margin:.25rem 0"><input value="${nsEsc(v)}" placeholder="${nsEsc(opts.placeholder||'')}" aria-label="entry ${i+1}" style="flex:1;min-width:0;font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line,#33332f);border-radius:5px;background:var(--bg,#151514);color:var(--fg,#ececea)">
      ${opts.browse===false?'':`<button type="button" data-a="browse" style="${btn}">Browse…</button>`}<button type="button" data-a="up" title="Move up" aria-label="Move up" style="${btn}" ${i?'':'disabled'}>↑</button><button type="button" data-a="down" title="Move down" aria-label="Move down" style="${btn}" ${i<vals.length-1?'':'disabled'}>↓</button><button type="button" data-a="rm" title="Remove" aria-label="Remove" style="${btn};color:var(--no,#ff7a6b)">×</button></div>`).join('')
      +`<button type="button" data-a="add" style="${btn};margin-top:.2rem">${nsEsc(opts.addLabel||'+ Add')}</button>`;
    el.querySelectorAll('[data-r]').forEach(row=>{ const i=+row.dataset.r, inp=row.querySelector('input');
      inp.oninput=inp.onchange=()=>{ vals[i]=inp.value.trim(); changed(); };
      if(opts.browse!==false) row.querySelector('[data-a=browse]').onclick=async()=>{ const p=await nsPickPath({want:opts.want||'dir',ext:opts.ext,start:vals[i]||undefined}); if(p){ vals[i]=p; draw(); changed(); } };
      row.querySelector('[data-a=up]').onclick=()=>{ [vals[i-1],vals[i]]=[vals[i],vals[i-1]]; draw(); changed(); };
      row.querySelector('[data-a=down]').onclick=()=>{ [vals[i+1],vals[i]]=[vals[i],vals[i+1]]; draw(); changed(); };
      row.querySelector('[data-a=rm]').onclick=()=>{ vals.splice(i,1); draw(); changed(); }; });
    el.querySelector('[data-a=add]').onclick=async()=>{ if(opts.browse===false){ get(); vals.push(''); draw(); const ins=el.querySelectorAll('[data-r] input'); ins[ins.length-1].focus(); return; } const p=await nsPickPath({want:opts.want||'dir',ext:opts.ext}); if(p&&!vals.includes(p)){ vals.push(p); draw(); changed(); } }; };
  const get=()=>{ el.querySelectorAll('[data-r] input').forEach((inp,i)=>vals[i]=inp.value.trim()); return vals.filter(Boolean); };
  draw();
  return {el, get, set:v=>{ vals=[...v]; draw(); }}; }
// Inputs marked data-pick="dir|file|any" (data-ext="gguf,npz") get a Browse… button, also when added later.
// Textareas marked data-lines="dir|file|text" become per-entry lists; the textarea stays (hidden) as the value.
function nsEnhancePickers(root){ (root||document).querySelectorAll('input[data-pick]').forEach(i=>nsPathField(i,{want:i.dataset.pick,ext:(i.dataset.ext||'').split(',').filter(Boolean)}));
  (root||document).querySelectorAll('textarea[data-lines]').forEach(t=>{ if(t.dataset.nsPick==='1') return; t.dataset.nsPick='1'; t.hidden=true;
    const d=document.createElement('div'); t.after(d); const w=t.dataset.lines;
    nsListEditor(d,{values:t.value.split('\n').map(x=>x.trim()).filter(Boolean),want:w,browse:w!=='text',placeholder:t.placeholder,
      addLabel:t.dataset.add||(w==='dir'?'+ Add folder':w==='file'?'+ Add file':'+ Add'),empty:t.dataset.empty||'None yet.',
      onchange:v=>{ t.value=v.join('\n'); t.dispatchEvent(new Event('input',{bubbles:true})); }}); }); }
(function(){ const start=()=>{ nsEnhancePickers(document); try{ new MutationObserver(ms=>{ for(const m of ms) for(const n of m.addedNodes) if(n.nodeType===1){ if(n.matches&&n.matches('input[data-pick]')) nsEnhancePickers(n.parentNode); else nsEnhancePickers(n); } }).observe(document.body,{childList:true,subtree:true}); }catch(e){} };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start); else start(); })();
// ---- text that does not fit: long unbroken text (paths, ids, hashes) in a cell is cut with … and every cut
// text, cell or field gets a tooltip with all of it, also for what pages add later
function nsFit(root){
  root=root||document;
  root.querySelectorAll('td,th').forEach(el=>{ if(!el.childElementCount&&!el.classList.contains('ns-clip')&&/\S{40,}/.test(el.textContent)) el.classList.add('ns-clip'); });
  root.querySelectorAll('td,th,code,a,button,summary,h2,h3,.chip,.name,.note,label,option').forEach(el=>{
    if(el.dataset.nsTip==='own'||(el.title&&el.dataset.nsTip!=='auto')) { if(el.title) el.dataset.nsTip='own'; return; }
    const full=(el.textContent||'').trim();
    if(full&&el.clientWidth&&el.scrollWidth>el.clientWidth+1){ el.title=full; el.dataset.nsTip='auto'; }
    else if(el.dataset.nsTip==='auto'){ el.removeAttribute('title'); delete el.dataset.nsTip; } });
  root.querySelectorAll('input:not([type]),input[type=text],input[type=url],input[type=search]').forEach(nsFitInput); }
function nsFitInput(i){
  if(i.dataset.nsHelp===undefined){ i.dataset.nsHelp=i.title||''; i.addEventListener('input',()=>nsFitInput(i)); i.addEventListener('change',()=>nsFitInput(i)); }
  const help=i.dataset.nsHelp, over=i.value&&i.clientWidth&&i.scrollWidth>i.clientWidth+1;
  i.title=over?(help?help+'\n\n':'')+i.value:help; }
(function(){ const css=document.createElement('style');
  css.textContent='.ns-clip{max-width:min(26rem,55vw);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}';
  let t=null; const later=()=>{ clearTimeout(t); t=setTimeout(()=>{ if(!document.hidden) nsFit(document); },250); };
  const start=()=>{ document.head.appendChild(css); nsFit(document);
    try{ new MutationObserver(ms=>{ if(ms.some(m=>[...m.addedNodes].some(n=>n.nodeType===1||n.nodeType===3))) later(); })
      .observe(document.body,{childList:true,subtree:true,characterData:true}); }catch(e){}
    addEventListener('resize',later); };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start); else start(); })();
// ---- the tour: optional, offered once, restartable from the ? button on every page
const NS_TOUR=[
 {path:'/', sel:'header', title:'Welcome to OmniCortex Studio', body:'Studio runs your local models, lets you chat with them, and shows what happens inside them: which neurons fire, and how likely a reply is to be a confident guess. Two minutes, eleven stops. Skip any time.'},
 {path:'/', sel:'#ns-rail', title:'The sidebar', body:'Everything Studio does is one click away here, on every page, in a few groups. Models: the GGUF files in your folders (load one, see its measured accuracy, set its options) and Discover to download more. Tasks: jobs, projects and the workers that run them. Studio: the analysis tools (features and emotions, neuron review, labs). Connect: other apps, other computers, your devices. Hover a group to see its pages; inside one, tabs at the top switch between them.'},
 {path:'/', sel:'#in', title:'Chat', body:'Ask anything. The model picker below chooses which model answers; auto picks the one with the best measured record for the question.'},
 {path:'/', sel:'label:has(#autoCheck),label:has(#confOn)', title:'How sure was it?', body:'With confidence on, Studio marks the words the model was unsure about. check replies scores each reply with the model\'s hallucination classifier and opens a 3D view of its neurons.'},
 {path:'/setup', sel:'#status-card', title:'Setup', body:'What is ready and the one thing to do next. Build llama.cpp here with one button, add model folders, and run the health check when something seems off.'},
 {path:'/jobs', sel:'#kind', title:'Jobs', body:'Longer work runs here: TestQA to measure models, retraining on what they get wrong, benchmarks, the logit lens and the decoding lab. Each job has a form; its log shows live.'},
 {path:'/features', sel:'#ftabs', title:'Features', body:'What a model represents, as features instead of neurons: train or import a dictionary, read any reply token by token, trace which features produced an output, explore emotion vectors, and steer with both. Start here walks through it.'},
 {path:'/review', sel:'#heat', title:'Neuron review', body:'Every checked reply leaves its activations here. Colour neurons by how they relate to right and wrong answers, per subject, over time.'},
 {path:'/projects', sel:'main', title:'Projects', body:'Paste a project\'s notes; the director splits it into tasks by skill. You approve the plan, worker models do the tasks, you review the results.'},
 {path:'/workers', sel:'#workers', title:'Workers', body:'Where those tasks run: models on this computer, servers you started elsewhere, and other computers running OmniCortex Remote. Pause, limit, route subjects, and watch GPU temperatures.'},
 {path:'/connect', sel:'main', title:'Connect your other apps', body:'Use these models from Cline, Claude Desktop or any OpenAI-compatible app, and give agents OmniCortex\'s tools over MCP. This page shows who controls whom.'},
 {path:'/settings', sel:'#appearance', title:'Make it yours', body:'Themes, workspaces, models per subject, tool approval, plugins and opt-in sharing. The ? button at the bottom right restarts this tour, and opens the help pages.'}];
function nsTourGet(){ try{ return JSON.parse(localStorage.getItem('ns-tour')||'null'); }catch(e){ return null; } }
function nsTourSet(v){ try{ localStorage.setItem('ns-tour',JSON.stringify(v)); }catch(e){} }
function nsTourEnd(state){ nsTourSet({state}); document.querySelectorAll('.ns-tour').forEach(e=>e.remove()); }
function nsTourStart(i){ nsTourSet({state:'running',i:i||0}); nsTourShow(); }
function nsTourShow(){ const t=nsTourGet(); if(!t||t.state!=='running') return; const st=NS_TOUR[t.i]; if(!st){ nsTourEnd('done'); return; }
  if(location.pathname!==st.path){ location.href=st.path; return; }
  document.querySelectorAll('.ns-tour').forEach(e=>e.remove());
  // The spotlight covers every element the step names (a comma list), or the screen's centre if none is here.
  const els=[...document.querySelectorAll(st.sel)].filter(e=>e.getClientRects().length), el=els[0], r=el?el.getBoundingClientRect():null;
  if(el&&el.scrollIntoView&&(r.top<0||r.bottom>innerHeight)) el.scrollIntoView({block:'center'});
  const rs=els.map(e=>e.getBoundingClientRect()), pad=6;
  const rr=rs.length?{left:Math.min(...rs.map(x=>x.left)),top:Math.min(...rs.map(x=>x.top)),right:Math.max(...rs.map(x=>x.right)),bottom:Math.max(...rs.map(x=>x.bottom))}:null;
  if(rr){ rr.width=rr.right-rr.left; rr.height=rr.bottom-rr.top; }
  const hole=document.createElement('div'); hole.className='ns-tour';
  hole.style.cssText=rr?`position:fixed;z-index:9998;left:${rr.left-pad}px;top:${rr.top-pad}px;width:${Math.min(rr.width+2*pad,innerWidth)}px;height:${Math.min(rr.height+2*pad,innerHeight*0.7)}px;border-radius:8px;box-shadow:0 0 0 9999px rgba(0,0,0,.55);outline:2px solid var(--acc,#6f93ff);pointer-events:none`
    :'position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,.55);pointer-events:none';
  const card=document.createElement('div'); card.className='ns-tour'; card.setAttribute('role','dialog'); card.setAttribute('aria-label',st.title);
  const below=rr&&rr.bottom+220<innerHeight, top=rr?(below?rr.bottom+14:Math.max(12,rr.top-14-200)):innerHeight/2-110;
  card.style.cssText=`position:fixed;z-index:9999;left:${Math.max(12,Math.min((rr?rr.left:innerWidth/2-170),innerWidth-352))}px;top:${Math.max(12,Math.min(top,innerHeight-230))}px;width:min(340px,calc(100vw - 24px));background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:10px;padding:.85rem .95rem;font:14px/1.5 ui-sans-serif,system-ui;box-shadow:0 8px 30px rgba(0,0,0,.35)`;
  const n=NS_TOUR.length;
  card.innerHTML=`<div style="font-size:11px;color:var(--mut,#9a9a94)">${t.i+1} of ${n}</div><b style="font-size:15px">${st.title}</b><div style="margin:.35rem 0 .7rem">${st.body}</div>
   <div style="display:flex;gap:.4rem;align-items:center"><button data-t="skip" style="background:none;border:none;color:var(--mut,#9a9a94);cursor:pointer;font:inherit;padding:0">End tour</button><span style="flex:1"></span>
   ${t.i?'<button data-t="back">Back</button>':''}<button data-t="next" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.3rem .8rem;cursor:pointer;font:600 13px ui-sans-serif,system-ui">${t.i+1<n?'Next':'Finish'}</button></div>`;
  card.querySelectorAll('button[data-t=back]').forEach(b=>b.style.cssText='border:1px solid var(--line,#33332f);background:none;color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer');
  document.body.append(hole,card);
  const go=d=>{ const k=t.i+d; if(k>=n) return nsTourEnd('done'); nsTourSet({state:'running',i:Math.max(0,k)}); nsTourShow(); };
  card.querySelector('[data-t=next]').onclick=()=>go(1); card.querySelector('[data-t=skip]').onclick=()=>nsTourEnd('skipped');
  const back=card.querySelector('[data-t=back]'); if(back) back.onclick=()=>go(-1);
  card.querySelector('[data-t=next]').focus();
  card.onkeydown=e=>{ if(e.key==='Escape') nsTourEnd('skipped'); if(e.key==='ArrowRight') go(1); if(e.key==='ArrowLeft'&&t.i) go(-1); }; }
function nsHelpButton(){ if(document.getElementById('ns-help')||window.self!==window.top) return;
  const b=document.createElement('button'); b.id='ns-help'; b.textContent='?'; b.title='Tour and help'; b.setAttribute('aria-label','Tour and help');
  const lift=innerWidth<=700?58:0;
  b.style.cssText='position:fixed;right:14px;bottom:'+((location.pathname==='/'?150:14)+lift)+'px;z-index:9000;width:34px;height:34px;border-radius:50%;border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);font:700 15px ui-sans-serif,system-ui;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25)';
  const m=document.createElement('div'); m.id='ns-help-menu'; m.hidden=true;
  m.style.cssText='position:fixed;right:14px;bottom:'+((location.pathname==='/'?190:54)+lift)+'px;z-index:9000;background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:8px;padding:.35rem;font:13px ui-sans-serif,system-ui;box-shadow:0 4px 16px rgba(0,0,0,.3)';
  const here=NS_TOUR.findIndex(s=>s.path===location.pathname);
  m.innerHTML=`<a href="#" data-h="tour" style="display:block;padding:.3rem .6rem;color:inherit">Take the tour</a>${here>=0?'<a href="#" data-h="here" style="display:block;padding:.3rem .6rem;color:inherit">This page in the tour</a>':''}<a href="/help/GETTING_STARTED" style="display:block;padding:.3rem .6rem;color:inherit">Getting started</a><a href="/help/UI_AND_CLI" style="display:block;padding:.3rem .6rem;color:inherit">Where everything is</a><a href="/help/GLOSSARY" style="display:block;padding:.3rem .6rem;color:inherit">Glossary</a><a href="/help/ROLES" style="display:block;padding:.3rem .6rem;color:inherit">Roles and ports</a><a href="/help/CREDITS" style="display:block;padding:.3rem .6rem;color:inherit">Credits</a>`;
  b.onclick=()=>{ m.hidden=!m.hidden; };
  m.querySelector('[data-h=tour]').onclick=e=>{ e.preventDefault(); m.hidden=true; nsTourStart(0); };
  const h=m.querySelector('[data-h=here]'); if(h) h.onclick=e=>{ e.preventDefault(); m.hidden=true; nsTourStart(here); };
  document.body.append(b,m); }
function nsTourOffer(){ if(location.pathname!=='/'||nsTourGet()||window.self!==window.top) return;
  const o=document.createElement('div'); o.className='ns-tour'; o.setAttribute('role','dialog');
  o.style.cssText='position:fixed;right:14px;bottom:'+(location.pathname==='/'?'192':'56')+'px;z-index:9001;width:min(320px,calc(100vw - 28px));background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--acc,#6f93ff);border-radius:10px;padding:.8rem .9rem;font:14px/1.5 ui-sans-serif,system-ui;box-shadow:0 8px 30px rgba(0,0,0,.35)';
  o.innerHTML='<b>New here?</b><div style="margin:.3rem 0 .6rem">A two-minute tour shows where everything is. It is optional; the ? button brings it back later.</div><div style="display:flex;gap:.4rem;flex-wrap:wrap"><button data-o="go" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.3rem .8rem;cursor:pointer;font:600 13px ui-sans-serif,system-ui">Show me around</button><button data-o="later" style="border:1px solid var(--line,#33332f);background:none;color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer">Not now</button><button data-o="never" style="border:none;background:none;color:var(--mut,#9a9a94);cursor:pointer">Don\'t ask again</button></div>';
  o.querySelector('[data-o=go]').onclick=()=>{ o.remove(); nsTourStart(0); };
  o.querySelector('[data-o=later]').onclick=()=>o.remove();
  o.querySelector('[data-o=never]').onclick=()=>{ nsTourSet({state:'declined'}); o.remove(); };
  document.body.appendChild(o); }
(function(){ const init=()=>{ nsHelpButton(); const t=nsTourGet(); if(t&&t.state==='running') setTimeout(nsTourShow,350); else nsTourOffer(); };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init(); })();
// Structured links a server reply may carry: [{href, label}].
function linkList(links){ return (links||[]).filter(l=>l&&typeof l.href==='string'&&l.href.startsWith('/'))
  .map(l=>`<a href="${l.href.replace(/"/g,'&quot;')}">${String(l.label||l.href).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</a>`).join(' · '); }
"""


def js() -> str:
    import json
    return JS.replace("__TARGETS__", json.dumps([[rx, href] for rx, href in TARGETS]))


def inject(page: str) -> str:
    """Add linkify() to a page, just before its last </script>."""
    i = page.rindex("</script>")
    return page[:i] + js() + page[i:]
