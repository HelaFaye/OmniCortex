"""
Widgets for OmniCortex's views: small, self-contained pieces of UI that take
data and settings from outside and can be put on any page (docs/WIDGETS.md).

The Godot client has the same widgets in plugins/godot-client/client/widgets/, with the same
settings keys and the same drawing, so a theme reads the same in both.

    jewel          the emotion jewel: colour, size and motion from the emotion
                   vectors at the current token. Styles: core, icosphere,
                   tesseract, off. Click it to hide or show the emotion graph.
    emotion_graph  one lane per emotion over the reply.
    nav_gizmo      (Godot) Blender-style axis gizmo, zoom, pan, view presets.
    chat_box       (Godot) search the trace, or ask the model.

Settings come from three places, later ones winning: DEFAULTS here, the 3D
theme's own "widgets" block in themes.json, and the owner's choices saved by
Studio (Settings -> Appearance -> Widgets). The merged block is served with
the theme (api/theme) as "widgets"; a viewer's own picks in a client (the
jewel's right-click, its toggle) are kept by that client.
"""
from __future__ import annotations

import copy

# Every setting a widget reads, with its kind and range, so Settings can draw
# a form and Studio can refuse bad values.
SPEC = {
    "jewel": {
        "style": {"kind": "choice", "choices": ["core", "off"], "default": "core",
                  "help": "core: a reactor core with a ribbon per emotion; add-ons add more (jewel-styles: "
                          "icosphere, tesseract)"},
        "speed": {"kind": "float", "min": 0.1, "max": 4.0, "default": 1.0, "help": "animation speed"},
        "glow": {"kind": "float", "min": 0.0, "max": 2.0, "default": 1.0, "help": "halo and core brightness"},
        "scale": {"kind": "float", "min": 0.5, "max": 1.0, "default": 0.9, "help": "size within its box"},
    },
    "emotion_graph": {
        "lanes": {"kind": "int", "min": 1, "max": 12, "default": 6, "help": "emotions drawn, strongest first"},
        "labels": {"kind": "bool", "default": True, "help": "names in their lane colours"},
    },
    "nav_gizmo": {
        "visible": {"kind": "bool", "default": True, "help": "show the navigation gizmo (Godot)"},
        "sway": {"kind": "bool", "default": True,
                 "help": "the slow automatic sway; it pauses when you move the view, and the gizmo's sway button "
                         "or S turns it back on"},
    },
    "chat_box": {
        "max_hits": {"kind": "int", "min": 3, "max": 30, "default": 8, "help": "search results shown"},
    },
}

DEFAULTS = {w: {k: v["default"] for k, v in spec.items()} for w, spec in SPEC.items()}


def spec(extra_styles=()) -> dict:
    """SPEC with the jewel styles that add-ons provide (plugin.json "widgets": {"jewel_styles": [...]})."""
    s = copy.deepcopy(SPEC)
    s["jewel"]["style"]["choices"] = ["core", *[x for x in extra_styles if x not in ("core", "off")], "off"]
    return s


def validate(block: dict, extra_styles=()) -> dict:
    """Keep only known widgets and settings with valid values; raise on a bad value."""
    SPEC = spec(extra_styles)                                   # noqa: N806 - the spec with add-on styles
    out = {}
    for w, vals in (block or {}).items():
        if w not in SPEC or not isinstance(vals, dict):
            raise ValueError(f"unknown widget: {w}")
        for k, v in vals.items():
            s = SPEC[w].get(k)
            if s is None:
                raise ValueError(f"{w}: unknown setting {k}")
            if s["kind"] == "choice" and v not in s["choices"]:
                raise ValueError(f"{w}.{k}: one of {', '.join(s['choices'])}")
            if s["kind"] == "bool" and not isinstance(v, bool):
                raise ValueError(f"{w}.{k}: true or false")
            if s["kind"] in ("int", "float"):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not s["min"] <= v <= s["max"]:
                    raise ValueError(f"{w}.{k}: a number from {s['min']} to {s['max']}")
                v = int(v) if s["kind"] == "int" else float(v)
            out.setdefault(w, {})[k] = v
    return out


def merged(theme: dict | None, owner: dict | None = None) -> dict:
    out = copy.deepcopy(DEFAULTS)
    for block in ((theme or {}).get("widgets") or {}, owner or {}):
        for w, vals in block.items():
            if w in out and isinstance(vals, dict):
                out[w].update({k: v for k, v in vals.items() if k in SPEC[w]})
    return out


def with_widgets(theme: dict, owner: dict | None = None) -> dict:
    """The theme as served to a 3D view: itself plus the merged widgets block."""
    t = dict(theme)
    t["widgets"] = merged(theme, owner)
    return t


# The web widgets. Each page includes WIDGETS_JS once; the drawing matches
# plugins/godot-client/client/widgets/jewel.gd so the two clients agree.
WIDGETS_JS = r"""
const NSWidgets = (() => {
  const reg = {}, STYLES = {};
  // Jewel styles from add-ons (the jewel-styles add-on: icosphere, tesseract). An add-on's script calls
  // NSWidgets.jewelStyle(name, (ctx, J) => …) with J = {cx, cy, r, cur, mix, mag, t, color, css, dark, light}.
  function jewelStyle(name, fn){ STYLES[name] = fn; }
  // Inside Studio, load the scripts of the widget add-ons that are on; elsewhere (bloom.py alone) there are none.
  try{ fetch('/api/addons/widgets').then(r => r.ok ? r.json() : null).then(j => { (j && j.js || []).forEach(x => {
    const el = document.createElement('script'); el.src = x.url; document.head.appendChild(el); }); }).catch(() => {}); }catch(e){}
  const store = { get(k, d){ try{ const v = localStorage.getItem('ns-w-'+k); return v===null ? d : JSON.parse(v); }catch(e){ return d; } },
                  set(k, v){ try{ localStorage.setItem('ns-w-'+k, JSON.stringify(v)); }catch(e){} } };
  function register(name, factory){ reg[name] = factory; }
  function create(name, el, opts){ return reg[name](el, opts || {}); }
  const hex = c => { c = String(c||'#888'); if(c[0]!=='#') c = '#888';
    if(c.length===4) c = '#'+[...c.slice(1)].map(x=>x+x).join(''); const n = parseInt(c.slice(1,7),16);
    return [(n>>16)/255, ((n>>8)&255)/255, (n&255)/255]; };
  const css = (c, a) => `rgba(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)},${a})`;
  const dark = (c, f) => c.map(v => v*(1-f));
  const light = (c, f) => c.map(v => v + (1-v)*f);

  // ---------------------------------------------------------------- jewel
  register('jewel', (canvas, opts) => {
    const ctx = canvas.getContext('2d');
    let S = Object.assign({style:'core', speed:1, glow:1, scale:0.9}, opts.settings||{}, store.get('jewel', {}));
    let items = [], cur = [], mag = 0, mix = [0.4,0.55,1], t = 0, last = performance.now();
    function set(list){ items = list || []; if(cur.length !== items.length) cur = items.map(()=>0); }
    function step(){
      const now = performance.now(), dt = Math.min(0.1, (now-last)/1000); last = now; t += dt*S.speed;
      const a = 1 - Math.exp(-dt*5); let mx = 0, acc = [0,0,0], ws = 0;
      items.forEach((it,k) => { cur[k] += (it.z - cur[k])*a; mx = Math.max(mx, Math.abs(cur[k])); const w = cur[k]*cur[k];
        let c = hex(it.color); if(cur[k] < 0) c = dark(c, 0.5); acc = acc.map((v,i)=>v + c[i]*w); ws += w; });
      mag += (Math.min(1, mx/3) - mag)*a; if(ws > 1e-4) mix = mix.map((v,i)=> v + (acc[i]/ws - v)*a);
      draw(); requestAnimationFrame(step); }
    function color(k){ return hex(items[k] && items[k].color); }
    function draw(){
      const r0 = canvas.getBoundingClientRect(), dpr = devicePixelRatio||1;
      if(canvas.width !== Math.round(r0.width*dpr)){ canvas.width = r0.width*dpr; canvas.height = r0.height*dpr; }
      ctx.setTransform(dpr,0,0,dpr,0,0); const W = r0.width, H = r0.height, cx = W/2, cy = H/2, r = Math.min(W,H)/2*S.scale;
      ctx.clearRect(0,0,W,H);
      if(S.style==='off' || !items.length){ ctx.strokeStyle='rgba(102,102,115,.5)'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(cx,cy,r*0.35,0,7); ctx.stroke(); return; }
      for(let i=0;i<6;i++){ const f = 1-i/6; ctx.fillStyle = css(mix, 0.05*S.glow*(0.4+mag)*f);
        ctx.beginPath(); ctx.arc(cx,cy, r*(0.45+0.55*(1-f))*(0.7+0.3*mag), 0, 7); ctx.fill(); }
      // a style from an add-on draws itself; one whose add-on is gone falls back to the core
      if(STYLES[S.style]) STYLES[S.style](ctx, {cx, cy, r, cur, mix, mag, t, color, css, dark, light}); else drawCore(cx,cy,r);
    }
    // A reactor core in the manner of the original Xbox's start-up: a glowing heart with energy swirling
    // inside it, and one ribbon of light per emotion on its own tilted orbit. A ribbon's length, width and
    // speed follow its |z|; positive z turns one way, negative the other. Light adds up where it overlaps.
    function drawCore(cx,cy,r){
      const n = cur.length, back = [], front = [];
      cur.forEach((z,k) => { const s = Math.min(1, Math.abs(z)/3), tilt = 0.35 + 1.1*k/Math.max(1,n), spin = k*2*Math.PI/Math.max(1,n);
        const head = t*(0.8+1.6*s)*(z>=0?1:-1) + spin*1.7, len = 0.9+2.6*s, steps = 34, col = color(k), rr = r*(0.66+0.14*k/Math.max(1,n));
        const orb = u => { const p = [Math.cos(u)*rr, Math.sin(u)*rr*Math.cos(tilt), Math.sin(u)*rr*Math.sin(tilt)];
          return [cx + p[0]*Math.cos(spin) - p[1]*Math.sin(spin), cy + p[0]*Math.sin(spin) + p[1]*Math.cos(spin), p[2]]; };
        for(let j=0;j<steps;j++){ const p0 = orb(head - len*j/steps), p1 = orb(head - len*(j+1)/steps), f = 1-j/steps;
          (p0[2] < 0 ? back : front).push([p0, p1, col, f*(0.45+0.55*s), (2.2+4.5*s)*(0.35+0.65*f)]); } });
      ctx.save(); ctx.globalCompositeOperation = 'lighter'; ctx.lineCap = 'butt';     // round caps would bead where segments overlap
      const seg = (sg, a, w, c) => { ctx.strokeStyle = css(c, a); ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(sg[0][0], sg[0][1]); ctx.lineTo(sg[1][0], sg[1][1]); ctx.stroke(); };
      const ribbons = (list, f) => { list.forEach(sg => seg(sg, sg[3]*f*0.22, sg[4]*3.2, sg[2])); list.forEach(sg => seg(sg, sg[3]*f, sg[4], light(sg[2], 0.3))); };
      ribbons(back, 0.35);
      const pulse = 1 + 0.08*Math.sin(t*(2+5*mag))*(0.3+mag), cr = r*(0.27+0.18*mag)*pulse;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, cr*1.7);
      g.addColorStop(0, css(light(mix, 0.92), 1)); g.addColorStop(0.22, css(light(mix, 0.55), 0.95));
      g.addColorStop(0.55, css(mix, 0.55*Math.min(1, 0.6+0.4*S.glow))); g.addColorStop(1, css(mix, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, cr*1.7, 0, 7); ctx.fill();
      for(let i=0;i<3;i++){ const a0 = t*(1.3+0.4*i)*(i%2?-1:1) + i*2.1;     // energy turning inside the heart
        ctx.strokeStyle = css(light(mix, 0.6), 0.25+0.35*mag); ctx.lineWidth = 1.5+2*mag;
        ctx.beginPath(); ctx.arc(cx, cy, cr*(0.45+0.17*i), a0, a0 + 1.4 + mag); ctx.stroke(); }
      ribbons(front, 1);
      ctx.restore();
    }
    canvas.title = 'Emotion jewel: click to hide or show the emotion graph, right-click for the next style';
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('contextmenu', e => { e.preventDefault(); S.style = styles[(styles.indexOf(S.style)+1)%styles.length];
      store.set('jewel', Object.assign(store.get('jewel', {}), {style: S.style})); });
    canvas.addEventListener('click', () => opts.onToggle && opts.onToggle());
    requestAnimationFrame(step);
    return { set, settings(s){ S = Object.assign(S, s||{}, store.get('jewel', {})); }, get style(){ return S.style; } };
  });

  // Rows of z values per emotion -> what the jewel shows at token i.
  function emotionsAt(E, i, palette){ if(!E || !E.z || !E.z[i]) return [];
    return E.names.map((n,k) => ({name: n, z: E.z[i][k], color: palette[k % palette.length]})); }
  return { register, create, store, emotionsAt, jewelStyle, styles: () => ['core', ...Object.keys(STYLES), 'off'] };
})();
"""
