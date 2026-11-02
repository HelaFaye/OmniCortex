#!/usr/bin/env python3
"""
OmniCortex frontend backend, and the three.js client.

This serves the API that all three frontends consume:

    GET /api/meta     frames, cells, layers, per-frame z, token labels, flagged
    GET /api/theme    the selected theme, from themes.json
    GET /api/trace    binary: i32 T, N, L | i32 layer[N] | i32 neuron[N]
                              | f32 intensity[T][N] | u8 state[T][N]

State is 0 idle, 1 active, 2 flagged, decided here in Python. The clients only
draw. That is what makes parity between pygfx, Godot and three.js maintainable:
one implementation of the thresholding, the H-Neuron mapping and the scoring,
three renderers. It also keeps the GPU free -- classification is CPU work done
once at load, not per frame on the device running the model.

The page served at / is the three.js client: UnrealBloomPass, additive
blending, depth fog.

viz/timeline.py is the analytical view -- pygfx, exact point sizes, no effects.
This is its sibling for showing people. three.js with UnrealBloomPass, additive
blending, depth fog. Same records, same themes, same flagging logic; different
job.

Keep both. Bloom is actively bad for diagnosis: it blooms neighbours together,
so a bright cell reads as a smear and you lose the spatial precision that made
the view worth having. It is excellent for a screenshot, a talk, or seeing the
shape of a burst at a glance. Use timeline.py to decide anything.

This does not need ROCm. ROCm is compute; a Vega iGPU does Vulkan graphics
fine, which is what the browser uses. With no GPU at all, Chrome falls back to
SwiftShader and it still runs, slowly.

    python viz/bloom.py runs/trace-abc
    python viz/bloom.py --demo            # synthetic trace: try the clients without a model
    python viz/bloom.py runs/trace-abc --theme cool --port 7880 --host 0.0.0.0

Serves a page; open the URL it prints. Positions are sent once and only
intensities stream per frame, so a 40-frame trace is a few hundred kilobytes
rather than a few hundred megabytes.
"""

import argparse
import json
import os
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ns_security as sec  # noqa: E402
from records import Session  # noqa: E402
from timeline import THEMES, add_flag_args, classify_frames, load_trace, resolve_mask  # noqa
import widgets  # noqa: E402

PAYLOAD = {}


DEMO_TEXT = ("The Eiffel Tower was completed in 1889 and stands in Paris . It was designed by "
             "the engineer Gustave Eiffel , and in 1923 it was moved to Lyon for the World Fair , "
             "where it remained until 1931 .").split()


def demo_trace(T=160, L=32, N=512, seed=0):
    """A synthetic trace for trying the clients without a model: a quiet field
    with a few dozen designated neurons that burst during an invented claim
    ("moved to Lyon"). Nothing here was measured; the HUD says "demo"."""
    rng = np.random.default_rng(seed)
    frames = rng.gamma(2.0, 0.02, size=(T, L, N)).astype(np.float32)
    frames *= (1 + 0.6 * np.sin(np.linspace(0, 3.1, L)))[None, :, None]   # mid layers busier
    hl = rng.integers(L // 3, L, 24)
    hn = rng.integers(0, N, 24)
    words = [DEMO_TEXT[i % len(DEMO_TEXT)] for i in range(T)]
    burst = np.zeros(T, np.float32)
    for i, w in enumerate(words):
        if w in ("moved", "to", "Lyon", "1923", "1931"):
            burst[i] = 1.0
    burst = np.convolve(burst, [0.3, 1.0, 0.6], "same")
    frames[:, hl, hn] += (0.05 + 1.2 * burst)[:, None] * rng.uniform(0.6, 1.0, 24)[None, :]
    # A stand-in classifier: positive weight on the designated neurons, so token
    # logits rise with their activity and cross 0 (p = 0.5) during the burst.
    act = frames[:, hl, hn].mean(1)
    base = np.median(act)
    scores = (act - base) / (act.max() - base + 1e-6) * 6.0 - 2.0
    col_w = np.zeros((L, N), np.float32)
    col_w[hl, hn] = rng.uniform(0.2, 1.0, 24)
    fields = {"scores": scores.tolist(), "pieces": [" " + w for w in words], "stride": 1,
              "meta": {"model": "demo (synthetic, not measured)"}, "n_frames": T, "n_layers": L,
              "h_cells": [[int(l), int(n)] for l, n in zip(hl, hn)], "col_weight": col_w.tolist()}
    return frames, fields


def column_order(fields, mask, L, N):
    """-> [L, N] int: the x position of each column within its layer. Columns
    are sorted by classifier weight (H-neurons first, strongest leftmost), so
    the H-neurons form a band on the left instead of scattering by index.
    Neuron index order carries no meaning; the layer axis is the real one."""
    w = fields.get("col_weight")
    if w is not None:
        w = np.asarray(w, dtype=np.float32)
        if w.shape != (L, N):
            w = None
    if w is None and mask is not None:
        w = mask.astype(np.float32)
    if w is None:
        return None
    order = np.argsort(-w, axis=1, kind="stable")       # per layer: strongest first
    pos = np.empty_like(order)
    rows = np.arange(L)[:, None]
    pos[rows, order] = np.arange(N)[None, :]
    return pos


def build_payload(session, h_neurons, active_pct, max_cells, demo=False, flag_prob=0.5, smooth=1,
                  relative_z=None, order="weight", subjects=None):
    """subjects: neuron_review.subject_map() for this model, to colour active
    cells by the subject they go with."""
    frames, fields = demo_trace() if demo else load_trace(session)
    T, L, N = frames.shape
    mask = resolve_mask(h_neurons, fields, (L, N))
    active, halluc, info = classify_frames(frames, fields, mask, active_pct, flag_prob, smooth, relative_z)
    z = info["z"]

    # Only cells that are ever active get geometry. At ~97% inactive, sending
    # the rest would be almost entirely wasted bandwidth and fill rate.
    ever = active.any(axis=0)
    ly, nx = np.nonzero(ever)
    if len(ly) > max_cells:
        # Keep the strongest by peak intensity rather than truncating
        # arbitrarily, so the visible structure survives the cap.
        peak = frames[:, ly, nx].max(axis=0)
        keep = np.argsort(-peak)[:max_cells]
        ly, nx = ly[keep], nx[keep]
    n = len(ly)

    vmax = float(np.percentile(frames, 99.5)) or 1.0
    inten = np.clip(frames[:, ly, nx] / vmax, 0, 1).astype(np.float32)
    # State per cell per frame: 0 idle, 1 active, 2 flagged.
    state = np.zeros((T, n), dtype=np.uint8)
    state[active[:, ly, nx]] = 1
    state[halluc[:, ly, nx]] = 2

    pos = column_order(fields, mask, L, N) if order == "weight" else None
    xs = pos[ly, nx] if pos is not None else nx
    n_h = int(mask.sum(axis=1).max()) if mask is not None else 0

    blob = bytearray()
    blob += struct.pack("<iii", T, n, L)
    blob += ly.astype("<i4").tobytes()
    blob += xs.astype("<i4").tobytes()
    blob += inten.tobytes()
    blob += state.tobytes()

    cell_subject, subj_names = None, []
    if subjects and subjects.get("subjects") and subjects.get("layers") == L:
        B = int(subjects["bins"])
        if B >= N:
            bins = nx
        else:
            edges = np.linspace(0, N, B + 1).astype(int)
            bins = np.clip(np.searchsorted(edges, nx, side="right") - 1, 0, B - 1)
        subj_names = list(subjects["subjects"])
        lookup = {}
        for k, name in enumerate(subj_names):
            for l, b, _d in subjects["cells"].get(name, []):
                lookup[(l, b)] = k
        cell_subject = [lookup.get((int(l), int(b)), -1) for l, b in zip(ly, bins)]
        if not any(c >= 0 for c in cell_subject):
            cell_subject, subj_names = None, []

    # tracks the decoding lab records beside the neurons (scripts/live_watch.py): emotions and features
    tracks = {}
    emo = fields.get("emotions") or {}
    if emo.get("z") and len(emo["z"]) == T:
        Z = np.asarray(emo["z"], dtype=np.float32)
        pick = np.argsort(-np.abs(Z).max(axis=0))[:6]
        tracks["emotions"] = {"names": [emo["names"][int(k)] for k in pick], "layer": emo.get("layer"),
                              "z": np.round(Z[:, pick], 2).tolist()}
    if fields.get("features") and len(fields["features"]) == T:
        tracks["features"] = fields["features"]
    pieces = fields.get("pieces") or []
    stride = int(fields.get("stride", 1))
    labels = ["".join(pieces[i * stride:(i + 1) * stride]) for i in range(T)] \
        if pieces else [str(i) for i in range(T)]
    return bytes(blob), {
        "frames": T, "cells": n, "layers": L, "neurons": N,
        "z": [round(float(v), 3) for v in z], "labels": labels,
        # Tokens whose risk crossed the threshold (whether or not an H-neuron
        # cell is drawn for them), and the per-token risk itself.
        "flagged": [int(i) for i in np.where(info["flagged_tokens"])[0]],
        "prob": None if info["prob"] is None else [round(float(v), 4) for v in info["prob"]],
        "threshold": info["threshold"], "mode": info["mode"], "relative_z": info["relative_z"],
        "cells_note": info["cells"], "order": "weight" if pos is not None else "index",
        "h_band": n_h,
        "model": fields.get("meta", {}).get("model"),
        "verdict": fields.get("verdict"),
        "question": (fields.get("question") or "")[:160],
        "subjects": subj_names, "cell_subject": cell_subject, **tracks, "source": fields.get("source"),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/meta":
            return self._send(200, json.dumps(PAYLOAD["meta"]),
                              "application/json")
        if self.path == "/api/trace":
            return self._send(200, PAYLOAD["blob"], "application/octet-stream")
        if self.path == "/api/theme":
            return self._send(200, json.dumps(widgets.with_widgets(PAYLOAD["theme"])),
                              "application/json")
        self._send(404, json.dumps({"error": "not found"}), "application/json")


PAGE = r"""<!DOCTYPE html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Bloom</title>
<style>
html,body{margin:0;height:100%;background:#05050a;color:#c9c9d2;
font:13px ui-sans-serif,system-ui,sans-serif;overflow:hidden}
#c{position:absolute;inset:0}
#hud{position:absolute;left:14px;top:12px;z-index:2;line-height:1.6;text-shadow:0 1px 3px #000;max-width:min(560px,90vw)}
#hud b{font-weight:500;color:#fff}
#cambar{position:absolute;right:12px;top:12px;z-index:3;display:flex;gap:4px;flex-wrap:wrap;justify-content:flex-end;max-width:60vw}
#cambar button{font:12px system-ui;padding:3px 8px;border:1px solid #3a3a48;border-radius:6px;background:rgba(16,16,24,.8);color:#d6d6e0;cursor:pointer}
#cambar button.on{border-color:#9d7be8;color:#fff}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle}
.ring{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle;box-sizing:border-box;border:2px solid}
.note{color:#8a8a92;font-size:12px}.bad{color:#ff9b8a}
#panel{position:absolute;left:0;right:0;bottom:0;z-index:2;padding:8px 14px 10px;
background:linear-gradient(transparent,#05050af2 22%)}
#row{display:flex;gap:12px;align-items:center}
#spark{flex:1;height:46px;cursor:pointer;display:block;min-width:0}
#jewel{width:64px;height:64px;flex:none;margin:-10px 0}
#spark.off{visibility:hidden}
#tok{font-family:ui-monospace,monospace;min-width:150px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#text{margin-top:6px;max-height:22vh;overflow-y:auto;line-height:1.75;font-size:13.5px;white-space:pre-wrap;word-break:break-word}
#text span{cursor:pointer;border-radius:3px;padding:1px 0}
#text span.cur{outline:1.5px solid #fff;outline-offset:1px}
button{background:transparent;border:1px solid #3a3a46;color:#c9c9d2;border-radius:6px;padding:5px 12px;cursor:pointer;font:inherit}
button:hover{background:#16161e}
.warn{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:2rem;z-index:3}
</style>
<canvas id="c"></canvas>
<div id="hud"></div>
<div id="cambar" title="camera: drag to orbit, right-drag to pan, wheel to zoom">
<button data-v="front" title="1">Front</button><button data-v="side" title="2">Side</button><button data-v="top" title="3">Top</button>
<button data-v="reset" title="R">Reset</button><button id="orbit" title="O: slow automatic orbit">Orbit</button>
<button id="follow" title="F: glide to the flagged neurons on flagged tokens">Follow flags</button></div>
<div id="panel"><div id="row"><button id="play">Pause</button><canvas id="spark"></canvas><canvas id="jewel" hidden></canvas><div id="tok"></div></div>
<div id="ef" style="margin-top:4px;font-size:12.5px;color:#8a8a92;min-height:1em"></div>
<div id="text"></div></div>
<script>/*NS_WIDGETS*/</script>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script>
// The renderer comes from unpkg. If it cannot load (offline, a blocking proxy),
// the module below never runs; say so instead of leaving a black page.
setTimeout(()=>{ if(!window.__nsReady){ const w=document.createElement('div'); w.className='warn';
  w.innerHTML='Could not load three.js from unpkg.com.<br>This viewer needs that CDN, or use the Godot '+
    'client (plugins/godot-client/client) or viz/timeline.py, which work offline.'; document.body.appendChild(w); } }, 10000);
</script>
<script type="module">
import * as THREE from 'three';
window.__nsReady = true;
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';

// Relative URLs: the same page is served at / by bloom.py and at /viz/<id>/ by Studio.
const meta = await (await fetch('api/meta')).json();
const theme = await (await fetch('api/theme')).json();
const buf = await (await fetch('api/trace')).arrayBuffer();

const dv = new DataView(buf);
const T = dv.getInt32(0,true), N = dv.getInt32(4,true), L = dv.getInt32(8,true);
let o = 12;
const ly = new Int32Array(buf, o, N); o += N*4;
const nx = new Int32Array(buf, o, N); o += N*4;
const inten = new Float32Array(buf, o, T*N); o += T*N*4;
const state = new Uint8Array(buf, o, T*N);
const prob = meta.prob, thr = meta.threshold ?? 0.5;
const flaggedSet = new Set(meta.flagged);

// Sizes and rate shared with the Godot client (plugins/godot-client/client/main.gd); a test keeps them equal.
const SIZE_IDLE = 1.1, SIZE_ACTIVE = 2.4, SIZE_FLAG = 7.0, FPS = 8;

const hex = s => new THREE.Color(s);
const cIdle = hex('#2a2a3a'), cAct = hex(theme.active), cHal = hex(theme.halluc);
// Subject colours (from the neuron review's subject map): active cells that go with a subject
// take that subject's colour; flagged cells keep the hallucination colour.
const SUBJ = meta.subjects||[], cellSubj = meta.cell_subject;
const subjPal = (theme.subjects||['#E7B65C','#5FCF8F','#B48EAD','#5DCAA5','#F28FAD','#9AD0F5','#C3E88D','#FFCB6B']);
const cSubj = SUBJ.map((_,k)=>hex(subjPal[k%subjPal.length]));
let subjOn = !!(cellSubj && SUBJ.length);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05050a, 0.0016);
const cam = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 6000);
// Fit the view to this trace: neuron columns span W units whatever the trace's
// width (512 bins or 14336 neurons), layers are 8 units apart.
const W = 640, XS = W/Math.max(1, meta.neurons), H = L*8;
const D = 0.95*Math.max(W, H);
cam.position.set(W/2 + 0.3*D, H/2 + 0.25*D, 0.95*D);
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));
renderer.setSize(innerWidth, innerHeight);
const controls = new OrbitControls(cam, canvas);
controls.enableDamping = true;
controls.target.set(W/2, H/2 - 40, 0);

function pointsMaterial(ring, onTop){
  return new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, depthTest:!onTop,
    blending: onTop ? THREE.NormalBlending : THREE.AdditiveBlending,
    // size 0 means hidden: some GL implementations clamp point size to 1 pixel,
    // so a hidden point is moved outside the clip volume instead.
    vertexShader:`attribute float size; varying vec3 vC;
      void main(){ vC=color; vec4 mv=modelViewMatrix*vec4(position,1.0);
      gl_PointSize=size*(300.0/-mv.z); gl_Position = size > 0.0 ? projectionMatrix*mv : vec4(2.0,2.0,2.0,1.0); }`,
    fragmentShader: ring
      // Flagged H-neurons: a solid core inside a ring, so they read as a different
      // kind of mark, not just a brighter dot, whatever the colour vision.
      ? `varying vec3 vC; void main(){ float r=length(gl_PointCoord-vec2(0.5)); if(r>0.5) discard;
         float a = r<0.22 ? 1.0 : (r>0.36 ? smoothstep(0.5,0.42,r) : 0.15); gl_FragColor=vec4(vC, a); }`
      : `varying vec3 vC; void main(){ float r=length(gl_PointCoord-vec2(0.5)); if(r>0.5) discard;
         gl_FragColor=vec4(vC, smoothstep(0.5,0.0,r)); }`,
    vertexColors:true});
}
function cloud(ring, onTop){
  const pos = new Float32Array(N*3), col = new Float32Array(N*3), siz = new Float32Array(N);
  for(let i=0;i<N;i++){ pos[i*3]=nx[i]*XS; pos[i*3+1]=ly[i]*8; pos[i*3+2]=0; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  g.setAttribute('color', new THREE.BufferAttribute(col,3));
  g.setAttribute('size', new THREE.BufferAttribute(siz,1));
  const pts = new THREE.Points(g, pointsMaterial(ring, onTop)); scene.add(pts); return pts;
}
const field = cloud(false, false);
// Trail: earlier frames persist as a dim wake so a burst reads as motion.
const trailN = 6, trails = [];
for(let k=0;k<trailN;k++){ const p = cloud(false, false); p.material.opacity = 0.5*(1-k/trailN); trails.push(p); }
const flags = cloud(true, true);     // drawn last, over everything
flags.renderOrder = 10;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, cam));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight), 1.0, 0.55, 0.12);
composer.addPass(bloom);

const tmp = new THREE.Color();
// The current token sits at z=0; trail k (an earlier token) sits k+1 steps behind it.
function paint(pts, t, dim, dz){
  const g = pts.geometry, c = g.getAttribute('color'), s = g.getAttribute('size'), p = g.getAttribute('position');
  const base = t*N;
  for(let i=0;i<N;i++){
    const v = inten[base+i], st = state[base+i];
    tmp.copy(st===0?cIdle:(subjOn && cellSubj[i]>=0 ? cSubj[cellSubj[i]] : cAct));   // flagged: the overlay
    const gn = (st===0?0.35:0.6+0.8*v)*dim;
    c.array[i*3]=tmp.r*gn; c.array[i*3+1]=tmp.g*gn; c.array[i*3+2]=tmp.b*gn;
    s.array[i] = (st===0?SIZE_IDLE:SIZE_ACTIVE)*(0.6+0.9*v);
    p.array[i*3+2] = dz;
  }
  c.needsUpdate=true; s.needsUpdate=true; p.needsUpdate=true;
}
function paintFlags(t){
  const g = flags.geometry, c = g.getAttribute('color'), s = g.getAttribute('size');
  const base = t*N;
  for(let i=0;i<N;i++){
    const on = state[base+i]===2;
    // Full colour, never pushed past 1: the hue survives instead of washing to white.
    c.array[i*3]=cHal.r; c.array[i*3+1]=cHal.g; c.array[i*3+2]=cHal.b;
    s.array[i] = on ? SIZE_FLAG*(0.8+0.5*inten[base+i]) : 0;
  }
  c.needsUpdate=true; s.needsUpdate=true;
}

let t=0, playing=true, acc=0;
const hud=document.getElementById('hud'), tok=document.getElementById('tok');
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const rgba=(c,a)=>`rgba(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)},${a})`;
document.getElementById('play').onclick=e=>{playing=!playing; e.target.textContent=playing?'Pause':'Play';};

// ---- the reply, token by token, shaded by risk
const textEl = document.getElementById('text');
textEl.innerHTML = meta.labels.map((s,i)=>{
  const p = prob ? prob[i] : 0, a = prob ? Math.max(0,(p-0.15)/0.85)*0.55 : 0;
  const style = `background:${rgba(cHal,a.toFixed(3))};` + (flaggedSet.has(i)?`text-decoration:underline 2px ${theme.halluc};text-underline-offset:3px;`:'');
  return `<span data-i="${i}" style="${style}" title="token ${i+1}${prob?` · risk ${(p*100).toFixed(0)}%`:''}">${esc(s)||'·'}</span>`;
}).join('');
textEl.onclick = e => { const i = e.target.dataset?.i; if(i!==undefined){ t=+i; update(); } };

// ---- risk over the reply: line, threshold, flagged stretches, playhead
const spark = document.getElementById('spark'), sctx = spark.getContext('2d');
function drawSpark(){
  const r = spark.getBoundingClientRect(), dpr = devicePixelRatio||1;
  spark.width = r.width*dpr; spark.height = r.height*dpr; sctx.setTransform(dpr,0,0,dpr,0,0);
  const w = r.width, h = r.height, x = i => (T>1 ? i/(T-1) : 0)*w, y = p => h-2-(h-4)*p;
  sctx.clearRect(0,0,w,h); sctx.fillStyle='#16161e'; sctx.fillRect(0,0,w,h);
  if(!prob && meta.emotions){            // a decoding-lab recording: emotion lanes instead of risk
    const E = meta.emotions, pal = EPAL;
    const mx = Math.max(3, ...E.z.flat().map(Math.abs)), ye = v => h/2 - (h/2-3)*v/mx;
    sctx.strokeStyle='#2a2a34'; sctx.beginPath(); sctx.moveTo(0,h/2); sctx.lineTo(w,h/2); sctx.stroke();
    E.names.forEach((n,k)=>{ sctx.strokeStyle=pal[k%6]; sctx.lineWidth=1.4; sctx.beginPath();
      E.z.forEach((row,i)=> i?sctx.lineTo(x(i),ye(row[k])):sctx.moveTo(x(i),ye(row[k]))); sctx.stroke(); });
  } else if(!prob){ sctx.fillStyle='#8a8a92'; sctx.font='12px ui-sans-serif,system-ui';
    sctx.fillText('no classifier scores in this trace: nothing can be flagged', 8, h/2+4); }
  else {
    sctx.fillStyle=rgba(cHal,0.18);
    for(const i of meta.flagged) sctx.fillRect(x(i)-Math.max(1,w/T/2), 0, Math.max(2,w/T), h);
    sctx.strokeStyle='#8a8a92'; sctx.setLineDash([4,4]); sctx.beginPath(); sctx.moveTo(0,y(thr)); sctx.lineTo(w,y(thr)); sctx.stroke(); sctx.setLineDash([]);
    sctx.strokeStyle=theme.halluc; sctx.lineWidth=1.5; sctx.beginPath();
    prob.forEach((p,i)=> i?sctx.lineTo(x(i),y(p)):sctx.moveTo(x(i),y(p))); sctx.stroke();
  }
  sctx.fillStyle='#fff'; sctx.fillRect(x(t)-1,0,2,h);
}
spark.onclick = e => { const r = spark.getBoundingClientRect(); t = Math.round((e.clientX-r.left)/r.width*(T-1)); update(); };

const modeNote = meta.mode==='review' ? `${esc(meta.question||'')} · cells: ${esc(meta.cells_note||'')}<br>strip: ${esc(meta.strip||'')}`
  : meta.mode==='absolute' ? `tokens flagged at risk ≥ ${Math.round(thr*100)}% (the classifier's own threshold)`
  : meta.mode==='relative' ? `<span class="bad">relative mode: tokens ${meta.relative_z} SD above this reply's mean, so some are always flagged</span>`
  : meta.source==='decode_lab' ? `decoding-lab recording: CETT${meta.emotions?', emotions':''}${meta.features?', features':''} for each generated token (no classifier, so nothing is flagged)`
  : `<span class="bad">no classifier scores: nothing can be flagged</span>`;
const cellsNote = meta.mode!=='review' && meta.cells_note && meta.cells_note.startsWith('none') ? `<br><span class="note">${esc(meta.cells_note)}: risky tokens are shown, H-neuron cells are not</span>` : '';
const orderNote = meta.order==='weight' ? `x: neurons ordered by classifier weight, H-neurons at the left · y: layer` : `x: neuron index · y: layer`;

// the emotion jewel (viz/widgets.py) at the right of the strip; click it to hide or show the lanes
const EPAL = ['#FF4D1A','#3D8BFF','#1D9E75','#E7B65C','#B07CFF','#c9c9d2'];
let jewel = null;
if(meta.emotions){ const jc = document.getElementById('jewel'); jc.hidden = false;
  if(NSWidgets.store.get('graph_hidden', false)) spark.classList.add('off');
  jewel = NSWidgets.create('jewel', jc, {settings: (theme.widgets||{}).jewel, onToggle(){
    spark.classList.toggle('off'); NSWidgets.store.set('graph_hidden', spark.classList.contains('off')); }}); }

let lastCur = null;
function update(){
  if(jewel) jewel.set(NSWidgets.emotionsAt(meta.emotions, t, EPAL));
  paint(field, t, 1.0, 0);
  for(let k=0;k<trailN;k++) paint(trails[k], Math.max(0,t-(k+1)), 0.45, -(k+1)*6);
  paintFlags(t);
  const flagged = flaggedSet.has(t);
  // Review views (scripts/neuron_review.py) bring their own legend: frames are time buckets, not tokens.
  const lg = meta.legend || {active:'active', halluc:'H-neuron on a flagged token'};
  hud.innerHTML = `<b>${esc(meta.model||'trace')}</b> <span class="note">${meta.frames} ${meta.mode==='review'?'time buckets':'tokens'} · ${meta.layers} layers</span><br>`+
    `<span class="dot" style="background:${theme.active};margin-left:0"></span>${esc(lg.active)}`+
    (lg.halluc?`<span class="ring" style="border-color:${theme.halluc}"></span>${esc(lg.halluc)}`:'')+'<br>'+
    `<span class="note">${modeNote}</span>${cellsNote}<br><span class="note">${orderNote}</span>`+
    (SUBJ.length?`<br><label class="note" style="cursor:pointer"><input type="checkbox" id="subjT"${subjOn?' checked':''}> colour by subject</label> `+
      SUBJ.map((n,k)=>`<span class="dot" style="background:${subjPal[k%subjPal.length]}"></span>${esc(n)}`).join(' '):'');
  const sT=document.getElementById('subjT'); if(sT) sT.onchange=()=>{ subjOn=sT.checked; update(); };
  tok.innerHTML = `${t+1}/${T} <span style="color:${flagged?theme.halluc:'#c9c9d2'}">${esc(meta.labels[t]||'·')}</span>`+
    (prob?` <span class="note">${meta.mode==='review'?'error rate':'risk'} ${(prob[t]*100).toFixed(0)}%</span>`:'')+(flagged?` <b style="color:${theme.halluc}">flagged</b>`:'');
  if(lastCur) lastCur.classList.remove('cur');
  lastCur = textEl.children[t]; if(lastCur){ lastCur.classList.add('cur');
    const r = lastCur.offsetTop - textEl.offsetTop; if(r < textEl.scrollTop || r > textEl.scrollTop + textEl.clientHeight - 24) textEl.scrollTop = r - 20; }
  const ef = document.getElementById('ef'); let efh = '';
  if(meta.emotions){ const E = meta.emotions, pal = ['#FF4D1A','#3D8BFF','#1D9E75','#E7B65C','#B07CFF','#c9c9d2'];
    efh += 'emotions: ' + E.names.map((n,k)=>[n,E.z[t][k],pal[k%6]]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,4)
      .map(([n,z,c])=>`<span style="color:${c}">${esc(n)} ${z>0?'+':''}${z.toFixed(1)}</span>`).join(' · '); }
  if(meta.features && meta.features[t] && meta.features[t].length){
    efh += (efh?'<br>':'') + 'features: ' + meta.features[t].slice(0,6).map(([d,id,r,l])=>`<span title="${esc(d)} #${id}" style="color:#c9c9d2">${esc(l||('#'+id))}</span> ${(+r).toFixed(2)}`).join(' · '); }
  ef.innerHTML = efh;
  bloom.strength = flagged ? 1.5 : 1.0;
  drawSpark();
}
update();

// ---- camera: presets (manual) and orbit / follow-flags (automatic), with smooth moves
const HOME = {pos: cam.position.clone(), tgt: controls.target.clone()};
let glide = null, follow = false;
function goTo(pos, tgt){ glide = {pos: pos.clone(), tgt: tgt.clone()}; }
function preset(v){ const c = new THREE.Vector3(W/2, H/2, 0);
  if(v==='front') goTo(new THREE.Vector3(W/2, H/2, 1.15*D), c);
  else if(v==='side') goTo(new THREE.Vector3(W/2 + 1.15*D, H/2, 0), c);
  else if(v==='top') goTo(new THREE.Vector3(W/2, H/2 + 1.25*D, 1), c);
  else goTo(HOME.pos, HOME.tgt); }
document.querySelectorAll('#cambar [data-v]').forEach(b=>b.onclick=()=>preset(b.dataset.v));
const orbitB = document.getElementById('orbit'), followB = document.getElementById('follow');
orbitB.onclick=()=>{ controls.autoRotate=!controls.autoRotate; controls.autoRotateSpeed=0.6; orbitB.classList.toggle('on', controls.autoRotate); };
followB.onclick=()=>{ follow=!follow; followB.classList.toggle('on', follow); if(!follow) preset('reset'); };
let followed = -1;
function followFlags(){
  if(!follow || followed===t) return; followed = t;
  const base = t*N; let x=0, y=0, k=0;
  for(let i=0;i<N;i++) if(state[base+i]===2){ x+=nx[i]*XS; y+=ly[i]*8; k++; }
  if(k){ const c = new THREE.Vector3(x/k, y/k, 0); goTo(c.clone().add(new THREE.Vector3(0.15*D, 0.1*D, 0.45*D)), c); }
  else if(!flaggedSet.has(t)) goTo(HOME.pos, HOME.tgt);
}
addEventListener('keydown', e=>{ if(e.target.tagName==='INPUT') return;
  const k=e.key.toLowerCase(); if(k==='1') preset('front'); else if(k==='2') preset('side'); else if(k==='3') preset('top');
  else if(k==='r') preset('reset'); else if(k==='o') orbitB.click(); else if(k==='f') followB.click(); });
controls.addEventListener('start', ()=>{ glide=null; });          // a hand on the camera wins over any glide

renderer.setAnimationLoop(()=>{
  if(playing){ acc++; if(acc % Math.round(60/FPS) === 0){ t=(t+1)%T; update(); } }
  followFlags();
  if(glide){ cam.position.lerp(glide.pos, 0.08); controls.target.lerp(glide.tgt, 0.08);
    if(cam.position.distanceTo(glide.pos) < 0.5 && controls.target.distanceTo(glide.tgt) < 0.5) glide=null; }
  controls.update(); composer.render();
});
addEventListener('resize', ()=>{
  cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight); composer.setSize(innerWidth,innerHeight); drawSpark();
});
</script>"""
PAGE = PAGE.replace("/*NS_WIDGETS*/", widgets.WIDGETS_JS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", nargs="?", help="trace session from scripts/trace_sample.py")
    p.add_argument("--demo", action="store_true",
                   help="a synthetic trace, to try the three.js and Godot clients without a model")
    p.add_argument("--theme", default="dark", choices=sorted(THEMES))
    p.add_argument("--h-neurons")
    p.add_argument("--active-pct", type=float, default=97.0)
    add_flag_args(p)
    p.add_argument("--order", choices=["weight", "index"], default="weight",
                   help="column order within a layer: by classifier weight (H-neurons as a band) or by index")
    p.add_argument("--max-cells", type=int, default=120000,
                   help="cap on instantiated points; the strongest are kept")
    p.add_argument("--port", type=int, default=7880)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--dump", metavar="BIN", help="write the payload and exit")
    p.add_argument("--allow-unauthenticated", action="store_true", help="permit a non-loopback bind (no auth)")
    a = p.parse_args()
    if not a.session and not a.demo:
        p.error("give a trace session, or --demo")
    if not a.dump:
        sec.loopback_only(a.host, "the trace viewer", a.allow_unauthenticated)

    blob, meta = build_payload(a.session, a.h_neurons, a.active_pct, a.max_cells, demo=a.demo,
                               flag_prob=a.flag_prob, smooth=a.smooth, relative_z=a.relative_z, order=a.order)
    PAYLOAD.update({"blob": blob, "meta": meta, "theme": THEMES[a.theme]})
    print(f"{meta['frames']} frames, {meta['cells']} cells, "
          f"{len(blob) / 1e6:.1f} MB payload, theme '{a.theme}'")
    if meta["mode"] == "unscored":
        print("no classifier scores in this trace: nothing can be flagged "
              "(trace_sample.py --classifier records them)")
    else:
        print(f"{len(meta['flagged'])} tokens flagged ({meta['mode']}"
              + (f", p >= {meta['threshold']}" if meta["mode"] == "absolute" else "")
              + f"); cells: {meta['cells_note']}")

    if a.dump:
        with open(a.dump, "wb") as f:
            f.write(blob)
        with open(a.dump + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {a.dump} and {a.dump}.json")
        return

    print(f"\nopen http://{a.host}:{a.port}  (three.js client)")
    print(f"Godot client: set NS_API=http://{a.host}:{a.port} and run "
          "plugins/godot-client/client/")
    print("This is the presentation view. Bloom blurs neighbours together, so "
          "use\nviz/timeline.py to decide anything.")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
