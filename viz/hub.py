#!/usr/bin/env python3
"""
One process that owns the others.

Before this, every mode was its own terminal: studio on 7870, stream on 7890,
control on 7861. The tab bar implied they were one application and they were
not. This supervises them -- open a tab and its service starts, close the tab
and it stops.

    python viz/hub.py --root . --port 7860
    python viz/hub.py --root . --host 0.0.0.0 --token-file hub.token --tls-cert c.pem --tls-key k.pem

Services are declared in SERVICES, not discovered, because a supervisor that
starts arbitrary commands from a config file is a remote shell. Each entry says
how to build its argv and how to know it is ready.

Stopping is reference-counted: a service stays up while any tab holds it, and
shuts down `idle_after` seconds after the last one goes. Without the delay a
page refresh would kill the model you just loaded.
"""

import argparse
import hmac
import http.cookies
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import oc_security as sec  # noqa: E402
try:
    import shell            # the mode registry
except ImportError:
    shell = None

STATE = {"root": ".", "token": None, "idle_after": 90}
LOCK = threading.RLock()
SERVICES = {}
RUNNING = {}        # name -> {proc, port, started, holders:set, log:[], last_release}


def _free_port(preferred):
    """Prefer the documented port; fall back if something already has it."""
    for p in (preferred, 0):
        try:
            s = socket.socket()
            s.bind(("127.0.0.1", p))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError:
            continue
    return preferred


def declare(root, py):
    """Service table. Fixed argv builders -- nothing here is user-supplied."""
    return {
        "chat": {
            "mode": "chat",
            "port": 7870,
            "script": "viz/studio.py",
            "args": lambda port, cfg: [
                py, "-u", os.path.join(root, "viz/studio.py"),
                "--port", str(port),
                *(["--models-dir", cfg["models_dir"]] if cfg.get("models_dir") else []),
                *(["--server", cfg["server_bin"]] if cfg.get("server_bin") else []),
            ],
            "health": "/api/status",
            "needs": ["server_bin"],
            "desc": "Model manager and chat, over llama-server.",
        },
        "live": {
            "mode": "live",
            "port": 7890,
            "script": "viz/stream.py",
            "args": lambda port, cfg: [
                py, "-u", os.path.join(root, "viz/stream.py"), "--port", str(port),
                # a llama-server patched by llama-tools/server-activations, else a simulation
                *(["--source", cfg["live_source"]] if cfg.get("live_source") else ["--simulate"]),
            ],
            "health": "/api/tiers",
            "needs": [],
            "desc": "Live activation stream, from a patched llama-server (or simulated).",
        },
        "replay": {
            "mode": "replay",
            "port": 7880,
            "script": "viz/bloom.py",
            "args": lambda port, cfg: [
                py, "-u", os.path.join(root, "viz/bloom.py"),
                cfg.get("trace", ""), "--port", str(port),
            ],
            "health": "/api/meta",
            "needs": ["trace"],
            "desc": "Token-resolved 3D trace, scrubbable.",
        },
        # Labs and dashboards: fixed argv, loopback only, no user-supplied values.
        **{name: {"mode": name, "port": port, "script": script, "health": health, "needs": [], "desc": desc,
                  "args": (lambda script, extra: lambda port, cfg: [
                      py, "-u", os.path.join(root, script), "--port", str(port), *extra])(script, extra)}
           for name, port, script, health, extra, desc in (
               ("pipeline", 7861, "viz/control.py", "/api/modes", ["--root", root],
                "Pipeline dashboard: what each stage has produced, stage launching, live logs."),
               ("quant_lab", 8796, "viz/quant_lab.py", "/", [],
                "Capability-preserving quantization: compare quants of one model, keep the H-Neuron edit."),
               ("scale_sweep", 8797, "viz/scale_sweep_lab.py", "/", [],
                "Sweep the suppression scale and see accuracy, abstention and damage at each step."),
               ("adaptive_tuning", 8800, "viz/adaptive_tuning_lab.py", "/", [],
                "Tune the scale on a remote worker, adaptively."),
               ("transfer", 8798, "viz/transfer_lab.py", "/health", [],
                "Send models between machines: encrypted browser-to-browser transfer."))},
    }


TRACE_DIRS = ["~/.omnicortex/traces", "runs", "runs/traces"]     # runs/traces: the decoding lab's --record


def find_traces(limit=200, extra=()):
    """Trace sessions for the Replay picker: Studio's per-reply checks,
    scripts/trace_sample.py output and decoding-lab recordings, newest first.
    extra: more folders, e.g. the running Studio's traces folder."""
    out, seen = [], set()
    for base in [*TRACE_DIRS, *(x for x in extra if x)]:
        base = os.path.realpath(os.path.join(STATE["root"], os.path.expanduser(base)))
        if not os.path.isdir(base) or base in seen:
            continue
        seen.add(base)
        for name in os.listdir(base):
            d = os.path.join(base, name)
            try:
                with open(os.path.join(d, "manifest.json")) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            if meta.get("kind") != "trace":
                continue
            rel = os.path.relpath(d, STATE["root"])
            out.append({"path": d if rel.startswith("..") else rel, "model": meta.get("model"),
                        "created": meta.get("created") or "", "source": meta.get("source", "trace_sample")})
    out.sort(key=lambda x: x["created"], reverse=True)
    return out[:limit]


def _pump(name, proc):
    rec = RUNNING[name]
    for line in proc.stdout:
        line = line.rstrip("\n")
        with LOCK:
            rec["log"].append(line)
            del rec["log"][:-300]
            for q in rec["subs"]:
                try:
                    q.put_nowait(line)
                except queue.Full:
                    pass
    rec["rc"] = proc.wait()


def _healthy(port, path, proc, timeout=45):
    end = time.time() + timeout
    url = f"http://127.0.0.1:{port}{path}"
    while time.time() < end:
        if proc.poll() is not None:
            return False, f"exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status in (200, 401):     # 401 = up, just gated
                    return True, "ready"
        except Exception:
            pass
        time.sleep(0.4)
    return False, f"no response on {path} after {timeout}s"


def acquire(name, holder, cfg=None):
    """Start the service if needed and register a holder."""
    with LOCK:
        svc = SERVICES.get(name)
        if not svc:
            return False, f"unknown service {name!r}", None
        rec = RUNNING.get(name)
        if rec and rec["proc"].poll() is None:
            rec["holders"].add(holder)
            rec["last_release"] = None
            return True, "already running", rec["port"]

        script = os.path.join(STATE["root"], svc["script"])
        if not os.path.exists(script):
            return False, f"{svc['script']} not found", None
        cfg = {k: str(v).strip() for k, v in (cfg or {}).items()
               if k in ("server_bin", "models_dir", "trace", "live_source")}
        for k, v in cfg.items():
            # These become argv values: one starting with "-" would be read as a flag.
            if v.startswith("-") or any(ord(c) < 32 for c in v):
                return False, f"{k}: invalid value", None
        if cfg.get("live_source") and not re.match(r"^https?://[^\s]+$", cfg["live_source"]):
            return False, "live source must be an http(s) URL of a patched llama-server", None
        missing = [n for n in svc["needs"] if not cfg.get(n)]
        if missing:
            return False, f"needs {', '.join(missing)}", None

        port = _free_port(svc["port"])
        cmd = svc["args"](port, cfg)
        proc = subprocess.Popen(
            cmd, cwd=STATE["root"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            # Its own group, so stopping takes the children with it. A chat
            # service owns a llama-server holding several GB.
            start_new_session=True)
        RUNNING[name] = {"proc": proc, "port": port, "started": time.time(),
                         "holders": {holder}, "log": [], "subs": [],
                         "rc": None, "last_release": None, "cmd": cmd}
        threading.Thread(target=_pump, args=(name, proc), daemon=True).start()

    ok, why = _healthy(port, svc["health"], proc)
    if not ok:
        with LOCK:
            tail = "\n".join(RUNNING.get(name, {}).get("log", [])[-8:])
            stop(name, force=True)
        return False, f"{why}\n{tail}", None
    return True, "started", port


def release(name, holder):
    with LOCK:
        rec = RUNNING.get(name)
        if not rec:
            return
        rec["holders"].discard(holder)
        if not rec["holders"]:
            # Grace period: a refresh releases and re-acquires within a second,
            # and reloading a 7GB model for that would be absurd.
            rec["last_release"] = time.time()


def stop(name, force=False):
    with LOCK:
        rec = RUNNING.pop(name, None)
    if not rec:
        return False
    p = rec["proc"]
    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except OSError:
            p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                p.kill()
    return True


def reaper():
    while True:
        time.sleep(5)
        now = time.time()
        with LOCK:
            due = [n for n, r in RUNNING.items()
                   if not r["holders"] and r["last_release"]
                   and now - r["last_release"] > STATE["idle_after"]]
        for n in due:
            print(f"[hub] stopping {n}: idle {STATE['idle_after']}s")
            stop(n)


def service_state():
    with LOCK:
        out = {}
        for name, svc in SERVICES.items():
            rec = RUNNING.get(name)
            up = bool(rec and rec["proc"].poll() is None)
            out[name] = {
                "mode": svc["mode"], "desc": svc["desc"],
                "needs": svc["needs"],
                "available": os.path.exists(
                    os.path.join(STATE["root"], svc["script"])),
                "running": up,
                "port": rec["port"] if up else None,
                "holders": len(rec["holders"]) if rec else 0,
                "secs": int(time.time() - rec["started"]) if rec else 0,
                "rc": rec.get("rc") if rec else None,
                "tail": rec["log"][-3:] if rec else [],
            }
        return out


# ---------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _ok(self):
        tok = STATE["token"]
        if not tok:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Bearer ") and hmac.compare_digest(h[7:], tok):
            return True
        raw = self.headers.get("Cookie", "")
        if raw:
            try:
                c = http.cookies.SimpleCookie(raw)
                if "oc" in c and hmac.compare_digest(c["oc"].value, tok):
                    return True
            except Exception:
                pass
        if "token=" in self.path:
            q = self.path.split("token=", 1)[1].split("&")[0]
            if hmac.compare_digest(q, tok):
                return True
        return False

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._ok():
            return self._send(401, LOGIN, "text/html; charset=utf-8")
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, page(), "text/html; charset=utf-8")
        if path == "/api/services":
            return self._send(200, json.dumps(service_state()))
        if path == "/api/theme":
            return self._send(200, json.dumps(ui_theme()))
        if path == "/api/traces":
            return self._send(200, json.dumps({"traces": find_traces()}))
        if path == "/api/modes":
            return self._send(200, shell.as_json() if shell
                              else json.dumps({"modes": []}))
        if path == "/api/log":
            name = self.path.split("name=", 1)[-1].split("&")[0]
            return self._stream(name)
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/login":
            if STATE["token"] and hmac.compare_digest(str(req.get("token", "")),
                                                      STATE["token"]):
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie", f"oc={STATE['token']}; Path=/; "
                                               "HttpOnly; SameSite=Strict")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(401, json.dumps({"error": "bad token"}))
            return
        if not self._ok():
            return self._send(401, json.dumps({"error": "unauthorized"}))
        if self.path == "/api/acquire":
            ok, msg, port = acquire(req.get("name"), req.get("holder", "?"),
                                    req.get("config"))
            return self._send(200 if ok else 500,
                              json.dumps({"ok": ok, "message": msg,
                                          "port": port}))
        if self.path == "/api/release":
            release(req.get("name"), req.get("holder", "?"))
            return self._send(200, json.dumps({"ok": True}))
        if self.path == "/api/stop":
            return self._send(200, json.dumps({"ok": stop(req.get("name"))}))
        self._send(404, json.dumps({"error": "not found"}))

    def _stream(self, name):
        with LOCK:
            rec = RUNNING.get(name)
        if not rec:
            return self._send(404, json.dumps({"error": "not running"}))
        q = queue.Queue(maxsize=200)
        with LOCK:
            rec["subs"].append(q)
            backlog = list(rec["log"][-100:])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for line in backlog:
                self.wfile.write(f"data: {json.dumps(line)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    line = q.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(line)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with LOCK:
                if q in rec["subs"]:
                    rec["subs"].remove(q)


def ui_theme():
    try:
        with open(os.path.join(HERE, "themes.json")) as f:
            t = json.load(f).get("_ui", {})
        return t.get("omnicortex-dark", {})
    except Exception:
        return {}


def css_vars():
    t = ui_theme()
    keep = ("bg", "panel", "panel_2", "line", "line_strong", "fg", "fg_dim",
            "fg_faint", "accent", "accent_dim", "teal", "teal_dim", "emerald",
            "emerald_dim", "danger", "danger_fg", "warn", "warn_fg")
    return "\n".join(f"  --{k.replace('_','-')}: {t[k]};"
                     for k in keep if k in t)


LOGIN = """<!DOCTYPE html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex</title><style>body{font:16px system-ui;background:#07070b;
color:#d6d6e0;display:flex;align-items:center;justify-content:center;
height:100vh;margin:0}input,button{width:280px;padding:.8rem;font-size:16px;
border-radius:8px;border:1px solid #22222e;background:#101018;color:#fff}
button{margin-top:.6rem;background:#9D7BE8;border:0;color:#07070b;
font-weight:500}</style>
<form onsubmit="event.preventDefault();fetch('/api/login',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({token:t.value})})
.then(r=>r.ok?location.reload():alert('no'))">
<div><input id="t" type="password" placeholder="token" autofocus>
<button>Unlock</button></div></form>"""


def page():
    tabs = shell.tabs_html("hub", "run") if shell else ""
    tabcss = shell.shell_css() if shell else ""
    return PAGE.replace("<!--VARS-->", css_vars()) \
               .replace("<!--TABCSS-->", tabcss).replace("<!--TABS-->", tabs)


PAGE = r"""<!DOCTYPE html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>OmniCortex</title>
<style>
:root{
<!--VARS-->
  --radius:9px;
}
<!--TABCSS-->
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui;
padding:14px;padding-top:calc(14px + env(safe-area-inset-top,0px))}
.ns-tabs{background:var(--panel);border-bottom:1px solid var(--line)}
.ns-tab{color:var(--fg-dim)} .ns-tab.on{color:#fff;border-bottom-color:var(--accent)}
h1{font-size:15px;margin:0 0 12px;font-weight:500;color:#fff}
.card{background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:12px;margin-bottom:12px}
.mut{color:var(--fg-dim);font-size:12px}
.svc{display:flex;align-items:center;gap:10px;padding:9px 0;
border-bottom:1px solid var(--line)}
.svc:last-child{border-bottom:0}
.dot{width:9px;height:9px;border-radius:50%;background:var(--fg-faint);flex:none}
.dot.up{background:var(--emerald)}
.dot.busy{background:var(--warn-fg)}
button{background:transparent;border:1px solid var(--line-strong);
color:var(--fg);border-radius:var(--radius);padding:7px 13px;
font:14px system-ui;min-height:38px;cursor:pointer}
button:hover{background:var(--panel-2);border-color:var(--accent-dim)}
button.on{background:var(--accent);border-color:var(--accent);color:var(--bg);
font-weight:500}
button:disabled{opacity:.4;cursor:not-allowed}
pre{background:#050508;border:1px solid var(--line);border-radius:var(--radius);
padding:10px;font:11px ui-monospace,monospace;max-height:300px;overflow:auto;
white-space:pre-wrap;color:var(--fg-dim)}
.err{color:var(--danger-fg);border-left:2px solid var(--danger);
padding-left:9px;border-radius:0;font-size:12px;margin-top:6px}
.warn{color:var(--warn-fg);border-left:2px solid var(--warn);
padding-left:9px;border-radius:0;font-size:12px;margin-top:6px}
iframe{width:100%;height:70vh;border:1px solid var(--line);
border-radius:12px;background:var(--panel)}
input{background:var(--panel-2);border:1px solid var(--line);color:var(--fg);
border-radius:var(--radius);padding:7px 9px;font:13px ui-monospace,monospace;
width:100%}
label{display:block;font-size:12px;color:var(--fg-dim);margin:8px 0 3px}
</style>
<!--TABS-->
<h1>OmniCortex</h1>

<div class="card">
  <div class="mut" style="margin-bottom:6px">services</div>
  <div id="svcs"></div>
  <div class="mut" style="margin-top:9px">
    A service starts when you open its tab and stops 90s after the last tab
    closes.</div>
</div>

<div class="card" id="cfgcard">
  <div class="mut" style="margin-bottom:4px">configuration</div>
  <label>llama-server binary</label>
  <input id="server_bin" placeholder="~/llama.cpp/build/bin/llama-server">
  <label>models directory</label>
  <input id="models_dir" placeholder="/path/to/.models">
  <label>trace session (for Replay): pick a checked reply, a trace_sample.py run or a decoding-lab recording, or type a path</label>
  <input id="trace" list="traces" placeholder="runs/trace-abc">
  <datalist id="traces"></datalist>
  <label>live source (for Live): a llama-server patched with llama-tools/server-activations; empty = simulated</label>
  <input id="live_source" placeholder="http://127.0.0.1:8080">
</div>

<div class="card" id="framecard" style="display:none">
  <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px">
    <span class="mut" id="framename"></span>
    <button id="close" style="margin-left:auto">Close tab</button>
  </div>
  <iframe id="frame"></iframe>
</div>

<div class="card">
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
    <span class="mut" id="logname">no service selected</span>
  </div>
  <pre id="log">service output appears here</pre>
</div>

<script>
const $=s=>document.querySelector(s);
const HOLDER = Math.random().toString(36).slice(2);
let open=null, es=null;

const cfg=()=>({server_bin:$('#server_bin').value.trim(),
  models_dir:$('#models_dir').value.trim(), trace:$('#trace').value.trim(),
  live_source:$('#live_source').value.trim()});
for(const k of ['server_bin','models_dir','trace','live_source']){
  try{ const v=localStorage.getItem('ns_'+k); if(v) $('#'+k).value=v; }catch{}
  $('#'+k).oninput=e=>{ try{ localStorage.setItem('ns_'+k,e.target.value) }catch{} };
}

fetch('/api/traces').then(r=>r.json()).then(j=>{
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  $('#traces').innerHTML=j.traces.map(t=>`<option value="${esc(t.path)}">${esc(t.model||'')} · ${esc(t.created)} · ${({studio:'checked reply',decode_lab:'decoding lab'})[t.source]||'trace run'}</option>`).join('');
  if(!$('#trace').value && j.traces.length) $('#trace').value=j.traces[0].path;
}).catch(()=>{});

async function tick(){
  const s=await (await fetch('/api/services')).json();
  $('#svcs').innerHTML=Object.entries(s).map(([n,v])=>{
    const cls=v.running?'dot up':(v.available?'dot':'dot');
    const state=v.running?`running on :${v.port} · ${v.holders} tab(s) · ${v.secs}s`
      :(v.available?'stopped':'script missing');
    return `<div class="svc"><span class="${cls}"></span>
      <span style="flex:1"><b>${n}</b> <span class="mut">— ${state}</span><br>
      <span class="mut">${v.desc}</span></span>
      <button data-open="${n}" ${v.available?'':'disabled'}
        class="${open===n?'on':''}">${open===n?'Opened':'Open'}</button></div>`;
  }).join('');
  document.querySelectorAll('[data-open]').forEach(b=>
    b.onclick=()=>openTab(b.dataset.open));
}

async function openTab(name){
  if(open===name) return;
  if(open) await closeTab();
  const r=await (await fetch('/api/acquire',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name,holder:HOLDER,config:cfg()})})).json();
  if(!r.ok){
    $('#log').innerHTML='<span class="err">'+
      (r.message||'failed').replace(/</g,'&lt;')+'</span>';
    $('#logname').textContent=name+' — failed to start';
    tick(); return;
  }
  open=name;
  $('#framecard').style.display='block';
  $('#framename').textContent=name+' · :'+r.port;
  $('#frame').src='http://'+location.hostname+':'+r.port+'/';
  follow(name); tick();
}

async function closeTab(){
  if(!open) return;
  await fetch('/api/release',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:open,holder:HOLDER})});
  $('#frame').src='about:blank';
  $('#framecard').style.display='none';
  open=null; tick();
}
$('#close').onclick=closeTab;

// Closing the browser tab must release too, or the service outlives the UI.
addEventListener('pagehide',()=>{ if(open) navigator.sendBeacon('/api/release',
  new Blob([JSON.stringify({name:open,holder:HOLDER})],
           {type:'application/json'})); });

function follow(name){
  if(es) es.close();
  $('#logname').textContent=name;
  $('#log').textContent='';
  es=new EventSource('/api/log?name='+encodeURIComponent(name));
  es.onmessage=e=>{
    const l=JSON.parse(e.data), p=$('#log');
    p.textContent+=l+'\n'; p.scrollTop=p.scrollHeight;
  };
}
tick(); setInterval(tick,4000);
</script>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--idle-after", type=int, default=90,
                   help="seconds to keep a service alive after its last tab")
    sec.add_server_security_args(p)
    a = p.parse_args()

    os.environ.setdefault(sec.TOKEN_ENV, os.environ.get("OC_STUDIO_TOKEN", ""))
    token = sec.resolve_token(a.token, a.token_file) or None
    tls = bool(a.tls_cert and a.tls_key)
    try:
        for w in sec.check_bind(a.host, token or "", tls=tls, allow_plaintext=a.allow_plaintext):
            print("warning:", w)
    except sec.SecurityConfigError as e:
        raise SystemExit(f"error: {e}")

    STATE.update({"root": os.path.abspath(a.root),
                  "token": token,
                  "idle_after": a.idle_after})
    SERVICES.update(declare(STATE["root"], sys.executable))

    print(f"OmniCortex hub on http://{a.host}:{a.port}")
    print(f"root: {STATE['root']}")
    for n, s in SERVICES.items():
        ok = os.path.exists(os.path.join(STATE["root"], s["script"]))
        print(f"  {n:<8} {s['script']:<18} {'ok' if ok else 'MISSING'}")

    threading.Thread(target=reaper, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    if tls:
        srv.socket = sec.server_ssl_context(a.tls_cert, a.tls_key).wrap_socket(
            srv.socket, server_side=True, do_handshake_on_connect=False)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for n in list(RUNNING):
            print(f"[hub] stopping {n}")
            stop(n)


if __name__ == "__main__":
    main()
