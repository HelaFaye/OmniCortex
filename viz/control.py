#!/usr/bin/env python3
"""
OmniCortex control: run the pipeline and watch it, from a browser.

viz/server.py shows where the pipeline is. This runs it. One page with the
stage list, a live view of whatever is currently going, and a Run button per
stage that streams the subprocess output back.

    python viz/control.py --root . --port 7861
    python viz/control.py --root . --host 0.0.0.0 --token-file control.token --tls-cert c.pem --tls-key k.pem

It reads state off the filesystem rather than keeping its own, so it tells the
truth about a run started from a terminal, and a run started here survives the
page being closed. Collection in particular is long enough that watching it
from a phone is the point.

No authentication unless --token is given. --host 0.0.0.0 without one puts
process launching on your network; do not do that.
"""

import argparse
import hmac
import sys
import http.cookies
import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import oc_security as sec  # noqa: E402
try:
    import shell            # shared tab registry, viz/modes.json
except ImportError:
    shell = None

STATE = {"root": ".", "token": None, "py": "python"}
JOBS = {}          # name -> {proc, log deque, started, rc}
JOBS_LOCK = threading.Lock()


# ------------------------------------------------------------------- reading

def _count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _count_label(path, label):
    needle = f'"judge": "{label}"'.encode()
    try:
        with open(path, "rb") as f:
            return sum(1 for line in f if needle in line)
    except OSError:
        return 0


def collection_state():
    """Live counts, straight off the files the collector appends to."""
    root = STATE["root"]
    out = os.path.join(root, "data", "consistency_samples.jsonl")
    att = out + ".attempted"
    t = _count_label(out, "true")
    f = _count_label(out, "false")
    attempted = _count_lines(att)
    kept = t + f
    return {
        "attempted": attempted,
        "kept": kept,
        "correct": t,
        "hallucinated": f,
        "pairs": min(t, f),
        "yield": round(kept / attempted, 3) if attempted else None,
        # 300 pairs is the rough target for a powered causal test.
        "pairs_target": 300,
    }


def throughput_rows(limit=40):
    root = STATE["root"]
    logs = os.path.join(root, "logs")
    rows = []
    try:
        files = sorted(x for x in os.listdir(logs) if x.startswith("throughput-"))
    except OSError:
        return rows
    if not files:
        return rows
    try:
        with open(os.path.join(logs, files[-1])) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == len(head):
                    rows.append(dict(zip(head, parts)))
    except OSError:
        pass
    return rows[-limit:]


STAGES = [
    ("collect", "data/consistency_samples.jsonl",
     "Generate and label responses. The bottleneck; runs on the fast box."),
    ("split", "data/train_qids.json",
     "Balanced train/test split by question id."),
    ("extract", "data/activations/neuron_index.json",
     "CETT per neuron, via llama-cett-dump on Vulkan."),
    ("classify", "models_1v1/h_neurons.json",
     "L1 logistic regression; writes the H-Neuron set."),
    ("trace", "runs", "Token-resolved trace for the 3D viewers."),
    ("profile", "profiles", "Suppression profile, for the causal test."),
]


def pipeline_state():
    root = STATE["root"]
    out = []
    for name, rel, desc in STAGES:
        p = os.path.join(root, rel)
        done = os.path.exists(p) and (not os.path.isdir(p) or bool(os.listdir(p)))
        info = ""
        if done and rel.endswith("h_neurons.json"):
            try:
                with open(p) as f:
                    d = json.load(f)
                info = f"{d.get('total', 0)} neurons"
            except Exception:
                pass
        elif done and rel.endswith("neuron_index.json"):
            d = os.path.dirname(p)
            info = f"{len([x for x in os.listdir(d) if x.endswith('.npy')])} files"
        out.append({"name": name, "path": rel, "done": done,
                    "desc": desc, "info": info})
    return out


# ------------------------------------------------------------------- running

def start_job(name, cmd):
    with JOBS_LOCK:
        j = JOBS.get(name)
        if j and j["proc"].poll() is None:
            return False, "already running"
    proc = subprocess.Popen(
        cmd, cwd=STATE["root"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        # Own process group, so stopping kills the children too -- a collection
        # run spawns a python that spawns HTTP workers.
        start_new_session=True)
    job = {"proc": proc, "log": [], "started": time.time(), "rc": None,
           "cmd": " ".join(shlex.quote(c) for c in cmd), "subs": []}
    with JOBS_LOCK:
        JOBS[name] = job

    def pump():
        for line in proc.stdout:
            line = line.rstrip("\n")
            with JOBS_LOCK:
                job["log"].append(line)
                del job["log"][:-400]          # keep the tail only
                for q in job["subs"]:
                    try:
                        q.put_nowait(line)
                    except queue.Full:
                        pass
        job["rc"] = proc.wait()
        with JOBS_LOCK:
            for q in job["subs"]:
                try:
                    q.put_nowait(f"__exit__ {job['rc']}")
                except queue.Full:
                    pass

    threading.Thread(target=pump, daemon=True).start()
    return True, "started"


def stop_job(name):
    with JOBS_LOCK:
        j = JOBS.get(name)
    if not j or j["proc"].poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(j["proc"].pid), signal.SIGTERM)
    except OSError:
        j["proc"].terminate()
    return True


def job_state():
    with JOBS_LOCK:
        return {n: {"running": j["proc"].poll() is None, "rc": j["rc"],
                    "cmd": j["cmd"], "secs": int(time.time() - j["started"]),
                    "tail": j["log"][-3:]}
                for n, j in JOBS.items()}


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
        if self.path.split("?")[0] == "/":
            page = PAGE
            if shell:
                # Other surfaces run on their own ports; point tabs we do not
                # implement at them rather than leaving dead ends.
                links = {"chat": "http://%s:7870/" % self.headers.get(
                             "Host", "localhost").split(":")[0],
                         "live": "http://%s:7890/" % self.headers.get(
                             "Host", "localhost").split(":")[0]}
                page = page.replace("<!--OC_SHELL_CSS-->", shell.shell_css())
                page = page.replace("<!--OC_TABS-->",
                                    shell.tabs_html("control", "run", links))
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path.startswith("/api/modes"):
            if not shell:
                return self._send(200, json.dumps({"modes": []}))
            return self._send(200, shell.as_json("control"))
        if self.path.startswith("/api/state"):
            return self._send(200, json.dumps({
                "collection": collection_state(),
                "pipeline": pipeline_state(),
                "jobs": job_state(),
                "throughput": throughput_rows(),
            }))
        if self.path.startswith("/api/log"):
            name = self.path.split("name=", 1)[-1].split("&")[0]
            return self._stream(name)
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/login":
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
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
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/run":
            cmd = req.get("cmd")
            if not isinstance(cmd, list) or not cmd:
                return self._send(400, json.dumps({"error": "cmd must be a list"}))
            ok, msg = start_job(req.get("name", "job"), cmd)
            return self._send(200 if ok else 409, json.dumps({"ok": ok,
                                                              "message": msg}))
        if self.path == "/api/stop":
            return self._send(200, json.dumps({"ok": stop_job(req["name"])}))
        self._send(404, json.dumps({"error": "not found"}))

    def _stream(self, name):
        with JOBS_LOCK:
            job = JOBS.get(name)
        if not job:
            return self._send(404, json.dumps({"error": "no such job"}))
        q = queue.Queue(maxsize=200)
        with JOBS_LOCK:
            job["subs"].append(q)
            backlog = list(job["log"][-120:])
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
            with JOBS_LOCK:
                if q in job["subs"]:
                    job["subs"].remove(q)


LOGIN = """<!DOCTYPE html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex</title><style>body{font:16px system-ui;background:#0b0b10;
color:#c9c9d2;display:flex;align-items:center;justify-content:center;
height:100vh;margin:0}input,button{width:280px;padding:.8rem;font-size:16px;
border-radius:8px;border:1px solid #2a2a34;background:#14141c;color:#fff}
button{margin-top:.6rem;background:#2c5f8a;border:0}</style>
<form onsubmit="event.preventDefault();fetch('/api/login',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({token:t.value})})
.then(r=>r.ok?location.reload():alert('no'))">
<div><input id="t" type="password" placeholder="token" autofocus>
<button>Unlock</button></div></form>"""

PAGE = r"""<!DOCTYPE html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>OmniCortex control</title>
<style>
<!--OC_SHELL_CSS-->
*{box-sizing:border-box}
body{margin:0;background:#0b0b10;color:#c9c9d2;font:14px system-ui;padding:14px;
padding-top:calc(14px + env(safe-area-inset-top,0px))}
h1{font-size:15px;margin:0 0 12px;font-weight:600;color:#fff}
.card{background:#14141c;border:1px solid #23232e;border-radius:10px;
padding:12px;margin-bottom:12px}
.big{font:600 28px ui-monospace,monospace;color:#fff}
.mut{color:#8a8a92;font-size:12px}
.bar{height:6px;background:#23232e;border-radius:3px;overflow:hidden;margin:8px 0}
.bar>div{height:6px;background:#5DCAA5}
table{width:100%;border-collapse:collapse;font:12px ui-monospace,monospace}
td,th{text-align:left;padding:3px 6px;border-bottom:1px solid #1c1c24}
th{color:#8a8a92;font-weight:500}
.stage{display:flex;align-items:center;gap:10px;padding:7px 0;
border-bottom:1px solid #1c1c24}
.dot{width:9px;height:9px;border-radius:50%;background:#3a3a46;flex:none}
.dot.on{background:#1D9E75}.dot.run{background:#EF9F27}
button{background:#1d2530;border:1px solid #2f3a49;color:#c9c9d2;
border-radius:7px;padding:7px 13px;font:14px system-ui;min-height:38px}
button:hover{background:#243040}
pre{background:#08080c;border:1px solid #1c1c24;border-radius:8px;padding:10px;
font:11px ui-monospace,monospace;max-height:320px;overflow:auto;white-space:pre-wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
</style>
<!--OC_TABS-->
<h1>OmniCortex control</h1>
<div class="card">
  <div class="mut">balanced pairs</div>
  <div class="big" id="pairs">–</div>
  <div class="bar"><div id="pbar" style="width:0%"></div></div>
  <div class="mut" id="psub"></div>
  <div class="grid" style="margin-top:12px">
    <div><div class="mut">attempted</div><div id="att" class="big" style="font-size:20px">–</div></div>
    <div><div class="mut">kept</div><div id="kept" class="big" style="font-size:20px">–</div></div>
    <div><div class="mut">yield</div><div id="yld" class="big" style="font-size:20px">–</div></div>
  </div>
</div>
<div class="card"><div class="mut" style="margin-bottom:6px">pipeline</div>
  <div id="stages"></div></div>
<div class="card"><div class="mut" style="margin-bottom:6px">throughput</div>
  <div style="overflow:auto"><table id="tp"></table></div></div>
<div class="card">
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
    <span class="mut" id="jobname">no job</span>
    <button id="stop" style="margin-left:auto">Stop</button>
  </div>
  <pre id="log">select a running stage to follow its output</pre>
</div>
<script>
const $=s=>document.querySelector(s);
let es=null, following=null;
async function tick(){
  const s=await (await fetch('/api/state')).json();
  const c=s.collection;
  $('#pairs').textContent=c.pairs;
  const pct=Math.min(100,100*c.pairs/c.pairs_target);
  $('#pbar').style.width=pct+'%';
  $('#psub').textContent=`${c.correct} correct / ${c.hallucinated} hallucinated `
    + `· target ${c.pairs_target} for a powered causal test`;
  $('#att').textContent=c.attempted;
  $('#kept').textContent=c.kept;
  $('#yld').textContent=c.yield===null?'–':(c.yield*100).toFixed(0)+'%';

  $('#stages').innerHTML=s.pipeline.map(p=>{
    const j=s.jobs[p.name];
    const cls=j&&j.running?'dot run':(p.done?'dot on':'dot');
    const state=j&&j.running?`running ${j.secs}s`:(p.done?(p.info||'done'):'not run');
    return `<div class="stage"><span class="${cls}"></span>
      <span style="flex:1"><b>${p.name}</b>
      <span class="mut"> — ${state}</span><br>
      <span class="mut">${p.desc}</span></span>
      ${j?`<button data-f="${p.name}">Follow</button>`:''}</div>`;
  }).join('');
  document.querySelectorAll('[data-f]').forEach(b=>b.onclick=()=>follow(b.dataset.f));

  const rows=s.throughput;
  if(rows.length){
    const cols=Object.keys(rows[0]);
    $('#tp').innerHTML='<tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>'
      + rows.slice(-12).map(r=>'<tr>'+cols.map(c=>`<td>${r[c]}</td>`).join('')+'</tr>').join('');
  }
}
function follow(name){
  if(es) es.close();
  following=name; $('#jobname').textContent=name; $('#log').textContent='';
  es=new EventSource('/api/log?name='+encodeURIComponent(name));
  es.onmessage=e=>{
    const l=JSON.parse(e.data);
    const p=$('#log');
    // a tqdm bar rewrites its own line; keep the newest rather than stacking
    if(l.includes('\r')||/\d+%\|/.test(l)){
      const lines=p.textContent.split('\n');
      if(/\d+%\|/.test(lines[lines.length-1])) lines.pop();
      lines.push(l); p.textContent=lines.join('\n');
    } else p.textContent+=l+'\n';
    p.scrollTop=p.scrollHeight;
  };
}
$('#stop').onclick=()=>following&&fetch('/api/stop',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({name:following})});
tick(); setInterval(tick,4000);
</script>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--host", default="127.0.0.1")
    sec.add_server_security_args(p)
    a = p.parse_args()
    STATE["root"] = os.path.abspath(a.root)
    os.environ.setdefault(sec.TOKEN_ENV, os.environ.get("OC_STUDIO_TOKEN", ""))
    STATE["token"] = sec.resolve_token(a.token, a.token_file) or None
    tls = bool(a.tls_cert and a.tls_key)
    try:
        for w in sec.check_bind(a.host, STATE["token"] or "", tls=tls, allow_plaintext=a.allow_plaintext):
            print("warning:", w)
    except sec.SecurityConfigError as e:
        raise SystemExit(f"error: {e}")

    print(f"OmniCortex control on http://{a.host}:{a.port}")
    print(f"root: {STATE['root']}")
    c = collection_state()
    print(f"collection: {c['attempted']} attempted, {c['pairs']} pairs")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    if tls:
        srv.socket = sec.server_ssl_context(a.tls_cert, a.tls_key).wrap_socket(
            srv.socket, server_side=True, do_handshake_on_connect=False)
    srv.serve_forever()


if __name__ == "__main__":
    main()
