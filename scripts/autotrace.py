#!/usr/bin/env python3
"""
A tracing proxy: near-realtime visualisation with no fork and no new C++.

The assumption that held this up was that activations have to arrive *with* the
tokens. They do not. This sits in front of llama-server, forwards chat requests
untouched, streams the response back with no added latency, and then traces the
completed text in the background. The visualisation lands a beat behind the
words instead of alongside them -- one prefill pass, a few seconds for a typical
response.

    llama-server -m model.gguf --port 8080 &
    python viz/stream.py --token-file viewer.token --host 0.0.0.0 --allow-plaintext &          # viewer
    python scripts/autotrace.py --upstream http://127.0.0.1:8080 \\
        --binary ~/llama.cpp/build/bin/llama-cett-dump \\
        --gguf model.gguf --tokenizer Qwen/Qwen3-8B \\
        --n-layers 36 --publish http://127.0.0.1:7890 --port 8088

Then point Cline, Studio or anything else at :8088 instead of :8080. Nothing
downstream knows it is being traced.

Cost: one extra prefill per response, on whichever machine holds the GGUF. That
competes with the next generation for the GPU, so --trace-every lets you sample
rather than trace everything, and traces are skipped while a request is in
flight.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "viz"))

CFG = {}
JOBS = queue.Queue(maxsize=4)
STATS = {"requests": 0, "traced": 0, "skipped": 0, "failed": 0,
         "last_ms": 0, "busy": False}


def forward(path, body, headers, stream):
    """Pass the request through unchanged. The proxy must not alter behaviour,
    or the thing being traced is not the thing being served."""
    req = urllib.request.Request(
        CFG["upstream"].rstrip("/") + path, method="POST", data=body,
        headers={k: v for k, v in headers.items()
                 if k.lower() in ("content-type", "authorization")})
    return urllib.request.urlopen(req, timeout=900)


def trace_worker():
    from records import Recorder
    while True:
        job = JOBS.get()
        if job is None:
            return
        STATS["busy"] = True
        t0 = time.time()
        try:
            _trace(job, Recorder)
            STATS["traced"] += 1
        except Exception as e:
            STATS["failed"] += 1
            print(f"[autotrace] {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            STATS["busy"] = False
            STATS["last_ms"] = int((time.time() - t0) * 1000)


def _get_tokenizer():
    """Load the configured tokenizer without trusting Hub code by default."""
    tok = CFG.get("_tok")
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            CFG["tokenizer"],
            trust_remote_code=bool(CFG.get("trust_remote_code", False)),
        )
        CFG["_tok"] = tok
    return tok


def _trace(job, Recorder):
    """One prefill over prompt+response, reduced, published."""
    tok = _get_tokenizer()
    prompt = tok.apply_chat_template(job["messages"],
                                     add_generation_prompt=True, tokenize=False)
    text = prompt + job["response"]

    with tempfile.TemporaryDirectory() as work:
        m1 = os.path.join(work, "t.jsonl")
        qid = f"live{int(time.time() * 1000) % 10 ** 9}"
        with open(m1, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": qid, "text": text},
                               ensure_ascii=False) + "\n")
        base = [CFG["binary"], "-m", CFG["gguf"]]
        subprocess.run(base + ["--tokenize-only", "-ngl", "0",
                               "--manifest", m1, "--outdir", work],
                       check=True, capture_output=True)

        from extract_activations_gguf import read_aggregate, read_tokens
        ids = read_tokens(os.path.join(work, f"{qid}.toks"))
        n_tok = len(ids)
        if n_tok > CFG["max_tokens"]:
            STATS["skipped"] += 1
            return
        stride = max(1, n_tok // CFG["max_frames"])
        spans = [[t, min(t + stride, n_tok)] for t in range(0, n_tok, stride)]

        m2 = os.path.join(work, "s.jsonl")
        with open(m2, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": qid, "text": text, "spans": spans},
                               ensure_ascii=False) + "\n")
        subprocess.run(base + ["-ngl", str(CFG["ngl"]), "-b", str(CFG["batch"]),
                               "-c", str(CFG["batch"]),
                               "--n-layers", str(CFG["n_layers"]),
                               "--manifest", m2, "--outdir", work],
                       check=True, capture_output=True)

        _, agg, seen, counts, n_exp = read_aggregate(
            os.path.join(work, f"{qid}.bin"))

    pieces = [tok.decode([int(i)]) for i in ids]
    _publish(agg, pieces, stride, qid)


def _publish(agg, pieces, stride, qid):
    """Push frames to viz/stream.py at its chosen tier."""
    import numpy as np
    from stream import reduce_frame
    arr = np.asarray(agg, dtype=np.float32)
    if arr.ndim == 4:                       # MoE: fold experts into the row
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    lo, hi = np.percentile(arr, [50, 99.5])
    scale = float(hi - lo) or 1.0

    url = CFG.get("publish")
    for i, frame in enumerate(arr):
        z = float((frame.max() - lo) / scale)
        payload = reduce_frame(frame.tolist(), CFG["tier"], score=z * 3 - 1)
        payload.update({"i": i, "tok": "".join(
            pieces[i * stride:(i + 1) * stride]), "qid": qid})
        if not url:
            continue
        try:
            req = urllib.request.Request(
                url.rstrip("/") + "/api/push", method="POST",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {CFG['token']}"}
                            if CFG.get("token") else {})})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            return          # viewer gone; drop the rest rather than blocking


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/autotrace":
            body = json.dumps({**STATS, "queued": JOBS.qsize()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def _score(self, raw):
        """POST /api/score {messages, response} -> H-Neuron classifier score.

        Used by delegate.py's gate. Synchronous: one prefill on this GGUF."""
        try:
            if not CFG.get("classifier"):
                raise ValueError("start autotrace with --classifier to enable /api/score")
            if "_scorer" not in CFG:
                from hscore import HScorer
                CFG["_scorer"] = HScorer(CFG["binary"], CFG["gguf"], CFG["classifier"],
                                         CFG["ngl"], CFG["batch"])
            req = json.loads(raw or b"{}")
            res = CFG["_scorer"].score(req.get("messages", []), str(req.get("response", "")))
            res["detail"] = f"prob {res['prob']:.2f} over {res['n_tokens']} response tokens"
            code = 200
        except Exception as e:
            res, code = {"error": f"{type(e).__name__}: {e}"[:300]}, 400
        body = json.dumps(res).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if self.path == "/api/score":
            return self._score(raw)
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            req = {}
        STATS["requests"] += 1
        streaming = bool(req.get("stream"))

        try:
            up = forward(self.path, raw, self.headers, streaming)
        except Exception as e:
            self.send_error(502, str(e)[:120])
            return

        self.send_response(200)
        for h in ("Content-Type",):
            if up.headers.get(h):
                self.send_header(h, up.headers[h])
        self.send_header("Connection", "close")
        self.end_headers()

        # Reassemble the text while forwarding it, so tracing costs the client
        # nothing in latency.
        collected = []
        try:
            if streaming:
                for line in up:
                    self.wfile.write(line)
                    self.wfile.flush()
                    s = line.decode(errors="replace").strip()
                    if s.startswith("data:") and "[DONE]" not in s:
                        try:
                            d = json.loads(s[5:])
                            delta = d["choices"][0].get("delta", {})
                            collected.append(delta.get("content") or "")
                            if delta.get("reasoning_content"):
                                collected.append(delta["reasoning_content"])
                        except Exception:
                            pass
            else:
                body = up.read()
                self.wfile.write(body)
                try:
                    d = json.loads(body)
                    collected.append(
                        d["choices"][0]["message"].get("content") or "")
                except Exception:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            return

        text = "".join(collected).strip()
        if not text or self.path.endswith("/embeddings"):
            return
        STATS["requests"] = STATS["requests"]
        if STATS["requests"] % CFG["trace_every"]:
            STATS["skipped"] += 1
            return
        if STATS["busy"]:
            # Never queue behind a trace: the next generation matters more than
            # visualising the last one.
            STATS["skipped"] += 1
            return
        try:
            JOBS.put_nowait({"messages": req.get("messages", []),
                             "response": text})
        except queue.Full:
            STATS["skipped"] += 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="the real llama-server")
    p.add_argument("--binary", required=True, help="llama-cett-dump")
    p.add_argument("--gguf", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--publish", help="viz/stream.py base URL")
    p.add_argument("--token", help="bearer token for the viewer")
    p.add_argument("--classifier", help="classifier.npz for this GGUF; enables POST /api/score")
    p.add_argument("--tier", default="sparse",
                   choices=["raw", "binned", "sparse"])
    p.add_argument("--trace-every", type=int, default=1,
                   help="trace one request in N; raise it to sample")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--trust-remote-code", action="store_true",
                   help="allow execution of custom Python code from the tokenizer repository")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--allow-unauthenticated", action="store_true", help="permit a non-loopback bind (no auth)")
    return p.parse_args(argv)


def main():
    a = parse_args()
    import oc_security as sec
    sec.loopback_only(a.host, "the tracing proxy", a.allow_unauthenticated)
    CFG.update(vars(a))
    CFG["max_frames"] = a.max_frames
    CFG["max_tokens"] = a.max_tokens

    threading.Thread(target=trace_worker, daemon=True).start()
    print(f"tracing proxy on http://{a.host}:{a.port} -> {a.upstream}")
    print(f"point clients here instead of upstream; tier {a.tier}, "
          f"tracing 1 in {a.trace_every}")
    print("cost: one extra prefill per traced response, on the machine holding "
          "the GGUF")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
