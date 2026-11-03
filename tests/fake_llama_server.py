#!/usr/bin/env python3
"""Stand-in for llama-server in tests: same CLI shape, same HTTP surface."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m")
p.add_argument("--port", type=int)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--alias", default="")
p.add_argument("--mmproj", default="")
a, _ = p.parse_known_args()


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"status": "ok"})
        self._send({"data": [{"id": a.alias}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            # deterministic bag-of-words vectors: similar wording -> similar vectors
            import hashlib
            import re
            out = []
            for i, t in enumerate(body["input"] if isinstance(body["input"], list) else [body["input"]]):
                v = [0.0] * 64
                for w in re.findall(r"[a-z]+", t.lower()):
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
                out.append({"index": i, "embedding": v})
            return self._send({"data": out, "model": a.alias})
        if self.path == "/tokenize":
            return self._send({"tokens": list(range(len(body.get("content", "").split())))})   # one per word
        msgs = body.get("messages", [])
        if body.get("tools") and body.get("stream") and (not msgs or msgs[-1].get("role") != "tool"):
            # call the first tool named like *__add, the way llama-server streams tool calls
            fn = next((t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("__add")),
                      body["tools"][0]["function"]["name"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for d in ({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                       "function": {"name": fn, "arguments": '{"a": 2,'}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": ' "b": 40}'}}]}):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
                             "data: [DONE]\n\n".encode())
            return
        if msgs and msgs[-1].get("role") == "tool":
            text = f"tool said: {msgs[-1]['content']}"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\ndata: [DONE]\n\n".encode())
            return
        user_last = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "")
        if user_last == "loop please" and body.get("stream"):
            # a reply that loops; resumed (a trailing assistant message), it finishes and shows what it was sent
            resumed = msgs and msgs[-1].get("role") == "assistant"
            pieces = ([f" Done. bias={json.dumps(body.get('logit_bias'))} dry={body.get('dry_multiplier')}"] if resumed
                      else ["Start. "] + ["again and again. "] * 12)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in pieces:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        has_img = any(isinstance(m.get("content"), list) for m in msgs)
        ctx = any(m.get("role") == "system" and "[1] (" in str(m.get("content")) for m in body.get("messages", []))
        text = f"model={a.alias} mmproj={'yes' if a.mmproj else 'no'} image={'yes' if has_img else 'no'}"
        if ctx:
            text += " ctx=yes"
        if any(m.get("role") == "system" and str(m.get("content")).startswith("Summary of the first") for m in msgs):
            text += f" summary=yes turns={sum(1 for m in msgs if m.get('role') != 'system')}"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for n, piece in enumerate([text[:10], text[10:]]):
                ch = {"delta": {"content": piece}}
                if body.get("logprobs"):
                    # first piece certain, second spread over five candidates
                    tops = ([{"token": piece, "logprob": -0.01}] if n == 0 else
                            [{"token": piece, "logprob": math.log(0.2)}] * 5)
                    ch["logprobs"] = {"content": [{"token": piece, "logprob": tops[0]["logprob"],
                                                   "top_logprobs": tops[:body.get("top_logprobs", 5)]}]}
                self.wfile.write(f"data: {json.dumps({'choices': [ch]})}\n\n".encode())
            final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "timings": {"predicted_n": 7, "predicted_per_second": 42.0}}
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            return
        self._send({"choices": [{"message": {"role": "assistant", "content": text}}],
                    "model": a.alias})


ThreadingHTTPServer((a.host, a.port), H).serve_forever()
