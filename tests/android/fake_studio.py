#!/usr/bin/env python3
"""
A stand-in Studio for the client's self-test (clients/kivy/omnicortex_client/selftest.py): pairing, models, a
streamed chat and every reply check, over TLS with a self-signed certificate, the way a real Studio pairs.

    python tests/android/fake_studio.py --host 0.0.0.0 --port 8790 --link-host 10.0.2.2 --out link.txt

Writes the pairing link (with the certificate's fingerprint) to --out and serves until killed. --link-host is
the address the client uses: 10.0.2.2 is the computer running an Android emulator, seen from inside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import ssl
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

CODE, TOKEN = "ABCDEFGHJKMN", "selftest-device-token"
SEEN: list[str] = []          # the paths the client called, for the harness


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _authed(self):
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_GET(self):
        SEEN.append("GET " + self.path.split("?")[0])
        if not self._authed():
            return self._send(401, {"error": "token needed"})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "qwen2.5-1.5b"}, {"id": "acme/coder-7b"}]})
        if self.path.startswith("/api/lookup?"):
            return self._send(200, {"model": "Qwen/Qwen2.5-1.5B-Instruct", "available": True, "stats": [
                {"key": "accuracy", "group": "quality", "label": "Accuracy (TestQA, verified)", "value": 0.71,
                 "unit": "ratio", "n": 400, "variant": "Q4_K_M · llama.cpp"},
                {"key": "tg", "group": "performance", "label": "Generation on RTX 3060", "value": 88.4,
                 "unit": "tok/s"}]})
        self._send(404, {"error": "no"})

    def do_POST(self):
        SEEN.append("POST " + self.path)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/api/pair/claim":
            if req.get("code") != CODE:
                return self._send(403, {"error": "pairing code is wrong, used or expired"})
            return self._send(200, {"token": TOKEN, "expires": None, "fingerprint": self.server.fp})
        if not self._authed():
            return self._send(401, {"error": "token needed"})
        if self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for p in ["The capital ", "of Australia ", "is Sydney."]:
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": p}}]}).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if self.path == "/api/trace":
            return self._send(200, {"pieces": ["The", " capital", " of", " Australia", " is", " Syd", "ney", "."],
                                    "prob": [0.05, 0.1, 0.05, 0.2, 0.3, 0.86, 0.62, 0.1], "flagged": [5, 6],
                                    "threshold": 0.5})
        if self.path == "/api/truth":
            return self._send(200, {"verdict": "knows_but_says_otherwise", "stage": "model.layers.25",
                                    "stage_auc": 0.93, "answer": {"text": "Sydney", "p_true": 0.03},
                                    "alternatives": [{"text": "Canberra", "p_true": 0.74}],
                                    "best_alternative": "Canberra", "alt_leads_at": ["model.layers.18"]})
        if self.path == "/api/emotions":
            n = 8
            return self._send(200, {"names": ["calm", "worried", "happy", "desperate"],
                                    "colors": ["#ff6b6b", "#4dabf7", "#ffd43b", "#69db7c"],
                                    "z": [[round(1.5 * math.sin(t / 2 + k), 2) for k in range(4)] for t in range(n)],
                                    "mean": [1.2, -0.8, 0.6, -0.3], "tokens": ["x"] * n, "top": [[]] * n})
        if self.path == "/api/harvest":
            return self._send(200, {"id": "p1", "pending": 1, "auto_train": False, "min_pairs": 20})
        self._send(404, {"error": "no"})


def serve(host="127.0.0.1", port=0, link_host=None):
    """Start the fake Studio on a thread -> (server, pairing link)."""
    import oc_security as sec
    d = Path(tempfile.mkdtemp())
    cert, key = d / "c.pem", d / "k.pem"
    hosts = []
    for n in sorted({"127.0.0.1", link_host or host} - {"0.0.0.0"}):
        hosts += ["--host", n]
    if sec.main(["selfsigned", *hosts, "--cert", str(cert), "--key", str(key)]) != 0:
        raise SystemExit("could not make a certificate")
    pem = cert.read_text()
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.fp = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem[pem.index("-----BEGIN"):])).hexdigest()
    srv.socket = sec.server_ssl_context(str(cert), str(key)).wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = CODE
    link = f"https://{link_host or host}:{srv.server_port}/pair#c={c[:4]}-{c[4:8]}-{c[8:]}&fp={srv.fp}&a=p"
    return srv, link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--link-host", help="the address the client reaches this host at (10.0.2.2 from an emulator)")
    ap.add_argument("--out", help="write the pairing link here")
    a = ap.parse_args()
    srv, link = serve(a.host, a.port, a.link_host)
    if a.out:
        Path(a.out).write_text(link)
    print(link, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
