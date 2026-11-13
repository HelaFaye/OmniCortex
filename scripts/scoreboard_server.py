#!/usr/bin/env python3
"""
A reference scoreboard for score cards (score_cards.py): anyone can host one.

    python scripts/scoreboard_server.py --db board.sqlite --port 8765 --admin-token-file admin.token
    # behind TLS (a reverse proxy, or --tls-cert/--tls-key) when it is on the internet

What it does with a submission:
1. **Verifies the card against its log.**
   - The bank fingerprint of the items must match this server's TestQA
     bank.
   - Every reply is graded again with the same grader. Items that need code
     execution are counted, not re-run.
   - The counts per subject must equal the card's.
   - A card whose replies do not grade to its numbers is rejected and never
     shown.
2. **Stores the log privately.**
   - It goes in a separate table that no public route reads.
   - Only the admin token can fetch a log, to investigate a disputed result.
3. **Publishes only public, verified cards.**
   - A card is public when the submitter asked for it.
   - It is marked *listed* when the model file's SHA-256 is in its Hugging
     Face repo; with --verify-hf this server checks that itself.
   - Results group by model file and *variant*: the H-Neuron set, the
     suppression scale, the LoRA, the lineage, the engine and its
     answer-changing options, and the sampling. Different setups of one
     file are never mixed.

Routes: POST /api/submit; GET /api/board (public aggregates);
GET /api/card/<id> (a public card); GET /api/admin/log/<id> (admin only);
GET / (a page).
"""
from __future__ import annotations

import argparse
import hmac
import json
import secrets
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
MAX_BODY = 32 * 1024 * 1024

SCHEMA = """
create table if not exists submissions (
  id text primary key, received real, model_sha text, model_file text, quant text, hf_repo text,
  listed integer, visibility text, variant_id text, card text, status text, note text);
create table if not exists logs (id text primary key, log text);
"""


def verify_card(tasks: dict, card: dict, log: dict, verify_hf: bool = False) -> tuple[str, str]:
    """(status, note): verified, unverified (another bank) or rejected. tasks: the bank by id."""
    import score_cards as sc
    import testqa as tq
    items = log.get("items") or []
    if not items:
        return "rejected", "no log items"
    unknown = [i["id"] for i in items if i.get("id") not in tasks]
    if unknown:
        return "unverified", f"{len(unknown)} items are not in this server's bank"
    asked = [tasks[i["id"]] for i in items]
    if sc.bank_fingerprint(asked) != card.get("test", {}).get("bank"):
        return "unverified", "the bank differs from this server's"
    args = SimpleNamespace(allow_exec=False, exec_timeout=5)
    per = {}
    mismatched, not_regraded = 0, 0
    for it in items:
        t = tasks[it["id"]]
        claimed = it.get("verdict")
        if claimed in ("error", "skipped"):
            continue
        if t["kind"] == "code_exec":
            not_regraded += 1
            v = claimed
        else:
            v = tq.grade(t, str(it.get("reply", "")), args)[0]
            if v != claimed:
                mismatched += 1
        c = per.setdefault(t["subject"], {"n": 0, "correct": 0, "wrong": 0, "abstained": 0, "other": 0})
        c["n"] += 1
        key = "correct" if v in ("correct", "answered") else v if v in ("wrong", "abstained") else "other"
        c[key] += 1
    if mismatched:
        return "rejected", f"{mismatched} replies do not grade to the verdicts claimed"
    if per != (card.get("test") or {}).get("per_subject"):
        return "rejected", "the card's counts differ from its log"
    if card.get("model", {}).get("listed") and verify_hf:
        repo = card["model"].get("hf_repo")
        if not repo or not _hf_has(repo, card["model"]["sha256"]):
            return "rejected", "the model is claimed to be listed but its file is not in that repo"
    return "verified", (f"{not_regraded} code items taken as reported" if not_regraded else "all replies regraded")


class Board:
    def __init__(self, db: str, tasks_paths=None, verify_hf: bool = False):
        import testqa as tq
        self.tq = tq
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.tasks = {t["id"]: t for t in tq.load_tasks(tasks_paths or [str(ROOT / "qa" / "bank")], None, None, 0)}
        self.verify_hf = verify_hf

    def verify(self, card: dict, log: dict) -> tuple[str, str]:
        return verify_card(self.tasks, card, log, self.verify_hf)

    def submit(self, card: dict, log: dict) -> dict:
        if not isinstance(card, dict) or card.get("card_version") != 1:
            raise ValueError("unknown card version")
        m = card.get("model") or {}
        if not isinstance(m.get("sha256"), str) or len(m["sha256"]) != 64:
            raise ValueError("card has no model SHA-256")
        status, note = self.verify(card, log)
        sid = secrets.token_hex(8)
        vis = "public" if card.get("visibility") == "public" else "private"
        with self.lock:
            self.db.execute("insert into submissions values (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (sid, time.time(), m["sha256"], str(m.get("file", ""))[:200], m.get("quant"),
                             m.get("hf_repo"), int(bool(m.get("listed"))), vis, str(card.get("variant_id", ""))[:32],
                             json.dumps(card), status, note))
            self.db.execute("insert into logs values (?,?)", (sid, json.dumps(log)))
            self.db.commit()
        shown = vis == "public" and status == "verified"
        return {"id": sid, "status": status, "note": note, "visibility": vis, "shown_publicly": shown}

    def board(self, model: str | None = None) -> list[dict]:
        rows = self.db.execute("select model_sha, model_file, quant, hf_repo, listed, variant_id, card from submissions "
                               "where visibility='public' and status='verified'"
                               + (" and (model_sha=? or model_file=? or hf_repo=?)" if model else ""),
                               (model, model, model) if model else ()).fetchall()
        return aggregate([json.loads(r[6]) for r in rows])


def aggregate(cards: list[dict]) -> list[dict]:
    """Public rows: one per model file and variant, over its verified cards."""
    groups = defaultdict(list)
    for c in cards:
        groups[(c["model"]["sha256"], c.get("variant_id", ""))].append(c)
    out = []
    for (sha, vid), cs in groups.items():
        per = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0, "abstained": 0, "other": 0})
        for c in cs:
            for s, v in c["test"]["per_subject"].items():
                for k in per[s]:
                    per[s][k] += v.get(k, 0)
        n = sum(v["n"] for v in per.values())
        corr = sum(v["correct"] for v in per.values())
        wrong = sum(v["wrong"] for v in per.values())
        m, v0 = cs[0]["model"], cs[-1]["variant"]
        out.append({"model_sha": sha, "file": m.get("file"), "quant": m.get("quant"), "hf_repo": m.get("hf_repo"),
                    "listed": bool(m.get("listed")), "variant_id": vid, "submissions": len(cs), "items": n,
                    "accuracy": round(corr / n, 4) if n else None,
                    "hallucination_rate": round(wrong / n, 4) if n else None,
                    "per_subject": {s: {**v, "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else None}
                                    for s, v in per.items()},
                    "variant": {"engine": v0["engine"], "suppression": v0.get("suppression"),
                                "lora": v0.get("lora"), "lineage": v0.get("lineage"), "sampling": v0.get("sampling")},
                    "hardware": sorted({", ".join(g["name"] for g in c["hardware"].get("gpus") or []) or
                                        c["hardware"].get("cpu", "cpu") for c in cs}),
                    "performance": [dict(p, hardware=c["hardware"], model_bytes=m.get("size"), quant=m.get("quant"))
                                    for c in cs for p in (c.get("performance") or [])]})
        ex = _extras(cs)
        if ex:
            out[-1]["extras"] = ex
    out.sort(key=lambda e: -(e["accuracy"] or 0))
    return out


def _extras(cards):
    try:
        import unified_review
    except ImportError:
        return {}
    return unified_review.aggregate_extras(cards)


def _hf_has(repo: str, sha: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=15) as r:
            info = json.loads(r.read())
    except Exception:
        return False
    return any((s.get("lfs") or {}).get("sha256") == sha for s in info.get("siblings") or [])


PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex scoreboard</title><style>body{font:14px system-ui;margin:0;padding:1rem;background:#0f1115;color:#e6e8eb}
table{border-collapse:collapse;width:100%}td,th{padding:.35rem .4rem;border-bottom:1px solid #2a2f37;text-align:left;font-size:13px}
.n{color:#9aa1ab;font-size:12px}code{font-size:11px}</style><h1>Scoreboard</h1>
<p class="n">Verified public results only. Each row is one model file in one setup (H-Neuron set, suppression, LoRA, lineage,
engine options that change answers, sampling). Verification logs are private and used only to check results.</p>
<table id="t"><tr><th>model</th><th>setup</th><th>accuracy</th><th>hallucination</th><th>items</th><th>runs</th><th>hardware</th></tr></table>
<script>fetch('api/board').then(r=>r.json()).then(j=>{const e=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('t').innerHTML+=j.board.map(b=>`<tr><td>${e(b.hf_repo||b.file)}<div class="n">${e(b.quant||'')} ${b.listed?'listed':'unlisted, published by its submitter'}</div></td>
<td><code>${e(b.variant_id)}</code><div class="n">${e(b.variant.engine.name)} ${e(b.variant.engine.version||'')}${b.variant.suppression?' · H-Neuron suppression':''}${b.variant.lora?' · LoRA '+e(b.variant.lora.scale):''}${b.variant.lineage&&b.variant.lineage.length?' · '+b.variant.lineage.map(x=>e(x.kind)).join(', '):''}</div></td>
<td>${b.accuracy==null?'':(b.accuracy*100).toFixed(1)+'%'}</td><td>${b.hallucination_rate==null?'':(b.hallucination_rate*100).toFixed(1)+'%'}</td><td>${b.items}</td><td>${b.submissions}</td><td class="n">${e(b.hardware.join('; '))}</td></tr>`).join('')})</script>"""


def make_handler(board: Board, admin_token: str):
    hits: dict = defaultdict(list)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, ctype="application/json"):
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/board":
                q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&") if "=" in x) \
                    if "?" in self.path else {}
                return self._send(200, {"board": board.board(q.get("model"))})
            if path.startswith("/api/card/"):
                r = board.db.execute("select card from submissions where id=? and visibility='public' and "
                                     "status='verified'", (path[10:],)).fetchone()
                return self._send(200, json.loads(r[0])) if r else self._send(404, {"error": "not found"})
            if path.startswith("/api/admin/log/"):
                auth = self.headers.get("Authorization", "")
                if not admin_token or not hmac.compare_digest(auth, f"Bearer {admin_token}"):
                    return self._send(401, {"error": "admin only"})
                r = board.db.execute("select log from logs where id=?", (path[15:],)).fetchone()
                return self._send(200, json.loads(r[0])) if r else self._send(404, {"error": "not found"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/submit":
                return self._send(404, {"error": "not found"})
            ip = self.client_address[0]
            now = time.time()
            hits[ip] = [t for t in hits[ip] if now - t < 3600]
            if len(hits[ip]) >= 60:
                return self._send(429, {"error": "too many submissions from this address; try later"})
            hits[ip].append(now)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n))
                return self._send(200, board.submit(body.get("card"), body.get("log") or {}))
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
    return H


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default="scoreboard.sqlite")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--admin-token-file", help="file with the token that may read verification logs")
    p.add_argument("--tasks", nargs="+", help="the TestQA bank to verify against (default: this checkout's)")
    p.add_argument("--verify-hf", action="store_true", help="check listed models' files against Hugging Face")
    p.add_argument("--tls-cert")
    p.add_argument("--tls-key")
    a = p.parse_args(argv)
    token = Path(a.admin_token_file).read_text().strip() if a.admin_token_file else ""
    board = Board(a.db, a.tasks, a.verify_hf)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(board, token))
    if a.tls_cert and a.tls_key:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.tls_cert, a.tls_key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"scoreboard on {'https' if a.tls_cert else 'http'}://{a.host}:{a.port}  ({len(board.tasks)} bank items)")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
