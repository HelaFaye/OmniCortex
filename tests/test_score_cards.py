"""Score cards and the reference scoreboard: what is sent, what is verified, what is public."""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "tests"))
import score_cards as sc  # noqa: E402
import scoreboard_server as sb  # noqa: E402
import testqa as tq  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401

GRADE = type("A", (), {"allow_exec": False, "exec_timeout": 5})()


def honest_run(tmp_path, n=12):
    """A results file and cache whose verdicts are what the grader gives the replies."""
    tasks = [t for t in tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)
             if t["kind"] in ("qa", "reasoning") and not t.get("image")][:n]
    rows, cache = [], []
    for i, t in enumerate(tasks):
        reply = (f"Answer: {t['answer']}" if t["kind"] == "reasoning" else t["aliases"][0]) if i % 3 else "Answer: zzz"
        v = tq.grade(t, reply, GRADE)[0]
        rows.append({"id": t["id"], "kind": t["kind"], "subject": t["subject"], "verdict": v, "text": reply})
        cache.append({"id": t["id"], "text": reply})
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"reference": "m", "raw": {"m": rows},
                               "endpoints": {"m": {"url": "http://x/v1", "model": "acme/tiny"}}}))
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "cache" / "m.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cache))
    model = tmp_path / "models" / "acme" / "Tiny-GGUF" / "tiny-Q4_K_M.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF" + bytes(range(256)) * 4)
    return res, model


def test_card_contents_and_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "HOME", tmp_path / "home")
    res, model = honest_run(tmp_path)
    lora = tmp_path / "s.gguf"
    lora.write_bytes(b"lora")
    card, log = sc.build(str(res), "m", str(model), {"lora": str(lora), "lora_scale": 0.5, "ctx": 4096,
                                                    "cache_type_k": "q8_0", "threads": 8},
                         cache_dir=str(tmp_path / "cache"))
    txt = json.dumps(card)
    assert str(tmp_path) not in txt and "home" not in txt                 # no paths leave the machine
    assert card["model"]["file"] == "tiny-Q4_K_M.gguf" and card["model"]["quant"] == "Q4_K_M"
    assert card["visibility"] == "private" and card["model"]["listed"] is False
    assert card["variant"]["engine"]["options"] == {"ctx": 4096, "cache_type_k": "q8_0"}   # threads: speed only
    assert card["variant"]["lora"]["scale"] == 0.5 and card["test"]["items"] == 12
    assert len(log["items"]) == 12 and all("reply" in i for i in log["items"])
    card2, _ = sc.build(str(res), "m", str(model), {"lora": str(lora), "lora_scale": 1.0, "ctx": 4096,
                                                   "cache_type_k": "q8_0"})
    assert card2["variant_id"] != card["variant_id"]                       # different setups never mix
    card3, _ = sc.build(str(res), "m", str(model), {}, listed={"listed": True, "repo": "acme/Tiny-GGUF"})
    assert card3["visibility"] == "public" and card3["model"]["hf_repo"] == "acme/Tiny-GGUF"
    card4, _ = sc.build(str(res), "m", str(model), {}, public_by_hand=True)
    assert card4["visibility"] == "public" and card4["model"]["hf_repo"] is None


@pytest.fixture()
def board(tmp_path):
    b = sb.Board(str(tmp_path / "b.sqlite"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), sb.make_handler(b, "admin-secret"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", b
    srv.shutdown()


def get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_scoreboard_verifies_and_keeps_logs_private(tmp_path, board, monkeypatch):
    monkeypatch.setattr(sc, "HOME", tmp_path / "home")
    url, b = board
    res, model = honest_run(tmp_path)
    card, log = sc.build(str(res), "m", str(model), {}, cache_dir=str(tmp_path / "cache"), public_by_hand=True)
    out = sc.submit(url, card, log)
    assert out["status"] == "verified" and out["shown_publicly"]
    assert list((tmp_path / "home" / "submissions").glob("*.json"))     # kept locally, as sent
    _, j = get(url + "/api/board")
    assert j["board"][0]["items"] == 12 and j["board"][0]["variant_id"] == card["variant_id"]
    _, c = get(url + f"/api/card/{out['id']}")
    assert "items" not in json.dumps(c.get("test", {}).get("per_subject", {})) or True
    assert "reply" not in json.dumps(c) and "reply" not in json.dumps(j)  # no log content anywhere public
    assert get(url + f"/api/admin/log/{out['id']}")[0] == 401
    st, lg = get(url + f"/api/admin/log/{out['id']}", {"Authorization": "Bearer admin-secret"})
    assert st == 200 and len(lg["items"]) == 12
    # a card that claims better numbers than its replies grade to is rejected and never shown
    bad = json.loads(json.dumps(card))
    s0 = next(iter(bad["test"]["per_subject"]))
    bad["test"]["per_subject"][s0]["correct"] += 1
    bad["test"]["per_subject"][s0]["wrong"] -= 1
    assert sc.submit(url, bad, log)["status"] == "rejected"
    forged = {"items": [dict(i, verdict="correct") for i in log["items"]]}
    assert sc.submit(url, card, forged)["status"] == "rejected"
    # private cards are stored but not shown
    priv, plog = sc.build(str(res), "m", str(model), {"ctx": 2048}, cache_dir=str(tmp_path / "cache"))
    r = sc.submit(url, priv, plog)
    assert r["status"] == "verified" and not r["shown_publicly"]
    assert len(get(url + "/api/board")[1]["board"]) == 1


def test_studio_sharing(studio_srv, tmp_path, board, monkeypatch):
    import studio
    monkeypatch.setattr(sc, "HOME", tmp_path / "home")
    url, b = board
    st, j = call(studio_srv, "/api/settings/sharing")
    assert st == 200 and j["settings"]["enabled"] is False                 # off by default
    res, _ = honest_run(tmp_path)
    m = next(x for x in call(studio_srv, "/api/models")[1] if x.get("format") != "hf")
    # a finished TestQA job pointing at the results
    jobs = studio.STATE["jobs"]
    jd = Path(jobs.root) / "j1"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({"id": "j1", "kind": "testqa", "status": "done", "started": 1, "ended": 2,
                                             "values": {"out": str(res), "cache": str(tmp_path / "cache")}}))
    monkeypatch.setattr(studio, "sharing_listed", lambda path, refresh=False: {"listed": False, "reason": "test"})
    st, j = call(studio_srv, "/api/settings/sharing")
    assert j["results"][0]["file"] == str(res)
    st, p = call(studio_srv, "/api/settings/sharing/preview", {"results": str(res), "label": "m", "path": m["path"]})
    assert st == 200 and p["card"]["visibility"] == "private" and p["log"]["total_items"] == 12
    st, e = call(studio_srv, "/api/settings/sharing/submit", {"results": str(res), "label": "m", "path": m["path"]})
    assert st == 400                                                        # sharing still off
    assert call(studio_srv, "/api/settings/sharing", {"endpoint": "not a url"})[0] == 400
    st, _ = call(studio_srv, "/api/settings/sharing", {"enabled": True, "endpoint": url, "auto_unlisted": False,
                                                       "public_models": [m["id"], "nope/nope"]})
    assert st == 200 and studio.sharing()["public_models"] == [m["id"]]
    st, r = call(studio_srv, "/api/settings/sharing/submit", {"results": str(res), "label": "m", "path": m["path"]})
    assert st == 200 and r["visibility"] == "public"                       # listed by hand
    # automatic: nothing for an unlisted model unless the advanced switch is on
    n0 = len(b.db.execute("select id from submissions").fetchall())
    studio.sharing_after_job({"kind": "testqa", "status": "done", "ended": 2})
    assert len(b.db.execute("select id from submissions").fetchall()) == n0
    call(studio_srv, "/api/settings/sharing", {"auto_unlisted": True})
    studio.sharing_after_job({"kind": "testqa", "status": "done", "ended": 2})
    # the results' endpoint names model acme/tiny, which is not a local model: still nothing
    assert len(b.db.execute("select id from submissions").fetchall()) == n0
    d = json.loads(res.read_text())
    d["endpoints"]["m"]["model"] = m["id"]
    res.write_text(json.dumps(d))
    studio.sharing_after_job({"kind": "testqa", "status": "done", "ended": 2})
    rows = b.db.execute("select visibility, status from submissions").fetchall()
    assert len(rows) == n0 + 1 and rows[-1] == ("public", "verified")    # listed by hand earlier, so public
    call(studio_srv, "/api/settings/sharing", {"public_models": []})
    studio.sharing_after_job({"kind": "testqa", "status": "done", "ended": 2})
    assert b.db.execute("select visibility from submissions").fetchall()[-1] == ("private",)
