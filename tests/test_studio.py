"""Studio's OpenAI-compatible API, JIT loading, routing, TTL and chat storage
against a fake llama-server."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import model_stats  # noqa: E402
import studio  # noqa: E402

CODER = "coder-gguf/coder-7b-q4_k_m"
VLM = "vision-gguf/vlm-3b-q8_0"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def studio_srv(tmp_path):
    models = tmp_path / "models"
    (models / "acme" / "Coder-GGUF").mkdir(parents=True)
    (models / "acme" / "Vision-GGUF").mkdir(parents=True)
    (models / "acme" / "Coder-GGUF" / "coder-7b-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    (models / "acme" / "Vision-GGUF" / "vlm-3b-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    (models / "acme" / "Vision-GGUF" / "mmproj-vlm-3b-f16.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    studio.STATE.update({
        "models_dirs": [str(models)], "server_bin": str(ROOT / "tests" / "fake_llama_server.py"),
        "settings_path": str(tmp_path / "studio.json"), "chats_dir": str(tmp_path / "chats"),
        "token": None, "stats": model_stats.StatsStore(tmp_path / "stats"), "min_graded": 20,
        "min_subject": 5, "halluc_cost": 1.0, "cett": None, "idle_ttl": 0, "jit": True, "backend_port": free_port(), "tls": False,
        "jobs": studio.oc_jobs.JobRunner(tmp_path / "jobs", 1), "jobs_off": None,
        "rag": None, "rag_dir": str(tmp_path / "rag"), "rag_embed": None, "rag_embed_gguf": None,
        "devices": studio.oc_pairing.DeviceRegistry(tmp_path / "devices.json"),
        "links_path": str(tmp_path / "links.json"), "fingerprint": None,
        "review_dir": str(tmp_path / "review"), "review": None,
        "tool_rules_path": str(tmp_path / "tool-rules.json"), "subject_models": {},
        "config_path": str(tmp_path / "config.json"), "workspace_name": "default", "work_log": None})
    srv = studio.BoundedThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    studio.stop_server()


def test_studio_http_server_rejects_connections_above_worker_limit():
    entered = threading.Event()
    release = threading.Event()

    class SlowHandler(studio.Handler):
        def do_GET(self):
            entered.set()
            release.wait(5)
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = studio.BoundedThreadingHTTPServer(("127.0.0.1", 0), SlowHandler, max_workers=1)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    first = socket.create_connection(("127.0.0.1", srv.server_port), timeout=3)
    second = None
    try:
        first.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        assert entered.wait(3)
        second = socket.create_connection(("127.0.0.1", srv.server_port), timeout=3)
        second.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        response = second.recv(1024)
        assert b"503 Service Unavailable" in response
    finally:
        release.set()
        # Allow the active handler to finish before closing its client socket.
        try:
            first.settimeout(3)
            while first.recv(4096):
                pass
        except (OSError, socket.timeout):
            pass
        first.close()
        if second is not None:
            second.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=3)


def test_studio_body_reader_rejects_chunked_encoding():
    class Headers(dict):
        def get_all(self, key, default=None):
            value = self.get(key)
            return default if value is None else [value]

    class Dummy:
        headers = Headers({"Transfer-Encoding": "chunked"})
    with pytest.raises(ValueError, match="Transfer-Encoding"):
        studio.Handler._read(Dummy())


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_models_skip_mmproj_and_pair_it(studio_srv):
    _, ms = call(studio_srv, "/v1/models")
    ids = [m["id"] for m in ms["data"]]
    assert "auto" not in ids                      # nothing has stats yet
    assert CODER in ids and VLM in ids
    assert not any("mmproj" in i for i in ids)
    assert next(m for m in ms["data"] if m["id"] == VLM)["vision"] is True


def seed(base, model, subject, correct, wrong, abstained=0):
    recs = ([{"subject": subject, "verdict": "correct"}] * correct + [{"subject": subject, "verdict": "wrong"}] * wrong
            + [{"subject": subject, "verdict": "abstained"}] * abstained)
    code, r = call(base, "/api/stats/ingest", {"model": model, "records": recs})
    assert code == 200
    return r


def test_jit_load_swap_and_vision(studio_srv):
    code, r = call(studio_srv, "/v1/chat/completions",
                   {"model": "coder-gguf/coder-7b-q4_k_m", "messages": [{"role": "user", "content": "hi"}]})
    assert code == 200 and "model=coder-gguf/coder-7b-q4_k_m" in r["choices"][0]["message"]["content"]
    img = [{"type": "text", "text": "what is this"},
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                                        "messages": [{"role": "user", "content": img}]})
    assert code == 400 and "mmproj" in r["error"]["message"]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "vision-gguf/vlm-3b-q8_0",
                                                        "messages": [{"role": "user", "content": img}]})
    text = r["choices"][0]["message"]["content"]
    assert code == 200 and "mmproj=yes" in text and "image=yes" in text
    _, st = call(studio_srv, "/api/status")
    assert st["id"] == "vision-gguf/vlm-3b-q8_0" and st["vision"]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "nope", "messages": []})
    assert code == 404


def test_auto_refuses_without_stats_but_manual_still_works(studio_srv):
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": "Write a Python function"}]})
    assert code == 409 and "performance stats" in r["error"]["message"]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": CODER, "messages": [
        {"role": "user", "content": "hi"}]})
    assert code == 200


def test_auto_routes_on_stats(studio_srv):
    assert seed(studio_srv, CODER, "code", 40, 5)["recorded"] == 45
    seed(studio_srv, CODER, "writing", 2, 18)
    seed(studio_srv, VLM, "writing", 18, 2)
    seed(studio_srv, VLM, "vision", 20, 2)
    _, ms = call(studio_srv, "/v1/models")
    assert ms["data"][0]["id"] == "auto"
    _, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": "Write a Python function that parses JSON and fix this bug"}]})
    assert f"model={CODER}" in r["choices"][0]["message"]["content"]
    _, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": "Write a short poem about autumn, in the tone of a letter"}]})
    assert f"model={VLM}" in r["choices"][0]["message"]["content"]
    _, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Fix this Python bug, see the screenshot"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]})
    assert f"model={VLM}" in r["choices"][0]["message"]["content"]   # only vision models are candidates


def test_stats_surface_and_warning_reason(studio_srv):
    seed(studio_srv, CODER, "code", 10, 2)
    _, models = call(studio_srv, "/api/models")
    by = {m["id"]: m["stats"] for m in models}
    assert not by[CODER]["eligible"] and "12 graded" in by[CODER]["why_not"]
    assert by[VLM]["why_not"].startswith("No performance stats")
    code, r = call(studio_srv, "/api/stats/ingest", {"model": "nope", "records": []})
    assert code == 404


def test_live_and_activation_recording(studio_srv, monkeypatch):
    import hscore

    class FakeScorer:
        def __init__(self, *a, **k):
            pass

        def score(self, messages, text):
            return {"score": 1.5, "prob": 0.8, "n_tokens": 7}
    monkeypatch.setattr(hscore, "HScorer", FakeScorer)
    studio.STATE["cett"] = "/bin/true"
    threading.Thread(target=studio.score_worker, daemon=True).start()
    settings = {studio._key(m["path"]): {"classifier": "x.npz"} for m in studio.scan_models()}
    studio.save_settings(settings)
    for _ in range(3):
        call(studio_srv, "/v1/chat/completions", {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
    deadline = time.time() + 10
    while time.time() < deadline:
        _, st = call(studio_srv, "/api/stats")
        if st["models"][CODER]["activation"]["n"] >= 1:
            break
        time.sleep(0.2)
    s = st["models"][CODER]
    studio.STATE["cett"] = None
    assert s["live"]["n"] == 3 and s["live"]["abstention_rate"] == 0.0
    assert s["activation"]["n"] >= 1 and s["activation"]["mean"] == 1.5


def test_streaming_passthrough(studio_srv):
    req = urllib.request.Request(studio_srv + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"model": "coder-gguf/coder-7b-q4_k_m", "stream": True,
                                                  "messages": [{"role": "user", "content": "x"}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode()
    assert body.count("data:") >= 3 and "[DONE]" in body and "predicted_per_second" in body


def test_idle_ttl_unloads(studio_srv):
    call(studio_srv, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                              "messages": [{"role": "user", "content": "x"}]})
    assert studio.server_running()
    studio.STATE["idle_ttl"] = 1
    studio.ACTIVITY["last"] = time.time() - 10
    threading.Thread(target=studio.idle_reaper, daemon=True).start()
    deadline = time.time() + 15
    while studio.server_running() and time.time() < deadline:
        time.sleep(0.5)
    studio.STATE["idle_ttl"] = 0
    assert not studio.server_running()


def test_chat_history_roundtrip(studio_srv):
    _, saved = call(studio_srv, "/api/chats", {"title": "t1", "messages": [{"role": "user", "content": "hi"}]})
    _, lst = call(studio_srv, "/api/chats")
    assert [c["id"] for c in lst] == [saved["id"]] and lst[0]["n"] == 1
    _, got = call(studio_srv, f"/api/chats/{saved['id']}")
    assert got["messages"][0]["content"] == "hi"
    code, _ = call(studio_srv, "/api/chats/../../etc")
    assert code == 404
    call(studio_srv, "/api/chats/delete", {"id": saved["id"]})
    assert call(studio_srv, "/api/chats")[1] == []


def test_refuses_open_lan_bind():
    assert studio.main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_jobs_run_log_and_reject_injection(studio_srv, tmp_path):
    code, specs = call(studio_srv, "/api/jobs/specs")
    assert code == 200 and {"testqa", "deficits", "finetune", "swe_train"} <= set(specs)
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"resolved_ids": ["a"], "unresolved_ids": ["b"], "empty_patch_ids": []}))
    code, job = call(studio_srv, "/api/jobs", {"kind": "swe_import", "values": {"report": str(rep), "model": "m"}})
    assert code == 200 and job["status"] == "running"
    for _ in range(100):
        _, js = call(studio_srv, "/api/jobs")
        if js[0]["status"] != "running":
            break
        time.sleep(0.1)
    assert js[0]["status"] == "done"
    _, log = call(studio_srv, f"/api/jobs/{job['id']}/log")
    assert "1/2 resolved" in log["log"]
    code, err = call(studio_srv, "/api/jobs", {"kind": "swe_import", "values": {"report": "--help", "model": "m"}})
    assert code == 400 and "may not start" in err["error"]
    code, _ = call(studio_srv, "/api/jobs", {"kind": "rm", "values": {}})
    assert code == 400
    # a cross-site form post cannot start a job
    req = urllib.request.Request(studio_srv + "/api/jobs", data=b"kind=swe_import", method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 415


def test_jobs_cancel_and_off_on_network_bind(studio_srv, tmp_path, monkeypatch):
    runner = studio.STATE["jobs"]
    monkeypatch.setitem(studio.oc_jobs.SPECS, "sleep", {"title": "sleep", "group": "t", "help": "",
                        "argv": ["-c", "import time; time.sleep(30)"], "fields": []})
    code, job = call(studio_srv, "/api/jobs", {"kind": "sleep", "values": {}})
    assert code == 200
    code, err = call(studio_srv, "/api/jobs", {"kind": "sleep", "values": {}})
    assert code == 429                                        # --max-jobs 1
    call(studio_srv, "/api/jobs/cancel", {"id": job["id"]})
    for _ in range(50):
        if runner.get(job["id"])["ended"]:
            break
        time.sleep(0.1)
    assert runner.get(job["id"])["status"] == "cancelled"
    monkeypatch.setitem(studio.STATE, "jobs", None)
    monkeypatch.setitem(studio.STATE, "jobs_off", "jobs are off on a network bind")
    code, err = call(studio_srv, "/api/jobs")
    assert code == 403 and "network bind" in err["error"]


def sse_chat(base, body):
    req = urllib.request.Request(base + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                events.append(json.loads(line[5:]))
    return events


def _upload(base, coll, name, text):
    import base64
    return call(base, "/api/rag/upload", {"collection": coll, "filename": name,
                                          "data": base64.b64encode(text.encode()).decode()})


def test_rag_collections_and_chat_citations(studio_srv):
    code, _ = call(studio_srv, "/api/rag/create", {"collection": "kb"})
    assert code == 200
    code, r = _upload(studio_srv, "kb", "falcon.md", "The Falcon 9 booster lands on a drone ship.\n\n"
                                                        "Paris is the capital of France.")
    assert code == 200 and r["chunks"] >= 1
    _upload(studio_srv, "kb", "cats.txt", "Cats sleep for most of the day.")
    _, info = call(studio_srv, "/api/rag")
    (kb,) = info["collections"]
    assert kb["docs"] == 2 and not kb["dense"]
    code, hits = call(studio_srv, "/api/rag/search", {"collection": "kb", "query": "where does the booster land"})
    assert hits[0]["source"] == "falcon.md"
    ev = sse_chat(studio_srv, {"model": CODER, "rag": {"collection": "kb", "k": 2},
                               "messages": [{"role": "user", "content": "Where does the Falcon booster land?"}]})
    assert ev[0]["sources"][0]["source"] == "falcon.md"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if "choices" in e)
    assert "ctx=yes" in text                              # the passages reached the model
    ev = sse_chat(studio_srv, {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
    assert "sources" not in ev[0]
    # bad names cannot escape the collection root
    code, err = call(studio_srv, "/api/rag/create", {"collection": "../etc"})
    assert code == 400
    _, docs = call(studio_srv, "/api/rag/docs?c=kb")
    call(studio_srv, "/api/rag/delete", {"collection": "kb", "doc": next(d["doc"] for d in docs if d["source"] == "cats.txt")})
    _, info = call(studio_srv, "/api/rag")
    assert info["collections"][0]["docs"] == 1
    call(studio_srv, "/api/rag/delete", {"collection": "kb"})
    assert call(studio_srv, "/api/rag")[1]["collections"] == []


def test_rag_dense_with_embedding_sidecar(studio_srv, monkeypatch):
    monkeypatch.setitem(studio.STATE, "rag_embed_gguf", "/models/embed.gguf")   # the fake server ignores -m
    try:
        _upload(studio_srv, "dense", "a.txt", "alpha bravo charlie")
        _upload(studio_srv, "dense", "b.txt", "rocket booster landing")
        _, info = call(studio_srv, "/api/rag")
        assert info["embedder"] and info["collections"][0]["dense"]
        _, hits = call(studio_srv, "/api/rag/search", {"collection": "dense", "query": "booster landing", "k": 1})
        assert hits[0]["source"] == "b.txt"
    finally:
        p = studio.EMBED["proc"]
        if p is not None:
            p.terminate()
            studio.EMBED["proc"] = None


def _mcp_cfg(tmp_path, auto):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"calc": {"command": sys.executable,
                                                        "args": [str(ROOT / "tests" / "fake_mcp_server.py")],
                                                        "autoApprove": auto}}}))
    return str(cfg)


@pytest.mark.parametrize("auto,allow", [(["add"], None), ([], True), ([], False)])
def test_mcp_tool_calls_with_approval(studio_srv, tmp_path, monkeypatch, auto, allow):
    pytest.importorskip("mcp")
    monkeypatch.setitem(studio.STATE, "mcp_config", _mcp_cfg(tmp_path, auto))
    monkeypatch.setitem(studio.STATE, "mcp", None)
    try:
        _, st = call(studio_srv, "/api/mcp")
        (srv,) = st["servers"]
        assert srv["connected"] and {t["name"] for t in srv["tools"]} == {"add", "echo", "boom"}
        approver = None
        if allow is not None:
            def approve():
                for _ in range(200):
                    with studio.APPROVALS_LOCK:
                        keys = list(studio.APPROVALS)
                    if keys:
                        call(studio_srv, "/api/tools/approve", {"key": keys[0], "allow": allow})
                        return
                    time.sleep(0.05)
            approver = threading.Thread(target=approve)
            approver.start()
        ev = sse_chat(studio_srv, {"model": CODER, "tools": True,
                                   "messages": [{"role": "user", "content": "what is 2+40?"}]})
        if approver:
            approver.join()
        (tc,) = [e["tool_call"] for e in ev if "tool_call" in e]
        (tr,) = [e["tool_result"] for e in ev if "tool_result" in e]
        assert tc["name"] == "calc__add" and tc["arguments"] == {"a": 2, "b": 40}
        assert tc["needs_approval"] is (allow is not None)
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if "choices" in e)
        if allow is False:
            assert not tr["ok"] and "declined" in tr["text"] and "declined" in text
        else:
            assert tr["ok"] and tr["text"] == "42" and text == "tool said: 42"
    finally:
        if studio.STATE.get("mcp"):
            studio.STATE["mcp"].close()



def test_mcp_tool_rules_deny_and_always(studio_srv, tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    import tool_rules as tr
    monkeypatch.setitem(studio.STATE, "mcp_config", _mcp_cfg(tmp_path, []))
    monkeypatch.setitem(studio.STATE, "mcp", None)
    rules = tr.Rules(studio.STATE["tool_rules_path"])
    try:
        call(studio_srv, "/api/mcp")
        # A deny rule blocks the call without asking, and says which rule.
        rules.add({"id": "noadd", "action": "deny", "server": "calc", "tool": "add",
                   "args": [{"path": "b", "regex": "^40$"}], "note": "no forty"})
        ev = sse_chat(studio_srv, {"model": CODER, "tools": True, "messages": [{"role": "user", "content": "2+40?"}]})
        (tc,) = [e["tool_call"] for e in ev if "tool_call" in e]
        (res,) = [e["tool_result"] for e in ev if "tool_result" in e]
        assert not tc["needs_approval"] and tc["decision"]["action"] == "deny"
        assert not res["ok"] and "rule noadd" in res["text"]
        rules.remove("noadd")

        # "always allow this tool" on the approval card: allowed now, and a rule from then on.
        def approve_always():
            for _ in range(200):
                with studio.APPROVALS_LOCK:
                    keys = list(studio.APPROVALS)
                if keys:
                    code, r = call(studio_srv, "/api/tools/approve", {"key": keys[0], "allow": True, "remember": "tool"})
                    assert code == 200 and r["rule"]["tool"] == "add"
                    return
                time.sleep(0.05)
        t = threading.Thread(target=approve_always)
        t.start()
        ev = sse_chat(studio_srv, {"model": CODER, "tools": True, "messages": [{"role": "user", "content": "2+40?"}]})
        t.join()
        assert [e["tool_result"]["ok"] for e in ev if "tool_result" in e] == [True]
        ev = sse_chat(studio_srv, {"model": CODER, "tools": True, "messages": [{"role": "user", "content": "2+40?"}]})
        (tc,) = [e["tool_call"] for e in ev if "tool_call" in e]
        assert not tc["needs_approval"] and tc["decision"]["by"] == "rule"
        code, st = call(studio_srv, "/api/settings")
        assert code == 200 and {t["tool"] for t in st["tools"]} == {"add", "echo", "boom"}
        assert st["tool_rules"]["rules"][0]["hits"] >= 1
        code, d = call(studio_srv, "/api/settings/tool-rules/test", {"server": "calc", "tool": "boom", "args": "{}"})
        assert code == 200 and d["action"] == "ask"
        assert call(studio_srv, "/api/settings/tool-rules", {"rules": {"default": "maybe"}})[0] == 400
    finally:
        if studio.STATE.get("mcp"):
            studio.STATE["mcp"].close()


def test_subject_default_beats_measured_best(studio_srv):
    code, st = call(studio_srv, "/api/settings")
    assert code == 200 and "code" in st["subjects"] and st["subjects"]["code"]["chosen"] is None
    assert call(studio_srv, "/api/settings/subjects", {"subject_models": {"code": "nope/nope"}})[0] == 400
    code, r = call(studio_srv, "/api/settings/subjects", {"subject_models": {"code": VLM}})
    assert code == 200 and r["subjects"]["code"]["uses"] == VLM
    cfg = json.loads(Path(studio.STATE["config_path"]).read_text())
    assert cfg["studio"]["subject_models"] == {"code": VLM}
    # Not enough graded results for a measured pick: the setting still routes code prompts.
    target, pick = studio.auto_pick({"messages": [{"role": "user", "content":
                                     "Write a Python function that parses a CSV file and returns rows"}]},
                                    studio.scan_models())
    assert target["id"] == VLM and pick["by"] == "setting" and "your default for code" in pick["reason"]
    target, pick = studio.auto_pick({"messages": [{"role": "user", "content": "Who was the first emperor of Rome?"}]},
                                    studio.scan_models())
    assert pick.get("by") == "measured"


def test_workspaces_api_and_setup_paths(studio_srv, tmp_path):
    code, ws = call(studio_srv, "/api/workspaces")
    assert code == 200 and ws["active"] == "default" and ws["config"] == studio.STATE["config_path"]
    models = tmp_path / "ext-models"
    models.mkdir()
    code, r = call(studio_srv, "/api/workspaces/save", {"name": "ext", "values": {"models_dirs": [str(models)],
                                                                                  "env": {"VK_DRIVER_FILES": "/x.json"}}})
    assert code == 200 and any(w["name"] == "ext" for w in r["workspaces"])
    assert call(studio_srv, "/api/workspaces/save", {"name": "ext", "values": {"root": str(tmp_path / "nope")}})[0] == 400
    assert call(studio_srv, "/api/workspaces/save", {"name": "ext", "values": {"env": {"PATH": "/evil"}}})[0] == 400
    assert call(studio_srv, "/api/workspaces/save", {"name": "../x", "values": {}})[0] == 400
    code, r = call(studio_srv, "/api/workspaces/use", {"name": "ext"})
    assert code == 200 and r["active"] == "ext" and r["restart_needed"]
    assert call(studio_srv, "/api/workspaces/delete", {"name": "default"})[0] == 409     # Studio runs in it
    call(studio_srv, "/api/workspaces/use", {"name": "default"})
    assert call(studio_srv, "/api/workspaces/delete", {"name": "ext"})[0] == 200
    # Setup's models folder lands in the running workspace, not the studio section.
    code, r = call(studio_srv, "/api/setup", {"values": {"models_dir": [str(models)], "idle_ttl": 5}})
    assert code == 200, r
    cfg = json.loads(Path(studio.STATE["config_path"]).read_text())
    assert cfg["workspaces"]["default"]["models_dirs"] == [str(models)]
    assert "models_dir" not in cfg["studio"] and cfg["studio"]["idle_ttl"] == 5
    code, page = call(studio_srv, "/settings")
    assert code == 200 and "Studio · Settings" in page


@pytest.fixture()
def remote_studio(tmp_path):
    """A second Studio in its own process: token + self-signed TLS, one model."""
    import subprocess
    pytest.importorskip("cryptography")
    d = tmp_path / "remote"
    (d / "models" / "acme" / "Big-GGUF").mkdir(parents=True)
    (d / "models" / "acme" / "Big-GGUF" / "big-70b-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    sec = __import__("oc_security")
    token = sec.generate_token()
    sec.write_secret_file(d / "token", token)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "oc_security.py"), "selfsigned", "--host", "127.0.0.1",
                    "--cert", str(d / "c.pem"), "--key", str(d / "k.pem")], check=True, capture_output=True)
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "viz" / "studio.py"), "--models-dir", str(d / "models"),
                             "--server", str(ROOT / "tests" / "fake_llama_server.py"), "--port", str(port),
                             "--backend-port", str(free_port()), "--token-file", str(d / "token"),
                             "--tls-cert", str(d / "c.pem"), "--tls-key", str(d / "k.pem"),
                             "--devices", str(d / "devices.json"), "--links", str(d / "links.json"),
                             "--mcp-config", str(d / "none.json"), "--rag-dir", str(d / "rag"),
                             "--jobs-dir", str(d / "jobs"), "--settings", str(d / "s.json"),
                             "--chats-dir", str(d / "chats"), "--stats-dir", str(d / "stats")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"https://127.0.0.1:{port}"
    import oc_pairing
    fp = oc_pairing.cert_fingerprint(str(d / "c.pem"))
    for _ in range(100):
        try:
            oc_pairing.request(base, "GET", "/api/status", token=token, fingerprint=fp, timeout=2)
            break
        except Exception:
            time.sleep(0.1)
    yield base, token, fp
    proc.terminate()
    proc.wait(10)


def test_pairing_and_linked_host(studio_srv, remote_studio):
    import oc_pairing
    base, token, fp = remote_studio
    # the owner creates a link; it carries the certificate fingerprint
    r = oc_pairing.request(base, "POST", "/api/pair/start", {}, token=token, fingerprint=fp)
    assert r["fingerprint"] == fp and f"fp={fp}" in r["link"]
    link = r["link"].replace(r["link"].split("/pair")[0], base)
    # a wrong fingerprint is refused before the code is ever sent
    bad = link.replace(fp, "0" * 64)
    code, err = call(studio_srv, "/api/links/add", {"link": bad, "name": "gpu"})
    assert code == 400 and "fingerprint" in err["error"]
    code, r2 = call(studio_srv, "/api/links/add", {"link": link, "name": "gpu"})
    assert code == 200 and r2["models"] == ["gpu:big-gguf/big-70b-q4_k_m"], r2
    _, ms = call(studio_srv, "/v1/models")
    remote_ids = [m["id"] for m in ms["data"] if m.get("owned_by") == "link:gpu"]
    assert len(remote_ids) == 1 and remote_ids[0].startswith("gpu:")
    # chat through the link: the remote Studio loads and answers
    ev = sse_chat(studio_srv, {"model": remote_ids[0], "messages": [{"role": "user", "content": "hi"}]})
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if "choices" in e)
    assert text.startswith("model=") and "big" in text
    code, out = call(studio_srv, "/v1/chat/completions", {"model": remote_ids[0],
                                                          "messages": [{"role": "user", "content": "hi"}]})
    assert code == 200 and "big" in out["choices"][0]["message"]["content"]
    # the code was single use
    with pytest.raises(RuntimeError, match="403"):
        oc_pairing.claim(link, "again")
    # the device token works for models but not for owner actions
    links = json.loads(open(studio.STATE["links_path"]).read())["links"]
    dev_tok = links[0]["token"]
    assert oc_pairing.request(base, "GET", "/v1/models", token=dev_tok, fingerprint=fp)["data"]
    with pytest.raises(RuntimeError, match="403"):
        oc_pairing.request(base, "POST", "/api/pair/start", {}, token=dev_tok, fingerprint=fp)
    for path, body in [("/api/jobs", {"kind": "swe_import", "values": {}}), ("/api/load", {"id": "x"}),
                       ("/api/download", {"repo": "a/b"}), ("/api/mcp/reload", {})]:
        with pytest.raises(RuntimeError, match="403"):
            oc_pairing.request(base, "POST", path, body, token=dev_tok, fingerprint=fp)
    with pytest.raises(RuntimeError, match="403"):
        oc_pairing.request(base, "GET", "/api/jobs", token=dev_tok, fingerprint=fp)
    # revoking the device cuts the link off
    devs = oc_pairing.request(base, "GET", "/api/devices", token=token, fingerprint=fp)["devices"]
    assert [d["name"] for d in devs] == ["studio"]
    oc_pairing.request(base, "POST", "/api/devices/revoke", {"id": devs[0]["id"]}, token=token, fingerprint=fp)
    with pytest.raises(RuntimeError, match="401"):
        oc_pairing.request(base, "GET", "/v1/models", token=dev_tok, fingerprint=fp)
    # the token file on this side is private
    assert oct(os.stat(studio.STATE["links_path"]).st_mode & 0o777) == "0o600"


def test_device_registry_codes_expire_and_are_hashed(tmp_path, monkeypatch):
    import oc_pairing
    reg = oc_pairing.DeviceRegistry(tmp_path / "d.json")
    code, _, _ = reg.new_code()
    assert reg.claim("WRONGCODE000", "x") is None
    r = reg.claim(code.lower()[:4] + "-" + code[4:], "laptop")          # dashes and case do not matter
    assert r and reg.check(r["token"])["name"] == "laptop"
    assert r["token"] not in (tmp_path / "d.json").read_text()          # only the hash is stored
    code2, _, _ = reg.new_code()
    monkeypatch.setattr(oc_pairing.time, "time", lambda: 10 ** 12)      # far future
    assert reg.claim(code2, "late") is None


def test_multi_gpu_flags_and_vram_fit(studio_srv, monkeypatch):
    seen = {}

    class P:
        def __init__(self, cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw.get("env")
        def poll(self):
            return None
        def terminate(self):
            pass
        def wait(self, timeout=None):
            return 0
        kill = terminate
    monkeypatch.setattr(studio.subprocess, "Popen", P)
    monkeypatch.setattr(studio.time, "sleep", lambda s: None)
    monkeypatch.setattr(studio, "wait_healthy", lambda port, proc: (True, "ready"))
    monkeypatch.setattr(studio, "port_busy", lambda port: False)
    # the engine's --help probe would run through the fake Popen too: answer it here, as other tests' cache did
    import engine_options
    monkeypatch.setattr(engine_options, "probe", lambda binary, engine: set())
    m = {"path": "/m.gguf", "id": "m", "name": "m.gguf"}
    ok, _ = studio.start_server(m, {**studio.DEFAULTS, "gpus": "0,1,2,3", "split_mode": "layer",
                                    "tensor_split": "1,1,1,1", "main_gpu": 0}, 9999)
    assert ok and seen["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    c = seen["cmd"]
    assert c[c.index("-sm") + 1] == "layer" and c[c.index("-ts") + 1] == "1,1,1,1" and c[c.index("-mg") + 1] == "0"
    for bad in ({"gpus": "0;rm"}, {"split_mode": "fast"}, {"tensor_split": "1,a"}):
        ok, why = studio.start_server(m, {**studio.DEFAULTS, **bad}, 9999)
        assert not ok, bad
    studio.PROC.update({"proc": None, "model": None, "port": None})
    import cuda_info
    m10 = cuda_info.parse_smi("".join(f"{i}, Tesla M10, 5.0, 8192, 8000, 580.95.05, b{i}\n" for i in range(4)))
    monkeypatch.setattr(studio, "nvidia_gpus", lambda: m10)

    class FewRam:
        ram, ram_available = 16 * 2**30, 12 * 2**30
    monkeypatch.setattr(studio, "Host", FewRam)
    f20 = studio.fit_estimate(20 * 2**30)
    assert f20["ok"] is True and f20["level"] == "gpu" and "4 GPUs" in f20["note"]
    f35 = studio.fit_estimate(35 * 2**30)                   # 40 GiB needed: 31 on the GPUs, the rest in RAM
    assert f35["ok"] is True and f35["level"] == "partial"
    assert studio.fit_estimate(60 * 2**30)["level"] == "no"
    code, adv = call(studio_srv, "/api/gpus")
    assert code == 200 and adv["count"] == 4 and adv["multi_gpu"]["llama_server"] == "-sm layer -ts 1,1,1,1"


def test_pairing_policy_persistent_and_temporary(tmp_path, monkeypatch):
    import json as _json
    import oc_pairing
    clock = [1_000_000.0]
    monkeypatch.setattr(oc_pairing.time, "time", lambda: clock[0])
    pol = oc_pairing.PairingPolicy(code_ttl="2m", presets=["1h", "8h", "7d"], max_ttl="30d")
    reg = oc_pairing.DeviceRegistry(tmp_path / "d.json", pol)
    # persistent (the default) and temporary pairings side by side
    c1, code_exp, acc = reg.new_code()
    assert acc is None and code_exp == clock[0] + 120
    keep = reg.claim(c1, "desktop")
    c2, _, acc = reg.new_code(persistent=False, ttl="8h")
    assert acc == 8 * 3600
    temp = reg.claim(c2, "phone")
    assert keep["expires"] is None and temp["expires"] == clock[0] + 8 * 3600
    clock[0] += 8 * 3600 - 1
    assert reg.check(temp["token"]) and reg.check(keep["token"])
    clock[0] += 2
    assert reg.check(temp["token"]) is None                  # temporary access ran out
    assert reg.check(keep["token"])                          # persistent access did not
    rows = {d["name"]: d for d in reg.list()}
    assert rows["phone"]["expired"] and rows["desktop"]["persistent"]
    # the owner renews, extends or makes persistent; the token is unchanged
    reg.update(temp["device_id"], persistent=False, ttl="1h")
    assert reg.check(temp["token"])
    reg.update(temp["device_id"], persistent=True)
    clock[0] += 400 * 86400
    assert reg.check(temp["token"])
    # host limits
    for kw in ({"persistent": False, "ttl": "31d"}, {"persistent": False, "ttl": "10s"}):
        try:
            reg.new_code(**kw)
            raise AssertionError(kw)
        except ValueError:
            pass
    strict = oc_pairing.DeviceRegistry(tmp_path / "s.json", oc_pairing.PairingPolicy(allow_persistent=False))
    try:
        strict.new_code(persistent=True)
        raise AssertionError("persistent allowed")
    except ValueError as e:
        assert "temporary" in str(e)
    assert strict.policy.describe()["default"] == "30d"      # default falls back to a temporary preset
    _, _, acc = strict.new_code()
    assert acc == 30 * 86400
    # pairing codes expire on the host's schedule
    c3, _, _ = reg.new_code(persistent=False, ttl="1h")
    clock[0] += 121
    assert reg.claim(c3, "late") is None
    # long-expired devices are pruned on the next pairing
    c4, _, _ = reg.new_code(persistent=False, ttl="1h")
    gone = reg.claim(c4, "short")
    clock[0] += 3600 + oc_pairing.EXPIRED_KEEP + 1
    reg.claim(reg.new_code()[0], "next")
    assert "short" not in {d["name"] for d in reg.list()}
    # device files from before expiry existed load as persistent
    (tmp_path / "old.json").write_text(_json.dumps({"devices": [{"id": "a", "name": "old", "token_sha256": "x",
                                                                 "created": 1, "last_seen": None}]}))
    assert oc_pairing.DeviceRegistry(tmp_path / "old.json").list()[0]["persistent"]
    assert oc_pairing.parse_duration("90m") == 5400 and oc_pairing.fmt_duration(7 * 86400) == "7d"


def test_temporary_pairing_end_to_end(remote_studio):
    import oc_pairing
    base, token, fp = remote_studio
    r = oc_pairing.request(base, "POST", "/api/pair/start", {"persistent": False, "ttl": "2h"},
                           token=token, fingerprint=fp)
    assert r["access_seconds"] == 7200 and not r["persistent"] and "&a=2h" in r["link"]
    code = oc_pairing.parse_link(r["link"].replace(r["link"].split("/pair")[0], base))[1]
    resp = oc_pairing.request(base, "POST", "/api/pair/claim", {"code": code, "name": "tablet", "cookie": True},
                              fingerprint=fp, stream=True)
    cookie = resp.getheader("Set-Cookie")
    claimed = json.loads(resp.read())
    assert 7100 < int(cookie.split("Max-Age=")[1].split(";")[0]) <= 7200      # the cookie ends with the access
    assert claimed["expires"] and oc_pairing.request(base, "GET", "/v1/models", token=claimed["token"],
                                                     fingerprint=fp)["data"]
    devs = oc_pairing.request(base, "GET", "/api/devices", token=token, fingerprint=fp)
    (dev,) = [d for d in devs["devices"] if d["name"] == "tablet"]
    assert not dev["persistent"] and devs["policy"]["allow_persistent"]
    up = oc_pairing.request(base, "POST", "/api/devices/update", {"id": dev["id"], "persistent": True},
                            token=token, fingerprint=fp)
    assert up["persistent"] and up["expires"] is None
    with pytest.raises(RuntimeError, match="400"):          # beyond the host's --pair-max (90d)
        oc_pairing.request(base, "POST", "/api/pair/start", {"persistent": False, "ttl": "400d"},
                           token=token, fingerprint=fp)
    with pytest.raises(RuntimeError, match="403"):          # a device cannot change its own access
        oc_pairing.request(base, "POST", "/api/devices/update", {"id": dev["id"], "persistent": True},
                           token=claimed["token"], fingerprint=fp)


def test_expired_link_is_reported_not_called(studio_srv):
    import json as _json
    studio.LINKS["cache"].clear()
    with open(studio.STATE["links_path"], "w") as f:
        _json.dump({"links": [{"name": "old", "url": "https://192.0.2.1:7870", "token": "t", "fingerprint": "",
                               "expires": time.time() - 60}]}, f)
    assert studio.link_models(refresh=True) == []          # no request to a host that would refuse it
    _, d = call(studio_srv, "/api/devices")
    (ln,) = d["links"]
    assert "expired" in ln["error"] and ln["expires"]


class _CharTok:
    """One token per character, as tests/fake_cett_dump.py tokenizes."""
    def render_chat(self, messages, add_generation_prompt=True):
        return "<u>" + messages[-1]["content"] + "</u><a>"

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def _fake_scorer(tmp_path, max_frames=None):
    import numpy as np
    import hscore
    n_layers, n_ff = 3, 4
    coef = np.zeros(n_layers * n_ff, dtype=np.float32)
    coef[5] = 1.0                                     # layer 1, neuron 1
    np.savez(tmp_path / "clf.npz", coef=coef, intercept=-5.0, n_layers=n_layers, n_neurons=n_ff)
    sc = hscore.HScorer(str(ROOT / "tests" / "fake_cett_dump.py"), "model.gguf", str(tmp_path / "clf.npz"),
                        tokenizer=_CharTok())
    sc._norms = np.full((n_layers, n_ff), 2.0, dtype=np.float32)
    return sc


def test_hscore_trace_scores_each_token(tmp_path):
    import numpy as np
    sc = _fake_scorer(tmp_path)
    prompt, reply = "<u>hi</u><a>", "abcdef"
    r = sc.trace([{"role": "user", "content": "hi"}], reply)
    t = np.arange(len(prompt), len(prompt) + len(reply))
    want = 2.0 * (1 + (t + 1 + 1) % 3) - 5.0          # per token: layer 1, neuron 1
    assert r["pieces"] == list(reply) and r["stride"] == 1
    assert np.allclose(r["scores"], want, atol=1e-2)
    assert r["frames"].shape[0] == len(reply)
    assert r["h_cells"] == [[1, 1]]                    # the one positive weight
    # A long reply is strided so the trace stays a bounded size.
    r2 = sc.trace([{"role": "user", "content": "hi"}], reply, max_frames=2)
    assert r2["stride"] == 3 and r2["pieces"] == ["abc", "def"] and len(r2["scores"]) == 2


def test_check_reply_saves_trace_and_serves_3d_view(studio_srv, tmp_path):
    sc = _fake_scorer(tmp_path)
    m = studio.find_model(CODER)
    st = studio.load_settings()
    st.setdefault(studio._key(m["path"]), {})["classifier"] = str(tmp_path / "clf.npz")
    studio.save_settings(st)
    studio.STATE.update(cett=sc.binary, traces_dir=str(tmp_path / "traces"))
    studio._SCORERS[(m["path"], str(tmp_path / "clf.npz"))] = sc
    try:
        code, r = call(studio_srv, "/api/trace", {"model": CODER, "text": "abcdef",
                                                  "messages": [{"role": "user", "content": "hi"}]})
        assert code == 200, r
        # Tokens alternate 1, 2, 3 on the H-neuron: logits -3, -1, 1 -> only the 3s cross 0.5.
        assert r["pieces"] == list("abcdef") and len(r["prob"]) == 6
        assert r["flagged"] == [i for i, p in enumerate(r["prob"]) if p >= 0.5] and r["flagged"]
        assert r["url"] == f"viz/{r['id']}/"
        code, page = call(studio_srv, "/" + r["url"])
        assert code == 200 and "three" in page
        code, meta = call(studio_srv, f"/viz/{r['id']}/api/meta")
        assert meta["flagged"] == r["flagged"] and meta["labels"] == list("abcdef")
        assert meta["mode"] == "absolute"
        code, lst = call(studio_srv, "/api/traces")
        assert [x["id"] for x in lst["traces"]] == [r["id"]]
        assert call(studio_srv, "/viz/../api/meta")[0] == 404
        assert call(studio_srv, "/viz/nosuchtrace/api/meta")[0] == 404
        # A model without a classifier gets a reason, not a crash.
        code, r = call(studio_srv, "/api/trace", {"model": VLM, "text": "x", "messages": []})
        assert code == 400 and "classifier" in r["error"]
    finally:
        studio._SCORERS.clear()
        studio.STATE.update(cett=None)



def test_checked_replies_feed_the_review_page(studio_srv, tmp_path):
    sc = _fake_scorer(tmp_path)
    m = studio.find_model(CODER)
    st = studio.load_settings()
    st.setdefault(studio._key(m["path"]), {})["classifier"] = str(tmp_path / "clf.npz")
    studio.save_settings(st)
    studio.STATE.update(cett=sc.binary, traces_dir=str(tmp_path / "traces"))
    studio._SCORERS[(m["path"], str(tmp_path / "clf.npz"))] = sc
    try:
        code, info = call(studio_srv, "/api/review")
        assert code == 200 and info["models"] == []
        ids = []
        for i, text in enumerate(["abcdef", "abcdefgh", "xyz", "pqrstu", "hello"]):
            code, r = call(studio_srv, "/api/trace", {"model": CODER, "text": text,
                                                      "messages": [{"role": "user", "content": f"q{i}"}]})
            assert code == 200 and r["review"]["model"] == CODER
            ids.append(r["review"]["id"])
        for oid, v in zip(ids, ["correct", "wrong", "correct", "wrong"]):
            assert call(studio_srv, "/api/review/label", {"model": CODER, "id": oid, "verdict": v})[0] == 200
        assert call(studio_srv, "/api/review/label", {"model": CODER, "id": "nope", "verdict": "wrong"})[0] == 404
        assert call(studio_srv, "/api/review/label", {"model": CODER, "id": ids[0], "verdict": "maybe"})[0] == 400
        code, info = call(studio_srv, "/api/review")
        f = info["facets"][CODER]
        assert f["n"] == 5 and f["sources"] == {"chat-check": {"kind": "observed", "n": 5}}
        assert f["verdicts"] == {"correct": 2, "wrong": 2, "unknown": 1}
        code, r = call(studio_srv, "/api/review/summary", {"model": CODER, "stat": "association", "by": "all"})
        assert code == 200 and r["n_right"] == 2 and r["n_wrong"] == 2 and len(r["grid"]) == 3
        code, r = call(studio_srv, "/api/review/summary", {"model": CODER, "stat": "risk", "verdicts": ["unknown"]})
        assert code == 200 and r["n"] == 1
        assert call(studio_srv, "/api/review/summary", {"model": "nobody"})[0] == 404
        assert call(studio_srv, "/api/review/summary", {"model": CODER, "stat": "vibes"})[0] == 400
        code, v = call(studio_srv, "/api/review/view", {"model": CODER, "stat": "firing", "by": "day"})
        assert code == 200 and v["url"].startswith("/review/view/")
        code, page = call(studio_srv, v["url"])
        assert code == 200 and "three" in page
        code, meta = call(studio_srv, v["url"] + "api/meta")
        assert meta["mode"] == "review" and meta["frames"] == 1
        assert call(studio_srv, "/review/view/000000000000/api/meta")[0] == 404
        code, page = call(studio_srv, "/review")
        assert code == 200 and "Studio · Review" in page
    finally:
        studio._SCORERS.clear()
        studio.STATE.update(cett=None)



def test_requirements_api(studio_srv, tmp_path, monkeypatch):
    import oc_requirements
    monkeypatch.setattr(oc_requirements, "SNAP_DIR", tmp_path / "env")
    code, r = call(studio_srv, "/api/requirements")
    assert code == 200 and r["features"]["core"]["status"] in ("ok", "warn") and "hardware" in r
    assert call(studio_srv, "/api/requirements/diff", {})[0] == 400          # no snapshots yet
    for _ in range(2):
        code, s = call(studio_srv, "/api/requirements/snapshot", {})
        assert code == 200
        time.sleep(1.1)                                                       # ids are per second
    code, lst = call(studio_srv, "/api/requirements/snapshots")
    assert len(lst["snapshots"]) == 2
    code, d = call(studio_srv, "/api/requirements/diff", {})
    assert code == 200 and d["packages"]["changed"] == {}
    assert call(studio_srv, "/api/requirements/diff", {"a": "../x", "b": "y"})[0] == 400
    assert call(studio_srv, "/api/requirements/diff", {"a": "20000101-000000", "b": s["id"]})[0] == 404
    code, txt = call(studio_srv, "/api/requirements/freeze")
    assert code == 200 and "numpy==" in txt
    code, j = call(studio_srv, "/api/jobs", {"kind": "install_package", "values": {"package": "evil"}})
    assert code == 400


_LOOP = {}


def _start_director(tmp_path):
    import director
    studio.DIRECTOR["store"] = director.ProjectStore(tmp_path / "projects")
    studio.STATE.update(max_workers=2, worker_idle=300, director_tick=0.2)
    if not _LOOP:
        _LOOP["t"] = threading.Thread(target=studio.director_loop, daemon=True)
        _LOOP["t"].start()


def _wait(pred, timeout=40):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_director_end_to_end(studio_srv, tmp_path):
    _start_director(tmp_path)
    text = ("- Shaders: write GLSL vertex and fragment shaders for the terrain mesh.\n"
            "- Build: set up the CMake build with the toolchain and dependencies.\n")
    code, r = call(studio_srv, "/api/projects", {"title": "Demo", "goal": "a renderer", "text": text})
    assert code == 200, r
    pid = r["id"]
    code, p = call(studio_srv, f"/api/projects/{pid}")
    assert [t["labels"][0] for t in p["tasks"]] == ["graphics", "systems"]
    assert set(p["skills"]) == {"graphics", "systems"} and p["status"] == "draft"
    # Nothing runs from a draft.
    assert call(studio_srv, f"/api/projects/{pid}/start", {})[0] == 400
    call(studio_srv, f"/api/projects/{pid}/edit", {"changes": [{"op": "update", "id": "T1",
                                                                "fields": {"depends_on": ["T2"]}}]})
    code, r = call(studio_srv, f"/api/projects/{pid}/approve", {})
    assert code == 200, r
    code, p = call(studio_srv, f"/api/projects/{pid}")
    # No graded results: the largest model that fits, and the plan says why.
    chosen = p["tasks"][0]["assignee"]["model"]
    assert chosen in (CODER, VLM) and all(t["assignee"]["model"] == chosen for t in p["tasks"])
    assert "largest" in p["tasks"][0]["assignee"]["reason"]
    call(studio_srv, f"/api/projects/{pid}/start", {})
    p = _wait(lambda: (lambda p: p if d_status(p, "T2") == "review" else None)(call(studio_srv, f"/api/projects/{pid}")[1]))
    a = p["tasks"][1]["attempts"][0]
    assert a["text"].startswith(f"model={chosen}") and a["report"]["status"] == "unreported"
    assert d_status(p, "T1") == "todo"                      # waits for T2's review
    _, w = call(studio_srv, "/api/workers")
    assert [x["model"] for x in w["workers"]] == [chosen] and w["workers"][0]["busy"] == 0
    call(studio_srv, f"/api/projects/{pid}/review", {"task": "T2", "accept": False, "feedback": "add a CI job"})
    p = _wait(lambda: (lambda p: p if d_status(p, "T2") == "review" and len(p["tasks"][1]["attempts"]) == 2
                       else None)(call(studio_srv, f"/api/projects/{pid}")[1]))
    call(studio_srv, f"/api/projects/{pid}/review", {"task": "T2", "accept": True})
    p = _wait(lambda: (lambda p: p if d_status(p, "T1") == "review" else None)(call(studio_srv, f"/api/projects/{pid}")[1]))
    call(studio_srv, f"/api/projects/{pid}/review", {"task": "T1", "accept": True})
    _, p = call(studio_srv, f"/api/projects/{pid}")
    assert p["status"] == "done"
    assert [h["action"] for h in p["history"]][:3] == ["created", "edit", "approved"]
    call(studio_srv, "/api/workers/stop", {})
    assert call(studio_srv, "/api/workers")[1]["workers"] == []
    page = urllib.request.urlopen(studio_srv + "/projects").read().decode()
    assert "Approve plan" in page


def test_project_requirements_gate_start(studio_srv, tmp_path):
    import director
    studio.DIRECTOR["store"] = director.ProjectStore(tmp_path / "projects")
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)\nfind_package(Git)\n")
    code, r = call(studio_srv, "/api/projects", {"title": "Req", "text": "- Build: set up CMake.", "repo": str(repo)})
    assert code == 200, r
    pid = r["id"]
    code, p = call(studio_srv, f"/api/projects/{pid}")
    names = [i["name"] for i in p["requirements"]["items"]]
    assert "cmake" in names and "git" in names and p["requirements"]["last"] is None
    code, meta = call(studio_srv, "/api/projects")
    assert "tool" in meta["req_kinds"] and "amd_vulkan" in meta["req_features"]
    items = p["requirements"]["items"] + [{"kind": "tool", "name": "no-such-tool-xyz"},
                                          {"kind": "note", "name": "needs the dev board plugged in"}]
    code, r = call(studio_srv, f"/api/projects/{pid}/requirements", {"items": items, "reason": "board"})
    assert code == 200, r
    assert call(studio_srv, f"/api/projects/{pid}/requirements", {"items": [{"kind": "tool", "name": "a|b"}]})[0] == 400
    code, rq = call(studio_srv, f"/api/projects/{pid}/requirements")
    assert code == 200 and rq["last"]["missing"] == ["no-such-tool-xyz"]
    assert call(studio_srv, f"/api/projects/{pid}/approve", {})[0] == 200
    code, r = call(studio_srv, f"/api/projects/{pid}/start", {})
    assert code == 409 and r["missing"] == ["no-such-tool-xyz"]
    _, p = call(studio_srv, f"/api/projects/{pid}")
    assert p["status"] == "approved"
    code, r = call(studio_srv, f"/api/projects/{pid}/start", {"force": True})
    assert code == 200 and r["status"] == "running"
    call(studio_srv, f"/api/projects/{pid}/pause", {})
    code, r = call(studio_srv, f"/api/projects/{pid}/requirements/scan", {"root": str(tmp_path / "nope")})
    assert code == 400
    code, r = call(studio_srv, f"/api/projects/{pid}/requirements/scan", {})
    assert code == 200
    _, p = call(studio_srv, f"/api/projects/{pid}")
    assert "no-such-tool-xyz" in [i["name"] for i in p["requirements"]["items"]]   # the person's items stay
    assert any(h["action"] == "requirements" for h in p["history"])
    call(studio_srv, "/api/workers/stop", {})


def d_status(p, tid):
    return next(t["status"] for t in p["tasks"] if t["id"] == tid)


def test_projects_are_owner_only(studio_srv, tmp_path):
    """Plans hold private project text and start models: paired devices get neither."""
    _start_director(tmp_path)
    reg = studio.STATE["devices"]
    code, _, _ = reg.new_code()
    dev = reg.claim(code, "phone")
    studio.STATE["token"] = "o" * 40
    try:
        for path, body in (("/api/projects", None), ("/projects", None), ("/api/workers", None),
                           ("/api/projects", {"text": "- do things with the build system"})):
            req = urllib.request.Request(studio_srv + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {dev['token']}",
                                                  "Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=10)
            assert e.value.code == 403, path
    finally:
        studio.STATE["token"] = None


def _rpc(base, payload, headers=None, raw=False):
    req = urllib.request.Request(base + "/mcp", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, (json.loads(body) if body[:1] == b"{" else body)


def test_mcp_over_http(studio_srv, tmp_path):
    _start_director(tmp_path)
    code, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                                           "clientInfo": {"name": "t", "version": "0"}}})
    assert code == 200 and r["result"]["protocolVersion"] == "2025-03-26"
    assert r["result"]["serverInfo"]["name"] == "omnicortex" and "tools" in r["result"]["capabilities"]
    assert _rpc(studio_srv, {"jsonrpc": "2.0", "method": "notifications/initialized"})[0] == 202
    code, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = {t["name"]: t for t in r["result"]["tools"]}
    assert {"chat", "check_reply", "start_job", "create_project", "review_task", "devices"} <= set(tools)
    assert tools["list_models"]["annotations"]["readOnlyHint"] and tools["start_job"]["annotations"]["destructiveHint"]
    # A batch: models, a chat through the fake llama-server, routing, a project.
    code, r = _rpc(studio_srv, [
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "chat", "arguments": {"model": CODER, "prompt": "hi"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "route_prompt", "arguments": {"text": "Write a GLSL shader for the terrain mesh"}}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "create_project", "arguments": {
            "title": "t", "text": "- Build: set up the CMake build with the toolchain and dependencies."}}}])
    out = {x["id"]: x["result"] for x in r}
    assert CODER in [m["id"] for m in out[3]["structuredContent"]["models"]]
    assert out[4]["structuredContent"]["reply"].startswith(f"model={CODER}")
    assert out[5]["structuredContent"]["subjects"] == ["graphics"]
    pid = out[6]["structuredContent"]["id"]
    _, p = call(studio_srv, f"/api/projects/{pid}")
    assert p["tasks"][0]["labels"] == ["systems"]
    # Errors: a tool's failure is a result the model can read; protocol errors are JSON-RPC errors.
    _, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                             "params": {"name": "job_log", "arguments": {"id": "000000000000"}}})
    assert r["result"]["isError"] and "404" in r["result"]["content"][0]["text"]
    _, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "nope"}})
    assert r["error"]["code"] == -32602
    _, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "chat",
                                                                                         "arguments": {}}})
    assert "missing model" in r["error"]["message"]
    _, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 10, "method": "frobnicate"})
    assert r["error"]["code"] == -32601


def test_mcp_refuses_browser_tricks(studio_srv):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert _rpc(studio_srv, msg, {"Origin": "https://evil.example"})[0] == 403
    req = urllib.request.Request(studio_srv + "/mcp", data=json.dumps(msg).encode(),
                                 headers={"Content-Type": "text/plain"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 415                       # a "simple" cross-site form post cannot reach it
    host = studio_srv.split("//")[1]
    assert _rpc(studio_srv, msg, {"Origin": f"http://{host}"})[0] == 200
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(studio_srv + "/mcp", timeout=10)
    assert e.value.code == 405


def test_mcp_and_connect_with_tokens(studio_srv, tmp_path):
    _start_director(tmp_path)
    owner = "o" * 40
    studio.STATE["token"] = owner
    try:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_projects", "arguments": {}}}
        assert _rpc(studio_srv, msg)[0] == 401
        code, r = _rpc(studio_srv, msg, {"Authorization": f"Bearer {owner}"})
        assert code == 200 and not r["result"]["isError"]
        # The Connect page mints an app token: a paired device, so it can chat but not run jobs.
        req = urllib.request.Request(studio_srv + "/api/connect/token", data=b'{"name": "Cline"}',
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {owner}"})
        app = json.loads(urllib.request.urlopen(req, timeout=10).read())["token"]
        hdr = {"Authorization": f"Bearer {app}"}
        _, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": "chat", "arguments": {"model": CODER, "prompt": "hi"}}}, hdr)
        assert not r["result"]["isError"]
        _, r = _rpc(studio_srv, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "start_job", "arguments": {"kind": "testqa"}}}, hdr)
        assert r["result"]["isError"] and "403" in r["result"]["content"][0]["text"]
        req = urllib.request.Request(studio_srv + "/api/connect", headers={"Authorization": f"Bearer {owner}"})
        sn = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert sn["token_required"] and sn["cline_provider"]["API Provider"] == "OpenAI Compatible"
        entry = sn["cline_mcp_http"]["mcpServers"]["omnicortex"]
        assert entry["url"].endswith("/mcp") and entry["headers"]["Authorization"].startswith("Bearer <")
        assert "list_models" in entry["autoApprove"] and "start_job" not in entry["autoApprove"]
    finally:
        studio.STATE["token"] = None


def test_mcp_with_the_official_client(studio_srv):
    """Both transports against the reference implementation of the protocol."""
    mcp = pytest.importorskip("mcp")
    import asyncio
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    async def run(transport):
        async with Client(transport) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            r = await c.call_tool("route_prompt", {"text": "Set up the CMake build and the CI toolchain"})
            return names, json.loads(r.content[0].text)

    for tr in (streamable_http_client(studio_srv + "/mcp"),
               stdio_client(StdioServerParameters(command=sys.executable, args=[
                   str(ROOT / "scripts" / "oc_mcp.py"), "--studio", studio_srv]))):
        names, out = asyncio.run(run(tr))
        assert "check_reply" in names and out["subjects"] == ["systems"]


def test_connect_writes_cline_settings_and_keeps_others(tmp_path, monkeypatch):
    import oc_connect
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    path = oc_connect.cline_settings_path("cursor")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    assert oc_connect.main(["--studio", "http://127.0.0.1:9", "cline", "--editor", "cursor", "--write"]) == 0
    cfg = json.loads(path.read_text())
    assert cfg["mcpServers"]["other"] == {"command": "x"}
    assert cfg["mcpServers"]["omnicortex"]["url"] == "http://127.0.0.1:9/mcp"
    assert list(path.parent.glob("*.bak-*"))
    tok = tmp_path / "tok"
    tok.write_text("secret-token\n")
    oc_connect.main(["--studio", "http://127.0.0.1:9", "--token-file", str(tok), "cline", "--editor", "cursor",
                     "--write"])
    e = json.loads(path.read_text())["mcpServers"]["omnicortex"]
    assert e["headers"] == {"Authorization": "Bearer secret-token"}
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_setup_settings_validate_apply_and_persist(studio_srv, tmp_path):
    studio.STATE["config_path"] = str(tmp_path / "config.json")
    studio.STATE.setdefault("max_workers", 2)
    studio.STATE.setdefault("worker_idle", 300)
    studio.STATE.setdefault("score_every", 1)
    studio.STATE.setdefault("score_ngl", 0)
    code, s = call(studio_srv, "/api/setup")
    assert code == 200 and s["ready"]["server"] and s["ready"]["models"]
    assert s["fields"]["server"]["value"] == studio.STATE["server_bin"]
    # Bad values are refused per field, and nothing is written.
    code, r = call(studio_srv, "/api/setup", {"values": {"idle_ttl": -1, "models_dir": ["/no/such/dir"],
                                                         "host": "0.0.0.0"}})
    assert code == 400 and set(r["fields"]) == {"idle_ttl", "models_dir", "host"}
    assert not (tmp_path / "config.json").exists()
    # Good values apply at once and land in the config file.
    code, r = call(studio_srv, "/api/setup", {"values": {"idle_ttl": 600, "max_workers": 3, "max_jobs": 4}})
    assert code == 200 and r["restart_needed"] == ["max_jobs"]
    assert studio.STATE["idle_ttl"] == 600 and studio.STATE["max_workers"] == 3
    cfg = json.loads((tmp_path / "config.json").read_text())["studio"]
    assert cfg == {"idle_ttl": 600, "max_workers": 3, "max_jobs": 4}
    studio.STATE["idle_ttl"] = 0
    # Every flag worth a setting is here, grouped; network and pairing are advanced.
    _, s = call(studio_srv, "/api/setup")
    assert {"host", "port", "tls_cert", "pair_max", "rag_embed", "mcp_config", "no_jobs", "vllm"} <= set(s["fields"])
    assert s["fields"]["host"]["group"] == "Network and security" and s["fields"]["host"]["advanced"]
    assert not s["fields"]["idle_ttl"]["advanced"]
    # A network address is accepted only with what Studio needs to start on it.
    code, r = call(studio_srv, "/api/setup", {"values": {"host": "0.0.0.0", "allow_plaintext": True}})
    assert code == 400 and "token" in r["fields"]["host"]
    tok = tmp_path / "studio.token"
    tok.write_text("x" * 48)
    code, r = call(studio_srv, "/api/setup", {"values": {"host": "0.0.0.0", "token_file": str(tok)}})
    assert code == 400 and "plaintext" in r["fields"]["host"]
    code, r = call(studio_srv, "/api/setup", {"values": {"host": "0.0.0.0", "token_file": str(tok),
                                                         "allow_plaintext": True}})
    assert code == 200 and set(r["restart_needed"]) == {"host", "token_file", "allow_plaintext"}
    code, r = call(studio_srv, "/api/setup", {"values": {"pair_max": "soon", "pair_durations": "1h,8h,1d",
                                                         "rag_embed": "ftp://x", "port": 70000}})
    assert code == 400 and set(r["fields"]) == {"pair_max", "rag_embed"}
    code, r = call(studio_srv, "/api/setup", {"values": {"port": 70000}})
    assert code == 400 and "port" in r["fields"]
    code, r = call(studio_srv, "/api/setup", {"values": {"pair_max": "30d", "vllm": "/opt/vllm/bin/vllm"}})
    assert code == 200 and studio.STATE["vllm_bin"] == "/opt/vllm/bin/vllm"
    cfg = json.loads((tmp_path / "config.json").read_text())["studio"]
    assert cfg["host"] == "0.0.0.0" and cfg["pair_max"] == "30d" and cfg["token_file"] == str(tok)
    studio.STATE["vllm_bin"] = "vllm"


def test_config_file_supplies_studio_defaults(tmp_path, monkeypatch):
    """main() reads the Setup page's file before parsing flags; flags still win."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"studio": {"idle_ttl": 77, "models_dir": [str(tmp_path)], "port": 1}}))
    seen = {}

    class Stop(Exception):
        pass

    def fake_check_bind(*a, **k):
        seen["idle_ttl"], seen["dirs"] = studio.STATE["idle_ttl"], list(studio.STATE["models_dirs"])
        raise Stop
    monkeypatch.setattr(studio.sec, "check_bind", fake_check_bind)
    with pytest.raises(Stop):
        studio.main(["--config", str(cfg), "--settings", str(tmp_path / "s.json"), "--projects-dir",
                     str(tmp_path / "p"), "--idle-ttl", "5"])
    assert seen == {"idle_ttl": 5, "dirs": [str(tmp_path)]}


def test_hardware_config_from_the_ui(studio_srv, tmp_path):
    studio.STATE["hardware_path"] = str(tmp_path / "hardware.json")
    code, h = call(studio_srv, "/api/hardware")
    assert code == 200 and any(d["id"] == "cpu:0" for d in h["devices"])
    code, r = call(studio_srv, "/api/hardware", {"config": {"servers": {"rocm": "/no/llama-server"}}})
    assert code == 400 and "does not exist" in r["error"]
    code, r = call(studio_srv, "/api/hardware", {"config": {"evil": 1}})
    assert code == 400
    code, r = call(studio_srv, "/api/hardware", {"config": {"devices": {"cpu:0": {"reserve_gib": 3}}}})
    assert code == 200
    assert json.loads((tmp_path / "hardware.json").read_text())["devices"]["cpu:0"]["reserve_gib"] == 3


def test_lab_services_start_and_stop_from_studio(studio_srv):
    code, lab = call(studio_srv, "/api/lab")
    assert code == 200 and {"pipeline", "quant_lab", "scale_sweep", "replay", "live"} <= set(lab["services"])
    assert "chat" not in lab["services"]                       # Studio is the chat
    code, r = call(studio_srv, "/api/lab/start", {"name": "scale_sweep"})
    try:
        assert code == 200 and r["port"]
        with urllib.request.urlopen(f"http://127.0.0.1:{r['port']}/", timeout=10) as resp:
            assert b"Scale Sweep" in resp.read()
        assert call(studio_srv, "/api/lab")[1]["services"]["scale_sweep"]["running"]
    finally:
        call(studio_srv, "/api/lab/stop", {"name": "scale_sweep"})
    assert not call(studio_srv, "/api/lab")[1]["services"]["scale_sweep"]["running"]
    assert call(studio_srv, "/api/lab/start", {"name": "replay"})[0] == 400      # needs a trace
    assert call(studio_srv, "/api/lab/start", {"name": "../../bin/sh"})[0] == 404


def test_launcher_starts_studio_and_opens_setup_first(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path))
    port = free_port()
    launch = [sys.executable, str(ROOT / "scripts" / "launch.py"), "--no-browser", "--port", str(port)]
    try:
        out = subprocess.run(launch, env=env, capture_output=True, text=True, timeout=90).stdout
        assert "starting OmniCortex Studio" in out and out.strip().endswith("/setup")   # nothing set up yet
        out = subprocess.run(launch, env=env, capture_output=True, text=True, timeout=30).stdout
        assert "already running" in out
    finally:
        subprocess.run(launch[:2] + ["--stop"], env=env, timeout=30)


def test_chat_asks_for_token_probabilities_only_when_asked(studio_srv):
    msgs = [{"role": "user", "content": "hi"}]
    ev = sse_chat(studio_srv, {"model": CODER, "messages": msgs, "confidence": True})
    lps = [t for e in ev if e.get("choices") for t in (e["choices"][0].get("logprobs") or {}).get("content", [])]
    assert len(lps) == 2 and len(lps[1]["top_logprobs"]) == 5
    ev = sse_chat(studio_srv, {"model": CODER, "messages": msgs})
    assert not any(e.get("choices") and e["choices"][0].get("logprobs") for e in ev)
