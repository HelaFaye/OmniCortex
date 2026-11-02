"""OmniCortex Remote: the host agent on its own, and Studio as its controller."""
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "remote"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
from neuronscope_remote import agent as nsr  # noqa: E402
from test_studio import call, free_port, studio_srv  # noqa: E402,F401


@pytest.fixture()
def backend():
    port = free_port()
    p = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_llama_server.py"), "--port", str(port),
                          "--alias", "tiny-chat"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2)
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    p.terminate()


def make_agent(tmp_path, backend, mode="vpn", **kw):
    ag = nsr.Agent(tmp_path / "host")
    ag.cfg.update(detect=False, backends=[{"kind": "llama-server", "url": backend}],
                  models_dir=str(tmp_path / "lmstudio-models"), name="gpu-box")
    (tmp_path / "lmstudio-models").mkdir(exist_ok=True)
    ag.save_config()
    rt = nsr.Runtime(ag, beacon_targets=kw.pop("beacon_targets", None))
    urls = rt.listen(mode, kw.pop("bind", "127.0.0.1" if mode == "vpn" else ""), 0, **kw)
    return ag, rt, urls[0]


def rpc(url, token, method, params=None, fp=None, mid=1):
    st, out, _ = nsr.http_json(url, "POST", "/mcp", {"jsonrpc": "2.0", "id": mid, "method": method,
                                                    "params": params or {}}, token=token, fingerprint=fp)
    assert st == 200, out
    return out["result"]


def test_codes_proofs_and_invites():
    c = nsr.new_code()
    assert len(c) == nsr.CODE_LEN and nsr.norm_code(nsr.fmt_code(c).lower()) == c
    p = nsr.pair_proof(c, "ab:CD", "claim")
    assert p == nsr.pair_proof(c, "abcd", "claim")            # fingerprint format does not matter
    assert p != nsr.pair_proof(c, "abce", "claim")            # but which certificate does
    assert p != nsr.pair_proof(c, "abcd", "accept")
    inv = nsr.parse_invite(nsr.make_invite("https://studio:7870", c, "ff" * 32, "Studio"))
    assert inv == {"url": "https://studio:7870", "code": c, "fingerprint": "ff" * 32, "name": "Studio"}
    link = nsr.parse_invite(f"nsr://10.0.0.5:47871/#c={nsr.fmt_code(c)}&fp={'ee' * 32}&s=https&n=box")
    assert link["url"] == "https://10.0.0.5:47871" and link["fingerprint"] == "ee" * 32 and link["name"] == "box"
    with pytest.raises(ValueError):
        nsr.parse_invite("hello")


def test_pairing_tls_pinning_and_mcp(tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend, mode="lan", bind="127.0.0.1")
    try:
        assert url.startswith("https://") and len(ag.fingerprint) == 64
        st, info, _ = nsr.http_json(url, "GET", "/info", fingerprint="")
        assert info["role"] == "host" and info["tls"] and not info["pairing_open"]
        # nothing works before pairing
        st, out, _ = nsr.http_json(url, "POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                   fingerprint="")
        assert st == 401 and "pair" in out["error"]
        code = nsr.norm_code(ag.pairing.open()["code"])
        # a proof bound to another certificate (a man in the middle) is refused and costs an attempt
        st, out, _ = nsr.http_json(url, "POST", "/pair", {"proof": nsr.pair_proof(code, "00" * 32, "claim")},
                                   fingerprint="")
        assert st == 403
        rec = nsr.pair_direct("127.0.0.1:%d" % ag.link["port"], code, "studio-test")
        assert rec["fingerprint"] == ag.fingerprint and rec["url"] == url
        assert ag.pairing.current() is None                        # single use
        with pytest.raises(nsr.LinkError):
            nsr.pair_direct(url, code, "again")
        # pinned: another fingerprint is refused before anything is sent
        with pytest.raises(Exception):
            nsr.http_json(url, "GET", "/info", fingerprint="11" * 32)
        init = rpc(url, rec["token"], "initialize", {"protocolVersion": "2025-06-18"}, rec["fingerprint"])
        assert "HOST" in init["instructions"] and "CONTROLLER" in init["instructions"]
        names = [t["name"] for t in rpc(url, rec["token"], "tools/list", fp=rec["fingerprint"])["tools"]]
        assert {"host_info", "list_models", "chat", "load_model", "receive_file", "install_model"} <= set(names)
        info = rpc(url, rec["token"], "tools/call", {"name": "host_info", "arguments": {}}, rec["fingerprint"])
        assert info["structuredContent"]["role"] == "host" and info["structuredContent"]["you_are"] == "controller"
        ms = rpc(url, rec["token"], "tools/call", {"name": "list_models"}, rec["fingerprint"])["structuredContent"]
        assert ms["models"][0]["id"] == "tiny-chat" and ms["models"][0]["backend"] == "llama-server"
        r = rpc(url, rec["token"], "tools/call", {"name": "chat", "arguments": {
            "model": "tiny-chat", "messages": [{"role": "user", "content": "hi"}]}}, rec["fingerprint"])
        assert not r["isError"] and r["content"][0]["text"] == r["structuredContent"]["reply"]
        # the host owner sees what controllers did, and can revoke them
        assert any(a["what"] == "chat" and a["who"] == "studio-test" for a in ag.activity)
        cid = ag.controllers.list()[0]["id"]
        assert ag.controllers.revoke(cid)
        st, _, _ = nsr.http_json(url, "POST", "/mcp", {"jsonrpc": "2.0", "id": 9, "method": "tools/list"},
                                 token=rec["token"], fingerprint=rec["fingerprint"])
        assert st == 401
    finally:
        rt.close_api()


def test_code_dies_after_five_wrong_tries(tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend)
    try:
        code = nsr.norm_code(ag.pairing.open()["code"])
        for _ in range(nsr.MAX_ATTEMPTS):
            st, _, _ = nsr.http_json(url, "POST", "/pair", {"proof": "x"})
            assert st == 403
        st, _, _ = nsr.http_json(url, "POST", "/pair", {"proof": nsr.pair_proof(code, "", "claim")})
        assert st in (403, 429)
        assert ag.pairing.current() is None
    finally:
        rt.close_api()


def test_permissions_uploads_and_install(tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend)
    try:
        code = nsr.norm_code(ag.pairing.open()["code"])
        rec = nsr.pair_direct(url, code, "studio")
        tok = rec["token"]

        def tool(name, args=None):
            return rpc(url, tok, "tools/call", {"name": name, "arguments": args or {}})

        r = tool("delete_file", {"file": "x.gguf"})
        assert r["isError"] and "does not allow" in r["content"][0]["text"]
        ag.cfg["allow"]["chat"] = False
        assert tool("chat", {"messages": [{"role": "user", "content": "x"}]})["isError"]
        st, _, _ = nsr.http_json(url, "POST", "/v1/chat/completions", {"model": "tiny-chat", "messages": []},
                                 token=tok)
        assert st == 403
        ag.cfg["allow"]["chat"] = True
        # resumable upload with a checksum
        data = b"GGUF" + os.urandom(3000)
        sha = hashlib.sha256(data).hexdigest()

        def put(path, body):
            conn = nsr.open_conn(url, None, 10)
            conn.request("PUT", path, body=body, headers={"Authorization": f"Bearer {tok}"})
            r = conn.getresponse()
            out = json.loads(r.read())
            conn.close()
            return r.status, out
        assert put("/upload/m-q4.gguf?offset=0", data[:1000])[1]["size"] == 1000
        st, out = put("/upload/m-q4.gguf?offset=0", data[1000:])
        assert st == 409 and out["size"] == 1000                  # resume from where the host is
        st, out = put(f"/upload/m-q4.gguf?offset=1000&final=1&sha256={sha}", data[1000:])
        assert st == 200 and out["complete"]
        st, out = put("/upload/..%2F..%2Fevil.gguf?offset=0&final=1", b"x")
        assert st == 200 and out["file"] == "evil.gguf"             # names cannot leave the incoming folder
        assert (ag.incoming() / "evil.gguf").exists()
        files = tool("list_files")["structuredContent"]["files"]
        assert {"file": "m-q4.gguf", "size": len(data), "partial": False} in files
        r = tool("install_model", {"file": "m-q4.gguf"})["structuredContent"]
        dest = Path(r["installed"])
        assert dest == tmp_path / "lmstudio-models" / "neuronscope" / "m-q4" / "m-q4.gguf"
        assert dest.read_bytes() == data
        assert tool("list_files", {"where": "models"})["structuredContent"]["files"][0]["file"] == \
            "neuronscope/m-q4/m-q4.gguf"
        ag.cfg["allow"]["delete"] = True
        assert tool("delete_file", {"file": "../outside", "where": "models"})["isError"]
        assert not tool("delete_file", {"file": "evil.gguf"})["isError"]
        # OpenAI pass-through: models and chat
        st, out, _ = nsr.http_json(url, "GET", "/v1/models", token=tok)
        assert [m["id"] for m in out["data"]] == ["tiny-chat"]
        st, out, _ = nsr.http_json(url, "POST", "/v1/chat/completions",
                                   {"model": "tiny-chat", "messages": [{"role": "user", "content": "hi"}]}, token=tok)
        assert st == 200 and out["choices"][0]["message"]["content"]
    finally:
        rt.close_api()


def test_receive_file_downloads_and_checks(tmp_path, backend):
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import functools
    import threading
    src = tmp_path / "srv"
    src.mkdir()
    data = os.urandom(5000)
    (src / "m.gguf").write_bytes(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(src)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ag = nsr.Agent(tmp_path / "host")
    ag.cfg.update(detect=False)
    try:
        base = f"http://127.0.0.1:{srv.server_port}/m.gguf"
        bad = ag.call_tool("receive_file", {"url": base, "name": "bad.gguf", "sha256": "0" * 64}, "t")
        good = ag.call_tool("receive_file", {"url": base, "sha256": hashlib.sha256(data).hexdigest()}, "t")
        for _ in range(100):
            st = {t["file"]: t["status"] for t in ag.t_transfers({})["transfers"]}
            if st.get("bad.gguf") not in ("running", "checking") and st.get("m.gguf") not in ("running", "checking"):
                break
            time.sleep(0.05)
        assert st == {"bad.gguf": "failed", "m.gguf": "done"}, st
        assert (ag.incoming() / "m.gguf").read_bytes() == data and not (ag.incoming() / "bad.gguf").exists()
        assert not bad["isError"] and not good["isError"]
    finally:
        srv.shutdown()


def test_console_needs_key_and_host(tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend)
    try:
        curl = rt.console(0)
        port = int(curl.split(":")[2].split("/")[0])
        key = curl.split("#k=")[1]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "This computer: HOST" in page and key not in page

        def get(path, headers):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        assert get("/api/state", {})[0] == 403
        assert get("/api/state", {"X-Console-Key": key, "Host": "evil.example:80"})[0] == 403
        st, s = get("/api/state", {"X-Console-Key": key})
        assert st == 200 and s["role"] == "host" and s["link"]["mode"] == "vpn"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/pair/open", data=b"{}", method="POST",
                                     headers={"X-Console-Key": key, "Content-Type": "application/json"})
        s = json.loads(urllib.request.urlopen(req).read())
        assert s["pairing"]["code"] and s["pairing"]["link"].startswith("nsr://127.0.0.1:")
    finally:
        rt.close_api()


def test_lan_beacon_is_heard_while_pairing(tmp_path, backend):
    import remote_hosts
    port = free_port()
    ag, rt, url = make_agent(tmp_path, backend, mode="lan", bind="127.0.0.1", beacon_targets=[("127.0.0.1", port)])
    try:
        assert remote_hosts.discover(1.0, port) == []              # pairing closed: silent
        ag.pairing.open()
        found = remote_hosts.discover(3.0, port)
        assert found and found[0]["name"] == "gpu-box" and found[0]["url"] == url
        assert found[0]["fingerprint"] == ag.fingerprint
    finally:
        rt.close_api()


def test_stdio_and_bridge(tmp_path, backend):
    home = tmp_path / "host"
    ag, rt, url = make_agent(tmp_path, backend)
    try:
        msgs = "\n".join(json.dumps(m) for m in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_models"}})) + "\n"
        env = {**os.environ, "NSR_HOME": str(home), "PYTHONPATH": str(ROOT / "remote")}
        out = subprocess.run([sys.executable, "-m", "neuronscope_remote", "stdio"], input=msgs, capture_output=True,
                             text=True, env=env, timeout=30).stdout.splitlines()
        assert [json.loads(x)["id"] for x in out] == [1, 2]
        assert json.loads(out[1])["result"]["structuredContent"]["models"][0]["id"] == "tiny-chat"
        # the controller side of the package: pair from the command line, then bridge an MCP client
        code = ag.pairing.open()["code"]
        ctl = {**env, "NSR_HOME": str(tmp_path / "controller")}
        r = subprocess.run([sys.executable, "-m", "neuronscope_remote", "add-host", url, "--code", code],
                           capture_output=True, text=True, env=ctl, timeout=30)
        assert r.returncode == 0, r.stderr
        assert "CONTROLLER" in r.stdout and '"bridge"' in r.stdout and "gpu-box" in r.stdout
        out = subprocess.run([sys.executable, "-m", "neuronscope_remote", "bridge", "gpu-box"], input=msgs,
                             capture_output=True, text=True, env=ctl, timeout=30).stdout.splitlines()
        got = {json.loads(x)["id"]: json.loads(x) for x in out}
        assert set(got) == {1, 2} and "HOST" in got[1]["result"]["instructions"]
    finally:
        rt.close_api()


def test_official_mcp_client(tmp_path, backend):
    pytest.importorskip("mcp")
    import asyncio

    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    ag, rt, url = make_agent(tmp_path, backend)
    try:
        rec = nsr.pair_direct(url, ag.pairing.open()["code"], "mcp-sdk")

        async def go():
            http = create_mcp_http_client(headers={"Authorization": f"Bearer {rec['token']}"})
            async with streamable_http_client(url + "/mcp", http_client=http) as streams:
                r, w = streams[0], streams[1]
                async with ClientSession(r, w) as s:
                    init = await s.initialize()
                    tools = await s.list_tools()
                    res = await s.call_tool("chat", {"model": "tiny-chat",
                                                     "messages": [{"role": "user", "content": "hi"}]})
                return init, tools, res
        init, tools, res = asyncio.run(go())
        g = lambda o, *names: next(getattr(o, n) for n in names if hasattr(o, n))  # noqa: E731  SDK 1.x and 2.x
        assert g(init, "server_info", "serverInfo").name == "neuronscope-remote" and "HOST" in init.instructions
        ann = {t.name: t.annotations for t in tools.tools}
        assert g(ann["delete_file"], "destructive_hint", "destructiveHint")
        assert g(ann["list_models"], "read_only_hint", "readOnlyHint")
        assert not g(res, "is_error", "isError") and res.content[0].text
    finally:
        rt.close_api()


# ------------------------------------------------------------------ Studio as the controller

def test_studio_pairs_lists_and_chats_through_a_host(studio_srv, tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend, mode="lan", bind="127.0.0.1")
    try:
        st, page = call(studio_srv, "/remote")
        assert st == 200 and "This computer: CONTROLLER" in page
        code = ag.pairing.open()["code"]
        st, out = call(studio_srv, "/api/remote/pair", {"method": "lan", "address": url, "code": "WRONG-CODE0"})
        assert st == 400
        code = ag.pairing.open()["code"]
        st, out = call(studio_srv, "/api/remote/pair", {"method": "lan", "address": url, "code": code})
        assert st == 200, out
        assert out["name"] == "gpu-box" and out["models"] == ["tiny-chat"] and out["short_fingerprint"]
        assert ag.controllers.list()[0]["name"].startswith("OmniCortex Studio")
        st, s = call(studio_srv, "/api/remote")
        h = s["hosts"][0]
        assert s["role"] == "controller" and h["pinned"] and h["method"] == "lan" and h["models"] == ["tiny-chat"]
        # the host's models are Studio models now
        _, ms = call(studio_srv, "/v1/models")
        assert "gpu-box:tiny-chat" in [m["id"] for m in ms["data"]]
        st, r = call(studio_srv, "/v1/chat/completions", {"model": "gpu-box:tiny-chat",
                                                         "messages": [{"role": "user", "content": "hi"}]})
        assert st == 200 and r["choices"][0]["message"]["content"]
        # and every host tool is reachable through Studio
        st, r = call(studio_srv, "/api/remote/call", {"host": "gpu-box", "tool": "host_info"})
        assert st == 200 and r["role"] == "host"
        st, r = call(studio_srv, "/api/remote/call", {"host": "gpu-box", "tool": "delete_file", "args": {"file": "x"}})
        assert st == 400 and "does not allow" in r["error"]
        # send a model: resumable upload, then installed into LM Studio's folder
        _, s = call(studio_srv, "/api/remote")
        m = s["local_models"][0]
        st, r = call(studio_srv, "/api/remote/send", {"host": "gpu-box", "path": m["path"], "install": True})
        assert st == 200
        for _ in range(100):
            _, s = call(studio_srv, "/api/remote")
            if s["sends"][0]["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert s["sends"][0]["status"] == "done", s["sends"]
        assert list((tmp_path / "lmstudio-models" / "neuronscope").rglob("*.gguf"))
        st, r = call(studio_srv, "/api/remote/send", {"host": "gpu-box", "path": "/etc/passwd"})
        assert st == 400
        call(studio_srv, "/api/remote/remove", {"host": "gpu-box"})
        assert call(studio_srv, "/api/remote")[1]["hosts"] == []
    finally:
        rt.close_api()


def test_host_that_calls_in(studio_srv, tmp_path, backend):
    st, inv = call(studio_srv, "/api/remote/invite", {"name": "behind-nat"})
    assert st == 200 and inv["invite"].startswith("nsr1:") and inv["warning"]   # Studio is loopback-only here
    ag = nsr.Agent(tmp_path / "host")
    ag.cfg.update(detect=False, backends=[{"kind": "llama-server", "url": backend}], name="laptop")
    # a wrong proof is refused; the invite stays usable
    bad = nsr.make_invite(studio_srv, "AAAAAAAAAA")
    with pytest.raises(RuntimeError):
        nsr.claim_invite(ag, bad)
    entry = nsr.claim_invite(ag, inv["invite"])
    assert entry["controller"].startswith("OmniCortex Studio")
    with pytest.raises(RuntimeError):                        # single use
        nsr.claim_invite(nsr.Agent(tmp_path / "other"), inv["invite"])
    rt = nsr.Runtime(ag)
    rt.start_callhome(entry)
    try:
        for _ in range(100):
            _, s = call(studio_srv, "/api/remote")
            if s["hosts"] and s["hosts"][0].get("online"):
                break
            time.sleep(0.05)
        h = s["hosts"][0]
        assert h["name"] == "behind-nat" and h["method"] == "callhome" and h["online"]
        st, r = call(studio_srv, "/api/remote/call", {"host": "behind-nat", "tool": "list_models"})
        assert st == 200 and r["models"][0]["id"] == "tiny-chat"
        _, ms = call(studio_srv, "/v1/models")
        assert "behind-nat:tiny-chat" in [m["id"] for m in ms["data"]]
        st, r = call(studio_srv, "/v1/chat/completions", {"model": "behind-nat:tiny-chat",
                                                         "messages": [{"role": "user", "content": "hi"}]})
        assert st == 200 and r["choices"][0]["message"]["content"]
        # streamed: the reply comes through as the host's backend writes it
        req = urllib.request.Request(studio_srv + "/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": "behind-nat:tiny-chat", "stream": True,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            lines = [ln.decode().strip() for ln in resp if ln.strip()]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(ln[5:]) for ln in lines[:-1]]
        text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
        assert text == "model=tiny-chat mmproj=no image=no"
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        # Studio serves a model file to it by a one-time ticket; the host fetches it
        _, s = call(studio_srv, "/api/remote")
        st, r = call(studio_srv, "/api/remote/send", {"host": "behind-nat", "path": s["local_models"][0]["path"]})
        for _ in range(200):
            _, s = call(studio_srv, "/api/remote")
            if s["sends"][-1]["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert s["sends"][-1]["status"] == "done", s["sends"]
        assert any(p.suffix == ".gguf" for p in ag.incoming().iterdir())
        # host routes take the host's token only
        req = urllib.request.Request(studio_srv + "/api/remote/next?wait=0", headers={"Authorization": "Bearer x"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req)
    finally:
        for t in ag.callhome.values():
            t.stop.set()


def test_callhome_hub_streams_cancels_and_checks_hosts():
    import threading
    import remote_hosts
    hub = remote_hosts.CallHomeHub()
    hub.seen["a"] = hub.seen["b"] = time.time()
    got, out = [], {}

    def ask(on_progress):
        try:
            out["resp"] = hub.request("a", {"method": "tools/call", "params": {}}, timeout=5, on_progress=on_progress)
        except Exception as e:
            out["err"] = e

    t = threading.Thread(target=ask, args=(got.append,))
    t.start()
    msg = hub.next("a", wait=2)[0]
    mid = msg["id"]
    assert msg["message"]["params"]["_meta"]["progressToken"] == mid
    assert hub.progress(mid, ["x"], host="b") == {"ok": False, "cancel": True}   # not b's message
    assert not hub.reply(mid, {"r": 1}, host="b")
    assert hub.progress(mid, ["Hel", "lo"], host="a") == {"ok": True, "cancel": False}
    time.sleep(0.2)
    assert hub.reply(mid, {"r": 1}, host="a") and not hub.reply(mid, {"r": 2}, host="a")
    t.join(5)
    assert got == ["Hel", "lo"] and out["resp"] == {"r": 1}

    # the reader goes away: the host learns to stop at its next progress post
    def gone(_):
        raise BrokenPipeError

    out.clear()
    t = threading.Thread(target=ask, args=(gone,))
    t.start()
    mid = hub.next("a", wait=2)[0]["id"]
    hub.progress(mid, ["x"], host="a")
    t.join(5)
    assert isinstance(out["err"], BrokenPipeError)
    assert hub.progress(mid, ["y"], host="a")["cancel"]


def test_agent_stream_posts_pieces_and_stops_when_cancelled(backend):
    b = nsr.Backend("llama-server", backend)
    seen = []
    r = b.chat_stream({"model": "tiny-chat", "messages": [{"role": "user", "content": "hi"}]}, seen.append)
    assert "".join(seen) == r["choices"][0]["message"]["content"] == "model=tiny-chat mmproj=no image=no"
    seen.clear()
    r = b.chat_stream({"model": "tiny-chat", "messages": [{"role": "user", "content": "hi"}]}, seen.append,
                      cancelled=lambda: bool(seen))
    assert len(seen) == 1 and r["choices"][0]["finish_reason"] == "cancelled"


def test_package_is_standalone_and_downloadable(studio_srv):
    src = (ROOT / "remote" / "neuronscope_remote" / "agent.py").read_text()
    for line in src.splitlines():
        if line.startswith(("import ", "from ")) and "__future__" not in line:
            mod = line.split()[1].split(".")[0]
            assert mod in sys.stdlib_module_names or mod == "", line   # stdlib only
    st, _ = call(studio_srv, "/remote")
    with urllib.request.urlopen(studio_srv + "/remote/download") as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    names = set(z.namelist())
    assert {"neuronscope-remote/neuronscope_remote/agent.py", "neuronscope-remote/pyproject.toml",
            "neuronscope-remote/README.md", "neuronscope-remote/OmniCortex Remote.bat"} <= names
    assert not any("__pycache__" in n for n in names)


FAKE_TRACER = '''
import argparse, sys
import numpy as np
sys.path.insert(0, {scripts!r})
import hscore
p = argparse.ArgumentParser()
for f in ("--binary", "--gguf", "--classifier", "--messages", "--response-file", "--trace-out", "--ngl"):
    p.add_argument(f)
a = p.parse_args()
text = open(a.response_file, encoding="utf-8").read()
n = len(text)
hscore.save_trace(a.trace_out, {{"frames": np.ones((n, 3, 4), np.float32), "scores": np.linspace(-3, 3, n),
                  "prob": 1 / (1 + np.exp(-np.linspace(-3, 3, n))), "tokens": [ord(c) for c in text],
                  "col_weight": np.ones(4), "pieces": list(text), "stride": 1, "h_cells": [[1, 1]], "n_layers": 3}})
'''


def test_check_a_hosts_reply_here_or_on_the_host(studio_srv, tmp_path, backend):
    import shutil
    import studio
    from test_studio import CODER, _fake_scorer
    st, inv = call(studio_srv, "/api/remote/invite", {"name": "rig"})
    ag = nsr.Agent(tmp_path / "host")
    ag.cfg.update(detect=False, backends=[{"kind": "llama-server", "url": backend}], name="rig")
    rt = nsr.Runtime(ag)
    rt.start_callhome(nsr.claim_invite(ag, inv["invite"]))
    sc = _fake_scorer(tmp_path)
    m = studio.find_model(CODER)
    studio.STATE.update(cett=sc.binary, traces_dir=str(tmp_path / "traces"))
    studio._SCORERS[(m["path"], str(tmp_path / "clf.npz"))] = sc
    try:
        for _ in range(100):
            if call(studio_srv, "/api/remote")[1]["hosts"][0].get("online"):
                break
            time.sleep(0.05)
        ask = {"model": "rig:tiny-chat", "text": "abcdef", "messages": [{"role": "user", "content": "hi"}]}
        # 1. the host has the very file Studio has: Studio traces it itself, with that model's classifier
        same = tmp_path / "host-models" / "tiny-chat.gguf"
        same.parent.mkdir()
        shutil.copy(m["path"], same)
        ag.cfg["model_files"] = {"tiny-chat": str(same)}
        st, r = call(studio_srv, "/api/trace", ask)
        assert st == 400 and "classifier" in r["error"]
        cfg = studio.load_settings()
        cfg.setdefault(studio._key(m["path"]), {})["classifier"] = str(tmp_path / "clf.npz")
        studio.save_settings(cfg)
        st, r = call(studio_srv, "/api/trace", ask)
        assert st == 200, r
        assert r["traced_on"] == "this computer (same file)" and r["host"] == "rig" and r["pieces"] == list("abcdef")
        # 2. a different file: the host runs its tracer, with the classifier Studio sends
        other = same.with_name("tiny-chat-other.gguf")
        other.write_bytes(b"GGUF" + b"\1" * 64)
        ag.cfg["model_files"] = {"tiny-chat": str(other)}
        st, r = call(studio_srv, "/api/trace", ask)
        assert st == 400 and "classifier" in r["error"]           # no twin and none set for the host model
        st, r = call(studio_srv, "/api/remote/classifier", {"host": "rig", "model": "tiny-chat",
                                                             "classifier": str(tmp_path / "clf.npz")})
        assert st == 200
        ag.cfg["tracer"] = {"neuronscope": str(tmp_path / "no-neuronscope-here")}     # whatever this machine has
        st, r = call(studio_srv, "/api/trace", ask)
        assert st == 502 and "no OmniCortex tracer" in r["error"]
        fake = tmp_path / "ns" / "scripts" / "hscore.py"
        fake.parent.mkdir(parents=True)
        fake.write_text(FAKE_TRACER.format(scripts=str(ROOT / "scripts")))
        ag.cfg["tracer"] = {"neuronscope": str(tmp_path / "ns"), "cett": sc.binary}
        st, r = call(studio_srv, "/api/trace", ask)
        assert st == 200, r
        assert r["traced_on"] == "rig" and r["pieces"] == list("abcdef") and len(r["prob"]) == 6
        code, meta = call(studio_srv, f"/viz/{r['id']}/api/meta")
        assert code == 200 and meta["labels"] == list("abcdef")
        # the host owner can switch it off
        ag.cfg["allow"]["trace"] = False
        st, r = call(studio_srv, "/api/trace", ask)
        assert st == 502 and "does not allow" in r["error"]
    finally:
        studio._SCORERS.clear()
        studio.STATE.update(cett=None)
        for t in ag.callhome.values():
            t.stop.set()


def test_jobs_run_on_a_host_when_its_owner_allows(studio_srv, tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend, mode="lan", bind="127.0.0.1")
    try:
        code = ag.pairing.open()["code"]
        st, out = call(studio_srv, "/api/remote/pair", {"method": "lan", "address": url, "code": code})
        assert st == 200, out
        _, targets = call(studio_srv, "/api/jobs/targets")
        host = next(t for t in targets if t["id"] == "host:gpu-box")
        assert targets[0]["id"] == "local" and not host["ok"] and "not allowed" in host["why"]
        st, r = call(studio_srv, "/api/jobs", {"kind": "split", "values": {}, "target": "host:gpu-box"})
        assert st == 502 and "does not run jobs" in r["error"]
        # the host's owner allows jobs; the agent finds the OmniCortex folder it runs from
        ag.cfg["allow"]["jobs"] = True
        ag.cfg["jobs_cwd"] = str(tmp_path)
        import studio
        studio._RCACHE.clear()
        _, targets = call(studio_srv, "/api/jobs/targets")
        host = next(t for t in targets if t["id"] == "host:gpu-box")
        assert host["ok"] and "split" in host["kinds"]
        st, j = call(studio_srv, "/api/jobs", {"kind": "split", "target": "host:gpu-box",
                                               "values": {"input_path": str(tmp_path / "missing.jsonl"),
                                                          "output_path": str(tmp_path / "ids.json")}})
        assert st == 200 and j["host"] == "gpu-box" and j["status"] == "running"
        for _ in range(100):
            studio._RCACHE.pop("jobs:gpu-box", None)
            _, js = call(studio_srv, "/api/jobs")
            mine = [x for x in js if x.get("host") == "gpu-box" and x["id"] == j["id"]]
            if mine and mine[0]["status"] != "running":
                break
            time.sleep(0.1)
        assert mine[0]["status"] == "failed" and mine[0]["hint"]          # a reason, from the host's own runner
        st, log = call(studio_srv, f"/api/jobs/host/gpu-box/{j['id']}/log")
        assert st == 200 and "missing.jsonl" in log["log"]
        # bad job kinds are refused by the host's own checks
        st, r = call(studio_srv, "/api/jobs", {"kind": "rm -rf", "values": {}, "target": "host:gpu-box"})
        assert st == 502 and "unknown job kind" in r["error"]
        # the System page lists the host's agent; restarting is up to the host's owner
        _, sv = call(studio_srv, "/api/services")
        assert any(s["id"] == "host:gpu-box" and "restart" in s["actions"] for s in sv["services"])
        ag.cfg["allow"]["restart"] = False
        st, r = call(studio_srv, "/api/services", {"id": "host:gpu-box", "action": "restart"})
        assert st == 400 and "does not allow" in r["error"]
    finally:
        rt.close_api()


def test_host_gives_the_computer_back_to_its_owner(tmp_path, backend):
    ag, rt, url = make_agent(tmp_path, backend)
    try:
        snap = {"idle_s": 999, "gamepad": False, "vr": False}
        g = nsr.PresenceGuard(ag, read=lambda: dict(snap))
        ag.presence = g
        g.check()
        assert ag.busy is None                                         # off unless the owner turns it on
        ag.cfg["yield"] = {"enabled": True, "idle_under": 60, "cooldown": 300}
        snap["idle_s"] = 4
        g.check()
        assert ag.busy and ag.busy["source"] == "owner" and "keyboard or mouse" in ag.busy["why"]
        r = ag.call_tool("chat", {"messages": [{"role": "user", "content": "hi"}]}, "t")
        assert r["isError"] and r["content"][0]["text"].startswith(nsr.BUSY_PREFIX)
        assert ag.call_tool("set_busy", {"busy": False}, "t")["structuredContent"]["busy"]      # not a controller's to end
        snap["idle_s"] = 120
        g.check()
        assert ag.busy                                                 # quiet, but not for the cooldown yet
        g.last_seen -= 400
        g.check()
        assert ag.busy is None
        assert not ag.call_tool("chat", {"messages": [{"role": "user", "content": "hi"}]}, "t")["isError"]
        snap.update(idle_s=None, vr=True)
        g.check()
        assert "VR" in ag.busy["why"]
        # a controller's rule (power) can make it rest, for a while
        ag.set_busy(None)
        ag.cfg["yield"]["enabled"] = False
        r = ag.call_tool("set_busy", {"busy": True, "minutes": 30, "reason": "monthly energy cap"}, "t")
        assert not r["isError"] and ag.busy["until"] > time.time()
        assert ag.call_tool("t" and "telemetry", {}, "t")["structuredContent"]["busy"]["why"] == "monthly energy cap"
        ag.busy["until"] = time.time() - 1                                 # its time is up
        assert not ag.call_tool("chat", {"messages": [{"role": "user", "content": "hi"}]}, "t")["isError"]
    finally:
        rt.close_api()
