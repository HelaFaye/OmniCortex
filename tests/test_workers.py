"""Spreading work across workers: the registry, routing, pause/drain/limits, endpoints, the
Workers page's API, GPU telemetry and the remote agent's thermal guard."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "remote"))
import worker_registry as wr  # noqa: E402
from omnicortex_remote import agent as nsr  # noqa: E402
from test_studio import CODER, _start_director, _wait, call, free_port, studio_srv  # noqa: E402,F401


def test_registry_endpoints_controls_and_routes(tmp_path):
    reg = wr.Registry(tmp_path / "workers.json")
    with pytest.raises(wr.RegistryError):
        reg.put_endpoint({"name": "bad name", "url": "http://x"})
    with pytest.raises(wr.RegistryError):
        reg.put_endpoint({"name": "x", "url": "ftp://x"})
    out = reg.put_endpoint({"name": "heavy", "url": "http://10.0.0.5:8082", "token": "s3cret", "max_parallel": 2})
    assert out["url"] == "http://10.0.0.5:8082/v1" and out["has_token"] and "token" not in out
    assert oct(os.stat(tmp_path / "workers.json").st_mode & 0o777) == "0o600"
    reg.put_endpoint({"name": "heavy", "url": "http://10.0.0.5:8082/v1"})        # token kept when not given
    assert reg.endpoint("ep:heavy")["token"] == "s3cret"
    # limits: endpoints default to their max_parallel; local models to none of their own
    assert reg.limit("ep:heavy") == 1 and reg.limit("some-model") is None
    reg.set_control("ep:heavy", max_parallel=2)
    t0, t1 = reg.begin("ep:heavy", "a"), reg.begin("ep:heavy", "b")
    assert not reg.has_slot("ep:heavy")
    reg.end("ep:heavy", t0, "a", tokens=50)
    assert reg.has_slot("ep:heavy")
    # drain: no new work, and paused once the last task ends
    reg.set_control("ep:heavy", draining=True)
    assert not reg.has_slot("ep:heavy")
    reg.end("ep:heavy", t1, "b", error="boom")
    c = reg.control("ep:heavy")
    assert c["paused"] and not c["draining"]
    s = reg.summary("ep:heavy")
    assert s["done"] == 1 and s["failed"] == 1 and s["last_error"] == "boom" and s["inflight"] == 0
    # routes: first worker on the subject's route that is not paused or down
    reg.set_routes({"code": ["ep:heavy", "local-7b"], "math": ["ep:heavy"]}, ["code", "math", "writing"])
    with pytest.raises(wr.RegistryError):
        reg.set_routes({"cooking": ["x"]}, ["code"])
    assert reg.route(["code"]) == {"model": "local-7b", "reason": "route for code"}       # heavy is paused
    reg.set_control("ep:heavy", paused=False)
    assert reg.route(["code"])["model"] == "ep:heavy"
    assert reg.route(["code"], alive=lambda w: w != "ep:heavy")["model"] == "local-7b"
    assert reg.route(["writing"]) is None
    # subjects a worker takes
    reg.set_control("local-7b", takes=["writing"])
    assert reg.takes("local-7b", ["writing", "code"]) and not reg.takes("local-7b", ["math"])
    reg.remove_endpoint("heavy")
    assert reg.endpoints() == [] and "ep:heavy" not in reg.load()["routes"]["code"]
    assert [e["what"] for e in reg.events][:2] == ["endpoint saved", "endpoint saved"]


def test_thermal_guard_pauses_and_resumes(tmp_path):
    ag = nsr.Agent(tmp_path / "host")
    ag.cfg.update(detect=False, backends=[], thermal={"enabled": True, "limit_c": 95, "margin_c": 10})
    temps = {"j": 70.0}
    reads = lambda: [{"name": "Test GPU", "junction_c": temps["j"], "temp_c": 60.0, "mem_c": None}]  # noqa: E731
    g = nsr.ThermalGuard(ag, read=reads)
    ag.guard = g
    unloaded = []
    g.unload_all = lambda: unloaded.append(1)
    g.check()
    assert g.tripped is None
    temps["j"] = 96.0
    g.check()
    assert g.tripped and g.tripped["temp"] == 96.0 and unloaded == [1]
    with pytest.raises(nsr.ToolError, match="too hot"):
        ag.t_chat({"messages": [{"role": "user", "content": "hi"}]})
    out = ag.call_tool("chat", {"messages": [{"role": "user", "content": "hi"}]}, "test")
    assert out["isError"] and "too hot" in out["content"][0]["text"]
    temps["j"] = 90.0
    g.check()
    assert g.tripped                                  # still within the margin
    temps["j"] = 84.0
    g.check()
    assert g.tripped is None
    # unreadable sensors never trip it
    g.read = lambda: [{"name": "Test GPU", "junction_c": None, "temp_c": None, "mem_c": None}]
    g.check()
    assert g.tripped is None
    tel = ag.t_telemetry({})
    assert tel["thermal"]["enabled"] and tel["thermal"]["limit_c"] == 95 and tel["gpus"][0]["name"] == "Test GPU"
    # off unless turned on
    ag.cfg["thermal"] = {}
    temps["j"] = 120.0
    g.read = reads
    g.check()
    assert g.tripped is None


def test_gpu_telemetry_never_guesses():
    rows = nsr.gpu_telemetry()
    for r in rows:
        assert set(r) >= {"name", "temp_c", "junction_c", "vram_used_mb", "source"}
    assert nsr.hottest({"junction_c": None, "temp_c": None, "mem_c": None}) is None
    assert nsr.hottest({"junction_c": 80.0, "temp_c": 60.0, "mem_c": 85.0}) == 85.0


@pytest.fixture()
def endpoint_server():
    port = free_port()
    p = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_llama_server.py"), "--port", str(port),
                          "--alias", "heavy-model"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2)
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    p.terminate()


def test_workers_page_routes_pauses_and_runs_on_an_endpoint(studio_srv, tmp_path, endpoint_server):
    import studio
    studio.STATE.update(workers_reg=None)
    _start_director(tmp_path)
    code, r = call(studio_srv, "/api/workers/endpoint", {"name": "heavy", "url": endpoint_server, "max_parallel": 1,
                                                         "note": "the big GPU"})
    assert code == 200, r
    assert call(studio_srv, "/api/workers/endpoint", {"name": "x y", "url": endpoint_server})[0] == 400
    code, r = call(studio_srv, "/api/workers/test", {"id": "ep:heavy"})
    assert code == 200 and r["reply"].startswith("model=heavy-model")
    code, r = call(studio_srv, "/api/workers/test", {"id": CODER})
    assert code == 400
    code, r = call(studio_srv, "/api/workers/routes", {"routes": {"graphics": ["ep:heavy"]}})
    assert code == 200 and r["routes"] == {"graphics": ["ep:heavy"]}
    text = ("- Shaders: write GLSL vertex and fragment shaders for the terrain mesh.\n"
            "- Build: set up the CMake build with the toolchain and dependencies.\n")
    _, r = call(studio_srv, "/api/projects", {"title": "Spread", "text": text})
    pid = r["id"]
    assert call(studio_srv, f"/api/projects/{pid}/approve", {})[0] == 200
    _, p = call(studio_srv, f"/api/projects/{pid}")
    t1, t2 = p["tasks"]
    assert t1["assignee"] == {"model": "ep:heavy", "reason": "route for graphics"}
    assert t2["assignee"]["model"] != "ep:heavy"
    # paused: its task waits, the other runs
    call(studio_srv, "/api/workers/control", {"id": "ep:heavy", "paused": True})
    call(studio_srv, f"/api/projects/{pid}/start", {})
    _wait(lambda: call(studio_srv, f"/api/projects/{pid}")[1]["tasks"][1]["status"] == "review")
    _, ov = call(studio_srv, "/api/workers/overview")
    q = {x["task"]: x for x in ov["queue"]}
    assert q["T1"]["why"] == "worker paused" and q["T1"]["worker"] == "ep:heavy"
    ep = next(w for w in ov["workers"] if w["id"] == "ep:heavy")
    assert ep["kind"] == "endpoint" and ep["control"]["paused"] and ep["note"] == "the big GPU"
    assert ov["subjects"] and "graphics" in ov["routes"]
    # resume: it runs there
    call(studio_srv, "/api/workers/control", {"id": "ep:heavy", "paused": False})
    p = _wait(lambda: (lambda p: p if p["tasks"][0]["status"] == "review" else None)(
        call(studio_srv, f"/api/projects/{pid}")[1]))
    assert p["tasks"][0]["attempts"][0]["text"].startswith("model=heavy-model")
    _, ov = call(studio_srv, "/api/workers/overview")
    ep = next(w for w in ov["workers"] if w["id"] == "ep:heavy")
    assert ep["stats"]["done"] == 1 and ep["stats"]["inflight"] == 0
    whats = [e["what"] for e in ov["events"] if e["worker"] == "ep:heavy"]
    assert {"assigned", "started", "finished", "tested"} <= set(whats)
    local = next(w for w in ov["workers"] if w["id"] == t2["assignee"]["model"])
    assert local["kind"] == "local" and local["stats"]["done"] >= 1
    # move a task to another worker from the page
    call(studio_srv, f"/api/projects/{pid}/review", {"task": "T1", "accept": False, "feedback": "again"})
    code, r = call(studio_srv, f"/api/projects/{pid}/edit", {"changes": [{"op": "assign", "id": "T1", "model": CODER}]})
    assert code == 200
    call(studio_srv, f"/api/projects/{pid}/pause", {})
    page = urllib.request.urlopen(studio_srv + "/workers").read().decode()
    assert "Routing" in page and "controller" in page
    call(studio_srv, "/api/workers/stop", {})
    studio.STATE.update(workers_reg=None)
