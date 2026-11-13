"""System page: what runs, restarting parts, stale-code notice, forgiving page paths."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_studio import call, studio_srv  # noqa: E402,F401


def get_html(base, path):
    req = urllib.request.Request(base + path, headers={"Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_services_and_actions(studio_srv):
    code, j = call(studio_srv, "/api/services")
    ids = [s["id"] for s in j["services"]]
    assert code == 200 and ids[:2] == ["studio", "chat"] and {"workers", "embed", "mcp", "labs", "jobs"} <= set(ids)
    assert j["stale"] is False
    code, r = call(studio_srv, "/api/services", {"id": "workers", "action": "stop"})
    assert code == 200 and "stopped" in r["note"]
    code, r = call(studio_srv, "/api/services", {"id": "chat", "action": "reload"})
    assert code == 400 and "no model" in r["error"]
    code, r = call(studio_srv, "/api/services", {"id": "nope", "action": "explode"})
    assert code == 400


def test_page_paths_and_not_found(studio_srv, monkeypatch):
    import studio
    for p in ("/workers", "/workers/", "/workers?tab=1", "/system", "/jobs/"):
        assert get_html(studio_srv, p)[0] == 200, p
    code, page = get_html(studio_srv, "/no-such-page")
    assert code == 404 and "not in the running Studio" in page and "nsRestart" in page
    assert call(studio_srv, "/api/no-such")[0] == 404                       # APIs still answer JSON
    monkeypatch.setitem(studio.STARTED, "checked", 0)
    monkeypatch.setitem(studio.STARTED, "code", 0)                          # as if the code changed since start
    _, st = call(studio_srv, "/api/status")
    assert st["stale"] is True
    code, page = get_html(studio_srv, "/new-page")
    assert "changed after this Studio started" in page


def test_monitor_and_automation_api(studio_srv, monkeypatch):
    import json as _json
    import threading as _th
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import studio

    class HA(BaseHTTPRequestHandler):                     # a stand-in Home Assistant
        def log_message(self, *a):
            pass

        def do_GET(self):
            ok = self.headers.get("Authorization") == "Bearer secret"
            if self.path.startswith("/api/states/sensor.rig_power"):
                body = {"state": "420", "attributes": {"unit_of_measurement": "W", "friendly_name": "Rig power"}}
            elif self.path.startswith("/api/history/"):
                body = [[]]
            else:
                body = {}
            b = _json.dumps(body).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
    ha = ThreadingHTTPServer(("127.0.0.1", 0), HA)
    _th.Thread(target=ha.serve_forever, daemon=True).start()
    monkeypatch.setattr(studio, "BUSY", {})
    try:
        code, m = call(studio_srv, "/api/system/monitor")
        assert code == 200 and "local" in m and "presence" in m
        code, a = call(studio_srv, "/api/automation")
        assert code == 200 and a["rules"] == [] and a["targets"][0] == "local" and "month" in a["periods"]
        code, r = call(studio_srv, "/api/automation/ha", {"url": f"http://127.0.0.1:{ha.server_port}", "token": "secret",
                                                          "test_entity": "sensor.rig_power"})
        assert code == 200 and r["reading"]["value"] == 420.0 and r["reading"]["unit"] == "W"
        _, a = call(studio_srv, "/api/automation")
        assert a["ha"]["has_token"] and "secret" not in str(a)                     # the token never comes back
        rule = {"name": "cap", "target": "local", "then": ["busy", "throttle"],
                "when": {"kind": "power", "entity": "sensor.rig_power", "measure": "power", "period": "minute", "limit": 350}}
        code, r = call(studio_srv, "/api/automation/rules", {"rules": [rule]})
        assert code == 200, r
        res = list(r["results"].values())[0]
        assert res["active"] and "420 W" in res["why"]
        assert studio.target_busy("local") and not studio.worker_alive("coder-gguf/coder-7b-q4_k_m")
        call(studio_srv, "/api/automation/rules", {"rules": []})                     # rule gone: released
        studio.automation_tick()
        assert not studio.target_busy("local")
        # the automation API, as Home Assistant or a script would call it
        code, r = call(studio_srv, "/api/automation/action", {"target": "local", "action": "busy", "minutes": 5,
                                                              "reason": "gaming"})
        assert code == 200 and r["busy"]["why"] == "gaming" and r["busy"]["until"]
        _, a = call(studio_srv, "/api/automation")
        assert a["busy"]["local"]["why"] == "gaming" and a["log"]
        call(studio_srv, "/api/automation/action", {"target": "local", "action": "free"})
        assert not studio.target_busy("local")
        for bad in ({"target": "elsewhere", "action": "busy"}, {"target": "local", "action": "explode"}):
            assert call(studio_srv, "/api/automation/action", bad)[0] == 400
        assert call(studio_srv, "/api/automation/rules", {"rules": [{"when": {"kind": "weather"}}]})[0] == 400
    finally:
        ha.shutdown()


def test_yielded_attempt_is_not_a_failure():
    import director as dr
    plan = {"policy": {"max_attempts": 2}, "tasks": [], "updated": 0}
    t = {"id": "t1", "status": "todo", "attempts": [], "assignee": {"model": "rig:m"}, "depends_on": [], "questions": []}
    plan["tasks"].append(t)
    for _ in range(3):                                   # taken back three times: still not blocked
        dr.start_attempt(plan, t)
        dr.yield_attempt(plan, t, "the host's owner is using it")
    assert t["status"] == "todo" and t["not_before"] == 0 and t["avoid"] == ["rig:m"]
    assert all(a["yielded"] and not a.get("error") for a in t["attempts"])
