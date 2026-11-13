"""Studio plugins: discovery, opt-in with a content hash, jobs, reply checks, pages."""
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import plugins as oc_plugins  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401

EXAMPLE = ROOT / "plugins" / "claim-spotter"


def make(tmp_path, manifest, files=None):
    d = tmp_path / "plugins" / manifest.get("name", "x")
    d.mkdir(parents=True)
    (d / "plugin.json").write_text(json.dumps(manifest))
    for k, v in (files or {}).items():
        (d / k).parent.mkdir(parents=True, exist_ok=True)
        (d / k).write_text(v)
    return d


def test_manifests_are_checked(tmp_path):
    make(tmp_path, {"name": "Bad Name"})
    make(tmp_path, {"name": "escape", "checks": [{"name": "c", "script": "../../evil.py"}]})
    make(tmp_path, {"name": "badfield", "jobs": [{"kind": "j", "script": "j.py", "fields": [{"name": "x", "kind": "list"}]}]})
    pm = oc_plugins.Plugins([tmp_path / "plugins"], tmp_path / "state.json")
    rows = {r["folder"].rsplit("/", 1)[-1]: r for r in pm.scan()}
    assert all(r["status"] == "broken" for r in rows.values())
    assert "outside" in rows["escape"]["error"] and "kind" in rows["badfield"]["error"]
    with pytest.raises(oc_plugins.PluginError):
        pm.set_enabled("escape", True)


def test_opt_in_hash_jobs_and_checks(tmp_path):
    d = tmp_path / "plugins" / "claim-spotter"
    shutil.copytree(EXAMPLE, d)
    pm = oc_plugins.Plugins([tmp_path / "plugins"], tmp_path / "state.json")
    assert pm.get("claim-spotter", need_on=False)["status"] == "off"
    assert pm.job_specs() == {} and pm.checks() == []
    with pytest.raises(oc_plugins.PluginError, match="off"):
        pm.run_check("claim-spotter", "claims", {"text": "x"})
    pm.set_enabled("claim-spotter", True)
    r = pm.run_check("claim-spotter", "claims", {"text": "It opened in 1889 and is 330 m tall."})
    assert r["label"] == "2 claim(s) to verify" and [s[2] for s in r["spans"]] == ["year", "number"]
    spec = pm.job_specs()["plugin:claim-spotter:collect"]
    assert spec["group"] == "Plugins: claim-spotter" and [f["name"] for f in spec["fields"]] == ["chats", "out", "model"]
    assert pm.page_file("claim-spotter", "").name == "index.html"
    assert pm.page_file("claim-spotter", "../plugin.json") is None
    # a changed file switches it off until it is turned on again
    (d / "check.py").write_text((d / "check.py").read_text() + "\n# changed\n")
    assert pm.get("claim-spotter", need_on=False)["status"] == "changed"
    assert pm.checks() == []
    pm.set_enabled("claim-spotter", True)
    assert pm.get("claim-spotter")["status"] == "on"


def test_bad_check_output_is_reported(tmp_path):
    make(tmp_path, {"name": "noisy", "checks": [{"name": "c", "script": "c.py"}, {"name": "boom", "script": "b.py"}]},
         {"c.py": "print('hello')", "b.py": "raise SystemExit('broken here')"})
    pm = oc_plugins.Plugins([tmp_path / "plugins"], tmp_path / "state.json")
    pm.set_enabled("noisy", True)
    with pytest.raises(oc_plugins.PluginError, match="no JSON"):
        pm.run_check("noisy", "c", {})
    with pytest.raises(oc_plugins.PluginError, match="broken here"):
        pm.run_check("noisy", "boom", {})


def test_studio_plugins(studio_srv, tmp_path):
    import studio
    shutil.copytree(EXAMPLE, tmp_path / "plugins" / "claim-spotter")
    studio.STATE.update(plugin_dirs=[str(tmp_path / "plugins")], plugins=None)
    try:
        code, r = call(studio_srv, "/api/plugins")
        assert code == 200 and r["plugins"][0]["status"] == "off" and r["checks"] == []
        assert r["plugins"][0]["adds"]["checks"] == ["Claims to verify"]
        code, r = call(studio_srv, "/api/plugins/check", {"plugin": "claim-spotter", "check": "claims", "text": "1889"})
        assert code == 400 and "off" in r["error"]
        assert call(studio_srv, "/plugins/claim-spotter/")[0] == 404
        code, r = call(studio_srv, "/api/plugins/enable", {"name": "claim-spotter", "on": True})
        assert code == 200 and r["status"] == "on"
        code, r = call(studio_srv, "/api/plugins/check", {"plugin": "claim-spotter", "check": "claims",
                                                          "text": "Founded in 1901 with 12 members."})
        assert code == 200 and r["label"] == "2 claim(s) to verify"
        code, page = call(studio_srv, "/plugins/claim-spotter/")
        assert code == 200 and "Claim spotter" in page
        assert call(studio_srv, "/plugins/claim-spotter/..%2Fplugin.json")[0] == 404
        # its job is a Studio job now
        _, specs = call(studio_srv, "/api/jobs/specs")
        assert "plugin:claim-spotter:collect" in specs
        chats = tmp_path / "chats-in"
        chats.mkdir()
        (chats / "c1.json").write_text(json.dumps({"messages": [
            {"role": "user", "content": "When was it built?"},
            {"role": "assistant", "model": "m", "content": "It was finished in 1889, at a cost of 7.8 million francs."}]}))
        code, job = call(studio_srv, "/api/jobs", {"kind": "plugin:claim-spotter:collect",
                                                   "values": {"chats": str(chats), "out": str(tmp_path / "claims.csv")}})
        assert code == 200, job
        for _ in range(100):
            _, js = call(studio_srv, "/api/jobs")
            if js[0]["status"] != "running":
                break
            time.sleep(0.1)
        assert js[0]["status"] == "done", call(studio_srv, f"/api/jobs/{job['id']}/log")
        rows = (tmp_path / "claims.csv").read_text().splitlines()
        assert len(rows) == 3 and "1889" in rows[1] and "7.8 million" in rows[2]
        call(studio_srv, "/api/plugins/enable", {"name": "claim-spotter", "on": False})
        _, specs = call(studio_srv, "/api/jobs/specs")
        assert "plugin:claim-spotter:collect" not in specs
    finally:
        studio.STATE.update(plugin_dirs=None, plugins=None)


def test_plugin_mcp_servers_join_the_hub(tmp_path):
    make(tmp_path, {"name": "tools", "mcp": {"command": "python", "args": ["server.py"]}}, {"server.py": ""})
    pm = oc_plugins.Plugins([tmp_path / "plugins"], tmp_path / "state.json")
    assert pm.mcp_servers() == {}
    pm.set_enabled("tools", True)
    srv = pm.mcp_servers()["plugin-tools"]
    assert srv["command"] == "python" and srv["cwd"] == str(tmp_path / "plugins" / "tools")
    import mcp_client
    hub = mcp_client.MCPHub(tmp_path / "none.json")
    hub.extra = pm.mcp_servers
    assert "plugin-tools" in hub.load()
