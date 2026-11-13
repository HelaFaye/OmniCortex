"""Connectors: MCP servers added, edited, turned off and removed from Settings; Claude set-up on Connect."""
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import mcp_client  # noqa: E402
import oc_connect  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401

FAKE = str(ROOT / "tests" / "fake_mcp_server.py")


def test_config_edits_keep_secrets_and_other_keys(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"other": 1, "mcpServers": {"web": {"url": "https://h/mcp", "timeout": 9,
                                                                 "headers": {"X-Key": "s3cret"}}}}))
    assert mcp_client.public_spec(json.loads(p.read_text())["mcpServers"]["web"])["headers"] == {"X-Key": mcp_client.MASK}
    # The page sends the mask back for a header it did not change: the saved value stays.
    mcp_client.save_server(p, "web", {"url": "https://h2/mcp", "headers": {"X-Key": mcp_client.MASK}})
    cfg = json.loads(p.read_text())
    assert cfg["other"] == 1 and cfg["mcpServers"]["web"] == {"url": "https://h2/mcp", "timeout": 9,
                                                             "headers": {"X-Key": "s3cret"}}
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    mcp_client.save_server(p, "files", {"command": "npx", "args": ["-y", "srv"], "env": {"A": "${env:HOME}"}})
    assert mcp_client.public_spec(json.loads(p.read_text())["mcpServers"]["files"])["env"] == {"A": "${env:HOME}"}
    mcp_client.save_server(p, "files2", {"command": "npx"}, old_name="files")      # rename
    assert set(json.loads(p.read_text())["mcpServers"]) == {"web", "files2"}
    mcp_client.set_disabled(p, "web", True)
    assert json.loads(p.read_text())["mcpServers"]["web"]["disabled"] is True
    mcp_client.remove_server(p, "files2")
    for bad in [("x y", {"url": "https://h"}), ("ok", {"url": "ftp://h"}), ("ok", {}),
                ("ok", {"url": "https://h", "headers": {"A": "1\r\nB: 2"}}), ("ok", {"command": "x", "env": {"1A": "v"}}),
                ("web", {"url": "https://h", "headers": {"New": mcp_client.MASK}})]:
        try:
            mcp_client.save_server(p, *bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_connectors_api(studio_srv, tmp_path):
    import studio
    studio.STATE.update(mcp_config=None, mcp_config_path=str(tmp_path / "mcp.json"), mcp=None)
    code, c = call(studio_srv, "/api/connectors")
    assert code == 200 and c["servers"] == [] and not c["exists"]
    code, c = call(studio_srv, "/api/connectors/save",
                   {"name": "calc", "spec": {"command": sys.executable, "args": [FAKE]}})
    assert code == 200 and c["exists"] and c["servers"][0]["name"] == "calc"
    assert c["servers"][0]["connected"] and c["servers"][0]["tools"]
    code, t = call(studio_srv, "/api/mcp/test", {"server": "calc"})
    assert code == 200 and t["ok"]
    code, c = call(studio_srv, "/api/connectors/toggle", {"name": "calc", "disabled": True})
    assert c["servers"][0]["disabled"] and not c["servers"][0]["connected"]
    code, r = call(studio_srv, "/api/connectors/save", {"name": "bad name", "spec": {"url": "https://h"}})
    assert code == 400
    code, c = call(studio_srv, "/api/connectors/remove", {"name": "calc"})
    assert code == 200 and c["servers"] == []
    _, page = call(studio_srv, "/settings")
    assert 'id="connectors"' in page and "Add connector" in page
    studio.STATE.update(mcp=None, mcp_config=None)


def test_claude_instructions():
    s = oc_connect.snippets("http://127.0.0.1:7870", token="t")
    assert s["claude_code"]["http"].startswith("claude mcp add --transport http --scope user omnicortex "
                                               "http://127.0.0.1:7870/mcp")
    assert '--header "Authorization: Bearer <your Studio token>"' in s["claude_code"]["http"]
    assert " -- " in s["claude_code"]["stdio"] and "oc_mcp.py" in s["claude_code"]["stdio"]
    assert "--header" not in oc_connect.snippets("http://127.0.0.1:7870")["claude_code"]["http"]
