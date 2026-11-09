"""Workspaces (scripts/oc_workspace.py) and tool approval rules
(scripts/tool_rules.py)."""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import oc_workspace as ws  # noqa: E402
import tool_rules as tr  # noqa: E402


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    monkeypatch.setenv("OC_CONFIG", str(p))
    monkeypatch.delenv("OC_WORKSPACE", raising=False)
    return p


def test_default_workspace_and_legacy_paths(cfg, tmp_path):
    a = ws.active()
    assert a["name"] == "default" and a["root"] == str(ROOT)
    assert a["traces_dir"].endswith("/.omnicortex/traces")
    # Paths the Setup page saved before workspaces existed still count.
    models = tmp_path / "m"
    models.mkdir()
    cfg.write_text(json.dumps({"studio": {"models_dir": [str(models)], "idle_ttl": 3}}))
    assert ws.active()["models_dirs"] == [str(models)]


def test_add_use_env_and_validation(cfg, tmp_path):
    root, models, llama = tmp_path / "root", tmp_path / "models", tmp_path / "llama.cpp"
    for d in (root, models, llama / "build" / "bin"):
        d.mkdir(parents=True)
    srv = llama / "build" / "bin" / "llama-server"
    srv.write_text("#!/bin/sh\n")
    srv.chmod(0o755)
    ws.save("ext", {"root": str(root), "models_dirs": [str(models)], "llama_dir": str(llama),
                    "state_dir": str(tmp_path / "state"), "env": {"VK_DRIVER_FILES": "/usr/share/x.json"}})
    assert "ext" in ws.names() and ws.active_name() == "default"
    ws.use("ext")
    a = ws.active()
    assert a["name"] == "ext" and a["server"] == str(srv) and a["review_dir"] == str(tmp_path / "state" / "review")
    env = ws.env_for(a)
    assert env["OC_ROOT"] == str(root) and env["OC_LLAMA_SERVER"] == str(srv) and env["VK_DRIVER_FILES"]
    assert "OC_CETT" not in env                         # not built: not exported
    for bad in ({"root": str(tmp_path / "missing")}, {"env": {"PATH": "/x"}}, {"env": {"lower": "x"}},
                {"server": str(models)}, {"nonsense": 1}, {"models_dirs": [str(tmp_path / "nope")]}):
        with pytest.raises(ValueError):
            ws.save("ext", bad)
    with pytest.raises(ValueError):
        ws.save("../evil", {})
    ws.save("ext", {"env": None})                       # empty clears
    assert ws.raw("ext").get("env") is None
    with pytest.raises(ValueError):
        ws.delete("default")
    ws.delete("ext")
    assert ws.active_name() == "default"


def test_cli_env(cfg, tmp_path, capsys):
    (tmp_path / "r").mkdir()
    assert ws.main(["add", "w2", "--root", str(tmp_path / "r"), "--use"]) == 0
    assert ws.main(["set", "w2", "env.HSA_OVERRIDE_GFX_VERSION", "9.0.0"]) == 0
    capsys.readouterr()
    assert ws.main(["env"]) == 0
    out = capsys.readouterr().out
    assert f"export OC_ROOT={tmp_path / 'r'}" in out and "export HSA_OVERRIDE_GFX_VERSION=9.0.0" in out
    assert ws.main(["set", "w2", "env.PATH", "/x"]) == 2


def test_rules_order_kinds_and_conditions(tmp_path):
    R = tr.Rules(tmp_path / "rules.json")
    assert R.decide("fs", "read_file", {"path": "/x"}, {"readOnlyHint": True})["action"] == "ask"
    d = R.load()
    d["categories"]["read_only"] = "allow"
    d["categories"]["destructive"] = "deny"
    R.replace(d)
    assert R.decide("fs", "read_file", {}, {"readOnlyHint": True})["by"] == "category"
    assert R.decide("fs", "rm", {}, {"readOnlyHint": False})["action"] == "deny"        # MCP default: destructive
    assert R.decide("fs", "write", {}, {"readOnlyHint": False, "destructiveHint": False})["action"] == "ask"
    # Read-only but open world: the stricter of the two kinds' settings.
    d = R.load()
    d["categories"]["open_world"] = "ask"
    R.replace(d)
    assert R.decide("web", "search", {}, {"readOnlyHint": True, "openWorldHint": True})["action"] == "ask"
    # Rules come first, top to bottom; argument conditions with paths, wildcards and negation.
    R.add({"id": "notes", "action": "allow", "server": "fs", "tool": "read_*",
           "args": [{"path": "path", "regex": "^/home/me/notes/"}]})
    R.add({"id": "secrets", "action": "deny", "server": "*", "tool": "*",
           "args": [{"path": "*", "regex": r"\.ssh/"}]})
    assert R.decide("fs", "read_file", {"path": "/home/me/notes/a"}, {})["rule"] == "notes"
    assert R.decide("fs", "read_file", {"path": "/home/me/.ssh/id"}, {})["rule"] == "secrets"
    assert R.decide("fs", "read_file", {"path": "/etc/passwd"}, {})["by"] == "default"
    R.add({"id": "batch", "action": "allow", "tool": "move",
           "args": [{"path": "files.*.to", "regex": "^/tmp/"}, {"path": "force", "regex": "true", "negate": True}]})
    assert R.decide("fs", "move", {"files": [{"to": "/etc/x"}, {"to": "/tmp/y"}]}, {})["rule"] == "batch"
    assert R.decide("fs", "move", {"files": [{"to": "/tmp/y"}], "force": True}, {})["by"] != "rule"
    # autoApprove from mcp.json sits between rules and quick settings.
    assert R.decide("calc", "add", {}, {}, auto_approve=["add"])["by"] == "autoApprove"
    assert R.load()["rules"][-1]["hits"] >= 1


def test_rate_limit_expiry_and_validation(tmp_path):
    R = tr.Rules(tmp_path / "rules.json")
    R.add({"id": "two", "action": "allow", "tool": "ping", "max_per_hour": 2})
    t0 = 1_000_000.0
    assert [R.decide("s", "ping", {}, {}, now=t0 + i)["action"] for i in range(3)] == ["allow", "allow", "ask"]
    assert R.decide("s", "ping", {}, {}, now=t0 + 3700)["action"] == "allow"
    R.add({"id": "old", "action": "deny", "tool": "pong", "expires": time.time() - 1})
    assert R.decide("s", "pong", {}, {})["by"] == "default"
    for bad in ({"action": "maybe"}, {"action": "allow", "args": [{"path": "x", "regex": "("}]},
                {"action": "deny", "max_per_hour": 3}, {"action": "allow", "kind": "weird"},
                {"action": "allow", "args": [{"path": "a b", "regex": "x"}]},
                {"action": "allow", "args": [{"path": "x", "regex": "a" * 600}]}):
        with pytest.raises(ValueError):
            tr.validate_rule(bad)
    with pytest.raises(ValueError):
        R.replace({"default": "ask", "rules": [{"id": "a", "action": "allow"}, {"id": "a", "action": "deny"}]})
    # A hand-edited broken rule is skipped, not fatal.
    d = json.loads((tmp_path / "rules.json").read_text())
    d["rules"].append({"id": "broken", "action": "allow", "args": [{"path": "x", "regex": "("}]})
    (tmp_path / "rules.json").write_text(json.dumps(d))
    assert "broken" not in [r["id"] for r in R.load()["rules"]]


def test_rule_from_call_matches_exactly(tmp_path):
    R = tr.Rules(tmp_path / "rules.json")
    R.add(tr.rule_from_call("args", "fs", "read_file", {"path": "/a/b.txt", "n": 3}))
    assert R.decide("fs", "read_file", {"path": "/a/b.txt", "n": 3}, {})["action"] == "allow"
    assert R.decide("fs", "read_file", {"path": "/a/b.txt.evil", "n": 3}, {})["by"] == "default"
    R.add(tr.rule_from_call("deny", "fs", "delete", {}))
    assert R.decide("fs", "delete", {"path": "x"}, {})["action"] == "deny"


def test_tool_rules_cli(tmp_path, capsys):
    f = str(tmp_path / "r.json")
    assert tr.main(["--file", f, "add", "deny", "--tool", "*", "--arg", "*=rm -rf", "--note", "no"]) == 0
    assert tr.main(["--file", f, "add", "allow", "--server", "fs", "--arg", "path!=^/etc", "--expires-in", "2h"]) == 0
    assert tr.main(["--file", f, "category", "read_only", "allow"]) == 0
    capsys.readouterr()
    assert tr.main(["--file", f, "test", "x", "shell", '{"cmd": "rm -rf /"}']) == 0
    assert capsys.readouterr().out.startswith("deny")
    assert tr.main(["--file", f, "list"]) == 0
    assert "read_only allow" in capsys.readouterr().out
