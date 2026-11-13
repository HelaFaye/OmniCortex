"""The transfer MCP server over HTTP (--http): token, origin and throttling rules, real MCP
clients (the official one and Studio's hub with token_file), tool annotations, and refusals."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("mcp")
pytest.importorskip("uvicorn")
import mcp_client  # noqa: E402
import oc_transfer_mcp as transfer_mcp  # noqa: E402

TOKEN = "t" * 48


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def server(tmp_path):
    tok = tmp_path / "mcp.token"
    tok.write_text(TOKEN + "\n")
    port = _port()
    env = {**os.environ, "OC_CONFIG_DIR": str(tmp_path / "cfg")}
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "oc_transfer_mcp.py"), "--http", "--port", str(port),
                          "--token-file", str(tok)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    for _ in range(150):
        try:
            urllib.request.urlopen(base + "/health", timeout=0.3)
            break
        except OSError:
            if p.poll() is not None:
                raise AssertionError(p.stderr.read().decode())
            time.sleep(0.1)
    yield base, tok
    p.terminate()
    p.wait(10)


def _post(url, headers):
    req = urllib.request.Request(url, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                                          **headers})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def test_http_mode_refuses_without_the_token_or_from_other_sites(server):
    base, _ = server
    assert _post(base + "/mcp", {}) == 401
    assert _post(base + "/mcp", {"Authorization": "Bearer wrong"}) == 401
    assert _post(base + "/mcp", {"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example"}) == 403
    for _ in range(20):
        _post(base + "/mcp", {"Authorization": "Bearer guess"})
    assert _post(base + "/mcp", {"Authorization": "Bearer guess"}) == 429           # too many wrong tokens


def test_official_client_and_studios_hub(server, tmp_path):
    base, tok = server

    async def official():
        from mcp import Client
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {TOKEN}"})
        async with Client(streamable_http_client(base + "/mcp", http_client=http)) as c:
            tools = (await c.list_tools()).tools
            res = await c.call_tool("list_targets", {})
            return tools, res
    tools, res = asyncio.run(official())
    names = {t.name for t in tools}
    assert len(names) == 9 and {"list_targets", "send_file"} <= names
    lt = next(t for t in tools if t.name == "list_targets")
    assert mcp_client._attr(lt.annotations, "read_only_hint", "readOnlyHint") is True
    assert not mcp_client._attr(res, "is_error", "isError")
    assert json.loads(res.content[0].text) == {"targets": {}}

    # Studio's hub, with the token named by file rather than written into mcp.json
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"box": {"url": base + "/mcp", "token_file": str(tok)}}}))
    hub = mcp_client.MCPHub(cfg)
    hub.connect()
    st = hub.status()[0]
    assert st["connected"] and len(st["tools"]) == 9, st
    t = hub.test("box")
    assert t["ok"] and t["tools"] == 9 and t["called"] in ("list_targets",) and t["call_ok"]
    info = hub.tool_info("box__list_targets")
    assert info["annotations"]["readOnlyHint"] is True                     # mcp 2.x snake_case is read
    assert hub.tool_info("box__remove_target")["annotations"]["destructiveHint"] is True
    assert hub.test("nope")["ok"] is False
    # a wrong token file: the server reports down, with the reason
    bad = tmp_path / "bad.token"
    bad.write_text("x" * 48)
    cfg.write_text(json.dumps({"mcpServers": {"box": {"url": base + "/mcp", "token_file": str(bad)}}}))
    hub.connect()
    st = hub.status()[0]
    assert not st["connected"] and st["error"]


def test_token_references():
    os.environ["OC_TEST_TOKEN"] = "abc"
    assert mcp_client.expand("Bearer ${env:OC_TEST_TOKEN}") == "Bearer abc"
    with pytest.raises(ValueError):
        mcp_client.expand("${env:OC_NO_SUCH_VAR_123}")
    with pytest.raises(ValueError):
        mcp_client.expand("${file:/no/such/file}")
    assert mcp_client.http_headers({"token_env": "OC_TEST_TOKEN", "headers": {"X-A": "1"}}) == {
        "X-A": "1", "Authorization": "Bearer abc"}


def test_http_mode_refuses_unsafe_setups(tmp_path):
    script = str(ROOT / "scripts" / "oc_transfer_mcp.py")
    short = tmp_path / "short.token"
    short.write_text("abc")
    r = subprocess.run([sys.executable, script, "--http", "--token-file", str(short)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode != 0 and "at least" in (r.stderr + r.stdout)
    good = tmp_path / "good.token"
    good.write_text(TOKEN)
    r = subprocess.run([sys.executable, script, "--http", "--host", "0.0.0.0", "--port", str(_port()),
                        "--token-file", str(good)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "plaintext" in (r.stderr + r.stdout)



def test_mcp_rejects_unsafe_target_names(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer_mcp, "SECRETS", tmp_path / "secrets")
    monkeypatch.setattr(transfer_mcp, "_keyring", lambda: None)

    victim = tmp_path / "victim.token"
    victim.write_text("protected-token")

    invalid_names = (
        "../victim",
        "..",
        ".",
        "/tmp/victim",
        "nested/name",
        "nested\\\\name",
        "",
    )

    for name in invalid_names:
        with pytest.raises(ValueError):
            transfer_mcp.validate_target_name(name)
        with pytest.raises(ValueError):
            transfer_mcp.store_token(name, TOKEN)
        with pytest.raises(ValueError):
            transfer_mcp.load_token(name)
        with pytest.raises(ValueError):
            transfer_mcp.delete_token(name)
        with pytest.raises(ValueError):
            transfer_mcp.get_target(name)

    assert victim.read_text() == "protected-token"
    assert not (tmp_path / "secrets").exists()


def test_mcp_state_save_does_not_follow_predictable_tmp_symlink(tmp_path, monkeypatch):
    state = tmp_path / "config" / "transfer-mcp.json"
    monkeypatch.setattr(transfer_mcp, "STATE", state)

    state.parent.mkdir(parents=True)
    victim = tmp_path / "victim.json"
    victim.write_text("protected-state")

    # This was the old predictable temporary filename.
    predictable_tmp = state.with_suffix(".tmp")
    try:
        predictable_tmp.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    transfer_mcp.save({
        "targets": {
            "box": {
                "url": "https://example.invalid",
                "cafile": "",
                "destination_hint": "",
            }
        }
    })

    assert json.loads(state.read_text())["targets"]["box"]["url"] == "https://example.invalid"
    assert predictable_tmp.is_symlink()
    assert victim.read_text() == "protected-state"

    if os.name == "posix":
        assert state.stat().st_mode & 0o777 == 0o600
        assert state.parent.stat().st_mode & 0o777 == 0o700


def test_mcp_state_save_replaces_state_symlink_safely(tmp_path, monkeypatch):
    state = tmp_path / "transfer-mcp.json"
    monkeypatch.setattr(transfer_mcp, "STATE", state)

    victim = tmp_path / "victim.json"
    victim.write_text("do not overwrite")
    try:
        state.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    transfer_mcp.save({"targets": {}})

    assert not state.is_symlink()
    assert json.loads(state.read_text()) == {"targets": {}}
    assert victim.read_text() == "do not overwrite"


@pytest.mark.parametrize(
    "contents",
    [
        "{malformed json",
        "[]",
        '{"targets": []}',
        '{"targets": null}',
    ],
)
def test_mcp_load_fails_closed_on_malformed_state(tmp_path, monkeypatch, contents):
    state = tmp_path / "transfer-mcp.json"
    monkeypatch.setattr(transfer_mcp, "STATE", state)
    state.write_text(contents)

    assert transfer_mcp.load() == {"targets": {}}


def test_mcp_load_removes_unsafe_legacy_target_names(tmp_path, monkeypatch):
    state = tmp_path / "transfer-mcp.json"
    secrets = tmp_path / "secrets"
    monkeypatch.setattr(transfer_mcp, "STATE", state)
    monkeypatch.setattr(transfer_mcp, "SECRETS", secrets)
    monkeypatch.setattr(transfer_mcp, "_keyring", lambda: None)

    victim = tmp_path / "victim.token"
    victim.write_text("protected-token")

    state.write_text(json.dumps({
        "targets": {
            "../victim": {"url": "https://invalid.example", "token": TOKEN},
            "box": {"url": "https://example.invalid", "cafile": ""},
        }
    }))

    loaded = transfer_mcp.load()

    assert set(loaded["targets"]) == {"box"}
    assert "token" not in loaded["targets"]["box"]
    assert victim.read_text() == "protected-token"
    assert json.loads(state.read_text())["targets"].keys() == {"box"}


def test_secret_writer_does_not_follow_predictable_tmp_symlink(tmp_path):
    target = tmp_path / "secret.token"
    victim = tmp_path / "victim.txt"
    victim.write_text("protected")

    predictable_tmp = target.with_name(target.name + ".tmp")
    try:
        predictable_tmp.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    transfer_mcp.sec.write_secret_file(target, TOKEN)

    assert target.read_text().strip() == TOKEN
    assert predictable_tmp.is_symlink()
    assert victim.read_text() == "protected"

    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o600
        assert target.parent.stat().st_mode & 0o777 == 0o700
