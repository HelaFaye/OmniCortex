import json
import os
import stat
import sys
import threading
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import oc_security as sec  # noqa: E402
import oc_transfer as nt  # noqa: E402


def test_generated_token_is_256_bits():
    t = sec.generate_token()
    assert len(t) >= 43
    assert t != sec.generate_token()


def test_bearer_constant_time_semantics():
    assert sec.bearer_ok("Bearer abc", "abc")
    assert not sec.bearer_ok("Bearer abd", "abc")
    assert not sec.bearer_ok("abc", "abc")
    assert not sec.bearer_ok(None, "abc")
    assert sec.bearer_ok(None, "")  # auth disabled (only allowed on loopback)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_may_run_without_token(host):
    assert sec.check_bind(host, "", tls=False, allow_plaintext=False) == []


def test_remote_bind_requires_token():
    with pytest.raises(sec.SecurityConfigError):
        sec.check_bind("0.0.0.0", "", tls=True, allow_plaintext=False)
    with pytest.raises(sec.SecurityConfigError):
        sec.check_bind("0.0.0.0", "short", tls=True, allow_plaintext=False)


def test_remote_bind_requires_tls_or_explicit_plaintext():
    tok = sec.generate_token()
    with pytest.raises(sec.SecurityConfigError):
        sec.check_bind("0.0.0.0", tok, tls=False, allow_plaintext=False)
    assert sec.check_bind("0.0.0.0", tok, tls=True, allow_plaintext=False) == []
    assert sec.check_bind("0.0.0.0", tok, tls=False, allow_plaintext=True)  # warns


def test_serve_refuses_unauthenticated_remote_bind(tmp_path):
    assert nt.main(["serve", "--host", "0.0.0.0", "--port", "0", "--root", str(tmp_path)]) == 2


def test_secret_file_permissions(tmp_path):
    p = sec.write_secret_file(tmp_path / "d" / "t.token", "s3cret")
    assert sec.read_secret_file(p) == "s3cret"
    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_resolve_token_prefers_file_then_env(tmp_path, monkeypatch):
    f = sec.write_secret_file(tmp_path / "t", "fromfile")
    monkeypatch.setenv(sec.TOKEN_ENV, "fromenv")
    assert sec.resolve_token("fromargv", str(f)) == "fromfile"
    assert sec.resolve_token("fromargv", "") == "fromenv"
    monkeypatch.delenv(sec.TOKEN_ENV)
    assert sec.resolve_token("fromargv", "", warn_argv=False) == "fromargv"


def test_throttle_locks_out():
    t = sec.FailureThrottle(max_failures=3, window=60, lockout=60)
    for _ in range(3):
        t.fail("a")
    assert t.blocked("a") and not t.blocked("b")


def _server(tmp, token):
    nt.Handler.receiver = nt.Receiver(tmp)
    nt.Handler.token = token
    nt.Handler.throttle = sec.FailureThrottle(max_failures=3, window=60, lockout=60)
    s = ThreadingHTTPServer(("127.0.0.1", 0), nt.Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def test_transfer_ids_are_128_bit_hex(tmp_path):
    s, base = _server(tmp_path, "t" * 40)
    m = nt.request(base, "POST", "/v1/transfers/init", "t" * 40,
                   {"name": "a.gguf", "size": 3, "sha256": nt.sha256_bytes(b"abc")})
    assert len(m["id"]) == 32 and int(m["id"], 16) >= 0
    s.shutdown()


def test_bad_token_is_throttled(tmp_path):
    s, base = _server(tmp_path, "t" * 40)
    codes = []
    for _ in range(4):
        try:
            nt.request(base, "GET", "/health", "wrong")
        except nt.TransferError as e:
            codes.append(str(e)[5:8])
    assert codes[:3] == ["401"] * 3 and codes[3] == "429"
    s.shutdown()


def test_oversized_bodies_rejected(tmp_path):
    tok = "t" * 40
    s, base = _server(tmp_path, tok)
    # The server answers before reading the body, so the client may see either
    # the 400 or a reset while it is still writing.
    with pytest.raises((nt.TransferError, urllib.error.URLError, ConnectionError)):
        nt.request(base, "POST", "/v1/transfers/init", tok, {"pad": "x" * (sec.MAX_JSON_BYTES + 10)})
    m = nt.request(base, "POST", "/v1/transfers/init", tok,
                   {"name": "big.gguf", "size": nt.MAX_CHUNK * 2, "sha256": "0" * 64})
    n = nt.MAX_CHUNK + 1
    with pytest.raises((nt.TransferError, urllib.error.URLError, ConnectionError)):
        nt.request(base, "PUT", f"/v1/transfers/{m['id']}/chunk", tok, b"\0" * n,
                   headers={"Content-Range": f"bytes 0-{n - 1}/{nt.MAX_CHUNK * 2}", "X-Chunk-SHA256": "0" * 64},
                   raw=True)
    s.shutdown()


def test_tls_roundtrip(tmp_path):
    pytest.importorskip("cryptography")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    assert sec.main(["selfsigned", "--host", "127.0.0.1", "--cert", str(cert), "--key", str(key)]) == 0
    tok = sec.generate_token()
    nt.Handler.receiver = nt.Receiver(tmp_path / "recv")
    nt.Handler.token = tok
    nt.Handler.throttle = sec.FailureThrottle()
    s = ThreadingHTTPServer(("127.0.0.1", 0), nt.Handler)
    s.socket = sec.server_ssl_context(str(cert), str(key)).wrap_socket(
        s.socket, server_side=True, do_handshake_on_connect=False)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    nt._CLIENT_SSL.update(cafile=str(cert), insecure=False)
    try:
        base = f"https://127.0.0.1:{s.server_port}"
        src = tmp_path / "m.gguf"
        src.write_bytes(os.urandom(50000))
        got = nt.send_one(base, src, tok, chunk=8192)
        assert Path(got["path"]).read_bytes() == src.read_bytes()
    finally:
        nt._CLIENT_SSL.update(cafile="", insecure=False)
        s.shutdown()


def test_mcp_never_puts_token_in_state_or_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("OC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("OC_TRANSFER_MCP_STATE", str(tmp_path / "state.json"))
    import importlib
    import oc_transfer_mcp as m
    m = importlib.reload(m)
    monkeypatch.setattr(m, "_keyring", lambda: None)
    # Legacy state with an inline token is migrated out on load.
    (tmp_path / "state.json").write_text(json.dumps({"targets": {"box": {"url": "https://h:1", "token": "SECRET"}}}))
    d = m.load()
    assert "SECRET" not in (tmp_path / "state.json").read_text()
    assert m.load_token("box") == "SECRET"
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "state.json").stat().st_mode) == 0o600
    seen = {}

    class P:
        returncode = 0
        stdout = "token was SECRET"
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return P()
    monkeypatch.setattr(m.subprocess, "run", fake_run)
    out = m.run("box", ["health"])
    assert "SECRET" not in " ".join(seen["cmd"])
    assert seen["env"][sec.TOKEN_ENV] == "SECRET"
    assert "SECRET" not in out["stdout"]
    assert d["targets"]["box"].get("token") is None
