#!/usr/bin/env python3
"""Optional MCP control plane for OmniCortex transfers.

Two ways to serve it:
* stdio (default): an MCP client on this computer starts it as a process;
* streamable HTTP (--http): an MCP client on another computer connects to
  http(s)://this-computer:PORT/mcp with a bearer token. Every request needs the
  token; requests from web pages of other sites are refused; a network address
  needs TLS, or --allow-plaintext behind a VPN or a TLS proxy (tailscale serve).

    python scripts/ns_transfer_mcp.py --http --token-file ~/.config/neuronscope/secrets/transfer-mcp-http.token
    python scripts/ns_transfer_mcp.py --http --host 0.0.0.0 --port 8765 --token-file T --tls-cert C --tls-key K

The tools act on the computer that runs this server: its targets, its files.

Secrets handling:
* the state file holds target identity only (url, CA file, hints), mode 0600;
* each target's bearer token lives in the OS keyring when ``keyring`` is
  installed, otherwise in a 0600 file under ~/.config/neuronscope/secrets/;
* tokens reach the CLI through the NS_TRANSFER_TOKEN environment variable of
  the child process, never argv, and are never returned by any tool.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_security as sec  # noqa: E402

MIN_MCP = (2, 3)

try:
    # mcp 2.x renamed FastMCP to MCPServer; the decorator API is unchanged.
    from mcp.server.mcpserver import MCPServer as FastMCP
    IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on optional package
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x, refused by ensure()
        IMPORT_ERROR = None
    except Exception:
        FastMCP = None
        IMPORT_ERROR = e

ROOT = Path(__file__).resolve().parent
CLI = ROOT / "ns_transfer.py"
CONFIG = Path(os.environ.get("NS_CONFIG_DIR", str(Path.home() / ".config" / "neuronscope")))
STATE = Path(os.environ.get("NS_TRANSFER_MCP_STATE", str(CONFIG / "transfer-mcp.json")))
SECRETS = CONFIG / "secrets"
KEYRING_SERVICE = "neuronscope-transfer"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def mcp_version_ok() -> tuple[bool, str]:
    try:
        from importlib.metadata import version
        v = version("mcp")
    except Exception:
        return False, "not installed"
    nums = tuple(int(x) for x in re.findall(r"\d+", v)[:2])
    return nums >= MIN_MCP, v


# ---------------------------------------------------------------- secrets

def _keyring():
    try:
        import keyring  # type: ignore
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def store_token(name: str, token: str) -> str:
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, name, token)
            return "keyring"
        except Exception:
            pass
    sec.write_secret_file(SECRETS / f"{name}.token", token)
    return "file"


def load_token(name: str) -> str:
    kr = _keyring()
    if kr is not None:
        try:
            v = kr.get_password(KEYRING_SERVICE, name)
            if v:
                return v
        except Exception:
            pass
    p = SECRETS / f"{name}.token"
    return sec.read_secret_file(p) if p.exists() else ""


def delete_token(name: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, name)
        except Exception:
            pass
    (SECRETS / f"{name}.token").unlink(missing_ok=True)


# ---------------------------------------------------------------- state

def load() -> dict:
    if STATE.exists():
        try:
            d = json.loads(STATE.read_text())
            # Migrate state written by older versions that stored tokens inline.
            changed = False
            for name, t in d.get("targets", {}).items():
                if t.get("token"):
                    store_token(name, t.pop("token"))
                    changed = True
            if changed:
                save(d)
            return d
        except Exception:
            pass
    return {"targets": {}}


def save(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, STATE)


def get_target(name: str) -> dict:
    t = load()["targets"].get(name)
    if t is None:
        raise KeyError(f"unknown target {name!r}; call register_target first")
    return t


def _redact(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def run(name: str, args: list[str], timeout: int = 30) -> dict:
    """Run the transfer CLI against a target with the token in the environment."""
    t = get_target(name)
    token = load_token(name)
    cmd = [sys.executable, str(CLI), *args, "--url", t["url"]]
    if t.get("cafile"):
        cmd += ["--cafile", t["cafile"]]
    env = {**os.environ, sec.TOKEN_ENV: token}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return {"returncode": p.returncode,
            "stdout": _redact(p.stdout[-12000:], token),
            "stderr": _redact(p.stderr[-12000:], token)}


def public(name: str, t: dict) -> dict:
    return {**t, "has_token": bool(load_token(name))}


def ensure() -> None:
    if FastMCP is None:
        raise RuntimeError(f"mcp package unavailable: {IMPORT_ERROR}  (pip install 'mcp>=2.3,<3')")
    ok, v = mcp_version_ok()
    if not ok:
        raise RuntimeError(f"mcp {v} is older than {'.'.join(map(str, MIN_MCP))}, which has known "
                           "security advisories; pip install -U 'mcp>=2.3,<3'")


INSTRUCTIONS = ("OmniCortex transfers. You, the MCP client, are the controller; every tool acts on the computer "
                "running this server: the targets registered here, and file paths on this computer.")

def _ann(**kw):
    """Tool annotations (read-only, destructive…), so clients can approve by kind."""
    try:
        from mcp.types import ToolAnnotations
        return ToolAnnotations(**kw)
    except Exception:                       # pragma: no cover - older SDKs
        return None


READ = dict(read_only_hint=True, destructive_hint=False, open_world_hint=True)
CHANGE = dict(read_only_hint=False, destructive_hint=False, open_world_hint=True)
DESTROY = dict(read_only_hint=False, destructive_hint=True, open_world_hint=False)

if FastMCP:
    mcp = FastMCP("neuronscope-transfer", instructions=INSTRUCTIONS)

    @mcp.tool(annotations=_ann(**CHANGE))
    def register_target(name: str, url: str, token: str = "", cafile: str = "",
                        destination_hint: str = "") -> dict:
        """Register a remote OmniCortex transfer receiver. The token is moved to
        the OS keyring (or a 0600 file) and is never echoed back."""
        if not NAME_RE.match(name):
            raise ValueError("name must match [A-Za-z0-9_.-]{1,64}")
        if not url.startswith(("https://", "http://")):
            raise ValueError("url must be http(s)://")
        d = load()
        d["targets"][name] = {"url": url, "cafile": cafile, "destination_hint": destination_hint}
        save(d)
        where = store_token(name, token) if token else "none"
        warn = []
        if url.startswith("http://") and not sec.is_loopback(url.split("/")[2].rsplit(":", 1)[0]):
            warn.append("plaintext http:// target: only safe through a VPN tunnel")
        return {"ok": True, "name": name, "token_store": where, "warnings": warn,
                "target": public(name, d["targets"][name])}

    @mcp.tool(annotations=_ann(**DESTROY))
    def remove_target(name: str) -> dict:
        """Forget a target and delete its stored token."""
        d = load()
        d["targets"].pop(name, None)
        save(d)
        delete_token(name)
        return {"ok": True}

    @mcp.tool(annotations=_ann(**dict(READ, open_world_hint=False)))
    def list_targets() -> dict:
        """List configured transfer targets (tokens are never returned)."""
        return {"targets": {k: public(k, v) for k, v in load()["targets"].items()}}

    @mcp.tool(annotations=_ann(**READ))
    def probe_target(name: str) -> dict:
        """Check a target receiver."""
        return run(name, ["health"])

    @mcp.tool(annotations=_ann(**CHANGE))
    def send_file(target: str, file_path: str, subdir: str = "") -> dict:
        """Start one reliable resumable file transfer."""
        return run(target, ["send", "--file", file_path, "--subdir", subdir], timeout=86400)

    @mcp.tool(annotations=_ann(**CHANGE))
    def send_batch(target: str, directory: str, glob_pattern: str = "*.gguf", workers: int = 2,
                   subdir: str = "") -> dict:
        """Transfer a batch of GGUFs to a receiver."""
        return run(target, ["batch", "--dir", directory, "--glob", glob_pattern,
                            "--workers", str(max(1, min(workers, 8))), "--subdir", subdir],
                   timeout=86400)

    @mcp.tool(annotations=_ann(**CHANGE))
    def start_watch(target: str, directory: str, glob_pattern: str = "*.gguf", workers: int = 1,
                    subdir: str = "") -> dict:
        """Start a background watcher that sends new GGUFs as they appear."""
        t = get_target(target)
        logdir = Path.home() / ".local" / "state" / "neuronscope-transfer"
        logdir.mkdir(parents=True, exist_ok=True)
        log = logdir / f"{target}.log"
        cmd = [sys.executable, str(CLI), "watch", "--url", t["url"], "--dir", directory,
               "--glob", glob_pattern, "--workers", str(max(1, min(workers, 8))), "--subdir", subdir]
        if t.get("cafile"):
            cmd += ["--cafile", t["cafile"]]
        env = {**os.environ, sec.TOKEN_ENV: load_token(target)}
        with log.open("a") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        return {"ok": True, "pid": p.pid, "log": str(log), "command": cmd}

    @mcp.tool(annotations=_ann(**READ))
    def transfer_status(target: str) -> dict:
        """List receiver-side transfers."""
        return run(target, ["status"])

    @mcp.tool(annotations=_ann(**CHANGE))
    def pull_completed(target: str, transfer_id: str, out_directory: str) -> dict:
        """Pull a completed remote transfer back to the controller."""
        return run(target, ["pull", "--id", transfer_id, "--out", out_directory], timeout=86400)
else:
    mcp = None


class Guard:
    """ASGI wrapper for the HTTP mode: bearer token on every request, no other site's web pages,
    and a cool-off for addresses that keep sending wrong tokens. Lifespan events pass through
    untouched, so the MCP session manager starts and stops with the server."""

    def __init__(self, app, token: str, allowed_origins=(), path: str = "/mcp", max_failures: int = 20,
                 window: float = 600.0):
        self.app, self.token, self.path = app, token, path
        self.origins = {o.rstrip("/") for o in allowed_origins}
        self.max_failures, self.window = max_failures, window
        self.failures: dict = {}

    async def _deny(self, send, code: int, msg: str):
        body = json.dumps({"error": msg}).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                *([(b"www-authenticate", b"Bearer")] if code == 401 else [])]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)            # lifespan and the rest
        import hmac
        import time as _t
        h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        ip = (scope.get("client") or ("?",))[0]
        now = _t.time()
        recent = [t for t in self.failures.get(ip, []) if now - t < self.window]
        self.failures[ip] = recent
        if len(recent) >= self.max_failures:
            return await self._deny(send, 429, "too many wrong tokens from this address; wait a few minutes")
        if scope.get("path") == "/health":
            return await self._ok(send)
        origin = h.get("origin")
        if origin and origin.rstrip("/") not in self.origins:
            return await self._deny(send, 403, "requests from web pages of other sites are refused")
        auth = h.get("authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not given or not hmac.compare_digest(given.encode(), self.token.encode()):
            self.failures[ip].append(now)
            return await self._deny(send, 401, "a valid bearer token is required")
        return await self.app(scope, receive, send)

    async def _ok(self, send):
        body = b'{"status": "ok"}'
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def http_app(token: str, allowed_origins=(), path: str = "/mcp"):
    """The streamable-HTTP MCP app behind the Guard. Host-header checks are off because the
    token already protects every request (and a proxy such as tailscale serve sends its own name)."""
    from mcp.server.transport_security import TransportSecuritySettings
    inner = mcp.streamable_http_app(streamable_http_path=path,
                                    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    return Guard(inner, token, allowed_origins, path)


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="MCP server for OmniCortex transfers. Default: stdio (an MCP client on this computer starts "
                    "it; run on its own it waits silently). --http: serve over the network with a token.",
        epilog='stdio client config: {"mcpServers": {"neuronscope-transfer": {"command": "python", '
               '"args": ["/path/to/neuronscope/scripts/ns_transfer_mcp.py"]}}}. '
               'HTTP client config: {"mcpServers": {"transfer-box": {"url": "https://box:8765/mcp", '
               '"token_file": "~/.config/neuronscope/secrets/transfer-box.token"}}}')
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1", help="listen address (127.0.0.1: this computer, or a proxy "
                   "such as tailscale serve in front)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--path", default="/mcp")
    p.add_argument("--token-file", default="", help="file with the bearer token (default: "
                   "~/.config/neuronscope/secrets/transfer-mcp-http.token, else $NS_TRANSFER_MCP_TOKEN); "
                   "make one with: python scripts/ns_security.py token --out FILE")
    p.add_argument("--tls-cert", default="")
    p.add_argument("--tls-key", default="")
    p.add_argument("--allow-plaintext", action="store_true",
                   help="HTTP without TLS on a network address: only behind a VPN or a TLS proxy")
    p.add_argument("--allowed-origin", action="append", default=[],
                   help="a web origin allowed to call it from a browser (none by default)")
    a = p.parse_args(argv)
    ensure()
    if not a.http:
        mcp.run("stdio")  # no network listener
        return
    default_tok = SECRETS / "transfer-mcp-http.token"
    tf = Path(a.token_file).expanduser() if a.token_file else (default_tok if default_tok.is_file() else None)
    if a.token_file and not tf.is_file():
        raise SystemExit(f"no token file at {tf}")
    token = tf.read_text().strip() if tf else os.environ.get("NS_TRANSFER_MCP_TOKEN", "").strip()
    if len(token) < sec.MIN_TOKEN_CHARS:
        raise SystemExit("--http needs a token of at least %d characters: put one in %s (or give --token-file, "
                         "or set NS_TRANSFER_MCP_TOKEN). Make one: python scripts/ns_security.py token --out \"%s\""
                         % (sec.MIN_TOKEN_CHARS, default_tok, default_tok))
    tls = bool(a.tls_cert and a.tls_key)
    try:
        for w in sec.check_bind(a.host, token, tls=tls, allow_plaintext=a.allow_plaintext):
            print("warning:", w, file=sys.stderr)
    except sec.SecurityConfigError as e:
        raise SystemExit(f"error: {e}")
    import uvicorn
    scheme = "https" if tls else "http"
    print(f"neuronscope-transfer MCP at {scheme}://{a.host}:{a.port}{a.path} (bearer token required)", file=sys.stderr)
    uvicorn.run(http_app(token, a.allowed_origin, a.path), host=a.host, port=a.port, log_level="warning",
                ssl_certfile=a.tls_cert or None, ssl_keyfile=a.tls_key or None)


if __name__ == "__main__":
    main()
