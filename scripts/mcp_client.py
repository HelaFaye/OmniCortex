#!/usr/bin/env python3
"""
MCP client for Studio: connect to MCP servers and expose their tools to chat.

Servers are configured in the format LM Studio and Claude Desktop use
(default ~/.omnicortex/mcp.json):

    {"mcpServers": {
        "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
        "search": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."}},
        "remote": {"url": "https://box:8765/mcp", "token_file": "~/.config/omnicortex/secrets/box.token"},
        "clock":  {"command": "python", "args": ["clock_server.py"], "autoApprove": ["now"]},
        "old":    {"command": "...", "disabled": true}}}

Each enabled server keeps one connection (stdio subprocess or streamable HTTP)
in a background event loop. Tools are offered to the model as OpenAI
functions named `<server>__<tool>`. Studio asks before every call unless the
server sets `"autoApprove": true` or lists the tool names it may run unasked.

Secrets need not sit in this file: an HTTP server's `"token_file"` (or
`"token_env"`) becomes its `Authorization: Bearer` header, and any header or
env value may say `${file:PATH}` or `${env:NAME}`. They are read when the
server connects, so rotating a token is a reload, not an edit here.

    python scripts/mcp_client.py tools               # what the configured servers offer
    python scripts/mcp_client.py call clock__now '{}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
from concurrent.futures import Future
from pathlib import Path

DEFAULT_CONFIG = Path.home() / ".omnicortex" / "mcp.json"
SEP = "__"
MAX_RESULT_CHARS = 20000


def _attr(obj, *names):
    """mcp 2.x uses snake_case model fields; older releases used camelCase."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


REF = re.compile(r"\$\{(file|env):([^}]+)\}")


def expand(value: str) -> str:
    """${file:PATH} -> the file's contents (stripped); ${env:NAME} -> the variable. Missing ones raise."""
    def one(m):
        kind, ref = m.group(1), m.group(2).strip()
        if kind == "env":
            if ref not in os.environ:
                raise ValueError(f"${{env:{ref}}}: no such environment variable")
            return os.environ[ref]
        f = Path(ref).expanduser()
        if not f.is_file():
            raise ValueError(f"${{file:{ref}}}: no such file")
        return f.read_text(encoding="utf-8").strip()
    return REF.sub(one, str(value))


def http_headers(spec: dict) -> dict:
    """The headers an HTTP server's spec asks for, with token_file / token_env and ${...} resolved."""
    headers = {k: expand(v) for k, v in (spec.get("headers") or {}).items()}
    tok = None
    if spec.get("token_file"):
        tok = expand("${file:" + str(spec["token_file"]) + "}")
    elif spec.get("token_env"):
        tok = expand("${env:" + str(spec["token_env"]) + "}")
    if tok:
        if any(c in tok for c in "\r\n"):
            raise ValueError("the token has a line break inside it")
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


class _Conn:
    """One server: a task that holds the connection and serves queued calls."""

    def __init__(self, name: str, spec: dict, log_dir: Path):
        self.name, self.spec, self.log_dir = name, spec, log_dir
        self.errlog = None
        self.queue: asyncio.Queue | None = None
        self.tools: list = []
        self.error: str | None = None
        self.ready = asyncio.Event()
        self.task: asyncio.Task | None = None

    def transport(self):
        if self.spec.get("url"):
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
            return streamable_http_client(self.spec["url"],
                                          http_client=create_mcp_http_client(headers=http_headers(self.spec) or None))
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = None
        if self.spec.get("env"):
            env = {**os.environ, **{k: expand(v) for k, v in self.spec["env"].items()}}
        params = StdioServerParameters(command=self.spec["command"], args=[str(a) for a in self.spec.get("args", [])],
                                       env=env, cwd=self.spec.get("cwd"))
        # The server's stderr (its logs) goes to a file, not into the host's console.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.errlog = open(self.log_dir / f"{self.name}.log", "a", encoding="utf-8")
        return stdio_client(params, errlog=self.errlog)

    async def run(self):
        from mcp import Client
        self.queue = asyncio.Queue()
        try:
            async with Client(self.transport(), read_timeout_seconds=float(self.spec.get("timeout", 120))) as c:
                self.tools = list((await c.list_tools()).tools)
                self.error = None
                self.ready.set()
                while True:
                    op, args, fut = await self.queue.get()
                    if op == "stop":
                        fut.set_result(None)
                        return
                    try:
                        if op == "call":
                            fut.set_result(await c.call_tool(*args))
                        elif op == "list":
                            self.tools = list((await c.list_tools()).tools)
                            fut.set_result(self.tools)
                    except Exception as e:           # a failing call must not drop the connection
                        fut.set_exception(e)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"[:500]
            self.ready.set()
            while self.queue is not None and not self.queue.empty():
                _, _, fut = self.queue.get_nowait()
                if not fut.done():
                    fut.set_exception(RuntimeError(self.error))


class MCPHub:
    def __init__(self, config: str | Path | None = None):
        self.config_path = Path(config or DEFAULT_CONFIG).expanduser()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="mcp-hub")
        self.thread.start()
        self.conns: dict[str, _Conn] = {}
        self.specs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.extra = None     # callable -> more servers (Studio: from enabled plugins)

    # ------------------------------------------------------------ config
    def load(self) -> dict:
        servers = {}
        if self.config_path.exists():
            cfg = json.loads(self.config_path.read_text())
            servers = dict(cfg.get("mcpServers", cfg if isinstance(cfg, dict) else {}))
        if self.extra:
            servers.update(self.extra())
        out = {}
        for name, spec in servers.items():
            if not isinstance(spec, dict) or not (spec.get("command") or spec.get("url")):
                continue
            out[_safe(name)] = spec
        return out

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def connect(self, timeout: float = 60) -> None:
        """(Re)read the config; start new or changed servers, stop removed ones."""
        specs = {n: s for n, s in self.load().items() if not s.get("disabled")}
        with self.lock:
            for name in list(self.conns):
                if name not in specs or specs[name] != self.specs.get(name):
                    self._stop(name)
            for name, spec in specs.items():
                if name not in self.conns:
                    conn = _Conn(name, spec, self.config_path.parent / "mcp-logs")
                    self.conns[name] = conn
                    self.specs[name] = spec

                    async def start(c=conn):
                        c.task = asyncio.ensure_future(c.run())
                    self._run(start(), 5)
            conns = list(self.conns.values())
        for c in conns:
            try:
                self._run(asyncio.wait_for(c.ready.wait(), timeout), timeout + 5)
            except Exception:
                c.error = c.error or f"no answer within {timeout}s"

    def _stop(self, name: str) -> None:
        c = self.conns.pop(name, None)
        self.specs.pop(name, None)
        if c is None or c.queue is None or c.task is None or c.task.done():
            return
        fut = Future()

        async def put():
            await c.queue.put(("stop", None, _Bridge(fut)))
        try:
            self._run(put(), 5)
            fut.result(10)
        except Exception:
            self.loop.call_soon_threadsafe(c.task.cancel)

    def close(self) -> None:
        with self.lock:
            for name in list(self.conns):
                self._stop(name)

    # ------------------------------------------------------------ tools
    def status(self) -> list[dict]:
        out = []
        for name, spec in self.load().items():
            c = self.conns.get(name)
            out.append({"name": name, "transport": "http" if spec.get("url") else "stdio",
                        "disabled": bool(spec.get("disabled")), "connected": bool(c and not c.error and c.ready.is_set()),
                        "error": c.error if c else None, "auto_approve": spec.get("autoApprove", False),
                        "tools": [{"name": t.name, "description": (t.description or "")[:300]} for t in (c.tools if c else [])]})
        return out

    def openai_tools(self) -> list[dict]:
        out = []
        for name, c in self.conns.items():
            if c.error:
                continue
            for t in c.tools:
                schema = _attr(t, "input_schema", "inputSchema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                out.append({"type": "function", "function": {
                    "name": f"{name}{SEP}{_safe(t.name)}"[:64],
                    "description": (t.description or f"{t.name} from MCP server {name}")[:1024],
                    "parameters": schema}})
        return out

    def resolve(self, fn_name: str) -> tuple[str, str]:
        server, _, tool = fn_name.partition(SEP)
        c = self.conns.get(server)
        if c is None:
            raise KeyError(f"no MCP server {server!r}")
        for t in c.tools:
            if _safe(t.name) == tool:
                return server, t.name
        raise KeyError(f"MCP server {server!r} has no tool {tool!r}")

    def tool_info(self, fn_name: str) -> dict:
        """-> {server, tool, annotations, auto_approve} for a function name the model used."""
        server, tool = self.resolve(fn_name)
        ann = {}
        for t in self.conns[server].tools:
            if t.name == tool:
                a = getattr(t, "annotations", None)
                if a is not None:
                    # mcp 2.x names them read_only_hint…; older releases and plain dicts readOnlyHint…
                    for k, snake in (("readOnlyHint", "read_only_hint"), ("destructiveHint", "destructive_hint"),
                                     ("idempotentHint", "idempotent_hint"), ("openWorldHint", "open_world_hint")):
                        v = (a.get(k, a.get(snake)) if isinstance(a, dict)
                             else _attr(a, snake, k))
                        if v is not None:
                            ann[k] = bool(v)
        return {"server": server, "tool": tool, "annotations": ann,
                "auto_approve": self.specs.get(server, {}).get("autoApprove", False)}

    def tool_list(self) -> list[dict]:
        """Every connected tool with its annotations: for the approval settings."""
        out = []
        for name, c in self.conns.items():
            if c.error:
                continue
            for t in c.tools:
                try:
                    info = self.tool_info(f"{name}{SEP}{_safe(t.name)}")
                except KeyError:
                    continue
                out.append({**info, "description": (t.description or "")[:200]})
        return out

    def auto_approved(self, fn_name: str) -> bool:
        try:
            server, tool = self.resolve(fn_name)
        except KeyError:
            return False
        aa = self.specs.get(server, {}).get("autoApprove", False)
        return aa is True or (isinstance(aa, list) and tool in aa)

    def test(self, name: str, timeout: float = 60) -> dict:
        """Check one server for real: ask it for its tools again, and call one read-only tool that
        needs no arguments if it has one. -> {ok, tools, seconds, called, result | error}."""
        import time as _t
        c = self.conns.get(name)
        if c is None:
            return {"ok": False, "error": f"no server {name!r} (or it is disabled)"}
        if c.error:
            return {"ok": False, "error": c.error}
        t0 = _t.time()
        fut: Future = Future()

        async def put():
            await c.queue.put(("list", (), _Bridge(fut)))
        try:
            self._run(put(), 5)
            tools = fut.result(timeout)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "seconds": round(_t.time() - t0, 2)}
        out = {"ok": True, "tools": len(tools), "seconds": round(_t.time() - t0, 2), "called": None}
        for t in tools:
            ann = _attr(t, "annotations")
            ro = bool(ann is not None and _attr(ann, "read_only_hint", "readOnlyHint"))
            schema = _attr(t, "input_schema", "inputSchema") or {}
            if ro and not (schema.get("required") if isinstance(schema, dict) else None):
                try:
                    ok, text = self.call(f"{name}{SEP}{_safe(t.name)}", {}, timeout)
                    out.update(called=t.name, call_ok=ok, result=text[:600])
                except Exception as e:
                    out.update(called=t.name, call_ok=False, result=f"{type(e).__name__}: {e}"[:300])
                break
        out["seconds"] = round(_t.time() - t0, 2)
        return out

    def call(self, fn_name: str, arguments: dict, timeout: float = 120) -> tuple[bool, str]:
        """-> (ok, text). Text content is joined; other content is summarised."""
        server, tool = self.resolve(fn_name)
        c = self.conns[server]
        fut: Future = Future()

        async def put():
            await c.queue.put(("call", (tool, arguments), _Bridge(fut)))
        self._run(put(), 5)
        res = fut.result(timeout)
        parts = []
        for item in getattr(res, "content", None) or []:
            kind = getattr(item, "type", "")
            if kind == "text":
                parts.append(item.text)
            elif kind == "resource" and getattr(getattr(item, "resource", None), "text", None):
                parts.append(item.resource.text)
            else:
                parts.append(f"[{kind or 'content'} omitted]")
        sc = _attr(res, "structured_content", "structuredContent")
        if not parts and sc is not None:
            parts.append(json.dumps(sc, ensure_ascii=False))
        text = "\n".join(parts)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n[truncated {len(text) - MAX_RESULT_CHARS} characters]"
        return not _attr(res, "is_error", "isError"), text


class _Bridge:
    """Lets the event loop resolve a concurrent.futures.Future from another thread."""

    def __init__(self, fut: Future):
        self.fut = fut

    def done(self):
        return self.fut.done()

    def set_result(self, v):
        if not self.fut.done():
            self.fut.set_result(v)

    def set_exception(self, e):
        if not self.fut.done():
            self.fut.set_exception(e)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools")
    c = sub.add_parser("call")
    c.add_argument("name")
    c.add_argument("arguments", nargs="?", default="{}")
    a = p.parse_args(argv)
    hub = MCPHub(a.config)
    hub.connect()
    try:
        if a.cmd == "tools":
            print(json.dumps(hub.status(), indent=1))
        else:
            ok, text = hub.call(a.name, json.loads(a.arguments))
            print(("" if ok else "error: ") + text)
    finally:
        hub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------ editing mcp.json (Studio's Connectors page)
MASK = "••••••"
SERVER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")


def read_config(path: str | Path) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        return {"mcpServers": {}}
    cfg = json.loads(p.read_text() or "{}")
    if not isinstance(cfg, dict):
        raise ValueError(f"{p} is not a JSON object")
    if "mcpServers" not in cfg:                     # the bare {name: spec} form: keep it readable
        cfg = {"mcpServers": {k: v for k, v in cfg.items() if isinstance(v, dict)}}
    return cfg


def write_config(path: str | Path, cfg: dict) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.chmod(tmp, 0o600)                            # headers and env may hold keys
    os.replace(tmp, p)


def _masked(d: dict) -> dict:
    """Values shown to the page: ${file:...} / ${env:...} references as they are, anything else hidden."""
    return {k: (v if isinstance(v, str) and REF.fullmatch(v.strip()) else MASK) for k, v in (d or {}).items()}


def public_spec(spec: dict) -> dict:
    out = {k: spec[k] for k in ("command", "args", "cwd", "url", "token_file", "token_env", "autoApprove", "disabled")
           if k in spec}
    if spec.get("env"):
        out["env"] = _masked(spec["env"])
    if spec.get("headers"):
        out["headers"] = _masked(spec["headers"])
    return out


def _pairs(d, old: dict, what: str, key_rx) -> dict:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f"{what}: expected NAME: value pairs")
    out = {}
    for k, v in d.items():
        k, v = str(k).strip(), "" if v is None else str(v)
        if not key_rx.match(k):
            raise ValueError(f"{what}: bad name {k!r}")
        if v == MASK:                               # unchanged secret: keep what the file has
            if k not in (old or {}):
                raise ValueError(f"{what}: {k} has no saved value to keep")
            v = old[k]
        if any(c in v for c in "\r\n\0"):
            raise ValueError(f"{what}: {k} has a line break in it")
        out[k] = v
    return out


def clean_spec(spec: dict, old: dict | None = None) -> dict:
    """Check a server entry from the page. Keys this page does not manage are kept from the old entry."""
    old = old or {}
    out = {k: v for k, v in old.items() if k not in ("command", "args", "cwd", "env", "url", "headers", "token_file",
                                                     "token_env", "autoApprove", "disabled")}
    if spec.get("url"):
        url = str(spec["url"]).strip()
        if not re.match(r"^https?://[^\s/]+", url):
            raise ValueError("url: http(s)://host[:port]/path")
        out["url"] = url
        h = _pairs(spec.get("headers"), old.get("headers"), "headers", re.compile(r"^[A-Za-z0-9-]{1,100}$"))
        if h:
            out["headers"] = h
        if spec.get("token_file"):
            out["token_file"] = str(spec["token_file"]).strip()
        elif spec.get("token_env"):
            if not ENV_NAME.match(str(spec["token_env"])):
                raise ValueError("token_env: an environment variable name")
            out["token_env"] = str(spec["token_env"])
    elif spec.get("command"):
        cmd = str(spec["command"]).strip()
        args = spec.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, (str, int, float)) for a in args):
            raise ValueError("args: a list of words")
        out["command"], out["args"] = cmd, [str(a) for a in args]
        if spec.get("cwd"):
            out["cwd"] = str(spec["cwd"])
        e = _pairs(spec.get("env"), old.get("env"), "env", ENV_NAME)
        if e:
            out["env"] = e
    else:
        raise ValueError("give a server URL, or a command to start")
    aa = spec.get("autoApprove", False)
    if isinstance(aa, list):
        out["autoApprove"] = [str(t) for t in aa if str(t)]
    elif aa:
        out["autoApprove"] = True
    if spec.get("disabled"):
        out["disabled"] = True
    return out


def save_server(path: str | Path, name: str, spec: dict, old_name: str | None = None) -> dict:
    name = str(name).strip()
    if not SERVER_NAME.match(name):
        raise ValueError("name: letters, digits, - and _ (up to 40)")
    cfg = read_config(path)
    servers = cfg.setdefault("mcpServers", {})
    old_name = old_name or name
    if old_name != name and name in servers:
        raise ValueError(f"there is already a server named {name}")
    entry = clean_spec(spec, servers.get(old_name))
    servers.pop(old_name, None)
    servers[name] = entry
    write_config(path, cfg)
    return entry


def remove_server(path: str | Path, name: str) -> None:
    cfg = read_config(path)
    if name not in cfg.get("mcpServers", {}):
        raise ValueError(f"no server named {name}")
    del cfg["mcpServers"][name]
    write_config(path, cfg)


def set_disabled(path: str | Path, name: str, disabled: bool) -> None:
    cfg = read_config(path)
    s = cfg.get("mcpServers", {}).get(name)
    if s is None:
        raise ValueError(f"no server named {name}")
    if disabled:
        s["disabled"] = True
    else:
        s.pop("disabled", None)
    write_config(path, cfg)
