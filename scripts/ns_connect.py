#!/usr/bin/env python3
"""
Connect coding agents and chat apps to OmniCortex: Cline, Claude Desktop, and
any OpenAI-compatible client.

Two connections, set up separately in most apps:

  * the model provider: Studio's OpenAI-compatible API, so the app's chat runs
    on your local models (Cline: API Provider "OpenAI Compatible");
  * the MCP server: OmniCortex's harness and lab as tools (list and route
    models, check replies for hallucination risk, run TestQA and retraining
    jobs, plan and review projects). See scripts/ns_mcp.py.

    python scripts/ns_connect.py show                  # every snippet, for this Studio
    python scripts/ns_connect.py cline --write         # add OmniCortex to Cline's MCP settings
    python scripts/ns_connect.py cline --write --editor cursor --transport stdio
    python scripts/ns_connect.py claude-desktop --write

--write merges one "neuronscope" entry into the app's config (a backup is kept
next to it) and leaves every other server untouched. Studio's Connect page
(/connect) shows the same settings with copy buttons.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "neuronscope"
CLINE_EXT = "saoudrizwan.claude-dev"
EDITORS = {"code": "Code", "code-insiders": "Code - Insiders", "codium": "VSCodium", "cursor": "Cursor",
           "windsurf": "Windsurf"}


def _appdata() -> Path:
    sysname = platform.system()
    if sysname == "Darwin":
        return Path.home() / "Library" / "Application Support"
    if sysname == "Windows":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def cline_settings_path(editor: str = "code") -> Path:
    return _appdata() / EDITORS[editor] / "User" / "globalStorage" / CLINE_EXT / "settings" / "cline_mcp_settings.json"


def claude_desktop_path() -> Path:
    return _appdata() / "Claude" / "claude_desktop_config.json"


def read_only_tools() -> list[str]:
    sys.path.insert(0, str(HERE))
    import ns_mcp
    return [n for n, _, _, ann, _ in ns_mcp.TOOLS if ann.get("readOnlyHint")]


def mcp_http_entry(studio: str, token: str | None = None) -> dict:
    """Cline's remote server entry: Studio serves MCP itself at /mcp."""
    e = {"type": "streamableHttp", "url": studio.rstrip("/") + "/mcp", "disabled": False,
         "autoApprove": read_only_tools()}
    if token:
        e["headers"] = {"Authorization": f"Bearer {token}"}
    return e


def mcp_stdio_entry(studio: str, token_file: str | None = None, python: str | None = None) -> dict:
    """An entry that starts scripts/ns_mcp.py; works in every MCP client."""
    args = [str(HERE / "ns_mcp.py"), "--studio", studio.rstrip("/")]
    if token_file:
        args += ["--token-file", str(Path(token_file).expanduser())]
    return {"command": python or sys.executable, "args": args, "disabled": False,
            "autoApprove": read_only_tools()}


def _q(word: str) -> str:
    """Quote a word for a shell command line, the same on Windows, macOS and Linux."""
    return f'"{word}"' if any(c in word for c in " \t'&()") else word


def claude_code_commands(studio: str, token: str | None = None, token_file: str | None = None) -> dict:
    """`claude mcp add` lines for Claude Code: over HTTP to Studio's /mcp, or starting ns_mcp.py."""
    base = studio.rstrip("/")
    http = f"claude mcp add --transport http --scope user {NAME} {base}/mcp"
    if token:
        http += f' --header "Authorization: Bearer {token}"'
    e = mcp_stdio_entry(base, token_file)
    stdio = f"claude mcp add --scope user {NAME} -- " + " ".join(_q(w) for w in [e["command"], *e["args"]])
    return {"http": http, "stdio": stdio}


def snippets(studio: str, token: str | None = None, token_file: str | None = None) -> dict:
    """Everything a client needs, as data (Studio's /connect page renders this)."""
    base = studio.rstrip("/")
    return {
        "openai": {"base_url": base + "/v1", "api_key": "your Studio token" if token else "any text (Studio has no token)",
                   "model": "a model id from Studio, or auto"},
        "cline_provider": {"API Provider": "OpenAI Compatible", "Base URL": base + "/v1",
                           "API Key": "your Studio token" if token else "local",
                           "Model ID": "a model id from Studio's list, or auto"},
        "cline_mcp_http": {"mcpServers": {NAME: mcp_http_entry(base, "<your Studio token>" if token else None)}},
        "cline_mcp_stdio": {"mcpServers": {NAME: mcp_stdio_entry(base, token_file or ("<path to a file with your "
                                                                                       "Studio token>" if token else None))}},
        "claude_desktop": {"mcpServers": {NAME: {k: v for k, v in mcp_stdio_entry(base, token_file).items()
                                                 if k in ("command", "args")}}},
        "claude_code": claude_code_commands(base, "<your Studio token>" if token else None,
                                            token_file or ("<path to a file with your Studio token>" if token else None)),
        "paths": {"cline": {e: str(cline_settings_path(e)) for e in EDITORS},
                  "claude_desktop": str(claude_desktop_path())},
    }


def merge_into(path: Path, entry: dict) -> Path | None:
    """Add or replace mcpServers.neuronscope in a JSON config. -> backup path."""
    cfg, backup = {}, None
    if path.exists():
        try:
            cfg = json.loads(path.read_text() or "{}")
        except ValueError:
            raise SystemExit(f"{path} is not valid JSON; fix or move it first")
        backup = path.with_name(path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
    cfg.setdefault("mcpServers", {})[NAME] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, path)
    if "headers" in entry:
        os.chmod(path, 0o600)          # it now holds a token
    return backup


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--studio", default=os.environ.get("NS_STUDIO_URL", "http://127.0.0.1:7870"))
    p.add_argument("--token-file", help="Studio's token (or a paired device's), if Studio needs one")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="print every connection snippet")
    c = sub.add_parser("cline", help="Cline's MCP settings")
    c.add_argument("--editor", choices=sorted(EDITORS), default="code")
    c.add_argument("--transport", choices=["http", "stdio"], default="http",
                   help="http: Cline talks to Studio's /mcp (Studio must be running); stdio: Cline starts "
                        "scripts/ns_mcp.py")
    c.add_argument("--write", action="store_true", help="merge into Cline's cline_mcp_settings.json")
    d = sub.add_parser("claude-desktop", help="Claude Desktop's config (stdio)")
    d.add_argument("--write", action="store_true", help="merge into claude_desktop_config.json")
    a = p.parse_args(argv)
    token = Path(a.token_file).expanduser().read_text().strip() if a.token_file else None

    if a.cmd == "show":
        sn = snippets(a.studio, token, a.token_file)
        print("== Cline: Settings > API Provider")
        for k, v in sn["cline_provider"].items():
            print(f"   {k:<13} {v}")
        print("\n== Cline: MCP Servers > Configure (cline_mcp_settings.json), Studio's own endpoint")
        print(json.dumps(sn["cline_mcp_http"], indent=2))
        print("\n== or, if your Cline cannot reach Studio over HTTP: a local process (stdio)")
        print(json.dumps(sn["cline_mcp_stdio"], indent=2))
        print("\n== Claude Desktop (claude_desktop_config.json)")
        print(json.dumps(sn["claude_desktop"], indent=2))
        print(f"\n== Any OpenAI client: base_url={sn['openai']['base_url']}  api_key={sn['openai']['api_key']}")
        return 0
    if a.cmd == "cline":
        entry = mcp_http_entry(a.studio, token) if a.transport == "http" else mcp_stdio_entry(a.studio, a.token_file)
        path = cline_settings_path(a.editor)
    else:
        e = mcp_stdio_entry(a.studio, a.token_file)
        entry = {k: e[k] for k in ("command", "args")}
        path = claude_desktop_path()
    if not a.write:
        print(f"# {path}\n" + json.dumps({"mcpServers": {NAME: entry}}, indent=2))
        print("\n(add --write to merge this into that file)")
        return 0
    backup = merge_into(path, entry)
    print(f"wrote the {NAME} entry to {path}" + (f" (previous file kept as {backup.name})" if backup else ""))
    print("Cline reloads its MCP settings on its own." if a.cmd == "cline"
          else "Restart Claude Desktop to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
