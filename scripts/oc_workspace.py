#!/usr/bin/env python3
"""
Workspaces: where things are, by name, instead of hard-coded paths.

A workspace says where the models are, where llama.cpp was built, which
folder holds the pipeline's data (data/, models/, runs/, profiles/,
adapters/ live under its root), where Studio keeps its own state, and any
environment the GPU stack needs (VK_DRIVER_FILES, HSA_OVERRIDE_GFX_VERSION…).
Studio, its jobs, doctor.py and the launcher all read the active one, so a
repository on an external drive, models on another disk, or a second machine
layout is a setting, not a code change.

Stored in ~/.omnicortex/config.json ($OC_CONFIG) next to Studio's settings:

    {"active_workspace": "laptop",
     "workspaces": {"laptop": {"root": "/run/media/me/disk/omnicortex", "models_dirs": ["~/models"], …}}}

    python scripts/oc_workspace.py                      # the active workspace, resolved
    python scripts/oc_workspace.py list
    python scripts/oc_workspace.py add ext --root /mnt/ext/ns --models-dir /mnt/ext/models --llama-dir ~/llama.cpp
    python scripts/oc_workspace.py set ext env.VK_DRIVER_FILES /usr/share/vulkan/icd.d/radeon_icd.x86_64.json
    python scripts/oc_workspace.py use ext
    eval "$(python scripts/oc_workspace.py env)"        # NS_* variables for scripts run by hand
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "default"
NAME_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}")
ENV_RX = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
# Variables a workspace may not set: they would change how Python, the shell or
# OmniCortex itself run rather than where things are.
ENV_DENY = {"PATH", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "HOME", "SHELL", "OC_STUDIO_TOKEN",
            "OC_CONFIG", "BASH_ENV", "ENV"}
ENV_SECRET_MARKERS = ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                      "AUTH", "PRIVATE_KEY", "ACCESS_KEY")

def _workspace_env_allowed(name: str) -> bool:
    """Workspace env is for runtime paths/options, never for storing credentials."""
    raw = str(name)
    key = raw.upper()
    return (raw == key and bool(ENV_RX.fullmatch(key)) and key not in ENV_DENY
            and not any(marker in key for marker in ENV_SECRET_MARKERS))

def _safe_workspace_env(values) -> dict:
    if not isinstance(values, dict):
        return {}
    return {str(k): str(v) for k, v in values.items()
            if _workspace_env_allowed(k) and "\n" not in str(v) and "\0" not in str(v)}

# name: (kind, help). kind: dir, newdir (created when needed; its parent must exist), dirs, file, exe, env, text
FIELDS = {
    "root": ("dir", "Pipeline data root: data/, models/, runs/, profiles/ and adapters/ live here; jobs run here"),
    "models_dirs": ("dirs", "Folders searched for .gguf models"),
    "download_dir": ("newdir", "Where model downloads go (default: the first models folder)"),
    "llama_dir": ("dir", "llama.cpp checkout (binaries are looked for in its build/bin)"),
    "server": ("exe", "llama-server (default: <llama_dir>/build/bin/llama-server)"),
    "cett": ("exe", "llama-cett-dump (default: <llama_dir>/build/bin/llama-cett-dump)"),
    "gguf": ("file", "The model the H-Neuron pipeline and doctor.py work on"),
    "state_dir": ("newdir", "Studio's own state: chats, traces, review, projects, jobs, stats, documents"),
    "env": ("env", "Environment for jobs and model servers, e.g. VK_DRIVER_FILES"),
    "note": ("text", "What this workspace is for"),
}
# Studio state folders, under state_dir.
STATE_SUBDIRS = {"chats_dir": "chats", "traces_dir": "traces", "review_dir": "review", "projects_dir": "projects",
                 "jobs_dir": "jobs", "stats_dir": "stats", "rag_dir": "rag", "graphs_dir": "graphs"}


def config_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("OC_CONFIG", "~/.omnicortex/config.json")))


def _load(path: Path | None = None) -> dict:
    try:
        d = json.loads((path or config_path()).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict, path: Path | None = None) -> None:
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _x(p):
    return os.path.expanduser(str(p)) if p else p


def _legacy(d: dict) -> dict:
    """Paths saved by Studio's Setup page before workspaces existed."""
    st = d.get("studio") or {}
    out = {}
    if st.get("models_dir"):
        out["models_dirs"] = st["models_dir"] if isinstance(st["models_dir"], list) else [st["models_dir"]]
    for k in ("download_dir", "server", "cett"):
        if st.get(k):
            out[k] = st[k]
    return out


def names(path: Path | None = None) -> list[str]:
    d = _load(path)
    return sorted(set(d.get("workspaces") or {}) | {DEFAULT_NAME})


def active_name(path: Path | None = None, use_env: bool = True) -> str:
    """The workspace in use: $OC_WORKSPACE (set by --workspace, and passed to
    Studio's jobs) or else the file's active_workspace."""
    env = os.environ.get("OC_WORKSPACE") if use_env else None
    d = _load(path)
    name = env or d.get("active_workspace") or DEFAULT_NAME
    return name if name in names(path) else DEFAULT_NAME


def raw(name: str | None = None, path: Path | None = None) -> dict:
    d = _load(path)
    name = name or active_name(path)
    ws = dict((d.get("workspaces") or {}).get(name) or {})
    if name == DEFAULT_NAME:
        ws = {**_legacy(d), **ws}
    return ws


def resolve(ws: dict) -> dict:
    """Fill in defaults and expand ~: the paths everything else uses."""
    root = _x(ws.get("root")) or str(REPO)
    llama = _x(ws.get("llama_dir")) or _x(os.environ.get("OC_LLAMA")) or os.path.expanduser("~/llama.cpp")
    bin_dir = os.path.join(llama, "build", "bin")
    models = [_x(m) for m in (ws.get("models_dirs") or []) if m] or [
        os.path.join(root, "models", "gguf"), os.path.expanduser("~/models"),
        os.path.expanduser("~/.lmstudio/models"), os.path.expanduser("~/.cache/lm-studio/models")]
    state = _x(ws.get("state_dir")) or os.path.expanduser("~/.omnicortex")
    out = {"root": root, "models_dirs": models, "download_dir": _x(ws.get("download_dir")), "llama_dir": llama,
           "server": _x(ws.get("server")) or os.path.join(bin_dir, "llama-server"),
           "cett": _x(ws.get("cett")) or os.path.join(bin_dir, "llama-cett-dump"),
           "gguf": _x(ws.get("gguf")) or _x(os.environ.get("OC_GGUF")), "state_dir": state,
           "env": _safe_workspace_env(ws.get("env") or {}), "note": ws.get("note") or ""}
    for k, sub in STATE_SUBDIRS.items():
        out[k] = os.path.join(state, sub)
    out["set"] = sorted(k for k in ws if k in FIELDS)          # what the person chose, vs defaults
    return out


def active(path: Path | None = None) -> dict:
    name = active_name(path)
    return {"name": name, **resolve(raw(name, path))}


def env_for(ws: dict) -> dict:
    """NS_* variables (what the scripts read) plus the workspace's own environment."""
    out = {"OC_ROOT": ws["root"], "OC_LLAMA": ws["llama_dir"], "OC_WORKSPACE": ws.get("name") or ""}
    if ws.get("server") and os.path.exists(ws["server"]):
        out["OC_LLAMA_SERVER"] = ws["server"]
    if ws.get("cett") and os.path.exists(ws["cett"]):
        out["OC_CETT"] = ws["cett"]
    if ws.get("gguf"):
        out["OC_GGUF"] = ws["gguf"]
    out["OC_MODELS_DIRS"] = os.pathsep.join(ws["models_dirs"])
    out["OC_STATS_DIR"] = ws["stats_dir"]
    out["OC_REVIEW_DIR"] = ws["review_dir"]
    out.update(ws.get("env") or {})
    return out


def validate(values: dict, check_exists: bool = True) -> tuple[dict, dict]:
    """-> (clean, errors). Unknown keys are errors; empty values clear a field."""
    clean, errs = {}, {}
    for k, v in values.items():
        if k not in FIELDS:
            errs[k] = "unknown setting"
            continue
        kind = FIELDS[k][0]
        try:
            if v in (None, "", []):
                clean[k] = None
                continue
            if kind == "dirs":
                v = [str(x).strip() for x in (v if isinstance(v, list) else str(v).split("\n")) if str(x).strip()]
                bad = [x for x in v if check_exists and not os.path.isdir(_x(x))]
                if bad:
                    raise ValueError("not a folder: " + ", ".join(bad))
            elif kind in ("dir", "newdir", "file", "exe"):
                v = str(v).strip()
                if "\n" in v or "\0" in v:
                    raise ValueError("one path")
                p = _x(v)
                if check_exists and kind == "dir" and not os.path.isdir(p):
                    raise ValueError("not a folder")
                if check_exists and kind == "newdir" and not os.path.isdir(p) and \
                        not os.path.isdir(os.path.dirname(os.path.abspath(p))):
                    raise ValueError("neither it nor its parent folder exists")
                if check_exists and kind == "file" and not os.path.isfile(p):
                    raise ValueError("no such file")
                if check_exists and kind == "exe" and not (os.path.isfile(p) and os.access(p, os.X_OK)):
                    raise ValueError("not an executable file")
            elif kind == "env":
                if not isinstance(v, dict):
                    raise ValueError("a map of VARIABLE: value")
                for ek, ev in v.items():
                    if not ENV_RX.fullmatch(str(ek)):
                        raise ValueError(f"{ek!r}: upper-case letters, digits and _ only")
                    if str(ek).upper() in ENV_DENY:
                        raise ValueError(f"{ek} cannot be set from a workspace")
                    if any(marker in str(ek).upper() for marker in ENV_SECRET_MARKERS):
                        raise ValueError(f"{ek} looks like a credential; store secrets outside workspace config")
                    if "\n" in str(ev) or "\0" in str(ev):
                        raise ValueError(f"{ek}: one line")
                    if not _workspace_env_allowed(str(ek)):
                        raise ValueError(f"{ek} is not allowed in workspace environment settings")
                v = {str(ek): str(ev) for ek, ev in v.items()}
            elif kind == "text":
                v = str(v)[:500]
            clean[k] = v
        except ValueError as e:
            errs[k] = str(e)
    return clean, errs


def save(name: str, values: dict, path: Path | None = None, check_exists: bool = True, replace: bool = False) -> dict:
    if not NAME_RX.fullmatch(name or ""):
        raise ValueError("workspace names: letters, digits, . _ - (up to 41)")
    clean, errs = validate(values, check_exists)
    if errs:
        raise ValueError("; ".join(f"{k}: {v}" for k, v in errs.items()))
    d = _load(path)
    wss = d.setdefault("workspaces", {})
    cur = {} if replace else dict(wss.get(name) or (raw(name, path) if name == DEFAULT_NAME else {}))
    for k, v in clean.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    wss[name] = cur
    _save(d, path)
    return cur


def use(name: str, path: Path | None = None) -> None:
    if name not in names(path):
        raise ValueError(f"no workspace {name!r}; have {', '.join(names(path))}")
    d = _load(path)
    d["active_workspace"] = name
    _save(d, path)


def delete(name: str, path: Path | None = None) -> None:
    if name == DEFAULT_NAME:
        raise ValueError("the default workspace cannot be deleted (clear its fields instead)")
    d = _load(path)
    if name not in (d.get("workspaces") or {}):
        raise ValueError(f"no workspace {name!r}")
    del d["workspaces"][name]
    if d.get("active_workspace") == name:
        d["active_workspace"] = DEFAULT_NAME
    _save(d, path)


def describe(path: Path | None = None, env_override: str | None = None) -> dict:
    """Every workspace, resolved, with what exists on disk: for the Settings page.
    "active" is the file's choice (what the next start uses)."""
    act = active_name(path, use_env=False)
    out = []
    for n in names(path):
        r = raw(n, path)
        res = resolve(r)
        out.append({"name": n, "active": n == act, "values": r, "resolved": res,
                    "exists": {"root": os.path.isdir(res["root"]), "llama_dir": os.path.isdir(res["llama_dir"]),
                               "server": os.path.isfile(res["server"]), "cett": os.path.isfile(res["cett"]),
                               "gguf": bool(res["gguf"]) and os.path.isfile(res["gguf"]),
                               "models_dirs": [os.path.isdir(m) for m in res["models_dirs"]]}})
    return {"active": act, "workspaces": out, "fields": {k: {"kind": v[0], "help": v[1]} for k, v in FIELDS.items()},
            "config": str(path or config_path()), "env_override": env_override}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("show", help="the active workspace, resolved (the default)").add_argument("name", nargs="?")
    sub.add_parser("list")
    a_ = sub.add_parser("add", help="create or update a workspace")
    a_.add_argument("name")
    a_.add_argument("--root")
    a_.add_argument("--models-dir", action="append", dest="models_dirs")
    a_.add_argument("--download-dir")
    a_.add_argument("--llama-dir")
    a_.add_argument("--server")
    a_.add_argument("--cett")
    a_.add_argument("--gguf")
    a_.add_argument("--state-dir")
    a_.add_argument("--note")
    a_.add_argument("--use", action="store_true", help="make it the active workspace")
    s_ = sub.add_parser("set", help="one field: root, models_dirs (comma-separated), env.NAME, …; empty value clears")
    s_.add_argument("name")
    s_.add_argument("key")
    s_.add_argument("value", nargs="?", default="")
    sub.add_parser("use").add_argument("name")
    sub.add_parser("delete").add_argument("name")
    e_ = sub.add_parser("env", help="export lines for the shell: eval \"$(python scripts/oc_workspace.py env)\"")
    e_.add_argument("name", nargs="?")
    a = p.parse_args(argv)
    try:
        if a.cmd in (None, "show"):
            name = getattr(a, "name", None) or active_name()
            print(json.dumps({"name": name, **resolve(raw(name))}, indent=1))
        elif a.cmd == "list":
            act = active_name()
            for n in names():
                r = resolve(raw(n))
                print(f"{'*' if n == act else ' '} {n:<16} root {r['root']}  models {', '.join(r['models_dirs'][:2])}"
                      + (f"  ({r['note']})" if r["note"] else ""))
        elif a.cmd == "add":
            vals = {k: getattr(a, k) for k in ("root", "models_dirs", "download_dir", "llama_dir", "server", "cett",
                                               "gguf", "state_dir", "note") if getattr(a, k)}
            save(a.name, vals)
            if a.use:
                use(a.name)
            print(f"saved workspace {a.name}" + (" (active)" if a.use else ""))
        elif a.cmd == "set":
            if a.key.startswith("env."):
                env = dict(raw(a.name).get("env") or {})
                if a.value:
                    env[a.key[4:]] = a.value
                else:
                    env.pop(a.key[4:], None)
                save(a.name, {"env": env or None})
            else:
                v = a.value.split(",") if a.key == "models_dirs" and a.value else a.value
                save(a.name, {a.key: v})
            print(f"{a.name}: {a.key} {'set' if a.value else 'cleared'}")
        elif a.cmd == "use":
            use(a.name)
            print(f"active workspace: {a.name} (restart Studio to apply)")
        elif a.cmd == "delete":
            delete(a.name)
            print(f"deleted {a.name}")
        elif a.cmd == "env":
            ws = {"name": a.name or active_name(), **resolve(raw(a.name))}
            for k, v in env_for(ws).items():
                print(f"export {k}={shlex.quote(v)}")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
