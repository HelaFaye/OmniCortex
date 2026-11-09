#!/usr/bin/env python3
"""
When a model calls an MCP tool in Studio's chat: ask, allow or deny, by rule.

Every call is decided in this order, and the first answer wins:

  1. Your rules, top to bottom. A rule matches on the server and tool (globs),
     the kind of operation, and regular expressions on the arguments; it
     allows, asks or denies. Rules can expire, and an allow rule can be
     limited to N calls an hour (after that it asks).
  2. A server's autoApprove list in mcp.json (as in Cline and Claude Desktop).
  3. Quick settings per kind of operation, from the tool's own MCP
     annotations: read-only, writes, destructive, reaches the outside world,
     or unannotated.
  4. The default (ask).

Kinds: read_only (readOnlyHint), destructive (destructiveHint true), write (not
read-only, destructiveHint false), open_world (openWorldHint true, checked as
well as the others: a read-only web search is both), unknown (no annotations).
A tool's annotations are its author's claim; a rule on the tool itself is the
stronger statement.

Argument conditions: {"path": "a.b", "regex": "...", "negate": false}. path
is a dotted path into the arguments; "*" in it matches any key or list item
("files.*.path"), and the path "*" alone means any string anywhere in the
arguments. Every condition of a rule must hold (negate inverts one). With
"any" path matching several values, the condition holds when one of them
matches (for negate: when none does).

    python scripts/tool_rules.py                 # rules, quick settings, default
    python scripts/tool_rules.py add allow --server filesystem --tool read_file \\
        --arg 'path=^/home/me/notes/' --note "my notes are fine"
    python scripts/tool_rules.py add deny --tool '*' --arg '*=rm -rf'
    python scripts/tool_rules.py category destructive deny
    python scripts/tool_rules.py test filesystem read_file '{"path": "/home/me/notes/a.md"}'
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

DEFAULT_PATH = Path.home() / ".omnicortex" / "tool-rules.json"
ACTIONS = ("allow", "ask", "deny")
CAT_ACTIONS = ("default",) + ACTIONS          # "default": this kind follows the overall default
KINDS = ("read_only", "write", "destructive", "open_world", "unknown")
MAX_REGEX = 500
MAX_MATCH_CHARS = 20_000
_LOCK = threading.Lock()


def blank() -> dict:
    return {"default": "ask", "categories": {k: "default" for k in KINDS}, "rules": []}


def kinds_of(annotations: dict | None) -> list[str]:
    a = annotations or {}
    ro, de, ow = a.get("readOnlyHint"), a.get("destructiveHint"), a.get("openWorldHint")
    if ro is None and de is None and ow is None:
        return ["unknown"]
    out = []
    if ro:
        out.append("read_only")
    elif de is False:
        out.append("write")
    else:
        out.append("destructive")          # the MCP default for a tool that is not read-only
    if ow:
        out.append("open_world")
    return out


def _values(obj, parts):
    if not parts:
        yield obj
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []
        for v in items:
            yield from _values(v, rest)
    elif isinstance(obj, dict) and head in obj:
        yield from _values(obj[head], rest)
    elif isinstance(obj, list) and head.isdigit() and int(head) < len(obj):
        yield from _values(obj[int(head)], rest)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif obj is not None:
        yield json.dumps(obj)


def _text(v) -> str:
    return (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))[:MAX_MATCH_CHARS]


def cond_holds(cond: dict, args: dict) -> bool:
    rx = re.compile(cond["regex"])
    path = cond.get("path") or "*"
    vals = list(_strings(args)) if path == "*" else [_text(v) for v in _values(args, path.split("."))]
    hit = any(rx.search(v) for v in vals)
    return (not hit) if cond.get("negate") else hit


def rule_matches(rule: dict, server: str, tool: str, kinds: list[str], args: dict) -> bool:
    if not fnmatch.fnmatchcase(server, rule.get("server") or "*"):
        return False
    if not fnmatch.fnmatchcase(tool, rule.get("tool") or "*"):
        return False
    k = rule.get("kind") or "any"
    if k != "any" and k not in kinds:
        return False
    return all(cond_holds(c, args) for c in rule.get("args") or [])


def validate_rule(r: dict) -> dict:
    if not isinstance(r, dict):
        raise ValueError("a rule is an object")
    action = r.get("action")
    if action not in ACTIONS:
        raise ValueError(f"action is one of {ACTIONS}")
    out = {"id": str(r.get("id") or uuid.uuid4().hex[:8]), "action": action,
           "server": str(r.get("server") or "*")[:200], "tool": str(r.get("tool") or "*")[:200],
           "kind": r.get("kind") or "any", "args": [], "note": str(r.get("note") or "")[:300],
           "enabled": r.get("enabled", True) is not False}
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", out["id"]):
        raise ValueError("rule ids: letters, digits, _ and -")
    if out["kind"] not in KINDS + ("any",):
        raise ValueError(f"kind is one of {KINDS + ('any',)}")
    for c in r.get("args") or []:
        if not isinstance(c, dict) or not isinstance(c.get("regex"), str):
            raise ValueError("argument conditions are {path, regex, negate}")
        if len(c["regex"]) > MAX_REGEX:
            raise ValueError(f"regex longer than {MAX_REGEX} characters")
        try:
            re.compile(c["regex"])
        except re.error as e:
            raise ValueError(f"bad regex {c['regex']!r}: {e}")
        path = str(c.get("path") or "*")
        if not re.fullmatch(r"\*|[A-Za-z0-9_*-]+(\.[A-Za-z0-9_*-]+)*", path):
            raise ValueError(f"bad argument path {path!r}")
        out["args"].append({"path": path, "regex": c["regex"], "negate": bool(c.get("negate"))})
    if r.get("max_per_hour") not in (None, "", 0):
        n = int(r["max_per_hour"])
        if n < 1:
            raise ValueError("max_per_hour is 1 or more")
        if action != "allow":
            raise ValueError("max_per_hour limits allow rules")
        out["max_per_hour"] = n
    if r.get("expires") not in (None, ""):
        out["expires"] = float(r["expires"])
    out["hits"] = int(r.get("hits") or 0)
    out["recent"] = [float(t) for t in (r.get("recent") or [])][-500:]
    return out


class Rules:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("OC_TOOL_RULES") or DEFAULT_PATH).expanduser()

    def load(self) -> dict:
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return blank()
        out = blank()
        if d.get("default") in ACTIONS:
            out["default"] = d["default"]
        for k, v in (d.get("categories") or {}).items():
            if k in KINDS and v in CAT_ACTIONS:
                out["categories"][k] = v
        for r in d.get("rules") or []:
            try:
                out["rules"].append(validate_rule(r))
            except (ValueError, TypeError):
                continue                      # a hand-edited bad rule is skipped, never fatal
        return out

    def save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, self.path)

    def replace(self, d: dict) -> dict:
        """Save a whole rule set from the Settings page (validated)."""
        out = blank()
        if d.get("default") not in ACTIONS:
            raise ValueError(f"default is one of {ACTIONS}")
        out["default"] = d["default"]
        for k, v in (d.get("categories") or {}).items():
            if k not in KINDS or v not in CAT_ACTIONS:
                raise ValueError(f"categories: {k} -> {v}?")
            out["categories"][k] = v
        out["rules"] = [validate_rule(r) for r in d.get("rules") or []]
        if len({r["id"] for r in out["rules"]}) != len(out["rules"]):
            raise ValueError("rule ids must be unique")
        with _LOCK:
            self.save(out)
        return out

    def add(self, rule: dict, first: bool = True) -> dict:
        r = validate_rule(rule)
        with _LOCK:
            d = self.load()
            d["rules"] = [x for x in d["rules"] if x["id"] != r["id"]]
            d["rules"].insert(0 if first else len(d["rules"]), r)
            self.save(d)
        return r

    def remove(self, rid: str) -> None:
        with _LOCK:
            d = self.load()
            n = len(d["rules"])
            d["rules"] = [x for x in d["rules"] if x["id"] != rid]
            if len(d["rules"]) == n:
                raise KeyError(rid)
            self.save(d)

    def decide(self, server: str, tool: str, args: dict, annotations: dict | None = None,
               auto_approve=None, record: bool = True, now: float | None = None) -> dict:
        """-> {action: allow|ask|deny, by: rule|autoApprove|category|default, rule?, reason}."""
        now = time.time() if now is None else now
        kinds = kinds_of(annotations)
        with _LOCK:
            d = self.load()
            for r in d["rules"]:
                if not r["enabled"] or (r.get("expires") and r["expires"] < now):
                    continue
                try:
                    if not rule_matches(r, server, tool, kinds, args or {}):
                        continue
                except re.error:
                    continue
                action, reason = r["action"], f"rule {r['id']}" + (f" ({r['note']})" if r["note"] else "")
                if action == "allow" and r.get("max_per_hour"):
                    recent = [t for t in r["recent"] if t > now - 3600]
                    if len(recent) >= r["max_per_hour"]:
                        action, reason = "ask", f"rule {r['id']} allows {r['max_per_hour']} an hour; limit reached"
                    elif record:
                        r["recent"] = recent + [now]
                if record:
                    r["hits"] += 1
                    r["last_hit"] = now
                    self.save(d)
                return {"action": action, "by": "rule", "rule": r["id"], "reason": reason, "kinds": kinds}
        if auto_approve is True or (isinstance(auto_approve, list) and tool in auto_approve):
            return {"action": "allow", "by": "autoApprove", "reason": f"{server}'s autoApprove in mcp.json",
                    "kinds": kinds}
        # The strictest quick setting among the tool's kinds applies; a kind left
        # at "default" counts as the overall default.
        order = {"deny": 2, "ask": 1, "allow": 0}
        set_ = [(d["categories"].get(k, "default"), k) for k in kinds]
        if all(v == "default" for v, _ in set_):
            return {"action": d["default"], "by": "default", "reason": f"default: {d['default']}", "kinds": kinds}
        act, k = max((((d["default"] if v == "default" else v), k) for v, k in set_), key=lambda x: order[x[0]])
        return {"action": act, "by": "category", "reason": f"{k.replace('_', ' ')} tools: {act}", "kinds": kinds}


def rule_from_call(kind: str, server: str, tool: str, args: dict) -> dict:
    """The rule an 'always' button on an approval card creates.
    kind: tool (this tool, any arguments), args (these exact top-level argument
    values), deny (never this tool)."""
    if kind == "tool":
        return {"action": "allow", "server": server, "tool": tool, "note": "always allowed from a chat"}
    if kind == "deny":
        return {"action": "deny", "server": server, "tool": tool, "note": "always denied from a chat"}
    if kind == "args":
        conds = [{"path": k, "regex": "^" + re.escape(_text(v)) + "$"} for k, v in sorted((args or {}).items())
                 if re.fullmatch(r"[A-Za-z0-9_-]+", k)]
        return {"action": "allow", "server": server, "tool": tool, "args": conds,
                "note": "these arguments, allowed from a chat"}
    raise ValueError("kind is tool, args or deny")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--file", help=f"rules file (default {DEFAULT_PATH}, or $OC_TOOL_RULES)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list")
    a_ = sub.add_parser("add", help="add a rule at the top (it wins over later ones)")
    a_.add_argument("action", choices=ACTIONS)
    a_.add_argument("--server", default="*")
    a_.add_argument("--tool", default="*")
    a_.add_argument("--kind", default="any", choices=KINDS + ("any",))
    a_.add_argument("--arg", action="append", default=[], help="PATH=REGEX, or PATH!=REGEX for 'does not match'")
    a_.add_argument("--max-per-hour", type=int)
    a_.add_argument("--expires-in", help="e.g. 8h, 30m, 2d")
    a_.add_argument("--note", default="")
    a_.add_argument("--last", action="store_true", help="add at the bottom instead")
    sub.add_parser("remove").add_argument("id")
    c_ = sub.add_parser("category", help="quick setting for a kind of operation")
    c_.add_argument("kind", choices=KINDS)
    c_.add_argument("action", choices=CAT_ACTIONS)
    sub.add_parser("default").add_argument("action", choices=ACTIONS)
    t_ = sub.add_parser("test", help="what would happen to this call (nothing is recorded)")
    t_.add_argument("server")
    t_.add_argument("tool")
    t_.add_argument("args", nargs="?", default="{}")
    t_.add_argument("--annotations", default="{}", help='e.g. \'{"readOnlyHint": true}\'')
    a = p.parse_args(argv)
    R = Rules(a.file)
    try:
        if a.cmd in (None, "list"):
            d = R.load()
            print(f"default: {d['default']}    file: {R.path}")
            print("kinds: " + ", ".join(f"{k} {v}" for k, v in d["categories"].items()))
            for i, r in enumerate(d["rules"], 1):
                cond = " and ".join(f"{c['path']}{'!~' if c['negate'] else '~'}/{c['regex']}/" for c in r["args"])
                extra = [f"{r['max_per_hour']}/h" if r.get("max_per_hour") else "",
                         time.strftime("until %Y-%m-%d %H:%M", time.localtime(r["expires"])) if r.get("expires") else "",
                         "" if r["enabled"] else "disabled", f"{r['hits']} hits"]
                print(f"{i:>3}. [{r['id']}] {r['action']:<5} {r['server']}/{r['tool']}"
                      + (f" ({r['kind']})" if r["kind"] != "any" else "") + (f" if {cond}" if cond else "")
                      + "  " + " ".join(x for x in extra if x) + (f"  # {r['note']}" if r["note"] else ""))
        elif a.cmd == "add":
            conds = []
            for s in a.arg:
                m = re.match(r"^([^=!]+)(!?)=(.*)$", s)
                if not m:
                    raise ValueError(f"--arg {s!r}: PATH=REGEX or PATH!=REGEX")
                conds.append({"path": m.group(1), "regex": m.group(3), "negate": m.group(2) == "!"})
            expires = None
            if a.expires_in:
                m = re.fullmatch(r"(\d+)([mhd])", a.expires_in)
                if not m:
                    raise ValueError("--expires-in like 30m, 8h, 2d")
                expires = time.time() + int(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]
            r = R.add({"action": a.action, "server": a.server, "tool": a.tool, "kind": a.kind, "args": conds,
                       "max_per_hour": a.max_per_hour, "expires": expires, "note": a.note}, first=not a.last)
            print(f"added rule {r['id']}")
        elif a.cmd == "remove":
            R.remove(a.id)
            print(f"removed {a.id}")
        elif a.cmd in ("category", "default"):
            d = R.load()
            if a.cmd == "category":
                d["categories"][a.kind] = a.action
            else:
                d["default"] = a.action
            R.replace(d)
            print("saved")
        elif a.cmd == "test":
            r = R.decide(a.server, a.tool, json.loads(a.args), json.loads(a.annotations), record=False)
            print(f"{r['action']}: {r['reason']}  (kinds: {', '.join(r['kinds'])})")
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
