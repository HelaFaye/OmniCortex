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
import math
import os
import re
import itertools
import regex as timeout_regex
import sys
import tempfile
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
MAX_RULES = 500
MAX_CONDITIONS = 12
MAX_ARGUMENT_VALUES = 128
MAX_ARGUMENT_NODES = 4096
MAX_ARGUMENT_DEPTH = 64
MAX_ARGUMENT_PATH_CHARS = 256
MAX_RULE_FILE_BYTES = 2 * 1024 * 1024
MAX_RATE_HISTORY = 500
REGEX_TIMEOUT_SECONDS = 0.025
RULE_MATCH_BUDGET_SECONDS = 0.5
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


class RuleRegexTimeout(TimeoutError):
    """A rule could not be evaluated safely within the configured time budget."""


def _walk_values(obj, parts=None, deadline=None):
    """Iteratively traverse JSON-like arguments under node, depth and time limits."""
    stack = [("visit", obj, 0, 0)]
    visited = 0

    while stack:
        if deadline is not None and time.monotonic() >= deadline:
            raise RuleRegexTimeout("tool-rule evaluation budget exceeded")

        kind, value, part_index, depth = stack.pop()

        if kind == "iter":
            try:
                child = next(value)
            except StopIteration:
                continue
            stack.append(("iter", value, part_index, depth))
            stack.append(("visit", child, part_index, depth + 1))
            continue

        visited += 1
        if visited > MAX_ARGUMENT_NODES:
            raise RuleRegexTimeout(
                f"argument traversal exceeded {MAX_ARGUMENT_NODES} nodes"
            )
        if depth > MAX_ARGUMENT_DEPTH:
            raise RuleRegexTimeout(
                f"argument traversal exceeded depth {MAX_ARGUMENT_DEPTH}"
            )

        if parts is None:
            if isinstance(value, dict):
                stack.append(("iter", iter(value.values()), 0, depth))
            elif isinstance(value, list):
                stack.append(("iter", iter(value), 0, depth))
            elif value is not None:
                yield value
            continue

        if part_index >= len(parts):
            yield value
            continue

        head = parts[part_index]
        if head == "*":
            if isinstance(value, dict):
                stack.append(("iter", iter(value.values()), part_index + 1, depth))
            elif isinstance(value, list):
                stack.append(("iter", iter(value), part_index + 1, depth))
        elif isinstance(value, dict) and head in value:
            stack.append(("visit", value[head], part_index + 1, depth + 1))
        elif isinstance(value, list) and head.isdigit():
            index = int(head)
            if index < len(value):
                stack.append(("visit", value[index], part_index + 1, depth + 1))


def _values(obj, parts, deadline=None):
    yield from _walk_values(obj, parts, deadline)


def _strings(obj, deadline=None):
    yield from _walk_values(obj, None, deadline)


def _text(value, deadline=None) -> str:
    """Convert a value to bounded text without serializing an unlimited object."""
    if isinstance(value, str):
        return value[:MAX_MATCH_CHARS]

    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
    pieces = []
    length = 0

    try:
        for chunk in encoder.iterencode(value):
            if deadline is not None and time.monotonic() >= deadline:
                raise RuleRegexTimeout("tool-rule evaluation budget exceeded")

            remaining = MAX_MATCH_CHARS - length
            if remaining <= 0:
                break

            pieces.append(chunk[:remaining])
            length += min(len(chunk), remaining)
            if len(chunk) >= remaining:
                break

        return "".join(pieces)
    except RuleRegexTimeout:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuleRegexTimeout(
            "argument value could not be serialized safely"
        ) from exc


def _compile_safe_regex(pattern: str):
    if len(pattern) > MAX_REGEX:
        raise ValueError(f"regex longer than {MAX_REGEX} characters")
    try:
        return timeout_regex.compile(pattern)
    except timeout_regex.error as exc:
        raise ValueError(f"bad regex {pattern!r}: {exc}") from exc


def cond_holds(cond: dict, args: dict, deadline: float | None = None) -> bool:
    rx = _compile_safe_regex(cond["regex"])
    if deadline is not None and time.monotonic() >= deadline:
        raise RuleRegexTimeout("tool-rule evaluation budget exceeded")

    path = cond.get("path") or "*"
    source = (
        _strings(args, deadline)
        if path == "*"
        else _values(args, path.split("."), deadline)
    )

    hit = False
    count = 0
    for value in source:
        count += 1
        if count > MAX_ARGUMENT_VALUES:
            raise RuleRegexTimeout(
                f"rule argument path matched more than {MAX_ARGUMENT_VALUES} values"
            )

        text = _text(value, deadline)
        remaining = REGEX_TIMEOUT_SECONDS
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuleRegexTimeout("tool-rule evaluation budget exceeded")
            remaining = min(remaining, REGEX_TIMEOUT_SECONDS)

        try:
            if not hit and rx.search(text, timeout=remaining):
                hit = True
        except TimeoutError as exc:
            raise RuleRegexTimeout(
                "regular-expression evaluation timed out"
            ) from exc

    return (not hit) if cond.get("negate") else hit


def rule_matches(rule: dict, server: str, tool: str, kinds: list[str], args: dict,
                 deadline: float | None = None) -> bool:
    if not fnmatch.fnmatchcase(server, rule.get("server") or "*"):
        return False
    if not fnmatch.fnmatchcase(tool, rule.get("tool") or "*"):
        return False
    k = rule.get("kind") or "any"
    if k != "any" and k not in kinds:
        return False
    return all(cond_holds(c, args, deadline) for c in rule.get("args") or [])


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
    conditions = r.get("args") or []
    if not isinstance(conditions, list) or len(conditions) > MAX_CONDITIONS:
        raise ValueError(f"a rule may have at most {MAX_CONDITIONS} argument conditions")
    for c in conditions:
        if not isinstance(c, dict) or not isinstance(c.get("regex"), str):
            raise ValueError("argument conditions are {path, regex, negate}")
        try:
            _compile_safe_regex(c["regex"])
        except ValueError as e:
            raise ValueError(str(e)) from e
        path = str(c.get("path") or "*")
        if len(path) > MAX_ARGUMENT_PATH_CHARS:
            raise ValueError(
                f"argument path exceeds {MAX_ARGUMENT_PATH_CHARS} characters"
            )
        if not re.fullmatch(
            r"(?:\*|[A-Za-z0-9_-]+)(?:\.(?:\*|[A-Za-z0-9_-]+))*",
            path,
        ):
            raise ValueError(f"bad argument path {path!r}")
        out["args"].append({"path": path, "regex": c["regex"], "negate": bool(c.get("negate"))})
    if r.get("max_per_hour") not in (None, "", 0):
        try:
            n = int(r["max_per_hour"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_per_hour must be an integer") from exc
        if not 1 <= n <= MAX_RATE_HISTORY:
            raise ValueError(f"max_per_hour must be between 1 and {MAX_RATE_HISTORY}")
        if action != "allow":
            raise ValueError("max_per_hour limits allow rules")
        out["max_per_hour"] = n

    if r.get("expires") not in (None, ""):
        try:
            expires = float(r["expires"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("expires must be a finite timestamp") from exc
        if not math.isfinite(expires):
            raise ValueError("expires must be a finite timestamp")
        out["expires"] = expires

    try:
        hits = int(r.get("hits") or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hits must be a nonnegative integer") from exc
    if hits < 0:
        raise ValueError("hits must be a nonnegative integer")
    out["hits"] = min(hits, 2**31 - 1)

    raw_recent = r.get("recent") or []
    if not isinstance(raw_recent, list):
        raise ValueError("recent rate-limit history must be a list")
    recent = []
    for timestamp in raw_recent[-MAX_RATE_HISTORY:]:
        try:
            value = float(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("recent rate-limit timestamps must be finite") from exc
        if not math.isfinite(value):
            raise ValueError("recent rate-limit timestamps must be finite")
        recent.append(value)
    out["recent"] = recent
    return out


class Rules:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("OC_TOOL_RULES") or DEFAULT_PATH).expanduser()

    def load(self) -> dict:
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(MAX_RULE_FILE_BYTES + 1)
            if len(raw) > MAX_RULE_FILE_BYTES:
                return blank()
            d = json.loads(raw)
        except (OSError, ValueError, TypeError, UnicodeError):
            return blank()

        if not isinstance(d, dict):
            return blank()

        out = blank()
        default = d.get("default")
        if isinstance(default, str) and default in ACTIONS:
            out["default"] = default

        categories = d.get("categories")
        if isinstance(categories, dict):
            for key, value in categories.items():
                if isinstance(key, str) and isinstance(value, str):
                    if key in KINDS and value in CAT_ACTIONS:
                        out["categories"][key] = value

        raw_rules = d.get("rules")
        if not isinstance(raw_rules, list):
            raw_rules = []

        for rule in raw_rules[:MAX_RULES]:
            try:
                out["rules"].append(validate_rule(rule))
            except (ValueError, TypeError, OverflowError):
                continue

        return out

    def save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        tmp = Path(tmp_name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                fd = -1
                json.dump(d, stream, indent=1)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if fd >= 0:
                os.close(fd)
            tmp.unlink(missing_ok=True)

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
        raw_rules = d.get("rules") or []
        if not isinstance(raw_rules, list) or len(raw_rules) > MAX_RULES:
            raise ValueError(f"a ruleset may contain at most {MAX_RULES} rules")
        out["rules"] = [validate_rule(r) for r in raw_rules]
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
            if len(d["rules"]) >= MAX_RULES:
                raise ValueError(f"a ruleset may contain at most {MAX_RULES} rules")
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
        """Return allow, ask or deny; evaluation failures require manual approval."""
        now = time.time() if now is None else now
        kinds = kinds_of(annotations)
        deadline = time.monotonic() + RULE_MATCH_BUDGET_SECONDS

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not _LOCK.acquire(timeout=max(0.0, remaining)):
            return {
                "action": "ask",
                "by": "rule",
                "reason": "tool-rule evaluation lock is busy; manual approval required",
                "kinds": kinds,
            }

        try:
            d = self.load()
            if time.monotonic() >= deadline:
                return {
                    "action": "ask",
                    "by": "rule",
                    "reason": "tool-rule evaluation budget exceeded; manual approval required",
                    "kinds": kinds,
                }

            for rule in d["rules"]:
                if time.monotonic() >= deadline:
                    return {
                        "action": "ask",
                        "by": "rule",
                        "reason": "tool-rule evaluation budget exceeded; manual approval required",
                        "kinds": kinds,
                    }

                if not rule["enabled"] or (
                    rule.get("expires") and rule["expires"] < now
                ):
                    continue

                try:
                    if not rule_matches(
                        rule, server, tool, kinds, args or {}, deadline
                    ):
                        continue
                except RuleRegexTimeout as exc:
                    return {
                        "action": "ask",
                        "by": "rule",
                        "rule": rule["id"],
                        "reason": (
                            "rule evaluation timed out; manual approval required "
                            f"({exc})"
                        ),
                        "kinds": kinds,
                    }
                except re.error:
                    continue

                action = rule["action"]
                reason = f"rule {rule['id']}" + (
                    f" ({rule['note']})" if rule["note"] else ""
                )

                if action == "allow" and rule.get("max_per_hour"):
                    recent = [
                        timestamp for timestamp in rule["recent"]
                        if now - 3600 < timestamp <= now
                    ]
                    if len(recent) >= rule["max_per_hour"]:
                        action = "ask"
                        reason = (
                            f"rule {rule['id']} allows "
                            f"{rule['max_per_hour']} an hour; limit reached"
                        )
                    elif record:
                        rule["recent"] = recent + [now]

                if record:
                    rule["hits"] += 1
                    rule["last_hit"] = now
                    self.save(d)

                return {
                    "action": action,
                    "by": "rule",
                    "rule": rule["id"],
                    "reason": reason,
                    "kinds": kinds,
                }

            if time.monotonic() >= deadline:
                return {
                    "action": "ask",
                    "by": "rule",
                    "reason": "tool-rule evaluation budget exceeded; manual approval required",
                    "kinds": kinds,
                }
        finally:
            _LOCK.release()

        if auto_approve is True or (
            isinstance(auto_approve, list) and tool in auto_approve
        ):
            return {
                "action": "allow",
                "by": "autoApprove",
                "reason": f"{server}'s autoApprove in mcp.json",
                "kinds": kinds,
            }

        order = {"deny": 2, "ask": 1, "allow": 0}
        settings = [
            (d["categories"].get(kind, "default"), kind)
            for kind in kinds
        ]
        if all(value == "default" for value, _ in settings):
            return {
                "action": d["default"],
                "by": "default",
                "reason": f"default: {d['default']}",
                "kinds": kinds,
            }

        action, kind = max(
            (
                (d["default"] if value == "default" else value, kind)
                for value, kind in settings
            ),
            key=lambda item: order[item[0]],
        )
        return {
            "action": action,
            "by": "category",
            "reason": f"{kind.replace('_', ' ')} tools: {action}",
            "kinds": kinds,
        }


def rule_from_call(kind: str, server: str, tool: str, args: dict) -> dict:
    """Create an approval rule without silently broadening exact-argument rules."""
    if kind == "tool":
        return {
            "action": "allow",
            "server": server,
            "tool": tool,
            "note": "always allowed from a chat",
        }
    if kind == "deny":
        return {
            "action": "deny",
            "server": server,
            "tool": tool,
            "note": "always denied from a chat",
        }
    if kind == "args":
        if not isinstance(args, dict) or not args:
            raise ValueError("an exact-argument rule requires a non-empty argument object")
        if len(args) > MAX_CONDITIONS:
            raise ValueError(
                f"exact-argument rules support at most {MAX_CONDITIONS} arguments"
            )

        keys = list(args)
        for key in keys:
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", key)
            ):
                raise ValueError(
                    f"argument key {key!r} cannot be represented as a safe exact-match path"
                )

        conds = []
        for key in sorted(keys):
            text = _text(args[key])
            if len(text) >= MAX_MATCH_CHARS:
                raise ValueError(
                    f"argument {key!r} exceeds the exact-match text limit"
                )

            pattern = "^" + re.escape(text) + "$"
            if len(pattern) > MAX_REGEX:
                raise ValueError(
                    f"argument {key!r} is too complex for an exact-match rule"
                )

            conds.append({
                "path": key,
                "regex": pattern,
            })

        return {
            "action": "allow",
            "server": server,
            "tool": tool,
            "args": conds,
            "note": "these arguments, allowed from a chat",
        }

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
