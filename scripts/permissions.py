#!/usr/bin/env python3
"""
What a model may do: read, write, run commands, start programs, reach the
network, touch system components, call tools. Per model and per config
(docs/PERMISSIONS.md).

A **rule** allows, asks or denies one action on targets that match a pattern:

    {"action": "run", "effect": "deny", "match": "regex", "pattern": "\\brm\\s+-rf\\s+/"}
    {"action": "write", "effect": "allow", "match": "glob", "pattern": "{workdir}/**"}

    actions  read (a path), write (a path, also deletes), run (a command line),
             exec (a program, by name or path), net (a host or URL),
             system (a component: gpu, docker, packages, services, power,
             drivers, users, registry, firewall, clipboard, screen, camera,
             microphone), tool (an MCP tool as server/tool), or * for any
    match    exact, prefix, glob (* within a path segment, ** across them) or regex
    pattern  may use {workdir}, {repo}, {home}, {tmp} and {omnicortex}, filled
             in from the context of the check

Allow rules are the whitelist and deny rules the blacklist. A **profile** is
a named set of rules plus a default effect per action. A check stacks:

  1. the base profile (default: "standard");
  2. the profiles bound to the model (glob on the model id), in order;
  3. the job's profile: each job kind has a preset that fits it (a QA run
     needs no actions; a SWE-bench run needs git, Python and Docker inside
     its checkout), which the owner can change;
  4. a profile asked for explicitly (a project task, a node in a graph).

Every matching rule in every layer counts, and the most restrictive wins:
deny over ask over allow. So a job can open up what it needs but never
lift a deny from the base or the model's profile. With no matching rule, the
default comes from the most specific layer that sets one for the action, then
"ask".

    python scripts/permissions.py                          # the policy
    python scripts/permissions.py check --model qwen --job swe_evaluate run "git apply x.patch" --repo /work/r
    python scripts/permissions.py presets
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import re
import threading
from pathlib import Path

DEFAULT_PATH = Path.home() / ".omnicortex" / "permissions.json"
ACTIONS = {
    "read": "files read (a path)",
    "write": "files written, changed or deleted (a path)",
    "run": "shell command lines (the whole line)",
    "exec": "programs started (name or path)",
    "net": "network destinations (host or URL)",
    "system": "system components: gpu, docker, packages, services, power, drivers, users, registry, firewall, "
              "clipboard, screen, camera, microphone",
    "tool": "MCP tools (server/tool)",
}
EFFECTS = ("allow", "ask", "deny")
RANK = {"allow": 0, "ask": 1, "deny": 2}
MATCHES = ("exact", "prefix", "glob", "regex")
MAX_PATTERN = 500
MAX_TARGET = 8000
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")
_LOCK = threading.Lock()

# Commands that are never what a model should be doing, whatever the job.
_NEVER_RUN = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|~|\$HOME)(\s|$)", "removes a home or root folder"),
    (r"\bmkfs(\.\w+)?\b|\bwipefs\b", "formats a disk"),
    (r"\bdd\b[^|;&]*\bof=/dev/", "writes a raw device"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "powers the machine down"),
    (r"(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", "pipes a download into a shell"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "opens permissions on the root"),
    (r"\bsudo\b", "asks for administrator rights"),
]
_SECRETS = ["{home}/.ssh/**", "{home}/.gnupg/**", "{home}/.aws/**", "{home}/.config/gh/**",
            "{home}/.omnicortex/*token*", "{home}/.netrc", "**/.env", "**/*.pem", "**/id_rsa*", "**/id_ed25519*"]


def _r(action, effect, match, pattern, note=""):
    return {"action": action, "effect": effect, "match": match, "pattern": pattern, "note": note}


def presets() -> dict:
    """The built-in profiles. Copies, so callers may change them."""
    base_deny = ([_r("run", "deny", "regex", rx, why) for rx, why in _NEVER_RUN]
                 + [_r("read", "deny", "glob", g, "secrets") for g in _SECRETS]
                 + [_r("write", "deny", "glob", g, "secrets") for g in _SECRETS]
                 + [_r("system", "deny", "exact", c, "not for a model") for c in ("power", "users", "registry", "drivers", "firewall")])
    return copy.deepcopy({
        "standard": {
            "description": "The base for every check: never the dangerous commands or your secrets; reading is "
                           "fine, everything else asks.",
            "defaults": {"read": "allow", "write": "ask", "run": "ask", "exec": "ask", "net": "ask", "system": "ask",
                         "tool": "allow"},
            "rules": base_deny},
        "read_only": {
            "description": "Look, do not touch: reading only.",
            "defaults": {"read": "allow", "write": "deny", "run": "deny", "exec": "deny", "net": "deny",
                         "system": "deny", "tool": "ask"},
            "rules": []},
        "no_actions": {
            "description": "Answering questions needs no actions at all (TriviaQA, LiveBench, TestQA without code).",
            "defaults": {a: "deny" for a in ACTIONS},
            "rules": []},
        "code_sandboxed": {
            "description": "Run candidate code in a container: Python, Docker or Podman, writing only in the job's "
                           "folder and explicitly assigned scratch directory, no network.",
            "defaults": {"read": "allow", "write": "deny", "run": "deny", "exec": "deny", "net": "deny",
                         "system": "deny", "tool": "deny"},
            "rules": [_r("exec", "allow", "regex", r"^(.*[/\\])?(python[\d.]*|docker|podman)(\.exe)?$", "the sandbox"),
                      _r("run", "allow", "regex", r"^(python[\d.]*|docker|podman)(\.exe)?\s", "the sandbox"),
                      _r("write", "allow", "glob", "{workdir}/**", "the job's own folder"),
                      _r("write", "allow", "glob", "{tmp}/**", "scratch files"),
                      _r("system", "allow", "exact", "docker", "containers")]},
        "swe_bench": {
            "description": "Fix a repository and run its tests: git, Python, pytest, pip and Docker inside the "
                           "checkout and explicitly assigned scratch directory; packages from PyPI; nothing else.",
            "defaults": {"read": "deny", "write": "deny", "run": "deny", "exec": "deny", "net": "deny",
                         "system": "deny", "tool": "deny"},
            "rules": [_r("read", "allow", "glob", "{repo}/**", "the checkout"),
                      _r("read", "allow", "glob", "{tmp}/**", "scratch files"),
                      _r("write", "allow", "glob", "{repo}/**", "the checkout"),
                      _r("write", "allow", "glob", "{tmp}/**", "scratch files"),
                      _r("write", "deny", "glob", "{repo}/.git/hooks/**", "git hooks run code later"),
                      _r("exec", "allow", "regex", r"^(.*[/\\])?(git|python[\d.]*|pytest|pip[\d.]*|docker|tox|make)(\.exe)?$", "build and test"),
                      _r("run", "allow", "regex", r"^(git|python[\d.]*|pytest|pip[\d.]*|docker|tox|make)(\.exe)?\s", "build and test"),
                      _r("run", "deny", "regex", r"\bgit\s+push\b", "nothing leaves the machine"),
                      _r("net", "allow", "regex", r"^(https?://)?(pypi\.org|files\.pythonhosted\.org|github\.com|"
                                                  r"codeload\.github\.com)(/|:|$)", "packages and sources"),
                      _r("system", "allow", "exact", "docker", "the evaluation harness")]},
        "developer": {
            "description": "A coding agent in your projects: read and write inside the project, run its build "
                           "tools, ask for anything else.",
            "defaults": {"read": "allow", "write": "ask", "run": "ask", "exec": "ask", "net": "ask", "system": "ask",
                         "tool": "allow"},
            "rules": [_r("write", "allow", "glob", "{workdir}/**", "the project"),
                      _r("exec", "allow", "regex", r"^(.*[/\\])?(git|python[\d.]*|pytest|node|npm|cargo|make|cmake|go)(\.exe)?$", "build tools"),
                      _r("run", "allow", "regex", r"^(git (status|diff|log|add|commit)|pytest|npm test|cargo test|make)\b",
                         "routine build and test commands")]},
        "full": {
            "description": "Everything allowed, except what the base profile denies (dangerous commands, secrets). "
                           "For a model you trust on a machine you can rebuild.",
            "defaults": {a: "allow" for a in ACTIONS},
            "rules": []},
    })


# The preset each job kind starts with ("" for jobs whose model takes no actions of its own).
JOB_PRESETS = {
    "testqa": "code_sandboxed", "livebench": "no_actions", "eval_lmstudio": "no_actions", "tune_adaptive": "no_actions",
    "swe_predict": "no_actions", "swe_evaluate": "swe_bench", "clip_bench": "no_actions", "compare_models": "no_actions",
    "review_testqa": "no_actions", "judge": "no_actions", "emo_stories": "no_actions", "feat_interpret": "no_actions",
    "project_task": "developer",
}


def blank() -> dict:
    return {"base": "standard", "profiles": {}, "bindings": [], "jobs": {}}


# ------------------------------------------------------------------ matching

def expand(pattern: str, ctx: dict) -> str:
    def sub(m):
        v = ctx.get(m.group(1))
        return str(v).rstrip("/") if v else "\0unset\0"        # an unset variable matches nothing
    return re.sub(r"\{(workdir|repo|home|tmp|omnicortex)\}", sub, pattern)


def _glob_rx(g: str) -> str:
    out, i = "", 0
    while i < len(g):
        if g.startswith("**", i):
            out += ".*"
            i += 2
        elif g[i] == "*":
            out += "[^/]*"
            i += 1
        elif g[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(g[i])
            i += 1
    return "^" + out + "$"


def matches(rule: dict, target: str, ctx: dict) -> bool:
    pat = expand(rule["pattern"], ctx)
    if "\0unset\0" in pat:
        return False
    t = target[:MAX_TARGET]
    m = rule.get("match", "glob")
    if m == "exact":
        return t == pat
    if m == "prefix":
        return t.startswith(pat)
    if m == "regex":
        return re.search(pat, t) is not None
    return re.match(_glob_rx(pat), t) is not None


def normalise(action: str, target: str, ctx: dict) -> str:
    """Paths absolute and without .. (so a pattern cannot be walked around), programs by name or path."""
    target = str(target).strip()
    if action in ("read", "write"):
        p = os.path.expanduser(target)
        if not os.path.isabs(p) and ctx.get("workdir"):
            p = os.path.join(ctx["workdir"], p)
        # Resolve existing symlinks before policy matching. Lexical normalization alone
        # lets an allowed workspace path point outside the workspace through a symlink.
        # strict=False also handles not-yet-created output files while resolving their
        # existing parent directories.
        return str(Path(p).resolve(strict=False))
    if action == "system":
        return target.lower()
    return target


def context(**kw) -> dict:
    # Scratch access is deliberately absent unless the caller supplies a
    # job-owned directory. tempfile.gettempdir() is shared by unrelated jobs.
    ctx = {"home": str(Path.home().resolve(strict=False)),
           "omnicortex": str(Path(__file__).resolve().parent.parent)}
    for k, v in kw.items():
        if v:
            ctx[k] = str(Path(os.path.expanduser(str(v))).resolve(strict=False))
    return ctx


# ------------------------------------------------------------------ validation

def validate_rule(r: dict) -> dict:
    if not isinstance(r, dict):
        raise ValueError("a rule is an object")
    a, e, m, p = r.get("action"), r.get("effect"), r.get("match", "glob"), r.get("pattern")
    if a != "*" and a not in ACTIONS:
        raise ValueError(f"action: one of {', '.join(ACTIONS)} or *")
    if e not in EFFECTS:
        raise ValueError(f"effect: one of {', '.join(EFFECTS)}")
    if m not in MATCHES:
        raise ValueError(f"match: one of {', '.join(MATCHES)}")
    if not isinstance(p, str) or not p or len(p) > MAX_PATTERN:
        raise ValueError(f"pattern: 1 to {MAX_PATTERN} characters")
    if m == "regex":
        try:
            re.compile(p)
        except re.error as err:
            raise ValueError(f"pattern: not a regular expression ({err})")
    return {"action": a, "effect": e, "match": m, "pattern": p, "note": str(r.get("note") or "")[:200]}


def validate_profile(name: str, prof: dict) -> dict:
    if not NAME.match(name or ""):
        raise ValueError(f"profile name {name!r}: lowercase letters, digits, - _ .")
    d = prof.get("defaults") or {}
    for a, e in d.items():
        if a not in ACTIONS or e not in EFFECTS:
            raise ValueError(f"{name}: default {a}={e}")
    return {"description": str(prof.get("description") or "")[:300], "defaults": dict(d),
            "rules": [validate_rule(r) for r in prof.get("rules") or []]}


def validate(policy: dict) -> dict:
    out = blank()
    out["profiles"] = {n: validate_profile(n, p) for n, p in (policy.get("profiles") or {}).items()}
    names = set(out["profiles"]) | set(presets())
    base = policy.get("base") or "standard"
    if base not in names:
        raise ValueError(f"base: no profile {base!r}")
    out["base"] = base
    for b in policy.get("bindings") or []:
        if not isinstance(b, dict) or not b.get("model") or b.get("profile") not in names:
            raise ValueError(f"binding {b!r}: needs model (a glob) and an existing profile")
        out["bindings"].append({"model": str(b["model"])[:200], "profile": b["profile"]})
    for k, v in (policy.get("jobs") or {}).items():
        if v and v not in names:
            raise ValueError(f"job {k}: no profile {v!r}")
        out["jobs"][str(k)] = v or ""
    return out


# ------------------------------------------------------------------ the policy

class Policy:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("OC_PERMISSIONS") or DEFAULT_PATH).expanduser()

    def load(self) -> dict:
        try:
            return validate(json.loads(self.path.read_text()))
        except FileNotFoundError:
            return blank()
        except (OSError, ValueError):
            return blank()

    def save(self, policy: dict) -> dict:
        clean = validate(policy)
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(clean, indent=2))
            os.replace(tmp, self.path)
        return clean

    def profiles(self, policy: dict | None = None) -> dict:
        p = policy or self.load()
        return {**presets(), **p["profiles"]}

    def layers(self, model: str = "", job: str = "", profile: str = "", policy: dict | None = None) -> list:
        """[(why, name, profile)], least specific first."""
        pol = policy or self.load()
        profs = self.profiles(pol)
        out = [("base", pol["base"], profs[pol["base"]])]
        for b in pol["bindings"]:
            if model and fnmatch.fnmatchcase(model, b["model"]):
                out.append((f"model {b['model']}", b["profile"], profs[b["profile"]]))
        jp = pol["jobs"].get(job, JOB_PRESETS.get(job, "")) if job else ""
        if jp:
            out.append((f"job {job}", jp, profs[jp]))
        if profile:
            if profile not in profs:
                raise ValueError(f"no profile {profile!r}")
            out.append(("asked for", profile, profs[profile]))
        return out

    def check(self, action: str, target: str, model: str = "", job: str = "", profile: str = "",
              ctx: dict | None = None, policy: dict | None = None) -> dict:
        """-> {"effect", "why", "rule", "target", "layers"}. A command line is checked whole and then
        segment by segment (split on ; && || | newlines and command substitutions), so an allowed
        command cannot carry a disallowed one; the strictest answer wins."""
        if action not in ACTIONS:
            raise ValueError(f"action: one of {', '.join(ACTIONS)}")
        ctx = ctx or context()
        pol = policy or self.load()
        whole = self._check1(action, target, model, job, profile, ctx, pol)
        if action == "run":
            parts = [x.strip() for x in re.split(r";|&&|\|\||\||\n|\$\(|`|\)", str(target)) if x.strip()]
            if len(parts) > 1:
                for part in parts:
                    d = self._check1(action, part, model, job, profile, ctx, pol)
                    if RANK[d["effect"]] > RANK[whole["effect"]]:
                        whole = {**d, "why": f"{part[:60]!r}: {d['why']}", "target": whole["target"]}
        return whole

    def _check1(self, action, target, model, job, profile, ctx, policy) -> dict:
        t = normalise(action, target, ctx)
        layers = self.layers(model, job, profile, policy)
        best = None
        for why, name, prof in layers:
            for r in prof["rules"]:
                if r["action"] in (action, "*") and matches(r, t, ctx):
                    if best is None or RANK[r["effect"]] > RANK[best[0]["effect"]]:
                        best = (r, f"{why} ({name})")
        names = [n for _, n, _ in layers]
        if best:
            r, where = best
            return {"effect": r["effect"], "why": f"{r['note'] or r['pattern']}: {where}", "rule": r,
                    "target": t, "layers": names}
        for why, name, prof in reversed(layers):
            if action in prof["defaults"]:
                return {"effect": prof["defaults"][action], "why": f"default for {action}: {why} ({name})",
                        "rule": None, "target": t, "layers": names}
        return {"effect": "ask", "why": "no rule or default", "rule": None, "target": t, "layers": names}


def most(*effects: str) -> str:
    return max(effects, key=lambda e: RANK[e]) if effects else "allow"


# ------------------------------------------------------------------ tool calls

_PATH_KEYS = ("path", "file", "filename", "filepath", "file_path", "target", "destination", "dest", "source", "src",
              "directory", "dir", "folder", "cwd")
_CMD_KEYS = ("command", "cmd", "script", "shell", "commandline", "command_line")
_URL_KEYS = ("url", "uri", "href", "endpoint", "host")


def actions_of_call(server: str, tool: str, args: dict, read_only: bool | None = None) -> list:
    """What an MCP tool call would do, as (action, target) pairs: the call itself, plus any paths it
    names (read for a read-only tool, else write), command lines and programs, and URLs."""
    out = [("tool", f"{server}/{tool}")]
    path_act = "read" if read_only else "write"

    def walk(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, str(k).lower())
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, str) and o:
            if key in _PATH_KEYS or key.endswith(("_path", "_file", "_dir")):
                out.append((path_act, o))
            elif key in _CMD_KEYS:
                out.append(("run", o))
                prog = o.strip().split()[0] if o.strip() else ""
                if prog:
                    out.append(("exec", prog))
            elif key in _URL_KEYS:
                out.append(("net", o))
    walk(args or {})
    return out


def check_call(policy: Policy, model: str, server: str, tool: str, args: dict, read_only=None, job: str = "",
               profile: str = "", ctx: dict | None = None) -> dict:
    """The most restrictive answer over everything a tool call would do."""
    worst, details = None, []
    for action, target in actions_of_call(server, tool, args, read_only):
        d = policy.check(action, target, model, job, profile, ctx)
        details.append({"action": action, "target": d["target"], "effect": d["effect"], "why": d["why"]})
        if worst is None or RANK[d["effect"]] > RANK[worst["effect"]]:
            worst = details[-1]
    return {"effect": worst["effect"], "why": f"{worst['action']} {worst['target'][:120]}: {worst['why']}",
            "checks": details}


def remember(policy: Policy, model: str, items: list, effect: str = "allow") -> str:
    """"Always allow" from a prompt: exact-target rules in this model's own profile ("remembered-<model>",
    bound to the model's id; "remembered" for all models when there is no id). -> the profile's name."""
    slug = re.sub(r"[^a-z0-9_.-]+", "-", (model or "").lower()).strip("-.")[:28]
    name = f"remembered-{slug}" if slug else "remembered"
    pol = policy.load()
    prof = pol["profiles"].setdefault(name, {"description": f"Choices remembered from prompts for {model or 'every model'}.",
                                              "defaults": {}, "rules": []})
    have = {(r["action"], r["pattern"]) for r in prof["rules"]}
    for action, target in items:
        if action in ACTIONS and target and (action, target) not in have:
            prof["rules"].append({"action": action, "effect": effect, "match": "exact", "pattern": str(target)[:MAX_PATTERN],
                                  "note": "remembered from a prompt"})
    bind = {"model": model or "*", "profile": name}
    if bind not in pol["bindings"]:
        pol["bindings"].append(bind)
    policy.save(pol)
    return name


def prompt_text(policy: Policy, model: str = "", job: str = "", profile: str = "", ctx: dict | None = None) -> str:
    """A short statement of what the model may do, for its system prompt (delegation, graph nodes)."""
    layers = policy.layers(model, job, profile)
    lines = []
    for a in ACTIONS:
        d = "ask"
        for _, _, p in reversed(layers):
            if a in p["defaults"]:
                d = p["defaults"][a]
                break
        allow = [expand(r["pattern"], ctx or context()) for _, _, p in layers for r in p["rules"]
                 if r["action"] == a and r["effect"] == "allow"]
        deny = [r["note"] or r["pattern"] for _, _, p in layers for r in p["rules"] if r["action"] == a and r["effect"] == "deny"]
        s = f"- {a}: {d}"
        if allow:
            s += "; allowed: " + ", ".join(dict.fromkeys(allow))[:300]
        if deny:
            s += "; never: " + ", ".join(dict.fromkeys(deny))[:300]
        lines.append(s)
    return "Permissions for this task (" + " + ".join(n for _, n, _ in layers) + "):\n" + "\n".join(lines)


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help=f"policy file (default $OC_PERMISSIONS or {DEFAULT_PATH})")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="what would happen")
    c.add_argument("action", choices=sorted(ACTIONS))
    c.add_argument("target")
    c.add_argument("--model", default="")
    c.add_argument("--job", default="")
    c.add_argument("--profile", default="")
    c.add_argument("--workdir")
    c.add_argument("--repo")
    sub.add_parser("presets", help="the built-in profiles")
    b = sub.add_parser("bind", help="use a profile for models matching a glob")
    b.add_argument("model")
    b.add_argument("profile")
    j = sub.add_parser("job", help="the profile a job kind uses (empty: its preset)")
    j.add_argument("kind")
    j.add_argument("profile", nargs="?", default="")
    a = p.parse_args(argv)
    pol = Policy(a.file)
    if a.cmd == "check":
        print(json.dumps(pol.check(a.action, a.target, a.model, a.job, a.profile,
                                   context(workdir=a.workdir, repo=a.repo)), indent=1))
    elif a.cmd == "presets":
        for n, pr in presets().items():
            print(f"{n:<15} {pr['description']}")
    elif a.cmd == "bind":
        d = pol.load()
        d["bindings"].append({"model": a.model, "profile": a.profile})
        pol.save(d)
    elif a.cmd == "job":
        d = pol.load()
        if a.profile:
            d["jobs"][a.kind] = a.profile
        else:
            d["jobs"].pop(a.kind, None)
        pol.save(d)
    else:
        print(json.dumps(pol.load(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
