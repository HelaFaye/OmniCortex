#!/usr/bin/env python3
"""
OmniCortex from a terminal: everything Studio's web pages do, against a
running Studio on this machine or another.

Every command is one of the MCP tools (scripts/ns_mcp.py), so the terminal,
agents and the web pages all go through the same Studio API, plus an
interactive chat and job-log following:

    python scripts/ns.py                                  # the commands
    python scripts/ns.py studio-status
    python scripts/ns.py list-models
    python scripts/ns.py chat                             # interactive; /check, /right, /wrong, /model, /quit
    python scripts/ns.py chat --model auto "Is 2**61-1 prime?"
    python scripts/ns.py start-job --kind testqa --values '{"endpoint": ["m=http://127.0.0.1:7870/v1@MODEL"], "out": "runs/t.json"}'
    python scripts/ns.py follow JOB_ID                    # a job's log as it runs
    python scripts/ns.py get-project --id 20261012-5a19c04a
    python scripts/ns.py review-summary --model acme/model-q4_k_m --subjects math --by month
    python scripts/ns.py --studio https://gpu-box:7870 --token-file ~/.config/neuronscope/studio.token --cafile studio.pem list-models

--json prints the raw answer. Settings, workspaces and tool rules are local
files: scripts/ns_workspace.py and scripts/tool_rules.py.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_mcp  # noqa: E402

KIND = {"string": str, "integer": int, "number": float}


class Client(ns_mcp.Studio):
    def __init__(self, base, token=None, cafile=None):
        super().__init__(base, token)
        if cafile and base.startswith("https://"):
            self.ctx = ssl.create_default_context(cafile=os.path.expanduser(cafile))


def _cmd(name: str) -> str:
    return name.replace("_", "-")


def _add_tool(sub, name, desc, schema):
    p = sub.add_parser(_cmd(name), help=desc.split(". ")[0][:100], description=desc)
    req = set(schema.get("required") or [])
    for prop, spec in (schema.get("properties") or {}).items():
        flag = "--" + prop.replace("_", "-")
        t = spec.get("type")
        if t == "boolean":
            p.add_argument(flag, dest=prop, action="store_true")
        elif t == "array":
            item = (spec.get("items") or {}).get("type")
            p.add_argument(flag, dest=prop, action="append", required=prop in req,
                           type=(json.loads if item == "object" else str),
                           help="repeatable" + (" (JSON objects)" if item == "object" else ""))
        elif t == "object":
            p.add_argument(flag, dest=prop, type=json.loads, required=prop in req, help="JSON")
        else:
            p.add_argument(flag, dest=prop, type=KIND.get(t, str), required=prop in req,
                           choices=spec.get("enum"), help=spec.get("description"))
    return p


# ------------------------------------------------------------------ output

def show(name: str, out) -> None:
    """Readable output for the common commands; JSON for the rest."""
    if name == "list_models" and isinstance(out, dict) and "models" in out:
        for m in out["models"]:
            stats = (f"right {m['accuracy']:.0%} · halluc {m['hallucination_rate']:.0%} · n {m['graded']}"
                     if m.get("graded") else "no graded results")
            flags = " ".join(x for x in ("loaded" if m.get("loaded") else "", "vision" if m.get("vision") else "",
                                         "auto" if m.get("auto_eligible") else "") if x)
            print(f"{m['id']:<44} {m.get('quant') or '':<8} {stats}" + (f"  [{flags}]" if flags else ""))
        return
    if name == "job_status" and isinstance(out, dict):
        rows = out.get("jobs") or [out]
        for j in rows:
            print(f"{j.get('id')}  {j.get('status', ''):<9} {j.get('title', '')}")
        return
    if name == "list_projects" and isinstance(out, dict):
        for p in out.get("projects", []):
            print(f"{p['id']}  {p['status']:<9} {p['title']}")
        return
    if name == "list_features" and isinstance(out, dict):
        for d in out.get("dicts", []):
            kept = f" · keeps {d['recovered']:.0%}" if d.get("recovered") is not None else ""
            print(f"dictionary  {d['name']:<24} {d.get('arch') or ''} {d.get('site')}.{d.get('layer')} "
                  f"{d.get('n_features')} features{kept}" + ("" if d.get("interpreted") else " · not described"))
        for e in out.get("emotions", []):
            acc = f" · {e['best_accuracy']:.0%} at layer {e.get('best_layer')}" if e.get("best_accuracy") is not None else ""
            print(f"emotions    {e['name']:<24} {e.get('n')} emotions{acc}")
        for g in out.get("graphs", []):
            print(f"graph       {g['name']:<24} {(g.get('prompt') or '')[:60]}")
        if not any(out.get(k) for k in ("dicts", "emotions", "graphs")):
            print("none yet: start-job --kind feat_collect, then feat_train (docs/FEATURES.md)")
        return
    if name == "chat" and isinstance(out, dict):
        print(out.get("reply", ""))
        return
    print(json.dumps(out, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ chat

def chat(st: Client, model: str, first: str | None) -> int:
    msgs: list[dict] = []
    last_check = None
    print(f"chatting with {model or 'the loaded model'} on {st.base}. /check /right /wrong /model NAME /new /quit")

    def say(text):
        nonlocal last_check
        msgs.append({"role": "user", "content": text})
        try:
            r = st.call("POST", "/v1/chat/completions", {"model": model, "messages": msgs, "stream": False})
        except ns_mcp.ToolError as e:
            msgs.pop()
            print(f"! {e}")
            return
        reply = r["choices"][0]["message"].get("content") or ""
        msgs.append({"role": "assistant", "content": reply})
        route = (r.get("route") or {}).get("reason")
        print(f"\n[{r.get('model', model)}]" + (f" ({route})" if route else "") + f"\n{reply}\n")
        last_check = None

    if first:
        say(first)
        return 0
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/new":
            msgs, last_check = [], None
            continue
        if line.startswith("/model"):
            model = line.split(maxsplit=1)[1] if " " in line else ""
            print(f"model: {model or 'the loaded model'}")
            continue
        if line == "/check":
            if not msgs or msgs[-1]["role"] != "assistant":
                print("nothing to check yet")
                continue
            try:
                r = st.call("POST", "/api/trace", {"model": model if model not in ("", "auto") else None,
                                                   "messages": msgs[:-1], "text": msgs[-1]["content"]}, timeout=1800)
            except ns_mcp.ToolError as e:
                print(f"! {e}")
                continue
            flagged = [r["pieces"][i] for i in r["flagged"]]
            print(f"risk: {len(flagged)} of {len(r['prob'])} tokens flagged · peak {r['max']:.0%} · "
                  f"3D: {st.base}/{r['url']}")
            if flagged:
                print("  flagged: " + "".join(flagged)[:300])
            last_check = r.get("review")
            continue
        if line in ("/right", "/wrong"):
            if not last_check:
                print("check the reply first (/check)")
                continue
            st.call("POST", "/api/review/label", {"model": last_check["model"], "id": last_check["id"],
                                                  "verdict": "correct" if line == "/right" else "wrong"})
            print("noted for the neuron review")
            continue
        say(line)


def follow(st: Client, jid: str) -> int:
    seen = 0
    while True:
        log = st.call("GET", f"/api/jobs/{jid}/log").get("log", "")
        sys.stdout.write(log[seen:])
        sys.stdout.flush()
        seen = len(log)
        jobs = st.call("GET", "/api/jobs")
        j = next((x for x in jobs if x.get("id") == jid), None)
        if j is None or j.get("status") not in ("running", "queued"):
            print(f"\n[{(j or {}).get('status', 'gone')}]")
            return 0 if j and j.get("status") == "done" else 1
        time.sleep(2)


def settings(st: Client, a) -> int:
    if a.cmd == "restart":
        st.call("POST", "/api/restart", {})
        print("Studio is restarting")
        return 0
    if a.cmd == "workspaces":
        d = st.call("POST", "/api/workspaces/use", {"name": a.use}) if a.use else st.call("GET", "/api/workspaces")
        for w in d["workspaces"]:
            r = w["resolved"]
            print(f"{'*' if w['active'] else ' '} {w['name']:<14} root {r['root']}  models {', '.join(r['models_dirs'][:2])}")
        if d.get("restart_needed"):
            print("Studio switches after a restart: python scripts/ns.py restart")
        return 0
    if a.cmd == "subject-defaults":
        d = st.call("GET", "/api/settings")
        if a.set:
            vals = {k: v["chosen"] for k, v in d["subjects"].items()}
            for item in a.set:
                k, _, v = item.partition("=")
                vals[k] = v or None
            d = st.call("POST", "/api/settings/subjects", {"subject_models": vals})
        for k, v in d["subjects"].items():
            print(f"{k:<20} uses {v['uses'] or '-':<40} measured best {v['best'] or '-'}"
                  + ("  (your pick)" if v["chosen"] else ""))
        return 0
    d = st.call("GET", "/api/settings")["tool_rules"]
    if a.default or a.kind:
        if a.default:
            d["default"] = a.default
        for item in a.kind:
            k, _, v = item.partition("=")
            d["categories"][k] = v
        d = st.call("POST", "/api/settings/tool-rules", {"rules": d})["tool_rules"]
    print(f"default: {d['default']}  ·  " + ", ".join(f"{k} {v}" for k, v in d["categories"].items()))
    print(f"{len(d['rules'])} rule(s) first: " + "; ".join(f"{r['action']} {r['server']}/{r['tool']}" for r in d["rules"]))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="commands: " + ", ".join(
        ["chat", "follow"] + [_cmd(t[0]) for t in ns_mcp.TOOLS]))
    p.add_argument("--studio", default=os.environ.get("NS_STUDIO_URL", "http://127.0.0.1:7870"),
                   help="Studio's address (default $NS_STUDIO_URL or http://127.0.0.1:7870)")
    p.add_argument("--token-file", help="Studio's token (or a paired device's); default $NS_STUDIO_TOKEN")
    p.add_argument("--cafile", help="certificate to trust for a self-signed Studio")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("chat", help="interactive chat (or one message)")
    c.add_argument("--model", default="", help="a model id, or auto (default: the loaded model)")
    c.add_argument("message", nargs="?")
    f = sub.add_parser("follow", help="print a job's log until it ends")
    f.add_argument("id")
    w = sub.add_parser("workspaces", help="Studio's workspaces; --use NAME switches (after a restart)")
    w.add_argument("--use")
    sd = sub.add_parser("subject-defaults", help="the model auto uses per subject; --set SUBJECT=MODEL (empty "
                                                 "MODEL: back to the measured best)")
    sd.add_argument("--set", action="append", default=[])
    tq = sub.add_parser("tool-approval", help="quick settings for chat tool calls: --default ACTION, "
                                              "--kind KIND=ACTION (rules: scripts/tool_rules.py on the Studio machine)")
    tq.add_argument("--default", choices=("ask", "allow", "deny"))
    tq.add_argument("--kind", action="append", default=[])
    sub.add_parser("restart", help="restart Studio")
    tools = {}
    for name, desc, schema, _ann, fn in ns_mcp.TOOLS:
        if name == "chat":
            continue
        _add_tool(sub, name, desc, schema)
        tools[_cmd(name)] = (name, fn)
    argv = list(sys.argv[1:] if argv is None else argv)
    want_json = "--json" in argv                       # accepted before or after the command
    argv = [x for x in argv if x != "--json"]
    a = p.parse_args(argv)
    a.json = a.json or want_json
    if not a.cmd:
        p.print_help()
        return 0
    token = os.environ.get("NS_STUDIO_TOKEN") or None
    if a.token_file:
        token = Path(os.path.expanduser(a.token_file)).read_text().strip()
    st = Client(a.studio, token, a.cafile)
    try:
        if a.cmd == "chat":
            return chat(st, a.model, a.message)
        if a.cmd == "follow":
            return follow(st, a.id)
        if a.cmd in ("workspaces", "subject-defaults", "tool-approval", "restart"):
            return settings(st, a)
        name, fn = tools[a.cmd]
        args = {k: v for k, v in vars(a).items() if k not in ("studio", "token_file", "cafile", "json", "cmd")
                and v not in (None, False)}
        out = fn(st, args)
    except ns_mcp.ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
    else:
        show(name, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
