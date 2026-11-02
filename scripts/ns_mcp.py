#!/usr/bin/env python3
"""
OmniCortex as an MCP server: the harness and lab as tools for Cline, Claude
Desktop, LM Studio or any other MCP client.

Every tool is a call to Studio's HTTP API, so an agent sees exactly what the
UI sees and is held to the same rules: Studio's token, owner-only actions,
paired-device limits, jobs only from typed forms. Two ways to connect:

  * streamable HTTP, served by Studio itself at  http://127.0.0.1:7870/mcp
    (nothing to install; send Studio's token as a Bearer header if it has one);
  * stdio, for clients that start a process:
        python scripts/ns_mcp.py --studio http://127.0.0.1:7870 [--token-file F]

No MCP library is needed: the protocol (JSON-RPC 2.0: initialize, tools/list,
tools/call, ping) is implemented here, and tests check it against the
official `mcp` client.

    python scripts/ns_mcp.py --list        # the tools, and what each does
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

SERVER_NAME = "neuronscope"
SERVER_VERSION = "1.0"
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "OmniCortex Studio: local models (llama.cpp), graded evaluation (TestQA), hallucination checks from "
    "neuron activations, retraining and benchmark jobs, and projects split into tasks for worker models. "
    "Start with studio_status and list_models. Model ids from list_models go in every 'model' argument. "
    "Jobs and projects run on the user's machine and can take hours; start them only when asked, then "
    "poll job_status or get_project. Results of check_reply are a risk signal (classifier AUROC ~0.7), "
    "not a verdict. Roles: you, the MCP client, are the CONTROLLER. This server is OmniCortex Studio, and every "
    "tool acts on the computer Studio runs on (studio_status names it), not on the computer you run on. Studio "
    "may in turn send project tasks to worker models on other computers (its Workers page).")


class ToolError(Exception):
    pass


class Studio:
    """HTTP client for Studio's API, carrying the caller's token."""

    def __init__(self, base: str, token: str | None = None, insecure_loopback: bool = False):
        self.base = base.rstrip("/")
        self.token = token
        self.ctx = None
        if base.startswith("https://") and insecure_loopback:
            # Studio calling itself over TLS with a self-signed certificate.
            self.ctx = ssl.create_default_context()
            self.ctx.check_hostname = False
            self.ctx.verify_mode = ssl.CERT_NONE

    def call(self, method: str, path: str, body=None, timeout: float = 600):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error")
                msg = msg.get("message") if isinstance(msg, dict) else msg
            except Exception:
                msg = raw[:300].decode(errors="replace")
            raise ToolError(f"Studio said {e.code}: {msg}")
        except urllib.error.URLError as e:
            raise ToolError(f"cannot reach Studio at {self.base}: {e.reason}. Is it running?")
        except (OSError, http.client.HTTPException) as e:
            raise ToolError(f"Studio at {self.base} closed the connection ({type(e).__name__}); see its log")
        try:
            return json.loads(raw)
        except ValueError:
            return {"text": raw.decode(errors="replace")}


# ------------------------------------------------------------------ tools

def _s(props: dict, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
READ = {"readOnlyHint": True, "openWorldHint": False}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}
RISKY = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False}


def _chat(st, a):
    msgs = a.get("messages") or [{"role": "user", "content": a.get("prompt", "")}]
    if a.get("system"):
        msgs = [{"role": "system", "content": a["system"]}] + msgs
    r = st.call("POST", "/v1/chat/completions", {
        "model": a.get("model") or "", "messages": msgs, "stream": False,
        "max_tokens": int(a.get("max_tokens") or 1024), "temperature": float(a.get("temperature", 0.7))})
    msg = (r.get("choices") or [{}])[0].get("message", {})
    return {"model": r.get("model") or a.get("model"), "reply": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or None, "usage": r.get("usage")}


def _models(st, a):
    r = st.call("GET", "/v1/models")
    out = []
    for m in r.get("data", []):
        s = m.get("stats") or {}
        out.append({"id": m["id"], "loaded": m.get("loaded"), "vision": m.get("vision"), "quant": m.get("quant"),
                    "graded": s.get("graded"), "accuracy": s.get("accuracy"),
                    "hallucination_rate": s.get("hallucination_rate"), "auto_eligible": s.get("auto_eligible")})
    return {"models": out}


def _load(st, a):
    ms = st.call("GET", "/api/models")
    m = next((x for x in ms if a["model"] in (x["id"], x["name"], x["path"])), None)
    if m is None:
        raise ToolError(f"no model {a['model']!r}; list_models shows the ids")
    return st.call("POST", "/api/load", {"path": m["path"], "settings": m.get("settings") or {}}, timeout=900)


def _check(st, a):
    msgs = a.get("messages") or [{"role": "user", "content": a.get("prompt", "")}]
    r = st.call("POST", "/api/trace", {"model": a.get("model"), "messages": msgs, "text": a["reply"]}, timeout=1800)
    flagged = [r["pieces"][i] for i in r.get("flagged", [])]
    return {"flagged_tokens": len(flagged), "of_tokens": len(r.get("prob", [])), "peak_risk": r.get("max"),
            "mean_risk": r.get("mean"), "threshold": r.get("threshold"), "flagged_text": flagged[:80],
            "view_3d": st.base + "/" + r["url"], "trace_id": r.get("id"), "review": r.get("review")}


def _run_graph(st, a):
    import time
    r = st.call("POST", "/api/graphs/run", {k: a[k] for k in ("name", "inputs", "model", "profile") if a.get(k) is not None})
    end = time.time() + max(1, min(1800, int(a.get("wait_seconds") or 300)))
    while True:
        run = st.call("GET", f"/api/graphs/runs/{r['id']}")
        if run.get("status") not in ("running", "queued") or time.time() > end:
            break
        time.sleep(1)
    return {"run": r["id"], "status": run.get("status"), "outputs": run.get("outputs"), "error": run.get("error"),
            "waiting": run.get("waiting"), "nodes": [(t["node"], t["status"]) for t in run.get("trace") or []]}


def _requirements(st, a):
    r = st.call("GET", "/api/requirements")
    feats = r["features"]
    if a.get("feature"):
        if a["feature"] not in feats:
            raise ToolError(f"no feature {a['feature']!r}; known: {', '.join(feats)}")
        feats = {a["feature"]: feats[a["feature"]]}
    # Keep it short for the model: what is wrong, and how to fix it.
    return {"os": r["os"], "package_manager": r["package_manager"], "hardware": r["hardware"],
            "features": {k: {"title": f["title"], "status": f["status"], "note": f.get("note"),
                             "problems": [{x: i.get(x) for x in ("name", "have", "need", "fix", "optional", "installable")}
                                          for i in f["items"] if i["status"] != "ok"]}
                         for k, f in feats.items()}}


def _job_status(st, a):
    jobs = st.call("GET", "/api/jobs")
    if a.get("id"):
        j = next((x for x in jobs if x.get("id") == a["id"]), None)
        if j is None:
            raise ToolError(f"no job {a['id']}")
        return j
    return {"jobs": jobs[: int(a.get("limit") or 20)]}


def _job_log(st, a):
    log = st.call("GET", f"/api/jobs/{a['id']}/log").get("log", "")
    n = int(a.get("tail_chars") or 4000)
    return {"id": a["id"], "log": log[-n:]}


IV_OPS = {"trace": "iv_trace", "patch": "iv_patch", "act_patch": "iv_act_patch", "path_patch": "iv_path_patch",
          "circuit": "iv_circuit"}


def _intervene(st, a):
    """Run one interventions.py operation as a job and return its JSON document (or the job id to poll)."""
    import time as _t
    kind = IV_OPS.get(a.get("operation"))
    if kind is None:
        raise ToolError(f"operation: one of {', '.join(IV_OPS)}")
    j = st.call("POST", "/api/jobs", {"kind": kind, "values": a.get("values") or {}})
    jid, deadline = j.get("id"), _t.time() + min(int(a.get("wait_seconds") or 300), 1800)
    while jid and _t.time() < deadline:
        cur = _job_status(st, {"id": jid})
        if cur.get("status") not in ("running", "queued"):
            log = st.call("GET", f"/api/jobs/{jid}/log").get("log", "")
            i = log.find('{\n "schema_version"')
            if cur.get("status") == "done" and i >= 0:
                try:
                    return json.loads(log[i:log.rindex("}") + 1])
                except ValueError:
                    pass
            return {"id": jid, "status": cur.get("status"), "log": log[-3000:]}
        _t.sleep(2)
    return {"id": jid, "status": "running", "note": "still running: poll job_status / job_log with this id"}


def _project(st, a):
    p = st.call("GET", f"/api/projects/{a['id']}")
    for t in p.get("tasks", []):
        for at in t.get("attempts", []):           # keep replies short; the full text is in the UI
            if len(at.get("text") or "") > 3000:
                at["text"] = at["text"][:3000] + f"… [{len(at['text'])} chars; full text on /projects]"
    p.pop("history", None)
    return p


def _pact(action):
    def f(st, a):
        body = {k: v for k, v in a.items() if k != "id"}
        return st.call("POST", f"/api/projects/{a['id']}/{action}", body)
    return f


TOOLS = [
    ("studio_status", "What Studio is serving right now: the loaded model, its uptime, whether auto routing "
     "has stats to work with.", _s({}), READ, lambda st, a: st.call("GET", "/api/status")),
    ("list_models", "Local GGUF models Studio can serve, with their measured accuracy and hallucination rate "
     "when TestQA has graded them. The 'id' is what every other tool's 'model' takes.", _s({}), READ, _models),
    ("load_model", "Load a model into Studio's chat server (unloads the current one). Chat requests also load "
     "models on demand, so this is only needed to preload.", _s({"model": STR}, ["model"]), WRITE, _load),
    ("unload_model", "Unload Studio's chat model to free memory.", _s({}), WRITE,
     lambda st, a: st.call("POST", "/api/unload", {})),
    ("chat", "Ask one of the user's local models. 'model' may be 'auto' to let Studio pick the model with the "
     "best measured record for the prompt's subject.",
     _s({"model": STR, "prompt": STR, "system": STR,
         "messages": {"type": "array", "items": {"type": "object"}}, "max_tokens": INT,
         "temperature": {"type": "number"}}, ["model"]), WRITE, _chat),
    ("check_reply", "Score each token of a reply for hallucination risk from the model's own neuron activations "
     "(needs a classifier for that model). Returns the flagged text and a link to a 3D view.",
     _s({"model": STR, "prompt": STR, "reply": STR, "messages": {"type": "array", "items": {"type": "object"}}},
        ["model", "reply"]), WRITE, _check),
    ("route_prompt", "Which subjects a prompt or task involves, and which model 'auto' would pick for it and "
     "why.", _s({"text": STR}, ["text"]), READ, lambda st, a: st.call("POST", "/api/route", {"text": a["text"]})),
    ("model_stats", "Per-model, per-subject results from graded runs: accuracy, hallucination and abstention "
     "rates with confidence intervals.", _s({}), READ, lambda st, a: st.call("GET", "/api/stats")),
    ("devices", "Accelerators on this machine (ROCm, Vulkan, CUDA, Metal, CPU) with free memory, and the "
     "worker models running on them.", _s({}), READ, lambda st, a: st.call("GET", "/api/workers")),
    ("doctor", "Check the installation: packages, GPUs, llama.cpp build, model, pipeline progress, with a fix "
     "for each problem.", _s({}), READ, lambda st, a: st.call("GET", "/api/doctor", timeout=180)),
    ("job_kinds", "Jobs Studio can run (TestQA, retraining, benchmarks, pipeline stages, builds) and the "
     "typed fields each takes.", _s({}), READ, lambda st, a: st.call("GET", "/api/jobs/specs")),
    ("start_job", "Start a job of one of the kinds from job_kinds, with its fields as 'values'. Runs on the "
     "user's machine; long jobs can take hours.",
     _s({"kind": STR, "values": {"type": "object"}}, ["kind"]), RISKY,
     lambda st, a: st.call("POST", "/api/jobs", {"kind": a["kind"], "values": a.get("values") or {}})),
    ("intervene", "Which attention heads and neurons drove a model's output, and what changes without them, on a "
     "Hugging Face model: operation 'trace' (attribution), 'patch' (zero/mean ablation of a component, head, "
     "neuron or SAE feature), 'act_patch' (clean vs corrupted activation patching), 'path_patch' (a head's effect "
     "through chosen receivers only) or 'circuit' (candidates, each ablated). 'values' are that job's fields "
     "(job_kinds lists them: iv_trace, iv_patch…). Waits up to wait_seconds and returns the result document.",
     _s({"operation": {"type": "string", "enum": list(IV_OPS)}, "values": {"type": "object"}, "wait_seconds": INT},
        ["operation", "values"]), RISKY, _intervene),
    ("job_status", "One job's state, or the most recent jobs.", _s({"id": STR, "limit": INT}), READ, _job_status),
    ("job_log", "The end of a job's output.", _s({"id": STR, "tail_chars": INT}, ["id"]), READ, _job_log),
    ("cancel_job", "Stop a running job.", _s({"id": STR}, ["id"]), RISKY,
     lambda st, a: st.call("POST", "/api/jobs/cancel", {"id": a["id"]})),
    ("list_projects", "Projects the director is planning or running.", _s({}), READ,
     lambda st, a: st.call("GET", "/api/projects")),
    ("create_project", "Turn a description (notes, a task list, a README) into a draft plan of skill-labelled "
     "tasks. Nothing runs until a person approves it.",
     _s({"title": STR, "goal": STR, "text": STR, "repo": STR}, ["text"]), WRITE,
     lambda st, a: st.call("POST", "/api/projects", a)),
    ("get_project", "A project's plan: tasks, their status, assigned models, results awaiting review, pending "
     "proposals.", _s({"id": STR}, ["id"]), READ, _project),
    ("edit_project", "Change a plan: changes is a list of {op: add|update|drop|reopen|assign, ...} (see "
     "docs/DIRECTOR.md). Logged as the person's edit.",
     _s({"id": STR, "changes": {"type": "array", "items": {"type": "object"}}, "reason": STR}, ["id", "changes"]),
     WRITE, _pact("edit")),
    ("approve_project", "Approve a draft plan: assigns a model to every task and freezes it.",
     _s({"id": STR}, ["id"]), WRITE, _pact("approve")),
    ("start_project", "Start (or resume) an approved project: worker models begin on ready tasks. Refused while "
     "the machine lacks something the project requires (see project_requirements) unless force is true; only "
     "force when the user agreed.",
     _s({"id": STR, "force": BOOL}, ["id"]), RISKY, _pact("start")),
    ("project_requirements", "Check a project's own requirements on this machine: the toolchain, libraries and "
     "Python packages its build files name, hardware, OmniCortex features, notes; each with what is installed "
     "and the fix for this OS.", _s({"id": STR}, ["id"]), READ,
     lambda st, a: st.call("GET", f"/api/projects/{a['id']}/requirements", timeout=120)),
    ("set_project_requirements", "Replace a project's requirement list. items: [{kind: tool|library|package|"
     "hardware|feature|note, name, need (version spec, or GiB for hardware ram_gib/vram_gib/disk_gib, or a "
     "vendor for hardware gpu), why, optional}]. Read project_requirements first and send the whole list back.",
     _s({"id": STR, "items": {"type": "array", "items": {"type": "object"}}, "reason": STR}, ["id", "items"]),
     WRITE, _pact("requirements")),
    ("scan_project_requirements", "Infer a project's requirements again from its checkout's build files "
     "(CMake, Meson, xmake, Cargo, go.mod, package.json, pyproject, requirements.txt, …). Items a person added "
     "are kept.", _s({"id": STR, "root": STR}, ["id"]), WRITE, _pact("requirements/scan")),
    ("pause_project", "Pause a running project.", _s({"id": STR}, ["id"]), WRITE, _pact("pause")),
    ("review_task", "Accept a task's result, or send it back with feedback for the next attempt.",
     _s({"id": STR, "task": STR, "accept": BOOL, "feedback": STR}, ["id", "task", "accept"]), WRITE,
     _pact("review")),
    ("decide_proposal", "Accept or reject a change the director proposed.",
     _s({"id": STR, "proposal": STR, "accept": BOOL, "note": STR}, ["id", "proposal", "accept"]), WRITE,
     _pact("decide")),
    ("answer_task", "Answer a blocked task's questions; it goes back in the queue.",
     _s({"id": STR, "task": STR, "answers": {"type": "array", "items": STR}}, ["id", "task", "answers"]), WRITE,
     _pact("answer")),
    ("list_traces", "Saved per-reply checks and traces, viewable in 3D at <studio>/viz/<id>/.", _s({}), READ,
     lambda st, a: st.call("GET", "/api/traces")),
    ("check_permission", "Would the permission policy let a model do this? action: read, write, run (a command "
     "line), exec (a program), net (host or URL), system (gpu, docker, packages, power…) or tool (server/tool); "
     "target: the path, command, program, URL, component or tool. Optional model, job (a job kind), profile, and "
     "workdir/repo for patterns that use them. Returns allow, ask or deny and the rule that decided.",
     _s({"action": {"type": "string", "enum": ["read", "write", "run", "exec", "net", "system", "tool"]},
         "target": STR, "model": STR, "job": STR, "profile": STR, "workdir": STR, "repo": STR}, ["action", "target"]),
     READ, lambda st, a: st.call("POST", "/api/settings/permissions/check", a)),
    ("list_graphs", "Node graphs that screen, verify and route model inputs and outputs (Studio's /graphs), "
     "the ready-made templates, and recent runs.", _s({}), READ,
     lambda st, a: {k: v for k, v in st.call("GET", "/api/graphs").items() if k != "types"}),
    ("run_graph", "Run a saved graph (or template:<name>) with inputs by name, e.g. {\"question\": \"...\"}, and "
     "wait up to wait_seconds for it. Returns its outputs and each node's status; a graph that stops for a "
     "person's approval returns status waiting and its run id (answer with decide_graph).",
     _s({"name": STR, "inputs": {"type": "object"}, "model": STR, "profile": STR, "wait_seconds": INT}, ["name"]),
     WRITE, _run_graph),
    ("decide_graph", "Approve or reject a graph run that waits for a person, optionally with an edited value.",
     _s({"run": STR, "node": STR, "approve": {"type": "boolean"}, "value": STR, "note": STR}, ["run", "node", "approve"]),
     WRITE, lambda st, a: st.call("POST", f"/api/graphs/runs/{a['run']}/decide", a)),
    ("list_features", "Feature dictionaries (sparse autoencoders and transcoders), emotion vectors, recorded "
     "activations and attribution graphs in the workspace, with their checks (how much of the model's behaviour a "
     "dictionary keeps, how accurately the emotion vectors classify). Make them with start_job kinds feat_* and "
     "emo_*; read replies, trace and steer on Studio's /features page.", _s({}), READ,
     lambda st, a: st.call("GET", "/api/features/list")),
    ("requirements", "What each OmniCortex feature needs (Python packages, programs, llama.cpp builds and their GPU "
     "backends, drivers, device access, memory) and what this machine is missing, with the fix for its OS. "
     "Install a missing Python package with start_job kind 'install_package'.",
     _s({"feature": STR}), READ, _requirements),
    ("review_sources", "Neuron review: the models with recorded replies, and for each the sources (tests, "
     "benchmarks, observed chat checks), subjects and verdicts that review_summary can filter by.", _s({}), READ,
     lambda st, a: {k: v for k, v in st.call("GET", "/api/review").items() if k != "store"}),
    ("review_summary", "Neuron review: which neurons fire more on wrong answers than right ones (stat "
     "'association', Cohen's d), track the classifier's risk ('risk'), or fire most ('firing'), for a filter of "
     "sources, kinds (test, benchmark, observed), subjects and verdicts; plus the error rate per day, week or "
     "month. Open it in 3D from Studio's /review page.",
     _s({"model": STR, "stat": {"type": "string", "enum": ["association", "risk", "firing"]},
         "by": {"type": "string", "enum": ["day", "week", "month", "all"]},
         "sources": {"type": "array", "items": STR}, "kinds": {"type": "array", "items": STR},
         "subjects": {"type": "array", "items": STR}, "verdicts": {"type": "array", "items": STR},
         "top": INT}, ["model"]), READ,
     lambda st, a: {k: v for k, v in st.call("POST", "/api/review/summary", a).items() if k != "grid"}),
    ("review_label", "Neuron review: record whether a checked reply was right or wrong (verdict 'correct', "
     "'wrong' or 'abstained'), so it counts in the association. The id is the 'review' id a check returns.",
     _s({"model": STR, "id": STR, "verdict": {"type": "string", "enum": ["correct", "wrong", "abstained"]}},
        ["model", "id", "verdict"]), WRITE, lambda st, a: st.call("POST", "/api/review/label", a)),
    ("remote_hosts", "Computers this Studio controls that run models but not OmniCortex (LM Studio, Ollama or "
     "llama.cpp with the OmniCortex Remote agent): how each is reached, whether it answers, its models. Studio is "
     "the controller, they are hosts. Their models work everywhere as '<host>:<model>'.", _s({}), READ,
     lambda st, a: st.call("GET", "/api/remote?refresh=1")),
    ("remote_call", "Run one of a remote host's own tools: host_info, list_backends, list_models, chat, load_model, "
     "unload_model, list_files, receive_file, transfers, install_model, delete_file. The host's owner decides which "
     "of them controllers may use.", _s({"host": STR, "tool": STR, "args": {"type": "object"}}, ["host", "tool"]),
     RISKY, lambda st, a: st.call("POST", "/api/remote/call", a, timeout=900)),
]
BY_NAME = {t[0]: t for t in TOOLS}


def tool_list() -> list[dict]:
    return [{"name": n, "description": d, "inputSchema": s, "annotations": {"title": n.replace("_", " "), **ann}}
            for n, d, s, ann, _ in TOOLS]


# ------------------------------------------------------------------ protocol

def _err(mid, code, msg):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": msg}}


def handle(msg, studio: Studio):
    """One JSON-RPC message -> response dict, or None for a notification."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    mid, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
    if "id" not in msg:                       # notifications (initialized, cancelled, …) need no answer
        return None
    try:
        if method == "initialize":
            want = params.get("protocolVersion")
            res = {"protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                   "instructions": INSTRUCTIONS + f" This Studio: {studio.base}."}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": tool_list()}
        elif method == "tools/call":
            name = params.get("name")
            if name not in BY_NAME:
                return _err(mid, -32602, f"unknown tool {name!r}")
            args = params.get("arguments") or {}
            if not isinstance(args, dict):
                return _err(mid, -32602, "arguments must be an object")
            schema = BY_NAME[name][2]
            missing = [k for k in schema["required"] if k not in args]
            if missing:
                return _err(mid, -32602, f"{name}: missing {', '.join(missing)}")
            try:
                out = BY_NAME[name][4](studio, args)
                res = {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}],
                       "structuredContent": out if isinstance(out, dict) else {"result": out}, "isError": False}
            except ToolError as e:
                res = {"content": [{"type": "text", "text": str(e)}], "isError": True}
        elif method in ("resources/list", "prompts/list"):
            res = {method.split("/")[0]: []}
        else:
            return _err(mid, -32601, f"method not found: {method}")
    except Exception as e:                     # a bug must not kill the session
        return _err(mid, -32603, f"{type(e).__name__}: {e}")
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def handle_payload(payload, studio: Studio):
    """A message or a batch -> response(s), None when nothing needs an answer."""
    if isinstance(payload, list):
        out = [r for r in (handle(m, studio) for m in payload) if r is not None]
        return out or None
    return handle(payload, studio)


def serve_stdio(studio: Studio) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            resp = _err(None, -32700, "parse error")
        else:
            resp = handle_payload(payload, studio)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OmniCortex MCP server (stdio). An MCP client starts this; "
                                            "run alone it waits for requests on stdin.")
    p.add_argument("--studio", default=os.environ.get("NS_STUDIO_URL", "http://127.0.0.1:7870"),
                   help="Studio URL (default $NS_STUDIO_URL or http://127.0.0.1:7870)")
    p.add_argument("--token-file", help="file holding Studio's token (or a paired device's)")
    p.add_argument("--list", action="store_true", help="print the tools and exit")
    a = p.parse_args(argv)
    if a.list:
        for n, d, s, ann, _ in TOOLS:
            kind = "read" if ann.get("readOnlyHint") else ("changes things" if not ann.get("destructiveHint")
                                                           else "starts work / stops work")
            print(f"{n:<16} [{kind}] {d}")
        return 0
    token = os.environ.get("NS_STUDIO_TOKEN") or None
    if a.token_file:
        token = open(os.path.expanduser(a.token_file)).read().strip()
    serve_stdio(Studio(a.studio, token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
