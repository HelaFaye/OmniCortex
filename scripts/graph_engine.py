#!/usr/bin/env python3
"""
Node graphs for the harness: route inputs and outputs between models,
screens, verifiers, tools and people (docs/GRAPHS.md).

A graph is JSON:

    {"name": "screen-and-verify", "description": "...", "profile": "read_only",
     "nodes": [{"id": "in", "type": "input", "params": {"name": "question"}, "x": 40, "y": 80}, ...],
     "edges": [{"from": ["in", "value"], "to": ["ask", "prompt"]}, ...]}

How it runs (dataflow):
  - A value travels along every edge leaving an output port. A node runs once all
    its required inputs that have an edge hold a value; it runs again when a new
    value arrives (that is how loops work).
  - A node may leave an output empty: that branch is **skipped**, and a node whose
    required input is skipped is skipped too, without running. So pass / fail
    ports route a value one way or the other.
  - Every node has an implicit `error` output. When a node fails and its error
    port has an edge, the error text goes there and the run carries on;
    otherwise the run stops with the error.
  - `approval` nodes pause the run until a person approves (optionally editing
    the value) or rejects it.
  - `loop` nodes bound a cycle: start -> value, again -> value (up to max), then done.
  - Every execution is traced: inputs, outputs, time, error. A run is capped
    at MAX_STEPS executions.

Nodes that act on the machine (files, network, programs, MCP tools) check the
permission policy (scripts/permissions.py) for job "graph", the graph's
profile and the node's own profile; "ask" counts as no inside a graph unless an
approval node stands in front.

Node types are registered with @node; plugins add more (plugins.py, "nodes").

    python scripts/graph_engine.py types
    python scripts/graph_engine.py run graph.json --input question="Is water wet?" --studio http://127.0.0.1:7870
    python scripts/graph_engine.py eval graph.json --cases cases.jsonl --expect answer
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import uuid
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_STEPS = 500
MAX_TEXT = 200_000
SKIP = object()            # an output a node chose not to produce
TYPES: dict[str, dict] = {}
GROUPS = ["Input and output", "Models", "Screening", "Verification", "Routing", "Data", "Tools", "Delegation", "Plugins"]


class GraphError(ValueError):
    pass


class Pause(Exception):
    """An approval node waiting for a person."""


def node(name, title, group, inputs=(), outputs=(), params=(), help="", acts=()):
    """Register a node type. inputs: (name, type, required); outputs: (name, type);
    params: dicts {name, kind (str|text|int|float|bool|choice|json|model|endpoint|profile|graph), default,
    choices, help}. acts: permission actions it may need, for the editor to show."""
    def deco(fn):
        TYPES[name] = {"name": name, "title": title, "group": group, "help": help,
                       "inputs": [{"name": n, "type": t, "required": r} for n, t, r in inputs],
                       "outputs": [{"name": n, "type": t} for n, t in outputs] + [{"name": "error", "type": "text"}],
                       "params": [{"default": None, "help": "", **p} for p in params], "acts": list(acts), "run": fn}
        return fn
    return deco


HIDDEN = None             # (node type) -> True when its add-on is removed (viz/addons.py); set by Studio


def types_public() -> list[dict]:
    return [{k: v for k, v in t.items() if k != "run"} for t in TYPES.values() if not (HIDDEN and HIDDEN(t["name"]))]


# ------------------------------------------------------------------ the environment nodes run in

class Env:
    """What nodes may use. In Studio, chat/trace/rag/mcp calls go in-process or over loopback; from the CLI
    they go to --studio over HTTP."""

    def __init__(self, studio=None, policy=None, model: str = "", profile: str = "", workdir: str | None = None,
                 mcp_call=None, graph_loader=None, log=None):
        self.studio, self.model, self.profile, self.workdir = studio, model, profile, workdir
        self.mcp_call, self.graph_loader = mcp_call, graph_loader
        self.log = log or (lambda *_: None)
        if policy is None:
            import permissions as P
            policy = P.Policy()
        self.policy = policy
        self.memory: dict[str, list] = {}

    def need_studio(self):
        if self.studio is None:
            raise GraphError("this node needs Studio (run the graph from Studio, or pass --studio)")
        return self.studio

    def chat(self, messages, model="", endpoint="", temperature=0.7, max_tokens=1024, json_mode=False) -> dict:
        body = {"model": model or self.model or "", "messages": messages, "stream": False,
                "temperature": float(temperature), "max_tokens": int(max_tokens)}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if endpoint:
            import oc_mcp
            url, _, m = endpoint.partition("@")
            body["model"] = m or body["model"]
            st = oc_mcp.Studio(url.rstrip("/").removesuffix("/v1"))
            r = st.call("POST", "/v1/chat/completions", body, timeout=1800)
        else:
            r = self.need_studio().call("POST", "/v1/chat/completions", body, timeout=1800)
        msg = (r.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", text, re.S)          # reasoning inline, split it out
        if m and not reasoning:
            reasoning, text = m.group(1).strip(), m.group(2)
        return {"text": text, "reasoning": reasoning, "model": r.get("model") or body["model"]}

    def permit(self, action: str, target: str, node_profile: str = "") -> None:
        import permissions as P
        ctx = P.context(workdir=self.workdir, repo=self.workdir)
        effects = [self.policy.check(action, target, self.model, "graph", self.profile, ctx)]
        if node_profile:
            effects.append(self.policy.check(action, target, self.model, "graph", node_profile, ctx))
        worst = max(effects, key=lambda d: P.RANK[d["effect"]])
        if worst["effect"] != "allow":
            raise GraphError(f"not permitted: {action} {str(target)[:120]} ({worst['effect']}: {worst['why']}); "
                             "Settings -> Permissions, or put an approval node in front")


# ------------------------------------------------------------------ helpers

def _text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v[:MAX_TEXT]
    return json.dumps(v, ensure_ascii=False)[:MAX_TEXT]


def _json(v):
    if isinstance(v, (dict, list)):
        return v
    s = _text(v).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        i, j = min([k for k in (s.find("{"), s.find("[")) if k >= 0] or [-1]), max(s.rfind("}"), s.rfind("]"))
        if i >= 0 and j > i:
            return json.loads(s[i:j + 1])
        raise GraphError("not JSON: " + s[:120])


def _path(obj, path: str):
    for part in [p for p in re.split(r"\.|\[(\d+)\]", path or "") if p]:
        if isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise GraphError(f"no {part!r} in {type(obj).__name__}")
    return obj


def _schema_errors(value, schema: dict, where="$") -> list[str]:
    """A useful subset of JSON Schema: type, properties, required, items, enum, minLength, maxLength,
    minimum, maximum, pattern."""
    errs = []
    t = schema.get("type")
    kinds = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool,
             "null": type(None)}
    if t and t in kinds and (not isinstance(value, kinds[t]) or (t in ("integer", "number") and isinstance(value, bool))):
        return [f"{where}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{where}: not one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{where}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{where}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{where}: does not match {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{where}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{where}: above {schema['maximum']}")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{where}.{k}: missing")
        for k, sub in (schema.get("properties") or {}).items():
            if k in value:
                errs += _schema_errors(value[k], sub, f"{where}.{k}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value[:200]):
            errs += _schema_errors(v, schema["items"], f"{where}[{i}]")
    return errs


def _fmt(template: str, values: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: _text(values.get(m.group(1), m.group(0))), template)


def _score(text: str) -> float | None:
    m = re.search(r"(?i)score\W{0,3}(\d+(?:\.\d+)?)", text) or re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b", text)
    return float(m.group(1)) if m else None


def _verdict(ok: bool, value, why: str = "") -> dict:
    return {"pass": value if ok else SKIP, "fail": SKIP if ok else (why or value), "ok": ok}


# ------------------------------------------------------------------ input and output

@node("input", "Input", "Input and output", outputs=[("value", "any")],
      params=[{"name": "name", "kind": "str", "default": "input", "help": "the run's input of this name"},
              {"name": "default", "kind": "text", "default": "", "help": "used when the run gives none"}],
      help="A value given when the graph runs (a question, a reply, a task).")
def _n_input(n, i, env, run):
    p = n["params"]
    v = run["inputs"].get(p.get("name") or "input", p.get("default") or "")
    return {"value": v}


@node("output", "Output", "Input and output", inputs=[("value", "any", True)],
      params=[{"name": "name", "kind": "str", "default": "result"}],
      help="A result of the run. The last value to arrive wins.")
def _n_output(n, i, env, run):
    run["outputs"][n["params"].get("name") or "result"] = i["value"]
    return {}


@node("template", "Template", "Input and output", inputs=[("a", "any", True), ("b", "any", False), ("c", "any", False),
                                                          ("d", "any", False)],
      outputs=[("text", "text")], params=[{"name": "template", "kind": "text", "default": "{a}",
                                           "help": "text with {a} {b} {c} {d} and the run's inputs by name"}],
      help="Build text from the inputs. It runs when a arrives (b, c and d are used as they stand).")
def _n_template(n, i, env, run):
    return {"text": _fmt(n["params"].get("template") or "", {**run["inputs"], **{k: v for k, v in i.items() if v is not None}})}


# ------------------------------------------------------------------ models

@node("llm", "Model", "Models", inputs=[("prompt", "text", True), ("system", "text", False), ("context", "any", False)],
      outputs=[("text", "text"), ("reasoning", "text"), ("json", "json")],
      params=[{"name": "model", "kind": "model", "default": "", "help": "empty: the graph's model; auto: Studio picks"},
              {"name": "endpoint", "kind": "endpoint", "default": "", "help": "another OpenAI-compatible URL@model; empty: Studio"},
              {"name": "system", "kind": "text", "default": ""},
              {"name": "temperature", "kind": "float", "default": 0.7},
              {"name": "max_tokens", "kind": "int", "default": 1024},
              {"name": "json_schema", "kind": "json", "default": None,
               "help": "ask for JSON matching this schema; it is checked and asked once more if wrong"},
              {"name": "memory", "kind": "bool", "default": False, "help": "keep the conversation across a loop's rounds"}],
      help="Ask a model. Reasoning is split from the answer.")
def _n_llm(n, i, env, run):
    p = n["params"]
    sys_p = "\n\n".join(x for x in (p.get("system"), _text(i.get("system"))) if x)
    schema = p.get("json_schema")
    if schema:
        sys_p += ("\n\n" if sys_p else "") + "Answer with JSON only, matching this schema:\n" + json.dumps(schema)
    user = _text(i["prompt"]) + (("\n\nContext:\n" + _text(i["context"])) if i.get("context") not in (None, "") else "")
    hist = env.memory.setdefault(n["id"], []) if p.get("memory") else []
    msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + hist + [{"role": "user", "content": user}]
    out = env.chat(msgs, p.get("model"), p.get("endpoint"), p.get("temperature", 0.7), p.get("max_tokens", 1024))
    res = {"text": out["text"], "reasoning": out["reasoning"] or SKIP}
    if schema:
        for attempt in range(2):
            try:
                val = _json(out["text"])
                errs = _schema_errors(val, schema)
            except (GraphError, ValueError) as e:
                val, errs = None, [str(e)]
            if not errs:
                res["json"] = val
                break
            if attempt == 0:
                msgs = msgs + [{"role": "assistant", "content": out["text"]},
                               {"role": "user", "content": "That does not match the schema: " + "; ".join(errs[:5])
                                + ". Answer again with JSON only."}]
                out = env.chat(msgs, p.get("model"), p.get("endpoint"), 0.2, p.get("max_tokens", 1024))
                res["text"] = out["text"]
            else:
                raise GraphError("the model's JSON does not match the schema: " + "; ".join(errs[:5]))
    else:
        try:
            res["json"] = _json(out["text"])
        except (GraphError, ValueError):
            res["json"] = SKIP
    if p.get("memory"):
        hist += [{"role": "user", "content": user}, {"role": "assistant", "content": out["text"]}]
    return res


@node("route", "Pick a model", "Models", inputs=[("text", "text", True)], outputs=[("model", "text"), ("why", "text")],
      help="Studio's auto routing: the model with the best measured record for the text's subject.")
def _n_route(n, i, env, run):
    r = env.need_studio().call("POST", "/api/route", {"text": _text(i["text"])})
    return {"model": r.get("model") or r.get("pick") or "", "why": _text(r.get("reason") or r)}


# ------------------------------------------------------------------ screening

_SECRET_RX = [
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("API key", r"\b(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|"
                r"github_pat_[A-Za-z0-9_]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,})\b"),
    ("password assignment", r"(?i)\b(password|passwd|secret|token)\s*[:=]\s*\S{6,}"),
    ("email address", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("card number", r"\b(?:\d[ -]?){13,19}\b"),
]


@node("regex_screen", "Regex screen", "Screening", inputs=[("text", "text", True)],
      outputs=[("pass", "text"), ("fail", "text"), ("matches", "list")],
      params=[{"name": "pattern", "kind": "str", "default": "", "help": "a regular expression"},
              {"name": "mode", "kind": "choice", "choices": ["block", "require"], "default": "block",
               "help": "block: fail when it matches; require: fail when it does not"},
              {"name": "ignore_case", "kind": "bool", "default": True}],
      help="Pass or fail text on a regular expression (whitelist or blacklist).")
def _n_regex(n, i, env, run):
    p = n["params"]
    t = _text(i["text"])
    hits = re.findall(p.get("pattern") or "(?!)", t, re.I if p.get("ignore_case", True) else 0)
    hits = [h if isinstance(h, str) else h[0] for h in hits][:50]
    ok = (not hits) if p.get("mode", "block") == "block" else bool(hits)
    return {**_verdict(ok, t, f"{'matched' if hits else 'did not match'} {p.get('pattern')!r}: {hits[:5]}"), "matches": hits}


@node("secrets_screen", "Secrets and personal data", "Screening", inputs=[("text", "text", True)],
      outputs=[("pass", "text"), ("fail", "text"), ("redacted", "text"), ("findings", "list")],
      params=[{"name": "kinds", "kind": "str", "default": "private key, API key, password assignment",
               "help": "which to look for: " + ", ".join(k for k, _ in _SECRET_RX)}],
      help="Finds keys, passwords, emails and card numbers; passes clean text and gives a redacted copy either way.")
def _n_secrets(n, i, env, run):
    t = _text(i["text"])
    want = {k.strip() for k in (n["params"].get("kinds") or "").split(",") if k.strip()}
    found, red = [], t
    for k, rx in _SECRET_RX:
        if want and k not in want:
            continue
        for m in re.finditer(rx, t):
            found.append({"kind": k, "at": m.start(), "text": m.group(0)[:6] + "…"})
        red = re.sub(rx, f"[{k}]", red)
    return {**_verdict(not found, t, f"found {', '.join(sorted({f['kind'] for f in found}))}"), "redacted": red,
            "findings": found}


@node("length_screen", "Length screen", "Screening", inputs=[("text", "text", True)], outputs=[("pass", "text"), ("fail", "text")],
      params=[{"name": "min", "kind": "int", "default": 1}, {"name": "max", "kind": "int", "default": 20000}],
      help="Pass text whose length is within bounds (empty and runaway replies fail).")
def _n_length(n, i, env, run):
    t = _text(i["text"]).strip()
    lo, hi = int(n["params"].get("min") or 0), int(n["params"].get("max") or 10**9)
    return _verdict(lo <= len(t) <= hi, t, f"{len(t)} characters, wanted {lo} to {hi}")


@node("json_check", "JSON schema check", "Screening", inputs=[("value", "any", True)],
      outputs=[("pass", "json"), ("fail", "text")], params=[{"name": "schema", "kind": "json", "default": {"type": "object"}}],
      help="Parse JSON (also out of a fenced block) and check it against a schema.")
def _n_json_check(n, i, env, run):
    try:
        v = _json(i["value"])
    except (GraphError, ValueError) as e:
        return _verdict(False, None, str(e))
    errs = _schema_errors(v, n["params"].get("schema") or {})
    return _verdict(not errs, v, "; ".join(errs[:8]))


@node("permission_gate", "Permission gate", "Screening", inputs=[("target", "text", True)],
      outputs=[("allow", "text"), ("ask", "text"), ("deny", "text"), ("why", "text")],
      params=[{"name": "action", "kind": "choice", "choices": ["read", "write", "run", "exec", "net", "system", "tool"],
               "default": "run"},
              {"name": "profile", "kind": "profile", "default": ""}],
      help="Route a path, command, program, URL or tool by what the permission policy says about it.")
def _n_perm(n, i, env, run):
    import permissions as P
    t = _text(i["target"])
    d = env.policy.check(n["params"].get("action") or "run", t, env.model, "graph",
                         n["params"].get("profile") or env.profile, P.context(workdir=env.workdir, repo=env.workdir))
    return {e: (t if e == d["effect"] else SKIP) for e in P.EFFECTS} | {"why": d["why"]}


def _thresholds(spec: str) -> list[tuple[str, str, float]]:
    out = []
    for part in [x.strip() for x in (spec or "").split(",") if x.strip()]:
        m = re.match(r"([\w#.-]+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$", part)
        if not m:
            raise GraphError(f"threshold {part!r}: name > number, e.g. desperate > 2")
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def _cmp(v, op, x):
    return {">": v > x, ">=": v >= x, "<": v < x, "<=": v <= x}[op]


@node("emotion_screen", "Emotion screen", "Screening", inputs=[("text", "text", True)],
      outputs=[("pass", "text"), ("fail", "text"), ("peaks", "json")],
      params=[{"name": "emotions", "kind": "str", "default": "", "help": "emotion vectors folder (emotions.py build)"},
              {"name": "fail_when", "kind": "str", "default": "desperate > 2.5, hostile > 2.5",
               "help": "comma-separated: emotion > z (the peak over the text)"},
              {"name": "lab", "kind": "str", "default": "http://127.0.0.1:8090", "help": "the decoding lab"}],
      help="Read the text through the model and fail it when an emotion vector peaks past a threshold "
           "(docs/FEATURES.md). Catches desperate or hostile turns that the words may hide.")
def _n_emotion(n, i, env, run):
    p = n["params"]
    r = env.need_studio().call("POST", "/api/features/lab/inspect",
                               {"lab": p.get("lab"), "text": _text(i["text"]), "emotions": p.get("emotions")}, timeout=900)
    emo = r.get("emotions") or {}
    names, z = emo.get("names") or [], emo.get("z") or []
    peaks = {nm: max((row[k] for row in z), default=0.0) for k, nm in enumerate(names)}
    bad = [f"{nm} {peaks.get(nm, 0):.1f} {op} {x}" for nm, op, x in _thresholds(p.get("fail_when")) if _cmp(peaks.get(nm, 0.0), op, x)]
    top = dict(sorted(peaks.items(), key=lambda kv: -abs(kv[1]))[:8])
    return {**_verdict(not bad, _text(i["text"]), "; ".join(bad)), "peaks": top}


@node("feature_screen", "Feature screen", "Screening", inputs=[("text", "text", True)],
      outputs=[("pass", "text"), ("fail", "text"), ("peaks", "json")],
      params=[{"name": "dictionary", "kind": "str", "default": "", "help": "feature dictionary folder"},
              {"name": "fail_when", "kind": "str", "default": "", "help": "e.g. #877 > 0.5 (relative to the feature's maximum)"},
              {"name": "lab", "kind": "str", "default": "http://127.0.0.1:8090"}],
      help="Fail text on which chosen features of a dictionary fire past a threshold.")
def _n_feature(n, i, env, run):
    p = n["params"]
    r = env.need_studio().call("POST", "/api/features/lab/inspect",
                               {"lab": p.get("lab"), "text": _text(i["text"]), "saes": [p.get("dictionary")]}, timeout=900)
    peaks: dict[str, float] = {}
    for d in r.get("dicts") or []:
        for row in d.get("top") or []:
            for fid, rel, *_ in row:
                k = f"#{int(fid)}"
                peaks[k] = max(peaks.get(k, 0.0), float(rel))
    bad = [f"{nm} {peaks.get(nm, 0):.2f} {op} {x}" for nm, op, x in _thresholds(p.get("fail_when")) if _cmp(peaks.get(nm, 0.0), op, x)]
    return {**_verdict(not bad, _text(i["text"]), "; ".join(bad)), "peaks": dict(sorted(peaks.items(), key=lambda kv: -kv[1])[:10])}


# ------------------------------------------------------------------ verification

@node("judge", "Judge", "Verification", inputs=[("answer", "text", True), ("question", "text", False), ("reference", "text", False)],
      outputs=[("pass", "text"), ("fail", "text"), ("score", "number"), ("feedback", "text")],
      params=[{"name": "model", "kind": "model", "default": ""}, {"name": "endpoint", "kind": "endpoint", "default": ""},
              {"name": "rubric", "kind": "text", "default": "Is the answer correct, complete and free of invented facts?"},
              {"name": "pass_score", "kind": "float", "default": 7.0}],
      help="A model grades the answer against a rubric (and a reference, if given) from 0 to 10.")
def _n_judge(n, i, env, run):
    p = n["params"]
    prompt = (f"Rubric: {p.get('rubric')}\n\n" + (f"Question:\n{_text(i.get('question'))}\n\n" if i.get("question") else "")
              + (f"Reference answer:\n{_text(i.get('reference'))}\n\n" if i.get("reference") else "")
              + f"Answer to grade:\n{_text(i['answer'])}\n\n"
              "Reply with one line 'Score: N' (0 to 10), then what is wrong or missing, briefly.")
    out = env.chat([{"role": "user", "content": prompt}], p.get("model"), p.get("endpoint"), 0.0, 400)
    s = _score(out["text"])
    if s is None:
        raise GraphError("the judge gave no score: " + out["text"][:200])
    ok = s >= float(p.get("pass_score") or 7)
    fb = re.sub(r"(?i)^\s*score\W*\d+(\.\d+)?\s*", "", out["text"]).strip()
    return {**_verdict(ok, _text(i["answer"]), fb or f"score {s}"), "score": s, "feedback": fb}


@node("hallucination_check", "Hallucination check", "Verification",
      inputs=[("reply", "text", True), ("prompt", "text", False)],
      outputs=[("pass", "text"), ("fail", "text"), ("risk", "number"), ("flagged", "list"), ("view", "text")],
      params=[{"name": "model", "kind": "model", "default": "", "help": "the model that wrote the reply (needs its classifier)"},
              {"name": "max_risk", "kind": "float", "default": 0.5, "help": "fail when any token's risk reaches this"},
              {"name": "max_flagged", "kind": "int", "default": 0, "help": "…or more than this many tokens are flagged"}],
      help="OmniCortex's own check: each token's hallucination risk from the model's neuron activations. "
           "Fails with the flagged words, and links the 3D view.")
def _n_halluc(n, i, env, run):
    p = n["params"]
    st = env.need_studio()
    r = st.call("POST", "/api/trace", {"model": p.get("model") or env.model,
                                       "messages": [{"role": "user", "content": _text(i.get("prompt"))}],
                                       "text": _text(i["reply"])}, timeout=1800)
    flagged = [r["pieces"][k] for k in r.get("flagged", [])]
    risk = float(r.get("max") or 0.0)
    ok = risk < float(p.get("max_risk", 0.5)) and len(flagged) <= int(p.get("max_flagged") or 0)
    return {**_verdict(ok, _text(i["reply"]), f"peak risk {risk:.2f}; flagged: {''.join(flagged)[:300]}"), "risk": risk,
            "flagged": flagged, "view": st.base + "/" + r.get("url", "")}


@node("self_consistency", "Self-consistency", "Verification", inputs=[("prompt", "text", True)],
      outputs=[("pass", "text"), ("fail", "text"), ("agreement", "number"), ("answers", "list")],
      params=[{"name": "model", "kind": "model", "default": ""}, {"name": "samples", "kind": "int", "default": 5},
              {"name": "temperature", "kind": "float", "default": 0.8},
              {"name": "min_agreement", "kind": "float", "default": 0.6},
              {"name": "answer_regex", "kind": "str", "default": r"(?is)(?:final answer|answer)\s*[:=]\s*(.+?)\s*$",
               "help": "how to pull the short answer out of each sample (else its last line)"}],
      help="Ask several times and pass the majority answer when enough samples agree: a cheap guard against guesses.")
def _n_consistency(n, i, env, run):
    p = n["params"]
    k = max(2, min(15, int(p.get("samples") or 5)))
    answers = []
    for _ in range(k):
        out = env.chat([{"role": "user", "content": _text(i["prompt"])}], p.get("model"), "", p.get("temperature", 0.8), 600)
        m = re.search(p.get("answer_regex") or "(?!)", out["text"])
        a = (m.group(1) if m else (out["text"].strip().splitlines() or [""])[-1]).strip().lower().rstrip(".")
        answers.append(a[:200])
    top, cnt = Counter(answers).most_common(1)[0]
    agree = cnt / k
    return {**_verdict(agree >= float(p.get("min_agreement") or 0.6), top, f"only {cnt}/{k} agree: {Counter(answers).most_common(3)}"),
            "agreement": agree, "answers": answers}


@node("cross_check", "Cross-check by another model", "Verification",
      inputs=[("answer", "text", True), ("question", "text", True)],
      outputs=[("pass", "text"), ("fail", "text"), ("explanation", "text")],
      params=[{"name": "model", "kind": "model", "default": "", "help": "a different model from the one that answered"},
              {"name": "endpoint", "kind": "endpoint", "default": ""}],
      help="A second model says whether the answer is right, with its reason.")
def _n_cross(n, i, env, run):
    p = n["params"]
    out = env.chat([{"role": "user", "content": f"Question:\n{_text(i['question'])}\n\nProposed answer:\n{_text(i['answer'])}\n\n"
                     "Is the proposed answer correct? Start your reply with YES or NO, then explain in two sentences."}],
                   p.get("model"), p.get("endpoint"), 0.0, 300)
    ok = out["text"].strip().upper().startswith("YES")
    return {**_verdict(ok, _text(i["answer"]), out["text"]), "explanation": out["text"]}


@node("grounding", "Grounded in sources", "Verification", inputs=[("answer", "text", True), ("question", "text", False)],
      outputs=[("pass", "text"), ("fail", "text"), ("sources", "list")],
      params=[{"name": "collection", "kind": "str", "default": "", "help": "Studio document collection"},
              {"name": "k", "kind": "int", "default": 4}, {"name": "model", "kind": "model", "default": ""}],
      help="Look the answer up in a document collection and have a model say whether the sources support it.")
def _n_grounding(n, i, env, run):
    p = n["params"]
    r = env.need_studio().call("POST", "/api/rag/search", {"collection": p.get("collection"),
                                                            "query": _text(i.get("question")) or _text(i["answer"]), "k": int(p.get("k") or 4)})
    hits = r.get("hits") if isinstance(r, dict) else r
    src = "\n\n".join(f"[{j + 1}] {_text(h.get('text'))[:1500]}" for j, h in enumerate(hits or []))
    if not src:
        return {**_verdict(False, None, "no sources found"), "sources": []}
    out = env.chat([{"role": "user", "content": f"Sources:\n{src}\n\nClaim:\n{_text(i['answer'])}\n\n"
                     "Do the sources support every factual statement in the claim? Start with SUPPORTED or "
                     "UNSUPPORTED, then name what is not supported."}], p.get("model"), "", 0.0, 400)
    ok = out["text"].strip().upper().startswith("SUPPORTED")
    return {**_verdict(ok, _text(i["answer"]), out["text"]), "sources": [h.get("source") for h in hits or []]}


@node("code_tests", "Run code against tests", "Verification", inputs=[("code", "text", True), ("tests", "text", True)],
      outputs=[("pass", "text"), ("fail", "text")],
      params=[{"name": "timeout", "kind": "float", "default": 10.0},
              {"name": "sandbox", "kind": "choice", "choices": ["none", "docker", "podman"], "default": "docker"}],
      acts=["exec"],
      help="Run the code with the tests (one assertion block per line group) in TestQA's sandbox; needs the "
           "permission to start the sandbox.")
def _n_code(n, i, env, run):
    import testqa
    p = n["params"]
    testqa.configure_sandbox(p.get("sandbox") if p.get("sandbox") != "none" else "none", None)
    env.permit("exec", testqa.SANDBOX["engine"] or sys.executable, n.get("profile", ""))
    code = testqa._code_blocks(_text(i["code"])) or _text(i["code"])
    tests = [t for t in re.split(r"\n\s*\n", _text(i["tests"])) if t.strip()]
    verdict, detail = testqa.run_code(code, tests, float(p.get("timeout") or 10))
    return _verdict(verdict == "correct", code, f"{verdict}: {detail}")


# ------------------------------------------------------------------ routing

@node("gate", "Gate", "Routing", inputs=[("value", "any", True), ("condition", "any", True)],
      outputs=[("pass", "any"), ("fail", "any")],
      help="Send the value to pass when the condition is true (non-empty, not 'false'/'no'/0), else to fail.")
def _n_gate(n, i, env, run):
    c = i["condition"]
    ok = bool(c) and _text(c).strip().lower() not in ("false", "no", "0", "fail", "none", "null")
    return {"pass": i["value"] if ok else SKIP, "fail": SKIP if ok else i["value"]}


@node("switch", "Switch", "Routing", inputs=[("value", "any", True)],
      outputs=[("a", "any"), ("b", "any"), ("c", "any"), ("other", "any")],
      params=[{"name": "a", "kind": "str", "default": "", "help": "regex for output a"},
              {"name": "b", "kind": "str", "default": ""}, {"name": "c", "kind": "str", "default": ""}],
      help="Send the value to the first output whose regular expression matches it.")
def _n_switch(n, i, env, run):
    t = _text(i["value"])
    out = {k: SKIP for k in ("a", "b", "c", "other")}
    for k in ("a", "b", "c"):
        rx = n["params"].get(k)
        if rx and re.search(rx, t, re.I):
            out[k] = i["value"]
            return out
    out["other"] = i["value"]
    return out


@node("merge", "First available", "Routing", inputs=[("a", "any", False), ("b", "any", False), ("c", "any", False)],
      outputs=[("value", "any")], help="Pass on whichever input has a value (a, then b, then c): joins branches.")
def _n_merge(n, i, env, run):
    for k in ("a", "b", "c"):
        if i.get(k) is not None:
            return {"value": i[k]}
    return {"value": SKIP}


@node("loop", "Loop", "Routing", inputs=[("start", "any", False), ("again", "any", False)],
      outputs=[("value", "any"), ("round", "number"), ("done", "any")],
      params=[{"name": "max", "kind": "int", "default": 3, "help": "rounds after the first"}],
      help="Bound a cycle: start sends the value on; again sends it round once more, up to max, then to done. "
           "Wire a verifier's fail back into again to retry with its feedback.")
def _n_loop(n, i, env, run):
    st = run["state"].setdefault(n["id"], {"round": 0})
    if run["_trigger"].get(n["id"]) == "again" and i.get("again") is not None:
        if st["round"] >= int(n["params"].get("max") or 3):
            return {"value": SKIP, "round": st["round"], "done": i["again"]}
        st["round"] += 1
        return {"value": i["again"], "round": st["round"], "done": SKIP}
    st["round"] = 0
    return {"value": i.get("start"), "round": 0, "done": SKIP}


@node("approval", "Ask a person", "Routing", inputs=[("value", "any", True)],
      outputs=[("approved", "any"), ("rejected", "any")],
      params=[{"name": "question", "kind": "text", "default": "Approve this?"}],
      help="Pause the run until a person approves (they may edit the value first) or rejects it.")
def _n_approval(n, i, env, run):
    dec = run["decisions"].pop(n["id"], None)
    if dec is None:
        run["waiting"] = {"node": n["id"], "question": n["params"].get("question") or "Approve this?", "value": i["value"]}
        raise Pause()
    v = dec.get("value", i["value"])
    return {"approved": v if dec.get("approve") else SKIP, "rejected": SKIP if dec.get("approve") else (dec.get("note") or v)}


@node("fallback", "Fallback", "Routing", inputs=[("primary", "any", False), ("backup", "any", False)],
      outputs=[("value", "any")], help="The primary value, or the backup when the primary branch was skipped or failed.")
def _n_fallback(n, i, env, run):
    return {"value": i["primary"] if i.get("primary") is not None else i.get("backup", SKIP)}


# ------------------------------------------------------------------ data

@node("json_extract", "JSON: extract", "Data", inputs=[("value", "any", True)], outputs=[("value", "any")],
      params=[{"name": "path", "kind": "str", "default": "", "help": "e.g. items[0].name"}],
      help="Pull one value out of JSON (also out of a fenced block in a reply).")
def _n_jx(n, i, env, run):
    return {"value": _path(_json(i["value"]), n["params"].get("path") or "")}


@node("json_build", "JSON: build", "Data", inputs=[("a", "any", False), ("b", "any", False), ("c", "any", False), ("d", "any", False)],
      outputs=[("json", "json")], params=[{"name": "keys", "kind": "str", "default": "a, b, c, d", "help": "a key for each input"}],
      help="An object from the inputs, under the keys given.")
def _n_jb(n, i, env, run):
    keys = [k.strip() for k in (n["params"].get("keys") or "a,b,c,d").split(",")]
    return {"json": {k: i.get(p) for k, p in zip(keys, ("a", "b", "c", "d")) if k and i.get(p) is not None}}


@node("json_combine", "JSON: combine", "Data", inputs=[("a", "json", False), ("b", "json", False)], outputs=[("json", "json")],
      help="Merge two objects (b wins), or join two lists.")
def _n_jc(n, i, env, run):
    a, b = i.get("a"), i.get("b")
    a = _json(a) if a is not None else None
    b = _json(b) if b is not None else None
    if isinstance(a, list) or isinstance(b, list):
        return {"json": (a or []) + (b or [])}
    return {"json": {**(a or {}), **(b or {})}}


@node("join", "Join text", "Data", inputs=[("a", "any", False), ("b", "any", False), ("c", "any", False)],
      outputs=[("text", "text")], params=[{"name": "separator", "kind": "str", "default": "\n\n"}],
      help="Join the inputs that have values.")
def _n_join(n, i, env, run):
    sep = (n["params"].get("separator") or "").encode().decode("unicode_escape")
    return {"text": sep.join(_text(i[k]) for k in ("a", "b", "c") if i.get(k) is not None)}


@node("translate", "Translate", "Data", inputs=[("text", "text", True)], outputs=[("text", "text")],
      params=[{"name": "to", "kind": "str", "default": "English"}, {"name": "model", "kind": "model", "default": ""}],
      help="Translate with a model, keeping formatting.")
def _n_translate(n, i, env, run):
    out = env.chat([{"role": "system", "content": f"Translate the user's text into {n['params'].get('to') or 'English'}. "
                     "Keep formatting and code unchanged. Reply with the translation only."},
                    {"role": "user", "content": _text(i["text"])}], n["params"].get("model"), "", 0.2, 2000)
    return {"text": out["text"]}


# ------------------------------------------------------------------ tools

@node("mcp_tool", "MCP tool", "Tools", inputs=[("args", "json", False)], outputs=[("result", "text")],
      params=[{"name": "tool", "kind": "str", "default": "", "help": "server/tool, as in chat"},
              {"name": "args", "kind": "json", "default": {}, "help": "arguments; the args input is merged over them"},
              {"name": "profile", "kind": "profile", "default": ""}],
      acts=["tool", "read", "write", "run", "net"],
      help="Call one of Studio's MCP tools. The permission policy and tool approval both apply; ask counts as no.")
def _n_mcp(n, i, env, run):
    import permissions as P
    p = n["params"]
    server, _, tool = (p.get("tool") or "").partition("/")
    args = {**(p.get("args") or {}), **(_json(i["args"]) if i.get("args") is not None else {})}
    d = P.check_call(env.policy, env.model, server, tool, args, job="graph", profile=p.get("profile") or env.profile,
                     ctx=P.context(workdir=env.workdir))
    if d["effect"] != "allow":
        raise GraphError(f"not permitted: {d['why']}")
    if env.mcp_call is None:
        raise GraphError("MCP tools run inside Studio")
    ok, text = env.mcp_call(server, tool, args)
    if not ok:
        raise GraphError(text)
    return {"result": text}


@node("http_get", "Fetch a URL", "Tools", inputs=[("url", "text", True)], outputs=[("text", "text")],
      params=[{"name": "max_chars", "kind": "int", "default": 20000}, {"name": "profile", "kind": "profile", "default": ""}],
      acts=["net"], help="GET a URL (the net permission applies); HTML is reduced to text.")
def _n_http(n, i, env, run):
    import urllib.request
    url = _text(i["url"]).strip()
    if not re.match(r"https?://", url):
        raise GraphError("http or https URLs only")
    env.permit("net", url, n["params"].get("profile", ""))
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OmniCortex-graph"}), timeout=30) as r:
        body = r.read(2_000_000).decode("utf-8", "replace")
    if "<html" in body[:500].lower():
        body = re.sub(r"(?s)<(script|style).*?</\1>", " ", body)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body)
    return {"text": body[:int(n["params"].get("max_chars") or 20000)]}


@node("file_read", "Read a file", "Tools", inputs=[("path", "text", False)], outputs=[("text", "text")],
      params=[{"name": "path", "kind": "str", "default": ""}, {"name": "profile", "kind": "profile", "default": ""}],
      acts=["read"], help="Read a text file (the read permission applies).")
def _n_fread(n, i, env, run):
    import permissions as P
    path = P.normalise("read", _text(i.get("path")) or n["params"].get("path") or "", P.context(workdir=env.workdir))
    env.permit("read", path, n["params"].get("profile", ""))
    return {"text": Path(path).read_text(encoding="utf-8", errors="replace")[:MAX_TEXT]}


@node("file_write", "Write a file", "Tools", inputs=[("text", "text", True), ("path", "text", False)], outputs=[("path", "text")],
      params=[{"name": "path", "kind": "str", "default": ""}, {"name": "profile", "kind": "profile", "default": ""}],
      acts=["write"], help="Write text to a file (the write permission applies).")
def _n_fwrite(n, i, env, run):
    import permissions as P
    path = P.normalise("write", _text(i.get("path")) or n["params"].get("path") or "", P.context(workdir=env.workdir))
    env.permit("write", path, n["params"].get("profile", ""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(_text(i["text"]), encoding="utf-8")
    return {"path": path}


@node("rag_search", "Search documents", "Tools", inputs=[("query", "text", True)], outputs=[("text", "text"), ("hits", "list")],
      params=[{"name": "collection", "kind": "str", "default": ""}, {"name": "k", "kind": "int", "default": 4}],
      help="Search a Studio document collection; the text output is the passages, numbered, for a model's context.")
def _n_rag(n, i, env, run):
    r = env.need_studio().call("POST", "/api/rag/search", {"collection": n["params"].get("collection"),
                                                            "query": _text(i["query"]), "k": int(n["params"].get("k") or 4)})
    hits = r.get("hits") if isinstance(r, dict) else r
    return {"text": "\n\n".join(f"[{j + 1}] {h.get('source', '')}: {_text(h.get('text'))}" for j, h in enumerate(hits or [])),
            "hits": hits or []}


# ------------------------------------------------------------------ delegation

@node("worker", "Worker", "Delegation", inputs=[("task", "text", True), ("inputs", "text", False), ("feedback", "text", False)],
      outputs=[("result", "text"), ("summary", "text"), ("blocked", "text")],
      params=[{"name": "model", "kind": "model", "default": "", "help": "empty or auto: Studio picks by subject"},
              {"name": "acceptance", "kind": "text", "default": "", "help": "what done means"},
              {"name": "profile", "kind": "profile", "default": "developer", "help": "what the worker may do; it is told"}],
      help="Do a task the way the director's workers do: the worker prompt, the task's permissions, earlier "
           "feedback, and a done / blocked report.")
def _n_worker(n, i, env, run):
    import director
    import permissions as P
    p = n["params"]
    lines = [f"Your task: {_text(i['task'])}"]
    if p.get("acceptance"):
        lines.append(f"Done means: {p['acceptance']}")
    if i.get("inputs") is not None:
        lines += ["", "Input from earlier steps:", _text(i["inputs"])]
    if i.get("feedback") is not None:
        lines += ["", "Your last attempt was sent back. Feedback:", _text(i["feedback"])]
    lines += ["", P.prompt_text(env.policy, env.model, "project_task", p.get("profile") or "", P.context(workdir=env.workdir))]
    out = env.chat([{"role": "system", "content": director.WORKER_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
                   p.get("model") or "auto", "", 0.4, 4096)
    rep = {}
    try:
        rep = _json(out["text"][out["text"].rfind("```json"):]) if "```json" in out["text"] else {}
    except (GraphError, ValueError):
        pass
    rep = rep if isinstance(rep, dict) else {}
    blocked = rep.get("status") == "blocked"
    return {"result": SKIP if blocked else out["text"], "summary": _text(rep.get("summary") or ""),
            "blocked": ("; ".join(map(str, rep.get("questions") or [])) or out["text"]) if blocked else SKIP}


@node("subgraph", "Graph", "Delegation", inputs=[("input", "any", False)], outputs=[("result", "any"), ("outputs", "json")],
      params=[{"name": "graph", "kind": "graph", "default": ""},
              {"name": "input_name", "kind": "str", "default": "input"},
              {"name": "result_name", "kind": "str", "default": "result"}],
      help="Run another saved graph as one step.")
def _n_subgraph(n, i, env, run):
    if env.graph_loader is None:
        raise GraphError("saved graphs are loaded by Studio")
    if run.get("depth", 0) >= 4:
        raise GraphError("graphs nest at most 4 deep")
    g = env.graph_loader(n["params"].get("graph") or "")
    sub = Runner(g, env).start({n["params"].get("input_name") or "input": i.get("input")}, depth=run.get("depth", 0) + 1)
    if sub["status"] != "done":
        raise GraphError(f"graph {g.get('name')}: {sub['status']} {sub.get('error') or ''}")
    return {"result": sub["outputs"].get(n["params"].get("result_name") or "result", SKIP), "outputs": sub["outputs"]}


# ------------------------------------------------------------------ plugin nodes

def sync_plugins(plugins) -> None:
    """Node types from enabled plugins (viz/plugins.py, "nodes" in plugin.json), as plugin:<plugin>:<type>.
    Each runs its script with {"params", "inputs"} on stdin; turned-off plugins' types go away."""
    for k in [k for k in TYPES if k.startswith("plugin:")]:
        del TYPES[k]
    for n in plugins.graph_nodes() if plugins else []:
        name = f"plugin:{n['plugin']}:{n['type']}"

        def run(nd, ins, env, run_, plugin=n["plugin"], typ=n["type"], outs=[p[0] for p in n.get("outputs") or []]):
            got = plugins.run_node(plugin, typ, {"params": nd.get("params") or {},
                                                 "inputs": {k: v for k, v in ins.items() if v is not None}})
            return {k: got[k] if k in got else SKIP for k in outs}
        node(name, n.get("title") or n["type"], "Plugins",
             inputs=[(p[0], p[1] if len(p) > 1 else "any", bool(p[2]) if len(p) > 2 else False) for p in n.get("inputs") or []],
             outputs=[(p[0], p[1] if len(p) > 1 else "any") for p in n.get("outputs") or []],
             params=[p for p in n.get("params") or [] if isinstance(p, dict) and p.get("name")],
             help=f"{n.get('help', '')} (plugin {n['plugin']})".strip())(run)


# ------------------------------------------------------------------ validation and running

def validate(g: dict) -> dict:
    if not isinstance(g, dict):
        raise GraphError("a graph is an object")
    nodes = g.get("nodes") or []
    ids = set()
    for nd in nodes:
        if not re.match(r"^[A-Za-z0-9_-]{1,40}$", str(nd.get("id", ""))):
            raise GraphError(f"node id {nd.get('id')!r}")
        if nd["id"] in ids:
            raise GraphError(f"two nodes are called {nd['id']}")
        ids.add(nd["id"])
        if nd.get("type") not in TYPES:
            raise GraphError(f"{nd['id']}: unknown node type {nd.get('type')!r}")
        if HIDDEN and HIDDEN(nd["type"]):
            raise GraphError(f"{nd['id']}: {TYPES[nd['type']]['title']} belongs to an add-on that is removed (Add-ons)")
        nd.setdefault("params", {})
    by = {nd["id"]: nd for nd in nodes}
    for e in g.get("edges") or []:
        (a, ap), (b, bp) = e["from"], e["to"]
        if a not in by or b not in by:
            raise GraphError(f"edge {e}: no such node")
        if ap not in [o["name"] for o in TYPES[by[a]["type"]]["outputs"]]:
            raise GraphError(f"{a} has no output {ap!r}")
        if bp not in [x["name"] for x in TYPES[by[b]["type"]]["inputs"]]:
            raise GraphError(f"{b} has no input {bp!r}")
    _no_free_cycles(by, g.get("edges") or [])
    return g


def _no_free_cycles(by, edges):
    """Cycles are allowed only through a loop node, so every cycle is bounded."""
    nxt = {k: [] for k in by}
    for e in edges:
        if by[e["to"][0]]["type"] != "loop":
            nxt[e["from"][0]].append(e["to"][0])
    seen, stack = set(), set()

    def dfs(u):
        seen.add(u)
        stack.add(u)
        for v in nxt[u]:
            if v in stack:
                raise GraphError(f"a cycle through {v} has no loop node to bound it")
            if v not in seen:
                dfs(v)
        stack.discard(u)
    for k in by:
        if k not in seen:
            dfs(k)


class Runner:
    def __init__(self, graph: dict, env: Env):
        self.g = validate(copy.deepcopy(graph))
        self.env = env
        self.by = {nd["id"]: nd for nd in self.g["nodes"]}
        self.out_edges: dict[tuple, list] = {}
        self.wired: dict[str, set] = {k: set() for k in self.by}
        self.required = {k: {x["name"]: x["required"] for x in TYPES[nd["type"]]["inputs"]} for k, nd in self.by.items()}
        for e in self.g.get("edges") or []:
            self.out_edges.setdefault(tuple(e["from"]), []).append(tuple(e["to"]))
            self.wired[e["to"][0]].add(e["to"][1])

    def start(self, inputs: dict, depth: int = 0) -> dict:
        run = {"id": uuid.uuid4().hex[:12], "graph": self.g.get("name", ""), "status": "running", "inputs": dict(inputs or {}),
               "outputs": {}, "trace": [], "values": {k: {} for k in self.by}, "state": {}, "decisions": {},
               "waiting": None, "queue": [], "_trigger": {}, "steps": 0, "started": time.time(), "depth": depth}
        # sources: nodes with no wired inputs
        run["queue"] = [k for k in self.by if not self.wired[k]]
        return self._go(run)

    def resume(self, run: dict, node_id: str, approve: bool, value=None, note: str = "") -> dict:
        if run.get("status") != "waiting" or (run.get("waiting") or {}).get("node") != node_id:
            raise GraphError("that run is not waiting on that node")
        dec = {"approve": bool(approve), "note": note}
        if value is not None:
            dec["value"] = value
        run["decisions"][node_id] = dec
        run["status"], run["waiting"] = "running", None
        run["queue"].insert(0, node_id)
        return self._go(run)

    def _ready(self, run, k) -> bool:
        spec = TYPES[self.by[k]["type"]]
        vals = run["values"][k]
        return all(x["name"] in vals for x in spec["inputs"] if x["required"] and x["name"] in self.wired[k]) and \
            (bool(vals) or not self.wired[k])

    def _go(self, run: dict) -> dict:
        q = deque(run["queue"])
        run["queue"] = []
        try:
            while q:
                k = q.popleft()
                if not self._ready(run, k):
                    continue
                run["steps"] += 1
                if run["steps"] > MAX_STEPS:
                    raise GraphError(f"stopped after {MAX_STEPS} steps (a loop without an end?)")
                nd, spec = self.by[k], TYPES[self.by[k]["type"]]
                vals = run["values"][k]
                skipped = any(vals.get(x["name"]) is SKIP for x in spec["inputs"] if x["required"] and x["name"] in self.wired[k])
                ins = {x["name"]: (None if vals.get(x["name"]) is SKIP else vals.get(x["name"])) for x in spec["inputs"]}
                t0 = time.time()
                rec = {"node": k, "type": nd["type"], "inputs": {a: _preview(b) for a, b in ins.items() if b is not None}}
                if skipped:
                    outs, rec["status"] = {o["name"]: SKIP for o in spec["outputs"]}, "skipped"
                else:
                    try:
                        outs = spec["run"](nd, ins, self.env, run) or {}
                        rec["status"] = "ok"
                    except Pause:
                        run["status"] = "waiting"
                        run["queue"] = list(q)
                        rec["status"] = "waiting"
                        run["trace"].append(rec)
                        return run
                    except Exception as e:                                 # noqa: BLE001 - shown on the node
                        msg = f"{type(e).__name__}: {e}" if not isinstance(e, GraphError) else str(e)
                        rec.update({"status": "error", "error": msg[:2000]})
                        if (k, "error") not in self.out_edges:
                            run["trace"].append(rec)
                            raise GraphError(f"{k} ({spec['title']}): {msg}")
                        outs = {o["name"]: SKIP for o in spec["outputs"]}
                        outs["error"] = msg
                    outs = {o["name"]: outs.get(o["name"], SKIP) for o in spec["outputs"]}
                rec["ms"] = int((time.time() - t0) * 1000)
                rec["outputs"] = {a: ("(skipped)" if b is SKIP else _preview(b)) for a, b in outs.items()
                                  if (k, a) in self.out_edges or b is not SKIP}
                run["trace"].append(rec)
                run["_trigger"].pop(k, None)
                for port, v in outs.items():
                    for (b, bp) in self.out_edges.get((k, port), []):
                        run["values"][b][bp] = v
                        # a skip only travels through required inputs; on an optional one it is just "no value"
                        if v is SKIP and not self.required[b].get(bp):
                            continue
                        run["_trigger"][b] = bp
                        if b not in q:
                            q.append(b)
            run["status"] = "done"
        except GraphError as e:
            run["status"], run["error"] = "failed", str(e)
        run["finished"] = time.time()
        run["outputs"] = {k: v for k, v in run["outputs"].items() if v is not SKIP}
        return run


def _preview(v, n=2000):
    if v is SKIP:
        return "(skipped)"
    if isinstance(v, str):
        return v[:n] + ("…" if len(v) > n else "")
    try:
        s = json.dumps(v, ensure_ascii=False)
        return v if len(s) <= n else s[:n] + "…"
    except (TypeError, ValueError):
        return str(v)[:n]


def public_run(run: dict) -> dict:
    """A run as stored and shown: without the live values (they can be large or not JSON)."""
    out = {k: v for k, v in run.items() if k not in ("values", "_trigger", "queue", "state", "decisions")}
    out["outputs"] = {k: _preview(v, 20000) for k, v in run.get("outputs", {}).items()}
    if run.get("waiting"):
        out["waiting"] = {**run["waiting"], "value": _preview(run["waiting"].get("value"), 20000)}
    return out


# ------------------------------------------------------------------ evals

def run_eval(graph: dict, cases: list[dict], env: Env, expect: str = "", output: str = "result",
             grade: str = "contains") -> dict:
    """Run the graph once per case (each case's keys are the run's inputs). With expect, compare
    outputs[output] with case[expect]: exact, contains, or regex. Without, a case passes when the run
    finishes with that output."""
    rows = []
    for c in cases:
        r = Runner(graph, env).start({k: v for k, v in c.items() if k != expect})
        got = _text(r["outputs"].get(output, ""))
        ok = r["status"] == "done" and output in r["outputs"]
        if expect and ok:
            want = _text(c.get(expect))
            ok = (got.strip().lower() == want.strip().lower() if grade == "exact" else
                  bool(re.search(want, got, re.I)) if grade == "regex" else want.strip().lower() in got.lower())
        rows.append({"inputs": {k: _preview(v, 300) for k, v in c.items()}, "status": r["status"], "ok": ok,
                     "got": got[:1000], "error": r.get("error"), "steps": r["steps"]})
    n = len(rows)
    return {"cases": n, "passed": sum(r["ok"] for r in rows), "rate": (sum(r["ok"] for r in rows) / n) if n else 0.0,
            "rows": rows}


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("types", help="the node types")
    r = sub.add_parser("run", help="run a graph")
    r.add_argument("graph")
    r.add_argument("--input", action="append", default=[], help="name=value (repeatable)")
    e = sub.add_parser("eval", help="run a graph over test cases")
    e.add_argument("graph")
    e.add_argument("--cases", required=True, help="JSONL, one case per line: the run's inputs and the expected value")
    e.add_argument("--expect", default="")
    e.add_argument("--output", default="result")
    e.add_argument("--grade", choices=["contains", "exact", "regex"], default="contains")
    for x in (r, e):
        x.add_argument("--studio", default=os.environ.get("OC_STUDIO_URL", "http://127.0.0.1:7870"))
        x.add_argument("--token-file")
        x.add_argument("--model", default="")
        x.add_argument("--profile", default="")
        x.add_argument("--workdir")
    a = p.parse_args(argv)
    if a.cmd == "types":
        for g in GROUPS:
            print(g)
            for t in TYPES.values():
                if t["group"] == g:
                    print(f"  {t['name']:<20} {t['help'][:90]}")
        return 0
    if a.cmd not in ("run", "eval"):
        p.print_help()
        return 0
    import oc_mcp
    tok = Path(a.token_file).read_text().strip() if a.token_file else os.environ.get("OC_STUDIO_TOKEN")
    env = Env(oc_mcp.Studio(a.studio, tok), model=a.model, profile=a.profile, workdir=a.workdir)
    g = json.loads(Path(a.graph).read_text())
    if a.cmd == "run":
        run = Runner(g, env).start(dict(x.split("=", 1) for x in a.input))
        print(json.dumps(public_run(run), indent=1, ensure_ascii=False))
        return 0 if run["status"] == "done" else 1
    cases = [json.loads(line) for line in Path(a.cases).read_text().splitlines() if line.strip()]
    res = run_eval(g, cases, env, a.expect, a.output, a.grade)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
