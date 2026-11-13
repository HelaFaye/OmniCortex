"""Node graphs (scripts/graph_engine.py): routing, skips, errors, loops, approvals, permissions, evals."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import graph_engine as G  # noqa: E402
import permissions as P  # noqa: E402


class FakeEnv(G.Env):
    """Answers from a script: each call pops the next reply (or uses a function of the prompt)."""

    def __init__(self, tmp_path, replies=None, fn=None, **kw):
        super().__init__(policy=P.Policy(tmp_path / "perm.json"), **kw)
        self.replies, self.fn, self.calls = list(replies or []), fn, []

    def chat(self, messages, model="", endpoint="", temperature=0.7, max_tokens=1024, json_mode=False):
        self.calls.append(messages)
        text = self.fn(messages) if self.fn else self.replies.pop(0)
        return {"text": text, "reasoning": "", "model": model or "m"}


def N(id, type, **params):
    return {"id": id, "type": type, "params": params}


def E(a, ap, b, bp):
    return {"from": [a, ap], "to": [b, bp]}


def test_env_scratch_is_private_and_cleaned(tmp_path):
    env = G.Env(policy=P.Policy(tmp_path / "permissions.json"), workdir=str(tmp_path))
    scratch = Path(env.open_scratch())
    try:
        ctx = env.permission_context()
        assert Path(ctx["tmp"]) == scratch.resolve()
        assert P.Policy(tmp_path / "permissions.json").check(
            "read", str(scratch / "inside.txt"), job="swe_evaluate", ctx=ctx
        )["effect"] == "allow"
        # A sibling of the checkout, not a child of workdir (which is allowed).
        outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
        assert P.Policy(tmp_path / "permissions.json").check(
            "read", str(outside), job="swe_evaluate", ctx=ctx
        )["effect"] == "deny"
    finally:
        env.close_scratch()
    assert not scratch.exists()


def test_graph_api_scratch_survives_approval_then_cleans(tmp_path):
    import sys
    import time

    sys.path.insert(0, str(ROOT / "viz"))
    import graphs_api as GA

    graph = {
        "nodes": [N("in", "input"), N("okq", "approval"), N("o", "output"), N("no", "output", name="no")],
        "edges": [E("in", "value", "okq", "value"), E("okq", "approved", "o", "value"),
                  E("okq", "rejected", "no", "value")],
    }
    env = G.Env(policy=P.Policy(tmp_path / "permissions.json"))
    store = GA.Store(tmp_path / "graphs")
    rid = GA.start(store, graph, {"input": "review"}, env)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = store.run(rid)
        if current.get("status") == "waiting":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("graph did not reach its approval pause")

    scratch = Path(env.scratch_dir)
    assert scratch.is_dir()
    GA.decide(store, rid, "okq", True, value="approved")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with GA.LOCK:
            active = GA.RUNS.get(rid)
        if active is None:
            current = store.run(rid)
            if current.get("status") == "done":
                break
        time.sleep(0.01)
    else:
        raise AssertionError("graph did not finish and clean up after approval")
    current = store.run(rid)

    assert current["status"] == "done"
    assert not scratch.exists()


def test_screen_routes_pass_and_fail(tmp_path):
    g = {"nodes": [N("in", "input", name="q"), N("ask", "llm"), N("sec", "secrets_screen", kinds="API key"),
                   N("ok", "output", name="result"), N("bad", "output", name="blocked")],
         "edges": [E("in", "value", "ask", "prompt"), E("ask", "text", "sec", "text"), E("sec", "pass", "ok", "value"),
                   E("sec", "redacted", "bad", "value")]}
    env = FakeEnv(tmp_path, ["here: sk-abcdefghijklmnopqrstuvwxyz123"])
    r = G.Runner(g, env).start({"q": "key?"})
    assert r["status"] == "done" and "result" not in r["outputs"] and "[API key]" in r["outputs"]["blocked"]
    assert [t["status"] for t in r["trace"] if t["node"] == "ok"] == ["skipped"]
    r = G.Runner(g, FakeEnv(tmp_path, ["all clean"])).start({"q": "x"})
    assert r["outputs"]["result"] == "all clean"


def test_judge_loop_retries_with_feedback(tmp_path):
    # ask -> judge; fail goes back round the loop as feedback, up to 2 more rounds
    g = {"nodes": [N("in", "input", name="q"), N("loop", "loop", max=2), N("w", "template", template="{a}\n{b}"),
                   N("ask", "llm"), N("j", "judge", pass_score=7), N("out", "output"), N("gave_up", "output", name="gave_up")],
         "edges": [E("in", "value", "loop", "start"), E("loop", "value", "w", "a"), E("w", "text", "ask", "prompt"),
                   E("ask", "text", "j", "answer"), E("in", "value", "j", "question"), E("j", "pass", "out", "value"),
                   E("j", "fail", "loop", "again"), E("loop", "done", "gave_up", "value")]}
    script = iter(["draft 1", "Score: 3\nwrong", "draft 2", "Score: 8\nfine"])
    env = FakeEnv(tmp_path, fn=lambda m: next(script))
    r = G.Runner(g, env).start({"q": "what?"})
    assert r["status"] == "done" and r["outputs"]["result"] == "draft 2" and "gave_up" not in r["outputs"]
    assert "wrong" in env.calls[2][-1]["content"]                       # the feedback reached the retry
    script2 = iter(["d", "Score: 1\nno"] * 5)
    r = G.Runner(g, FakeEnv(tmp_path, fn=lambda m: next(script2))).start({"q": "what?"})
    assert r["outputs"].get("gave_up") and "result" not in r["outputs"]


def test_errors_go_to_the_error_port_or_stop(tmp_path):
    g = {"nodes": [N("in", "input"), N("x", "json_extract", path="a.b"), N("o", "output"), N("err", "output", name="err")],
         "edges": [E("in", "value", "x", "value"), E("x", "value", "o", "value")]}
    r = G.Runner(g, FakeEnv(tmp_path)).start({"input": "not json"})
    assert r["status"] == "failed" and "x (JSON: extract)" in r["error"]
    g["edges"].append(E("x", "error", "err", "value"))
    r = G.Runner(g, FakeEnv(tmp_path)).start({"input": "not json"})
    assert r["status"] == "done" and "not JSON" in r["outputs"]["err"]
    r = G.Runner(g, FakeEnv(tmp_path)).start({"input": '{"a": {"b": 5}}'})
    assert r["outputs"]["result"] == 5


def test_approval_pauses_and_resumes(tmp_path):
    g = {"nodes": [N("in", "input"), N("okq", "approval"), N("o", "output"), N("no", "output", name="no")],
         "edges": [E("in", "value", "okq", "value"), E("okq", "approved", "o", "value"), E("okq", "rejected", "no", "value")]}
    rn = G.Runner(g, FakeEnv(tmp_path))
    r = rn.start({"input": "ship it"})
    assert r["status"] == "waiting" and r["waiting"]["node"] == "okq"
    r = rn.resume(r, "okq", True, value="ship it, edited")
    assert r["status"] == "done" and r["outputs"]["result"] == "ship it, edited"
    r = rn.resume(rn.start({"input": "x"}), "okq", False, note="not yet")
    assert r["outputs"] == {"no": "not yet"}


def test_permissions_apply_to_tool_nodes(tmp_path):
    g = {"nodes": [N("p", "input"), N("w", "file_write"), N("t", "input", name="t"), N("o", "output")],
         "edges": [E("t", "value", "w", "text"), E("p", "value", "w", "path"), E("w", "path", "o", "value")]}
    target = tmp_path / "out.txt"
    r = G.Runner(g, FakeEnv(tmp_path)).start({"input": str(target), "t": "hi"})
    assert r["status"] == "failed" and "not permitted" in r["error"]           # the base profile asks for writes
    r = G.Runner(g, FakeEnv(tmp_path, profile="full")).start({"input": str(target), "t": "hi"})
    assert r["status"] == "done" and target.read_text() == "hi"
    r = G.Runner(g, FakeEnv(tmp_path, profile="full")).start({"input": str(Path.home() / ".ssh" / "x"), "t": "hi"})
    assert r["status"] == "failed"                                              # the base still denies secrets
    g2 = {"nodes": [N("c", "input"), N("gate", "permission_gate", action="run"), N("a", "output", name="a"), N("d", "output", name="d")],
          "edges": [E("c", "value", "gate", "target"), E("gate", "allow", "a", "value"), E("gate", "deny", "d", "value")]}
    r = G.Runner(g2, FakeEnv(tmp_path)).start({"input": "rm -rf /"})
    assert r["outputs"] == {"d": "rm -rf /"}


def test_validation_and_structured_output(tmp_path):
    with pytest.raises(G.GraphError):
        G.validate({"nodes": [N("a", "nope")], "edges": []})
    with pytest.raises(G.GraphError):                                           # an unbounded cycle
        G.validate({"nodes": [N("a", "template"), N("b", "template")],
                    "edges": [E("a", "text", "b", "a"), E("b", "text", "a", "a")]})
    g = {"nodes": [N("in", "input"), N("ask", "llm", json_schema={"type": "object", "required": ["n"],
                                                                  "properties": {"n": {"type": "integer"}}}),
                   N("o", "output")], "edges": [E("in", "value", "ask", "prompt"), E("ask", "json", "o", "value")]}
    env = FakeEnv(tmp_path, ['{"n": "five"}', '```json\n{"n": 5}\n```'])
    r = G.Runner(g, env).start({"input": "count"})
    assert r["outputs"]["result"] == {"n": 5} and len(env.calls) == 2           # asked once more after a bad answer


def test_eval_and_templates(tmp_path):
    g = {"nodes": [N("in", "input", name="q"), N("ask", "llm"), N("o", "output")],
         "edges": [E("in", "value", "ask", "prompt"), E("ask", "text", "o", "value")]}
    env = FakeEnv(tmp_path, fn=lambda m: "Paris" if "France" in m[-1]["content"] else "dunno")
    res = G.run_eval(g, [{"q": "capital of France?", "a": "paris"}, {"q": "capital of Mars?", "a": "none"}], env, expect="a")
    assert res["cases"] == 2 and res["passed"] == 1
    import graph_templates
    for name, t in graph_templates.TEMPLATES.items():
        G.validate(t)


def test_verification_template_and_director_backport(tmp_path):
    import director as dr
    import graph_templates
    g = graph_templates.TEMPLATES["verify-delegated-result"]
    good = FakeEnv(tmp_path, ["Score: 9\nfine"])
    r = G.Runner(g, good).start({"task": "Write a haiku about rain", "result": "Soft rain on the roof,\nthe gutters hum..."})
    assert set(r["outputs"]) == {"accepted"}
    bad = FakeEnv(tmp_path, ["Score: 2\nit is not a haiku"])
    r = G.Runner(g, bad).start({"task": "Write a haiku about rain", "result": "Rain is wet and also falls down a lot."})
    assert "not a haiku" in r["outputs"]["feedback"]
    r = G.Runner(g, FakeEnv(tmp_path)).start({"task": "x", "result": "key sk-abcdefghijklmnopqrstuvwxyz0123"})
    assert "API key" in r["outputs"]["feedback"]

    # the director: a failed screen with verify_auto goes straight back to the worker with the feedback
    plan = dr.new_plan("p", "t", [{"title": "Write a haiku", "detail": "about rain", "labels": []}]) if hasattr(dr, "new_plan") else None
    if plan is None:
        pytest.skip("no plan constructor")
    dr.set_policy(plan, {"verify_graph": "template:verify-delegated-result", "verify_auto": True})
    t = plan["tasks"][0]
    t["status"], t["attempts"] = "running", [{"n": 1, "model": "m", "started": 0}]
    dr.finish_attempt(plan, t, "Rain is wet.\n```json\n{\"status\": \"done\", \"summary\": \"s\"}\n```",
                      screen={"status": "failed", "feedback": "not a haiku", "graph": "g"})
    assert t["status"] == "rework" and "not a haiku" in t["attempts"][-1]["review"]["feedback"]
    assert t["attempts"][-1]["review"]["by"] == "graph" and t["attempts"][-1]["screen"]["status"] == "failed"
    with pytest.raises(dr.PlanError):
        dr.set_policy(plan, {"verify_graph": "../etc"})


sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "viz"))
from test_studio import studio_srv  # noqa: E402,F401


def _post(base, path, body):
    import urllib.request
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base, path):
    import urllib.request
    with urllib.request.urlopen(base + path, timeout=20) as r:
        return json.loads(r.read())


def _wait(base, rid):
    import time
    for _ in range(100):
        r = _get(base, f"/api/graphs/runs/{rid}")
        if r["status"] not in ("running", "queued"):
            return r
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_studio_graphs(studio_srv, tmp_path):  # noqa: F811
    import studio
    studio.STATE["graphs_dir"] = str(tmp_path / "graphs")
    cat = _get(studio_srv, "/api/graphs")
    assert len(cat["types"]) > 30 and any(t["name"] == "screen-and-verify" for t in cat["templates"])
    g = {"name": "approve-me", "nodes": [N("in", "input"), N("sec", "secrets_screen"), N("ask", "approval"), N("o", "output")],
         "edges": [E("in", "value", "sec", "text"), E("sec", "pass", "ask", "value"), E("ask", "approved", "o", "value")]}
    assert _post(studio_srv, "/api/graphs/save", {"graph": g})[0] == 200
    assert _post(studio_srv, "/api/graphs/save", {"graph": {**g, "name": "../x"}})[0] == 400
    code, r = _post(studio_srv, "/api/graphs/run", {"name": "approve-me", "inputs": {"input": "hello"}})
    run = _wait(studio_srv, r["id"])
    assert run["status"] == "waiting" and run["waiting"]["node"] == "ask"
    _post(studio_srv, f"/api/graphs/runs/{r['id']}/decide", {"node": "ask", "approve": True, "value": "hello!"})
    run = _wait(studio_srv, r["id"])
    assert run["status"] == "done" and run["outputs"]["result"] == "hello!"
    code, d = _post(studio_srv, "/api/graphs/node", {"graph": g, "node": "sec", "inputs": {"text": "a@b.co"}})
    assert d["ok"] and d["outputs"]["fail"]
    code, r = _post(studio_srv, "/api/graphs/eval", {"name": "template:screen-and-verify", "cases": '{"question": "hi"}'})
    ev = _wait(studio_srv, r["id"])
    assert ev["cases"] == 1 and ev["passed"] == 0                       # no model is loaded in the test Studio
    assert any(x["id"] == r["id"] for x in _get(studio_srv, "/api/graphs/runs")["runs"])
    import urllib.request
    assert "Graphs" in urllib.request.urlopen(studio_srv + "/graphs", timeout=20).read().decode()


def test_plugin_nodes(tmp_path):
    import plugins as PL
    d = tmp_path / "plugins" / "counter"
    d.mkdir(parents=True)
    (d / "plugin.json").write_text(json.dumps({"name": "counter", "nodes": [
        {"type": "words", "title": "Count words", "script": "node.py", "inputs": [["text", "text", True]],
         "outputs": [["count", "number"], ["long", "text"]], "params": [{"name": "over", "kind": "int", "default": 3}]}]}))
    (d / "node.py").write_text("import json,sys\nd=json.load(sys.stdin)\nn=len(d['inputs']['text'].split())\n"
                               "out={'count': n}\nif n > d['params']['over']: out['long'] = d['inputs']['text']\nprint(json.dumps(out))\n")
    pl = PL.Plugins([tmp_path / "plugins"], tmp_path / "state.json")
    G.sync_plugins(pl)
    assert "plugin:counter:words" not in G.TYPES                     # off until the owner turns it on
    pl.set_enabled("counter", True)
    G.sync_plugins(pl)
    g = {"nodes": [N("in", "input"), N("c", "plugin:counter:words", over=3), N("o", "output"), N("l", "output", name="long")],
         "edges": [E("in", "value", "c", "text"), E("c", "count", "o", "value"), E("c", "long", "l", "value")]}
    r = G.Runner(g, FakeEnv(tmp_path)).start({"input": "one two"})
    assert r["outputs"] == {"result": 2}
    r = G.Runner(g, FakeEnv(tmp_path)).start({"input": "one two three four five"})
    assert r["outputs"]["long"] == "one two three four five"
    G.sync_plugins(None)
    assert not any(k.startswith("plugin:") for k in G.TYPES)


def test_graph_api_resume_keeps_published_state_stable(tmp_path, monkeypatch):
    import sys
    import threading
    import time

    sys.path.insert(0, str(ROOT / "viz"))
    import graphs_api as GA

    graph = {
        "nodes": [N("in", "input"), N("okq", "approval"), N("o", "output")],
        "edges": [E("in", "value", "okq", "value"), E("okq", "approved", "o", "value")],
    }
    env = G.Env(policy=P.Policy(tmp_path / "permissions.json"))
    store = GA.Store(tmp_path / "graphs")
    entered, release = threading.Event(), threading.Event()
    original_resume = G.Runner.resume

    def held_resume(self, run, *args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test resume gate timed out")
        return original_resume(self, run, *args, **kwargs)

    monkeypatch.setattr(G.Runner, "resume", held_resume)
    rid = GA.start(store, graph, {"input": "review"}, env)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = store.run(rid)
        if current.get("status") == "waiting":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("graph did not reach its approval pause")

    scratch = Path(env.scratch_dir)
    assert scratch.is_dir()
    GA.decide(store, rid, "okq", True, value="approved")
    assert entered.wait(10), "resume thread did not enter Runner.resume"

    with GA.LOCK:
        live = GA.RUNS.get(rid)
        assert live is not None
        assert live["status"] == "running"
        assert live["run"]["status"] == "waiting"  # published state was not mutated in place
    assert store.run(rid)["status"] == "running"
    assert scratch.is_dir()

    release.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with GA.LOCK:
            active = GA.RUNS.get(rid)
        if active is None:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("graph did not finish after releasing the resume gate")

    assert store.run(rid)["status"] == "done"
    assert not scratch.exists()


def test_graph_api_scratch_cleans_after_resume_error(tmp_path, monkeypatch):
    import sys
    import time

    sys.path.insert(0, str(ROOT / "viz"))
    import graphs_api as GA

    graph = {
        "nodes": [N("in", "input"), N("okq", "approval"), N("o", "output")],
        "edges": [E("in", "value", "okq", "value"), E("okq", "approved", "o", "value")],
    }
    env = G.Env(policy=P.Policy(tmp_path / "permissions.json"))
    store = GA.Store(tmp_path / "graphs")

    def fail_resume(*args, **kwargs):
        raise RuntimeError("intentional resume failure")

    monkeypatch.setattr(G.Runner, "resume", fail_resume)
    rid = GA.start(store, graph, {"input": "review"}, env)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = store.run(rid)
        if current.get("status") == "waiting":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("graph did not reach its approval pause")

    scratch = Path(env.scratch_dir)
    assert scratch.is_dir()
    GA.decide(store, rid, "okq", True, value="approved")

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with GA.LOCK:
            active = GA.RUNS.get(rid)
        if active is None:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("failed resume was not finalized")

    final = store.run(rid)
    assert final["status"] == "failed"
    assert "intentional resume failure" in final["error"]
    assert not scratch.exists()
