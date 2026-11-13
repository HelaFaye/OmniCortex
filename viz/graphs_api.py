"""Studio's side of node graphs (scripts/graph_engine.py): saved graphs, runs in the background, approvals,
evals and running one node to debug it. Owner only; routes in studio.py, page in graphs_page.py."""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import graph_engine as G  # noqa: E402
import graph_templates  # noqa: E402

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,59}$")
RUNS: dict[str, dict] = {}          # id -> {"runner", "run", "kind", "result"} (waiting runs live here only)
LOCK = threading.Lock()
MAX_KEPT = 200


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def _path(self, name: str) -> Path:
        if not NAME.match(name or ""):
            raise ValueError("a graph's name: lowercase letters, digits, - and _")
        return self.root / f"{name}.json"

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json")) if self.root.is_dir() else []:
            try:
                g = json.loads(p.read_text())
                out.append({"name": p.stem, "description": g.get("description", ""), "nodes": len(g.get("nodes") or []),
                            "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
            except (OSError, ValueError):
                continue
        return out

    def get(self, name: str) -> dict:
        if name.startswith("template:"):
            t = graph_templates.TEMPLATES.get(name[9:])
            if t is None:
                raise FileNotFoundError(name)
            return json.loads(json.dumps(t))
        return json.loads(self._path(name).read_text())

    def save(self, g: dict) -> dict:
        p = self._path(str(g.get("name") or ""))
        G.validate(json.loads(json.dumps(g)))
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(g, indent=1))
        os.replace(tmp, p)
        return {"name": p.stem}

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    # runs: kept as JSON (trace and outputs); a run waiting on a person also lives in RUNS
    def save_run(self, rec: dict) -> None:
        d = self.root / "runs"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str))
        old = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in old[:-MAX_KEPT]:
            p.unlink(missing_ok=True)

    def runs(self, limit: int = 50) -> list[dict]:
        d = self.root / "runs"
        out = []
        for p in sorted(d.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit] if d.is_dir() else []:
            try:
                r = json.loads(p.read_text())
                out.append({k: r.get(k) for k in ("id", "graph", "status", "kind", "started", "error", "rate")})
            except (OSError, ValueError):
                continue
        return out

    def run(self, rid: str) -> dict:
        if not re.match(r"^[0-9a-f]{12}$", rid or ""):
            raise ValueError("bad run id")
        with LOCK:
            live = RUNS.get(rid)
            if live:
                return _public(live)
        return json.loads((self.root / "runs" / f"{rid}.json").read_text())


def _public(live: dict) -> dict:
    if live["kind"] == "eval":
        return {"id": live["id"], "kind": "eval", "graph": live["graph"], "status": live["status"],
                "started": live["started"], **(live.get("result") or {}), "error": live.get("error")}
    r = G.public_run(live["run"]) if live.get("run") else {"id": live["id"], "status": live["status"]}
    # live.status is the synchronized lifecycle state. The run object is a
    # completed snapshot; during approval resume it must not dictate status.
    r["status"] = live.get("status") or r.get("status", "running")
    return {**r, "kind": "run", "id": live["id"], "error": live.get("error") or r.get("error")}


def _finish(store: Store, live: dict) -> None:
    # Keep the run registered until its owned scratch directory is removed.
    # A poller observing absence from RUNS can then rely on cleanup having ended.
    with LOCK:
        if RUNS.get(live["id"]) is not live:
            return
        rec = _public(live)
        terminal = rec.get("status") not in ("waiting", "queued", "running")
        try:
            store.save_run(rec)
        finally:
            if terminal:
                try:
                    env = live.get("env")
                    if env is not None:
                        env.close_scratch()
                finally:
                    RUNS.pop(live["id"], None)


def start(store: Store, graph: dict, inputs: dict, env: G.Env) -> str:
    """Run a graph in a thread; poll with Store.run(id)."""
    G.validate(json.loads(json.dumps(graph)))
    env.open_scratch()
    rid = uuid.uuid4().hex[:12]
    live = {"id": rid, "kind": "run", "graph": graph.get("name", ""), "status": "running",
            "started": time.time(), "env": env}
    with LOCK:
        RUNS[rid] = live

    def go():
        try:
            live["runner"] = G.Runner(graph, env)
            run = live["runner"].start(inputs)
            run["id"] = rid
            with LOCK:
                live["run"] = run
                live["status"] = run["status"]
        except Exception as e:                                             # noqa: BLE001
            with LOCK:
                live["status"], live["error"] = "failed", f"{type(e).__name__}: {e}"
        finally:
            _finish(store, live)

    try:
        threading.Thread(target=go, daemon=True).start()
    except Exception:
        with LOCK:
            RUNS.pop(rid, None)
        env.close_scratch()
        raise
    return rid


def decide(store: Store, rid: str, node: str, approve: bool, value=None, note: str = "") -> dict:
    with LOCK:
        live = RUNS.get(rid)
        if not live or live.get("status") != "waiting":
            raise ValueError("that run is not waiting (a run waiting on a person does not survive a Studio restart)")
        live["status"] = "running"

    def go():
        try:
            # Runner.resume mutates its run argument in place. Resume a copy so
            # concurrent polling/finalization never observes a half-mutated snapshot.
            resume_state = copy.deepcopy(live["run"], {id(G.SKIP): G.SKIP})
            run = live["runner"].resume(resume_state, node, approve, value, note)
            with LOCK:
                live["run"] = run
                live["status"] = run["status"]
        except Exception as e:                                             # noqa: BLE001
            with LOCK:
                live["status"], live["error"] = "failed", f"{type(e).__name__}: {e}"
        finally:
            _finish(store, live)

    try:
        threading.Thread(target=go, daemon=True).start()
    except Exception:
        with LOCK:
            if RUNS.get(rid) is live:
                live["status"] = "waiting"
        raise
    return {"id": rid, "status": "running"}


def start_eval(store: Store, graph: dict, cases: list, env: G.Env, expect="", output="result", grade="contains") -> str:
    G.validate(json.loads(json.dumps(graph)))
    if not cases or len(cases) > 2000:
        raise ValueError("1 to 2000 cases")
    env.open_scratch()
    rid = uuid.uuid4().hex[:12]
    live = {"id": rid, "kind": "eval", "graph": graph.get("name", ""), "status": "running",
            "started": time.time(), "env": env}
    with LOCK:
        RUNS[rid] = live

    def go():
        try:
            result = G.run_eval(graph, cases, env, expect, output, grade)
            with LOCK:
                if RUNS.get(rid) is live:
                    live["result"] = result
                    live["status"] = "done"
        except Exception as e:                                             # noqa: BLE001
            with LOCK:
                if RUNS.get(rid) is live:
                    live["status"], live["error"] = "failed", f"{type(e).__name__}: {e}"
        finally:
            _finish(store, live)

    try:
        threading.Thread(target=go, daemon=True).start()
    except Exception:
        with LOCK:
            RUNS.pop(rid, None)
        env.close_scratch()
        raise
    return rid


def run_node(graph: dict, node_id: str, inputs: dict, env: G.Env) -> dict:
    """Debug one node: run it alone with the inputs given (as the editor's 'run this node')."""
    by = {n["id"]: n for n in graph.get("nodes") or []}
    if node_id not in by:
        raise ValueError(f"no node {node_id}")
    nd = json.loads(json.dumps(by[node_id]))
    spec = G.TYPES.get(nd.get("type"))
    if spec is None:
        raise ValueError(f"unknown node type {nd.get('type')}")
    g = {"name": "debug", "nodes": [nd], "edges": []}
    run = {"inputs": {}, "outputs": {}, "state": {}, "decisions": {}, "_trigger": {}, "depth": 0}
    ins = {x["name"]: inputs.get(x["name"]) for x in spec["inputs"]}
    G.validate(g)
    t0 = time.time()
    env.open_scratch()
    try:
        try:
            out = spec["run"](nd, ins, env, run) or {}
            return {"ok": True, "ms": int((time.time() - t0) * 1000),
                    "outputs": {k: ("(skipped)" if v is G.SKIP else G._preview(v, 20000)) for k, v in out.items()
                                if k in {o["name"] for o in spec["outputs"]}}}
        except G.Pause:
            return {"ok": True, "outputs": {}, "note": "an approval node waits for a person in a real run"}
        except Exception as e:                                             # noqa: BLE001
            return {"ok": False, "ms": int((time.time() - t0) * 1000), "error": f"{type(e).__name__}: {e}"}
    finally:
        env.close_scratch()


def catalog(store: Store) -> dict:
    return {"types": G.types_public(), "groups": G.GROUPS, "graphs": store.list(),
            "templates": [{"name": k, "description": t.get("description", "")} for k, t in graph_templates.TEMPLATES.items()],
            "runs": store.runs(20)}
