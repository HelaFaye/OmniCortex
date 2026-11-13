"""Workers for the director, and how work is spread across them.

A worker is anywhere a task can run:

  local     a model in Studio's model folders, started on this computer's
            devices when a task needs it (Studio's worker pool)
  endpoint  an OpenAI-compatible server you started yourself, here or on
            another computer (llama-server, vLLM, LM Studio, Ollama...)
  remote    a model on a computer that runs OmniCortex Remote, or on a
            linked Studio ("host:model")

Worker ids: a local model's id; "ep:<name>" for an endpoint; "<host>:<model>"
for a remote one.

The registry keeps, in workers.json:
  endpoints  name, URL, model, token, note, max parallel
  controls   per worker: paused, draining, max parallel, subjects it takes
  routes     per subject: an ordered list of workers to try first

and, in memory, what each worker is doing: tasks in flight, done, failed,
time per task, tokens per second, the last error, and an event log.

Routing a task: the route of its first subject that names a worker with a
free slot wins; otherwise the director's own choice (measured results, then
the default model, then the largest that fits) among the workers that take
the task's subjects and are not paused.
"""
from __future__ import annotations

import collections
import json
import os
import re
import threading
import time
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


class RegistryError(ValueError):
    pass


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.lock = threading.RLock()
        self.inflight: dict = collections.Counter()
        self.stats: dict = collections.defaultdict(lambda: {"done": 0, "failed": 0, "seconds": 0.0, "tokens": 0,
                                                            "last_error": None, "last_at": None})
        self.events: collections.deque = collections.deque(maxlen=300)
        self.health: dict = {}            # worker id -> (time, ok, detail)

    # -- the file
    def load(self) -> dict:
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            d = {}
        return {"endpoints": d.get("endpoints") or [], "controls": d.get("controls") or {},
                "routes": d.get("routes") or {}}

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.chmod(tmp, 0o600)                # endpoint tokens live here
        tmp.replace(self.path)

    # -- endpoints
    def endpoints(self) -> list[dict]:
        return self.load()["endpoints"]

    def endpoint(self, wid: str) -> dict | None:
        name = wid[3:] if wid.startswith("ep:") else wid
        return next((e for e in self.endpoints() if e["name"] == name), None)

    def put_endpoint(self, rec: dict) -> dict:
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip().rstrip("/")
        if not NAME.match(name):
            raise RegistryError("name: letters, digits, . _ - (up to 40)")
        if not re.match(r"https?://[^\s/]+", url):
            raise RegistryError("url: http(s)://host:port/v1")
        if not url.endswith("/v1"):
            url += "/v1"
        out = {"name": name, "url": url, "model": str(rec.get("model") or "").strip()[:200],
               "note": str(rec.get("note") or "")[:300], "max_parallel": max(1, int(rec.get("max_parallel") or 1))}
        with self.lock:
            d = self.load()
            old = next((e for e in d["endpoints"] if e["name"] == name), None)
            tok = rec.get("token")
            out["token"] = (old or {}).get("token", "") if tok is None else str(tok)
            d["endpoints"] = [e for e in d["endpoints"] if e["name"] != name] + [out]
            self._save(d)
        self.event("ep:" + name, "endpoint saved", url)
        return {k: v for k, v in out.items() if k != "token"} | {"has_token": bool(out["token"])}

    def remove_endpoint(self, name: str) -> None:
        with self.lock:
            d = self.load()
            d["endpoints"] = [e for e in d["endpoints"] if e["name"] != name]
            d["controls"].pop("ep:" + name, None)
            for s, ids in d["routes"].items():
                d["routes"][s] = [i for i in ids if i != "ep:" + name]
            self._save(d)
        self.event("ep:" + name, "endpoint removed")

    # -- controls
    def control(self, wid: str) -> dict:
        c = self.load()["controls"].get(wid) or {}
        return {"paused": bool(c.get("paused")), "draining": bool(c.get("draining")),
                "max_parallel": c.get("max_parallel"), "takes": list(c.get("takes") or [])}

    def set_control(self, wid: str, **kw) -> dict:
        with self.lock:
            d = self.load()
            c = dict(d["controls"].get(wid) or {})
            for k in ("paused", "draining"):
                if k in kw and kw[k] is not None:
                    c[k] = bool(kw[k])
            if kw.get("max_parallel") is not None:
                c["max_parallel"] = max(1, int(kw["max_parallel"]))
            if kw.get("takes") is not None:
                c["takes"] = [str(s) for s in kw["takes"]]
            if c.get("draining") and self.inflight[wid] == 0:
                c["draining"], c["paused"] = False, True      # nothing to finish: drained at once
            d["controls"][wid] = c
            self._save(d)
        self.event(wid, "settings changed", ", ".join(f"{k}={v}" for k, v in kw.items() if v is not None))
        return self.control(wid)

    def set_routes(self, routes: dict, subjects: list) -> dict:
        clean = {}
        for s, ids in (routes or {}).items():
            if s not in subjects:
                raise RegistryError(f"unknown subject {s!r}")
            ids = [str(i) for i in ids or [] if str(i)]
            if ids:
                clean[s] = list(dict.fromkeys(ids))
        with self.lock:
            d = self.load()
            d["routes"] = clean
            self._save(d)
        self.event("-", "routes changed", "; ".join(f"{s}: {' > '.join(v)}" for s, v in clean.items()) or "none")
        return clean

    # -- live state
    def event(self, wid: str, what: str, detail: str = "") -> None:
        self.events.append({"t": time.time(), "worker": wid, "what": what, "detail": str(detail)[:300]})

    def limit(self, wid: str) -> int | None:
        """Tasks this worker may run at once; None: no limit of its own (the project's applies)."""
        c = self.control(wid)
        if c["max_parallel"]:
            return int(c["max_parallel"])
        ep = self.endpoint(wid) if wid.startswith("ep:") else None
        return int(ep.get("max_parallel") or 1) if ep else None

    def has_slot(self, wid: str) -> bool:
        c = self.control(wid)
        if c["paused"] or c["draining"]:
            return False
        lim = self.limit(wid)
        with self.lock:
            return lim is None or self.inflight[wid] < lim

    def begin(self, wid: str, task: str = "") -> float:
        with self.lock:
            self.inflight[wid] += 1
        self.event(wid, "started", task)
        return time.time()

    def end(self, wid: str, t0: float, task: str = "", error: str | None = None, tokens: int = 0) -> None:
        with self.lock:
            self.inflight[wid] = max(0, self.inflight[wid] - 1)
            s = self.stats[wid]
            dt = time.time() - t0
            s["last_at"] = time.time()
            if error:
                s["failed"] += 1
                s["last_error"] = str(error)[:300]
            else:
                s["done"] += 1
                s["seconds"] += dt
                s["tokens"] += int(tokens or 0)
            finished_draining = self.inflight[wid] == 0 and self.control(wid)["draining"]
        self.event(wid, "failed" if error else "finished", f"{task} in {dt:.1f}s" + (f": {error}"[:200] if error else ""))
        if finished_draining:
            self.set_control(wid, draining=False, paused=True)
            self.event(wid, "drained", "no tasks left; paused")

    def summary(self, wid: str) -> dict:
        with self.lock:
            s = dict(self.stats[wid])
            n = s["done"]
            return {"inflight": self.inflight[wid], "done": n, "failed": s["failed"],
                    "avg_s": round(s["seconds"] / n, 1) if n else None,
                    "tok_s": round(s["tokens"] / s["seconds"], 1) if s["seconds"] and s["tokens"] else None,
                    "last_error": s["last_error"], "last_at": s["last_at"]}

    # -- routing
    def takes(self, wid: str, labels: list) -> bool:
        t = self.control(wid)["takes"]
        return not t or not labels or bool(set(t) & set(labels))

    def route(self, labels: list, alive=lambda wid: True) -> dict | None:
        """The first worker on the route of the task's subjects that is free. -> {model, reason} or None."""
        routes = self.load()["routes"]
        for s in labels or []:
            for wid in routes.get(s) or []:
                c = self.control(wid)
                if c["paused"] or c["draining"] or not alive(wid):
                    continue
                return {"model": wid, "reason": f"route for {s}"}
        return None
