#!/usr/bin/env python3
"""
Spread requests over OpenAI-compatible servers, each at its own concurrency (docs/TESTQA.md#concurrency).

- Each server gets a Limiter: a fixed number of requests in flight, or "auto". Auto starts at one, adds a
  slot while throughput rises by more than 15%, gives one back when it falls, and halves on an error. A
  server whose single requests take longer than `slow_s` stays at one. llama-server's /props `total_slots`
  caps it, since requests past that just queue inside the server.
- Labels that name the same server share its Limiter, so two runs against one GPU do not double its load.
- run_pool() hands items from one queue to every member of a pool: whichever server frees a slot first
  takes the next item, so fast servers do more of the work. A failed item goes back to the queue once per
  other member; a member that fails three times in a row is dropped from the pool.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request

AUTO_MAX = 8


class Limiter:
    def __init__(self, name: str, limit: int | str = "auto", cap: int = AUTO_MAX, slow_s: float = 120.0):
        self.name = name
        self.auto = limit == "auto"
        self.cap = max(1, int(cap if self.auto else limit))
        self.limit = 1 if self.auto else self.cap
        self.slow_s = slow_s
        self.cond = threading.Condition()
        self.inflight = 0
        self.frozen = not self.auto
        self.why = "fixed" if not self.auto else "probing"
        self.best = 0.0
        self.win = {"t0": None, "n": 0, "work": 0.0}
        self.stats = {"done": 0, "failed": 0, "seconds": 0.0, "tokens": 0, "peak": 0}

    def set_cap(self, cap: int, why: str) -> None:
        with self.cond:
            if cap < self.cap:
                self.cap = max(1, cap)
                self.limit = min(self.limit, self.cap)
                if not self.auto:
                    self.limit = self.cap
                self.why = why

    def acquire(self) -> None:
        with self.cond:
            while self.inflight >= self.limit:
                self.cond.wait()
            self.inflight += 1
            if self.win["t0"] is None:
                self.win["t0"] = time.time()
            self.stats["peak"] = max(self.stats["peak"], self.inflight)

    def release(self, ok: bool = True, seconds: float = 0.0, tokens: int | None = None, counted: bool = True) -> None:
        with self.cond:
            self.inflight -= 1
            if counted:
                s = self.stats
                s["done" if ok else "failed"] += 1
                s["seconds"] += seconds
                s["tokens"] += tokens or 0
                if self.auto and not self.frozen:
                    self._adapt(ok, seconds, tokens)
            self.cond.notify_all()

    def _adapt(self, ok: bool, seconds: float, tokens: int | None) -> None:
        if not ok:
            self.limit, self.frozen, self.why = max(1, self.limit // 2), True, "an error: backed off"
            return
        if self.limit == 1 and seconds > self.slow_s:
            self.frozen, self.why = True, f"slow ({seconds:.0f} s a request): one at a time"
            return
        w = self.win
        w["n"] += 1
        w["work"] += tokens if tokens else 1
        if w["n"] < max(2, 2 * self.limit):
            return
        thr = w["work"] / max(1e-6, time.time() - w["t0"])
        self.win = {"t0": time.time(), "n": 0, "work": 0.0}
        if thr > self.best * 1.15 and self.limit < self.cap:
            self.best, self.limit, self.why = thr, self.limit + 1, "throughput still rising"
        elif thr < self.best * 0.9 and self.limit > 1:
            self.limit, self.frozen, self.why = self.limit - 1, True, "throughput fell: settled one lower"
        else:
            self.best = max(self.best, thr)
            self.frozen = True
            self.why = "at the server's slots" if self.limit >= self.cap else "throughput levelled off"

    def summary(self) -> dict:
        s = self.stats
        n = s["done"] + s["failed"]
        return {"server": self.name, "limit": self.limit, "mode": "auto" if self.auto else "fixed", "why": self.why,
                "peak_in_flight": s["peak"], "done": s["done"], "failed": s["failed"],
                "avg_s": round(s["seconds"] / n, 2) if n else None,
                "gen_tok_s": round(s["tokens"] / s["seconds"], 1) if s["seconds"] and s["tokens"] else None}


def server_slots(url: str, timeout: float = 5.0) -> int | None:
    """llama-server's parallel slots (GET /props, beside /v1), or None for servers that do not say."""
    base = url[:-3] if url.endswith("/v1") else url
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/props", timeout=timeout) as r:
            n = json.loads(r.read()).get("total_slots")
        return int(n) if n else None
    except Exception:
        return None


class Servers:
    """One Limiter per server URL, shared by every label that uses it."""

    def __init__(self, default: int | str = "auto", cap: int = AUTO_MAX, slow_s: float = 120.0, probe=server_slots):
        self.default, self.cap, self.slow_s, self.probe = default, cap, slow_s, probe
        self.by_url: dict[str, Limiter] = {}
        self.lock = threading.Lock()

    def get(self, url: str, limit: int | str | None = None) -> Limiter:
        with self.lock:
            lim = self.by_url.get(url)
            if lim is None:
                lim = self.by_url[url] = Limiter(url, limit or self.default, self.cap, self.slow_s)
                slots = self.probe(url) if self.probe else None
                if slots and lim.auto:
                    lim.set_cap(slots, f"the server has {slots} slot(s)")
            elif limit is not None and limit != "auto" and int(limit) < lim.cap:
                lim.set_cap(int(limit), "fixed")
            return lim

    def summary(self) -> list[dict]:
        return [lim.summary() for lim in self.by_url.values()]


def run_pool(members: list[dict], items: list, fn, on_done=None) -> list:
    """members: [{"name", "limiter", ...}]; fn(member, item) -> (result, ok, tokens). Returns the results in
    item order; on_done(item, result, member) after each."""
    q: queue.Queue = queue.Queue()
    for i, x in enumerate(items):
        q.put((i, x, 0))
    results: list = [None] * len(items)
    left = [len(items)]
    lock = threading.Lock()
    alive = {m["name"]: True for m in members}
    strikes = {m["name"]: 0 for m in members}

    def finish(i, x, r, m):
        with lock:
            results[i] = r
            left[0] -= 1
        if on_done:
            on_done(x, r, m)

    def worker(m):
        lim = m["limiter"]
        while alive[m["name"]]:
            lim.acquire()
            try:
                i, x, tries = q.get_nowait()
            except queue.Empty:
                lim.release(counted=False)
                with lock:
                    if left[0] <= 0:
                        return
                time.sleep(0.05)            # another member's failure may put an item back
                continue
            t0 = time.time()
            r, ok, tokens = fn(m, x)
            lim.release(ok, time.time() - t0, tokens)
            with lock:
                strikes[m["name"]] = 0 if ok else strikes[m["name"]] + 1
                if strikes[m["name"]] >= 3 and sum(alive.values()) > 1:
                    alive[m["name"]] = False
                others = sum(alive.values()) - (1 if alive[m["name"]] else 0)
            if not ok and tries < others:
                q.put((i, x, tries + 1))
                continue
            finish(i, x, r, m)
        # dropped from the pool: the last member standing takes what is left

    threads = []
    for m in members:
        for _ in range(m["limiter"].cap):
            t = threading.Thread(target=worker, args=(m,), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return results
