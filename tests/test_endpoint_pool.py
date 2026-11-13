"""Per-server concurrency and pooled servers (scripts/endpoint_pool.py), and TestQA's use of them."""
import json
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import endpoint_pool as EP  # noqa: E402
import testqa as tq  # noqa: E402


def server_like(capacity: int, per_request: float):
    """fn(member, item) for a fake server that runs `capacity` requests at once, each per_request seconds;
    past that, requests queue (so more in flight stops helping). Records the most in flight seen."""
    sem, seen, lock = threading.Semaphore(capacity), {"now": 0, "max": 0}, threading.Lock()

    def fn(m, x):
        with lock:
            seen["now"] += 1
            seen["max"] = max(seen["max"], seen["now"])
        with sem:
            time.sleep(per_request)
        with lock:
            seen["now"] -= 1
        return x, True, None
    return fn, seen


def test_fixed_limit_is_respected():
    fn, seen = server_like(10, 0.02)
    lim = EP.Limiter("s", 3)
    out = EP.run_pool([{"name": "s", "limiter": lim}], list(range(30)), fn)
    assert out == list(range(30)) and seen["max"] == 3 and lim.summary()["done"] == 30


def test_auto_grows_to_capacity_and_stops():
    fn, seen = server_like(3, 0.03)
    lim = EP.Limiter("s", "auto", cap=8)
    EP.run_pool([{"name": "s", "limiter": lim}], list(range(120)), fn)
    s = lim.summary()
    assert 3 <= s["limit"] <= 5, s          # past three slots throughput stops rising
    assert s["mode"] == "auto" and lim.frozen


def test_auto_keeps_a_slow_server_at_one():
    fn, seen = server_like(8, 0.05)
    lim = EP.Limiter("s", "auto", cap=8, slow_s=0.01)
    EP.run_pool([{"name": "s", "limiter": lim}], list(range(10)), fn)
    assert lim.limit == 1 and seen["max"] == 1 and "one at a time" in lim.why


def test_auto_backs_off_on_errors():
    lim = EP.Limiter("s", "auto", cap=8)
    lim.limit = 6
    lim.acquire()
    lim.release(ok=False, seconds=1)
    assert lim.limit == 3 and lim.frozen


def test_pool_gives_more_to_the_fast_server():
    fast_fn, _ = server_like(4, 0.01)
    slow_fn, _ = server_like(1, 0.12)
    fns = {"fast": fast_fn, "slow": slow_fn}
    members = [{"name": "fast", "limiter": EP.Limiter("fast", 4)}, {"name": "slow", "limiter": EP.Limiter("slow", 1)}]
    who = Counter()
    out = EP.run_pool(members, list(range(60)), lambda m, x: fns[m["name"]](m, x),
                      on_done=lambda x, r, m: who.update([m["name"]]))
    assert sorted(out) == list(range(60)) and sum(who.values()) == 60
    assert who["fast"] > 5 * who["slow"] and who["slow"] >= 1


def test_pool_moves_work_off_a_failing_server():
    ok_fn, _ = server_like(2, 0.005)

    def fn(m, x):
        if m["name"] == "down":
            time.sleep(0.002)
            return {"error": x}, False, None
        return ok_fn(m, x)
    members = [{"name": "down", "limiter": EP.Limiter("down", 2)}, {"name": "up", "limiter": EP.Limiter("up", 2)}]
    out = EP.run_pool(members, list(range(20)), fn)
    assert out == list(range(20))            # every item answered by the server that works


def test_servers_share_a_limiter_and_read_slots():
    sv = EP.Servers("auto", cap=8, probe=lambda url: 2)
    a, b = sv.get("http://x/v1"), sv.get("http://x/v1")
    assert a is b and a.cap == 2 and "2 slot" in a.why
    fixed = sv.get("http://y/v1", 1)
    assert fixed.limit == 1 and not fixed.auto
    assert {s["server"] for s in sv.summary()} == {"http://x/v1", "http://y/v1"}


class Echo(BaseHTTPRequestHandler):
    delay = 0.0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(type(self).delay)
        out = json.dumps({"choices": [{"message": {"content": "Answer: 0"}}],
                          "usage": {"completion_tokens": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = json.dumps({"total_slots": 2}).encode()
        self.send_response(200 if self.path == "/props" else 404)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_testqa_pool_and_parallel(tmp_path):
    class Slow(Echo):
        delay = 0.15
    fast = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    slow = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    for s in (fast, slow):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    out = tmp_path / "r.json"
    try:
        assert tq.main(["--endpoint", f"m=http://127.0.0.1:{fast.server_port}@m",
                        "--pool", f"m=http://127.0.0.1:{slow.server_port}@m", "--parallel", "m=auto",
                        "--endpoint", f"one=http://127.0.0.1:{slow.server_port}@m", "--parallel", "one=1",
                        "--kind", "reasoning", "--limit", "30", "--out", str(out)]) == 0
    finally:
        fast.shutdown()
        slow.shutdown()
    rep = json.loads(out.read_text())
    took = Counter(r["served_by"] for r in rep["raw"]["m"])
    fast_name = f"http://127.0.0.1:{fast.server_port}/v1 @ m"
    assert sum(took.values()) == 30 and took[fast_name] >= 20       # the slow server is also busy with "one"
    assert len(rep["endpoints"]["m"]["pool"]) == 2 and "served_by" not in rep["raw"]["one"][0]
    sched = {s["server"]: s for s in rep["scheduling"]}
    slow_s = sched[f"http://127.0.0.1:{slow.server_port}/v1"]
    assert slow_s["limit"] == 1                          # "one=1" pins the shared slow server
    assert sched[f"http://127.0.0.1:{fast.server_port}/v1"]["limit"] <= 2   # its /props says two slots
