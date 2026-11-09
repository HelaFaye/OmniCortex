#!/usr/bin/env python3
"""
Boss/minion delegation, gated on measured hallucination.

Any orchestrator can route subtasks to cheaper models. What this adds is the
part only this project can do: before the boss accepts a minion's answer, the
answer is scored by the H-Neuron classifier, and a high score sends it back.
Delegation you have evidence for, rather than delegation you hope worked.

    python scripts/delegate.py --config delegation.json --task "..."
    python scripts/delegate.py --config delegation.json --serve 8100

The config names endpoints; nothing here constructs a command line, so a
config file cannot become a remote shell:

    {
      "boss":    {"url": "http://GPU-HOST:1234/v1", "model": "qwen3-8b"},
      "minions": [
        {"name": "local",  "url": "http://127.0.0.1:8080/v1", "model": "minion",
         "good_at": ["summarise", "extract", "rewrite"]},
        {"name": "remote", "url": "http://GPU-HOST:1234/v1",
         "model": "qwen3-8b", "good_at": ["reason", "code"]}
      ],
      "gate": {"trace_endpoint": "http://127.0.0.1:8088",
               "classifier": "models_1v1/classifier.npz", "max_score": 1.5,
               "retries": 1}
    }

The gate is optional and OFF unless `classifier` is set, because a gate that
cannot actually score anything would be theatre.
"""

import argparse
import json
import sys
import time
import urllib.request

CFG = {}


def chat(ep, messages, max_tokens=2048, temperature=0.4):
    body = json.dumps({"model": ep["model"], "messages": messages,
                       "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(ep["url"].rstrip("/") + "/chat/completions",
                                 data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def plan(task, minions):
    """Ask the boss to split the task. Falls back to one step on bad JSON --
    a malformed plan should degrade to 'do it yourself', not crash."""
    names = ", ".join(f'{m["name"]} (good at: {", ".join(m.get("good_at", []))})'
                      for m in minions)
    sys_p = (
        "Split the task into at most 5 independent subtasks and assign each to "
        f"a worker. Workers: {names}. Reply with JSON only: "
        '{"steps":[{"worker":"<name>","instruction":"<self-contained>"}]}. '
        "Each instruction must stand alone: the worker sees only that text, "
        "not the original task or the other steps.")
    raw = chat(CFG["boss"], [{"role": "system", "content": sys_p},
                             {"role": "user", "content": task}],
               max_tokens=1200, temperature=0.2)
    body = raw
    if "```" in body:
        body = body.split("```")[1].lstrip("json").strip()
    if "{" in body:
        body = body[body.index("{"):body.rindex("}") + 1]
    try:
        steps = json.loads(body).get("steps", [])
    except Exception as e:
        print(f"  plan unparseable ({e}); doing it in one step", file=sys.stderr)
        return [{"worker": minions[0]["name"], "instruction": task}]
    valid = {m["name"] for m in minions}
    return [s for s in steps
            if s.get("worker") in valid and s.get("instruction")] or \
           [{"worker": minions[0]["name"], "instruction": task}]


def score(instruction, answer):
    """-> (score, reason). None means no gate is configured.

    Routes through scripts/autotrace.py, which traces a completed response and
    scores it with the classifier. The gate is only as good as that classifier:
    at AUROC ~0.69 it is a weak signal, so `max_score` should be set loose
    enough to catch the obvious cases rather than to arbitrate close ones.
    """
    g = CFG.get("gate") or {}
    if not g.get("classifier") or not g.get("trace_endpoint"):
        return None, "no gate configured"
    try:
        body = json.dumps({"messages": [{"role": "user", "content": instruction}],
                           "response": answer}).encode()
        req = urllib.request.Request(
            g["trace_endpoint"].rstrip("/") + "/api/score", data=body,
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return float(d["score"]), d.get("detail", "")
    except Exception as e:
        # A gate that fails closed would stall the whole run on a flaky
        # endpoint; a gate that fails open silently would be a lie. Say so.
        return None, f"gate unreachable: {type(e).__name__}"


def run_step(step, minions, verbose=True):
    m = next(x for x in minions if x["name"] == step["worker"])
    g = CFG.get("gate") or {}
    tries = int(g.get("retries", 1)) + 1
    limit = float(g.get("max_score", 1.5))
    attempts = []
    for i in range(tries):
        extra = ("\n\nYour previous answer was flagged as likely fabricated. "
                 "Answer only what you can support; say what you do not know."
                 if i else "")
        out = chat(m, [{"role": "user",
                        "content": step["instruction"] + extra}])
        s, why = score(step["instruction"], out)
        attempts.append({"attempt": i + 1, "answer": out, "score": s,
                         "note": why})
        if verbose:
            tag = "ungated" if s is None else f"score {s:+.2f}"
            print(f"    [{m['name']}] attempt {i+1}: {tag}"
                  + ("" if s is None else
                     (" FLAGGED, retrying" if s > limit and i + 1 < tries
                      else " FLAGGED, kept anyway" if s > limit else " ok")))
        if s is None or s <= limit:
            break
    best = min(attempts, key=lambda a: (a["score"] is None, a["score"] or 0))
    return {"worker": m["name"], "instruction": step["instruction"],
            "answer": best["answer"], "score": best["score"],
            "flagged": best["score"] is not None and best["score"] > limit,
            "attempts": attempts}


def delegate(task, verbose=True):
    minions = CFG["minions"]
    t0 = time.time()
    if verbose:
        print(f"boss: {CFG['boss']['model']}  minions: "
              f"{', '.join(m['name'] for m in minions)}")
    steps = plan(task, minions)
    if verbose:
        print(f"\nplan: {len(steps)} step(s)")
        for s in steps:
            print(f"  -> {s['worker']}: {s['instruction'][:74]}")
    results = [run_step(s, minions, verbose) for s in steps]

    parts = []
    for r in results:
        mark = "  [FLAGGED as likely fabricated]" if r["flagged"] else ""
        parts.append(f"### {r['instruction']}{mark}\n{r['answer']}")
    final = chat(CFG["boss"], [
        {"role": "system", "content":
            "Combine the worker results into one answer for the original task. "
            "Anything marked FLAGGED was scored as likely fabricated -- do not "
            "repeat its claims as fact; say what is uncertain."},
        {"role": "user", "content":
            f"Task: {task}\n\n" + "\n\n".join(parts)}])

    flagged = sum(1 for r in results if r["flagged"])
    if verbose:
        print(f"\n{len(results)} step(s), {flagged} flagged, "
              f"{time.time() - t0:.0f}s")
    return {"task": task, "steps": results, "answer": final,
            "flagged": flagged, "secs": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task")
    p.add_argument("--serve", type=int, metavar="PORT")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    with open(a.config) as f:
        CFG.update(json.load(f))
    for k in ("boss", "minions"):
        if k not in CFG:
            raise SystemExit(f"config needs a {k!r} entry")
    if not (CFG.get("gate") or {}).get("classifier"):
        print("note: no classifier in gate -- results are NOT scored. "
              "Delegation without the gate is just routing.\n", file=sys.stderr)

    if a.serve:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *x):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                msgs = req.get("messages", [])
                task = msgs[-1]["content"] if msgs else req.get("task", "")
                out = delegate(task, verbose=True)
                # OpenAI-shaped, so Cline or anything else can point at it.
                body = json.dumps({
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": out["answer"]}}],
                    "x_omnicortex": {"steps": len(out["steps"]),
                                      "flagged": out["flagged"]},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        print(f"delegating endpoint on http://127.0.0.1:{a.serve}/v1/chat/completions")
        ThreadingHTTPServer(("127.0.0.1", a.serve), H).serve_forever()
        return

    if not a.task:
        raise SystemExit("pass --task or --serve")
    out = delegate(a.task)
    print(json.dumps(out, indent=2) if a.json else "\n" + out["answer"])


if __name__ == "__main__":
    main()
