#!/usr/bin/env python3
"""
Sweep a looped model's loop count (docs/DORY.md): what each extra pass through the recurrent block buys.

A looped (latent-recurrent) model such as NVIDIA's Dory runs one block of layers several times per token; the count
is set at load time (llama.cpp: --override-kv dory.num_loops=int:N) without retraining. That makes "how much thinking
per token" a dial, and this measures the dial: for each loop count (and KV cache mode) it starts llama-server, runs the
same tasks, and reports accuracy, confident mistakes, declines and speed, plus the paired changes against the first
setting (gained, regressed, McNemar). The question it answers: do more loops mostly fix wrong answers, or do they
also make the model more confidently wrong somewhere?

    python scripts/loop_sweep.py --gguf ~/models/dory-q5_k_m.gguf --server ~/llama.cpp-dory/build/bin/llama-server \\
        --tasks qa/bank --loops 1 2 3 4 --kv-mode per_loop --out runs/dory-loops.json

--tasks takes a TestQA bank folder or a JSONL of TestQA tasks (qa and reasoning items; graded by TestQA's grader).
Reasoning stays on (Dory's recipe); answers are taken at temperature 0 so settings differ only in the loops.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_eval import ask, mcnemar  # noqa: E402
from sweep_experts import measure_rate, wait_healthy  # noqa: E402

VERDICTS = ("correct", "wrong", "abstained", "refused")


def arch_of(gguf_path: str) -> str:
    try:
        import gguf
        import gguf_utils
        return gguf_utils.read_kv(gguf.GGUFReader(gguf_path), "general.architecture") or "dory"
    except Exception:
        return "dory"


def settings(loops: list[int], modes: list[str]) -> list[tuple[str, int, str]]:
    return [(f"loops={n}" + (f",{m}" if len(modes) > 1 else ""), n, m) for n in loops for m in modes]


def command(server, gguf_path, arch, n, mode, port, ngl, ctx, extra) -> list[str]:
    return [server, "-m", gguf_path, "-ngl", str(ngl), "-c", str(ctx), "--port", str(port), "--host", "127.0.0.1",
            "--jinja", "--reasoning", "on", "--temp", "1.0", "--top-p", "0.95",
            "--override-kv", f"{arch}.num_loops=int:{n},{arch}.recurrent_kv_cache_mode=str:{mode}", *extra]


def run(url: str, tasks: list[dict], max_tokens: int, concurrency: int) -> list[dict]:
    """Every task through the server, graded by TestQA's own grader."""
    from concurrent.futures import ThreadPoolExecutor
    import testqa
    gargs = argparse.Namespace(allow_exec=False, exec_timeout=5)

    def one(t):
        try:
            text = ask(url, t, None, max_tokens)
        except Exception as e:
            return {"id": t["id"], "verdict": "error", "note": str(e)[:160]}
        v = testqa.grade(t, text, gargs)[0]
        return {"id": t["id"], "verdict": "correct" if v in ("correct", "answered") else v, "text": text[-400:]}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, tasks))


def paired(ref: list[dict], var: list[dict]) -> tuple[list[str], list[str]]:
    """Items that became correct, and items that stopped being correct."""
    r = {x["id"]: x["verdict"] for x in ref}
    gained, lost = [], []
    for x in var:
        a, b = r.get(x["id"]), x["verdict"]
        if a is None or "error" in (a, b) or (a == "correct") == (b == "correct"):
            continue
        (gained if b == "correct" else lost).append(x["id"])
    return gained, lost


def summary(rows: list[dict]) -> dict:
    c = {v: 0 for v in VERDICTS}
    for r in rows:
        c[r["verdict"]] = c.get(r["verdict"], 0) + 1
    n = max(1, len(rows))
    return {**c, "n": len(rows), "accuracy": round(c["correct"] / n, 4), "confident_wrong": round(c["wrong"] / n, 4)}


def load_tasks(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        import testqa
        return [t for t in testqa.load_tasks([str(p)], {"qa", "reasoning"}, None, 0)]
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    import testqa
    for t in rows:
        t["kind"] = testqa.kind_of(t)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--server", required=True, help="llama-server built with the model's support")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--kv-mode", nargs="+", default=["per_loop"], choices=["per_loop", "last_loop"])
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    arch = arch_of(a.gguf)
    tasks = load_tasks(a.tasks)
    print(f"{arch}: {len(tasks)} tasks; settings: " + ", ".join(s[0] for s in settings(a.loops, a.kv_mode)))

    results, rates = {}, {}
    for label, n, mode in settings(a.loops, a.kv_mode):
        cmd = command(a.server, a.gguf, arch, n, mode, a.port, a.ngl, a.ctx, a.extra)
        print(f"\n--- {label} ---\n" + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not wait_healthy(a.port, proc, a.startup_timeout):
                print("  did not start; skipped")
                continue
            rates[label] = measure_rate(a.port, 128)
            results[label] = run(f"http://127.0.0.1:{a.port}", tasks, a.max_tokens, a.concurrency)
            s = summary(results[label])
            print(f"  {s['accuracy']:.1%} correct, {s['confident_wrong']:.1%} confidently wrong, "
                  f"{s['abstained']} declined" + (f", {rates[label]:.1f} tok/s" if rates[label] else ""), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(2)
    if not results:
        raise SystemExit("no setting started")
    ref = next(iter(results))
    out = {"gguf": a.gguf, "arch": arch, "reference": ref, "rates": rates,
           "summary": {k: summary(v) for k, v in results.items()}, "comparisons": []}
    for label, rows in results.items():
        if label == ref:
            continue
        g, l = paired(results[ref], rows)
        chi2, disc = mcnemar(len(g), len(l))
        out["comparisons"].append({"setting": label, "gained": len(g), "regressed": len(l), "chi2": round(chi2, 3),
                                   "verdict": ("significant" if chi2 > 3.84 else "not significant") if disc >= 10
                                   else "too few changes to call"})
        print(f"{label} vs {ref}: gained {len(g)}, regressed {len(l)} ({out['comparisons'][-1]['verdict']})")
    if a.out:
        Path(a.out).write_text(json.dumps({**out, "raw": results}, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
