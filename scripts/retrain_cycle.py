#!/usr/bin/env python3
"""
One retraining cycle, start to verdict: measure, curate, train one or more
arms, convert, measure each on the holdout, and keep or reject each.

    python scripts/retrain_cycle.py --out runs/cycle1 \\
        --base-model org/my-model --base-gguf models/my-model-Q4_K_M.gguf \\
        --teacher http://gpu-host:1234/v1@strong-model \\
        --arms all,plan --review-model acme/my-model-q4_k_m --method lora --dpo --expand 3

Steps, each skipped when its output already exists (so an interrupted cycle
resumes where it stopped; --redo STEP forces one again):

  1. baseline   TestQA on the base model (served from --base-gguf by
                llama-server, or --base-endpoint), with replies cached
  2. data       deficits.py: verified targets, knowledge probing against the
                base model (--student), honest DPO pairs, the holdout, focus
                layers when --review-model is given
  3. per arm    finetune.py --layers <arm>, then merge_export.py to a GGUF
  4. per arm    TestQA on the holdout items, served by llama-server
  5. gate       retrain_gate.py: each arm against the baseline; with several
                arms, each later arm against the first too

Arms are layer choices: `all`, `plan` (the review's focus layers) or a list
like `10-17`. Two arms, `all,plan`, answer whether focusing on the layers the
neuron review points at beats training everywhere, on the same data.

The baseline and the arms are served the same way (llama-server, same
settings, temperature 0) when --base-gguf is given, so the comparison is not
confounded by serving. With only --base-endpoint, the baseline comes from that
server instead; the report says so.

Writes <out>/cycle.json (every step, command, and result) and report.txt.
The cycle recommends the best arm the gate keeps, and nothing when none is
kept. Nothing is installed or deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


def log(msg: str) -> None:
    print(f"[cycle] {msg}", flush=True)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def llama_dir(a) -> Path:
    return Path(a.llama or os.environ.get("OC_LLAMA") or "~/llama.cpp").expanduser()


def server_bin(a) -> str:
    cands = [a.llama_server, os.environ.get("OC_LLAMA_SERVER"), str(llama_dir(a) / "build" / "bin" / "llama-server"),
             shutil.which("llama-server")]
    for c in cands:
        if c and Path(c).expanduser().exists():
            return str(Path(c).expanduser())
    raise SystemExit("no llama-server: pass --llama-server, set OC_LLAMA_SERVER, or build llama.cpp "
                     "(Setup → Build llama.cpp)")


class Served:
    """llama-server on a free port for one GGUF, for as long as the block runs."""

    def __init__(self, a, gguf: str, alias: str, logdir: Path):
        self.a, self.gguf, self.alias, self.logdir = a, gguf, alias, logdir
        self.proc = None

    def __enter__(self) -> str:
        port = free_port()
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.logdir / f"server-{self.alias}.log", "w")
        cmd = [server_bin(self.a), "-m", self.gguf, "--host", "127.0.0.1", "--port", str(port), "--alias", self.alias,
               "-c", str(self.a.ctx), *self.a.server_arg]
        log("serving: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        end = time.time() + self.a.load_timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({self.proc.returncode}); see {self.logf.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return f"http://127.0.0.1:{port}/v1@{self.alias}"
            except Exception:
                pass
            time.sleep(0.5)
        raise SystemExit(f"llama-server did not become ready in {self.a.load_timeout}s; see {self.logf.name}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()


class Cycle:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "cycle.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}}
        self.state["args"] = {k: v for k, v in vars(a).items() if k != "api_key"}

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def done(self, step: str, *outputs: Path) -> bool:
        if step in self.a.redo or "all" in self.a.redo:
            return False
        return self.state["steps"].get(step, {}).get("status") == "done" and all(p.exists() for p in outputs)

    def run(self, step: str, cmd: list[str]) -> None:
        log(f"{step}: " + " ".join(str(c) for c in cmd))
        rec = {"cmd": [str(c) for c in cmd], "started": time.time(), "status": "running"}
        self.state["steps"][step] = rec
        self.save()
        logp = self.out / "logs" / f"{step.replace('/', '-')}.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "w") as f:
            r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        rec.update(status="done" if r.returncode == 0 else "failed", code=r.returncode, log=str(logp),
                   seconds=round(time.time() - rec["started"], 1))
        self.save()
        if r.returncode != 0:
            tail = logp.read_text(errors="replace")[-1500:]
            raise SystemExit(f"{step} failed ({r.returncode}); log {logp}:\n{tail}")

    # ---------------------------------------------------------- steps
    def testqa(self, step: str, label: str, endpoint: str, out: Path, cache: Path | None, only: Path | None):
        a = self.a
        cmd = [sys.executable, HERE / "testqa.py", "--endpoint", f"{label}={endpoint}", "--out", out,
               "--seed", str(a.seed), "--max_tokens", str(a.max_tokens)]
        if cache:
            cmd += ["--cache", cache]
        if only:
            cmd += ["--only-ids", only]
        if a.tasks:
            cmd += ["--tasks", *a.tasks]
        if only is None:
            # the baseline samples; evaluations ask exactly the baseline's holdout items
            if a.per_subject:
                cmd += ["--per-subject", str(a.per_subject)]
            if a.subject:
                cmd += ["--subject", *a.subject]
        if a.allow_exec:
            cmd += ["--allow-exec", "--sandbox", a.sandbox]
        if a.api_key:
            cmd += ["--api-key", a.api_key]
        self.run(step, cmd)

    def check_answers(self, path: Path, label: str, what: str) -> None:
        rows = json.loads(path.read_text())["raw"][label]
        errs = [r for r in rows if r["verdict"] == "error"]
        if rows and len(errs) > len(rows) / 2:
            raise SystemExit(f"{what}: {len(errs)} of {len(rows)} requests failed, e.g. {errs[0].get('note', '')[:200]}. "
                             f"Server logs are in {self.out / 'logs'} (a model without a chat template needs "
                             "--server-arg=--chat-template --server-arg=chatml).")

    def baseline(self) -> str:
        """-> the endpoint the base model is reachable at for probing (may be served
        later again); writes base.json with the base model's answers."""
        out, cache = self.out / "base.json", self.out / "cache"
        if self.done("baseline", out):
            log("baseline: done already")
            return self.state.get("base_how", "")
        if self.a.base_gguf:
            with Served(self.a, self.a.base_gguf, "base", self.out / "logs") as ep:
                self.testqa("baseline", "base", ep, out, cache, None)
            self.state["base_how"] = "llama-server"
        else:
            self.testqa("baseline", "base", self.a.base_endpoint, out, cache, None)
            self.state["base_how"] = "endpoint"
        self.save()
        self.check_answers(out, "base", "baseline")
        return self.state["base_how"]

    def data(self):
        a = self.a
        d = self.out / "data"
        if self.done("data", d / "sft.jsonl", d / "plan.json"):
            log("data: done already")
            return
        cmd = [sys.executable, HERE / "deficits.py", "--results", self.out / "base.json", "--label", "base",
               "--cache", self.out / "cache", "--out", d, "--deficit-fraction", str(a.deficit_fraction),
               "--expand", str(a.expand), "--unknown", a.unknown, "--probe-samples", str(a.probe_samples),
               "--seed", str(a.seed)]
        if a.tasks:
            cmd += ["--tasks", *a.tasks]
        if a.teacher:
            cmd += ["--teacher", a.teacher]
        if a.allow_exec:
            cmd += ["--allow-exec", "--sandbox", a.sandbox]
        if a.review_model:
            cmd += ["--review-model", a.review_model]
            if a.review_dir:
                cmd += ["--review-dir", a.review_dir]
        for g in a.general or []:
            cmd += ["--general", g]
        if a.api_key:
            cmd += ["--api-key", a.api_key]
        if a.base_gguf:
            with Served(a, a.base_gguf, "base", self.out / "logs") as ep:
                self.run("data", cmd + ["--student", ep])
        elif a.base_endpoint:
            self.run("data", cmd + ["--student", a.base_endpoint])
        else:
            self.run("data", cmd)

    def eval_ids(self) -> Path:
        """The holdout items the baseline answered: what every arm is asked."""
        p = self.out / "data" / "eval_ids.json"
        hold = set(json.loads((self.out / "data" / "holdout_ids.json").read_text())["ids"])
        base = json.loads((self.out / "base.json").read_text())["raw"]["base"]
        ids = sorted(r["id"] for r in base if r["id"] in hold)
        p.write_text(json.dumps({"ids": ids}))
        return p

    def has_data(self) -> bool:
        plan = json.loads((self.out / "data" / "plan.json").read_text())
        n = plan.get("deficit_examples") or 0
        if n < self.a.min_examples:
            log(f"data: {n} deficit examples, fewer than --min-examples {self.a.min_examples}; not training. "
                "Test more items (--per-subject), add --expand with a --teacher, or see data/report.txt for "
                "what was skipped and why")
            self.state["stopped"] = f"{n} deficit examples"
            self.save()
            return False
        return True

    def arm(self, name: str) -> dict:
        a = self.a
        d = self.out / "arms" / safe(name)
        rec = self.state.setdefault("arms", {}).setdefault(name, {"layers": name})
        if name == "plan":
            plan = json.loads((self.out / "data" / "plan.json").read_text())
            fl = (plan.get("focus_layers") or {})
            if not fl.get("layers"):
                rec.update(status="skipped", why="no focus layers: " + (fl.get("note") or "pass --review-model"))
                self.save()
                log(f"arm {name}: skipped ({rec['why']})")
                return rec
            rec["focus_layers"] = fl["layers"]
        adapter, merged = d / "adapter", d / "merged"
        if not self.done(f"train/{name}", adapter / "omnicortex-finetune.json"):
            cmd = [sys.executable, HERE / "finetune.py", "--model", a.base_model, "--data", self.out / "data",
                   "--out", adapter, "--method", a.method, "--layers", name, "--seed", str(a.seed), *a.finetune_arg]
            if a.dpo:
                cmd.append("--dpo")
            self.run(f"train/{name}", cmd)
        meta = merged / "omnicortex-export.json"
        if not self.done(f"convert/{name}", meta):
            self.run(f"convert/{name}", [sys.executable, HERE / "merge_export.py", "--base", a.base_model,
                                         "--adapter", adapter, "--out", merged, "--gguf", a.quant,
                                         "--dtype", a.merge_dtype,
                                         "--llama", llama_dir(a)])
        rec["gguf"] = json.loads(meta.read_text())["gguf"]
        ev = d / "eval.json"
        if not self.done(f"eval/{name}", ev):
            with Served(a, rec["gguf"], safe(name), self.out / "logs") as ep:
                self.testqa(f"eval/{name}", safe(name), ep, ev, None, self.eval_ids())
            self.check_answers(ev, safe(name), f"arm {name}")
        rec.update(status="evaluated", eval=str(ev))
        self.save()
        return rec

    def gate(self, arms: list[str]) -> dict:
        import retrain_gate as rg
        holdout = set(json.loads((self.out / "data" / "holdout_ids.json").read_text())["ids"])
        _, base = rg.load(str(self.out / "base.json"), "base")
        results, verdicts = {}, {}
        for name in arms:
            rec = self.state["arms"][name]
            if rec.get("status") != "evaluated":
                continue
            _, results[name] = rg.load(rec["eval"], safe(name))
            g = rg.gate(base, results[name], holdout, self.a.alpha)
            g.update(before="base", after=name)
            (self.out / "arms" / safe(name) / "gate.json").write_text(json.dumps(g, indent=1))
            rec["verdict"], rec["reasons"] = g["verdict"], g["reasons"]
            rec["holdout"] = {k: g["holdout"][k] for k in ("n", "fixed", "broke", "net", "p", "accuracy_before",
                                                           "accuracy_after", "wrong_before", "wrong_after")}
            verdicts[name] = g
        ran = [n for n in arms if n in results]
        head = []
        if len(ran) > 1:
            first = ran[0]
            for other in ran[1:]:
                g = rg.gate(results[first], results[other], holdout, self.a.alpha)
                head.append({"before": first, "after": other, "verdict": g["verdict"], "reasons": g["reasons"],
                             "fixed": g["holdout"]["fixed"], "broke": g["holdout"]["broke"], "p": g["holdout"]["p"]})
        kept = [n for n in ran if verdicts[n]["verdict"] == "keep"]
        best = max(kept, key=lambda n: (verdicts[n]["holdout"]["net"], -verdicts[n]["holdout"]["p"])) if kept else None
        self.state.update(head_to_head=head, recommended=best)
        self.save()
        return {"recommended": best, "head_to_head": head}

    def report(self) -> str:
        st = self.state
        lines = [f"Retraining cycle in {self.out}", ""]
        if st.get("stopped"):
            lines.append(f"stopped before training: {st['stopped']} (see data/report.txt)")
        if st.get("base_how") == "endpoint":
            lines.append("note: the baseline came from --base-endpoint, the arms from llama-server; serving "
                         "differences can show up as changes. Give --base-gguf for a like-for-like comparison.")
        try:
            plan = json.loads((self.out / "data" / "plan.json").read_text())
            lines.append(f"data: {plan['deficit_examples']} deficit examples, {plan['replay_examples']} replay, "
                         f"{plan['dpo_pairs']} DPO pairs; holdout: {plan['holdout_items']} bank items, "
                         f"{len(json.loads((self.out / 'data' / 'eval_ids.json').read_text())['ids']) if (self.out / 'data' / 'eval_ids.json').exists() else 0} of them tested"
                         + (f"; facts probed: {plan['knowledge']}" if plan.get("knowledge") else ""))
        except (OSError, KeyError, ValueError):
            pass
        lines.append("")
        for name, rec in st.get("arms", {}).items():
            if rec.get("status") == "skipped":
                lines.append(f"arm {name:<10} skipped: {rec['why']}")
                continue
            h = rec.get("holdout") or {}
            pct = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
            lines.append(f"arm {name:<10} {str(rec.get('verdict', '?')).upper():<13} holdout n={h.get('n')} "
                         f"accuracy {pct(h.get('accuracy_before'))} -> {pct(h.get('accuracy_after'))}, "
                         f"fixed {h.get('fixed')} broke {h.get('broke')} (p={h.get('p')}), "
                         f"wrong {pct(h.get('wrong_before'))} -> {pct(h.get('wrong_after'))}"
                         + (f"  layers {rec['focus_layers']}" if rec.get("focus_layers") else ""))
            for r in rec.get("reasons", []):
                lines.append(f"{'':<15}- {r}")
        for h in st.get("head_to_head", []):
            lines.append(f"\n{h['after']} vs {h['before']} on the holdout: {h['verdict']} "
                         f"(fixed {h['fixed']}, broke {h['broke']}, p={h['p']})")
        rec = st.get("recommended")
        lines += ["", f"RECOMMENDED: {rec} -> {st['arms'][rec]['gguf']}" if rec else
                  "RECOMMENDED: none; no arm passed the gate. Keep the base model."]
        text = "\n".join(lines)
        (self.out / "report.txt").write_text(text + "\n")
        return text


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="the cycle's folder (resumes when it exists)")
    p.add_argument("--base-model", required=True, help="HF id or path of the base model's weights (for training)")
    p.add_argument("--base-gguf", help="the base model as GGUF, served by llama-server for the baseline (recommended)")
    p.add_argument("--base-endpoint", help="URL@model where the base model is already served (instead of --base-gguf)")
    p.add_argument("--teacher", help="URL@model of a stronger model for verified targets and variations")
    p.add_argument("--arms", default="all", help="comma list of layer choices: all, plan, or ranges like 10-17 "
                   "(default all)")
    p.add_argument("--review-model", help="neuron review model id, for the plan arm's focus layers")
    p.add_argument("--review-dir")
    p.add_argument("--method", choices=["qlora", "lora", "full"], default="lora")
    p.add_argument("--dpo", action="store_true")
    p.add_argument("--finetune-arg", action="append", default=[], metavar="ARG",
                   help="extra finetune.py argument, repeatable, e.g. --finetune-arg=--max-steps=200")
    p.add_argument("--quant", default="Q4_K_M", help="GGUF type for the arms (default Q4_K_M)")
    p.add_argument("--merge-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--expand", type=int, default=0)
    p.add_argument("--unknown", choices=["abstain", "skip", "teach"], default="abstain")
    p.add_argument("--probe-samples", type=int, default=4)
    p.add_argument("--deficit-fraction", type=float, default=0.25)
    p.add_argument("--min-examples", type=int, default=8, help="do not train on fewer deficit examples")
    p.add_argument("--general", action="append", help="extra anchor JSONL (repeatable)")
    p.add_argument("--tasks", nargs="+", help="TestQA task files or folders (default: the bank)")
    p.add_argument("--per-subject", type=int, default=0)
    p.add_argument("--subject", nargs="+")
    p.add_argument("--allow-exec", action="store_true")
    p.add_argument("--sandbox", choices=["none", "docker", "podman"], default="docker")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--llama", help="llama.cpp checkout (default $OC_LLAMA or ~/llama.cpp)")
    p.add_argument("--llama-server", help="llama-server binary (default from the checkout)")
    p.add_argument("--server-arg", action="append", default=[], metavar="ARG",
                   help="extra llama-server argument, repeatable, e.g. --server-arg=-ngl --server-arg=99")
    p.add_argument("--ctx", type=int, default=8192, help="llama-server context, shared by its parallel requests")
    p.add_argument("--max-tokens", type=int, default=2048, help="per TestQA answer")
    p.add_argument("--load-timeout", type=float, default=600)
    p.add_argument("--api-key", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--redo", action="append", default=[],
                   help="run a step again: baseline, data, train/ARM, convert/ARM, eval/ARM, or all")
    a = p.parse_args(argv)
    if not (a.base_gguf or a.base_endpoint):
        p.error("give --base-gguf (served here, recommended) or --base-endpoint")
    a.finetune_arg = [x for arg in a.finetune_arg for x in arg.split("=", 1)] if a.finetune_arg else []
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    c = Cycle(a)
    c.baseline()
    c.data()
    if not c.has_data():
        print(c.report())
        return 0
    for arm in arms:
        c.arm(arm)
    c.gate(arms)
    print(c.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
