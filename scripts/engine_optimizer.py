#!/usr/bin/env python3
"""
Find this machine's fastest engine settings for one model by measuring, not
guessing.

Each trial starts llama-server with one combination of engine options
(threads, batch, ubatch, flash attention, KV cache types, speculative
decoding, CPU placement...). It sends the same request a few times, reads
the server's own timings, and stops it. The search is Bayesian (Optuna's
TPE) when optuna is installed. Otherwise it is random search followed by a
one-option-at-a-time refinement around the best trial.

    python scripts/engine_optimizer.py --model models/m-Q4_K_M.gguf --trials 30 --out runs/opt-m
    python scripts/engine_optimizer.py --model ... --objective request --prompt-tokens 2000 --gen-tokens 200
    python scripts/engine_optimizer.py --model ... --apply ~/.omnicortex/studio.json    # Studio uses the winner

Objectives:
  gen       generation tokens per second (the default; what chat feels like)
  prompt    prompt tokens per second (long documents, RAG)
  request   time for the whole request shape given by --prompt-tokens and
            --gen-tokens (lower is better)

Only speed is measured here. Settings that can change answers are left
out of the search unless you add them: KV cache quantization (--kv), and
speculative decoding, which is exact for greedy sampling but changes
timing (--spec). The suppression scale α is a quality setting; tune it
with tune_scale_server.py and the gate, not for tokens per second.

Trials are cached in <out>/trials.jsonl, so a run resumes and a longer run
builds on a shorter one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine_options as eo  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def cores() -> tuple[int, int]:
    """(physical, logical) CPU counts, best effort."""
    logical = os.cpu_count() or 4
    phys = logical
    try:
        ids = set()
        block = {}
        for line in open("/proc/cpuinfo"):
            if not line.strip():
                if block:
                    ids.add((block.get("physical id"), block.get("core id")))
                block = {}
                continue
            k, _, v = line.partition(":")
            block[k.strip()] = v.strip()
        if block:
            ids.add((block.get("physical id"), block.get("core id")))
        if ids and None not in {i[1] for i in ids}:
            phys = len(ids)
    except OSError:
        phys = max(1, logical // 2) if logical >= 8 else logical
    return phys, logical


def space(a, supported: set[str]) -> dict:
    """option -> candidate values (the first is the engine's default where there is one)."""
    phys, logical = cores()
    threads = sorted({max(1, phys // 2), phys, min(logical, phys + phys // 2), logical} - {0})
    sp = {"threads": threads, "batch": [512, 1024, 2048], "ubatch": [128, 256, 512],
          "flash_attn": ["on", "off"]}
    if a.kv:
        sp["cache_type_k"] = ["f16", "q8_0"]
        sp["cache_type_v"] = ["f16", "q8_0"]
    if a.spec:
        sp["spec_type"] = ["none", "ngram-mod", "ngram-simple"]
    if a.placement:
        sp["cpu_strict"] = [False, True]
        sp["poll"] = [0, 50]
    for k in a.fix or []:
        name, _, v = k.partition("=")
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        sp[name] = [val]
    # leave out what this build cannot do
    return {k: v for k, v in sp.items() if k in ("threads", "batch") or eo.spelling(k, "llama", supported)}


def valid(cfg: dict) -> bool:
    if cfg.get("ubatch") and cfg.get("batch") and cfg["ubatch"] > cfg["batch"]:
        return False
    if cfg.get("cache_type_v") not in (None, "f16") and cfg.get("flash_attn") == "off":
        return False                     # a quantized V cache needs flash attention
    return True


def key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def filler(n_tokens: int) -> str:
    words = ("the model reads this sentence about rivers mountains numbers and small facts so that the "
             "prompt has a known length ").split()
    return " ".join(words[i % len(words)] for i in range(int(n_tokens * 0.8)))


class Bench:
    def __init__(self, a):
        self.a = a
        self.supported = eo.probe(a.server, "llama")
        self.prompt = filler(a.prompt_tokens)

    def cmd(self, cfg: dict, port: int) -> list[str]:
        a = self.a
        cmd = [a.server, "-m", a.model, "-c", str(a.ctx), "-ngl", str(a.ngl), "--port", str(port),
               "--host", "127.0.0.1", "-np", "1"]
        if cfg.get("threads"):
            cmd += ["-t", str(cfg["threads"])]
        if cfg.get("batch"):
            cmd += ["-b", str(cfg["batch"])]
        opts = {k: v for k, v in cfg.items() if k in eo.OPTION}
        if opts.get("spec_type") == "none":
            opts.pop("spec_type")
        args, _ = eo.build_args(opts, "llama", self.supported)
        return cmd + args + list(a.server_arg)

    def run(self, cfg: dict) -> dict:
        a = self.a
        port = free_port()
        cmd = self.cmd(cfg, port)
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            if not self._wait(p, port):
                err = ""
                if p.poll() is not None:
                    err = (p.stderr.read() or b"").decode(errors="replace").strip().splitlines()[-1:] or [""]
                    err = err[0][-300:]
                return {"ok": False, "error": err or "did not start", "cmd": cmd}
            load = time.time() - t0
            self._ask(port, 8)                                        # warm-up
            runs = [self._ask(port, a.gen_tokens) for _ in range(a.repeats)]
            pps = sorted(r.get("prompt_per_second") or 0 for r in runs)[len(runs) // 2]
            gps = sorted(r.get("predicted_per_second") or 0 for r in runs)[len(runs) // 2]
            n_prompt = runs[0].get("prompt_n") or a.prompt_tokens
            n_gen = runs[0].get("predicted_n") or a.gen_tokens
            req = (n_prompt / pps if pps else 1e9) + (n_gen / gps if gps else 1e9)
            return {"ok": True, "prompt_tps": round(pps, 2), "gen_tps": round(gps, 2),
                    "request_s": round(req, 3), "load_s": round(load, 1), "prompt_n": n_prompt, "gen_n": n_gen,
                    "cmd": cmd}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "cmd": cmd}
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()

    def _wait(self, p, port) -> bool:
        end = time.time() + self.a.load_timeout
        while time.time() < end:
            if p.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(0.3)
        return False

    def _ask(self, port, n) -> dict:
        body = {"prompt": self.prompt, "n_predict": n, "temperature": 0, "cache_prompt": False,
                "ignore_eos": True, "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.a.request_timeout) as r:
            return json.loads(r.read()).get("timings") or {}


def score(res: dict, objective: str) -> float:
    if not res.get("ok"):
        return float("-inf")
    if objective == "prompt":
        return res["prompt_tps"]
    if objective == "request":
        return -res["request_s"]
    return res["gen_tps"]


class Search:
    def __init__(self, a, bench: Bench, sp: dict):
        self.a, self.bench, self.sp = a, bench, sp
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "trials.jsonl"
        self.trials = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                t = json.loads(line)
                if t.get("model") == a.model:
                    self.trials[t["key"]] = t
        self.rng = random.Random(a.seed)
        self.new = 0

    def measure(self, cfg: dict) -> float:
        k = key(cfg)
        if k in self.trials:
            return score(self.trials[k]["result"], self.a.objective)
        if self.new >= self.a.trials:
            raise StopIteration
        self.new += 1
        res = self.bench.run(cfg)
        t = {"key": k, "model": self.a.model, "settings": cfg, "result": {k2: v for k2, v in res.items() if k2 != "cmd"},
             "cmd": res.get("cmd"), "t": time.time()}
        self.trials[k] = t
        with open(self.path, "a") as f:
            f.write(json.dumps(t) + "\n")
        r = t["result"]
        print(f"[{self.new}/{self.a.trials}] " + (
            f"gen {r['gen_tps']:.1f} t/s  prompt {r['prompt_tps']:.1f} t/s  request {r['request_s']:.2f}s"
            if r.get("ok") else f"failed: {r.get('error')}") + "  " + json.dumps(cfg), flush=True)
        return score(r, self.a.objective)

    def baseline(self) -> dict:
        """The engine's defaults, measured first: what everything is compared against."""
        return {"threads": self.sp["threads"][-1] if len(self.sp["threads"]) < 2 else cores()[0],
                "batch": 2048}

    def random_cfg(self) -> dict:
        while True:
            cfg = {k: self.rng.choice(v) for k, v in self.sp.items()}
            if valid(cfg):
                return cfg

    def run(self) -> None:
        try:
            self.measure(self.baseline())
            if self._optuna():
                return
            n_random = max(1, self.a.trials // 2)
            for _ in range(n_random):
                self.measure(self.random_cfg())
            # refine: change one option at a time around the best, until nothing improves
            improved = True
            while improved:
                improved = False
                best = self.best()
                if best is None:
                    break
                base = dict(best["settings"])
                for k, vals in self.sp.items():
                    for v in vals:
                        if base.get(k) == v:
                            continue
                        cfg = {**base, k: v}
                        if valid(cfg) and self.measure(cfg) > score(best["result"], self.a.objective):
                            improved = True
                            best = self.trials[key(cfg)]
                            base = dict(cfg)
        except StopIteration:
            pass

    def _optuna(self) -> bool:
        try:
            import optuna
        except ImportError:
            return False
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.a.seed))
        for t in self.trials.values():            # what is already known
            dist = {k: optuna.distributions.CategoricalDistribution(v) for k, v in self.sp.items()}
            if all(k in t["settings"] and t["settings"][k] in v for k, v in self.sp.items()):
                s = score(t["result"], self.a.objective)
                if s != float("-inf"):
                    study.add_trial(optuna.trial.create_trial(params={k: t["settings"][k] for k in self.sp},
                                                              distributions=dist, value=s))

        def objective(trial):
            cfg = {k: trial.suggest_categorical(k, v) for k, v in self.sp.items()}
            if not valid(cfg):
                raise optuna.TrialPruned()
            s = self.measure(cfg)
            if s == float("-inf"):
                raise optuna.TrialPruned()
            return s
        try:
            study.optimize(objective, n_trials=self.a.trials * 3, catch=())
        except StopIteration:
            pass
        return True

    def best(self) -> dict | None:
        ok = [t for t in self.trials.values() if t["result"].get("ok")]
        return max(ok, key=lambda t: score(t["result"], self.a.objective)) if ok else None


def report(search: Search, base_key: str) -> dict:
    a = search.a
    ok = sorted((t for t in search.trials.values() if t["result"].get("ok")),
                key=lambda t: -score(t["result"], a.objective))
    best = ok[0] if ok else None
    base = search.trials.get(base_key)
    lines = [f"{len(search.trials)} trials for {a.model}, objective {a.objective}", ""]
    for t in ok[:10]:
        r = t["result"]
        lines.append(f"gen {r['gen_tps']:>7.1f}  prompt {r['prompt_tps']:>8.1f}  request {r['request_s']:>6.2f}s  "
                     + " ".join(f"{k}={v}" for k, v in sorted(t["settings"].items())))
    failed = [t for t in search.trials.values() if not t["result"].get("ok")]
    if failed:
        lines.append(f"\n{len(failed)} failed to start or answer (e.g. {failed[0]['result'].get('error')})")
    gain = None
    if best and base and base["result"].get("ok"):
        b, w = score(base["result"], a.objective), score(best["result"], a.objective)
        gain = (w / b - 1) if a.objective != "request" else (b / w - 1)
        lines.append(f"\nbest vs the engine's defaults: {gain:+.0%}")
    out = {"model": a.model, "objective": a.objective, "best": best and {"settings": best["settings"],
                                                                         "result": best["result"]},
           "baseline": base and {"settings": base["settings"], "result": base["result"]}, "gain": gain,
           "trials": len(search.trials), "search": "optuna-tpe" if _has_optuna() else "random+refine"}
    (search.out / "best.json").write_text(json.dumps(out, indent=1))
    (search.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def apply(settings_path: str, model: str, best: dict) -> None:
    """Merge the winner into Studio's per-model settings (the same file and key Studio uses)."""
    p = Path(settings_path).expanduser()
    allset = json.loads(p.read_text()) if p.exists() else {}
    k = hashlib.sha256(model.encode()).hexdigest()[:16]
    cur = allset.get(k, {})
    cur.update(best)
    if cur.get("spec_type") == "none":
        cur.pop("spec_type")
    allset[k] = cur
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(allset, indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="the GGUF to tune for")
    p.add_argument("--server", default=os.environ.get("OC_LLAMA_SERVER")
                   or str(Path(os.environ.get("OC_LLAMA", "~/llama.cpp")).expanduser() / "build" / "bin" / "llama-server"))
    p.add_argument("--out", required=True, help="folder for trials.jsonl, best.json, report.txt")
    p.add_argument("--trials", type=int, default=24, help="new measurements to make (cached ones are free)")
    p.add_argument("--objective", choices=["gen", "prompt", "request"], default="gen")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--kv", action="store_true", help="also try quantized KV caches (can change answers slightly)")
    p.add_argument("--spec", action="store_true", help="also try draft-free speculative decoding")
    p.add_argument("--placement", action="store_true", help="also try strict CPU placement and busy-wait levels")
    p.add_argument("--fix", action="append", metavar="OPTION=VALUE",
                   help="hold an option at a value, e.g. --fix device=Vulkan0 (repeatable)")
    p.add_argument("--server-arg", action="append", default=[], metavar="ARG", help="passed to every trial")
    p.add_argument("--load-timeout", type=float, default=600)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apply", metavar="STUDIO_SETTINGS", help="write the winner into Studio's settings file")
    a = p.parse_args(argv)
    if not Path(a.server).exists():
        raise SystemExit(f"no llama-server at {a.server} (--server, or Setup → Build llama.cpp)")
    a.model = os.path.abspath(os.path.expanduser(a.model))     # the path Studio keys its settings by
    bench = Bench(a)
    sp = space(a, bench.supported)
    print("searching: " + ", ".join(f"{k} {v}" for k, v in sp.items())
          + f"  ({'Optuna TPE' if _has_optuna() else 'random + refinement; pip install optuna for TPE'})")
    search = Search(a, bench, sp)
    base_key = key(search.baseline())
    search.run()
    out = report(search, base_key)
    if a.apply and out["best"]:
        apply(a.apply, a.model, out["best"]["settings"])
        print(f"applied to {a.apply}: Studio uses these settings the next time it loads the model")
    return 0 if out["best"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
