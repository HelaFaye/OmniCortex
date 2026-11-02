#!/usr/bin/env python3
"""
Pick the quantization of a model that suits this computer, by measuring.

Give it several GGUF files of the same model in different quantizations
(Q8_0, Q6_K, Q4_K_M, IQ3_XS, the ternary TQ2_0 / TQ1_0...), or let it make
them from a high-precision GGUF with llama-quantize. For each one it
measures, on this hardware:

  fit        does the file fit in the memory you give it (or the largest
             accelerator's free memory, or RAM)?
  speed      engine_optimizer.py's benchmark; with --tune N, its Bayesian
             search (Optuna TPE) over engine options for that file as well
  fidelity   how closely it follows the highest-precision file: llama.cpp's
             KL divergence and "same top token" rate (llama-perplexity
             --kl-divergence) when that tool is found, otherwise how much of
             each greedy answer to a fixed set of prompts matches the
             reference word for word

and picks the fastest one whose fidelity clears the bar (--max-kld,
--min-same-top or --min-agreement). The choice is kept per hardware in
~/.neuronscope/quant_choices.json, so the same model can run as Q8_0 on a
big GPU and Q4_K_M on a laptop. --apply writes the winner's engine settings
into Studio's settings too.

No switch between quantizations is lossless: each file loses something
against the original weights, and this measures how much. Ternary types
(TQ1_0, TQ2_0) only keep a model's quality if it was trained ternary
(BitNet b1.58 and its relatives). Quantizing an ordinary model to ternary
wrecks it, and the fidelity column shows that.

    python scripts/quant_select.py --model m-Q8_0.gguf m-Q4_K_M.gguf m-IQ3_XS.gguf --out runs/qsel-m
    python scripts/quant_select.py --from m-F16.gguf --make Q8_0,Q6_K,Q4_K_M --out runs/qsel-m --tune 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine_optimizer as eopt  # noqa: E402

QUANT = re.compile(r"(?<![A-Za-z0-9])(TQ[12]_0|IQ\d_(?:XXS|XS|S|M|NL)|Q\d_K(?:_[SML])?|Q\d_[01]|BF16|FP?16|FP?32)"
                   r"(?![A-Za-z0-9])", re.I)
BPW = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.75, "Q5_K_S": 5.5, "Q5_0": 5.5,
       "Q5_1": 6, "Q4_K_M": 4.75, "Q4_K_S": 4.5, "Q4_0": 4.5, "Q4_1": 5, "IQ4_XS": 4.25, "IQ4_NL": 4.5,
       "Q3_K_L": 4.27, "Q3_K_M": 3.91, "Q3_K_S": 3.5, "IQ3_M": 3.66, "IQ3_S": 3.44, "IQ3_XS": 3.3, "IQ3_XXS": 3.06,
       "Q2_K": 2.63, "IQ2_M": 2.7, "IQ2_S": 2.5, "IQ2_XS": 2.31, "IQ2_XXS": 2.06, "IQ1_M": 1.75, "IQ1_S": 1.56,
       "TQ2_0": 2.06, "TQ1_0": 1.69}
TERNARY = {"TQ1_0", "TQ2_0"}
CHOICES = Path.home() / ".neuronscope" / "quant_choices.json"
PROMPTS = [
    "The capital of Australia is", "Water boils at sea level at a temperature of",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: If a train travels 120 km in 1.5 hours, what is its average speed?\nA:",
    "Write one sentence describing autumn:", "The three primary colours of light are",
    "SELECT name, COUNT(*) FROM orders", "In 1969, the Apollo 11 mission",
    "A haiku about the sea:\n", "The derivative of x squared is",
    "Translate to French: 'Good morning, how are you?'\n", "Photosynthesis converts",
]


def quant_of(path: str) -> str | None:
    m = QUANT.search(Path(path).name)
    return m.group(1).upper().replace("FP", "F") if m else None       # fp16 -> F16


def family(path: str) -> str:
    """The model's name without the quantization: m-Q4_K_M.gguf -> m."""
    name = Path(path).name
    if name.endswith(".gguf"):
        name = name[:-5]
    return re.sub(r"[-_.]+$", "", QUANT.sub("", name)).strip("-_.") or name


def hardware() -> dict:
    """What decides which file suits this computer: CPU, accelerators and their memory (no names of people
    or machines)."""
    devs = []
    try:
        import accelerators
        devs = [{"backend": d.get("backend"), "name": d.get("name"), "memory_gib": round((d.get("memory_total") or 0)
                                                                                           / 2 ** 30)}
                for d in accelerators.detect() if d.get("enabled", True)]
    except Exception:
        pass
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    hw = {"cpu": cpu, "system": platform.system(), "devices": devs}
    hw["id"] = hashlib.sha256(json.dumps(hw, sort_keys=True).encode()).hexdigest()[:12]
    return hw


def memory_budget(gib: float | None) -> tuple[int, str]:
    if gib:
        return int(gib * 2 ** 30), f"{gib:g} GiB (given)"
    try:
        import accelerators
        devs = [d for d in accelerators.detect() if d.get("enabled", True) and d.get("backend") != "cpu"]
        if devs:
            d = max(devs, key=lambda d: d.get("memory_free") or 0)
            if d.get("memory_free"):
                return int(d["memory_free"] - (d.get("reserve") or 0)), f"{d.get('name')} free memory"
    except Exception:
        pass
    try:
        pages, size = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return int(pages * size * 0.8), "80% of free RAM"
    except (ValueError, OSError, AttributeError):
        return 0, "unknown"


def make_variants(src: str, types: list[str], out: Path, quantize: str, imatrix: str | None) -> list[str]:
    """llama-quantize SRC into each type (kept in OUT/variants; existing files are reused)."""
    dst_dir = out / "variants"
    dst_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for t in types:
        t = t.upper()
        if t not in BPW:
            raise SystemExit(f"unknown quantization type {t}")
        if t in TERNARY and not re.search(r"bitnet|1\.58|ternary|tq", Path(src).name, re.I):
            print(f"warning: {t} is ternary; it only suits models trained ternary (BitNet b1.58). An ordinary model "
                  f"loses most of its quality. The fidelity measurement will show how much.", flush=True)
        dst = dst_dir / f"{family(src)}-{t}.gguf"
        if not dst.exists():
            cmd = [quantize] + (["--imatrix", imatrix] if imatrix else []) + [src, str(dst), t]
            print("$ " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        made.append(str(dst))
    return made


# ---------------------------------------------------------------- fidelity

def find_tool(server: str, name: str) -> str | None:
    cand = Path(server).with_name(name + Path(server).suffix)
    return str(cand) if cand.exists() else shutil.which(name)


def eval_text(path: str | None, out: Path) -> str:
    if path:
        return path
    dst = out / "eval.txt"
    if not dst.exists():
        docs = sorted((HERE.parent / "docs").glob("*.md"))
        dst.write_text("\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in docs)[:80_000])
    return str(dst)


def kld(tool: str, ref: str, cand: str, text: str, out: Path, ngl: int, chunks: int) -> dict:
    """llama-perplexity's KL divergence of CAND from REF over TEXT. -> {kld, same_top, ppl}"""
    base = out / f"ref-{hashlib.sha256(ref.encode()).hexdigest()[:8]}.kld"
    common = ["-c", "512", "--chunks", str(chunks), "-ngl", str(ngl)]
    if not base.exists():
        r = subprocess.run([tool, "-m", ref, "-f", text, "--kl-divergence-base", str(base)] + common,
                           capture_output=True, text=True)
        if r.returncode != 0 or not base.exists():
            raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1:] or ["llama-perplexity failed"])
    if os.path.abspath(cand) == os.path.abspath(ref):
        return {"kld": 0.0, "same_top": 100.0, "method": "kld"}
    r = subprocess.run([tool, "-m", cand, "--kl-divergence-base", str(base), "--kl-divergence"] + common,
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    m_k = re.search(r"Mean\s+KLD:\s+(-?[\d.]+)", log)
    m_t = re.search(r"Same top p:\s+([\d.]+)", log)
    if r.returncode != 0 or not m_k:
        raise RuntimeError(log.strip().splitlines()[-1:] or ["llama-perplexity failed"])
    return {"kld": float(m_k.group(1)), "same_top": float(m_t.group(1)) if m_t else None, "method": "kld"}


def answers(server: str, model: str, ngl: int, prompts: list[str], n: int, timeout: float) -> list[str]:
    """Greedy answers from MODEL to each prompt (llama-server, started and stopped here)."""
    port = eopt.free_port()
    p = subprocess.Popen([server, "-m", model, "-c", "2048", "-ngl", str(ngl), "--port", str(port),
                          "--host", "127.0.0.1", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        end = time.time() + timeout
        while True:
            if p.poll() is not None or time.time() > end:
                raise RuntimeError("llama-server did not start for " + Path(model).name)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        out = []
        for q in prompts:
            body = {"prompt": q, "n_predict": n, "temperature": 0, "seed": 0, "cache_prompt": False}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out.append(json.loads(r.read()).get("content") or "")
        return out
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()


def agreement(ref: list[str], cand: list[str]) -> dict:
    """How much of each reference answer the candidate repeats word for word before it diverges."""
    fr, exact = [], 0
    for r, c in zip(ref, cand):
        rw, cw = r.split(), c.split()
        k = 0
        while k < min(len(rw), len(cw)) and rw[k] == cw[k]:
            k += 1
        fr.append(k / len(rw) if rw else 1.0)
        exact += r.strip() == c.strip()
    return {"agreement": round(sum(fr) / len(fr), 4) if fr else None, "exact": exact, "n": len(fr),
            "method": "greedy agreement"}


# ---------------------------------------------------------------- speed

def speed(model: str, a, out: Path) -> dict:
    """engine_optimizer.py on this file: its defaults only, or --tune N trials of its search."""
    d = out / "speed" / re.sub(r"[^\w.-]+", "_", Path(model).stem)
    argv = ["--model", model, "--server", a.server, "--out", str(d), "--trials", str(max(1, a.tune)),
            "--objective", a.objective, "--ngl", str(a.ngl), "--ctx", str(a.ctx),
            "--prompt-tokens", str(a.prompt_tokens), "--gen-tokens", str(a.gen_tokens), "--repeats", str(a.repeats)]
    try:
        eopt.main(argv)
    except SystemExit:
        pass
    try:
        best = json.loads((d / "best.json").read_text()).get("best")
    except (OSError, ValueError):
        best = None
    if not best:
        return {"ok": False}
    return {"ok": True, **{k: best["result"].get(k) for k in ("gen_tps", "prompt_tps", "request_s")},
            "settings": best["settings"]}


def faster(x: dict, y: dict, objective: str) -> bool:
    if objective == "request":
        return (x.get("request_s") or 1e9) < (y.get("request_s") or 1e9)
    k = "prompt_tps" if objective == "prompt" else "gen_tps"
    return (x.get(k) or 0) > (y.get(k) or 0)


def good_enough(f: dict, a) -> bool:
    if f.get("method") == "kld":
        return f["kld"] <= a.max_kld and (f.get("same_top") is None or f["same_top"] >= a.min_same_top)
    return f.get("agreement") is not None and f["agreement"] >= a.min_agreement


def choose(rows: list[dict], a) -> dict | None:
    """The fastest file that fits and stays faithful; failing that, the most faithful one that fits."""
    fits = [r for r in rows if r["fits"] and r["speed"].get("ok")]
    ok = [r for r in fits if r.get("fidelity") and good_enough(r["fidelity"], a)]
    if ok:
        best = ok[0]
        for r in ok[1:]:
            if faster(r["speed"], best["speed"], a.objective):
                best = r
        return {**best, "why": "fastest that fits and stays faithful"}
    scored = [r for r in fits if r.get("fidelity")]
    if scored:
        f = (lambda r: -r["fidelity"]["kld"]) if scored[0]["fidelity"].get("method") == "kld" else \
            (lambda r: r["fidelity"].get("agreement") or 0)
        return {**max(scored, key=f), "why": "none clears the fidelity bar; the most faithful that fits"}
    return None


def remember(fam: str, hw: dict, pick: dict, path: Path = CHOICES) -> None:
    allc = json.loads(path.read_text()) if path.exists() else {}
    allc.setdefault(fam, {})[hw["id"]] = {"model": pick["model"], "quant": pick["quant"], "hardware": hw,
                                          "speed": pick["speed"], "fidelity": pick.get("fidelity"), "t": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(allc, indent=1))


def table(rows: list[dict], pick: dict | None, objective: str) -> str:
    k = {"prompt": "prompt_tps", "request": "request_s"}.get(objective, "gen_tps")
    lines = [f"{'quant':<9}{'size':>8}  {'fits':<5}{k:>11}  fidelity"]
    for r in rows:
        f = r.get("fidelity") or {}
        fid = (f"KLD {f['kld']:.4f}" + (f", same top {f['same_top']:.1f}%" if f.get("same_top") is not None else "")
               if f.get("method") == "kld" else
               f"{f['agreement']:.0%} of answers match" if f.get("agreement") is not None else f.get("error", "-"))
        sp = r["speed"].get(k)
        lines.append(f"{(r['quant'] or '?'):<9}{r['size'] / 2 ** 30:>7.2f}G  {'yes' if r['fits'] else 'no':<5}"
                     f"{(f'{sp:.2f}' if sp else '-'):>11}  {fid}" + ("   <- chosen" if pick and pick["model"] == r["model"]
                                                                      else "")
                     + ("   (reference)" if r.get("reference") else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", nargs="*", default=[], help="GGUF files of the same model in different quantizations")
    p.add_argument("--from", dest="src", help="a high-precision GGUF to make variants from (with --make)")
    p.add_argument("--make", help="comma-separated types to make with llama-quantize, e.g. Q8_0,Q6_K,Q4_K_M,TQ2_0")
    p.add_argument("--imatrix", help="importance matrix for llama-quantize (recommended below 4 bits)")
    p.add_argument("--reference", help="the file others are compared with (default: the highest precision)")
    p.add_argument("--server", default=os.environ.get("NS_LLAMA_SERVER") or shutil.which("llama-server") or "llama-server")
    p.add_argument("--out", required=True)
    p.add_argument("--objective", choices=["gen", "prompt", "request"], default="gen")
    p.add_argument("--tune", type=int, default=0, help="Bayesian engine-option trials per file (0: defaults only)")
    p.add_argument("--memory-gb", type=float, help="memory the model may use (default: largest accelerator, or RAM)")
    p.add_argument("--max-kld", type=float, default=0.1, help="fidelity bar with llama-perplexity")
    p.add_argument("--min-same-top", type=float, default=85.0, help="per cent; fidelity bar with llama-perplexity")
    p.add_argument("--min-agreement", type=float, default=0.6, help="fidelity bar without it (0..1)")
    p.add_argument("--fidelity", choices=["auto", "kld", "agreement"], default="auto")
    p.add_argument("--text", help="text for the KL divergence (default: OmniCortex's docs)")
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--apply", metavar="STUDIO_SETTINGS", help="also write the winner's engine settings into Studio's")
    p.add_argument("--choices", default=str(CHOICES), help="where choices per hardware are kept")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = list(a.model)
    if a.make:
        if not a.src:
            raise SystemExit("--make needs --from")
        quantize = find_tool(a.server, "llama-quantize") or "llama-quantize"
        models += make_variants(a.src, a.make.split(","), out, quantize, a.imatrix)
        models.append(a.src)
    models = list(dict.fromkeys(os.path.abspath(m) for m in models))
    if len(models) < 2:
        raise SystemExit("give at least two quantizations of the model (--model ...), or --from with --make")
    fams = {family(m) for m in models}
    if len(fams) > 1:
        print(f"note: the files look like different models ({', '.join(sorted(fams))}); comparing them anyway")
    ref = os.path.abspath(a.reference) if a.reference else max(models, key=lambda m: BPW.get(quant_of(m) or "", 0))
    budget, budget_from = memory_budget(a.memory_gb)
    hw = hardware()
    print(f"{len(models)} files; reference {Path(ref).name}; memory {budget / 2 ** 30:.1f} GiB ({budget_from})")
    tool = find_tool(a.server, "llama-perplexity") if a.fidelity != "agreement" else None
    if a.fidelity == "kld" and not tool:
        raise SystemExit("llama-perplexity not found next to llama-server or on PATH")
    text = eval_text(a.text, out) if tool else None
    ref_answers = None
    rows = []
    for m in sorted(models, key=lambda m: -BPW.get(quant_of(m) or "", 0)):
        size = os.path.getsize(m)
        row = {"model": m, "quant": quant_of(m), "size": size, "fits": not budget or size <= budget * 0.9,
               "reference": m == ref, "ternary": quant_of(m) in TERNARY}
        print(f"\n== {Path(m).name} ({size / 2 ** 30:.2f} GiB{', does not fit' if not row['fits'] else ''})", flush=True)
        row["speed"] = speed(m, a, out) if row["fits"] else {"ok": False, "error": "does not fit"}
        try:
            if tool:
                row["fidelity"] = kld(tool, ref, m, text, out, a.ngl, a.chunks)
            else:
                if ref_answers is None:
                    ref_answers = answers(a.server, ref, a.ngl, PROMPTS, 48, a.timeout)
                row["fidelity"] = agreement(ref_answers, ref_answers if m == ref else
                                            answers(a.server, m, a.ngl, PROMPTS, 48, a.timeout))
        except Exception as e:
            row["fidelity"] = {"error": str(e)[:300]}
        rows.append(row)
    pick = choose(rows, a)
    txt = table(rows, pick, a.objective)
    print("\n" + txt)
    if pick:
        print(f"\nchosen: {Path(pick['model']).name} ({pick['why']}) for this hardware ({hw['cpu']}"
              + "".join(f", {d['name']}" for d in hw["devices"]) + ")")
        remember(family(pick["model"]), hw, pick, Path(a.choices).expanduser())
        if a.apply and pick["speed"].get("settings"):
            eopt.apply(a.apply, pick["model"], pick["speed"]["settings"])
            print(f"applied its engine settings to {a.apply}")
    else:
        print("\nno file fits and runs here")
    report = {"hardware": hw, "reference": ref, "memory_budget": budget, "objective": a.objective,
              "rows": rows, "chosen": pick and {k: pick[k] for k in ("model", "quant", "why")}}
    (out / "quant_select.json").write_text(json.dumps(report, indent=1))
    (out / "report.txt").write_text(txt + "\n")
    return 0 if pick else 1


if __name__ == "__main__":
    sys.exit(main())
