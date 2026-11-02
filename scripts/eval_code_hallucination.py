#!/usr/bin/env python3
"""
Measure code hallucination against a real package index, across suppression
scales, with paired statistics.

Why not an LLM judge: for code you have ground truth. A model that writes
`requests.get_json(...)` has hallucinated, and `hasattr(requests, "get_json")`
settles it with no judge, no rubric and no second model's opinion. That makes
this cheap enough to run at every scale and honest enough to report.

What it measures per task:

    clean         code produced, every referenced symbol resolves
    hallucinated  code produced, at least one symbol does not exist
    abstained     no code produced (declined, or emitted prose only)
    unparsable    code produced but syntactically invalid

The suppression trade shows up as hallucinated -> abstained. That is the whole
question: does OmniCortex convert wrong answers into declined ones, and at what
cost in clean answers lost.

METHODOLOGY WARNING. Do not add tasks here in response to failures you saw
while tuning, and do not adjust the profile based on results from this script.
It is a held-out measurement. If it informs the neuron set or the scale, it
stops measuring anything.

    python scripts/eval_code_hallucination.py \\
        --base_url http://GPU-HOST:8080 \\
        --tasks data/code_tasks.jsonl \\
        --alphas 0 0.5 1.0 --adapter_id 0 \\
        --out results.json

Task file, one JSON object per line:
    {"id": "req-01",
     "prompt": "Using the requests library, fetch JSON from a URL and ...",
     "modules": ["requests", "json"]}

`modules` must be importable in this interpreter -- that is the ground truth.
"""

import argparse
import ast
import importlib
import json
from concurrent.futures import ThreadPoolExecutor

import requests as _http
from tqdm import tqdm

THINK_CLOSE = "</think>"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0],
                   help="Adapter scales. Include 0.0 as the control.")
    p.add_argument("--adapter_id", type=int, default=0)
    p.add_argument("--model", default=None, help="for OpenAI-compatible servers")
    p.add_argument("--no_adapter", action="store_true",
                   help="Skip /lora-adapters. Use for a reference model such "
                        "as Claude, where there is nothing to sweep.")
    p.add_argument("--label", default=None,
                   help="Name for this endpoint in the output, e.g. 'claude'")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=1500)
    p.add_argument("--out", default="results.json")
    return p.parse_args()


def strip_think(t):
    return t.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in t else t


def extract_code(text):
    """Fenced python blocks, else the whole thing if it parses as python."""
    blocks, cur, inside = [], [], False
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("```"):
            if inside:
                blocks.append("\n".join(cur))
                cur, inside = [], False
            elif st[3:].strip().lower() in ("", "python", "py", "python3"):
                inside = True
            else:
                inside = True   # other language: captured, will fail to parse
            continue
        if inside:
            cur.append(line)
    if inside and cur:
        blocks.append("\n".join(cur))
    if blocks:
        return "\n\n".join(blocks)
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        return ""


def referenced_symbols(code, allowed_modules):
    """-> {(module, attribute)} for attributes reached through a known module.

    Deliberately conservative. Only attributes on names bound by an import of a
    module in `allowed_modules` are checked, and locally reassigned names are
    dropped. Anything ambiguous is not counted, so this under-reports rather
    than inventing hallucinations.
    """
    tree = ast.parse(code)
    alias_to_mod, from_imports, assigned = {}, [], set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in allowed_modules:
                    alias_to_mod[a.asname or root] = a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in allowed_modules \
                    and node.level == 0:
                for a in node.names:
                    if a.name != "*":
                        from_imports.append((node.module, a.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        assigned.add(sub.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            assigned.add(node.name)
            args = getattr(node, "args", None)
            if args:
                for a in (args.args + args.kwonlyargs + args.posonlyargs):
                    assigned.add(a.arg)

    out = set(from_imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.value.id
            if name in alias_to_mod and name not in assigned:
                out.add((alias_to_mod[name], node.attr))
    return out


_mod_cache = {}


def resolves(module, attr):
    if module not in _mod_cache:
        try:
            _mod_cache[module] = importlib.import_module(module)
        except Exception:
            _mod_cache[module] = None
    m = _mod_cache[module]
    if m is None:
        return None      # cannot verify: not counted either way
    return hasattr(m, attr)


def grade(text, modules):
    code = extract_code(strip_think(text))
    if not code.strip():
        return "abstained", []
    try:
        symbols = referenced_symbols(code, set(modules))
    except SyntaxError:
        return "unparsable", []
    bad = [f"{m}.{a}" for m, a in sorted(symbols) if resolves(m, a) is False]
    return ("hallucinated" if bad else "clean"), bad


def set_alpha(base_url, adapter_id, alpha):
    r = _http.post(f"{base_url}/lora-adapters", timeout=30,
                   json=[{"id": adapter_id, "scale": alpha}])
    if r.status_code != 200:
        raise SystemExit(
            f"could not set adapter scale ({r.status_code}): {r.text[:200]}\n"
            "llama-server with --lora-scaled is required for a sweep; pass "
            "--no_adapter for a plain reference endpoint."
        )


def ask(args, prompt):
    body = {"messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": args.max_tokens}
    if args.model:
        body["model"] = args.model
    r = _http.post(f"{args.base_url}/v1/chat/completions", timeout=900, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def mcnemar(base, other, key):
    """Discordant-pair test. Same tasks, greedy decoding, so only changed
    verdicts carry information; two independent proportions would overstate n."""
    fixed = sum(1 for b, x in zip(base, other) if b == key and x != key)
    broke = sum(1 for b, x in zip(base, other) if b != key and x == key)
    disc = fixed + broke
    chi2 = (fixed - broke) ** 2 / disc if disc else 0.0
    return fixed, broke, chi2, disc


def main():
    args = parse_args()
    tasks = []
    with open(args.tasks, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if not tasks:
        raise SystemExit("no tasks")

    unverifiable = set()
    for t in tasks:
        for m in t.get("modules", []):
            if resolves(m, "__name__") is None:
                unverifiable.add(m)
    if unverifiable:
        print(f"warning: not importable here, so symbols in them cannot be "
              f"checked: {sorted(unverifiable)}")
        print("         install them or those tasks will under-report")

    alphas = [None] if args.no_adapter else args.alphas
    label = args.label or ("reference" if args.no_adapter else "adapter")
    print(f"{len(tasks)} tasks x {len(alphas)} setting(s) against {label}\n")

    runs = {}
    hdr = f"{'setting':>10} {'clean':>7} {'halluc':>7} {'abstain':>8} {'unparse':>8}"
    print(hdr)
    print("-" * len(hdr))
    for alpha in alphas:
        if alpha is not None:
            set_alpha(args.base_url, args.adapter_id, alpha)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            texts = list(tqdm(pool.map(lambda t: ask(args, t["prompt"]), tasks),
                              total=len(tasks), leave=False,
                              desc=f"  {alpha}"))
        verdicts, details = [], []
        for t, txt in zip(tasks, texts):
            v, bad = grade(txt, t.get("modules", []))
            verdicts.append(v)
            details.append({"id": t["id"], "verdict": v, "bad_symbols": bad})
        key = "a=%.2f" % alpha if alpha is not None else label
        runs[key] = {"verdicts": verdicts, "details": details}
        n = len(verdicts)
        c = {k: verdicts.count(k) for k in
             ("clean", "hallucinated", "abstained", "unparsable")}
        print(f"{key:>10} {c['clean']/n:>7.1%} {c['hallucinated']/n:>7.1%} "
              f"{c['abstained']/n:>8.1%} {c['unparsable']/n:>8.1%}")

    if len(runs) > 1:
        base_key = next(iter(runs))
        base = runs[base_key]["verdicts"]
        print(f"\nvs {base_key}, discordant pairs only:")
        for k, r in runs.items():
            if k == base_key:
                continue
            f_, b_, chi2, disc = mcnemar(base, r["verdicts"], "hallucinated")
            traded = sum(1 for x, y in zip(base, r["verdicts"])
                         if x == "clean" and y == "abstained")
            note = ("significant" if chi2 > 3.84 else "not significant") \
                if disc >= 10 else "too few changes to call"
            print(f"  {k}: {f_} hallucinated->not, {b_} not->hallucinated, "
                  f"{traded} clean->abstained  (chi2={chi2:.1f}, {note})")
        if len(tasks) < 100:
            print(f"\n  {len(tasks)} tasks is a pilot. Discordant pairs are what "
                  "carry power here,\n  and at this size only a large effect "
                  "will clear significance.")

    with open(args.out, "w") as f:
        json.dump({"tasks": len(tasks), "label": label, "runs": runs}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("Compare endpoints by running this twice with different --base_url "
          "and --label,\nthen read the clean rates against each other. The "
          "informative axis is\nsuppressed vs unsuppressed on one model; a "
          "frontier model is a ceiling, not a rival.")


if __name__ == "__main__":
    main()
