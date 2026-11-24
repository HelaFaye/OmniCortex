#!/usr/bin/env python3
"""
One acceptance gate for any change to a model: a retrain, a repository adapter, a new quantization. It reads the
reports the measuring tools already write and gives one verdict, with every reason.

    python scripts/accept_gate.py --preset strict \\
        --retrain runs/cycle/gate.json \\
        --fidelity runs/fidelity/fidelity.json --quant model-Q4_K_M \\
        --adapt-manifest runs/adapt-app/manifest.json --repo ~/src/app --adapter out/app-lora.gguf \\
        --out runs/accept

| Input | From | Passes when |
|---|---|---|
| `--retrain` | `retrain_gate.py --out` | its verdict is *keep* (the holdout improved, hallucination and no subject got worse) |
| `--fidelity` | `quant_fidelity.py` (`fidelity.json`) | the quant *keeps*, judged again here with the preset's limits |
| `--adapt-manifest` with `--repo` | `repo_adapt.py run` | the adapter is current (no core interface changed) and within `--max-adapter-mb` |

The verdict:
- **accept**: every check given passed (exit 0);
- **review**: none failed, but something needs a person, such as an inconclusive retrain, a quant past a limit
  but not twice past it, or a warning in the adapter's manifest (exit 2);
- **reject**: a check failed (exit 1).

Presets (`--preset`) are quant_fidelity's: `default`, or `strict`, which holds the truth probe within 0.02 AUROC
and tightens the other limits. Any single limit can be overridden (`--max-auc-drop 0.03`). `--require` names
checks that must be present, so a pipeline can't pass by leaving a report out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_fidelity as QF  # noqa: E402

CHECKS = ("retrain", "fidelity", "adapter")
ORDER = {"pass": 0, "review": 1, "fail": 2}


def check_retrain(g: dict) -> dict:
    v = g.get("verdict")
    status = {"keep": "pass", "inconclusive": "review", "reject": "fail"}.get(v, "fail")
    why = list(g.get("reasons") or [])
    if v not in ("keep", "inconclusive", "reject"):
        why = [f"not a retrain_gate report (verdict {v!r})"]
    return {"status": status, "verdict": v, "reasons": why}


def check_fidelity(rep: dict, quant: str | None, t: dict) -> dict:
    qs = rep.get("quants") or {}
    if quant is None:
        if len(qs) != 1:
            return {"status": "fail", "reasons": [f"the report has {len(qs)} quants: name one with --quant "
                                                  f"({', '.join(qs) or 'none'})"]}
        quant = next(iter(qs))
    if quant not in qs:
        return {"status": "fail", "reasons": [f"no quant {quant!r} in the report ({', '.join(qs)})"]}
    row = {k: v for k, v in qs[quant].items() if k not in ("verdict", "reasons")}
    measured = [k for k in ("next_token", "knowledge", "hneurons", "truth") if k in row]
    if not measured:
        return {"status": "fail", "quant": quant, "reasons": ["nothing was measured for this quant"]}
    v, why = QF.verdict(row, t)
    status = {"keeps": "pass", "check": "review", "loses": "fail"}[v]
    nt = row.get("next_token") or {}
    if nt.get("retried_after"):
        why = why + [f"llama-perplexity crashed once and was run again ({nt['retried_after'][0][:120]})"]
        status = max(status, "review", key=ORDER.get)
    return {"status": status, "quant": quant, "verdict": v, "measured": measured, "reasons": why}


def check_adapter(manifest: dict, repo: str, adapter: str | None, max_mb: float | None) -> dict:
    import repo_adapt as RA
    c = RA.check(manifest, repo, adapter, max_mb)
    why, status = [], "pass"
    if not c["fresh"]:
        status = "fail"
        parts = [f"{k}: " + ", ".join(c[k][:5]) for k in ("changed", "removed", "added") if c[k]]
        why.append("the repository's core interfaces changed since the adapter was trained (" + "; ".join(parts) + ")")
    if c.get("adapter_ok") is False:
        status = "fail"
        why.append(f"the adapter is {c['adapter_mb']} MB, over its {max_mb} MB budget")
    for w in manifest.get("warnings") or []:
        why.append("manifest: " + w)
        status = max(status, "review", key=ORDER.get)
    if manifest.get("dirty"):
        why.append("the data came from a working tree with uncommitted changes")
        status = max(status, "review", key=ORDER.get)
    return {"status": status, **{k: c[k] for k in ("fresh", "commit_then", "commit_now")},
            **({"adapter_mb": c["adapter_mb"]} if "adapter_mb" in c else {}), "reasons": why}


def decide(results: dict, require: list[str]) -> tuple[str, list[str]]:
    reasons = [f"{name}: {r}" for name, c in results.items() for r in c["reasons"] if c["status"] != "pass"]
    missing = [r for r in require if r not in results]
    if missing:
        return "reject", [f"required but not given: {', '.join(missing)}"] + reasons
    if not results:
        return "reject", ["nothing to judge: give at least one report"]
    worst = max((c["status"] for c in results.values()), key=ORDER.get)
    return {"pass": "accept", "review": "review", "fail": "reject"}[worst], reasons


def markdown(g: dict) -> str:
    L = [f"# Acceptance gate: **{g['verdict']}**", "",
         f"Preset `{g['preset']}`; truth-probe AUROC drop at most {g['limits']['max_auc_drop']}, KL divergence at "
         f"most {g['limits']['max_kld']}, TestQA answers lost at most {g['limits']['max_lost']:.0%}.", "",
         "| Check | Result | Why |", "|---|---|---|"]
    for name, c in g["checks"].items():
        L.append(f"| {name} | {c['status']} | {'; '.join(c['reasons']) or '-'} |")
    for name in CHECKS:
        if name not in g["checks"]:
            L.append(f"| {name} | not given | |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(QF.PRESETS), default="default")
    p.add_argument("--retrain", help="retrain_gate.py's JSON report")
    p.add_argument("--fidelity", help="quant_fidelity.py's fidelity.json")
    p.add_argument("--quant", help="which quant in the fidelity report (needed if it has several)")
    p.add_argument("--adapt-manifest", help="repo_adapt.py's manifest.json")
    p.add_argument("--repo", help="the repository, for the adapter check")
    p.add_argument("--adapter", help="the adapter file, for the size check")
    p.add_argument("--max-adapter-mb", type=float)
    p.add_argument("--require", action="append", default=[], choices=list(CHECKS),
                   help="a check that must be given (repeatable)")
    for k in QF.DEFAULTS:
        p.add_argument("--" + k.replace("_", "-"), type=float, help=f"override the preset's {k}")
    p.add_argument("--out", help="folder for accept.json and accept.md")
    a = p.parse_args(argv)
    t = QF.limits(a.preset, **{k: getattr(a, k) for k in QF.DEFAULTS})

    results = {}
    if a.retrain:
        results["retrain"] = check_retrain(json.loads(Path(a.retrain).read_text()))
    if a.fidelity:
        results["fidelity"] = check_fidelity(json.loads(Path(a.fidelity).read_text()), a.quant, t)
    if a.adapt_manifest:
        if not a.repo:
            p.error("--adapt-manifest needs --repo")
        results["adapter"] = check_adapter(json.loads(Path(a.adapt_manifest).read_text()), a.repo, a.adapter,
                                           a.max_adapter_mb)
    verdict, reasons = decide(results, a.require)
    g = {"schema": "omnicortex-accept-gate/1", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": verdict,
         "preset": a.preset, "limits": t, "reasons": reasons, "checks": results,
         "inputs": {k: v for k, v in {"retrain": a.retrain, "fidelity": a.fidelity, "quant": a.quant,
                                      "adapt_manifest": a.adapt_manifest, "adapter": a.adapter}.items() if v}}
    md = markdown(g)
    print(md)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "accept.json").write_text(json.dumps(g, indent=1))
        (out / "accept.md").write_text(md)
    return {"accept": 0, "reject": 1, "review": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
