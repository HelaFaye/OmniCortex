#!/usr/bin/env python3
"""
Did a quantization keep what the model knows? Compares each quantized GGUF with its reference (the original,
or the highest-precision file) on four counts, each optional, and gives a verdict per quant:

  1. next token   llama.cpp's KL divergence and "same top token" rate (llama-perplexity --kl-divergence); without
                  that tool, how much of each greedy answer to a fixed prompt set matches the reference
  2. knowledge    TestQA on both: which questions the reference gets right and the quant does not, split into
                  confident mistakes (now answered wrongly) and abstentions (now declined)
  3. H-neurons    the same texts profiled on both models: how similar each text's MLP neuron activity (CETT)
                  stays, layer by layer, and, with the model's H-neuron classifier, how far its risk score moves
                  and how many flags flip
  4. truth probe  a probe trained on the reference (truth_probe.py --gguf), scored on the quant: does its AUROC
                  hold on the same labelled rows?

    python scripts/quant_fidelity.py --ref models/m-F16.gguf --quant models/m-Q4_K_M.gguf --quant models/m-IQ3_XS.gguf \\
        --testqa qa/bank --testqa-limit 200 --classifier models/m.classifier.npz \\
        --truth-probe models/m.truth.npz --truth-data runs/truth-heldout.jsonl --out runs/fidelity-m

Models run one at a time, so a quant and its reference never have to fit in memory together. Results are cached
per model in --out, so an interrupted run resumes. The verdict thresholds are options (--max-kld, --min-same-top,
--max-lost, --min-cett-cos, --max-flag-flips, --max-auc-drop). "keeps" means every measured count is within them;
"check", that one is past its limit; "loses", that one is past twice its limit (the exit status is then 1).

Use truth rows the probe was not trained on: both AUROCs are measured on the same rows, so the drop is fair
either way, but held-out rows say how the probe will do on new replies.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULTS = {"max_kld": 0.1, "min_same_top": 85.0, "min_agreement": 0.8, "max_lost": 0.03, "min_cett_cos": 0.95, "max_flag_flips": 0.05,
            "max_auc_drop": 0.05}
# Limit presets. "strict" is for a model whose own instruments must keep working (the truth probe within 0.02
# AUROC, from the 2026-11-23 spec) and for small models, which lose more to the same quant type.
PRESETS = {"default": dict(DEFAULTS),
           "strict": {"max_kld": 0.05, "min_same_top": 90.0, "min_agreement": 0.9, "max_lost": 0.02,
                      "min_cett_cos": 0.97, "max_flag_flips": 0.03, "max_auc_drop": 0.02}}


def limits(preset: str = "default", **override) -> dict:
    """A preset's limits, with any given (non-None) values in place of its own."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}: one of {', '.join(PRESETS)}")
    return {**PRESETS[preset], **{k: v for k, v in override.items() if k in DEFAULTS and v is not None}}


def label_of(path: str) -> str:
    return Path(path).stem


# ---------------------------------------------------------------- a llama-server for one model at a time

@contextlib.contextmanager
def serve(server: str, model: str, ngl: int, timeout: float = 300, ctx: int = 4096):
    """llama-server for MODEL on a free loopback port -> its base URL; stopped on exit."""
    import engine_optimizer
    port = engine_optimizer.free_port()
    p = subprocess.Popen([server, "-m", model, "-c", str(ctx), "-ngl", str(ngl), "--port", str(port),
                          "--host", "127.0.0.1", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        end = time.time() + timeout
        while True:
            if p.poll() is not None or time.time() > end:
                raise RuntimeError(f"llama-server did not start for {Path(model).name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        yield f"http://127.0.0.1:{port}"
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()


# ---------------------------------------------------------------- 1. next token

def next_token(a, out: Path) -> dict:
    """{label: {kld, same_top} or {agreement, exact}} for every quant, against the reference (cached per quant)."""
    cache = out / "next-token.json"
    res = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [q for q in a.quant if label_of(q) not in res or "error" in res[label_of(q)]]
    if todo and not a.skip_next_token:
        res.update(_next_token(a, out, todo))
        cache.write_text(json.dumps(res, indent=1))
    return {label_of(q): res[label_of(q)] for q in a.quant if label_of(q) in res}


def _next_token(a, out: Path, quants: list[str]) -> dict:
    import quant_select as qs
    res = {}
    tool = a.perplexity or qs.find_tool(a.server, "llama-perplexity")
    if tool:
        text = qs.eval_text(a.eval_text, out)
        for q in quants:
            try:
                res[label_of(q)] = qs.kld(tool, a.ref, q, text, out, a.ngl, a.chunks)
            except (RuntimeError, OSError) as e:
                res[label_of(q)] = {"error": str(e)}
        return res
    ref = qs.answers(a.server, a.ref, a.ngl, qs.PROMPTS, 64, a.timeout)
    for q in quants:
        res[label_of(q)] = qs.agreement(ref, qs.answers(a.server, q, a.ngl, qs.PROMPTS, 64, a.timeout))
    return res


# ---------------------------------------------------------------- 2. knowledge (TestQA)

def testqa_rows(a, model: str, out: Path) -> list[dict]:
    """TestQA on one model through its own llama-server -> testqa's per-item rows (cached)."""
    import testqa
    label = label_of(model)
    path = out / f"testqa-{label}.json"
    if not path.exists():
        with serve(a.server, model, a.ngl, a.timeout) as url:
            args = ["--endpoint", f"{label}={url}/v1", "--tasks", *a.testqa, "--out", str(path),
                    "--cache", str(out / "testqa-cache"), "--max_tokens", str(a.max_tokens), "--concurrency", "1"]
            if a.testqa_limit:
                args += ["--per-subject", str(a.testqa_limit)] if a.per_subject else ["--limit", str(a.testqa_limit)]
            if testqa.main(args) != 0:
                raise RuntimeError(f"TestQA failed on {label}")
    return json.loads(path.read_text())["raw"][label]


def knowledge(ref_rows: list[dict], quant_rows: list[dict]) -> dict:
    """What the quant lost and gained against the reference, item by item (canaries and errors left out)."""
    import testqa
    good = testqa.GOOD
    q = {r["id"]: r for r in quant_rows}
    n = mistakes = abstains = gained = 0
    examples = []
    for r in ref_rows:
        x = q.get(r["id"])
        if x is None or r.get("kind") == "canary" or {"error", "skipped"} & {r["verdict"], x["verdict"]}:
            continue
        n += 1
        if r["verdict"] in good and x["verdict"] not in good:
            if x["verdict"] == "abstained":
                abstains += 1
            else:
                mistakes += 1
                if len(examples) < 10:
                    examples.append({"id": r["id"], "subject": r.get("subject"), "now": x["verdict"],
                                     "answer": str(x.get("text", ""))[:160]})
        elif r["verdict"] not in good and x["verdict"] in good:
            gained += 1
    ref_good = sum(1 for r in ref_rows if r["verdict"] in good)
    from model_stats import wilson
    lost = mistakes + abstains
    return {"n": n, "reference_correct": ref_good, "confident_mistakes": mistakes, "now_abstains": abstains,
            "gained": gained, "lost_rate": round(lost / n, 4) if n else None,
            # 95% interval: the verdict judges its lower bound, so one flip in a small set is not a finding
            "lost_ci": [round(x, 4) for x in wilson(lost, n)] if n else None,
            "mistake_examples": examples}


# ---------------------------------------------------------------- 3. H-neurons

def texts_for(a, ref_rows: list[dict] | None) -> list[tuple[list[dict], str]]:
    """The texts both models are profiled on: the reference's own TestQA replies, else the truth rows."""
    out = []
    if ref_rows:
        import testqa
        tasks = {t["id"]: t for t in testqa.load_tasks(a.testqa, None, None, 0)}
        for r in ref_rows:
            t = tasks.get(r["id"])
            if t and r.get("text") and r["verdict"] not in ("error", "skipped"):
                out.append(([{"role": "user", "content": testqa.prompt_for(t)}], r["text"][:2000]))
    elif a.truth_data:
        import truth_probe
        for r in truth_probe.build_rows([a.truth_data], borrow=False):
            out.append((truth_probe.messages_for(r["question"]), r["answer"]))
    return out[:a.hneuron_n]


def profiles(a, model: str, texts, out: Path, classifier: str | None) -> dict:
    """CETT [L, N] per text, and the classifier's probability when given (cached as .npz)."""
    path = out / f"cett-{label_of(model)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"cett": list(z["cett"]), "prob": list(z["prob"]) if "prob" in z else None}
    from hscore import HScorer
    h = HScorer(a.cett, model, classifier, ngl=a.ngl)
    cett, prob = [], []
    for msgs, resp in texts:
        p = h.profile(msgs, resp)
        cett.append(p["cett"])
        if "prob" in p:
            prob.append(p["prob"])
    np.savez_compressed(path, cett=np.stack(cett), **({"prob": np.array(prob)} if prob else {}))
    return {"cett": cett, "prob": prob or None}


def cos(x, y) -> float:
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d else 1.0


def hneurons(ref: dict, quant: dict, threshold: float = 0.5) -> dict:
    """How far the quant's neuron activity drifts from the reference's on the same texts."""
    whole = [cos(r, q) for r, q in zip(ref["cett"], quant["cett"])]
    L = np.asarray(ref["cett"][0]).shape[0]
    per_layer = [round(float(np.mean([cos(r[i], q[i]) for r, q in zip(ref["cett"], quant["cett"])])), 4)
                 for i in range(L)]
    out = {"texts": len(whole), "cett_cos_mean": round(float(np.mean(whole)), 4),
           "cett_cos_min": round(float(np.min(whole)), 4), "cett_cos_per_layer": per_layer,
           "weakest_layer": int(np.argmin(per_layer))}
    if ref.get("prob") is not None and quant.get("prob") is not None:
        pr, pq = np.asarray(ref["prob"]), np.asarray(quant["prob"])
        flips = int(np.sum((pr >= threshold) != (pq >= threshold)))
        out.update(risk_delta_mean=round(float(np.mean(np.abs(pq - pr))), 4),
                   risk_delta_max=round(float(np.max(np.abs(pq - pr))), 4),
                   flag_flips=flips, flag_flip_rate=round(flips / len(pr), 4),
                   flagged_ref=int(np.sum(pr >= threshold)), flagged_quant=int(np.sum(pq >= threshold)))
    return out


# ---------------------------------------------------------------- 4. truth probe

def auroc(scores, labels) -> float | None:
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None
    return round(float(roc_auc_score(labels, scores)), 4)


def truth_scores(a, model: str, probe, rows, out: Path) -> dict:
    """The probe's probability on every row at every stage, read from MODEL (cached)."""
    path = out / f"truth-{label_of(model)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"p": z["p"], "y": z["y"]}
    import truth_probe
    feat = truth_probe.GGUFFeatures(model, a.cett, ngl=a.ngl)
    stages, X, y = truth_probe.featurize(feat, rows, probe.meta.get("pool", "last"),
                                         framing=probe.meta.get("framing", "answer"))
    if list(stages) != list(probe.stages):
        raise ValueError(f"{Path(model).name} has different layers from the probe's model")
    p = np.array([[probe.p(i, x[i]) for i in range(len(stages))] for x in X], np.float32)   # [rows, stages]
    np.savez_compressed(path, p=p, y=y)
    return {"p": p, "y": y}


def truth_transfer(probe, ref: dict, quant: dict) -> dict:
    b = probe.best
    out = {"stage": probe.stages[b], "rows": int(len(ref["y"])),
           "auroc_ref": auroc(ref["p"][:, b], ref["y"]), "auroc_quant": auroc(quant["p"][:, b], quant["y"])}
    if out["auroc_ref"] is not None and out["auroc_quant"] is not None:
        out["auroc_drop"] = round(out["auroc_ref"] - out["auroc_quant"], 4)
    good = [i for i, v in enumerate(probe.auc) if v >= 0.7]
    if good:
        out["best_stage_on_quant"] = probe.stages[max(good, key=lambda i: auroc(quant["p"][:, i], quant["y"]) or 0)]
    return out


# ---------------------------------------------------------------- verdict and report

def verdict(row: dict, t: dict) -> tuple[str, list[str]]:
    """keeps / check / loses, with the reasons."""
    why, bad = [], []
    nt = row.get("next_token") or {}
    if "kld" in nt and nt["kld"] is not None:
        (bad if nt["kld"] > 2 * t["max_kld"] else why if nt["kld"] > t["max_kld"] else []).append(
            f"KL divergence {nt['kld']:.4f} (limit {t['max_kld']})")
        if nt.get("same_top") is not None and nt["same_top"] < t["min_same_top"]:
            why.append(f"same top token {nt['same_top']:.1f}% (limit {t['min_same_top']}%)")
    elif nt.get("agreement") is not None and nt["agreement"] < t["min_agreement"]:
        why.append(f"greedy agreement {nt['agreement']:.2f} (limit {t['min_agreement']})")
    k = row.get("knowledge") or {}
    low = (k.get("lost_ci") or [k.get("lost_rate")])[0]        # the 95% lower bound, when there is one
    if low is not None and low > t["max_lost"]:
        (bad if low > 2 * t["max_lost"] else why).append(
            f"lost {k['confident_mistakes']} answers to confident mistakes and {k['now_abstains']} to abstentions "
            f"of {k['n']} ({100 * k['lost_rate']:.1f}%, at least {100 * low:.1f}% at 95% confidence; "
            f"limit {100 * t['max_lost']:.1f}%)")
    h = row.get("hneurons") or {}
    if h.get("cett_cos_mean") is not None and h["cett_cos_mean"] < t["min_cett_cos"]:
        why.append(f"H-neuron activity similarity {h['cett_cos_mean']:.3f} (limit {t['min_cett_cos']}); "
                   f"weakest layer {h['weakest_layer']}")
    if h.get("flag_flip_rate") is not None and h["flag_flip_rate"] > t["max_flag_flips"]:
        why.append(f"risk flags flipped on {h['flag_flips']} of {h['texts']} texts")
    tp = row.get("truth") or {}
    if tp.get("auroc_drop") is not None and tp["auroc_drop"] > t["max_auc_drop"]:
        (bad if tp["auroc_drop"] > 2 * t["max_auc_drop"] else why).append(
            f"truth probe AUROC {tp['auroc_ref']:.3f} -> {tp['auroc_quant']:.3f}")
    errors = [f"{k}: {v['error']}" for k, v in row.items() if isinstance(v, dict) and v.get("error")]
    if bad:
        return "loses", bad + why + errors
    if why or errors:
        return "check", why + errors
    return "keeps", []


def markdown(report: dict) -> str:
    L = [f"# Quantization fidelity: {Path(report['reference']).name}", ""]
    L.append("| Quant | Verdict | KLD / same top | Lost (mistakes + abstains) | H-neuron cos | Flag flips | Truth AUROC |")
    L.append("|---|---|---|---|---|---|---|")
    for q, r in report["quants"].items():
        nt, k, h, tp = (r.get(x) or {} for x in ("next_token", "knowledge", "hneurons", "truth"))
        ntx = (f"{nt['kld']:.4f} / {nt.get('same_top') or 0:.1f}%" if nt.get("kld") is not None else
               f"agreement {nt['agreement']:.2f}" if nt.get("agreement") is not None else "-")
        kx = f"{k['confident_mistakes']} + {k['now_abstains']} of {k['n']}" if k.get("n") else "-"
        hx = f"{h['cett_cos_mean']:.3f}" if h.get("cett_cos_mean") is not None else "-"
        fx = f"{h['flag_flips']}/{h['texts']}" if h.get("flag_flips") is not None else "-"
        tx = (f"{tp['auroc_ref']:.3f} → {tp['auroc_quant']:.3f}" if tp.get("auroc_quant") is not None else "-")
        L.append(f"| {q} | **{r['verdict']}** | {ntx} | {kx} | {hx} | {fx} | {tx} |")
    for q, r in report["quants"].items():
        if r["reasons"]:
            L += ["", f"**{q}**: " + "; ".join(r["reasons"]) + "."]
        ex = (r.get("knowledge") or {}).get("mistake_examples") or []
        if ex:
            L += ["", f"Confident mistakes of {q} (the reference was right):"]
            L += [f"- `{e['id']}` ({e['subject']}): {e['answer']!r}" for e in ex[:5]]
    return "\n".join(L) + "\n"


def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = limits(getattr(a, "preset", "default") or "default", **{k: getattr(a, k, None) for k in DEFAULTS})
    rows = {label_of(q): {} for q in a.quant}

    for lbl, r in next_token(a, out).items():          # --skip-next-token still reports cached results
        rows[lbl]["next_token"] = r

    ref_rows = None
    if a.testqa:
        ref_rows = testqa_rows(a, a.ref, out)
        for q in a.quant:
            try:
                rows[label_of(q)]["knowledge"] = knowledge(ref_rows, testqa_rows(a, q, out))
            except RuntimeError as e:
                rows[label_of(q)]["knowledge"] = {"error": str(e)}

    if a.cett and (ref_rows or a.truth_data):
        texts = texts_for(a, ref_rows)
        if texts:
            ref_p = profiles(a, a.ref, texts, out, a.classifier)
            for q in a.quant:
                rows[label_of(q)]["hneurons"] = hneurons(ref_p, profiles(a, q, texts, out, a.classifier))

    if a.truth_probe and a.truth_data and a.cett:
        import truth_probe
        probe = truth_probe.Probe.load(a.truth_probe)
        data = truth_probe.build_rows([a.truth_data], borrow=False)[:a.truth_n]
        ref_s = truth_scores(a, a.ref, probe, data, out)
        for q in a.quant:
            rows[label_of(q)]["truth"] = truth_transfer(probe, ref_s, truth_scores(a, q, probe, data, out))

    for lbl, r in rows.items():
        r["verdict"], r["reasons"] = verdict(r, t)
    report = {"schema": "omnicortex-quant-fidelity/1", "reference": a.ref, "preset": getattr(a, "preset", "default"), "thresholds": t, "quants": rows}
    (out / "fidelity.json").write_text(json.dumps(report, indent=1))
    (out / "fidelity.md").write_text(markdown(report))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference GGUF: the original, or the highest-precision file")
    p.add_argument("--quant", action="append", required=True, help="a quantized GGUF of the same model (repeat)")
    p.add_argument("--out", default="runs/quant-fidelity")
    llama = os.environ.get("OC_LLAMA", "~/llama.cpp")
    p.add_argument("--server", default=os.path.expanduser(os.environ.get("OC_LLAMA_SERVER")
                                                           or f"{llama}/build/bin/llama-server"))
    p.add_argument("--perplexity", help="llama-perplexity (default: next to llama-server)")
    p.add_argument("--cett", default=os.path.expanduser(os.environ.get("OC_CETT") or f"{llama}/build/bin/llama-cett-dump"),
                   help="llama-cett-dump, for the H-neuron and truth-probe counts")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--timeout", type=float, default=600)
    g = p.add_argument_group("1. next token")
    g.add_argument("--eval-text", help="text for the KL divergence (default: OmniCortex's docs)")
    g.add_argument("--chunks", type=int, default=20, help="512-token chunks of it")
    g.add_argument("--skip-next-token", action="store_true", help="measure nothing new for it (cached results still show)")
    g = p.add_argument_group("2. knowledge (TestQA)")
    g.add_argument("--testqa", nargs="*", default=[], help="TestQA bank files or folders (e.g. qa/bank)")
    g.add_argument("--testqa-limit", type=int, default=200, help="items (0: all)")
    g.add_argument("--per-subject", action="store_true", help="--testqa-limit items per subject instead")
    g.add_argument("--max-tokens", type=int, default=1024)
    g = p.add_argument_group("3. H-neurons")
    g.add_argument("--classifier", help="the reference model's H-neuron classifier (.npz)")
    g.add_argument("--hneuron-n", type=int, default=100, help="texts to profile")
    g = p.add_argument_group("4. truth probe")
    g.add_argument("--truth-probe", help="a probe trained on the reference with truth_probe.py --gguf")
    g.add_argument("--truth-data", help="labelled rows (truth_probe.py data), ideally not the training rows")
    g.add_argument("--truth-n", type=int, default=200)
    g = p.add_argument_group("verdict")
    g.add_argument("--preset", choices=list(PRESETS), default="default",
                   help="limits: default, or strict (truth-probe AUROC drop at most 0.02); options below override it")
    g.add_argument("--max-kld", type=float, help=f"default {DEFAULTS['max_kld']}")
    g.add_argument("--min-same-top", type=float, help=f"percent; default {DEFAULTS['min_same_top']}")
    g.add_argument("--min-agreement", type=float, help=f"default {DEFAULTS['min_agreement']}")
    g.add_argument("--max-lost", type=float, help=f"fraction of TestQA items lost; default {DEFAULTS['max_lost']}")
    g.add_argument("--min-cett-cos", type=float, help=f"default {DEFAULTS['min_cett_cos']}")
    g.add_argument("--max-flag-flips", type=float, help=f"default {DEFAULTS['max_flag_flips']}")
    g.add_argument("--max-auc-drop", type=float, help=f"default {DEFAULTS['max_auc_drop']}")
    a = p.parse_args(argv)
    for f in [a.ref, *a.quant]:
        if not Path(f).exists():
            p.error(f"no such file: {f}")
    report = run(a)
    print(markdown(report))
    print(f"wrote {Path(a.out) / 'fidelity.json'} and fidelity.md")
    return 0 if all(r["verdict"] != "loses" for r in report["quants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
