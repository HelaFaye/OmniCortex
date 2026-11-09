#!/usr/bin/env python3
"""
Tune the suppression scale and save the result as a profile.

Suppressing H-Neurons trades hallucinations for abstentions and, past some
point, for general capability. This sweeps the scale and measures all three so
you pick a point rather than guess one:

  correct     answer matches a gold alias
  abstained   model declined instead of answering
  wrong       confidently wrong: the thing you are trying to reduce
  canary ppl  perplexity on held-out text, as a capability damage proxy

A good profile lowers `wrong` substantially while `canary ppl` stays flat.
If perplexity climbs, you selected too many neurons: retrain the classifier
with a lower --C rather than reaching for a gentler scale.

    python scripts/tune_scale.py \
        --model_path Qwen/Qwen3-8B \
        --h_neurons models/h_neurons.json \
        --eval_path data/consistency_samples.jsonl \
        --scales 1.0 0.5 0.25 0.1 0.0 \
        --n_eval 50 --gpu_mem 14GiB \
        --config_name trivia-q8 --save
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import THINK_CLOSE, load_model  # noqa: E402
from profiles import Profile, SuppressionHandle, fingerprint  # noqa: E402

# Deliberately generic. Anything matching counts as a decline, not a wrong
# answer, which is the distinction the whole sweep turns on.
ABSTAIN_MARKERS = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "not certain", "cannot determine", "can't determine", "no information",
    "unable to answer", "unsure", "i don't have", "i do not have",
]

CANARY = (
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. It generates most of the cell's supply of adenosine "
    "triphosphate, used as a source of chemical energy. In addition to "
    "supplying cellular energy, mitochondria are involved in signaling, "
    "cellular differentiation, and cell death, as well as maintaining control "
    "of the cell cycle and cell growth."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--h_neurons", required=True)
    p.add_argument("--eval_path", required=True,
                   help="jsonl from collect_responses_lmstudio.py")
    p.add_argument("--scales", nargs="+", type=float,
                   default=[1.0, 0.5, 0.25, 0.1, 0.0])
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--canary_file", help="Text file for the perplexity probe")
    p.add_argument("--gpu_mem", default=None)
    p.add_argument("--cpu_mem", default="40GiB")
    p.add_argument("--config_name", default="default")
    p.add_argument("--profiles_root", default="profiles")
    p.add_argument("--save", action="store_true",
                   help="Write the best scale as a profile")
    p.add_argument("--no_trust_remote_code", action="store_true")
    return p.parse_args()


def strip_think(text):
    return text.split(THINK_CLOSE, 1)[1].strip() if THINK_CLOSE in text \
        else text.strip()


def classify(answer, aliases):
    low = answer.lower()
    if not low.strip():
        return "abstained"
    if any(m in low for m in ABSTAIN_MARKERS):
        return "abstained"
    norm = "".join(c if c.isalnum() or c.isspace() else " " for c in low)
    norm = " ".join(norm.split())
    for a in aliases:
        an = "".join(c if c.isalnum() or c.isspace() else " " for c in a.lower())
        an = " ".join(an.split())
        if an and an in norm:
            return "correct"
    return "wrong"


@torch.no_grad()
def canary_ppl(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


@torch.no_grad()
def run_eval(model, tokenizer, cases, device, max_new_tokens):
    counts = {"correct": 0, "abstained": 0, "wrong": 0}
    for case in tqdm(cases, leave=False, desc="  generating"):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["question"]}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        counts[classify(strip_think(text), case["aliases"])] += 1
    return counts


def main():
    args = parse_args()

    with open(args.h_neurons) as f:
        spec = json.load(f)

    cases, skipped_no_gold = [], 0
    with open(args.eval_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            d = next(iter(rec.values()))
            aliases = d.get("aliases") or []
            if not aliases:
                skipped_no_gold += 1
                continue
            cases.append({"question": d["question"], "aliases": aliases})
            if len(cases) >= args.n_eval:
                break
    if not cases:
        raise SystemExit(
            f"no evaluable cases ({skipped_no_gold} entries had no gold "
            "aliases). Re-run collect_responses_lmstudio.py: versions before "
            "the alias fix did not persist them."
        )
    if skipped_no_gold:
        print(f"skipped {skipped_no_gold} entries with no gold aliases")
    print(f"evaluating on {len(cases)} questions")

    model, tokenizer = load_model(
        args.model_path, args.gpu_mem, args.cpu_mem,
        trust_remote_code=not args.no_trust_remote_code,
    )
    device = next(model.parameters()).device
    fp, geom = fingerprint(model.config)

    canary = CANARY
    if args.canary_file:
        with open(args.canary_file) as f:
            canary = f.read()

    profile = Profile.create(
        fp, geom, getattr(model.config, "_name_or_path", args.model_path),
        spec["by_layer"], scale=1.0,
        n_layers=spec["n_layers"], n_neurons=spec["n_neurons"],
        config_name=args.config_name,
        provenance={"h_neurons": os.path.abspath(args.h_neurons),
                    "total_neurons": spec.get("total")},
    )

    handle = SuppressionHandle(model, profile, scale=1.0)
    print(handle.summary())

    results = []
    print(f"\n{'scale':>7} {'correct':>8} {'abstain':>8} {'wrong':>7} "
          f"{'canary ppl':>11}")
    print("-" * 46)
    for scale in args.scales:
        handle.set_scale(scale)
        ppl = canary_ppl(model, tokenizer, canary, device)
        counts = run_eval(model, tokenizer, cases, device, args.max_new_tokens)
        n = sum(counts.values())
        rec = {"scale": scale, "ppl": ppl, "n": n, **counts}
        results.append(rec)
        print(f"{scale:>7.2f} {counts['correct'] / n:>8.1%} "
              f"{counts['abstained'] / n:>8.1%} {counts['wrong'] / n:>7.1%} "
              f"{ppl:>11.3f}")

    handle.remove()

    has_unity = any(r["scale"] == 1.0 for r in results)
    base = next(r for r in results if r["scale"] == 1.0) if has_unity \
        else results[0]
    if not has_unity:
        print("\nwarning: 1.0 was not in --scales, so there is no unsuppressed "
              f"baseline; comparing against scale {base['scale']} instead.")

    print(f"\nrelative to scale {base['scale']}"
          f"{' (no suppression)' if has_unity else ''}:")
    for r in results:
        if r["scale"] == base["scale"]:
            continue
        dw = (r["wrong"] - base["wrong"]) / base["n"]
        dppl = (r["ppl"] - base["ppl"]) / base["ppl"]
        flag = "  <-- capability damage" if dppl > 0.05 else ""
        print(f"  scale {r['scale']:.2f}: wrong {dw:+.1%}, "
              f"canary ppl {dppl:+.1%}{flag}")

    # Pick the scale that removes the most wrong answers while keeping the
    # canary within 5%. Deliberately conservative: a model that abstains on
    # everything scores well on `wrong` and is useless.
    viable = [r for r in results
              if r["ppl"] <= base["ppl"] * 1.05 and r["scale"] != base["scale"]]
    if viable:
        best = min(viable, key=lambda r: r["wrong"])
        if best["wrong"] >= base["wrong"]:
            print("\nNo scale reduced wrong answers without capability damage.")
            best = None
    else:
        print("\nEvery scale damaged the canary. Retrain with a lower --C.")
        best = None

    if best:
        print(f"\nsuggested scale: {best['scale']} "
              f"(wrong {base['wrong'] / base['n']:.1%} -> "
              f"{best['wrong'] / best['n']:.1%}, "
              f"abstain {base['abstained'] / base['n']:.1%} -> "
              f"{best['abstained'] / best['n']:.1%})")

    if args.save:
        profile.data["scale"] = best["scale"] if best else 1.0
        profile.add_evaluation({
            "eval_path": os.path.abspath(args.eval_path),
            "n_eval": len(cases),
            "sweep": results,
            "selected": best["scale"] if best else None,
        })
        path = profile.save(args.profiles_root)
        print(f"\nwrote {path}")
        if not best:
            print("saved at scale 1.0 (inert). Fix the neuron set before using it.")


if __name__ == "__main__":
    main()
