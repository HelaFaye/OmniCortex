#!/usr/bin/env python3
"""
Turn collected pairs into an abstention-tuning dataset.

OmniCortex induces abstention by scaling neurons down. A small LoRA can induce it
directly, and your consistency-filtered data is already the right shape for
that: questions the model got right every time become "answer it", questions it
got wrong every time become "decline". That is the same behavioural target,
reached by training rather than by clamping, and it is far more precise --
gradient descent can shape when to decline, whereas a scalar on a neuron column
cannot.

Running both and comparing on the same held-out set is the experiment worth
doing. They are not redundant: OmniCortex needs no GPU and no training, the LoRA
needs a rented 24GB card for an afternoon. If the LoRA wins clearly, OmniCortex
was scaffolding. If it does not, that is a real finding about how much of this
behaviour is localised in those neurons.

Output is the `messages` format that TRL, Axolotl and LLaMA-Factory all read.

HELD-OUT DISCIPLINE. Pass --exclude with the qid files used for anything you
will report on. Training on questions you later evaluate against is the same
mistake as tuning on observed failures, and it produces the same flattering,
meaningless number.

    python scripts/export_sft_dataset.py \\
        --input_path data/consistency_samples.jsonl \\
        --output_path data/abstention_sft.jsonl \\
        --exclude data/test_qids.json \\
        --val_fraction 0.1
"""

import argparse
import json
import random

# Varied so the model learns the behaviour rather than one string. Kept plain:
# elaborate hedging teaches verbosity, not calibration.
ABSTENTIONS = [
    "I don't know.",
    "I'm not sure about this one.",
    "I don't know the answer to that.",
    "I'm not confident enough to answer that.",
    "I don't have a reliable answer for that.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="qid json files to hold out (test_qids.json etc.)")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap per class. Classes are balanced regardless.")
    p.add_argument("--keep_think", action="store_true",
                   help="Keep the reasoning block in the target. Off by "
                        "default: training on a decline should not also train "
                        "a long rationalisation for it.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


THINK_CLOSE = "</think>"


def strip_think(t):
    return t.split(THINK_CLOSE, 1)[1].strip() if THINK_CLOSE in t else t.strip()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    held_out = set()
    for path in args.exclude:
        with open(path) as f:
            ids = json.load(f)
        held_out |= set(ids.get("t", [])) | set(ids.get("f", []))
    if held_out:
        print(f"holding out {len(held_out)} qids from {len(args.exclude)} file(s)")
    else:
        print("warning: nothing held out. Anything you evaluate on later will "
              "have been trained on.")

    keep, decline, skipped = [], [], 0
    with open(args.input_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = next(iter(rec))
            if qid in held_out:
                skipped += 1
                continue
            d = rec[qid]
            q = d["question"]
            if d.get("judge") == "true":
                target = d["response"] if args.keep_think \
                    else strip_think(d["response"])
                if target.strip():
                    keep.append((qid, q, target))
            elif d.get("judge") == "false":
                decline.append((qid, q, None))

    n = min(len(keep), len(decline))
    if args.max_pairs:
        n = min(n, args.max_pairs)
    if n == 0:
        raise SystemExit(
            f"nothing usable ({len(keep)} correct, {len(decline)} hallucinated "
            f"available). Collect more before training.")
    print(f"{len(keep)} correct, {len(decline)} hallucinated available; "
          f"using {n} of each")
    if n < 200:
        print("  note: under ~200 per class a LoRA will mostly learn to decline "
              "everything.\n  Collect more before spending GPU time.")

    rows = []
    for qid, q, target in rng.sample(keep, n):
        rows.append({"qid": qid, "label": "answer", "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": target}]})
    for qid, q, _ in rng.sample(decline, n):
        rows.append({"qid": qid, "label": "decline", "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": rng.choice(ABSTENTIONS)}]})
    rng.shuffle(rows)

    n_val = int(len(rows) * args.val_fraction)
    val, train = rows[:n_val], rows[n_val:]

    def dump(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(args.output_path, train)
    val_path = args.output_path.replace(".jsonl", ".val.jsonl")
    dump(val_path, val)

    print(f"\nwrote {len(train)} train -> {args.output_path}")
    print(f"      {len(val)} val   -> {val_path}")
    print(f"      {skipped} rows skipped as held out")
    print("""
Both classes matter. Training only on declines teaches declining; the correct
answers are what stops the model declining everything. That balance is why the
counts above are equalised.

Then, on a rented 24GB card (this will not run on a Vega iGPU or a 16GB card):

  pip install trl peft transformers datasets bitsandbytes
  python -m trl.scripts.sft \\
      --model_name_or_path Qwen/Qwen3-8B \\
      --dataset_name json --dataset_train_split train \\
      --use_peft --lora_r 16 --lora_alpha 32 \\
      --load_in_4bit --learning_rate 1e-4 --num_train_epochs 2 \\
      --output_dir out/abstention-lora

Then evaluate it against OmniCortex on the same held-out set with
eval_code_hallucination.py. That comparison is the point of the exercise.""")


if __name__ == "__main__":
    main()
