#!/usr/bin/env python3
"""
Stage 4: CETT activation extraction.

Replaces scripts/extract_activations.py from the upstream repo. See oc_common.py
for the substantive differences (device-safe hooks, text-layer filtering, exact
sequence construction, think-block-aware region indexing).

    python scripts/extract_activations.py \
        --model_path Qwen/Qwen3-8B \
        --input_path data/answer_tokens.jsonl \
        --ids_path data/train_qids.json \
        --output_root data/activations \
        --gpu_mem 14GiB --cpu_mem 40GiB \
        --locations answer_tokens all_except_answer_tokens
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import CETTManager, build_sequence, find_regions, load_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--input_path", required=True, help="answer_tokens.jsonl")
    p.add_argument("--ids_path", required=True, help="train_qids.json / test_qids.json")
    p.add_argument("--output_root", required=True)
    p.add_argument("--locations", nargs="+", default=["answer_tokens"],
                   choices=["input", "output", "answer_tokens",
                            "all_except_answer_tokens"])
    p.add_argument("--method", choices=["mean", "max"], default="mean")
    p.add_argument("--gpu_mem", default=None, help="e.g. 14GiB. Omit for CPU only.")
    p.add_argument("--cpu_mem", default="40GiB")
    p.add_argument("--max_tokens", type=int, default=2048,
                   help="Skip longer samples. Transient RAM is "
                        "layers * tokens * intermediate * 4 bytes.")
    p.add_argument("--keep_think", action="store_true",
                   help="Search for answer tokens from the start of the response "
                        "instead of after </think>.")
    p.add_argument("--no_trust_remote_code", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(
        args.model_path, args.gpu_mem, args.cpu_mem,
        trust_remote_code=not args.no_trust_remote_code,
    )
    mgr = CETTManager(model)
    entry_device = next(model.parameters()).device
    print(f"hooked {mgr.n_layers} layers x {mgr.n_neurons} neurons "
          f"= {mgr.n_layers * mgr.n_neurons:,} features")

    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, "neuron_index.json"), "w") as f:
        json.dump({
            "n_layers": mgr.n_layers,
            "n_neurons": mgr.n_neurons,
            "order": "flat = layer * n_neurons + neuron",
            "model_path": args.model_path,
        }, f, indent=2)

    with open(args.ids_path) as f:
        ids = json.load(f)
    targets = set(ids["t"] + ids["f"])

    for loc in args.locations:
        os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

    with open(args.input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    skipped = {"not_target": 0, "too_long": 0, "no_answer_span": 0, "empty": 0}
    written = 0

    for sample in tqdm(samples, desc="extracting"):
        qid = next(iter(sample))
        if qid not in targets:
            skipped["not_target"] += 1
            continue
        data = sample[qid]

        full_ids, prompt_len = build_sequence(
            tokenizer, data["question"], data["response"]
        )
        if len(full_ids) > args.max_tokens:
            skipped["too_long"] += 1
            continue

        mgr.clear()
        with torch.no_grad():
            model(full_ids.unsqueeze(0).to(entry_device))
        cett = mgr.cett()

        regions = find_regions(
            tokenizer, full_ids, prompt_len,
            data.get("answer_tokens", []), skip_think=not args.keep_think,
        )
        if regions["answer_tokens"] is None:
            skipped["no_answer_span"] += 1

        wrote_any = False
        for loc in args.locations:
            if loc == "all_except_answer_tokens":
                span = regions["answer_tokens"]
                if span is None:
                    continue
                s, e = span
                out_start = regions["output"][0]
                sel = torch.cat([cett[:, out_start:s, :], cett[:, e:, :]], dim=1)
            else:
                span = regions[loc]
                if span is None or span[1] <= span[0]:
                    continue
                sel = cett[:, span[0]:span[1], :]

            if sel.shape[1] == 0:
                continue
            agg = sel.mean(dim=1) if args.method == "mean" else sel.max(dim=1)[0]
            np.save(
                os.path.join(args.output_root, loc, f"act_{qid}.npy"),
                agg.numpy().astype(np.float16),
            )
            wrote_any = True

        if wrote_any:
            written += 1
        else:
            skipped["empty"] += 1

    mgr.remove()
    print(f"wrote {written} samples; skipped {skipped}")


if __name__ == "__main__":
    main()
