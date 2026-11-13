#!/usr/bin/env python3
"""
Stage 2/3 bridge: turn consistency_samples.jsonl into answer_tokens.jsonl.

The upstream repo uses GPT-4o to tag which tokens in a response are the factual
answer. That is an API bill for something we already know: the collector stored
the post-</think> answer string, so we just tokenise it and record the pieces.

Falls back to an empty answer_tokens list when the answer cannot be located,
which is fine as long as you extract the "output" location too.

    python scripts/make_answer_tokens.py \
        --input_path data/consistency_samples.jsonl \
        --output_path data/answer_tokens.jsonl \
        --model_path Qwen/Qwen3-8B
"""
import argparse, json
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_path", required=True)
    try:
        from oc_cli import add_trust_remote_code_argument
    except ImportError:
        from scripts.oc_cli import add_trust_remote_code_argument
    add_trust_remote_code_argument(p)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(
        a.model_path, trust_remote_code=a.trust_remote_code)

    tagged = untagged = 0
    with open(a.input_path, encoding="utf-8") as fin, \
         open(a.output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="tagging"):
            rec = json.loads(line)
            qid = next(iter(rec))
            d = rec[qid]
            answer = (d.get("answer") or "").strip()
            pieces = []
            if answer:
                ids = tok(answer, add_special_tokens=False)["input_ids"]
                pieces = [tok.decode([i]) for i in ids]
            # Count what was produced, not what was attempted: a tokenizer
            # missing its vocab returns nothing for every string, and counting
            # attempts reported that as total success.
            if pieces:
                tagged += 1
            else:
                untagged += 1
            fout.write(json.dumps({qid: {
                "question": d["question"],
                "response": d["response"],
                "answer_tokens": pieces,
                "judge": d["judge"],
                # carried through so this file is also usable as an eval set
                "answer": d.get("answer"),
                "aliases": d.get("aliases", []),
                "task": d.get("task"),
            }}, ensure_ascii=False) + "\n")
    print(f"tagged {tagged}, left empty {untagged}")
    if tagged == 0 and untagged:
        raise SystemExit(
            "\nno answer produced any tokens -- the tokenizer is not loading. "
            "Check the --model_path directory has tokenizer.json or "
            "tokenizer.model, not just config.json. Or skip this stage: "
            "extract_activations_gguf.py finds spans from the answer string.")

if __name__ == "__main__":
    main()
