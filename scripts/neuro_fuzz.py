#!/usr/bin/env python3
"""
Neuro-fuzzing (docs/INTEGRATED_OPTIONS.md): new test items aimed at the model's known weak spots, so evaluation
keeps measuring the weakness rather than a memorised bank.

Seeds, the weak spots:
    --results R.json     TestQA items the model got wrong, and the ones it got right only narrowly (an activation
                         risk at or above --near when the run was scored)
    --harvest            Fix This Mistake corrections
    --review MODEL       observations in the neuron review with a risk at or above --near (their prompts)
A fuzzer model (any OpenAI-compatible URL@model) writes --per-seed new items per seed: the same skill, different
surface (new entities and numbers, false premises, distractors, edge cases), in TestQA's own format (qa with
aliases, reasoning with an answer). Each new item is
    deduplicated  against the bank and against each other (word overlap at or above --dup, 0.9)
    verified      with --verify URL@model: a second model answers it, graded by TestQA's grader; items it agrees
                  with are kept as verified, the rest go to a separate unverified file for you to look at

Writes a bank folder testqa.py reads like qa/bank (--tasks DIR). Item ids start fz-, with their seed recorded,
so results can be traced back to the weakness that produced them.

    python scripts/neuro_fuzz.py --results runs/tq.json --fuzzer http://127.0.0.1:1234/v1@writer \\
        --verify http://gpu:8080/v1@judge --per-seed 5 --out qa/fuzz/round1

Not done: reading "neural coordinates" straight off the model to aim the fuzzer. Seeds are behaviours (wrong or
risky answers); which neurons they involve is what the neuron review and the classifier work out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

PROMPT = """You write test questions for evaluating a language model. The model failed or nearly failed this one:

{seed}

Write {n} NEW questions that test the same skill with different surface details: new names, numbers and
settings; include a false premise, a distractor or an edge case where it fits. Each must have one short,
checkable answer. Do not copy the original. Output a JSON array only, each element either
  {{"kind": "qa", "prompt": "...? Respond with the answer only.", "aliases": ["answer", "other accepted form"]}}
or
  {{"kind": "reasoning", "prompt": "...", "answer": "42", "answer_type": "number"}}   (answer_type: number, choice or text)
For a question with a false premise, the accepted answer explains the premise is false, in a few words."""


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def overlap(a: str, b: str) -> float:
    x, y = words(a), words(b)
    return len(x & y) / max(1, len(x | y))


def chat(endpoint: str, messages: list[dict], max_tokens: int = 1500, temperature: float = 0.8) -> str:
    url, _, model = endpoint.partition("@")
    body = {"model": model or "default", "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_items(text: str) -> list[dict]:
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for x in arr if isinstance(arr, list) else []:
        if not isinstance(x, dict) or not str(x.get("prompt") or "").strip():
            continue
        if x.get("kind") == "qa" and isinstance(x.get("aliases"), list) and x["aliases"]:
            out.append({"kind": "qa", "prompt": str(x["prompt"]).strip(), "aliases": [str(a) for a in x["aliases"]][:6]})
        elif x.get("kind") == "reasoning" and str(x.get("answer") or "").strip():
            at = x.get("answer_type") if x.get("answer_type") in ("number", "choice", "text") else "text"
            out.append({"kind": "reasoning", "prompt": str(x["prompt"]).strip(), "answer": str(x["answer"]).strip(),
                        "answer_type": at})
    return out


def seeds(a) -> list[dict]:
    import testqa
    bank = {t["id"]: t for t in testqa.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)}
    out = []
    for r in a.results or []:
        d = json.loads(Path(r).read_text())
        for rows in (d.get("raw") or {}).values():
            for row in rows:
                t = bank.get(row.get("id"))
                risky = isinstance(row.get("risk"), (int, float)) and row["risk"] >= a.near
                if t and t["kind"] in ("qa", "reasoning") and (row.get("verdict") == "wrong" or risky):
                    out.append({"seed": t["id"], "text": t["prompt"], "subject": t.get("subject"),
                                "why": "wrong" if row.get("verdict") == "wrong" else "risky"})
    if a.harvest:
        import harvest
        for it in harvest.items():
            out.append({"seed": "harvest:" + it["id"], "text": it["context"][-1]["content"] +
                        f"\n(The model answered: {it['rejected'][:300]}\nThe correct answer: {it['chosen'][:300]})",
                        "subject": None, "why": "corrected"})
    if a.review:
        import neuron_review
        st = neuron_review.ReviewStore(a.review_dir)
        for row in st.query(a.review):
            if isinstance(row.get("risk"), (int, float)) and row["risk"] >= a.near and row.get("item"):
                out.append({"seed": f"review:{row['id']}", "text": str(row["item"])[:1500], "subject": (row.get("subjects") or [None])[0],
                            "why": "risky"})
    uniq, seen = [], set()
    for s in out:
        if s["text"] not in seen:
            seen.add(s["text"])
            uniq.append(s)
    return uniq[:a.max_seeds]


def verify(item: dict, endpoint: str) -> tuple[bool, str]:
    import testqa
    reply = chat(endpoint, [{"role": "user", "content": item["prompt"]}], max_tokens=300, temperature=0.0)
    t = dict(item)
    t.setdefault("subject", "factual")
    verdict = testqa.grade(t, reply, argparse.Namespace(allow_exec=False, exec_timeout=5))[0]
    return verdict in ("correct", "answered"), reply


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", action="append")
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--review", help="a model id in the neuron review")
    ap.add_argument("--review-dir")
    ap.add_argument("--near", type=float, default=0.4, help="activation risk that counts as nearly failing")
    ap.add_argument("--fuzzer", required=True, help="URL@model that writes the new items")
    ap.add_argument("--verify", help="URL@model that answers them; only items it gets right count as verified")
    ap.add_argument("--per-seed", type=int, default=5)
    ap.add_argument("--max-seeds", type=int, default=40)
    ap.add_argument("--dup", type=float, default=0.9, help="word overlap that counts as a duplicate (bank questions share boilerplate, so keep it high)")
    ap.add_argument("--out", required=True, help="bank folder to write (testqa.py --tasks DIR)")
    a = ap.parse_args(argv)
    import testqa
    bank = [t["prompt"] for t in testqa.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)]
    sd = seeds(a)
    if not sd:
        raise SystemExit("no seeds: give --results with failures, --harvest or --review")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, unverified, made, dups = [], [], 0, 0
    for s in sd:
        try:
            items = parse_items(chat(a.fuzzer, [{"role": "user", "content": PROMPT.format(seed=s["text"], n=a.per_seed)}]))
        except Exception as e:
            print(f"fuzzer failed on {s['seed']}: {e}", file=sys.stderr)
            continue
        for it in items[:a.per_seed]:
            made += 1
            if any(overlap(it["prompt"], b) >= a.dup for b in bank + [k["prompt"] for k in kept + unverified]):
                dups += 1
                continue
            it["id"] = "fz-" + hashlib.sha1(it["prompt"].encode()).hexdigest()[:10]
            it["subject"] = s.get("subject") or "factual"
            it["fuzz"] = {"seed": s["seed"], "why": s["why"]}
            ok = False
            if a.verify:
                try:
                    ok, reply = verify(it, a.verify)
                    it["fuzz"]["verifier_reply"] = reply[:200]
                except Exception as e:
                    it["fuzz"]["verifier_error"] = str(e)[:200]
            (kept if ok else unverified).append(it)
    (out / "fuzz.jsonl").write_text("".join(json.dumps(x) + "\n" for x in kept))
    (out / "unverified.jsonl.txt").write_text("".join(json.dumps(x) + "\n" for x in unverified))
    print(f"{len(sd)} seeds, {made} items written by the fuzzer, {dups} duplicates dropped, "
          f"{len(kept)} verified -> {out / 'fuzz.jsonl'}, {len(unverified)} unverified -> {out / 'unverified.jsonl.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
