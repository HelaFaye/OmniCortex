# Truth probe starter set

`basic.jsonl`: short, well-established facts, each with its true answer and plausible wrong ones
(`{"question", "answer", "wrong": [...]}`). It is enough to try `scripts/truth_probe.py` on a model.

For a probe you rely on, train on a few hundred rows or more, close to the replies you will check: your TestQA
bank (`truth_probe.py data --bank qa/bank`), Fix This Mistake pairs, or your own question/answer/wrong files.
See docs/TRUTH_PROBE.md.
