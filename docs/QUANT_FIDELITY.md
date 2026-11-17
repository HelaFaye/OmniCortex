# Quantization fidelity: did the quant keep what the model knows?

`scripts/quant_fidelity.py` (Jobs → **Did a quantization keep what the model knows?**) compares quantized GGUFs
of a model with a reference: the original, or the highest-precision file. It checks four things; each is
optional, and each needs only what it reads.

| Count | What it measures | Needs |
|---|---|---|
| **Next token** | llama.cpp's KL divergence from the reference and how often the most likely next token is the same, over a text (default: OmniCortex's docs) | `llama-perplexity` (built by `scripts/build_llama_tools.sh`); without it, how much of each greedy answer to a fixed prompt set matches word for word |
| **Knowledge** | TestQA on both. Of the questions the reference gets right, which the quant now gets wrong (**confident mistakes**) and which it declines (**now abstains**), plus any it newly gets right | a TestQA bank (`--testqa qa/bank`) and `llama-server` |
| **H-neurons** | the same texts (the reference's own TestQA replies, else the truth rows) profiled on both models: how similar each text's MLP neuron activity (CETT) stays, overall and layer by layer; with the model's H-neuron classifier, how far its risk score moves and how many reply flags flip | `llama-cett-dump`; `--classifier` for the risk part |
| **Truth probe** | a probe trained on the reference (`truth_probe.py train --gguf`), scored on each quant: its AUROC on both models, on the same labelled rows | `llama-cett-dump`, `--truth-probe`, `--truth-data` |

Together these answer different questions:
- **Next token** shows how far the output distribution moved.
- **Knowledge** shows whether that movement cost right answers, and whether the cost is a confident wrong answer
  (the worse kind) or a refusal.
- **H-neurons and the truth probe** show whether OmniCortex's own instruments still read the quant correctly: a
  classifier or probe trained on the original is only trustworthy on a quant that keeps those signals.

## Use

```bash
python scripts/quant_fidelity.py --ref models/m-F16.gguf \
    --quant models/m-Q8_0.gguf --quant models/m-Q4_K_M.gguf --quant models/m-Q2_K.gguf \
    --testqa qa/bank --testqa-limit 200 \
    --classifier models/m.classifier.npz \
    --truth-probe models/m.truth.npz --truth-data runs/truth-heldout.jsonl \
    --out runs/fidelity-m
```

- Models run one at a time, so a quant and its reference never need to fit in memory together.
- Every count is cached per model in `--out`, so an interrupted run resumes, and adding a quant measures only
  that quant.
- The report is `fidelity.json` and `fidelity.md`.

**The truth rows:** use rows the probe wasn't trained on. Both AUROCs are measured on the same rows, so the
*drop* is fair either way, but held-out rows show how the probe will do on new replies.

## The verdict

Each quant gets one verdict:

| Verdict | Meaning |
|---|---|
| **keeps** | every count that was measured is within its limit |
| **check** | a count is past its limit; the reasons are listed |
| **loses** | a count is past twice its limit (the script then exits with status 1) |

| Limit | Default | Option |
|---|---|---|
| KL divergence | 0.1 | `--max-kld` |
| same top token | 85% | `--min-same-top` |
| greedy agreement (without `llama-perplexity`) | 0.8 | `--min-agreement` |
| TestQA answers lost (mistakes + abstentions), the 95% lower bound | 3% | `--max-lost` |
| H-neuron activity similarity (mean cosine) | 0.95 | `--min-cett-cos` |
| risk flags flipped | 5% of texts | `--max-flag-flips` |
| truth probe AUROC drop | 0.05 | `--max-auc-drop` |

The defaults are a starting point. A small model loses more to the same quant type than a large one, and a
task that leans on exact recall (facts, code) loses more than chat.

## Results

On the tiny random Llama the llama.cpp integration test builds (4 layers; `tests/test_llamacpp_integration.py`),
Q8_0 kept everything and Q4_0 lost the truth probe:

| Quant | Verdict | Same top token | H-neuron cos | Truth AUROC |
|---|---|---|---|---|
| Q8_0 | keeps | 98.8% | 1.000 | 0.921 → 0.936 |
| Q4_0 | loses | 82.7% | 0.991 | 0.921 → 0.778 |

Even on a toy model, the truth probe was the most sensitive count: activity stayed 99% similar while the probe's
AUROC fell by 0.14.

**Qwen2.5-1.5B-Instruct**, F16 against three llama.cpp quants, on a CPU:
- KL over 10 chunks of OmniCortex's docs;
- TestQA's 30 factual items (the F16 got 26 right);
- H-neuron activity over the F16's own 30 replies;
- a truth probe trained on the F16 with 30 of the starter set's 44 questions (65 rows; cross-validated AUROC
  0.93 at `layer.24`), scored on the other 14 questions (28 rows).

| Quant | Size | Verdict | KLD / same top | Lost (mistakes + abstains) | H-neuron cos | Truth AUROC (held out) |
|---|---|---|---|---|---|---|
| F16 (reference) | 3.09 GB | | 0 / 100% | | 1 | 0.939 |
| Q8_0 | 1.65 GB | keeps | 0.0024 / 96.9% | 1 + 0 of 30 | 1.000 | 0.939 |
| Q4_K_M | 0.99 GB | keeps | 0.0560 / 86.8% | 1 + 0 of 30 | 0.988 | 0.934 |
| Q2_K | 0.68 GB | **loses** | 1.2327 / 46.7% | 5 + 0 of 30 | 0.863 | **0.679** |

- **Q8_0 and Q4_K_M kept what the model knows.** Each lost one answer of 30. Its 95% interval (0.6% to 17%)
  doesn't rule out no loss, so the verdict doesn't count it. Q8_0's lost item was in fact a sensible refusal
  of an unanswerable question that the grader didn't count as an abstention.
- **Q2_K lost it**, on every count. Its five confident mistakes include "the largest country in the world by
  area is Antarctica", the Colosseum built "in Alexandria, Egypt", and a made-up painter for the Sistine Chapel.
  Its activity drifted most at layer 15. A probe trained on the F16 dropped from 0.94 to 0.68 on it, so neither
  the probe nor the H-neuron classifier should be trusted on Q2_K without retraining.
- **The truth probe barely moved on Q4_K_M** (0.939 to 0.934), although its next-token agreement was only 86.8%.
  A probe trained on the original can be used on a good quant.

**Lost answers are judged by their 95% interval.** The verdict compares the interval's lower bound with
`--max-lost`, so one flipped answer in a small set isn't a finding, and five of 30 is (at least 7.3%). Use
more TestQA items (`--testqa qa/bank`, all subjects) to see smaller losses.

## Related

- `scripts/quant_select.py` (Jobs → *Choose a quantization*) picks the fastest quant that clears a fidelity
  bar on this computer. It measures next-token fidelity only; this report says what a quant loses, and why.
- `scripts/quantization_lab.py` makes mixed-precision quants that protect the layers OmniCortex measures as
  important. This report is how to check that the protection worked.
