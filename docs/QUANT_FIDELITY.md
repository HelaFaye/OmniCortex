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
| TestQA answers lost (mistakes + abstentions) | 3% | `--max-lost` |
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
AUROC fell by 0.14. A run on a real model is in progress and will be added here.

## Related

- `scripts/quant_select.py` (Jobs → *Choose a quantization*) picks the fastest quant that clears a fidelity
  bar on this computer. It measures next-token fidelity only; this report says what a quant loses, and why.
- `scripts/quantization_lab.py` makes mixed-precision quants that protect the layers OmniCortex measures as
  important. This report is how to check that the protection worked.
