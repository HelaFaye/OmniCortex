# Truth probes: does the model believe what it said?

A model can write one answer while its own activations favour another. OmniCortex can look for that mismatch on
most models, with a **truth probe**: a linear classifier that reads, from the model's internal state, whether it
takes an answer to be true.

The method follows published work:
- Azaria & Mitchell (2023), "The Internal State of an LLM Knows When It's Lying";
- Marks & Tegmark (2023), "The Geometry of Truth";
- Burns et al. (2022), contrast-consistent search;
- Zou et al. (2023), representation engineering.

Deception probes on the same principle have flagged strategic deception (Apollo Research, 2025) and backdoored
"sleeper agent" behaviour (Anthropic, 2024).

## What it tells you, and what it cannot

For a question, the model's answer and optionally other answers, the check gives one verdict:

| Verdict | Meaning |
|---|---|
| `knows_but_says_otherwise` | the probe rates an alternative true and the model's own answer false |
| `doubts_own_answer` | the probe rates the answer false, with no alternative clearly preferred |
| `consistent` | the probe rates the answer true |
| `unsure` | in between |

It also gives a **trajectory**: the answer's (and the best alternative's) probability at every layer where the probe
is reliable. `alt_leads_at` lists the layers where the alternative was ahead. An alternative that leads in middle
layers and loses at the end is an answer that formed and was then overridden.

It cannot establish intent. A flag means the activations and the words disagree, which can happen because:
- the probe is wrong (check its AUROC);
- the model is writing fiction, role-play or a requested argument, where saying false things is the task;
- the model is uncertain rather than deceptive;
- a model trained against a probe can learn to hide from it.

Use it as one more risk signal beside the H-neuron risk strip, and look at the flagged replies yourself.

## Many kinds of models

There are two backends, and both handle any model their runtime handles.

**`--hf` (Hugging Face weights).** Every transformer layer's output is a stage, in the order the layers run.
- This covers causal LMs and the language half of most vision-language models.
- Models whose layer lists are not recognised fall back to the model's own `hidden_states`.
- Looped models (docs/DORY.md) run some layers more than once. Each pass is its own stage (`model.layers.12@pass2`),
  so the trajectory shows every pass.

**`--gguf` (any GGUF llama.cpp runs).** Goes through `llama-cett-dump`; build it with `scripts/build_llama_tools.sh`.
- The stages are the layers' MLP neuron contributions (CETT), the same features the H-neuron classifier uses.
- No PyTorch is needed. This is also the backend Studio uses.

A probe belongs to one model and one backend. Train a new one for every model; even a fine-tune of the same base
should get its own.

## Use

```bash
# training data: the bundled starter set, and/or your TestQA bank
python scripts/truth_probe.py data --bank qa/truth --bank qa/bank --out runs/truth.jsonl

# one probe per layer, each with its cross-validated AUROC; the best layer is the default
python scripts/truth_probe.py train --gguf models/my-model.gguf --data runs/truth.jsonl --out models/my-model.truth.npz
python scripts/truth_probe.py train --hf Qwen/Qwen3-8B --data runs/truth.jsonl --out models/qwen3-8b.truth.npz

# check an answer against alternatives (with --hf, leave out --answer to have the model answer)
python scripts/truth_probe.py check --gguf models/my-model.gguf --probe models/my-model.truth.npz \
    --question "What is the capital of Australia?" --answer "Sydney." --alt "Canberra."
```

The same steps are in Jobs as **Truth probe: train** and **Truth probe: does the model believe its answer?**.

### Training data

`data` reads any of these formats:

| Row | Becomes |
|---|---|
| `{"question", "answer", "label": true/false}` | one labelled answer |
| `{"statement", "label"}` | a claim on its own |
| `{"question", "answer", "wrong": [...]}` | the true answer and each wrong one |
| a TestQA `qa` item `{"prompt", "aliases"}` | true; another item's answer from the same file is the false one |

The last form makes easy negatives (the wrong answer is often the wrong kind of thing). Real distractors make a
better probe; use `--no-borrow` to keep only explicit wrong answers.

`qa/truth/basic.jsonl` is a starter set of 44 questions with 49 wrong answers (93 rows). It is enough to try the probe. For a
probe you rely on, use a few hundred rows or more, close to the replies you will check.

### Reading the training output

`train` prints the best layers and their AUROC, measured by cross-validation on held-out rows. In published results,
truth is usually most readable in the middle layers, and above 0.8 is common for factual questions on capable
models. Near 0.5 means the probe found nothing: more or better data, `--pool mean`, or a larger model. The check
only reports layers at 0.7 or above in the trajectory.

## In Studio

Set **Truth probe** in a model's settings (a probe trained with `--gguf` on that model's file). Then:

```
POST /api/truth  {"model": "...", "messages": [...], "text": "<the reply>", "alternatives": ["...", "..."]}
```

The question is the last user message. Paired devices may call it, as they may `/api/trace`. It needs Studio's
`llama-cett-dump`, like the reply checks.

## Limits

- One probe reads one kind of truth. A probe trained on short factual answers transfers poorly to code, opinions or
  long arguments; train on what you will check.
- Long replies: use `--pool mean`, or check the claim on its own as the answer.
- MoE GGUFs are not supported by the GGUF backend yet, as with live H-neuron scoring.
