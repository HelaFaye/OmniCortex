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

### How the answer is read

Two settings decide what the probe reads. Both are saved in the probe file, so `check` reads replies the same way.

- **`--framing judge`** (default): the model is shown the question and the answer and asked "Is this answer
  correct?", and it is read at the start of its verdict ("The answer is"), before it says which.
  **`--framing answer`** reads the answer as the model's own reply, at its last token. Probes made before framings
  existed load as `answer`.
- **`--method massmean`** (default): the direction from the false answers' average activation to the true answers',
  calibrated to a probability. **`--method logreg`** is L2 logistic regression (`--C`).

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

## Results on real models

These are from the starter set, on a CPU. Qwen2.5-0.5B and 1.5B are small, so treat this as a first measurement.

**Cross-validated AUROC on the starter set (93 rows, best layer):**

| Model | `answer`, last token | `answer`, mean | `judge`, last token | `judge`, mean |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.56–0.61 | 0.52–0.56 | 0.69–0.72 | 0.65–0.69 |
| Qwen2.5-1.5B-Instruct | 0.90 | 0.80–0.86 | **0.95** | 0.95 |

Each range spans mass-mean and logistic regression. Ending the answer with a full stop made `answer` at the last
token worse on both models, and made no clear difference to the mean. Two things stand out:
- **Model size matters most.** The 0.5B model barely holds a readable truth signal.
- **`judge` beats `answer`**, on both models. Truth is easiest to read where the model evaluates a claim, as the
  literature reports.

The best-of-layers figures are optimistic. So, separately, a probe with the defaults on Qwen2.5-1.5B (best layer:
`model.layers.25`, cross-validated 0.93) was run on 16 questions outside the training set:

| Test | Result |
|---|---|
| correct answer rated above a wrong one | 16 of 16 pairs |
| correct answers flagged (false alarms) | 0 of 16 (2 `unsure`: "54" for 9 × 6, "8" spider legs) |
| wrong answers flagged | 14 of 16 (missed "Buzz Aldrin", "Lead") |
| the model's own answers | 15 of 16 right, 0 flagged. The one mistake ("12 months" for the days in a leap year) was flagged `knows_but_says_otherwise`: own answer p = 0.01, the correct "366" p = 0.70 |

Sixteen questions are too few for a reliable error rate. They do show the intended behaviour on a real model: the
probe flagged a confident mistake whose correct answer the model rated true, and raised no alarm on correct answers.

### Reading the training output

`train` prints the best layers and their AUROC, measured by cross-validation on held-out rows. In published results,
truth is usually most readable in the middle layers, and above 0.8 is common for factual questions on capable
models. Near 0.5 means the probe found nothing: try a larger model first (see Results), then more or better data. The check
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
