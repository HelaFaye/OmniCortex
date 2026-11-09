# Research notes

What the method does and does not claim, where results come from, and what changed from upstream H-Neurons.

## Read this before you start

**LM Studio cannot do the analysis.** It is a llama.cpp/MLX front-end. Its API
is chat, completions, embeddings, tokenization and plugins — there is no hidden
state endpoint and no way to register a forward hook. Stages 2 through 5 need
PyTorch. LM Studio covers stage 1 (generation) and stage 7 (serving the edited
model), and nothing in between.

**Vulkan is not a compute path here.** PyTorch has no usable Vulkan backend.
Vulkan exists for you only inside llama.cpp. Everything hooked runs on ROCm or
CPU.

**Do not quantize the model you extract from.** CETT measures activation
magnitude. Quantization perturbs exactly the quantity being measured. Generate
from a quant if you like; extract from bf16.

**Suppression is not correction.** The paper's finding is that H-Neurons are
causally linked to *over-compliance*. Scaling them down makes the model more
willing to decline, not more likely to be right. You are trading hallucinations
for abstentions. On short factual QA that is usually worth it. On a coding
workload it may just make the model less useful. Measure both sides.

**Scope is narrower than "hallucination" suggests.** The labels come from
TriviaQA: a short factual answer that does not match a gold string. This is not
RAG groundedness, not fabricated citations, not long-form confabulation. If your
real failure mode is one of those, the neurons you find may not transfer.

---

## What was changed from upstream, and why

| Change | Reason |
|---|---|
| Hooks filtered to text decoder layers | `'down_proj' in name` also catches a multimodal model's vision tower, corrupting the flat-index → (layer, neuron) map |
| Hook captures to CPU float32 | Upstream stacks hooked tensors and calls `.to(model.device)`, which raises the moment `device_map` splits layers |
| Layer index read from module path | Firing order is not guaranteed to be layer order |
| Sequence built as prompt + response ids | Chat templates for reasoning models strip `<think>` from prior assistant turns, so the upstream round-trip does not reproduce what was generated |
| Answer search starts after `</think>` | Otherwise the span matches inside the reasoning trace |
| `neuron_index.json` written explicitly | Intervention no longer has to assume uniform `intermediate_size` |
| Preallocated fp32 matrix, saga solver | `np.array(list_of_arrays)` copies, then liblinear promotes to fp64; ~4× the peak RAM |
| vLLM replaced with any OpenAI endpoint | Generation runs on a different machine, OS and GPU vendor than extraction |
| Truncated generations discarded | Counting an unclosed think block as a hallucination poisons the labels |
| `--max_layer_fraction` guard | Aborts intervention if a bad `C` selected enough neurons to damage general capability |
| Pre-hook instead of weight edit | Same maths, no weight mutation, scale changeable at runtime |
| Portable profiles keyed to a fingerprint | Reuse a tuned intervention; refuse to apply it to the wrong model |
| LoRA export | The edit is rank-k, so llama.cpp can hot-swap and rescale it |
| `--gguf` weight source | Building B from the served quant makes the adapter exact, not approximate |

## Status

Written against the papers and the upstream source, but **not executed against
real weights** — treat the first run of each stage as a debugging session. The
likeliest thing to need adjusting is `TEXT_LAYER_RE` in `oc_common.py` if your
model's remote code names its modules unconventionally. `preflight.py` will
show you the names.


## Improving results: where the leverage actually is

There is no training loop in this pipeline. The only free parameters are which
neurons and how hard, so a competitive signal against a stronger model has
nothing to optimise against. What does move the numbers, in order:

**1. Label quality.** Every mislabelled pair is noise the L1 classifier fits,
and it puts the wrong neurons in the profile -- which no amount of scale tuning
undoes. `rule_judge` is substring matching on gold aliases; with a verbose
reasoning model it errs in both directions. Measure before you trust it:

```bash
python scripts/judge_agreement.py --input_path data/consistency_samples.jsonl \
    --base_url http://host:8080/v1 --model <judge> --n 150
```

Cohen's kappa above 0.8 means keep the rule judge. Below 0.6 means your labels
are noise and nothing downstream is meaningful yet. Read the disagreements
rather than assuming the model judge is right -- it has its own failure modes,
and disagreement marks where labels are ambiguous, not which judge is correct.

**2. Domain match.** Neurons from TriviaQA short-form recall may do nothing for
hallucinated APIs in agentic coding. If `eval_code_hallucination.py` comes back
null, the fix is collecting stage 1 from code tasks, not a different alpha.

**3. Sample size.** More balanced pairs sharpens the neuron set, and the cost
lands on whichever machine runs stage 1.

A stronger model is useful here as a judge and as a source of adversarial eval
items. Using its outputs to fine-tune this model is a different project, and
one that most providers' terms prohibit.

---

## Training

Retraining on measured deficits is now built in: see [RETRAINING.md](RETRAINING.md). The notes below explain where it sits relative to suppression.

### Why suppression alone is not training

There is no training loop here. OmniCortex suppresses; it cannot add capability.
If the goal is a better local model over time, the order of leverage is:

**1. Upgrade the checkpoint.** A newer open-weight release in the same size
class costs zero GPU-hours and usually beats anything you can train at home.
Do this before anything else.

**2. Train an abstention LoRA.** `export_sft_dataset.py` turns the collected
pairs into it directly: consistently-correct questions teach "answer",
consistently-hallucinated teach "decline". Same behavioural target as OmniCortex,
reached by gradient descent instead of a scalar on a neuron column, so it can
shape *when* to decline in a way clamping cannot. Needs a rented 24GB card for
an afternoon; integrated GPUs and 16 GB cards are not enough.

**3. Compare the two** on the same held-out set with
`eval_code_hallucination.py`. If the LoRA wins clearly, OmniCortex was scaffolding.
If it does not, that is a real result about how localised this behaviour is.

Replicating a vendor's RL self-improvement loop (GRPO and friends) is a
different project at a different scale -- it wants datacentre GPUs, and the
vendors already run it and ship you the checkpoints. What is worth keeping from this repo either way is the data and
verifier layer: consistency-filtered labels, a validated judge, and an
executable hallucination metric that is already shaped like a reward function.
The GPU is the cheap, rentable part; that layer is not.

---
