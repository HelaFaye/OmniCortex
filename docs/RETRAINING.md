# Retraining on measured deficits

Suppression (the H-Neuron edit) makes a model decline more. Retraining can
make it *know* more or *reason* better on what it currently gets wrong. This
is targeted retraining, also called error-driven data curation or hard-negative
mining, built on what TestQA (and the benchmark importers) already measure.

```text
testqa.py ──► deficits.py ──► finetune.py ──► merge_export.py ──► testqa.py ──► retrain_gate.py
 measure       curate          QLoRA/LoRA/      merge + GGUF        measure       keep, reject or
               (verified,      full, SFT+DPO,   (or GGUF LoRA)      again         "not enough items
               probed)         focus layers                                       to tell"
```

## The whole cycle in one command

```bash
python scripts/retrain_cycle.py --out runs/cycle1 \
    --base-model org/my-model --base-gguf models/my-model-Q4_K_M.gguf \
    --teacher http://gpu-host:1234/v1@strong-model \
    --arms all,plan --review-model acme/my-model-q4_k_m --method lora --dpo --expand 3 --allow-exec
```

It is also in Studio: Jobs → Retrain → *Retraining cycle*. It runs every step
below in order:

1. Serve the base GGUF with llama-server and measure it with TestQA.
2. Build the dataset, probing the base model as `--student`.
3. For each arm: train, convert to GGUF, serve it, and ask it exactly the
   baseline's holdout items.
4. Gate each arm against the baseline, and later arms against the first.

The baseline and every arm are served the same way, at temperature 0, so
differences come from training, not from serving.

**Arms** are layer choices: `all`, `plan` (the review's focus layers), or a
range like `10-17`. With `--arms all,plan`, one run answers whether focusing
on the layers the neuron review points at beats training everywhere, on the
same data and the same items.

**Results:**
- `report.txt` has a verdict per arm and a recommendation: the best arm the
  gate keeps, or *keep the base model*.
- `cycle.json` has every command, log and result.

**Resuming:** an interrupted cycle continues where it stopped, because
finished steps are skipped. `--redo train/plan` (or `data`, `all`, …) runs
a step again.

**When it stops early, and why:**
- the baseline is mostly request errors, with a pointer to the server log;
- fewer than `--min-examples` (8) deficit examples to train on;
- a `plan` arm without focus layers, which is skipped.

Nothing is installed or deleted. Copy the recommended GGUF into your models
folder when you are satisfied.

The sections below are the same steps run by hand.

## 1. Measure

```bash
python scripts/testqa.py --endpoint m=http://127.0.0.1:7870/v1@my-model --allow-exec \
    --cache runs/cache --out runs/base.json
```

`--cache` matters: it keeps the model's full replies, which become the
"rejected" side of preference pairs.

## 2. Curate the deficits

```bash
python scripts/deficits.py --results runs/base.json --label m --cache runs/cache \
    --teacher http://gpu-host:1234/v1@strong-model --expand 5 --allow-exec \
    --general data/general-chat.jsonl --deficit-fraction 0.25 --out runs/retrain
```

What it does:

| step | detail |
|---|---|
| **Categorise** | `factual_gap` (factual answers wrong or declined), `reasoning_failure` (reasoning, executable code, API code wrong), `format_violation` (instruction following, unparsable output), `perception_failure` (vision items wrong) |
| **Correction pairs (SFT)** | gold answers where the bank has them; otherwise a teacher model's reply. **Every target must pass the same grader that failed the model**, so nothing wrong is trained in |
| **Contrastive pairs (DPO)** | `prompt / chosen (verified target) / rejected (the model's own failing reply)` |
| **Synthetic expansion** | `--expand N`: the teacher writes N variations of each failure. Code variations must ship tests that their own solution passes in the interpreter. Reasoning and factual variations must be answered identically by a second, independent teacher call. Anything unverified is dropped |
| **Replay buffer** | deficits are `--deficit-fraction` (default 25%) of the SFT set; the rest is anchor data: what the model already answers correctly, in its own words, plus `--general` chat data. This is the guard against catastrophic forgetting |
| **Holdout** | half the bank (deterministic by id) is never trained on, not even through variations. `holdout_ids.json` lists it |

### Probe what the model knows (`--student`)

```bash
python scripts/deficits.py --results runs/base.json --label m --cache runs/cache \
    --student http://127.0.0.1:7870/v1@my-model --teacher http://gpu-host:1234/v1@strong-model \
    --expand 5 --out runs/retrain
```

`--student` is the model being retrained, served. With it, two things change.

**Failed facts are sorted into weakly known and unknown.**
- Each one is sampled `--probe-samples` times (default 4) at temperature
  0.8.
- **Weakly known** (some sample right): the model holds the fact but does
  not reliably say it. The gold answer is trained; this is where
  fine-tuning on facts helps.
- **Unknown** (no sample right): the model does not hold the fact.
  - Training it to state the fact anyway teaches it to answer confidently
    from nothing, and raises hallucination on unrelated questions.
    Gekhman et al. (2024), *Does Fine-Tuning LLMs on New Knowledge
    Encourage Hallucinations?*, measured exactly this, and found such
    examples are learned late, mostly once the model has started to
    overfit.
  - So by default (`--unknown abstain`) the target becomes a decline, and
    the DPO pair prefers the decline over the made-up answer.
  - An unknown fact the model already declines is left alone.
  - `--unknown skip` leaves unknown facts out.
  - `--unknown teach` keeps the gold answer. Use it only for a full
    fine-tune meant to add knowledge, and judge the result with the gate.

**Synthetic preference pairs become honest.** The *rejected* side of a
variation's DPO pair must be a wrong answer to that variation. So the
student answers each variation:
- if the student fails it, its reply is the rejected side;
- if it already answers correctly, the variation is used for SFT only.

Without `--student`, variations get no DPO pairs. Earlier versions paired
the original item's failing reply with the variation's prompt, which taught
preference between answers to two different questions.

### The holdout stays held out

Synthetic variations are written by a teacher that has seen the failed
item, and occasionally restate a holdout question. Any variation that
shares an eight-word run with a holdout prompt is dropped, and so is any
whose word trigrams overlap one by 60% or more. The report counts what was
dropped.

### Focus layers from the neuron review (`--review-model`)

```bash
python scripts/deficits.py ... --review-model acme/my-model-q4_k_m --out runs/retrain
python scripts/finetune.py --model org/my-model --data runs/retrain --layers plan --method lora --out runs/adapter
```

The neuron review ([REVIEW.md](REVIEW.md)) records each checked reply's
activations, so it can say which layers behave differently on wrong answers
than on right ones, per subject. `--review-model` asks it which layers do
that most for the subjects that failed:
- each layer's evidence is the association above the review's
  Bonferroni-corrected floor, summed over its bins;
- with too few observations, no layer clears the floor and none is
  suggested.

The result is written to `plan.json` as `focus_layers`. Then
`finetune.py --layers plan` trains only those layers: adapters only there
with LoRA, and only their weights with `--method full`. `--layers 10-17` sets
them by hand.

Fewer layers changed means less of what already works is disturbed, and
the change goes where the difference was measured. It is a hypothesis to
test, not a guarantee, so compare it against `--layers all` with the gate.
The layer numbers belong to the GGUF the review recorded. They fit the
Hugging Face checkpoint of the same model; finetune.py refuses layer
numbers the model does not have.

`plan.json` and `report.txt` give counts by category and subject, and the
method each category calls for. The teacher can be any OpenAI-compatible
endpoint: a larger local model, or a hosted one if its terms allow training
on its outputs.

### Vision deficits

Vision items go to `sft_vision.jsonl` and `dpo_vision.jsonl`, with their
images rendered into `images/`, in the image + messages format VLM trainers
take. Their targets are the bank's gold answers. `--expand N` needs no teacher
for them: `scripts/vision_synth.py` draws N new scenes in the failed item's
skill family (counting, colour, shape, position, size, reading text and
numbers, arithmetic), and the answer is known because the program drew it.
Every bank vision item maps to a family. Replay anchors are the model's own
correct vision answers, topped up with text anchors when there are too few.

```bash
python scripts/vision_synth.py --family count_shape -n 4 --out /tmp/preview   # look at what it draws
```

### SWE-bench deficits

Repository-level coding failures need longer, real-world data than the bank
has. `benchmarks.py swebench train-data` builds `sft.jsonl` and `dpo.jsonl` in
the same layout from SWE-bench's train split: gold patches as targets, the
model's own failed patches as DPO rejections, failure modes from a test run
setting priorities, and this directory's `sft.jsonl` as replay. See
[BENCHMARKS.md](BENCHMARKS.md#retraining-on-swe-bench-deficits).

## 3. Train

```bash
python scripts/finetune.py --check                                    # GPUs, precision, bitsandbytes
python scripts/finetune.py --model org/base-model --data runs/retrain --method qlora --dpo --out runs/adapter
python scripts/finetune.py --launch 4 --model ... --method lora --dpo --out runs/adapter      # 4 GPUs, data parallel
python scripts/finetune.py --launch 4 --model ... --method full --fsdp --out runs/full        # 4 GPUs, sharded
```

| deficit | method | why | multi-GPU |
|---|---|---|---|
| behaviour, logic, format | `qlora` / `lora`, SFT then `--dpo` | habits, not knowledge; adapters capture them cheaply | data parallel (DDP): each GPU holds the model plus a tiny adapter |
| factual / knowledge | `full` (or high-rank LoRA on the MLPs) | facts are spread across the MLP weights | `--fsdp` shards weights, gradients and optimizer state across GPUs |

- **QLoRA** loads the base in 4-bit NF4 and trains 16-bit adapters on all
  linear layers (`--lora-targets` to narrow). It needs CUDA and bitsandbytes.
- **LoRA** is the fallback for ROCm, CPU and GPUs without 4-bit kernels.
- Train from the bf16/fp16 Hugging Face checkpoint, not a GGUF.
- SFT loss is on the reply only. DPO uses the adapter-disabled model as its
  reference, so no second copy of the weights is loaded.
- Precision is chosen per GPU: bf16 where supported, otherwise fp16, and fp32
  on CPU. Older cards (Maxwell/Pascal, e.g. a Tesla M10) have no bf16, and
  recent PyTorch wheels may not include their `sm_XX`; `--check` shows
  whether yours is supported before you start.

### Vision-language models

```bash
python scripts/finetune.py --vision --model org/vlm --data runs/retrain --method lora --dpo --out runs/vlm-adapter
```

`--vision` loads the model with its processor and trains on the vision files.
Adapters go on the language model and the projector; the vision encoder is
always frozen. `--freeze-projector` also leaves the projector alone, which
means the base model's existing `mmproj` GGUF still matches the result: use it
when llama.cpp cannot convert your model's projector (below).

## 4. Merge and convert

```bash
python scripts/merge_export.py --base org/base-model --adapter runs/adapter --out runs/merged --gguf Q4_K_M
python scripts/merge_export.py --base org/base-model --adapter runs/adapter --out runs/lora --lora-gguf
```

The first merges the adapter into the base, converts it with llama.cpp's
converter and quantizes it. The second converts only the adapter to a GGUF
LoRA: load it in Studio's adapter field and the α slider blends base and
retrained behaviour live, the same way it does for suppression.

With `--vision`, merging writes the language GGUF and also tries to convert the
projector to `mmproj-model-f16.gguf`. llama.cpp converts projectors for the VLM
families it serves (Gemma 3, Qwen-VL, SmolVLM, Pixtral and others) but not
every Hugging Face layout, HF-format LLaVA included. When it cannot, the
command exits with status 2, records `mmproj_error` in
`omnicortex-export.json`, and points you at `--freeze-projector`.

## 5. Measure again, honestly, and decide

```bash
python scripts/testqa.py --endpoint before=...@my-model --endpoint after=...@my-model-retrained \
    --allow-exec --out runs/after.json
python scripts/retrain_gate.py --results runs/after.json --before before --after after \
    --holdout runs/retrain/holdout_ids.json --out runs/gate.json
```

Only the holdout says whether retraining generalised. The trained half will
look better whether or not anything was learned.

`retrain_gate.py` (Jobs → Retrain → *Keep or reject a retrain*) compares
the same items before and after. For each split and subject it reports:
- items fixed and items broken;
- an exact McNemar test on those changes;
- wrong-answer and abstention rates.

The verdict:
- **keep**: the holdout improved significantly, and neither hallucination
  nor any subject got significantly worse.
- **reject**: any of those got significantly worse.
- **inconclusive**: otherwise, with roughly how many holdout items it
  would take to tell.

It also flags **memorisation**: a large gain on the trained items with
little or none on the holdout. `--strict` makes the verdict the exit status,
for scripts that chain runs. Check the canaries and the
per-subject table too: a model that improved on code but regressed on writing
has traded one deficit for another. Re-run with `--publish-stats` so Studio's
`auto` routing sees the new numbers.

## Tested

`tests/test_retrain.py` runs the whole chain on a tiny Llama: verified
deficit dataset with holdout and replay share, LoRA SFT plus DPO,
merge, a Q8_0 GGUF that llama.cpp loads, and a GGUF LoRA adapter. The vision
path runs on a tiny LLaVA: vision dataset with verified variations, LoRA SFT
plus DPO with and without a frozen projector, merge and language GGUF. `tests/test_deficit_quality.py` covers:
- knowledge probing against a fake student;
- synthetic DPO pairs built from the student's own replies;
- the holdout guard;
- focus layers from a recorded review;
- LoRA adapters landing only on the chosen layers of a tiny Llama;
- the gate's three verdicts.

QLoRA, FSDP and real VLM projector conversion need GPUs or downloads and
have not been run here. Whether focus layers beat all layers on real models
has not been measured yet; that is what the gate is for.
