# Text-model H-Neuron pipeline

The stage-by-stage guide for decoder LLMs. For image-text models see [VISION.md](VISION.md); for the no-PyTorch GGUF path see the section *Extraction without PyTorch* below.

## The idea, in one pass

1. **Collect** many answers to factual questions, several per question, and
   keep the ones the model gets *consistently* right and consistently wrong:
   it knows the first, and is confidently mistaken about the second.
2. **Tag** where the answer sits in each reply, and **split** the questions
   into balanced training and test sets.
3. **Extract** CETT: for every MLP neuron, how much it contributed on the
   answer tokens. This is the expensive step; llama.cpp's `cett-dump` does it
   from a GGUF, PyTorch from Hugging Face weights.
4. **Classify**: a sparse classifier learns which few neurons predict "wrong
   but confident". Those are the H-Neurons (`models/h_neurons.json`).
5. **Visualise** them, then **tune** how far to scale them down: far enough to
   cut confident errors, not so far that the model stops answering what it
   knows.
6. **Apply and ship**: edit the GGUF in place, export a LoRA, or apply live in
   Studio.

Terms are explained in [GLOSSARY.md](GLOSSARY.md). `python scripts/doctor.py`
shows which stages already have output and the command for the next one.

## PIPELINE

### 0. Preflight

```bash
python scripts/preflight.py --model_path Qwen/Qwen3-8B --n_pairs 400
```

Loads only the config and a meta-device skeleton — no VRAM, no weights. Tells
you the layer count, the intermediate size, whether a vision tower pollutes the
`down_proj` match, and how much RAM the classifier will want. That last number
is what usually ends these runs. Writes `preflight.json`.

### 1. Collect responses (LM Studio)

Point at any OpenAI-compatible server, local or over LAN (Studio's
`http://127.0.0.1:7870/v1` works). The questions come from TriviaQA; how to get
the files is in [data/README.md](../data/README.md).

```bash
python scripts/collect_responses_lmstudio.py \
  --base_url http://GPU-HOST:1234/v1 \
  --model qwen3-8b \
  --data_path data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet \
  --output_path data/consistency_samples.jsonl \
  --sample_num 5 --max_questions 200 --concurrency 4
```

Start small and read the `balanced pairs available` line. That yield rate tells
you how many questions you actually need to screen; scale from a measured number
rather than a guess.

For LM Studio, set **context length to 8192** (not 47k or 110k — the KV cache
reservation is what forces layers onto CPU) and raise **Max Concurrent
Predictions** to 4–8. Leave **Seed at -1**: a fixed seed makes all `sample_num`
generations identical and the consistency filter meaningless.

Resumable. Two servers can be driven in parallel with different
`--max_questions` slices and output paths; `cat` the results together.

### 2. Tag answer tokens

```bash
python scripts/make_answer_tokens.py \
  --input_path data/consistency_samples.jsonl \
  --output_path data/answer_tokens.jsonl \
  --model_path Qwen/Qwen3-8B
```

Upstream pays GPT-4o to find the factual span. The collector already stored the
post-`</think>` answer, so this just tokenizes it. Free.

### 3. Balanced splits

```bash
python scripts/sample_balanced_ids.py --input_path data/answer_tokens.jsonl \
  --output_path data/train_qids.json --num_samples 400
python scripts/sample_balanced_ids.py --input_path data/answer_tokens.jsonl \
  --output_path data/test_qids.json --num_samples 150 --exclude data/train_qids.json
```

### 4. Extract activations

```bash
python scripts/extract_activations.py \
  --model_path Qwen/Qwen3-8B \
  --input_path data/answer_tokens.jsonl \
  --ids_path data/train_qids.json \
  --output_root data/activations \
  --gpu_mem 14GiB --cpu_mem 40GiB \
  --locations answer_tokens all_except_answer_tokens
```

The slow stage. With `--gpu_mem` set, layers split across GPU and CPU; omit it
for pure CPU. Writes fp16 `.npy` per sample plus `neuron_index.json`.

Repeat with `--ids_path data/test_qids.json --output_root data/activations_test`.

### 5. Train and identify

```bash
python scripts/classifier.py \
  --acts_root data/activations --train_ids data/train_qids.json \
  --train_mode 3-vs-1 --penalty l1 --C 1.0 \
  --test_ids data/test_qids.json --test_acts_root data/activations_test \
  --out_dir models
```

`--C` is the sparsity knob and needs a sweep. Too high and you sweep in neurons
the model needs for general language, which wrecks it at intervention time. The
paper lands under 0.1% of all neurons; the script warns if you are far off.

Writes `models/classifier.npz` and `models/h_neurons.json`.

### 6. Visualise

```bash
python scripts/score_tokens.py \
  --model_path Qwen/Qwen3-8B \
  --classifier models/classifier.npz \
  --input_path data/consistency_samples.jsonl \
  --n 8 --gpu_mem 14GiB --out report.html
```

Standalone HTML: every token shaded by its H-Neuron score, reasoning block
separated from the final answer, plus a per-layer histogram. The question worth
asking of it — does the signal spike *inside* the reasoning trace, before the
model commits? If so you have an early warning signal rather than a post-hoc
flag, which is the most interesting thing this whole exercise could produce.

### 7. Intervene and ship

```bash
python scripts/intervene_model.py \
  --model_path Qwen/Qwen3-8B \
  --h_neurons models/h_neurons.json --scale 0.1   # dry run

python scripts/intervene_model.py \
  --model_path Qwen/Qwen3-8B \
  --h_neurons models/h_neurons.json --scale 0.1 \
  --output_path models/model-suppressed
```

Then convert and quantize for LM Studio:

```bash
python llama.cpp/convert_hf_to_gguf.py models/model-suppressed \
  --outfile model-suppressed-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize \
  model-suppressed-f16.gguf model-suppressed-Q8_0.gguf Q8_0
```

Drop the GGUF into `~/.lmstudio/models/local/model-suppressed/`.

Because the edit is a static weight change it survives conversion — llama.cpp
control vectors could not express this, since they only add to the residual
stream and cannot scale individual MLP neurons. Verify it survived quantization
by comparing abstention rates on held-out questions before and after.

---

---

## Profiles: tune once, reuse everywhere

A suppression profile is the neuron set plus a tuned scale, keyed to a model
fingerprint. It is a few kilobytes of JSON, so you can keep one per model per
deployment config and check them into version control.

### 7. Tune the scale

```bash
python scripts/tune_scale.py \
  --model_path Qwen/Qwen3-8B \
  --h_neurons models/h_neurons.json \
  --eval_path data/consistency_samples.jsonl \
  --scales 1.0 0.5 0.25 0.1 0.0 --n_eval 50 \
  --gpu_mem 14GiB --config_name trivia-q8 --save
```

Sweeps the scale and reports four numbers at each point: correct, abstained,
wrong, and perplexity on a canary text. The trade you are looking at is `wrong`
falling while `canary ppl` stays flat. If perplexity climbs, the classifier
selected too many neurons — go back and lower `--C`. Reaching for a gentler
scale instead just hides the problem.

The tuner will not pick a scale that damages the canary by more than 5%, and it
says so rather than silently choosing one.

### 8. Apply at runtime

```bash
python scripts/apply_profile.py --model_path <m> --list
python scripts/apply_profile.py --model_path <m> --config_name trivia-q8 \
  --prompt "Who wrote the novel Stoner?" --compare 1.0 0.25
```

Profiles apply as forward pre-hooks, not weight edits. Scaling column *j* of
`down_proj` is identical to scaling input element *j* before the matmul, so the
hook gives the same result with no weight mutation and a scale you can change
between generations. As a library:

```python
from profiles import Profile, SuppressionHandle
with SuppressionHandle(model, Profile.load(path)) as h:
    ...                  # suppressed
    h.set_scale(1.0)     # off, no reload
```

Profiles are keyed on architecture geometry (model type, layers, hidden,
intermediate, heads, vocab). Applying one to a mismatched model raises rather
than silently scaling arbitrary neurons. A fine-tune with identical geometry
gets a warning, because its neurons are different even though the shapes match.

### 9. Export

**As a LoRA adapter (hot-swappable):** the intervention is exactly a rank-*k*
update, `dW = W[:, S] @ diag(s - 1) @ E_S^T`, with *k* the H-Neuron count in
that layer. So it converts to a standard LoRA:

```bash
python scripts/export_lora.py --model_path <m> \
  --profile profiles/<fp>/trivia-q8.json --output_dir adapters/suppress
python llama.cpp/convert_lora_to_gguf.py adapters/suppress \
  --base <m> --outfile suppress-lora.gguf
./llama.cpp/build/bin/llama-server -m base-Q8_0.gguf \
  --lora-scaled suppress-lora.gguf 1.0
```

Because `dW` is linear in `(s - 1)`, the adapter's runtime scale a gives an
effective neuron scale of `1 + a*(s - 1)`. a=0 is the untouched model, a=1 is
the profile as tuned. `llama-server` exposes `/lora-adapters` to change it on a
running server, and `/completion` accepts per-request overrides — so suppression
strength becomes a dial rather than a rebuild.

**If you serve a fixed quant, pass `--gguf`.** llama.cpp applies the delta
against the *quantized* base. With `B` built from bf16 weights you get
`Q(W)[:,S] + W_bf16[:,S]*(s-1)`, so the base's quantization error rides along
unscaled. Point `--gguf` at the exact file you will serve and `B` is built from
`Q(W)` instead:

```
Q(W)[:,S] + Q(W)[:,S]*(s-1) = s*Q(W)[:,S]
```

Exactly the intended edit on the weights actually in use. Only the `ffn_down`
tensors are read, so it costs a fraction of a model load.

```bash
python scripts/export_lora.py --model_path <m> \
  --profile profiles/<fp>/trivia-q6.json \
  --gguf ~/.lmstudio/models/.../Qwen3-8B-Q6_K.gguf \
  --output_dir adapters/suppress
```

**As a baked model (maximum compatibility):**

```bash
python scripts/intervene_model.py --model_path <m> \
  --profile profiles/<fp>/trivia-q8.json --output_path models/suppressed
```

Or merge the adapter into a standalone GGUF with `llama-export-lora`. LM
Studio's LoRA support is less clearly documented than llama.cpp's; if it will
not load the adapter, merge and ship one file.

### What varies per config, and what doesn't

Neuron indices are a property of the model and do not transfer across models,
not even same-family ones. The *scale* is a deployment choice and is worth
re-tuning per config, because a Q4 base and a bf16 base do not respond
identically to the same suppression. Keep one profile per `(model, config)`
pair — `trivia-q8`, `trivia-q4`, `strict`, `lenient` — and record what each was
tuned against. `tune_scale.py` stores the full sweep in the profile's
`evaluations` list, so the numbers behind a choice stay with it.


---

## Vulkan and fixed quants

**Vulkan serves, it cannot analyse.** PyTorch has no usable Vulkan compute
backend, so stages 4 through 7 still need ROCm or CPU. Everything downstream of
the profile — LoRA adapter, baked GGUF, llama.cpp, LM Studio — runs on Vulkan
fine. LoRA is applied in the ggml graph rather than in backend code, so it is
backend-agnostic in principle; adapters have had backend-specific bugs though,
so load yours at scale 0 and confirm the output matches the plain base before
trusting it.

**A fixed quant makes the LoRA path strictly better than baking.** The base
GGUF is never touched or re-quantized, so there is no "did the edit survive
quantization" question at all. With `--gguf` the adapter is exact rather than
approximate. And because suppression strength is a runtime dial, you can do the
coarse sweep in PyTorch with `tune_scale.py` and then re-tune live against the
real Q6_K Vulkan server without re-exporting anything:

```bash
curl -X POST http://127.0.0.1:8080/lora-adapters \
  -H 'Content-Type: application/json' -d '[{"id":0,"scale":0.6}]'
```

That second pass matters. The scale `tune_scale.py` picks is measured on
dequantized or bf16 weights; the quant you deploy will not respond identically.
Treat the PyTorch number as a starting point and the live sweep as the answer.

**Quantizing does not shrink the extraction step.** Dequantized Q6_K weights
occupy the same memory as bf16 once they are PyTorch tensors, so a 9B model is
still ~18 GB and still needs `--gpu_mem` offload on a 16 GB card. Choosing a
6-bit deployment target does not make stage 4 fit.



## Extraction without PyTorch (llama.cpp, Vulkan)

`llama-tools/cett-dump/` is a llama.cpp tool that captures the same quantity the
PyTorch hooks do, via `cb_eval` -- the callback `llama-cvector-generator` uses.
For the node named `ffn_down-<il>`, `t->src[1]` is the input to down_proj, which
is exactly what `CETTManager` hooks.

| | weights | runtime | footprint | 16GB card |
|---|---|---|---|---|
| `extract_activations.py` | bf16 safetensors | PyTorch + ROCm/CPU | ~18 GB | needs offload |
| `extract_activations_gguf.py` | Q6_K GGUF | llama.cpp + Vulkan | ~8 GB | fits |

That is the whole point: the quantized path fits where the bf16 path does not,
and it removes ROCm from the extraction step entirely.

Extraction runs in two phases. Spans must be expressed in the tokenization that
produces the activations, so phase 1 gets the token ids with no forward passes,
the driver computes the regions, and phase 2 does the real work.

Reduction happens inside the tool. Writing raw `[n_tok, n_ff]` per layer would
be **918 MB per sample** at 500 tokens x 14336 x 32 layers; reducing over the
spans first makes it **1.8 MB**. The tool accumulates `|a| / ||layer output||`
and the driver applies the weight column norm afterwards -- that constant is
per-neuron and non-negative, so both mean and max commute with it and the split
is exact, not an approximation. It also means the tool never touches weights.

```bash
# build (see llama-tools/cett-dump/BUILD.md)
cp -r llama-tools/cett-dump ~/llama.cpp/tools/
echo 'add_subdirectory(cett-dump)' >> ~/llama.cpp/tools/CMakeLists.txt
cd ~/llama.cpp && cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cett-dump -j

# extract
python scripts/extract_activations_gguf.py \
  --binary ~/llama.cpp/build/bin/llama-cett-dump \
  --gguf ~/models/Qwen3-8B-Q6_K.gguf \
  --tokenizer Qwen/Qwen3-8B \
  --input_path data/answer_tokens.jsonl --ids_path data/train_qids.json \
  --output_root data/activations --ngl 99 --batch 4096 \
  --locations answer_tokens all_except_answer_tokens
```

Output is byte-compatible with the PyTorch path, so stages 5-9 are unchanged.

The tool emits its own token ids and the driver decodes those, so answer spans
are located against the tokenization that actually produced the activations
rather than a re-tokenization that might disagree at the boundaries.

**Do not mix the two paths in one training set.** This measures the quantized
model, the other measures bf16. For a fixed-quant deployment measuring the quant
is arguably more honest, but the neuron sets are calibrated to different weights.

### MoE

Supported, via the `ffn_moe_down-<il>` node. `ggml_mul_mat_id` puts the expert
ids in `src[2]`, so each (token, slot) already knows which expert produced it.
The index becomes three-axis:

```
flat = (layer * n_experts + expert) * n_neurons + neuron
```

Dense models report `n_experts = 1` and the axis collapses, so nothing
downstream changes. Means are divided by each expert's own routed-token count
rather than the span length -- a token only reaches the experts it was routed
to, so the span length is the wrong divisor.

**The feature count is the constraint, not the extraction.** A 48x128x768 MoE
is 4.7M features: 9.4 MB per sample, and a 400-pair 3-vs-1 classifier matrix of
**30.2 GB**. That does not fit 32 GB of RAM. Use `--top-experts`:

| kept | features | classifier |
|---|---|---|
| 128 (all) | 4,718,592 | 30.2 GB |
| 32 | 1,179,648 | 7.5 GB |
| 16 | 589,824 | 3.8 GB |
| 8 | 294,912 | 1.9 GB |

It ranks experts by mean activation over a sample of the training set and keeps
the busiest per layer. Rarely-routed experts are the right thing to drop: few
tokens reach them, so their per-cell estimates are the noisiest in the matrix
anyway.

There is a second, subtler cost. With top-8-of-128 routing each expert sees
roughly 6% of tokens, so per-expert estimates need far more data than the dense
case for the same reliability. Budget more collection, not just more RAM.

Selected neurons are written to `h_neurons_moe.json` keyed by `"layer:expert"`,
alongside the flat form in `h_neurons.json`.


## Without CETT: older GPUs and CPUs

CETT needs a forward pass with hooks over bf16 weights (PyTorch), or a
llama.cpp build with the `cett-dump` tool. On an older GPU (no bfloat16, too
little memory, no Vulkan 1.2) or on a CPU, use one of these instead
(`scripts/neuron_attrib.py`, or **Jobs → H-Neurons**):

| measure | what it says about a neuron | needs | cost |
|---|---|---|---|
| **saliency** (`grad_x_act`) | how much the model's log-probability of its own answer would change if the neuron went quiet: gradient × activation, a first-order estimate of patching each neuron | PyTorch, Hugging Face weights; float32 (any CPU or GPU) or float16 | one forward and one backward pass per sample |
| **saliency** (`integrated`) | the same, with the gradient averaged along the path from zero to the actual activation (integrated gradients over the neuron Jacobian). It is more faithful where a neuron is far from linear | as above | `--steps` (8) times saliency |
| **weights** | how strongly the neuron's output column writes each answer token's unembedding, after the final norm's scale (direct logit attribution). It reads the weights only, with no forward pass, so it shows what a neuron *can* promote, not what it did in a given reply | a GGUF and numpy | seconds per layer |
| **quant-noise** | how far the neuron's gate and up rows and its down column moved between a reference GGUF and a quant. With `h_neurons.json`, it also says whether the H-Neurons were hit harder than the rest | two GGUFs | seconds per layer |

`saliency` and `weights` write what extraction writes:
`neuron_index.json` and `<location>/act_<qid>.npy`, float16
`[layers, neurons]`. Stage 5 (the classifier) reads them unchanged. Point
`--acts_root` at their folder. Each file records its measure
(`"measure": "saliency-grad_x_act"`, `"weights-dla"`). **Train on one measure
only**: their scales and meanings differ from CETT's and from each other's.

```bash
# CPU, a small model, float32
python scripts/neuron_attrib.py saliency --model_path Qwen/Qwen3-0.6B \
  --input_path data/answer_tokens.jsonl --ids_path data/train_qids.json \
  --output_root data/saliency --device cpu
# an old GPU without bfloat16
python scripts/neuron_attrib.py saliency ... --device cuda --dtype float16
# no forward pass at all
python scripts/neuron_attrib.py weights --gguf model-Q8_0.gguf \
  --input_path data/answer_tokens.jsonl --output_root data/dla
# did the quant damage the H-Neurons?
python scripts/neuron_attrib.py quant-noise --ref model-F16.gguf \
  --quant model-Q4_K_M.gguf --neurons models/h_neurons.json --out runs/qnoise
```

Saliency stores every layer's activations for the backward pass, so its
memory grows with the reply length. `--max_tokens` (1024) skips longer
samples. Gradients through very small float16 values can underflow; use
float32 when results look sparse or all zero. `weights` and `quant-noise`
read dense FFNs; for mixture-of-experts models, use saliency or CETT with
`--top-experts`.

---

## Keeping LM Studio (and LM Link)

Nothing here touches LM Studio's install, config or model directory. The one
capability it lacks is adapter loading, so the GUI needs a pre-merged GGUF.

**Merge late.** Tune with base + adapter against `llama-server` -- one full-size
model file plus a ~30MB adapter, with the scale as a runtime dial -- and merge only
the scale you settle on. Merging every candidate costs a full model copy each.

```bash
source env.sh
./scripts/install_to_lmstudio.sh --lora adapters/suppress-lora.gguf \
    --alpha 1.0 --tag s010 --dry-run     # check the destination first
./scripts/install_to_lmstudio.sh --lora adapters/suppress-lora.gguf \
    --alpha 1.0 --tag s010
```

It infers LM Studio's model root from the base model's location, lays the file
out as `<publisher>/<repo>/<file>.gguf` so the GUI picks it up on rescan, refuses
to overwrite, checks free space first, and drops an `omnicortex.json` beside the
model recording which adapter and scale produced it. The tag goes in the name
because several suppression strengths will coexist in the model list and are
otherwise indistinguishable.

**LM Link** shares models that are *loaded on a peer device*, keyed by
`deviceIdentifier`. So the merged file has to exist on whichever machine
actually runs it. If you tune on the fast box and want the result reachable from
the laptop over LM Link, install it there, not locally.

---

## Comparing models

`viz/compare.py` sorts every pair into a comparability tier and refuses
index-level arithmetic outside the ones where indices mean the same thing.

| tier | condition | index-level? |
|---|---|---|
| IDENTICAL | same fingerprint and quant | yes -- a repeat measurement |
| REQUANTIZED | same fingerprint, different quant | yes -- same neurons, perturbed values |
| LINEAGE | same geometry, asserted post-training relationship | yes, but unverified |
| UNRELATED | different geometry, or different weights unasserted | no |

Layer 5 neuron 1234 in one model has no relationship to the same index in an
unrelated model. Differencing two unrelated maps produces noise that looks convincingly
like structure, so the tool declines rather than drawing it.

Distribution comparisons (depth profile, concentration, p99) and behavioural
ones (which items each model got wrong, Jaccard over failure sets) are valid at
every tier, because neither depends on indices lining up. Depth profiles are
resampled onto ten relative-depth bins and normalized afterwards, so a 32-layer
and a 48-layer model are directly comparable.

```bash
python viz/compare.py runs/model-q6 runs/model-q4
python viz/compare.py runs/base runs/finetune --assert-lineage
```

The REQUANTIZED tier is where the interesting experiment lives: does Q4 recruit
different neurons than Q6? That is the mechanistic version of "low quants
hallucinate more", and unlike the folklore it is measurable.

`--assert-lineage` is a hypothesis you are asserting, not a fact the tool can
check -- post-training does not permute neurons, so indices *plausibly* still
correspond. The reported correlation is how you test it: near zero means the
assertion was wrong and the shifts are meaningless.

---


## Which neurons for which task

`scripts/task_neurons.py` trains one classifier per task type and asks whether
they select the same neurons.

```bash
# label each collection run
python scripts/collect_responses_lmstudio.py ... --task trivia
python scripts/collect_responses_lmstudio.py ... --task code

python scripts/task_neurons.py --acts_root data/activations \
    --ids data/train_qids.json --samples data/consistency_samples.jsonl \
    --out_dir models/by_task
```

**Overlap is reported against chance, not raw**, because raw Jaccard is not
comparable across set sizes. Two independent sets of 400 drawn from 458,752
neurons share 0.35 cells by accident; two sets of 40,000 share 3,488. Measured
on planted data: two random 4,000-neuron sets score 0.77x -- chance -- while
two sets sharing a 1,200-neuron core score **35x** despite a Jaccard of only
0.18. Jaccard alone would have called that "mostly different".

Three outcomes, all worth having:

- **mostly shared** — hallucination is a general mechanism in this model; one
  profile transfers, and a task specialist buys nothing on this axis
- **mostly disjoint** — task-specific failure modes; a profile per domain is
  warranted, and delegating to a specialist has something underneath it
- **partly shared** — the intersection is the general mechanism, the remainder
  is where a domain profile earns its keep

Each task gets a normal `h_neurons_<task>.json`, so you tune, export and
evaluate it exactly like any other profile.


## VRAM budget

`scripts/vram_budget.py` computes the three competing costs from the model's
own architecture rather than a rule of thumb.

```bash
python scripts/vram_budget.py --gguf model-Q6_K.gguf --ctx 8192
python scripts/vram_budget.py --gguf ... --ctx 110592 --cache-type q8_0 --viz godot
```

The KV cache is the term people get wrong. It scales linearly with context and
is **independent of model quantization** -- a Q4 and a Q8 of one architecture
have identical cache costs.

```
bytes = 2 * layers * kv_heads * head_dim * ctx * elem_size
```

`kv_heads` is the grouped-query count, not the attention head count. Using the
latter overestimates by the GQA ratio, commonly 4x or 8x. Verified against
Llama-3-8B: 32 layers, 32/8 heads, head_dim 128, f16 at 8192 ctx = exactly
1.00 GiB.

On a 9B Q6_K with a 12 GiB carve-out:

| context | KV cache | layers on GPU | visualizer |
|---|---|---|---|
| 8192 | 1.00 GiB | 32/32 | GPU |
| 47104 | 5.75 GiB | 21/32 | CPU |
| 110592 | 13.50 GiB | 0/32 | CPU |

At 110K the cache alone exceeds the whole carve-out before a single weight.
When things do not fit, the visualizer is evicted to CPU **first** -- it being
slower to draw does not slow generation, and generation is the long pole.

## Hardware profiles

A config says what you want; a host says what a machine can do. Keeping them
apart is what lets one config be checked against several machines.

```bash
python scripts/hostprofiles.py detect --save laptop --vram 12
python scripts/hostprofiles.py add rig --vram 16 --ram 32 --cores 16 \
    --gpu "RX 9060 XT" --backend vulkan --dedicated
python scripts/hostprofiles.py plan --gguf model.gguf --ctx 47104
```

`plan` runs one config against every saved machine and prints a verdict per
host. `detect` fingerprints the current box so the right profile is selected
automatically, and configs can name a target host -- so you can check what will
fit on the rig from the laptop.

**Dedicated versus shared matters more than the number.** A discrete card's
VRAM is exclusively yours; an iGPU carve-out comes out of system RAM and the
desktop compositor draws from the same pool, so shared profiles reserve 1.6x
more headroom. `detect` will not guess an iGPU's carve-out -- it is a BIOS
setting and is not reported -- so pass `--vram` for those.

## Named configs

A config is a complete reusable setup under a name: model, load settings,
preset, visualizer, cache type and hardware limits. The name is passed to
llama-server as `--alias`, so **Cline sees the config name** rather than a
filename.

Every setting is audited against the model's real architecture, and warnings
attach to the field that caused them so the UI marks the setting rather than
listing complaints at the bottom. `turboquant` is flagged as an **error**: it
has no implementation in llama.cpp, being a research method for KV-cache
compression rather than a switch. `cache_type q8_0` is the same lever at lower
sophistication and does exist.


## Doctor

`scripts/doctor.py` walks every prerequisite in dependency order and reports
what works, what is missing, what that blocks, and the fix.

```bash
python scripts/doctor.py
python scripts/doctor.py --gguf ~/models/model-Q6_K.gguf
python scripts/doctor.py --json
```

It checks python, packages, torch and Vulkan devices (including the ICD-conflict
warning), the llama.cpp binaries this project needs, the model's architecture
and whether its `ffn_down` tensors are dense or MoE, which pipeline stages have
output, and whether the ports are free. It ends with a single next action rather
than a wall of status.

## Host capability checks

`scripts/hostcheck.py` estimates a mode's peak cost before it starts. Several
stages here fail in ways that are obvious in hindsight and invisible in advance
-- a MoE classifier matrix is 30GB on a model whose weights are 20GB, and you
find out an hour in.

```bash
python scripts/hostcheck.py                      # what this machine has
python scripts/hostcheck.py --mode classifier \
    --layers 48 --experts 128 --neurons 768 --pairs 400
```

It is wired into `classifier.py` and `merge_selective.py`, which warn by
default and refuse with `--strict-host`. A warning you can override beats a
crash four steps in, and beats a hard limit that is wrong about your machine.

It reads `MemAvailable` rather than `MemFree`, since page cache is reclaimable
and counting it as used would understate what a large allocation can get.

---
