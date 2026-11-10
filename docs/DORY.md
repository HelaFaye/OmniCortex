# Looped models: NVIDIA Dory

[NVIDIA-Nemotron-Labs-Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) is a 4B
**looped** (latent-recurrent) language model. Its 36 layers are:
- 8 input layers;
- a block of 20 recurrent layers, run several times per token, with the input reinjected on every pass;
- 8 output layers.

Its attention mixes sliding-window and full layers, it handles up to 256K tokens, and it reasons and calls tools.
The loop count (1–4, default 2) is set at load time **without retraining**: more loops mean more computation per
token. The KV cache keeps one copy per loop (`per_loop`), or one shared copy the size of a single loop
(`last_loop`).

How much a model computes per token is usually fixed by its size. Here it is a setting, so OmniCortex can measure
what each extra pass through the same weights changes, and watch it happen inside the model.

## Running it in Studio

Dory's GGUF needs a llama.cpp fork until llama.cpp merges its support (ggml-org/llama.cpp#30306). The model card
pins one commit. Build it with the **Build llama.cpp** job (Setup), into its own folder:

```bash
scripts/build_llama_tools.sh --repo https://github.com/brianchmielNV/llama.cpp.git \
    --ref 98a74c97ea09ac72f6ceda1559808fc0a2f447ce --backend cuda --dir ~/llama.cpp-dory
```

- Use `--backend metal` on Apple silicon, or `vulkan`.
- `cett-dump` is built into the fork as well, so activation profiles and H-neuron checks work on Dory too.
- Point Studio's llama-server at `~/llama.cpp-dory/build/bin/llama-server` (Settings → Workspaces), or keep it
  per workspace.

Download `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` from the
[GGUF repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF). Its SHA-256 is
`96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`.

Studio recognises looped models from the GGUF (`dory.num_loops`) and tags them **looped**. The model's settings
then show:
- **Loops** (1–4; empty keeps the model's default, 2);
- **Recurrent KV cache** (`per_loop` or `last_loop`).

Both are passed as `--override-kv dory.num_loops=int:N,dory.recurrent_kv_cache_mode=str:MODE`. For Dory, Studio
also adds the model card's `--jinja` (tool calls through the chat template) and `--reasoning on`, unless your
extra arguments set them.

The card recommends sampling at temperature 1.0, top-p 0.95. Set that in a chat preset.

Memory: with `per_loop`, the KV cache grows with the loop count. On a small card, use `last_loop` for 3–4 loops.

## What the loops buy: the loop sweep

The **Looped model: sweep the loop count** job (`scripts/loop_sweep.py`) runs TestQA once per loop count, and per
KV mode if you give both. It reports:
- accuracy;
- **confident mistakes**;
- declines;
- tokens per second;
- the paired changes against the first setting (gained, regressed, with a McNemar test).

Read it as a curve:
- If 3 loops match 4 at better speed, the fourth loop buys nothing.
- If more loops raise accuracy but also confident mistakes in one subject, that subject wants fewer loops. That
  is what per-subject routing is for: give each subject a model entry with its own loop count.

## Where the answer forms: the per-loop lens

The **Looped model: what each loop adds** job (`scripts/loop_lens.py`) runs on Hugging Face weights
(`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`; a transformers version that knows the architecture). After each
pass through the recurrent block, it reads the state through the final norm and output head, a logit lens with
loops in place of layers. It reports:
- **after loop k**: the top tokens, the entropy, and the expected answer's probability and rank;
- **decided at loop**: the first loop whose top token is the final answer and stays it;
- **work per loop**: each recurrent layer's MLP activity on every pass;
- with the model's `h_neurons.json`, the **H-neurons' share** of that activity, per loop.

For a confident mistake, this shows whether:
- the wrong answer is there from loop 1, so more loops will not fix it and retraining must;
- it forms in a later loop, so fewer loops would avoid it and the loop sweep should confirm that;
- the H-neurons fire on one particular pass, which would make the H-neuron picture depend on the loop.

That last question is new for H-neuron research: the same neurons are reused on every pass. Whether
hallucination-associated neurons fire on the first pass, or build up across passes, is open. OmniCortex now has
the tools to answer it.

## Training with NeMo RL and Gym

Dory was post-trained with GRPO in [NeMo RL](https://github.com/NVIDIA-NeMo/RL), on environments from
[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). Pre-training used [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
OmniCortex plugs into that stack as an environment:

- `integrations/nemo_gym/omnicortex_testqa` is a Gym resources server. Its reward is TestQA's grade, shaped not to
  pay for guessing:

  | Answer | Reward |
  |---|---|
  | correct | +1 |
  | declined | +0.25 |
  | confidently wrong | −1 |

  With a Studio and the policy's classifier, it can also take `risk_weight` × the model's own H-neuron risk off a
  correct answer, so the policy learns to answer what it actually knows.
- **NeMo Gym: export TestQA** (`scripts/gym_bridge.py export`) writes the dataset: bank items, neuro-fuzzing
  banks, and Fix This Mistake pairs, the last scored by closeness to your correction.
- Train with NeMo RL's GRPO through Gym, as for any Gym environment. NeMo RL supports Nemotron models; check its
  model list for Dory before planning a run.

The reward is tested here. The Gym server follows the interfaces of Gym's own example servers, but it has not been
run inside Gym in this repository's CI.

**Megatron-LM** is not integrated. OmniCortex works with released checkpoints (Hugging Face and GGUF), not with
pre-training. Its papers and the model card's training notes (NVFP4 for most of pre-training, then BF16, and context
extended to 256K) are background for reading the loop results, not something OmniCortex runs.

## Activations from the GGUF: cett-dump and the loops

Reading the fork's graph (`src/models/dory.cpp` at the pinned commit) answers whether `cett-dump` can tell the loops
apart:

- **The loops are separable.** The fork unrolls the model into virtual layers: 8 input, 20 × loops recurrent and
  8 output, so 56 at 2 loops. Every pass gets its own layer index, so a loop's activations never overwrite the
  previous loop's.
- **But the MLP output is unnamed.** cett-dump normally finds each layer's down projection by the graph node's name
  (`ffn_down-<layer>`). Dory's graph calls the matrix multiply directly and names only `l_out`, so name matching
  finds nothing.
- **Not every block has an MLP.** Dory alternates attention-only and MLP-only blocks, so only some blocks have an
  `ffn_down` weight.

So for any GGUF whose metadata has `<arch>.num_loops`, OmniCortex runs cett-dump with **`--by-weight <blocks>`**:
- the down projection is the matrix multiply whose weight is `blk.N.ffn_down.weight`;
- the k-th time block N runs within one forward pass is recorded as layer **k × blocks + N**.

Dory has 36 blocks, so at 2 loops there are 72 layer slots. Pass 1 is slots 0–35, and pass 2 of recurrent block N
is slot 36 + N. Slots that never run (the input and output blocks' second pass, and attention-only blocks) stay
unseen, with zeros. Weight norms repeat per pass. The neuron index records `looped: {blocks, passes}`.

A classifier or profile made at one loop count therefore has blocks × passes features, and fits only that loop
count. Studio passes the model's **Loops** setting to cett-dump (`--override-kv`), so live scoring matches what
llama-server runs.

`--by-weight` records exactly what name matching does on ordinary models: the integration tests compare the two,
across several micro-batches. It has not yet run on Dory itself. Building and running the fork was not possible in
the environment where this was written. Run `extract_activations_gguf.py` on the real file and check that pass 2's
recurrent slots are seen and differ from pass 1's.

## What is not done yet

- llama-server cannot change the loop count per request. Switching loops reloads the model, which takes a few
  seconds for a 4B model. A per-request loop count would need a llama.cpp change.
- `loop_lens.py` needs PyTorch and Dory's transformers support.
- `cett-dump` on Dory is built for it but not yet run on the real model (see below). Until it is, use the per-loop
  lens for loop-resolved questions.
