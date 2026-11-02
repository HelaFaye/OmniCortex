# Engine options: llama.cpp and vLLM

Studio serves models with one of two engines, chosen per model in the model
panel (**Engine**):

| engine | serves | best for |
|---|---|---|
| **llama.cpp** (`llama-server`) | GGUF files | one machine, CPU + any GPU (Vulkan, CUDA, ROCm, Metal), mixed vendors, small memory; the only engine OmniCortex's activation tools read (cett-dump) |
| **vLLM** (`vllm serve`) | Hugging Face checkpoint folders (also AWQ / GPTQ / fp8), GGUF experimentally | many parallel requests on NVIDIA or AMD Instinct/RDNA GPUs; prefix caching, chunked prefill, tensor and pipeline parallelism |

Hugging Face folders under your models folders (a `config.json` with
`*.safetensors` next to it) show up in the model list and default to vLLM.
Install vLLM separately (`pip install vllm`, or AMD's ROCm build) and tell
Studio its command with `--vllm` or `$NS_VLLM` if it is not on the PATH.

## The options

Open a model → **Engine options**. Each option shows:
- its flag;
- what it does and when it helps;
- whether the installed engine has it. Studio reads `llama-server --help`;
  options the build lacks are folded away.

They are grouped:
- **Speed**: flash attention (auto/on/off), prompt threads, physical
  batch, parallel slots, busy-wait, priority.
- **Memory and KV cache**:
  - K and V cache types (`q8_0` halves the cache; a quantized V needs
    flash attention);
  - the KV cache in system RAM;
  - loading mode: mmap, mlock or read in.
- **Long context**:
  - context shift with *tokens to keep*: StreamingLLM-style. The first
    tokens (the attention sinks) stay, old middle tokens drop, and memory
    stays flat in long chats;
  - prompt cache reuse, so a repeated system prompt or document is not
    computed again;
  - vLLM: prefix caching and chunked prefill.
- **Speculative decoding**:
  - a draft model, its limits and its own device. For example, the draft
    on the iGPU and the big model on the dGPU;
  - **draft-free n-gram methods** (`ngram-mod`, `ngram-map-k`…), which
    propose from text already in the context. They need no extra memory
    and work well on code and edits;
  - EAGLE-3 / MTP heads where the model ships them;
  - vLLM: a JSON speculative config.
- **Devices**:
  - which devices hold layers (`Vulkan0,Vulkan1`);
  - RPC servers on other machines;
  - vLLM tensor / pipeline parallel.
- **CPU placement**: core range or mask, strict placement, NUMA mode.
- **Environment and launch**: a closed list of documented variables
  (ROCm, Vulkan, CUDA, vLLM), and an optional `numactl` or `taskset`
  wrapper.

Older Studio settings keep working:
- *flash attention on* becomes `-fa on`;
- a config's `cache_type` becomes both cache types.

The flag spellings follow the build: llama.cpp renamed `--draft-max` to
`--spec-draft-n-max`, and Studio passes whichever the binary accepts. After
**Load**, the panel shows the exact command and anything it left out.

From a terminal, `python scripts/engine_options.py --server <llama-server>`
lists the options this build has. With `--args '{...}'` it prints the flags
a setting produces.

## Recipes

Each of these is a starting point, not a promise. Hardware differs, so
measure the result.

### A small GPU and long chats
- K and V cache types `q8_0`, flash attention `on`;
- context shift on, *tokens to keep* `-1`;
- prompt cache reuse `256`.

### Faster generation without a second model
- Speculative decoding → method `ngram-mod`.

For code and repetitive text this often gives 1.5–2× and costs nothing.

### Faster generation with a draft model
- Draft model: the smallest model of the same family, which shares the
  tokenizer.
- Draft tokens at most: 8–16.

If acceptance is low, lower the limit.

### A draft on another device

The draft model proposes a few tokens, and the big model checks all of them
in one pass. Putting the draft on a device the big model does not use (an
iGPU, a second GPU, or the CPU) keeps the big model's GPU free for that
checking.

In a model's settings, **Speculative decoding → Suggest a draft model and
device** does the matching (`scripts/draft_pick.py`):

- **Drafts.** It reads the vocabulary of every model in your folders and keeps
  the ones llama-server will accept: the same tokenizer, vocab sizes within
  128 of each other, the same BOS and EOS tokens, and the same token text at
  the ids it compares. Of those, models near a tenth of the big model's size
  rank first; anything over 40% of its size is left out.
- **Device.** From `llama-server --list-devices`, it picks the GPU with the
  most free memory that the big model does not use and that holds the draft.
  If the big model is not pinned to a device and there are several GPUs, it
  pins the big model to the largest and drafts on another. With nowhere else
  to go, the draft runs on the CPU (draft GPU layers 0).
- **Use** fills in the draft model, `draft-simple`, the draft device and its
  layers. Save, then load the model.

From a terminal: `python scripts/draft_pick.py big.gguf --models-dir ~/models
--server llama-server`.

A draft on another computer works through RPC: list its `rpc-server` under
Devices → RPC servers, then set the draft device to the RPC device that
`--list-devices` shows. Only do this on a trusted network, because
rpc-server has no authentication. Every drafted token crosses the network,
so this pays off only on a fast link.

### An NVIDIA dGPU and an AMD iGPU in one laptop
CUDA and ROCm cannot share a model, so:
1. Build llama.cpp for **Vulkan** (Setup → Build llama.cpp → vulkan). Both
   chips are then Vulkan devices.
2. Devices: `Vulkan0,Vulkan1`.
3. Split mode: `layer`, which hands the work over once per token.
   Not `row`, which syncs inside every layer over a slow bus.
4. Tensor split weighted to the fast chip, e.g. `85,15`.

Or keep the whole model on the dGPU, and put only the draft model on the
iGPU (draft model device `Vulkan1`).

### AMD with vLLM (ROCm)
- Environment `VLLM_USE_TRITON_FLASH_ATTN=0`: Composable Kernel flash
  attention instead of Triton, often faster prefill.
- `VLLM_ROCM_USE_AITER=1` where the build has AITER.
- Multi-GPU: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (e.g. `INT4`) for
  quick all-reduce.
- On transfer hangs, `HSA_ENABLE_SDMA=0`.
- For a GPU the runtime does not list, `HSA_OVERRIDE_GFX_VERSION`.

### A Ryzen laptop (one CCX, SMT)
- CPU cores: the physical cores, e.g. CPU mask `0x5555` on an 8-core / 16-
  thread part where siblings are adjacent. Check `lscpu -e` for your
  numbering.
- Strict placement on, threads = physical cores.

This keeps threads on one L3 cache instead of migrating.

### Several NUMA nodes (multi-socket, Threadripper/EPYC)
- Launch wrapper `numactl`, node 0, and NUMA `numactl`.

The OS's automatic NUMA balancing (`kernel.numa_balancing`) moves pages
around during inference. Turning it off is a system setting that needs
root (`sysctl -w kernel.numa_balancing=0`), so Studio does not change it.

## Let the machine find its own settings

```bash
python scripts/engine_optimizer.py --model models/m-Q4_K_M.gguf --trials 30 --out runs/opt-m --kv --spec \
    --apply ~/.neuronscope/studio.json
```

It is also in Studio: Jobs → Setup → *Find the fastest engine settings*.
Each trial:
1. starts llama-server with one combination of settings (threads, batch,
   ubatch, flash attention, and with `--kv`, `--spec`, `--placement`: KV
   types, n-gram speculation, CPU placement);
2. sends the same request a few times and reads the server's own timings;
3. stops it.

The search:
- **Bayesian (Optuna TPE)** when optuna is installed (`pip install
  optuna`); otherwise random search, then one option at a time around the
  best;
- the engine's defaults are measured first, so the report says how much
  faster the winner is;
- trials are cached, so a run resumes and a longer run builds on a shorter
  one;
- `--apply` writes the winner into Studio's settings for that model;
- `--fix device=Vulkan0` holds an option at a value.

**Objectives:**
- `gen`: generation tokens/s, the default;
- `prompt`: prompt tokens/s;
- `request`: the time of a whole request shaped by `--prompt-tokens` and
  `--gen-tokens`.

Only speed is measured. A quantized KV cache can change answers slightly,
so check one with TestQA or the gate. The suppression α is a quality
setting: tune it with `tune_scale_server.py`, not for tokens per second.

## Choosing a quantization for each computer

`scripts/quant_select.py` (Jobs → *Choose a quantization*, or **Measure
which suits this computer** under a model in My Models) takes several GGUF
files of one model in different quantizations, or makes them with
`llama-quantize` from a high-precision file (`--from m-F16.gguf --make
Q8_0,Q6_K,Q4_K_M,TQ2_0`). For each one it measures on this computer:

| | how |
|---|---|
| fit | the file against the memory you give (`--memory-gb`), or the largest accelerator's free memory, or 80% of free RAM |
| speed | the engine optimizer's benchmark; `--tune N` also runs its Bayesian search over engine options for that file |
| fidelity | against the highest-precision file: llama.cpp's KL divergence and same-top-token rate (`llama-perplexity --kl-divergence`, found next to llama-server) or, without that tool, how much of each greedy answer to twelve fixed prompts matches word for word |

It keeps the fastest file that fits and clears the fidelity bar (`--max-kld
0.1`, `--min-same-top 85`, or `--min-agreement 0.6`). If none clears it,
it keeps the most faithful file that fits and says so. The choice is stored per hardware
(CPU, accelerators and their memory, never the computer's name) in
`~/.neuronscope/quant_choices.json`, and My Models marks it with ★. A
laptop and a GPU desktop can then each run the file that suits them.

One real run (Qwen2.5-0.5B-Instruct, 4 CPU cores, no GPU, default bars;
KL divergence over 20 chunks of text against the fp16 file):

| quant | size | generation tok/s | KLD | same top token |
|---|---|---|---|---|
| F16 (reference) | 1.18 GB | 26.1 | 0 | 100% |
| Q8_0 | 0.63 GB | 38.8 | 0.002 | 97.3% |
| **Q4_K_M** (chosen) | 0.46 GB | 51.0 | 0.032 | 89.1% |
| Q3_K_M | 0.40 GB | 58.5 | 0.066 | 84.9% |
| Q2_K | 0.39 GB | 57.7 | 0.119 | 79.9% |

On a CPU, smaller files are faster because memory bandwidth is the limit.
Q3_K_M fell just short of the same-top bar, and Q2_K was no faster than
Q3_K_M while losing more. A bigger model, or a GPU, moves these numbers,
which is why the choice is kept per computer.

Switching between quantizations is never lossless. Each file loses
something against the original weights, and the fidelity column measures
how much. The ternary types TQ1_0 and TQ2_0 keep quality only for models
trained ternary (BitNet b1.58 and its relatives). Quantizing an ordinary
model to ternary wrecks it, and the numbers will show that.

## What Studio does not do for you

These are real gains in some setups, but outside what an application should
change, or they come with the engine already:

- **BIOS memory profiles, HugePages, kernel settings:** system
  administration. Measure before and after with the engine optimizer
  (below).
- **CUDA/HIP graphs:** vLLM captures them by default (*No CUDA/HIP graphs*
  turns that off). llama.cpp uses CUDA graphs on its own where they apply.
- **Tree speculative decoding, lookahead/Jacobi decoding:** not in either
  engine's server as a switch. The n-gram methods are the draft-free
  option that exists today.
- **H2O / heavy-hitter KV eviction:** not in llama.cpp or vLLM. Context
  shift with *tokens to keep* is the attention-sink half of StreamingLLM,
  and is available.
- **Mamba / SSM hybrids, Mixture-of-Depths, ternary (BitNet b1.58)
  models:** these are model choices. llama.cpp loads the architectures it
  supports (Mamba, Mamba-2, Jamba, several hybrids, and BitNet through its
  TQ1/TQ2 types). Load such a GGUF like any other. TestQA and the gate tell
  you whether it is better for your work, and OmniCortex's neuron tools
  read activations the same way.
