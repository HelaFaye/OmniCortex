# Glossary

The terms OmniCortex's docs and UI use, in plain language, with why each one
matters here. Ordered so that each entry only leans on the ones above it.

## Models and files

**GGUF.** The single-file model format llama.cpp, LM Studio and Ollama load.
It holds the weights (usually quantized) plus the tokenizer and chat template.
Most of OmniCortex reads and writes GGUFs directly, so it works with the
models you already have.

**Quantization (Q4_K_M, Q6_K, Q8_0, F16).** Storing weights in fewer bits to
save memory. Q4 is about a quarter of F16's size and loses a little quality; Q8
is close to lossless. *Here:* run and test any quant you like, but find
H-Neurons on a higher-precision copy (Q8 or F16), because measuring activation
sizes on a heavily quantized model measures the rounding as well.

**mmproj.** The vision half of a multimodal GGUF model: a separate file
(`mmproj-*.gguf`) that turns an image into tokens the language model can read.
Studio pairs it with its model automatically; `suppress_mmproj.py` edits it.

**Context length (ctx).** How many tokens a model can see at once: prompt plus
reply. It is fixed by training (a model "trained to 8k" is unreliable past it)
and costs memory (the KV cache grows with it). Studio's director skips models
whose context is too short for a task.

**llama.cpp / llama-server.** The C++ engine that runs GGUFs on CPUs and GPUs,
and its HTTP server. Studio starts `llama-server` for you; OmniCortex adds two
tools to it (`cett-dump`, and an optional live `/activations` stream).

**`-ngl` (GPU layers).** How many of the model's layers llama.cpp puts on the
GPU. `99` means all of them. Lower it when a model does not fit; the rest run
on the CPU, slower.

**Backend: CUDA, ROCm, Vulkan, Metal, CPU.** The way llama.cpp or PyTorch
reaches the hardware: CUDA for NVIDIA, ROCm (HIP) for AMD, Vulkan for almost
any GPU including AMD APUs, Metal for Apple. Vulkan is the most portable; ROCm
and CUDA are usually faster on the cards they support. See
[HARDWARE.md](HARDWARE.md).

**APU / unified memory / GTT.** An APU is a CPU with a built-in GPU (e.g. AMD
Ryzen laptops). It has no memory of its own: a small "VRAM" carve-out plus GTT,
system RAM the GPU can map. OmniCortex counts both, and remembers that the CPU
uses the same RAM.

**OpenAI-compatible API (`/v1`).** The HTTP interface most tools speak
(`/v1/chat/completions`, `/v1/models`). Studio serves one, so Cline, Open WebUI,
scripts and TestQA can all use your local models. The model id (shown at
Studio's startup and in `/v1/models`) is what goes in `"model"`.

**JIT loading / idle unload (TTL).** Studio loads a model when a request names
it ("just in time") and can unload it after a period without requests, as LM
Studio does.

**MoE (mixture of experts).** A model whose feed-forward layer is many small
"experts", of which only a few run per token. OmniCortex handles their neurons
per expert, and Studio can change how many experts run.

**MCP (Model Context Protocol).** A standard for giving a model tools: an MCP
*server* offers tools (read files, search, run a job), an MCP *client* (Studio's
chat, Cline, Claude Desktop) lets the model call them, with your approval.

## What OmniCortex measures

**Hallucination (as used here).** A confident answer that is wrong. The method
targets the *confident* part: it cannot add missing knowledge, only make a model
more willing to say it does not know.

**Abstain / abstention.** The model declining to answer ("I don't know").
TestQA counts it separately from right and wrong: an abstention costs nothing,
a wrong answer costs something, so a model that abstains when unsure can rank
above one that guesses.

**Consistency sampling.** Asking the same question several times. If the model
gives the same right answer every time it "knows" it; the same wrong answer every
time is a confident error. Those two groups are what stage 1 collects.

**Judge.** A model (or rule) that labels answers right or wrong.
`judge_agreement.py` checks it against hand labels before you trust anything
built on its labels.

**MLP neuron.** One unit in a layer's feed-forward (MLP) block, the part of each
transformer layer that stores most facts. An 8B model has about 14,000 per layer
and 32 layers.

**CETT.** The quantity OmniCortex measures for each neuron: how much it
contributes to its layer's output on the answer tokens: its activation times
the size of its outgoing weights, relative to the layer's whole output,
`|a| × ‖W_down[:, neuron]‖ / ‖layer output‖`. It is what the H-Neurons paper
uses, and what `cett-dump` captures from llama.cpp.

**H-Neurons.** The small set of neurons (often well under 0.1%) whose CETT
predicts a confident wrong answer. Found by training a sparse classifier on
CETT from right versus wrong answers; scaling them down makes a model abstain
more and assert falsehoods less.

**Sparse classifier (L1 logistic regression).** A classifier that is pushed to
use as few neurons as possible, so the neurons it keeps are the ones that matter.
Its weights are the H-Neuron list (`h_neurons.json`).

**AUROC.** How well a score separates two groups: 1.0 perfect, 0.5 a coin toss.
OmniCortex's classifier is around 0.7 on whole replies: a useful signal, not a
verdict, which is why the UI says "risk" rather than "wrong".

**Suppression scale (alpha).** The factor the H-Neurons' outputs are multiplied
by: 1.0 unchanged, 0.0 silenced. Tuned per model, because too low damages the
model. Applied by editing the GGUF, a LoRA, or live in Studio.

**Risk (per token).** In Studio's check and the 3D views: the classifier's
probability that a token's activity looks like a confident error. Tokens at or
above 0.5, the classifier's own boundary, are flagged.

**Canary.** A TestQA item with an obvious right answer. If a model fails it, the
problem is the setup (template, endpoint), not the model.

## Training and testing

**TestQA.** OmniCortex's graded question bank (210 items over seven subjects)
for any OpenAI-compatible endpoint. Its results become the per-subject stats
that `model: "auto"` and the director use to pick models.

**Subject / skill labels.** The subject classifier sorts a prompt or a task
into code, math, logic, science, factual, writing, vision, graphics, systems or
reverse-engineering (or several, or "unknown") so routing can pick the model
with the best record for it.

**LoRA / QLoRA.** Training a small add-on (adapter) instead of the whole model;
QLoRA does it on a 4-bit copy to fit in less memory. A LoRA can be merged into
the model or loaded next to it.

**SFT / DPO.** Two ways to fine-tune. SFT (supervised fine-tuning) shows the
model good answers. DPO (direct preference optimization) shows it a better and
a worse answer to the same prompt. Retraining uses both, built from the model's
own measured failures.

**Replay buffer / holdout.** Old, unrelated examples mixed into retraining so
the model does not forget what it could already do (replay), and examples kept
out of training to check that it really improved (holdout).
