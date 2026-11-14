# OmniCortex

A cross-platform workbench for finding, visualizing and editing the neurons
behind confident mistakes in local models. It also covers serving, routing,
benchmarking and moving those models between machines.

OmniCortex began as a reworking of [thunlp/H-Neurons](https://github.com/thunlp/H-Neurons)
(Gao et al., [arXiv:2512.01797](https://arxiv.org/abs/2512.01797)). That paper
shows a sparse set of MLP neurons ("H-Neurons") is causally linked to
over-confident answers, and scaling them down makes a model more willing to
abstain. OmniCortex extends the method in several directions:

- **any hardware**: llama.cpp (Vulkan, CUDA, Metal, CPU) or PyTorch (CUDA, ROCm, MPS, CPU);
- **any serving stack**: LM Studio, llama-server, or any OpenAI-compatible endpoint;
- **text and image-text models**: decoder LLMs, plus CLIP and SigLIP;
- **ships the result**: edited GGUFs, LoRA adapters, `mmproj` projectors and HF checkpoints that load where you already run models.

**The name.** This project was called NeuronScope until November 2026. It is now OmniCortex, because
[RudrenduPaul/NeuronScope](https://github.com/RudrenduPaul/NeuronScope) is a different, earlier project with
that name (a TransformerLens-based CLI for single-component ablation and circuit experiments, whose ideas
OmniCortex has since taken up: [docs/INTERVENTIONS.md](docs/INTERVENTIONS.md)). Everything now carries the new
name. For an existing install:
- start it with `./omnicortex` (`./neuronscope` is gone);
- `~/.neuronscope` and `~/.config/neuronscope` are moved to `~/.omnicortex` and `~/.config/omnicortex` the first
  time OmniCortex starts, unless the new folders already exist;
- environment variables are `OC_*` (was `NS_*`), and `OCR_*` for the remote agent (was `NSR_*`): rename them in
  `env.local.sh` and anywhere else you set them;
- the MCP server is `omnicortex`: write the Cline / Claude Desktop entry again (Connect → MCP) or rename the
  `neuronscope` key in its settings;
- the route header is `X-OmniCortex-Route`, and the ComfyUI nodes are `OmniCortex…`: re-add them in saved
  workflows;
- the remote agent is `omnicortex-remote` (the Python package `omnicortex_remote`); update it on each host.

Prebuilt binaries for Windows, macOS and Linux (x64, x86, ARM64, ARM): [docs/BINARIES.md](docs/BINARIES.md).

Progress, what has been checked on real hardware, and the plan: [docs/STATUS.md](docs/STATUS.md).

## What's in the box

| Area | What it does | Guide |
|---|---|---|
| **H-Neuron pipeline** | collect → label → extract CETT → sparse classifier → tune scale → export (GGUF in place, LoRA, HF) | [docs/PIPELINE.md](docs/PIPELINE.md) |
| **Features and emotion vectors** | sparse autoencoders and transcoders (TopK, ReLU, JumpReLU; or imported from SAELens / Gemma Scope), feature descriptions and labels, feature-to-neuron maps with H-Neuron overlap, per-token reading of any reply, attribution graphs with interventions, emotion vectors after Anthropic's emotions paper, and steering with both | [docs/FEATURES.md](docs/FEATURES.md) |
| **Interventions** | which attention heads and neurons drove an output (direct logit attribution), zero and mean ablation of components, heads, neurons and SAE features, clean-vs-corrupted activation patching by layer, position or head, path patching, circuit sketches, Neuronpedia labels; as jobs and the MCP tool `intervene` | [docs/INTERVENTIONS.md](docs/INTERVENTIONS.md) |
| **Permissions** | what each model may read, write, run, start, reach and call: allow and deny lists with exact, glob and regex patterns, per model and per job (presets for QA runs, TestQA code and SWE-bench), enforced in chat, TestQA and project tasks, with Allow once / Always allow / Deny prompts that show the command | [docs/PERMISSIONS.md](docs/PERMISSIONS.md) |
| **Graphs** | node graphs that screen, verify and route model inputs and outputs: judges, hallucination checks from the model's own neurons, emotion and feature screens, secrets screens, self-consistency, cross-model checks, grounding, sandboxed code tests, loops, human approvals, evals and plugin nodes; projects can screen every delegated result with one | [docs/GRAPHS.md](docs/GRAPHS.md) |
| **Add-ons** | remove the parts of Studio you do not use (the decoding lab, features, graphs, retraining, SWE-bench, …) and install packages from a file or a repository (SHA-256 checked); default packages for the Godot client, jewel styles, themes and graph screens; ComfyUI as a Studio tab with LLM node packs and OmniCortex's own ComfyUI nodes | [docs/ADDONS.md](docs/ADDONS.md) |
| **Widgets** | the emotion jewel (reactor core, icosphere or tesseract driven by the emotion vectors), emotion lanes, Blender-style 3D navigation and a search line, the same in the browser and the Godot client, set under Settings → Appearance | [docs/WIDGETS.md](docs/WIDGETS.md) |
| **Vision H-Neurons** | the same method for CLIP/SigLIP: confident image-caption mismatches, `mmproj` editing | [docs/VISION.md](docs/VISION.md) |
| **Studio** | LM Studio-style model library, chat with images, saved chats, Hugging Face downloads, and an OpenAI-compatible `/v1` server with JIT loading, idle unload and `model: "auto"` routing; chat with your documents (RAG), MCP tools with per-call approval, paired devices and linked machines' models, a jobs page for evaluation and retraining; plus a live suppression slider and MoE expert control | [docs/STUDIO.md](docs/STUDIO.md) |
| **TestQA** | 210 graded items, 30 in each of seven subjects (reasoning, executable code, API existence, false-premise facts, instruction-following writing, generated-image vision) against any endpoints; per-subject right/hallucinated/abstained rates; paired comparison; feeds routing stats | [docs/TESTQA.md](docs/TESTQA.md) |
| **Subject classifier + stats routing** | labels prompts (code, math, logic, science, factual, writing, vision); `model: "auto"` picks the model with the best measured record for that subject, from rolling per-model stats | [docs/TESTQA.md](docs/TESTQA.md#subject-classifier) |
| **Transfer** | encrypted P2P browser transfer, resumable HTTPS CLI with watch mode, remote tuning worker, MCP control plane | [docs/TRANSFER.md](docs/TRANSFER.md) |
| **External benchmarks** | CLIP_benchmark (ImageNetV2, ImageNet-Sketch, VTAB, ...) for CLIP and its edits; LiveBench and SWE-bench for chat/coding models, feeding stats; leaderboard (BenchLM) imports as reference | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| **Retraining on deficits** | turns measured failures into verified SFT and DPO data with replay and a holdout, trains with QLoRA, LoRA or full fine-tuning (DDP/FSDP), merges and converts to GGUF | [docs/RETRAINING.md](docs/RETRAINING.md) |
| **Labs** | capability-preserving quantization, scale sweeps, adaptive remote tuning | [docs/LABS.md](docs/LABS.md) |
| **Connect** | your models as the AI of Cline, Claude Desktop or any OpenAI-compatible app; OmniCortex's harness and lab as MCP tools (models, reply checks, jobs, projects), with a Connect page that writes the settings | [docs/CONNECT.md](docs/CONNECT.md) |
| **Neuron review** | every neuron's link to wrong answers across tests, benchmarks and checked chat replies, filtered by source, subject and time, played over time in 3D | [docs/REVIEW.md](docs/REVIEW.md) |
| **Unified review and browser extension** | one record per model from review boards, hardware reports, SAELens and your own Studio; a Firefox/Chromium/Opera extension whose icon glows blue on Hugging Face pages that have data, with a dropdown per statistic and source | [docs/UNIFIED_REVIEW.md](docs/UNIFIED_REVIEW.md), [docs/BROWSER_EXTENSION.md](docs/BROWSER_EXTENSION.md) |
| **Looped models (NVIDIA Dory)** | serve a looped model with its loop count and KV mode as settings; sweep the loop count against TestQA (accuracy, confident mistakes, speed); a per-loop lens showing which pass decides the answer and where the H-neurons fire; a NeMo Gym environment whose reward charges for confident mistakes | [docs/DORY.md](docs/DORY.md) |
| **Truth probes** | does the model believe what it said? A probe on every layer, for Hugging Face models or any GGUF, flags replies whose activations favour another answer ("knows but says otherwise"), with a per-layer trajectory; a signal, not proof of intent | [docs/TRUTH_PROBE.md](docs/TRUTH_PROBE.md) |
| **Proxmox** | run the remote agent or Studio in an LXC container with the host's GPUs (AMD or NVIDIA, legacy or current); the only GPU route on boards with non-coherent PCIe such as RK3588; plus an MI50 VBIOS check | [docs/PROXMOX.md](docs/PROXMOX.md) |
| **Clients** | the web pages, a terminal client (`./omnicortex cli`) and a Godot client (`./omnicortex godot`) over the same API; the Godot client is a full UI with a native 3D view, built to run on another computer so drawing never takes GPU memory from the model | [docs/CLIENTS.md](docs/CLIENTS.md) |
| **Remote hosts** | use the models on computers that run LM Studio, Ollama or llama.cpp but not OmniCortex: a small agent there (distributed on its own) is controlled over MCP; same network, VPN (Tailscale, ZeroTier, WireGuard…), SSH, a public tunnel, or the host calling in from behind NAT; pairing with a short code, and the host's owner sees and revokes every controller | [docs/REMOTE.md](docs/REMOTE.md), [remote/](remote/README.md) |
| **Settings** | named workspaces for every folder and binary (no fixed layout), the default model per subject for `auto`, and rules for when chat tool calls are allowed, asked or denied (by tool, kind of operation, argument regexes, rate limits) | [docs/SETTINGS.md](docs/SETTINGS.md) |
| **Requirements** | what each feature needs (packages, programs, llama.cpp GPU backends, drivers, device access, memory), fixes for your OS, one-click Python installs, environment snapshots and diffs | [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) |
| **Visualization** | 2D/3D activation maps, token-resolved timelines, live tracing proxy, weight views | [docs/VISUALIZATION.md](docs/VISUALIZATION.md) |
| **Merging** | neuron-granular selective merges, mergekit plugin, MoE expert-count sweeps | [docs/MERGING.md](docs/MERGING.md) |

New to the terms (GGUF, CETT, H-Neurons, abstention, MCP…)? The
[glossary](docs/GLOSSARY.md) explains each one and why it matters here.

## Quick start

Double-click the installer for your system: **Install OmniCortex.bat**
(Windows), **Install OmniCortex.command** (macOS) or **install-gui.sh**
(Linux). A wizard opens in your browser, checks the computer, picks the
right PyTorch build for your GPU (or none), installs, and adds a shortcut
([details](docs/GETTING_STARTED.md#the-short-way)). In a terminal:

```bash
python3 scripts/installer.py --yes   # or ./install.sh on Linux and macOS
./omnicortex                        # or the shortcut: starts Studio and opens it in your browser
```

From there everything is in the browser: **Setup** builds llama.cpp and finds
your models, **Jobs** runs evaluations, retraining and every pipeline stage,
**Lab** opens the labs and 3D views, **Projects** plans work for worker models,
**Connect** sets up Cline. [docs/UI_AND_CLI.md](docs/UI_AND_CLI.md) maps every
feature to its place in the UI, its command and its MCP tool.

From a terminal instead:

```bash
source venv/bin/activate
#   ./install.sh --no-torch                  # lighter: Studio, TestQA, Projects and the llama.cpp path need no PyTorch
python scripts/doctor.py                     # checks Python, packages, GPUs, llama.cpp, your model; says what to do next
scripts/build_llama_tools.sh --backend vulkan  # llama.cpp + the cett-dump extractor (or cpu, cuda, hip, metal); ~5 min
```

Full setup: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). Hardware notes,
including AMD APUs: [docs/HARDWARE.md](docs/HARDWARE.md). OmniCortex supports legacy and modern
hardware alike; [hardware support profiles](docs/HARDWARE_PROFILES.md) record the right setup per family and purpose.

Each block below is safe to paste as a whole. Set your folders once (or set
them on Studio's **Settings** page and skip the flags):

```bash
MODELS=~/models                               # the folder that holds your .gguf files
LLAMA=~/llama.cpp/build/bin                   # where build_llama_tools.sh put llama-server
```

Serve and chat with local GGUFs. This keeps running; OpenAI clients use
`http://127.0.0.1:7870/v1`. It prints the model ids it found.

```bash
python viz/studio.py --models-dir "$MODELS" --server "$LLAMA/llama-server"
```

In a second terminal, benchmark a model on reasoning, coding, facts and
canaries. `--allow-exec` runs the model's code to grade it; `--sandbox docker`
(or `podman`) keeps it in a container. Replace the id with one Studio printed:

```bash
MODEL_ID=acme/model-q4_k_m
python scripts/testqa.py --endpoint "base=http://127.0.0.1:7870/v1@$MODEL_ID" --allow-exec --sandbox docker
```

Find H-Neurons in a CLIP model, from a folder of images:

```bash
IMAGES=~/Pictures/samples
python scripts/clip_neurons.py collect --model openai/clip-vit-base-patch32 --images "$IMAGES" --out runs/clip
```

## Read before you trust a result

- **Suppression is not correction.** Scaling H-Neurons down trades confident
  errors for abstentions. It does not add knowledge. Measure both sides; on
  coding workloads the trade may not be worth it.
- **Scope is narrower than "hallucination".** Text labels come from short-form
  factual QA by default. Long-form confabulation, RAG groundedness and
  invented APIs may involve different neurons; collect from the domain you
  care about.
- **Do not extract from a quantized model.** CETT measures activation
  magnitudes, which quantization perturbs. Generate from a quant if you like.
- **Labels are the leverage.** Validate the judge (`judge_agreement.py`) before
  tuning anything downstream.

More in [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).

The research, datasets and software OmniCortex uses are credited in
[docs/CREDITS.md](docs/CREDITS.md).

## Security

Every network service follows one policy. Loopback binds work without
credentials. Any other bind requires a 256-bit token, and TLS unless you
explicitly state the port is behind a VPN or TLS proxy. WebRTC transfers are
always DTLS-encrypted and can be verified with an out-of-band code. See
[docs/SECURITY.md](docs/SECURITY.md) for the threat model and checklist.

## Repository layout

```
install.sh, env.sh           environment setup (env.local.sh for your paths, gitignored)
scripts/                     pipeline stages, exporters, evaluators, transfer, security
  oc_common.py, profiles.py    shared hooks, sequence building, portable profiles
  clip_neurons.py              vision H-Neurons; suppress_mmproj.py edits mmproj GGUFs
  testqa.py                    evaluation harness; subject_classifier.py routing
  oc_transfer*.py, model_transfer.py, tuning_worker.py, oc_security.py
viz/                         browser and native frontends (studio, hub, labs, explorers)
web/                         WebRTC transfer page
qa/                          TestQA bank (one file per subject), subject seed data
llama-tools/                 cett-dump extractor and the optional activations server patch
tests/                       pytest suite; no model or GPU needed
docs/                        guides
```

## Tests

```bash
pip install pytest && python -m pytest -q
```

Set `OC_TEST_CACHE=~/.cache/omnicortex-tests` to build the tiny test models once and reuse them; CI caches that
folder. With `OC_LLAMA` pointing at a llama.cpp build with cett-dump, the llama.cpp integration tests run too.

## Licence

MIT, following upstream H-Neurons.
