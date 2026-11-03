# Hardware notes

OmniCortex supports legacy and modern hardware alike. For a ready-made setup per hardware family and
purpose (inference, retraining), see [HARDWARE_PROFILES.md](HARDWARE_PROFILES.md).

OmniCortex does two very different kinds of work, and they want different
hardware:

| Work | Bound by | Runs well on |
|---|---|---|
| Generation (collect responses, evaluate, chat) | memory bandwidth per token | a discrete GPU, via llama.cpp or LM Studio |
| Activation extraction (`cett-dump`) | one prefill per sample | any llama.cpp backend, including integrated GPUs |
| Classifier training | RAM | CPU; ~3 GB at 400 pairs, ~7 GB at 1000 |
| PyTorch hook extraction | VRAM at bf16 | a GPU that fits the full bf16 model, or CPU |

The pipeline is split so each stage can run on the machine that suits it.
Collection and scale tuning talk to any OpenAI-compatible server over HTTP, so
a laptop can drive a GPU box elsewhere on the network; see
[TRANSFER.md](TRANSFER.md) for moving the resulting files.

## Low-power hosts

On an integrated GPU or a CPU-only machine:

| stage | where | why |
|---|---|---|
| 1 collect | a GPU host, over HTTP | generation is bandwidth-bound; an iGPU manages a few tokens/s |
| 4 extract | locally, `cett-dump` on Vulkan/CPU | prefill only, no generation |
| 5 classify | locally, CPU | sklearn |
| 7 tune | a GPU host, `tune_scale_server.py` | many generations per scale |

`cett-dump` loads the model once for the whole manifest and resumes from
existing dumps, so it is the stage that runs well on modest hardware.

## Backends

**Vulkan** (llama.cpp) is the most portable GPU path: AMD, Intel and NVIDIA,
Linux and Windows. PyTorch has no usable Vulkan backend, so Vulkan exists only
inside llama.cpp here. Systems with more than one Vulkan driver installed
(e.g. AMDVLK and RADV) sometimes pick one that fails to initialise; set
`NS_VULKAN_ICD` to the driver JSON you want before `source env.sh`.

**Metal**: build llama.cpp with `--backend metal`; `install.sh` installs the
default wheel, which uses MPS.

### NVIDIA / CUDA

```bash
python scripts/cuda_info.py                    # every GPU, and what fits it
./install.sh                                   # picks the PyTorch wheel from that
scripts/build_llama_tools.sh --backend cuda    # llama.cpp for exactly these GPUs
python scripts/doctor.py                       # checks torch kernels, toolkit, driver
```

`scripts/cuda_info.py` reads `nvidia-smi` (it works before PyTorch is
installed) and makes every CUDA decision in one place. The installer, the
build script, Studio, `doctor.py` and `finetune.py` all use it.

| GPUs | PyTorch | llama.cpp toolkit | train in | QLoRA |
|---|---|---|---|---|
| Ampere, Ada, Hopper, Blackwell (sm_80+) | default PyPI wheel | any current CUDA | bf16 | yes |
| Turing (sm_75) | default PyPI wheel | any current CUDA | fp16 | yes |
| Volta (sm_70), P100 (sm_60) | CUDA 12.6 wheel, `torch<2.15` | CUDA 12.x | fp16 | check `finetune.py --check` |
| Pascal consumer/P40 (sm_61), Maxwell (sm_50/52) | CUDA 12.6 wheel, `torch<2.15` | CUDA 12.x | fp32 | Maxwell: no |

Why the split: PyTorch's CUDA 12.8+ wheels dropped Maxwell, Pascal and Volta,
and 2.14 is the last release that still publishes the CUDA 12.6 wheels that
have them ([pytorch#190385](https://github.com/pytorch/pytorch/issues/190385)).
CUDA 13 cannot compile for anything below sm_75, so llama.cpp for those cards
needs a CUDA 12.x toolkit (12.9 is the newest), and the build script refuses a
CUDA 13 build for them instead of producing binaries that fail with "no kernel
image". The R580 driver branch is the last that supports them: pin it.
bitsandbytes dropped Maxwell, so QLoRA is unavailable there; `finetune.py`
says so and LoRA works. Precision follows the slowest GPU: bf16 needs Ampere,
fast fp16 needs Volta/Turing or the P100, and Maxwell and consumer Pascal run
fp16 at a small fraction of fp32 speed, so they train in fp32.

No local toolkit, or the wrong one: build in a container.
`docker/llama-cuda.Dockerfile` builds cett-dump and the activation-streaming
llama-server with CUDA 12.9 for any architecture list (default sm_50):

```bash
docker build -f docker/llama-cuda.Dockerfile --build-arg CUDA_ARCH="50-real" -t neuronscope-llama:cuda .
docker run --rm --gpus all -v ~/models:/models -p 8080:8080 neuronscope-llama:cuda \
    llama-server -m /models/model.gguf -ngl 99 -sm layer --host 0.0.0.0 --port 8080
```

**Several GPUs.** Layer split (`-sm layer`, llama.cpp's default) puts whole
layers on each GPU and passes one activation between them per token, so it
works over plain PCIe. Row split (`-sm row`) splits every matrix and needs fast
GPU-to-GPU links. Studio's load panel has **Visible GPUs**
(`CUDA_VISIBLE_DEVICES`), **Split**, **Tensor split** and **Main GPU**, and
its fit estimate counts free VRAM across all of them. `vram_budget.py`,
`hostcheck.py` and host profiles sum VRAM over every GPU too. Training spreads
over GPUs with `finetune.py --launch N` (DDP); each GPU holds the whole model
plus adapters, so per-GPU memory is the limit there, not the total.

**Tesla M10.** One board, four Maxwell GPUs (sm_50) with 8 GB each. In the
table above it is the last row: `torch<2.15` from the CUDA 12.6 index, llama.cpp
built with CUDA 12.x for `50-real`, driver R580, fp32 LoRA. Serve one model
across all four (`-sm layer -ts 1,1,1,1`), or up to four small models side by
side, each with its own **Visible GPUs**. It has no display outputs and a
passive heatsink, so it needs server airflow.

Verified here, without a GPU:

- `docker build -f docker/llama-cuda.Dockerfile .` builds end to end: it fetches
  the pinned llama.cpp commit, applies the activation patch, compiles 144 sm_50
  kernel images into `libggml-cuda.so`, and ships cett-dump, the patched
  llama-server, llama-quantize and llama-eval-callback in a 6.4 GB runtime image.
- The image's binaries run: with the toolkit's stub `libcuda.so.1` mounted,
  CUDA reports no devices and llama.cpp falls back to the CPU, so the container's
  llama-server streams `/activations` frames that match PyTorch on every layer,
  and its cett-dump matches the host build
  (`NS_DOCKER_IMAGE=neuronscope-llama:cuda NS_DOCKER_LIBCUDA=<stub>` runs these
  in `tests/test_llamacpp_integration.py`).
- The CUDA decision table is unit-tested against M10, RTX 4090 and mixed-GPU
  `nvidia-smi` output.

Not verified: the CUDA kernels themselves on NVIDIA hardware.

Behind a proxy that re-signs TLS, pass its CA to the build with
`--secret id=ca,src=/path/ca.pem`. `--build-arg LLAMA_REF=master` builds the
newest llama.cpp instead of the tested commit; the patch stops with a clear
message if llama.cpp has moved the code it hooks into.

**ROCm (AMD)**: `install.sh` detects the gfx target and picks a PyTorch ROCm
wheel index, then runs a bf16 matmul against CPU to prove the result is
correct, not merely that a kernel ran.

- RDNA4 (gfx1200/1201) needs ROCm 7.x wheels.
- RDNA2 dies below gfx1030 need `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- RDNA1 and Vega APUs are unsupported by the wheels; `install.sh` installs CPU
  PyTorch and the Vulkan path does the heavy lifting.
- Distros AMD does not officially support usually work anyway, because the pip
  wheels bundle the ROCm userspace and only need the mainline `amdgpu` driver.
  `./install.sh --docker` prints a container recipe if they don't.

Diagnostics:

```bash
./scripts/rocm_inventory.sh      # real gfx target, system vs wheel rocBLAS kernels
./scripts/try_rocm.sh            # correctness check of an override against CPU
./scripts/try_rocm.sh --install  # install a ROCm torch first
```

If system rocBLAS has kernels for your GPU but the wheel does not, prefer a
distro-built PyTorch (`./install.sh --system-torch` creates a venv with system
site packages) over mixing a wheel with kernels from another ROCm major
version.

## Worker devices

The director's worker models (see [DIRECTOR.md](DIRECTOR.md)) go wherever
`scripts/accelerators.py` finds room, AMD first. It reads `nvidia-smi`,
`/sys/class/drm` (AMD VRAM and GTT, with or without ROCm), `rocminfo` (run
with `HSA_OVERRIDE_GFX_VERSION` unset, so it reports the real chip),
`vulkaninfo --summary`, and the platform (Metal on Apple silicon). Each device
gets an id like `rocm:0` or `vulkan:1`.

```bash
python scripts/accelerators.py      # devices, free memory, notes, which server drives each
```

Configuration lives in `~/.neuronscope/hardware.json` (Studio `--hardware`):

```json
{
  "prefer": ["amd", "nvidia", "intel", "apple", "arm"],
  "servers": {"vulkan": "~/llama.cpp/build-vulkan/bin/llama-server",
              "rocm":   "~/llama.cpp/build-rocm/bin/llama-server"},
  "devices": {
    "rocm:0":   {"enabled": true, "reserve_gib": 3,
                 "rocm_path": "/opt/rocm-6.2.4",
                 "server": "~/llama.cpp/build-rocm624/bin/llama-server",
                 "settings": {"threads": 8, "ctx": 8192},
                 "env": {"HSA_OVERRIDE_GFX_VERSION": "9.0.0"}},
    "vulkan:0": {"enabled": false}
  },
  "manual": [{"id": "vulkan:2", "backend": "vulkan", "index": 2, "name": "eGPU", "memory_total_gib": 8}]
}
```

- `servers`: a llama-server build per backend; a device's own `server` wins.
  Without either, Studio's `--server` is used.
- `rocm_path`: pin a ROCm/HIP release for one device. Its llama-server gets
  `ROCM_PATH`, `HIP_PATH` and `LD_LIBRARY_PATH` for that release; nothing else
  on the machine changes. A path that does not exist turns the device off.
- `env`: only `HSA_*`, `HIP_*`, `ROCR_*`, `ROCM_*`, `GGML_*`, `CUDA_*`, `MTL_*`,
  `VK_*`, `AMD_*` and `RADV_*` variables pass; anything else (`LD_PRELOAD`, …)
  is dropped.
- `settings`: `ngl`, `ctx`, `batch`, `threads`, `flash_attn`, `cache_type`,
  `parallel`, `extra`. A project can override these per device in its policy.
- `reserve_gib`: memory to leave alone (default 1 GiB on shared-memory
  devices, 0.5 GiB on discrete GPUs).
- `manual`: devices detection misses.

**One GPU, two entries.** A GPU reachable natively and through Vulkan is
listed twice; the Vulkan entry is off while the native one is on, so one GPU is
never booked twice.

**AMD APUs.** Usable memory is the VRAM carve-out plus GTT, which is system
RAM: the APU and CPU workers share it, and placement counts it once. On Linux
the GTT size is a kernel setting (`amdgpu.gttsize`, in MiB); raise it if
models that fit in RAM do not fit on the iGPU.

- *Vega-based APUs* (gfx90c: Renoir, Cezanne, Barcelo, e.g. Ryzen 5000U and
  7030U such as the 7730U): ROCm does not support them. The ROCm entry is
  **off** and the Vulkan entry for the same GPU is used. To try ROCm, enable
  `rocm:N`; Studio then applies `HSA_OVERRIDE_GFX_VERSION=9.0.0`, the gfx900
  masquerade, which can hang the GPU or compute wrong results. Check it with
  `scripts/try_rocm.sh` first, and pin the ROCm release that worked for you
  with `rocm_path` and a matching `server`.
- *RDNA2/RDNA3 APUs and small dies* (gfx1031–1036, gfx1103): the usual
  override (10.3.0 / 11.0.0) is applied automatically and the ROCm entry is on.
  On an APU, ROCm workers also get `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` so they
  can allocate from GTT, not only the carve-out.

**Arm boards with a PCIe GPU** (e.g. a Tesla M10 on an RK3588 with the
`m10-arm` driver patches): the board's Mali GPU shows up through Vulkan with
estimated memory, so a model too big for one M10 is split across M10s before
the Mali is considered. When the patched driver maps system memory uncached,
the device note suggests trying `GGML_CUDA_NO_PINNED=1` if prompt processing
is slow (untested on hardware).

## Sizing

```bash
python scripts/vram_budget.py --gguf "$NS_GGUF" --ctx 32768 --vram 12
python scripts/hostcheck.py
```

`vram_budget.py` reads the model's real attention shape and reports what fits
at a given context. `hostcheck.py` estimates whether a stage will fit this
machine before it starts, rather than after it fails; the classifier and
Studio both call it.

For MoE models on consumer hardware, keep routed experts in system RAM and
everything else on the GPU:

```bash
--override-tensor '\.ffn_.*_exps\.=CPU'
```

Generation is bound by *active* parameters, so a Q4 30B-A3B can be faster than
a dense Q6 9B despite being three times larger on disk.

## Speed predictions

**Tasks → Distribution → Predicted speed** shows how fast each of your models
should generate on each of your devices (`scripts/perf_predict.py`).

**The estimate.** Generating one token reads every active weight once, so the
starting point is *efficiency × memory bandwidth ÷ model size*. Bandwidths
come from a table of published specs. A device not in the table is treated as
a typical CPU, integrated GPU or discrete GPU until it has been measured.

**What it learns.** Every measurement corrects that estimate:
- **Measurements.** Studio's work log records the server's own timings for
  every reply on this computer. Cards on the scoreboard carry the speed their
  runs measured on other people's hardware.
- **Shared weights.** A ridge regression in log space learns how much each of
  these changes speed:
  - the backend (CUDA, ROCm, Vulkan, Metal, SYCL or CPU);
  - bits per weight;
  - model size;
  - layers left on the CPU;
  - flash attention;
  - speculative decoding;
  - parallel requests;
  - unified memory.

  Results from many different machines teach these weights faster than
  results from one machine.
- **Per-device offsets.** Each device gets its own correction for everything
  the features miss, such as drivers, clocks and cooling. The correction is
  shrunk toward zero until the device has several measurements
  (n ÷ (n + 3)), so one odd run cannot swing it.

**How sure.** Each prediction is a value and an 80% range:
- The range is narrow on devices with many measurements, and wide on hardware
  never measured.
- **Typical error** is the median error on measured hardware.
- **Unseen hardware** error comes from leaving each device out in turn and
  predicting it from the rest. It is shown once three devices have
  measurements. It says how far to trust a prediction for a card you do not
  own yet.

**What the prediction has learned** lists each feature's effect, as a factor
on tokens per second, and each device's correction. These move as more
results arrive.

A grey cell is a model too large for that device's memory. Prompt processing
(compute-bound) is not predicted: the measurements rarely separate it from
generation.
