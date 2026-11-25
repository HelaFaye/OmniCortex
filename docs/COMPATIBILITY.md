# Hardware compatibility

What each part of OmniCortex does on each family of hardware. This page is generated from
`profiles/compatibility.json` by `python scripts/compat.py doc`, and a test fails if the two disagree.
Reviewed 2026-11-24.

**Statuses:**

- **verified**: run on this hardware by OmniCortex's tests or a recorded trial; the evidence says which
- **expected**: should work from the engine's or library's own support, but not run here yet
- **partial**: works with a limit the note states
- **unsupported**: does not work; the note says why
- **unknown**: not known either way
- **n/a**: does not apply to this hardware

**Only the CPU rows have been run on real hardware so far**: everything OmniCortex tests runs in CI on an
x86 CPU. Every GPU cell is *expected*, *partial* or *unsupported* until it has been run on that card and its
evidence recorded here.

| Hardware | llama.cpp | mx-llama.cpp | Radiance | PyTorch | H-neurons (GGUF) | Truth probe (GGUF) | LoRA | QLoRA | bf16 | Quant fidelity |
|---|---|---|---|---|---|---|---|---|---|---|
| `cpu-x86` | ✅ verified | – | expected | ✅ verified | ✅ verified | ✅ verified | ✅ verified | ✗ no | ◐ partial | ✅ verified |
| `arm-board` | expected | – | ✗ no | ◐ partial | expected | expected | ◐ partial | ✗ no | ◐ partial | expected |
| `amd-gfx906-arm` | expected | expected | ✗ no | ◐ partial | expected | expected | ◐ partial | ✗ no | ✗ no | expected |
| `amd-gfx906-mx` | expected | ◐ partial | ✗ no | ◐ partial | ◐ partial | expected | ◐ partial | ✗ no | ✗ no | expected |
| `amd-legacy-gcn` | expected | – | ✗ no | ◐ partial | expected | expected | ◐ partial | ✗ no | ✗ no | expected |
| `amd-rocm` | expected | – | ◐ partial | expected | expected | expected | expected | ✗ no | ◐ partial | expected |
| `nvidia-modern-sm75` | expected | – | ✗ no | expected | expected | expected | expected | expected | ◐ partial | expected |
| `nvidia-legacy-sm50-70` | expected | – | ✗ no | expected | expected | expected | expected | ? | ✗ no | expected |
| `nvidia-maxwell-arm` | expected | – | ✗ no | ◐ partial | expected | expected | ◐ partial | ✗ no | ✗ no | expected |
| `apple-silicon` | expected | – | ✗ no | expected | expected | expected | expected | ✗ no | ◐ partial | expected |
| `intel-gpu` | expected | – | ✗ no | expected | expected | expected | expected | ✗ no | expected | expected |
| `shared-igpu` | expected | – | ◐ partial | ◐ partial | expected | expected | ◐ partial | ✗ no | ◐ partial | expected |

## The reasons, by hardware

### `cpu-x86`: x86 CPU only

- **llama.cpp**: verified. CI builds llama.cpp on x86-64 and runs tests/test_llamacpp_integration.py on every push.
- **mx-llama.cpp**: n/a. The fork exists for gfx906; its CPU build passed the integration tests here, which checks the patches, not the fork's purpose.
- **Radiance**: expected. Its host backend (libavx: AVX-512 and AVX2) serves every model it ships; slowly for 27B and up; not run here yet.
- **PyTorch**: verified. CI runs the PyTorch tests on CPU wheels.
- **H-neurons (GGUF)**: verified. CI (tests/test_llamacpp_integration.py).
- **Truth probe (GGUF)**: verified. CI, and Qwen2.5 0.5B and 1.5B on a CPU (docs/TRUTH_PROBE.md).
- **LoRA**: verified. CI trains a tiny model (tests/test_retrain.py, tests/test_train_guards.py).
- **QLoRA**: unsupported. Bitsandbytes' 4-bit kernels need a CUDA GPU; finetune.py says so.
- **bf16**: partial. CPUs with AVX-512 BF16 or AMX run it; others emulate it slowly. Training on a CPU uses float32.
- **Quant fidelity**: verified. CI (tiny model) and Qwen2.5-1.5B on a CPU (docs/QUANT_FIDELITY.md).

### `arm-board`: ARM boards and servers

- **llama.cpp**: expected. CI builds the arm64 and armv7 binaries but runs no tests on them.
- **Radiance**: unsupported. Its production host kernels (libavx) are x86 only; on ARM only the reference kernels remain, which it refuses to serve with.
- **PyTorch**: partial. 64-bit ARM has CPU wheels; 32-bit ARM has none.
- **H-neurons (GGUF)**: expected. Builds wherever llama.cpp does.
- **LoRA**: partial. On the CPU, small models only; none on 32-bit ARM.
- **QLoRA**: unsupported. Needs CUDA.
- **bf16**: partial. ARMv8.6 and newer cores (and some v8.2 with the BF16 extension) only.

### `amd-gfx906-arm`: AMD Instinct MI50/MI60 and Radeon VII (gfx906) on an ARM host

- **llama.cpp**: expected. Vulkan (RADV); HIP needs ROCm built from source for gfx906 on aarch64 (the mi50-arm guide).
- **mx-llama.cpp**: expected. Needs ROCm built from source on aarch64; see amd-gfx906-mx.
- **Radiance**: unsupported. No gfx906 kernel library, and no ARM host kernels.
- **PyTorch**: partial. On the ARM CPU only: there are no PyTorch ROCm wheels for aarch64.
- **H-neurons (GGUF)**: expected. Builds with the same backend as llama.cpp; not run on this card here.
- **LoRA**: partial. On the ARM CPU, small models only.
- **QLoRA**: unsupported. Bitsandbytes doesn't run on gfx906.
- **bf16**: unsupported. Gfx906 has no bfloat16 arithmetic; fp16 has a smaller range (the roadmap's fp16 range check).

### `amd-gfx906-mx`: AMD Instinct MI50/MI60 and Radeon VII (gfx906) with mx-llama.cpp

- **llama.cpp**: expected. Upstream llama.cpp with HIP for gfx906 also works; this profile builds the fork instead.
- **mx-llama.cpp**: partial. Built here and its binaries passed the integration tests on the CPU; HIP on a real MI50 not run here yet (docs/MX_LLAMA.md).
- **Radiance**: unsupported. No gfx906 kernel library; an x86 host can run its CPU backend instead.
- **PyTorch**: partial. Recent ROCm releases have dropped gfx906; use a ROCm wheel that still includes it, or the CPU.
- **H-neurons (GGUF)**: partial. Built into the fork and passed the integration tests on the CPU; not on the card here.
- **LoRA**: partial. FP16 LoRA where a gfx906 PyTorch build exists; a 27B FP16 base (about 54 GB) doesn't fit one 32 GB card (docs/ROADMAP.md, phase 3).
- **QLoRA**: unsupported. Bitsandbytes doesn't run on gfx906; finetune.py refuses QLoRA on ROCm.
- **bf16**: unsupported. Gfx906 has no bfloat16 arithmetic.

### `amd-legacy-gcn`: AMD GCN, Polaris and Vega (gfx803, gfx900, gfx906)

- **llama.cpp**: expected. Vulkan; ROCm has dropped or deprecated these chips.
- **mx-llama.cpp**: n/a. Gfx906 only; see amd-gfx906-mx.
- **Radiance**: unsupported. No kernel library for GCN; an x86 host can run its CPU backend instead.
- **PyTorch**: partial. CPU wheels.
- **LoRA**: partial. On the CPU.
- **QLoRA**: unsupported. Needs CUDA.
- **bf16**: unsupported. No bfloat16 arithmetic on GCN.

### `amd-rocm`: AMD with ROCm (RDNA 2–4, CDNA)

- **llama.cpp**: expected. HIP or Vulkan.
- **Radiance**: partial. RDNA4 (gfx1200, gfx1201), RDNA3 (gfx11xx) and Strix Halo (gfx1151) have kernel libraries; RDNA2 and CDNA don't. Not run here yet.
- **PyTorch**: expected. ROCm wheels.
- **QLoRA**: unsupported. Finetune.py refuses QLoRA on ROCm.
- **bf16**: partial. RDNA3 and CDNA2 or newer; not RDNA2.

### `nvidia-modern-sm75`: NVIDIA Turing and newer (compute 7.5+)

- **llama.cpp**: expected. CUDA; CI builds the CUDA binaries but runs no GPU tests.
- **Radiance**: unsupported. AMD GPUs only; an x86 host can run its CPU backend instead.
- **QLoRA**: expected. Bitsandbytes on CUDA.
- **bf16**: partial. Ampere (compute 8.0) and newer; Turing has none.

### `nvidia-legacy-sm50-70`: NVIDIA Maxwell, Pascal and Volta (compute 5.0–7.0)

- **llama.cpp**: expected. CUDA 12.x; CUDA 13 can't build for these cards.
- **Radiance**: unsupported. AMD GPUs only.
- **PyTorch**: expected. The CUDA 12.6 wheels, the last with kernels for compute 5.0 to 6.1.
- **LoRA**: expected. In float32.
- **QLoRA**: unknown. This profile doesn't use it; bitsandbytes' support for these cards depends on its version.
- **bf16**: unsupported. No bfloat16 arithmetic before Ampere.

### `nvidia-maxwell-arm`: NVIDIA Maxwell (Tesla M10, M40) on an ARM host

- **llama.cpp**: expected. CUDA 12.9 for arm64 servers (sbsa), the last that builds for Maxwell; the m10-arm guide.
- **Radiance**: unsupported. AMD GPUs only, and no ARM host kernels.
- **PyTorch**: partial. On the ARM CPU only: PyTorch's aarch64 CUDA wheels don't target Maxwell.
- **LoRA**: partial. On the ARM CPU, small models only.
- **QLoRA**: unsupported. No CUDA PyTorch for this combination.
- **bf16**: unsupported. No bfloat16 arithmetic on Maxwell.

### `apple-silicon`: Apple silicon (M1 and newer)

- **llama.cpp**: expected. Metal; CI builds the macOS binaries but runs no tests on them.
- **Radiance**: unsupported. AMD GPUs and x86 hosts only.
- **PyTorch**: expected. MPS.
- **LoRA**: expected. In float16 on MPS.
- **QLoRA**: unsupported. Needs CUDA.
- **bf16**: partial. M2 and newer; not M1.

### `intel-gpu`: Intel Arc and Xe

- **llama.cpp**: expected. SYCL or Vulkan.
- **Radiance**: unsupported. AMD GPUs only; an x86 host can run its CPU backend instead.
- **PyTorch**: expected. XPU wheels.
- **QLoRA**: unsupported. Needs CUDA.

### `shared-igpu`: Integrated GPUs sharing system memory (AMD APUs, Intel iGPUs)

- **llama.cpp**: expected. Vulkan, within the memory the BIOS gives the GPU.
- **Radiance**: partial. Strix Halo (gfx1151) only; on other x86 machines its CPU backend.
- **PyTorch**: partial. CPU wheels.
- **LoRA**: partial. On the CPU.
- **QLoRA**: unsupported. Needs CUDA.
- **bf16**: partial. Depends on the chip.

## Changing it

- **A cell moves to verified** only with evidence: a CI test that runs on that hardware, or a trial
  recorded with its date and results (Setup → Hardware support profiles → *It worked here* records one).
- **A new hardware profile** needs a row; `compat.py check` fails without one.
- **Every partial or unsupported cell** says why.
