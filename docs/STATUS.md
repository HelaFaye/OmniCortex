# Progress and plan

Where OmniCortex stands, what has and hasn't been checked, and what comes next. Updated 2026-11-17.

## Where things stand

Everything below is on `main`, except quantization fidelity, which is in the pull request after #6. It came in
through these pull requests:
- **#3** (merged 2026-11-13): the bulk of it, with the hardening of execution permissions, network services,
  remote transfers, tool rules and the tuning worker;
- **#4**: the audit fixes and the Kivy client's API layer;
- **#5**: mx-llama.cpp;
- **#6** (merged 2026-11-16): the Kivy client app, desktop and Android.

Each area has its own document.

The **Checked** column says how far each piece has been verified:
- **CI**: the GitHub Actions test or build runs it on every push;
- **local**: run in a CPU-only container;
- **simulated**: tested against fake hosts or files;
- **not on hardware**: untested on the real card or model it targets.

| Area | What it is | Checked |
|---|---|---|
| The name ([README](../README.md)) | Everything says OmniCortex; old folders move once on first start | CI |
| Hardware profiles ([HARDWARE_PROFILES.md](HARDWARE_PROFILES.md)) | What to install per hardware family and purpose, legacy and current, including MI50 and Maxwell on ARM | CI; not on hardware |
| Unified review and browser extension ([UNIFIED_REVIEW.md](UNIFIED_REVIEW.md), [BROWSER_EXTENSION.md](BROWSER_EXTENSION.md)) | One record per model; stats on model pages | CI; Chromium by hand; Firefox untested |
| Integrated options ([INTEGRATED_OPTIONS.md](INTEGRATED_OPTIONS.md)) | 17 options under Settings → Integrated options | CI |
| Default add-ons and the pose add-on ([ADDONS.md](ADDONS.md), [POSE.md](POSE.md)) | Five default add-ons; video or OpenPose to 3D poses with Godot and Blender clients | CI; the engine clients by hand only |
| Looped models: NVIDIA Dory ([DORY.md](DORY.md)) | Loop count settings, loop sweep, per-loop lens, NeMo Gym environment | CI; not on the real model |
| cett-dump `--by-weight` ([DORY.md](DORY.md#activations-from-the-gguf-cett-dump-and-the-loops)) | Activations for models whose graph doesn't name the MLP output; separates a looped model's passes | CI, identical to name matching; not on Dory |
| Truth probes ([TRUTH_PROBE.md](TRUTH_PROBE.md)) | "Knows but says otherwise", Hugging Face or GGUF, Studio `/api/truth` | CI; local on Qwen2.5 0.5B and 1.5B (results in the doc) |
| Faster activation capture ([PIPELINE.md](PIPELINE.md)) | `device`, `stream` and `auto` capture modes | CI on the CPU; GPU modes not on hardware |
| Proxmox backend ([PROXMOX.md](PROXMOX.md)) | OmniCortex in an LXC container with the host's GPUs; MI50 VBIOS check | Simulated hosts |
| mx-llama.cpp ([MX_LLAMA.md](MX_LLAMA.md)) | The MI50-tuned llama.cpp fork as the `amd-gfx906-mx` profile; split mode `tensor`, `-tps`, `-lm dio` in Studio | CI; cett-dump and the server activations patch built into the fork and its binaries passed the integration tests (CPU); HIP not on hardware |
| Quantization fidelity ([QUANT_FIDELITY.md](QUANT_FIDELITY.md)) | Did a quant keep what the model knows: KL and top token, TestQA answers lost, H-neuron drift, truth probe transfer; a verdict per quant | CI (tiny model's Q8_0 and Q4_0); local on Qwen2.5-1.5B (F16 against Q8_0, Q4_K_M, Q2_K; results in the doc) |
| CI (`.github/workflows/tests.yml`, `binaries.yml`, `android.yml`) | Tests with cached models and a cached llama.cpp build; binaries for 10 targets; the Android APK, and the client's self-test in an Android 13 emulator | Green except 32-bit Windows (best-effort, llama.cpp's own source) |
| Kivy client ([MOBILE_CLIENTS.md](MOBILE_CLIENTS.md)) | Python/Kivy app for Studio | `clients/kivy`: pairing by link or QR code with certificate pinning, model choice, streamed chat, the risk strip, the truth probe, the emotion jewel, Fix This Mistake, the review lookup. CI against a fake Studio over TLS; the real app driven on a virtual display against a fake Studio; `/api/emotions` on a tiny real model. CI builds the Android APK. Not yet against a real Studio or on a phone |

Companion repositories:
- **mi50-arm #1** (merged): a VBIOS check in `check-mi50.sh`, and which MI50 VBIOSes can shrink BAR0 on RK3588.
- **m10-arm #1** (merged): a Proxmox guide for sharing the Tesla M10 with an LXC container.
- **mi50-arm #2 and m10-arm #2** (open, ready to merge), after Llaminar's ARM work:
  - `arm-cpu.sh`: what the CPU can execute, and whether a build uses instructions it lacks;
  - `gpu-parity.sh`: every GPU operation checked against the CPU;
  - `cpu-threads-bench.sh`: fast cores against every core, and a longer thread spin, measured on a model;
  - mx-llama.cpp builds (mi50-arm).

## Audit, 2026-11-14

An audit of `main` (bfad43d) plus the commits after it found no problem that blocks the next stage. What was
checked:

| Check | Result |
|---|---|
| Full test suite, CPU-only container with PyTorch and a llama.cpp build with cett-dump | 797 passed, 0 failed. The 19 skips need a GPU, Docker, Godot, `clip_benchmark`, IPv6 loopback, or a `llama-quantize` build |
| CI on `main` | tests green; binaries green except best-effort 32-bit Windows |
| Every Python file compiles; pyflakes undefined names | Clean; the 2 hits are false positives (a string annotation, Blender's property syntax) |
| Every Studio job's fields against its script's flags | All 102 Python jobs match (`tests/test_job_specs.py` checks this on every run) |
| Markdown links; paths named in the docs | No broken links. Missing paths appear only in historical CHANGELOG entries, or belong to a review board's repository |
| Secrets, large files | None. The only key-like strings are the sanitizer's own detection patterns |
| Unsafe loading, shell, archives | **Fixed:** five `torch.load(..., weights_only=False)` calls could run code from a `.pt` file from elsewhere (a downloaded feature dictionary, shared steering or emotion vectors). They now go through `scripts/safe_load.py` (weights only; `OC_TRUST_PICKLE=1` for a file you made). The `shell=True` evaluator commands come from the operator's command line, never the API. The one zip extraction relies on Python's own path sanitising |
| Old names | One user-visible log line in the llama-server activations patch now says OmniCortex. Its marker and C++ names keep the old spelling on purpose: renaming them would make the patch apply twice to an existing checkout |

Minor, not fixed: two of the three tuning scripts substitute file paths into the evaluator command unquoted (the
third quotes `{profile}`), so a path containing a space breaks that command. Passing paths through the
`OC_MODEL`/`OC_PROFILE` environment variables, which all three already set, avoids it.

## Still to check on real hardware

None of these can be done in a cloud container. Each turns an "untested" note into a result:

1. **Dory:**
   - build the fork (`docs/DORY.md`);
   - run `extract_activations_gguf.py` on the real GGUF, and check that pass 2's recurrent slots are seen and differ from pass 1's;
   - run the loop sweep.
2. **MI50 (RK3588 or x86):**
   - `check-mi50.sh` and `scripts/amd_vbios.py`;
   - `scripts/proxmox_lxc.py host` and `plan` on the Proxmox host;
   - the `amd-gfx906-arm` profile.
3. **Tesla M10:**
   - m10-arm's `check-pcie.sh` and CUDA self-test;
   - the `nvidia-maxwell-arm` profile;
   - the Proxmox container.
4. **Any CUDA or ROCm GPU:**
   - the `stream` and `device` capture modes (`tests/test_capture_modes.py` runs its GPU test when a GPU is present);
   - a truth probe on a 7B-or-larger model.
5. **Windows ARM64 and macOS builds:** CI builds them; nobody has run them yet.
6. **The Kivy client on Android:** install the APK CI builds, scan Studio's QR code, chat, and try each reply
   check against a real model.

## Next stage, after merging

In this order unless you reorder it.

1. **Kivy client interface:**
   - **done:** pairing by link or QR code, chat with model choice and streaming, the reply checks (risk strip,
     truth probe), the emotion jewel, Fix This Mistake, the review lookup (`clients/kivy/README.md`);
   - **built, not yet run:** the Android APK (Buildozer, in CI). Next: install it on a phone and pair with a real
     Studio.
2. **Quantization fidelity (from Project Maya's quality reports):** *done:* `scripts/quant_fidelity.py`,
   [QUANT_FIDELITY.md](QUANT_FIDELITY.md). Compare a quantized model with its original:
   - top-1 next-token agreement and KL divergence (llama.cpp's `llama-perplexity --kl-divergence`);
   - TestQA's confident mistakes;
   - H-neuron activity;
   - whether the truth probe still works.

   Together these say whether a quant kept what the model knows. Also: does greedy output match across backends
   (CPU, Vulkan, ROCm, CUDA)? Llaminar gets identical text from Qwen3-0.6B Q8_0 on ARM NEON and x86, so a
   mismatch is worth a look rather than being assumed normal.
3. **A size budget per layer (from IST-DASLab's RCO):** pick each layer's quantization type to hit an exact size,
   protecting the layers OmniCortex measures as important. The output is llama.cpp `--tensor-type` overrides.
   Item 2 measures the result.
4. **fp16 range check, for every card without bf16 (from mi50-vllm-lab and Llaminar):**
   - **Which cards:** every hardware profile whose card has no bfloat16 arithmetic:
     - AMD gfx906 (MI50, MI60, Radeon VII) and older GCN;
     - NVIDIA before Ampere (Maxwell, including the Tesla M10; Pascal; Volta; Turing);
     - Apple M1.

     Each of those profiles gets a field saying so, and the check runs by default for them.
   - **The risk:** a model trained in bf16 can hold values outside fp16's range:
     - **Too large:** above 65504 a value becomes inf, and then NaN.
     - **Too small:** below about 6.1e-5 a value loses precision, and below about 6e-8 it becomes zero.

     A model can also be fine in one tensor and broken in the next.
   - **The check:** for each tensor and each layer's activations, on a short calibration text, it reports:
     - the peak against fp16's maximum;
     - the share of values that would lose precision or become zero.
   - **One rounding rule:** wherever OmniCortex narrows a value to fp16, it uses one rule (round to nearest, ties
     to even, keeping the small values fp16 can still hold), the same on CPU and GPU. Llaminar converts fp16 in
     software for this reason: the hardware instruction rounds the smallest values by a different rule from the
     one its other devices use, which would break their agreement.
   - **The fix:**
     - **Scaling** by a power of two at load time, which leaves the output unchanged. It has to be chosen against
       both limits, because scaling down to avoid overflow pushes small values toward zero.
     - **Per-tensor choice** when one scale can't fit a tensor: keep that tensor in bf16 or fp32 and make the rest
       fp16, as llama.cpp `--tensor-type` overrides (the same output as item 3).
   - **The proof:** item 2's KL divergence and top-1 agreement against the original model decide each choice. Llaminar
     chose fp16 over bf16 for one model's sensitive values the same way, by measuring how much the output changed.
5. **A/B tests of settings with an output check (from the Strata MI50 fork and Llaminar's ARM performance
   review):** a fixed greedy prompt set, answer hashes and timing. A setting is kept only if it's faster and changes
   no answer, or if the changed answers are shown. Llaminar's method, adopted as the rules:
   - **Alternate:** run A B B A, at least two fresh processes per side, and average, so warm-up and heat don't
     favour whichever ran first.
   - **Check the setting took effect** in each run (the run records it), rather than trusting the command line.
   - **Judge on the whole model only.** Llaminar's per-shape kernel tables won 20–25% in isolated benchmarks but
     lost 4–7% inside the model. Removing them gave +4% generation and +9–11% prompt processing.
   - **Report noise:** the spread across repeats goes beside every difference, and a difference smaller than the
     spread isn't reported as a win.
7. **gfx906 profile updates:**
   - **done:** mx-llama.cpp, as the `amd-gfx906-mx` profile ([MX_LLAMA.md](MX_LLAMA.md)), with the ROCm 7 note.
     cett-dump builds into it, and its binaries pass the llama.cpp integration tests;
   - Castagna Veloce the same way, if wanted: a profile with its `llama.repo`.
8. **Llaminar ARM64 backports to mi50-arm and m10-arm.** Those are separate repositories; see their own docs. Open
   there: `arm-cpu.sh` (what the CPU can run), `gpu-parity.sh` (GPU against CPU) and `cpu-threads-bench.sh`
   (fast cores against every core, and a longer thread spin, measured on a model).

## Decisions on record

- **32-bit Windows binaries are best-effort**, like 32-bit Linux. Current llama.cpp uses a 64-bit-only MSVC
  intrinsic; the job no longer fails the run.
- **The Claude GitHub App isn't installed** on the repositories, so CI and review events don't reach the assistant.
  It checks pull requests itself.
- **The fp16 range check applies to every card without bf16**, not only the MI50 (item 4).
- **Not taken from Llaminar:**
  - its NEON and gfx906 kernels: llama.cpp already uses the same nibble lookups (`vqtbl1q_s8` on ARM, `v_perm_b32`
    on AMD), for MXFP4 too;
  - running Llaminar as an engine: on GB10, llama.cpp's CUDA backend is ahead at prompt processing (1063 against
    319 tokens/s) and generation (28.8 against 22.8).
- **No refusal-removal weight edits** (like Project Maya's "Derisked" quant) are planned.
