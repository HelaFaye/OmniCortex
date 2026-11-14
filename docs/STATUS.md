# Progress and plan

Where OmniCortex stands, what has and hasn't been checked, and what comes next. Updated 2026-11-14.

## Where things stand

Everything below is on the branch behind pull request #3 (`claude/determined-ritchie-xlemxv`), which is being
prepared for merging into `main`. Each area has its own document.

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
| CI (`.github/workflows/tests.yml`, `binaries.yml`) | Tests with cached models and a cached llama.cpp build; binaries for 10 targets | Green except 32-bit Windows (best-effort, llama.cpp's own source) |
| Kivy client ([MOBILE_CLIENTS.md](MOBILE_CLIENTS.md)) | Python/Kivy app for Studio | The API layer only (`clients/kivy`): pairing with certificate pinning, models, streaming chat, reply checks, review lookup. CI against a fake Studio over TLS. No interface yet |

Companion repositories, both merged:
- **mi50-arm #1:** a VBIOS check in `check-mi50.sh`, and which MI50 VBIOSes can shrink BAR0 on RK3588.
- **m10-arm #1:** a Proxmox guide for sharing the Tesla M10 with an LXC container.

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

## Next stage, after merging

In this order unless you reorder it.

1. **Kivy client interface:**
   - pair by link or QR code;
   - chat with model choice and streaming;
   - the reply checks (risk strip, truth probe);
   - the review lookup.

   Desktop first, then Android through Buildozer.
2. **Quantization fidelity (from Project Maya's quality reports):** compare a quantized model with its original:
   - top-1 next-token agreement and KL divergence (llama.cpp's `llama-perplexity --kl-divergence`);
   - TestQA's confident mistakes;
   - H-neuron activity;
   - whether the truth probe still works.

   Together these say whether a quant kept what the model knows.
3. **A size budget per layer (from IST-DASLab's RCO):** pick each layer's quantization type to hit an exact size,
   protecting the layers OmniCortex measures as important. The output is llama.cpp `--tensor-type` overrides.
   Item 2 measures the result.
4. **fp16 headroom check (from mi50-vllm-lab):**
   - **The risk:** on cards without bf16 (MI50, Maxwell), a bf16-trained model's hidden state can pass fp16's
     maximum and turn into inf/NaN.
   - **The check:** reports each layer's peak against that limit.
   - **The fix:** scaling by a power of two at load time, which leaves the output unchanged.
   - Tied into the gfx906 and Maxwell profiles.
5. **A/B tests of settings with an output check (from the Strata MI50 fork):** a fixed greedy prompt set, answer
   hashes and timing, so a setting is kept only if it's faster and changes no answer, or if the changed answers are
   shown.
6. **A benchmark shaped like coding agents (from Maya's agent benchmark and Strata's protocol):** concurrent
   conversations, shared system prompts, cold restarts and frozen prompts with a checksum, so community speed reports
   compare like with like.
7. **gfx906 profile updates:**
   - optional MI50-tuned llama.cpp forks (Castagna Veloce, mx-llama.cpp), built with `--repo`, with a check that
     cett-dump still builds into them;
   - a note that official ROCm 7 images no longer include gfx906 rocBLAS kernels.
8. **Llaminar ARM64 backports to mi50-arm and m10-arm.** Those are separate repositories; see their own docs.

## Decisions on record

- **32-bit Windows binaries are best-effort**, like 32-bit Linux. Current llama.cpp uses a 64-bit-only MSVC
  intrinsic; the job no longer fails the run.
- **The Claude GitHub App isn't installed** on the repositories, so CI and review events don't reach the assistant.
  It checks pull requests itself.
- **No refusal-removal weight edits** (like Project Maya's "Derisked" quant) are planned.
