# Hardware support profiles

OmniCortex is meant to run on legacy and modern hardware alike: a 2015 Maxwell
Tesla card or a gfx906 Instinct deserves the same care as a current RTX or
RDNA 4 board, an Apple laptop, an APU sharing system memory, or a plain CPU.
A **hardware support profile** writes that care down once, for a family of
hardware, so nobody has to rediscover the right build flags and wheel index.

## What a profile says

A profile is a small JSON file (schema `omnicortex-hw-profile/1`):

- **match**: the hardware it is for: vendor, compute capability range (NVIDIA),
  `gfx` prefix (AMD), unified memory, backend, CPU architecture.
- **purposes**: what to install and how to configure it, stored separately per purpose:

| Purpose | Covers |
|---|---|
| `inference` | serving and chatting: the llama.cpp backend, its CMake flags (e.g. `--cuda-arch`), engine defaults |
| `retraining` | fine-tuning and the PyTorch side: the wheel index, training precision, method |
| `general` | one setup for everything. A profile with a single purpose must use this label |

Each purpose has a one-line `summary` and optional `notes`. The validator refuses
shell metacharacters in build flags and any PyTorch index that is not
`download.pytorch.org`, so a shared profile cannot run arbitrary commands.

## Where they come from

| Source | Location | Shown as |
|---|---|---|
| Built-in | `profiles/hardware/` (legacy and modern NVIDIA, AMD ROCm and legacy GCN, Intel, Apple, iGPU/APU, x86 CPU, ARM) | built-in |
| Yours | `~/.omnicortex/hw-profiles/` | yours |
| Community | an accepted board's `profiles/accepted/` | community |

## Using them

In Studio: **Setup → Hardware support profiles** shows which profiles fit this
computer and why, lets you pick a profile and purpose, previews the exact steps,
and runs them as a job (`/jobs`, kind `hw_apply`). From the command line:

```bash
python scripts/hw_profiles.py match                      # which profiles fit, and why
python scripts/hw_profiles.py plan --id nvidia-legacy-sm50-70 --purpose inference
python scripts/hw_profiles.py apply --id nvidia-legacy-sm50-70 --purpose inference
python scripts/hw_profiles.py check my-profile.json      # validate before sharing
```

`apply` builds llama.cpp through `scripts/build_llama_tools.sh` and installs
PyTorch from the profile's wheel index into the project's venv. Nothing runs
until you ask; `plan` shows every command first.

## Sharing and community review

This uses the same GitHub board as model reviews ([REVIEW.md](REVIEW.md), [SHARING.md](SHARING.md)):

1. **Make your own**: paste or edit JSON in the Setup section and save it, or export a built-in one and change it.
2. **Submit**: opens a board issue marked `<!-- omnicortex-hw-profile v1 -->`. The board's workflow
   validates it and files it under `profiles/pending/`; maintainers move good ones to `profiles/accepted/`.
3. **Report**: *Worked* / *Did not work* on a profile and purpose opens an issue marked
   `<!-- omnicortex-hw-report v1 -->` with the devices (names, backends, compute) and optional tokens/s.
   No paths, hostnames or prompts are included. Reports are aggregated in the board's
   `board.json` under `hw_profiles.reports` (counts and median tokens/s per profile and purpose),
   and Studio shows them beside each profile.

Submissions need no board key; anyone can propose a profile or report, and the
workflow only accepts files that pass validation.
