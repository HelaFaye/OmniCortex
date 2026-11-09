# Requirements: libraries, programs, drivers and hardware

OmniCortex touches many layers: Python packages, llama.cpp builds for
different GPU backends, kernel drivers and device permissions, and the
machine's memory. `scripts/oc_requirements.py` keeps track of all of it per
feature, and **Setup → Requirements** in Studio shows the same.

## What each feature needs

| feature | needs |
|---|---|
| Core (Studio, TestQA, Projects, review) | Python ≥ 3.10 and `requirements-core.txt` |
| Run GGUF models in Studio | a built `llama-server`; 8 GiB RAM or more |
| Per-reply checks, live scoring, neuron review | a built `llama-cett-dump` |
| Build llama.cpp | git, cmake, a C++ compiler |
| AMD GPUs through ROCm | `rocminfo`, `/dev/kfd`, the render group; a llama.cpp built with HIP |
| AMD/Intel GPUs through Vulkan (incl. APUs like the Vega 8 / gfx90c) | `vulkaninfo`, a `/dev/dri/renderD*` node, the render group; a llama.cpp built with Vulkan (`glslc` to build it) |
| NVIDIA GPUs through CUDA | `nvidia-smi` and a GPU it sees; a llama.cpp built with CUDA (`nvcc` to build it) |
| PyTorch pipeline | torch (from `install.sh`), accelerate; 16 GiB RAM or more |
| TestQA code tasks in a container | podman or docker |
| Transfer GUI and MCP extras | aiohttp; optionally mcp, keyring, cryptography |
| Desktop viewers / Godot client | fastplotlib and pygfx / Godot 4 |

GPU groups for vendors the machine doesn't have show as *n/a*. For each gap,
the check prints the fix for this OS (apt, dnf, pacman, zypper or brew). It
also shows which GPU backends the llama.cpp build has, from the ggml
libraries next to `llama-server`. A build that runs on the CPU only, on a
machine with an AMD GPU, is the commonest reason for slow models.

Version floors come from `requirements*.txt`; those files are the one place
a version is decided.

## Installing

Missing Python packages have an **Install** button (a job, `install_package`)
that runs `pip install` into Studio's Python at the version the files ask
for. Only the catalog's packages can be installed this way. torch is not
among them: its wheel depends on the GPU, so `./install.sh` picks it. System
packages and drivers need your password, so the page shows the command
instead of running it.

## Snapshots

**Take a snapshot** records every installed Python package, the programs
above with their versions, the llama.cpp builds (time and backends), the
devices and the OmniCortex commit, to `~/.omnicortex/env/<time>.json`.
**Compare** two snapshots after an update, a driver change or a rebuild:
"it was faster last week" usually turns out to be a llama.cpp rebuilt
without its GPU backend, or a package that moved a major version.
**Download exact package versions** gives a requirements file that
reproduces this setup on another machine.

## Per project

Each project in **Projects** has its own requirement list, inferred from
its checkout's build files (CMake, Meson, xmake, Cargo, go.mod,
package.json, pyproject, requirements.txt and more), editable, and checked
the same way; Start waits until required items are present. See
[DIRECTOR.md](DIRECTOR.md#requirements-of-a-project).

## Command line

    python scripts/oc_requirements.py                       # every feature
    python scripts/oc_requirements.py check --feature amd_vulkan -v
    python scripts/oc_requirements.py check --json
    python scripts/oc_requirements.py install peft
    python scripts/oc_requirements.py snapshot
    python scripts/oc_requirements.py diff                  # the last two
    python scripts/oc_requirements.py freeze > env.txt
    python scripts/oc_requirements.py project ~/src/game    # a project's own needs

MCP: the `requirements` tool (optionally one `feature`), and `start_job`
with `install_package` or `env_snapshot`. `doctor.py` remains the "can I run
the pipeline right now, and what's next" check; this one is about what each
feature needs and what changed.
