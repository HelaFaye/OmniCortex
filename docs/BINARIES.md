# Prebuilt binaries

`.github/workflows/binaries.yml` builds two kinds of binary for every platform:

- **The llama.cpp tools OmniCortex drives:**
  - `llama-cett-dump`, the activation extractor;
  - `llama-server`;
  - `llama-quantize`;
  - `llama-eval-callback`.
- **The remote agent** as a single executable, `omnicortex-remote`. Copy it to another computer and run
  `omnicortex-remote pair` there ([REMOTE.md](REMOTE.md)). That computer needs no Python and no checkout.

| package | built on | llama.cpp backend | notes |
|---|---|---|---|
| `windows-x64` | Windows Server 2022 | CPU | |
| `windows-x86` | Windows Server 2022, Win32 | CPU (no AVX) | 32-bit: models up to about 2 GB |
| `windows-arm64` | Windows 11 on ARM | CPU | Snapdragon and other ARM laptops |
| `macos-arm64` | macOS 14, Apple silicon | Metal | |
| `macos-x64` | macOS 14, cross-compiled for Intel | Metal | the agent runs under Rosetta on the build machine as its start check |
| `linux-x64` | Ubuntu 24.04 | CPU | |
| `linux-arm64` | Ubuntu 24.04 on ARM | CPU | Raspberry Pi 4/5 (64-bit OS), ARM servers |
| `linux-x86` | Debian 12 i386 container | CPU (no AVX) | best effort: 32-bit |
| `linux-armv7` | Debian 12 armhf container (QEMU) | CPU | best effort: 32-bit ARM, slow to build |

**Static executables.** The tools are linked statically, so they run from any folder with nothing beside
them; the Metal shader file comes along on macOS.

**GPU builds.** The packages are CPU builds, and Metal on Macs. CUDA, ROCm and Vulkan builds depend on the
GPU and its driver, so OmniCortex builds them on the machine that has the GPU: **Setup → Build llama.cpp**,
or `scripts/build_llama_tools.sh --backend cuda|hip|vulkan`.

**32-bit targets** (`windows-x86`, `linux-x86`, `linux-armv7`) are built without AVX, FMA and F16C, and
llama.cpp supports them on a best-effort basis. If one fails to build, the rest of the run still
publishes.

## When it runs

- **A tag `v*`** builds everything. It attaches every package and a `SHA256SUMS` file to that
  release.
- **By hand**, from Actions → binaries → Run workflow. You can pin a llama.cpp tag or commit there. The
  default is llama.cpp's latest commit.
- **Pull requests** that change the build scripts, `llama-tools/` or `remote/` run it too, so a broken
  build shows up before release.

## Building one locally

```bash
bash scripts/ci/build_unix.sh linux-x64            # or macos-arm64, linux-arm64…  -> dist/omnicortex-linux-x64.tar.gz
SKIP_LLAMA=1 bash scripts/ci/build_unix.sh linux-x64   # only the agent
pwsh scripts/ci/build_windows.ps1 -Target windows-x64 -Arch x64
```

`BACKEND=metal|vulkan|cuda|hip` and `CMAKE_EXTRA="…"` pass through to `scripts/build_llama_tools.sh`.
