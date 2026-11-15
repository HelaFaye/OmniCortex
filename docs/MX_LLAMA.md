# mx-llama.cpp: a llama.cpp fork for the MI50

[mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) is a llama.cpp fork tuned for AMD's gfx906 (Instinct MI50
and MI60, Radeon VII):
- weights repacked for gfx906;
- tensor parallelism across several GPUs (`-sm tensor -tps N`);
- its own AllReduce between GPUs, which writes directly to the peer GPUs' memory.

OmniCortex builds and serves it like llama.cpp itself. cett-dump and the server activations patch build into it
unchanged, so the H-neuron checks and the truth probe work with it.

## Build

The hardware profile `amd-gfx906-mx` (Setup → Hardware support profiles) does this for you:

```bash
python scripts/hw_profiles.py plan --id amd-gfx906-mx     # shows the command
python scripts/hw_profiles.py apply --id amd-gfx906-mx
```

which runs:

```bash
scripts/build_llama_tools.sh --backend hip --repo https://github.com/mxxm-t/mx-llama.cpp --dir ~/mx-llama.cpp \
    --portable --cmake "-DAMDGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF"
```

- **Its own directory:** the fork builds in `~/mx-llama.cpp`, next to any `~/llama.cpp`. `build_llama_tools.sh`
  refuses to build a fork inside a clone of another repository, or the reverse.
- **ROCm:** you need ROCm 6.x with gfx906 kernels. Official ROCm 7 images no longer ship rocBLAS kernels for gfx906.
  On an ARM board, build ROCm from source as the mi50-arm guide does.
- **Pinning a version:** add `--ref TAG_OR_COMMIT`. A profile can also set `llama.repo` and `llama.ref`
  ([HARDWARE_PROFILES.md](HARDWARE_PROFILES.md)).

Point Studio at the build in `env.local.sh`:

```bash
export OC_LLAMA="$HOME/mx-llama.cpp"
export OC_CETT="$HOME/mx-llama.cpp/build/bin/llama-cett-dump"
export OC_LLAMA_SERVER="$HOME/mx-llama.cpp/build/bin/llama-server"
```

## Settings

All of these are in a model's settings in Studio:

| Setting | Value | Why |
|---|---|---|
| **Loading mode** | `dio` | Loading with mmap hangs with this fork; direct I/O doesn't |
| **Split** | `tensor` | each layer split across the GPUs (several GPUs only) |
| **Tensor parallel GPUs** (`-tps`) | `0`, or a number that divides the GPU count | `0` puts every GPU in one group; `2` with four GPUs makes two groups of two, run as two pipeline stages |
| Environment `GGML_ENABLE_CUSTOM_AR` | `0` at first | turns off the fork's peer-write AllReduce until you've checked peer-to-peer access works on this host |

- Studio refuses **Tensor parallel GPUs** without split mode `tensor`.
- A llama-server without `-tps` (llama.cpp itself) leaves that setting out, with a note, instead of failing to start.

**Never set `HSA_FORCE_FINE_GRAIN_PCIE=1`** on a host without cache-coherent PCIe, such as an RK3588 board.

## Check it before trusting it

A fork's kernels differ from llama.cpp's. Before relying on its answers:
1. Run llama.cpp's `test-backend-ops` from the fork's build, which compares every GPU operation with the CPU.
   mi50-arm's `scripts/gpu-parity.sh` wraps it and catches the case where no GPU was found.
2. Compare a few greedy answers with llama.cpp itself on the same GGUF.

## What has been checked

| Check | Result |
|---|---|
| cett-dump and the server activations patch build into the fork (`cdc81da`, CPU backend) | builds, no errors |
| `tests/test_llamacpp_integration.py` with the fork's binaries: cett-dump, `--by-weight`, the extraction pipeline and live server activations against PyTorch | 5 passed; the 2 Docker tests were skipped |
| The fork's `llama-server` has `-sm tensor`, `-tps` and `-lm dio` | yes |
| The HIP build, and anything on an MI50 | not yet: no gfx906 card was available |
