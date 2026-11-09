#!/usr/bin/env bash
# Build llama.cpp with OmniCortex's cett-dump extractor.
#
#   scripts/build_llama_tools.sh                     # CPU, ~/llama.cpp
#   scripts/build_llama_tools.sh --backend vulkan    # or cuda, metal, hip
#   scripts/build_llama_tools.sh --backend cuda      # archs from this machine's GPUs
#   scripts/build_llama_tools.sh --backend cuda --cuda-arch "50-real;61-real"   # for other GPUs
#   scripts/build_llama_tools.sh --dir /opt/llama.cpp --ref b6500
#   scripts/build_llama_tools.sh --portable          # no -march=native: runs on other CPUs
#   scripts/build_llama_tools.sh --server-activations  # llama-server streams /activations
#   scripts/build_llama_tools.sh --cmake "-DGGML_AVX=OFF"  # extra CMake flags (CI: 32-bit targets)
#   scripts/build_llama_tools.sh --rpc                 # also rpc-server, for layer streaming across computers
#   scripts/build_llama_tools.sh --repo URL --ref SHA  # a fork, e.g. one with a model llama.cpp has not merged yet
#
# Clones (or updates) llama.cpp, copies llama-tools/cett-dump into tools/,
# registers it, and builds llama-cett-dump, llama-server, llama-quantize and
# llama-eval-callback. --server-activations also applies
# llama-tools/server-activations (live per-token CETT over SSE).
# Prints the env exports for env.local.sh at the end.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${HOME}/llama.cpp"; BACKEND="cpu"; REF=""; REPO="https://github.com/ggml-org/llama.cpp"; NATIVE=ON; ACTS=0; RPC=0; CUDA_ARCH=""; EXTRA=""; JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    --portable) NATIVE=OFF; shift ;;
    --server-activations) ACTS=1; shift ;;
    --rpc) RPC=1; shift ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --cmake) EXTRA="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
case "$BACKEND" in
  cpu) FLAGS=() ;;
  vulkan) FLAGS=(-DGGML_VULKAN=ON) ;;
  cuda)
    # Architectures: --cuda-arch, else the GPUs in this machine (scripts/cuda_info.py),
    # else llama.cpp's own default list. Maxwell/Pascal/Volta need a CUDA 12.x
    # toolkit: CUDA 13 cannot compile for anything below sm_75.
    NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    [[ -x "$NVCC" ]] || { echo "nvcc not found: install a CUDA toolkit, or build in docker/llama-cuda.Dockerfile" >&2; exit 1; }
    CUDA_MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
    if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null; then
      CUDA_ARCH="$(python3 "$here/scripts/cuda_info.py" --cmake-archs || true)"
    fi
    if [[ -n "$CUDA_ARCH" && "${CUDA_MAJOR:-0}" -ge 13 ]]; then
      for a in ${CUDA_ARCH//;/ }; do
        n="${a%%-*}"
        if [[ "$n" =~ ^[0-9]+$ && "$n" -lt 75 ]]; then
          echo "CUDA $CUDA_MAJOR cannot build for sm_$n (CUDA 13 dropped Maxwell, Pascal and Volta)." >&2
          echo "Use a CUDA 12.x toolkit (12.9), e.g. docker/llama-cuda.Dockerfile." >&2
          exit 1
        fi
      done
    fi
    FLAGS=(-DGGML_CUDA=ON ${CUDA_ARCH:+"-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"})
    # No driver here (a container or CI build): libcuda.so.1 only exists where the
    # GPU is, so let the executables link with it unresolved, as llama.cpp's own
    # CUDA Dockerfile does. It resolves at run time on the GPU host.
    if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
      FLAGS+=(-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined)
      echo "no NVIDIA driver library on this machine: linking for a GPU host"
    fi
    echo "CUDA ${CUDA_MAJOR:-?}, architectures: ${CUDA_ARCH:-llama.cpp default}" ;;
  metal) FLAGS=(-DGGML_METAL=ON) ;;
  hip|rocm) FLAGS=(-DGGML_HIP=ON) ;;
  *) echo "backend must be cpu, vulkan, cuda, metal or hip" >&2; exit 1 ;;
esac
TARGETS=(llama-cett-dump llama-server llama-quantize llama-eval-callback)
if [[ "$RPC" == 1 ]]; then FLAGS+=(-DGGML_RPC=ON); TARGETS+=(rpc-server); fi
if [[ ! -d "$DIR/.git" ]]; then
  # Shallow: the history is ~1 GB and nothing here needs it (--ref fetches a pinned commit).
  git clone --depth 1 "$REPO" "$DIR"
fi
if [[ -n "$REF" ]]; then git -C "$DIR" fetch --depth 1 "$REPO" "$REF" && git -C "$DIR" checkout FETCH_HEAD; fi
rm -rf "$DIR/tools/cett-dump"
cp -r "$here/llama-tools/cett-dump" "$DIR/tools/"
grep -q 'add_subdirectory(cett-dump)' "$DIR/tools/CMakeLists.txt" || echo 'add_subdirectory(cett-dump)' >> "$DIR/tools/CMakeLists.txt"
if [[ "$ACTS" == 1 ]]; then python3 "$here/llama-tools/server-activations/apply_patch.py" "$DIR"; fi
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE="$NATIVE" "${FLAGS[@]}" $EXTRA
cmake --build "$DIR/build" --config Release -j "$JOBS" \
  --target "${TARGETS[@]}"
cat <<MSG

Built in $DIR/build/bin. Add to env.local.sh:
  export OC_LLAMA="$DIR"
  export OC_CETT="$DIR/build/bin/llama-cett-dump"
  export OC_LLAMA_SERVER="$DIR/build/bin/llama-server"
MSG
