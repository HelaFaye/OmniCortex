#!/usr/bin/env bash
# Installer for OmniCortex: creates ./venv with a PyTorch build that matches
# this machine, installs requirements, and smoke-tests the GPU.
#
#   NVIDIA (nvidia-smi present)  -> a CUDA wheel for the GPU's architecture:
#                                   Turing+ the default PyPI wheel; Maxwell,
#                                   Pascal, Volta (e.g. Tesla M10) the CUDA 12.6
#                                   wheels, torch<2.15 (scripts/cuda_info.py)
#   macOS                        -> default PyPI wheel (Metal/MPS)
#   AMD (/dev/kfd present)       -> ROCm wheel index chosen from the gfx target;
#                                   the wheels bundle ROCm userspace, so distros
#                                   AMD does not officially support usually work
#   otherwise                    -> CPU wheel
#
#   ./install.sh              # detect, install into ./venv, verify
#   ./install.sh --rocm 6.4   # force a ROCm wheel index
#   ./install.sh --cuda       # force a CUDA wheel (chosen by scripts/cuda_info.py)
#   ./install.sh --cpu        # no GPU, CPU-only torch
#   ./install.sh --system-torch  # venv that reuses the distro's torch
#   ./install.sh --no-torch   # skip PyTorch: Studio, TestQA, Projects, transfer and
#                             # the llama.cpp (GGUF) path do not need it
#   ./install.sh --docker     # just print the ROCm Docker recipe and exit

set -euo pipefail

ROCM_VERSION=""
MODE="auto"
VENV="${VENV:-venv}"

FORCE_ROCM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm) ROCM_VERSION="$2"; FORCE_ROCM=1; shift 2 ;;
    --cpu) MODE="cpu"; shift ;;
    --cuda) MODE="cuda"; shift ;;
    --system-torch) MODE="system"; shift ;;
    --docker) MODE="docker"; shift ;;
    --no-torch) MODE="none"; shift ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx \033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- environment

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  info "distro: ${PRETTY_NAME:-$ID}"
  case "${ID_LIKE:-$ID}" in
    *arch*|arch) warn "Arch is not an AMD-supported ROCm distro. The pip wheels" ;
                 warn "usually work anyway; ./install.sh --docker if they don't." ;;
  esac
fi

command -v python3 >/dev/null || die "python3 not found"
PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
info "python: $PYVER"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "python 3.10+ required"

# ------------------------------------------------------------ docker shortcut

if [[ "$MODE" == "docker" ]]; then
  cat <<'DOCKER'

Docker fallback. Kernel driver comes from your host, all of ROCm userspace
comes from the container, so Arch's lack of official support stops mattering.

  docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size 16G \
    -v "$PWD":/work -w /work \
    rocm/pytorch:latest bash

  # then, inside:
  pip install -r requirements.txt
  python scripts/preflight.py --model_path <model>

DOCKER
  exit 0
fi

# ------------------------------------------------------------ GPU detection

GFX=""
if [[ "$MODE" == "auto" ]] && (( ! FORCE_ROCM )); then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    info "macOS: installing the default wheel (MPS backend)"
    MODE="cuda"   # same install path: the default PyPI wheel
  elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    info "NVIDIA GPU(s): $(nvidia-smi -L | wc -l), $(nvidia-smi -L | head -1)"
    MODE="cuda"
  fi
fi
if [[ "$MODE" == "auto" ]]; then
  if [[ ! -e /dev/kfd ]]; then
    info "no NVIDIA or AMD compute device found (/dev/kfd missing); using CPU torch."
    info "llama.cpp's Vulkan backend can still use the GPU for serving and extraction."
    MODE="cpu"
  else
    for grp in video render; do
      id -nG | tr ' ' '\n' | grep -qx "$grp" \
        || warn "you are not in the '$grp' group: sudo usermod -aG $grp \$USER (re-login)"
    done
    if command -v rocminfo >/dev/null; then
      # Read the real target: HSA_OVERRIDE_GFX_VERSION changes what rocminfo
      # reports, so leaving it set detects the masquerade instead. The {3,}
      # also stops a truncated match like a bare "gfx9".
      GFX=$(env -u HSA_OVERRIDE_GFX_VERSION rocminfo 2>/dev/null \
            | grep -oP 'gfx[0-9a-f]{3,}' | sort -u | grep -v gfx000 | head -1 || true)
    fi
    if [[ -n "$GFX" ]]; then
      info "detected: $GFX"
    else
      warn "could not detect gfx target (rocminfo not installed?)"
      if command -v lspci >/dev/null && lspci 2>/dev/null | grep -qiE "vega|cezanne|barcelo|renoir|picasso|raven"; then
        warn "lspci suggests a Vega-class integrated GPU, which ROCm does not"
        warn "support. Installing CPU torch; use the Vulkan path for extraction."
        MODE="cpu"
      fi
    fi
  fi
fi

# Map architecture to guidance. Only the wheel index is decided here; the
# override advice is printed for the user to apply if the smoke test fails.
OVERRIDE=""
case "$GFX" in
  gfx1200|gfx1201)
    info "RDNA4. Needs ROCm 7.x; older wheels will not have kernels for this."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="7.0" ;;
  gfx1100|gfx1101|gfx1102)
    info "RDNA3, officially supported."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
    info "RDNA2. gfx1030 is supported; the smaller dies need an override."
    [[ "$GFX" != "gfx1030" ]] && OVERRIDE="10.3.0"
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  gfx1010|gfx1011|gfx1012)
    warn "RDNA1 is not supported by ROCm. Community builds exist but expect"
    warn "to compile PyTorch yourself. CPU extraction may be the faster path."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  gfx90c|gfx902|gfx909|gfx90b)
    if (( FORCE_ROCM )); then
      # The gfx900 masquerade does work on some Vega APUs -- verify with
      # scripts/try_rocm.sh, which checks results against CPU rather than
      # just checking that a matmul returned.
      info "Vega APU ($GFX) with --rocm given: installing ROCm torch."
      info "Export HSA_OVERRIDE_GFX_VERSION=9.0.0 before using it."
    else
      warn "Vega-based APU ($GFX). ROCm has no support for this target."
      warn "Installing CPU torch, which is correct for the main pipeline:"
      warn "cett-dump runs on Vulkan and a 9B in bf16 would not fit anyway."
      warn ""
      warn "The gfx900 masquerade does work on some of these. To try it:"
      warn "  ./scripts/try_rocm.sh          then  ./install.sh --rocm 6.4"
      MODE="cpu"
    fi ;;
  gfx90a|gfx942|gfx908) info "CDNA, fully supported."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  "") [[ -z "$ROCM_VERSION" && "$MODE" != "cpu" ]] && ROCM_VERSION="6.4" ;;
  *) warn "unrecognised target $GFX; defaulting the wheel index"
     [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
esac

# ---------------------------------------------------------------- launcher

install_launcher() {
  # An app-menu entry, so nobody needs a terminal after this script.
  local here; here="$(pwd)"
  if [[ "$(uname -s)" == "Linux" ]]; then
    local apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$apps"
    cat > "$apps/omnicortex.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=OmniCortex
Comment=Run, test and look inside local models
Exec=$here/omnicortex
Terminal=false
Categories=Development;Science;
DESKTOP
    info "added OmniCortex to the app menu ($apps/omnicortex.desktop)"
  fi
}

# ---------------------------------------------------------------- install

if [[ "$MODE" == "none" ]]; then
  info "creating venv at $VENV (no PyTorch)"
  python3 -m venv "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip wheel
  info "installing requirements (without the PyTorch-only ones)"
  pip install --quiet -r requirements-core.txt
  cat <<EOF

Done, without PyTorch.

Start it from the app menu (OmniCortex), or:  ./omnicortex
It opens Studio in your browser; the Setup page there takes care of the rest.
Command line: source $VENV/bin/activate

Works now: Studio (viz/studio.py), TestQA, Projects, transfer, the subject
classifier, and everything that runs through llama.cpp (cett-dump extraction,
GGUF editing). Needs PyTorch later: the PyTorch extraction path, CLIP/SigLIP
H-Neurons, fine-tuning. Re-run ./install.sh without --no-torch to add it.

Next:  python scripts/doctor.py      # what is ready, and the one thing to do next
EOF
  install_launcher
  exit 0
fi

if [[ "$MODE" == "system" ]]; then
  # Use the distro's torch instead of a wheel. On Arch, python-pytorch-rocm is
  # built against the same ROCm as /opt/rocm, so its rocBLAS and its Tensile
  # kernels match -- no ABI mixing across major versions.
  if ! python3 -c 'import torch' 2>/dev/null; then
    die "no system torch found. Install it first:
   sudo pacman -S python-pytorch-rocm"
  fi
  info "creating venv at $VENV with access to system packages"
  python3 -m venv --system-site-packages "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip wheel
  info "using system torch: $(python -c 'import torch;print(torch.__version__)')"
  info "installing the rest"
  pip install --quiet -r requirements.txt
  info "verifying"
  # fall through to the smoke test below
  SKIP_TORCH_INSTALL=1
fi

if [[ -z "${SKIP_TORCH_INSTALL:-}" ]]; then
if [[ -d "$VENV" ]]; then
  info "reusing the venv at $VENV"
else
  info "creating venv at $VENV"
fi
python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel
# The packages torch does not bring first: in a reused venv this repairs any
# half-installed state, so pip's resolver has nothing to complain about while
# torch and the rest go in.
info "installing the core packages"
pip install --quiet -r requirements-core.txt

TORCH_SPEC="torch"
if [[ "$MODE" == "cuda" ]]; then
  INDEX=""
  if [[ "$(uname -s)" != "Darwin" ]] && command -v nvidia-smi >/dev/null; then
    python3 scripts/cuda_info.py | sed 's/^/    /'
    read -r -a TORCH_ARGS <<< "$(python3 scripts/cuda_info.py --torch-pip)"
    if [[ ${#TORCH_ARGS[@]} -gt 0 ]]; then
      TORCH_SPEC="${TORCH_ARGS[0]}"
      [[ "${TORCH_ARGS[1]:-}" == "--index-url" ]] && INDEX="${TORCH_ARGS[2]}"
    fi
  fi
  info "installing $TORCH_SPEC from ${INDEX:-PyPI}"
elif [[ "$MODE" == "cpu" ]]; then
  INDEX="https://download.pytorch.org/whl/cpu"
  info "installing CPU torch"
else
  INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
  info "installing torch from $INDEX"
fi

if ! pip install "$TORCH_SPEC" ${INDEX:+--index-url "$INDEX"}; then
  warn "installing $TORCH_SPEC from ${INDEX:-PyPI} failed (see pip's message above)."
  case "$MODE" in
    cpu)  warn "If download.pytorch.org is unreachable from here, PyPI's torch also runs on CPU"
          warn "(a larger download):  source $VENV/bin/activate && pip install torch && pip install -r requirements.txt" ;;
    cuda) warn "python scripts/cuda_info.py shows which wheel this GPU needs; install it by hand with"
          warn "  pip install <spec> --index-url <index>  inside $VENV, then pip install -r requirements.txt" ;;
    *)    warn "Available ROCm indexes are listed at https://pytorch.org/get-started/locally/"
          warn "Retry with ./install.sh --rocm <version>, or use ./install.sh --docker" ;;
  esac
  warn "Or skip PyTorch for now: ./install.sh --no-torch (Studio, TestQA, Projects and the"
  warn "llama.cpp path do not need it)."
  exit 1
fi

info "installing the rest"
pip install --quiet -r requirements.txt
fi

# ---------------------------------------------------------------- smoke test

info "verifying"
cat > /tmp/oc_smoke.py <<'PY'
import torch
print("torch     :", torch.__version__)
print("hip       :", getattr(torch.version, "hip", None))
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("device    : Apple MPS")
if torch.cuda.is_available():
    # Every GPU, not just the first: a Tesla M10 is four. Check that the wheel has
    # kernels for each card (a wheel without them fails with "no kernel image"),
    # and that results match the CPU, not merely that a kernel ran.
    archs = torch.cuda.get_arch_list()
    total_all = 0
    a = torch.randn(512, 512, dtype=torch.float32)
    ref = a @ a
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = f"sm_{p.major}{p.minor}"
        free, total = torch.cuda.mem_get_info(i)
        total_all += total
        print(f"device {i}  : {p.name}  {cc}  {total/2**30:.1f} GiB ({free/2**30:.1f} free)")
        if not getattr(torch.version, "hip", None) and cc not in archs and \
                not any(x.startswith("compute_") for x in archs):
            raise SystemExit(f"this torch wheel has no kernels for {cc} (has {', '.join(archs)}); "
                             "re-run ./install.sh, which picks the wheel with scripts/cuda_info.py")
        got = (a.to(f"cuda:{i}") @ a.to(f"cuda:{i}")).cpu()
        err = float((got - ref).abs().max() / ref.abs().max())
        if err > 1e-3:
            raise SystemExit(f"device {i}: fp32 matmul differs from CPU by {err:.2e}")
        dt = "bf16" if p.major >= 8 else "fp16" if (p.major >= 7 or (p.major, p.minor) == (6, 0)) else "fp32"
        if dt != "fp32":
            x = torch.randn(1024, 1024, device=f"cuda:{i}", dtype=getattr(torch, "bfloat16" if dt == "bf16" else "float16"))
            _ = x @ x
        print(f"            fp32 matches CPU (rel err {err:.1e}); training precision {dt}")
    print("\nsuggested --gpu_mem: %dGiB" % max(1, int(torch.cuda.mem_get_info(0)[1]/2**30) - 2))
else:
    print("no GPU visible to torch; extraction will run on CPU")
PY

if python /tmp/oc_smoke.py; then
  :
else
  warn "smoke test failed."
  if [[ -n "$OVERRIDE" ]]; then
    warn "Your $GFX needs an architecture override. Try:"
    warn "  export HSA_OVERRIDE_GFX_VERSION=$OVERRIDE"
    warn "and re-run. Add it to ~/.bashrc if it works."
  else
    warn "Try ./install.sh --docker, or ./install.sh --cpu to proceed without GPU."
  fi
  exit 1
fi
rm -f /tmp/oc_smoke.py

cat <<EOF

Done.

Start it from the app menu (OmniCortex), or:  ./omnicortex
It opens Studio in your browser; the Setup page there takes care of the rest.
Command line: source $VENV/bin/activate

Next:  python scripts/doctor.py      # what is ready, and the one thing to do next

Then docs/GETTING_STARTED.md. Before a long extraction run,
  python scripts/preflight.py --model_path <hf-model> --n_pairs 400
estimates time and memory without loading the weights.
EOF
install_launcher
