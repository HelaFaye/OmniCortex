#!/usr/bin/env bash
# Test whether PyTorch will run on a Vega-class APU via the gfx900 override.
#
# gfx90c (Renoir/Cezanne/Barcelo) is not a ROCm-supported target. The community
# workaround is HSA_OVERRIDE_GFX_VERSION=9.0.0, which makes the runtime treat it
# as gfx900 (Vega 64). Same GCN generation, so the kernels are often compatible.
# Often, not always: it can also hang the GPU or produce silently wrong results.
#
# Read this before running it: nothing in OmniCortex currently needs it. The
# bf16 PyTorch extraction path was replaced by cett-dump on Vulkan, which is
# faster on this hardware and fits in 12GB where bf16 never would. This is here
# because "unreliable" was not a good enough answer, not because you need it.
#
#   ./scripts/try_rocm.sh            # check only
#   ./scripts/try_rocm.sh --install  # install ROCm torch into the venv first

set -uo pipefail
INSTALL=0
[[ "${1:-}" == "--install" ]] && INSTALL=1

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*"; }

# HSA_OVERRIDE_GFX_VERSION changes what rocminfo reports, so reading it with
# the override active tells you the masquerade, not the hardware. Unset it for
# detection or this script congratulates you on owning a Vega 64.
PRESET="${HSA_OVERRIDE_GFX_VERSION:-}"
GFX=$(env -u HSA_OVERRIDE_GFX_VERSION rocminfo 2>/dev/null \
      | grep -oP 'gfx[0-9a-f]+' | grep -v gfx000 | head -1)
if [[ -n "$PRESET" ]]; then
  MASQ=$(rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | grep -v gfx000 | head -1)
  say "HSA_OVERRIDE_GFX_VERSION=$PRESET is already set in your environment"
  say "  real hardware: ${GFX:-unknown}   reported as: ${MASQ:-unknown}"
else
  say "gfx target: ${GFX:-not detected (rocminfo missing?)}"
fi
case "$GFX" in
  gfx90c|gfx902|gfx909)
    say "Vega APU. The override is the only path; odds are moderate." ;;
  gfx900|gfx906|gfx908|gfx90a|gfx942|gfx1030|gfx110*|gfx120*)
    warn "$GFX is supported natively; you do not need this script." ;;
  "") warn "install rocminfo (pacman -S rocminfo) or this cannot tell you much" ;;
  *)  warn "unrecognised target $GFX; proceeding anyway" ;;
esac

if [[ ! -e /dev/kfd ]]; then
  warn "/dev/kfd is missing: the amdgpu kernel driver is not exposing compute."
  warn "Without it no override will help. Check 'lsmod | grep amdgpu'."
  exit 1
fi
for g in video render; do
  id -nG | tr ' ' '\n' | grep -qx "$g" || warn "not in group '$g': sudo usermod -aG $g \$USER"
done

if (( INSTALL )); then
  # Arch blocks pip against system python under PEP 668, and it is right to:
  # a --break-system-packages torch would shadow pacman's python-pytorch-rocm.
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    warn "no virtualenv active. Arch blocks pip outside one (PEP 668)."
    warn ""
    warn "  source venv/bin/activate && ./scripts/try_rocm.sh --install"
    warn ""
    warn "Or skip pip entirely -- Arch packages a ROCm build:"
    warn "  sudo pacman -S python-pytorch-rocm"
    warn "then run this script without --install, using system python."
    exit 1
  fi
  say "installing ROCm PyTorch into $VIRTUAL_ENV (several GB)"
  pip install --force-reinstall torch \
      --index-url https://download.pytorch.org/whl/rocm6.4 \
    || { warn "install failed; try a different ROCm index from pytorch.org"
         warn "or use Arch's package: sudo pacman -S python-pytorch-rocm"; exit 1; }
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  say "python: $VIRTUAL_ENV"
else
  say "python: system (no venv active)"
fi

# rocBLAS needs a Tensile kernel library for whatever target it thinks it is
# running on. A wheel that lacks one for your masquerade gives a device that
# enumerates fine and then aborts on the first matmul -- which looks like the
# override failing when it is really a packaging gap.
TLIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','rocblas','library'))" 2>/dev/null)
if [[ -d "$TLIB" ]]; then
  AVAIL=$(ls "$TLIB" | grep -oP 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')
  say "rocBLAS kernels in this wheel: ${AVAIL:-none found}"
  TARGET="gfx$(echo "${PRESET:-9.0.0}" | tr -d '.')"
  if [[ -n "$AVAIL" ]] && ! grep -qw "$TARGET" <<< "$AVAIL"; then
    warn "no rocBLAS kernels for $TARGET in this wheel -- matmul will abort."
    warn ""
    warn "Options, cheapest first:"
    warn "  1. an older wheel that still ships gfx900:"
    warn "       ./install.sh --rocm 6.0     (or 5.7)"
    warn "  2. Arch's build, often compiled for more targets:"
    warn "       sudo pacman -Syu python-pytorch-rocm"
    warn "  3. masquerade as a target that IS present. None of"
    warn "     $AVAIL"
    warn "     is GCN5 like gfx90c, so expect this to fail or be wrong."
    warn ""
    warn "None of this blocks the pipeline: cett-dump uses Vulkan."
  fi
fi

say "running a real matmul under the override"
_NS_PRESET="$PRESET" HSA_OVERRIDE_GFX_VERSION="${PRESET:-9.0.0}" python - <<'PY'
import sys
try:
    import torch
except BaseException as e:
    # ImportError also covers a shared library that failed to load, so
    # reporting "not installed" hides the actual cause. Show it.
    import traceback
    traceback.print_exc()
    sys.exit(f"\nimport torch failed: {type(e).__name__}: {e}")
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
if not torch.cuda.is_available():
    sys.exit("no device visible even with the override -- this path is closed")
print("device:", torch.cuda.get_device_name(0))
try:
    a = torch.randn(1024, 1024, device="cuda")
    b = a @ a
    # Correctness matters more than "it ran": a masqueraded target can produce
    # plausible garbage rather than failing outright.
    ref = (a.cpu() @ a.cpu())
    err = (b.cpu() - ref).abs().max().item()
    print(f"matmul ok, max abs error vs CPU: {err:.2e}")
    if err > 1e-2:
        sys.exit("results diverge from CPU -- the override is NOT safe here")
    h = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    _ = h @ h
    print("bf16 ok")
    free, total = torch.cuda.mem_get_info()
    print(f"vram: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
except Exception as e:
    sys.exit(f"failed: {type(e).__name__}: {e}")
print("\nWorks.")
import os
if not os.environ.get("_NS_PRESET"):
    print("Add to ~/.bashrc to keep it:  export HSA_OVERRIDE_GFX_VERSION=9.0.0")
print("""
What this unlocks, honestly:
  - the PyTorch scripts (merge_selective, export_lora, intervene_model) can
    use the GPU, though they are one-off weight edits where it barely matters
  - small models can use the bf16 hook path in extract_activations.py
What it does not:
  - a 9B in bf16 is ~18GB, more than most integrated GPUs can address;
    cett-dump on Vulkan remains the extraction path for large models.""")
PY
rc=$?
if (( rc != 0 )); then
  warn "the override did not work here."
  if [[ -n "${AVAIL:-}" ]] && ! grep -qw "gfx$(echo "${PRESET:-9.0.0}" | tr -d '.')" <<< "$AVAIL"; then
    warn "Cause: no rocBLAS kernels for your target in this wheel, not the"
    warn "override itself -- HIP saw the device. Try ./install.sh --rocm 6.0"
  fi
  warn "Either way it costs you nothing: cett-dump on Vulkan already uses"
  warn "this GPU, and a 9B in bf16 would not fit regardless."
fi
exit $rc
