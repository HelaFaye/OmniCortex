#!/usr/bin/env bash
# Source this before running local stages:  source env.sh
#
# Every script reads the model path from $OC_GGUF instead of embedding it, so
# this is the one place to point OmniCortex at your model. Put your values in
# env.local.sh (gitignored) or export them before sourcing:
#
#   export OC_GGUF=~/models/Qwen3-8B-Q6_K.gguf
#   export OC_LLAMA=~/llama.cpp            # llama.cpp checkout with cett-dump built
#
# Optional: if the model lives on a removable drive whose mount point changes,
# set OC_VOLUME_UUID and OC_MODEL_REL (path relative to the volume root) and the
# path is resolved from the filesystem UUID instead.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -r "${_here}/env.local.sh" ]] && source "${_here}/env.local.sh"

if [[ -z "${OC_GGUF:-}" && -n "${OC_VOLUME_UUID:-}" && -n "${OC_MODEL_REL:-}" ]]; then
    if [[ -e "/dev/disk/by-uuid/${OC_VOLUME_UUID}" ]]; then
        _dev=$(readlink -f "/dev/disk/by-uuid/${OC_VOLUME_UUID}")
        _mnt=$(findmnt -n -o TARGET --source "${_dev}" 2>/dev/null | head -1)
    fi
    if [[ -n "${_mnt:-}" ]]; then
        export OC_GGUF="${_mnt}/${OC_MODEL_REL}"
    else
        echo "!! volume ${OC_VOLUME_UUID} is not mounted:" >&2
        echo "   udisksctl mount -b /dev/disk/by-uuid/${OC_VOLUME_UUID}" >&2
    fi
fi

export OC_LLAMA="${OC_LLAMA:-${HOME}/llama.cpp}"
export OC_CETT="${OC_CETT:-${OC_LLAMA}/build/bin/llama-cett-dump}"
export OC_LLAMA_SERVER="${OC_LLAMA_SERVER:-${OC_LLAMA}/build/bin/llama-server}"
# The tokenizer and chat template are read from the GGUF itself. Set
# OC_TOKENIZER (an HF id or path) only to force the transformers path.
export OC_TOKENIZER="${OC_TOKENIZER:-}"
# Must exceed the longest sequence: cett-dump needs one decode per sample.
export OC_BATCH="${OC_BATCH:-4096}"

# Some systems ship several Vulkan drivers (e.g. AMDVLK and RADV) and the wrong
# one can fail to initialise. OC_VULKAN_ICD pins one explicitly, e.g.
#   OC_VULKAN_ICD=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
[[ -n "${OC_VULKAN_ICD:-}" && -r "${OC_VULKAN_ICD}" ]] && export VK_DRIVER_FILES="${OC_VULKAN_ICD}"

if [[ -z "${OC_GGUF:-}" ]]; then
    echo "OC_GGUF is not set. Export it (or create env.local.sh) to point at your model." >&2
elif [[ ! -r "${OC_GGUF}" ]]; then
    echo "!! model not readable at: ${OC_GGUF}" >&2
else
    echo "model : ${OC_GGUF} ($(( $(stat -c %s "${OC_GGUF}" 2>/dev/null || stat -f %z "${OC_GGUF}") / 1024 / 1024 )) MB)"
    if [[ -z "${OC_LAYERS:-}" ]] && python3 -c "import gguf" 2>/dev/null; then
        OC_LAYERS=$(python3 - "${OC_GGUF}" <<'PY' 2>/dev/null
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
for f in r.fields.values():
    if f.name.endswith(".block_count"):
        print(int(f.parts[f.data[0]][0])); break
PY
)
        [[ -n "${OC_LAYERS}" ]] && export OC_LAYERS && echo "layers: ${OC_LAYERS} (from GGUF)"
    fi
fi
unset _here _dev _mnt
