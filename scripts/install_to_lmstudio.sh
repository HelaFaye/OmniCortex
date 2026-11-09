#!/usr/bin/env bash
# Merge a suppression adapter into a standalone GGUF and place it where
# LM Studio will find it.
#
# LM Studio has no adapter loading -- the runtime scale dial is a llama-server
# feature -- so the model you use in the GUI has to be pre-merged. That is fine
# as long as you merge late: tune with base + adapter against llama-server,
# which costs one 8.3GB file plus a ~30MB adapter, and only merge the scale you
# settle on. Merging every candidate would cost 8.3GB each.
#
#   ./scripts/install_to_lmstudio.sh \
#       --lora adapters/suppress-lora.gguf \
#       --alpha 1.0 --tag s010
#
# Reads OC_GGUF and OC_LLAMA from env.sh. Override with --base / --llama.

set -euo pipefail

BASE="${OC_GGUF:-}"
LLAMA="${OC_LLAMA:-$HOME/llama.cpp}"
LORA=""
ALPHA="1.0"
TAG=""
PUBLISHER="omnicortex"
MODELS_DIR=""
DRY=0

usage() { sed -n '2,16p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)       BASE="$2"; shift 2 ;;
    --lora)       LORA="$2"; shift 2 ;;
    --alpha)      ALPHA="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    --publisher)  PUBLISHER="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --llama)      LLAMA="$2"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

die() { echo "!! $*" >&2; exit 1; }

[[ -n "$BASE" ]] || die "no base model. source env.sh, or pass --base"
[[ -r "$BASE" ]] || die "base not readable: $BASE"
[[ -n "$LORA" ]] || die "--lora is required"
[[ -r "$LORA" ]] || die "adapter not readable: $LORA"

EXPORT_BIN="${LLAMA}/build/bin/llama-export-lora"
[[ -x "$EXPORT_BIN" ]] || die "llama-export-lora not found at $EXPORT_BIN
   build it:  cmake --build ${LLAMA}/build --target llama-export-lora -j"

# Find LM Studio's model root. It is wherever the GUI points, which is not
# necessarily under $HOME -- on this setup it lives beside the base model.
if [[ -z "$MODELS_DIR" ]]; then
  for cand in "$(dirname "$(dirname "$(dirname "$BASE")")")" \
              "$HOME/.lmstudio/models" \
              "$HOME/.cache/lm-studio/models"; do
    if [[ -d "$cand" ]]; then MODELS_DIR="$cand"; break; fi
  done
fi
[[ -n "$MODELS_DIR" && -d "$MODELS_DIR" ]] \
  || die "could not locate the LM Studio model directory; pass --models-dir"

BASENAME="$(basename "$BASE" .gguf)"
[[ -n "$TAG" ]] || TAG="a${ALPHA//./}"
# The tag lands in the name because several suppression strengths will coexist
# in the GUI and they are otherwise indistinguishable.
REPO="${BASENAME}-nscope-${TAG}-GGUF"
OUTNAME="${BASENAME}-nscope-${TAG}.gguf"
DEST_DIR="${MODELS_DIR}/${PUBLISHER}/${REPO}"
DEST="${DEST_DIR}/${OUTNAME}"

echo "base       : $BASE"
echo "adapter    : $LORA  (scale $ALPHA)"
echo "destination: $DEST"

if [[ -e "$DEST" ]]; then
  die "already exists: $DEST
   remove it or choose a different --tag"
fi

avail=$(df -Pk "$MODELS_DIR" | awk 'NR==2{print $4*1024}')
need=$(stat -c %s "$BASE")
if (( avail < need + need/10 )); then
  die "only $(( avail/1024/1024 ))MB free at $MODELS_DIR, need ~$(( need/1024/1024 ))MB.
   Each merged variant is a full copy of the base model."
fi

if (( DRY )); then
  echo "(dry run, nothing written)"
  exit 0
fi

mkdir -p "$DEST_DIR"
echo
echo "merging..."
"$EXPORT_BIN" -m "$BASE" --lora-scaled "$LORA" "$ALPHA" -o "$DEST"

[[ -s "$DEST" ]] || die "merge produced no output"

# Provenance beside the model: six months from now the tag alone will not tell
# you which neuron set or which sweep produced this file.
cat > "${DEST_DIR}/omnicortex.json" <<EOF
{
  "base_model": "$(basename "$BASE")",
  "adapter": "$(basename "$LORA")",
  "adapter_scale": ${ALPHA},
  "merged": "$(date -Iseconds)",
  "note": "OmniCortex suppression baked in. Effective neuron scale is 1 + alpha*(s-1) where s is the profile scale the adapter was exported at."
}
EOF

echo
echo "done. $(( $(stat -c %s "$DEST") / 1024 / 1024 )) MB"
echo
echo "It will appear in LM Studio as ${PUBLISHER}/${REPO} after a rescan"
echo "(the GUI picks up new folders automatically; 'lms ls' confirms)."
echo
echo "For LM Link: peers serve models loaded on their own device, so this file"
echo "has to exist on whichever machine actually runs it, not just the one you"
echo "are sitting at."
