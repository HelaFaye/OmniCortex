#!/usr/bin/env bash
# CI: build one target's binaries on Linux or macOS (also inside the 32-bit and ARM containers).
#
#   scripts/ci/build_unix.sh <target> [llama.cpp ref]
#
# <target> names the package, e.g. linux-x64, linux-arm64, linux-x86, linux-armv7, macos-arm64, macos-x64.
# Writes dist/omnicortex-<target>/ with
#   bin/      llama-cett-dump, llama-server, llama-quantize, llama-eval-callback (and their shared libraries)
#   omnicortex-remote   the remote agent as one executable
# and dist/omnicortex-<target>.tar.gz. Environment: BACKEND (default cpu), CMAKE_EXTRA, SKIP_LLAMA, SKIP_AGENT.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:?target}"; REF="${2:-}"
OUT="$here/dist/omnicortex-$TARGET"
rm -rf "$OUT"; mkdir -p "$OUT/bin"

if [[ -z "${SKIP_LLAMA:-}" ]]; then
  args=(--dir "$here/build/llama.cpp" --backend "${BACKEND:-cpu}" --portable)
  [[ -n "$REF" ]] && args+=(--ref "$REF")
  # static: the executables then run from any folder, with no shared libraries beside them
  args+=(--cmake "-DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF ${CMAKE_EXTRA:-}")
  bash "$here/scripts/build_llama_tools.sh" "${args[@]}"
  b="$here/build/llama.cpp/build/bin"
  for f in llama-cett-dump llama-server llama-quantize llama-eval-callback; do cp "$b/$f" "$OUT/bin/"; done
  find "$b" -maxdepth 1 -name '*.metal' -exec cp {} "$OUT/bin/" \; || true
fi

if [[ -z "${SKIP_AGENT:-}" ]]; then
  python3 -m pip install --quiet pyinstaller
  python3 -m PyInstaller --onefile --clean --name omnicortex-remote --paths "$here/remote" \
    --distpath "$OUT" --workpath "$here/build/pyi-$TARGET" --specpath "$here/build/pyi-$TARGET" \
    "$here/scripts/ci/remote_entry.py"
  "$OUT/omnicortex-remote" --help >/dev/null          # it starts on this target
fi

cp "$here/README.md" "$OUT/"
cp "$here/remote/README.md" "$OUT/REMOTE.md" 2>/dev/null || true
(cd "$here/dist" && tar czf "omnicortex-$TARGET.tar.gz" "omnicortex-$TARGET")
ls -la "$OUT" "$OUT/bin"
