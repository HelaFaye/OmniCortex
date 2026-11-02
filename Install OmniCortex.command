#!/bin/bash
# Double-click to install OmniCortex on macOS. If macOS says it cannot be opened,
# right-click it and choose Open (it is a script you downloaded, not a signed app).
cd "$(dirname "$0")" || exit 1
ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; }
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && ok "$c"; then PY="$(command -v "$c")"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "OmniCortex needs Python 3.10 or newer. The python3 that comes with macOS's"
  echo "command-line tools is 3.9, which is too old."
  echo
  echo "Opening python.org: install the macOS package, then double-click this file again."
  echo "(With Homebrew instead:  brew install python@3.12)"
  open "https://www.python.org/downloads/macos/"
  read -r -p "Press Return to close."
  exit 1
fi
echo "Starting the OmniCortex installer in your browser (keep this window open)..."
exec "$PY" scripts/installer.py "$@"
