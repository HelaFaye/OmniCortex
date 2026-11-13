#!/usr/bin/env bash
# Install OmniCortex with the installer wizard in your browser (Linux; macOS works too).
# The terminal-only installer is ./install.sh, or: python3 scripts/installer.py --yes
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; }
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && ok "$c"; then PY="$(command -v "$c")"; break; fi
done
hint() {
  if command -v apt >/dev/null; then echo "  sudo apt install $1"
  elif command -v dnf >/dev/null; then echo "  sudo dnf install $2"
  elif command -v pacman >/dev/null; then echo "  sudo pacman -S $3"
  elif command -v zypper >/dev/null; then echo "  sudo zypper install $2"
  else echo "  (your distribution's package for $1)"; fi
}
if [[ -z "$PY" ]]; then
  echo "OmniCortex needs Python 3.10 or newer. Install it with:"; hint python3 python3 python; exit 1
fi
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  echo "Python's venv module is missing. Install it with:"; hint python3-venv python3 python; exit 1
fi
exec "$PY" scripts/installer.py "$@"
