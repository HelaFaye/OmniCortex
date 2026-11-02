#!/usr/bin/env python3
"""OmniCortex Model Transfer GUI/server launcher."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_transfer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
