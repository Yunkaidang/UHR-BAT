#!/usr/bin/env python3
"""Backward-compatible single-image inference wrapper."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uhr_bat.infer import main


if __name__ == "__main__":
    main()
