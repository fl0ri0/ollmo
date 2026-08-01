#!/usr/bin/env python3
"""Entry point for the Ollmo unified client CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.ollmoctl import main


if __name__ == '__main__':
    sys.exit(main())
