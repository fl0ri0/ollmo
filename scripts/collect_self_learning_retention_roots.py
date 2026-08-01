#!/usr/bin/env python3
"""CLI wrapper for self-learning response-frame sidecar retention roots."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.self_learning_retention import _main


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
