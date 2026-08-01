#!/usr/bin/env python3
"""Operator entrypoint for explicit external-client provider unsync."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_integrations import provider_unsync


if __name__ == '__main__':
    raise SystemExit(provider_unsync.main())
