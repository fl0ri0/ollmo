#!/usr/bin/env python3
"""Operator entrypoint for Codex provider cleanup."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_integrations.codex.provider_cleanup import cleanup_providers


def main() -> None:
    cleanup_providers()


if __name__ == "__main__":
    main()
