#!/usr/bin/env python3
"""Explicitly attest complete response-index coverage from ledger truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.response_frames import (
    DEFAULT_RESPONSE_FRAME_INDEX,
    DEFAULT_RESPONSE_FRAME_LEDGER,
    DEFAULT_RESPONSE_FRAMES_DIR,
    attest_response_frame_index,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Stream the complete response-frame ledger and add v2 absence-proof '
            'fields only when the existing index matches it exactly.'
        ),
    )
    parser.add_argument(
        '--frames-dir',
        default=str(DEFAULT_RESPONSE_FRAMES_DIR),
        help='Directory containing the response ledger and current index.',
    )
    parser.add_argument(
        '--ledger-name',
        default=DEFAULT_RESPONSE_FRAME_LEDGER,
        help='Response-frame ledger filename inside --frames-dir.',
    )
    parser.add_argument(
        '--index-name',
        default=DEFAULT_RESPONSE_FRAME_INDEX,
        help='Response-frame index filename inside --frames-dir.',
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Verify the complete mapping but do not replace the index.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = attest_response_frame_index(
        frames_dir=Path(args.frames_dir),
        ledger_name=args.ledger_name,
        index_name=args.index_name,
        write=not args.check_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get('ok') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
