#!/usr/bin/env python3
"""Verify one response-frame epoch and optionally retain readiness evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.graph_rebase_readiness_registry import (
    DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH,
    sync_graph_rebase_readiness_epoch,
)
from ollmo_services.response_frames import (
    DEFAULT_RESPONSE_FRAME_INDEX,
    DEFAULT_RESPONSE_FRAME_LEDGER,
    DEFAULT_RESPONSE_FRAMES_DIR,
)


_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


def _sha256(value: str) -> str:
    normalized = str(value or '').strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError('expected a lowercase 64-character SHA-256')
    return normalized


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('expected a non-negative integer')
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify an exact response-frame epoch and, only with --write, '
            'append its settled bounded graph-rebase observations.'
        ),
    )
    parser.add_argument(
        '--response-frames-dir',
        '--frames-dir',
        dest='frames_dir',
        default=str(DEFAULT_RESPONSE_FRAMES_DIR),
        help='Directory containing responses.jsonl, current_index.json, and snapshots.',
    )
    parser.add_argument(
        '--registry',
        default=str(DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH),
        help='Append-only readiness registry path.',
    )
    parser.add_argument('--ledger-name', default=DEFAULT_RESPONSE_FRAME_LEDGER)
    parser.add_argument('--index-name', default=DEFAULT_RESPONSE_FRAME_INDEX)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--write',
        action='store_true',
        help='Append verified settled observations. Default is check-only.',
    )
    mode.add_argument(
        '--check-only',
        action='store_true',
        help='Explicitly select the default read-only behavior.',
    )
    parser.add_argument(
        '--require-all-settled',
        action='store_true',
        help='Fail when any selected graph-rebase observation is not settled.',
    )
    parser.add_argument(
        '--no-relocated-index',
        action='store_true',
        help='Require the stored index ledger path to resolve to --frames-dir.',
    )
    parser.add_argument(
        '--expected-index-sha256',
        type=_sha256,
        help='Fail unless current_index.json has this exact SHA-256.',
    )
    parser.add_argument(
        '--expected-ledger-sha256',
        type=_sha256,
        help='Fail unless responses.jsonl has this exact SHA-256.',
    )
    parser.add_argument('--expected-ledger-line-count', type=_non_negative_int)
    parser.add_argument('--expected-response-count', type=_non_negative_int)
    parser.add_argument('--expected-selected-count', type=_non_negative_int)
    parser.add_argument('--expected-settled-count', type=_non_negative_int)
    parser.add_argument('--expected-active-count', type=_non_negative_int)
    parser.add_argument(
        '--shell-summary',
        action='store_true',
        help='Print stable key=value lines for cleanup/archive preflight scripts.',
    )
    return parser.parse_args(argv)


def _print_shell_summary(result: dict) -> None:
    fields = (
        'status',
        'selected_observation_count',
        'settled_observation_count',
        'registered_observation_count',
        'missing_settled_observation_count',
        'active_observation_count',
        'scan_error_count',
        'hydration_error_count',
        'registry_error_count',
        'appended_record_count',
        'already_present_count',
        'error_count',
    )
    for field in fields:
        value = result.get(field)
        if field == 'status':
            value = re.sub(r'[^a-z0-9_-]+', '_', str(value or 'rejected').lower())
        else:
            try:
                value = int(value or 0)
            except (TypeError, ValueError):
                value = 0
        print(f'{field}={value}')


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = sync_graph_rebase_readiness_epoch(
        frames_dir=Path(args.frames_dir),
        registry_path=Path(args.registry),
        ledger_name=args.ledger_name,
        index_name=args.index_name,
        write=bool(args.write),
        require_all_settled=bool(args.require_all_settled),
        allow_relocated=not bool(args.no_relocated_index),
        expected_index_sha256=args.expected_index_sha256,
        expected_ledger_sha256=args.expected_ledger_sha256,
        expected_ledger_line_count=args.expected_ledger_line_count,
        expected_response_count=args.expected_response_count,
        expected_selected_count=args.expected_selected_count,
        expected_settled_count=args.expected_settled_count,
        expected_active_count=args.expected_active_count,
    )
    if args.shell_summary:
        _print_shell_summary(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get('ok') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
