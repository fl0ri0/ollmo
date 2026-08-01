#!/usr/bin/env python3
"""Audit or explicitly compact historical Response Frame Ghost previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.response_frame_ledger_maintenance import (
    compact_response_frame_ledger,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Audit response-frame sidecar integrity and compact oversized '
            'historical request.ghost_preview payloads through the existing '
            'SHA-256 content-addressed snapshot store.'
        )
    )
    parser.add_argument(
        '--frames-dir',
        default='state/response_frames',
        help='Response-frame state root (default: state/response_frames).',
    )
    parser.add_argument(
        '--ledger-name',
        default='responses.jsonl',
        help='Ledger filename within the frames root.',
    )
    parser.add_argument(
        '--index-name',
        default='current_index.json',
        help='Derived current-index filename within the frames root.',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help=(
            'Perform the guarded ledger/index rewrite. Without this flag the '
            'command is strictly read-only.'
        ),
    )
    parser.add_argument(
        '--writers-stopped',
        action='store_true',
        help=(
            'Required with --execute: assert that the control plane and every '
            'external response-frame writer are stopped for the full command.'
        ),
    )
    parser.add_argument(
        '--backup-dir',
        default=None,
        help=(
            'Optional new backup directory. By default a timestamped sibling '
            'under state/response_frame_ledger_backups is created.'
        ),
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Print the complete machine-readable report.',
    )
    return parser


def _number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _human_summary(report: Mapping[str, Any]) -> str:
    preflight = (
        report.get('preflight')
        if isinstance(report.get('preflight'), Mapping)
        else report
    )
    lines = [
        f"ok={str(report.get('ok') is True).lower()}",
        f"mode={report.get('mode') or 'audit'}",
        f"changed={str(report.get('changed') is True).lower()}",
        f"ledger_line_count={_number(preflight.get('ledger_line_count'))}",
        f"response_count={_number(preflight.get('response_count'))}",
        (
            'inline_ghost_preview_frame_count='
            f"{_number(preflight.get('inline_ghost_preview_frame_count'))}"
        ),
        (
            'eligible_ghost_preview_frame_count='
            f"{_number(preflight.get('eligible_ghost_preview_frame_count'))}"
        ),
        (
            'inline_ghost_preview_bytes='
            f"{_number(preflight.get('inline_ghost_preview_bytes'))}"
        ),
        (
            'estimated_reclaimable_inline_bytes='
            f"{_number(preflight.get('estimated_reclaimable_inline_bytes'))}"
        ),
        (
            'authoritative_missing_sidecar_count='
            f"{_number(preflight.get('authoritative_missing_sidecar_count'))}"
        ),
        (
            'digest_only_audit_identity_unique_count='
            f"{_number(preflight.get('digest_only_audit_identity_unique_count'))}"
        ),
        'digest_only_audit_identities_are_sidecars=false',
    ]
    rewrite = report.get('rewrite')
    if isinstance(rewrite, Mapping):
        lines.extend(
            [
                f"target_size_bytes={_number(rewrite.get('target_size_bytes'))}",
                (
                    'reclaimed_ledger_bytes='
                    f"{_number(rewrite.get('reclaimed_ledger_bytes'))}"
                ),
                (
                    'changed_frame_count='
                    f"{_number(rewrite.get('changed_frame_count'))}"
                ),
            ]
        )
    backup = report.get('backup')
    if isinstance(backup, Mapping):
        lines.append(f"backup_directory={backup.get('directory')}")
    error = report.get('error')
    if isinstance(error, Mapping):
        lines.append(f"error_code={error.get('code')}")
        lines.append(f"error_message={error.get('message')}")
    return '\n'.join(lines)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.writers_stopped and not args.execute:
        parser.error('--writers-stopped is meaningful only with --execute')
    report = compact_response_frame_ledger(
        frames_dir=Path(args.frames_dir),
        ledger_name=args.ledger_name,
        index_name=args.index_name,
        execute=args.execute,
        backup_dir=Path(args.backup_dir) if args.backup_dir else None,
        writers_stopped=args.writers_stopped,
    )
    if args.json:
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
    else:
        print(_human_summary(report))
    return 0 if report.get('ok') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
