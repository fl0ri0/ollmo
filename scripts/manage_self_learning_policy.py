#!/usr/bin/env python3
"""Manage the disabled-by-default accepted-learning policy snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.self_learning import (
    DEFAULT_ACCEPTED_POLICY_SNAPSHOT,
    DEFAULT_SELF_LEARNING_DIR,
    build_default_accepted_learning_policy_snapshot,
    build_self_learning_report,
    load_accepted_learning_policy_snapshot,
    persist_accepted_learning_policy_snapshot,
    promote_policy_improvement_candidate_from_report,
    set_accepted_learning_policy_enabled,
)


def _snapshot_path(value: str | None) -> Path:
    return Path(value or (DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT))


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise SystemExit(f'{path} must contain a JSON object')
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Manage Ollmo accepted-learning policy snapshots without opening them by default.',
    )
    parser.add_argument(
        '--snapshot',
        default=str(DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT),
        help='Accepted-learning policy snapshot path.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('show', help='Show the current snapshot or disabled default.')
    subparsers.add_parser('init', help='Create a disabled snapshot if needed.')

    promote = subparsers.add_parser('promote', help='Promote one improvement candidate into accepted learnings.')
    promote.add_argument('--candidate-id', required=True, help='Candidate id from a self-learning report.')
    promote.add_argument(
        '--report',
        default=str(DEFAULT_SELF_LEARNING_DIR / 'report.json'),
        help='Self-learning report JSON to read candidates from.',
    )
    promote.add_argument('--reviewer', default='operator', help='Reviewer/operator name.')
    promote.add_argument('--note', default='', help='Review note to store with the accepted learning.')

    enable = subparsers.add_parser('enable', help='Explicitly enable accepted learnings as bounded hints.')
    enable.add_argument('--reviewer', default='operator', help='Reviewer/operator name.')
    enable.add_argument('--reason', default='', help='Reason recorded with the activation change.')

    disable = subparsers.add_parser('disable', help='Disable accepted learnings.')
    disable.add_argument('--reviewer', default='operator', help='Reviewer/operator name.')
    disable.add_argument('--reason', default='', help='Reason recorded with the activation change.')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    snapshot_path = _snapshot_path(args.snapshot)

    if args.command == 'show':
        payload = load_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)
    elif args.command == 'init':
        payload = build_default_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)
        persist_accepted_learning_policy_snapshot(payload, output_path=snapshot_path)
    elif args.command == 'promote':
        report_path = Path(args.report)
        if report_path.exists():
            report = _load_json_object(report_path)
        else:
            report = build_self_learning_report(include_cases=False)
        snapshot = load_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)
        payload = promote_policy_improvement_candidate_from_report(
            report,
            candidate_id=args.candidate_id,
            snapshot=snapshot,
            reviewer=args.reviewer,
            review_note=args.note,
        )
        persist_accepted_learning_policy_snapshot(payload, output_path=snapshot_path)
    elif args.command == 'enable':
        snapshot = load_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)
        payload = set_accepted_learning_policy_enabled(
            snapshot,
            enabled=True,
            reviewer=args.reviewer,
            reason=args.reason,
        )
        persist_accepted_learning_policy_snapshot(payload, output_path=snapshot_path)
    elif args.command == 'disable':
        snapshot = load_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)
        payload = set_accepted_learning_policy_enabled(
            snapshot,
            enabled=False,
            reviewer=args.reviewer,
            reason=args.reason,
        )
        persist_accepted_learning_policy_snapshot(payload, output_path=snapshot_path)
    else:
        raise SystemExit(f'unsupported command: {args.command}')

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
