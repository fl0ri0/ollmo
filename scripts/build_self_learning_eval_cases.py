#!/usr/bin/env python3
"""Build proposal-only self-learning eval cases from frozen response frames."""

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
    DEFAULT_EVAL_CASE_LEDGER,
    DEFAULT_SELF_LEARNING_DIR,
    DEFAULT_SELF_LEARNING_REPORT,
    build_self_learning_report,
    persist_eval_cases,
    persist_accepted_learning_policy_snapshot,
    persist_self_learning_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build Ollmo self-learning eval cases from frozen response-frame truth traces.',
    )
    parser.add_argument(
        '--frames',
        action='append',
        default=None,
        help=(
            'Response-frame JSONL ledger to scan. Repeat for older ledger epochs; '
            'the first ledger wins when the same response id appears more than once.'
        ),
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_SELF_LEARNING_DIR / DEFAULT_EVAL_CASE_LEDGER),
        help='JSONL path for extracted eval cases.',
    )
    parser.add_argument(
        '--report',
        default=str(DEFAULT_SELF_LEARNING_DIR / DEFAULT_SELF_LEARNING_REPORT),
        help='JSON path for the compact self-learning report.',
    )
    parser.add_argument(
        '--monitor-reports',
        default=None,
        help='Optional monitor reports JSONL path to use as supporting evidence.',
    )
    parser.add_argument(
        '--graph-rebase-corpus-dir',
        default='state/graph_rebase_shadow_corpus',
        help='Directory containing version-1 graph-rebase shadow corpus manifests.',
    )
    parser.add_argument(
        '--accepted-policy',
        default=str(DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT),
        help='Path for the disabled accepted-learning policy snapshot template.',
    )
    parser.add_argument(
        '--self-learning-dir',
        default=str(DEFAULT_SELF_LEARNING_DIR),
        help='Directory containing active self-learning JSON/JSONL state for retention diagnostics.',
    )
    parser.add_argument(
        '--frame-limit',
        type=int,
        default=200,
        help='Maximum recent response frames to scan.',
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=80,
        help='Maximum eval cases to extract.',
    )
    parser.add_argument(
        '--no-persist',
        action='store_true',
        help='Print the summary without writing eval-case or report files.',
    )
    parser.add_argument(
        '--no-shadow-hints',
        action='store_true',
        help='Omit diagnostic-only shadow hints from the report.',
    )
    parser.add_argument(
        '--init-accepted-policy-snapshot',
        action='store_true',
        help='Also create a disabled accepted-learning policy snapshot template.',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    frame_paths = [Path(value) for value in (args.frames or ['state/response_frames/responses.jsonl'])]
    report = build_self_learning_report(
        response_frame_ledger_path=frame_paths[0],
        additional_response_frame_ledger_paths=frame_paths[1:],
        monitor_report_path=Path(args.monitor_reports) if args.monitor_reports else None,
        graph_rebase_corpus_dir=Path(args.graph_rebase_corpus_dir),
        self_learning_dir=Path(args.self_learning_dir),
        frame_limit=args.frame_limit,
        max_cases=args.max_cases,
        include_cases=True,
        include_shadow_hints=not args.no_shadow_hints,
    )
    cases = report.get('eval_cases') if isinstance(report.get('eval_cases'), list) else []
    output_path = Path(args.output)
    report_path = Path(args.report)
    accepted_policy_path = Path(args.accepted_policy)

    if not args.no_persist:
        output_path = persist_eval_cases(cases, output_path=output_path)
        report_path = persist_self_learning_report(report, output_path=report_path)
        if args.init_accepted_policy_snapshot:
            accepted_policy_path = persist_accepted_learning_policy_snapshot(output_path=accepted_policy_path)

    summary = {
        'status': report.get('status'),
        'optimization_policy': report.get('optimization_policy'),
        'frame_count': report.get('frame_count', 0),
        'evaluated_response_count': report.get('evaluated_response_count', 0),
        'superseded_frame_count': report.get('superseded_frame_count', 0),
        'case_count': report.get('case_count', 0),
        'improvement_candidate_count': report.get('improvement_candidate_count', 0),
        'shadow_hint_count': report.get('shadow_hint_count', 0),
        'unique_case_count_before_cap': report.get('unique_case_count_before_cap', 0),
        'case_truncated_count': report.get('case_truncated_count', 0),
        'graph_rebase_corpus': report.get('graph_rebase_corpus', {}),
        'counts_by_layer': report.get('counts_by_layer', {}),
        'counts_by_kind': report.get('counts_by_kind', {}),
        'counts_by_severity': report.get('counts_by_severity', {}),
        'retention': report.get('retention', {}),
        'eval_case_output': None if args.no_persist else str(output_path),
        'report_output': None if args.no_persist else str(report_path),
        'accepted_policy_snapshot_output': (
            str(accepted_policy_path)
            if args.init_accepted_policy_snapshot and not args.no_persist
            else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
