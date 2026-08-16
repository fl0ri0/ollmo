#!/usr/bin/env python3
"""Build proposal-only self-learning eval cases from frozen response frames."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.self_learning import (
    DEFAULT_ACCEPTED_POLICY_SNAPSHOT,
    DEFAULT_EVAL_CASE_LEDGER,
    DEFAULT_RESPONSE_FRAME_LEDGER,
    DEFAULT_RESPONSE_FRAMES_DIR,
    DEFAULT_SELF_LEARNING_DIR,
    DEFAULT_SELF_LEARNING_REPORT,
    build_self_learning_report,
    persist_accepted_learning_policy_snapshot,
    persist_self_learning_outputs,
    self_learning_output_update_lock,
)
from ollmo_services.self_learning_retention import (
    DEFAULT_RETAINED_SIDECARS_DIR,
    DEFAULT_RETENTION_MANIFEST,
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_merge_output_paths(
    *,
    output_path: Path,
    report_path: Path,
    accepted_policy_path: Path,
    self_learning_dir: Path,
    frame_paths: list[Path],
    monitor_report_path: Path | None,
    graph_rebase_corpus_dir: Path,
) -> None:
    """Keep merge persistence away from every read-only learning input."""

    destinations = {
        '--output': output_path.resolve(strict=False),
        '--report': report_path.resolve(strict=False),
    }
    if len(set(destinations.values())) != len(destinations):
        raise ValueError('--output and --report must resolve to distinct paths')

    resolved_self_learning_dir = self_learning_dir.resolve(strict=False)
    protected_files: dict[str, Path] = {
        'accepted policy snapshot': accepted_policy_path.resolve(strict=False),
        'default accepted policy snapshot': (
            resolved_self_learning_dir / DEFAULT_ACCEPTED_POLICY_SNAPSHOT
        ).resolve(strict=False),
        'repository accepted policy snapshot': (
            REPO_ROOT / DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT
        ).resolve(strict=False),
        'retention manifest': (
            resolved_self_learning_dir / DEFAULT_RETENTION_MANIFEST.name
        ).resolve(strict=False),
        'default retention manifest': DEFAULT_RETENTION_MANIFEST.resolve(strict=False),
        'repository retention manifest': (
            REPO_ROOT / DEFAULT_RETENTION_MANIFEST
        ).resolve(strict=False),
        'default response-frame ledger': (
            DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER
        ).resolve(strict=False),
        'default response-frame index': (
            DEFAULT_RESPONSE_FRAMES_DIR / 'current_index.json'
        ).resolve(strict=False),
        'repository response-frame ledger': (
            REPO_ROOT / DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER
        ).resolve(strict=False),
        'repository response-frame index': (
            REPO_ROOT / DEFAULT_RESPONSE_FRAMES_DIR / 'current_index.json'
        ).resolve(strict=False),
        'default monitor report ledger': Path(
            'state/ollmo_run_monitor/reports.jsonl'
        ).resolve(strict=False),
        'repository monitor report ledger': (
            REPO_ROOT / 'state/ollmo_run_monitor/reports.jsonl'
        ).resolve(strict=False),
    }
    for index, frame_path in enumerate(frame_paths):
        resolved_frame_path = frame_path.resolve(strict=False)
        protected_files[f'response-frame ledger {index + 1}'] = resolved_frame_path
        protected_files[f'response-frame index {index + 1}'] = (
            resolved_frame_path.parent / 'current_index.json'
        ).resolve(strict=False)
    if monitor_report_path is not None:
        protected_files['monitor report ledger'] = monitor_report_path.resolve(strict=False)

    protected_directories: dict[str, Path] = {
        'retained sidecars': (
            resolved_self_learning_dir / DEFAULT_RETAINED_SIDECARS_DIR.name
        ).resolve(strict=False),
        'graph-rebase corpus': graph_rebase_corpus_dir.resolve(strict=False),
        'default retained sidecars': (
            DEFAULT_SELF_LEARNING_DIR / DEFAULT_RETAINED_SIDECARS_DIR.name
        ).resolve(strict=False),
        'repository retained sidecars': (
            REPO_ROOT / DEFAULT_SELF_LEARNING_DIR / DEFAULT_RETAINED_SIDECARS_DIR.name
        ).resolve(strict=False),
        'default response-frame CAS': (
            DEFAULT_RESPONSE_FRAMES_DIR / 'snapshots'
        ).resolve(strict=False),
        'repository response-frame CAS': (
            REPO_ROOT / DEFAULT_RESPONSE_FRAMES_DIR / 'snapshots'
        ).resolve(strict=False),
        'default graph-rebase corpus': Path('state/graph_rebase_shadow_corpus').resolve(
            strict=False
        ),
        'repository graph-rebase corpus': (
            REPO_ROOT / 'state/graph_rebase_shadow_corpus'
        ).resolve(strict=False),
    }
    for index, frame_path in enumerate(frame_paths):
        protected_directories[f'response-frame CAS {index + 1}'] = (
            frame_path.resolve(strict=False).parent / 'snapshots'
        ).resolve(strict=False)

    for destination_name, destination in destinations.items():
        for protected_name, protected_path in protected_files.items():
            if destination == protected_path:
                raise ValueError(
                    f'--merge-existing refuses {destination_name} at protected '
                    f'{protected_name} path: {destination}'
                )
        for protected_name, protected_root in protected_directories.items():
            if _is_within(destination, protected_root):
                raise ValueError(
                    f'--merge-existing refuses {destination_name} inside protected '
                    f'{protected_name}: {destination}'
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
        help='JSONL output path and, with --merge-existing, the existing eval-case input ledger.',
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
        help='Maximum freshly extracted eval cases; merge mode preserves existing cases beyond this limit.',
    )
    parser.add_argument(
        '--no-persist',
        action='store_true',
        help='Print the summary and any merge preview counts without writing files.',
    )
    parser.add_argument(
        '--merge-existing',
        action='store_true',
        help=(
            'Read --output and union existing plus freshly generated cases by case_id; '
            'old-only cases are preserved and fresh duplicates win.'
        ),
    )
    parser.add_argument(
        '--no-shadow-hints',
        action='store_true',
        help='Omit diagnostic-only shadow hints from the report.',
    )
    parser.add_argument(
        '--init-accepted-policy-snapshot',
        action='store_true',
        help='Also create a disabled accepted-learning policy snapshot template (replacement mode only).',
    )
    args = parser.parse_args()
    if args.merge_existing and args.init_accepted_policy_snapshot:
        parser.error(
            '--merge-existing cannot be combined with --init-accepted-policy-snapshot; '
            'merge mode never writes accepted learning policy state'
        )
    return args


def main() -> int:
    args = _parse_args()
    frame_paths = [Path(value) for value in (args.frames or ['state/response_frames/responses.jsonl'])]
    output_path = Path(args.output)
    report_path = Path(args.report)
    accepted_policy_path = Path(args.accepted_policy)
    monitor_report_path = Path(args.monitor_reports) if args.monitor_reports else None
    graph_rebase_corpus_dir = Path(args.graph_rebase_corpus_dir)
    self_learning_dir = Path(args.self_learning_dir)
    if args.merge_existing and not args.no_persist:
        try:
            _validate_merge_output_paths(
                output_path=output_path,
                report_path=report_path,
                accepted_policy_path=accepted_policy_path,
                self_learning_dir=self_learning_dir,
                frame_paths=frame_paths,
                monitor_report_path=monitor_report_path,
                graph_rebase_corpus_dir=graph_rebase_corpus_dir,
            )
        except ValueError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 2
    update_context = (
        self_learning_output_update_lock([output_path, report_path])
        if not args.no_persist
        else nullcontext()
    )
    with update_context:
        if args.merge_existing and not args.no_persist:
            try:
                _validate_merge_output_paths(
                    output_path=output_path,
                    report_path=report_path,
                    accepted_policy_path=accepted_policy_path,
                    self_learning_dir=self_learning_dir,
                    frame_paths=frame_paths,
                    monitor_report_path=monitor_report_path,
                    graph_rebase_corpus_dir=graph_rebase_corpus_dir,
                )
            except ValueError as exc:
                print(f'error: {exc}', file=sys.stderr)
                return 2
        report = build_self_learning_report(
            response_frame_ledger_path=frame_paths[0],
            additional_response_frame_ledger_paths=frame_paths[1:],
            monitor_report_path=monitor_report_path,
            graph_rebase_corpus_dir=graph_rebase_corpus_dir,
            self_learning_dir=self_learning_dir,
            existing_eval_case_ledger_path=output_path,
            merge_existing=args.merge_existing,
            frame_limit=args.frame_limit,
            max_cases=args.max_cases,
            include_cases=True,
            include_shadow_hints=not args.no_shadow_hints,
        )
        cases = report.get('eval_cases') if isinstance(report.get('eval_cases'), list) else []

        if not args.no_persist:
            output_path, report_path = persist_self_learning_outputs(
                cases,
                report,
                eval_case_output_path=output_path,
                report_output_path=report_path,
            )
            if args.init_accepted_policy_snapshot:
                accepted_policy_path = persist_accepted_learning_policy_snapshot(
                    output_path=accepted_policy_path
                )

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
        'previous_case_count': report.get('previous_case_count', 0),
        'new_case_count': report.get('new_case_count', 0),
        'preserved_case_count': report.get('preserved_case_count', 0),
        'replaced_case_count': report.get('replaced_case_count', 0),
        'removed_case_count': report.get('removed_case_count', 0),
        'merge_policy': report.get('merge_policy', {}),
        'merge_preview': bool(args.merge_existing and args.no_persist),
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
