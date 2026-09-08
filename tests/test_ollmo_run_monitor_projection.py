import subprocess
import sys
import json
from pathlib import Path

import pytest

import scripts.ollmo_run_monitor as monitor
from scripts.ollmo_run_monitor import (
    _collect_runtime_repair_authority,
    _render_human,
    _wave_diagnostic_from_history,
)


def test_late_fill_skip_is_clean_only_when_already_fulfilled_without_work() -> None:
    late_fill = {
        'status': 'skipped',
        'skip_reason': 'already_fulfilled',
        'skip_kind': 'no_work_needed',
        'skip_source': 'late_fill_prune',
    }

    assert monitor._late_fill_is_clean(late_fill) is True


def test_other_late_fill_skip_is_not_clean() -> None:
    late_fill = {
        'status': 'skipped',
        'skip_reason': 'already_fulfilled',
        'skip_kind': 'no_work_needed',
        'skip_source': 'manual_override',
    }

    assert monitor._late_fill_is_clean(late_fill) is False


def _synthetic_terminal_report(
    response_id: str,
    frame_id: str,
    frame_sequence: int,
    *,
    verdict: str,
    lifecycle_state: str,
) -> dict:
    return {
        'response_id': response_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'reported_at': '2026-09-03T12:00:00Z',
        'verdict': verdict,
        'lifecycle_state': lifecycle_state,
        'human_report': f'{verdict} {frame_id}',
    }


def test_monitor_emits_report_for_same_response_successor_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_id = 'resp_same_response_retry'
    latest = {
        'line': 1,
        'bytes': 1,
        'frame_id': f'{response_id}:frame-1',
        'frame_sequence': 1,
    }

    monkeypatch.setattr(
        monitor,
        '_scan_responses',
        lambda _root: ([response_id], {response_id: dict(latest)}),
    )

    def fake_build_report(_root, _response_id, _next_response_id, latest_line):
        frame_sequence = int(latest_line['frame_sequence'])
        if frame_sequence == 1:
            return _synthetic_terminal_report(
                response_id,
                latest_line['frame_id'],
                frame_sequence,
                verdict='needs_attention',
                lifecycle_state='repair_needed',
            )
        return _synthetic_terminal_report(
            response_id,
            latest_line['frame_id'],
            frame_sequence,
            verdict='clean',
            lifecycle_state='completed',
        )

    monkeypatch.setattr(monitor, '_build_report', fake_build_report)
    state_dir = tmp_path / 'monitor'

    assert monitor.run_once(tmp_path, state_dir, quiet=True) == 0
    latest.update(
        {
            'line': 2,
            'frame_id': f'{response_id}:frame-2',
            'frame_sequence': 2,
        }
    )
    assert monitor.run_once(tmp_path, state_dir, quiet=True) == 0

    reports = [
        json.loads(line)
        for line in (state_dir / 'reports.jsonl').read_text().splitlines()
    ]
    assert [(item['frame_id'], item['verdict']) for item in reports] == [
        (f'{response_id}:frame-1', 'needs_attention'),
        (f'{response_id}:frame-2', 'clean'),
    ]


def test_monitor_does_not_duplicate_same_response_same_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_id = 'resp_same_response_no_duplicate'
    latest = {
        'line': 1,
        'bytes': 1,
        'frame_id': f'{response_id}:frame-1',
        'frame_sequence': 1,
    }
    monkeypatch.setattr(
        monitor,
        '_scan_responses',
        lambda _root: ([response_id], {response_id: dict(latest)}),
    )
    monkeypatch.setattr(
        monitor,
        '_build_report',
        lambda _root, _response_id, _next_response_id, latest_line: (
            _synthetic_terminal_report(
                response_id,
                latest_line['frame_id'],
                latest_line['frame_sequence'],
                verdict='clean',
                lifecycle_state='completed',
            )
        ),
    )
    state_dir = tmp_path / 'monitor'

    assert monitor.run_once(tmp_path, state_dir, quiet=True) == 0
    assert monitor.run_once(tmp_path, state_dir, quiet=True) == 0
    assert len((state_dir / 'reports.jsonl').read_text().splitlines()) == 1


def test_monitor_legacy_frameless_report_remains_response_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_id = 'resp_legacy_frameless'
    latest = {
        'line': 2,
        'bytes': 1,
        'frame_id': f'{response_id}:frame-2',
        'frame_sequence': 2,
    }
    monkeypatch.setattr(
        monitor,
        '_scan_responses',
        lambda _root: ([response_id], {response_id: dict(latest)}),
    )
    state_dir = tmp_path / 'monitor'
    state_dir.mkdir(parents=True)
    (state_dir / 'reports.jsonl').write_text(
        json.dumps(
            {
                'response_id': response_id,
                'reported_at': '2026-09-03T12:00:00Z',
                'verdict': 'clean',
                'lifecycle_state': 'completed',
                'human_report': 'legacy frame-less report',
            }
        )
        + '\n'
    )

    assert monitor.run_once(tmp_path, state_dir, quiet=True) == 0
    assert monitor.run_once(tmp_path, state_dir, quiet=True) == 0
    assert len((state_dir / 'reports.jsonl').read_text().splitlines()) == 1


def test_monitor_compatibility_entrypoint_imports_from_any_working_directory(
    tmp_path: Path,
) -> None:
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / 'state'
        / 'ollmo_run_monitor'
        / 'monitor_once.py'
    )

    result = subprocess.run(
        [sys.executable, str(entrypoint), '--help'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'Append Ollmo run monitor reports from local runtime truth.' in result.stdout


def _base_report(wave: dict) -> dict:
    return {
        'response_id': 'resp_test_branch_progress_dispatch',
        'frame_sequence': 3,
        'verdict': 'clean',
        'lifecycle_state': 'completed',
        'late_fill_status': 'completed',
        'final_materialization_contract_status': 'fulfilled',
        'branch_counts': {'pending': 0, 'active': 0, 'failed': 0, 'completed': 1},
        'failed_branch_count': 0,
        'materialization_contract_unmet': False,
        'timing': {
            'start': '2026-06-15T19:29:14Z',
            'initial_chat_finished': '2026-06-15T19:35:26Z',
            'initial_chat_seconds': 372.0,
            'image_wave_start': '2026-06-15T19:36:22Z',
            'image_wave_end': '2026-06-15T19:47:11Z',
            'image_wave_seconds': 649.0,
            'terminal_output': '2026-06-15T19:50:51Z',
            'terminal_seconds': 1297.0,
            'hygiene_finished': '2026-06-15T19:51:46Z',
            'hygiene_seconds': 1352.0,
        },
        'sizes': {
            'state_total_bytes': 1000,
            'state_add_approx_bytes': 100,
            'snapshots_total_bytes': 900,
            'snapshots_add_bytes': 90,
            'artifacts_total_bytes': 800,
            'artifacts_add_bytes': 80,
            'latest_response_line_bytes': 70,
            'new_snapshot_file_count': 1,
            'new_artifact_file_count': 1,
        },
        'artifacts': {
            'missing_files': [],
            'sha_mismatches': [],
            'html_issues': [],
            'css_issues': [],
            'html_image_links': [],
            'weak_viewport_tags': [],
            'artifact_kind_counts': {},
            'artifact_file_count_by_suffix': {},
            'output_ref_counts': {},
            'audio_artifact_count': 0,
        },
        'registry_parse_clean': True,
        'image_models': {},
        'branch_role_counts': {},
        'branch_capability_counts': {},
        'output_obligation_counts': {},
        'learning_healing': {},
        'notes': ['No repair/requeue/materialization failure events observed.'],
        'timing_diagnostics': {
            'phase_gaps': {},
            'scheduling_policy': {},
            'waves': [wave],
            'branches': [],
        },
    }


def test_wave_diagnostic_preserves_branch_progress_dispatch_fields() -> None:
    wave = _wave_diagnostic_from_history(
        {
            'elapsed_ms': 668000,
            'planning_elapsed_ms': 20900,
            'worker_count': 1,
            'max_parallel_workers': 4,
            'prepared_branch_count': 16,
            'distinct_instance_ids': ['image-runtime'],
            'branch_progress_dispatch': 'async_ordered',
            'branch_progress_callback_count': 16,
        }
    )

    assert wave['branch_progress_dispatch'] == 'async_ordered'
    assert wave['branch_progress_callback_count'] == 16


def test_human_wave_line_reports_branch_progress_dispatch_when_present() -> None:
    wave = _wave_diagnostic_from_history(
        {
            'elapsed_ms': 668000,
            'planning_elapsed_ms': 20900,
            'worker_count': 1,
            'max_parallel_workers': 4,
            'default_worker_count': 1,
            'worker_count_source': 'default',
            'scheduling_capacity_units': 1,
            'prepared_branch_count': 16,
            'distinct_instance_count': 1,
            'gpu_heavy_guard': 'not_serialized',
            'branch_progress_dispatch': 'async_ordered',
            'branch_progress_callback_count': 16,
        }
    )

    rendered = _render_human(_base_report(wave))

    assert 'progress dispatch async_ordered, callbacks 16' in rendered


def test_human_report_includes_backend_finalize_timing() -> None:
    report = _base_report({})
    report['timing_diagnostics']['backend_finalize'] = {
        'post_wave': {
            'phase': 'nonterminal',
            'status': 'running',
            'finalize_seconds': 143.0,
            'touch_response_lookup_seconds': 0.4,
            'pending_branch_count': 2,
            'active_branch_count': 0,
            'completed_branch_count': 16,
            'failed_branch_count': 0,
            'response_frame_finalize': {
                'phase': 'nonterminal_late_fill',
                'persist_effective': True,
                'total_seconds': 142.5,
                'steps': [
                    {'name': 'build_working_frame', 'elapsed_seconds': 40.0},
                    {'name': 'persist_response_frame', 'elapsed_seconds': 90.0},
                ],
            },
        }
    }

    rendered = _render_human(report)

    assert 'Backend post-wave finalize nonterminal status running' in rendered
    assert 'finalize 2m23s' in rendered
    assert 'lookup touch 0.4s' in rendered
    assert 'Backend response-frame finalize timing: nonterminal_late_fill total 2m22s' in rendered
    assert 'build_working_frame 40.0s' in rendered
    assert 'persist_response_frame 1m30s' in rendered


def test_human_report_includes_nonterminal_and_terminal_backend_finalize_events() -> None:
    report = _base_report({})
    report['timing_diagnostics']['backend_finalize'] = {
        'post_wave_events': [
            {
                'phase': 'nonterminal',
                'status': 'running',
                'finalize_seconds': 120.2,
                'touch_response_lookup_seconds': 21.2,
                'pending_branch_count': 3,
                'active_branch_count': 3,
                'completed_branch_count': 13,
                'failed_branch_count': 0,
            },
            {
                'phase': 'terminal',
                'status': 'completed',
                'finalize_seconds': 48.5,
                'touch_response_lookup_seconds': 12.5,
                'pending_branch_count': 0,
                'active_branch_count': 0,
                'completed_branch_count': 16,
                'failed_branch_count': 0,
            },
        ],
        'event_count': 2,
    }

    rendered = _render_human(report)

    assert 'Backend post-wave finalize #1 nonterminal status running' in rendered
    assert 'finalize 2m00s' in rendered
    assert 'lookup touch 21.2s' in rendered
    assert 'branches pending=3, active=3, completed=13, failed=0' in rendered
    assert 'Backend post-wave finalize #2 terminal status completed' in rendered
    assert 'finalize 48.5s' in rendered
    assert 'lookup touch 12.5s' in rendered


def _partial_rebase_execution() -> dict:
    return {
        'kind': 'ollmo.graph_rebase_partial_successor_execution',
        'status': 'queued',
        'execution_key': 'partial-execution-monitor-1',
        'successor_key': 'partial-successor-monitor-1',
        'proposal_id': 'proposal-partial-monitor-1',
        'rebase_id': 'rebase-partial-monitor-1',
        'authorization_record_id': 'operator-authorization-monitor-1',
        'scheduled_branch_ids': ['branch-image', 'branch-page'],
        'root_prompt_replay': False,
    }


def test_runtime_authority_observes_partial_rebase_execution_without_gaining_authority() -> None:
    execution = _partial_rebase_execution()
    authority = _collect_runtime_repair_authority(
        [{'developer_diagnostics': {
            'graph_rebase_partial_successor_execution': dict(execution),
        }}],
        [{'successor_rebase_executions': [dict(execution)]}],
        [{'graph_rebase_partial_successor_execution': dict(execution)}],
        {},
    )

    successor = authority['successor_runtime']
    observer = successor['partial_rebase_execution_observer']
    assert successor['rebase_execution_count'] == 1
    assert successor['parent_frozen_unmutated'] is True
    assert observer['execution_count'] == 1
    assert observer['status_counts'] == {'queued': 1}
    assert observer['scheduled_branch_count'] == 2
    assert observer['root_prompt_replay_count'] == 0
    assert observer['authorization_record_count'] == 1
    assert observer['authority'] == 'observer_only'
    assert observer['runtime_effect'] == 'none'
    assert authority['graph_rebase_enforcement']['allowed_count'] == 0
    assert authority['graph_rebase_enforcement']['blocked_count'] == 0


def test_human_report_names_partial_rebase_execution_as_observer_only() -> None:
    execution = _partial_rebase_execution()
    authority = _collect_runtime_repair_authority(
        [],
        [{'successor_rebase_executions': [execution]}],
        [],
        {},
    )
    report = _base_report({})
    report['learning_healing'] = {
        'runtime_repair_authority': authority,
    }

    rendered = _render_human(report)

    assert 'Partial rebase execution observer: 1 executions' in rendered
    assert 'statuses queued=1' in rendered
    assert 'scheduled branches 2' in rendered
    assert 'root prompt replays 0' in rendered
    assert 'authority observer_only, runtime effect none' in rendered
