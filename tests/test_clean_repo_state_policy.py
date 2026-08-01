from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_isolated_cleanup_repo(
    tmp_path: Path,
    *,
    readiness_summary: Mapping[str, str | int] | None = None,
    readiness_exit_code: int = 0,
    include_readiness_tool: bool = True,
) -> tuple[Path, dict[str, str], Path]:
    repo_root = tmp_path / 'repo'
    scripts_dir = repo_root / 'scripts'
    response_frames_dir = repo_root / 'state' / 'response_frames'
    registry = repo_root / 'state' / 'graph_rebase' / 'readiness_observations.jsonl'
    fake_bin = tmp_path / 'bin'
    scripts_dir.mkdir(parents=True)
    response_frames_dir.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    fake_bin.mkdir()

    cleanup_script = repo_root / 'clean_repo_state.sh'
    shutil.copy2(REPO_ROOT / 'clean_repo_state.sh', cleanup_script)
    (repo_root / 'ollmo').write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    (repo_root / 'ollmo').chmod(0o755)
    (repo_root / 'model_ports.json').write_text('[]\n', encoding='utf-8')
    (repo_root / 'artifacts' / 'bundles').mkdir(parents=True)
    (repo_root / 'artifacts' / 'bundles' / 'keep.html').write_text(
        '<p>fixture</p>\n',
        encoding='utf-8',
    )
    (response_frames_dir / 'responses.jsonl').write_text(
        '{"response_id":"resp-fixture"}\n',
        encoding='utf-8',
    )
    registry.write_bytes(b'{"record_id":"readiness-fixture"}\n')

    collector = scripts_dir / 'collect_self_learning_retention_roots.py'
    collector.write_text(
        """import sys
assert '--verify' in sys.argv
print('status=complete')
print('retained_sidecar_count=0')
print('missing_sidecar_count=0')
print('external_or_unsafe_ref_count=0')
print('retained_copy_count=0')
""",
        encoding='utf-8',
    )

    readiness_args_file = tmp_path / 'readiness-args.txt'
    if include_readiness_tool:
        summary = dict(
            readiness_summary
            or {
                'status': 'verified',
                'selected_observation_count': 1,
                'settled_observation_count': 1,
                'registered_observation_count': 1,
                'missing_settled_observation_count': 0,
                'active_observation_count': 0,
                'scan_error_count': 0,
                'hydration_error_count': 0,
                'registry_error_count': 0,
                'appended_record_count': 0,
                'already_present_count': 1,
                'error_count': 0,
            }
        )
        readiness_tool = scripts_dir / 'sync_graph_rebase_readiness_registry.py'
        readiness_tool.write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['READINESS_ARGS_FILE']).write_text('\\n'.join(sys.argv[1:]), encoding='utf-8')\n"
            + ''.join(f'print({f"{key}={value}"!r})\n' for key, value in summary.items())
            + f'raise SystemExit({readiness_exit_code})\n',
            encoding='utf-8',
        )

    fake_lsof = fake_bin / 'lsof'
    fake_lsof.write_text('#!/bin/sh\nexit 1\n', encoding='utf-8')
    fake_lsof.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = f'{fake_bin}{os.pathsep}{env.get("PATH", "")}'
    env['READINESS_ARGS_FILE'] = str(readiness_args_file)
    return repo_root, env, readiness_args_file


def run_isolated_cleanup(
    repo_root: Path,
    env: Mapping[str, str],
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(repo_root / 'clean_repo_state.sh'), *args],
        cwd=repo_root,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )


def run_ollmo_dry_run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir(exist_ok=True)
    fake_lsof = fake_bin / 'lsof'
    fake_lsof.write_text('#!/bin/sh\nexit 1\n', encoding='utf-8')
    fake_lsof.chmod(0o755)

    env = os.environ.copy()
    env['PATH'] = f'{fake_bin}{os.pathsep}{env.get("PATH", "")}'
    return subprocess.run(
        [str(REPO_ROOT / 'ollmo'), *args, '--dry-run'],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_archive_full_dry_run_preserves_protected_ghost_state(tmp_path: Path) -> None:
    result = run_ollmo_dry_run(tmp_path, 'archive', '--full')

    assert result.returncode == 0, result.stderr or result.stdout
    assert 'mode: archive' in result.stdout
    assert 'full reset: enabled' in result.stdout
    assert 'Removing Ghost preference state' not in result.stdout
    assert '[dry-run] remove: state/ghost_preferences.json' not in result.stdout
    assert '[dry-run] remove: state/ghost_compiled_memory.json' not in result.stdout
    assert '[dry-run] remove: state/ghost_compiled_memory.md' not in result.stdout
    assert '[dry-run] clear directory contents: state/self_learning' not in result.stdout
    assert '- state/ghost_preferences.json preserved' in result.stdout
    assert 'protected Ghost state snapshotted when present' in result.stdout


def test_archive_and_clean_preserve_artifact_bucket_structure_in_output(tmp_path: Path) -> None:
    archive_result = run_ollmo_dry_run(tmp_path, 'archive', '--full')
    clean_result = run_ollmo_dry_run(tmp_path, 'clean')

    assert archive_result.returncode == 0, archive_result.stderr or archive_result.stdout
    assert clean_result.returncode == 0, clean_result.stderr or clean_result.stdout
    assert 'copy artifacts tree before cleanup' in archive_result.stdout
    assert 'clear artifacts contents while preserving standard bucket dirs' in archive_result.stdout
    assert 'clear artifacts contents while preserving standard bucket dirs' in clean_result.stdout
    assert '[dry-run] ensure dir: artifacts/bundles' in archive_result.stdout
    assert '[dry-run] ensure dir: artifacts/bundles' in clean_result.stdout


def test_clean_and_archive_dry_run_report_learning_retained_sidecars(tmp_path: Path) -> None:
    archive_result = run_ollmo_dry_run(tmp_path, 'archive', '--full')
    clean_result = run_ollmo_dry_run(tmp_path, 'clean')

    assert archive_result.returncode == 0, archive_result.stderr or archive_result.stdout
    assert clean_result.returncode == 0, clean_result.stderr or clean_result.stdout
    assert '[dry-run] preserve learning-retained response-frame sidecars:' in archive_result.stdout
    assert '[dry-run] missing learning-retained sidecars:' in archive_result.stdout
    assert '[dry-run] preserve learning-retained response-frame sidecars:' in clean_result.stdout
    assert '[dry-run] missing learning-retained sidecars:' in clean_result.stdout
    assert 'learning-retained response-frame sidecars' in archive_result.stdout


def test_clean_fail_closes_response_frames_when_retention_is_partial(tmp_path: Path) -> None:
    repo_root = tmp_path / 'repo'
    scripts_dir = repo_root / 'scripts'
    fake_bin = tmp_path / 'bin'
    scripts_dir.mkdir(parents=True)
    fake_bin.mkdir()
    cleanup_script = repo_root / 'clean_repo_state.sh'
    shutil.copy2(REPO_ROOT / 'clean_repo_state.sh', cleanup_script)
    collector = scripts_dir / 'collect_self_learning_retention_roots.py'
    collector.write_text(
        """import sys
assert '--verify' in sys.argv
print('status=partial')
print('retained_sidecar_count=1')
print('missing_sidecar_count=1')
print('external_or_unsafe_ref_count=0')
print('retained_copy_count=0')
""",
        encoding='utf-8',
    )
    fake_lsof = fake_bin / 'lsof'
    fake_lsof.write_text('#!/bin/sh\nexit 1\n', encoding='utf-8')
    fake_lsof.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = f'{fake_bin}{os.pathsep}{env.get("PATH", "")}'

    result = subprocess.run(
        ['bash', str(cleanup_script), '--dry-run'],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert 'self-learning retention is partial' in result.stdout
    assert 'preserving state/response_frames for safety' in result.stdout
    assert '[dry-run] clear directory contents: state/response_frames' not in result.stdout


def test_clean_full_and_forget_ghost_remain_explicit_forget_paths(tmp_path: Path) -> None:
    clean_full = run_ollmo_dry_run(tmp_path, 'clean', '--full')
    clean_forget = run_ollmo_dry_run(tmp_path, 'clean', '--forget-ghost')

    assert clean_full.returncode == 0, clean_full.stderr or clean_full.stdout
    assert clean_forget.returncode == 0, clean_forget.stderr or clean_forget.stdout
    assert 'Removing Ghost preference state' in clean_full.stdout
    assert '- state/ghost_preferences.json removed' in clean_full.stdout
    assert 'Removing Ghost preference state' in clean_forget.stdout
    assert '- state/ghost_preferences.json removed' in clean_forget.stdout


def test_clean_dry_run_warns_about_chrome_artifact_file_access(tmp_path: Path) -> None:
    result = run_ollmo_dry_run(tmp_path, 'archive', '--full')

    assert result.returncode == 0, result.stderr or result.stdout
    assert 'Chrome/macOS artifact preview note' in result.stdout
    assert 'net::ERR_ACCESS_DENIED' in result.stdout
    assert '--reset-chrome-file-access-prompt' in result.stdout


def test_chrome_file_access_prompt_reset_flag_is_opt_in(tmp_path: Path) -> None:
    without_flag = run_ollmo_dry_run(tmp_path, 'archive', '--full')
    with_flag = run_ollmo_dry_run(
        tmp_path,
        'archive',
        '--full',
        '--reset-chrome-file-access-prompt',
    )

    assert without_flag.returncode == 0, without_flag.stderr or without_flag.stdout
    assert with_flag.returncode == 0, with_flag.stderr or with_flag.stdout
    assert '[dry-run] tccutil reset SystemPolicyDesktopFolder com.google.Chrome' not in without_flag.stdout
    assert '[dry-run] tccutil reset SystemPolicyDesktopFolder com.google.Chrome' in with_flag.stdout


def test_readiness_dry_run_is_check_only_and_reports_pending_registry_append(
    tmp_path: Path,
) -> None:
    repo_root, env, args_file = make_isolated_cleanup_repo(
        tmp_path,
        readiness_summary={
            'status': 'verified',
            'selected_observation_count': 2,
            'settled_observation_count': 2,
            'registered_observation_count': 1,
            'missing_settled_observation_count': 1,
            'active_observation_count': 0,
            'scan_error_count': 0,
            'hydration_error_count': 0,
            'registry_error_count': 0,
            'appended_record_count': 0,
            'already_present_count': 1,
            'error_count': 0,
        },
    )

    result = run_isolated_cleanup(repo_root, env, '--dry-run')

    assert result.returncode == 0, result.stderr or result.stdout
    args = args_file.read_text(encoding='utf-8').splitlines()
    assert '--check-only' in args
    assert '--write' not in args
    assert '--require-all-settled' in args
    assert '--shell-summary' in args
    assert args[args.index('--response-frames-dir') + 1] == str(
        repo_root / 'state' / 'response_frames'
    )
    assert args[args.index('--registry') + 1] == str(
        repo_root / 'state' / 'graph_rebase' / 'readiness_observations.jsonl'
    )
    assert '[dry-run] graph-rebase readiness observations: 2 settled / 2 selected' in result.stdout
    assert '[dry-run] graph-rebase readiness registry: 1 registered, 1 pending append' in result.stdout
    assert '[dry-run] clear directory contents: state/response_frames' in result.stdout


def test_readiness_missing_tool_fail_closes_response_frame_cleanup(tmp_path: Path) -> None:
    repo_root, env, _args_file = make_isolated_cleanup_repo(
        tmp_path,
        include_readiness_tool=False,
    )

    result = run_isolated_cleanup(repo_root, env, '--dry-run')

    assert result.returncode == 0, result.stderr or result.stdout
    assert 'readiness retention tool missing' in result.stdout
    assert 'preserving state/response_frames for safety' in result.stdout
    assert '[dry-run] clear directory contents: state/response_frames' not in result.stdout


def test_readiness_invalid_or_unsettled_summary_fail_closes_response_frames(
    tmp_path: Path,
) -> None:
    repo_root, env, _args_file = make_isolated_cleanup_repo(
        tmp_path,
        readiness_summary={
            'status': 'verified',
            'selected_observation_count': 2,
            'settled_observation_count': 1,
            'registered_observation_count': 1,
            'missing_settled_observation_count': 0,
            'active_observation_count': 0,
            'scan_error_count': 0,
            'hydration_error_count': 0,
            'registry_error_count': 1,
            'appended_record_count': 0,
            'already_present_count': 1,
            'error_count': 1,
        },
    )

    result = run_isolated_cleanup(repo_root, env, '--dry-run')

    assert result.returncode == 0, result.stderr or result.stdout
    assert 'readiness evidence is active, incomplete, or invalid' in result.stdout
    assert '[dry-run] clear directory contents: state/response_frames' not in result.stdout


def test_readiness_preflight_failure_does_not_archive_or_remove_response_frames(
    tmp_path: Path,
) -> None:
    repo_root, env, _args_file = make_isolated_cleanup_repo(
        tmp_path,
        readiness_summary={
            'status': 'rejected',
            'selected_observation_count': 1,
            'settled_observation_count': 0,
            'registered_observation_count': 0,
            'missing_settled_observation_count': 0,
            'active_observation_count': 1,
            'scan_error_count': 0,
            'hydration_error_count': 0,
            'registry_error_count': 0,
            'appended_record_count': 0,
            'already_present_count': 0,
            'error_count': 1,
        },
        readiness_exit_code=1,
    )
    live_frame = repo_root / 'state' / 'response_frames' / 'responses.jsonl'

    result = run_isolated_cleanup(repo_root, env, '--archive')

    assert result.returncode == 0, result.stderr or result.stdout
    assert live_frame.read_text(encoding='utf-8') == '{"response_id":"resp-fixture"}\n'
    archives = list((repo_root / '.ollmo_archiv').iterdir())
    assert len(archives) == 1
    assert not (archives[0] / 'state' / 'response_frames' / 'responses.jsonl').exists()
    assert 'skipping archive of state/response_frames' in result.stdout
    assert 'preserving state/response_frames because an evidence-retention preflight failed' in result.stdout


def test_archive_write_snapshots_but_preserves_readiness_registry_under_full_forget(
    tmp_path: Path,
) -> None:
    repo_root, env, args_file = make_isolated_cleanup_repo(
        tmp_path,
        readiness_summary={
            'status': 'unchanged',
            'selected_observation_count': 1,
            'settled_observation_count': 1,
            'registered_observation_count': 1,
            'missing_settled_observation_count': 0,
            'active_observation_count': 0,
            'scan_error_count': 0,
            'hydration_error_count': 0,
            'registry_error_count': 0,
            'appended_record_count': 0,
            'already_present_count': 1,
            'error_count': 0,
        },
    )
    registry = repo_root / 'state' / 'graph_rebase' / 'readiness_observations.jsonl'
    registry_bytes = registry.read_bytes()

    result = run_isolated_cleanup(repo_root, env, '--archive', '--full', '--forget-ghost')

    assert result.returncode == 0, result.stderr or result.stdout
    args = args_file.read_text(encoding='utf-8').splitlines()
    assert '--write' in args
    assert '--check-only' not in args
    assert registry.read_bytes() == registry_bytes
    archives = list((repo_root / '.ollmo_archiv').iterdir())
    assert len(archives) == 1
    archived_registry = (
        archives[0] / 'state' / 'graph_rebase' / 'readiness_observations.jsonl'
    )
    assert archived_registry.read_bytes() == registry_bytes
    manifest = (archives[0] / 'manifest.txt').read_text(encoding='utf-8')
    assert 'protected_graph_rebase_registry_snapshot_path=state/graph_rebase/readiness_observations.jsonl' in manifest
    assert 'graph_rebase_readiness_retention_status=unchanged' in manifest
    assert 'graph_rebase_readiness_registered_observations=1' in manifest
    assert 'graph-rebase readiness registry preserved: 1 / 1 settled observations registered' in result.stdout


def test_docs_do_not_describe_archive_full_as_forget_ghost_equivalent() -> None:
    checked_paths = [
        REPO_ROOT / 'README.md',
        REPO_ROOT / 'OLLMO_FOR_AGENTS.md',
        REPO_ROOT / 'clean_repo_state.sh',
    ]

    for path in checked_paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            normalized = line.lower()
            assert not (
                'archive --full' in normalized
                and 'equivalent' in normalized
                and '--forget-ghost' in normalized
            ), f'{path} still equates archive --full with --forget-ghost: {line}'
