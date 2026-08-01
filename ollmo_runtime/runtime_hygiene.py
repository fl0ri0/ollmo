"""Startup/runtime hygiene helpers for stale registry, status, and log cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ollmo_core.registry import DEFAULT_REGISTRY_PATH, list_runtime_entries, write_registry_entries
from ollmo_core.status import DEFAULT_RUNTIME_STATUS_PATH, refresh_runtime_status_entries
from ollmo_runtime.runtime_log_hygiene import (
    collect_active_runtime_log_paths,
    sweep_stale_global_logs,
    sweep_stale_runtime_logs,
)


def cleanup_runtime_hygiene(
    *,
    registry_path: Path | str | None = None,
    status_path: Path | str | None = None,
    log_dir: Path | str = Path('logs'),
    sync_external: bool = False,
    active_global_log_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    resolved_registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    resolved_status_path = Path(status_path) if status_path else DEFAULT_RUNTIME_STATUS_PATH
    resolved_log_dir = Path(log_dir)

    live_entries = list_runtime_entries(
        prune=True,
        path=resolved_registry_path,
        sync_external=sync_external,
    )
    refreshed_status = refresh_runtime_status_entries(
        live_entries,
        path=resolved_status_path,
    )
    log_report = sweep_stale_runtime_logs(
        collect_active_runtime_log_paths(
            live_entries,
            base_dir=resolved_log_dir.parent,
        ),
        log_dir=resolved_log_dir,
        base_dir=resolved_log_dir.parent,
    )
    global_log_report = sweep_stale_global_logs(
        active_global_log_paths,
        log_dir=resolved_log_dir,
        base_dir=resolved_log_dir.parent,
    )

    return {
        'live_instance_count': len(live_entries),
        'runtime_status_count': len(refreshed_status),
        'runtime_archived_count': log_report.get('archived_count', 0),
        'runtime_archived_paths': log_report.get('archived_paths', []),
        'global_archived_count': global_log_report.get('archived_count', 0),
        'global_archived_paths': global_log_report.get('archived_paths', []),
        'archived_count': log_report.get('archived_count', 0) + global_log_report.get('archived_count', 0),
        'archived_paths': [
            *log_report.get('archived_paths', []),
            *global_log_report.get('archived_paths', []),
        ],
    }


def finalize_runtime_shutdown(
    *,
    registry_path: Path | str | None = None,
    status_path: Path | str | None = None,
    log_dir: Path | str = Path('logs'),
    sync_external: bool = False,
    preserve_agents: bool = True,
    active_global_log_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    resolved_registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    resolved_status_path = Path(status_path) if status_path else DEFAULT_RUNTIME_STATUS_PATH
    resolved_log_dir = Path(log_dir)

    write_registry_entries(
        [],
        path=resolved_registry_path,
        preserve_agents=preserve_agents,
        sync_external=sync_external,
    )
    refreshed_status = refresh_runtime_status_entries([], path=resolved_status_path)
    log_report = sweep_stale_runtime_logs(
        [],
        log_dir=resolved_log_dir,
        base_dir=resolved_log_dir.parent,
    )
    global_log_report = sweep_stale_global_logs(
        active_global_log_paths,
        log_dir=resolved_log_dir,
        base_dir=resolved_log_dir.parent,
    )

    return {
        'live_instance_count': 0,
        'runtime_status_count': len(refreshed_status),
        'runtime_archived_count': log_report.get('archived_count', 0),
        'runtime_archived_paths': log_report.get('archived_paths', []),
        'global_archived_count': global_log_report.get('archived_count', 0),
        'global_archived_paths': global_log_report.get('archived_paths', []),
        'archived_count': log_report.get('archived_count', 0) + global_log_report.get('archived_count', 0),
        'archived_paths': [
            *log_report.get('archived_paths', []),
            *global_log_report.get('archived_paths', []),
        ],
    }
