"""Helpers for resetting persisted Ghost learning state."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_EVENT_LOG_PATH = Path('state/events.jsonl')
DEFAULT_GHOST_LEARNING_ARCHIVE_ROOT = Path('state/ghost_learning_archives')
DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH = Path(
    'state/graph_rebase/readiness_observations.jsonl'
)
DEFAULT_RESPONSE_FRAME_LEDGER_PATH = Path('state/response_frames/responses.jsonl')
DEFAULT_RUNTIME_LOG_PATH = Path('logs/flask_webserver.log')
DEFAULT_RETIRED_COMPILED_MEMORY_PATH = Path('state/ghost_compiled_memory.json')
DEFAULT_RETIRED_COMPILED_MEMORY_MARKDOWN_PATH = Path('state/ghost_compiled_memory.md')
DEFAULT_SELF_LEARNING_DIR = Path('state/self_learning')


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _timestamp_slug(moment: dt.datetime | None = None) -> str:
    return (moment or _now_utc()).strftime('%Y%m%dT%H%M%SZ')


def _archive_relative_path(source: Path) -> Path:
    parts = source.parts
    for anchor_name in ('state', 'logs'):
        if anchor_name in parts:
            anchor = parts.index(anchor_name)
            return Path(*parts[anchor:])
    return Path(source.name)


def _archive_file(source: Path, archive_root: Path) -> str | None:
    if not source.exists():
        return None
    destination = archive_root / _archive_relative_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return str(destination)


def reset_ghost_learning_state(
    *,
    archive_root: Path | str | None = None,
    event_log_path: Path | str | None = None,
    runtime_log_path: Path | str | None = None,
    compiled_memory_path: Path | str | None = None,
    compiled_memory_markdown_path: Path | str | None = None,
    self_learning_dir_path: Path | str | None = None,
    response_frame_ledger_path: Path | str | None = None,
    graph_rebase_readiness_registry_path: Path | str | None = None,
) -> dict[str, Any]:
    moment = _now_utc()
    archive_dir = Path(archive_root) if archive_root else DEFAULT_GHOST_LEARNING_ARCHIVE_ROOT / _timestamp_slug(moment)
    event_log = Path(event_log_path) if event_log_path else DEFAULT_EVENT_LOG_PATH
    runtime_log = Path(runtime_log_path) if runtime_log_path else DEFAULT_RUNTIME_LOG_PATH
    retired_route_memory_json = event_log.parent / 'ghost_learned_policy.json'
    retired_route_memory_markdown = event_log.parent / 'ghost_learned_policy.md'
    compiled_memory = Path(compiled_memory_path) if compiled_memory_path else DEFAULT_RETIRED_COMPILED_MEMORY_PATH
    compiled_memory_markdown = (
        Path(compiled_memory_markdown_path)
        if compiled_memory_markdown_path
        else DEFAULT_RETIRED_COMPILED_MEMORY_MARKDOWN_PATH
    )
    self_learning_dir = Path(self_learning_dir_path) if self_learning_dir_path else DEFAULT_SELF_LEARNING_DIR
    response_frame_ledger = (
        Path(response_frame_ledger_path)
        if response_frame_ledger_path
        else DEFAULT_RESPONSE_FRAME_LEDGER_PATH
    )
    graph_rebase_readiness_registry = (
        Path(graph_rebase_readiness_registry_path)
        if graph_rebase_readiness_registry_path
        else DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH
    )

    archived_files: dict[str, str] = {}
    for label, source in (
        ('event_log', event_log),
        ('runtime_log', runtime_log),
        ('retired_route_memory_json', retired_route_memory_json),
        ('retired_route_memory_markdown', retired_route_memory_markdown),
        ('compiled_memory_json', compiled_memory),
        ('compiled_memory_markdown', compiled_memory_markdown),
        ('self_learning_dir', self_learning_dir),
    ):
        archived = _archive_file(source, archive_dir)
        if archived:
            archived_files[label] = archived

    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'kind': 'ollmo.ghost_learning_reset',
        'reset_at': moment.isoformat().replace('+00:00', 'Z'),
        'repo_relative_archive': str(archive_dir),
        'archived_files': archived_files,
        'preserved_paths': {
            'response_frame_ledger': str(response_frame_ledger),
            'graph_rebase_readiness_registry': str(
                graph_rebase_readiness_registry
            ),
        },
    }
    (archive_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    event_log.parent.mkdir(parents=True, exist_ok=True)
    event_log.write_text('', encoding='utf-8')
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_log.write_text('', encoding='utf-8')

    for stale_path in (retired_route_memory_json, retired_route_memory_markdown):
        if stale_path.exists():
            stale_path.unlink()

    for stale_path in (compiled_memory, compiled_memory_markdown):
        if stale_path.exists():
            stale_path.unlink()

    return {
        'ok': True,
        'reset_at': manifest['reset_at'],
        'archive_dir': str(archive_dir),
        'archived_files': archived_files,
        'preserved_paths': manifest['preserved_paths'],
        'reset_paths': {
            'event_log': str(event_log),
            'runtime_log': str(runtime_log),
        },
    }
