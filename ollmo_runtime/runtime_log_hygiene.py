"""Helpers for keeping runtime diagnostic logs clean and clearly archived."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

RUNTIME_LOG_PREFIXES = (
    'mlx_',
    'mlx_vlm_',
    'mlx_audio_',
    'mlx_whisper_',
    'llama_cpp_server_',
    'ollama_server_',
)
LOAD_LOG_TOKEN = '_load_on_'
ARCHIVE_SUBDIR = Path('archive/runtime')
GLOBAL_ARCHIVE_SUBDIR = Path('archive/global')
ARCHIVE_README_NAME = 'README.md'
ARCHIVE_MANIFEST_NAME = 'manifest.jsonl'
PORT_SUFFIX_RE = re.compile(r'_(\d+)\.log$')
LOAD_PORT_RE = re.compile(r'_load_on_(\d+)_')
SAFE_LOG_FRAGMENT_RE = re.compile(r'[^A-Za-z0-9_.-]+')
GLOBAL_LOG_NAMES = (
    'flask_webserver.log',
    'ollama_default_server_11434.log',
)
LEGACY_GLOBAL_LOG_NAMES = (
    'flask_webserver_auto.log',
    'flask_webserver_manual.log',
    'ollmo_webserver.log',
)
ARCHIVE_README = """# Runtime Log Archive

This directory stores archived runtime diagnostic logs that are no longer the active
top-level log for a running backend instance.

Important:
- Canonical conversation history lives in `state/chat_history/`.
- User artifacts live under `artifacts/`.
- Files archived here are stale or superseded runtime diagnostics, not current chat history.
"""
GLOBAL_ARCHIVE_README = """# Global Log Archive

This directory stores archived global service logs that are no longer the active
top-level log for a running Ollmo control-plane service.

Important:
- Canonical conversation history lives in `state/chat_history/`.
- User artifacts live under `artifacts/`.
- Files archived here are stale or superseded global diagnostics, not current chat history.
"""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _absolute_path(path: Path | str, *, base_dir: Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    anchor = base_dir or Path.cwd()
    return anchor / raw


def _archive_root_for(log_path: Path, archive_subdir: Path) -> Path:
    return log_path.parent / archive_subdir


def _ensure_archive_metadata(archive_root: Path, *, readme_text: str) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    readme_path = archive_root / ARCHIVE_README_NAME
    if not readme_path.exists():
        readme_path.write_text(readme_text, encoding='utf-8')


def _dedupe_archive_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    index = 2
    while True:
        candidate = target.with_name(f'{stem}__{index}{suffix}')
        if not candidate.exists():
            return candidate
        index += 1


def is_runtime_diagnostic_log(path: Path | str) -> bool:
    name = Path(path).name
    if not name.endswith('.log'):
        return False
    if LOAD_LOG_TOKEN in name:
        return True
    return any(name.startswith(prefix) for prefix in RUNTIME_LOG_PREFIXES)


def is_global_diagnostic_log(path: Path | str) -> bool:
    name = Path(path).name
    if not name.endswith('.log'):
        return False
    return name in GLOBAL_LOG_NAMES or name in LEGACY_GLOBAL_LOG_NAMES


def _safe_log_fragment(value: Any) -> str:
    fragment = SAFE_LOG_FRAGMENT_RE.sub('_', str(value or '').strip())
    return fragment.strip('._') or 'model'


def infer_runtime_log_path(
    entry: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> Optional[Path]:
    if not isinstance(entry, dict):
        return None

    raw_path = str(entry.get('log') or '').strip()
    if raw_path:
        return _absolute_path(raw_path, base_dir=base_dir)

    backend = str(entry.get('backend') or '').strip().lower()
    instance_id = str(entry.get('instance_id') or '').strip()
    port = entry.get('port')
    if backend != 'ollama' or not instance_id or not port:
        return None

    try:
        port_value = int(port)
    except (TypeError, ValueError):
        return None

    inferred = Path('logs') / f'ollama_server_{_safe_log_fragment(instance_id)}_{port_value}.log'
    return _absolute_path(inferred, base_dir=base_dir)


def runtime_log_port_hint(path: Path | str) -> Optional[int]:
    name = Path(path).name
    matcher = LOAD_PORT_RE.search(name) if LOAD_LOG_TOKEN in name else PORT_SUFFIX_RE.search(name)
    if not matcher:
        return None
    try:
        return int(matcher.group(1))
    except (TypeError, ValueError):
        return None


def collect_active_runtime_log_paths(
    entries: Iterable[dict[str, Any]],
    *,
    base_dir: Path | None = None,
) -> set[str]:
    active: set[str] = set()
    for entry in entries:
        inferred_path = infer_runtime_log_path(entry, base_dir=base_dir)
        if inferred_path is None:
            continue
        active.add(str(inferred_path))
    return active


def _archive_log(
    log_path: Path | str,
    *,
    reason: str,
    label: Optional[str] = None,
    archive_subdir: Path,
    archive_readme: str,
    base_dir: Path | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    absolute_log_path = _absolute_path(log_path, base_dir=base_dir)
    if not absolute_log_path.exists() or not absolute_log_path.is_file():
        return None

    try:
        size_bytes = absolute_log_path.stat().st_size
    except OSError:
        size_bytes = 0

    if size_bytes <= 0:
        absolute_log_path.unlink(missing_ok=True)
        return None

    archive_root = _archive_root_for(absolute_log_path, archive_subdir)
    _ensure_archive_metadata(archive_root, readme_text=archive_readme)
    dated_dir = archive_root / dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    dated_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f'{_now_stamp()}__{reason}__{absolute_log_path.name}'
    archive_path = _dedupe_archive_path(dated_dir / archive_name)
    absolute_log_path.rename(archive_path)

    manifest_entry = {
        'archived_at': _now_iso(),
        'reason': str(reason or '').strip() or 'archived_runtime_log',
        'label': str(label or reason or 'archived_runtime_log').strip(),
        'original_path': str(absolute_log_path),
        'archived_path': str(archive_path),
        'size_bytes': size_bytes,
    }
    if isinstance(metadata, dict) and metadata:
        manifest_entry['metadata'] = metadata

    manifest_path = archive_root / ARCHIVE_MANIFEST_NAME
    with manifest_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')

    return archive_path


def archive_runtime_log(
    log_path: Path | str,
    *,
    reason: str,
    label: Optional[str] = None,
    base_dir: Path | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    return _archive_log(
        log_path,
        reason=reason,
        label=label,
        archive_subdir=ARCHIVE_SUBDIR,
        archive_readme=ARCHIVE_README,
        base_dir=base_dir,
        metadata=metadata,
    )


def archive_global_log(
    log_path: Path | str,
    *,
    reason: str,
    label: Optional[str] = None,
    base_dir: Path | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    return _archive_log(
        log_path,
        reason=reason,
        label=label,
        archive_subdir=GLOBAL_ARCHIVE_SUBDIR,
        archive_readme=GLOBAL_ARCHIVE_README,
        base_dir=base_dir,
        metadata=metadata,
    )


def prepare_clean_runtime_log(
    log_path: Path | str,
    *,
    base_dir: Path | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    absolute_log_path = _absolute_path(log_path, base_dir=base_dir)
    absolute_log_path.parent.mkdir(parents=True, exist_ok=True)
    return archive_runtime_log(
        absolute_log_path,
        reason='superseded_launch',
        label='superseded_runtime_log',
        metadata=metadata,
    )


def prepare_clean_global_log(
    log_path: Path | str,
    *,
    base_dir: Path | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    absolute_log_path = _absolute_path(log_path, base_dir=base_dir)
    absolute_log_path.parent.mkdir(parents=True, exist_ok=True)
    return archive_global_log(
        absolute_log_path,
        reason='superseded_global_session',
        label='superseded_global_log',
        metadata=metadata,
    )


def sweep_stale_runtime_logs(
    active_log_paths: Iterable[Path | str],
    *,
    log_dir: Path | str = Path('logs'),
    base_dir: Path | None = None,
) -> dict[str, Any]:
    target_log_dir = _absolute_path(log_dir, base_dir=base_dir)
    if not target_log_dir.exists():
        return {
            'archived_count': 0,
            'archived_paths': [],
        }

    normalized_active = {
        str(_absolute_path(path, base_dir=base_dir))
        for path in active_log_paths
        if str(path or '').strip()
    }
    archived_paths: list[str] = []

    for candidate in sorted(target_log_dir.glob('*.log')):
        if not is_runtime_diagnostic_log(candidate):
            continue
        if str(candidate) in normalized_active:
            continue
        archived = archive_runtime_log(
            candidate,
            reason='stale_runtime_log',
            label='stale_runtime_log',
            base_dir=base_dir,
        )
        if archived is not None:
            archived_paths.append(str(archived))

    return {
        'archived_count': len(archived_paths),
        'archived_paths': archived_paths,
    }


def sweep_stale_global_logs(
    active_log_paths: Iterable[Path | str],
    *,
    log_dir: Path | str = Path('logs'),
    base_dir: Path | None = None,
) -> dict[str, Any]:
    target_log_dir = _absolute_path(log_dir, base_dir=base_dir)
    if not target_log_dir.exists():
        return {
            'archived_count': 0,
            'archived_paths': [],
        }

    normalized_active = {
        str(_absolute_path(path, base_dir=base_dir))
        for path in active_log_paths
        if str(path or '').strip()
    }
    archived_paths: list[str] = []

    for candidate in sorted(target_log_dir.glob('*.log')):
        if not is_global_diagnostic_log(candidate):
            continue
        if str(candidate) in normalized_active:
            continue
        if candidate.name in GLOBAL_LOG_NAMES:
            reason = 'stale_global_log'
            label = 'stale_global_log'
        else:
            reason = 'legacy_global_log'
            label = 'legacy_global_log'
        archived = archive_global_log(
            candidate,
            reason=reason,
            label=label,
            base_dir=base_dir,
        )
        if archived is not None:
            archived_paths.append(str(archived))

    return {
        'archived_count': len(archived_paths),
        'archived_paths': archived_paths,
    }
