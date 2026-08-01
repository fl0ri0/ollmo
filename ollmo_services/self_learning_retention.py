"""Retention manifest helpers for self-learning evidence sidecars."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

DEFAULT_SELF_LEARNING_DIR = Path('state/self_learning')
DEFAULT_RESPONSE_FRAMES_DIR = Path('state/response_frames')
DEFAULT_RETENTION_MANIFEST = DEFAULT_SELF_LEARNING_DIR / 'retention_manifest.json'
DEFAULT_RETAINED_SIDECARS_DIR = DEFAULT_SELF_LEARNING_DIR / 'retained_sidecars'
RETENTION_MANIFEST_KIND = 'ollmo.self_learning_retention_manifest'

_REF_KEYS = {
    'snapshot_ref',
    'full_snapshot_ref',
    'review_snapshot_ref',
    'runtime_snapshot_ref',
    'late_fill_snapshot_ref',
    'work_tree_snapshot_ref',
    'request_phase_graph_snapshot_ref',
    'graph_snapshot_ref',
}
_CONTAINER_KEYS = {
    'sidecar_manifest',
    'items',
    'child_refs',
    'children',
    'refs',
}


def _now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if item not in (None, '', [], {})}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value if item not in (None, '', [], {})]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_snapshot_payload_file(path: Path) -> str:
    """Hash CAS payload bytes while ignoring at most one terminal newline."""

    digest = hashlib.sha256()
    trailing_byte = b''
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            buffered = trailing_byte + chunk
            if len(buffered) > 1:
                digest.update(buffered[:-1])
                trailing_byte = buffered[-1:]
            else:
                trailing_byte = buffered
    if trailing_byte != b'\n':
        digest.update(trailing_byte)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_self_learning_files(self_learning_dir: Path) -> Iterable[Path]:
    if not self_learning_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(self_learning_dir.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.jsonl'}:
            continue
        if DEFAULT_RETAINED_SIDECARS_DIR.name in path.parts:
            continue
        files.append(path)
    return files


def _iter_json_payloads(path: Path) -> Iterable[Any]:
    if path.suffix == '.jsonl':
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except OSError:
            return
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {'_source_line': line_number, 'payload': payload}
        return
    payload = _read_json(path)
    if payload is not None:
        yield {'payload': payload}


def _looks_like_snapshot_ref(value: Mapping[str, Any]) -> bool:
    raw_path = _clean_text(value.get('path'))
    if not raw_path:
        return False
    if value.get('sha256') not in (None, ''):
        return True
    normalized = raw_path.replace('\\', '/')
    return (
        normalized.startswith('snapshots/')
        or '/snapshots/' in normalized
        or normalized.startswith('state/response_frames/')
    )


def _iter_snapshot_refs(value: Any, *, max_depth: int = 12) -> Iterable[Mapping[str, Any]]:
    if max_depth <= 0:
        return
    if isinstance(value, Mapping):
        if _looks_like_snapshot_ref(value):
            yield value
        for key, item in value.items():
            key_text = _clean_text(key)
            if (
                key_text in _REF_KEYS
                or key_text.endswith('_snapshot_ref')
                or key_text in _CONTAINER_KEYS
                or isinstance(item, (Mapping, list, tuple))
            ):
                yield from _iter_snapshot_refs(item, max_depth=max_depth - 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_snapshot_refs(item, max_depth=max_depth - 1)


def _normalize_ref_path(
    ref: Mapping[str, Any],
    *,
    response_frames_dir: Path,
) -> tuple[Optional[str], Optional[Path], Optional[dict[str, Any]]]:
    raw_path = _clean_text(ref.get('path')).replace('\\', '/')
    if not raw_path:
        return None, None, {'path': raw_path, 'reason': 'missing_path'}
    response_root = response_frames_dir.resolve(strict=False)
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        if not _is_relative_to(resolved, response_root):
            return None, None, {'path': raw_path, 'reason': 'absolute_path_outside_response_frames'}
        return str(resolved.relative_to(response_root)).replace('\\', '/'), resolved, None
    parts = [part for part in candidate.parts if part not in ('', '.')]
    if any(part == '..' for part in parts):
        return None, None, {'path': raw_path, 'reason': 'relative_path_escapes_response_frames'}
    if parts[:2] == ['state', 'response_frames']:
        parts = parts[2:]
    elif parts[:1] == ['response_frames']:
        parts = parts[1:]
    rel_path = Path(*parts) if parts else Path(raw_path)
    resolved = (response_root / rel_path).resolve(strict=False)
    if not _is_relative_to(resolved, response_root):
        return None, None, {'path': raw_path, 'reason': 'resolved_path_outside_response_frames'}
    return str(rel_path).replace('\\', '/'), resolved, None


def _record_for_ref(
    ref: Mapping[str, Any],
    *,
    rel_path: str,
    source_file: Path,
    self_learning_dir: Path,
    path: Path,
    exists: bool,
    retained_sidecars_dir: Path,
    storage_source: str,
) -> dict[str, Any]:
    expected_sha = _clean_text(ref.get('sha256'))
    record = {
        'path': rel_path,
        'source_file': str(source_file),
        'source_file_relative': str(source_file.relative_to(self_learning_dir)) if _is_relative_to(source_file, self_learning_dir) else str(source_file),
        'sha256': expected_sha or None,
        'storage_source': storage_source,
    }
    if exists:
        actual_sha = _sha256_snapshot_payload_file(path)
        record['actual_sha256'] = actual_sha
        if expected_sha and expected_sha != actual_sha:
            record['sha256_mismatch'] = True
        record['retained_copy_path'] = str(retained_sidecars_dir / rel_path)
    return _json_safe(record)


def collect_self_learning_retention_roots(
    self_learning_dir: Path | str = DEFAULT_SELF_LEARNING_DIR,
    response_frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    *,
    retained_sidecars_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe manifest for response-frame sidecars referenced by active learning."""

    learning_root = Path(self_learning_dir)
    frames_root = Path(response_frames_dir)
    retained_root = Path(retained_sidecars_dir) if retained_sidecars_dir else learning_root / 'retained_sidecars'
    files = list(_iter_self_learning_files(learning_root))
    retained_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    missing_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    unsafe_by_path: dict[str, dict[str, Any]] = {}
    retained_copy_sha256_mismatches: dict[tuple[str, str], dict[str, Any]] = {}
    discovered_ref_count = 0

    for source_file in files:
        for wrapper in _iter_json_payloads(source_file):
            payload = wrapper.get('payload') if isinstance(wrapper, Mapping) else wrapper
            for ref in _iter_snapshot_refs(payload):
                discovered_ref_count += 1
                rel_path, resolved, unsafe = _normalize_ref_path(ref, response_frames_dir=frames_root)
                if unsafe is not None:
                    unsafe_record = {
                        **unsafe,
                        'source_file': str(source_file),
                        'source_file_relative': str(source_file.relative_to(learning_root)) if _is_relative_to(source_file, learning_root) else str(source_file),
                        'sha256': _clean_text(ref.get('sha256')) or None,
                    }
                    unsafe_by_path[_clean_text(unsafe.get('path')) or f'unsafe-{len(unsafe_by_path) + 1}'] = _json_safe(unsafe_record)
                    continue
                if not rel_path or resolved is None:
                    continue
                identity = (rel_path, _clean_text(ref.get('sha256')))
                source_exists = resolved.exists() and resolved.is_file()
                retained_copy = (retained_root / rel_path).resolve(strict=False)
                retained_root_resolved = retained_root.resolve(strict=False)
                retained_copy_is_safe = _is_relative_to(retained_copy, retained_root_resolved)
                retained_copy_exists = (
                    retained_copy_is_safe
                    and retained_copy.exists()
                    and retained_copy.is_file()
                )
                storage_path = resolved if source_exists else retained_copy
                storage_source = 'response_frames' if source_exists else 'retained_copy'
                exists = source_exists or retained_copy_exists
                record = _record_for_ref(
                    ref,
                    rel_path=rel_path,
                    source_file=source_file,
                    self_learning_dir=learning_root,
                    path=storage_path,
                    exists=exists,
                    retained_sidecars_dir=retained_root,
                    storage_source=storage_source,
                )
                if exists and record.get('sha256_mismatch') is not True:
                    retained_by_identity.setdefault(identity, record)
                else:
                    if exists and record.get('sha256_mismatch') is True and storage_source == 'retained_copy':
                        record['reason'] = 'retained_copy_sha256_mismatch'
                        retained_copy_sha256_mismatches.setdefault(identity, record)
                    missing_by_identity.setdefault(identity, record)

    retained = sorted(retained_by_identity.values(), key=lambda item: (item.get('path', ''), item.get('sha256', '')))
    missing = sorted(missing_by_identity.values(), key=lambda item: (item.get('path', ''), item.get('sha256', '')))
    unsafe = sorted(unsafe_by_path.values(), key=lambda item: item.get('path', ''))
    retained_copy_mismatches = sorted(
        retained_copy_sha256_mismatches.values(),
        key=lambda item: (item.get('path', ''), item.get('sha256', '')),
    )
    if not discovered_ref_count:
        status = 'empty'
    elif missing or unsafe:
        status = 'partial'
    else:
        status = 'complete'
    manifest = {
        'kind': RETENTION_MANIFEST_KIND,
        'generated_at': _now_iso_utc(),
        'self_learning_dir': str(learning_root),
        'response_frames_dir': str(frames_root),
        'retained_sidecars_dir': str(retained_root),
        'self_learning_files_scanned': [str(path) for path in files],
        'self_learning_file_count': len(files),
        'discovered_ref_count': discovered_ref_count,
        'retained_response_frame_sidecars': retained,
        'missing_response_frame_sidecars': missing,
        'external_or_unsafe_refs': unsafe,
        'retention_root_count': len(retained),
        'retained_sidecar_count': len(retained),
        'source_sidecar_count': len(
            [item for item in retained if item.get('storage_source') == 'response_frames']
        ),
        'retained_copy_sidecar_count': len(
            [item for item in retained if item.get('storage_source') == 'retained_copy']
        ),
        'retained_copy_sha256_mismatches': retained_copy_mismatches,
        'retained_copy_sha256_mismatch_count': len(retained_copy_mismatches),
        'missing_sidecar_count': len(missing),
        'external_or_unsafe_ref_count': len(unsafe),
        'status': status,
    }
    return _json_safe(manifest)


def copy_retained_sidecars(
    manifest: Mapping[str, Any],
    *,
    response_frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    retained_sidecars_dir: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy retained sidecars into self-learning-owned storage and return a copy manifest."""

    frames_root = Path(response_frames_dir)
    retained_root = (
        Path(retained_sidecars_dir)
        if retained_sidecars_dir
        else Path(_clean_text(manifest.get('retained_sidecars_dir')) or DEFAULT_RETAINED_SIDECARS_DIR)
    )
    copied: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in manifest.get('retained_response_frame_sidecars') or []:
        if not isinstance(item, Mapping):
            continue
        rel_path = _clean_text(item.get('path')).replace('\\', '/')
        if not rel_path:
            continue
        source = (frames_root / rel_path).resolve(strict=False)
        try:
            source.relative_to(frames_root.resolve(strict=False))
        except ValueError:
            errors.append({'path': rel_path, 'error': 'source_outside_response_frames'})
            continue
        destination = retained_root / rel_path
        expected_sha = _clean_text(item.get('actual_sha256') or item.get('sha256'))
        already_retained = destination.exists() and destination.is_file()
        if already_retained and expected_sha:
            already_retained = _sha256_snapshot_payload_file(destination) == expected_sha
        if not dry_run and not already_retained:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied.append(
            {
                'path': rel_path,
                'source_path': str(source),
                'retained_copy_path': str(destination),
                'sha256': expected_sha or None,
                'copy_status': 'already_retained' if already_retained else 'would_copy' if dry_run else 'copied',
            }
        )
    return _json_safe(
        {
            **dict(manifest),
            'retained_copy_count': len(copied),
            'retained_copies': copied,
            'retained_copy_errors': errors,
        }
    )


def write_retention_manifest(manifest: Mapping[str, Any], path: Path | str = DEFAULT_RETENTION_MANIFEST) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return target


def load_retention_manifest(path: Path | str = DEFAULT_RETENTION_MANIFEST) -> dict[str, Any]:
    target = Path(path)
    payload = _read_json(target)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _retained_copy_path(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> Path | None:
    raw_copy_path = _clean_text(record.get('retained_copy_path'))
    if raw_copy_path:
        return Path(raw_copy_path)
    rel_path = _clean_text(record.get('path')).replace('\\', '/')
    if not rel_path:
        return None
    retained_root = Path(_clean_text(manifest.get('retained_sidecars_dir')) or DEFAULT_RETAINED_SIDECARS_DIR)
    return retained_root / rel_path


def _retention_storage_diagnostics(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | str,
) -> dict[str, Any]:
    retained_records = [
        item
        for item in (manifest.get('retained_response_frame_sidecars') or [])
        if isinstance(item, Mapping)
    ]
    target_manifest = Path(manifest_path)
    manifest_exists = target_manifest.exists() and target_manifest.is_file()
    missing_copies: list[dict[str, Any]] = []
    sha_mismatches: list[dict[str, Any]] = []
    for record in retained_records:
        copy_path = _retained_copy_path(record, manifest)
        rel_path = _clean_text(record.get('path'))
        if copy_path is None:
            missing_copies.append({'path': rel_path or 'unknown', 'reason': 'missing_retained_copy_path'})
            continue
        if not copy_path.exists() or not copy_path.is_file():
            missing_copies.append(
                {
                    'path': rel_path or str(copy_path),
                    'retained_copy_path': str(copy_path),
                    'reason': 'retained_copy_missing',
                }
            )
            continue
        expected_sha = _clean_text(record.get('actual_sha256') or record.get('sha256'))
        if expected_sha:
            actual_sha = _sha256_snapshot_payload_file(copy_path)
            if actual_sha != expected_sha:
                sha_mismatches.append(
                    {
                        'path': rel_path or str(copy_path),
                        'retained_copy_path': str(copy_path),
                        'sha256': expected_sha,
                        'actual_sha256': actual_sha,
                        'reason': 'retained_copy_sha256_mismatch',
                    }
                )
    if not retained_records:
        storage_status = 'empty'
    elif not manifest_exists:
        storage_status = 'missing_manifest'
    elif missing_copies or sha_mismatches:
        storage_status = 'missing_retained_copies'
    else:
        storage_status = 'complete'
    return _json_safe(
        {
            'storage_status': storage_status,
            'manifest_exists': manifest_exists,
            'retained_copy_missing_count': len(missing_copies),
            'retained_copy_sha256_mismatch_count': len(sha_mismatches),
            'missing_retained_copies': missing_copies,
            'retained_copy_sha256_mismatches': sha_mismatches,
        }
    )


def retention_summary(manifest: Mapping[str, Any], *, manifest_path: Path | str = DEFAULT_RETENTION_MANIFEST) -> dict[str, Any]:
    source_status = _clean_text(manifest.get('status')) or 'empty'
    storage = _retention_storage_diagnostics(manifest, manifest_path=manifest_path)
    retained_count = manifest.get('retained_sidecar_count') or len(manifest.get('retained_response_frame_sidecars') or [])
    status = source_status
    if source_status == 'complete' and retained_count and storage.get('storage_status') != 'complete':
        status = 'partial'
    return _json_safe(
        {
            'status': status,
            'source_status': source_status,
            'storage_status': storage.get('storage_status'),
            'manifest_exists': storage.get('manifest_exists'),
            'retained_sidecar_count': retained_count,
            'missing_sidecar_count': manifest.get('missing_sidecar_count') or len(manifest.get('missing_response_frame_sidecars') or []),
            'external_or_unsafe_ref_count': manifest.get('external_or_unsafe_ref_count') or len(manifest.get('external_or_unsafe_refs') or []),
            'manifest_path': str(manifest_path),
            'retained_sidecars_dir': manifest.get('retained_sidecars_dir'),
            'retained_copy_missing_count': storage.get('retained_copy_missing_count'),
            'retained_copy_sha256_mismatch_count': storage.get('retained_copy_sha256_mismatch_count'),
            'missing_response_frame_sidecars': manifest.get('missing_response_frame_sidecars') or [],
            'external_or_unsafe_refs': manifest.get('external_or_unsafe_refs') or [],
            'missing_retained_copies': storage.get('missing_retained_copies') or [],
            'retained_copy_sha256_mismatches': storage.get('retained_copy_sha256_mismatches') or [],
        }
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect self-learning response-frame sidecar retention roots.')
    parser.add_argument('--self-learning-dir', default=str(DEFAULT_SELF_LEARNING_DIR))
    parser.add_argument('--response-frames-dir', default=str(DEFAULT_RESPONSE_FRAMES_DIR))
    parser.add_argument('--retained-sidecars-dir', default=None)
    parser.add_argument('--write-manifest', default=None)
    parser.add_argument('--copy-retained', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--shell-summary', action='store_true')
    return parser.parse_args(argv)


def _main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    manifest = collect_self_learning_retention_roots(
        self_learning_dir=Path(args.self_learning_dir),
        response_frames_dir=Path(args.response_frames_dir),
        retained_sidecars_dir=Path(args.retained_sidecars_dir) if args.retained_sidecars_dir else None,
    )
    if args.copy_retained:
        manifest = copy_retained_sidecars(
            manifest,
            response_frames_dir=Path(args.response_frames_dir),
            retained_sidecars_dir=Path(args.retained_sidecars_dir) if args.retained_sidecars_dir else None,
            dry_run=args.dry_run,
        )
    if args.write_manifest and not args.dry_run:
        write_retention_manifest(manifest, args.write_manifest)
    if args.shell_summary:
        print(f"status={manifest.get('status') or 'empty'}")
        print(f"retained_sidecar_count={manifest.get('retained_sidecar_count') or 0}")
        print(f"missing_sidecar_count={manifest.get('missing_sidecar_count') or 0}")
        print(f"external_or_unsafe_ref_count={manifest.get('external_or_unsafe_ref_count') or 0}")
        print(f"retained_copy_count={manifest.get('retained_copy_count') or 0}")
    elif args.json or not args.verify:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if args.verify and manifest.get('status') == 'partial':
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
