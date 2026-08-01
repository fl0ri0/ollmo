"""Scoped internal file tools for Ollmo policy, memory, and artifact files."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

from ollmo_core.transports import is_path_within

DEFAULT_SCOPED_FILE_ROOTS = (Path('state'), Path('artifacts'))
DEFAULT_MAX_READ_BYTES = 1_000_000


def _repo_root(repo_root: Path | str | None = None) -> Path:
    return Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent.parent


def expand_scoped_roots(
    allowed_roots: Optional[Iterable[Path | str]] = None,
    *,
    repo_root: Path | str | None = None,
) -> set[Path]:
    base = _repo_root(repo_root)
    roots: set[Path] = set()
    for raw_root in allowed_roots or DEFAULT_SCOPED_FILE_ROOTS:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = base / root
        roots.add(root.resolve(strict=False))
    return roots


def resolve_scoped_path(
    raw_path: Path | str,
    *,
    allowed_roots: Optional[Iterable[Path | str]] = None,
    repo_root: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    base = _repo_root(repo_root)
    raw_text = str(raw_path or '').strip()
    if not raw_text:
        raise ValueError('path is required')
    path = Path(raw_text).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve(strict=False)
    roots = expand_scoped_roots(allowed_roots, repo_root=base)
    if not any(is_path_within(resolved, root) for root in roots):
        allowed = ', '.join(str(root) for root in sorted(roots, key=str))
        raise ValueError(f'path is outside Ollmo scoped file roots: {resolved} (allowed: {allowed})')
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f'file not found: {resolved}')
    return resolved


def read_scoped_text(
    raw_path: Path | str,
    *,
    allowed_roots: Optional[Iterable[Path | str]] = None,
    repo_root: Path | str | None = None,
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
    encoding: str = 'utf-8',
) -> tuple[str, dict]:
    path = resolve_scoped_path(raw_path, allowed_roots=allowed_roots, repo_root=repo_root, must_exist=True)
    if not path.is_file():
        raise IsADirectoryError(f'path is not a file: {path}')
    raw = path.read_bytes()
    limit = max(1, int(max_bytes or DEFAULT_MAX_READ_BYTES))
    data = raw[:limit]
    text = data.decode(encoding, errors='replace')
    return text, {
        'operation': 'read',
        'path': str(path),
        'bytes': len(data),
        'total_bytes': len(raw),
        'truncated': len(raw) > len(data),
    }


def write_scoped_text(
    raw_path: Path | str,
    text: str,
    *,
    allowed_roots: Optional[Iterable[Path | str]] = None,
    repo_root: Path | str | None = None,
    overwrite: bool = True,
    encoding: str = 'utf-8',
) -> dict:
    path = resolve_scoped_path(raw_path, allowed_roots=allowed_roots, repo_root=repo_root)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f'path is a directory: {path}')
    if path.exists() and not overwrite:
        raise FileExistsError(f'file already exists: {path}')
    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = str(text)
    path.write_text(payload, encoding=encoding)
    return {
        'operation': 'write',
        'path': str(path),
        'bytes': len(payload.encode(encoding)),
        'created': created,
    }


def copy_scoped_file(
    source_path: Path | str,
    destination_path: Path | str,
    *,
    allowed_roots: Optional[Iterable[Path | str]] = None,
    repo_root: Path | str | None = None,
    overwrite: bool = False,
) -> dict:
    source = resolve_scoped_path(source_path, allowed_roots=allowed_roots, repo_root=repo_root, must_exist=True)
    destination = resolve_scoped_path(destination_path, allowed_roots=allowed_roots, repo_root=repo_root)
    if not source.is_file():
        raise IsADirectoryError(f'source is not a file: {source}')
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f'destination is a directory: {destination}')
    if destination.exists() and not overwrite:
        raise FileExistsError(f'file already exists: {destination}')
    created = not destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        'operation': 'copy',
        'source_path': str(source),
        'path': str(destination),
        'bytes': destination.stat().st_size,
        'created': created,
    }


def replace_scoped_text(
    raw_path: Path | str,
    old: str,
    new: str,
    *,
    allowed_roots: Optional[Iterable[Path | str]] = None,
    repo_root: Path | str | None = None,
    count: int = -1,
    encoding: str = 'utf-8',
) -> dict:
    if old == '':
        raise ValueError('old text must not be empty')
    path = resolve_scoped_path(raw_path, allowed_roots=allowed_roots, repo_root=repo_root, must_exist=True)
    if not path.is_file():
        raise IsADirectoryError(f'path is not a file: {path}')
    original = path.read_text(encoding=encoding)
    updated = original.replace(str(old), str(new), int(count))
    replacements = original.count(str(old)) if int(count) < 0 else min(original.count(str(old)), int(count))
    path.write_text(updated, encoding=encoding)
    return {
        'operation': 'replace',
        'path': str(path),
        'replacements': replacements,
        'bytes': len(updated.encode(encoding)),
        'changed': updated != original,
    }
