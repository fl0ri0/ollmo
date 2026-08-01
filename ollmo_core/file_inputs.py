"""Shared file and local-path intake helpers for Ollmo."""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse


def file_kind_from_name(filename: str) -> str:
    suffix = Path(filename or '').suffix.lower()
    if suffix in {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm'}:
        return 'audio'
    if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}:
        return 'image'
    if suffix in {
        '.txt',
        '.md',
        '.markdown',
        '.html',
        '.htm',
        '.css',
        '.json',
        '.csv',
        '.log',
        '.py',
        '.js',
        '.mjs',
        '.cjs',
        '.ts',
        '.tsx',
        '.jsx',
        '.yaml',
        '.yml',
        '.xml',
        '.svg',
        '.sh',
        '.sql',
    }:
        return 'text'
    if suffix == '.pdf':
        return 'pdf'
    return 'binary'


def save_upload_to_temp(upload) -> Path:
    suffix = Path(upload.filename or '').suffix
    temp = tempfile.NamedTemporaryFile(prefix='ollmo_upload_', suffix=suffix, delete=False)
    try:
        upload.save(temp.name)
    finally:
        temp.close()
    return Path(temp.name)


def normalize_local_path_input(raw_path: str) -> str:
    candidate = str(raw_path or '').strip().strip('"').strip("'")
    if not candidate:
        return ''
    if candidate.lower().startswith('file://'):
        parsed = urlparse(candidate)
        path_text = unquote(parsed.path or '')
        if parsed.netloc and not path_text.startswith('/'):
            path_text = f'/{parsed.netloc}{path_text}'
        if re.match(r'^/[A-Za-z]:/', path_text):
            path_text = path_text[1:]
        return path_text
    return candidate


def resolve_existing_local_path(raw_path: str) -> Path:
    normalized = normalize_local_path_input(raw_path)
    if not normalized:
        raise ValueError("Parameter 'file_path' is empty.")
    source = Path(normalized).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    else:
        source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f'File was not found: {source}')
    return source


def save_local_path_to_temp(raw_path: str) -> tuple[Path, Path]:
    source = resolve_existing_local_path(raw_path)
    if source.is_dir():
        raise IsADirectoryError(f'Path is a directory, not a file: {source}')
    suffix = source.suffix
    temp = tempfile.NamedTemporaryFile(prefix='ollmo_upload_', suffix=suffix, delete=False)
    temp_path = Path(temp.name)
    temp.close()
    shutil.copy2(source, temp_path)
    return source, temp_path


def expand_local_paths(raw_paths: list[str], *, max_items: int = 1000) -> tuple[list[str], list[dict], bool]:
    resolved_files: list[str] = []
    skipped: list[dict] = []
    seen = set()
    truncated = False
    limit = max(1, min(10_000, int(max_items)))

    def add_file(path_obj: Path) -> None:
        nonlocal truncated
        if truncated:
            return
        canonical = str(path_obj.resolve())
        if canonical in seen:
            return
        if len(resolved_files) >= limit:
            truncated = True
            return
        seen.add(canonical)
        resolved_files.append(canonical)

    for raw in raw_paths:
        text = str(raw or '').strip()
        if not text:
            continue
        try:
            root = resolve_existing_local_path(text)
        except (ValueError, FileNotFoundError, OSError) as exc:
            skipped.append({'path': text, 'reason': str(exc)})
            continue

        if root.is_file():
            add_file(root)
            if truncated:
                break
            continue

        try:
            for child in sorted(root.rglob('*'), key=lambda item: str(item).lower()):
                if not child.is_file():
                    continue
                add_file(child)
                if truncated:
                    break
        except OSError as exc:
            skipped.append({'path': str(root), 'reason': str(exc)})
        if truncated:
            break

    return resolved_files, skipped, truncated


def read_text_file_with_metadata(path: Path, max_bytes: int = 500_000) -> tuple[str, bool, int, int]:
    raw_bytes = path.read_bytes()
    total_bytes = len(raw_bytes)
    data = raw_bytes[:max_bytes]
    truncated = total_bytes > max_bytes
    for encoding in ('utf-8', 'utf-16', 'latin-1'):
        try:
            return data.decode(encoding), truncated, len(data), total_bytes
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace'), truncated, len(data), total_bytes


def read_text_file(path: Path, max_bytes: int = 500_000) -> str:
    text, _truncated, _inline_bytes, _total_bytes = read_text_file_with_metadata(path, max_bytes=max_bytes)
    return text


def to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('utf-8')


def parse_bool(raw_value: Any, *, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def parse_int_with_bounds(
    raw_value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
