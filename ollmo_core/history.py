"""Shared infer-history and dedupe-slot helpers for Ollmo."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Optional


def append_infer_history(entry: dict, *, history_path: Path) -> None:
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError as exc:
        logging.warning('Could not write infer history: %s', exc)


def truncate_for_history(text: str, max_chars: int = 120_000) -> str:
    raw = str(text or '')
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + '\n...[truncated]'


def read_infer_history(history_path: Path, limit: int = 200) -> list[dict]:
    if limit <= 0 or not history_path.exists():
        return []
    try:
        lines = history_path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        logging.warning('Could not read infer history: %s', exc)
        return []

    entries: list[dict] = []
    for raw_line in reversed(lines):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            entries.append(item)
        if len(entries) >= limit:
            break
    return entries


def find_cached_pdf_insight(
    *,
    history_path: Path,
    file_sha256: str,
    model_name: str,
    backend: str,
    capability: str,
    prompt: str,
    looks_like_ocr_prompt_echo,
) -> Optional[dict]:
    if not file_sha256:
        return None
    history = read_infer_history(history_path, limit=1000)
    for entry in history:
        if entry.get('status') != 'ok':
            continue
        if entry.get('file_kind') != 'pdf':
            continue
        if entry.get('file_sha256') != file_sha256:
            continue
        if str(entry.get('model') or '') != model_name:
            continue
        if str(entry.get('backend') or '') != backend:
            continue
        if str(entry.get('capability') or '') != capability:
            continue
        if str(entry.get('prompt') or '') != prompt:
            continue
        content_value = str(entry.get('content') or '').strip()
        if not content_value:
            continue
        if looks_like_ocr_prompt_echo(content_value, user_hint=prompt):
            continue
        if str(entry.get('mode') or '').strip() == 'vision_analysis_pdf_scan':
            if int(entry.get('pdf_processed_pages') or 0) == 0:
                continue
        return entry
    return None


def log_pdf_infer_event(
    *,
    history_path: Path,
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    prompt: str,
    file_name: str,
    file_sha256: str,
    status: str,
    mode: Optional[str] = None,
    content: Optional[str] = None,
    error: Optional[str] = None,
    warnings: Optional[list[str]] = None,
    pdf_source: Optional[str] = None,
    pdf_total_pages: Optional[int] = None,
    pdf_processed_pages: Optional[int] = None,
    artifact_path: Optional[str] = None,
) -> None:
    entry = {
        'id': f"infer-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        'timestamp': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
        'instance_id': instance_id,
        'model': model_name,
        'backend': backend,
        'capability': capability,
        'prompt': prompt,
        'file_kind': 'pdf',
        'file_name': file_name,
        'file_sha256': file_sha256,
        'status': status,
        'mode': mode,
        'content': truncate_for_history(content or ''),
        'error': error,
        'warnings': warnings or [],
        'pdf_source': pdf_source,
        'pdf_total_pages': pdf_total_pages,
        'pdf_processed_pages': pdf_processed_pages,
        'artifact_path': artifact_path,
    }
    append_infer_history(entry, history_path=history_path)


def build_infer_dedupe_key(
    *,
    instance_id: str,
    backend: str,
    capability: str,
    model_name: str,
    prompt: str,
    upload,
    local_file_path: str = '',
) -> str:
    upload_name = ''
    upload_len = ''
    if upload is not None:
        upload_name = str(getattr(upload, 'filename', '') or '')
        upload_len = str(getattr(upload, 'content_length', '') or '')
    normalized_local_path = str(local_file_path or '').strip()
    base = '|'.join([
        instance_id.strip(),
        backend.strip(),
        capability.strip(),
        model_name.strip(),
        prompt.strip(),
        upload_name.strip(),
        upload_len.strip(),
        normalized_local_path,
    ])
    return hashlib.sha256(base.encode('utf-8', errors='ignore')).hexdigest()


def acquire_infer_slot(
    key: str,
    *,
    slots: dict[str, float],
    lock: Lock,
    now: Optional[float] = None,
    ttl_sec: int,
) -> bool:
    current = now if now is not None else time.time()
    with lock:
        expired = [k for k, ts in slots.items() if (current - ts) > ttl_sec]
        for slot_key in expired:
            slots.pop(slot_key, None)
        if key in slots:
            return False
        slots[key] = current
        return True


def release_infer_slot(key: Optional[str], *, slots: dict[str, float], lock: Lock) -> None:
    if not key:
        return
    with lock:
        slots.pop(key, None)
