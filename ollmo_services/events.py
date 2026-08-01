"""Unified durable event/history layer for canonical Ollmo flows."""

from __future__ import annotations

import datetime as dt
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_EVENT_LOG_PATH = Path('state/events.jsonl')
_EVENT_LOG_LOCK = threading.Lock()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _truncate(value: Any, max_chars: int = 4000) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + '...[truncated]'
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_truncate(item, max_chars=max_chars) for item in value[:50]]
    if isinstance(value, dict):
        return {str(k): _truncate(v, max_chars=max_chars) for k, v in list(value.items())[:50]}
    return _truncate(str(value), max_chars=max_chars)


def make_event(
    *,
    category: str,
    action: str,
    status: str,
    message: Optional[str] = None,
    **fields: Any,
) -> dict:
    payload = {
        'id': f"event-{uuid.uuid4().hex}",
        'timestamp': _now_iso(),
        'category': str(category).strip(),
        'action': str(action).strip(),
        'status': str(status).strip(),
    }
    if message:
        payload['message'] = _truncate(message, max_chars=1200)
    for key, value in fields.items():
        if value is None or value == '':
            continue
        payload[str(key)] = _truncate(value)
    return payload


def append_event(entry: dict, *, path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_EVENT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    with _EVENT_LOG_LOCK:
        with target.open('a', encoding='utf-8') as handle:
            handle.write(line)


def log_event(
    *,
    category: str,
    action: str,
    status: str,
    path: Path | str | None = None,
    message: Optional[str] = None,
    **fields: Any,
) -> dict:
    entry = make_event(
        category=category,
        action=action,
        status=status,
        message=message,
        **fields,
    )
    append_event(entry, path=path)
    return entry


def read_events(
    *,
    path: Path | str | None = None,
    limit: int = 200,
    category: Optional[str] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict]:
    target = Path(path) if path else DEFAULT_EVENT_LOG_PATH
    if limit <= 0 or not target.exists():
        return []
    lines = target.read_text(encoding='utf-8').splitlines()
    entries: list[dict] = []
    for raw_line in reversed(lines):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if category and str(item.get('category') or '') != category:
            continue
        if action and str(item.get('action') or '') != action:
            continue
        if status and str(item.get('status') or '') != status:
            continue
        entries.append(item)
        if len(entries) >= limit:
            break
    return entries
