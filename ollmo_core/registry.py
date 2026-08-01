"""Shared runtime registry helpers for Ollmo.

The shared registry is `model_ports.json`. Runtime model instances and agent
entries coexist in the same file, so callers can preserve one while updating
the other.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from helpers.model_capabilities import build_registry_metadata

CONFIG_FILE_NAME = 'model_ports.json'
DEFAULT_REGISTRY_PATH = Path(CONFIG_FILE_NAME)
VOLATILE_REGISTRY_KEYS = {
    'activity',
    'backend_runtime',
    'health',
    'last_error',
    'last_error_at',
    'last_latency_sec',
    'last_success_at',
    'port_listening',
    'process_alive',
    'readiness',
    'runtime_status',
    'session_controls',
}


def resolve_registry_path(path: Path | str | None = None) -> Path:
    if path is None:
        return DEFAULT_REGISTRY_PATH
    return Path(path)


def is_port_listening(port: int, host: str = 'localhost') -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def pid_is_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def enrich_registry_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(entry)
    model_name = str(enriched.get('modelName') or enriched.get('model') or '').strip()
    if model_name:
        enriched['model'] = model_name
    metadata = build_registry_metadata(
        model_name,
        enriched.get('backend'),
        enriched.get('capability'),
        metadata=enriched,
    )
    enriched.update(metadata)
    return enriched


def sanitize_registry_entry_for_persistence(entry: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = enrich_registry_entry(entry)
    for key in VOLATILE_REGISTRY_KEYS:
        sanitized.pop(key, None)
    return sanitized


def _normalise_raw_entries(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [
            enrich_registry_entry(entry)
            for entry in raw
            if isinstance(entry, dict)
        ]
    if isinstance(raw, dict):
        normalised = []
        for key, value in raw.items():
            entry: Dict[str, Any] = {'model': key}
            if isinstance(value, dict):
                entry.update(value)
            else:
                entry['port'] = value
            entry.setdefault('instance_id', key)
            normalised.append(enrich_registry_entry(entry))
        return normalised
    return []


def read_registry_entries(path: Path | str | None = None) -> List[Dict[str, Any]]:
    target = resolve_registry_path(path)
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    return _normalise_raw_entries(raw)


def _merge_preserved_agents(
    runtime_entries: Iterable[Dict[str, Any]],
    existing_entries: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    seen_ids = set()

    for entry in runtime_entries:
        if not isinstance(entry, dict):
            continue
        normalized = sanitize_registry_entry_for_persistence(entry)
        combined.append(normalized)
        instance_id = normalized.get('instance_id')
        if instance_id:
            seen_ids.add(instance_id)

    for entry in existing_entries:
        if not isinstance(entry, dict) or not entry.get('agent'):
            continue
        instance_id = entry.get('instance_id')
        if instance_id and instance_id in seen_ids:
            continue
        combined.append(enrich_registry_entry(entry))

    return combined


def _sync_downstream_integrations(entries: Iterable[Dict[str, Any]]) -> None:
    try:
        from ollmo_integrations.downstream_sync import sync_downstream_integrations
    except ImportError:
        return
    try:
        sync_downstream_integrations(entries)
    except Exception as exc:  # noqa: BLE001
        print(f'⚠️ Downstream config sync skipped: {exc}')


def write_registry_entries(
    entries: Iterable[Dict[str, Any]],
    *,
    path: Path | str | None = None,
    preserve_agents: bool = False,
    sync_external: bool = False,
) -> None:
    target = resolve_registry_path(path)
    existing = read_registry_entries(target) if preserve_agents else []
    payload = (
        _merge_preserved_agents(entries, existing)
        if preserve_agents
        else [
            sanitize_registry_entry_for_persistence(entry)
            for entry in entries
            if isinstance(entry, dict)
        ]
    )
    try:
        target.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except OSError as exc:
        print(f'⚠️ Konnte {target} nicht schreiben: {exc}')
        return
    if sync_external:
        _sync_downstream_integrations(payload)


def filter_active_registry_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized = enrich_registry_entry(entry)
        if normalized.get('agent'):
            filtered.append(normalized)
            continue
        port = normalized.get('port')
        pid = normalized.get('pid')
        instance_id = normalized.get('instance_id')
        if port:
            if is_port_listening(int(port)):
                filtered.append(normalized)
                continue
            print(f"⚠️ Removing inactive instance '{instance_id}' from {resolve_registry_path()}.")
            continue
        if pid and pid_is_running(int(pid)):
            filtered.append(normalized)
            continue
        print(f"⚠️ Removing inactive instance '{instance_id}' from {resolve_registry_path()}.")
    return filtered


def load_registry_entries(
    *,
    prune: bool = True,
    path: Path | str | None = None,
    sync_external: bool = False,
) -> List[Dict[str, Any]]:
    entries = read_registry_entries(path)
    if not prune:
        return entries
    filtered = filter_active_registry_entries(entries)
    if filtered != entries:
        write_registry_entries(
            filtered,
            path=path,
            preserve_agents=False,
            sync_external=sync_external,
        )
    return filtered


def list_runtime_entries(
    *,
    prune: bool = True,
    path: Path | str | None = None,
    sync_external: bool = False,
) -> List[Dict[str, Any]]:
    return [
        entry
        for entry in load_registry_entries(
            prune=prune,
            path=path,
            sync_external=sync_external,
        )
        if not entry.get('agent')
    ]


def list_agent_entries(path: Path | str | None = None) -> List[Dict[str, Any]]:
    return [entry for entry in read_registry_entries(path) if entry.get('agent')]
