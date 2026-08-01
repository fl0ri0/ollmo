"""Dynamic runtime status registry that sits beside the stable model registry."""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import threading
from pathlib import Path
from typing import Any, Optional
from urllib import error as _urlerror
from urllib import request as _urlrequest

from helpers.ollama_client import fetch_ollama_ps
from helpers.model_capabilities import build_ollama_ps_summary, normalize_capability
from ollmo_core.runtime_liveness import (
    DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC,
    format_runtime_timestamp,
    runtime_failure_error_class,
    runtime_utc_now,
)

DEFAULT_RUNTIME_STATUS_PATH = Path('state/runtime_status.json')
_RUNTIME_STATUS_LOCK = threading.Lock()
OLLAMA_PS_TIMEOUT_SECONDS = 2


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def default_runtime_status() -> dict:
    return {
        'schema_version': 1,
        'updated_at': _now_iso(),
        'instances': {},
    }


def read_runtime_status(path: Path | str | None = None) -> dict:
    target = Path(path) if path else DEFAULT_RUNTIME_STATUS_PATH
    if not target.exists():
        return default_runtime_status()
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except Exception:
        return default_runtime_status()
    if not isinstance(payload, dict):
        return default_runtime_status()
    base = default_runtime_status()
    base.update({k: v for k, v in payload.items() if k != 'instances'})
    raw_instances = payload.get('instances') if isinstance(payload.get('instances'), dict) else {}
    base['instances'] = {
        str(key): dict(value)
        for key, value in raw_instances.items()
        if isinstance(value, dict)
    }
    return base


def write_runtime_status(payload: dict, *, path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_RUNTIME_STATUS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    with _RUNTIME_STATUS_LOCK:
        target.write_text(serialized, encoding='utf-8')


def _port_listening(port: Any, host: str = '127.0.0.1') -> Optional[bool]:
    try:
        numeric_port = int(port)
    except (TypeError, ValueError):
        return None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, numeric_port)) == 0
    except OSError:
        return False


def _fetch_json(url: str, *, timeout: float = 2.0) -> Optional[dict[str, Any]]:
    try:
        with _urlrequest.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except (_urlerror.URLError, _urlerror.HTTPError, TimeoutError, ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _process_alive(pid: Any) -> Optional[bool]:
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return None
    if numeric_pid <= 0:
        return None
    try:
        os.kill(numeric_pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _base_instance_status(instance: dict) -> dict:
    return {
        'instance_id': instance.get('instance_id'),
        'model': instance.get('model') or instance.get('modelName'),
        'backend': instance.get('backend'),
        'capability': instance.get('capability'),
        'port': instance.get('port'),
        'pid': instance.get('pid'),
    }


def _merge_instance_metadata(target: dict, instance: Optional[dict]) -> dict:
    if not instance:
        return target
    for key, value in _base_instance_status(instance).items():
        if value is not None and value != '':
            target[key] = value
    for key in (
        'backend_package',
        'backend_contract',
        'mlx_server',
        'log',
        'model_path',
        'request_model',
        'start_source',
        'start_audit',
    ):
        value = instance.get(key)
        if value is not None and value != '':
            target[key] = value
    launch_defaults = instance.get('launch_defaults')
    if isinstance(launch_defaults, dict) and launch_defaults:
        target['launch_defaults'] = dict(launch_defaults)
    return target


def _fetch_backend_runtime_metadata(instance: Optional[dict]) -> dict:
    if not isinstance(instance, dict):
        return {}
    backend = str(instance.get('backend') or '').strip().lower()
    if backend == 'llama_cpp':
        port = instance.get('port')
        try:
            numeric_port = int(port)
        except (TypeError, ValueError):
            numeric_port = None
        base_url = f'http://127.0.0.1:{numeric_port}' if numeric_port is not None else None
        runtime: dict[str, Any] = {
            'source': 'llama_cpp_runtime_registry',
            'backend_package': str(instance.get('backend_package') or '').strip() or 'llama_cpp',
            'backend_contract': str(instance.get('backend_contract') or '').strip() or 'llama.cpp.server',
            'native_base_url': base_url,
            'log': str(instance.get('log') or '').strip() or None,
            'model_path': str(instance.get('model_path') or '').strip() or None,
            'request_model': str(instance.get('request_model') or '').strip() or None,
            'hf_repo': str(instance.get('hf_repo') or '').strip() or None,
            'hf_file': str(instance.get('hf_file') or '').strip() or None,
            'models_url': f'{base_url}/v1/models' if base_url else None,
            'chat_completions_url': f'{base_url}/v1/chat/completions' if base_url else None,
            'request_model_strategy': 'model_bound_at_launch',
        }
        live_payload = _fetch_json(runtime['models_url'], timeout=2.0) if runtime.get('models_url') else None
        if isinstance(live_payload, dict):
            data_items = live_payload.get('data') if isinstance(live_payload.get('data'), list) else []
            live_ids = [
                str(item.get('id') or '').strip()
                for item in data_items
                if isinstance(item, dict) and str(item.get('id') or '').strip()
            ]
            if live_ids:
                runtime['source'] = 'llama_cpp_live_server'
                runtime['live_model_ids'] = live_ids
                runtime['active_model_id'] = live_ids[0]
        if normalize_capability(instance.get('capability')) == 'embedding':
            runtime['embeddings_url'] = f'{base_url}/v1/embeddings' if base_url else None
        return {
            key: value
            for key, value in runtime.items()
            if value not in (None, '', [])
        }
    if backend == 'mlx':
        port = instance.get('port')
        try:
            numeric_port = int(port)
        except (TypeError, ValueError):
            numeric_port = None

        base_url = f'http://127.0.0.1:{numeric_port}' if numeric_port is not None else None
        backend_package = str(instance.get('backend_package') or '').strip() or None
        backend_contract = str(instance.get('backend_contract') or '').strip() or None
        server_kind = str(instance.get('mlx_server') or '').strip() or None
        runtime: dict[str, Any] = {
            'source': 'mlx_runtime_registry',
            'backend_package': backend_package,
            'backend_contract': backend_contract,
            'server_kind': server_kind,
            'native_base_url': base_url,
            'log': str(instance.get('log') or '').strip() or None,
            'model_path': str(instance.get('model_path') or '').strip() or None,
            'request_model': str(instance.get('request_model') or '').strip() or None,
            'launch_defaults': instance.get('launch_defaults') if isinstance(instance.get('launch_defaults'), dict) else None,
        }

        if backend_package == 'mlx_lm':
            runtime.update(
                {
                    'request_model_strategy': 'model_bound_at_launch',
                    'runtime_knobs': ['prefill_step_size', 'prompt_cache_size', 'prompt_cache_bytes'],
                    'cache_features': ['rotating_kv_cache', 'prompt_cache'],
                }
            )
        elif backend_package == 'mlx_vlm':
            runtime.update(
                {
                    'health_url': f'{base_url}/health' if base_url else None,
                    'models_url': f'{base_url}/v1/models' if base_url else None,
                    'responses_url': f'{base_url}/v1/responses' if base_url else None,
                    'chat_completions_url': f'{base_url}/v1/chat/completions' if base_url else None,
                    'unload_url': f'{base_url}/unload' if base_url else None,
                    'lazy_loads_model': True,
                    'single_loaded_model': True,
                    'supports_unload': True,
                    'runtime_knobs': [
                        'prefill_step_size',
                        'kv_bits',
                        'kv_quant_scheme',
                        'kv_group_size',
                        'max_kv_size',
                        'quantized_kv_start',
                        'token_queue_timeout',
                    ],
                }
            )
        elif backend_package == 'mlx_audio':
            runtime.update(
                {
                    'health_url': f'{base_url}/health' if base_url else None,
                    'speech_url': f'{base_url}/v1/audio/speech' if base_url else None,
                    'transcriptions_url': f'{base_url}/v1/audio/transcriptions' if base_url else None,
                    'request_model_strategy': 'model_selected_per_request',
                    'lazy_loads_model': True,
                }
            )
        elif backend_package == 'mlx_whisper_shim':
            runtime.update(
                {
                    'health_url': f'{base_url}/healthz' if base_url else None,
                    'transcriptions_url': f'{base_url}/v1/audio/transcriptions' if base_url else None,
                    'legacy_transcribe_url': f'{base_url}/api/transcribe' if base_url else None,
                    'shim_kind': 'local_http_compatibility_shim',
                    'request_model_strategy': 'model_bound_at_launch',
                }
            )

        return {
            key: value
            for key, value in runtime.items()
            if value not in (None, '', [])
        }
    if backend != 'ollama':
        return {}
    port = instance.get('port')
    model_name = str(instance.get('model') or instance.get('modelName') or '').strip()
    try:
        numeric_port = int(port)
    except (TypeError, ValueError):
        return {}
    try:
        payload = fetch_ollama_ps(
            numeric_port,
            timeout=OLLAMA_PS_TIMEOUT_SECONDS,
        )
    except Exception:
        return {}
    summary = build_ollama_ps_summary(payload, model_name=model_name)
    if summary:
        summary['model_active'] = True
        return summary
    return {
        'source': 'ollama_api_ps',
        'model': model_name,
        'model_active': False,
    }


def _backend_runtime_indicates_idle(backend_runtime: dict[str, Any]) -> bool:
    if not isinstance(backend_runtime, dict) or not backend_runtime:
        return False
    if backend_runtime.get('model_active') is False:
        return True
    for key in ('busy', 'is_busy', 'runner_busy', 'active'):
        value = backend_runtime.get(key)
        if isinstance(value, bool):
            return not value
    state = str(
        backend_runtime.get('activity')
        or backend_runtime.get('state')
        or backend_runtime.get('status')
        or backend_runtime.get('runner_state')
        or ''
    ).strip().lower()
    if state:
        return state in {'idle', 'ready', 'completed', 'complete', 'done', 'finished', 'stopped'}
    return backend_runtime.get('model_active') is True


def _error_is_timeout_like(raw_error: Any) -> bool:
    token = str(raw_error or '').strip().lower()
    if not token:
        return False
    return any(
        marker in token
        for marker in (
            'timeout',
            'timed out',
            'waiting for the next generated token',
        )
    )


def _update_entry(
    instance_id: str,
    *,
    path: Path | str | None = None,
    instance: Optional[dict] = None,
    remove: bool = False,
    **updates: Any,
) -> tuple[Optional[dict], Optional[dict]]:
    payload = read_runtime_status(path)
    instances = payload['instances']
    previous = dict(instances.get(instance_id) or {}) or None

    if remove:
        instances.pop(instance_id, None)
        payload['updated_at'] = _now_iso()
        write_runtime_status(payload, path=path)
        return previous, None

    current = dict(previous or {})
    if not current:
        current['instance_id'] = instance_id
        current['created_at'] = _now_iso()
    _merge_instance_metadata(current, instance)
    for key, value in updates.items():
        if value is None:
            continue
        current[key] = value
    current['last_checked_at'] = _now_iso()
    instances[instance_id] = current
    payload['updated_at'] = _now_iso()
    write_runtime_status(payload, path=path)
    return previous, current


def record_instance_started(instance: dict, *, path: Path | str | None = None) -> tuple[Optional[dict], dict]:
    instance_id = str(instance.get('instance_id') or '').strip()
    if not instance_id:
        raise ValueError("Instance requires 'instance_id'.")
    port_listening = _port_listening(instance.get('port'))
    process_alive = _process_alive(instance.get('pid'))
    readiness = 'ready'
    if process_alive is False or port_listening is False:
        readiness = 'unreachable'
    previous, current = _update_entry(
        instance_id,
        path=path,
        instance=instance,
        readiness=readiness,
        activity='idle',
        process_alive=process_alive,
        port_listening=port_listening,
        last_started_at=_now_iso(),
    )
    return previous, current or {}


def record_instance_activity(
    instance_id: str,
    *,
    path: Path | str | None = None,
    instance: Optional[dict] = None,
    activity: str,
) -> tuple[Optional[dict], dict]:
    previous, current = _update_entry(
        instance_id,
        path=path,
        instance=instance,
        activity=activity,
    )
    return previous, current or {}


def record_instance_success(
    instance_id: str,
    *,
    path: Path | str | None = None,
    instance: Optional[dict] = None,
    latency_sec: Optional[float] = None,
) -> tuple[Optional[dict], dict]:
    port = instance.get('port') if instance else None
    pid = instance.get('pid') if instance else None
    previous, current = _update_entry(
        instance_id,
        path=path,
        instance=instance,
        readiness='ready',
        activity='idle',
        process_alive=_process_alive(pid) if pid is not None else True,
        port_listening=_port_listening(port) if port is not None else True,
        last_success_at=_now_iso(),
        last_latency_sec=latency_sec,
        last_error='',
    )
    if current is not None:
        current.pop('last_error', None)
        current.pop('cooldown_until', None)
        current.pop('failure_cooldown_until', None)
        current.pop('cooldown_capability', None)
        current.pop('cooldown_error_class', None)
        current.pop('cooldown_ttl_sec', None)
        payload = read_runtime_status(path)
        payload['instances'][instance_id] = current
        payload['updated_at'] = _now_iso()
        write_runtime_status(payload, path=path)
    return previous, current or {}


def record_instance_failure(
    instance_id: str,
    *,
    path: Path | str | None = None,
    instance: Optional[dict] = None,
    message: str,
    latency_sec: Optional[float] = None,
) -> tuple[Optional[dict], dict]:
    port = instance.get('port') if instance else None
    pid = instance.get('pid') if instance else None
    port_listening = _port_listening(port) if port is not None else None
    process_alive = _process_alive(pid) if pid is not None else None
    readiness = 'degraded'
    if process_alive is False or port_listening is False:
        readiness = 'unreachable'
    now = runtime_utc_now()
    cooldown_until = now + dt.timedelta(seconds=DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC)
    cooldown_capability = normalize_capability(instance.get('capability')) if instance else ''
    previous, current = _update_entry(
        instance_id,
        path=path,
        instance=instance,
        readiness=readiness,
        activity='idle',
        process_alive=process_alive,
        port_listening=port_listening,
        last_error_at=format_runtime_timestamp(now),
        last_error=message,
        last_latency_sec=latency_sec,
        cooldown_until=format_runtime_timestamp(cooldown_until),
        failure_cooldown_until=format_runtime_timestamp(cooldown_until),
        cooldown_capability=cooldown_capability,
        cooldown_error_class=runtime_failure_error_class(message),
        cooldown_ttl_sec=DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC,
    )
    return previous, current or {}


def remove_instance_status(instance_id: str, *, path: Path | str | None = None) -> Optional[dict]:
    previous, _ = _update_entry(instance_id, path=path, remove=True)
    return previous


def refresh_runtime_status_entries(
    instances: list[dict],
    *,
    path: Path | str | None = None,
) -> dict[str, dict]:
    payload = read_runtime_status(path)
    existing = payload.get('instances', {})
    refreshed: dict[str, dict] = {}

    for instance in instances:
        instance_id = str(instance.get('instance_id') or '').strip()
        if not instance_id:
            continue
        entry = dict(existing.get(instance_id) or {})
        _merge_instance_metadata(entry, instance)
        process_alive = _process_alive(entry.get('pid'))
        port_listening = _port_listening(entry.get('port'))
        entry['process_alive'] = process_alive
        entry['port_listening'] = port_listening
        entry['last_checked_at'] = _now_iso()

        backend_runtime = {}
        if process_alive is False or port_listening is False:
            entry['readiness'] = 'unreachable'
            entry['activity'] = 'idle'
        else:
            backend_runtime = _fetch_backend_runtime_metadata(entry)
            if backend_runtime:
                entry['backend_runtime'] = backend_runtime
            if str(entry.get('activity') or '').strip().lower() == 'busy':
                if _backend_runtime_indicates_idle(backend_runtime):
                    entry['activity'] = 'idle'
                    entry['busy_clear_reason'] = 'backend_runtime_idle'
                    entry['last_busy_cleared_at'] = _now_iso()
            else:
                entry['activity'] = 'idle'
            last_error_at = str(entry.get('last_error_at') or '')
            last_recovery_at = max(
                (
                    str(entry.get('last_success_at') or ''),
                    str(entry.get('last_started_at') or ''),
                )
            )
            if last_error_at and (not last_recovery_at or last_error_at > last_recovery_at):
                if (
                    _backend_runtime_indicates_idle(backend_runtime)
                    and _error_is_timeout_like(entry.get('last_error'))
                ):
                    entry['readiness'] = 'ready'
                    entry['degraded_clear_reason'] = 'backend_runtime_idle_after_timeout'
                    entry['last_degraded_cleared_at'] = _now_iso()
                    entry.pop('cooldown_until', None)
                    entry.pop('failure_cooldown_until', None)
                    entry.pop('cooldown_capability', None)
                    entry.pop('cooldown_error_class', None)
                    entry.pop('cooldown_ttl_sec', None)
                else:
                    entry['readiness'] = 'degraded'
            else:
                entry['readiness'] = 'ready'

        refreshed[instance_id] = entry

    payload['instances'] = refreshed
    payload['updated_at'] = _now_iso()
    write_runtime_status(payload, path=path)
    return refreshed


def merge_instances_with_runtime_status(
    instances: list[dict],
    *,
    path: Path | str | None = None,
    refresh: bool = False,
) -> list[dict]:
    statuses = (
        refresh_runtime_status_entries(instances, path=path)
        if refresh
        else read_runtime_status(path).get('instances', {})
    )
    merged = []
    for instance in instances:
        entry = dict(instance)
        status = dict(statuses.get(str(instance.get('instance_id') or '')) or {})
        entry['runtime_status'] = status
        if status:
            entry['readiness'] = status.get('readiness')
            entry['activity'] = status.get('activity')
            entry['process_alive'] = status.get('process_alive')
            entry['port_listening'] = status.get('port_listening')
            entry['last_success_at'] = status.get('last_success_at')
            entry['last_error'] = status.get('last_error')
            entry['last_error_at'] = status.get('last_error_at')
            entry['last_latency_sec'] = status.get('last_latency_sec')
            entry['backend_runtime'] = status.get('backend_runtime')
            entry['cooldown_until'] = status.get('cooldown_until')
            entry['failure_cooldown_until'] = status.get('failure_cooldown_until')
            entry['cooldown_capability'] = status.get('cooldown_capability')
            entry['cooldown_error_class'] = status.get('cooldown_error_class')
            entry['cooldown_ttl_sec'] = status.get('cooldown_ttl_sec')
            entry['start_source'] = status.get('start_source')
            entry['start_audit'] = status.get('start_audit')
        merged.append(entry)
    return merged
