"""Shared runtime liveness and cooldown policy helpers."""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Optional

from helpers.model_capabilities import normalize_capability


DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC = 300

BUSY_RUNTIME_ACTIVITIES = {
    'active',
    'busy',
    'executing',
    'generating',
    'in_progress',
    'loading',
    'running',
    'starting',
    'streaming',
    'working',
}
HARD_UNAVAILABLE_RUNTIME_STATES = {
    'error',
    'failed',
    'offline',
    'stopped',
    'stopping',
    'unreachable',
}


def runtime_utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def format_runtime_timestamp(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_runtime_timestamp(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        token = str(value or '').strip()
        if not token:
            return None
        if token.endswith('Z'):
            token = f'{token[:-1]}+00:00'
        try:
            parsed = dt.datetime.fromisoformat(token)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def runtime_status_value(instance: Mapping[str, Any], key: str) -> Any:
    runtime_status = (
        instance.get('runtime_status')
        if isinstance(instance.get('runtime_status'), Mapping)
        else {}
    )
    if key in instance and instance.get(key) not in (None, ''):
        return instance.get(key)
    if isinstance(runtime_status, Mapping) and runtime_status.get(key) not in (None, ''):
        return runtime_status.get(key)
    return None


def runtime_text_value(instance: Mapping[str, Any], key: str) -> str:
    return str(runtime_status_value(instance, key) or '').strip().lower()


def runtime_bool_value(instance: Mapping[str, Any], key: str) -> Optional[bool]:
    value = runtime_status_value(instance, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    token = str(value or '').strip().lower()
    if not token:
        return None
    if token in {'1', 'true', 'yes', 'on'}:
        return True
    if token in {'0', 'false', 'no', 'off'}:
        return False
    return None


def runtime_instance_capability(instance: Mapping[str, Any]) -> str:
    return normalize_capability(runtime_status_value(instance, 'capability')) or ''


def runtime_failure_error_class(message: Any) -> str:
    token = str(message or '').strip().lower()
    if not token:
        return 'unknown'
    if any(marker in token for marker in ('oom', 'out of memory', 'memory pressure', 'metal oom')):
        return 'oom'
    if any(marker in token for marker in ('timeout', 'timed out', 'deadline')):
        return 'timeout'
    if any(marker in token for marker in ('connection refused', 'connection reset', 'connect_ex', 'port')):
        return 'connection'
    if any(marker in token for marker in ('unavailable', 'not reachable', 'unreachable', 'no route')):
        return 'backend_unavailable'
    return 'backend_error'


def _cooldown_capability_matches(instance: Mapping[str, Any], capability: Any) -> bool:
    requested = normalize_capability(capability) or ''
    cooldown_capability = normalize_capability(runtime_status_value(instance, 'cooldown_capability')) or ''
    instance_capability = runtime_instance_capability(instance)
    if requested and cooldown_capability:
        return cooldown_capability == requested
    if requested and instance_capability:
        return instance_capability == requested
    return True


def runtime_failure_cooldown_expires_at(
    instance: Mapping[str, Any],
    *,
    capability: Any = None,
    ttl_sec: int = DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC,
) -> Optional[dt.datetime]:
    if not _cooldown_capability_matches(instance, capability):
        return None
    explicit_until = (
        parse_runtime_timestamp(runtime_status_value(instance, 'cooldown_until'))
        or parse_runtime_timestamp(runtime_status_value(instance, 'failure_cooldown_until'))
    )
    if explicit_until is not None:
        return explicit_until
    last_error_at = parse_runtime_timestamp(runtime_status_value(instance, 'last_error_at'))
    if last_error_at is None or ttl_sec <= 0:
        return None
    return last_error_at + dt.timedelta(seconds=ttl_sec)


def runtime_instance_liveness(
    instance: Mapping[str, Any],
    *,
    capability: Any = None,
    now: Optional[dt.datetime] = None,
    ttl_sec: int = DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC,
) -> dict[str, Any]:
    current_time = now or runtime_utc_now()
    process_alive = runtime_bool_value(instance, 'process_alive')
    port_listening = runtime_bool_value(instance, 'port_listening')
    readiness = runtime_text_value(instance, 'readiness')
    status = runtime_text_value(instance, 'status')
    activity = runtime_text_value(instance, 'activity')
    cooldown_until_dt = runtime_failure_cooldown_expires_at(
        instance,
        capability=capability,
        ttl_sec=ttl_sec,
    )
    cooldown_until = format_runtime_timestamp(cooldown_until_dt) if cooldown_until_dt else ''
    fresh_cooldown = bool(cooldown_until_dt and cooldown_until_dt > current_time)
    state_token = readiness or status
    hard_state_without_live_evidence = (
        state_token in HARD_UNAVAILABLE_RUNTIME_STATES
        and process_alive is not True
        and port_listening is not True
    )
    hard_unavailable = (
        process_alive is False
        or port_listening is False
        or hard_state_without_live_evidence
    )
    busy = activity in BUSY_RUNTIME_ACTIVITIES
    advisory_degraded = readiness == 'degraded' or status == 'degraded'
    last_error = runtime_status_value(instance, 'last_error')
    selectable = not hard_unavailable and not fresh_cooldown
    return {
        'process_alive': process_alive,
        'port_listening': port_listening,
        'readiness': readiness,
        'status': status,
        'activity': activity,
        'busy': busy,
        'advisory_degraded': advisory_degraded,
        'last_error': last_error,
        'cooldown_until': cooldown_until,
        'fresh_cooldown': fresh_cooldown,
        'hard_unavailable': hard_unavailable,
        'selectable': selectable,
    }


def runtime_instance_is_selectable(
    instance: Mapping[str, Any],
    *,
    capability: Any = None,
    now: Optional[dt.datetime] = None,
    ttl_sec: int = DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC,
) -> bool:
    return bool(
        runtime_instance_liveness(
            instance,
            capability=capability,
            now=now,
            ttl_sec=ttl_sec,
        )['selectable']
    )


def runtime_instance_score(
    instance: Mapping[str, Any],
    *,
    capability: Any = None,
    now: Optional[dt.datetime] = None,
    ttl_sec: int = DEFAULT_INSTANCE_FAILURE_COOLDOWN_TTL_SEC,
) -> tuple[int, int, int, int, int, int, int, str]:
    liveness = runtime_instance_liveness(
        instance,
        capability=capability,
        now=now,
        ttl_sec=ttl_sec,
    )
    readiness = liveness['readiness']
    activity = liveness['activity']
    readiness_rank = 0
    if readiness == 'ready':
        readiness_rank = 3
    elif readiness in {'started', 'idle'}:
        readiness_rank = 2
    elif readiness == '':
        readiness_rank = 1
    activity_rank = 2 if activity in {'idle', 'ready', ''} else 0 if liveness['busy'] else 1
    last_error_rank = 0 if liveness.get('last_error') else 1
    port_rank = 1 if liveness['port_listening'] is True else 0
    process_rank = 1 if liveness['process_alive'] is True else 0
    return (
        1 if liveness['selectable'] else 0,
        1 if not liveness['busy'] else 0,
        readiness_rank,
        activity_rank,
        last_error_rank,
        port_rank,
        process_rank,
        str(instance.get('instance_id') or ''),
    )


def runtime_liveness_summary(instance: Mapping[str, Any], *, capability: Any = None) -> str:
    instance_id = str(instance.get('instance_id') or '').strip() or '<unknown>'
    liveness = runtime_instance_liveness(instance, capability=capability)
    details: list[str] = []
    for key in ('readiness', 'activity'):
        value = liveness.get(key)
        if value:
            details.append(f'{key}={value}')
    if liveness['process_alive'] is not None:
        details.append(f"process_alive={str(liveness['process_alive']).lower()}")
    if liveness['port_listening'] is not None:
        details.append(f"port_listening={str(liveness['port_listening']).lower()}")
    if liveness['fresh_cooldown'] and liveness['cooldown_until']:
        details.append(f"cooldown_until={liveness['cooldown_until']}")
    if liveness['hard_unavailable']:
        details.append('hard_unavailable=true')
    if liveness['advisory_degraded'] and not liveness['hard_unavailable']:
        details.append('degraded=advisory')
    suffix = f" ({', '.join(details)})" if details else ''
    return f'{instance_id}{suffix}'
