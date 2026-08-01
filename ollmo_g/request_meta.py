"""Structured request-meta normalization for Ghost diagnostics and future control hints."""

from __future__ import annotations

import json
from typing import Any, Optional

from helpers.model_capabilities import SUPPORTED_CAPABILITIES, normalize_capability

from ollmo_g.ghost_mode_compat import normalize_legacy_mode_hint

DEFAULT_DEVELOPER_FLAGS: dict[str, Any] = {
    'embedding_signals_enabled': True,
    'planner_timeout_ms': None,
    'accepted_learning_authority': 'soft_hint',
}
MIN_PLANNER_TIMEOUT_MS = 500
MAX_PLANNER_TIMEOUT_MS = 7_200_000
ACCEPTED_LEARNING_AUTHORITY_LEVELS = {
    'soft_hint',
    'advisory',
    'preferred',
    'enforced',
}


def _clean_text(value: Any, *, lower: bool = False) -> Optional[str]:
    token = str(value or '').strip()
    if not token:
        return None
    return token.lower() if lower else token


def _parse_jsonish(raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, (dict, list)):
        return raw_value
    token = str(raw_value or '').strip()
    if not token:
        return None
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        return None


def _parse_bool_like(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value or '').strip().lower()
    if token in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if token in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def _coerce_timeout_ms(value: Any, *, minimum: int, maximum: int) -> Optional[int]:
    if value in (None, ''):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return max(minimum, min(maximum, parsed))


def normalize_accepted_learning_authority(value: Any) -> str:
    token = str(value or '').strip().lower().replace('-', '_')
    if token in ACCEPTED_LEARNING_AUTHORITY_LEVELS:
        return token
    return str(DEFAULT_DEVELOPER_FLAGS['accepted_learning_authority'])


def normalize_developer_flags(raw_value: Any) -> dict[str, Any]:
    payload = raw_value if isinstance(raw_value, dict) else _parse_jsonish(raw_value)
    source = payload if isinstance(payload, dict) else {}
    embedding_signals_enabled = _parse_bool_like(
        source.get('embedding_signals_enabled'),
        default=bool(DEFAULT_DEVELOPER_FLAGS['embedding_signals_enabled']),
    )
    return {
        'embedding_signals_enabled': embedding_signals_enabled,
        'planner_timeout_ms': _coerce_timeout_ms(
            source.get('planner_timeout_ms'),
            minimum=MIN_PLANNER_TIMEOUT_MS,
            maximum=MAX_PLANNER_TIMEOUT_MS,
        ),
        'accepted_learning_authority': normalize_accepted_learning_authority(
            source.get('accepted_learning_authority')
            if source.get('accepted_learning_authority') is not None
            else source.get('acceptedLearningAuthority')
        ),
    }

def _coerce_request_meta_source(payload: Any) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    nested_payload = source.get('request_meta') if source.get('request_meta') is not None else source.get('requestMeta')
    nested = nested_payload if isinstance(nested_payload, dict) else _parse_jsonish(nested_payload)
    if not isinstance(nested, dict):
        return source
    merged = dict(nested)
    for key in (
        'ghost_mode',
        'ghostMode',
        'capability_hint',
        'language_hint',
        'developer_flags',
        'developerFlags',
        'accepted_learning_authority',
        'acceptedLearningAuthority',
    ):
        if source.get(key) is not None:
            merged[key] = source.get(key)
    return merged


def normalize_request_meta(payload: Any) -> dict[str, Any]:
    source = _coerce_request_meta_source(payload)

    request_ghost_mode = normalize_legacy_mode_hint(
        source.get('ghost_mode') if source.get('ghost_mode') is not None else source.get('ghostMode')
    )

    request_capability_hint = normalize_capability(source.get('capability_hint'))
    capability_hint = request_capability_hint
    if capability_hint not in SUPPORTED_CAPABILITIES:
        capability_hint = None

    request_language_hint = _clean_text(source.get('language_hint'), lower=True)
    language_hint = request_language_hint

    raw_developer_flags = (
        source.get('developer_flags') if source.get('developer_flags') is not None else source.get('developerFlags')
    )
    if source.get('accepted_learning_authority') is not None or source.get('acceptedLearningAuthority') is not None:
        parsed_flags = raw_developer_flags if isinstance(raw_developer_flags, dict) else _parse_jsonish(raw_developer_flags)
        raw_developer_flags = dict(parsed_flags) if isinstance(parsed_flags, dict) else {}
        raw_developer_flags['accepted_learning_authority'] = (
            source.get('accepted_learning_authority')
            if source.get('accepted_learning_authority') is not None
            else source.get('acceptedLearningAuthority')
        )
    developer_flags = normalize_developer_flags(raw_developer_flags)
    return {
        'ghost_mode': request_ghost_mode,
        'ghost_mode_source': 'request' if request_ghost_mode else None,
        'capability_hint': capability_hint,
        'capability_hint_source': 'request' if request_capability_hint else None,
        'language_hint': language_hint,
        'language_hint_source': 'request' if request_language_hint else None,
        'developer_flags': developer_flags,
    }


def compact_request_meta(request_meta: Any) -> dict[str, Any]:
    source = request_meta if isinstance(request_meta, dict) else {}
    compact = {
        'ghost_mode': normalize_legacy_mode_hint(source.get('ghost_mode')),
        'capability_hint': normalize_capability(source.get('capability_hint')),
        'language_hint': _clean_text(source.get('language_hint'), lower=True),
        'developer_flags': normalize_developer_flags(source.get('developer_flags')),
    }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, '', [], {})
    }


def attach_request_meta(payload: Any) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else dict(payload or {})
    request_meta = normalize_request_meta(source)
    updated = dict(source)
    for key in ('pipeline', 'mode', 'role', 'execution_profile', 'executionProfile', 'requestMeta'):
        updated.pop(key, None)
    updated['_ollmo_request_meta'] = request_meta
    compact = compact_request_meta(request_meta)
    if compact:
        updated['request_meta'] = compact
    else:
        updated.pop('request_meta', None)
    if compact.get('ghost_mode'):
        updated['ghost_mode'] = compact['ghost_mode']
    if compact.get('capability_hint'):
        updated['capability_hint'] = compact['capability_hint']
    if compact.get('language_hint'):
        updated['language_hint'] = compact['language_hint']
    if 'developer_flags' in compact:
        updated['developer_flags'] = compact.get('developer_flags')
    return updated


def extract_request_meta(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get('_ollmo_request_meta'), dict):
        return payload.get('_ollmo_request_meta') or {}
    return normalize_request_meta(payload if isinstance(payload, dict) else dict(payload or {}))


def effective_developer_flags(payload: Any) -> dict[str, Any]:
    request_meta = extract_request_meta(payload)
    return normalize_developer_flags(request_meta.get('developer_flags'))


def apply_request_meta_to_route_context(route_context: dict[str, Any], request_meta: Any) -> dict[str, Any]:
    updated = dict(route_context or {})
    compact_meta = compact_request_meta(request_meta)
    if compact_meta:
        updated['request_meta'] = compact_meta
    runtime = dict(updated.get('runtime') or {})
    if compact_meta:
        runtime['request_meta'] = compact_meta
    updated['runtime'] = runtime
    return updated
