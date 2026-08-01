"""Model-start source validation and audit metadata."""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Optional


VALID_START_SOURCES = {
    'api_start_model',
    'frontend_button',
    'startup_policy',
    'explicit_recovery',
}

FORBIDDEN_START_SOURCES = {
    'ghost_carried',
    'ghost_route',
    'late_fill',
    'phase_continuation',
    'route_selection',
}


class StartSourcePolicyError(ValueError):
    def __init__(self, message: str, *, policy_violation: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.policy_violation = policy_violation or {}


def normalize_start_source(value: Any) -> str:
    return str(value or '').strip().lower()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def build_start_source_policy_violation(
    value: Any,
    *,
    context: str,
    reason: str,
) -> dict[str, Any]:
    return {
        'kind': 'ollmo.model_start_policy_violation',
        'policy': 'explicit_lifecycle_start_required',
        'context': str(context or 'model_start').strip() or 'model_start',
        'start_source': normalize_start_source(value),
        'reason': reason,
        'valid_start_sources': sorted(VALID_START_SOURCES),
        'forbidden_start_sources': sorted(FORBIDDEN_START_SOURCES),
        'created_at': _now_iso(),
    }


def validate_start_source(
    value: Any,
    *,
    default: Any = None,
    context: str = 'model_start',
) -> str:
    start_source = normalize_start_source(value)
    if not start_source and default is not None:
        start_source = normalize_start_source(default)
    if not start_source:
        violation = build_start_source_policy_violation(
            start_source,
            context=context,
            reason='missing_start_source',
        )
        raise StartSourcePolicyError(
            'start_source is required for model starts.',
            policy_violation=violation,
        )
    if start_source in FORBIDDEN_START_SOURCES:
        violation = build_start_source_policy_violation(
            start_source,
            context=context,
            reason='forbidden_route_source',
        )
        raise StartSourcePolicyError(
            f"Forbidden start_source '{start_source}' for model starts.",
            policy_violation=violation,
        )
    if start_source not in VALID_START_SOURCES:
        violation = build_start_source_policy_violation(
            start_source,
            context=context,
            reason='unknown_start_source',
        )
        raise StartSourcePolicyError(
            f"Unsupported start_source '{start_source}' for model starts.",
            policy_violation=violation,
        )
    return start_source


def build_start_audit(
    start_source: Any,
    *,
    context: str,
    requested_by: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    normalized_source = validate_start_source(start_source, context=context)
    audit = {
        'kind': 'ollmo.model_start_audit',
        'policy': 'explicit_lifecycle_start_required',
        'start_source': normalized_source,
        'context': str(context or 'model_start').strip() or 'model_start',
        'created_at': _now_iso(),
    }
    requested_by_text = str(requested_by or '').strip()
    if requested_by_text:
        audit['requested_by'] = requested_by_text
    if isinstance(extra, Mapping):
        audit.update(
            {
                str(key): value
                for key, value in extra.items()
                if value not in (None, '', [], {})
            }
        )
    return audit


def attach_start_audit(
    instance: Mapping[str, Any],
    *,
    start_source: Any,
    context: str,
    requested_by: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    audit = build_start_audit(
        start_source,
        context=context,
        requested_by=requested_by,
        extra=extra,
    )
    updated = dict(instance or {})
    updated['start_source'] = audit['start_source']
    updated['start_audit'] = audit
    return updated
