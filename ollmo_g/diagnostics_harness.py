"""Offline fixture utilities for Ghost diagnostics and routing validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from helpers.model_capabilities import SUPPORTED_CAPABILITIES, normalize_capability

DEFAULT_GHOST_DIAGNOSTICS_FIXTURE_DIR = Path(__file__).resolve().parent.parent / 'tests' / 'testdata' / 'ghost_diagnostics'


def _clean_text(value: Any, *, lower: bool = False) -> Optional[str]:
    token = str(value or '').strip()
    if not token:
        return None
    return token.lower() if lower else token


def _normalize_artifact(raw_artifact: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw_artifact, dict):
        return None
    artifact_type = _clean_text(raw_artifact.get('type'), lower=True)
    path = _clean_text(raw_artifact.get('path'))
    if not artifact_type and not path:
        return None
    payload = {}
    if artifact_type:
        payload['type'] = artifact_type
    if path:
        payload['path'] = path
    if raw_artifact.get('image_state') and isinstance(raw_artifact.get('image_state'), dict):
        payload['image_state'] = dict(raw_artifact.get('image_state') or {})
    return payload or None


def normalize_diagnostic_fixture(raw_fixture: Any, *, source_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    if not isinstance(raw_fixture, dict):
        return None
    case_id = _clean_text(raw_fixture.get('case_id') or raw_fixture.get('id'))
    prompt = _clean_text(raw_fixture.get('prompt'))
    expected_capability = normalize_capability(raw_fixture.get('expected_capability'))
    if not case_id or not prompt or expected_capability not in SUPPORTED_CAPABILITIES:
        return None
    payload = {
        'case_id': case_id,
        'prompt': prompt,
        'attachments': [
            artifact
            for artifact in (_normalize_artifact(item) for item in (raw_fixture.get('attachments') or []))
            if artifact
        ],
        'expected_capability': expected_capability,
        'expected_instance_traits': dict(raw_fixture.get('expected_instance_traits') or {})
        if isinstance(raw_fixture.get('expected_instance_traits'), dict) else {},
        'expected_route_metadata': dict(raw_fixture.get('expected_route_metadata') or {})
        if isinstance(raw_fixture.get('expected_route_metadata'), dict) else {},
        'expected_control_hints': dict(raw_fixture.get('expected_control_hints') or {})
        if isinstance(raw_fixture.get('expected_control_hints'), dict) else {},
        'meta': dict(raw_fixture.get('meta') or {}) if isinstance(raw_fixture.get('meta'), dict) else {},
    }
    if source_path:
        payload['source_path'] = str(source_path)
    return payload


def load_diagnostic_fixtures(*, fixture_dir: Path | str | None = None) -> list[dict[str, Any]]:
    target_dir = Path(fixture_dir) if fixture_dir else DEFAULT_GHOST_DIAGNOSTICS_FIXTURE_DIR
    if not target_dir.exists():
        return []
    fixtures: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob('*.json')):
        try:
            raw_fixture = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        normalized = normalize_diagnostic_fixture(raw_fixture, source_path=path)
        if normalized:
            fixtures.append(normalized)
    return fixtures


def build_fixture_request_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_diagnostic_fixture(fixture)
    if not normalized:
        raise ValueError('Invalid diagnostic fixture.')
    meta = dict(normalized.get('meta') or {})
    developer_flags = dict(meta.get('developer_flags') or {})
    payload: dict[str, Any] = {
        'ghost_route': True,
        'prompt': normalized['prompt'],
        'ghost_messages': [{'role': 'user', 'content': normalized['prompt']}],
        'developer_flags': developer_flags,
    }
    for key in ('ghost_mode', 'capability_hint', 'language_hint'):
        value = meta.get(key)
        if value not in (None, '', [], {}):
            payload[key] = value
    if normalized.get('attachments'):
        payload['attachments'] = list(normalized['attachments'])
    return payload


def _lookup_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for segment in str(dotted_path or '').split('.'):
        if not segment:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def evaluate_fixture_result(
    fixture: dict[str, Any],
    *,
    route_payload: Optional[dict[str, Any]] = None,
    control_hints: Optional[dict[str, Any]] = None,
) -> list[str]:
    normalized = normalize_diagnostic_fixture(fixture)
    if not normalized:
        return ['invalid fixture']
    failures: list[str] = []
    route_payload = route_payload if isinstance(route_payload, dict) else {}
    resolved_capability = normalize_capability(
        route_payload.get('capability')
        or route_payload.get('response_capability')
        or route_payload.get('mode')
    )
    if resolved_capability != normalized['expected_capability']:
        failures.append(
            f"expected capability {normalized['expected_capability']}, got {resolved_capability or 'none'}"
        )
    for dotted_key, expected_value in (normalized.get('expected_route_metadata') or {}).items():
        actual_value = _lookup_path(route_payload, dotted_key)
        if actual_value != expected_value:
            failures.append(
                f'expected route metadata {dotted_key}={expected_value!r}, got {actual_value!r}'
            )
    effective_control_hints = control_hints if isinstance(control_hints, dict) else (
        _lookup_path(route_payload, 'runtime.control_hints') if isinstance(route_payload, dict) else {}
    )
    effective_control_hints = effective_control_hints if isinstance(effective_control_hints, dict) else {}
    for key, expected_value in (normalized.get('expected_control_hints') or {}).items():
        actual_value = effective_control_hints.get(key)
        if actual_value != expected_value:
            failures.append(
                f'expected control hint {key}={expected_value!r}, got {actual_value!r}'
            )
    return failures
