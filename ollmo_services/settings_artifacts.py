"""Reusable settings artifacts promoted from response-frame controls."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


DEFAULT_SETTINGS_ARTIFACTS_DIR = Path('artifacts/settings')
SETTINGS_ARTIFACT_VERSION = 1

_REQUEST_OVERRIDE_CATEGORIES = ('generation', 'audio', 'image', 'document', 'cache', 'execution')
_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.:-]+$')


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _is_empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _clean_text(raw_key)
            if not key or _is_empty(raw_value):
                continue
            payload[key] = _json_safe(raw_value)
        return payload
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value if not _is_empty(item)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:16]


def _compact_filename_token(value: str) -> str:
    token = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('-._')
    return token[:80] or 'settings'


def _extract_control_snapshot(source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    kind = _clean_text(source.get('kind'))
    if kind == 'ollmo.settings_artifact':
        controls = source.get('controls')
        if not isinstance(controls, Mapping) or not controls:
            raise ValueError('settings_artifact must include non-empty controls')
        return {
            'kind': 'ollmo.control_snapshot',
            'status': 'tracked',
            'values': dict(controls),
            'target': dict(source.get('target') or {}) if isinstance(source.get('target'), Mapping) else {},
            'route': dict(source.get('route') or {}) if isinstance(source.get('route'), Mapping) else {},
        }, {
            'kind': kind,
            'artifact_id': _clean_text(source.get('artifact_id')) or None,
        }
    if kind == 'ollmo.response_frame':
        controls = source.get('controls')
        if not isinstance(controls, Mapping) or not controls:
            raise ValueError('response_frame does not include controls to promote')
        snapshot = dict(controls)
        if 'target' not in snapshot and isinstance(source.get('target'), Mapping):
            snapshot['target'] = dict(source.get('target') or {})
        if 'route' not in snapshot and isinstance(source.get('route'), Mapping):
            snapshot['route'] = dict(source.get('route') or {})
        return snapshot, {
            'kind': kind,
            'response_id': _clean_text(source.get('response_id')) or None,
            'frame_version': source.get('frame_version'),
        }
    if kind == 'ollmo.control_snapshot':
        return dict(source), {'kind': kind}
    controls = source.get('controls')
    if isinstance(controls, Mapping) and controls:
        return dict(controls), {
            'kind': kind or 'payload',
            'response_id': _clean_text(source.get('response_id')) or None,
            'frame_version': source.get('frame_version'),
        }
    if isinstance(source.get('values'), Mapping):
        return {
            'kind': 'ollmo.control_snapshot',
            'status': _clean_text(source.get('status')) or 'tracked',
            'values': dict(source.get('values') or {}),
            'target': dict(source.get('target') or {}) if isinstance(source.get('target'), Mapping) else {},
            'route': dict(source.get('route') or {}) if isinstance(source.get('route'), Mapping) else {},
        }, {'kind': kind or 'values_payload'}
    raise ValueError('settings artifact source must include response_frame controls or a control snapshot')


def _build_request_overrides(control_values: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for category in _REQUEST_OVERRIDE_CATEGORIES:
        values = control_values.get(category)
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if _is_empty(value):
                continue
            overrides[_clean_text(key)] = _json_safe(value)
    return overrides


def _select_snapshot_mapping(snapshot: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = snapshot.get(key)
    return _json_safe(value) if isinstance(value, Mapping) else {}


def build_settings_artifact(
    source: Mapping[str, Any],
    *,
    label: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Build a reusable settings artifact from a response frame or controls snapshot."""

    if not isinstance(source, Mapping):
        raise ValueError('settings artifact source must be a mapping')
    snapshot, source_meta = _extract_control_snapshot(source)
    values = snapshot.get('values') if isinstance(snapshot.get('values'), Mapping) else {}
    if not values:
        raise ValueError('control snapshot has no reusable values')

    controls = _json_safe(values)
    request_overrides = _build_request_overrides(controls)
    if not request_overrides and not controls.get('batch'):
        raise ValueError('control snapshot has no request overrides to reuse')

    target = _select_snapshot_mapping(snapshot, 'target')
    route = _select_snapshot_mapping(snapshot, 'route')
    digest_payload = {
        'controls': controls,
        'request_overrides': request_overrides,
        'target': target,
        'route': route,
    }
    digest = _stable_digest(digest_payload)
    created = _clean_text(created_at) or _utc_now_iso()
    artifact_label = _clean_text(label) or _clean_text(target.get('capability') or target.get('mode')) or 'Reusable settings'
    artifact_id = f'settings_{digest}'
    artifact: dict[str, Any] = {
        'kind': 'ollmo.settings_artifact',
        'artifact_version': SETTINGS_ARTIFACT_VERSION,
        'artifact_id': artifact_id,
        'created_at': created,
        'label': artifact_label,
        'digest': digest,
        'controls': controls,
        'request_overrides': request_overrides,
        'source': {key: value for key, value in source_meta.items() if value not in (None, '', [], {})},
        'replay': {
            'merge_strategy': 'request_overrides',
            'description': 'Merge request_overrides into a future /api/responses request to reuse these settings.',
        },
    }
    if target:
        artifact['target'] = target
    if route:
        artifact['route'] = route
    if isinstance(controls.get('references'), Mapping):
        artifact['references'] = controls.get('references')
    if isinstance(controls.get('batch'), Mapping):
        artifact['batch'] = controls.get('batch')
    return artifact


def _settings_artifact_path(artifact: Mapping[str, Any], artifacts_dir: Path | str) -> Path:
    target_dir = Path(artifacts_dir)
    timestamp = _clean_text(artifact.get('created_at')).replace(':', '').replace('-', '')
    timestamp = _compact_filename_token(timestamp.replace('+', 'Z')) or dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label = _compact_filename_token(_clean_text(artifact.get('label')))
    artifact_id = _compact_filename_token(_clean_text(artifact.get('artifact_id')))
    return target_dir / f'{timestamp}_{label}_{artifact_id}.settings.json'


def persist_settings_artifact(
    source: Mapping[str, Any],
    *,
    label: Optional[str] = None,
    artifacts_dir: Path | str = DEFAULT_SETTINGS_ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Persist a settings artifact and return the saved artifact payload."""

    artifact = build_settings_artifact(source, label=label)
    target = _settings_artifact_path(artifact, artifacts_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact['path'] = str(target)
    target.write_text(json.dumps(_json_safe(artifact), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return artifact


def _read_settings_artifact(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get('kind') != 'ollmo.settings_artifact':
        return None
    payload['path'] = str(path)
    return _json_safe(payload)


def list_settings_artifacts(
    *,
    artifacts_dir: Path | str = DEFAULT_SETTINGS_ARTIFACTS_DIR,
) -> list[dict[str, Any]]:
    """List persisted settings artifacts, newest first."""

    target_dir = Path(artifacts_dir)
    if not target_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob('*.settings.json'), reverse=True):
        artifact = _read_settings_artifact(path)
        if artifact:
            artifacts.append(artifact)
    return artifacts


def load_settings_artifact(
    artifact_id: str,
    *,
    artifacts_dir: Path | str = DEFAULT_SETTINGS_ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Load a persisted settings artifact by artifact id or exact filename."""

    token = _clean_text(artifact_id)
    if not token or '/' in token or '\\' in token or not _ID_PATTERN.match(token):
        raise ValueError('invalid settings artifact id')
    target_dir = Path(artifacts_dir)
    candidates: list[Path] = []
    exact = target_dir / token
    if exact.suffix == '.json':
        candidates.append(exact)
    candidates.extend(sorted(target_dir.glob(f'*_{token}.settings.json'), reverse=True))
    if token.endswith('.settings.json'):
        candidates.append(target_dir / token)
    for path in candidates:
        artifact = _read_settings_artifact(path)
        if artifact:
            return artifact
    for artifact in list_settings_artifacts(artifacts_dir=target_dir):
        if artifact.get('artifact_id') == token:
            return artifact
    raise FileNotFoundError(token)
