"""Durable backend chat history for Ollmo conversations."""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from ollmo_services.artifact_contracts import sanitize_artifact_record
from ollmo_core.transports import (
    ARTIFACT_INPUTS_ROOT,
    ARTIFACT_OUTPUTS_AUDIO_DIR,
    ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
    ARTIFACT_OUTPUTS_IMAGES_DIR,
    ARTIFACT_OUTPUTS_OCR_DIR,
    ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
)

DEFAULT_CHAT_HISTORY_DIR = Path('state/chat_history')
DEFAULT_RESPONSE_FRAMES_DIR = DEFAULT_CHAT_HISTORY_DIR.parent / 'response_frames'
DEFAULT_RESPONSE_FRAME_LEDGER = 'responses.jsonl'
LINEAGE_LOG_FILE_NAME = 'conversation_lineage.jsonl'


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _slug(value: str) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip())
    return token.strip('._-') or 'chat'


def _history_path(instance_id: str, history_dir: Path | str | None = None) -> Path:
    base = Path(history_dir) if history_dir else DEFAULT_CHAT_HISTORY_DIR
    return base / f'{_slug(instance_id)}.json'


def _lineage_path(history_dir: Path | str | None = None) -> Path:
    base = Path(history_dir) if history_dir else DEFAULT_CHAT_HISTORY_DIR
    return base / LINEAGE_LOG_FILE_NAME


def _response_frame_ledger_path(history_dir: Path | str | None = None) -> Path:
    if history_dir:
        return Path(history_dir).parent / 'response_frames' / DEFAULT_RESPONSE_FRAME_LEDGER
    return DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER


def _read_lineage_events(history_dir: Path | str | None = None) -> list[dict[str, Any]]:
    target = _lineage_path(history_dir)
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for raw_line in target.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    except OSError:
        return []
    return events


def _trim_history_preview_text(value: Any) -> str:
    normalized = re.sub(r'<[^>]+>', ' ', str(value or ''))
    normalized = re.sub(r'\[[^\]]+:\s*[^\]]+\]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    if not normalized:
        return ''
    return f'{normalized[:65]}...' if len(normalized) > 68 else normalized


def _derive_history_preview_text(messages: list[dict[str, Any]], *, role: str | None = None, reverse: bool = False) -> str:
    normalized_role = str(role or '').strip().lower()
    iterable = reversed(messages) if reverse else messages
    for message in iterable:
        if not isinstance(message, dict):
            continue
        if normalized_role and str(message.get('role') or '').strip().lower() != normalized_role:
            continue
        preview = _trim_history_preview_text(message.get('content'))
        if preview:
            return preview
    return ''


def _build_slot_history_ids(
    *,
    workspace: str | None = None,
    slot_id: str | None = None,
    history_dir: Path | str | None = None,
    fallback_instance_id: str | None = None,
) -> list[str]:
    workspace_key = str(workspace or '').strip()
    slot_key = str(slot_id or '').strip()
    fallback_key = str(fallback_instance_id or '').strip()
    ids: list[str] = []
    seen: set[str] = set()

    def push(value: str | None) -> None:
        token = str(value or '').strip()
        if token and token not in seen:
            seen.add(token)
            ids.append(token)

    push(fallback_key)
    for event in _read_lineage_events(history_dir):
        if not isinstance(event, dict):
            continue
        if workspace_key and str(event.get('workspace') or '').strip() != workspace_key:
            continue
        if slot_key and str(event.get('slot_id') or '').strip() != slot_key:
            continue
        push(event.get('from_conversation_id'))
        push(event.get('to_conversation_id'))
    return ids


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sanitize_scalar_mapping(value: Any, *, allow_none: bool = False) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or '').strip()
        if not key:
            continue
        if raw_value is None:
            if allow_none:
                payload[key] = None
            continue
        if isinstance(raw_value, bool):
            payload[key] = raw_value
            continue
        if isinstance(raw_value, (int, float)):
            payload[key] = raw_value
            continue
        if isinstance(raw_value, str):
            payload[key] = raw_value
    return payload or None


def _sanitize_request_snapshot_target(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload = {
        'instance_id': str(value.get('instance_id') or value.get('instanceId') or '').strip() or None,
        'model': str(value.get('model') or value.get('modelName') or '').strip() or None,
        'backend': str(value.get('backend') or '').strip() or None,
        'capability': str(value.get('capability') or '').strip() or None,
    }
    return {key: item for key, item in payload.items() if item not in (None, '')} or None


def _sanitize_request_snapshot_attachment(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload = {
        'name': str(value.get('name') or '').strip() or None,
        'kind': str(value.get('kind') or '').strip() or None,
        'local_path': str(value.get('local_path') or value.get('localPath') or '').strip() or None,
    }
    return {key: item for key, item in payload.items() if item not in (None, '')} or None


def _normalize_input_artifact_type(raw_type: Any, raw_kind: Any) -> Optional[str]:
    artifact_type = str(raw_type or '').strip().lower()
    if artifact_type == 'pdf':
        return 'document'
    if artifact_type:
        return artifact_type
    kind = str(raw_kind or '').strip().lower()
    if kind == 'image':
        return 'image'
    if kind == 'audio':
        return 'audio'
    if kind == 'text':
        return 'text'
    if kind == 'pdf':
        return 'document'
    return kind or None


def _normalize_artifact_availability(value: Any) -> Optional[str]:
    normalized = str(value or '').strip().lower()
    if normalized in {'available', 'missing', 'purged'}:
        return normalized
    return None


def _history_artifact_roots() -> set[Path]:
    app_root = Path(__file__).resolve().parent.parent
    return {
        (app_root / root).resolve()
        for root in (
            ARTIFACT_OUTPUTS_AUDIO_DIR,
            ARTIFACT_OUTPUTS_IMAGES_DIR,
            ARTIFACT_OUTPUTS_OCR_DIR,
            ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
            ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
            ARTIFACT_INPUTS_ROOT,
        )
    }


def _path_is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _artifact_candidate_paths(raw_path: Any) -> list[Path]:
    raw_value = str(raw_path or '').strip()
    if not raw_value:
        return []
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return [candidate.resolve(strict=False)]
    app_root = Path(__file__).resolve().parent.parent
    return [
        (Path.cwd() / candidate).resolve(strict=False),
        (app_root / candidate).resolve(strict=False),
    ]


def _detect_artifact_availability(path_value: Any, explicit_availability: Any) -> Optional[str]:
    explicit = _normalize_artifact_availability(explicit_availability)
    if explicit == 'purged':
        return 'purged'
    candidates = _artifact_candidate_paths(path_value)
    if not candidates:
        return explicit
    allowed_roots = _history_artifact_roots()
    within_allowed_roots = False
    for candidate in candidates:
        if not any(_path_is_within_root(candidate, root) for root in allowed_roots):
            continue
        within_allowed_roots = True
        if candidate.exists() and candidate.is_file():
            return 'available'
    if within_allowed_roots:
        return 'missing'
    return explicit


def _annotate_artifact_availability(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = dict(artifact or {})
    availability = _detect_artifact_availability(payload.get('path'), payload.get('availability'))
    if availability:
        payload['availability'] = availability
    checked_at = str(payload.get('availability_checked_at') or '').strip() or None
    if checked_at:
        payload['availability_checked_at'] = checked_at
    if availability == 'purged':
        purged_at = str(payload.get('purged_at') or '').strip() or None
        if purged_at:
            payload['purged_at'] = purged_at
        purge_reason = str(payload.get('purge_reason') or '').strip() or None
        if purge_reason:
            payload['purge_reason'] = purge_reason
    return payload


def _sanitize_request_snapshot_input_artifact(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    sanitized = sanitize_artifact_record(
        {
            'type': _normalize_input_artifact_type(value.get('type'), value.get('kind')),
            'path': str(value.get('path') or '').strip() or None,
            'name': str(value.get('name') or '').strip() or None,
            'kind': str(value.get('kind') or '').strip().lower() or None,
            'origin': str(value.get('origin') or '').strip().lower() or None,
            'artifact_id': str(value.get('artifact_id') or value.get('artifactId') or '').strip() or None,
            'artifact_ref': str(value.get('artifact_ref') or value.get('artifactRef') or value.get('ref') or '').strip() or None,
            'source_path': str(
                value.get('source_path')
                or value.get('sourcePath')
                or value.get('local_path')
                or value.get('localPath')
                or ''
            ).strip() or None,
            'mime_type': str(value.get('mime_type') or value.get('mimeType') or '').strip() or None,
            'availability': _normalize_artifact_availability(value.get('availability')),
            'purged_at': str(value.get('purged_at') or value.get('purgedAt') or '').strip() or None,
            'purge_reason': str(value.get('purge_reason') or value.get('purgeReason') or '').strip() or None,
            'availability_checked_at': str(
                value.get('availability_checked_at')
                or value.get('availabilityCheckedAt')
                or ''
            ).strip() or None,
        },
        default_origin='user_input',
    )
    return _annotate_artifact_availability(sanitized) if sanitized else None


def _sanitize_request_snapshot_input_artifacts(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    payload: list[dict[str, Any]] = []
    for item in items:
        sanitized = _sanitize_request_snapshot_input_artifact(item)
        if sanitized:
            payload.append(sanitized)
    return payload


def _sanitize_message_artifact(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    artifact_type = _normalize_input_artifact_type(value.get('type'), value.get('kind'))
    if not artifact_type:
        return None
    payload = sanitize_artifact_record(
        {
            'type': artifact_type,
            'path': value.get('path'),
            'name': value.get('name'),
            'kind': value.get('kind'),
            'origin': value.get('origin'),
            'artifact_id': value.get('artifact_id') if value.get('artifact_id') is not None else value.get('artifactId'),
            'artifact_ref': (
                value.get('artifact_ref')
                if value.get('artifact_ref') is not None
                else value.get('artifactRef')
            ),
            'source_path': value.get('source_path') if value.get('source_path') is not None else value.get('sourcePath'),
            'mime_type': value.get('mime_type') if value.get('mime_type') is not None else value.get('mimeType'),
            'seed': value.get('seed'),
            'batch_index': value.get('batch_index'),
            'prompt': value.get('prompt'),
            'availability': _normalize_artifact_availability(value.get('availability')),
            'purged_at': value.get('purged_at') if value.get('purged_at') is not None else value.get('purgedAt'),
            'purge_reason': value.get('purge_reason') if value.get('purge_reason') is not None else value.get('purgeReason'),
            'availability_checked_at': (
                value.get('availability_checked_at')
                if value.get('availability_checked_at') is not None
                else value.get('availabilityCheckedAt')
            ),
            'image_state': value.get('image_state'),
        },
        default_origin='assistant_output',
    )
    if not payload:
        return None
    return _annotate_artifact_availability(payload)


def _artifact_identity_key(artifact: dict[str, Any]) -> tuple[str, str, str, str]:
    path_identity = str(artifact.get('path') or artifact.get('source_path') or '').strip()
    if path_identity:
        return (
            str(artifact.get('type') or '').strip().lower(),
            'path',
            path_identity,
            '',
        )
    identity = (
        str(artifact.get('artifact_id') or '').strip()
        or str(artifact.get('artifact_ref') or artifact.get('ref') or '').strip()
        or str(artifact.get('name') or '').strip()
    )
    return (
        str(artifact.get('type') or '').strip().lower(),
        'identity',
        identity,
        str(artifact.get('origin') or '').strip().lower(),
    )


def _merge_artifact_records(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value in (None, '', []):
            continue
        if merged.get(key) in (None, '', []):
            merged[key] = value
            continue
        if key == 'availability':
            current = _normalize_artifact_availability(merged.get(key))
            next_value = _normalize_artifact_availability(value)
            if current != 'purged' and next_value == 'purged':
                merged[key] = next_value
            elif current == 'available' and next_value == 'missing':
                merged[key] = next_value
    return _annotate_artifact_availability(merged)


def _append_canonical_artifact(target: list[dict[str, Any]], artifact: Any) -> None:
    sanitized = _sanitize_message_artifact(artifact)
    if not sanitized:
        return
    key = _artifact_identity_key(sanitized)
    for index, existing in enumerate(target):
        if _artifact_identity_key(existing) == key:
            target[index] = _merge_artifact_records(existing, sanitized)
            return
    target.append(sanitized)


def _build_canonical_message_artifacts(
    message: dict[str, Any],
    *,
    request_snapshot: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    role = str(message.get('role') or '').strip().lower()
    request_snapshot_input_artifacts = (
        _sanitize_request_snapshot_input_artifacts(request_snapshot.get('input_artifacts'))
        if isinstance(request_snapshot, dict)
        else []
    )
    if role == 'user':
        source_items = request_snapshot_input_artifacts or list(message.get('artifacts') or [])
        for item in source_items:
            _append_canonical_artifact(artifacts, item)
    else:
        for item in message.get('artifacts') or []:
            _append_canonical_artifact(artifacts, item)
        outputs = _sanitize_outputs(
            message.get('outputs')
            if message.get('outputs') is not None
            else message.get('canonical_outputs')
        )
        for output in outputs:
            for item in output.get('artifacts') or []:
                _append_canonical_artifact(artifacts, item)

    def append_fallback_artifact(artifact_type: str, *, path: Any = None) -> None:
        if not str(path or '').strip():
            return
        _append_canonical_artifact(
            artifacts,
            {
                'type': artifact_type,
                'path': str(path or '').strip() or None,
            },
        )

    append_fallback_artifact(
        'image',
        path=message.get('saved_image_path') or message.get('savedImagePath'),
    )
    append_fallback_artifact(
        'audio',
        path=message.get('saved_audio_path') or message.get('savedAudioPath'),
    )
    append_fallback_artifact(
        'text',
        path=message.get('saved_text_path') or message.get('savedTextPath'),
    )
    return artifacts


def _collect_response_frame_artifact_candidates(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if value in (None, '', [], {}) or depth > 4:
        return []
    if isinstance(value, list):
        candidates: list[dict[str, Any]] = []
        for item in value:
            candidates.extend(_collect_response_frame_artifact_candidates(item, depth=depth + 1))
        return candidates
    if not isinstance(value, dict):
        return []

    candidates: list[dict[str, Any]] = []
    direct_artifact = value.get('artifact') if isinstance(value.get('artifact'), dict) else None
    if direct_artifact:
        metadata = value.get('metadata') if isinstance(value.get('metadata'), dict) else {}
        candidates.append(
            {
                **direct_artifact,
                'availability': direct_artifact.get('availability') or metadata.get('availability'),
                'artifact_id': direct_artifact.get('artifact_id') or value.get('artifact_id') or value.get('artifactId'),
                'artifact_ref': (
                    direct_artifact.get('artifact_ref')
                    or direct_artifact.get('artifactRef')
                    or value.get('artifact_ref')
                    or value.get('artifactRef')
                ),
                'kind': direct_artifact.get('kind') or value.get('type') or value.get('kind'),
                'origin': direct_artifact.get('origin') or value.get('origin'),
            }
        )
    else:
        has_type = bool(str(value.get('type') or value.get('kind') or '').strip())
        has_payload = any(
            str(value.get(key) or '').strip()
            for key in (
                'path',
                'source_path',
                'sourcePath',
                'image_data_url',
                'imageDataUrl',
                'saved_image_path',
                'savedImagePath',
            )
        )
        if has_type and has_payload:
            artifact = dict(value)
            if not artifact.get('path'):
                artifact['path'] = artifact.get('saved_image_path') or artifact.get('savedImagePath')
            candidates.append(artifact)

    for nested_key in ('dossiers', 'output', 'outputs', 'artifacts'):
        nested = value.get(nested_key)
        if nested is not None:
            if nested_key == 'dossiers' and isinstance(nested, dict):
                nested = list(nested.values())
            candidates.extend(_collect_response_frame_artifact_candidates(nested, depth=depth + 1))
    return candidates


def _load_latest_response_frame(response_id: str, *, history_dir: Path | str | None = None) -> Optional[dict[str, Any]]:
    target_response_id = str(response_id or '').strip()
    if not target_response_id:
        return None
    ledger_path = _response_frame_ledger_path(history_dir)
    if not ledger_path.exists():
        return None
    latest: Optional[dict[str, Any]] = None
    try:
        for raw_line in ledger_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or target_response_id not in line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and str(payload.get('response_id') or '').strip() == target_response_id:
                latest = payload
    except OSError:
        return None
    return latest


def _rehydrate_message_from_response_frame(
    message: dict[str, Any],
    *,
    history_dir: Path | str | None = None,
) -> dict[str, Any]:
    if not isinstance(message, dict):
        return message
    response_id = str(message.get('response_id') or message.get('responseId') or '').strip()
    if not response_id:
        return message
    frame = _load_latest_response_frame(response_id, history_dir=history_dir)
    if not isinstance(frame, dict):
        return message

    if not message.get('outputs'):
        output_frame = frame.get('output') if isinstance(frame.get('output'), dict) else {}
        outputs = _sanitize_outputs(output_frame.get('outputs'))
        if outputs:
            message['outputs'] = outputs
    if not message.get('output_slots'):
        planning = frame.get('planning') if isinstance(frame.get('planning'), dict) else {}
        artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), dict) else {}
        output_slots = _sanitize_output_slots(artifact_flow.get('output_slots'))
        if output_slots:
            message['output_slots'] = output_slots
    if not message.get('output_branches'):
        output_branches = _sanitize_output_branches(frame.get('output_branches'))
        if output_branches:
            message['output_branches'] = output_branches

    artifacts = _build_canonical_message_artifacts(message, request_snapshot=message.get('request_snapshot'))
    if not artifacts:
        for item in _collect_response_frame_artifact_candidates(frame.get('artifacts')):
            _append_canonical_artifact(artifacts, item)
    if artifacts:
        message['artifacts'] = artifacts
    return message


def _sanitize_request_snapshot_ghost_preview(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload = {
        'instance_id': str(value.get('instance_id') or value.get('instanceId') or '').strip() or None,
        'capability': str(value.get('capability') or '').strip() or None,
        'reuse_last_artifact': bool(value.get('reuse_last_artifact')) if 'reuse_last_artifact' in value else None,
        'artifact_ref': str(value.get('artifact_ref') or value.get('artifactRef') or '').strip() or None,
        'artifact_path': str(value.get('artifact_path') or '').strip() or None,
        'confidence': value.get('confidence') if isinstance(value.get('confidence'), (int, float)) else None,
        'reason': str(value.get('reason') or '').strip() or None,
        'route_source': str(value.get('route_source') or value.get('routeSource') or '').strip() or None,
    }
    return {
        key: item
        for key, item in payload.items()
        if item not in (None, '') or (key == 'reuse_last_artifact' and item is False)
    } or None


def _sanitize_request_snapshot_reference_artifact(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    artifact_type = _normalize_input_artifact_type(value.get('type'), value.get('kind'))
    if not artifact_type:
        return None
    if artifact_type == 'message':
        message_id = str(value.get('message_id') or value.get('messageId') or '').strip() or None
        content = str(value.get('content') or value.get('text') or value.get('prompt') or '').strip()
        if not content and not message_id:
            return None
        payload = sanitize_artifact_record(
            {
                'type': 'message',
                'content': content or None,
                'message_role': str(value.get('message_role') or value.get('messageRole') or value.get('role') or 'assistant').strip() or 'assistant',
                'source_message_id': message_id,
                'response_model': str(value.get('response_model') or value.get('responseModel') or '').strip() or None,
                'response_instance_id': str(value.get('response_instance_id') or value.get('responseInstanceId') or '').strip() or None,
                'timestamp': str(value.get('timestamp') or '').strip() or None,
                'artifact_ref': str(value.get('artifact_ref') or value.get('artifactRef') or value.get('ref') or '').strip() or None,
                'artifact_id': str(value.get('artifact_id') or value.get('artifactId') or '').strip() or None,
                'origin': 'conversation_reference',
            },
            default_kind='message',
            default_origin='conversation_reference',
            include_content=not bool(message_id),
        )
        return payload
    path = str(value.get('path') or '').strip()
    if not path:
        return None
    return sanitize_artifact_record(
        {
            'type': artifact_type,
            'path': path,
            'artifact_ref': str(value.get('artifact_ref') or value.get('artifactRef') or value.get('ref') or '').strip() or None,
            'artifact_id': str(value.get('artifact_id') or value.get('artifactId') or '').strip() or None,
            'source_message_id': str(value.get('message_id') or value.get('messageId') or '').strip() or None,
            'name': str(value.get('name') or '').strip() or None,
            'kind': str(value.get('kind') or '').strip() or None,
            'mime_type': str(value.get('mime_type') or value.get('mimeType') or '').strip() or None,
            'seed': value.get('seed'),
            'origin': 'conversation_reference',
        },
        default_origin='conversation_reference',
    )


def _sanitize_request_snapshot_reference_artifacts(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    payload: list[dict[str, Any]] = []
    for item in items:
        sanitized = _sanitize_request_snapshot_reference_artifact(item)
        if sanitized:
            payload.append(sanitized)
    return payload


def _sanitize_request_snapshot(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    batch_prompts = [
        str(item).strip()
        for item in (value.get('batch_prompts') or value.get('batchPrompts') or [])
        if str(item).strip()
    ] if isinstance(value.get('batch_prompts') or value.get('batchPrompts') or [], list) else []
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ('request_id', 'request_id'),
        ('requestId', 'request_id'),
        ('created_at', 'created_at'),
        ('createdAt', 'created_at'),
        ('conversation_id', 'conversation_id'),
        ('conversationId', 'conversation_id'),
        ('response_id', 'response_id'),
        ('responseId', 'response_id'),
        ('transport', 'transport'),
        ('prompt_text', 'prompt_text'),
        ('promptText', 'prompt_text'),
        ('prompt_preview', 'prompt_preview'),
        ('promptPreview', 'prompt_preview'),
    ):
        item = str(value.get(source_key) or '').strip()
        if item:
            payload[target_key] = item
    target = _sanitize_request_snapshot_target(value.get('target'))
    if target:
        payload['target'] = target
    attachment = _sanitize_request_snapshot_attachment(value.get('attachment'))
    if attachment:
        payload['attachment'] = attachment
    input_artifacts = _sanitize_request_snapshot_input_artifacts(
        value.get('input_artifacts')
        if value.get('input_artifacts') is not None
        else value.get('inputArtifacts')
    )
    if input_artifacts:
        payload['input_artifacts'] = input_artifacts
    session_controls = _sanitize_scalar_mapping(value.get('session_controls') or value.get('sessionControls'))
    if session_controls:
        payload['session_controls'] = session_controls
    settings = _sanitize_scalar_mapping(value.get('settings'), allow_none=True)
    if settings:
        payload['settings'] = settings
    ghost_preview = _sanitize_request_snapshot_ghost_preview(value.get('ghost_preview') or value.get('ghostPreview'))
    if ghost_preview:
        payload['ghost_preview'] = ghost_preview
    reference_artifacts = _sanitize_request_snapshot_reference_artifacts(
        value.get('reference_artifacts')
        if value.get('reference_artifacts') is not None
        else (
            value.get('selected_reference_artifacts')
            if value.get('selected_reference_artifacts') is not None
            else (
                value.get('referenceArtifacts')
                if value.get('referenceArtifacts') is not None
                else (
                    value.get('selectedReferenceArtifacts')
                    if value.get('selectedReferenceArtifacts') is not None
                    else value.get('selected_reference_artifact') or value.get('selectedReferenceArtifact')
                )
            )
        )
    )
    if reference_artifacts:
        payload['reference_artifacts'] = reference_artifacts
    if batch_prompts:
        payload['batch_prompts'] = batch_prompts
    return payload or None


def _sanitize_output_slot(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ('slot_id', 'slot_id'),
        ('slotId', 'slot_id'),
        ('type', 'type'),
        ('status', 'status'),
        ('lifecycle', 'lifecycle'),
        ('artifact_ref', 'artifact_ref'),
        ('artifactRef', 'artifact_ref'),
        ('placeholder_ref', 'placeholder_ref'),
        ('placeholderRef', 'placeholder_ref'),
        ('blocked_reason', 'blocked_reason'),
        ('blockedReason', 'blocked_reason'),
        ('follow_up_capability', 'follow_up_capability'),
        ('followUpCapability', 'follow_up_capability'),
        ('follow_up_source', 'follow_up_source'),
        ('followUpSource', 'follow_up_source'),
        ('branch_id', 'branch_id'),
        ('branchId', 'branch_id'),
        ('phase_id', 'phase_id'),
        ('phaseId', 'phase_id'),
        ('parent_slot_id', 'parent_slot_id'),
        ('parentSlotId', 'parent_slot_id'),
    ):
        raw = value.get(source_key)
        text = str(raw or '').strip()
        if text:
            payload[target_key] = text
    batch_index = value.get('batch_index') if value.get('batch_index') is not None else value.get('batchIndex')
    if batch_index not in (None, ''):
        try:
            parsed_batch_index = int(batch_index)
        except (TypeError, ValueError):
            parsed_batch_index = None
        if parsed_batch_index is not None and parsed_batch_index >= 0:
            payload['batch_index'] = parsed_batch_index
    child_slot_ids = value.get('child_slot_ids') if value.get('child_slot_ids') is not None else value.get('childSlotIds')
    if isinstance(child_slot_ids, list):
        normalized_child_slot_ids = [
            str(item).strip()
            for item in child_slot_ids
            if str(item).strip()
        ]
        if normalized_child_slot_ids:
            payload['child_slot_ids'] = normalized_child_slot_ids
    error_ref = value.get('error_ref') if isinstance(value.get('error_ref'), dict) else value.get('errorRef')
    if isinstance(error_ref, dict):
        normalized_error_ref = {
            'branch_id': str(error_ref.get('branch_id') or error_ref.get('branchId') or '').strip() or None,
            'code': str(error_ref.get('code') or '').strip() or None,
            'stage': str(error_ref.get('stage') or '').strip() or None,
        }
        payload['error_ref'] = {
            key: value
            for key, value in normalized_error_ref.items()
            if value not in (None, '', [], {})
        }
    recovery_context = (
        value.get('recovery_context')
        if isinstance(value.get('recovery_context'), dict)
        else value.get('recoveryContext')
    )
    if isinstance(recovery_context, dict):
        normalized_recovery_context = {
            'can_retry': recovery_context.get('can_retry')
            if isinstance(recovery_context.get('can_retry'), bool)
            else recovery_context.get('canRetry')
            if isinstance(recovery_context.get('canRetry'), bool)
            else None,
            'retry_scope': str(recovery_context.get('retry_scope') or recovery_context.get('retryScope') or '').strip() or None,
            'suggested_action': str(recovery_context.get('suggested_action') or recovery_context.get('suggestedAction') or '').strip() or None,
            'preserve_intent': recovery_context.get('preserve_intent')
            if isinstance(recovery_context.get('preserve_intent'), bool)
            else recovery_context.get('preserveIntent')
            if isinstance(recovery_context.get('preserveIntent'), bool)
            else None,
            'exclude_instance_ids': [
                str(item).strip()
                for item in (
                    recovery_context.get('exclude_instance_ids')
                    if isinstance(recovery_context.get('exclude_instance_ids'), list)
                    else recovery_context.get('excludeInstanceIds')
                    if isinstance(recovery_context.get('excludeInstanceIds'), list)
                    else []
                )
                if str(item).strip()
            ],
        }
        payload['recovery_context'] = {
            key: value
            for key, value in normalized_recovery_context.items()
            if value not in (None, '', [], {})
        }
    return payload or None


def _sanitize_output_slots(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    payload: list[dict[str, Any]] = []
    for item in value:
        sanitized = _sanitize_output_slot(item)
        if sanitized:
            payload.append(sanitized)
    return payload


def _sanitize_output_branch(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ('slot_id', 'slot_id'),
        ('slotId', 'slot_id'),
        ('branch_id', 'branch_id'),
        ('branchId', 'branch_id'),
        ('phase_id', 'phase_id'),
        ('phaseId', 'phase_id'),
        ('type', 'type'),
        ('status', 'status'),
        ('lifecycle', 'lifecycle'),
        ('follow_up_capability', 'follow_up_capability'),
        ('followUpCapability', 'follow_up_capability'),
        ('artifact_ref', 'artifact_ref'),
        ('artifactRef', 'artifact_ref'),
        ('placeholder_ref', 'placeholder_ref'),
        ('placeholderRef', 'placeholder_ref'),
        ('blocked_reason', 'blocked_reason'),
        ('blockedReason', 'blocked_reason'),
        ('parent_slot_id', 'parent_slot_id'),
        ('parentSlotId', 'parent_slot_id'),
    ):
        raw = value.get(source_key)
        text = str(raw or '').strip()
        if text:
            payload[target_key] = text
    child_slot_ids = value.get('child_slot_ids') if value.get('child_slot_ids') is not None else value.get('childSlotIds')
    if isinstance(child_slot_ids, list):
        normalized_child_slot_ids = [
            str(item).strip()
            for item in child_slot_ids
            if str(item).strip()
        ]
        if normalized_child_slot_ids:
            payload['child_slot_ids'] = normalized_child_slot_ids
    error_ref = value.get('error_ref') if isinstance(value.get('error_ref'), dict) else value.get('errorRef')
    if isinstance(error_ref, dict):
        normalized_error_ref = {
            'branch_id': str(error_ref.get('branch_id') or error_ref.get('branchId') or '').strip() or None,
            'code': str(error_ref.get('code') or '').strip() or None,
            'stage': str(error_ref.get('stage') or '').strip() or None,
        }
        payload['error_ref'] = {
            key: value
            for key, value in normalized_error_ref.items()
            if value not in (None, '', [], {})
        }
    recovery_context = (
        value.get('recovery_context')
        if isinstance(value.get('recovery_context'), dict)
        else value.get('recoveryContext')
    )
    if isinstance(recovery_context, dict):
        normalized_recovery_context = {
            'can_retry': recovery_context.get('can_retry')
            if isinstance(recovery_context.get('can_retry'), bool)
            else recovery_context.get('canRetry')
            if isinstance(recovery_context.get('canRetry'), bool)
            else None,
            'retry_scope': str(recovery_context.get('retry_scope') or recovery_context.get('retryScope') or '').strip() or None,
            'suggested_action': str(recovery_context.get('suggested_action') or recovery_context.get('suggestedAction') or '').strip() or None,
            'preserve_intent': recovery_context.get('preserve_intent')
            if isinstance(recovery_context.get('preserve_intent'), bool)
            else recovery_context.get('preserveIntent')
            if isinstance(recovery_context.get('preserveIntent'), bool)
            else None,
            'exclude_instance_ids': [
                str(item).strip()
                for item in (
                    recovery_context.get('exclude_instance_ids')
                    if isinstance(recovery_context.get('exclude_instance_ids'), list)
                    else recovery_context.get('excludeInstanceIds')
                    if isinstance(recovery_context.get('excludeInstanceIds'), list)
                    else []
                )
                if str(item).strip()
            ],
        }
        payload['recovery_context'] = {
            key: value
            for key, value in normalized_recovery_context.items()
            if value not in (None, '', [], {})
        }
    return payload or None


def _sanitize_output_branches(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    payload: list[dict[str, Any]] = []
    for item in value:
        sanitized = _sanitize_output_branch(item)
        if sanitized:
            payload.append(sanitized)
    return payload


def _sanitize_output(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ('slot_id', 'slot_id'),
        ('slotId', 'slot_id'),
        ('branch_id', 'branch_id'),
        ('branchId', 'branch_id'),
        ('phase_id', 'phase_id'),
        ('phaseId', 'phase_id'),
        ('type', 'type'),
        ('status', 'status'),
        ('lifecycle', 'lifecycle'),
        ('artifact_ref', 'artifact_ref'),
        ('artifactRef', 'artifact_ref'),
        ('placeholder_ref', 'placeholder_ref'),
        ('placeholderRef', 'placeholder_ref'),
        ('parent_slot_id', 'parent_slot_id'),
        ('parentSlotId', 'parent_slot_id'),
        ('follow_up_capability', 'follow_up_capability'),
        ('followUpCapability', 'follow_up_capability'),
    ):
        raw = value.get(source_key)
        text = str(raw or '').strip()
        if text:
            payload[target_key] = text
    output_value = str(value.get('value') or '').strip()
    if output_value:
        payload['value'] = output_value
    batch_index = value.get('batch_index') if value.get('batch_index') is not None else value.get('batchIndex')
    if batch_index not in (None, ''):
        try:
            parsed_batch_index = int(batch_index)
        except (TypeError, ValueError):
            parsed_batch_index = None
        if parsed_batch_index is not None and parsed_batch_index >= 0:
            payload['batch_index'] = parsed_batch_index
    child_slot_ids = value.get('child_slot_ids') if value.get('child_slot_ids') is not None else value.get('childSlotIds')
    if isinstance(child_slot_ids, list):
        normalized_child_slot_ids = [
            str(item).strip()
            for item in child_slot_ids
            if str(item).strip()
        ]
        if normalized_child_slot_ids:
            payload['child_slot_ids'] = normalized_child_slot_ids
    artifacts = value.get('artifacts')
    if isinstance(artifacts, list):
        normalized_artifacts = []
        for item in artifacts:
            artifact = sanitize_artifact_record(item)
            if artifact:
                normalized_artifacts.append(artifact)
        if normalized_artifacts:
            payload['artifacts'] = normalized_artifacts
    return payload or None


def _sanitize_outputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    payload: list[dict[str, Any]] = []
    for item in value:
        sanitized = _sanitize_output(item)
        if sanitized:
            payload.append(sanitized)
    return payload


def _sanitize_bundle_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or '').strip()
            if key:
                payload[key] = _sanitize_bundle_json_value(raw_value)
        return payload
    if isinstance(value, list):
        return [_sanitize_bundle_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _sanitize_artifact_bundle(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ('status', 'status'),
        ('bundle_id', 'bundle_id'),
        ('bundleId', 'bundle_id'),
        ('bundle_path', 'bundle_path'),
        ('bundlePath', 'bundle_path'),
        ('entrypoint', 'entrypoint'),
        ('entrypoint_relative_path', 'entrypoint_relative_path'),
        ('entrypointRelativePath', 'entrypoint_relative_path'),
        ('manifest_path', 'manifest_path'),
        ('manifestPath', 'manifest_path'),
        ('source_response_id', 'source_response_id'),
        ('sourceResponseId', 'source_response_id'),
        ('source_artifact_refs', 'source_artifact_refs'),
        ('sourceArtifactRefs', 'source_artifact_refs'),
        ('copied_artifacts', 'copied_artifacts'),
        ('copiedArtifacts', 'copied_artifacts'),
        ('rewritten_links', 'rewritten_links'),
        ('rewrittenLinks', 'rewritten_links'),
        ('link_check', 'link_check'),
        ('linkCheck', 'link_check'),
    ):
        raw_value = value.get(source_key)
        if raw_value not in (None, '', [], {}):
            payload[target_key] = _sanitize_bundle_json_value(raw_value)
    if not (
        str(payload.get('bundle_id') or '').strip()
        or str(payload.get('bundle_path') or '').strip()
        or str(payload.get('entrypoint') or '').strip()
    ):
        return None
    return payload


def _artifact_bundle_key(bundle: dict[str, Any]) -> str:
    return str(
        bundle.get('bundle_id')
        or bundle.get('bundle_path')
        or bundle.get('entrypoint')
        or ''
    ).strip()


def _sanitize_artifact_bundles(value: Any) -> list[dict[str, Any]]:
    candidates = value if isinstance(value, list) else [value]
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        sanitized = _sanitize_artifact_bundle(item)
        if not sanitized:
            continue
        key = _artifact_bundle_key(sanitized)
        if not key or key in seen:
            continue
        seen.add(key)
        payload.append(sanitized)
    return payload


def _sanitize_message(message: dict) -> Optional[dict]:
    if not isinstance(message, dict):
        return None
    role = str(message.get('role') or '').strip()
    content = str(message.get('content') or '')
    timestamp = str(message.get('timestamp') or message.get('ts') or '').strip() or _now_iso()
    if not role:
        return None
    payload = {
        'role': role,
        'content': content,
        'timestamp': timestamp,
        'message_id': str(message.get('message_id') or message.get('messageId') or message.get('clientMessageId') or '').strip() or None,
    }
    for source_key, target_key in (
        ('savedImagePath', 'saved_image_path'),
        ('saved_image_path', 'saved_image_path'),
        ('savedAudioPath', 'saved_audio_path'),
        ('saved_audio_path', 'saved_audio_path'),
        ('savedTextPath', 'saved_text_path'),
        ('saved_text_path', 'saved_text_path'),
        ('responseId', 'response_id'),
        ('response_id', 'response_id'),
        ('responseModel', 'response_model'),
        ('response_model', 'response_model'),
        ('responseBackend', 'response_backend'),
        ('response_backend', 'response_backend'),
        ('responseCapability', 'response_capability'),
        ('response_capability', 'response_capability'),
        ('responseInstanceId', 'response_instance_id'),
        ('response_instance_id', 'response_instance_id'),
        ('routeSource', 'route_source'),
        ('route_source', 'route_source'),
        ('routeReason', 'route_reason'),
        ('route_reason', 'route_reason'),
        ('routeArtifactRef', 'route_artifact_ref'),
        ('route_artifact_ref', 'route_artifact_ref'),
        ('routeArtifactPath', 'route_artifact_path'),
        ('route_artifact_path', 'route_artifact_path'),
        ('routeReuseLastArtifact', 'route_reuse_last_artifact'),
        ('route_reuse_last_artifact', 'route_reuse_last_artifact'),
        ('referenceImageCount', 'reference_image_count'),
        ('reference_image_count', 'reference_image_count'),
        ('referenceImageKind', 'reference_image_kind'),
        ('reference_image_kind', 'reference_image_kind'),
        ('contextMode', 'context_mode'),
        ('context_mode', 'context_mode'),
        ('contextReason', 'context_reason'),
        ('context_reason', 'context_reason'),
    ):
        value = message.get(source_key)
        if value not in (None, '', []):
            payload[target_key] = value
    request_snapshot = _sanitize_request_snapshot(
        message.get('request_snapshot')
        if message.get('request_snapshot') is not None
        else message.get('requestSnapshot')
    )
    sanitized_artifacts = _build_canonical_message_artifacts(message, request_snapshot=request_snapshot)
    if sanitized_artifacts:
        payload['artifacts'] = sanitized_artifacts
    if request_snapshot:
        payload['request_snapshot'] = request_snapshot
    output_slots = _sanitize_output_slots(
        message.get('output_slots')
        if message.get('output_slots') is not None
        else message.get('outputSlots')
    )
    if output_slots:
        payload['output_slots'] = output_slots
    output_branches = _sanitize_output_branches(
        message.get('output_branches')
        if message.get('output_branches') is not None
        else message.get('outputBranches')
    )
    if output_branches:
        payload['output_branches'] = output_branches
    outputs = _sanitize_outputs(
        message.get('outputs')
        if message.get('outputs') is not None
        else message.get('canonical_outputs')
    )
    if outputs:
        payload['outputs'] = outputs
    artifact_bundles = _sanitize_artifact_bundles(
        message.get('artifact_bundles')
        if message.get('artifact_bundles') is not None
        else (
            message.get('artifactBundles')
            if message.get('artifactBundles') is not None
            else message.get('artifactBundle')
        )
    )
    if artifact_bundles:
        payload['artifact_bundles'] = artifact_bundles
    return payload


def _sanitize_conversation_metadata(
    metadata: Any,
    *,
    fallback_instance_id: str | None = None,
) -> Optional[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ('workspace', 'workspace'),
        ('slot_id', 'slot_id'),
        ('source_instance_id', 'source_instance_id'),
        ('label', 'label'),
        ('parent_conversation_id', 'parent_conversation_id'),
        ('root_conversation_id', 'root_conversation_id'),
        ('created_at', 'created_at'),
        ('display_title', 'display_title'),
        ('preview_text', 'preview_text'),
        ('last_message_at', 'last_message_at'),
    ):
        value = str(metadata.get(source_key) or '').strip()
        if value:
            payload[target_key] = value
    message_count = metadata.get('message_count')
    if message_count not in (None, ''):
        try:
            parsed_count = int(message_count)
        except (TypeError, ValueError):
            parsed_count = None
        if parsed_count is not None and parsed_count >= 0:
            payload['message_count'] = parsed_count
    fresh_root = metadata.get('fresh_root')
    if isinstance(fresh_root, bool):
        payload['fresh_root'] = fresh_root
    if payload.get('workspace') == 'instance' and not payload.get('source_instance_id') and fallback_instance_id:
        payload['source_instance_id'] = str(fallback_instance_id).strip()
    return payload or None


def _derive_history_index_metadata(
    instance_id: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    *,
    updated_at: Any = None,
) -> dict[str, Any] | None:
    normalized = _sanitize_conversation_metadata(metadata, fallback_instance_id=instance_id) or {}
    persisted_messages = [item for item in messages if isinstance(item, dict)]
    if not normalized.get('display_title'):
        normalized['display_title'] = (
            _derive_history_preview_text(persisted_messages, role='user')
            or _derive_history_preview_text(persisted_messages)
            or None
        )
    if not normalized.get('preview_text'):
        normalized['preview_text'] = (
            _derive_history_preview_text(persisted_messages, role='user', reverse=True)
            or _derive_history_preview_text(persisted_messages, reverse=True)
            or normalized.get('display_title')
            or None
        )
    if normalized.get('message_count') in (None, ''):
        normalized['message_count'] = len(persisted_messages)
    if not normalized.get('last_message_at'):
        for message in reversed(persisted_messages):
            timestamp = str(message.get('timestamp') or '').strip()
            if timestamp:
                normalized['last_message_at'] = timestamp
                break
    if not normalized.get('last_message_at'):
        timestamp = str(updated_at or '').strip()
        if timestamp:
            normalized['last_message_at'] = timestamp
    return normalized or None


def _default_rotation_workspace(instance_id: str) -> str:
    normalized = str(instance_id or '').strip()
    return 'responses' if normalized.startswith('__responses_workbench__') else 'instance'


def _default_slot_id(workspace: str, source_instance_id: str | None = None) -> str:
    if workspace == 'responses':
        return 'responses-workbench'
    source_key = str(source_instance_id or '').strip() or 'conversation'
    return f'instance:{source_key}'


def _build_rotated_conversation_id(workspace: str, source_instance_id: str | None = None) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    token = uuid.uuid4().hex[:8]
    if workspace == 'responses':
        return f'__responses_workbench__--{stamp}-{token}'
    source_key = _slug(source_instance_id or 'conversation')
    return f'__instance_chat__--{source_key}--{stamp}-{token}'


def read_chat_history(instance_id: str, *, history_dir: Path | str | None = None) -> dict:
    target = _history_path(instance_id, history_dir)
    if not target.exists():
        return {
            'instance_id': instance_id,
            'messages': [],
            'updated_at': None,
            'conversation_metadata': None,
        }
    payload = _load_payload(target)
    if not payload:
        return {
            'instance_id': instance_id,
            'messages': [],
            'updated_at': None,
            'conversation_metadata': None,
        }
    messages = []
    for item in payload.get('messages') or []:
        sanitized = _sanitize_message(item)
        if sanitized:
            messages.append(_rehydrate_message_from_response_frame(sanitized, history_dir=history_dir))
    conversation_metadata = _sanitize_conversation_metadata(
        payload.get('conversation_metadata'),
        fallback_instance_id=instance_id,
    )
    return {
        'instance_id': str(payload.get('instance_id') or instance_id),
        'model': payload.get('model'),
        'backend': payload.get('backend'),
        'capability': payload.get('capability'),
        'updated_at': payload.get('updated_at'),
        'conversation_metadata': conversation_metadata,
        'messages': messages,
    }


def write_chat_history(
    instance_id: str,
    messages: list[dict],
    *,
    history_dir: Path | str | None = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    capability: Optional[str] = None,
    conversation_metadata: Optional[dict[str, Any]] = None,
) -> dict:
    sanitized_messages = []
    for item in messages or []:
        sanitized = _sanitize_message(item)
        if sanitized and not item.get('isLoading'):
            sanitized_messages.append(sanitized)
    target = _history_path(instance_id, history_dir)
    existing_payload = _load_payload(target)
    existing_metadata = _sanitize_conversation_metadata(
        existing_payload.get('conversation_metadata'),
        fallback_instance_id=instance_id,
    )
    incoming_metadata = _sanitize_conversation_metadata(
        conversation_metadata,
        fallback_instance_id=instance_id,
    )
    merged_metadata = None
    if existing_metadata or incoming_metadata:
        merged_metadata = {
            **(existing_metadata or {}),
            **(incoming_metadata or {}),
        }
        if 'created_at' not in merged_metadata:
            merged_metadata['created_at'] = _now_iso()
    payload = {
        'instance_id': instance_id,
        'model': model,
        'backend': backend,
        'capability': capability,
        'updated_at': _now_iso(),
        'messages': sanitized_messages,
    }
    if merged_metadata:
        payload['conversation_metadata'] = merged_metadata
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return payload


def append_chat_history_lineage(event: dict[str, Any], *, history_dir: Path | str | None = None) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in event.items()
        if value not in (None, '', [], {})
    }
    target = _lineage_path(history_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
    return payload


def rotate_chat_history(
    current_instance_id: str,
    *,
    history_dir: Path | str | None = None,
    workspace: str | None = None,
    slot_id: str | None = None,
    source_instance_id: str | None = None,
    label: str | None = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    capability: Optional[str] = None,
    fresh_root: bool = False,
) -> dict:
    current_key = str(current_instance_id or '').strip()
    if not current_key:
        raise ValueError("current_instance_id is required")
    current_history = read_chat_history(current_key, history_dir=history_dir)
    current_metadata = current_history.get('conversation_metadata') or {}
    workspace_key = str(workspace or current_metadata.get('workspace') or '').strip() or _default_rotation_workspace(current_key)
    source_key = str(source_instance_id or current_metadata.get('source_instance_id') or '').strip()
    if workspace_key == 'instance' and not source_key:
        source_key = current_key
    slot_key = str(slot_id or current_metadata.get('slot_id') or '').strip() or _default_slot_id(workspace_key, source_key)
    label_key = str(label or current_metadata.get('label') or '').strip() or slot_key
    successor_id = _build_rotated_conversation_id(workspace_key, source_key)
    root_key = successor_id if fresh_root else (str(current_metadata.get('root_conversation_id') or '').strip() or current_key)
    successor_metadata = {
        'workspace': workspace_key,
        'slot_id': slot_key,
        'source_instance_id': source_key,
        'label': label_key,
        'root_conversation_id': root_key,
        'created_at': _now_iso(),
    }
    if fresh_root:
        successor_metadata['fresh_root'] = True
    else:
        successor_metadata['parent_conversation_id'] = current_key
    successor_history = write_chat_history(
        successor_id,
        [],
        history_dir=history_dir,
        model=model if model is not None else current_history.get('model'),
        backend=backend if backend is not None else current_history.get('backend'),
        capability=capability if capability is not None else current_history.get('capability'),
        conversation_metadata=successor_metadata,
    )
    lineage_event = append_chat_history_lineage(
        {
            'rotated_at': successor_metadata['created_at'],
            'from_conversation_id': current_key,
            'to_conversation_id': successor_id,
            'workspace': workspace_key,
            'slot_id': slot_key,
            'source_instance_id': source_key,
            'label': label_key,
            'root_conversation_id': root_key,
            'fresh_root': bool(fresh_root),
            'model': successor_history.get('model'),
            'backend': successor_history.get('backend'),
            'capability': successor_history.get('capability'),
        },
        history_dir=history_dir,
    )
    if not fresh_root:
        lineage_event['parent_conversation_id'] = current_key
    return {
        **successor_history,
        'rotation_event': lineage_event,
    }


def resolve_chat_history_slot(
    *,
    workspace: str | None = None,
    slot_id: str | None = None,
    history_dir: Path | str | None = None,
    fallback_instance_id: str | None = None,
) -> dict[str, Any]:
    workspace_key = str(workspace or '').strip()
    slot_key = str(slot_id or '').strip()
    resolved_instance_id = str(fallback_instance_id or '').strip()

    for event in reversed(_read_lineage_events(history_dir)):
        if not isinstance(event, dict):
            continue
        if workspace_key and str(event.get('workspace') or '').strip() != workspace_key:
            continue
        if slot_key and str(event.get('slot_id') or '').strip() != slot_key:
            continue
        candidate = str(event.get('to_conversation_id') or '').strip()
        if candidate:
            resolved_instance_id = candidate
            break

    if not resolved_instance_id and workspace_key == 'responses' and slot_key == 'responses-workbench':
        resolved_instance_id = '__responses_workbench__'

    history = read_chat_history(resolved_instance_id or (fallback_instance_id or ''), history_dir=history_dir)
    history['slot_history_ids'] = _build_slot_history_ids(
        workspace=workspace_key,
        slot_id=slot_key,
        history_dir=history_dir,
        fallback_instance_id=fallback_instance_id,
    )
    history['resolved_slot'] = {
        'workspace': workspace_key or None,
        'slot_id': slot_key or None,
        'fallback_instance_id': str(fallback_instance_id or '').strip() or None,
    }
    return history


def list_chat_history_index(
    *,
    history_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    base = Path(history_dir) if history_dir else DEFAULT_CHAT_HISTORY_DIR
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for target in base.glob('*.json'):
        if not target.is_file():
            continue
        payload = _load_payload(target)
        if not payload:
            continue
        instance_id = str(payload.get('instance_id') or '').strip()
        if not instance_id:
            continue
        messages = []
        for item in payload.get('messages') or []:
            sanitized = _sanitize_message(item)
            if sanitized:
                messages.append(sanitized)
        metadata = _derive_history_index_metadata(
            instance_id,
            messages,
            payload.get('conversation_metadata'),
            updated_at=payload.get('updated_at'),
        )
        message_count = int(metadata.get('message_count') or 0) if isinstance(metadata, dict) else 0
        if message_count <= 0:
            continue
        items.append({
            'instance_id': instance_id,
            'model': str(payload.get('model') or '').strip() or None,
            'backend': str(payload.get('backend') or '').strip() or None,
            'capability': str(payload.get('capability') or '').strip() or None,
            'updated_at': str(payload.get('updated_at') or '').strip() or None,
            'conversation_metadata': metadata,
        })

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        metadata = item.get('conversation_metadata') if isinstance(item, dict) else {}
        timestamp = ''
        if isinstance(metadata, dict):
            timestamp = str(metadata.get('last_message_at') or metadata.get('created_at') or '').strip()
        if not timestamp:
            timestamp = str(item.get('updated_at') or '').strip()
        try:
            parsed = dt.datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp() if timestamp else 0
        except ValueError:
            parsed = 0
        return (int(parsed), str(item.get('instance_id') or ''))

    items.sort(key=sort_key, reverse=True)
    return items


def delete_chat_history(instance_id: str, *, history_dir: Path | str | None = None) -> bool:
    target = _history_path(instance_id, history_dir)
    if not target.exists():
        return False
    target.unlink()
    return True
