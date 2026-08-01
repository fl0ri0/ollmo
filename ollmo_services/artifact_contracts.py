"""Shared durable artifact-contract helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Iterable, Optional


def clean_text(value: Any) -> str:
    return str(value or '').strip()


def clean_artifact_kind(raw_type: Any, raw_kind: Any = None) -> Optional[str]:
    artifact_type = clean_text(raw_type).lower()
    if artifact_type == 'pdf':
        return 'document'
    if artifact_type:
        return artifact_type
    kind = clean_text(raw_kind).lower()
    if kind == 'pdf':
        return 'document'
    return kind or None


def normalize_artifact_availability(value: Any) -> Optional[str]:
    normalized = clean_text(value).lower()
    if normalized in {'available', 'missing', 'provided', 'purged'}:
        return normalized
    return None


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]


def _clean_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = clean_text(item)
        if not token or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def build_artifact_id(value: Mapping[str, Any], *, kind: Optional[str] = None) -> str:
    for key in ('artifact_id', 'artifactId', 'id'):
        token = clean_text(value.get(key))
        if token:
            return token

    normalized_kind = kind or clean_artifact_kind(value.get('type'), value.get('kind')) or 'artifact'
    provenance_id = clean_text(value.get('provenance_id') or value.get('provenanceId'))
    if provenance_id:
        return f'{normalized_kind}_{provenance_id}'
    message_id = clean_text(
        value.get('message_id')
        or value.get('messageId')
        or value.get('source_message_id')
        or value.get('sourceMessageId')
    )
    if normalized_kind == 'message' and message_id:
        return f'message_{message_id}'

    identity_payload = {
        'kind': normalized_kind,
        'origin': clean_text(value.get('origin')).lower() or None,
        'path': clean_text(value.get('path')) or None,
        'source_path': clean_text(value.get('source_path') or value.get('sourcePath')) or None,
        'name': clean_text(value.get('name')) or None,
        'mime_type': clean_text(value.get('mime_type') or value.get('mimeType')) or None,
        'source_message_id': message_id or None,
        'source_response_id': clean_text(
            value.get('source_response_id')
            or value.get('sourceResponseId')
            or value.get('response_id')
            or value.get('responseId')
        ) or None,
        'seed': value.get('seed') if isinstance(value.get('seed'), (int, float)) else None,
        'content': clean_text(value.get('content') or value.get('text') or value.get('prompt')) or None,
    }
    digest = _content_digest(json.dumps(identity_payload, sort_keys=True, ensure_ascii=False))
    return f'{normalized_kind}_{digest}'


def build_artifact_ref(value: Mapping[str, Any], *, artifact_id: Optional[str] = None, kind: Optional[str] = None) -> str:
    for key in ('artifact_ref', 'artifactRef', 'ref'):
        token = clean_text(value.get(key))
        if token:
            return token
    normalized_kind = kind or clean_artifact_kind(value.get('type'), value.get('kind')) or 'artifact'
    resolved_id = artifact_id or build_artifact_id(value, kind=normalized_kind)
    prefix = 'message' if normalized_kind == 'message' else 'artifact'
    return f'{prefix}:{resolved_id}'


def sanitize_artifact_record(
    value: Any,
    *,
    default_kind: Optional[str] = None,
    default_origin: Optional[str] = None,
    include_content: bool = False,
    content_limit: int = 12_000,
    include_image_state: bool = True,
) -> Optional[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None

    kind = default_kind or clean_artifact_kind(value.get('type'), value.get('kind'))
    if not kind:
        return None

    payload: dict[str, Any] = {
        'type': kind,
        'kind': kind,
        'origin': clean_text(value.get('origin')).lower() or (clean_text(default_origin).lower() or None),
        'path': clean_text(value.get('path')) or None,
        'source_path': clean_text(
            value.get('source_path')
            or value.get('sourcePath')
            or value.get('local_path')
            or value.get('localPath')
        ) or None,
        'name': clean_text(value.get('name')) or None,
        'mime_type': clean_text(value.get('mime_type') or value.get('mimeType')) or None,
        'availability': normalize_artifact_availability(value.get('availability')),
        'purged_at': clean_text(value.get('purged_at') or value.get('purgedAt')) or None,
        'purge_reason': clean_text(value.get('purge_reason') or value.get('purgeReason')) or None,
        'availability_checked_at': clean_text(
            value.get('availability_checked_at')
            or value.get('availabilityCheckedAt')
        ) or None,
        'source_message_id': clean_text(value.get('source_message_id') or value.get('sourceMessageId') or value.get('message_id') or value.get('messageId')) or None,
        'source_response_id': clean_text(value.get('source_response_id') or value.get('sourceResponseId') or value.get('response_id') or value.get('responseId')) or None,
        'response_model': clean_text(value.get('response_model') or value.get('responseModel')) or None,
        'response_instance_id': clean_text(value.get('response_instance_id') or value.get('responseInstanceId')) or None,
        'message_role': clean_text(value.get('message_role') or value.get('messageRole') or value.get('role')).lower() or None,
        'provenance_id': clean_text(value.get('provenance_id') or value.get('provenanceId')) or None,
        'timestamp': clean_text(value.get('timestamp')) or None,
        'prompt': clean_text(value.get('prompt'))[:4000] or None,
    }

    seed = value.get('seed')
    if isinstance(seed, (int, float)) and not isinstance(seed, bool):
        payload['seed'] = int(seed)
    batch_index = value.get('batch_index')
    if isinstance(batch_index, (int, float)) and not isinstance(batch_index, bool):
        payload['batch_index'] = int(batch_index)

    derived_from = _clean_list(value.get('derived_from') or value.get('derivedFrom'))
    if derived_from:
        payload['derived_from'] = derived_from

    if include_content:
        content = clean_text(value.get('content') or value.get('text') or value.get('prompt'))
        if content:
            payload['content'] = content[:content_limit]

    image_state = value.get('image_state')
    if include_image_state and isinstance(image_state, Mapping) and image_state:
        payload['image_state'] = dict(image_state)

    artifact_id = build_artifact_id(value, kind=kind)
    artifact_ref = build_artifact_ref(value, artifact_id=artifact_id, kind=kind)
    payload['artifact_id'] = artifact_id
    payload['artifact_ref'] = artifact_ref
    payload['ref'] = artifact_ref
    if payload.get('source_message_id'):
        payload['message_id'] = payload['source_message_id']

    return {
        key: item
        for key, item in payload.items()
        if item not in (None, '', [], {})
    } or None


def sanitize_artifact_records(
    value: Any,
    *,
    default_kind: Optional[str] = None,
    default_origin: Optional[str] = None,
    include_content: bool = False,
    content_limit: int = 12_000,
    include_image_state: bool = True,
) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    payload: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for item in items:
        normalized = sanitize_artifact_record(
            item,
            default_kind=default_kind,
            default_origin=default_origin,
            include_content=include_content,
            content_limit=content_limit,
            include_image_state=include_image_state,
        )
        if not normalized:
            continue
        ref = clean_text(normalized.get('artifact_ref'))
        if ref and ref in seen_refs:
            continue
        if ref:
            seen_refs.add(ref)
        payload.append(normalized)
    return payload


def extract_artifact_ref(value: Mapping[str, Any], *, role: str = 'artifact', index: int = 1) -> str:
    ref = clean_text(value.get('artifact_ref') or value.get('artifactRef') or value.get('ref'))
    if ref:
        return ref
    artifact_id = clean_text(value.get('artifact_id') or value.get('artifactId') or value.get('id'))
    if artifact_id:
        prefix = 'message' if clean_artifact_kind(value.get('type'), value.get('kind')) == 'message' else 'artifact'
        return f'{prefix}:{artifact_id}'
    normalized = sanitize_artifact_record(value)
    if normalized:
        return clean_text(normalized.get('artifact_ref'))
    return f'{role}:{index}:{clean_artifact_kind(value.get("type"), value.get("kind")) or "artifact"}'
