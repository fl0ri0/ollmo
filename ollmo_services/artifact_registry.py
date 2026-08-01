"""Durable artifact registry records keyed by artifact_ref and artifact path."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ollmo_services.artifact_contracts import (
    clean_text,
    sanitize_artifact_record,
    sanitize_artifact_records,
)


DEFAULT_ARTIFACT_REGISTRY_LEDGER = Path('state/artifact_registry.jsonl')
ARTIFACT_REGISTRY_VERSION = 1
ARTIFACT_REGISTRY_CONTENT_LIMIT = 16_000
ARTIFACT_REGISTRY_REFERENCE_CONTENT_LIMIT = 4_000
ARTIFACT_REGISTRY_PROMPT_LIMIT = 4_000
ARTIFACT_REGISTRY_PROMPT_PREVIEW_LIMIT = 2_000
ARTIFACT_REGISTRY_ROUTE_REASON_LIMIT = 2_000
ARTIFACT_REGISTRY_LOOKUP_LIMIT = 2_000
FINAL_TEXT_ARTIFACT_MAX_BYTES = 512_000
TEXT_ARTIFACT_EXTENSIONS = {
    'css',
    'csv',
    'cjs',
    'htm',
    'html',
    'js',
    'json',
    'jsx',
    'md',
    'mjs',
    'py',
    'svg',
    'ts',
    'tsx',
    'txt',
    'xml',
    'yaml',
    'yml',
}
SYNTAX_SANITY_EXTENSIONS = {'html', 'htm', 'css', 'json'}
TEXT_MIME_TYPES = {
    'application/javascript',
    'application/json',
    'application/markdown',
    'application/x-javascript',
    'application/xml',
    'image/svg+xml',
}
ARTIFACT_INTEGRITY_FIELDS = {
    'content_length_chars',
    'content_preview_truncated',
    'content_sha256',
    'content_source',
    'extension',
    'file_sha256',
    'file_size_bytes',
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _is_empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = clean_text(raw_key)
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


def _normalize_path(value: Any) -> Optional[str]:
    token = clean_text(value)
    if not token:
        return None
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str(candidate.resolve(strict=False))


def _clean_list(value: Any, *, limit: int = 24) -> list[str]:
    items = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    for item in items:
        token = clean_text(item)
        if not token or token in cleaned:
            continue
        cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def _merge_unique_texts(*values: Any, limit: int = 24) -> list[str]:
    merged: list[str] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            token = clean_text(item)
            if not token or token in merged:
                continue
            merged.append(token)
            if len(merged) >= limit:
                return merged
    return merged


def _normalize_mapping(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    payload = _json_safe(dict(value))
    return payload if isinstance(payload, dict) and payload else None


def _merge_mapping_payload(
    base: Any,
    overlay: Any,
) -> Optional[dict[str, Any]]:
    payload = dict(_normalize_mapping(base) or {})
    overlay_payload = _normalize_mapping(overlay) or {}
    for raw_key, raw_value in overlay_payload.items():
        key = clean_text(raw_key)
        if not key or _is_empty(raw_value):
            continue
        existing = payload.get(key)
        if isinstance(existing, Mapping) and isinstance(raw_value, Mapping):
            payload[key] = _merge_mapping_payload(existing, raw_value)
            continue
        payload[key] = _json_safe(raw_value)
    return payload if payload else None


def _compact_text(value: Any, *, limit: int) -> Optional[str]:
    token = clean_text(value)
    if not token:
        return None
    return token[:limit]


def _copy_artifact_integrity_fields(
    normalized_artifact: dict[str, Any],
    source_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    for key in ARTIFACT_INTEGRITY_FIELDS:
        value = source_artifact.get(key)
        if _is_empty(value):
            continue
        normalized_artifact[key] = _json_safe(value)
    return normalized_artifact


def _path_extension(value: Any) -> str:
    path = clean_text(value)
    if not path:
        return ''
    return Path(path.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')


def _saved_text_mime_type(path: Any) -> str:
    normalized_path = clean_text(path)
    if not normalized_path:
        return 'text/markdown'
    guessed_mime, _encoding = mimetypes.guess_type(normalized_path)
    return clean_text(guessed_mime) or 'text/plain'


def _artifact_extension(artifact: Mapping[str, Any]) -> str:
    for value in (
        artifact.get('text_artifact_extension'),
        artifact.get('extension'),
        _path_extension(artifact.get('path')),
        _path_extension(artifact.get('source_path')),
        _path_extension(artifact.get('name')),
    ):
        extension = clean_text(value).lower().lstrip('.')
        if extension:
            return extension
    return ''


def _artifact_is_text_like(artifact: Mapping[str, Any]) -> bool:
    artifact_type = clean_text(artifact.get('type') or artifact.get('kind')).lower()
    if artifact_type == 'text':
        return True
    mime_type = clean_text(artifact.get('mime_type') or artifact.get('mimeType')).lower()
    if mime_type.startswith('text/') or mime_type in TEXT_MIME_TYPES:
        return True
    return _artifact_extension(artifact) in TEXT_ARTIFACT_EXTENSIONS


def _syntax_sanity_issues_for_text_artifact(extension: str, content: str) -> tuple[str, list[str]]:
    normalized_extension = clean_text(extension).lower().lstrip('.')
    if normalized_extension not in SYNTAX_SANITY_EXTENSIONS:
        return 'not_applicable', []
    try:
        from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner

        issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            normalized_extension,
            content,
        )
    except Exception as exc:  # noqa: BLE001
        return f'unavailable:{type(exc).__name__}', []
    return ('issues' if issues else 'ok'), list(issues or [])


def refresh_text_artifact_record_from_saved_path(
    artifact: Mapping[str, Any],
    *,
    content_limit: int = ARTIFACT_REGISTRY_CONTENT_LIMIT,
    max_bytes: int = FINAL_TEXT_ARTIFACT_MAX_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refresh a small saved text artifact record from its final local file."""

    if not isinstance(artifact, Mapping):
        return {}, {}
    refreshed = dict(artifact)
    if not _artifact_is_text_like(refreshed):
        return refreshed, {}
    normalized_path = _normalize_path(refreshed.get('path') or refreshed.get('source_path'))
    if not normalized_path:
        return refreshed, {'final_text_artifact_refresh_status': 'skipped_missing_path'}
    target = Path(normalized_path)
    if not target.is_file():
        return refreshed, {'final_text_artifact_refresh_status': 'skipped_missing_file'}
    try:
        raw = target.read_bytes()
    except OSError:
        return refreshed, {'final_text_artifact_refresh_status': 'skipped_unreadable'}
    if len(raw) > max(0, int(max_bytes)):
        return refreshed, {
            'final_text_artifact_refresh_status': 'skipped_too_large',
            'file_size_bytes': len(raw),
        }

    content = raw.decode('utf-8', errors='replace')
    extension = _artifact_extension({**refreshed, 'path': normalized_path})
    syntax_status, syntax_issues = _syntax_sanity_issues_for_text_artifact(extension, content)
    content_bytes = content.encode('utf-8')
    preview_limit = max(1, int(content_limit))
    refreshed['path'] = normalized_path
    refreshed['content'] = content[:preview_limit]
    refreshed['content_sha256'] = hashlib.sha256(content_bytes).hexdigest()
    refreshed['file_sha256'] = hashlib.sha256(raw).hexdigest()
    refreshed['file_size_bytes'] = len(raw)
    refreshed['content_length_chars'] = len(content)
    refreshed['content_preview_truncated'] = len(content) > preview_limit
    refreshed['content_source'] = 'final_saved_text_artifact'
    if extension and not clean_text(refreshed.get('extension')):
        refreshed['extension'] = extension

    metadata = {
        'final_text_artifact_refresh_status': 'refreshed',
        'final_text_artifact_content_source': 'final_saved_text_artifact',
        'content_sha256': refreshed['content_sha256'],
        'file_sha256': refreshed['file_sha256'],
        'file_size_bytes': len(raw),
        'content_length_chars': len(content),
        'content_preview_truncated': len(content) > preview_limit,
    }
    if extension:
        metadata['text_artifact_extension'] = extension
    if syntax_status:
        metadata['syntax_sanity_status'] = syntax_status
        metadata['syntax_sanity_issue_count'] = len(syntax_issues)
        if syntax_issues:
            metadata['syntax_sanity_issues'] = syntax_issues[:5]
    return refreshed, metadata


def _sanitize_artifact_reference(value: Any) -> Optional[dict[str, Any]]:
    normalized = sanitize_artifact_record(
        value,
        include_content=True,
        content_limit=ARTIFACT_REGISTRY_REFERENCE_CONTENT_LIMIT,
    )
    if not normalized:
        return None
    if normalized.get('path'):
        normalized['path'] = _normalize_path(normalized.get('path'))
    if normalized.get('source_path'):
        normalized['source_path'] = _normalize_path(normalized.get('source_path'))
    if normalized.get('prompt'):
        normalized['prompt'] = _compact_text(
            normalized.get('prompt'),
            limit=ARTIFACT_REGISTRY_PROMPT_LIMIT,
        )
    if normalized.get('content'):
        normalized['content'] = _compact_text(
            normalized.get('content'),
            limit=ARTIFACT_REGISTRY_REFERENCE_CONTENT_LIMIT,
        )
    return normalized or None


def _sanitize_artifact_references(value: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in sanitize_artifact_records(
        value if isinstance(value, list) else [],
        include_content=True,
        content_limit=ARTIFACT_REGISTRY_REFERENCE_CONTENT_LIMIT,
    ):
        normalized = _sanitize_artifact_reference(item)
        if normalized:
            sanitized.append(normalized)
        if len(sanitized) >= limit:
            break
    return sanitized


def _build_generated_image_provenance_id(image_path: str) -> str:
    digest = hashlib.sha256(image_path.encode('utf-8')).hexdigest()[:24]
    return f'generated_image_{digest}'


def build_artifact_registry_record(
    *,
    artifact: Mapping[str, Any],
    roles: Any = None,
    provenance: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    enrichments: Optional[Mapping[str, Any]] = None,
    linked_response_ids: Any = None,
    linked_message_ids: Any = None,
    created_at: Any = None,
) -> dict[str, Any]:
    normalized_artifact = sanitize_artifact_record(
        artifact,
        include_content=True,
        content_limit=ARTIFACT_REGISTRY_CONTENT_LIMIT,
    )
    if not normalized_artifact:
        raise ValueError('artifact must be a non-empty mapping with a valid type')
    if normalized_artifact.get('path'):
        normalized_artifact['path'] = _normalize_path(normalized_artifact.get('path'))
    if normalized_artifact.get('source_path'):
        normalized_artifact['source_path'] = _normalize_path(normalized_artifact.get('source_path'))
    normalized_artifact = _copy_artifact_integrity_fields(
        normalized_artifact,
        artifact,
    )
    if (
        clean_text(artifact.get('content_source')) == 'final_saved_text_artifact'
        and isinstance(artifact.get('content'), str)
    ):
        normalized_artifact['content'] = artifact.get('content')[:ARTIFACT_REGISTRY_CONTENT_LIMIT]

    payload = {
        'kind': 'ollmo.artifact_registry_record',
        'artifact_registry_version': ARTIFACT_REGISTRY_VERSION,
        'created_at': clean_text(created_at) or _utc_now_iso(),
        'artifact_ref': normalized_artifact.get('artifact_ref'),
        'artifact_id': normalized_artifact.get('artifact_id'),
        'type': normalized_artifact.get('type') or 'artifact',
        'roles': _clean_list(roles),
        'artifact': normalized_artifact,
        'provenance': _normalize_mapping(provenance),
        'metadata': _normalize_mapping(metadata),
        'enrichments': _normalize_mapping(enrichments),
        'linked_response_ids': _clean_list(linked_response_ids),
        'linked_message_ids': _clean_list(linked_message_ids),
    }
    return _json_safe(payload)


def build_generated_image_artifact_registry_record(
    provenance: Mapping[str, Any],
    *,
    created_at: Any = None,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError('provenance must be a non-empty mapping')
    source = provenance.get('source') if isinstance(provenance.get('source'), Mapping) else {}
    request = provenance.get('request') if isinstance(provenance.get('request'), Mapping) else {}
    output = provenance.get('output') if isinstance(provenance.get('output'), Mapping) else {}
    image_path = output.get('saved_image_path') or provenance.get('image_path')
    artifact = {
        'type': 'image',
        'path': image_path,
        'seed': output.get('seed') if output.get('seed') is not None else request.get('seed'),
        'origin': 'generated_output',
        'provenance_id': provenance.get('provenance_id'),
        'artifact_id': output.get('artifact_id'),
        'artifact_ref': output.get('artifact_ref'),
        'derived_from': provenance.get('derived_from'),
        'source_response_id': source.get('response_id'),
        'response_model': source.get('model'),
        'response_instance_id': source.get('instance_id'),
        'prompt': request.get('prompt_preview') or request.get('prompt_text'),
    }
    metadata = {
        'path': image_path,
        'prompt_preview': request.get('prompt_preview'),
        'width': request.get('width'),
        'height': request.get('height'),
        'seed': request.get('seed'),
        'file_path': request.get('file_path'),
        'response_model': source.get('model'),
        'response_instance_id': source.get('instance_id'),
        'backend': source.get('backend'),
        'capability': source.get('capability'),
        'mode': source.get('mode'),
        'request_origin': source.get('request_origin'),
        'conversation_id': source.get('conversation_id'),
        'request_id': source.get('request_id'),
    }
    return build_artifact_registry_record(
        artifact=artifact,
        roles=['output'],
        provenance=provenance,
        metadata=metadata,
        linked_response_ids=[source.get('response_id')],
        created_at=created_at or provenance.get('created_at'),
    )


def _artifact_records_from_response_payload(response_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = sanitize_artifact_records(
        response_payload.get('artifacts') if isinstance(response_payload.get('artifacts'), list) else [],
        include_content=True,
        content_limit=ARTIFACT_REGISTRY_CONTENT_LIMIT,
    )
    saved_text_artifacts = response_payload.get('saved_text_artifacts')
    if isinstance(saved_text_artifacts, list):
        for item in saved_text_artifacts:
            source = item if isinstance(item, Mapping) else {'path': item}
            source_path = source.get('path') or source.get('saved_text_path')
            artifact = sanitize_artifact_record(
                {
                    'type': 'text',
                    'path': source_path,
                    'name': (source.get('text_artifact_request') or {}).get('source_name')
                    if isinstance(source.get('text_artifact_request'), Mapping)
                    else source.get('source_name'),
                    'mime_type': source.get('mime_type') or _saved_text_mime_type(source_path),
                    'origin': 'assistant_output',
                    'source_response_id': response_payload.get('id') or response_payload.get('response_id'),
                },
                include_content=True,
                content_limit=ARTIFACT_REGISTRY_CONTENT_LIMIT,
            )
            if artifact:
                artifacts.append(artifact)
    saved_text_path = clean_text(response_payload.get('saved_text_path') or response_payload.get('savedTextPath'))
    if saved_text_path:
        artifact = sanitize_artifact_record(
            {
                'type': 'text',
                'path': saved_text_path,
                'mime_type': _saved_text_mime_type(saved_text_path),
                'origin': 'assistant_output',
                'source_response_id': response_payload.get('id') or response_payload.get('response_id'),
            },
            include_content=True,
            content_limit=ARTIFACT_REGISTRY_CONTENT_LIMIT,
        )
        if artifact:
            artifacts.append(artifact)
    saved_audio_path = clean_text(response_payload.get('saved_audio_path') or response_payload.get('savedAudioPath'))
    if saved_audio_path:
        artifact = sanitize_artifact_record(
            {
                'type': 'audio',
                'path': saved_audio_path,
                'mime_type': clean_text(response_payload.get('audio_mimetype') or response_payload.get('audioMimeType')) or 'audio/wav',
                'origin': 'assistant_output',
                'source_response_id': response_payload.get('id') or response_payload.get('response_id'),
            },
            include_content=True,
            content_limit=ARTIFACT_REGISTRY_CONTENT_LIMIT,
        )
        if artifact:
            artifacts.append(artifact)
    saved_image_path = clean_text(response_payload.get('saved_image_path') or response_payload.get('savedImagePath'))
    if saved_image_path:
        artifact = sanitize_artifact_record(
            {
                'type': 'image',
                'path': saved_image_path,
                'origin': 'assistant_output',
                'source_response_id': response_payload.get('id') or response_payload.get('response_id'),
                'image_state': response_payload.get('image_state')
                if isinstance(response_payload.get('image_state'), Mapping)
                else None,
            },
            include_content=True,
            content_limit=ARTIFACT_REGISTRY_CONTENT_LIMIT,
        )
        if artifact:
            artifacts.append(artifact)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in artifacts:
        ref = clean_text(artifact.get('artifact_ref'))
        path = clean_text(artifact.get('path'))
        key = ref or path
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(artifact)
    return deduped


def build_late_fill_artifact_source_index(response_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}
    fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
    index: dict[str, dict[str, Any]] = {}

    def remember(path_value: Any, result: Mapping[str, Any], artifact_type: Optional[str] = None) -> None:
        path = _normalize_path(path_value)
        if not path or path in index:
            return
        source = {
            'response_id': response_payload.get('id') or response_payload.get('response_id'),
            'conversation_id': clean_text(response_payload.get('conversation_id') or response_payload.get('conversationId')) or None,
            'request_id': clean_text(response_payload.get('request_id') or response_payload.get('requestId')) or None,
            'instance_id': clean_text(result.get('fill_instance_id') or result.get('instance_id')) or None,
            'model': clean_text(result.get('fill_model') or result.get('model') or result.get('request_model')) or None,
            'backend': clean_text(result.get('fill_backend') or result.get('backend')) or None,
            'capability': clean_text(result.get('capability')) or clean_text(artifact_type) or None,
            'mode': clean_text(result.get('fill_mode') or result.get('mode') or result.get('capability')) or None,
            'route_source': clean_text(result.get('route_source')) or None,
            'route_reason': _compact_text(
                result.get('route_reason'),
                limit=ARTIFACT_REGISTRY_ROUTE_REASON_LIMIT,
            ),
            'branch_id': clean_text(result.get('branch_id')) or None,
            'phase_id': clean_text(result.get('phase_id')) or None,
            'task_id': clean_text(result.get('task_id') or result.get('workload_task_id')) or None,
            'obligation_id': clean_text(result.get('obligation_id')) or None,
        }
        index[path] = {
            key: value
            for key, value in source.items()
            if not _is_empty(value)
        }

    for result in fill_results:
        if not isinstance(result, Mapping):
            continue
        remember(result.get('saved_text_path'), result, 'text')
        remember(result.get('saved_audio_path'), result, 'audio')
        remember(result.get('saved_image_path'), result, 'image')
        artifacts = result.get('artifacts') if isinstance(result.get('artifacts'), list) else []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            remember(
                artifact.get('path') or artifact.get('source_path'),
                result,
                clean_text(artifact.get('type') or artifact.get('kind')) or None,
            )
    return index


def find_late_fill_artifact_source(
    response_payload: Mapping[str, Any],
    artifact_path: Any,
) -> Optional[dict[str, Any]]:
    """Return the exact Late Fill producer bound to one materialized path."""

    normalized_path = _normalize_path(artifact_path)
    if not normalized_path:
        return None
    source = build_late_fill_artifact_source_index(response_payload).get(normalized_path)
    return dict(source) if isinstance(source, Mapping) else None


def build_output_artifact_registry_records(
    response_payload: Mapping[str, Any],
    *,
    created_at: Any = None,
) -> list[dict[str, Any]]:
    """Build generic output artifact registry records from a response payload."""

    if not isinstance(response_payload, Mapping) or not response_payload:
        return []
    response_id = clean_text(response_payload.get('id') or response_payload.get('response_id'))
    source = {
        'response_id': response_id or None,
        'conversation_id': clean_text(response_payload.get('conversation_id') or response_payload.get('conversationId')) or None,
        'request_id': clean_text(response_payload.get('request_id') or response_payload.get('requestId')) or None,
        'instance_id': clean_text(response_payload.get('instance_id')) or None,
        'model': clean_text(response_payload.get('model') or response_payload.get('request_model')) or None,
        'backend': clean_text(response_payload.get('backend')) or None,
        'capability': clean_text(response_payload.get('capability')) or None,
        'mode': clean_text(response_payload.get('mode')) or None,
        'route_source': clean_text(response_payload.get('route_source')) or None,
        'route_reason': _compact_text(
            response_payload.get('route_reason'),
            limit=ARTIFACT_REGISTRY_ROUTE_REASON_LIMIT,
        ),
    }
    late_fill_artifact_sources = build_late_fill_artifact_source_index(response_payload)
    records: list[dict[str, Any]] = []
    for artifact in _artifact_records_from_response_payload(response_payload):
        artifact, final_text_metadata = refresh_text_artifact_record_from_saved_path(artifact)
        artifact_type = clean_text(artifact.get('type') or artifact.get('kind')) or 'artifact'
        path = artifact.get('path')
        normalized_path = _normalize_path(path) if path else None
        artifact_source = dict(source)
        if normalized_path and normalized_path in late_fill_artifact_sources:
            artifact_source.update(late_fill_artifact_sources[normalized_path])
        provenance = {
            'kind': 'ollmo.output_artifact_provenance',
            'artifact_version': 1,
            'created_at': clean_text(created_at) or _utc_now_iso(),
            'source': {key: value for key, value in artifact_source.items() if not _is_empty(value)},
            'output': {
                'artifact_ref': artifact.get('artifact_ref'),
                'artifact_id': artifact.get('artifact_id'),
                'type': artifact_type,
                'path': path,
                'mime_type': artifact.get('mime_type'),
            },
        }
        metadata = {
            'path': path,
            'mime_type': artifact.get('mime_type'),
            'response_model': artifact_source.get('model'),
            'response_instance_id': artifact_source.get('instance_id'),
            'backend': artifact_source.get('backend'),
            'capability': artifact_source.get('capability'),
            'mode': artifact_source.get('mode'),
            'conversation_id': artifact_source.get('conversation_id'),
            'request_id': artifact_source.get('request_id'),
            'branch_id': artifact_source.get('branch_id'),
            'phase_id': artifact_source.get('phase_id'),
            'task_id': artifact_source.get('task_id'),
            'obligation_id': artifact_source.get('obligation_id'),
            'request_origin': clean_text(response_payload.get('request_origin') or response_payload.get('provenance_origin')) or None,
        }
        metadata.update(final_text_metadata)
        records.append(
            build_artifact_registry_record(
                artifact=artifact,
                roles=['output'],
                provenance=provenance,
                metadata=metadata,
                linked_response_ids=[response_id],
                created_at=created_at,
            )
        )
    return records


def build_input_artifact_registry_records(
    input_artifacts: Any,
    *,
    request_payload: Optional[Mapping[str, Any]] = None,
    created_at: Any = None,
) -> list[dict[str, Any]]:
    """Build registry records for true external input artifacts."""

    request_payload = request_payload if isinstance(request_payload, Mapping) else {}
    response_id = clean_text(request_payload.get('response_id') or request_payload.get('responseId'))
    source = {
        'response_id': response_id or None,
        'conversation_id': clean_text(request_payload.get('conversation_id') or request_payload.get('conversationId')) or None,
        'request_id': clean_text(request_payload.get('request_id') or request_payload.get('requestId')) or None,
        'instance_id': clean_text(request_payload.get('instance_id')) or None,
        'model': clean_text(request_payload.get('model') or request_payload.get('request_model')) or None,
        'backend': clean_text(request_payload.get('backend')) or None,
        'capability': clean_text(request_payload.get('capability')) or None,
        'request_origin': clean_text(request_payload.get('request_origin') or request_payload.get('provenance_origin')) or None,
    }
    records: list[dict[str, Any]] = []
    for artifact in sanitize_artifact_records(
        input_artifacts if isinstance(input_artifacts, list) else [],
        include_content=True,
        content_limit=ARTIFACT_REGISTRY_REFERENCE_CONTENT_LIMIT,
    ):
        origin = clean_text(artifact.get('origin')) or 'upload'
        artifact_type = clean_text(artifact.get('type') or artifact.get('kind')) or 'artifact'
        provenance = {
            'kind': 'ollmo.input_artifact_provenance',
            'artifact_version': 1,
            'created_at': clean_text(created_at) or _utc_now_iso(),
            'source': {key: value for key, value in source.items() if not _is_empty(value)},
            'input': {
                'artifact_ref': artifact.get('artifact_ref'),
                'artifact_id': artifact.get('artifact_id'),
                'type': artifact_type,
                'path': artifact.get('path'),
                'source_path': artifact.get('source_path'),
                'mime_type': artifact.get('mime_type'),
                'origin': origin,
            },
        }
        metadata = {
            'path': artifact.get('path'),
            'source_path': artifact.get('source_path'),
            'mime_type': artifact.get('mime_type'),
            'origin': origin,
            'conversation_id': source.get('conversation_id'),
            'request_id': source.get('request_id'),
            'response_id': source.get('response_id'),
            'capability': source.get('capability'),
            'request_origin': source.get('request_origin') or 'external_input',
        }
        records.append(
            build_artifact_registry_record(
                artifact=artifact,
                roles=['input'],
                provenance=provenance,
                metadata=metadata,
                linked_response_ids=[response_id],
                created_at=created_at,
            )
        )
    return records


def persist_input_artifact_registry_records(
    input_artifacts: Any,
    *,
    request_payload: Optional[Mapping[str, Any]] = None,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    created_at: Any = None,
) -> list[dict[str, Any]]:
    """Persist true external input artifacts without duplicating reused output refs."""

    persisted: list[dict[str, Any]] = []
    for record in build_input_artifact_registry_records(
        input_artifacts,
        request_payload=request_payload,
        created_at=created_at,
    ):
        artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
        existing = _find_artifact_registry_record_for_write(
            artifact_ref=record.get('artifact_ref') or artifact.get('artifact_ref'),
            artifact_path=artifact.get('path'),
            ledger_path=ledger_path,
        )
        merged = merge_artifact_registry_records(existing, record)
        if isinstance(existing, Mapping) and _json_safe(existing) == merged:
            persisted.append(dict(existing))
            continue
        persist_artifact_registry_record(merged, ledger_path=ledger_path)
        persisted.append(merged)
    return persisted


def persist_output_artifact_registry_records(
    response_payload: Mapping[str, Any],
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    created_at: Any = None,
) -> list[dict[str, Any]]:
    """Persist generic output artifacts so recovery need not mine response frames."""

    persisted: list[dict[str, Any]] = []
    for record in build_output_artifact_registry_records(response_payload, created_at=created_at):
        artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
        existing = _find_artifact_registry_record_for_write(
            artifact_ref=record.get('artifact_ref') or artifact.get('artifact_ref'),
            artifact_path=artifact.get('path'),
            ledger_path=ledger_path,
        )
        if isinstance(existing, Mapping):
            record = dict(record)
            record['created_at'] = clean_text(existing.get('created_at')) or record.get('created_at')
            existing_provenance = (
                existing.get('provenance')
                if isinstance(existing.get('provenance'), Mapping)
                else {}
            )
            if clean_text(existing_provenance.get('kind')) not in {'', 'ollmo.output_artifact_provenance'}:
                # Generic output provenance must not overwrite richer modality-specific
                # provenance such as generated-image provenance.
                record.pop('provenance', None)
                record.pop('artifact', None)
            elif isinstance(record.get('provenance'), Mapping):
                record_provenance = dict(record.get('provenance') or {})
                record_provenance['created_at'] = (
                    clean_text(existing_provenance.get('created_at'))
                    or clean_text(record_provenance.get('created_at'))
                    or _utc_now_iso()
                )
                record['provenance'] = record_provenance
        merged = merge_artifact_registry_records(existing, record)
        if isinstance(existing, Mapping) and _json_safe(existing) == merged:
            persisted.append(dict(existing))
            continue
        persist_artifact_registry_record(merged, ledger_path=ledger_path)
        persisted.append(merged)
    return persisted


def build_generated_image_provenance(
    *,
    image_path: Any,
    prompt_text: Any = None,
    prompt_preview: Any = None,
    instance_id: Any = None,
    model: Any = None,
    backend: Any = None,
    capability: Any = None,
    mode: Any = None,
    request_origin: Any = None,
    response_id: Any = None,
    conversation_id: Any = None,
    request_id: Any = None,
    width: Any = None,
    height: Any = None,
    seed: Any = None,
    file_path: Any = None,
    input_artifacts: Any = None,
    reference_artifacts: Any = None,
    selected_reference_artifacts: Any = None,
    route_source: Any = None,
    route_reason: Any = None,
    request_meta: Any = None,
    created_at: Any = None,
) -> dict[str, Any]:
    normalized_image_path = _normalize_path(image_path)
    if not normalized_image_path:
        raise ValueError('image_path is required')

    normalized_prompt_text = _compact_text(prompt_text, limit=120_000)
    normalized_prompt_preview = _compact_text(
        prompt_preview if prompt_preview not in (None, '') else normalized_prompt_text,
        limit=ARTIFACT_REGISTRY_PROMPT_PREVIEW_LIMIT,
    )
    normalized_reference_artifacts = (
        reference_artifacts
        if reference_artifacts is not None
        else selected_reference_artifacts
    )
    normalized_input_artifacts = _sanitize_artifact_references(input_artifacts)
    normalized_reference_payload = _sanitize_artifact_references(normalized_reference_artifacts)
    output_artifact = sanitize_artifact_record(
        {
            'type': 'image',
            'path': normalized_image_path,
            'seed': seed if isinstance(seed, int) else None,
            'origin': 'generated_output',
        }
    ) or {}
    provenance_id = _build_generated_image_provenance_id(normalized_image_path)
    derived_from = [
        str(item.get('artifact_ref') or '')
        for item in normalized_input_artifacts + normalized_reference_payload
        if str(item.get('artifact_ref') or '').strip()
    ]

    source = {
        'instance_id': clean_text(instance_id) or None,
        'model': clean_text(model) or None,
        'backend': clean_text(backend) or None,
        'capability': clean_text(capability) or None,
        'mode': clean_text(mode) or None,
        'request_origin': clean_text(request_origin) or None,
        'response_id': clean_text(response_id) or None,
        'conversation_id': clean_text(conversation_id) or None,
        'request_id': clean_text(request_id) or None,
        'route_source': clean_text(route_source) or None,
        'route_reason': _compact_text(
            route_reason,
            limit=ARTIFACT_REGISTRY_ROUTE_REASON_LIMIT,
        ),
    }
    request = {
        'prompt_text': normalized_prompt_text,
        'prompt_preview': normalized_prompt_preview,
        'width': width if isinstance(width, int) else None,
        'height': height if isinstance(height, int) else None,
        'seed': seed if isinstance(seed, int) else None,
        'file_path': _normalize_path(file_path),
        'request_meta': _json_safe(request_meta) if isinstance(request_meta, Mapping) else None,
        'input_artifacts': normalized_input_artifacts,
        'reference_artifacts': normalized_reference_payload,
    }
    output = {
        'saved_image_path': normalized_image_path,
        'seed': seed if isinstance(seed, int) else None,
        'artifact_id': output_artifact.get('artifact_id'),
        'artifact_ref': output_artifact.get('artifact_ref'),
    }

    payload = {
        'kind': 'ollmo.generated_image_provenance',
        'artifact_version': 2,
        'provenance_id': provenance_id,
        'created_at': clean_text(created_at) or _utc_now_iso(),
        'image_path': normalized_image_path,
        'source': {key: item for key, item in source.items() if not _is_empty(item)},
        'request': {key: item for key, item in request.items() if not _is_empty(item)},
        'output': {key: item for key, item in output.items() if not _is_empty(item)},
    }
    if derived_from:
        payload['derived_from'] = derived_from
    return _json_safe(payload)


def _extract_generated_image_provenance(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    provenance = payload.get('provenance') if isinstance(payload.get('provenance'), Mapping) else {}
    normalized_kind = clean_text(provenance.get('kind'))
    if normalized_kind == 'ollmo.generated_image_provenance':
        return _json_safe(dict(provenance))
    if clean_text(payload.get('kind')) == 'ollmo.generated_image_provenance':
        return _json_safe(dict(payload))
    return None


def persist_generated_image_provenance(
    provenance: Mapping[str, Any],
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
) -> Path:
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError('provenance must be a non-empty mapping')
    record = build_generated_image_artifact_registry_record(_json_safe(provenance))
    artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
    existing = _find_artifact_registry_record_for_write(
        artifact_ref=record.get('artifact_ref') or artifact.get('artifact_ref'),
        artifact_path=artifact.get('path'),
        ledger_path=ledger_path,
    )
    if isinstance(existing, Mapping):
        record = merge_artifact_registry_records(existing, record)
    return persist_artifact_registry_record(
        record,
        ledger_path=ledger_path,
    )


def persist_artifact_registry_record(
    record: Mapping[str, Any],
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
) -> Path:
    if not isinstance(record, Mapping) or not record:
        raise ValueError('record must be a non-empty mapping')
    target = Path(ledger_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_record = _json_safe(record)
    artifact_ref, artifact_path = _artifact_registry_identity(safe_record)
    if not target.exists() or not (artifact_ref or artifact_path):
        with target.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(safe_record, ensure_ascii=False, sort_keys=True) + '\n')
        return target

    kept_lines: list[str] = []
    merged_record: dict[str, Any] = dict(safe_record)
    matched = False
    for raw_line in _load_registry_lines(target):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(raw_line)
            continue
        if not isinstance(payload, dict):
            kept_lines.append(raw_line)
            continue
        matches_ref = bool(artifact_ref and _record_matches_artifact_ref(payload, artifact_ref))
        matches_path = bool(artifact_path and _record_matches_artifact_path(payload, artifact_path))
        if matches_ref or matches_path:
            merged_record = merge_artifact_registry_records(payload, merged_record)
            matched = True
            continue
        kept_lines.append(raw_line)
    kept_lines.append(json.dumps(_json_safe(merged_record), ensure_ascii=False, sort_keys=True))
    if matched:
        target.write_text('\n'.join(kept_lines) + '\n', encoding='utf-8')
    else:
        with target.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_json_safe(merged_record), ensure_ascii=False, sort_keys=True) + '\n')
    return target


def merge_artifact_registry_records(
    base_record: Optional[Mapping[str, Any]],
    overlay_record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(overlay_record, Mapping) or not overlay_record:
        raise ValueError('overlay_record must be a non-empty mapping')
    base = dict(_json_safe(dict(base_record or {}))) if isinstance(base_record, Mapping) else {}
    overlay = dict(_json_safe(dict(overlay_record))) if isinstance(overlay_record, Mapping) else {}
    artifact = _merge_mapping_payload(base.get('artifact'), overlay.get('artifact')) or {}
    artifact_ref = clean_text(
        overlay.get('artifact_ref')
        or artifact.get('artifact_ref')
        or base.get('artifact_ref')
    )
    artifact_id = clean_text(
        overlay.get('artifact_id')
        or artifact.get('artifact_id')
        or base.get('artifact_id')
    )
    artifact_type = clean_text(
        overlay.get('type')
        or artifact.get('type')
        or base.get('type')
    ) or 'artifact'
    artifact_alias_refs = _merge_unique_texts(
        base.get('artifact_alias_refs'),
        (base.get('artifact') or {}).get('artifact_alias_refs') if isinstance(base.get('artifact'), Mapping) else None,
        base.get('artifact_ref'),
        (base.get('artifact') or {}).get('artifact_ref') if isinstance(base.get('artifact'), Mapping) else None,
        overlay.get('artifact_alias_refs'),
        (overlay.get('artifact') or {}).get('artifact_alias_refs') if isinstance(overlay.get('artifact'), Mapping) else None,
        overlay.get('artifact_ref'),
        (overlay.get('artifact') or {}).get('artifact_ref') if isinstance(overlay.get('artifact'), Mapping) else None,
    )
    roles = _merge_unique_texts(base.get('roles'), overlay.get('roles'))
    payload = {
        'kind': 'ollmo.artifact_registry_record',
        'artifact_registry_version': ARTIFACT_REGISTRY_VERSION,
        'created_at': clean_text(overlay.get('created_at') or overlay.get('updated_at'))
        or clean_text(base.get('created_at'))
        or _utc_now_iso(),
        'artifact_ref': artifact_ref,
        'artifact_id': artifact_id,
        'type': artifact_type,
        'roles': roles,
        'artifact_alias_refs': artifact_alias_refs,
        'artifact': artifact,
        'provenance': _merge_mapping_payload(base.get('provenance'), overlay.get('provenance')),
        'metadata': _merge_mapping_payload(base.get('metadata'), overlay.get('metadata')),
        'enrichments': _merge_mapping_payload(base.get('enrichments'), overlay.get('enrichments')),
        'linked_response_ids': _merge_unique_texts(
            base.get('linked_response_ids'),
            overlay.get('linked_response_ids'),
        ),
        'linked_message_ids': _merge_unique_texts(
            base.get('linked_message_ids'),
            overlay.get('linked_message_ids'),
        ),
    }
    metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}
    if (
        clean_text(metadata.get('syntax_sanity_status')).lower() == 'ok'
        and metadata.get('syntax_sanity_issue_count') == 0
    ):
        metadata.pop('syntax_sanity_issues', None)
    artifact_metadata = (
        payload.get('artifact', {}).get('metadata')
        if isinstance(payload.get('artifact'), Mapping)
        and isinstance(payload.get('artifact', {}).get('metadata'), dict)
        else {}
    )
    if (
        clean_text(artifact_metadata.get('syntax_sanity_status')).lower() == 'ok'
        and artifact_metadata.get('syntax_sanity_issue_count') == 0
    ):
        artifact_metadata.pop('syntax_sanity_issues', None)
    return _json_safe(payload)


def _load_registry_lines(ledger_path: Path | str) -> list[str]:
    target = Path(ledger_path)
    if not target.exists():
        return []
    try:
        return target.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []


def _find_artifact_registry_record_for_write(
    *,
    artifact_ref: Any = None,
    artifact_path: Any = None,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    limit: int = ARTIFACT_REGISTRY_LOOKUP_LIMIT,
) -> Optional[dict[str, Any]]:
    normalized_artifact_ref = clean_text(artifact_ref)
    if normalized_artifact_ref:
        record = find_artifact_registry_record_by_artifact_ref(
            normalized_artifact_ref,
            ledger_path=ledger_path,
            limit=limit,
        )
        if isinstance(record, Mapping):
            return dict(record)
    normalized_artifact_path = _normalize_path(artifact_path)
    if normalized_artifact_path:
        record = find_artifact_registry_record(
            normalized_artifact_path,
            ledger_path=ledger_path,
            limit=limit,
        )
        if isinstance(record, Mapping):
            return dict(record)
    return None


def persist_artifact_registry_enrichment(
    *,
    artifact_ref: Any = None,
    artifact_path: Any = None,
    artifact_type: Any = None,
    enrichments: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    linked_response_ids: Any = None,
    linked_message_ids: Any = None,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    created_at: Any = None,
) -> Optional[dict[str, Any]]:
    normalized_artifact_ref = clean_text(artifact_ref)
    normalized_artifact_path = _normalize_path(artifact_path)
    normalized_enrichments = _normalize_mapping(enrichments)
    normalized_metadata = _normalize_mapping(metadata)
    response_ids = _clean_list(linked_response_ids)
    message_ids = _clean_list(linked_message_ids)
    existing = _find_artifact_registry_record_for_write(
        artifact_ref=normalized_artifact_ref,
        artifact_path=normalized_artifact_path,
        ledger_path=ledger_path,
    )
    if not any((normalized_artifact_ref, normalized_artifact_path, existing)):
        return None
    artifact_payload = {}
    existing_artifact = (
        dict(existing.get('artifact') or {})
        if isinstance((existing or {}).get('artifact'), Mapping)
        else {}
    )
    artifact_payload.update(existing_artifact)
    if normalized_artifact_ref:
        artifact_payload['artifact_ref'] = normalized_artifact_ref
    if normalized_artifact_path:
        artifact_payload['path'] = normalized_artifact_path
    artifact_payload['type'] = (
        clean_text(artifact_type)
        or clean_text(artifact_payload.get('type'))
        or clean_text((existing or {}).get('type'))
        or 'artifact'
    )
    update_record = {
        'kind': 'ollmo.artifact_registry_record',
        'artifact_registry_version': ARTIFACT_REGISTRY_VERSION,
        'created_at': clean_text(created_at) or _utc_now_iso(),
        'artifact_ref': normalized_artifact_ref or clean_text((existing or {}).get('artifact_ref')),
        'artifact_id': clean_text((existing or {}).get('artifact_id') or artifact_payload.get('artifact_id')),
        'type': artifact_payload.get('type'),
        'roles': list((existing or {}).get('roles') or []),
        'artifact': artifact_payload,
        'metadata': normalized_metadata,
        'enrichments': normalized_enrichments,
        'linked_response_ids': response_ids,
        'linked_message_ids': message_ids,
    }
    merged = merge_artifact_registry_records(existing, update_record)
    if isinstance(existing, Mapping) and _json_safe(existing) == merged:
        return dict(existing)
    persist_artifact_registry_record(
        merged,
        ledger_path=ledger_path,
    )
    return merged


def _record_matches_artifact_ref(payload: Mapping[str, Any], artifact_ref: str) -> bool:
    if clean_text(payload.get('artifact_ref')) == artifact_ref:
        return True
    artifact = payload.get('artifact') if isinstance(payload.get('artifact'), Mapping) else {}
    refs = _merge_unique_texts(
        payload.get('artifact_alias_refs'),
        artifact.get('artifact_alias_refs'),
        artifact.get('artifact_ref'),
        artifact.get('ref'),
    )
    return artifact_ref in refs


def _record_artifact_path(payload: Mapping[str, Any]) -> Optional[str]:
    artifact = payload.get('artifact') if isinstance(payload.get('artifact'), Mapping) else {}
    metadata = payload.get('metadata') if isinstance(payload.get('metadata'), Mapping) else {}
    candidates = [
        payload.get('image_path'),
        artifact.get('path'),
        metadata.get('path'),
        metadata.get('file_path'),
    ]
    for candidate in candidates:
        normalized = _normalize_path(candidate)
        if normalized:
            return normalized
    provenance = payload.get('provenance') if isinstance(payload.get('provenance'), Mapping) else {}
    output = provenance.get('output') if isinstance(provenance.get('output'), Mapping) else {}
    request = provenance.get('request') if isinstance(provenance.get('request'), Mapping) else {}
    return _normalize_path(output.get('saved_image_path') or request.get('file_path'))


def _record_matches_artifact_path(payload: Mapping[str, Any], artifact_path: str) -> bool:
    return _record_artifact_path(payload) == artifact_path


def _artifact_registry_identity(record: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
    artifact_ref = clean_text(record.get('artifact_ref') or artifact.get('artifact_ref') or artifact.get('ref'))
    artifact_path = _record_artifact_path(record)
    return artifact_ref or None, artifact_path


def find_artifact_registry_record(
    artifact_path: Any,
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    limit: int = ARTIFACT_REGISTRY_LOOKUP_LIMIT,
) -> Optional[dict[str, Any]]:
    normalized_artifact_path = _normalize_path(artifact_path)
    if not normalized_artifact_path:
        return None
    scanned = 0
    for raw_line in reversed(_load_registry_lines(ledger_path)):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if _record_matches_artifact_path(payload, normalized_artifact_path):
            return _json_safe(payload)
        scanned += 1
        if scanned >= max(1, int(limit)):
            break
    return None


def find_artifact_registry_record_by_artifact_ref(
    artifact_ref: Any,
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    limit: int = ARTIFACT_REGISTRY_LOOKUP_LIMIT,
) -> Optional[dict[str, Any]]:
    normalized_artifact_ref = clean_text(artifact_ref)
    if not normalized_artifact_ref:
        return None
    scanned = 0
    for raw_line in reversed(_load_registry_lines(ledger_path)):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if _record_matches_artifact_ref(payload, normalized_artifact_ref):
            return _json_safe(payload)
        scanned += 1
        if scanned >= max(1, int(limit)):
            break
    return None


def find_generated_image_provenance(
    image_path: Any,
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    limit: int = ARTIFACT_REGISTRY_LOOKUP_LIMIT,
) -> Optional[dict[str, Any]]:
    normalized_image_path = _normalize_path(image_path)
    if not normalized_image_path:
        return None
    payload = find_artifact_registry_record(
        normalized_image_path,
        ledger_path=ledger_path,
        limit=limit,
    )
    if not isinstance(payload, Mapping):
        return None
    return _extract_generated_image_provenance(payload)


def find_generated_image_provenance_by_artifact_ref(
    artifact_ref: Any,
    *,
    ledger_path: Path | str = DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    limit: int = ARTIFACT_REGISTRY_LOOKUP_LIMIT,
) -> Optional[dict[str, Any]]:
    normalized_artifact_ref = clean_text(artifact_ref)
    if not normalized_artifact_ref:
        return None
    payload = find_artifact_registry_record_by_artifact_ref(
        normalized_artifact_ref,
        ledger_path=ledger_path,
        limit=limit,
    )
    if not isinstance(payload, Mapping):
        return None
    return _extract_generated_image_provenance(payload)
