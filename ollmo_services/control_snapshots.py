"""Compact control snapshots for response-frame replay metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Optional

from ollmo_services.artifact_contracts import extract_artifact_ref, sanitize_artifact_record


_CONTROL_ALIASES: dict[str, tuple[str, ...]] = {
    'temperature': ('temperature',),
    'top_p': ('top_p', 'topP'),
    'max_tokens': ('max_tokens', 'maxTokens'),
    'reasoning_effort': ('reasoning_effort', 'reasoningEffort'),
    'language': ('language', 'stt_language', 'sttLanguage'),
    'task': ('task', 'stt_task', 'sttTask'),
    'voice': ('voice', 'tts_voice', 'ttsVoice'),
    'lang_code': ('lang_code', 'tts_language', 'ttsLanguage'),
    'response_format': ('response_format', 'responseFormat', 'tts_response_format', 'ttsResponseFormat'),
    'instruct': ('instruct', 'tts_instruct', 'ttsInstruct'),
    'speed': ('speed', 'tts_speed', 'ttsSpeed'),
    'pitch': ('pitch', 'tts_pitch', 'ttsPitch'),
    'width': ('width', 'image_width', 'imageWidth'),
    'height': ('height', 'image_height', 'imageHeight'),
    'aspect_ratio': ('aspect_ratio', 'aspectRatio', 'image_aspect_ratio', 'imageAspectRatio'),
    'image_count': ('image_count', 'imageCount'),
    'seed': ('seed',),
    'ocr_mode': ('ocr_mode', 'ocrMode'),
    'pdf_max_pages': ('pdf_max_pages', 'pdfMaxPages'),
    'pdf_dpi': ('pdf_dpi', 'pdfDpi'),
    'pdf_page_timeout_sec': ('pdf_page_timeout_sec', 'pdfPageTimeoutSec'),
    'pdf_synthesize': ('pdf_synthesize', 'pdfSynthesize'),
    'pdf_prefer_text': ('pdf_prefer_text', 'pdfPreferText'),
    'pdf_max_image_side': ('pdf_max_image_side', 'pdfMaxImageSide'),
    'pdf_page_retry_dpi': ('pdf_page_retry_dpi', 'pdfPageRetryDpi'),
    'reuse_cached': ('reuse_cached', 'reuseCached'),
    'infer_timeout_sec': ('infer_timeout_sec', 'inferTimeoutSec'),
}

_CONTROL_CATEGORIES: dict[str, tuple[str, ...]] = {
    'generation': ('temperature', 'top_p', 'max_tokens', 'reasoning_effort'),
    'audio': ('language', 'task', 'voice', 'lang_code', 'response_format', 'instruct', 'speed', 'pitch'),
    'image': ('width', 'height', 'aspect_ratio', 'image_count', 'seed'),
    'document': (
        'ocr_mode',
        'pdf_max_pages',
        'pdf_dpi',
        'pdf_page_timeout_sec',
        'pdf_synthesize',
        'pdf_prefer_text',
        'pdf_max_image_side',
        'pdf_page_retry_dpi',
    ),
    'cache': ('reuse_cached',),
    'execution': ('infer_timeout_sec',),
}

_BOOL_CONTROLS = {'pdf_synthesize', 'pdf_prefer_text', 'reuse_cached'}
_INT_CONTROLS = {
    'max_tokens',
    'width',
    'height',
    'image_count',
    'seed',
    'pdf_max_pages',
    'pdf_dpi',
    'pdf_page_timeout_sec',
    'pdf_max_image_side',
    'pdf_page_retry_dpi',
    'infer_timeout_sec',
}
_FLOAT_CONTROLS = {'temperature', 'top_p', 'speed', 'pitch'}

_DEFAULT_TEXT_VALUES: dict[str, set[str]] = {
    'language': {'auto'},
    'lang_code': {'auto'},
    'task': {'transcribe'},
    'ocr_mode': {'auto'},
    'aspect_ratio': {'auto', 'custom'},
    'reasoning_effort': {'off'},
}
_DEFAULT_NUMBER_VALUES: dict[str, float] = {
    'image_count': 1,
    'pdf_dpi': 300,
    'pdf_page_timeout_sec': 180,
    'speed': 1.0,
    'pitch': 1.0,
}
_DEFAULT_BOOL_VALUES: dict[str, bool] = {
    'pdf_synthesize': False,
    'pdf_prefer_text': False,
    'reuse_cached': False,
}

_TARGET_KEYS = ('instance_id', 'model', 'request_model', 'backend', 'capability', 'mode')
_ROUTE_KEYS = ('route_source', 'route_reason', 'route_confidence')


def _is_empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_control_sources(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = [source]
    for key in ('session_controls', 'sessionControls', 'settings'):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    return sources


def _candidate_sources(
    request_payload: Optional[Mapping[str, Any]],
    response_payload: Optional[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    request = _mapping(request_payload)
    response = _mapping(response_payload)
    if request:
        sources.extend(_nested_control_sources(request))
    if response:
        sources.extend(_nested_control_sources(response))
    return sources


def _first_present(sources: list[Mapping[str, Any]], aliases: tuple[str, ...]) -> Any:
    for source in sources:
        for key in aliases:
            if key not in source:
                continue
            value = source.get(key)
            if not _is_empty(value):
                return value
    return None


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if token in {'false', '0', 'no', 'n', 'off'}:
        return False
    return None


def _parse_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return number


def _normalize_control_value(control_key: str, value: Any) -> Any:
    if control_key in _BOOL_CONTROLS:
        parsed_bool = _parse_bool(value)
        return parsed_bool if parsed_bool is not None else value
    if control_key in _INT_CONTROLS:
        parsed_int = _parse_int(value)
        return parsed_int if parsed_int is not None else value
    if control_key in _FLOAT_CONTROLS:
        parsed_float = _parse_float(value)
        return parsed_float if parsed_float is not None else value
    text = _clean_text(value)
    return text if text else value


def _is_default_control_value(control_key: str, value: Any) -> bool:
    if _is_empty(value):
        return True
    if control_key in _DEFAULT_TEXT_VALUES:
        return _clean_text(value).lower() in _DEFAULT_TEXT_VALUES[control_key]
    if control_key in _DEFAULT_BOOL_VALUES:
        parsed_bool = _parse_bool(value)
        return parsed_bool is not None and parsed_bool == _DEFAULT_BOOL_VALUES[control_key]
    if control_key in _DEFAULT_NUMBER_VALUES:
        try:
            return float(value) == _DEFAULT_NUMBER_VALUES[control_key]
        except (TypeError, ValueError):
            return False
    return False


def _build_control_values(
    request_payload: Optional[Mapping[str, Any]],
    response_payload: Optional[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    sources = _candidate_sources(request_payload, response_payload)
    if not sources:
        return {}
    values: dict[str, dict[str, Any]] = {}
    for category, control_keys in _CONTROL_CATEGORIES.items():
        category_values: dict[str, Any] = {}
        for control_key in control_keys:
            raw_value = _first_present(sources, _CONTROL_ALIASES[control_key])
            if raw_value is None:
                continue
            value = _normalize_control_value(control_key, raw_value)
            if _is_default_control_value(control_key, value):
                continue
            category_values[control_key] = value
        if category_values:
            values[category] = category_values
    return values


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = _clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list[Any]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, list):
        return parsed
    if parsed is not None:
        return [parsed]
    if isinstance(value, list):
        return value
    if value not in (None, ''):
        return [value]
    return []


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _artifact_ref(value: Mapping[str, Any], *, index: int, role: str) -> str:
    return extract_artifact_ref(value, role=role, index=index)


def _compact_reference(value: Any, *, index: int, role: str) -> Optional[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    canonical = sanitize_artifact_record(
        value,
        include_content=True,
    )
    artifact_type = _clean_text((canonical or {}).get('type') or value.get('type') or value.get('kind')).lower() or 'artifact'
    payload: dict[str, Any] = {
        'artifact_id': (canonical or {}).get('artifact_id'),
        'artifact_ref': (canonical or {}).get('artifact_ref') or _artifact_ref(value, index=index, role=role),
        'ref': (canonical or {}).get('artifact_ref') or _artifact_ref(value, index=index, role=role),
        'type': artifact_type,
    }
    for key in (
        'path',
        'source_path',
        'name',
        'kind',
        'origin',
        'mime_type',
        'message_id',
        'response_model',
        'response_instance_id',
        'message_role',
        'seed',
    ):
        item = (canonical or {}).get(key)
        if not _is_empty(item):
            payload[key] = item
    content = _clean_text((canonical or {}).get('content') or value.get('content') or value.get('text') or value.get('prompt'))
    if content:
        payload['content_chars'] = len(content)
        payload['content_sha256'] = _content_digest(content)
    return payload


def _extract_selected_references(request_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_value = (
        request_payload.get('reference_artifacts')
        if request_payload.get('reference_artifacts') is not None
        else (
            request_payload.get('selected_reference_artifacts')
            if request_payload.get('selected_reference_artifacts') is not None
            else request_payload.get('selectedReferenceArtifacts')
        )
    )
    if raw_value is None:
        raw_value = (
            request_payload.get('selected_reference_artifact')
            if request_payload.get('selected_reference_artifact') is not None
            else request_payload.get('selectedReferenceArtifact')
        )
    references: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(raw_value), start=1):
        compacted = _compact_reference(item, index=index, role='selected_reference')
        if compacted:
            references.append(compacted)
    return references


def _build_reference_values(
    request_payload: Optional[Mapping[str, Any]],
    response_payload: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    request = _mapping(request_payload)
    response = _mapping(response_payload)
    values: dict[str, Any] = {}
    selected = _extract_selected_references(request)
    if selected:
        values['selected'] = selected
    file_path = _clean_text(request.get('file_path'))
    if file_path:
        values['file_path'] = file_path
    reference_image_count = _parse_int(response.get('reference_image_count'))
    reference_image_kind = _clean_text(response.get('reference_image_kind'))
    if reference_image_count and reference_image_count > 0:
        summary: dict[str, Any] = {'count': reference_image_count}
        if reference_image_kind:
            summary['kind'] = reference_image_kind
        values['image_reference'] = summary
    return values


def _build_batch_values(
    request_payload: Optional[Mapping[str, Any]],
    response_payload: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    request = _mapping(request_payload)
    response = _mapping(response_payload)
    batch_items = _as_list(request.get('batch_prompts') or request.get('batchPrompts'))
    batch_count = _parse_int(response.get('batch_count')) or len(batch_items)
    item_controls: list[dict[str, Any]] = []
    for index, item in enumerate(batch_items, start=1):
        if not isinstance(item, Mapping):
            continue
        controls: dict[str, Any] = {'index': index}
        for key, aliases in {
            'width': ('width', 'image_width', 'imageWidth'),
            'height': ('height', 'image_height', 'imageHeight'),
            'aspect_ratio': ('aspect_ratio', 'aspectRatio', 'image_aspect_ratio', 'imageAspectRatio'),
            'seed': ('seed',),
        }.items():
            raw_value = _first_present([item], aliases)
            if raw_value is None:
                continue
            value = _normalize_control_value(key, raw_value)
            if _is_default_control_value(key, value):
                continue
            controls[key] = value
        if len(controls) > 1:
            item_controls.append(controls)

    values: dict[str, Any] = {}
    if batch_count and batch_count > 1:
        values['count'] = batch_count
    if item_controls:
        values['item_controls'] = item_controls
    return values


def _select_keys(*sources: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        for source in sources:
            if key not in source:
                continue
            value = source.get(key)
            if _is_empty(value):
                continue
            payload[key] = value
            break
    return payload


def _count_values(values: Mapping[str, Any]) -> int:
    count = 0
    for value in values.values():
        if isinstance(value, Mapping):
            count += _count_values(value)
        elif isinstance(value, list):
            count += len(value)
        elif not _is_empty(value):
            count += 1
    return count


def build_control_snapshot(
    request_payload: Optional[Mapping[str, Any]],
    response_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build compact non-default control metadata for a frozen response frame."""

    request = _mapping(request_payload)
    response = _mapping(response_payload)
    values = _build_control_values(request, response)

    references = _build_reference_values(request, response)
    if references:
        values['references'] = references

    batch = _build_batch_values(request, response)
    if batch:
        values['batch'] = batch

    if not values:
        return {}

    snapshot: dict[str, Any] = {
        'kind': 'ollmo.control_snapshot',
        'status': 'tracked',
        'values': values,
        'control_count': _count_values(values),
        'replay': {
            'promotable': True,
            'settings_artifact': {'status': 'not_promoted'},
        },
    }
    target = _select_keys(response, request, keys=_TARGET_KEYS)
    if target:
        snapshot['target'] = target
    route = _select_keys(response, request, keys=_ROUTE_KEYS)
    if route:
        snapshot['route'] = route
    return snapshot
