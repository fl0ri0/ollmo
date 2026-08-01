"""Artifact-centered dossier helpers keyed by artifact_ref."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ollmo_services.artifact_registry import (
    find_artifact_registry_record,
    find_artifact_registry_record_by_artifact_ref,
    find_late_fill_artifact_source,
)
from ollmo_services.artifact_contracts import clean_text, sanitize_artifact_records


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _merge_unique_texts(values: list[str], *extra_values: Any) -> list[str]:
    items = [str(item).strip() for item in values if str(item).strip()]
    for value in extra_values:
        if isinstance(value, list):
            for item in value:
                token = clean_text(item)
                if token and token not in items:
                    items.append(token)
            continue
        token = clean_text(value)
        if token and token not in items:
            items.append(token)
    return items


def _sanitize_artifacts(value: Any) -> list[dict[str, Any]]:
    return sanitize_artifact_records(value if isinstance(value, list) else [], include_content=True, content_limit=4000)


def _find_artifact_registry_entry(artifact: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    artifact_ref = clean_text(artifact.get('artifact_ref') or artifact.get('ref'))
    if artifact_ref:
        record = find_artifact_registry_record_by_artifact_ref(artifact_ref)
        if isinstance(record, Mapping):
            return dict(record)
    artifact_path = clean_text(artifact.get('path'))
    if artifact_path:
        record = find_artifact_registry_record(artifact_path)
        if isinstance(record, Mapping):
            return dict(record)
    return None


def _artifact_matches_primary_response_output(
    artifact: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> bool:
    artifact_path = clean_text(artifact.get('path'))
    if not artifact_path:
        return False
    for key in ('saved_image_path', 'saved_audio_path', 'saved_text_path'):
        if artifact_path and artifact_path == clean_text(response_payload.get(key)):
            return True
    return False


def _build_provenance_payload(
    artifact: Mapping[str, Any],
    *,
    registry_record: Optional[Mapping[str, Any]],
    late_fill_source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'provenance_id': clean_text(artifact.get('provenance_id')),
        'derived_from': list(artifact.get('derived_from') or []) if isinstance(artifact.get('derived_from'), list) else [],
        'source_response_id': clean_text(artifact.get('source_response_id')),
        'source_message_id': clean_text(artifact.get('source_message_id')),
        'origin': clean_text(artifact.get('origin')),
    }
    provenance_record = (
        registry_record.get('provenance')
        if isinstance((registry_record or {}).get('provenance'), Mapping)
        else {}
    )
    if isinstance(provenance_record, Mapping):
        payload['provenance_id'] = clean_text(provenance_record.get('provenance_id')) or payload['provenance_id']
        payload['derived_from'] = _merge_unique_texts(
            payload.get('derived_from') if isinstance(payload.get('derived_from'), list) else [],
            provenance_record.get('derived_from'),
        )
        source = provenance_record.get('source') if isinstance(provenance_record.get('source'), Mapping) else {}
        request = provenance_record.get('request') if isinstance(provenance_record.get('request'), Mapping) else {}
        output = provenance_record.get('output') if isinstance(provenance_record.get('output'), Mapping) else {}
        payload['source_response_id'] = clean_text(source.get('response_id')) or payload['source_response_id']
        payload['origin'] = clean_text(source.get('request_origin')) or payload['origin']
        payload['record'] = {
            'source': _json_safe(source),
            'request': _json_safe(request),
            'output': _json_safe(output),
        }
    if isinstance(late_fill_source, Mapping) and late_fill_source:
        producer_source = dict(
            payload.get('record', {}).get('source')
            if isinstance(payload.get('record'), Mapping)
            and isinstance(payload.get('record', {}).get('source'), Mapping)
            else {}
        )
        producer_source.update(
            {
                key: value
                for key, value in late_fill_source.items()
                if value not in (None, '', [], {})
            }
        )
        record = dict(payload.get('record') or {})
        record['source'] = _json_safe(producer_source)
        payload['record'] = record
        payload['source_response_id'] = (
            clean_text(producer_source.get('response_id'))
            or payload.get('source_response_id')
        )
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, '', [], {})
    }


def _build_metadata_payload(
    artifact: Mapping[str, Any],
    *,
    response_payload: Mapping[str, Any],
    registry_record: Optional[Mapping[str, Any]],
    late_fill_source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(
        registry_record.get('metadata') or {}
    ) if isinstance((registry_record or {}).get('metadata'), Mapping) else {}
    for key in (
        'path',
        'source_path',
        'name',
        'mime_type',
        'availability',
        'availability_checked_at',
        'purged_at',
        'purge_reason',
        'seed',
        'batch_index',
        'timestamp',
        'prompt',
        'prompt_preview',
        'response_model',
        'response_instance_id',
    ):
        value = artifact.get(key)
        if value not in (None, '', [], {}):
            payload[key] = value
    provenance_record = (
        registry_record.get('provenance')
        if isinstance((registry_record or {}).get('provenance'), Mapping)
        else {}
    )
    if isinstance(provenance_record, Mapping):
        request = provenance_record.get('request') if isinstance(provenance_record.get('request'), Mapping) else {}
        source = provenance_record.get('source') if isinstance(provenance_record.get('source'), Mapping) else {}
        for key in ('width', 'height', 'seed'):
            value = request.get(key)
            if value not in (None, '', [], {}):
                payload[key] = value
        prompt_preview = request.get('prompt_preview')
        if prompt_preview not in (None, '', [], {}):
            payload.setdefault('prompt_preview', prompt_preview)
        if source.get('model') not in (None, ''):
            payload.setdefault('response_model', source.get('model'))
        if source.get('instance_id') not in (None, ''):
            payload.setdefault('response_instance_id', source.get('instance_id'))
        if source.get('backend') not in (None, ''):
            payload.setdefault('backend', source.get('backend'))
        if source.get('capability') not in (None, ''):
            payload.setdefault('capability', source.get('capability'))
        if source.get('mode') not in (None, ''):
            payload.setdefault('mode', source.get('mode'))
    if isinstance(late_fill_source, Mapping):
        for source_key, metadata_key in (
            ('model', 'response_model'),
            ('instance_id', 'response_instance_id'),
            ('backend', 'backend'),
            ('capability', 'capability'),
            ('mode', 'mode'),
            ('branch_id', 'branch_id'),
            ('phase_id', 'phase_id'),
            ('task_id', 'task_id'),
            ('obligation_id', 'obligation_id'),
        ):
            value = late_fill_source.get(source_key)
            if value not in (None, '', [], {}):
                payload[metadata_key] = value
    if _artifact_matches_primary_response_output(artifact, response_payload):
        for key in ('model', 'instance_id', 'backend', 'capability', 'mode'):
            value = response_payload.get(key)
            if value not in (None, '', [], {}):
                payload.setdefault(
                    {
                        'model': 'response_model',
                        'instance_id': 'response_instance_id',
                    }.get(key, key),
                    value,
                )
        document_output_kind = response_payload.get('document_output_kind')
        if document_output_kind not in (None, '', [], {}):
            payload['document_output_kind'] = document_output_kind
        route_runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), Mapping)
            else {}
        )
        control_hints = (
            route_runtime.get('control_hints')
            if isinstance(route_runtime.get('control_hints'), Mapping)
            else {}
        )
        selected_controls: dict[str, Any] = {}
        for key in ('lang_code', 'language', 'voice', 'response_format', 'instruct', 'speed', 'pitch', 'seed'):
            value = response_payload.get(key)
            if value not in (None, '', [], {}):
                selected_controls[key] = value
        for key in ('lang_code', 'language', 'voice', 'response_format', 'instruct', 'width', 'height', 'seed'):
            value = control_hints.get(key)
            if value not in (None, '', [], {}):
                selected_controls.setdefault(key, value)
        if selected_controls:
            payload['selected_controls'] = _json_safe(selected_controls)
    artifact_path = clean_text(payload.get('path') or artifact.get('path'))
    if artifact_path and payload.get('availability') in (None, '', [], {}):
        exists = Path(artifact_path).exists()
        payload['availability'] = 'available' if exists else 'missing'
        payload['availability_source'] = 'filesystem'
        if exists:
            try:
                payload.setdefault('size_bytes', Path(artifact_path).stat().st_size)
            except OSError:
                pass
    return _json_safe(payload)


def _build_enrichment_payload(
    artifact: Mapping[str, Any],
    *,
    response_payload: Mapping[str, Any],
    registry_record: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(
        registry_record.get('enrichments') or {}
    ) if isinstance((registry_record or {}).get('enrichments'), Mapping) else {}
    image_state = artifact.get('image_state')
    if isinstance(image_state, Mapping) and image_state:
        payload['image_state'] = dict(image_state)
    if _artifact_matches_primary_response_output(artifact, response_payload):
        response_image_state = response_payload.get('image_state')
        if isinstance(response_image_state, Mapping) and response_image_state:
            payload['image_state'] = dict(response_image_state)
        enrichment_state = response_payload.get('image_state_enrichment')
        if isinstance(enrichment_state, Mapping) and enrichment_state:
            payload['image_state_enrichment'] = dict(enrichment_state)
    return _json_safe(payload)


def _collect_linked_ids(
    artifact: Mapping[str, Any],
    *,
    registry_record: Optional[Mapping[str, Any]],
    late_fill_source: Optional[Mapping[str, Any]] = None,
) -> tuple[list[str], list[str]]:
    response_ids = _merge_unique_texts([], artifact.get('source_response_id'))
    message_ids = _merge_unique_texts([], artifact.get('source_message_id'))
    if isinstance(registry_record, Mapping):
        response_ids = _merge_unique_texts(response_ids, registry_record.get('linked_response_ids'))
        message_ids = _merge_unique_texts(message_ids, registry_record.get('linked_message_ids'))
    provenance_record = (
        registry_record.get('provenance')
        if isinstance((registry_record or {}).get('provenance'), Mapping)
        else {}
    )
    if isinstance(provenance_record, Mapping):
        source = provenance_record.get('source') if isinstance(provenance_record.get('source'), Mapping) else {}
        response_ids = _merge_unique_texts(response_ids, source.get('response_id'))
        message_ids = _merge_unique_texts(message_ids, artifact.get('message_id'))
    if isinstance(late_fill_source, Mapping):
        response_ids = _merge_unique_texts(response_ids, late_fill_source.get('response_id'))
    return response_ids, message_ids


def build_artifact_dossier_index(
    *,
    input_artifacts: Any = None,
    reference_artifacts: Any = None,
    output_artifacts: Any = None,
    response_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    response = dict(response_payload or {}) if isinstance(response_payload, Mapping) else {}
    dossier_index: dict[str, dict[str, Any]] = {}
    role_payloads = (
        ('input', _sanitize_artifacts(input_artifacts)),
        ('reference', _sanitize_artifacts(reference_artifacts)),
        ('output', _sanitize_artifacts(output_artifacts)),
    )
    for role, artifacts in role_payloads:
        for artifact in artifacts:
            artifact_ref = clean_text(artifact.get('artifact_ref') or artifact.get('ref'))
            if not artifact_ref:
                continue
            dossier = dossier_index.setdefault(
                artifact_ref,
                {
                    'kind': 'ollmo.artifact_dossier',
                    'artifact_ref': artifact_ref,
                    'artifact_id': clean_text(artifact.get('artifact_id')),
                    'type': clean_text(artifact.get('type') or artifact.get('kind')) or 'artifact',
                    'roles': [],
                    'artifact': {},
                },
            )
            if role not in dossier['roles']:
                dossier['roles'].append(role)
            registry_record = _find_artifact_registry_entry(artifact)
            late_fill_source = (
                find_late_fill_artifact_source(response, artifact.get('path'))
                if role == 'output'
                else None
            )
            registry_artifact = (
                dict(registry_record.get('artifact') or {})
                if isinstance((registry_record or {}).get('artifact'), Mapping)
                else {}
            )
            if not clean_text(dossier.get('artifact_id')):
                dossier['artifact_id'] = clean_text(
                    artifact.get('artifact_id')
                    or registry_artifact.get('artifact_id')
                )
            if not clean_text(dossier.get('type')):
                dossier['type'] = clean_text(artifact.get('type') or registry_artifact.get('type')) or 'artifact'
            dossier['artifact'] = _json_safe(
                {**registry_artifact, **dict(dossier.get('artifact') or {}), **artifact}
            )
            provenance_payload = _build_provenance_payload(
                artifact,
                registry_record=registry_record,
                late_fill_source=late_fill_source,
            )
            if provenance_payload:
                dossier['provenance'] = provenance_payload
            metadata_payload = _build_metadata_payload(
                artifact,
                response_payload=response,
                registry_record=registry_record,
                late_fill_source=late_fill_source,
            )
            if metadata_payload:
                dossier['metadata'] = metadata_payload
            enrichment_payload = _build_enrichment_payload(
                artifact,
                response_payload=response,
                registry_record=registry_record,
            )
            if enrichment_payload:
                existing = dict(dossier.get('enrichments') or {})
                dossier['enrichments'] = _json_safe({**existing, **enrichment_payload})
            linked_response_ids, linked_message_ids = _collect_linked_ids(
                artifact,
                registry_record=registry_record,
                late_fill_source=late_fill_source,
            )
            if linked_response_ids:
                dossier['linked_response_ids'] = _merge_unique_texts(
                    dossier.get('linked_response_ids') if isinstance(dossier.get('linked_response_ids'), list) else [],
                    linked_response_ids,
                )
            if linked_message_ids:
                dossier['linked_message_ids'] = _merge_unique_texts(
                    dossier.get('linked_message_ids') if isinstance(dossier.get('linked_message_ids'), list) else [],
                    linked_message_ids,
                )
    return {
        artifact_ref: _json_safe(dossier)
        for artifact_ref, dossier in dossier_index.items()
        if isinstance(dossier, Mapping)
    }
