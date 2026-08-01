"""Planning helpers for frozen response-frame snapshots."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Iterable, Optional

from ollmo_g.request_phase_graph import build_request_phase_graph, downstream_phase_records
from ollmo_services.artifact_contracts import extract_artifact_ref, sanitize_artifact_record
from ollmo_services.responses import extract_responses_current_turn_prompt


_TEXT_OUTPUT_CAPABILITIES = {'chat', 'ocr', 'stt', 'speech_to_text', 'vision', 'embedding'}
_DEFERRED_LATE_FILL_STATUSES = {'pending', 'queued', 'running', 'scheduled', 'accepted'}
_FAILED_RESPONSE_STATUSES = {'failed', 'error', 'cancelled'}
_WAIVED_SLOT_STATUSES = {
    'waived',
    'not_needed',
    'not-needed',
    'not_needed_verified',
    'skipped_verified',
    'unnecessary_verified',
}
_SUPERSEDED_SLOT_STATUSES = {
    'obsolete',
    'replaced',
    'superseded',
    'no-longer-relevant',
    'no_longer_relevant',
}
_TERMINAL_SLOT_STATUSES = {
    'blocked',
    'cancelled',
    'completed',
    'failed',
    'fulfilled',
    'skipped',
    'superseded',
    'waived',
}
_CANDIDATE_SLOT_STATES = {
    'candidate',
    'reserved',
    'possible',
    'draft',
    'optional',
    'not_promoted',
    'not-promoted',
    'unpromoted',
    'discarded',
    'rejected',
}
_PROMOTED_SLOT_STATES = {
    'promoted',
    'promoted_to_obligation',
    'promotion_accepted',
}
_NON_MATERIALIZABLE_ARTIFACT_STATUSES = {
    'blocked',
    'cancelled',
    'failed',
    'partial_failed',
    'repair_needed',
    'rejected',
    'skipped',
    'superseded',
    'waived',
}
_TEXT_ARTIFACT_EXTENSION_BY_MIME = {
    'application/javascript': 'js',
    'application/json': 'json',
    'text/css': 'css',
    'text/html': 'html',
    'text/javascript': 'js',
    'text/markdown': 'md',
    'text/plain': 'txt',
}
_TEXT_ARTIFACT_EXTENSION_ALIASES = {
    'htm': 'html',
    'markdown': 'md',
    'plain': 'txt',
    'text': 'txt',
    'xhtml': 'html',
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _clean_capability(value: Any) -> Optional[str]:
    token = _clean_text(value).lower().replace('-', '_')
    return token or None


def _clean_capability_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    items: list[str] = []
    for value in values:
        token = _clean_capability(value)
        if not token or token in items:
            continue
        items.append(token)
    return items


def _normalize_text_artifact_extension(value: Any) -> str:
    token = _clean_text(value).lower().lstrip('.')
    return _TEXT_ARTIFACT_EXTENSION_ALIASES.get(token, token)


def _extension_from_path_like(value: Any) -> str:
    token = _clean_text(value).split('?', 1)[0].split('#', 1)[0]
    filename = token.rsplit('/', 1)[-1]
    if not filename or '.' not in filename:
        return ''
    return _normalize_text_artifact_extension(filename.rsplit('.', 1)[-1])


def _text_artifact_extension_from_record(record: Mapping[str, Any]) -> str:
    request_payload = record.get('text_artifact_request') if isinstance(record.get('text_artifact_request'), Mapping) else {}
    for value in (
        record.get('text_artifact_extension'),
        request_payload.get('extension'),
        _TEXT_ARTIFACT_EXTENSION_BY_MIME.get(_clean_text(record.get('mime_type')).lower()),
        _extension_from_path_like(record.get('path') or record.get('source_path') or record.get('saved_text_path')),
        _extension_from_path_like(record.get('name')),
    ):
        normalized = _normalize_text_artifact_extension(value)
        if normalized:
            return normalized
    return ''


def _expected_text_artifact_extension(record: Mapping[str, Any]) -> str:
    request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
    for value in (
        record.get('text_artifact_extension'),
        request_payload.get('extension'),
    ):
        normalized = _normalize_text_artifact_extension(value)
        if normalized:
            return normalized
    return ''


def _expected_text_artifact_source_name(record: Mapping[str, Any]) -> str:
    request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
    return _clean_text(
        record.get('text_artifact_source_name')
        or request_payload.get('source_name')
    ).lower()


def _text_artifact_source_name_from_record(record: Mapping[str, Any]) -> str:
    request_payload = record.get('text_artifact_request') if isinstance(record.get('text_artifact_request'), Mapping) else {}
    return _clean_text(
        record.get('name')
        or record.get('source_name')
        or record.get('text_artifact_source_name')
        or request_payload.get('source_name')
    ).lower()


def _text_artifact_path_from_record(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('path') or record.get('source_path') or record.get('saved_text_path'))


def _expected_text_artifact_path(record: Mapping[str, Any]) -> str:
    request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
    return _clean_text(
        record.get('saved_text_path')
        or record.get('text_artifact_target_path')
        or request_payload.get('target_path')
    )


def _error_message_for_response(response_payload: Mapping[str, Any]) -> str:
    error_detail = response_payload.get('error_detail')
    if isinstance(error_detail, Mapping):
        message = _clean_text(error_detail.get('message'))
        if message:
            return message
    error = response_payload.get('error')
    if isinstance(error, Mapping):
        message = _clean_text(error.get('message'))
        if message:
            return message
    return _clean_text(error)


def _artifact_type(artifact: Mapping[str, Any]) -> str:
    return _clean_text(artifact.get('type') or artifact.get('kind')).lower() or 'artifact'


def _artifact_ref(artifact: Mapping[str, Any], *, index: int, role: str) -> str:
    return extract_artifact_ref(artifact, role=role, index=index)


def _artifact_availability(artifact: Mapping[str, Any]) -> str:
    explicit = _clean_text(artifact.get('availability')).lower()
    if explicit:
        return explicit
    if any(_clean_text(artifact.get(key)) for key in ('path', 'source_path', 'url')):
        return 'available'
    return 'provided'


def _artifact_lifecycle(artifact: Mapping[str, Any], *, role: str) -> str:
    if role == 'input':
        return 'source_input'
    if role == 'reference':
        return 'carried_reference'
    if _clean_text(artifact.get('derived_from')) or _clean_text(artifact.get('provenance_id')):
        return 'transformed_output'
    return 'materialized_output'


def _artifact_output_projection(
    artifact: Mapping[str, Any],
) -> tuple[str, str, str]:
    status = _clean_text(
        artifact.get('status') or artifact.get('output_status')
    ).lower()
    diagnostic_only = artifact.get('diagnostic_only') is True
    materialization_ineligible = (
        artifact.get('materialization_eligible') is False
    )
    if (
        status in _NON_MATERIALIZABLE_ARTIFACT_STATUSES
        or diagnostic_only
        or materialization_ineligible
    ):
        reason = _clean_text(
            artifact.get('integrity_reason_code')
            or artifact.get('blocked_reason')
            or status
        ) or 'artifact_not_materialization_eligible'
        return 'blocked', 'diagnostic_artifact', reason
    return (
        'fulfilled',
        _clean_text(artifact.get('lifecycle')) or 'materialized_output',
        '',
    )


def _route_hint_for_artifact(artifact_type: str, *, target_capability: Optional[str]) -> dict[str, Any]:
    if artifact_type in {'image', 'png', 'jpg', 'jpeg', 'webp'}:
        if target_capability in {'image_generation', 'image_edit', 'ocr', 'vision'}:
            return {
                'capability': target_capability,
                'kind': 'image_reference' if target_capability != 'ocr' else 'image_ocr_source',
                'reason': 'image input can anchor the selected visual or OCR route',
            }
        return {
            'capability': 'vision',
            'kind': 'image_understanding',
            'reason': 'image input needs visual understanding before text-only routing',
        }
    if artifact_type in {'pdf'}:
        return {
            'capability': 'ocr',
            'kind': 'document_extraction',
            'reason': 'PDF input normally needs text extraction before downstream work',
        }
    if artifact_type in {'audio', 'wav', 'mp3', 'm4a'}:
        return {
            'capability': 'stt',
            'kind': 'audio_transcription',
            'reason': 'audio input normally needs transcription before downstream work',
        }
    if artifact_type in {'text', 'markdown', 'md', 'json', 'csv'}:
        return {
            'capability': target_capability if target_capability in _TEXT_OUTPUT_CAPABILITIES else 'chat',
            'kind': 'text_context',
            'reason': 'text input can be routed as context for the selected text-capable path',
        }
    return {
        'capability': target_capability or 'chat',
        'kind': 'artifact_context',
        'reason': 'generic artifact is kept as request context until a narrower route claims it',
    }


def _normalize_artifacts(artifacts: Iterable[Mapping[str, Any]], *, role: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_artifact in enumerate(artifacts or [], start=1):
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_type = _artifact_type(raw_artifact)
        canonical = sanitize_artifact_record(
            raw_artifact,
            default_kind=artifact_type,
            default_origin={
                'input': 'user_input',
                'reference': 'conversation_reference',
                'output': 'assistant_output',
            }.get(role),
            include_content=artifact_type == 'message',
        ) or {}
        payload: dict[str, Any] = {
            'ref': _artifact_ref(raw_artifact, index=index, role=role),
            'artifact_id': canonical.get('artifact_id'),
            'artifact_ref': canonical.get('artifact_ref') or _artifact_ref(raw_artifact, index=index, role=role),
            'type': artifact_type,
            'role': role,
            'state': _artifact_availability(raw_artifact),
            'lifecycle': _artifact_lifecycle(canonical, role=role),
        }
        for key in (
            'path',
            'source_path',
            'name',
            'mime_type',
            'batch_index',
            'prompt',
            'origin',
            'source_message_id',
            'source_response_id',
            'provenance_id',
            'derived_from',
            'branch_id',
            'phase_id',
            'slot_id',
            'obligation_id',
            'output_type',
            'status',
            'output_status',
            'diagnostic_only',
            'materialization_eligible',
            'integrity_reason_code',
            'blocked_reason',
        ):
            value = canonical.get(key)
            if value in (None, '', [], {}):
                value = raw_artifact.get(key)
            if value not in (None, '', [], {}):
                payload[key] = value
        normalized.append(payload)
    return normalized


def _late_fill_payload(response_payload: Mapping[str, Any]) -> dict[str, Any]:
    late_fill = response_payload.get('late_fill')
    return dict(late_fill) if isinstance(late_fill, Mapping) else {}


def _planner_payload(response_payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(response_payload.get('execution_planner'), Mapping):
        return dict(response_payload.get('execution_planner') or {})
    runtime = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    planner = runtime.get('execution_planner')
    return dict(planner) if isinstance(planner, Mapping) else {}


def _output_type_for_capability(capability: Any) -> Optional[str]:
    token = _clean_capability(capability)
    if token in {'image_generation', 'image_edit', 'image'}:
        return 'image'
    if token in {'tts', 'text_to_speech', 'speech'}:
        return 'audio'
    if token in _TEXT_OUTPUT_CAPABILITIES:
        return 'text'
    return None


def _expected_output_type(response_payload: Mapping[str, Any]) -> Optional[str]:
    if _clean_text(response_payload.get('document_output_kind')).lower() == 'document':
        return 'document'
    capability = _clean_capability(response_payload.get('capability'))
    mode = _clean_capability(response_payload.get('mode'))
    return _output_type_for_capability(capability or mode)


def _expected_follow_up_output(response_payload: Mapping[str, Any]) -> tuple[Optional[str], Optional[str], str]:
    late_fill = _late_fill_payload(response_payload)
    late_fill_capability = _clean_capability(late_fill.get('expected_capability'))
    late_fill_type = _output_type_for_capability(late_fill_capability)
    if late_fill_type:
        return late_fill_type, late_fill_capability, 'late_fill'
    planner = _planner_payload(response_payload)
    deferred_capability = _clean_capability(planner.get('deferred_capability'))
    deferred_type = _output_type_for_capability(deferred_capability)
    if deferred_type:
        return deferred_type, deferred_capability, 'execution_planner'
    return None, None, ''


def _request_phase_graph_payload(
    request_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    existing = runtime.get('request_phase_graph') if isinstance(runtime.get('request_phase_graph'), Mapping) else {}
    if existing:
        return dict(existing)
    prompt = _clean_text(
        request_payload.get('prompt')
        or request_payload.get('input')
        or request_payload.get('instructions')
        or response_payload.get('output_text')
    )
    return build_request_phase_graph(
        prompt,
        intent_prompt=extract_responses_current_turn_prompt(dict(request_payload)),
        request_payload=request_payload,
        response_payload=response_payload,
    )


def _coerce_positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _slot_status_from_phase_status(status: Any) -> str:
    normalized = _clean_text(status).lower()
    if normalized in {'completed', 'fulfilled'}:
        return 'fulfilled'
    if normalized in {'blocked', 'failed', 'error'}:
        return 'blocked'
    if normalized == 'cancelled':
        return 'cancelled'
    if normalized in _WAIVED_SLOT_STATUSES:
        return 'waived'
    if normalized in _SUPERSEDED_SLOT_STATUSES:
        return 'superseded'
    if normalized in _PROMOTED_SLOT_STATES:
        return 'pending'
    return 'pending'


def _contract_state(record: Mapping[str, Any]) -> str:
    for key in ('contract_state', 'contract_status', 'obligation_state', 'intent_state'):
        token = _clean_text(record.get(key)).lower()
        if token:
            return token
    return _clean_text(record.get('status')).lower()


def _explicit_required_flag(record: Mapping[str, Any]) -> Optional[bool]:
    if 'required' not in record:
        return None
    value = record.get('required')
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', 'yes', '1', 'required'}:
        return True
    if token in {'false', 'no', '0', 'optional'}:
        return False
    return None


def _is_unpromoted_candidate_record(record: Mapping[str, Any]) -> bool:
    state = _contract_state(record)
    if state in _SUPERSEDED_SLOT_STATUSES and _clean_text(record.get('obligation_id') or record.get('promoted_from_candidate_id')):
        return False
    if state in _PROMOTED_SLOT_STATES or _clean_text(record.get('promoted_from_candidate_id')):
        return False
    if _clean_text(record.get('candidate_id')):
        return True
    if state in _CANDIDATE_SLOT_STATES:
        return True
    return _explicit_required_flag(record) is False


def _slot_lifecycle_from_status(status: str, *, pending_lifecycle: str = 'deferred_output') -> str:
    normalized = _clean_text(status).lower()
    if normalized == 'fulfilled':
        return 'materialized_output'
    if normalized == 'blocked':
        return 'blocked_output'
    if normalized == 'waived':
        return 'waived_output'
    if normalized == 'superseded':
        return 'superseded_output'
    if normalized == 'cancelled':
        return 'cancelled_output'
    return pending_lifecycle


def _slot_id_from_token(token: Any, *, fallback_index: int) -> str:
    normalized = _clean_text(token)
    if normalized:
        return f'output-{normalized}'
    return f'output-{fallback_index}'


def _output_node_id_from_token(token: Any, *, fallback_index: int) -> str:
    normalized = _clean_text(token)
    if normalized:
        return f'node-output-{normalized}'
    return f'node-output-{fallback_index}'


def _phase_records(phase_graph: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(phase_graph, Mapping):
        return []
    phases = phase_graph.get('phases')
    if not isinstance(phases, list):
        return []
    return [dict(item) for item in phases if isinstance(item, Mapping)]


def _current_phase_record(phase_graph: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(phase_graph, Mapping):
        return {}
    current_phase_id = _clean_text(phase_graph.get('current_phase_id'))
    for phase in _phase_records(phase_graph):
        if _clean_text(phase.get('phase_id')) != current_phase_id:
            continue
        return phase
    return {}


def _normalize_branch_record(raw_branch: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw_branch, Mapping):
        return None
    capability = _clean_capability(raw_branch.get('capability'))
    branch_id = _clean_text(raw_branch.get('branch_id') or raw_branch.get('phase_id'))
    phase_id = _clean_text(raw_branch.get('phase_id') or branch_id)
    output_type = _clean_text(raw_branch.get('output_type')).lower() or _output_type_for_capability(capability)
    if not capability or not branch_id or not output_type:
        return None
    payload: dict[str, Any] = {
        'branch_id': branch_id,
        'phase_id': phase_id or branch_id,
        'capability': capability,
        'output_type': output_type,
    }
    obligation_id = _clean_text(raw_branch.get('obligation_id'))
    if obligation_id:
        payload['obligation_id'] = obligation_id
    status = _clean_text(raw_branch.get('status')).lower()
    if status:
        payload['status'] = status
    error = raw_branch.get('error') if isinstance(raw_branch.get('error'), Mapping) else {}
    if error:
        normalized_error: dict[str, Any] = {}
        for key in ('code', 'message', 'stage', 'exception_type'):
            value = _clean_text(error.get(key))
            if value:
                normalized_error[key] = value
        retryable = error.get('retryable')
        if isinstance(retryable, bool):
            normalized_error['retryable'] = retryable
        status_code = error.get('status_code')
        if status_code not in (None, ''):
            try:
                normalized_error['status_code'] = int(status_code)
            except (TypeError, ValueError):
                pass
        if normalized_error:
            payload['error'] = normalized_error
    recovery_context = raw_branch.get('recovery_context') if isinstance(raw_branch.get('recovery_context'), Mapping) else {}
    if recovery_context:
        normalized_recovery: dict[str, Any] = {}
        for key in ('retry_scope', 'suggested_action'):
            value = _clean_text(recovery_context.get(key))
            if value:
                normalized_recovery[key] = value
        for key in ('can_retry', 'preserve_intent'):
            value = recovery_context.get(key)
            if isinstance(value, bool):
                normalized_recovery[key] = value
        exclude_instance_ids = [
            _clean_text(item)
            for item in (recovery_context.get('exclude_instance_ids') or [])
            if _clean_text(item)
        ] if isinstance(recovery_context.get('exclude_instance_ids'), list) else []
        if exclude_instance_ids:
            normalized_recovery['exclude_instance_ids'] = exclude_instance_ids
        if normalized_recovery:
            payload['recovery_context'] = normalized_recovery
    recovery_state = raw_branch.get('recovery_state') if isinstance(raw_branch.get('recovery_state'), Mapping) else {}
    if recovery_state:
        normalized_recovery_state: dict[str, Any] = {}
        for key in (
            'kind',
            'status',
            'trigger',
            'branch_id',
            'capability',
            'retry_scope',
            'suggested_action',
            'failed_instance_id',
        ):
            value = _clean_text(recovery_state.get(key))
            if value:
                normalized_recovery_state[key] = _clean_capability(value) if key == 'capability' else value
        for key in ('promotion_required', 'auto_execute', 'preserve_intent'):
            value = recovery_state.get(key)
            if isinstance(value, bool):
                normalized_recovery_state[key] = value
        exclude_instance_ids = [
            _clean_text(item)
            for item in (recovery_state.get('exclude_instance_ids') or [])
            if _clean_text(item)
        ] if isinstance(recovery_state.get('exclude_instance_ids'), list) else []
        if exclude_instance_ids:
            normalized_recovery_state['exclude_instance_ids'] = exclude_instance_ids
        if normalized_recovery_state:
            payload['recovery_state'] = normalized_recovery_state
    for key in (
        'content_payload',
        'content_payload_source',
        'stage_direction',
        'phase_summary',
        'requires_artifact',
        'text_artifact_extension',
        'text_artifact_source_name',
        'text_artifact_source',
        'text_artifact_target_path',
        'artifact_request',
        'role',
        'fulfillment_policy',
        'saved_text_path',
        'saved_audio_path',
        'saved_image_path',
        'image_data_url',
        'result',
        'superseded_by',
        'superseded_by_candidate_id',
        'superseded_by_obligation_id',
        'supersession_reason',
        'cancel_requested',
        'cancel_reason',
        'cancelled_by',
        'cancelled_at',
        'waiver_reason',
        'execution_gate',
    ):
        value = raw_branch.get(key)
        if value not in (None, '', [], {}):
            payload[key] = value
    return payload


def _normalize_branch_records(values: Any, *, status: str, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_value in values or []:
        normalized = _normalize_branch_record(raw_value)
        if not normalized:
            continue
        normalized['status'] = status
        normalized['source'] = source
        records.append(normalized)
    return records


def _response_artifact_type_count(response_payload: Mapping[str, Any], artifact_type: str) -> int:
    normalized_type = _clean_text(artifact_type).lower()
    if not normalized_type:
        return 0
    count = 0
    artifacts = response_payload.get('artifacts') if isinstance(response_payload.get('artifacts'), list) else []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        if _clean_text(artifact.get('type')).lower() == normalized_type:
            count += 1
    path_keys = {
        'audio': ('saved_audio_path',),
        'image': ('saved_image_path', 'image_path'),
        'text': ('saved_text_path',),
        'document': ('saved_text_path',),
    }.get(normalized_type, ())
    for key in path_keys:
        if _clean_text(response_payload.get(key)):
            count += 1
    return count


def _response_directly_fulfills_branch(
    response_payload: Mapping[str, Any],
    *,
    capability: str,
    output_type: str,
) -> bool:
    target_capability = _clean_capability(response_payload.get('capability') or response_payload.get('mode'))
    branch_capability = _clean_capability(capability)
    if not target_capability or target_capability == 'chat' or target_capability != branch_capability:
        return False
    if _clean_text(response_payload.get('status')).lower() not in {'completed', 'succeeded', 'ok'}:
        return False
    if isinstance(response_payload.get('error'), Mapping) or _clean_text(response_payload.get('error')):
        return False
    normalized_output_type = _clean_text(output_type).lower()
    if normalized_output_type in {'text', 'document'}:
        return bool(_clean_text(response_payload.get('output_text') or response_payload.get('content_payload')))
    if normalized_output_type == 'audio':
        return _response_artifact_type_count(response_payload, 'audio') > 0
    if normalized_output_type == 'image':
        return _response_artifact_type_count(response_payload, 'image') > 0
    return False


def _build_branch_output_specs(
    response_payload: Mapping[str, Any],
    request_phase_graph: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    late_fill = _late_fill_payload(response_payload)
    late_fill_status = _clean_text(late_fill.get('status')).lower()
    late_fill_expected_capability = _clean_capability(late_fill.get('expected_capability'))
    late_fill_completed_capabilities = _clean_capability_list(late_fill.get('completed_capabilities'))
    late_fill_failed_capabilities = _clean_capability_list(late_fill.get('failed_capabilities'))
    fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
    explicit_by_id: dict[str, dict[str, Any]] = {}
    explicit_order: list[str] = []
    phase_records = [
        phase
        for phase in _phase_records(request_phase_graph)
        if _clean_text(phase.get('phase_id')) != _clean_text((request_phase_graph or {}).get('current_phase_id'))
        and not _is_unpromoted_candidate_record(phase)
    ]
    phase_capability_counts: dict[str, int] = {}
    for phase in phase_records:
        capability = _clean_capability(phase.get('capability'))
        if capability:
            phase_capability_counts[capability] = phase_capability_counts.get(capability, 0) + 1

    def ingest(records: list[dict[str, Any]]) -> None:
        for record in records:
            branch_id = _clean_text(record.get('branch_id'))
            if not branch_id:
                continue
            if branch_id not in explicit_order:
                explicit_order.append(branch_id)
            previous = explicit_by_id.get(branch_id, {})
            explicit_by_id[branch_id] = {
                **previous,
                **record,
                'branch_id': branch_id,
                'phase_id': _clean_text(record.get('phase_id') or previous.get('phase_id') or branch_id) or branch_id,
            }

    ingest(_normalize_branch_records(late_fill.get('pending_branches'), status='pending', source='late_fill'))
    ingest(_normalize_branch_records(late_fill.get('active_branches'), status='pending', source='late_fill'))
    ingest(_normalize_branch_records(late_fill.get('completed_branches'), status='fulfilled', source='late_fill'))
    ingest(_normalize_branch_records(late_fill.get('failed_branches'), status='blocked', source='late_fill'))
    ingest(_normalize_branch_records(late_fill.get('cancelled_branches'), status='cancelled', source='late_fill'))
    for item in fill_results:
        if not isinstance(item, Mapping):
            continue
        normalized = _normalize_branch_record(item)
        if not normalized:
            continue
        normalized['status'] = 'fulfilled'
        normalized['source'] = 'late_fill'
        ingest([normalized])

    specs: list[dict[str, Any]] = []
    seen_branch_ids: set[str] = set()
    for phase in phase_records:
        phase_id = _clean_text(phase.get('phase_id'))
        if not phase_id or phase_id == _clean_text(request_phase_graph.get('current_phase_id')):
            continue
        capability = _clean_capability(phase.get('capability'))
        output_type = _clean_text(phase.get('output_type')).lower() or _output_type_for_capability(capability)
        if not capability or not output_type:
            continue
        phase_branch_id = _clean_text(phase.get('branch_id') or phase_id)
        explicit = explicit_by_id.get(phase_branch_id) or explicit_by_id.get(phase_id, {})
        explicit_status = _clean_text(explicit.get('status')).lower()
        unique_capability_phase = phase_capability_counts.get(capability, 0) <= 1
        direct_target_fulfilled = unique_capability_phase and _response_directly_fulfills_branch(
            response_payload,
            capability=capability,
            output_type=output_type,
        )
        branch_id = _clean_text(explicit.get('branch_id') or phase_branch_id or phase_id)
        if explicit_status:
            status = _slot_status_from_phase_status(explicit_status)
            if status == 'pending' and direct_target_fulfilled:
                status = 'fulfilled'
        elif direct_target_fulfilled:
            status = 'fulfilled'
        elif unique_capability_phase and capability in late_fill_completed_capabilities:
            status = 'fulfilled'
        elif unique_capability_phase and capability in late_fill_failed_capabilities:
            status = 'blocked'
        elif unique_capability_phase and late_fill_status == 'completed' and capability == late_fill_expected_capability:
            status = 'fulfilled'
        elif unique_capability_phase and late_fill_status == 'failed' and capability == late_fill_expected_capability:
            status = 'blocked'
        else:
            status = _slot_status_from_phase_status(phase.get('status'))
        spec = {
            'branch_id': branch_id,
            'phase_id': _clean_text(explicit.get('phase_id') or phase_id) or phase_id,
            'obligation_id': _clean_text(explicit.get('obligation_id') or phase.get('obligation_id')) or None,
            'capability': _clean_capability(explicit.get('capability') or capability),
            'output_type': _clean_text(explicit.get('output_type')).lower() or output_type,
            'status': status,
            'source': _clean_text(explicit.get('source')) or 'request_phase_graph',
        }
        if isinstance(explicit.get('error'), Mapping):
            spec['error'] = dict(explicit.get('error') or {})
        if isinstance(explicit.get('recovery_context'), Mapping):
            spec['recovery_context'] = dict(explicit.get('recovery_context') or {})
        if isinstance(explicit.get('recovery_state'), Mapping):
            spec['recovery_state'] = dict(explicit.get('recovery_state') or {})
        for key in (
            'content_payload',
            'content_payload_source',
            'stage_direction',
            'phase_summary',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'role',
            'fulfillment_policy',
            'saved_text_path',
            'saved_audio_path',
            'saved_image_path',
            'image_data_url',
            'result',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
        ):
            value = explicit.get(key)
            if value in (None, '', [], {}):
                value = phase.get(key)
            if value not in (None, '', [], {}):
                spec[key] = value
        if direct_target_fulfilled and _clean_text(spec.get('output_type')).lower() in {'text', 'document'}:
            output_text = _clean_text(response_payload.get('output_text') or response_payload.get('content_payload'))
            if output_text and spec.get('content_payload') in (None, '', [], {}):
                spec['content_payload'] = output_text
                spec['content_payload_source'] = 'direct_target_response'
        specs.append(spec)
        seen_branch_ids.add(branch_id)

    for branch_id in explicit_order:
        if branch_id in seen_branch_ids:
            continue
        explicit = explicit_by_id.get(branch_id, {})
        if not explicit:
            continue
        spec = {
            'branch_id': branch_id,
            'phase_id': _clean_text(explicit.get('phase_id') or branch_id) or branch_id,
            'obligation_id': _clean_text(explicit.get('obligation_id')) or None,
            'capability': _clean_capability(explicit.get('capability')),
            'output_type': _clean_text(explicit.get('output_type')).lower(),
            'status': _slot_status_from_phase_status(explicit.get('status')),
            'source': _clean_text(explicit.get('source')) or 'late_fill',
        }
        if isinstance(explicit.get('error'), Mapping):
            spec['error'] = dict(explicit.get('error') or {})
        if isinstance(explicit.get('recovery_context'), Mapping):
            spec['recovery_context'] = dict(explicit.get('recovery_context') or {})
        if isinstance(explicit.get('recovery_state'), Mapping):
            spec['recovery_state'] = dict(explicit.get('recovery_state') or {})
        for key in (
            'content_payload',
            'content_payload_source',
            'stage_direction',
            'phase_summary',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'role',
            'fulfillment_policy',
            'saved_text_path',
            'saved_audio_path',
            'saved_image_path',
            'image_data_url',
            'result',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
        ):
            value = explicit.get(key)
            if value not in (None, '', [], {}):
                spec[key] = value
        specs.append(spec)
    return [
        item
        for item in specs
        if item.get('branch_id') and item.get('capability') and item.get('output_type')
    ]


def _artifact_type_aliases(slot_type: str) -> list[str]:
    normalized = _clean_text(slot_type).lower()
    if normalized == 'image':
        return ['image', 'png', 'jpg', 'jpeg', 'webp']
    if normalized == 'audio':
        return ['audio', 'wav', 'mp3', 'm4a', 'flac']
    if normalized == 'document':
        return ['document', 'text', 'markdown', 'md', 'json', 'csv']
    if normalized == 'text':
        return ['text', 'markdown', 'md', 'json', 'csv', 'message']
    return [normalized] if normalized else []


def _take_output_artifact_by_type(
    available_artifacts: dict[str, list[dict[str, Any]]],
    slot_type: str,
) -> Optional[dict[str, Any]]:
    """Take only an unbound artifact for a generic type fallback.

    Callers already attempt exact branch/phase/path matching first. An artifact
    with explicit ownership must remain reserved for that owner instead of being
    consumed by an earlier compatible sibling.
    """
    for artifact_type in _artifact_type_aliases(slot_type):
        bucket = available_artifacts.get(artifact_type) or []
        for index, artifact in enumerate(bucket):
            if _artifact_binding_tokens(artifact):
                continue
            matched = bucket.pop(index)
            if not bucket:
                available_artifacts.pop(artifact_type, None)
            return matched
    return None


def _take_root_output_artifact_by_type(
    available_artifacts: dict[str, list[dict[str, Any]]],
    slot_type: str,
    root_slot: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Take an unbound artifact or one explicitly bound to the root slot.

    Downstream artifacts share broad output types with the preparation/root phase.
    Their explicit branch, phase, slot, or obligation binding is stronger truth than
    a type-only root fallback and must remain available for the owning branch.
    """
    root_tokens = _artifact_binding_tokens(root_slot)

    def pop_match(*, require_root_binding: bool) -> Optional[dict[str, Any]]:
        for artifact_type in _artifact_type_aliases(slot_type):
            bucket = available_artifacts.get(artifact_type) or []
            for index, artifact in enumerate(bucket):
                artifact_tokens = _artifact_binding_tokens(artifact)
                matches_root = bool(root_tokens.intersection(artifact_tokens))
                if require_root_binding and not matches_root:
                    continue
                if not require_root_binding and artifact_tokens:
                    continue
                matched = bucket.pop(index)
                if not bucket:
                    available_artifacts.pop(artifact_type, None)
                return matched
        return None

    explicitly_bound = pop_match(require_root_binding=True)
    if explicitly_bound:
        return explicitly_bound
    return pop_match(require_root_binding=False)


def _artifact_path_for_slot_match(record: Mapping[str, Any]) -> str:
    return _clean_text(
        record.get('path')
        or record.get('source_path')
        or record.get('saved_text_path')
        or record.get('saved_audio_path')
        or record.get('saved_image_path')
    )


def _artifact_binding_tokens(record: Mapping[str, Any]) -> set[str]:
    tokens = {
        _clean_text(record.get('branch_id')),
        _clean_text(record.get('phase_id')),
        _clean_text(record.get('slot_id')),
        _clean_text(record.get('obligation_id')),
    }
    tokens.discard('')
    return tokens


def _expected_saved_artifact_path(spec: Mapping[str, Any]) -> str:
    output_type = _clean_text(spec.get('output_type')).lower()
    if output_type == 'text':
        return _clean_text(spec.get('saved_text_path'))
    if output_type == 'audio':
        return _clean_text(spec.get('saved_audio_path'))
    if output_type == 'image':
        return _clean_text(spec.get('saved_image_path'))
    return _clean_text(
        spec.get('saved_text_path')
        or spec.get('saved_audio_path')
        or spec.get('saved_image_path')
    )


def _pop_artifact_for_spec_match(
    available_artifacts: dict[str, list[dict[str, Any]]],
    spec: Mapping[str, Any],
    *,
    match_path: bool,
) -> Optional[dict[str, Any]]:
    slot_type = _clean_text(spec.get('output_type')).lower()
    expected_path = _expected_saved_artifact_path(spec)
    spec_tokens = _artifact_binding_tokens(spec)
    for artifact_type in _artifact_type_aliases(slot_type):
        bucket = available_artifacts.get(artifact_type) or []
        for index, artifact in enumerate(bucket):
            if match_path:
                if not expected_path or _artifact_path_for_slot_match(artifact) != expected_path:
                    continue
            elif not spec_tokens.intersection(_artifact_binding_tokens(artifact)):
                continue
            matched = bucket.pop(index)
            if not bucket:
                available_artifacts.pop(artifact_type, None)
            return matched
    return None


def _take_output_artifact_for_spec(
    available_artifacts: dict[str, list[dict[str, Any]]],
    spec: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    return (
        _pop_artifact_for_spec_match(available_artifacts, spec, match_path=False)
        or _pop_artifact_for_spec_match(available_artifacts, spec, match_path=True)
    )


def _spec_requires_text_artifact(spec: Mapping[str, Any]) -> bool:
    if _clean_text(spec.get('output_type')).lower() != 'text':
        return False
    value = spec.get('requires_artifact')
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', 'yes', '1', 'required'}:
        return True
    role = _clean_text(spec.get('role')).lower()
    policy = _clean_text(spec.get('fulfillment_policy')).lower()
    return bool(_expected_text_artifact_extension(spec)) or role in {'text_artifact_output', 'document_output'} or policy == 'runtime_text_artifact'


def _text_artifact_matches_spec(artifact: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    if _clean_text(artifact.get('type')).lower() != 'text':
        return False
    if not _clean_text(artifact.get('path') or artifact.get('source_path')):
        return False
    expected_extension = _expected_text_artifact_extension(spec)
    artifact_extension = _text_artifact_extension_from_record(artifact)
    if expected_extension and artifact_extension != expected_extension:
        return False
    expected_source_name = _expected_text_artifact_source_name(spec)
    if expected_source_name:
        artifact_source_name = _text_artifact_source_name_from_record(artifact)
        if artifact_source_name and artifact_source_name != expected_source_name:
            return False
        if not artifact_source_name and not expected_extension:
            return False
    expected_path = _expected_text_artifact_path(spec)
    if expected_path:
        artifact_path = _text_artifact_path_from_record(artifact)
        if artifact_path and artifact_path != expected_path:
            return False
    return True


def _pop_text_artifact_by_path_for_spec(
    bucket: list[dict[str, Any]],
    spec: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    expected_path = _expected_text_artifact_path(spec)
    if not expected_path:
        return None
    for index, artifact in enumerate(bucket):
        if _text_artifact_path_from_record(artifact) != expected_path:
            continue
        if not _text_artifact_matches_spec(artifact, spec):
            continue
        return bucket.pop(index)
    return None


def _take_text_artifact_for_spec(
    available_artifacts: dict[str, list[dict[str, Any]]],
    spec: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    bucket = available_artifacts.get('text') or []
    matched_by_path = _pop_text_artifact_by_path_for_spec(bucket, spec)
    if matched_by_path:
        if not bucket:
            available_artifacts.pop('text', None)
        return matched_by_path
    for index, artifact in enumerate(bucket):
        if not _text_artifact_matches_spec(artifact, spec):
            continue
        matched = bucket.pop(index)
        if not bucket:
            available_artifacts.pop('text', None)
        return matched
    return None


def _blocked_reason_for_spec(
    spec: Mapping[str, Any],
    late_fill: Mapping[str, Any],
    response_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    error = spec.get('error') if isinstance(spec.get('error'), Mapping) else {}
    response_error = _error_message_for_response(response_payload or {}) if isinstance(response_payload, Mapping) else ''
    return _clean_text(error.get('message')) or _clean_text(late_fill.get('error')) or response_error or 'late_fill_failed'


def _error_ref_for_spec(spec: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    error = spec.get('error') if isinstance(spec.get('error'), Mapping) else {}
    branch_id = _clean_text(spec.get('branch_id'))
    code = _clean_text(error.get('code'))
    if not branch_id and not code:
        return None
    payload: dict[str, Any] = {}
    if branch_id:
        payload['branch_id'] = branch_id
    if code:
        payload['code'] = code
    stage = _clean_text(error.get('stage'))
    if stage:
        payload['stage'] = stage
    return payload


def _recovery_context_for_spec(spec: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    recovery = spec.get('recovery_context') if isinstance(spec.get('recovery_context'), Mapping) else {}
    if not recovery:
        return None
    payload: dict[str, Any] = {}
    for key in ('retry_scope', 'suggested_action'):
        value = _clean_text(recovery.get(key))
        if value:
            payload[key] = value
    for key in ('can_retry', 'preserve_intent'):
        value = recovery.get(key)
        if isinstance(value, bool):
            payload[key] = value
    exclude_instance_ids = [
        _clean_text(item)
        for item in (recovery.get('exclude_instance_ids') or [])
        if _clean_text(item)
    ] if isinstance(recovery.get('exclude_instance_ids'), list) else []
    if exclude_instance_ids:
        payload['exclude_instance_ids'] = exclude_instance_ids
    return payload or None


def _recovery_state_for_spec(spec: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    recovery = spec.get('recovery_state') if isinstance(spec.get('recovery_state'), Mapping) else {}
    if not recovery:
        return None
    payload: dict[str, Any] = {}
    for key in (
        'kind',
        'status',
        'trigger',
        'branch_id',
        'capability',
        'retry_scope',
        'suggested_action',
        'failed_instance_id',
    ):
        value = _clean_text(recovery.get(key))
        if value:
            payload[key] = _clean_capability(value) if key == 'capability' else value
    for key in ('promotion_required', 'auto_execute', 'preserve_intent'):
        value = recovery.get(key)
        if isinstance(value, bool):
            payload[key] = value
    exclude_instance_ids = [
        _clean_text(item)
        for item in (recovery.get('exclude_instance_ids') or [])
        if _clean_text(item)
    ] if isinstance(recovery.get('exclude_instance_ids'), list) else []
    if exclude_instance_ids:
        payload['exclude_instance_ids'] = exclude_instance_ids
    return payload or None


def _build_output_work_nodes(
    output_artifacts: list[dict[str, Any]],
    *,
    response_payload: Mapping[str, Any],
    request_phase_graph: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    late_fill = _late_fill_payload(response_payload)
    late_fill_status = _clean_text(late_fill.get('status')).lower()
    late_fill_completed_capabilities = _clean_capability_list(late_fill.get('completed_capabilities'))
    late_fill_failed_capabilities = _clean_capability_list(late_fill.get('failed_capabilities'))
    document_output_kind = _clean_text(response_payload.get('document_output_kind')).lower()
    response_status = _clean_text(response_payload.get('status')).lower()
    response_failed = response_status in _FAILED_RESPONSE_STATUSES
    response_error = _error_message_for_response(response_payload)
    branch_specs = _build_branch_output_specs(response_payload, request_phase_graph)
    available_artifacts: dict[str, list[dict[str, Any]]] = {}
    for artifact in output_artifacts:
        artifact_type = _clean_text(artifact.get('type')).lower()
        if not artifact_type:
            continue
        available_artifacts.setdefault(artifact_type, []).append(dict(artifact))
    nodes: list[dict[str, Any]] = []
    output_text = _clean_text(response_payload.get('output_text'))
    batch_count = _coerce_positive_int(response_payload.get('batch_count'))
    count = batch_count if batch_count > 0 else 1
    expected_type = _expected_output_type(response_payload)
    current_phase = _current_phase_record(request_phase_graph)
    current_phase_id = _clean_text(current_phase.get('phase_id')) or _clean_text(request_phase_graph.get('current_phase_id'))
    current_phase_output_type = (
        _clean_text(current_phase.get('output_type')).lower()
        or (expected_type if branch_specs else None)
        or ('text' if output_text else None)
    )
    branch_text_artifact_specs = [
        spec for spec in branch_specs if _spec_requires_text_artifact(spec)
    ]
    if branch_specs and current_phase_output_type:
        root_status = _slot_status_from_phase_status(current_phase.get('status'))
        if root_status == 'pending' and output_text:
            root_status = 'fulfilled'
        if (late_fill_status == 'failed' or response_failed) and root_status != 'fulfilled':
            root_status = 'blocked'
        root_node = {
            'node_id': _output_node_id_from_token(current_phase_id or 'current', fallback_index=1),
            'kind': 'output',
            'role': 'current_phase_output',
            'slot_id': _slot_id_from_token(current_phase_id or 'current', fallback_index=1),
            'type': 'document' if current_phase_output_type == 'text' and document_output_kind == 'document' else current_phase_output_type,
            'status': root_status,
            'lifecycle': _slot_lifecycle_from_status(
                root_status,
                pending_lifecycle='emerging_output',
            ),
        }
        obligation_id = _clean_text(current_phase.get('obligation_id'))
        if obligation_id:
            root_node['obligation_id'] = obligation_id
        if current_phase_id:
            root_node['phase_id'] = current_phase_id
        root_artifact = (
            None
            if branch_text_artifact_specs and _clean_text(root_node.get('type')).lower() in {'text', 'document'}
            else _take_root_output_artifact_by_type(
                available_artifacts,
                root_node['type'],
                root_node,
            )
        )
        if root_artifact:
            root_node['artifact_ref'] = root_artifact.get('artifact_ref') or root_artifact.get('ref')
            artifact_status, artifact_lifecycle, artifact_reason = (
                _artifact_output_projection(root_artifact)
            )
            if artifact_status != 'fulfilled':
                root_node['status'] = artifact_status
                root_node['lifecycle'] = artifact_lifecycle
                root_node['blocked_reason'] = artifact_reason
        elif root_status == 'pending':
            root_node['placeholder_ref'] = f'pending-output-{current_phase_id or "current"}'
        elif root_status == 'blocked':
            root_node['blocked_reason'] = _clean_text(late_fill.get('error')) or response_error or 'request_failed'
        nodes.append(root_node)
        child_node_ids: list[str] = []
        for index, spec in enumerate(branch_specs, start=2):
            slot_status = _slot_status_from_phase_status(spec.get('status'))
            branch_artifact: Optional[dict[str, Any]] = None
            spec_saved_text_path = _clean_text(spec.get('saved_text_path'))
            if _spec_requires_text_artifact(spec):
                branch_artifact = _take_text_artifact_for_spec(available_artifacts, spec)
                if branch_artifact or spec_saved_text_path:
                    slot_status = 'fulfilled'
                elif slot_status == 'fulfilled':
                    slot_status = 'pending'
            if response_failed and slot_status == 'pending':
                slot_status = 'blocked'
            child_node = {
                'node_id': _output_node_id_from_token(spec.get('phase_id') or spec.get('branch_id'), fallback_index=index),
                'kind': 'output',
                'role': 'branch_output',
                'slot_id': _slot_id_from_token(spec.get('phase_id') or spec.get('branch_id'), fallback_index=index),
                'type': spec.get('output_type'),
                'status': slot_status,
                'lifecycle': _slot_lifecycle_from_status(slot_status),
                'follow_up_capability': spec.get('capability'),
                'follow_up_source': spec.get('source'),
                'branch_id': spec.get('branch_id'),
                'phase_id': spec.get('phase_id'),
                'parent_node_id': root_node['node_id'],
            }
            if spec.get('obligation_id'):
                child_node['obligation_id'] = spec.get('obligation_id')
            if slot_status == 'fulfilled':
                if branch_artifact:
                    artifact = branch_artifact
                elif spec_saved_text_path and _clean_text(spec.get('output_type')).lower() == 'text':
                    artifact = _take_output_artifact_for_spec(available_artifacts, spec)
                    if not artifact:
                        artifact = sanitize_artifact_record(
                            {
                                'type': 'text',
                                'path': spec_saved_text_path,
                                'name': _expected_text_artifact_source_name(spec),
                                'origin': 'assistant_output',
                            }
                        ) or {
                            'type': 'text',
                            'path': spec_saved_text_path,
                        }
                elif (
                    branch_text_artifact_specs
                    and _clean_text(spec.get('output_type')).lower() == 'text'
                    and not _spec_requires_text_artifact(spec)
                ):
                    artifact = None
                else:
                    artifact = (
                        _take_output_artifact_for_spec(available_artifacts, spec)
                        or _take_output_artifact_by_type(available_artifacts, str(spec.get('output_type') or ''))
                    )
                if artifact:
                    child_node['artifact_ref'] = artifact.get('artifact_ref') or artifact.get('ref')
                    artifact_path = _clean_text(artifact.get('path') or artifact.get('source_path'))
                    if artifact_path:
                        child_node['artifact_path'] = artifact_path
                    artifact_status, artifact_lifecycle, artifact_reason = (
                        _artifact_output_projection(artifact)
                    )
                    if artifact_status != 'fulfilled':
                        child_node['status'] = artifact_status
                        child_node['lifecycle'] = artifact_lifecycle
                        child_node['blocked_reason'] = artifact_reason
                if spec.get('content_payload') not in (None, '', [], {}):
                    child_node['value'] = spec.get('content_payload')
                    if spec.get('saved_text_path'):
                        child_node['artifact_path'] = spec.get('saved_text_path')
            elif slot_status == 'pending':
                child_node['placeholder_ref'] = f'pending-output-{_clean_text(spec.get("branch_id") or spec.get("phase_id") or index)}'
            elif slot_status == 'blocked':
                child_node['blocked_reason'] = _blocked_reason_for_spec(spec, late_fill, response_payload)
                error_ref = _error_ref_for_spec(spec)
                if error_ref:
                    child_node['error_ref'] = error_ref
                recovery_context = _recovery_context_for_spec(spec)
                if recovery_context:
                    child_node['recovery_context'] = recovery_context
                recovery_state = _recovery_state_for_spec(spec)
                if recovery_state:
                    child_node['recovery_state'] = recovery_state
            elif slot_status == 'superseded':
                for key in ('superseded_by', 'superseded_by_candidate_id', 'superseded_by_obligation_id', 'supersession_reason'):
                    value = spec.get(key)
                    if value not in (None, '', [], {}):
                        child_node[key] = value
            nodes.append(child_node)
            child_node_ids.append(child_node['node_id'])
        if child_node_ids:
            root_node['child_node_ids'] = child_node_ids
    else:
        for index, artifact in enumerate(output_artifacts, start=1):
            slot_type = artifact.get('type') or 'artifact'
            if slot_type == 'text' and document_output_kind == 'document':
                slot_type = 'document'
            artifact_status, artifact_lifecycle, artifact_reason = (
                _artifact_output_projection(artifact)
            )
            node = {
                'node_id': _output_node_id_from_token(index, fallback_index=index),
                'kind': 'output',
                'role': 'materialized_output',
                'slot_id': f'output-{index}',
                'type': slot_type,
                'status': artifact_status,
                'artifact_ref': artifact.get('ref'),
                'lifecycle': artifact_lifecycle,
            }
            if artifact_reason:
                node['blocked_reason'] = artifact_reason
            if artifact.get('batch_index') not in (None, ''):
                node['batch_index'] = artifact.get('batch_index')
            nodes.append(node)
        # These artifacts are already represented as output nodes in the non-branch path.
        available_artifacts.clear()
    if not nodes and expected_type:
        if response_failed or (late_fill_status == 'failed' and expected_type in {'image', 'audio'}):
            status = 'blocked'
        else:
            status = 'fulfilled' if expected_type == 'text' and output_text else 'pending'
        if status == 'blocked':
            lifecycle = 'blocked_output'
        elif status == 'fulfilled':
            lifecycle = 'materialized_output'
        elif late_fill_status in _DEFERRED_LATE_FILL_STATUSES:
            lifecycle = 'deferred_output'
        else:
            lifecycle = 'emerging_output'
        for index in range(1, count + 1):
            node = {
                'node_id': _output_node_id_from_token(index, fallback_index=len(nodes) + 1),
                'kind': 'output',
                'role': 'expected_output',
                'slot_id': f'output-{len(nodes) + 1}',
                'type': expected_type,
                'status': status,
                'lifecycle': lifecycle,
            }
            if batch_count > 0:
                node['batch_index'] = index
            if status == 'pending':
                node['placeholder_ref'] = f'pending-output-{index}'
            if status == 'blocked':
                node['blocked_reason'] = _clean_text(late_fill.get('error')) or response_error or 'request_failed'
            nodes.append(node)
    if not branch_specs:
        phase_graph_follow_up_records = downstream_phase_records(request_phase_graph)
        follow_up_specs: list[dict[str, Any]] = []
        for record in phase_graph_follow_up_records:
            if _is_unpromoted_candidate_record(record):
                continue
            capability = _clean_capability(record.get('capability'))
            follow_up_type = _clean_text(record.get('output_type')).lower() or _output_type_for_capability(capability)
            phase_id = _clean_text(record.get('phase_id'))
            branch_id = _clean_text(record.get('branch_id') or phase_id)
            if not follow_up_type or not capability or not (branch_id or phase_id):
                continue
            follow_up_spec = {
                'type': follow_up_type,
                'capability': capability,
                'source': 'request_phase_graph',
                'phase_id': phase_id or branch_id,
                'branch_id': branch_id or phase_id,
                'obligation_id': _clean_text(record.get('obligation_id')) or None,
                'status': _slot_status_from_phase_status(record.get('status')),
            }
            for key in ('superseded_by', 'superseded_by_candidate_id', 'superseded_by_obligation_id', 'supersession_reason'):
                value = record.get(key)
                if value not in (None, '', [], {}):
                    follow_up_spec[key] = value
            follow_up_specs.append(follow_up_spec)
        legacy_follow_up_type, legacy_follow_up_capability, legacy_follow_up_source = _expected_follow_up_output(response_payload)
        if (
            legacy_follow_up_type
            and legacy_follow_up_capability
            and not phase_graph_follow_up_records
        ):
            follow_up_specs.append(
                {
                    'type': legacy_follow_up_type,
                    'capability': legacy_follow_up_capability,
                    'source': legacy_follow_up_source,
                }
            )
        existing_slot_keys = {
            (
                _clean_text(node.get('phase_id')),
                _clean_text(node.get('branch_id')),
                _clean_text(node.get('type')).lower(),
            )
            for node in nodes
            if isinstance(node, Mapping)
        }
        for follow_up in follow_up_specs:
            follow_up_type = _clean_text(follow_up.get('type')).lower()
            follow_up_capability = _clean_capability(follow_up.get('capability'))
            follow_up_source = _clean_text(follow_up.get('source')) or 'request_phase_graph'
            follow_up_phase_id = _clean_text(follow_up.get('phase_id'))
            follow_up_branch_id = _clean_text(follow_up.get('branch_id') or follow_up_phase_id)
            slot_identity = (follow_up_phase_id, follow_up_branch_id, follow_up_type)
            if slot_identity in existing_slot_keys:
                continue
            follow_up_status = _slot_status_from_phase_status(follow_up.get('status'))
            if follow_up_status == 'pending' and follow_up_capability in late_fill_completed_capabilities:
                follow_up_status = 'fulfilled'
            elif follow_up_status == 'pending' and (
                follow_up_capability in late_fill_failed_capabilities
                or (late_fill_status == 'failed' and _clean_capability(late_fill.get('expected_capability')) == follow_up_capability)
                or response_failed
            ):
                follow_up_status = 'blocked'
            next_index = len(nodes) + 1
            follow_up_slot = {
                'slot_id': f'output-{next_index}',
                'type': follow_up_type,
                'status': follow_up_status,
                'lifecycle': _slot_lifecycle_from_status(follow_up_status),
                'follow_up_capability': follow_up_capability,
                'follow_up_source': follow_up_source,
            }
            if _clean_text(follow_up.get('obligation_id')):
                follow_up_slot['obligation_id'] = _clean_text(follow_up.get('obligation_id'))
            if follow_up_phase_id:
                follow_up_slot['phase_id'] = follow_up_phase_id
            if follow_up_branch_id:
                follow_up_slot['branch_id'] = follow_up_branch_id
            if follow_up_status == 'pending':
                follow_up_slot['placeholder_ref'] = f'pending-output-{next_index}'
            if follow_up_status == 'blocked':
                follow_up_slot['blocked_reason'] = _clean_text(late_fill.get('error')) or response_error or 'request_failed'
            if follow_up_status == 'superseded':
                for key in ('superseded_by', 'superseded_by_candidate_id', 'superseded_by_obligation_id', 'supersession_reason'):
                    value = follow_up.get(key)
                    if value not in (None, '', [], {}):
                        follow_up_slot[key] = value
            nodes.append(
                {
                    'node_id': _output_node_id_from_token(next_index, fallback_index=next_index),
                    'kind': 'output',
                        'role': 'follow_up_output',
                        **follow_up_slot,
                    }
                )
            existing_slot_keys.add(slot_identity)
    for artifact_list in list(available_artifacts.values()):
        while artifact_list:
            artifact = artifact_list.pop(0)
            slot_type = artifact.get('type') or 'artifact'
            if slot_type == 'text' and document_output_kind == 'document':
                slot_type = 'document'
            artifact_status, artifact_lifecycle, artifact_reason = (
                _artifact_output_projection(artifact)
            )
            extra_slot = {
                'node_id': _output_node_id_from_token(len(nodes) + 1, fallback_index=len(nodes) + 1),
                'kind': 'output',
                'role': 'materialized_output',
                'slot_id': f'output-{len(nodes) + 1}',
                'type': slot_type,
                'status': artifact_status,
                'artifact_ref': artifact.get('artifact_ref') or artifact.get('ref'),
                'lifecycle': artifact_lifecycle,
            }
            if artifact_reason:
                extra_slot['blocked_reason'] = artifact_reason
            if artifact.get('batch_index') not in (None, ''):
                extra_slot['batch_index'] = artifact.get('batch_index')
            nodes.append(extra_slot)
    return nodes


def _build_work_tree(
    *,
    input_routes: list[dict[str, Any]],
    reference_routes: list[dict[str, Any]],
    output_artifacts: list[dict[str, Any]],
    request_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    request_phase_graph: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    request_root_id = 'node-request'
    response_status = _clean_text(response_payload.get('status')).lower()
    request_root_status = (
        'completed'
        if response_status == 'completed'
        else 'blocked'
        if response_status in _FAILED_RESPONSE_STATUSES
        else 'active'
    )
    request_root = {
        'node_id': request_root_id,
        'kind': 'request',
        'role': 'request_root',
        'status': request_root_status,
        'summary': _clean_text(
            request_payload.get('prompt')
            or request_payload.get('input')
            or request_payload.get('instructions')
        ) or 'request',
    }
    nodes.append(request_root)

    for index, route in enumerate(input_routes, start=1):
        routing_hint = route.get('routing_hint') if isinstance(route.get('routing_hint'), Mapping) else {}
        nodes.append(
            {
                'node_id': f'node-input-{index}',
                'kind': 'artifact_route',
                'role': 'source_input',
                'status': 'completed',
                'artifact_ref': route.get('artifact_ref'),
                'artifact_type': route.get('artifact_type'),
                'capability': _clean_text(routing_hint.get('capability')) or None,
                'parent_node_id': request_root_id,
            }
        )

    for index, route in enumerate(reference_routes, start=1):
        routing_hint = route.get('routing_hint') if isinstance(route.get('routing_hint'), Mapping) else {}
        nodes.append(
            {
                'node_id': f'node-reference-{index}',
                'kind': 'artifact_reference',
                'role': 'carried_reference',
                'status': 'completed',
                'artifact_ref': route.get('artifact_ref'),
                'artifact_type': route.get('artifact_type'),
                'capability': _clean_text(routing_hint.get('capability')) or None,
                'parent_node_id': request_root_id,
            }
        )

    for node in _build_output_work_nodes(
        output_artifacts,
        response_payload=response_payload,
        request_phase_graph=request_phase_graph,
    ):
        payload = dict(node)
        if _clean_text(payload.get('parent_node_id')):
            nodes.append(payload)
            continue
        payload['parent_node_id'] = request_root_id
        nodes.append(payload)

    node_map = {
        _clean_text(item.get('node_id')): item
        for item in nodes
        if isinstance(item, Mapping) and _clean_text(item.get('node_id'))
    }
    for node in node_map.values():
        node['child_node_ids'] = []
    for node in node_map.values():
        parent_id = _clean_text(node.get('parent_node_id'))
        if not parent_id or parent_id not in node_map:
            continue
        parent = node_map[parent_id]
        child_ids = parent.get('child_node_ids') if isinstance(parent.get('child_node_ids'), list) else []
        node_id = _clean_text(node.get('node_id'))
        if node_id and node_id not in child_ids:
            child_ids.append(node_id)
        parent['child_node_ids'] = child_ids

    pending_node_ids = [
        node_id
        for node_id, node in node_map.items()
        if _clean_text(node.get('status')).lower() == 'pending'
    ]
    blocked_node_ids = [
        node_id
        for node_id, node in node_map.items()
        if _clean_text(node.get('status')).lower() == 'blocked'
    ]
    output_root_node_ids = [
        _clean_text(node.get('node_id'))
        for node in node_map.values()
        if _clean_text(node.get('kind')) == 'output' and _clean_text(node.get('parent_node_id')) == request_root_id
    ]
    status = 'empty'
    if len(node_map) > 1:
        status = 'blocked' if blocked_node_ids else 'pending' if pending_node_ids else 'tracked'
    return {
        'kind': 'ollmo.work_tree',
        'status': status,
        'root_node_id': request_root_id,
        'node_order': [node_id for node_id in (_clean_text(item.get('node_id')) for item in nodes) if node_id],
        'nodes': nodes,
        'output_root_node_ids': output_root_node_ids,
        'pending_node_ids': pending_node_ids,
        'blocked_node_ids': blocked_node_ids,
    }


def _build_output_slots_from_work_tree(work_tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(work_tree, Mapping):
        return []
    raw_nodes = work_tree.get('nodes')
    if not isinstance(raw_nodes, list):
        return []
    node_map = {
        _clean_text(item.get('node_id')): dict(item)
        for item in raw_nodes
        if isinstance(item, Mapping) and _clean_text(item.get('node_id'))
    }
    slots: list[dict[str, Any]] = []
    for node_id in work_tree.get('node_order') if isinstance(work_tree.get('node_order'), list) else list(node_map.keys()):
        node = node_map.get(_clean_text(node_id))
        if not node or _clean_text(node.get('kind')) != 'output':
            continue
        slot = {
            'slot_id': _clean_text(node.get('slot_id')) or _slot_id_from_token(node_id, fallback_index=len(slots) + 1),
            'type': _clean_text(node.get('type')) or 'artifact',
            'status': _clean_text(node.get('status')) or 'pending',
            'lifecycle': _clean_text(node.get('lifecycle')) or _slot_lifecycle_from_status(node.get('status')),
        }
        for key in (
            'artifact_ref',
            'placeholder_ref',
            'blocked_reason',
            'follow_up_capability',
            'follow_up_source',
            'branch_id',
            'phase_id',
            'obligation_id',
            'artifact_path',
            'batch_index',
            'error_ref',
            'recovery_context',
            'recovery_state',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
            'value',
        ):
            value = node.get(key)
            if value not in (None, '', [], {}):
                slot[key] = value
        if _clean_text(slot.get('status')).lower() in _TERMINAL_SLOT_STATUSES:
            slot.pop('placeholder_ref', None)
        parent_node_id = _clean_text(node.get('parent_node_id'))
        if parent_node_id and parent_node_id in node_map and _clean_text(node_map[parent_node_id].get('kind')) == 'output':
            slot['parent_slot_id'] = _clean_text(node_map[parent_node_id].get('slot_id'))
        child_slot_ids = [
            _clean_text(node_map[child_id].get('slot_id'))
            for child_id in (node.get('child_node_ids') if isinstance(node.get('child_node_ids'), list) else [])
            if child_id in node_map and _clean_text(node_map[child_id].get('kind')) == 'output'
        ]
        if child_slot_ids:
            slot['child_slot_ids'] = child_slot_ids
        slots.append(slot)
    return slots


def _build_output_slots(
    output_artifacts: list[dict[str, Any]],
    *,
    response_payload: Mapping[str, Any],
    request_phase_graph: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    work_tree = _build_work_tree(
        input_routes=[],
        reference_routes=[],
        output_artifacts=output_artifacts,
        request_payload={},
        response_payload=response_payload,
        request_phase_graph=request_phase_graph,
    )
    return _build_output_slots_from_work_tree(work_tree)


def _provided_work_tree_from_mapping(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get('work_tree'), Mapping):
        candidate = value.get('work_tree')
        if _is_runtime_work_tree(candidate):
            return copy.deepcopy(dict(candidate))
    planning = value.get('planning') if isinstance(value.get('planning'), Mapping) else {}
    if isinstance(planning.get('work_tree'), Mapping) and _is_runtime_work_tree(planning.get('work_tree')):
        return copy.deepcopy(dict(planning.get('work_tree')))
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), Mapping) else {}
    if isinstance(artifact_flow.get('work_tree'), Mapping) and _is_runtime_work_tree(artifact_flow.get('work_tree')):
        return copy.deepcopy(dict(artifact_flow.get('work_tree')))
    runtime = value.get('runtime') if isinstance(value.get('runtime'), Mapping) else {}
    runtime_planning = runtime.get('planning') if isinstance(runtime.get('planning'), Mapping) else {}
    runtime_artifact_flow = (
        runtime_planning.get('artifact_flow')
        if isinstance(runtime_planning.get('artifact_flow'), Mapping)
        else {}
    )
    if (
        isinstance(runtime_artifact_flow.get('work_tree'), Mapping)
        and _is_runtime_work_tree(runtime_artifact_flow.get('work_tree'))
    ):
        return copy.deepcopy(dict(runtime_artifact_flow.get('work_tree')))
    response_frame = value.get('response_frame') if isinstance(value.get('response_frame'), Mapping) else {}
    frame_planning = response_frame.get('planning') if isinstance(response_frame.get('planning'), Mapping) else {}
    frame_artifact_flow = (
        frame_planning.get('artifact_flow')
        if isinstance(frame_planning.get('artifact_flow'), Mapping)
        else {}
    )
    if (
        isinstance(frame_artifact_flow.get('work_tree'), Mapping)
        and _is_runtime_work_tree(frame_artifact_flow.get('work_tree'))
    ):
        return copy.deepcopy(dict(frame_artifact_flow.get('work_tree')))
    if isinstance(frame_planning.get('work_tree'), Mapping) and _is_runtime_work_tree(frame_planning.get('work_tree')):
        return copy.deepcopy(dict(frame_planning.get('work_tree')))
    return None


def _is_runtime_work_tree(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    source = _clean_text(value.get('work_tree_source')).lower()
    if source and source != 'runtime_owned':
        return False
    if value.get('compatibility_derived') is True:
        return False
    if value.get('authoritative') is False:
        return False
    if isinstance(value.get('nodes'), list):
        return True
    return _clean_text(value.get('kind')) == 'ollmo.work_tree'


def _artifact_lookup_by_type(artifacts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        artifact_type = _clean_text(artifact.get('type')).lower() or 'artifact'
        lookup.setdefault(artifact_type, []).append(dict(artifact))
    return lookup


def _refresh_work_tree_indexes(work_tree: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = work_tree.get('nodes') if isinstance(work_tree.get('nodes'), list) else []
    node_map = {
        _clean_text(node.get('node_id')): node
        for node in raw_nodes
        if isinstance(node, dict) and _clean_text(node.get('node_id'))
    }
    for node in node_map.values():
        node['child_node_ids'] = []
    for node_id, node in node_map.items():
        parent_id = _clean_text(node.get('parent_node_id'))
        if not parent_id or parent_id not in node_map:
            continue
        child_ids = node_map[parent_id].get('child_node_ids') if isinstance(node_map[parent_id].get('child_node_ids'), list) else []
        if node_id not in child_ids:
            child_ids.append(node_id)
        node_map[parent_id]['child_node_ids'] = child_ids
    work_tree['node_order'] = [
        node_id for node_id in (_clean_text(node.get('node_id')) for node in raw_nodes if isinstance(node, Mapping)) if node_id
    ]
    work_tree['output_root_node_ids'] = [
        node_id
        for node_id, node in node_map.items()
        if _clean_text(node.get('kind')) == 'output'
        and _clean_text(node.get('parent_node_id')) == _clean_text(work_tree.get('root_node_id') or 'node-request')
    ]
    work_tree['pending_node_ids'] = [
        node_id for node_id, node in node_map.items() if _clean_text(node.get('status')).lower() == 'pending'
    ]
    work_tree['blocked_node_ids'] = [
        node_id for node_id, node in node_map.items() if _clean_text(node.get('status')).lower() in {'blocked', 'failed'}
    ]
    if len(node_map) > 1:
        work_tree['status'] = 'blocked' if work_tree['blocked_node_ids'] else 'pending' if work_tree['pending_node_ids'] else 'tracked'
    return work_tree


def _apply_late_fill_state_to_runtime_work_tree(
    work_tree: dict[str, Any],
    *,
    response_payload: Mapping[str, Any],
    request_phase_graph: Optional[Mapping[str, Any]],
    output_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(work_tree, dict):
        return work_tree
    specs = _build_branch_output_specs(response_payload, request_phase_graph)
    if not specs:
        return _refresh_work_tree_indexes(work_tree)
    spec_by_key: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for key in ('branch_id', 'phase_id', 'obligation_id'):
            token = _clean_text(spec.get(key))
            if token:
                spec_by_key[token] = spec
    available_artifacts = _artifact_lookup_by_type(output_artifacts)
    late_fill = _late_fill_payload(response_payload)
    response_failed = _clean_text(response_payload.get('status')).lower() in _FAILED_RESPONSE_STATUSES
    raw_nodes = work_tree.get('nodes') if isinstance(work_tree.get('nodes'), list) else []
    for node in raw_nodes:
        if not isinstance(node, dict) or _clean_text(node.get('kind')) != 'output':
            continue
        spec = None
        for key in ('branch_id', 'phase_id', 'obligation_id', 'slot_id'):
            token = _clean_text(node.get(key))
            if token and token in spec_by_key:
                spec = spec_by_key[token]
                break
        if not spec:
            continue
        slot_status = _slot_status_from_phase_status(spec.get('status'))
        if response_failed and slot_status == 'pending':
            slot_status = 'blocked'
        node['status'] = slot_status
        node['lifecycle'] = _slot_lifecycle_from_status(slot_status)
        for source_key, node_key in (
            ('capability', 'follow_up_capability'),
            ('source', 'follow_up_source'),
            ('branch_id', 'branch_id'),
            ('phase_id', 'phase_id'),
            ('obligation_id', 'obligation_id'),
        ):
            value = spec.get(source_key)
            if value not in (None, '', [], {}):
                node[node_key] = value
        if spec.get('output_type') not in (None, '', [], {}):
            node['type'] = spec.get('output_type')
        if slot_status in _TERMINAL_SLOT_STATUSES:
            node.pop('placeholder_ref', None)
        if slot_status == 'fulfilled':
            artifact = (
                _take_output_artifact_for_spec(available_artifacts, spec)
                or _take_output_artifact_by_type(available_artifacts, str(spec.get('output_type') or node.get('type') or ''))
            )
            if artifact:
                node['artifact_ref'] = artifact.get('artifact_ref') or artifact.get('ref')
                artifact_path = _clean_text(artifact.get('path') or artifact.get('source_path'))
                if artifact_path:
                    node['artifact_path'] = artifact_path
                artifact_status, artifact_lifecycle, artifact_reason = (
                    _artifact_output_projection(artifact)
                )
                if artifact_status != 'fulfilled':
                    node['status'] = artifact_status
                    node['lifecycle'] = artifact_lifecycle
                    node['blocked_reason'] = artifact_reason
            for key in ('content_payload', 'saved_text_path', 'saved_audio_path', 'saved_image_path'):
                value = spec.get(key)
                if value in (None, '', [], {}):
                    continue
                if key == 'content_payload':
                    node['value'] = value
                else:
                    node['artifact_path'] = value
        elif slot_status == 'blocked':
            node['blocked_reason'] = _blocked_reason_for_spec(spec, late_fill, response_payload)
            error_ref = _error_ref_for_spec(spec)
            if error_ref:
                node['error_ref'] = error_ref
            recovery_context = _recovery_context_for_spec(spec)
            if recovery_context:
                node['recovery_context'] = recovery_context
            recovery_state = _recovery_state_for_spec(spec)
            if recovery_state:
                node['recovery_state'] = recovery_state
        elif slot_status == 'pending':
            node.setdefault('placeholder_ref', f'pending-output-{_clean_text(node.get("branch_id") or node.get("phase_id") or node.get("slot_id"))}')
    return _refresh_work_tree_indexes(work_tree)


def _build_review_state(response_payload: Mapping[str, Any], output_slots: list[dict[str, Any]]) -> dict[str, Any]:
    status = _clean_text(response_payload.get('status')).lower()
    late_fill = _late_fill_payload(response_payload)
    late_fill_status = _clean_text(late_fill.get('status')).lower()
    if status in {'failed', 'error', 'cancelled'} or response_payload.get('error') or late_fill_status == 'failed':
        review_status = 'blocked'
    elif any(
        _clean_text(slot.get('status')).lower() in {'blocked', 'failed'}
        for slot in output_slots
        if isinstance(slot, Mapping)
    ):
        review_status = 'blocked'
    elif output_slots and all(_clean_text(slot.get('status')).lower() in {'fulfilled', 'superseded', 'waived', 'cancelled'} for slot in output_slots):
        review_status = 'ready'
    elif output_slots:
        review_status = 'pending_outputs'
    elif _clean_text(response_payload.get('output_text')):
        review_status = 'ready'
    else:
        review_status = 'pending'
    return {
        'status': review_status,
        'checkpoints': ['route', 'artifacts', 'outputs', 'memory_delta'],
        'pending_output_slot_ids': [
            str(slot.get('slot_id') or '').strip()
            for slot in output_slots
            if isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() == 'pending'
        ],
        'deferred_output_slot_ids': [
            str(slot.get('slot_id') or '').strip()
            for slot in output_slots
            if isinstance(slot, Mapping) and _clean_text(slot.get('lifecycle')).lower() == 'deferred_output'
        ],
        'blocked_output_slot_ids': [
            str(slot.get('slot_id') or '').strip()
            for slot in output_slots
            if isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() in {'blocked', 'failed'}
        ],
        'waived_output_slot_ids': [
            str(slot.get('slot_id') or '').strip()
            for slot in output_slots
            if isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() == 'waived'
        ],
        'superseded_output_slot_ids': [
            str(slot.get('slot_id') or '').strip()
            for slot in output_slots
            if isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() == 'superseded'
        ],
        'cancelled_output_slot_ids': [
            str(slot.get('slot_id') or '').strip()
            for slot in output_slots
            if isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() == 'cancelled'
        ],
    }


def build_artifact_flow_plan(
    input_artifacts: Iterable[Mapping[str, Any]],
    output_artifacts: Iterable[Mapping[str, Any]],
    *,
    reference_artifacts: Optional[Iterable[Mapping[str, Any]]] = None,
    request_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    response = response_payload if isinstance(response_payload, Mapping) else {}
    request = request_payload if isinstance(request_payload, Mapping) else {}
    target_capability = _clean_capability(response.get('capability') or request.get('capability'))
    request_phase_graph = _request_phase_graph_payload(request, response)
    normalized_inputs = _normalize_artifacts(input_artifacts, role='input')
    normalized_references = _normalize_artifacts(reference_artifacts or [], role='reference')
    normalized_outputs = _normalize_artifacts(output_artifacts, role='output')

    input_routes = []
    for artifact in normalized_inputs:
        input_routes.append(
            {
                'artifact_id': artifact.get('artifact_id'),
                'artifact_ref': artifact.get('artifact_ref') or artifact['ref'],
                'artifact_type': artifact['type'],
                'state': artifact['state'],
                'lifecycle': artifact.get('lifecycle'),
                'routing_hint': _route_hint_for_artifact(artifact['type'], target_capability=target_capability),
            }
        )

    reference_routes = []
    for artifact in normalized_references:
        reference_routes.append(
            {
                'artifact_id': artifact.get('artifact_id'),
                'artifact_ref': artifact.get('artifact_ref') or artifact['ref'],
                'artifact_type': artifact['type'],
                'state': artifact['state'],
                'lifecycle': artifact.get('lifecycle'),
                'routing_hint': _route_hint_for_artifact(artifact['type'], target_capability=target_capability),
            }
        )

    provided_work_tree = _provided_work_tree_from_mapping(response)
    work_tree = (
        _apply_late_fill_state_to_runtime_work_tree(
            provided_work_tree,
            response_payload=response,
            request_phase_graph=request_phase_graph,
            output_artifacts=normalized_outputs,
        )
        if provided_work_tree is not None
        else _build_work_tree(
            input_routes=input_routes,
            reference_routes=reference_routes,
            output_artifacts=normalized_outputs,
            request_payload=request,
            response_payload=response,
            request_phase_graph=request_phase_graph,
        )
    )
    work_tree_source = 'runtime_owned' if provided_work_tree is not None else 'derived_planning_snapshot'
    work_tree_authoritative = provided_work_tree is not None
    if isinstance(work_tree, dict):
        work_tree['work_tree_source'] = work_tree_source
        work_tree['authoritative'] = work_tree_authoritative
        work_tree['compatibility_derived'] = not work_tree_authoritative
    output_slots = _build_output_slots_from_work_tree(work_tree)
    steps: list[dict[str, Any]] = []
    for index, route in enumerate(input_routes, start=1):
        steps.append(
            {
                'step_id': f'input-route-{index}',
                'kind': 'artifact_route',
                'status': 'planned',
                'artifact_ref': route['artifact_ref'],
                'capability': route['routing_hint']['capability'],
            }
        )
    for index, route in enumerate(reference_routes, start=1):
        steps.append(
            {
                'step_id': f'reference-route-{index}',
                'kind': 'artifact_reference',
                'status': 'planned',
                'artifact_ref': route['artifact_ref'],
                'capability': route['routing_hint']['capability'],
            }
        )
    for index, slot in enumerate(output_slots, start=1):
        step = {
            'step_id': f'output-slot-{index}',
            'kind': 'output_materialization',
            'status': slot['status'],
            'slot_id': slot['slot_id'],
            'type': slot['type'],
            'lifecycle': slot.get('lifecycle'),
        }
        for key in ('parent_slot_id', 'child_slot_ids', 'branch_id', 'phase_id', 'follow_up_capability'):
            value = slot.get(key)
            if value in (None, '', [], {}):
                continue
            step[key] = value
        steps.append(step)

    memory_delta = response.get('memory_delta') if isinstance(response.get('memory_delta'), Mapping) else {}
    has_flow = bool(input_routes or reference_routes or output_slots)
    return {
        'kind': 'ollmo.artifact_flow_plan',
        'status': 'tracked' if has_flow else 'empty',
        'work_tree_source': work_tree_source,
        'authoritative': work_tree_authoritative,
        'compatibility_derived': not work_tree_authoritative,
        'work_tree': work_tree,
        'input_routes': input_routes,
        'reference_routes': reference_routes,
        'output_slots': output_slots,
        'steps': steps,
        'request_phase_graph': request_phase_graph,
        'review': _build_review_state(response, output_slots),
        'memory': {
            'delta_status': 'present' if memory_delta else 'empty',
            'artifact_refs': [
                artifact.get('artifact_ref') or artifact['ref']
                for artifact in normalized_inputs + normalized_references + normalized_outputs
            ],
        },
    }
