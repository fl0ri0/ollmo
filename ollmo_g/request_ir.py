"""Canonical request IR helpers for output obligations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    normalize_capability,
)
from ollmo_g.candidate_contracts import (
    build_candidate_graph,
    review_candidate_promotions,
)
from ollmo_g.decision_contracts import build_ghost_decision_contract

REQUEST_IR_VERSION = 1
REQUEST_WORKLOAD_GRAPH_VERSION = 1
_WAIVED_OBLIGATION_STATUSES = {
    'waived',
    'not_needed',
    'not-needed',
    'not_needed_verified',
    'skipped_verified',
    'unnecessary_verified',
}
_SUPERSEDED_OBLIGATION_STATUSES = {
    'obsolete',
    'replaced',
    'superseded',
    'no-longer-relevant',
    'no_longer_relevant',
}
_CANDIDATE_CONTRACT_STATES = {
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
_PROMOTED_CONTRACT_STATES = {
    'promoted',
    'promoted_to_obligation',
    'promotion_accepted',
}
_ALLOWED_PROPOSAL_INPUT_REF_KINDS = {
    'artifact_prompt',
    'content_payload',
    'phase_output',
    'runtime_evidence',
    'selected_reference',
    'user_prompt',
}
_ALLOWED_PROPOSAL_VISIBILITIES = {'evidence', 'user_visible'}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _clip_text(value: Any, *, max_chars: int = 240) -> str:
    text = _clean_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + '...'


def _clean_string_list(value: Any, *, limit: int = 16, max_chars: int = 160) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, list):
        raw_items = list(value)
    else:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        text = _clip_text(raw_item, max_chars=max_chars)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _output_type_for_capability(capability: Any) -> Optional[str]:
    token = normalize_capability(capability)
    if token == CAPABILITY_IMAGE_GENERATION:
        return 'image'
    if token == CAPABILITY_TEXT_TO_SPEECH:
        return 'audio'
    if token == CAPABILITY_SPEECH_TO_TEXT:
        return 'text'
    if token == CAPABILITY_CHAT:
        return 'text'
    return None


def _workload_output_type_for_capability(capability: Any) -> Optional[str]:
    token = normalize_capability(capability)
    if token == CAPABILITY_VISION_ANALYSIS:
        return 'text'
    return _output_type_for_capability(token)


def _obligation_status(status: Any) -> str:
    normalized = _clean_text(status).lower()
    if normalized in {'completed', 'fulfilled'}:
        return 'fulfilled'
    if normalized in {'blocked', 'failed', 'error'}:
        return 'blocked'
    if normalized in _WAIVED_OBLIGATION_STATUSES:
        return 'waived'
    if normalized in _SUPERSEDED_OBLIGATION_STATUSES:
        return 'superseded'
    if normalized in _PROMOTED_CONTRACT_STATES:
        return 'pending'
    if normalized in {'active', 'running', 'queued', 'scheduled', 'accepted'}:
        return 'pending'
    return normalized or 'pending'


def _contract_state(phase: Mapping[str, Any]) -> str:
    for key in ('contract_state', 'contract_status', 'obligation_state', 'intent_state'):
        token = _clean_text(phase.get(key)).lower()
        if token:
            return token
    return _clean_text(phase.get('status')).lower()


def _explicit_required_flag(phase: Mapping[str, Any]) -> Optional[bool]:
    if 'required' not in phase:
        return None
    value = phase.get('required')
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', 'yes', '1', 'required'}:
        return True
    if token in {'false', 'no', '0', 'optional'}:
        return False
    return None


def _candidate_id_for_phase(phase: Mapping[str, Any], *, index: int) -> str:
    explicit = _clean_text(phase.get('candidate_id') or phase.get('promoted_from_candidate_id'))
    if explicit:
        return explicit
    phase_id = _clean_text(phase.get('phase_id'))
    if phase_id:
        return f'candidate-{phase_id}'
    branch_id = _clean_text(phase.get('branch_id'))
    if branch_id:
        return f'candidate-{branch_id}'
    return f'candidate-{index}'


def _candidate_status(phase: Mapping[str, Any]) -> str:
    state = _contract_state(phase)
    if state in _SUPERSEDED_OBLIGATION_STATUSES:
        return 'superseded'
    if state in _PROMOTED_CONTRACT_STATES:
        return 'promoted'
    if state in {'reserved'}:
        return 'reserved'
    if state in {'not_promoted', 'not-promoted', 'unpromoted'}:
        return 'not_promoted'
    if state in {'discarded', 'rejected'}:
        return 'discarded'
    if state in _CANDIDATE_CONTRACT_STATES:
        return 'candidate'
    return 'candidate'


def _phase_has_candidate_identity(phase: Mapping[str, Any]) -> bool:
    if _clean_text(phase.get('candidate_id') or phase.get('promoted_from_candidate_id')):
        return True
    state = _contract_state(phase)
    if state in _CANDIDATE_CONTRACT_STATES or state in _PROMOTED_CONTRACT_STATES:
        return True
    required = _explicit_required_flag(phase)
    return required is False


def _phase_is_promoted_candidate(phase: Mapping[str, Any]) -> bool:
    if _clean_text(phase.get('promoted_from_candidate_id')):
        return True
    if _contract_state(phase) in _PROMOTED_CONTRACT_STATES:
        return True
    if _contract_state(phase) in _SUPERSEDED_OBLIGATION_STATUSES and _clean_text(phase.get('obligation_id')):
        return True
    promotion = phase.get('promotion')
    if isinstance(promotion, Mapping) and _clean_text(promotion.get('to_obligation_id')):
        return True
    return False


def _obligation_id_for_phase(phase: Mapping[str, Any], *, index: int) -> str:
    explicit = _clean_text(phase.get('obligation_id'))
    if explicit:
        return explicit
    phase_id = _clean_text(phase.get('phase_id'))
    if phase_id:
        return f'obligation-{phase_id}'
    branch_id = _clean_text(phase.get('branch_id'))
    if branch_id:
        return f'obligation-{branch_id}'
    return f'obligation-{index}'


def _task_id_for_phase(phase: Mapping[str, Any], *, index: int) -> str:
    explicit = _clean_text(phase.get('workload_task_id') or phase.get('task_id'))
    if explicit:
        return explicit
    phase_id = _clean_text(phase.get('phase_id'))
    if phase_id:
        return f'task-{phase_id}'
    branch_id = _clean_text(phase.get('branch_id'))
    if branch_id:
        return f'task-{branch_id}'
    return f'task-{index}'


def _phase_dependency_ids(phase: Mapping[str, Any]) -> list[str]:
    return [
        _clean_text(item)
        for item in (phase.get('depends_on') or [])
        if _clean_text(item)
    ]


def _phase_input_refs(phase: Mapping[str, Any], *, current_phase_id: str) -> list[dict[str, Any]]:
    phase_id = _clean_text(phase.get('phase_id'))
    refs: list[dict[str, Any]] = []
    depends_on = _phase_dependency_ids(phase)
    if not depends_on and phase_id == _clean_text(current_phase_id):
        refs.append({'kind': 'user_prompt', 'ref': 'intent_anchor'})
    for dependency_id in depends_on:
        refs.append({'kind': 'phase_output', 'phase_id': dependency_id, 'role': 'dependency'})
    if phase.get('content_payload') not in (None, '', [], {}):
        refs.append({
            'kind': 'content_payload',
            'source': _clean_text(phase.get('content_payload_source')) or 'phase_record',
        })
    if phase.get('artifact_prompt') not in (None, '', [], {}):
        refs.append({
            'kind': 'artifact_prompt',
            'source': _clean_text(phase.get('artifact_prompt_source')) or 'phase_record',
        })
    if not refs:
        refs.append({'kind': 'user_prompt', 'ref': 'intent_anchor'})
    return refs


def _phase_requires_text_artifact(phase: Mapping[str, Any]) -> bool:
    value = phase.get('requires_artifact')
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', 'yes', '1', 'required'}:
        return True
    role = _clean_text(phase.get('role')).lower()
    policy = _clean_text(phase.get('fulfillment_policy')).lower()
    return role in {'text_artifact_output', 'document_output'} or policy == 'runtime_text_artifact'


def _fulfillment_policy_for_task(
    capability: str,
    output_type: Optional[str],
    *,
    requires_text_artifact: bool = False,
) -> str:
    if output_type in {'image', 'audio'}:
        return 'runtime_artifact_or_branch_state'
    if capability in {CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_VISION_ANALYSIS}:
        return 'runtime_evidence_text'
    if output_type == 'text' and requires_text_artifact:
        return 'runtime_text_artifact'
    if output_type == 'text':
        return 'runtime_text'
    return 'runtime_branch_state'


def _task_status(phase: Mapping[str, Any]) -> str:
    if _phase_has_candidate_identity(phase) and not _phase_is_promoted_candidate(phase):
        return _candidate_status(phase)
    return _obligation_status(phase.get('status'))


def _task_lifecycle(status: str) -> dict[str, Any]:
    normalized = _clean_text(status).lower()
    if normalized in {'fulfilled', 'completed'}:
        stage_statuses = {
            'prepare': 'fulfilled',
            'execute': 'fulfilled',
            'verify': 'fulfilled',
            'freeze': 'fulfilled',
        }
    elif normalized in _SUPERSEDED_OBLIGATION_STATUSES:
        stage_statuses = {
            'prepare': 'fulfilled',
            'execute': 'superseded',
            'verify': 'superseded',
            'freeze': 'fulfilled',
        }
    elif normalized in {'blocked', 'failed', 'error'}:
        stage_statuses = {
            'prepare': 'fulfilled',
            'execute': 'blocked',
            'verify': 'blocked',
            'freeze': 'blocked',
        }
    elif normalized in {'active', 'running'}:
        stage_statuses = {
            'prepare': 'fulfilled',
            'execute': 'active',
            'verify': 'pending',
            'freeze': 'pending',
        }
    elif normalized in {'pending', 'queued', 'scheduled', 'accepted'}:
        stage_statuses = {
            'prepare': 'fulfilled',
            'execute': 'pending',
            'verify': 'pending',
            'freeze': 'pending',
        }
    else:
        stage_statuses = {
            'prepare': 'pending',
            'execute': 'pending',
            'verify': 'pending',
            'freeze': 'pending',
        }
    return {
        'policy': 'prepare_execute_verify_freeze',
        'recursive': True,
        'scope': 'branch_local',
        'cycle': [
            'prepare',
            'gather_evidence',
            'execute',
            'verify',
            'repair_or_freeze',
        ],
        'repair_policy': 'repair_contract_or_dependency_before_retry',
        'stages': [
            {'stage': stage, 'status': stage_statuses[stage]}
            for stage in ('prepare', 'execute', 'verify', 'freeze')
        ],
    }


def _phase_decomposition_levels(phases: list[Mapping[str, Any]]) -> dict[str, int]:
    phase_ids = [
        _clean_text(phase.get('phase_id'))
        for phase in phases
        if isinstance(phase, Mapping) and _clean_text(phase.get('phase_id'))
    ]
    depths = {phase_id: 0 for phase_id in phase_ids}
    if not depths:
        return {}
    for _ in range(len(depths)):
        changed = False
        for phase in phases:
            phase_id = _clean_text(phase.get('phase_id')) if isinstance(phase, Mapping) else ''
            if not phase_id or phase_id not in depths:
                continue
            parent_depths = [
                depths[parent_id]
                for parent_id in _phase_dependency_ids(phase)
                if parent_id in depths
            ]
            next_depth = (max(parent_depths) + 1) if parent_depths else 0
            if next_depth > depths[phase_id]:
                depths[phase_id] = next_depth
                changed = True
        if not changed:
            break
    return depths


def _visibility_for_phase(phase: Mapping[str, Any], *, capability: str) -> str:
    kind = _clean_text(phase.get('kind')).lower()
    role = _clean_text(phase.get('role')).lower()
    if capability == CAPABILITY_VISION_ANALYSIS or kind == 'evidence' or 'evidence' in role:
        return 'evidence'
    return 'user_visible'


def _review_criteria_for_phase(
    phase: Mapping[str, Any],
    *,
    capability: str,
    output_type: Optional[str],
    is_current_phase: bool,
    has_downstream: bool,
) -> list[str]:
    criteria = [
        'output_contract_matches_capability',
        'runtime_status_reaches_fulfilled_blocked_failed_waived_superseded_or_pending',
    ]
    if output_type in {'image', 'audio'}:
        criteria.append('runtime_artifact_exists_when_fulfilled')
    elif capability in {CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_VISION_ANALYSIS}:
        criteria.append('runtime_evidence_text_exists_when_fulfilled')
    elif output_type == 'text' and _phase_requires_text_artifact(phase):
        criteria.append('runtime_text_artifact_exists_when_fulfilled')
    elif output_type == 'text':
        criteria.append('runtime_text_exists_when_fulfilled')
    if is_current_phase and has_downstream:
        criteria.append('preparation_text_is_bounded_to_downstream_inputs')
    if _phase_dependency_ids(phase):
        criteria.append('consumes_declared_input_refs')
    role = _clean_text(phase.get('role')).lower()
    stage_direction = _clean_text(phase.get('stage_direction')).lower()
    if role == 'post_artifact_text_follow_up' or stage_direction == 'write_text_after_artifact_generation':
        criteria.append('uses_dependency_evidence')
        criteria.append('does_not_restart_root_request')
    return criteria


def _input_ref_key(ref: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _clean_text(ref.get('kind')),
        _clean_text(ref.get('phase_id')),
        _clean_text(ref.get('ref')),
        _clean_text(ref.get('source')),
        _clean_text(ref.get('role')),
    )


def _sanitize_proposed_input_refs(
    value: Any,
    *,
    task: Mapping[str, Any],
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    if value in (None, '', [], {}):
        return [], None
    if not isinstance(value, list):
        return None, 'input_refs_not_list'
    depends_on = {
        _clean_text(item)
        for item in (task.get('depends_on') or [])
        if _clean_text(item)
    }
    task_phase_id = _clean_text(task.get('phase_id'))
    refs: list[dict[str, Any]] = []
    for raw_ref in value:
        if not isinstance(raw_ref, Mapping):
            return None, 'input_ref_not_object'
        kind = _clean_text(raw_ref.get('kind')).lower()
        if kind not in _ALLOWED_PROPOSAL_INPUT_REF_KINDS:
            return None, f'input_ref_kind_not_allowed:{kind or "missing"}'
        ref: dict[str, Any] = {'kind': kind}
        if kind == 'phase_output':
            phase_id = _clean_text(raw_ref.get('phase_id') or raw_ref.get('ref'))
            if phase_id not in depends_on:
                return None, f'input_ref_phase_output_not_declared_dependency:{phase_id or "missing"}'
            ref['phase_id'] = phase_id
            ref['role'] = _clip_text(raw_ref.get('role'), max_chars=80) or 'dependency'
        elif kind == 'runtime_evidence':
            phase_id = _clean_text(raw_ref.get('phase_id') or raw_ref.get('ref'))
            if phase_id and phase_id not in depends_on and phase_id != task_phase_id:
                return None, f'input_ref_runtime_evidence_not_declared_dependency:{phase_id}'
            if phase_id:
                ref['phase_id'] = phase_id
            source = _clip_text(raw_ref.get('source'), max_chars=120)
            if source:
                ref['source'] = source
        elif kind == 'user_prompt':
            ref['ref'] = _clip_text(raw_ref.get('ref'), max_chars=80) or 'intent_anchor'
        elif kind in {'artifact_prompt', 'content_payload'}:
            ref['source'] = _clip_text(raw_ref.get('source'), max_chars=120) or 'phase_record'
        elif kind == 'selected_reference':
            ref['ref'] = _clip_text(raw_ref.get('ref') or raw_ref.get('source'), max_chars=120) or 'selected_reference'
        refs.append(ref)
    return refs, None


def _merge_input_refs(existing: list[dict[str, Any]], proposed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw_ref in [*existing, *proposed]:
        if not isinstance(raw_ref, Mapping):
            continue
        ref = dict(raw_ref)
        key = _input_ref_key(ref)
        if key in seen:
            continue
        merged.append(ref)
        seen.add(key)
    return merged


def _default_obligation_id_for_task(task: Mapping[str, Any]) -> str:
    phase_id = _clean_text(task.get('phase_id'))
    if phase_id:
        return f'obligation-{phase_id}'
    branch_id = _clean_text(task.get('branch_id'))
    if branch_id:
        return f'obligation-{branch_id}'
    task_id = _clean_text(task.get('task_id'))
    if task_id:
        return f'obligation-{task_id}'
    return ''


def _proposal_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', 'yes', '1', 'required'}:
        return True
    if token in {'false', 'no', '0', 'optional'}:
        return False
    return None


def _reject_contract_field(
    field: str,
    proposed: Any,
    expected: Any,
) -> Optional[str]:
    proposed_text = _clean_text(proposed)
    expected_text = _clean_text(expected)
    if proposed_text and expected_text and proposed_text != expected_text:
        return f'execution_contract_{field}_mismatch'
    return None


def _sanitize_proposed_artifact_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    request: dict[str, Any] = {}
    for key in ('extension', 'source_name', 'source'):
        text = _clip_text(value.get(key), max_chars=120)
        if text:
            request[key] = text
    return request


def _sanitize_proposed_decision_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    allowed_text_keys = (
        'candidate_id',
        'candidate_type',
        'task_id',
        'phase_id',
        'branch_id',
        'obligation_id',
        'action',
        'status',
        'reason',
        'evidence_ref',
        'target_ref',
        'promotion_target',
        'promotion_reason',
        'waiver_reason',
        'waiver_policy',
        'release_evidence',
        'repair_action',
        'recovery_action',
        'superseded_by',
        'superseded_by_candidate_id',
        'superseded_by_obligation_id',
        'supersession_reason',
    )
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            continue
        item: dict[str, Any] = {}
        for key in allowed_text_keys:
            text = _clip_text(raw_item.get(key), max_chars=160)
            if text:
                item[key] = text
        evidence_refs = _clean_string_list(raw_item.get('evidence_refs'), limit=12, max_chars=160)
        if evidence_refs:
            item['evidence_refs'] = evidence_refs
        if item:
            items.append(item)
        if len(items) >= 16:
            break
    return items


def _sanitize_proposed_execution_contract(
    value: Any,
    *,
    task: Mapping[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if value in (None, '', [], {}):
        return {}, None
    if not isinstance(value, Mapping):
        return None, 'execution_contract_not_object'

    task_id = _clean_text(task.get('task_id'))
    phase_id = _clean_text(task.get('phase_id'))
    branch_id = _clean_text(task.get('branch_id'))
    capability = normalize_capability(task.get('capability'))
    output_contract = task.get('output_contract') if isinstance(task.get('output_contract'), Mapping) else {}
    output_type = _clean_text(output_contract.get('output_type')).lower()
    obligation_id = _default_obligation_id_for_task(task)

    for field, proposed, expected in (
        ('phase_id', value.get('phase_id'), phase_id),
        ('branch_id', value.get('branch_id'), branch_id),
        ('task_id', value.get('task_id') or value.get('workload_task_id'), task_id),
        ('capability', normalize_capability(value.get('capability')), capability),
        ('output_type', _clean_text(value.get('output_type')).lower(), output_type),
        ('obligation_id', value.get('obligation_id'), obligation_id),
    ):
        mismatch = _reject_contract_field(field, proposed, expected)
        if mismatch:
            return None, mismatch

    workload_task_ref = value.get('workload_task_ref') if isinstance(value.get('workload_task_ref'), Mapping) else {}
    for field, expected in (('task_id', task_id), ('phase_id', phase_id), ('branch_id', branch_id)):
        mismatch = _reject_contract_field(f'workload_task_ref_{field}', workload_task_ref.get(field), expected)
        if mismatch:
            return None, mismatch

    output_obligation_ref = (
        value.get('output_obligation_ref')
        if isinstance(value.get('output_obligation_ref'), Mapping)
        else {}
    )
    for field, expected in (
        ('obligation_id', obligation_id),
        ('phase_id', phase_id),
        ('branch_id', branch_id),
        ('output_type', output_type),
    ):
        mismatch = _reject_contract_field(f'output_obligation_ref_{field}', output_obligation_ref.get(field), expected)
        if mismatch:
            return None, mismatch

    proposed_depends_on = _clean_string_list(value.get('depends_on'), limit=24, max_chars=96)
    task_depends_on = [
        _clean_text(item)
        for item in (task.get('depends_on') or [])
        if _clean_text(item)
    ]
    if proposed_depends_on and proposed_depends_on != task_depends_on:
        return None, 'execution_contract_depends_on_mismatch'

    proposed_output_contract = (
        value.get('output_contract')
        if isinstance(value.get('output_contract'), Mapping)
        else {}
    )
    proposed_contract_output_type = _clean_text(proposed_output_contract.get('output_type')).lower()
    if proposed_contract_output_type and proposed_contract_output_type != output_type:
        return None, 'execution_contract_output_contract_output_type_mismatch'
    proposed_required = _proposal_bool(proposed_output_contract.get('required'))
    if proposed_required is not None and proposed_required != bool(output_contract.get('required')):
        return None, 'execution_contract_output_contract_required_mismatch'
    proposed_policy = _clean_text(proposed_output_contract.get('fulfillment_policy'))
    task_policy = _clean_text(output_contract.get('fulfillment_policy'))
    if proposed_policy and task_policy and proposed_policy != task_policy:
        return None, 'execution_contract_fulfillment_policy_mismatch'

    proposed_input_refs, input_ref_error = _sanitize_proposed_input_refs(
        value.get('input_refs'),
        task=task,
    )
    if input_ref_error:
        return None, f'execution_contract_{input_ref_error}'

    normalized_output_contract = {
        key: output_contract.get(key)
        for key in ('output_type', 'required', 'fulfillment_policy')
        if output_contract.get(key) not in (None, '', [], {})
    }
    if 'requires_artifact' in proposed_output_contract and isinstance(proposed_output_contract.get('requires_artifact'), bool):
        normalized_output_contract['requires_artifact'] = proposed_output_contract.get('requires_artifact')

    contract: dict[str, Any] = {
        'kind': _clip_text(value.get('kind'), max_chars=80) or 'ollmo.execution_contract',
        'branch_id': branch_id or None,
        'phase_id': phase_id or None,
        'capability': capability or None,
        'output_type': output_type or None,
        'workload_task_ref': {
            'task_id': task_id or None,
            'phase_id': phase_id or None,
            'branch_id': branch_id or None,
        },
        'output_obligation_ref': {
            'obligation_id': obligation_id or None,
            'phase_id': phase_id or None,
            'branch_id': branch_id or None,
            'output_type': output_type or None,
        },
        'output_contract': normalized_output_contract,
        'depends_on': task_depends_on,
        'input_refs': proposed_input_refs
        or [
            dict(item)
            for item in (task.get('input_refs') or [])
            if isinstance(item, Mapping)
        ],
    }
    for key in (
        'role',
        'stage_direction',
        'content_payload_source',
        'artifact_prompt_source',
        'text_artifact_extension',
        'text_artifact_source_name',
        'text_artifact_source',
    ):
        text = _clip_text(value.get(key), max_chars=160)
        if text:
            contract[key] = text
    if isinstance(value.get('requires_artifact'), bool):
        contract['requires_artifact'] = value.get('requires_artifact')
    artifact_request = _sanitize_proposed_artifact_request(value.get('artifact_request'))
    if artifact_request:
        contract['artifact_request'] = artifact_request
    return {
        key: item
        for key, item in contract.items()
        if item not in (None, '', [], {})
    }, None


def _reject_workload_proposal(
    review: dict[str, Any],
    *,
    index: int,
    reason: str,
    proposal_id: Optional[str] = None,
    target: Optional[str] = None,
) -> None:
    rejection = {
        'index': index,
        'reason': reason,
    }
    if proposal_id:
        rejection['proposal_id'] = proposal_id
    if target:
        rejection['target'] = target
    review['rejections'].append(rejection)


def _workload_task_needs_semantic_proposal(task: Mapping[str, Any], *, total_task_count: int) -> bool:
    status = _clean_text(task.get('status')).lower()
    if status in {'reserved', 'candidate', 'possible', 'draft', 'optional', 'rejected', 'discarded'}:
        return False
    has_dependency = bool(
        [
            _clean_text(item)
            for item in (task.get('depends_on') or [])
            if _clean_text(item)
        ]
    )
    has_child_tasks = bool(
        [
            _clean_text(item)
            for item in (task.get('child_task_ids') or [])
            if _clean_text(item)
        ]
    )
    return bool(_clean_text(task.get('task_id'))) and (
        total_task_count > 1 or has_dependency or has_child_tasks
    )


def _attach_workload_proposal_coverage(
    review: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_task_ids = [
        _clean_text(task.get('task_id'))
        for task in tasks
        if _workload_task_needs_semantic_proposal(task, total_task_count=len(tasks))
    ]
    accepted_task_ids = [
        _clean_text(item.get('task_id'))
        for item in (review.get('accepted') or [])
        if isinstance(item, Mapping) and _clean_text(item.get('task_id'))
    ]
    accepted_unique = list(dict.fromkeys(accepted_task_ids))
    missing_task_ids = [
        task_id for task_id in expected_task_ids if task_id not in accepted_unique
    ]
    if not expected_task_ids:
        coverage_status = 'not_required'
    elif not missing_task_ids:
        coverage_status = 'complete'
    elif accepted_unique:
        coverage_status = 'partial'
    else:
        coverage_status = 'missing'
    review['coverage'] = {
        'kind': 'ollmo.workload_proposal_coverage',
        'policy': 'expected_for_multitask_or_dependent_work',
        'status': coverage_status,
        'expected_task_ids': expected_task_ids,
        'accepted_task_ids': accepted_unique,
        'missing_task_ids': missing_task_ids,
        'missing_count': len(missing_task_ids),
    }
    return review


def _apply_workload_task_proposals(
    tasks: list[dict[str, Any]],
    proposals: Optional[list[Mapping[str, Any]]],
) -> dict[str, Any]:
    review = {
        'kind': 'ollmo.workload_proposal_review',
        'proposal_version': 1,
        'status': 'no_proposals',
        'accepted_count': 0,
        'rejected_count': 0,
        'accepted': [],
        'rejections': [],
    }
    if not proposals:
        return _attach_workload_proposal_coverage(review, tasks)

    task_by_phase_id = {
        _clean_text(task.get('phase_id')): task
        for task in tasks
        if _clean_text(task.get('phase_id'))
    }
    task_by_task_id = {
        _clean_text(task.get('task_id')): task
        for task in tasks
        if _clean_text(task.get('task_id'))
    }
    task_by_branch_id = {
        _clean_text(task.get('branch_id')): task
        for task in tasks
        if _clean_text(task.get('branch_id'))
    }
    for index, raw_proposal in enumerate(proposals, start=1):
        if not isinstance(raw_proposal, Mapping):
            _reject_workload_proposal(review, index=index, reason='proposal_not_object')
            continue
        proposal_id = _clip_text(raw_proposal.get('proposal_id') or raw_proposal.get('id'), max_chars=96)
        phase_id = _clean_text(raw_proposal.get('phase_id'))
        task_id = _clean_text(raw_proposal.get('task_id') or raw_proposal.get('workload_task_id'))
        branch_id = _clean_text(raw_proposal.get('branch_id'))
        task = (
            task_by_phase_id.get(phase_id)
            or task_by_task_id.get(task_id)
            or task_by_branch_id.get(branch_id)
        )
        target = phase_id or task_id or branch_id or None
        if not task:
            _reject_workload_proposal(
                review,
                index=index,
                reason='target_task_not_found',
                proposal_id=proposal_id or None,
                target=target,
            )
            continue

        task_capability = normalize_capability(task.get('capability'))
        proposed_capability = normalize_capability(raw_proposal.get('capability'))
        if proposed_capability and proposed_capability != task_capability:
            _reject_workload_proposal(
                review,
                index=index,
                reason='capability_mismatch',
                proposal_id=proposal_id or None,
                target=_clean_text(task.get('task_id')),
            )
            continue

        output_contract = (
            raw_proposal.get('output_contract')
            if isinstance(raw_proposal.get('output_contract'), Mapping)
            else {}
        )
        proposed_output_type = _clean_text(
            output_contract.get('output_type') or raw_proposal.get('output_type')
        ).lower()
        task_output_contract = task.get('output_contract') if isinstance(task.get('output_contract'), Mapping) else {}
        task_output_type = _clean_text(task_output_contract.get('output_type')).lower()
        if proposed_output_type and proposed_output_type != task_output_type:
            _reject_workload_proposal(
                review,
                index=index,
                reason='output_type_mismatch',
                proposal_id=proposal_id or None,
                target=_clean_text(task.get('task_id')),
            )
            continue
        if 'required' in output_contract:
            required = _explicit_required_flag(output_contract)
            if required is not None and required != bool(task_output_contract.get('required')):
                _reject_workload_proposal(
                    review,
                    index=index,
                    reason='required_flag_mismatch',
                    proposal_id=proposal_id or None,
                    target=_clean_text(task.get('task_id')),
                )
                continue

        if raw_proposal.get('depends_on') not in (None, '', [], {}):
            proposed_depends_on = _clean_string_list(raw_proposal.get('depends_on'), limit=24, max_chars=96)
            task_depends_on = [
                _clean_text(item)
                for item in (task.get('depends_on') or [])
                if _clean_text(item)
            ]
            if proposed_depends_on != task_depends_on:
                _reject_workload_proposal(
                    review,
                    index=index,
                    reason='depends_on_mismatch',
                    proposal_id=proposal_id or None,
                    target=_clean_text(task.get('task_id')),
                )
                continue
        if raw_proposal.get('parent_task_ids') not in (None, '', [], {}):
            proposed_parent_ids = _clean_string_list(raw_proposal.get('parent_task_ids'), limit=24, max_chars=96)
            task_parent_ids = [
                _clean_text(item)
                for item in (task.get('parent_task_ids') or [])
                if _clean_text(item)
            ]
            if proposed_parent_ids != task_parent_ids:
                _reject_workload_proposal(
                    review,
                    index=index,
                    reason='parent_task_ids_mismatch',
                    proposal_id=proposal_id or None,
                    target=_clean_text(task.get('task_id')),
                )
                continue
        visibility = _clean_text(raw_proposal.get('visibility')).lower()
        if visibility:
            if visibility not in _ALLOWED_PROPOSAL_VISIBILITIES:
                _reject_workload_proposal(
                    review,
                    index=index,
                    reason='visibility_not_allowed',
                    proposal_id=proposal_id or None,
                    target=_clean_text(task.get('task_id')),
                )
                continue
            if visibility != _clean_text(task.get('visibility')).lower():
                _reject_workload_proposal(
                    review,
                    index=index,
                    reason='visibility_mismatch',
                    proposal_id=proposal_id or None,
                    target=_clean_text(task.get('task_id')),
                )
                continue

        proposed_input_refs, input_ref_error = _sanitize_proposed_input_refs(
            raw_proposal.get('input_refs'),
            task=task,
        )
        if input_ref_error:
            _reject_workload_proposal(
                review,
                index=index,
                reason=input_ref_error,
                proposal_id=proposal_id or None,
                target=_clean_text(task.get('task_id')),
            )
            continue

        proposed_execution_contract, execution_contract_error = _sanitize_proposed_execution_contract(
            raw_proposal.get('execution_contract'),
            task=task,
        )
        if execution_contract_error:
            _reject_workload_proposal(
                review,
                index=index,
                reason=execution_contract_error,
                proposal_id=proposal_id or None,
                target=_clean_text(task.get('task_id')),
            )
            continue

        semantic_intent = _clip_text(
            raw_proposal.get('semantic_intent')
            or raw_proposal.get('intent')
            or raw_proposal.get('task_intent'),
            max_chars=240,
        )
        if semantic_intent:
            task['semantic_intent'] = semantic_intent
        for key in ('objective', 'deliverable', 'rationale'):
            value = _clip_text(raw_proposal.get(key), max_chars=320)
            if value:
                task[key] = value
        for key in ('advisory_role', 'decision_notes', 'promotion_policy', 'reconsideration_policy'):
            value = _clip_text(raw_proposal.get(key), max_chars=320)
            if value:
                task[key] = value
        depth = raw_proposal.get('decomposition_level', raw_proposal.get('depth'))
        try:
            parsed_depth = int(depth)
        except (TypeError, ValueError):
            parsed_depth = None
        if parsed_depth is not None:
            task['decomposition_level'] = max(0, parsed_depth)

        proposed_review_criteria = [
            *(_clean_string_list(raw_proposal.get('review_criteria'), limit=24, max_chars=160)),
            *(_clean_string_list(raw_proposal.get('acceptance_criteria'), limit=24, max_chars=160)),
        ]
        if proposed_review_criteria:
            merged_criteria = _clean_string_list(
                [*(task.get('review_criteria') or []), *proposed_review_criteria],
                limit=48,
                max_chars=160,
            )
            task['review_criteria'] = merged_criteria
        semantic_review_criteria = _clean_string_list(
            raw_proposal.get('semantic_review_criteria'),
            limit=24,
            max_chars=160,
        )
        if semantic_review_criteria:
            task['semantic_review_criteria'] = _clean_string_list(
                [*(task.get('semantic_review_criteria') or []), *semantic_review_criteria],
                limit=48,
                max_chars=160,
            )
            task['review_criteria'] = _clean_string_list(
                [*(task.get('review_criteria') or []), *semantic_review_criteria],
                limit=48,
                max_chars=160,
            )
        for key in ('evidence_requirements', 'reconsideration_triggers', 'learning_hint_refs'):
            values = _clean_string_list(raw_proposal.get(key), limit=24, max_chars=160)
            if values:
                task[key] = _clean_string_list(
                    [*(task.get(key) or []), *values],
                    limit=48,
                    max_chars=160,
                )
        for key in ('promotion_suggestions', 'waiver_candidates', 'repair_candidates', 'supersession_candidates'):
            values = _sanitize_proposed_decision_mapping_list(raw_proposal.get(key))
            if values:
                existing = [
                    dict(item)
                    for item in (task.get(key) or [])
                    if isinstance(item, Mapping)
                ]
                task[key] = [*existing, *values][:32]
        if proposed_input_refs:
            task['input_refs'] = _merge_input_refs(
                [dict(item) for item in (task.get('input_refs') or []) if isinstance(item, Mapping)],
                proposed_input_refs,
            )
        if proposed_execution_contract:
            task['execution_contract'] = proposed_execution_contract
            workload_task_ref = (
                proposed_execution_contract.get('workload_task_ref')
                if isinstance(proposed_execution_contract.get('workload_task_ref'), Mapping)
                else {}
            )
            output_obligation_ref = (
                proposed_execution_contract.get('output_obligation_ref')
                if isinstance(proposed_execution_contract.get('output_obligation_ref'), Mapping)
                else {}
            )
            if workload_task_ref:
                task['workload_task_ref'] = dict(workload_task_ref)
            if output_obligation_ref:
                task['output_obligation_ref'] = dict(output_obligation_ref)
            if proposed_execution_contract.get('input_refs'):
                task['input_refs'] = _merge_input_refs(
                    [dict(item) for item in (task.get('input_refs') or []) if isinstance(item, Mapping)],
                    [
                        dict(item)
                        for item in (proposed_execution_contract.get('input_refs') or [])
                        if isinstance(item, Mapping)
                    ],
                )
            for key in (
                'role',
                'stage_direction',
                'content_payload_source',
                'artifact_prompt_source',
                'requires_artifact',
                'text_artifact_extension',
                'text_artifact_source_name',
                'text_artifact_source',
                'artifact_request',
            ):
                value = proposed_execution_contract.get(key)
                if value not in (None, '', [], {}):
                    task[key] = value
        artifact_request = _sanitize_proposed_artifact_request(raw_proposal.get('artifact_request'))
        if artifact_request:
            task['artifact_request'] = artifact_request
        if isinstance(raw_proposal.get('requires_artifact'), bool):
            task['requires_artifact'] = raw_proposal.get('requires_artifact')
        for key in (
            'stage_direction',
            'content_payload_source',
            'artifact_prompt_source',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
        ):
            value = _clip_text(raw_proposal.get(key), max_chars=160)
            if value:
                task[key] = value

        accepted_record = {
            'proposal_id': proposal_id or f'proposal-{index}',
            'task_id': _clean_text(task.get('task_id')),
            'phase_id': _clean_text(task.get('phase_id')),
            'source': _clip_text(raw_proposal.get('source'), max_chars=120) or 'ghost_workload_task_proposals_v1',
        }
        task.setdefault('accepted_proposals', []).append(accepted_record)
        review['accepted'].append(accepted_record)

    review['accepted_count'] = len(review['accepted'])
    review['rejected_count'] = len(review['rejections'])
    if review['accepted_count'] and review['rejected_count']:
        review['status'] = 'partial'
    elif review['accepted_count']:
        review['status'] = 'accepted'
    else:
        review['status'] = 'rejected'
    return _attach_workload_proposal_coverage(review, tasks)


_WORKLOAD_TASK_PROJECTION_KEYS = (
    'semantic_intent',
    'objective',
    'deliverable',
    'rationale',
    'advisory_role',
    'decision_notes',
    'promotion_policy',
    'reconsideration_policy',
    'review_criteria',
    'evidence_requirements',
    'reconsideration_triggers',
    'semantic_review_criteria',
    'promotion_suggestions',
    'waiver_candidates',
    'repair_candidates',
    'supersession_candidates',
    'learning_hint_refs',
    'input_refs',
    'execution_contract',
    'workload_task_ref',
    'output_obligation_ref',
    'output_contract',
    'accepted_proposals',
    'stage_direction',
    'content_payload_source',
    'artifact_prompt_source',
    'requires_artifact',
    'text_artifact_extension',
    'text_artifact_source_name',
    'text_artifact_source',
    'artifact_request',
)


def _project_workload_task_fields(
    records: list[dict[str, Any]],
    tasks: list[Mapping[str, Any]],
) -> None:
    lookup: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        for key in ('phase_id', 'branch_id', 'task_id', 'workload_task_id'):
            token = _clean_text(task.get(key))
            if token:
                lookup[token] = task
    if not lookup:
        return
    for record in records:
        task = (
            lookup.get(_clean_text(record.get('phase_id')))
            or lookup.get(_clean_text(record.get('branch_id')))
            or lookup.get(_clean_text(record.get('task_id') or record.get('workload_task_id')))
        )
        if not isinstance(task, Mapping):
            continue
        for key in _WORKLOAD_TASK_PROJECTION_KEYS:
            value = task.get(key)
            if value not in (None, '', [], {}):
                record[key] = value


def build_workload_graph(
    *,
    intent_prompt: str,
    prompt_intent: Mapping[str, Any],
    phases: list[Mapping[str, Any]],
    current_phase_id: str,
    graph_mode: str,
    workload_task_proposals: Optional[list[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Return the general task graph projected from request phases."""

    normalized_current_phase_id = _clean_text(current_phase_id)
    phase_task_ids: dict[str, str] = {}
    normalized_phases = [
        phase
        for phase in (phases or [])
        if isinstance(phase, Mapping)
    ]
    for index, phase in enumerate(normalized_phases, start=1):
        phase_id = _clean_text(phase.get('phase_id'))
        if phase_id:
            phase_task_ids[phase_id] = _task_id_for_phase(phase, index=index)
    phase_children: dict[str, list[str]] = {phase_id: [] for phase_id in phase_task_ids}
    for phase in normalized_phases:
        child_phase_id = _clean_text(phase.get('phase_id'))
        if not child_phase_id:
            continue
        for parent_phase_id in _phase_dependency_ids(phase):
            if parent_phase_id in phase_children and child_phase_id not in phase_children[parent_phase_id]:
                phase_children[parent_phase_id].append(child_phase_id)
    phase_depths = _phase_decomposition_levels(normalized_phases)

    tasks: list[dict[str, Any]] = []
    child_phase_ids: set[str] = set()
    for index, phase in enumerate(normalized_phases, start=1):
        phase_id = _clean_text(phase.get('phase_id')) or f'phase-{index}'
        branch_id = _clean_text(phase.get('branch_id') or phase_id)
        capability = normalize_capability(phase.get('capability'))
        output_type = _clean_text(phase.get('output_type')).lower() or _workload_output_type_for_capability(capability)
        depends_on = _phase_dependency_ids(phase)
        child_phase_ids.update(depends_on)
        required = _explicit_required_flag(phase)
        required = True if required is None else required
        status = _task_status(phase)
        is_current_phase = bool(phase_id == normalized_current_phase_id)
        has_downstream = bool(phase.get('downstream_phase_ids')) or any(
            phase_id in _phase_dependency_ids(other)
            for other in normalized_phases
            if isinstance(other, Mapping)
        )
        task = {
            'kind': 'ollmo.workload_task',
            'task_id': _task_id_for_phase(phase, index=index),
            'phase_id': phase_id,
            'branch_id': branch_id,
            'intent': (
                _clean_text(phase.get('phase_summary'))
                or _clean_text(phase.get('role'))
                or f'{capability or "unknown"} task'
            ),
            'capability': capability or None,
            'output_contract': {
                'output_type': output_type,
                'required': required,
                'status': status,
                'fulfillment_policy': _fulfillment_policy_for_task(
                    capability,
                    output_type,
                    requires_text_artifact=_phase_requires_text_artifact(phase),
                ),
            },
            'input_refs': _phase_input_refs(phase, current_phase_id=normalized_current_phase_id),
            'depends_on': depends_on,
            'parent_task_ids': [
                phase_task_ids[item]
                for item in depends_on
                if item in phase_task_ids
            ],
            'child_task_ids': [
                phase_task_ids[item]
                for item in phase_children.get(phase_id, [])
                if item in phase_task_ids
            ],
            'decomposition_level': phase_depths.get(phase_id, 0),
            'lifecycle': _task_lifecycle(status),
            'review_criteria': _review_criteria_for_phase(
                phase,
                capability=capability,
                output_type=output_type,
                is_current_phase=is_current_phase,
                has_downstream=has_downstream,
            ),
            'visibility': _visibility_for_phase(phase, capability=capability),
            'status': status,
        }
        for key in (
            'artifact_prompt_source',
            'content_payload_source',
            'stage_direction',
            'queue_index',
            'source',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
            'advisory_role',
            'decision_notes',
            'evidence_requirements',
            'reconsideration_triggers',
            'semantic_review_criteria',
            'promotion_suggestions',
            'waiver_candidates',
            'repair_candidates',
            'supersession_candidates',
            'learning_hint_refs',
        ):
            value = phase.get(key)
            if value not in (None, '', [], {}):
                task[key] = value
        if isinstance(phase.get('artifact_request'), Mapping):
            task['artifact_request'] = dict(phase.get('artifact_request') or {})
        if _phase_has_candidate_identity(phase):
            task['candidate_id'] = _candidate_id_for_phase(phase, index=index)
            task['contract_state'] = _contract_state(phase) or status
        tasks.append(task)

    task_ids = [
        _clean_text(task.get('task_id'))
        for task in tasks
        if _clean_text(task.get('task_id'))
    ]
    leaf_task_ids = [
        _clean_text(task.get('task_id'))
        for task in tasks
        if _clean_text(task.get('phase_id')) not in child_phase_ids and _clean_text(task.get('task_id'))
    ]
    visibility_summary = {
        'user_visible': sum(1 for task in tasks if task.get('visibility') == 'user_visible'),
        'evidence': sum(1 for task in tasks if task.get('visibility') == 'evidence'),
    }
    proposal_review = _apply_workload_task_proposals(tasks, workload_task_proposals)
    return {
        'kind': 'ollmo.workload_graph',
        'workload_graph_version': REQUEST_WORKLOAD_GRAPH_VERSION,
        'root_workload_id': 'workload-root',
        'intent_anchor': _clean_text(intent_prompt) or None,
        'graph_mode': _clean_text(graph_mode) or None,
        'task_ids': task_ids,
        'leaf_task_ids': leaf_task_ids,
        'visibility_summary': visibility_summary,
        'proposal_review': proposal_review,
        'prompt_intent': {
            key: value
            for key, value in dict(prompt_intent or {}).items()
            if value not in (None, '', [], {})
        },
        'tasks': tasks,
    }


def build_output_obligations(
    phases: list[Mapping[str, Any]],
    *,
    current_phase_id: str,
) -> list[dict[str, Any]]:
    """Return the canonical output obligations represented by phase records."""

    obligations: list[dict[str, Any]] = []
    normalized_current_phase_id = _clean_text(current_phase_id)
    for index, raw_phase in enumerate(phases or [], start=1):
        if not isinstance(raw_phase, Mapping):
            continue
        capability = normalize_capability(raw_phase.get('capability'))
        output_type = _clean_text(raw_phase.get('output_type')).lower() or _output_type_for_capability(capability)
        phase_id = _clean_text(raw_phase.get('phase_id'))
        branch_id = _clean_text(raw_phase.get('branch_id') or phase_id)
        if not capability or not output_type or not phase_id:
            continue
        if _phase_has_candidate_identity(raw_phase) and not _phase_is_promoted_candidate(raw_phase):
            continue
        is_current = bool(phase_id and phase_id == normalized_current_phase_id)
        obligation_status = _obligation_status(raw_phase.get('status'))
        if _phase_is_promoted_candidate(raw_phase) and obligation_status == 'planned':
            obligation_status = 'pending'
        raw_role = _clean_text(raw_phase.get('role'))
        requires_text_artifact = output_type == 'text' and _phase_requires_text_artifact(raw_phase)
        obligation_role = 'preparation_output' if is_current and len(phases or []) > 1 else 'final_output'
        if not is_current and raw_role in {'text_artifact_output', 'document_output'}:
            obligation_role = raw_role
        obligation = {
            'kind': 'ollmo.output_obligation',
            'obligation_id': _obligation_id_for_phase(raw_phase, index=index),
            'phase_id': phase_id,
            'capability': capability,
            'output_type': output_type,
            'role': obligation_role,
            'required': True,
            'status': obligation_status,
            'source': _clean_text(raw_phase.get('source')) or 'request_phase_graph',
            'fulfillment_policy': (
                'runtime_text_artifact'
                if requires_text_artifact
                else 'runtime_text'
                if output_type == 'text'
                else 'runtime_artifact_or_branch_state'
            ),
        }
        if _phase_is_promoted_candidate(raw_phase):
            obligation['promoted_from_candidate_id'] = _candidate_id_for_phase(raw_phase, index=index)
            for key in ('promotion_reason', 'promotion_source'):
                value = _clean_text(raw_phase.get(key))
                if value:
                    obligation[key] = value
        if branch_id:
            obligation['branch_id'] = branch_id
        depends_on = [
            _clean_text(item)
            for item in (raw_phase.get('depends_on') or [])
            if _clean_text(item)
        ]
        if depends_on:
            obligation['depends_on'] = depends_on
        queue_index = raw_phase.get('queue_index')
        if queue_index not in (None, ''):
            try:
                obligation['queue_index'] = int(queue_index)
            except (TypeError, ValueError):
                pass
        for key in (
            'artifact_prompt',
            'artifact_prompt_source',
            'content_payload',
            'content_payload_source',
            'phase_summary',
            'stage_direction',
            'batch_prompts',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
            'advisory_role',
            'decision_notes',
            'evidence_requirements',
            'reconsideration_triggers',
            'semantic_review_criteria',
            'repair_candidates',
            'supersession_candidates',
            'learning_hint_refs',
        ):
            value = raw_phase.get(key)
            if value not in (None, '', [], {}):
                obligation[key] = value
        if isinstance(raw_phase.get('artifact_request'), Mapping):
            obligation['artifact_request'] = dict(raw_phase.get('artifact_request') or {})
        for key in (
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'output_contract',
            'input_refs',
        ):
            value = raw_phase.get(key)
            if value not in (None, '', [], {}):
                obligation[key] = value
        review_criteria = _clean_string_list(
            raw_phase.get('review_criteria'),
            limit=48,
            max_chars=160,
        )
        if review_criteria:
            obligation['review_criteria'] = review_criteria
        obligations.append(obligation)
    return obligations


def build_output_candidates(
    phases: list[Mapping[str, Any]],
    *,
    current_phase_id: str,
) -> list[dict[str, Any]]:
    """Return graph-level output possibilities that are not necessarily contract obligations."""

    candidates: list[dict[str, Any]] = []
    normalized_current_phase_id = _clean_text(current_phase_id)
    for index, raw_phase in enumerate(phases or [], start=1):
        if not isinstance(raw_phase, Mapping) or not _phase_has_candidate_identity(raw_phase):
            continue
        capability = normalize_capability(raw_phase.get('capability'))
        output_type = _clean_text(raw_phase.get('output_type')).lower() or _output_type_for_capability(capability)
        phase_id = _clean_text(raw_phase.get('phase_id'))
        branch_id = _clean_text(raw_phase.get('branch_id') or phase_id)
        if not capability or not output_type or not phase_id:
            continue
        is_current = bool(phase_id and phase_id == normalized_current_phase_id)
        candidate = {
            'kind': 'ollmo.output_candidate',
            'candidate_id': _candidate_id_for_phase(raw_phase, index=index),
            'phase_id': phase_id,
            'capability': capability,
            'output_type': output_type,
            'role': 'preparation_candidate' if is_current and len(phases or []) > 1 else 'final_candidate',
            'required': False,
            'status': _candidate_status(raw_phase),
            'source': _clean_text(raw_phase.get('source')) or 'request_phase_graph',
            'promotion_policy': _clean_text(raw_phase.get('promotion_policy')) or 'requires_intent_or_user_confirmation',
        }
        if branch_id:
            candidate['branch_id'] = branch_id
        if _phase_is_promoted_candidate(raw_phase):
            candidate['promoted_obligation_id'] = _obligation_id_for_phase(raw_phase, index=index)
        depends_on = [
            _clean_text(item)
            for item in (raw_phase.get('depends_on') or [])
            if _clean_text(item)
        ]
        if depends_on:
            candidate['depends_on'] = depends_on
        for key in (
            'artifact_prompt',
            'artifact_prompt_source',
            'content_payload',
            'content_payload_source',
            'phase_summary',
            'stage_direction',
            'batch_prompts',
            'promotion_reason',
            'promotion_source',
            'advisory_role',
            'decision_notes',
            'evidence_requirements',
            'reconsideration_triggers',
            'semantic_review_criteria',
            'repair_candidates',
            'supersession_candidates',
            'learning_hint_refs',
            'superseded_by',
            'superseded_by_candidate_id',
            'superseded_by_obligation_id',
            'supersession_reason',
        ):
            value = raw_phase.get(key)
            if value not in (None, '', [], {}):
                candidate[key] = value
        candidates.append(candidate)
    return candidates


def _promotion_records_from_candidates(candidates: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    promotions: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        obligation_id = _clean_text(raw_candidate.get('promoted_obligation_id'))
        candidate_id = _clean_text(raw_candidate.get('candidate_id'))
        if not obligation_id or not candidate_id:
            continue
        promotion = {
            'kind': 'ollmo.output_candidate_promotion',
            'candidate_id': candidate_id,
            'obligation_id': obligation_id,
            'phase_id': _clean_text(raw_candidate.get('phase_id')) or None,
            'branch_id': _clean_text(raw_candidate.get('branch_id')) or None,
            'capability': normalize_capability(raw_candidate.get('capability')) or None,
            'output_type': _clean_text(raw_candidate.get('output_type')).lower() or None,
            'reason': _clean_text(raw_candidate.get('promotion_reason')) or None,
            'source': _clean_text(raw_candidate.get('promotion_source')) or _clean_text(raw_candidate.get('source')) or None,
        }
        promotions.append({key: value for key, value in promotion.items() if value not in (None, '', [], {})})
    return promotions


def build_request_ir(
    *,
    intent_prompt: str,
    prompt_intent: Mapping[str, Any],
    phases: list[Mapping[str, Any]],
    current_phase_id: str,
    graph_mode: str,
    workload_task_proposals: Optional[list[Mapping[str, Any]]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
    semantic_role_profile: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    obligations = build_output_obligations(phases, current_phase_id=current_phase_id)
    candidates = build_output_candidates(phases, current_phase_id=current_phase_id)
    promotions = _promotion_records_from_candidates(candidates)
    workload_graph = build_workload_graph(
        intent_prompt=intent_prompt,
        prompt_intent=prompt_intent,
        phases=phases,
        current_phase_id=current_phase_id,
        graph_mode=graph_mode,
        workload_task_proposals=workload_task_proposals,
    )
    workload_tasks = (
        workload_graph.get('tasks')
        if isinstance(workload_graph.get('tasks'), list)
        else []
    )
    workload_proposal_review = (
        workload_graph.get('proposal_review')
        if isinstance(workload_graph.get('proposal_review'), Mapping)
        else {}
    )
    _project_workload_task_fields(
        obligations,
        [item for item in workload_tasks if isinstance(item, Mapping)],
    )
    _project_workload_task_fields(
        candidates,
        [item for item in workload_tasks if isinstance(item, Mapping)],
    )
    candidate_graph = build_candidate_graph(
        output_candidates=candidates,
        output_obligations=obligations,
        workload_tasks=[
            item for item in workload_tasks if isinstance(item, Mapping)
        ],
        promotions=promotions,
        workload_proposal_review=workload_proposal_review,
        intent_ref='intent_anchor',
        source='request_ir',
    )
    promotion_review = review_candidate_promotions(
        candidate_graph,
        existing_contracts={'source': 'request_ir.output_obligations'},
    )
    decision_contract = build_ghost_decision_contract(
        candidate_graph=candidate_graph,
        promotion_review=promotion_review,
        workload_graph=workload_graph,
        workload_proposal_review=workload_proposal_review,
        output_obligations=obligations,
        accepted_learning_hints=accepted_learning_hints,
        semantic_role_profile=semantic_role_profile,
    )
    final_obligations = [item for item in obligations if item.get('role') == 'final_output']
    payload = {
        'kind': 'ollmo.request_ir',
        'ir_version': REQUEST_IR_VERSION,
        'workload_graph_version': REQUEST_WORKLOAD_GRAPH_VERSION,
        'intent_anchor': _clean_text(intent_prompt) or None,
        'graph_mode': _clean_text(graph_mode) or None,
        'output_obligations': obligations,
        'candidate_graph': candidate_graph,
        'promotion_review': promotion_review,
        'decision_contract': decision_contract,
        'workload_graph': workload_graph,
        'workload_task_ids': workload_graph.get('task_ids') if isinstance(workload_graph.get('task_ids'), list) else [],
        'workload_proposal_review': workload_proposal_review,
        'final_output_obligation_ids': [
            _clean_text(item.get('obligation_id'))
            for item in final_obligations
            if _clean_text(item.get('obligation_id'))
        ],
        'prompt_intent': {
            key: value
            for key, value in dict(prompt_intent or {}).items()
            if value not in (None, '', [], {})
        },
    }
    if candidates:
        payload['output_candidates'] = candidates
        payload['candidate_output_ids'] = [
            _clean_text(item.get('candidate_id'))
            for item in candidates
            if _clean_text(item.get('candidate_id'))
        ]
    if promotions:
        payload['promotions'] = promotions
    return payload


def output_obligations_from_graph(phase_graph: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(phase_graph, Mapping):
        return []
    request_ir = phase_graph.get('request_ir') if isinstance(phase_graph.get('request_ir'), Mapping) else {}
    obligations = request_ir.get('output_obligations') if isinstance(request_ir, Mapping) else None
    if isinstance(obligations, list) and obligations:
        return [dict(item) for item in obligations if isinstance(item, Mapping)]
    top_level = phase_graph.get('output_obligations')
    if isinstance(top_level, list) and top_level:
        return [dict(item) for item in top_level if isinstance(item, Mapping)]
    phases = phase_graph.get('phases') if isinstance(phase_graph.get('phases'), list) else []
    return build_output_obligations(
        [dict(item) for item in phases if isinstance(item, Mapping)],
        current_phase_id=_clean_text(phase_graph.get('current_phase_id')),
    )


def output_candidates_from_graph(phase_graph: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(phase_graph, Mapping):
        return []
    request_ir = phase_graph.get('request_ir') if isinstance(phase_graph.get('request_ir'), Mapping) else {}
    candidates = request_ir.get('output_candidates') if isinstance(request_ir, Mapping) else None
    if isinstance(candidates, list) and candidates:
        return [dict(item) for item in candidates if isinstance(item, Mapping)]
    top_level = phase_graph.get('output_candidates')
    if isinstance(top_level, list) and top_level:
        return [dict(item) for item in top_level if isinstance(item, Mapping)]
    phases = phase_graph.get('phases') if isinstance(phase_graph.get('phases'), list) else []
    return build_output_candidates(
        [dict(item) for item in phases if isinstance(item, Mapping)],
        current_phase_id=_clean_text(phase_graph.get('current_phase_id')),
    )


def promotion_records_from_graph(phase_graph: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(phase_graph, Mapping):
        return []
    request_ir = phase_graph.get('request_ir') if isinstance(phase_graph.get('request_ir'), Mapping) else {}
    promotions = request_ir.get('promotions') if isinstance(request_ir, Mapping) else None
    if isinstance(promotions, list) and promotions:
        return [dict(item) for item in promotions if isinstance(item, Mapping)]
    top_level = phase_graph.get('promotions')
    if isinstance(top_level, list) and top_level:
        return [dict(item) for item in top_level if isinstance(item, Mapping)]
    return _promotion_records_from_candidates(output_candidates_from_graph(phase_graph))
