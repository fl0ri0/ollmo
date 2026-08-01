"""Bounded graph repair proposal, validation, and patch helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from typing import Any, Optional

from ollmo_services.enforced_policy import (
    build_enforced_lineage_summary,
    build_enforced_policy_review,
    enforced_policy_allows_application,
)

GRAPH_PATCH_LIFECYCLE_KIND = 'ollmo.graph_patch_lifecycle'
GRAPH_REPAIR_PROPOSAL_KIND = 'ollmo.graph_repair_proposal'
GRAPH_REPAIR_PROPOSAL_REVIEW_KIND = 'ollmo.graph_repair_proposal_review'
GRAPH_REPAIR_PATCH_APPLICATION_KIND = 'ollmo.graph_repair_patch_application'
GRAPH_REPAIR_AUTONOMY_ENV = 'OLLMO_GRAPH_REPAIR_AUTONOMY'
GRAPH_REPAIR_AUTONOMY_PRODUCT_DEFAULT = 'apply_enforced'

PROPOSAL_ALLOWED_USE = 'proposal_only_until_validated'
PROPOSAL_FORBIDDEN_USE = 'do_not_execute_without_promotion_review_and_runtime_truth'

_SUPPORTED_PATCH_KEYS = {
    'add_phases',
    'add_dependencies',
    'update_phase_fields',
    'mark_reserved',
    'supersede_phases',
}
_RUNTIME_EVIDENCE_SOURCES = {
    'closure_review',
    'graph_closure_review',
    'runtime_closure_review',
    'promotion_review',
    'runtime',
    'monitor_evidence',
    'response_frame',
}
_PROMOTED_STATUSES = {'accepted', 'approved', 'promoted', 'fulfilled'}
_REJECTED_STATUSES = {'rejected', 'blocked', 'failed', 'error', 'invalid'}
_RESERVED_STATUSES = {
    'reserved',
    'deferred',
    'omitted',
    'not_promoted',
    'candidate_only',
    'waived',
    'superseded',
}
_CAPABILITY_OUTPUT_TYPES = {
    'chat': 'text',
    'completion': 'text',
    'embedding': 'embedding',
    'image_generation': 'image',
    'vision_analysis': 'text',
    'speech_to_text': 'text',
    'text_to_speech': 'audio',
}
_TERMINAL_RESPONSE_STATES = {
    'completed',
    'failed',
    'cancelled',
    'repair_needed',
    'late_fill_completed',
    'late_fill_failed',
}
_OPEN_SURFACE_STATES = {'pending', 'blocked', 'open', 'repair_needed', 'review_pending'}
_ADVISORY_SURFACE_CATEGORIES = {
    'active_reconsideration',
    'aspiration_advisory',
    'commitment_advisory',
    'controlled_attention_advisory',
    'completed',
    'repair_advisory',
    'reconsiderable',
    'semantic_review_advisory',
    'superseded',
    'waived',
}
_ADVISORY_CHECK_KINDS = {
    'active_reconsideration',
    'aspiration_review',
    'commitment_review',
    'controlled_attention_review',
    'decision_contract_aspiration_review',
    'decision_contract_commitment_review',
    'decision_contract_controlled_attention_review',
}
_ACTIONABLE_SURFACE_CATEGORIES = {
    'blocked',
    'late_fill_pending',
    'repair_pending',
    'semantic_review_pending',
}
_ACTIONABLE_CHECK_KINDS = {
    'artifact_dependency',
    'artifact_ref_identity',
    'branch_semantic_review',
    'global_semantic_closure',
    'linked_artifact_rebind',
    'materialization_contract',
    'output_obligation',
    'text_artifact_syntax_sanity',
}
_ACTIONABLE_SURFACE_STATUSES = {
    'blocked',
    'failed',
    'pending',
    'planned',
    'repair_needed',
    'repair_pending',
    'repair_required',
    'review_pending',
}
_RESOLVED_SURFACE_STATUSES = {
    'cancelled',
    'completed',
    'done',
    'fulfilled',
    'skipped',
    'superseded',
    'waived',
}
_GRAPH_REPAIR_AUTONOMY_LEVELS = {
    'off',
    'shadow',
    'stage',
    'apply_safe',
    'apply_reviewed',
    'apply_enforced',
}
_SAFE_GRAPH_PATCH_CLASSES = {
    'artifact_binding_repair_branch',
    'missing_dependency_edge',
    'missing_materialization_branch',
    'repairable_block_reopen',
    'semantic_review_branch',
    'text_io_repair_branch',
}
_REVIEW_REQUIRED_GRAPH_PATCH_CLASSES = {
    'branch_identity_merge',
    'branch_identity_split',
    'branch_priority_or_capability_change',
    'candidate_supersession',
    'terminal_contract_reopen',
}
_FORBIDDEN_GRAPH_PATCH_CLASSES = {
    'advisory_surface_only',
    'broad_provider_disablement',
    'degraded_liveness_only',
    'delete_or_hide_branch',
    'full_graph_redraw',
    'provider_' + 'family_ban',
    'replace_user_intent',
    'waive_obligation',
}
_REPAIR_TYPE_TO_PATCH_CLASS = {
    'repair_missing_materialization_contract': 'missing_materialization_branch',
    'resume_or_repair_pending_branch': 'repairable_block_reopen',
    'repair_artifact_ref_identity': 'artifact_binding_repair_branch',
    'rebind_artifact_dependency': 'artifact_binding_repair_branch',
    'reconcile_surface_state_or_reopen_contract': 'repairable_block_reopen',
}
_REVIEW_REQUIRED_REPAIR_TYPES = {
    'branch_merge',
    'branch_split',
    'change_branch_capability',
    'change_branch_output_class',
    'change_branch_priority',
    'merge_branch',
    'reopen_terminal_contract',
    'split_branch',
    'supersede_candidate_branch',
}
_FORBIDDEN_REPAIR_TYPES = {
    'advisory_surface_only',
    'broad_provider_disablement',
    'degraded_liveness_repair',
    'delete_branch',
    'full_graph_redraw',
    'provider_' + 'family_ban',
    'replace_user_intent',
    'waive_obligation',
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _status(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_')


def _normalize_capability(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_')


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        return []
    values: list[str] = []
    for item in raw_items:
        text = _clean_text(item)
        if text and text not in values:
            values.append(text)
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _clean_text(raw_key)
            if not key or raw_value in (None, '', [], {}):
                continue
            payload[key] = _json_safe(raw_value)
        return payload
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value if item not in (None, '', [], {})]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        _clean_text(key): _json_safe(value)
        for key, value in payload.items()
        if _clean_text(key) and value not in (None, '', [], {})
    }


def _stable_digest(value: Any, *, prefix: str = '') -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return f'{prefix}{digest}' if prefix else digest


def stable_graph_repair_graph_digest(graph: Mapping[str, Any]) -> str:
    """Return the canonical digest used by bounded graph-repair application."""

    return _stable_digest(graph if isinstance(graph, Mapping) else {}, prefix='graph-')


def _branch_id(value: Mapping[str, Any]) -> str:
    return _clean_text(value.get('branch_id') or value.get('phase_id'))


def _phase_id(value: Mapping[str, Any]) -> str:
    return _clean_text(value.get('phase_id') or value.get('branch_id'))


def _output_type_for_capability(capability: Any) -> str:
    return _CAPABILITY_OUTPUT_TYPES.get(_normalize_capability(capability), '')


def _phase_from_branch(branch: Mapping[str, Any]) -> dict[str, Any]:
    branch_id = _branch_id(branch)
    phase_id = _phase_id(branch) or branch_id
    capability = _normalize_capability(branch.get('capability'))
    output_type = _clean_text(branch.get('output_type')).lower() or _output_type_for_capability(capability)
    return _compact_payload(
        {
            **dict(branch),
            'phase_id': phase_id,
            'branch_id': branch_id,
            'capability': capability,
            'output_type': output_type or None,
            'status': _clean_text(branch.get('status')) or 'pending',
            'source': _clean_text(branch.get('source')) or 'closure_repair_contract',
        }
    )


def _proposal_id_for(payload: Mapping[str, Any]) -> str:
    return _stable_digest(payload, prefix='graph-repair-')


def _review_id_for(payload: Mapping[str, Any]) -> str:
    return _stable_digest(payload, prefix='graph-repair-review-')


def _runtime_evidence_refs(
    *,
    source: Any,
    evidence_refs: Sequence[Any],
    closure_review: Mapping[str, Any],
    promotion_review: Mapping[str, Any],
) -> list[str]:
    refs = _clean_string_list(evidence_refs)
    source_text = _status(source)
    if source_text in _RUNTIME_EVIDENCE_SOURCES:
        refs.append(f'source:{source_text}')
    feedback = (
        closure_review.get('ghost_repair_feedback')
        if isinstance(closure_review.get('ghost_repair_feedback'), Mapping)
        else {}
    )
    if _status(feedback.get('status')) == 'repair_required':
        refs.append('closure_review:ghost_repair_feedback')
    if _status(closure_review.get('status')) in {'repair_required', 'pending', 'blocked'}:
        refs.append(f'closure_review:status:{_status(closure_review.get("status"))}')
    if _status(promotion_review.get('status')) in _PROMOTED_STATUSES:
        refs.append(f'promotion_review:status:{_status(promotion_review.get("status"))}')
    values: list[str] = []
    for ref in refs:
        text = _clean_text(ref)
        if text and text not in values:
            values.append(text)
    return values


def _only_learning_evidence(source: Any, evidence_refs: Sequence[Any]) -> bool:
    source_text = _status(source)
    refs = _clean_string_list(evidence_refs)
    if source_text and source_text != 'accepted_learning':
        return False
    if not refs:
        return source_text == 'accepted_learning'
    return all(
        _status(ref).startswith('accepted_learning')
        or _status(ref).startswith('learning')
        or _status(ref).startswith('hint')
        for ref in refs
    )


def _contains_reserved_conflict(
    *,
    phase: Mapping[str, Any],
    request_phase_graph: Mapping[str, Any],
    candidate_graph: Mapping[str, Any],
    promotion_review: Mapping[str, Any],
) -> bool:
    direct_statuses = {
        _status(phase.get('status')),
        _status(phase.get('contract_state')),
        _status(phase.get('promotion_status')),
        _status(phase.get('decision')),
    }
    if direct_statuses & _RESERVED_STATUSES:
        return True

    target_tokens = {
        _branch_id(phase),
        _phase_id(phase),
        _clean_text(phase.get('obligation_id')),
    }
    target_tokens = {item for item in target_tokens if item}
    if not target_tokens:
        return False
    for surface in (request_phase_graph, candidate_graph, promotion_review):
        text = json.dumps(_json_safe(surface), sort_keys=True).lower()
        if not any(token.lower() in text for token in target_tokens):
            continue
        if any(status in text for status in _RESERVED_STATUSES):
            return True
    return False


def _phase_validation_errors(phase: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    capability = _normalize_capability(phase.get('capability'))
    output_type = _clean_text(phase.get('output_type')).lower()
    expected_output_type = _output_type_for_capability(capability)
    if not _phase_id(phase):
        errors.append('missing_phase_id')
    if not _branch_id(phase):
        errors.append('missing_branch_id')
    if not capability:
        errors.append('missing_capability')
    if expected_output_type and output_type and output_type != expected_output_type:
        errors.append('capability_output_contract_mismatch')
    return errors


def _dependency_targets(dependency: Mapping[str, Any]) -> tuple[str, list[str]]:
    target_id = _clean_text(
        dependency.get('phase_id')
        or dependency.get('branch_id')
        or dependency.get('target_phase_id')
        or dependency.get('target_branch_id')
    )
    depends_on = _clean_string_list(dependency.get('depends_on'))
    for key in ('dependency_phase_id', 'dependency_branch_id', 'source_phase_id', 'source_branch_id'):
        token = _clean_text(dependency.get(key))
        if token and token not in depends_on:
            depends_on.append(token)
    return target_id, depends_on


def _slug(value: Any, *, fallback: str = 'repair') -> str:
    token = _status(value).replace('_', '-')
    chars = [char if char.isalnum() or char == '-' else '-' for char in token]
    slug = '-'.join(part for part in ''.join(chars).split('-') if part)
    return slug or fallback


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, Mapping) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _status(value) in {'true', 'yes', '1', 'unmet', 'failed', 'blocked'}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _append_unique(values: list[str], value: Any) -> None:
    text = _clean_text(value)
    if text and text not in values:
        values.append(text)


def _surface_item_is_promoted_owed_work(item: Mapping[str, Any]) -> bool:
    category = _status(item.get('category'))
    item_status = _status(item.get('status') or item.get('state'))
    check_kind = _status(item.get('check_kind') or item.get('kind'))
    if category in _ADVISORY_SURFACE_CATEGORIES or check_kind in _ADVISORY_CHECK_KINDS:
        return False
    if item_status in _RESOLVED_SURFACE_STATUSES:
        return False
    if item.get('repair_required') is True or item.get('repair_action') not in (None, '', [], {}):
        return True
    if check_kind in _ACTIONABLE_CHECK_KINDS and item_status in _ACTIONABLE_SURFACE_STATUSES:
        return True
    if category in _ACTIONABLE_SURFACE_CATEGORIES and item_status in _ACTIONABLE_SURFACE_STATUSES | {''}:
        return True
    has_promoted_identity = any(
        item.get(key) not in (None, '', [], {})
        for key in (
            'obligation_id',
            'task_id',
            'phase_id',
            'branch_id',
            'output_slot_id',
            'slot_id',
            'dependency_id',
            'artifact_ref',
        )
    )
    return category == 'open' and has_promoted_identity and item_status in {'pending', 'planned', 'active', 'open', 'review_pending'}


def _surface_item_is_generic_open_identity(item: Mapping[str, Any]) -> bool:
    category = _status(item.get('category'))
    item_status = _status(item.get('status') or item.get('state'))
    check_kind = _status(item.get('check_kind') or item.get('kind'))
    if category != 'open':
        return False
    if check_kind in _ACTIONABLE_CHECK_KINDS or check_kind in _ADVISORY_CHECK_KINDS:
        return False
    if item.get('repair_required') is True or item.get('repair_action') not in (None, '', [], {}):
        return False
    has_promoted_identity = any(
        item.get(key) not in (None, '', [], {})
        for key in (
            'obligation_id',
            'task_id',
            'phase_id',
            'branch_id',
            'output_slot_id',
            'slot_id',
            'dependency_id',
            'artifact_ref',
        )
    )
    return has_promoted_identity and item_status in {'pending', 'planned', 'active', 'open', 'review_pending'}


def classify_surface_repair_actionability(
    surface_state: Mapping[str, Any] | None,
    *,
    closure_review: Optional[Mapping[str, Any]] = None,
    late_fill: Optional[Mapping[str, Any]] = None,
    monitor_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify whether a visible surface state is repair evidence or advisory movement."""

    surface = _mapping_or_empty(surface_state)
    review = _mapping_or_empty(closure_review)
    late = _mapping_or_empty(late_fill)
    monitor = _mapping_or_empty(monitor_report)
    surface_status = _status(surface.get('state') or surface.get('status'))
    category_counts = (
        surface.get('category_counts')
        if isinstance(surface.get('category_counts'), Mapping)
        else {}
    )
    items = surface.get('items') if isinstance(surface.get('items'), list) else []
    actionable_categories: list[str] = []
    advisory_categories: list[str] = []
    evidence_refs: list[str] = []
    fulfilled_contract = (
        _status(late.get('final_materialization_contract_status')) == 'fulfilled'
        or _status(review.get('status')) == 'fulfilled'
    )

    has_surface_items = bool(items)
    for raw_category, raw_count in category_counts.items():
        category = _status(raw_category)
        if _safe_int(raw_count) <= 0 or not category:
            continue
        if category in _ACTIONABLE_SURFACE_CATEGORIES:
            if has_surface_items:
                continue
            else:
                _append_unique(actionable_categories, category)
                _append_unique(evidence_refs, f'surface_state:category:{category}')
        elif category in _ADVISORY_SURFACE_CATEGORIES:
            _append_unique(advisory_categories, category)

    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        category = _status(item.get('category'))
        check_kind = _status(item.get('check_kind') or item.get('kind'))
        item_status = _status(item.get('status') or item.get('state'))
        if category in _ADVISORY_SURFACE_CATEGORIES or check_kind in _ADVISORY_CHECK_KINDS:
            _append_unique(advisory_categories, category or check_kind)
        if fulfilled_contract and _surface_item_is_generic_open_identity(item):
            _append_unique(advisory_categories, category or 'open')
            continue
        if _surface_item_is_promoted_owed_work(item):
            _append_unique(actionable_categories, category or check_kind or 'open')
            identity = _clean_text(
                item.get('obligation_id')
                or item.get('branch_id')
                or item.get('phase_id')
                or item.get('task_id')
                or item.get('slot_id')
                or item.get('artifact_ref')
                or index
            )
            _append_unique(evidence_refs, f'surface_state:item:{category or check_kind or item_status}:{identity}')

    if surface_status in {'blocked', 'repair_needed'} and not actionable_categories:
        _append_unique(actionable_categories, surface_status)
        _append_unique(evidence_refs, f'surface_state:state:{surface_status}')

    if _truthy(late.get('materialization_contract_unmet')):
        _append_unique(actionable_categories, 'materialization_contract_unmet')
        _append_unique(evidence_refs, 'late_fill:materialization_contract_unmet')
    if _status(late.get('final_materialization_contract_status')) == 'unmet':
        _append_unique(actionable_categories, 'materialization_contract_unmet')
        _append_unique(evidence_refs, 'late_fill:final_materialization_contract_status:unmet')
    if _truthy(monitor.get('materialization_contract_unmet')):
        _append_unique(actionable_categories, 'materialization_contract_unmet')
        _append_unique(evidence_refs, 'monitor:materialization_contract_unmet')

    for check in review.get('checks') or []:
        if not isinstance(check, Mapping):
            continue
        check_kind = _status(check.get('check_kind') or check.get('kind'))
        check_status = _status(check.get('status') or check.get('state'))
        if check_kind in _ADVISORY_CHECK_KINDS:
            _append_unique(advisory_categories, check_kind)
            continue
        if check_status in _RESOLVED_SURFACE_STATUSES:
            _append_unique(advisory_categories, check_kind or check_status)
            continue
        if check.get('repair_required') is True or check.get('repair_action') not in (None, '', [], {}):
            _append_unique(actionable_categories, 'repair_pending')
            _append_unique(evidence_refs, f'closure_review:repair_action:{check_kind or check_status}')
        elif check_kind in _ACTIONABLE_CHECK_KINDS and check_status in _ACTIONABLE_SURFACE_STATUSES:
            _append_unique(actionable_categories, check_kind)
            _append_unique(evidence_refs, f'closure_review:check:{check_kind}:{check_status}')

    if actionable_categories:
        status = 'actionable'
        reason = 'surface carries runtime-backed blocked, repair, semantic-review, or owed-work evidence'
    elif advisory_categories or surface_status in _OPEN_SURFACE_STATES:
        status = 'advisory'
        reason = 'surface carries advisory movement or attention state without runtime-backed repair evidence'
    else:
        status = 'neutral'
        reason = 'surface does not carry open repair evidence'
    return _json_safe(
        {
            'status': status,
            'surface_status': surface_status,
            'actionable_categories': actionable_categories,
            'advisory_categories': advisory_categories,
            'reason': reason,
            'evidence_refs': evidence_refs,
        }
    )


def _walk_strings(value: Any, *, max_depth: int = 6) -> list[str]:
    if max_depth <= 0:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_walk_strings(item, max_depth=max_depth - 1))
        return strings
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        strings: list[str] = []
        for item in value:
            strings.extend(_walk_strings(item, max_depth=max_depth - 1))
        return strings
    return []


def _fake_artifact_refs(*values: Any) -> list[str]:
    refs: list[str] = []
    for value in values:
        for text in _walk_strings(value):
            if 'artifact://' not in text:
                continue
            for raw_part in text.replace('"', ' ').replace("'", ' ').split():
                part = raw_part.strip(' ,;()[]{}<>')
                if part.startswith('artifact://') and part not in refs:
                    refs.append(part)
    return refs


def _pending_branches_from(*values: Any) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    for value in values:
        source = value if isinstance(value, Mapping) else {}
        for key in ('pending_branches', 'open_branches', 'blocked_branches'):
            raw_items = source.get(key) if isinstance(source.get(key), list) else []
            for item in raw_items:
                if isinstance(item, Mapping):
                    branch = dict(item)
                    identity = _branch_id(branch) or _phase_id(branch) or _stable_digest(branch)
                    if not any((_branch_id(existing) or _phase_id(existing) or _stable_digest(existing)) == identity for existing in branches):
                        branches.append(branch)
    return branches


def _runtime_repair_phase(
    *,
    repair_type: str,
    evidence_key: str,
    capability: Any = 'chat',
    output_type: Any = '',
    depends_on: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    capability_text = _normalize_capability(capability) or 'chat'
    output_type_text = _clean_text(output_type).lower() or _output_type_for_capability(capability_text) or 'text'
    phase_id = f'repair-{_slug(repair_type)}-{_stable_digest(evidence_key)}'
    return _compact_payload(
        {
            'phase_id': phase_id,
            'branch_id': phase_id,
            'capability': capability_text,
            'output_type': output_type_text,
            'status': 'pending',
            'required': True,
            'source': 'runtime_monitor_evidence',
            'repair_action': repair_type,
            'semantic_intent': f'Run bounded repair for {repair_type}.',
            'depends_on': _clean_string_list(depends_on),
            'repair_evidence': evidence_key,
            'runtime_repair_metadata': dict(metadata or {}),
        }
    )


def _phases_for_pending_branches(
    branches: Sequence[Mapping[str, Any]],
    *,
    repair_type: str,
    evidence_key: str,
) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for index, branch in enumerate(branches, 1):
        phase = _phase_from_branch(
            {
                **dict(branch),
                'phase_id': _phase_id(branch) or _branch_id(branch) or f'repair-{_slug(repair_type)}-{index}',
                'branch_id': _branch_id(branch) or _phase_id(branch) or f'repair-{_slug(repair_type)}-{index}',
                'capability': _normalize_capability(branch.get('capability')) or 'chat',
                'output_type': _clean_text(branch.get('output_type')).lower()
                or _output_type_for_capability(branch.get('capability'))
                or 'text',
                'status': 'pending',
                'source': 'runtime_monitor_evidence',
                'repair_action': repair_type,
                'repair_evidence': evidence_key,
            }
        )
        if phase not in phases:
            phases.append(phase)
    return phases


def _runtime_evidence_proposal(
    *,
    request_phase_graph: Mapping[str, Any],
    repair_type: str,
    reason: str,
    evidence_refs: Sequence[Any],
    add_phases: Optional[Sequence[Mapping[str, Any]]] = None,
    add_dependencies: Optional[Sequence[Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
    redraw_scope_review: Optional[Mapping[str, Any]] = None,
    source: str = 'monitor_evidence',
) -> Optional[dict[str, Any]]:
    refs = _clean_string_list(evidence_refs)
    if not refs:
        return None
    patch: dict[str, Any] = {}
    phases = [dict(item) for item in (add_phases or []) if isinstance(item, Mapping)]
    dependencies = [dict(item) for item in (add_dependencies or []) if isinstance(item, Mapping)]
    if phases:
        patch['add_phases'] = phases
    if dependencies:
        patch['add_dependencies'] = dependencies
    payload = {
        'kind': GRAPH_REPAIR_PROPOSAL_KIND,
        'status': 'proposed',
        'source': _clean_text(source) or 'monitor_evidence',
        'repair_type': repair_type,
        'repair_gap_code': repair_type,
        'repair_actions': [repair_type],
        'target_graph_id': _stable_digest(request_phase_graph, prefix='graph-'),
        'evidence_refs': refs,
        'reason': reason,
        'patch': patch,
        'allowed_use': PROPOSAL_ALLOWED_USE,
        'forbidden_use': PROPOSAL_FORBIDDEN_USE,
        'runtime_monitor_evidence': dict(metadata or {}),
    }
    if isinstance(accepted_learning_hints, Mapping) and accepted_learning_hints:
        payload['learning_orientation'] = {
            'authority': 'soft_hint_only',
            'allowed_use': 'orientation_only_not_patch_authority',
            'hint_count': accepted_learning_hints.get('hint_count'),
        }
    if isinstance(redraw_scope_review, Mapping) and redraw_scope_review:
        payload['redraw_scope_orientation'] = _compact_payload(
            {
                'review_id': redraw_scope_review.get('review_id'),
                'status': redraw_scope_review.get('status'),
                'selected_scope': redraw_scope_review.get('selected_scope'),
                'selected_scope_reason': redraw_scope_review.get('selected_scope_reason'),
                'intent_contract_digest': redraw_scope_review.get('intent_contract_digest'),
                'allowed_use': 'orientation_only_not_patch_authority',
                'runtime_effect': 'none',
            }
        )
    payload['proposal_id'] = _proposal_id_for(payload)
    return _json_safe(payload)


def _provider_ban_requested(value: Any) -> bool:
    text = json.dumps(_json_safe(value), sort_keys=True).lower()
    return any(
        token in text
        for token in (
            'provider_ban',
            'provider_bans',
            'ban_provider',
            'banned_provider',
            'provider_' + 'family_ban',
            'disable_provider_' + 'family',
            'quarantine_provider_' + 'family',
        )
    )


def build_graph_repair_proposal_from_repair_gap(
    *,
    request_phase_graph: Mapping[str, Any],
    repair_gap: Mapping[str, Any],
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build one additive graph repair proposal from a Closure repair gap."""

    if not isinstance(request_phase_graph, Mapping) or not isinstance(repair_gap, Mapping):
        return None
    pending_branches = [
        item for item in (repair_gap.get('pending_branches') or []) if isinstance(item, Mapping)
    ]
    add_phases = [_phase_from_branch(item) for item in pending_branches]
    add_phases = [item for item in add_phases if _phase_id(item) and _branch_id(item)]
    if not add_phases:
        return None

    evidence_refs: list[str] = ['closure_review:ghost_repair_feedback']
    for branch in add_phases:
        for key, prefix in (
            ('repair_contract_id', 'repair_contract'),
            ('repair_evidence', 'closure_check'),
            ('obligation_id', 'obligation'),
            ('branch_id', 'branch'),
            ('repair_action', 'repair_action'),
        ):
            text = _clean_text(branch.get(key))
            if text:
                evidence_refs.append(f'{prefix}:{text}')
    repair_loop = repair_gap.get('repair_loop') if isinstance(repair_gap.get('repair_loop'), Mapping) else {}
    if _status(repair_loop.get('status')) == 'promoted':
        evidence_refs.append('repair_loop:promoted')

    patch = {'add_phases': add_phases}
    payload = {
        'kind': GRAPH_REPAIR_PROPOSAL_KIND,
        'source': 'closure_review',
        'repair_type': 'additive_graph_patch',
        'target_graph_id': _stable_digest(request_phase_graph, prefix='graph-'),
        'evidence_refs': list(dict.fromkeys(evidence_refs)),
        'reason': _clean_text(repair_gap.get('trigger')) or 'Closure requested a bounded graph repair.',
        'patch': patch,
        'allowed_use': PROPOSAL_ALLOWED_USE,
        'forbidden_use': PROPOSAL_FORBIDDEN_USE,
    }
    if isinstance(accepted_learning_hints, Mapping) and accepted_learning_hints:
        payload['learning_orientation'] = {
            'authority': 'soft_hint_only',
            'allowed_use': 'orientation_only_not_patch_authority',
            'hint_count': accepted_learning_hints.get('hint_count'),
        }
    payload['proposal_id'] = _proposal_id_for(payload)
    return _json_safe(payload)


def _proposal_from_repair_item(
    *,
    request_phase_graph: Mapping[str, Any],
    item: Mapping[str, Any],
    source: str,
    target_graph_id: str,
) -> Optional[dict[str, Any]]:
    capability = _normalize_capability(item.get('capability'))
    phase_id = _clean_text(item.get('phase_id') or item.get('branch_id') or item.get('task_id'))
    branch_id = _clean_text(item.get('branch_id') or phase_id)
    if not capability or not phase_id:
        return None
    patch = {
        'add_phases': [
            _compact_payload(
                {
                    'phase_id': phase_id,
                    'branch_id': branch_id,
                    'capability': capability,
                    'output_type': _clean_text(item.get('output_type')).lower()
                    or _output_type_for_capability(capability),
                    'depends_on': item.get('depends_on'),
                    'status': 'pending',
                    'source': 'decision_contract_repair_candidate',
                    'repair_action': item.get('repair_action'),
                    'repair_candidate': item,
                }
            )
        ]
    }
    payload = {
        'kind': GRAPH_REPAIR_PROPOSAL_KIND,
        'source': source,
        'repair_type': 'additive_graph_patch',
        'target_graph_id': target_graph_id,
        'target_phase_id': phase_id,
        'evidence_refs': [
            ref for ref in (
                'decision_contract:repair_candidates',
                f'task:{_clean_text(item.get("task_id"))}',
                f'branch:{branch_id}',
            ) if ref.split(':', 1)[-1]
        ],
        'reason': _clean_text(item.get('reason')) or 'Advisory repair candidate from decision contract.',
        'patch': patch,
        'allowed_use': PROPOSAL_ALLOWED_USE,
        'forbidden_use': PROPOSAL_FORBIDDEN_USE,
    }
    payload['proposal_id'] = _proposal_id_for(payload)
    return _json_safe(payload)


def build_graph_repair_proposals(
    *,
    request_phase_graph: Mapping[str, Any],
    closure_review: Optional[Mapping[str, Any]] = None,
    decision_contract: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Return advisory graph repair proposals. They are not executable truth."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    target_graph_id = _stable_digest(graph, prefix='graph-')
    proposals: list[dict[str, Any]] = []

    review = closure_review if isinstance(closure_review, Mapping) else {}
    feedback = review.get('ghost_repair_feedback') if isinstance(review.get('ghost_repair_feedback'), Mapping) else {}
    if _status(feedback.get('status')) == 'repair_required':
        proposal = build_graph_repair_proposal_from_repair_gap(
            request_phase_graph=graph,
            repair_gap={
                'trigger': 'ghost_repair_feedback',
                'pending_branches': feedback.get('items') if isinstance(feedback.get('items'), list) else [],
                'repair_loop': feedback.get('repair_loop'),
            },
            accepted_learning_hints=accepted_learning_hints,
        )
        if proposal:
            proposals.append(proposal)

    adequacy = review.get('intent_graph_adequacy') if isinstance(review.get('intent_graph_adequacy'), Mapping) else {}
    adequacy_checks = adequacy.get('checks') if isinstance(adequacy.get('checks'), list) else []
    for check in adequacy_checks:
        if not isinstance(check, Mapping):
            continue
        evidence = _clean_text(check.get('evidence'))
        repair_action = _clean_text(check.get('repair_action'))
        if evidence != 'intent_graph_adequacy_missing_dependency_edge':
            continue
        if repair_action and repair_action != 'rebind_artifact_dependency':
            continue
        add_dependencies = [
            dict(item) for item in (check.get('add_dependencies') or [])
            if isinstance(item, Mapping)
        ]
        if not add_dependencies:
            continue
        obligation_id = _clean_text(
            check.get('intent_obligation_id')
            or check.get('obligation_id')
            or check.get('dependency_contract')
        )
        evidence_refs = ['intent_graph_adequacy:missing_dependency_edge']
        if obligation_id:
            evidence_refs.append(f'intent_obligation:{obligation_id}')
        dependency_contract = _clean_text(check.get('dependency_contract'))
        if dependency_contract:
            evidence_refs.append(f'dependency_contract:{dependency_contract}')
        proposal = _runtime_evidence_proposal(
            request_phase_graph=graph,
            source='graph_closure_review',
            repair_type='rebind_artifact_dependency',
            reason='Intent graph adequacy found a missing producer/consumer dependency edge.',
            evidence_refs=evidence_refs,
            add_dependencies=add_dependencies,
            metadata={
                'intent_graph_adequacy_check': dict(check),
                'dependency_contract': dependency_contract,
            },
            accepted_learning_hints=accepted_learning_hints,
        )
        if proposal:
            proposals.append(proposal)

    contract = decision_contract if isinstance(decision_contract, Mapping) else {}
    repair_candidates = (
        contract.get('repair_candidates')
        if isinstance(contract.get('repair_candidates'), list)
        else []
    )
    for item in repair_candidates:
        if not isinstance(item, Mapping):
            continue
        proposal = _proposal_from_repair_item(
            request_phase_graph=graph,
            item=item,
            source='decision_contract',
            target_graph_id=target_graph_id,
        )
        if proposal:
            proposals.append(proposal)

    if isinstance(accepted_learning_hints, Mapping):
        hints = accepted_learning_hints.get('hints') if isinstance(accepted_learning_hints.get('hints'), list) else []
        for hint in hints:
            if not isinstance(hint, Mapping):
                continue
            target_area = _status(hint.get('target_area'))
            if 'graph' not in target_area and 'repair' not in target_area:
                continue
            payload = {
                'kind': GRAPH_REPAIR_PROPOSAL_KIND,
                'source': 'accepted_learning',
                'repair_type': 'advisory_orientation_only',
                'target_graph_id': target_graph_id,
                'evidence_refs': [f'accepted_learning:{_clean_text(hint.get("learning_id"))}'],
                'reason': _clean_text(hint.get('hint')),
                'patch': {},
                'allowed_use': PROPOSAL_ALLOWED_USE,
                'forbidden_use': PROPOSAL_FORBIDDEN_USE,
                'learning_orientation': {
                    'authority': 'soft_hint_only',
                    'allowed_use': 'orientation_only_not_patch_authority',
                },
            }
            payload['proposal_id'] = _proposal_id_for(payload)
            proposals.append(_json_safe(payload))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = _clean_text(proposal.get('proposal_id'))
        if not proposal_id or proposal_id in seen:
            continue
        seen.add(proposal_id)
        unique.append(proposal)
    return unique


def build_graph_repair_proposals_from_runtime_evidence(
    *,
    response_frame: Optional[Mapping[str, Any]] = None,
    monitor_report: Optional[Mapping[str, Any]] = None,
    request_phase_graph: Optional[Mapping[str, Any]] = None,
    closure_review: Optional[Mapping[str, Any]] = None,
    late_fill: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Map runtime/Closure/monitor failures into proposal-only graph repairs."""

    frame = _mapping_or_empty(response_frame)
    monitor = _mapping_or_empty(monitor_report)
    runtime = _mapping_or_empty(frame.get('runtime'))
    current_state = _mapping_or_empty(frame.get('current_state'))
    graph = (
        _mapping_or_empty(request_phase_graph)
        or _mapping_or_empty(runtime.get('request_phase_graph'))
        or _nested_mapping(frame, 'planning', 'request_phase_graph')
        or _nested_mapping(frame, 'planning', 'artifact_flow', 'request_phase_graph')
    )
    developer_diagnostics = _mapping_or_empty(runtime.get('developer_diagnostics'))
    redraw_scope_review = (
        _mapping_or_empty(graph.get('redraw_scope_ladder_review'))
        or _mapping_or_empty(developer_diagnostics.get('redraw_scope_ladder_review'))
    )
    review = (
        _mapping_or_empty(closure_review)
        or _mapping_or_empty(runtime.get('graph_closure_review'))
        or _mapping_or_empty(current_state.get('graph_closure_review'))
    )
    late = (
        _mapping_or_empty(late_fill)
        or _mapping_or_empty(frame.get('late_fill'))
        or _mapping_or_empty(current_state.get('late_fill'))
    )
    artifacts = _mapping_or_empty(monitor.get('artifacts'))
    response_id = _clean_text(
        monitor.get('response_id')
        or frame.get('response_id')
        or frame.get('id')
        or monitor.get('frame_id')
    )
    proposals: list[dict[str, Any]] = []
    pending_branches = _pending_branches_from(late, monitor)

    def add(proposal: Optional[dict[str, Any]]) -> None:
        if proposal:
            proposals.append(proposal)

    final_contract_status = _status(
        monitor.get('final_materialization_contract_status')
        or late.get('final_materialization_contract_status')
    )
    materialization_unmet = (
        _truthy(monitor.get('materialization_contract_unmet'))
        or _truthy(late.get('materialization_contract_unmet'))
        or final_contract_status == 'unmet'
    )
    if materialization_unmet:
        refs = []
        if _truthy(monitor.get('materialization_contract_unmet')):
            refs.append('monitor:materialization_contract_unmet')
        if _truthy(late.get('materialization_contract_unmet')):
            refs.append('late_fill:materialization_contract_unmet')
        if final_contract_status == 'unmet':
            refs.append('late_fill:final_materialization_contract_status:unmet')
        evidence_key = f'{response_id}:materialization_contract_unmet:{len(pending_branches)}'
        phases = _phases_for_pending_branches(
            pending_branches,
            repair_type='repair_missing_materialization_contract',
            evidence_key=evidence_key,
        ) or [
            _runtime_repair_phase(
                repair_type='repair_missing_materialization_contract',
                evidence_key=evidence_key,
                metadata={'response_id': response_id, 'final_contract_status': final_contract_status},
            )
        ]
        add(
            _runtime_evidence_proposal(
                request_phase_graph=graph,
                repair_type='repair_missing_materialization_contract',
                reason='Final materialization contract is unmet; propose bounded materialization repair.',
                evidence_refs=refs,
                add_phases=phases,
                metadata={
                    'response_id': response_id,
                    'final_materialization_contract_status': final_contract_status,
                    'pending_branch_count': len(pending_branches),
                },
                accepted_learning_hints=accepted_learning_hints,
                redraw_scope_review=redraw_scope_review,
            )
        )

    branch_counts = _mapping_or_empty(monitor.get('branch_counts'))
    try:
        monitor_pending_count = int(branch_counts.get('pending') or 0)
    except (TypeError, ValueError):
        monitor_pending_count = 0
    pending_count = max(monitor_pending_count, len(pending_branches))
    lifecycle = _status(
        monitor.get('lifecycle_state')
        or frame.get('lifecycle_state')
        or current_state.get('lifecycle_state')
        or monitor.get('status')
        or frame.get('status')
    )
    terminal_with_pending = lifecycle in _TERMINAL_RESPONSE_STATES and pending_count > 0
    if terminal_with_pending:
        evidence_key = f'{response_id}:terminal_pending_branch:{pending_count}'
        refs = [
            f'monitor:terminal_lifecycle:{lifecycle}',
            f'monitor:branch_counts.pending:{pending_count}',
        ]
        phases = _phases_for_pending_branches(
            pending_branches,
            repair_type='resume_or_repair_pending_branch',
            evidence_key=evidence_key,
        ) or [
            _runtime_repair_phase(
                repair_type='resume_or_repair_pending_branch',
                evidence_key=evidence_key,
                metadata={'response_id': response_id, 'pending_count': pending_count},
            )
        ]
        add(
            _runtime_evidence_proposal(
                request_phase_graph=graph,
                repair_type='resume_or_repair_pending_branch',
                reason='A terminal response still carries pending branch work.',
                evidence_refs=refs,
                add_phases=phases,
                metadata={'response_id': response_id, 'lifecycle_state': lifecycle, 'pending_count': pending_count},
                accepted_learning_hints=accepted_learning_hints,
                redraw_scope_review=redraw_scope_review,
            )
        )

    duplicate_refs = _clean_string_list(artifacts.get('duplicate_artifact_refs'))
    if duplicate_refs:
        evidence_key = f'{response_id}:duplicate_artifact_refs:{"|".join(duplicate_refs)}'
        add(
            _runtime_evidence_proposal(
                request_phase_graph=graph,
                repair_type='repair_artifact_ref_identity',
                reason='Duplicate artifact refs were observed and need identity repair.',
                evidence_refs=[f'monitor:duplicate_artifact_ref:{item}' for item in duplicate_refs],
                add_phases=[
                    _runtime_repair_phase(
                        repair_type='repair_artifact_ref_identity',
                        evidence_key=evidence_key,
                        metadata={'response_id': response_id, 'duplicate_artifact_refs': duplicate_refs},
                    )
                ],
                metadata={'response_id': response_id, 'duplicate_artifact_refs': duplicate_refs},
                accepted_learning_hints=accepted_learning_hints,
                redraw_scope_review=redraw_scope_review,
            )
        )

    missing_dependency_refs: list[str] = []
    for item in _clean_string_list(artifacts.get('missing_files')):
        missing_dependency_refs.append(item)
    files = artifacts.get('files') if isinstance(artifacts.get('files'), list) else []
    for item in files:
        if not isinstance(item, Mapping):
            continue
        if item.get('exists') is False:
            ref = _clean_text(item.get('path') or item.get('artifact_ref') or item.get('src'))
            if ref and ref not in missing_dependency_refs:
                missing_dependency_refs.append(ref)
    links = artifacts.get('html_image_links') if isinstance(artifacts.get('html_image_links'), list) else []
    for item in links:
        if not isinstance(item, Mapping):
            continue
        if item.get('exists') is False:
            ref = _clean_text(item.get('src') or item.get('href') or item.get('path'))
            if ref and ref not in missing_dependency_refs:
                missing_dependency_refs.append(ref)
    for item in _fake_artifact_refs(monitor, late, review, frame):
        if item not in missing_dependency_refs:
            missing_dependency_refs.append(item)
    if missing_dependency_refs:
        evidence_key = f'{response_id}:artifact_dependency:{"|".join(missing_dependency_refs[:8])}'
        add(
            _runtime_evidence_proposal(
                request_phase_graph=graph,
                repair_type='rebind_artifact_dependency',
                reason='Artifact dependency refs are missing, broken, or fake and need rebind.',
                evidence_refs=[f'monitor:artifact_dependency:{item}' for item in missing_dependency_refs[:12]],
                add_phases=[
                    _runtime_repair_phase(
                        repair_type='rebind_artifact_dependency',
                        evidence_key=evidence_key,
                        metadata={'response_id': response_id, 'artifact_dependency_refs': missing_dependency_refs[:12]},
                    )
                ],
                metadata={'response_id': response_id, 'artifact_dependency_refs': missing_dependency_refs[:12]},
                accepted_learning_hints=accepted_learning_hints,
                redraw_scope_review=redraw_scope_review,
            )
        )

    surface_state = (
        _mapping_or_empty(review.get('surface_state'))
        or _mapping_or_empty(late.get('surface_state'))
        or _mapping_or_empty(monitor.get('surface_state'))
    )
    surface_status = _status(surface_state.get('state') or surface_state.get('status'))
    closure_status = _status(review.get('status'))
    fulfilled_contract = final_contract_status == 'fulfilled' or closure_status == 'fulfilled'
    surface_actionability = classify_surface_repair_actionability(
        surface_state,
        closure_review=review,
        late_fill=late,
        monitor_report=monitor,
    )
    if (
        fulfilled_contract
        and surface_status in _OPEN_SURFACE_STATES
        and surface_actionability.get('status') == 'actionable'
    ):
        evidence_key = f'{response_id}:surface_mismatch:{surface_status}'
        evidence_refs = [
            f'closure_review:status:{closure_status or "unknown"}',
            f'late_fill:final_materialization_contract_status:{final_contract_status or "unknown"}',
            f'surface_state:{surface_status}',
        ]
        for ref in surface_actionability.get('evidence_refs') or []:
            if isinstance(ref, str) and ref not in evidence_refs:
                evidence_refs.append(ref)
        metadata = {
            'response_id': response_id,
            'surface_state': dict(surface_state),
            'surface_actionability': surface_actionability,
            'closure_status': closure_status,
            'final_contract_status': final_contract_status,
        }
        add(
            _runtime_evidence_proposal(
                request_phase_graph=graph,
                repair_type='reconcile_surface_state_or_reopen_contract',
                reason='Contract is fulfilled but surface state still reports actionable blocked or review-pending runtime work.',
                evidence_refs=evidence_refs,
                add_phases=[
                    _runtime_repair_phase(
                        repair_type='reconcile_surface_state_or_reopen_contract',
                        evidence_key=evidence_key,
                        metadata=metadata,
                    )
                ],
                metadata=metadata,
                accepted_learning_hints=accepted_learning_hints,
                redraw_scope_review=redraw_scope_review,
            )
        )

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = _clean_text(proposal.get('proposal_id'))
        if not proposal_id or proposal_id in seen:
            continue
        seen.add(proposal_id)
        unique.append(proposal)
    return unique


def validate_graph_repair_proposal(
    proposal: Mapping[str, Any],
    *,
    request_phase_graph: Mapping[str, Any],
    closure_review: Optional[Mapping[str, Any]] = None,
    candidate_graph: Optional[Mapping[str, Any]] = None,
    promotion_review: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate a graph repair proposal without applying it."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    review = closure_review if isinstance(closure_review, Mapping) else {}
    candidates = candidate_graph if isinstance(candidate_graph, Mapping) else {}
    promotion = promotion_review if isinstance(promotion_review, Mapping) else {}
    proposal_payload = proposal if isinstance(proposal, Mapping) else {}
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    def add_check(name: str, status: str, reason: str) -> None:
        checks.append({'check': name, 'status': status, 'reason': reason})
        if status != 'passed' and reason not in reasons:
            reasons.append(reason)

    if not proposal_payload:
        add_check('proposal_shape', 'rejected', 'proposal_missing')
    elif proposal_payload.get('kind') != GRAPH_REPAIR_PROPOSAL_KIND:
        add_check('proposal_shape', 'rejected', 'proposal_kind_mismatch')
    else:
        add_check('proposal_shape', 'passed', 'proposal_schema_present')

    proposal_id = _clean_text(proposal_payload.get('proposal_id'))
    if not proposal_id:
        add_check('proposal_id', 'rejected', 'proposal_id_missing')
    else:
        add_check('proposal_id', 'passed', 'proposal_id_present')

    if proposal_payload.get('allowed_use') != PROPOSAL_ALLOWED_USE:
        add_check('allowed_use', 'rejected', 'proposal_allowed_use_mismatch')
    else:
        add_check('allowed_use', 'passed', 'proposal_only_until_validated')
    if proposal_payload.get('forbidden_use') != PROPOSAL_FORBIDDEN_USE:
        add_check('forbidden_use', 'rejected', 'proposal_forbidden_use_mismatch')
    else:
        add_check('forbidden_use', 'passed', 'runtime_truth_required')

    patch = proposal_payload.get('patch') if isinstance(proposal_payload.get('patch'), Mapping) else {}
    unsupported_patch_keys = [
        key for key, value in patch.items()
        if key not in _SUPPORTED_PATCH_KEYS and value not in (None, '', [], {})
    ]
    if unsupported_patch_keys:
        add_check('patch_scope', 'rejected', 'unsupported_graph_patch_operation')
    else:
        add_check('patch_scope', 'passed', 'patch_is_bounded_to_supported_operations')

    source = proposal_payload.get('source')
    source_status = _status(source)
    explicit_evidence_refs = _clean_string_list(proposal_payload.get('evidence_refs') or [])
    evidence_refs = _runtime_evidence_refs(
        source=source,
        evidence_refs=explicit_evidence_refs,
        closure_review=review,
        promotion_review=promotion,
    )
    if _only_learning_evidence(source, evidence_refs):
        add_check('runtime_evidence', 'rejected', 'accepted_learning_not_runtime_evidence')
    elif source_status in {'monitor_evidence', 'runtime', 'response_frame'} and not explicit_evidence_refs:
        add_check('runtime_evidence', 'rejected', 'runtime_evidence_missing')
    elif not evidence_refs:
        add_check('runtime_evidence', 'rejected', 'runtime_evidence_missing')
    else:
        add_check('runtime_evidence', 'passed', 'runtime_or_closure_evidence_present')

    promotion_status = _status(promotion.get('status'))
    if promotion_status in _REJECTED_STATUSES:
        add_check('promotion_review', 'rejected', 'promotion_review_rejected_or_blocked')
    elif promotion_status in _PROMOTED_STATUSES:
        add_check('promotion_review', 'passed', 'promotion_review_promoted')
    elif _status(source) in {'closure_review', 'graph_closure_review', 'runtime_closure_review'}:
        feedback = review.get('ghost_repair_feedback') if isinstance(review.get('ghost_repair_feedback'), Mapping) else {}
        adequacy = review.get('intent_graph_adequacy') if isinstance(review.get('intent_graph_adequacy'), Mapping) else {}
        adequacy_checks = adequacy.get('checks') if isinstance(adequacy.get('checks'), list) else []
        has_repairable_adequacy_check = any(
            isinstance(item, Mapping)
            and _clean_text(item.get('evidence')).startswith('intent_graph_adequacy_')
            and _clean_text(item.get('repair_action'))
            and (
                bool(item.get('add_dependencies'))
                or bool(item.get('add_phases'))
            )
            for item in adequacy_checks
        )
        if _status(feedback.get('status')) == 'repair_required':
            add_check('promotion_review', 'passed', 'closure_repair_feedback_is_runtime_review_input')
        elif has_repairable_adequacy_check:
            add_check('promotion_review', 'passed', 'intent_graph_adequacy_repair_check_is_runtime_review_input')
        else:
            add_check('promotion_review', 'blocked', 'closure_repair_feedback_missing')
    elif source_status in {'monitor_evidence', 'runtime', 'response_frame'} and explicit_evidence_refs:
        add_check('promotion_review', 'passed', 'runtime_monitor_evidence_is_review_input')
    else:
        add_check('promotion_review', 'blocked', 'promotion_review_missing')

    if _provider_ban_requested(proposal_payload):
        add_check('provider_scope', 'rejected', 'backend_route_health_signal_is_not_graph_patch_authority')
    else:
        add_check('provider_scope', 'passed', 'no_broad_provider_disablement_requested')

    add_phases = patch.get('add_phases') if isinstance(patch.get('add_phases'), list) else []
    add_dependencies = patch.get('add_dependencies') if isinstance(patch.get('add_dependencies'), list) else []
    if not add_phases and not add_dependencies:
        add_check('patch_material', 'blocked', 'patch_contains_no_additive_graph_work')
    planned_phase_ids = {
        _phase_id(phase)
        for phase in add_phases
        if isinstance(phase, Mapping) and _phase_id(phase)
    }
    existing_phase_ids = {
        _phase_id(phase)
        for phase in (graph.get('phases') or [])
        if isinstance(phase, Mapping) and _phase_id(phase)
    }
    existing_branch_ids = {
        _branch_id(branch)
        for branch in (graph.get('downstream_branches') or [])
        if isinstance(branch, Mapping) and _branch_id(branch)
    }
    for phase in add_phases:
        if not isinstance(phase, Mapping):
            add_check('phase_shape', 'rejected', 'add_phase_entry_not_mapping')
            continue
        for error in _phase_validation_errors(phase):
            add_check('phase_contract', 'rejected', error)
        if not _phase_validation_errors(phase):
            add_check(
                f'phase_contract:{_phase_id(phase)}',
                'passed',
                'phase_contract_matches_capability_and_output',
            )
        if _contains_reserved_conflict(
            phase=phase,
            request_phase_graph=graph,
            candidate_graph=candidates,
            promotion_review=promotion,
        ):
            add_check('reserved_or_deferred_intent', 'rejected', 'deferred_or_reserved_intent_conflict')
    for dependency in add_dependencies:
        if not isinstance(dependency, Mapping):
            add_check('dependency_shape', 'rejected', 'add_dependency_entry_not_mapping')
            continue
        target_id, depends_on = _dependency_targets(dependency)
        if not target_id:
            add_check('dependency_contract', 'rejected', 'dependency_target_missing')
            continue
        if target_id not in existing_phase_ids and target_id not in existing_branch_ids and target_id not in planned_phase_ids:
            add_check('dependency_contract', 'rejected', 'dependency_target_not_in_graph_or_patch')
        if not depends_on:
            add_check('dependency_contract', 'rejected', 'dependency_source_missing')
        if target_id and depends_on:
            add_check(
                f'dependency_contract:{target_id}',
                'passed',
                'dependency_edge_is_additive',
            )

    accepted_learning_summary: dict[str, Any] = {}
    if isinstance(accepted_learning_hints, Mapping):
        accepted_learning_summary = {
            'authority': _clean_text(accepted_learning_hints.get('authority')) or 'soft_hint',
            'runtime_effect': _clean_text(accepted_learning_hints.get('runtime_effect')) or 'none',
            'hint_count': accepted_learning_hints.get('hint_count'),
            'allowed_use': 'orientation_only_not_patch_authority',
        }

    hard_rejection = any(item.get('status') == 'rejected' for item in checks)
    blocked = any(item.get('status') == 'blocked' for item in checks)
    status = 'rejected' if hard_rejection else 'blocked' if blocked else 'accepted'
    accepted_patch = patch if status == 'accepted' else {}
    payload = {
        'kind': GRAPH_REPAIR_PROPOSAL_REVIEW_KIND,
        'review_id': _review_id_for({'proposal_id': proposal_id, 'checks': checks}),
        'proposal_id': proposal_id,
        'status': status,
        'authority': 'runtime_validation',
        'target_graph_id': _clean_text(proposal_payload.get('target_graph_id')),
        'graph_digest': _stable_digest(graph, prefix='graph-'),
        'proposal_digest': _stable_digest(proposal_payload, prefix='proposal-'),
        'runtime_evidence_refs': evidence_refs,
        'reasons': reasons,
        'validation_checks': checks,
        'accepted_patch': accepted_patch,
        'allowed_runtime_action': 'apply_additive_patch' if status == 'accepted' else 'none',
        'accepted_learning_boundary': accepted_learning_summary,
        'source_proposal': proposal_payload,
    }
    return _json_safe(payload)


def apply_validated_graph_repair_patch(
    request_phase_graph: Mapping[str, Any],
    proposal_review: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an accepted additive patch to a request phase graph idempotently."""

    graph = copy.deepcopy(dict(request_phase_graph or {}))
    review = proposal_review if isinstance(proposal_review, Mapping) else {}
    original_digest = _stable_digest(graph, prefix='graph-')
    proposal_id = _clean_text(review.get('proposal_id'))
    patch = review.get('accepted_patch') if isinstance(review.get('accepted_patch'), Mapping) else {}
    if review.get('kind') != GRAPH_REPAIR_PROPOSAL_REVIEW_KIND or _status(review.get('status')) != 'accepted':
        return _json_safe(
            {
                'kind': GRAPH_REPAIR_PATCH_APPLICATION_KIND,
                'status': 'rejected',
                'reason': 'proposal_review_not_accepted',
                'proposal_id': proposal_id,
                'original_graph_digest': original_digest,
                'graph': graph,
            }
        )

    add_phases = [item for item in (patch.get('add_phases') or []) if isinstance(item, Mapping)]
    downstream_branches = [
        dict(item) for item in (graph.get('downstream_branches') or []) if isinstance(item, Mapping)
    ]
    phases = [dict(item) for item in (graph.get('phases') or []) if isinstance(item, Mapping)]
    output_obligations = [
        dict(item) for item in (graph.get('output_obligations') or []) if isinstance(item, Mapping)
    ]
    existing_branch_ids = {_branch_id(item) for item in downstream_branches if _branch_id(item)}
    existing_phase_ids = {_phase_id(item) for item in phases if _phase_id(item)}
    existing_obligation_ids = {
        _clean_text(item.get('obligation_id'))
        for item in output_obligations
        if _clean_text(item.get('obligation_id'))
    }

    applied_branch_ids: list[str] = []
    applied_phase_ids: list[str] = []
    applied_obligation_ids: list[str] = []
    applied_dependency_edges: list[dict[str, str]] = []
    for raw_phase in add_phases:
        phase = _phase_from_branch(raw_phase)
        branch_id = _branch_id(phase)
        phase_id = _phase_id(phase)
        capability = _normalize_capability(phase.get('capability'))
        output_type = _clean_text(phase.get('output_type')).lower() or _output_type_for_capability(capability)
        if not branch_id or not phase_id or not capability:
            continue
        if branch_id not in existing_branch_ids:
            downstream_branches.append(_compact_payload({**phase, 'status': _clean_text(phase.get('status')) or 'pending'}))
            existing_branch_ids.add(branch_id)
            applied_branch_ids.append(branch_id)
        if phase_id not in existing_phase_ids:
            phases.append(
                _compact_payload(
                    {
                        **phase,
                        'phase_id': phase_id,
                        'branch_id': branch_id,
                        'capability': capability,
                        'output_type': output_type or None,
                        'status': _clean_text(phase.get('status')) or 'pending',
                        'required': phase.get('required') if phase.get('required') is not None else True,
                        'source': _clean_text(phase.get('source')) or 'graph_repair_patch',
                    }
                )
            )
            existing_phase_ids.add(phase_id)
            applied_phase_ids.append(phase_id)
        obligation_id = _clean_text(phase.get('obligation_id')) or f'obligation-{phase_id}'
        if obligation_id and output_type and obligation_id not in existing_obligation_ids:
            output_obligations.append(
                _compact_payload(
                    {
                        'obligation_id': obligation_id,
                        'phase_id': phase_id,
                        'branch_id': branch_id,
                        'capability': capability,
                        'output_type': output_type,
                        'status': 'pending',
                        'required': True,
                        'source': 'graph_repair_patch',
                        'repair_action': phase.get('repair_action'),
                        'repair_contract_id': phase.get('repair_contract_id'),
                        'repair_execution_policy': phase.get('repair_execution_policy'),
                        'depends_on': phase.get('depends_on'),
                        'input_refs': phase.get('input_refs'),
                        'review_criteria': phase.get('review_criteria'),
                        'semantic_review_criteria': phase.get('semantic_review_criteria'),
                    }
                )
            )
            existing_obligation_ids.add(obligation_id)
            applied_obligation_ids.append(obligation_id)

    for dependency in (patch.get('add_dependencies') or []):
        if not isinstance(dependency, Mapping):
            continue
        target_id, depends_on = _dependency_targets(dependency)
        if not target_id or not depends_on:
            continue
        for collection in (downstream_branches, phases, output_obligations):
            for record in collection:
                if not isinstance(record, dict):
                    continue
                record_ids = {
                    _clean_text(record.get('phase_id')),
                    _clean_text(record.get('branch_id')),
                    _clean_text(record.get('obligation_id')),
                }
                if target_id not in record_ids:
                    continue
                current_deps = _clean_string_list(record.get('depends_on'))
                for source_id in depends_on:
                    if source_id in current_deps:
                        continue
                    current_deps.append(source_id)
                    edge = {'target_id': target_id, 'source_id': source_id}
                    if edge not in applied_dependency_edges:
                        applied_dependency_edges.append(edge)
                if current_deps:
                    record['depends_on'] = current_deps

    graph['downstream_branches'] = downstream_branches
    graph['downstream_branch_ids'] = [_branch_id(item) for item in downstream_branches if _branch_id(item)]
    graph['downstream_capabilities'] = list(
        dict.fromkeys(
            _normalize_capability(item.get('capability'))
            for item in downstream_branches
            if _normalize_capability(item.get('capability'))
        )
    )
    graph['phases'] = phases
    graph['output_obligations'] = output_obligations
    if applied_branch_ids or applied_phase_ids or applied_obligation_ids or applied_dependency_edges:
        graph['continuation_required'] = True

    patch_digest = _stable_digest(patch, prefix='patch-')
    refinements = [
        dict(item) for item in (graph.get('graph_refinements') or []) if isinstance(item, Mapping)
    ]
    duplicate_refinement = any(
        _clean_text(item.get('proposal_id')) == proposal_id
        and _clean_text(item.get('patch_digest')) == patch_digest
        for item in refinements
    )
    changed = bool(
        applied_branch_ids
        or applied_phase_ids
        or applied_obligation_ids
        or applied_dependency_edges
    )
    status = 'applied' if changed else 'already_applied'
    if changed and not duplicate_refinement:
        refinements.append(
            {
                'kind': GRAPH_REPAIR_PATCH_APPLICATION_KIND,
                'status': status,
                'proposal_id': proposal_id,
                'review_id': review.get('review_id'),
                'patch_digest': patch_digest,
                'applied_branch_ids': applied_branch_ids,
                'applied_phase_ids': applied_phase_ids,
                'applied_obligation_ids': applied_obligation_ids,
                'applied_dependency_edges': applied_dependency_edges,
                'authority': 'runtime_validated_graph_patch',
            }
        )
    graph['graph_refinements'] = refinements

    return _json_safe(
        {
            'kind': GRAPH_REPAIR_PATCH_APPLICATION_KIND,
            'status': status,
            'proposal_id': proposal_id,
            'review_id': review.get('review_id'),
            'original_graph_digest': original_digest,
            'patch_digest': patch_digest,
            'patched_graph_digest': _stable_digest(graph, prefix='graph-'),
            'applied_branch_ids': applied_branch_ids,
            'applied_phase_ids': applied_phase_ids,
            'applied_obligation_ids': applied_obligation_ids,
            'applied_dependency_edges': applied_dependency_edges,
            'graph': graph,
        }
    )


def normalize_graph_repair_autonomy(value: Any) -> str:
    """Normalize graph repair autonomy rollout values."""

    token = _status(value)
    if token in _GRAPH_REPAIR_AUTONOMY_LEVELS:
        return token
    return 'off'


def describe_graph_repair_autonomy(
    value: Any,
    *,
    source: str = 'explicit_value',
    configured: bool = True,
) -> dict[str, Any]:
    """Return normalized autonomy plus safe diagnostics for invalid raw values."""

    raw_value = value
    token = _status(value)
    normalized = normalize_graph_repair_autonomy(value)
    invalid_value = bool(token and token not in _GRAPH_REPAIR_AUTONOMY_LEVELS)
    return _json_safe(
        {
            'raw_value': raw_value,
            'autonomy_level': normalized,
            'normalized': normalized,
            'invalid_value': invalid_value,
            'source': source,
            'configured': bool(configured),
        }
    )


def graph_repair_autonomy_from_env(env: Optional[Mapping[str, Any]] = None) -> str:
    source = env if isinstance(env, Mapping) else os.environ
    if GRAPH_REPAIR_AUTONOMY_ENV not in source:
        return GRAPH_REPAIR_AUTONOMY_PRODUCT_DEFAULT
    return normalize_graph_repair_autonomy(source.get(GRAPH_REPAIR_AUTONOMY_ENV))


def describe_graph_repair_autonomy_from_env(env: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    source = env if isinstance(env, Mapping) else os.environ
    configured = GRAPH_REPAIR_AUTONOMY_ENV in source
    raw_value = (
        source.get(GRAPH_REPAIR_AUTONOMY_ENV)
        if configured
        else GRAPH_REPAIR_AUTONOMY_PRODUCT_DEFAULT
    )
    return describe_graph_repair_autonomy(
        raw_value,
        source='environment' if configured else 'product_default',
        configured=configured,
    )


def _proposal_from_review(proposal_review: Mapping[str, Any]) -> dict[str, Any]:
    proposal = proposal_review.get('source_proposal')
    return dict(proposal) if isinstance(proposal, Mapping) else {}


def _patch_from_review(proposal_review: Mapping[str, Any]) -> dict[str, Any]:
    patch = proposal_review.get('accepted_patch')
    if isinstance(patch, Mapping) and patch:
        return dict(patch)
    proposal = _proposal_from_review(proposal_review)
    proposal_patch = proposal.get('patch')
    return dict(proposal_patch) if isinstance(proposal_patch, Mapping) else {}


def _patch_planned_branch_ids(patch: Mapping[str, Any]) -> list[str]:
    branch_ids: list[str] = []
    for phase in patch.get('add_phases') or []:
        if not isinstance(phase, Mapping):
            continue
        branch_id = _branch_id(phase) or _phase_id(phase)
        if branch_id and branch_id not in branch_ids:
            branch_ids.append(branch_id)
    return branch_ids


def _graph_has_applied_patch(graph: Mapping[str, Any], idempotency_key: str, patch_id: str) -> bool:
    for item in graph.get('applied_graph_patches') or []:
        if not isinstance(item, Mapping):
            continue
        if _clean_text(item.get('idempotency_key')) == idempotency_key:
            return True
        if patch_id and _clean_text(item.get('patch_id')) == patch_id:
            return True
    for item in graph.get('graph_patch_lifecycle') or []:
        if not isinstance(item, Mapping):
            continue
        if _status(item.get('status')) not in {'applied', 'already_applied'}:
            continue
        if _clean_text(item.get('idempotency_key')) == idempotency_key:
            return True
        if patch_id and _clean_text(item.get('patch_id')) == patch_id:
            return True
    return False


def _upsert_record(
    records: Sequence[Any],
    record: Mapping[str, Any],
    *identity_keys: str,
) -> list[dict[str, Any]]:
    payload = dict(record)
    identity = ''
    for key in identity_keys:
        identity = _clean_text(payload.get(key))
        if identity:
            break
    updated: list[dict[str, Any]] = []
    replaced = False
    for item in records or []:
        if not isinstance(item, Mapping):
            continue
        item_payload = dict(item)
        item_identity = ''
        for key in identity_keys:
            item_identity = _clean_text(item_payload.get(key))
            if item_identity:
                break
        if identity and item_identity == identity:
            updated.append(_json_safe({**item_payload, **payload}))
            replaced = True
        else:
            updated.append(_json_safe(item_payload))
    if not replaced:
        updated.append(_json_safe(payload))
    return updated


def classify_graph_repair_patch(
    proposal_review: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a validated graph repair patch into safe/review/forbidden classes."""

    review = proposal_review if isinstance(proposal_review, Mapping) else {}
    proposal = _proposal_from_review(review)
    patch = _patch_from_review(review)
    repair_actions = proposal.get('repair_actions')
    if isinstance(repair_actions, list):
        repair_action = repair_actions[0] if repair_actions else ''
    else:
        repair_action = repair_actions
    repair_type = _status(
        proposal.get('repair_type')
        or proposal.get('repair_gap_code')
        or repair_action
    )
    forbidden_reasons: list[str] = []

    if _provider_ban_requested(proposal):
        return {
            'repair_class': 'broad_provider_disablement',
            'risk_level': 'forbidden',
            'reasons': ['backend_route_health_signal_is_not_graph_patch_authority'],
        }
    if repair_type in _FORBIDDEN_REPAIR_TYPES:
        return {
            'repair_class': _REPAIR_TYPE_TO_PATCH_CLASS.get(repair_type, repair_type),
            'risk_level': 'forbidden',
            'reasons': [f'forbidden_repair_type:{repair_type}'],
        }
    if repair_type in _REVIEW_REQUIRED_REPAIR_TYPES:
        repair_class = {
            'branch_merge': 'branch_identity_merge',
            'merge_branch': 'branch_identity_merge',
            'branch_split': 'branch_identity_split',
            'split_branch': 'branch_identity_split',
            'reopen_terminal_contract': 'terminal_contract_reopen',
            'supersede_candidate_branch': 'candidate_supersession',
        }.get(repair_type, 'branch_priority_or_capability_change')
        return {
            'repair_class': repair_class,
            'risk_level': 'review_required',
            'reasons': ['patch_class_requires_reviewed_autonomy'],
        }

    if any(patch.get(key) not in (None, '', [], {}) for key in ('mark_reserved', 'supersede_phases')):
        forbidden_reasons.append('patch_attempts_reserved_or_supersession_mutation')
    if patch.get('delete_phases') not in (None, '', [], {}):
        forbidden_reasons.append('patch_attempts_delete_or_hide_branch')
    if forbidden_reasons:
        return {
            'repair_class': 'delete_or_hide_branch',
            'risk_level': 'forbidden',
            'reasons': forbidden_reasons,
        }
    if patch.get('update_phase_fields') not in (None, '', [], {}):
        return {
            'repair_class': 'branch_priority_or_capability_change',
            'risk_level': 'review_required',
            'reasons': ['patch_class_requires_reviewed_autonomy'],
        }

    add_dependencies = [
        item for item in patch.get('add_dependencies') or []
        if isinstance(item, Mapping)
    ]
    add_phases = [
        item for item in patch.get('add_phases') or []
        if isinstance(item, Mapping)
    ]
    if add_dependencies:
        return {
            'repair_class': 'missing_dependency_edge',
            'risk_level': 'safe_additive',
            'reasons': [],
        }
    mapped_class = _REPAIR_TYPE_TO_PATCH_CLASS.get(repair_type)
    if mapped_class:
        return {
            'repair_class': mapped_class,
            'risk_level': 'safe_additive' if mapped_class in _SAFE_GRAPH_PATCH_CLASSES else 'review_required',
            'reasons': [],
        }

    phase_actions = {
        _status(phase.get('repair_action') or phase.get('repair_evidence'))
        for phase in add_phases
        if isinstance(phase, Mapping)
    }
    phase_text = json.dumps(_json_safe(add_phases), sort_keys=True).lower()
    if any(action in phase_actions for action in {'semantic_review', 'branch_semantic_review', 'global_semantic_closure'}):
        repair_class = 'semantic_review_branch'
    elif any('artifact' in action or 'binding' in action for action in phase_actions) or 'artifact' in phase_text:
        repair_class = 'artifact_binding_repair_branch'
    elif any('dependency' in action for action in phase_actions) or 'depends_on' in phase_text:
        repair_class = 'missing_dependency_edge'
    elif any('syntax' in action or 'text_io' in action for action in phase_actions):
        repair_class = 'text_io_repair_branch'
    elif any('materialization' in action or 'promoted_obligation' in action for action in phase_actions):
        repair_class = 'missing_materialization_branch'
    else:
        repair_class = 'missing_materialization_branch' if add_phases else 'artifact_binding_repair_branch'
    return {
        'repair_class': repair_class,
        'risk_level': 'safe_additive' if repair_class in _SAFE_GRAPH_PATCH_CLASSES else 'review_required',
        'reasons': [],
    }


def _graph_patch_autonomy_allows(risk_level: str, autonomy_level: str) -> bool:
    if autonomy_level in {'shadow', 'stage'}:
        return True
    if autonomy_level == 'apply_safe':
        return risk_level == 'safe_additive'
    if autonomy_level == 'apply_reviewed':
        return risk_level in {'safe_additive', 'review_required'}
    return False


def _graph_patch_authorization(payload: Mapping[str, Any]) -> dict[str, Any]:
    authorization = payload.get('graph_patch_authorization')
    if not isinstance(authorization, Mapping):
        review = payload.get('validation_review')
        if isinstance(review, Mapping):
            authorization = review.get('graph_patch_authorization')
    return dict(authorization) if isinstance(authorization, Mapping) else {}


def _graph_patch_authorization_allows(payload: Mapping[str, Any], autonomy_level: str) -> bool:
    authorization = _graph_patch_authorization(payload)
    if not authorization:
        return False
    status = _status(authorization.get('status'))
    if status not in _PROMOTED_STATUSES | {'authorized'}:
        return False
    authority = _status(authorization.get('authority'))
    if authority not in {'runtime_review', 'runtime_validation', 'operator_review'}:
        return False
    allowed_autonomy = _clean_string_list(authorization.get('allowed_autonomy'))
    normalized_allowed = {normalize_graph_repair_autonomy(item) for item in allowed_autonomy}
    if '*' not in allowed_autonomy and autonomy_level not in normalized_allowed:
        return False
    if not _clean_string_list(authorization.get('evidence_refs')):
        return False
    return True


def _lifecycle_runtime_effect(*, status: str, autonomy_level: str) -> str:
    if status in {'blocked', 'rejected'}:
        return 'none'
    if autonomy_level == 'shadow':
        return 'shadow_no_mutation'
    if autonomy_level == 'stage':
        return 'staged_no_executable_mutation'
    return 'pending_application'


def build_graph_patch_lifecycle(
    *,
    request_phase_graph: Mapping[str, Any],
    proposal_review: Mapping[str, Any],
    autonomy_level: str = 'stage',
) -> dict[str, Any]:
    """Build a backend-visible graph patch lifecycle record without applying it."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    review = proposal_review if isinstance(proposal_review, Mapping) else {}
    level = normalize_graph_repair_autonomy(autonomy_level)
    proposal = _proposal_from_review(review)
    patch = _patch_from_review(review)
    classification = classify_graph_repair_patch(review)
    repair_class = _clean_text(classification.get('repair_class')) or 'missing_materialization_branch'
    risk_level = _clean_text(classification.get('risk_level')) or 'review_required'
    proposal_id = _clean_text(review.get('proposal_id') or proposal.get('proposal_id'))
    review_id = _clean_text(review.get('review_id'))
    before_digest = _stable_digest(graph, prefix='graph-')
    patch_digest = _stable_digest(patch, prefix='patch-')
    patch_id = _stable_digest(
        {
            'proposal_id': proposal_id,
            'review_id': review_id,
            'repair_class': repair_class,
            'patch_digest': patch_digest,
        },
        prefix='graph-patch-',
    )
    idempotency_key = _stable_digest(
        {
            'target_graph_id': review.get('target_graph_id') or proposal.get('target_graph_id') or before_digest,
            'proposal_id': proposal_id,
            'patch_digest': patch_digest,
            'repair_class': repair_class,
        },
        prefix='graph-patch-idem-',
    )
    blocked_reasons = _clean_string_list(classification.get('reasons'))
    if risk_level == 'review_required':
        blocked_reasons = []
    review_status = _status(review.get('status'))
    authorization = _graph_patch_authorization(review)
    evidence_refs = _clean_string_list(review.get('runtime_evidence_refs'))
    for ref in _clean_string_list(proposal.get('evidence_refs')):
        _append_unique(evidence_refs, ref)
    scheduled_branch_ids = _patch_planned_branch_ids(patch)
    enforced_policy_review: dict[str, Any] = {}
    if review.get('kind') != GRAPH_REPAIR_PROPOSAL_REVIEW_KIND or review_status != 'accepted':
        status = 'rejected' if review_status == 'rejected' else 'blocked'
        _append_unique(blocked_reasons, 'proposal_review_not_accepted')
    elif risk_level == 'forbidden':
        status = 'rejected'
        _append_unique(blocked_reasons, 'forbidden_graph_patch_class')
    elif level == 'off':
        status = 'blocked'
        _append_unique(blocked_reasons, 'graph_repair_autonomy_off')
    elif level == 'shadow':
        status = 'validated'
    elif level == 'stage':
        status = 'staged'
    elif level == 'apply_enforced':
        enforced_policy_review = build_enforced_policy_review(
            autonomy_level=level,
            lifecycle={
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'patch_id': patch_id,
                'proposal_id': proposal_id,
                'review_id': review_id,
                'repair_class': repair_class,
                'risk_level': risk_level,
                'source_evidence_refs': evidence_refs,
                'validation_review': review,
                'idempotency_key': idempotency_key,
                'scheduled_branch_ids': scheduled_branch_ids,
                'outcome': {'status': 'pending', 'runtime_effect': 'pending_application'},
            },
            request_phase_graph=graph,
        )
        if enforced_policy_allows_application(enforced_policy_review):
            status = 'staged'
        else:
            status = 'blocked'
            for reason in _clean_string_list(enforced_policy_review.get('blocked_reasons')):
                _append_unique(blocked_reasons, reason)
    elif level == 'apply_reviewed' and risk_level == 'review_required' and not _graph_patch_authorization_allows(review, level):
        status = 'blocked'
        _append_unique(blocked_reasons, 'apply_reviewed_requires_explicit_review_authorization')
    elif not _graph_patch_autonomy_allows(risk_level, level):
        status = 'blocked'
        _append_unique(blocked_reasons, 'patch_class_requires_reviewed_autonomy')
    else:
        status = 'staged'

    payload = {
        'kind': GRAPH_PATCH_LIFECYCLE_KIND,
        'patch_id': patch_id,
        'proposal_id': proposal_id,
        'review_id': review_id,
        'repair_class': repair_class,
        'status': status,
        'autonomy_level': level,
        'risk_level': risk_level,
        'source_evidence_refs': evidence_refs,
        'target_graph_version': review.get('target_graph_id') or proposal.get('target_graph_id') or before_digest,
        'before_graph_digest': before_digest,
        'patch_digest': patch_digest,
        'idempotency_key': idempotency_key,
        'preconditions': [
            'proposal_review_accepted',
            'graph_digest_matches_before_graph_digest',
            'risk_level_allowed_for_autonomy',
        ],
        'postconditions': [
            'additive_graph_patch_visible_in_backend_truth',
            'no_duplicate_patch_for_idempotency_key',
        ],
        'validation_review': review,
        'accepted_patch': patch,
        'graph_patch_authorization': authorization,
        'scheduled_branch_ids': scheduled_branch_ids,
        'blocked_reasons': blocked_reasons,
        'outcome': {
            'status': status,
            'runtime_effect': _lifecycle_runtime_effect(status=status, autonomy_level=level),
        },
    }
    if enforced_policy_review:
        lineage = build_enforced_lineage_summary(graph, payload)
        payload.update(
            _compact_payload(
                {
                    'authority': 'runtime_enforced_policy'
                    if enforced_policy_allows_application(enforced_policy_review)
                    else 'runtime_enforced_policy_denied',
                    'enforced_policy_review': enforced_policy_review,
                    'enforced_policy_id': enforced_policy_review.get('policy_id'),
                    'enforced_class': enforced_policy_review.get('enforced_class'),
                    'policy_mode': enforced_policy_review.get('policy_mode'),
                    'allowed_by_policy': enforced_policy_review.get('allowed'),
                    'current_evidence_refs': enforced_policy_review.get('current_evidence_refs'),
                    'forbidden_evidence_seen': enforced_policy_review.get('forbidden_evidence_seen'),
                    'redraw_scope_review_ref': enforced_policy_review.get('redraw_scope_review_ref'),
                    **lineage,
                }
            )
        )
    return _json_safe(payload)


def _with_lifecycle_record(graph: dict[str, Any], lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    graph['graph_patch_lifecycle'] = _upsert_record(
        graph.get('graph_patch_lifecycle') or [],
        lifecycle,
        'patch_id',
        'idempotency_key',
    )
    return graph


def apply_validated_graph_patch(
    request_phase_graph: Mapping[str, Any],
    patch_lifecycle: Mapping[str, Any],
    *,
    autonomy_level: str = 'stage',
) -> dict[str, Any]:
    """Apply or record a validated graph patch according to the autonomy level."""

    graph = copy.deepcopy(dict(request_phase_graph or {}))
    lifecycle = dict(patch_lifecycle) if isinstance(patch_lifecycle, Mapping) else {}
    level = normalize_graph_repair_autonomy(autonomy_level or lifecycle.get('autonomy_level'))
    patch_id = _clean_text(lifecycle.get('patch_id'))
    proposal_id = _clean_text(lifecycle.get('proposal_id'))
    idempotency_key = _clean_text(lifecycle.get('idempotency_key'))
    risk_level = _clean_text(lifecycle.get('risk_level')) or 'review_required'
    base_status = _status(lifecycle.get('status'))
    blocked_reasons = _clean_string_list(lifecycle.get('blocked_reasons'))

    if lifecycle.get('kind') != GRAPH_PATCH_LIFECYCLE_KIND:
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': 'rejected',
                'autonomy_level': level,
                'blocked_reasons': [*blocked_reasons, 'patch_lifecycle_kind_mismatch'],
                'outcome': {'status': 'rejected', 'runtime_effect': 'none'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': 'rejected',
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'blocked_reasons': result_lifecycle.get('blocked_reasons'),
                'graph': graph,
            }
        )

    if idempotency_key and _graph_has_applied_patch(graph, idempotency_key, patch_id):
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'already_applied',
                'autonomy_level': level,
                'after_graph_digest': _stable_digest(graph, prefix='graph-'),
                'outcome': {
                    'status': 'already_applied',
                    'runtime_effect': 'idempotency_guard_no_duplicate_work',
                },
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': 'already_applied',
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'idempotency_key': idempotency_key,
                'graph': graph,
            }
        )

    if base_status in {'rejected', 'blocked'}:
        graph = _with_lifecycle_record(graph, lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': base_status,
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'blocked_reasons': blocked_reasons,
                'graph': graph,
            }
        )

    enforced_policy_review: dict[str, Any] = {}
    if level == 'apply_enforced':
        enforced_policy_review = build_enforced_policy_review(
            autonomy_level=level,
            lifecycle=lifecycle,
            request_phase_graph=graph,
        )
        if not enforced_policy_allows_application(enforced_policy_review):
            for reason in _clean_string_list(enforced_policy_review.get('blocked_reasons')):
                _append_unique(blocked_reasons, reason)
    if risk_level == 'forbidden':
        _append_unique(blocked_reasons, 'forbidden_graph_patch_class')
    elif level == 'off':
        _append_unique(blocked_reasons, 'graph_repair_autonomy_off')
    elif level == 'apply_enforced':
        pass
    elif level == 'apply_reviewed' and risk_level == 'review_required' and not _graph_patch_authorization_allows(lifecycle, level):
        _append_unique(blocked_reasons, 'apply_reviewed_requires_explicit_review_authorization')
    elif not _graph_patch_autonomy_allows(risk_level, level):
        _append_unique(blocked_reasons, 'patch_class_requires_reviewed_autonomy')
    if blocked_reasons:
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'rejected' if risk_level == 'forbidden' else 'blocked',
                'autonomy_level': level,
                'blocked_reasons': blocked_reasons,
                **(
                    _compact_payload(
                        {
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_policy_review': enforced_policy_review,
                            'enforced_policy_id': enforced_policy_review.get('policy_id') if enforced_policy_review else None,
                            'enforced_class': enforced_policy_review.get('enforced_class') if enforced_policy_review else None,
                            'policy_mode': enforced_policy_review.get('policy_mode') if enforced_policy_review else None,
                            'allowed_by_policy': enforced_policy_review.get('allowed') if enforced_policy_review else None,
                            'current_evidence_refs': enforced_policy_review.get('current_evidence_refs') if enforced_policy_review else None,
                            'forbidden_evidence_seen': enforced_policy_review.get('forbidden_evidence_seen') if enforced_policy_review else None,
                        }
                    )
                    if enforced_policy_review
                    else {}
                ),
                'outcome': {'status': 'blocked', 'runtime_effect': 'none'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': result_lifecycle.get('status'),
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'blocked_reasons': blocked_reasons,
                'graph': graph,
            }
        )

    if level == 'shadow':
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'validated',
                'autonomy_level': level,
                'outcome': {'status': 'validated', 'runtime_effect': 'shadow_no_mutation'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': 'validated',
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'idempotency_key': idempotency_key,
                'graph': graph,
            }
        )

    if level == 'stage':
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'staged',
                'autonomy_level': level,
                'outcome': {'status': 'staged', 'runtime_effect': 'staged_no_executable_mutation'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        graph['staged_graph_patches'] = _upsert_record(
            graph.get('staged_graph_patches') or [],
            result_lifecycle,
            'patch_id',
            'idempotency_key',
        )
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': 'staged',
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'idempotency_key': idempotency_key,
                'graph': graph,
            }
        )

    current_digest = _stable_digest(graph, prefix='graph-')
    expected_digest = _clean_text(lifecycle.get('before_graph_digest'))
    if expected_digest and current_digest != expected_digest:
        _append_unique(blocked_reasons, 'graph_digest_precondition_mismatch')
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'blocked',
                'autonomy_level': level,
                'blocked_reasons': blocked_reasons,
                'outcome': {'status': 'blocked', 'runtime_effect': 'none'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'status': 'blocked',
                'proposal_id': proposal_id,
                'patch_id': patch_id,
                'blocked_reasons': blocked_reasons,
                'graph': graph,
            }
        )

    review = lifecycle.get('validation_review') if isinstance(lifecycle.get('validation_review'), Mapping) else {}
    application = apply_validated_graph_repair_patch(graph, review)
    patched_graph = (
        dict(application.get('graph') or {})
        if isinstance(application.get('graph'), Mapping)
        else graph
    )
    application_status = _status(application.get('status')) or 'blocked'
    result_lifecycle = _json_safe(
        {
            **lifecycle,
            'status': application_status,
            'autonomy_level': level,
            'authority': 'runtime_enforced_policy'
            if level == 'apply_enforced' and application_status in {'applied', 'already_applied'}
            else lifecycle.get('authority'),
            'after_graph_digest': application.get('patched_graph_digest')
            or _stable_digest(patched_graph, prefix='graph-'),
            'applied_branch_ids': application.get('applied_branch_ids'),
            'applied_phase_ids': application.get('applied_phase_ids'),
            'applied_obligation_ids': application.get('applied_obligation_ids'),
            'applied_dependency_edges': application.get('applied_dependency_edges'),
            **(
                _compact_payload(
                    {
                        'enforced_policy_review': enforced_policy_review or lifecycle.get('enforced_policy_review'),
                        'enforced_policy_id': (
                            enforced_policy_review.get('policy_id')
                            if enforced_policy_review
                            else lifecycle.get('enforced_policy_id')
                        ),
                        'enforced_class': (
                            enforced_policy_review.get('enforced_class')
                            if enforced_policy_review
                            else lifecycle.get('enforced_class')
                        ),
                        'policy_mode': (
                            enforced_policy_review.get('policy_mode')
                            if enforced_policy_review
                            else lifecycle.get('policy_mode')
                        ),
                        'allowed_by_policy': (
                            enforced_policy_review.get('allowed')
                            if enforced_policy_review
                            else lifecycle.get('allowed_by_policy')
                        ),
                        'current_evidence_refs': (
                            enforced_policy_review.get('current_evidence_refs')
                            if enforced_policy_review
                            else lifecycle.get('current_evidence_refs')
                        ),
                        'forbidden_evidence_seen': (
                            enforced_policy_review.get('forbidden_evidence_seen')
                            if enforced_policy_review
                            else lifecycle.get('forbidden_evidence_seen')
                        ),
                    }
                )
                if level == 'apply_enforced'
                else {}
            ),
            'outcome': {
                'status': application_status,
                'runtime_effect': 'graph_mutated' if application_status == 'applied' else 'idempotency_guard_no_duplicate_work',
            },
        }
    )
    patched_graph = _with_lifecycle_record(patched_graph, result_lifecycle)
    if application_status in {'applied', 'already_applied'}:
        patched_graph['applied_graph_patches'] = _upsert_record(
            patched_graph.get('applied_graph_patches') or [],
            {
                'kind': GRAPH_PATCH_LIFECYCLE_KIND,
                'patch_id': patch_id,
                'proposal_id': proposal_id,
                'review_id': lifecycle.get('review_id'),
                'repair_class': lifecycle.get('repair_class'),
                'risk_level': risk_level,
                'status': application_status,
                'idempotency_key': idempotency_key,
                'patch_digest': lifecycle.get('patch_digest'),
                'before_graph_digest': lifecycle.get('before_graph_digest'),
                'after_graph_digest': result_lifecycle.get('after_graph_digest'),
                'authority': 'runtime_enforced_policy'
                if level == 'apply_enforced'
                else lifecycle.get('authority'),
            },
            'idempotency_key',
            'patch_id',
        )

    return _json_safe(
        {
            **application,
            'kind': GRAPH_PATCH_LIFECYCLE_KIND,
            'status': application_status,
            'patch_id': patch_id,
            'idempotency_key': idempotency_key,
            'repair_class': lifecycle.get('repair_class'),
            'risk_level': risk_level,
            'graph': patched_graph,
        }
    )
