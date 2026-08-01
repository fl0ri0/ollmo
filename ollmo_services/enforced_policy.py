"""Default-deny policy gates for bounded apply_enforced runtime authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from typing import Any, Optional

ENFORCED_POLICY_REVIEW_KIND = 'ollmo.enforced_policy_review'
ENFORCED_POLICY_DEFAULT_ID = 'ollmo-enforced-policy-v1'
ENFORCED_POLICY_ENV = 'OLLMO_APPLY_ENFORCED_POLICY'
ENFORCED_POLICY_PRODUCT_DEFAULT_MODE = 'safe_v1'

_ENFORCED_POLICY_MODES = {'off', 'audit', 'safe_v1', 'custom'}
_ALLOWED_SAFE_V1_CLASSES = {
    'duplicate_artifact_alias_canonicalization',
    'safe_additive_artifact_binding_repair',
    'safe_additive_dependency_repair',
    'safe_additive_missing_branch',
}
_FORBIDDEN_EVIDENCE_TOKENS = {
    'accepted_learning',
    'advisory',
    'cache_liveness',
    'degraded',
    'degraded_liveness_only',
    'frontend',
    'ghost_prose',
    'learning_only',
    'liveness_only',
    'model_prose',
    'monitor_only',
    'provider_ban',
    'provider_family_ban',
    'route_health',
    'ui_label',
}
_CURRENT_RUNTIME_EVIDENCE_TOKENS = {
    'artifact_identity',
    'closure',
    'graph_closure_review',
    'intent_graph_adequacy',
    'materialization_contract',
    'promotion_review',
    'response_frame',
    'runtime',
    'runtime_closure_review',
}
_REPAIR_CLASS_TO_ENFORCED_CLASS = {
    'artifact_binding_repair_branch': 'safe_additive_artifact_binding_repair',
    'missing_dependency_edge': 'safe_additive_dependency_repair',
    'missing_materialization_branch': 'safe_additive_missing_branch',
}
_ENFORCED_CLASS_SCOPES = {
    'duplicate_artifact_alias_canonicalization': {'repair_artifact_ref_identity'},
    'safe_additive_artifact_binding_repair': {'repair_binding_dependency', 'repair_artifact_ref_identity'},
    'safe_additive_dependency_repair': {'repair_binding_dependency'},
    'safe_additive_missing_branch': {'add_missing_branch', 'fill_reserved_slot', 'promote_reserved_slot'},
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _status(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_').replace(' ', '_')


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


def _append_unique(values: list[str], value: str) -> None:
    text = _clean_text(value)
    if text and text not in values:
        values.append(text)


def normalize_enforced_policy_mode(value: Any) -> str:
    """Normalize enforced policy mode. Invalid values fail closed to off."""

    mode = _status(value)
    return mode if mode in _ENFORCED_POLICY_MODES else 'off'


def describe_enforced_policy(value: Any = None, *, env: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Describe the active enforced policy knob with invalid values visible."""

    raw = value
    configured = value is not None
    value_source = 'explicit_value'
    if raw is None:
        source = env if isinstance(env, Mapping) else os.environ
        configured = ENFORCED_POLICY_ENV in source
        raw = (
            source.get(ENFORCED_POLICY_ENV)
            if configured
            else ENFORCED_POLICY_PRODUCT_DEFAULT_MODE
        )
        value_source = 'environment' if configured else 'product_default'
    normalized = normalize_enforced_policy_mode(raw)
    raw_status = _status(raw)
    invalid_value = _clean_text(raw) if raw_status and raw_status not in _ENFORCED_POLICY_MODES else ''
    policy_id = ENFORCED_POLICY_DEFAULT_ID if normalized in {'off', 'audit', 'safe_v1'} else 'ollmo-enforced-policy-custom'
    return _compact_payload(
        {
            'policy_id': policy_id,
            'env_var': ENFORCED_POLICY_ENV,
            'raw_value': _clean_text(raw),
            'mode': normalized,
            'normalized': normalized,
            'invalid_value': invalid_value,
            'enabled': normalized == 'safe_v1',
            'allowed_classes': sorted(_ALLOWED_SAFE_V1_CLASSES),
            'default_action': 'deny',
            'source': value_source,
            'configured': configured,
        }
    )


def describe_enforced_policy_from_env(env: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Describe the policy mode from the environment."""

    return describe_enforced_policy(None, env=env)


def _mode_from_policy(policy: Optional[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    if isinstance(policy, Mapping) and policy:
        if 'mode' in policy:
            return normalize_enforced_policy_mode(policy.get('mode')), describe_enforced_policy(policy.get('mode'))
        if 'normalized' in policy or 'raw_value' in policy:
            raw_value = policy.get('raw_value') if policy.get('raw_value') not in (None, '') else policy.get('normalized')
            return normalize_enforced_policy_mode(policy.get('normalized') or raw_value), describe_enforced_policy(raw_value)
    description = describe_enforced_policy_from_env()
    return normalize_enforced_policy_mode(description.get('mode')), description


def _selected_scope(review: Mapping[str, Any]) -> str:
    selected = _clean_text(review.get('selected_scope'))
    if selected:
        return selected
    candidate = review.get('selected_candidate')
    if isinstance(candidate, Mapping):
        return _clean_text(candidate.get('scope'))
    return ''


def _redraw_scope_review_from_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    review = graph.get('redraw_scope_ladder_review') if isinstance(graph, Mapping) else {}
    if isinstance(review, Mapping):
        return dict(review)
    diagnostics = graph.get('developer_diagnostics') if isinstance(graph.get('developer_diagnostics'), Mapping) else {}
    review = diagnostics.get('redraw_scope_ladder_review') if isinstance(diagnostics, Mapping) else {}
    return dict(review) if isinstance(review, Mapping) else {}


def _source_evidence_refs(lifecycle: Mapping[str, Any], redraw_scope_review: Mapping[str, Any]) -> list[str]:
    refs = _clean_string_list(lifecycle.get('source_evidence_refs'))
    validation = lifecycle.get('validation_review')
    if isinstance(validation, Mapping):
        for ref in _clean_string_list(validation.get('runtime_evidence_refs')):
            _append_unique(refs, ref)
        source = validation.get('source_proposal')
        if isinstance(source, Mapping):
            for ref in _clean_string_list(source.get('evidence_refs')):
                _append_unique(refs, ref)
    candidate = redraw_scope_review.get('selected_candidate')
    if isinstance(candidate, Mapping):
        for ref in _clean_string_list(candidate.get('evidence_refs')):
            _append_unique(refs, ref)
    for ref in _clean_string_list(redraw_scope_review.get('current_evidence_refs')):
        _append_unique(refs, ref)
    return refs


def _forbidden_evidence_seen(
    lifecycle: Mapping[str, Any],
    redraw_scope_review: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> list[str]:
    seen: list[str] = []
    if bool(redraw_scope_review.get('forbidden_evidence_seen')):
        _append_unique(seen, 'redraw_scope_forbidden_evidence')
    source_values: list[str] = []
    validation = lifecycle.get('validation_review') if isinstance(lifecycle.get('validation_review'), Mapping) else {}
    source_proposal = validation.get('source_proposal') if isinstance(validation.get('source_proposal'), Mapping) else {}
    for source in (
        lifecycle.get('source'),
        validation.get('source'),
        source_proposal.get('source'),
        redraw_scope_review.get('source'),
    ):
        text = _status(source)
        if text:
            source_values.append(text)
    for ref in evidence_refs:
        text = _status(ref)
        if text:
            source_values.append(text)
    for text in source_values:
        for token in sorted(_FORBIDDEN_EVIDENCE_TOKENS):
            if token in text:
                _append_unique(seen, token)
    return seen


def _has_current_runtime_evidence(evidence_refs: Sequence[str], redraw_scope_review: Mapping[str, Any]) -> bool:
    if _status(redraw_scope_review.get('status')) == 'selected' and _selected_scope(redraw_scope_review):
        return True
    for ref in evidence_refs:
        text = _status(ref)
        if any(token in text for token in _CURRENT_RUNTIME_EVIDENCE_TOKENS):
            return True
    return False


def _requested_rebase_class(lifecycle: Mapping[str, Any]) -> str:
    for source in (
        lifecycle,
        lifecycle.get('validation_review') if isinstance(lifecycle.get('validation_review'), Mapping) else {},
    ):
        if not isinstance(source, Mapping):
            continue
        requested = _clean_text(source.get('requested_rebase_class'))
        if requested:
            return requested
        proposal = source.get('source_proposal')
        if isinstance(proposal, Mapping):
            requested = _clean_text(proposal.get('requested_rebase_class'))
            if requested:
                return requested
    return 'full_successor_rebase'


def _classify_enforced_class(lifecycle: Mapping[str, Any]) -> str:
    explicit = _clean_text(lifecycle.get('enforced_class'))
    if explicit:
        return explicit
    kind = _clean_text(lifecycle.get('kind'))
    if kind == 'ollmo.graph_rebase_lifecycle':
        requested = _requested_rebase_class(lifecycle)
        if requested == 'partial_subtree_rebase':
            return 'partial_subtree_rebase_strict'
        return 'full_successor_rebase'
    repair_class = _clean_text(lifecycle.get('repair_class'))
    if repair_class in _REPAIR_CLASS_TO_ENFORCED_CLASS:
        return _REPAIR_CLASS_TO_ENFORCED_CLASS[repair_class]
    return repair_class or 'unknown_enforced_class'


def build_enforced_lineage_summary(
    request_phase_graph: Mapping[str, Any],
    lifecycle: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Summarize existing placeholder/output-slot/work-tree identities touched by a lifecycle."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    payload = lifecycle if isinstance(lifecycle, Mapping) else {}
    target_ids: list[str] = []
    for key in ('scheduled_branch_ids', 'applied_branch_ids', 'applied_phase_ids'):
        for item in _clean_string_list(payload.get(key)):
            _append_unique(target_ids, item)
    validation = payload.get('validation_review') if isinstance(payload.get('validation_review'), Mapping) else {}
    patch = validation.get('accepted_patch') if isinstance(validation.get('accepted_patch'), Mapping) else {}
    for phase in patch.get('add_phases') or []:
        if not isinstance(phase, Mapping):
            continue
        for key in ('branch_id', 'phase_id'):
            _append_unique(target_ids, _clean_text(phase.get(key)))

    placeholder_refs: list[str] = []
    slot_ids: list[str] = []
    work_tree_node_ids: list[str] = []

    def matches(record: Mapping[str, Any]) -> bool:
        if not target_ids:
            return True
        return any(_clean_text(record.get(key)) in target_ids for key in ('branch_id', 'phase_id', 'node_id', 'slot_id'))

    for collection_key in ('phases', 'downstream_branches', 'output_obligations'):
        for record in graph.get(collection_key) or []:
            if isinstance(record, Mapping) and matches(record):
                _append_unique(placeholder_refs, _clean_text(record.get('placeholder_ref')))
                _append_unique(slot_ids, _clean_text(record.get('output_slot_id') or record.get('slot_id')))
    for slot in graph.get('output_slots') or []:
        if isinstance(slot, Mapping) and matches(slot):
            _append_unique(placeholder_refs, _clean_text(slot.get('placeholder_ref')))
            _append_unique(slot_ids, _clean_text(slot.get('slot_id') or slot.get('id')))
    ollmo_payload = graph.get('ollmo') if isinstance(graph.get('ollmo'), Mapping) else {}
    work_tree = ollmo_payload.get('work_tree') if isinstance(ollmo_payload.get('work_tree'), Mapping) else graph.get('work_tree')
    nodes = work_tree.get('nodes') if isinstance(work_tree, Mapping) else []
    for node in nodes or []:
        if isinstance(node, Mapping) and matches(node):
            _append_unique(placeholder_refs, _clean_text(node.get('placeholder_ref')))
            _append_unique(slot_ids, _clean_text(node.get('output_slot_id') or node.get('slot_id')))
            _append_unique(work_tree_node_ids, _clean_text(node.get('node_id') or node.get('id')))

    for phase in patch.get('add_phases') or []:
        if not isinstance(phase, Mapping):
            continue
        _append_unique(placeholder_refs, _clean_text(phase.get('placeholder_ref')))
        _append_unique(slot_ids, _clean_text(phase.get('output_slot_id') or phase.get('slot_id')))

    return {
        'placeholder_lineage': _compact_payload({'placeholder_refs': placeholder_refs}),
        'output_slot_lineage': _compact_payload({'slot_ids': slot_ids}),
        'work_tree_lineage': _compact_payload({'node_ids': work_tree_node_ids}),
    }


def build_enforced_policy_review(
    *,
    autonomy_level: str,
    lifecycle: Mapping[str, Any],
    request_phase_graph: Mapping[str, Any],
    redraw_scope_review: Optional[Mapping[str, Any]] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a default-deny review for apply_enforced lifecycle/application."""

    lifecycle_payload = lifecycle if isinstance(lifecycle, Mapping) else {}
    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    scope_review = (
        dict(redraw_scope_review)
        if isinstance(redraw_scope_review, Mapping) and redraw_scope_review
        else _redraw_scope_review_from_graph(graph)
    )
    mode, policy_description = _mode_from_policy(policy)
    enforced_class = _classify_enforced_class(lifecycle_payload)
    selected_scope = _selected_scope(scope_review)
    evidence_refs = _source_evidence_refs(lifecycle_payload, scope_review)
    forbidden_seen = _forbidden_evidence_seen(lifecycle_payload, scope_review, evidence_refs)
    blocked_reasons: list[str] = []
    level = _status(autonomy_level)

    if level != 'apply_enforced':
        _append_unique(blocked_reasons, 'apply_enforced_autonomy_required')
    if mode == 'off':
        _append_unique(blocked_reasons, 'enforced_policy_off')
    elif mode == 'audit':
        _append_unique(blocked_reasons, 'enforced_policy_audit_only')
    elif mode == 'custom':
        _append_unique(blocked_reasons, 'enforced_policy_custom_not_implemented')

    invalid_value = _clean_text(policy_description.get('invalid_value'))
    if invalid_value:
        _append_unique(blocked_reasons, 'invalid_enforced_policy_mode')

    if enforced_class == 'full_successor_rebase':
        _append_unique(blocked_reasons, 'full_successor_rebase_not_enforced_v1')
    elif enforced_class == 'partial_subtree_rebase_strict':
        _append_unique(blocked_reasons, 'partial_subtree_rebase_enforced_v1_audit_only')
    elif enforced_class not in _ALLOWED_SAFE_V1_CLASSES:
        _append_unique(blocked_reasons, 'apply_enforced_class_not_allowed')
    elif enforced_class.startswith('safe_additive_') and _status(lifecycle_payload.get('risk_level')) != 'safe_additive':
        _append_unique(blocked_reasons, 'safe_additive_risk_level_required')

    if forbidden_seen:
        _append_unique(blocked_reasons, 'forbidden_evidence_not_enforced_authority')

    if not _clean_text(lifecycle_payload.get('idempotency_key')):
        _append_unique(blocked_reasons, 'idempotency_key_required')

    if not _has_current_runtime_evidence(evidence_refs, scope_review):
        _append_unique(blocked_reasons, 'current_runtime_evidence_required')

    if enforced_class == 'duplicate_artifact_alias_canonicalization':
        artifact_identity = (
            scope_review.get('artifact_identity')
            if isinstance(scope_review.get('artifact_identity'), Mapping)
            else {}
        )
        if bool(artifact_identity.get('final_projection_blocked')) or _clean_string_list(artifact_identity.get('conflicting_refs')):
            _append_unique(blocked_reasons, 'conflicting_duplicate_artifact_ref_not_enforced')
        if not _clean_string_list(artifact_identity.get('duplicate_refs')):
            _append_unique(blocked_reasons, 'duplicate_artifact_identity_evidence_required')

    allowed_scopes = _ENFORCED_CLASS_SCOPES.get(enforced_class, set())
    if enforced_class in _ALLOWED_SAFE_V1_CLASSES:
        if not selected_scope:
            _append_unique(blocked_reasons, 'redraw_scope_review_required')
        elif selected_scope not in allowed_scopes:
            _append_unique(blocked_reasons, 'redraw_scope_not_smallest_allowed_for_enforced_class')

    allowed = mode == 'safe_v1' and not blocked_reasons and enforced_class in _ALLOWED_SAFE_V1_CLASSES
    status = 'allowed' if allowed else 'blocked'
    lineage = build_enforced_lineage_summary(graph, lifecycle_payload)
    review = {
        'kind': ENFORCED_POLICY_REVIEW_KIND,
        'review_id': _stable_digest(
            {
                'policy_id': policy_description.get('policy_id') or ENFORCED_POLICY_DEFAULT_ID,
                'mode': mode,
                'enforced_class': enforced_class,
                'lifecycle_id': lifecycle_payload.get('patch_id') or lifecycle_payload.get('rebase_id'),
                'idempotency_key': lifecycle_payload.get('idempotency_key'),
                'scope_review_id': scope_review.get('review_id'),
                'blocked_reasons': blocked_reasons,
            },
            prefix='enforced-policy-review-',
        ),
        'policy_id': policy_description.get('policy_id') or ENFORCED_POLICY_DEFAULT_ID,
        'status': status,
        'allowed': allowed,
        'authority': 'runtime_enforced_policy' if allowed else 'runtime_enforced_policy_denied',
        'policy_mode': mode,
        'mode': mode,
        'enforced_class': enforced_class,
        'allowed_classes': sorted(_ALLOWED_SAFE_V1_CLASSES),
        'blocked_reasons': blocked_reasons,
        'current_evidence_refs': evidence_refs,
        'forbidden_evidence_seen': forbidden_seen,
        'redraw_scope_review_ref': scope_review.get('review_id'),
        'selected_scope': selected_scope,
        'selected_scope_allowed': selected_scope in allowed_scopes if allowed_scopes else False,
        'policy': policy_description,
        **lineage,
    }
    return _json_safe(review)


def enforced_policy_allows_application(review: Mapping[str, Any]) -> bool:
    """Return true only when a policy review explicitly allows application."""

    return (
        isinstance(review, Mapping)
        and review.get('kind') == ENFORCED_POLICY_REVIEW_KIND
        and bool(review.get('allowed'))
        and _status(review.get('status')) == 'allowed'
        and _status(review.get('authority')) == 'runtime_enforced_policy'
    )
