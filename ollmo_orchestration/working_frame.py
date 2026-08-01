"""Mutable working-frame helpers for Ollmo orchestration state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ollmo_g.context_ir import build_context_ir
from ollmo_g.request_ir import (
    output_candidates_from_graph,
    output_obligations_from_graph,
    promotion_records_from_graph,
)
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_services.artifact_dossiers import build_artifact_dossier_index
from ollmo_services.control_snapshots import build_control_snapshot
from ollmo_services.frame_planning import build_artifact_flow_plan

WORKING_FRAME_VERSION = 4
_WAIVED_CONTRACT_STATUSES = {
    'waived',
    'not_needed',
    'not-needed',
    'not_needed_verified',
    'skipped_verified',
    'unnecessary_verified',
}
_SUPERSEDED_CONTRACT_STATUSES = {
    'obsolete',
    'replaced',
    'superseded',
    'no-longer-relevant',
    'no_longer_relevant',
}
_WORKING_FRAME_HEAVY_TOP_LEVEL_KEYS = (
    'request_phase_graph',
    'intent_contract',
    'context_contract',
    'work_tree',
    'artifact_dossiers',
    'artifact_flow',
)
_WORKING_FRAME_REQUEST_HEAVY_KEYS = {
    'prompt',
    'input',
    'instructions',
    'context_candidates',
    'memory_candidates',
    'history_candidates',
    'reference_candidates',
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _is_empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or '').strip()
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


def _json_digest(value: Any) -> tuple[str, int]:
    encoded = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _content_ref(value: Any, *, json_path: str, snapshot=None) -> dict[str, Any]:
    if callable(snapshot):
        return _json_safe(snapshot(value, json_path))
    digest, size_bytes = _json_digest(value)
    return {
        'kind': 'ollmo.working_frame_content_ref',
        'json_path': json_path,
        'sha256': digest,
        'size_bytes': size_bytes,
        'content_addressed': True,
    }


def _contract_logic_summary(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        'kind',
        'source',
        'status',
        'reason',
        'obligation_count',
        'candidate_count',
        'promotion_count',
        'counts',
        'pending_obligation_ids',
        'deferred_obligation_ids',
        'blocked_obligation_ids',
        'superseded_obligation_ids',
        'waived_obligation_ids',
        'fulfilled_obligation_ids',
        'promoted_candidate_ids',
        'active_context_candidate_ids',
        'active_reference_candidate_ids',
        'active_memory_candidate_ids',
        'active_reference_artifact_refs',
        'promoted_history_scan_candidate_ids',
    ):
        value = contract.get(key)
        if value not in (None, '', [], {}):
            summary[key] = _json_safe(value)
    return summary


def _artifact_flow_logic_summary(artifact_flow: Any) -> dict[str, Any]:
    if not isinstance(artifact_flow, Mapping):
        return {}
    review = artifact_flow.get('review') if isinstance(artifact_flow.get('review'), Mapping) else {}
    output_slots = artifact_flow.get('output_slots') if isinstance(artifact_flow.get('output_slots'), list) else []
    summary: dict[str, Any] = {
        'kind': artifact_flow.get('kind'),
        'status': artifact_flow.get('status'),
        'work_tree_source': artifact_flow.get('work_tree_source'),
        'authoritative': artifact_flow.get('authoritative')
        if isinstance(artifact_flow.get('authoritative'), bool)
        else None,
        'compatibility_derived': artifact_flow.get('compatibility_derived')
        if isinstance(artifact_flow.get('compatibility_derived'), bool)
        else None,
        'output_slot_count': len(output_slots),
    }
    for key in (
        'pending_output_slot_ids',
        'deferred_output_slot_ids',
        'blocked_output_slot_ids',
        'waived_output_slot_ids',
        'superseded_output_slot_ids',
        'cancelled_output_slot_ids',
    ):
        value = review.get(key)
        if isinstance(value, list) and value:
            summary[key] = _json_safe(value)
    return {
        key: value
        for key, value in summary.items()
        if value not in (None, '', [], {})
    }


def _compact_request_logic(request: Any, *, snapshot=None, base_json_path: str = 'working_frame.request') -> dict[str, Any]:
    if not isinstance(request, Mapping):
        return {}
    compact: dict[str, Any] = {}
    heavy_payload: dict[str, Any] = {}
    for key, value in request.items():
        if value in (None, '', [], {}):
            continue
        if key in _WORKING_FRAME_REQUEST_HEAVY_KEYS:
            heavy_payload[key] = value
            continue
        compact[str(key)] = _json_safe(value)
    if heavy_payload:
        compact['content_snapshot_ref'] = _content_ref(heavy_payload, json_path=f'{base_json_path}.content', snapshot=snapshot)
    return _json_safe(compact)


def _compact_goal_stack(goals: Any) -> list[dict[str, Any]]:
    if not isinstance(goals, list):
        return []
    compact_goals: list[dict[str, Any]] = []
    for item in goals:
        if not isinstance(item, Mapping):
            continue
        compact: dict[str, Any] = {}
        for key in (
            'goal_id',
            'kind',
            'status',
            'slot_id',
            'obligation_id',
            'artifact_ref',
            'editable',
        ):
            value = item.get(key)
            if value not in (None, '', [], {}):
                compact[key] = _json_safe(value)
        if compact:
            compact_goals.append(compact)
    return compact_goals


def _compact_route_logic(route: Any, *, snapshot=None, base_json_path: str = 'working_frame.route') -> dict[str, Any]:
    if not isinstance(route, Mapping):
        return {}
    compact = dict(route)
    context_strategy = compact.get('context_strategy') if isinstance(compact.get('context_strategy'), Mapping) else None
    if context_strategy:
        summary: dict[str, Any] = {}
        for key in ('kind', 'mode', 'reason', 'context_scope', 'selected_context_scope'):
            value = context_strategy.get(key)
            if value not in (None, '', [], {}):
                summary[key] = value
        summary['full_snapshot_ref'] = _content_ref(
            context_strategy,
            json_path=f'{base_json_path}.context_strategy',
            snapshot=snapshot,
        )
        compact['context_strategy'] = _json_safe(summary)
    return _json_safe(compact)


def compact_working_frame_for_serialization(
    working_frame: Mapping[str, Any],
    *,
    snapshot=None,
    base_json_path: str = 'working_frame',
) -> dict[str, Any]:
    """Return a logic-only working-frame payload with heavy state moved behind refs."""

    payload = _json_safe(working_frame)
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get('request'), Mapping):
        payload['request'] = _compact_request_logic(
            payload.get('request'),
            snapshot=snapshot,
            base_json_path=f'{base_json_path}.request',
        )
    if isinstance(payload.get('route'), Mapping):
        payload['route'] = _compact_route_logic(
            payload.get('route'),
            snapshot=snapshot,
            base_json_path=f'{base_json_path}.route',
        )
    if isinstance(payload.get('goal_stack'), list):
        payload['goal_stack'] = _compact_goal_stack(payload.get('goal_stack'))
    for key in _WORKING_FRAME_HEAVY_TOP_LEVEL_KEYS:
        value = payload.get(key)
        if value in (None, '', [], {}):
            continue
        if key in {'intent_contract', 'context_contract'}:
            summary = _contract_logic_summary(value)
            if summary:
                payload[f'{key}_summary'] = summary
        elif key == 'artifact_flow':
            summary = _artifact_flow_logic_summary(value)
            if summary:
                payload[f'{key}_summary'] = summary
        payload[f'{key}_snapshot_ref'] = _content_ref(
            value,
            json_path=f'{base_json_path}.{key}',
            snapshot=snapshot,
        )
        payload.pop(key, None)
    return _json_safe(payload)


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        token = _clean_text(value)
        if token:
            return token
    return None


def _unique_texts(values: list[Any], *, limit: int = 12) -> list[str]:
    items: list[str] = []
    for value in values:
        token = _clean_text(value)
        if not token or token in items:
            continue
        items.append(token)
        if len(items) >= limit:
            break
    return items


def _input_artifacts(request: Mapping[str, Any], response: Mapping[str, Any]) -> list[dict[str, Any]]:
    for source in (response, request):
        items = source.get('input_artifacts')
        if isinstance(items, list) and items:
            return [dict(item) for item in items if isinstance(item, Mapping)]
    return []


def _output_artifacts(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = response.get('artifacts')
    if isinstance(items, list) and items:
        return [dict(item) for item in items if isinstance(item, Mapping)]
    return []


def _reference_artifacts(request: Mapping[str, Any], response: Mapping[str, Any]) -> list[dict[str, Any]]:
    for source in (response, request):
        for key in ('reference_artifacts', 'selected_reference_artifacts'):
            items = source.get(key)
            if isinstance(items, list) and items:
                return [dict(item) for item in items if isinstance(item, Mapping)]
    return []


def _planning_response_payload(
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(response)
    if 'capability' not in payload or _is_empty(payload.get('capability')):
        payload['capability'] = route.get('capability') or request.get('capability')
    if 'mode' not in payload or _is_empty(payload.get('mode')):
        payload['mode'] = route.get('capability') or request.get('mode') or request.get('capability')
    if 'instance_id' not in payload or _is_empty(payload.get('instance_id')):
        payload['instance_id'] = route.get('instance_id')
    if 'model' not in payload or _is_empty(payload.get('model')):
        instance = route.get('instance') if isinstance(route.get('instance'), Mapping) else {}
        payload['model'] = instance.get('model')
    if 'backend' not in payload or _is_empty(payload.get('backend')):
        instance = route.get('instance') if isinstance(route.get('instance'), Mapping) else {}
        payload['backend'] = instance.get('backend')
    return payload


def _target_payload(route: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    instance = route.get('instance') if isinstance(route.get('instance'), Mapping) else {}
    return {
        'instance_id': _first_text(route.get('instance_id'), response.get('instance_id'), instance.get('instance_id')),
        'model': _first_text(response.get('model'), instance.get('model')),
        'backend': _first_text(response.get('backend'), instance.get('backend')),
        'capability': _first_text(route.get('capability'), response.get('capability')),
        'mode': _first_text(response.get('mode'), route.get('capability')),
    }


def _route_summary(route: Mapping[str, Any], response: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        'source': _first_text(route.get('route_source'), response.get('route_source')),
        'reason': _first_text(route.get('route_reason'), response.get('route_reason')),
        'confidence': route.get('route_confidence') if route.get('route_confidence') not in (None, '') else response.get('route_confidence'),
        'reuse_last_artifact': bool(
            route.get('route_reuse_last_artifact')
            if route.get('route_reuse_last_artifact') is not None
            else response.get('route_reuse_last_artifact')
        ),
        'artifact_path': _first_text(route.get('route_artifact_path'), response.get('route_artifact_path')),
        'router_instance_id': _first_text(route.get('route_router_instance_id'), response.get('route_router_instance_id')),
        'router_model': _first_text(route.get('route_router_model'), response.get('route_router_model')),
    }
    context_strategy = runtime.get('context_strategy') if isinstance(runtime.get('context_strategy'), Mapping) else {}
    if context_strategy:
        payload['context_strategy'] = dict(context_strategy)
    return payload


def _request_summary(request: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        'request_id': _first_text(request.get('request_id')),
        'conversation_id': _first_text(request.get('conversation_id')),
        'prompt': _first_text(request.get('prompt'), request.get('input'), request.get('instructions')),
        'ghost_route': bool(request.get('ghost_route')),
        'ghost_mode': _first_text(
            (request.get('request_meta') or {}).get('ghost_mode') if isinstance(request.get('request_meta'), Mapping) else None,
            request.get('ghost_mode'),
        ),
        'capability_hint': _first_text(request.get('capability'), target.get('capability')),
    }
    return summary


def _request_phase_graph(
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    existing = runtime.get('request_phase_graph') if isinstance(runtime.get('request_phase_graph'), Mapping) else {}
    if existing:
        return dict(existing)
    prompt = _first_text(
        request.get('prompt'),
        request.get('input'),
        request.get('instructions'),
        response.get('output_text'),
    ) or ''
    return build_request_phase_graph(
        prompt,
        request_payload=request,
        route_payload=route,
        response_payload=response,
    )


def _loop_state(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    freeze: bool,
) -> dict[str, Any]:
    semantic_role_profile = (
        runtime.get('semantic_role_profile')
        if isinstance(runtime.get('semantic_role_profile'), Mapping)
        else {}
    )
    loop_template = (
        semantic_role_profile.get('loop')
        if isinstance(semantic_role_profile.get('loop'), Mapping)
        else {}
    )
    previous_working_frame = runtime.get('working_frame') if isinstance(runtime.get('working_frame'), Mapping) else {}
    previous_loop = previous_working_frame.get('loop') if isinstance(previous_working_frame.get('loop'), Mapping) else {}
    retry_failure = runtime.get('retry_failure') if isinstance(runtime.get('retry_failure'), Mapping) else {}
    self_heal_attempted = bool(request.get('ghost_self_heal_attempted')) or bool(retry_failure)
    pass_index = int(previous_loop.get('pass_index') or 0)
    if pass_index <= 0:
        pass_index = 2 if self_heal_attempted else 1
    max_passes = max(1, int(loop_template.get('max_passes') or previous_loop.get('max_passes') or 1))
    critic_passes = max(0, int(loop_template.get('critic_passes') or previous_loop.get('critic_passes') or 0))
    chain_id = _first_text(
        previous_loop.get('chain_id'),
        request.get('conversation_id'),
        request.get('request_id'),
        response.get('id'),
    ) or 'working-frame'
    return {
        'chain_id': chain_id,
        'pass_index': pass_index,
        'max_passes': max_passes,
        'critic_passes': critic_passes,
        'remaining_passes': max(0, max_passes - pass_index),
        'can_continue': bool(not freeze and pass_index < max_passes),
        'self_heal_attempted': self_heal_attempted,
        'self_heal_active': bool(retry_failure),
    }


def _goal_status_from_output_slot(slot: Mapping[str, Any]) -> str:
    status = _clean_text(slot.get('status')).lower()
    if status in {'fulfilled', 'superseded', 'waived'}:
        return 'completed'
    if status in {'blocked', 'failed'}:
        return 'blocked'
    return 'pending'


def _contract_check_status(status: Any) -> str:
    token = _clean_text(status).lower()
    if token in {'completed', 'fulfilled'}:
        return 'fulfilled'
    if token in {'blocked', 'failed', 'error'}:
        return 'blocked'
    if token in _WAIVED_CONTRACT_STATUSES:
        return 'waived'
    if token in _SUPERSEDED_CONTRACT_STATUSES:
        return 'superseded'
    if token in {'deferred'}:
        return 'deferred'
    if token in {'planned', 'active', 'running', 'queued', 'scheduled', 'accepted'}:
        return 'pending'
    return token or 'pending'


def _contract_counts(checks: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        'fulfilled': sum(1 for item in checks if _contract_check_status(item.get('status')) == 'fulfilled'),
        'pending': sum(1 for item in checks if _contract_check_status(item.get('status')) == 'pending'),
        'deferred': sum(1 for item in checks if _contract_check_status(item.get('status')) == 'deferred'),
        'blocked': sum(1 for item in checks if _contract_check_status(item.get('status')) == 'blocked'),
        'superseded': sum(1 for item in checks if _contract_check_status(item.get('status')) == 'superseded'),
        'waived': sum(1 for item in checks if _contract_check_status(item.get('status')) == 'waived'),
    }


def _contract_status_from_counts(counts: Mapping[str, Any], *, has_checks: bool) -> str:
    if int(counts.get('blocked') or 0) > 0:
        return 'blocked'
    if int(counts.get('pending') or 0) > 0 or int(counts.get('deferred') or 0) > 0:
        return 'pending'
    if has_checks:
        return 'fulfilled'
    return 'not_applicable'


def _contract_check_from_obligation(obligation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'obligation_id': _first_text(obligation.get('obligation_id')),
        'phase_id': _first_text(obligation.get('phase_id')),
        'branch_id': _first_text(obligation.get('branch_id')),
        'capability': _first_text(obligation.get('capability')),
        'output_type': _first_text(obligation.get('output_type')),
        'role': _first_text(obligation.get('role')),
        'status': _contract_check_status(obligation.get('status')),
        'evidence': _first_text(obligation.get('evidence')) or 'output_obligation_status',
    }


def _contract_check_from_review(check: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(check)
    payload['status'] = _contract_check_status(payload.get('status'))
    if not _first_text(payload.get('evidence')):
        payload['evidence'] = 'graph_closure_review'
    return payload


def _ids_for_contract_status(checks: list[Mapping[str, Any]], status: str) -> list[str]:
    wanted = _contract_check_status(status)
    values: list[str] = []
    for check in checks:
        if _contract_check_status(check.get('status')) != wanted:
            continue
        obligation_id = _first_text(check.get('obligation_id'))
        if obligation_id and obligation_id not in values:
            values.append(obligation_id)
    return values


def _intent_contract(
    request_phase_graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    review = runtime.get('graph_closure_review') if isinstance(runtime.get('graph_closure_review'), Mapping) else {}
    request_ir = (
        request_phase_graph.get('request_ir')
        if isinstance(request_phase_graph.get('request_ir'), Mapping)
        else {}
    )
    candidate_graph = (
        request_ir.get('candidate_graph')
        if isinstance(request_ir.get('candidate_graph'), Mapping)
        else (
            request_phase_graph.get('candidate_graph')
            if isinstance(request_phase_graph.get('candidate_graph'), Mapping)
            else {}
        )
    )
    promotion_review = (
        request_ir.get('promotion_review')
        if isinstance(request_ir.get('promotion_review'), Mapping)
        else (
            request_phase_graph.get('promotion_review')
            if isinstance(request_phase_graph.get('promotion_review'), Mapping)
            else {}
        )
    )
    obligations = output_obligations_from_graph(request_phase_graph)
    candidates = output_candidates_from_graph(request_phase_graph)
    promotions = promotion_records_from_graph(request_phase_graph)
    review_checks = review.get('checks') if isinstance(review.get('checks'), list) else []
    if review_checks:
        checks = [
            _contract_check_from_review(item)
            for item in review_checks
            if isinstance(item, Mapping)
        ]
    else:
        checks = [
            _contract_check_from_obligation(item)
            for item in obligations
            if isinstance(item, Mapping)
        ]
    checks = [_json_safe(item) for item in checks if isinstance(item, Mapping)]
    if not checks and not obligations and not candidates and not promotions and not review:
        return {}
    counts_source = review.get('counts') if isinstance(review.get('counts'), Mapping) else {}
    counts = {
        key: int(counts_source.get(key) or fallback)
        for key, fallback in _contract_counts(checks).items()
    }
    status = _contract_check_status(review.get('status')) if review.get('status') else _contract_status_from_counts(
        counts,
        has_checks=bool(checks),
    )
    if status == 'deferred':
        status = 'pending'
    contract_source = _first_text(review.get('contract_source'))
    if not contract_source and obligations:
        contract_source = 'request_ir.output_obligations'
    payload = {
        'kind': 'ollmo.intent_contract',
        'source': contract_source or 'request_phase_graph',
        'status': status,
        'reason': _first_text(review.get('reason')),
        'obligation_count': int(review.get('obligation_count') or len(obligations) or len(checks)),
        'candidate_count': len(candidates),
        'promotion_count': len(promotions),
        'counts': counts,
        'pending_obligation_ids': _ids_for_contract_status(checks, 'pending'),
        'deferred_obligation_ids': _ids_for_contract_status(checks, 'deferred'),
        'blocked_obligation_ids': _ids_for_contract_status(checks, 'blocked'),
        'superseded_obligation_ids': _ids_for_contract_status(checks, 'superseded'),
        'waived_obligation_ids': _ids_for_contract_status(checks, 'waived'),
        'fulfilled_obligation_ids': _ids_for_contract_status(checks, 'fulfilled'),
        'checks': checks,
    }
    if candidates:
        payload['output_candidates'] = _json_safe(candidates)
        payload['candidate_output_ids'] = _unique_texts([item.get('candidate_id') for item in candidates if isinstance(item, Mapping)])
    if promotions:
        payload['promotions'] = _json_safe(promotions)
    if candidate_graph:
        payload['candidate_graph'] = _json_safe(candidate_graph)
        payload['general_candidate_count'] = int(candidate_graph.get('candidate_count') or 0)
        payload['general_candidate_status_counts'] = (
            _json_safe(candidate_graph.get('status_counts'))
            if isinstance(candidate_graph.get('status_counts'), Mapping)
            else {}
        )
        payload['general_candidate_type_counts'] = (
            _json_safe(candidate_graph.get('type_counts'))
            if isinstance(candidate_graph.get('type_counts'), Mapping)
            else {}
        )
    if promotion_review:
        payload['promotion_review'] = _json_safe(promotion_review)
        payload['general_promoted_count'] = int(promotion_review.get('promoted_count') or 0)
        payload['general_reserved_count'] = int(promotion_review.get('reserved_count') or 0)
        payload['general_omitted_count'] = int(promotion_review.get('omitted_count') or 0)
        payload['general_superseded_count'] = int(promotion_review.get('superseded_count') or 0)
        payload['general_waived_count'] = int(promotion_review.get('waived_count') or 0)
        payload['general_rejected_count'] = int(promotion_review.get('rejected_count') or 0)
    return _json_safe(payload)


def _context_contract(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    context_ir = build_context_ir(
        request_payload=request,
        response_payload=response,
        runtime_payload=runtime,
    )
    if not context_ir:
        return {}
    candidates = context_ir.get('context_candidates') if isinstance(context_ir.get('context_candidates'), list) else []
    promotions = context_ir.get('promotions') if isinstance(context_ir.get('promotions'), list) else []
    promoted_candidate_ids = [
        token
        for token in _unique_texts([item.get('candidate_id') for item in promotions if isinstance(item, Mapping)])
        if token
    ]
    payload = {
        'kind': 'ollmo.context_contract',
        'source': 'context_ir.context_candidates',
        'status': 'active' if promotions else 'candidate_only',
        'candidate_count': len(candidates),
        'promotion_count': len(promotions),
        'context_candidates': _json_safe(candidates),
        'promotions': _json_safe(promotions),
        'promoted_candidate_ids': promoted_candidate_ids,
        'active_context_candidate_ids': [
            item.get('candidate_id')
            for item in promotions
            if isinstance(item, Mapping) and item.get('target') == 'active_context'
        ],
        'active_reference_candidate_ids': [
            item.get('candidate_id')
            for item in promotions
            if isinstance(item, Mapping) and item.get('target') == 'active_reference'
        ],
        'active_memory_candidate_ids': [
            item.get('candidate_id')
            for item in promotions
            if isinstance(item, Mapping) and item.get('target') == 'active_memory'
        ],
        'active_reference_artifact_refs': context_ir.get('active_reference_artifact_refs'),
        'active_memory_ids': context_ir.get('active_memory_ids'),
        'promoted_history_scan_candidate_ids': context_ir.get('promoted_history_scan_candidate_ids'),
    }
    context_gate_review = context_ir.get('context_gate_review') if isinstance(context_ir.get('context_gate_review'), Mapping) else {}
    context_gate_reviews = context_ir.get('context_gate_reviews') if isinstance(context_ir.get('context_gate_reviews'), list) else []
    if context_gate_review:
        payload['context_gate_review'] = context_gate_review
    if context_gate_reviews:
        payload['context_gate_reviews'] = context_gate_reviews
    candidate_graph = context_ir.get('candidate_graph') if isinstance(context_ir.get('candidate_graph'), Mapping) else {}
    promotion_review = context_ir.get('promotion_review') if isinstance(context_ir.get('promotion_review'), Mapping) else {}
    if candidate_graph:
        payload['candidate_graph'] = candidate_graph
        payload['general_candidate_count'] = int(candidate_graph.get('candidate_count') or 0)
    if promotion_review:
        payload['promotion_review'] = promotion_review
        payload['general_promoted_count'] = int(promotion_review.get('promoted_count') or 0)
        payload['general_reserved_count'] = int(promotion_review.get('reserved_count') or 0)
        payload['general_omitted_count'] = int(promotion_review.get('omitted_count') or 0)
        payload['general_superseded_count'] = int(promotion_review.get('superseded_count') or 0)
        payload['general_waived_count'] = int(promotion_review.get('waived_count') or 0)
        payload['general_rejected_count'] = int(promotion_review.get('rejected_count') or 0)
    return _json_safe(payload)


def _working_status(
    response: Mapping[str, Any],
    artifact_flow: Mapping[str, Any],
    loop: Mapping[str, Any],
    *,
    freeze: bool,
) -> str:
    response_status = _clean_text(response.get('status')).lower()
    output_slots = artifact_flow.get('output_slots') if isinstance(artifact_flow.get('output_slots'), list) else []
    blocked_output = any(
        isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() in {'blocked', 'failed'}
        for slot in output_slots
    )
    pending_output = any(
        isinstance(slot, Mapping) and _clean_text(slot.get('status')).lower() not in {'', 'fulfilled', 'superseded', 'waived'}
        for slot in output_slots
    )
    if freeze and response_status not in {'failed', 'error', 'cancelled'}:
        return 'frozen'
    if response_status in {'failed', 'error', 'cancelled'}:
        return 'blocked'
    if bool(loop.get('self_heal_active')):
        return 'repairing'
    if blocked_output:
        return 'blocked'
    if pending_output:
        return 'active'
    if response_status == 'completed':
        return 'completed'
    if any(response.get(key) not in (None, '', [], {}) for key in ('output_text', 'artifacts', 'results')):
        return 'completed'
    return 'active'


def _goal_stack(
    request: Mapping[str, Any],
    target: Mapping[str, Any],
    artifact_flow: Mapping[str, Any],
    *,
    working_status: str,
    freeze: bool,
) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    prompt = _first_text(request.get('prompt'), request.get('input'), request.get('instructions'))
    target_capability = _first_text(target.get('capability'), target.get('mode')) or 'response'
    root_status = 'blocked' if working_status == 'blocked' else 'completed' if working_status in {'completed', 'frozen'} else 'active'
    goals.append(
        {
            'goal_id': 'goal-request',
            'kind': 'respond',
            'status': root_status,
            'summary': prompt or f'complete {target_capability} request',
            'editable': not freeze,
        }
    )
    input_routes = artifact_flow.get('input_routes') if isinstance(artifact_flow.get('input_routes'), list) else []
    for index, route in enumerate(input_routes, start=1):
        routing_hint = route.get('routing_hint') if isinstance(route.get('routing_hint'), Mapping) else {}
        route_capability = _first_text(routing_hint.get('capability')) or target_capability
        goals.append(
            {
                'goal_id': f'goal-input-{index}',
                'kind': 'integrate_input_artifact',
                'status': 'completed' if working_status in {'completed', 'frozen'} else 'planned',
                'artifact_ref': _first_text(route.get('artifact_ref')),
                'summary': f'route input artifact through {route_capability}',
                'editable': not freeze,
            }
        )
    reference_routes = artifact_flow.get('reference_routes') if isinstance(artifact_flow.get('reference_routes'), list) else []
    for index, route in enumerate(reference_routes, start=1):
        reference_type = _first_text(route.get('artifact_type')) or 'artifact'
        goals.append(
            {
                'goal_id': f'goal-reference-{index}',
                'kind': 'carry_reference_artifact',
                'status': 'completed' if working_status in {'completed', 'frozen'} else 'planned',
                'artifact_ref': _first_text(route.get('artifact_ref')),
                'summary': f'carry {reference_type} reference forward without rematerializing it',
                'editable': not freeze,
            }
        )
    output_slots = artifact_flow.get('output_slots') if isinstance(artifact_flow.get('output_slots'), list) else []
    for index, slot in enumerate(output_slots, start=1):
        goals.append(
            {
                'goal_id': f'goal-output-{index}',
                'kind': 'materialize_output',
                'status': _goal_status_from_output_slot(slot),
                'slot_id': _first_text(slot.get('slot_id')),
                'obligation_id': _first_text(slot.get('obligation_id')),
                'summary': f"materialize {(_first_text(slot.get('type')) or 'artifact')} output",
                'editable': not freeze,
            }
        )
    review_status = 'completed' if freeze else 'blocked' if working_status == 'blocked' else 'pending'
    goals.append(
        {
            'goal_id': 'goal-review',
            'kind': 'review_and_freeze',
            'status': review_status,
            'summary': 'review the current working state and freeze it into the response frame',
            'editable': not freeze,
        }
    )
    return goals


def _journal(
    runtime: Mapping[str, Any],
    route_summary: Mapping[str, Any],
    working_status: str,
    *,
    freeze: bool,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = [
        {
            'entry_id': 'orient-1',
            'phase': 'orient',
            'status': 'completed',
            'summary': 'Read runtime truth, request hints, and recent local context.',
        }
    ]
    route_reason = _first_text(route_summary.get('reason')) or 'selected a bounded route'
    entries.append(
        {
            'entry_id': 'plan-1',
            'phase': 'plan',
            'status': 'completed' if _first_text(route_summary.get('source')) else 'pending',
            'summary': route_reason,
        }
    )
    planner_meta = runtime.get('execution_planner') if isinstance(runtime.get('execution_planner'), Mapping) else {}
    if planner_meta.get('attempted'):
        planner_status = 'completed' if planner_meta.get('applied') else 'observed'
        entries.append(
            {
                'entry_id': 'revise-planner-1',
                'phase': 'revise',
                'status': planner_status,
                'summary': _first_text(planner_meta.get('reason')) or 'resolver revised the route payload',
            }
        )
    control_hints = runtime.get('control_hints') if isinstance(runtime.get('control_hints'), Mapping) else {}
    if control_hints:
        entries.append(
            {
                'entry_id': 'revise-controls-1',
                'phase': 'revise',
                'status': 'completed',
                'summary': 'Applied control-hint refinements for the selected route.',
                'fields': sorted(str(key) for key in control_hints.keys()),
            }
        )
    retry_failure = runtime.get('retry_failure') if isinstance(runtime.get('retry_failure'), Mapping) else {}
    if retry_failure:
        entries.append(
            {
                'entry_id': 'critic-self-heal-1',
                'phase': 'critic',
                'status': 'completed',
                'summary': (
                    _first_text(retry_failure.get('error_message'))
                    or 'entered a bounded self-heal pass after a previous failure'
                ),
            }
        )
    entries.append(
        {
            'entry_id': 'act-1',
            'phase': 'act',
            'status': 'completed' if working_status in {'completed', 'frozen'} else 'blocked' if working_status == 'blocked' else 'in_progress',
            'summary': 'Materialize the selected route into outputs and artifacts.',
        }
    )
    entries.append(
        {
            'entry_id': 'freeze-1',
            'phase': 'freeze',
            'status': 'completed' if freeze else 'pending',
            'summary': 'Freeze the mutable working state into an immutable response frame.',
        }
    )
    return entries


def _review(
    artifact_flow: Mapping[str, Any],
    goals: list[dict[str, Any]],
    intent_contract: Mapping[str, Any],
    context_contract: Mapping[str, Any],
    *,
    freeze: bool,
    working_status: str,
) -> dict[str, Any]:
    flow_review = artifact_flow.get('review') if isinstance(artifact_flow.get('review'), Mapping) else {}
    pending_goal_ids = [
        str(item.get('goal_id') or '').strip()
        for item in goals
        if str(item.get('status') or '').strip().lower() in {'pending', 'planned', 'active'}
    ]
    pending_output_slot_ids = flow_review.get('pending_output_slot_ids') if isinstance(flow_review.get('pending_output_slot_ids'), list) else []
    deferred_output_slot_ids = flow_review.get('deferred_output_slot_ids') if isinstance(flow_review.get('deferred_output_slot_ids'), list) else []
    blocked_output_slot_ids = flow_review.get('blocked_output_slot_ids') if isinstance(flow_review.get('blocked_output_slot_ids'), list) else []
    superseded_output_slot_ids = flow_review.get('superseded_output_slot_ids') if isinstance(flow_review.get('superseded_output_slot_ids'), list) else []
    waived_output_slot_ids = flow_review.get('waived_output_slot_ids') if isinstance(flow_review.get('waived_output_slot_ids'), list) else []
    pending_obligation_ids = intent_contract.get('pending_obligation_ids') if isinstance(intent_contract.get('pending_obligation_ids'), list) else []
    deferred_obligation_ids = intent_contract.get('deferred_obligation_ids') if isinstance(intent_contract.get('deferred_obligation_ids'), list) else []
    blocked_obligation_ids = intent_contract.get('blocked_obligation_ids') if isinstance(intent_contract.get('blocked_obligation_ids'), list) else []
    superseded_obligation_ids = intent_contract.get('superseded_obligation_ids') if isinstance(intent_contract.get('superseded_obligation_ids'), list) else []
    waived_obligation_ids = intent_contract.get('waived_obligation_ids') if isinstance(intent_contract.get('waived_obligation_ids'), list) else []
    open_obligation_ids = [*pending_obligation_ids, *deferred_obligation_ids, *blocked_obligation_ids]
    open_output_slot_ids = [*pending_output_slot_ids, *deferred_output_slot_ids, *blocked_output_slot_ids]
    if freeze:
        if working_status == 'blocked':
            status = 'blocked'
        elif open_obligation_ids or open_output_slot_ids:
            status = 'partial_frozen'
        else:
            status = 'frozen'
    elif working_status == 'blocked':
        status = 'blocked'
    else:
        status = _first_text(flow_review.get('status')) or 'pending'
    freeze_ready = bool(
        not open_obligation_ids
        and not open_output_slot_ids
        and (freeze or status in {'ready', 'completed', 'frozen'})
    )
    return {
        'status': status,
        'freeze_ready': freeze_ready,
        'intent_contract_status': _first_text(intent_contract.get('status')),
        'context_contract_status': _first_text(context_contract.get('status')),
        'pending_obligation_ids': pending_obligation_ids,
        'deferred_obligation_ids': deferred_obligation_ids,
        'blocked_obligation_ids': blocked_obligation_ids,
        'superseded_obligation_ids': superseded_obligation_ids,
        'waived_obligation_ids': waived_obligation_ids,
        'promoted_context_candidate_ids': context_contract.get('promoted_candidate_ids') if isinstance(context_contract.get('promoted_candidate_ids'), list) else [],
        'active_reference_artifact_refs': context_contract.get('active_reference_artifact_refs') if isinstance(context_contract.get('active_reference_artifact_refs'), list) else [],
        'promoted_history_scan_candidate_ids': context_contract.get('promoted_history_scan_candidate_ids') if isinstance(context_contract.get('promoted_history_scan_candidate_ids'), list) else [],
        'pending_goal_ids': pending_goal_ids[:12],
        'pending_output_slot_ids': pending_output_slot_ids[:12],
        'deferred_output_slot_ids': deferred_output_slot_ids[:12],
        'blocked_output_slot_ids': blocked_output_slot_ids[:12],
        'superseded_output_slot_ids': superseded_output_slot_ids[:12],
        'waived_output_slot_ids': waived_output_slot_ids[:12],
        'checkpoints': flow_review.get('checkpoints') if isinstance(flow_review.get('checkpoints'), list) else ['route', 'artifacts', 'outputs', 'memory_delta'],
    }


def _possibility_space(
    target: Mapping[str, Any],
    artifact_flow: Mapping[str, Any],
    goals: list[dict[str, Any]],
    review: Mapping[str, Any],
    loop: Mapping[str, Any],
    intent_contract: Mapping[str, Any],
    context_contract: Mapping[str, Any],
    *,
    freeze: bool,
    working_status: str,
) -> dict[str, Any]:
    input_routes = artifact_flow.get('input_routes') if isinstance(artifact_flow.get('input_routes'), list) else []
    output_slots = artifact_flow.get('output_slots') if isinstance(artifact_flow.get('output_slots'), list) else []
    pending_goal_ids = [
        str(item.get('goal_id') or '').strip()
        for item in goals
        if str(item.get('status') or '').strip().lower() in {'pending', 'planned', 'active'}
    ][:12]
    pending_output_slot_ids = [
        str(item.get('slot_id') or f'output-{index}').strip()
        for index, item in enumerate(output_slots, start=1)
        if isinstance(item, Mapping) and str(item.get('status') or '').strip().lower() not in {'fulfilled', 'superseded', 'waived'}
    ][:12]
    deferred_output_slot_ids = [
        str(item.get('slot_id') or f'output-{index}').strip()
        for index, item in enumerate(output_slots, start=1)
        if isinstance(item, Mapping) and str(item.get('lifecycle') or '').strip().lower() == 'deferred_output'
    ][:12]
    waived_output_slot_ids = [
        str(item.get('slot_id') or f'output-{index}').strip()
        for index, item in enumerate(output_slots, start=1)
        if isinstance(item, Mapping) and str(item.get('status') or '').strip().lower() == 'waived'
    ][:12]
    superseded_output_slot_ids = [
        str(item.get('slot_id') or f'output-{index}').strip()
        for index, item in enumerate(output_slots, start=1)
        if isinstance(item, Mapping) and str(item.get('status') or '').strip().lower() == 'superseded'
    ][:12]
    candidate_capabilities = _unique_texts(
        [
            target.get('capability'),
            target.get('mode'),
            *[
                ((route.get('routing_hint') or {}).get('capability') if isinstance(route, Mapping) else None)
                for route in input_routes
            ],
        ],
        limit=8,
    )
    candidate_output_types = _unique_texts(
        [item.get('type') if isinstance(item, Mapping) else None for item in output_slots],
        limit=6,
    )
    review_paths: list[str] = []
    if not freeze:
        review_paths.append('continue')
        if pending_goal_ids or pending_output_slot_ids or working_status in {'repairing', 'blocked'} or bool(loop.get('can_continue')):
            review_paths.append('revise')
        if bool(review.get('freeze_ready')) and working_status != 'blocked':
            review_paths.append('freeze')
    return {
        'state': 'closed' if freeze else 'open',
        'constraint': working_status if not freeze and working_status in {'repairing', 'blocked'} else None,
        'candidate_capabilities': candidate_capabilities,
        'candidate_output_types': candidate_output_types,
        'pending_goal_ids': pending_goal_ids,
        'pending_obligation_ids': intent_contract.get('pending_obligation_ids') if isinstance(intent_contract.get('pending_obligation_ids'), list) else [],
        'deferred_obligation_ids': intent_contract.get('deferred_obligation_ids') if isinstance(intent_contract.get('deferred_obligation_ids'), list) else [],
        'blocked_obligation_ids': intent_contract.get('blocked_obligation_ids') if isinstance(intent_contract.get('blocked_obligation_ids'), list) else [],
        'superseded_obligation_ids': intent_contract.get('superseded_obligation_ids') if isinstance(intent_contract.get('superseded_obligation_ids'), list) else [],
        'waived_obligation_ids': intent_contract.get('waived_obligation_ids') if isinstance(intent_contract.get('waived_obligation_ids'), list) else [],
        'context_candidate_count': context_contract.get('candidate_count'),
        'general_intent_candidate_count': intent_contract.get('general_candidate_count'),
        'general_context_candidate_count': context_contract.get('general_candidate_count'),
        'general_promoted_candidate_count': int(intent_contract.get('general_promoted_count') or 0)
        + int(context_contract.get('general_promoted_count') or 0),
        'general_reserved_candidate_count': int(intent_contract.get('general_reserved_count') or 0)
        + int(context_contract.get('general_reserved_count') or 0),
        'general_omitted_candidate_count': int(intent_contract.get('general_omitted_count') or 0)
        + int(context_contract.get('general_omitted_count') or 0),
        'general_superseded_candidate_count': int(intent_contract.get('general_superseded_count') or 0)
        + int(context_contract.get('general_superseded_count') or 0),
        'general_waived_candidate_count': int(intent_contract.get('general_waived_count') or 0)
        + int(context_contract.get('general_waived_count') or 0),
        'general_rejected_candidate_count': int(intent_contract.get('general_rejected_count') or 0)
        + int(context_contract.get('general_rejected_count') or 0),
        'promoted_context_candidate_ids': context_contract.get('promoted_candidate_ids') if isinstance(context_contract.get('promoted_candidate_ids'), list) else [],
        'active_reference_artifact_refs': context_contract.get('active_reference_artifact_refs') if isinstance(context_contract.get('active_reference_artifact_refs'), list) else [],
        'promoted_history_scan_candidate_ids': context_contract.get('promoted_history_scan_candidate_ids') if isinstance(context_contract.get('promoted_history_scan_candidate_ids'), list) else [],
        'pending_output_slot_ids': pending_output_slot_ids,
        'deferred_output_slot_ids': deferred_output_slot_ids,
        'superseded_output_slot_ids': superseded_output_slot_ids,
        'waived_output_slot_ids': waived_output_slot_ids,
        'review_paths': review_paths,
        'open_path_count': len(review_paths),
        'extensible': not freeze,
    }


def _closure(
    review: Mapping[str, Any],
    possibility_space: Mapping[str, Any],
    *,
    freeze: bool,
    working_status: str,
) -> dict[str, Any]:
    deferred_output_slot_ids = (
        review.get('deferred_output_slot_ids')
        if isinstance(review.get('deferred_output_slot_ids'), list)
        else []
    )
    pending_output_slot_ids = (
        review.get('pending_output_slot_ids')
        if isinstance(review.get('pending_output_slot_ids'), list)
        else []
    )
    pending_obligation_ids = (
        review.get('pending_obligation_ids')
        if isinstance(review.get('pending_obligation_ids'), list)
        else []
    )
    deferred_obligation_ids = (
        review.get('deferred_obligation_ids')
        if isinstance(review.get('deferred_obligation_ids'), list)
        else []
    )
    blocked_obligation_ids = (
        review.get('blocked_obligation_ids')
        if isinstance(review.get('blocked_obligation_ids'), list)
        else []
    )
    if freeze:
        if working_status == 'blocked':
            return {
                'status': 'closed',
                'postable': True,
                'close_authority': 'logic',
                'reason': 'request reached a terminal blocked state and was closed by response logic.',
            }
        if pending_output_slot_ids or pending_obligation_ids or deferred_obligation_ids or blocked_obligation_ids:
            return {
                'status': 'closed',
                'postable': True,
                'close_authority': 'ollmo',
                'continuation_expected': True,
                'pending_output_slot_ids': pending_output_slot_ids[:12],
                'deferred_output_slot_ids': deferred_output_slot_ids[:12],
                'pending_obligation_ids': pending_obligation_ids[:12],
                'deferred_obligation_ids': deferred_obligation_ids[:12],
                'blocked_obligation_ids': blocked_obligation_ids[:12],
                'reason': 'a truthful partial moment was frozen while promoted work remains queued, pending, or blocked for later resolution.',
            }
        return {
            'status': 'closed',
            'postable': True,
            'close_authority': 'ollmo',
            'reason': 'working review deemed the request complete and froze the final response.',
        }
    if bool(review.get('freeze_ready')) and working_status != 'blocked':
        return {
            'status': 'ready',
            'postable': True,
            'close_authority': None,
            'reason': 'review checkpoints are satisfied; Ollmo may freeze the request when the pass is complete.',
        }
    constraint = _first_text(possibility_space.get('constraint'))
    if constraint == 'repairing':
        reason = 'self-heal or revision is still in progress, so the working frame stays open.'
    elif constraint == 'blocked':
        reason = 'the frame remains open for repair or revision even though the current pass is blocked.'
    else:
        reason = 'route, artifact, and review possibilities remain open while the request is still fluid.'
    return {
        'status': 'open',
        'postable': False,
        'close_authority': None,
        'reason': reason,
    }


def build_working_frame(
    *,
    request_payload: Optional[Mapping[str, Any]] = None,
    route_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
    freeze: bool = False,
) -> dict[str, Any]:
    """Build a mutable orchestration-state snapshot for the current request pass."""

    request = _mapping(request_payload)
    route = _mapping(route_payload)
    response = _mapping(response_payload)
    runtime = (
        route.get('route_runtime')
        if isinstance(route.get('route_runtime'), Mapping)
        else response.get('runtime')
    )
    runtime = _mapping(runtime)
    planning_response = _planning_response_payload(request, route, response)
    artifact_flow = build_artifact_flow_plan(
        _input_artifacts(request, response),
        _output_artifacts(response),
        reference_artifacts=_reference_artifacts(request, response),
        request_payload=request,
        response_payload=planning_response,
    )
    artifact_dossiers = build_artifact_dossier_index(
        input_artifacts=_input_artifacts(request, response),
        reference_artifacts=_reference_artifacts(request, response),
        output_artifacts=_output_artifacts(response),
        response_payload=planning_response,
    )
    target = _target_payload(route, planning_response)
    route_summary = _route_summary(route, planning_response, runtime)
    request_summary = _request_summary(request, target)
    request_phase_graph = _request_phase_graph(request, route, planning_response, runtime)
    loop = _loop_state(request, response, runtime, freeze=freeze)
    working_status = _working_status(response, artifact_flow, loop, freeze=freeze)
    intent_contract = _intent_contract(request_phase_graph, runtime)
    context_contract = _context_contract(request, planning_response, runtime)
    goals = _goal_stack(request, target, artifact_flow, working_status=working_status, freeze=freeze)
    review = _review(
        artifact_flow,
        goals,
        intent_contract,
        context_contract,
        freeze=freeze,
        working_status=working_status,
    )
    possibility_space = _possibility_space(
        target,
        artifact_flow,
        goals,
        review,
        loop,
        intent_contract,
        context_contract,
        freeze=freeze,
        working_status=working_status,
    )
    closure = _closure(review, possibility_space, freeze=freeze, working_status=working_status)
    payload: dict[str, Any] = {
        'working_frame_version': WORKING_FRAME_VERSION,
        'kind': 'ollmo.working_frame',
        'status': working_status,
        'request': request_summary,
        'request_phase_graph': request_phase_graph,
        'intent_contract': intent_contract,
        'context_contract': context_contract,
        'work_tree': artifact_flow.get('work_tree') if isinstance(artifact_flow.get('work_tree'), Mapping) else {},
        'artifact_dossiers': artifact_dossiers,
        'target': target,
        'route': route_summary,
        'artifact_flow': artifact_flow,
        'goal_stack': goals,
        'loop': loop,
        'journal': _journal(runtime, route_summary, working_status, freeze=freeze),
        'review': review,
        'possibility_space': possibility_space,
        'closure': closure,
        'freeze': {
            'status': 'frozen' if freeze else 'fluid',
            'response_id': _first_text(response.get('id')),
        },
        'editability': {
            'mutable': not freeze,
            'editable_surfaces': ['goal_stack', 'work_tree', 'artifact_dossiers', 'artifact_flow', 'intent_contract', 'context_contract', 'route', 'review', 'possibility_space', 'closure', 'journal', 'controls', 'memory_delta'] if not freeze else [],
            'locked_surfaces': ['response_frame', 'closure', 'possibility_space', 'journal'] if freeze else [],
        },
    }
    control_snapshot = build_control_snapshot(request, planning_response)
    if control_snapshot:
        payload['controls'] = control_snapshot
    memory_delta = response.get('memory_delta') if isinstance(response.get('memory_delta'), Mapping) else {}
    if memory_delta:
        payload['memory_delta'] = dict(memory_delta)
    return _json_safe(payload)
