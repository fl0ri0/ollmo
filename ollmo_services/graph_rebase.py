"""Reviewed graph rebase proposal, validation, and successor helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
import unicodedata
from typing import Any, Optional

from ollmo_g.candidate_contracts import (
    build_candidate_graph,
    review_candidate_promotions,
)
from ollmo_g.decision_contracts import build_ghost_decision_contract
from ollmo_services.enforced_policy import (
    build_enforced_lineage_summary,
    build_enforced_policy_review,
    enforced_policy_allows_application,
)

GRAPH_REBASE_PROPOSAL_KIND = 'ollmo.graph_rebase_proposal'
GRAPH_REBASE_DIFF_KIND = 'ollmo.graph_rebase_diff'
GRAPH_REBASE_PRESERVATION_PROOF_KIND = 'ollmo.graph_rebase_preservation_proof'
GRAPH_REBASE_REVIEW_KIND = 'ollmo.graph_rebase_review'
GRAPH_REBASE_LIFECYCLE_KIND = 'ollmo.graph_rebase_lifecycle'
GRAPH_REBASE_SUCCESSOR_REQUEST_KIND = 'ollmo.graph_rebase_successor_request'
GRAPH_REBASE_EXECUTION_CONTRACT_PROOF_KIND = 'ollmo.graph_rebase_execution_contract_proof'
GRAPH_REBASE_AUTONOMY_ENV = 'OLLMO_GRAPH_REBASE_AUTONOMY'
GRAPH_REBASE_AUTONOMY_PRODUCT_DEFAULT = 'shadow'

PROPOSAL_ALLOWED_USE = 'proposal_only_until_runtime_rebase_review_accepts'
PROPOSAL_FORBIDDEN_USE = 'direct_graph_mutation_or_learning_only_authority'


def parse_graph_rebase_frame_sequence(value: Any) -> int:
    """Return an exact positive JSON integer for graph-rebase frame CAS."""

    if type(value) is not int or value <= 0:
        raise ValueError('Graph-rebase frame sequence must be a positive JSON integer.')
    return value


_GRAPH_REBASE_AUTONOMY_LEVELS = {
    'off',
    'shadow',
    'stage',
    'apply_reviewed',
    'apply_enforced',
}
_RUNTIME_REBASE_SOURCES = {
    'closure_review',
    'graph_closure_review',
    'operator_review',
    'promotion_review',
    'response_frame',
    'runtime',
    'runtime_closure_review',
    'runtime_review',
    'semantic_review_verdict',
}
_LEARNING_SOURCES = {
    'accepted_learning',
    'learning',
    'self_learning',
}
_ROUTE_HEALTH_EVIDENCE_TOKENS = {
    'backend_route_health_signal',
    'broad_provider_disablement',
    'cache_liveness',
    'degraded',
    'degraded_liveness_only',
    'liveness_only',
    'provider_ban',
    'provider_family_ban',
    'route_health',
}
_AUTHORIZED_REBASE_AUTHORITIES = {
    'operator_review',
    'runtime_review',
}
_ACCEPTED_AUTHORIZATION_STATUSES = {
    'accepted',
    'approved',
    'authorized',
    'granted',
}
_SMALLER_REDRAW_SCOPES = {
    'add_missing_branch',
    'fill_reserved_slot',
    'promote_reserved_slot',
    'repair_artifact_ref_identity',
    'repair_binding_dependency',
}
_DEPENDENCY_FIELDS = (
    'depends_on',
    'dependency_phase_ids',
    'source_phase_ids',
    'input_phase_ids',
)
_OBLIGATION_DEPENDENCY_FIELDS = (
    'depends_on_obligation_ids',
    'dependency_obligation_ids',
    'source_obligation_ids',
)
_GRAPH_REBASE_VOLATILE_KEYS = {
    'applied_graph_rebases',
    'applied_graph_patches',
    'graph_patch_lifecycle',
    'graph_rebase_lifecycle',
    'graph_rebase_proposals',
    'graph_rebase_reviews',
    'graph_repair_proposals',
    'graph_repair_reviews',
    'redraw_scope_evidence',
    'redraw_scope_ladder_review',
    'staged_graph_patches',
    'staged_graph_rebases',
    'successor_rebase_executions',
    'successor_rebase_requests',
}
_GRAPH_RECORD_COLLECTION_KEYS = {
    'downstream_branches',
    'intent_obligations',
    'output_obligations',
    'phases',
}
_GRAPH_DERIVED_PROJECTION_KEYS = {
    'candidate_graph',
    'continuation_required',
    'current_phase_capability',
    'current_phase_id',
    'current_phase_resolution',
    'decision_contract',
    'downstream_branch_ids',
    'downstream_capabilities',
    'downstream_phase_ids',
    'graph_refinements',
    'is_multi_phase',
    'output_candidates',
    'promotion_review',
    'promotions',
    'workload_graph',
    'workload_proposal_review',
    'workload_task_ids',
}
_REQUEST_IR_DERIVED_PROJECTION_KEYS = {
    'candidate_graph',
    'candidate_output_ids',
    'decision_contract',
    'final_output_obligation_ids',
    'output_candidates',
    'output_obligations',
    'promotion_review',
    'promotions',
    'workload_graph',
    'workload_proposal_review',
    'workload_task_ids',
}
_GRAPH_RECORD_DERIVED_TOPOLOGY_KEYS = {
    'downstream_branch_ids',
    'downstream_phase_ids',
}
_UNPROMOTED_CONTRACT_STATES = {
    'candidate',
    'discarded',
    'draft',
    'not_promoted',
    'omitted',
    'optional',
    'possible',
    'rejected',
    'reserved',
    'unpromoted',
}
_PROMOTED_CONTRACT_STATES = {
    'accepted',
    'active',
    'claimed',
    'completed',
    'fulfilled',
    'promoted',
    'promoted_to_obligation',
    'promotion_accepted',
    'selected',
    'used',
}
_REQUEST_IR_ALLOWED_KEYS = {
    'candidate_graph',
    'candidate_output_ids',
    'decision_contract',
    'final_output_obligation_ids',
    'graph_mode',
    'intent_anchor',
    'ir_version',
    'kind',
    'output_candidates',
    'output_obligations',
    'promotion_review',
    'promotions',
    'prompt_intent',
    'workload_graph',
    'workload_graph_version',
    'workload_proposal_review',
    'workload_task_ids',
}
_WORKLOAD_GRAPH_ALLOWED_KEYS = {
    'graph_mode',
    'intent_anchor',
    'kind',
    'leaf_task_ids',
    'prompt_intent',
    'proposal_review',
    'root_workload_id',
    'task_ids',
    'tasks',
    'visibility_summary',
    'workload_graph_version',
}
_WORKLOAD_TASK_ALLOWED_KEYS = {
    'accepted_proposals',
    'advisory_role',
    'artifact_prompt_source',
    'artifact_request',
    'branch_id',
    'candidate_id',
    'capability',
    'child_task_ids',
    'content_payload_source',
    'contract_state',
    'decision_notes',
    'decomposition_level',
    'deliverable',
    'depends_on',
    'evidence_requirements',
    'execution_contract',
    'input_refs',
    'intent',
    'kind',
    'learning_hint_refs',
    'lifecycle',
    'objective',
    'output_contract',
    'output_obligation_ref',
    'parent_task_ids',
    'phase_id',
    'promotion_policy',
    'promotion_suggestions',
    'queue_index',
    'rationale',
    'reconsideration_policy',
    'reconsideration_triggers',
    'repair_candidates',
    'requires_artifact',
    'review_criteria',
    'role',
    'semantic_intent',
    'semantic_review_criteria',
    'source',
    'stage_direction',
    'status',
    'superseded_by',
    'superseded_by_candidate_id',
    'superseded_by_obligation_id',
    'supersession_candidates',
    'supersession_reason',
    'task_id',
    'text_artifact_extension',
    'text_artifact_source',
    'text_artifact_source_name',
    'visibility',
    'waiver_candidates',
    'workload_task_ref',
    'workload_task_id',
}
_OUTPUT_CANDIDATE_ALLOWED_KEYS = {
    'accepted_proposals',
    'advisory_role',
    'artifact_prompt',
    'artifact_prompt_source',
    'artifact_request',
    'batch_prompts',
    'branch_id',
    'candidate_id',
    'capability',
    'content_payload',
    'content_payload_source',
    'decision_notes',
    'deliverable',
    'depends_on',
    'evidence_requirements',
    'execution_contract',
    'input_refs',
    'kind',
    'learning_hint_refs',
    'objective',
    'output_contract',
    'output_obligation_ref',
    'output_type',
    'phase_id',
    'phase_summary',
    'promoted_obligation_id',
    'promotion_policy',
    'promotion_reason',
    'promotion_source',
    'promotion_suggestions',
    'rationale',
    'reconsideration_triggers',
    'repair_candidates',
    'required',
    'review_criteria',
    'role',
    'semantic_intent',
    'semantic_review_criteria',
    'source',
    'stage_direction',
    'status',
    'superseded_by',
    'superseded_by_candidate_id',
    'superseded_by_obligation_id',
    'supersession_candidates',
    'supersession_reason',
    'text_artifact_extension',
    'text_artifact_source',
    'text_artifact_source_name',
    'waiver_candidates',
    'workload_task_ref',
}
_DECISION_CONTRACT_ALLOWED_KEYS = {
    'accepted_learning',
    'active_reconsideration_decisions',
    'active_reconsideration_review',
    'aspiration_frames',
    'aspiration_review',
    'authority_model',
    'blocked_obligation_ids',
    'block_resolution_reflex',
    'candidate_count',
    'candidate_policy',
    'closed_obligation_ids',
    'commitment_frames',
    'commitment_review',
    'controlled_attention_frames',
    'controlled_attention_review',
    'decision_contract_version',
    'graph_repair_proposals',
    'kind',
    'next_decision_priorities',
    'obligation_counts',
    'obligation_policy',
    'open_obligation_ids',
    'promotion_counts',
    'promotion_suggestions',
    'reconsideration_candidates',
    'reconsideration_reflex_signals',
    'recursive_cycle_review',
    'recursive_cycle_tasks',
    'repair_candidates',
    'semantic_decision_proposals',
    'semantic_decision_review',
    'semantic_planning_contract',
    'semantic_quality_contracts',
    'semantic_quality_review',
    'semantic_review_candidates',
    'semantic_review_lenses',
    'semantic_review_lens_review',
    'semantic_role_orientation_frames',
    'semantic_role_orientation_review',
    'supersession_candidates',
    'supersession_records',
    'waiver_candidates',
    'workload_proposal_coverage',
}
_GRAPH_RECORD_NON_SEMANTIC_KEYS = {
    'artifacts',
    'attempt',
    'attempt_count',
    'backend',
    'completed_at',
    'created_at',
    'duration_ms',
    'elapsed_ms',
    'error',
    'errors',
    'execution_result',
    'execution_status',
    'failed_at',
    'fulfillment_status',
    'instance_id',
    'last_error',
    'latency_ms',
    'lifecycle_state',
    'model',
    'output',
    'outputs',
    'output_status',
    'provider',
    'retry_count',
    'runtime',
    'runtime_state',
    'started_at',
    'state',
    'status',
    'timestamp',
    'timestamps',
    'updated_at',
}
_GRAPH_TOP_LEVEL_NON_SEMANTIC_KEYS = {
    *_GRAPH_RECORD_NON_SEMANTIC_KEYS,
    'developer_diagnostics',
    'frame_id',
    'id',
    'request_id',
    'response_id',
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
        return {_clean_text(key): _json_safe(raw_value) for key, raw_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
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


def _graph_digest_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(graph or {}))
    for key in _GRAPH_REBASE_VOLATILE_KEYS:
        payload.pop(key, None)
    return payload


def stable_graph_digest(graph: Mapping[str, Any]) -> str:
    """Return the canonical graph digest, excluding rebase bookkeeping fields."""

    return _stable_digest(_graph_digest_payload(graph), prefix='graph-')


def stable_graph_rebase_prompt_digest(prompt: Any) -> str:
    """Return the normalized one-way guard identity for root-prompt rejection."""

    normalized = _normalize_graph_rebase_prompt(prompt)
    return _stable_digest(normalized, prefix='root-prompt-') if normalized else ''


def _normalize_graph_rebase_prompt(prompt: Any) -> str:
    return ' '.join(
        unicodedata.normalize('NFKC', _clean_text(prompt)).casefold().split()
    )


def graph_rebase_prompt_contains_root(prompt: Any, root_prompt: Any) -> bool:
    """Return whether an executable prompt carrier replays current root truth."""

    normalized_prompt = _normalize_graph_rebase_prompt(prompt)
    normalized_root = _normalize_graph_rebase_prompt(root_prompt)
    return bool(
        normalized_prompt
        and normalized_root
        and (
            normalized_prompt == normalized_root
            or normalized_root in normalized_prompt
        )
    )


def _append_unique(values: list[str], value: str) -> None:
    text = _clean_text(value)
    if text and text not in values:
        values.append(text)


def _records(graph: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    values = graph.get(key) if isinstance(graph, Mapping) else []
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, Mapping)]


def _phase_id(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('phase_id') or record.get('branch_id'))


def _branch_id(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('branch_id') or record.get('phase_id'))


def _obligation_id(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('obligation_id') or record.get('id'))


def _obligation_identity(record: Mapping[str, Any]) -> str:
    obligation_id = _obligation_id(record)
    collection = _clean_text(record.get('_obligation_collection')) or 'obligations'
    return f'{collection}:{obligation_id}' if obligation_id else ''


def _index_by_id(records: Sequence[Mapping[str, Any]], id_fn) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = id_fn(record)
        if record_id:
            indexed[record_id] = dict(record)
    return indexed


def _obligations(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ('intent_obligations', 'output_obligations'):
        for item in _records(graph, key):
            obligation_id = _obligation_id(item)
            identity = f'{key}:{obligation_id}' if obligation_id else _stable_digest(item)
            if identity in seen:
                continue
            seen.add(identity)
            obligations.append({**item, '_obligation_collection': key})
    return obligations


def _dependency_sources(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in _DEPENDENCY_FIELDS:
        for item in _clean_string_list(record.get(field)):
            _append_unique(values, item)
    return values


def _dependency_edges(records: Sequence[Mapping[str, Any]], id_fn) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for record in records:
        target_id = id_fn(record)
        if not target_id:
            continue
        for source_id in _dependency_sources(record):
            edges.add((target_id, source_id))
    return edges


def _duplicate_record_ids(
    records: Sequence[Mapping[str, Any]],
    id_fn,
) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        record_id = id_fn(record)
        if not record_id:
            continue
        if record_id in seen:
            _append_unique(duplicates, record_id)
        seen.add(record_id)
    return duplicates


def _candidate_duplicate_id_summary(graph: Mapping[str, Any]) -> dict[str, list[str]]:
    summary = {
        'phases': _duplicate_record_ids(_records(graph, 'phases'), _phase_id),
        'branches': _duplicate_record_ids(_records(graph, 'downstream_branches'), _branch_id),
        'intent_obligations': _duplicate_record_ids(
            _records(graph, 'intent_obligations'),
            _obligation_id,
        ),
        'output_obligations': _duplicate_record_ids(
            _records(graph, 'output_obligations'),
            _obligation_id,
        ),
    }
    return {key: values for key, values in summary.items() if values}


def _projection_contract_state(record: Mapping[str, Any]) -> str:
    for key in (
        'contract_state',
        'contract_status',
        'obligation_state',
        'intent_state',
        'promotion_status',
    ):
        value = _status(record.get(key))
        if value:
            return value
    return _status(record.get('status'))


def _is_unpromoted_projection_record(record: Mapping[str, Any]) -> bool:
    state = _projection_contract_state(record)
    if state in _PROMOTED_CONTRACT_STATES or _clean_text(record.get('promoted_from_candidate_id')):
        return False
    if _clean_text(record.get('candidate_id')):
        return True
    if state in _UNPROMOTED_CONTRACT_STATES:
        return True
    return record.get('required') is False


def _dependency_cycle_issues(graph: Mapping[str, Any]) -> list[dict[str, str]]:
    phases = _records(graph, 'phases')
    phase_ids = {_phase_id(item) for item in phases if _phase_id(item)}
    phase_aliases: dict[str, str] = {}
    for phase in phases:
        phase_id = _phase_id(phase)
        if not phase_id:
            continue
        phase_aliases[phase_id] = phase_id
        branch_id = _clean_text(phase.get('branch_id'))
        if branch_id:
            phase_aliases[branch_id] = phase_id
    dependencies: dict[str, set[str]] = {phase_id: set() for phase_id in phase_ids}
    issues: list[dict[str, str]] = []
    for phase in phases:
        phase_id = _phase_id(phase)
        if not phase_id:
            continue
        for raw_source_id in _dependency_sources(phase):
            source_id = phase_aliases.get(raw_source_id, raw_source_id)
            if source_id == phase_id:
                issues.append(
                    {
                        'target_id': phase_id,
                        'source_id': raw_source_id,
                        'relation': 'self_dependency',
                    }
                )
            if source_id in phase_ids:
                dependencies[phase_id].add(source_id)

    visiting: set[str] = set()
    visited: set[str] = set()
    reported_cycles: set[tuple[str, ...]] = set()

    def visit(phase_id: str, path: list[str]) -> None:
        if phase_id in visited:
            return
        if phase_id in visiting:
            try:
                cycle = path[path.index(phase_id):]
            except ValueError:
                cycle = [phase_id]
            cycle_key = tuple(sorted(set(cycle)))
            if cycle_key and cycle_key not in reported_cycles:
                reported_cycles.add(cycle_key)
                issues.append(
                    {
                        'target_id': phase_id,
                        'source_id': phase_id,
                        'relation': 'dependency_cycle',
                        'cycle': ' -> '.join([*cycle, phase_id]),
                    }
                )
            return
        visiting.add(phase_id)
        for source_id in sorted(dependencies.get(phase_id) or []):
            visit(source_id, [*path, phase_id])
        visiting.discard(phase_id)
        visited.add(phase_id)

    for phase_id in sorted(phase_ids):
        visit(phase_id, [])
    return issues


def _candidate_dangling_dependency_edges(
    graph: Mapping[str, Any],
    base_graph: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, str]]:
    phases = _records(graph, 'phases')
    branches = _records(graph, 'downstream_branches')
    base = base_graph if isinstance(base_graph, Mapping) else {}
    base_phases = _records(base, 'phases')
    base_branches = _records(base, 'downstream_branches')
    phase_ids = {_phase_id(item) for item in phases if _phase_id(item)}
    branch_ids = {_branch_id(item) for item in branches if _branch_id(item)}
    identities = phase_ids | branch_ids
    lineage_phase_ids = phase_ids | {
        _phase_id(item) for item in base_phases if _phase_id(item)
    }
    lineage_branch_ids = branch_ids | {
        _branch_id(item) for item in base_branches if _branch_id(item)
    }
    obligation_ids = {
        _obligation_id(item)
        for collection in ('intent_obligations', 'output_obligations')
        for item in _records(graph, collection)
        if _obligation_id(item)
    }
    lineage_obligation_ids = obligation_ids | {
        _obligation_id(item)
        for collection in ('intent_obligations', 'output_obligations')
        for item in _records(base, collection)
        if _obligation_id(item)
    }
    dangling: list[dict[str, str]] = []
    for target_id, source_id in sorted(
        _dependency_edges(phases, _phase_id) | _dependency_edges(branches, _branch_id)
    ):
        if source_id not in identities:
            dangling.append({'target_id': target_id, 'source_id': source_id})
    for branch in branches:
        branch_id = _branch_id(branch)
        phase_id = _clean_text(branch.get('phase_id'))
        if phase_id and phase_id not in phase_ids:
            dangling.append(
                {
                    'target_id': branch_id,
                    'source_id': phase_id,
                    'relation': 'branch_phase_target',
                }
            )
    for phase in phases:
        phase_id = _phase_id(phase)
        lineage = _lineage(phase)
        for key, valid_ids in (
            ('parent_phase_id', lineage_phase_ids),
            ('parent_branch_id', lineage_branch_ids),
        ):
            parent_id = _clean_text(lineage.get(key))
            if parent_id and (parent_id not in valid_ids or parent_id == phase_id):
                dangling.append(
                    {
                        'target_id': phase_id,
                        'source_id': parent_id,
                        'relation': f'phase_lineage:{key}',
                    }
                )
    for branch in branches:
        branch_id = _branch_id(branch)
        lineage = _lineage(branch)
        for key, valid_ids in (
            ('parent_phase_id', lineage_phase_ids),
            ('parent_branch_id', lineage_branch_ids),
        ):
            parent_id = _clean_text(lineage.get(key))
            if parent_id and (parent_id not in valid_ids or parent_id == branch_id):
                dangling.append(
                    {
                        'target_id': branch_id,
                        'source_id': parent_id,
                        'relation': f'branch_lineage:{key}',
                    }
                )
    for collection in ('intent_obligations', 'output_obligations'):
        for obligation in _records(graph, collection):
            obligation_id = _obligation_id(obligation)
            parent_obligation_id = _clean_text(
                _lineage(obligation).get('parent_obligation_id')
            )
            if parent_obligation_id and (
                parent_obligation_id not in lineage_obligation_ids
                or parent_obligation_id == obligation_id
            ):
                dangling.append(
                    {
                        'target_id': obligation_id,
                        'source_id': parent_obligation_id,
                        'relation': f'{collection}_lineage:parent_obligation_id',
                    }
                )
            for key in ('phase_id', 'target_phase_id'):
                phase_id = _clean_text(obligation.get(key))
                if phase_id and phase_id not in phase_ids:
                    dangling.append(
                        {
                            'target_id': obligation_id,
                            'source_id': phase_id,
                            'relation': f'{collection}:{key}',
                        }
                    )
            for key in ('branch_id', 'target_branch_id'):
                branch_id = _clean_text(obligation.get(key))
                if branch_id and branch_id not in identities:
                    dangling.append(
                        {
                            'target_id': obligation_id,
                            'source_id': branch_id,
                            'relation': f'{collection}:{key}',
                        }
                    )
            for source_id in _dependency_sources(obligation):
                if source_id not in identities:
                    dangling.append(
                        {
                            'target_id': obligation_id,
                            'source_id': source_id,
                            'relation': f'{collection}:dependency',
                        }
                    )
            for key in _OBLIGATION_DEPENDENCY_FIELDS:
                for source_obligation_id in _clean_string_list(obligation.get(key)):
                    if source_obligation_id not in obligation_ids:
                        dangling.append(
                            {
                                'target_id': obligation_id,
                                'source_id': source_obligation_id,
                                'relation': f'{collection}:{key}',
                            }
                        )
    dangling.extend(_dependency_cycle_issues(graph))
    return dangling


def _candidate_projection_consistency_issues(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    phases = _records(graph, 'phases')
    branches = _records(graph, 'downstream_branches')
    phase_ids = {_phase_id(item) for item in phases if _phase_id(item)}
    branch_ids = {_branch_id(item) for item in branches if _branch_id(item)}
    request_ir = graph.get('request_ir') if isinstance(graph.get('request_ir'), Mapping) else {}
    issues: list[dict[str, Any]] = []

    def add_issue(field: str, reason: str, **details: Any) -> None:
        issues.append(_compact_payload({'field': field, 'reason': reason, **details}))

    def semantic_equal(left: Any, right: Any) -> bool:
        if left in (None, '', [], {}) and right in (None, '', [], {}):
            return True
        return _semantic_contract_value(left) == _semantic_contract_value(right)

    def unknown_keys(value: Mapping[str, Any], allowed: set[str]) -> list[str]:
        return sorted(
            _clean_text(key)
            for key in value
            if _clean_text(key) and _clean_text(key) not in allowed
        )

    promoted_branches = [
        item for item in branches if not _is_unpromoted_projection_record(item)
    ]
    expected_downstream_phase_ids = {
        _clean_text(item.get('phase_id'))
        for item in promoted_branches
        if _clean_text(item.get('phase_id'))
    }
    expected_downstream_branch_ids = {
        _branch_id(item) for item in promoted_branches if _branch_id(item)
    }

    def compare_declared_ids(key: str, expected: set[str]) -> None:
        if key not in graph:
            return
        declared = set(_clean_string_list(graph.get(key)))
        if declared != expected:
            issues.append(
                {
                    'field': key,
                    'declared': sorted(declared),
                    'expected': sorted(expected),
                }
            )

    compare_declared_ids(
        'downstream_phase_ids',
        expected_downstream_phase_ids,
    )
    compare_declared_ids('downstream_branch_ids', expected_downstream_branch_ids)
    if 'downstream_capabilities' in graph:
        declared_capabilities = set(_clean_string_list(graph.get('downstream_capabilities')))
        expected_capabilities = {
            _clean_text(item.get('capability'))
            for item in promoted_branches
            if _clean_text(item.get('capability'))
        }
        if declared_capabilities != expected_capabilities:
            issues.append(
                {
                    'field': 'downstream_capabilities',
                    'declared': sorted(declared_capabilities),
                    'expected': sorted(expected_capabilities),
                }
            )
    if 'is_multi_phase' in graph and bool(graph.get('is_multi_phase')) != bool(promoted_branches):
        issues.append({'field': 'is_multi_phase', 'reason': 'projection_mismatch'})
    if 'continuation_required' in graph and bool(graph.get('continuation_required')) != bool(
        promoted_branches
    ):
        add_issue('continuation_required', 'projection_mismatch')

    current_phase_id = _clean_text(graph.get('current_phase_id'))
    if current_phase_id:
        current_phase = next(
            (item for item in phases if _phase_id(item) == current_phase_id),
            None,
        )
        if current_phase is None:
            issues.append({'field': 'current_phase_id', 'reason': 'phase_missing'})
        elif graph.get('current_phase_capability') not in (None, '') and _status(
            graph.get('current_phase_capability')
        ) != _status(current_phase.get('capability')):
            issues.append({'field': 'current_phase_capability', 'reason': 'projection_mismatch'})
        elif graph.get('current_phase_resolution') not in (None, '') and _status(
            graph.get('current_phase_resolution')
        ) != _status(current_phase.get('resolution')):
            add_issue('current_phase_resolution', 'projection_mismatch')
        current_downstream_ids = set(
            _clean_string_list((current_phase or {}).get('downstream_phase_ids'))
        )
        if request_ir and current_downstream_ids != expected_downstream_phase_ids:
            add_issue(
                'phases.current_phase.downstream_phase_ids',
                'projection_mismatch',
                declared=sorted(current_downstream_ids),
                expected=sorted(expected_downstream_phase_ids),
            )

    phase_index = {_phase_id(item): item for item in phases if _phase_id(item)}
    branches_by_phase: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        branches_by_phase.setdefault(_clean_text(branch.get('phase_id')), []).append(branch)
    for phase in phases:
        phase_id = _phase_id(phase)
        if not request_ir or not phase_id or phase_id == current_phase_id:
            continue
        matching_branches = branches_by_phase.get(phase_id) or []
        if len(matching_branches) != 1:
            add_issue(
                'phases.downstream_branch',
                'phase_branch_cardinality_mismatch',
                phase_id=phase_id,
                matching_branch_count=len(matching_branches),
            )
            continue
        if _clean_text(phase.get('branch_id')) and _clean_text(phase.get('branch_id')) != _branch_id(
            matching_branches[0]
        ):
            add_issue(
                'phases.downstream_branch',
                'phase_branch_identity_mismatch',
                phase_id=phase_id,
            )
        if set(_dependency_sources(phase)) != set(_dependency_sources(matching_branches[0])):
            add_issue(
                'phases.downstream_branch.depends_on',
                'phase_branch_dependency_mismatch',
                phase_id=phase_id,
            )

    output_obligations = _records(graph, 'output_obligations')
    obligations_by_phase: dict[str, list[dict[str, Any]]] = {}
    for obligation in output_obligations:
        obligations_by_phase.setdefault(_clean_text(obligation.get('phase_id')), []).append(obligation)
    for phase in phases:
        phase_id = _phase_id(phase)
        if not request_ir or not phase_id or _is_unpromoted_projection_record(phase):
            continue
        matching_obligations = obligations_by_phase.get(phase_id) or []
        if len(matching_obligations) != 1:
            add_issue(
                'phases.output_obligation',
                'phase_obligation_cardinality_mismatch',
                phase_id=phase_id,
                matching_obligation_count=len(matching_obligations),
            )
            continue
        obligation = matching_obligations[0]
        for key in ('capability', 'output_type'):
            if _status(phase.get(key)) != _status(obligation.get(key)):
                add_issue(
                    f'phases.output_obligation.{key}',
                    'phase_obligation_contract_mismatch',
                    phase_id=phase_id,
                )
        phase_obligation_id = _clean_text(phase.get('obligation_id'))
        if phase_obligation_id and phase_obligation_id != _obligation_id(obligation):
            add_issue(
                'phases.output_obligation.obligation_id',
                'phase_obligation_identity_mismatch',
                phase_id=phase_id,
            )

    if request_ir:
        unknown_ir_keys = unknown_keys(request_ir, _REQUEST_IR_ALLOWED_KEYS)
        if unknown_ir_keys:
            add_issue(
                'request_ir',
                'unknown_projection_fields',
                unknown_fields=unknown_ir_keys,
            )

        for key in (
            'candidate_graph',
            'decision_contract',
            'output_candidates',
            'output_obligations',
            'promotion_review',
            'promotions',
            'workload_graph',
            'workload_proposal_review',
            'workload_task_ids',
        ):
            if not semantic_equal(graph.get(key), request_ir.get(key)):
                add_issue(f'request_ir.{key}', 'top_level_projection_mismatch')

        ir_prompt_intent = request_ir.get('prompt_intent')
        if (
            isinstance(graph.get('prompt_intent'), Mapping)
            and isinstance(ir_prompt_intent, Mapping)
        ):
            top_prompt_intent = _semantic_contract_value(graph.get('prompt_intent'))
            compact_ir_prompt_intent = _semantic_contract_value(ir_prompt_intent)
            if any(
                top_prompt_intent.get(key) != value
                for key, value in compact_ir_prompt_intent.items()
            ):
                issues.append({'field': 'request_ir.prompt_intent', 'reason': 'projection_mismatch'})
        final_output_ids = set(_clean_string_list(request_ir.get('final_output_obligation_ids')))
        expected_final_output_ids = {
            _obligation_id(item)
            for item in output_obligations
            if _obligation_id(item) and _status(item.get('role')) == 'final_output'
        }
        if final_output_ids != expected_final_output_ids:
            add_issue(
                'request_ir.final_output_obligation_ids',
                'projection_mismatch',
                declared=sorted(final_output_ids),
                expected=sorted(expected_final_output_ids),
            )

        output_candidates = _records(request_ir, 'output_candidates')
        for candidate in output_candidates:
            candidate_unknown_keys = unknown_keys(candidate, _OUTPUT_CANDIDATE_ALLOWED_KEYS)
            if candidate_unknown_keys:
                add_issue(
                    'request_ir.output_candidates',
                    'unknown_candidate_projection_fields',
                    candidate_id=_clean_text(candidate.get('candidate_id')),
                    unknown_fields=candidate_unknown_keys,
                )
            phase = phase_index.get(_clean_text(candidate.get('phase_id')))
            if phase is None:
                add_issue(
                    'request_ir.output_candidates',
                    'candidate_phase_missing',
                    candidate_id=_clean_text(candidate.get('candidate_id')),
                )
                continue
            for key in ('branch_id', 'capability', 'output_type'):
                expected_value = (
                    _branch_id(phase) if key == 'branch_id' else _status(phase.get(key))
                )
                declared_value = (
                    _clean_text(candidate.get(key)) if key == 'branch_id' else _status(candidate.get(key))
                )
                if declared_value and declared_value != expected_value:
                    add_issue(
                        f'request_ir.output_candidates.{key}',
                        'candidate_phase_contract_mismatch',
                        candidate_id=_clean_text(candidate.get('candidate_id')),
                    )

        promotions = _records(request_ir, 'promotions')
        candidate_ids = {
            _clean_text(item.get('candidate_id'))
            for item in output_candidates
            if _clean_text(item.get('candidate_id'))
        }
        output_obligation_ids = {
            _obligation_id(item) for item in output_obligations if _obligation_id(item)
        }
        for promotion in promotions:
            candidate_id = _clean_text(promotion.get('candidate_id'))
            obligation_id = _clean_text(
                promotion.get('obligation_id')
                or promotion.get('contract_ref')
                or promotion.get('target')
            )
            if candidate_id not in candidate_ids or obligation_id not in output_obligation_ids:
                add_issue(
                    'request_ir.promotions',
                    'promotion_target_missing',
                    candidate_id=candidate_id,
                    obligation_id=obligation_id,
                )

        workload_graph = (
            request_ir.get('workload_graph')
            if isinstance(request_ir.get('workload_graph'), Mapping)
            else {}
        )
        if workload_graph:
            workload_unknown_keys = unknown_keys(workload_graph, _WORKLOAD_GRAPH_ALLOWED_KEYS)
            if workload_unknown_keys:
                add_issue(
                    'request_ir.workload_graph',
                    'unknown_workload_projection_fields',
                    unknown_fields=workload_unknown_keys,
                )
            tasks = _records(workload_graph, 'tasks')
            task_ids = [_clean_text(item.get('task_id')) for item in tasks if _clean_text(item.get('task_id'))]
            task_phase_ids = [
                _clean_text(item.get('phase_id')) for item in tasks if _clean_text(item.get('phase_id'))
            ]
            if len(task_ids) != len(set(task_ids)):
                add_issue('request_ir.workload_graph.tasks', 'duplicate_task_id')
            if len(task_phase_ids) != len(set(task_phase_ids)):
                add_issue('request_ir.workload_graph.tasks', 'duplicate_task_phase_id')
            if set(task_phase_ids) != phase_ids:
                add_issue(
                    'request_ir.workload_graph.tasks',
                    'workload_phase_coverage_mismatch',
                    declared=sorted(set(task_phase_ids)),
                    expected=sorted(phase_ids),
                )
            task_by_phase = {
                _clean_text(item.get('phase_id')): item
                for item in tasks
                if _clean_text(item.get('phase_id'))
            }
            task_id_by_phase = {
                phase_id: _clean_text(task.get('task_id'))
                for phase_id, task in task_by_phase.items()
            }
            expected_children: dict[str, set[str]] = {
                task_id: set() for task_id in task_ids
            }
            for phase_id, task in task_by_phase.items():
                task_unknown_keys = unknown_keys(task, _WORKLOAD_TASK_ALLOWED_KEYS)
                if task_unknown_keys:
                    add_issue(
                        'request_ir.workload_graph.tasks',
                        'unknown_task_projection_fields',
                        task_id=_clean_text(task.get('task_id')),
                        unknown_fields=task_unknown_keys,
                    )
                phase = phase_index.get(phase_id)
                if phase is None:
                    continue
                expected_branch_id = _branch_id(phase)
                expected_dependencies = set(_dependency_sources(phase))
                if _clean_text(task.get('branch_id')) != expected_branch_id:
                    add_issue(
                        'request_ir.workload_graph.tasks.branch_id',
                        'task_phase_contract_mismatch',
                        task_id=_clean_text(task.get('task_id')),
                    )
                for key in ('capability',):
                    if _status(task.get(key)) != _status(phase.get(key)):
                        add_issue(
                            f'request_ir.workload_graph.tasks.{key}',
                            'task_phase_contract_mismatch',
                            task_id=_clean_text(task.get('task_id')),
                        )
                output_contract = (
                    task.get('output_contract')
                    if isinstance(task.get('output_contract'), Mapping)
                    else {}
                )
                if _status(output_contract.get('output_type')) != _status(phase.get('output_type')):
                    add_issue(
                        'request_ir.workload_graph.tasks.output_contract',
                        'task_phase_contract_mismatch',
                        task_id=_clean_text(task.get('task_id')),
                    )
                if set(_clean_string_list(task.get('depends_on'))) != expected_dependencies:
                    add_issue(
                        'request_ir.workload_graph.tasks.depends_on',
                        'task_phase_dependency_mismatch',
                        task_id=_clean_text(task.get('task_id')),
                    )
                expected_parent_task_ids = {
                    task_id_by_phase[item]
                    for item in expected_dependencies
                    if item in task_id_by_phase
                }
                if set(_clean_string_list(task.get('parent_task_ids'))) != expected_parent_task_ids:
                    add_issue(
                        'request_ir.workload_graph.tasks.parent_task_ids',
                        'task_parent_projection_mismatch',
                        task_id=_clean_text(task.get('task_id')),
                    )
                for parent_task_id in expected_parent_task_ids:
                    expected_children.setdefault(parent_task_id, set()).add(
                        _clean_text(task.get('task_id'))
                    )
            for task in tasks:
                task_id = _clean_text(task.get('task_id'))
                if set(_clean_string_list(task.get('child_task_ids'))) != expected_children.get(
                    task_id, set()
                ):
                    add_issue(
                        'request_ir.workload_graph.tasks.child_task_ids',
                        'task_child_projection_mismatch',
                        task_id=task_id,
                    )
            if set(_clean_string_list(workload_graph.get('task_ids'))) != set(task_ids):
                add_issue('request_ir.workload_graph.task_ids', 'projection_mismatch')
            expected_leaf_task_ids = {
                task_id for task_id, children in expected_children.items() if not children
            }
            if set(_clean_string_list(workload_graph.get('leaf_task_ids'))) != expected_leaf_task_ids:
                add_issue('request_ir.workload_graph.leaf_task_ids', 'projection_mismatch')
            if set(_clean_string_list(request_ir.get('workload_task_ids'))) != set(task_ids):
                add_issue('request_ir.workload_task_ids', 'projection_mismatch')
            if not semantic_equal(
                workload_graph.get('proposal_review'),
                request_ir.get('workload_proposal_review'),
            ):
                add_issue('request_ir.workload_proposal_review', 'workload_review_mismatch')

            candidate_graph = (
                request_ir.get('candidate_graph')
                if isinstance(request_ir.get('candidate_graph'), Mapping)
                else {}
            )
            expected_candidate_graph = build_candidate_graph(
                output_candidates=output_candidates,
                output_obligations=output_obligations,
                workload_tasks=tasks,
                promotions=promotions,
                workload_proposal_review=(
                    request_ir.get('workload_proposal_review')
                    if isinstance(request_ir.get('workload_proposal_review'), Mapping)
                    else {}
                ),
                intent_ref='intent_anchor',
                source='request_ir',
            )
            if not semantic_equal(candidate_graph, expected_candidate_graph):
                add_issue('request_ir.candidate_graph', 'canonical_candidate_graph_mismatch')
            expected_promotion_review = review_candidate_promotions(
                expected_candidate_graph,
                existing_contracts={'source': 'request_ir.output_obligations'},
            )
            promotion_review = (
                request_ir.get('promotion_review')
                if isinstance(request_ir.get('promotion_review'), Mapping)
                else {}
            )
            if not semantic_equal(promotion_review, expected_promotion_review):
                add_issue('request_ir.promotion_review', 'canonical_promotion_review_mismatch')

            decision_contract = (
                request_ir.get('decision_contract')
                if isinstance(request_ir.get('decision_contract'), Mapping)
                else {}
            )
            if decision_contract:
                decision_unknown_keys = unknown_keys(
                    decision_contract,
                    _DECISION_CONTRACT_ALLOWED_KEYS,
                )
                if decision_unknown_keys:
                    add_issue(
                        'request_ir.decision_contract',
                        'unknown_decision_projection_fields',
                        unknown_fields=decision_unknown_keys,
                    )
                expected_decision_contract = build_ghost_decision_contract(
                    candidate_graph=expected_candidate_graph,
                    promotion_review=expected_promotion_review,
                    workload_graph=workload_graph,
                    workload_proposal_review=(
                        request_ir.get('workload_proposal_review')
                        if isinstance(request_ir.get('workload_proposal_review'), Mapping)
                        else {}
                    ),
                    output_obligations=output_obligations,
                )
                for key in (
                    'authority_model',
                    'candidate_count',
                    'candidate_policy',
                    'obligation_counts',
                    'obligation_policy',
                    'promotion_counts',
                    'workload_proposal_coverage',
                ):
                    if not semantic_equal(
                        decision_contract.get(key),
                        expected_decision_contract.get(key),
                    ):
                        add_issue(
                            f'request_ir.decision_contract.{key}',
                            'canonical_decision_contract_mismatch',
                        )
                accepted_learning = (
                    decision_contract.get('accepted_learning')
                    if isinstance(decision_contract.get('accepted_learning'), Mapping)
                    else {}
                )
                if accepted_learning and (
                    _status(accepted_learning.get('authority')) != 'soft_hint'
                    or _status(accepted_learning.get('runtime_effect')) != 'none'
                    or _status(accepted_learning.get('allowed_use'))
                    not in {'none', 'orientation_only_not_promotion_authority'}
                ):
                    add_issue(
                        'request_ir.decision_contract.accepted_learning',
                        'learning_authority_boundary_mismatch',
                    )

        for refinement in _records(graph, 'graph_refinements'):
            refinement_unknown_keys = unknown_keys(
                refinement,
                {
                    'added_branch_ids',
                    'added_count',
                    'capability',
                    'existing_count',
                    'reason',
                    'refinement',
                    'requested_count',
                    'source',
                },
            )
            if refinement_unknown_keys:
                add_issue(
                    'graph_refinements',
                    'unknown_refinement_projection_fields',
                    unknown_fields=refinement_unknown_keys,
                )
    return issues


def _semantic_record_payload(
    record: Mapping[str, Any],
    *,
    exclude_dependencies: bool = False,
) -> dict[str, Any]:
    """Return stable record meaning, excluding execution/runtime observations."""

    excluded = set(_GRAPH_RECORD_NON_SEMANTIC_KEYS)
    if exclude_dependencies:
        excluded.update(_DEPENDENCY_FIELDS)
        excluded.update(_GRAPH_RECORD_DERIVED_TOPOLOGY_KEYS)
    return {
        _clean_text(key): _json_safe(value)
        for key, value in record.items()
        if _clean_text(key) and _status(key) not in excluded
    }


def _semantic_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _clean_text(key): _semantic_contract_value(nested)
            for key, nested in value.items()
            if _clean_text(key) and _status(key) not in _GRAPH_RECORD_NON_SEMANTIC_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_semantic_contract_value(item) for item in value]
    return _json_safe(value)


def _failure_visibility_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    failure_statuses = {
        'blocked',
        'cancelled',
        'error',
        'failed',
        'partial_failed',
        'repair_needed',
        'repair_required',
    }
    payload: dict[str, Any] = {}
    for key in ('status', 'execution_status', 'lifecycle_state'):
        value = record.get(key)
        if _status(value) in failure_statuses:
            payload[key] = _json_safe(value)
    for key in ('error', 'errors', 'last_error', 'failed_at'):
        value = record.get(key)
        if value not in (None, '', [], {}):
            payload[key] = _json_safe(value)
    return payload


def _hidden_failure_visibility_losses(
    base_index: Mapping[str, Mapping[str, Any]],
    candidate_index: Mapping[str, Mapping[str, Any]],
    *,
    record_type: str,
) -> list[dict[str, Any]]:
    losses: list[dict[str, Any]] = []
    for record_id in sorted(set(base_index) & set(candidate_index)):
        base_failure = _failure_visibility_payload(base_index[record_id])
        if not base_failure:
            continue
        candidate_failure = _failure_visibility_payload(candidate_index[record_id])
        if base_failure == candidate_failure:
            continue
        candidate_record = candidate_index[record_id]
        if candidate_record.get('prior_failure_evidence') not in (None, '', [], {}) or candidate_record.get(
            'failure_lineage'
        ) not in (None, '', [], {}):
            continue
        losses.append(
            {
                'type': record_type,
                'id': record_id,
                'base_failure_digest': _stable_digest(base_failure, prefix='failure-'),
                'candidate_failure_digest': _stable_digest(candidate_failure, prefix='failure-'),
                'lost_fields': sorted(
                    key
                    for key in base_failure
                    if candidate_failure.get(key) != base_failure.get(key)
                ),
            }
        )
    return losses


def _top_level_semantic_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return graph-wide meaning, denying silent changes to unknown fields by default."""

    payload = {
        _clean_text(key): _semantic_contract_value(value)
        for key, value in sorted(graph.items(), key=lambda item: _clean_text(item[0]))
        if _clean_text(key)
        and _status(key) not in _GRAPH_TOP_LEVEL_NON_SEMANTIC_KEYS
        and _clean_text(key) not in _GRAPH_RECORD_COLLECTION_KEYS
        and _clean_text(key) not in _GRAPH_REBASE_VOLATILE_KEYS
        and _clean_text(key) not in _GRAPH_DERIVED_PROJECTION_KEYS
        and _clean_text(key) != 'request_ir'
    }
    request_ir = graph.get('request_ir') if isinstance(graph.get('request_ir'), Mapping) else {}
    if request_ir:
        payload['request_ir'] = {
            _clean_text(key): _semantic_contract_value(value)
            for key, value in sorted(request_ir.items(), key=lambda item: _clean_text(item[0]))
            if _clean_text(key) not in _REQUEST_IR_DERIVED_PROJECTION_KEYS
        }
    return payload


def _derived_projection_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: _semantic_contract_value(graph.get(key))
        for key in sorted(_GRAPH_DERIVED_PROJECTION_KEYS)
        if key in graph
    }
    request_ir = graph.get('request_ir') if isinstance(graph.get('request_ir'), Mapping) else {}
    request_ir_projection = {
        key: _semantic_contract_value(request_ir.get(key))
        for key in sorted(_REQUEST_IR_DERIVED_PROJECTION_KEYS)
        if key in request_ir
    }
    if request_ir_projection:
        payload['request_ir_projection'] = request_ir_projection
    return payload


def _changed_semantic_records(
    base_index: Mapping[str, Mapping[str, Any]],
    candidate_index: Mapping[str, Mapping[str, Any]],
    *,
    record_type: str,
    exclude_dependencies: bool = False,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for record_id in sorted(set(base_index) & set(candidate_index)):
        base_payload = _semantic_record_payload(
            base_index[record_id],
            exclude_dependencies=exclude_dependencies,
        )
        candidate_payload = _semantic_record_payload(
            candidate_index[record_id],
            exclude_dependencies=exclude_dependencies,
        )
        if base_payload == candidate_payload:
            continue
        changed_fields = sorted(
            key
            for key in set(base_payload) | set(candidate_payload)
            if base_payload.get(key) != candidate_payload.get(key)
        )
        changes.append(
            _compact_payload(
                {
                    'type': record_type,
                    'id': record_id,
                    'phase_id': candidate_index[record_id].get('phase_id'),
                    'branch_id': candidate_index[record_id].get('branch_id'),
                    'changed_fields': changed_fields,
                    'base_semantic_digest': _stable_digest(base_payload, prefix='record-'),
                    'candidate_semantic_digest': _stable_digest(candidate_payload, prefix='record-'),
                }
            )
        )
    return changes


def _lineage(record: Mapping[str, Any]) -> dict[str, Any]:
    lineage = record.get('lineage')
    return dict(lineage) if isinstance(lineage, Mapping) else {}


def _lineage_parent_id(record: Mapping[str, Any], *keys: str) -> str:
    lineage = _lineage(record)
    for key in keys:
        value = _clean_text(lineage.get(key))
        if value:
            return value
    return ''


def _lineage_relation(record: Mapping[str, Any]) -> str:
    return _status(_lineage(record).get('relation'))


def _lineaged_replacement_exists(
    candidate_records: Sequence[Mapping[str, Any]],
    parent_id: str,
    *,
    parent_keys: Sequence[str],
) -> bool:
    for record in candidate_records:
        if _lineage_parent_id(record, *parent_keys) != parent_id:
            continue
        relation = _lineage_relation(record)
        if relation in {'split_branch', 'supersedes', 'supersede', 'replacement', 'merge_branch', 'merge_branches'}:
            return True
    return False


def _collect_artifact_refs(value: Any) -> list[str]:
    refs: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, nested in node.items():
                key_text = _status(key)
                if key_text in {'artifact_ref', 'artifact_refs', 'artifact_uri', 'artifact_uris'}:
                    for ref in _clean_string_list(nested):
                        if ref.startswith('artifact://'):
                            _append_unique(refs, ref)
                    if isinstance(nested, str) and nested.startswith('artifact://'):
                        _append_unique(refs, nested)
                walk(nested)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node.startswith('artifact://'):
            _append_unique(refs, node)

    walk(value)
    return refs


def _required_obligation_ids(
    graph: Mapping[str, Any],
    collection: str,
) -> list[str]:
    ids: list[str] = []
    for item in _records(graph, collection):
        if item.get('required') is False:
            continue
        obligation_id = _obligation_id(item)
        if obligation_id:
            _append_unique(ids, obligation_id)
    return ids


def _target_bound_phase_records(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for phase in _records(graph, 'phases'):
        target_path = _clean_text(phase.get('repair_target_path') or phase.get('target_path'))
        repair_contract_id = _clean_text(phase.get('repair_contract_id'))
        if not target_path and not repair_contract_id:
            continue
        phase_id = _phase_id(phase)
        if not phase_id:
            continue
        records.append(
            {
                'phase_id': phase_id,
                'repair_contract_id': repair_contract_id,
                'repair_target_path': target_path,
            }
        )
    return records


def _candidate_preserves_target_bound_phase(
    candidate_graph: Mapping[str, Any],
    base_record: Mapping[str, Any],
) -> bool:
    base_phase_id = _clean_text(base_record.get('phase_id'))
    base_target_path = _clean_text(base_record.get('repair_target_path'))
    for phase in _records(candidate_graph, 'phases'):
        phase_id = _phase_id(phase)
        target_path = _clean_text(phase.get('repair_target_path') or phase.get('target_path'))
        if phase_id == base_phase_id and (not base_target_path or target_path == base_target_path):
            return True
        if _lineage_parent_id(phase, 'parent_phase_id') == base_phase_id:
            if target_path == base_target_path:
                return True
    return False


def _blocked_review(
    *,
    proposal: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
    status: str,
    source_proposal: Mapping[str, Any],
    diff: Optional[Mapping[str, Any]] = None,
    proof: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    proposal_id = _clean_text(proposal.get('proposal_id'))
    payload = {
        'kind': GRAPH_REBASE_REVIEW_KIND,
        'review_id': _stable_digest({'proposal_id': proposal_id, 'checks': checks}, prefix='graph-rebase-review-'),
        'proposal_id': proposal_id,
        'status': status,
        'authority': 'runtime_rebase_validation',
        'blocked_reasons': list(reasons),
        'validation_checks': list(checks),
        'allowed_runtime_action': 'none',
        'runtime_effect': 'none',
        'source_proposal': source_proposal,
        'diff': diff or {},
        'preservation_proof': proof or {},
    }
    return _json_safe(payload)


def normalize_graph_rebase_autonomy(value: Any) -> str:
    """Normalize graph rebase autonomy rollout values."""

    token = _status(value)
    if token in _GRAPH_REBASE_AUTONOMY_LEVELS:
        return token
    return 'off'


def describe_graph_rebase_autonomy(
    value: Any,
    *,
    source: str = 'explicit_value',
    configured: bool = True,
) -> dict[str, Any]:
    """Return normalized rebase autonomy plus safe diagnostics."""

    token = _status(value)
    normalized = normalize_graph_rebase_autonomy(value)
    return _json_safe(
        {
            'raw_value': value,
            'autonomy_level': normalized,
            'normalized': normalized,
            'invalid_value': bool(token and token not in _GRAPH_REBASE_AUTONOMY_LEVELS),
            'source': source,
            'configured': bool(configured),
        }
    )


def graph_rebase_autonomy_from_env(env: Optional[Mapping[str, Any]] = None) -> str:
    source = env if isinstance(env, Mapping) else os.environ
    if GRAPH_REBASE_AUTONOMY_ENV not in source:
        return GRAPH_REBASE_AUTONOMY_PRODUCT_DEFAULT
    return normalize_graph_rebase_autonomy(source.get(GRAPH_REBASE_AUTONOMY_ENV))


def describe_graph_rebase_autonomy_from_env(env: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    source = env if isinstance(env, Mapping) else os.environ
    configured = GRAPH_REBASE_AUTONOMY_ENV in source
    raw_value = (
        source.get(GRAPH_REBASE_AUTONOMY_ENV)
        if configured
        else GRAPH_REBASE_AUTONOMY_PRODUCT_DEFAULT
    )
    return describe_graph_rebase_autonomy(
        raw_value,
        source='environment' if configured else 'product_default',
        configured=configured,
    )


def build_graph_rebase_proposal(
    *,
    request_phase_graph: Mapping[str, Any],
    candidate_graph: Mapping[str, Any],
    target_response_id: str = '',
    target_frame_id: str = '',
    source: str = 'runtime_closure_review',
    reason: str = '',
    intent_anchor: Optional[Mapping[str, Any]] = None,
    evidence_refs: Optional[Sequence[str]] = None,
    candidate_origin: str = '',
    requested_rebase_class: str = '',
    scope_root_ids: Optional[Sequence[str]] = None,
    scope_phase_ids: Optional[Sequence[str]] = None,
    scope_branch_ids: Optional[Sequence[str]] = None,
    scope_artifact_refs: Optional[Sequence[str]] = None,
    preserve_outside_scope: Optional[bool] = None,
    redraw_scope_review_ref: str = '',
    root_prompt: str = '',
) -> dict[str, Any]:
    """Build a proposal-only graph rebase candidate."""

    base_graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    candidate = (
        _graph_digest_payload(candidate_graph)
        if isinstance(candidate_graph, Mapping)
        else {}
    )
    base_digest = stable_graph_digest(base_graph)
    candidate_digest = stable_graph_digest(candidate)
    refs = _clean_string_list(evidence_refs or [])
    normalized_root_prompt = _normalize_graph_rebase_prompt(root_prompt)
    root_prompt_digest = stable_graph_rebase_prompt_digest(normalized_root_prompt)
    proposal_id = _stable_digest(
        {
            'target_response_id': target_response_id or base_graph.get('response_id'),
            'target_frame_id': target_frame_id or base_graph.get('frame_id'),
            'base_graph_digest': base_digest,
            'candidate_graph_digest': candidate_digest,
            'source': source,
            'evidence_refs': refs,
            'reason': reason,
            'root_prompt_digest': root_prompt_digest,
        },
        prefix='graph-rebase-proposal-',
    )
    payload = {
        'kind': GRAPH_REBASE_PROPOSAL_KIND,
        'proposal_id': proposal_id,
        'target_response_id': _clean_text(target_response_id or base_graph.get('response_id')),
        'target_frame_id': _clean_text(target_frame_id or base_graph.get('frame_id')),
        'target_graph_id': base_digest,
        'base_graph_digest': base_digest,
        'candidate_graph_digest': candidate_digest,
        'source': _clean_text(source),
        'allowed_use': PROPOSAL_ALLOWED_USE,
        'forbidden_use': PROPOSAL_FORBIDDEN_USE,
        'reason': _clean_text(reason),
        'intent_anchor': dict(intent_anchor or {}),
        'evidence_refs': refs,
        'candidate_graph': copy.deepcopy(dict(candidate)),
        'candidate_origin': _clean_text(candidate_origin),
        'requested_rebase_class': _clean_text(requested_rebase_class),
        'scope_root_ids': _clean_string_list(scope_root_ids or []),
        'scope_phase_ids': _clean_string_list(scope_phase_ids or []),
        'scope_branch_ids': _clean_string_list(scope_branch_ids or []),
        'scope_artifact_refs': _clean_string_list(scope_artifact_refs or []),
        'preserve_outside_scope': preserve_outside_scope,
        'redraw_scope_review_ref': _clean_text(redraw_scope_review_ref),
        'root_prompt_guard': {
            'kind': 'ollmo.graph_rebase_root_prompt_guard',
            'digest': root_prompt_digest,
            'normalized_length': len(normalized_root_prompt),
            'authority': 'runtime_request_truth',
            'runtime_effect': 'none',
        } if root_prompt_digest else {},
    }
    return _compact_payload(payload)


def build_graph_rebase_diff(
    base_graph: Mapping[str, Any],
    candidate_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute a runtime-owned graph rebase diff."""

    base = base_graph if isinstance(base_graph, Mapping) else {}
    candidate = candidate_graph if isinstance(candidate_graph, Mapping) else {}
    base_phases = _records(base, 'phases')
    candidate_phases = _records(candidate, 'phases')
    base_branches = _records(base, 'downstream_branches')
    candidate_branches = _records(candidate, 'downstream_branches')
    base_obligations = _obligations(base)
    candidate_obligations = _obligations(candidate)
    base_phase_index = _index_by_id(base_phases, _phase_id)
    candidate_phase_index = _index_by_id(candidate_phases, _phase_id)
    base_branch_index = _index_by_id(base_branches, _branch_id)
    candidate_branch_index = _index_by_id(candidate_branches, _branch_id)
    base_obligation_index = _index_by_id(base_obligations, _obligation_identity)
    candidate_obligation_index = _index_by_id(candidate_obligations, _obligation_identity)
    semantic_changes = [
        *_changed_semantic_records(
            base_phase_index,
            candidate_phase_index,
            record_type='phase',
            exclude_dependencies=True,
        ),
        *_changed_semantic_records(
            base_branch_index,
            candidate_branch_index,
            record_type='branch',
            exclude_dependencies=True,
        ),
        *_changed_semantic_records(
            base_obligation_index,
            candidate_obligation_index,
            record_type='obligation',
        ),
    ]
    base_top_level_semantics = _top_level_semantic_payload(base)
    candidate_top_level_semantics = _top_level_semantic_payload(candidate)
    changed_top_level_fields = sorted(
        key
        for key in set(base_top_level_semantics) | set(candidate_top_level_semantics)
        if base_top_level_semantics.get(key) != candidate_top_level_semantics.get(key)
    )
    top_level_semantic_changes = [
        {
            'type': 'graph_contract',
            'changed_fields': changed_top_level_fields,
            'base_semantic_digest': _stable_digest(
                base_top_level_semantics,
                prefix='graph-contract-',
            ),
            'candidate_semantic_digest': _stable_digest(
                candidate_top_level_semantics,
                prefix='graph-contract-',
            ),
        }
    ] if changed_top_level_fields else []
    base_derived_projection = _derived_projection_payload(base)
    candidate_derived_projection = _derived_projection_payload(candidate)
    changed_derived_projection_fields = sorted(
        key
        for key in set(base_derived_projection) | set(candidate_derived_projection)
        if base_derived_projection.get(key) != candidate_derived_projection.get(key)
    )
    derived_projection_changes = [
        {
            'type': 'derived_graph_projection',
            'changed_fields': changed_derived_projection_fields,
            'base_projection_digest': _stable_digest(
                base_derived_projection,
                prefix='graph-projection-',
            ),
            'candidate_projection_digest': _stable_digest(
                candidate_derived_projection,
                prefix='graph-projection-',
            ),
        }
    ] if changed_derived_projection_fields else []
    hidden_failure_visibility_losses = [
        *_hidden_failure_visibility_losses(
            base_phase_index,
            candidate_phase_index,
            record_type='phase',
        ),
        *_hidden_failure_visibility_losses(
            base_branch_index,
            candidate_branch_index,
            record_type='branch',
        ),
        *_hidden_failure_visibility_losses(
            base_obligation_index,
            candidate_obligation_index,
            record_type='obligation',
        ),
    ]
    operations: list[dict[str, Any]] = []

    def add_operation(op: str, **values: Any) -> None:
        operations.append(_compact_payload({'op': op, **values}))

    for phase_id in sorted(base_phase_index):
        if phase_id in candidate_phase_index:
            add_operation('preserve_phase', phase_id=phase_id)
    for branch_id in sorted(base_branch_index):
        if branch_id in candidate_branch_index:
            add_operation('preserve_branch', branch_id=branch_id)
    for obligation_identity in sorted(base_obligation_index):
        if obligation_identity in candidate_obligation_index:
            obligation = base_obligation_index[obligation_identity]
            add_operation(
                'preserve_obligation',
                obligation_id=_obligation_id(obligation),
                obligation_collection=obligation.get('_obligation_collection'),
            )
    for phase_id, phase in sorted(candidate_phase_index.items()):
        if phase_id not in base_phase_index:
            add_operation('add_phase', phase_id=phase_id)
        relation = _lineage_relation(phase)
        parent_phase_id = _lineage_parent_id(phase, 'parent_phase_id')
        if relation == 'split_branch' and parent_phase_id:
            add_operation('split_branch', parent_phase_id=parent_phase_id, phase_id=phase_id)
        if relation in {'supersedes', 'supersede', 'replacement'} and parent_phase_id:
            add_operation('supersede_with_replacement', parent_phase_id=parent_phase_id, phase_id=phase_id)
        if relation in {'merge_branch', 'merge_branches'} and parent_phase_id:
            add_operation('merge_with_replacement', parent_phase_id=parent_phase_id, phase_id=phase_id)
    for branch_id, branch in sorted(candidate_branch_index.items()):
        if branch_id not in base_branch_index:
            add_operation('add_branch', branch_id=branch_id, phase_id=branch.get('phase_id'))
        relation = _lineage_relation(branch)
        parent_branch_id = _lineage_parent_id(branch, 'parent_branch_id', 'parent_phase_id')
        if relation == 'split_branch' and parent_branch_id:
            add_operation(
                'split_branch',
                parent_branch_id=parent_branch_id,
                branch_id=branch_id,
                phase_id=branch.get('phase_id'),
            )
        if relation in {'merge_branch', 'merge_branches'} and parent_branch_id:
            add_operation(
                'merge_with_replacement',
                parent_branch_id=parent_branch_id,
                branch_id=branch_id,
                phase_id=branch.get('phase_id'),
            )
    for obligation_identity, obligation in sorted(candidate_obligation_index.items()):
        if obligation_identity not in base_obligation_index:
            add_operation(
                'add_obligation',
                obligation_id=_obligation_id(obligation),
                obligation_collection=obligation.get('_obligation_collection'),
                phase_id=obligation.get('phase_id'),
                branch_id=obligation.get('branch_id'),
            )

    for change in semantic_changes:
        add_operation(
            f"change_{_status(change.get('type'))}_semantics",
            record_id=change.get('id'),
            phase_id=change.get('phase_id'),
            branch_id=change.get('branch_id'),
            changed_fields=change.get('changed_fields'),
        )
    if top_level_semantic_changes:
        add_operation(
            'change_graph_semantics',
            changed_fields=changed_top_level_fields,
        )
    if derived_projection_changes:
        add_operation(
            'change_derived_graph_projection',
            changed_fields=changed_derived_projection_fields,
        )

    base_phase_edges = _dependency_edges(base_phases, _phase_id)
    candidate_phase_edges = _dependency_edges(candidate_phases, _phase_id)
    base_branch_edges = _dependency_edges(base_branches, _branch_id)
    candidate_branch_edges = _dependency_edges(candidate_branches, _branch_id)
    added_edges = sorted((candidate_phase_edges | candidate_branch_edges) - (base_phase_edges | base_branch_edges))
    removed_edges = sorted((base_phase_edges | base_branch_edges) - (candidate_phase_edges | candidate_branch_edges))
    added_edges_by_target: dict[str, list[str]] = {}
    for target_id, source_id in added_edges:
        added_edges_by_target.setdefault(target_id, []).append(source_id)
    for target_id, source_id in added_edges:
        add_operation('add_dependency', target_id=target_id, source_id=source_id)
    for target_id, source_id in removed_edges:
        replacement_source_ids = sorted(added_edges_by_target.get(target_id) or [])
        if replacement_source_ids:
            add_operation(
                'rebind_dependency',
                target_id=target_id,
                removed_source_id=source_id,
                replacement_source_ids=replacement_source_ids,
            )
        else:
            add_operation('remove_dependency', target_id=target_id, source_id=source_id)

    removed_without_preservation: list[dict[str, str]] = []
    for phase_id in sorted(set(base_phase_index) - set(candidate_phase_index)):
        if not _lineaged_replacement_exists(candidate_phases, phase_id, parent_keys=('parent_phase_id',)):
            removed_without_preservation.append({'type': 'phase', 'id': phase_id, 'reason': 'missing_successor_lineage'})
            add_operation('remove_phase', phase_id=phase_id)
    for branch_id in sorted(set(base_branch_index) - set(candidate_branch_index)):
        if not _lineaged_replacement_exists(candidate_branches, branch_id, parent_keys=('parent_branch_id', 'parent_phase_id')):
            removed_without_preservation.append({'type': 'branch', 'id': branch_id, 'reason': 'missing_successor_lineage'})
            add_operation('remove_branch', branch_id=branch_id)
    for obligation_identity in sorted(set(base_obligation_index) - set(candidate_obligation_index)):
        base_obligation = base_obligation_index[obligation_identity]
        obligation_id = _obligation_id(base_obligation)
        obligation_collection = _clean_text(base_obligation.get('_obligation_collection'))
        replacements = [
            item
            for item in candidate_obligations
            if _clean_text(item.get('_obligation_collection')) == obligation_collection
        ]
        if not _lineaged_replacement_exists(
            replacements,
            obligation_id,
            parent_keys=('parent_obligation_id',),
        ):
            removed_without_preservation.append(
                {
                    'type': 'obligation',
                    'id': obligation_id,
                    'obligation_collection': obligation_collection,
                    'reason': 'missing_successor_lineage',
                }
            )
            add_operation(
                'remove_obligation',
                obligation_id=obligation_id,
                obligation_collection=obligation_collection,
            )

    base_refs = _collect_artifact_refs(base)
    candidate_refs = _collect_artifact_refs(candidate)
    operation_counts: dict[str, int] = {}
    for operation in operations:
        op = _clean_text(operation.get('op'))
        operation_counts[op] = operation_counts.get(op, 0) + 1
    meaningful_operations = [
        operation
        for operation in operations
        if not _clean_text(operation.get('op')).startswith('preserve_')
    ]
    rebound_removed_edges = [
        {
            'target_id': target_id,
            'source_id': source_id,
            'replacement_source_ids': sorted(added_edges_by_target.get(target_id) or []),
        }
        for target_id, source_id in removed_edges
        if added_edges_by_target.get(target_id)
    ]
    lost_dependency_edges = [
        {'target_id': target_id, 'source_id': source_id}
        for target_id, source_id in removed_edges
    ]
    diff = {
        'kind': GRAPH_REBASE_DIFF_KIND,
        'base_graph_digest': stable_graph_digest(base),
        'candidate_graph_digest': stable_graph_digest(candidate),
        'operation_counts': operation_counts,
        'operations': operations,
        'meaningful_change_count': len(meaningful_operations),
        'meaningful_operations': meaningful_operations,
        'semantic_changes': semantic_changes,
        'top_level_semantic_changes': top_level_semantic_changes,
        'derived_projection_changes': derived_projection_changes,
        'hidden_failure_visibility_losses': hidden_failure_visibility_losses,
        'identity_map': {
            'phases': {phase_id: phase_id for phase_id in sorted(set(base_phase_index) & set(candidate_phase_index))},
            'branches': {branch_id: branch_id for branch_id in sorted(set(base_branch_index) & set(candidate_branch_index))},
            'obligations': {
                obligation_id: obligation_id
                for obligation_id in sorted(set(base_obligation_index) & set(candidate_obligation_index))
            },
        },
        'preserved_ids': {
            'phases': sorted(set(base_phase_index) & set(candidate_phase_index)),
            'branches': sorted(set(base_branch_index) & set(candidate_branch_index)),
            'obligations': sorted(set(base_obligation_index) & set(candidate_obligation_index)),
            'artifact_refs': sorted(set(base_refs) & set(candidate_refs)),
        },
        'added_ids': {
            'phases': sorted(set(candidate_phase_index) - set(base_phase_index)),
            'branches': sorted(set(candidate_branch_index) - set(base_branch_index)),
            'obligations': sorted(set(candidate_obligation_index) - set(base_obligation_index)),
        },
        'removed_ids': {
            'phases': sorted(set(base_phase_index) - set(candidate_phase_index)),
            'branches': sorted(set(base_branch_index) - set(candidate_branch_index)),
            'obligations': sorted(set(base_obligation_index) - set(candidate_obligation_index)),
        },
        'added_dependency_edges': [
            {'target_id': target_id, 'source_id': source_id}
            for target_id, source_id in added_edges
        ],
        'removed_dependency_edges': [
            {'target_id': target_id, 'source_id': source_id}
            for target_id, source_id in removed_edges
        ],
        'rebound_dependency_edges': rebound_removed_edges,
        'lost_dependency_edges': lost_dependency_edges,
        'superseded_with_evidence': [
            operation for operation in operations if operation.get('op') == 'supersede_with_replacement'
        ],
        'waived_with_evidence': [],
        'removed_without_preservation': removed_without_preservation,
        'artifact_ref_changes': {
            'carried': sorted(set(base_refs) & set(candidate_refs)),
            'added': sorted(set(candidate_refs) - set(base_refs)),
            'lost': sorted(set(base_refs) - set(candidate_refs)),
        },
        'obligation_changes': {
            'preserved': sorted(set(base_obligation_index) & set(candidate_obligation_index)),
            'added': sorted(set(candidate_obligation_index) - set(base_obligation_index)),
            'removed_without_preservation': [
                item for item in removed_without_preservation if item.get('type') == 'obligation'
            ],
        },
        'review_obligation_changes': {
            'carried_global_semantic_review': bool(
                isinstance(base.get('global_semantic_closure_review'), Mapping)
                and isinstance(candidate.get('global_semantic_closure_review'), Mapping)
            ),
        },
    }
    return _json_safe(diff)


def build_graph_rebase_preservation_proof(
    base_graph: Mapping[str, Any],
    candidate_graph: Mapping[str, Any],
    diff: Mapping[str, Any],
    *,
    closure_review: Optional[Mapping[str, Any]] = None,
    intent_lens_review: Optional[Mapping[str, Any]] = None,
    artifact_registry: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Prove that a candidate graph preserves current obligations and evidence."""

    base = base_graph if isinstance(base_graph, Mapping) else {}
    candidate = candidate_graph if isinstance(candidate_graph, Mapping) else {}
    blocked_reasons: list[str] = []
    lost_items: list[dict[str, Any]] = []
    base_intent_ids = _required_obligation_ids(base, 'intent_obligations')
    candidate_intent_ids = _required_obligation_ids(candidate, 'intent_obligations')
    base_output_ids = _required_obligation_ids(base, 'output_obligations')
    candidate_output_ids = _required_obligation_ids(candidate, 'output_obligations')

    for obligation_id in base_intent_ids:
        if obligation_id not in candidate_intent_ids:
            _append_unique(blocked_reasons, 'lost_required_intent_obligation')
            lost_items.append({'type': 'intent_obligation', 'id': obligation_id})
    for obligation_id in base_output_ids:
        if obligation_id not in candidate_output_ids:
            _append_unique(blocked_reasons, 'lost_required_output_obligation')
            lost_items.append({'type': 'output_obligation', 'id': obligation_id})

    base_refs = _collect_artifact_refs(
        {'phases': _records(base, 'phases'), 'output_obligations': _records(base, 'output_obligations')}
    )
    candidate_refs = _collect_artifact_refs(
        {'phases': _records(candidate, 'phases'), 'output_obligations': _records(candidate, 'output_obligations')}
    )
    lost_refs = sorted(set(base_refs) - set(candidate_refs))
    for ref in lost_refs:
        _append_unique(blocked_reasons, 'lost_artifact_ref')
        lost_items.append({'type': 'artifact_ref', 'id': ref})

    for base_record in _target_bound_phase_records(base):
        if not _candidate_preserves_target_bound_phase(candidate, base_record):
            _append_unique(blocked_reasons, 'target_bound_repair_target_lost')
            lost_items.append(
                {
                    'type': 'target_bound_repair',
                    'id': base_record.get('phase_id'),
                    'repair_target_path': base_record.get('repair_target_path'),
                }
            )

    if diff.get('removed_without_preservation'):
        _append_unique(blocked_reasons, 'removed_without_preservation')
        for item in diff.get('removed_without_preservation') or []:
            if isinstance(item, Mapping):
                lost_items.append(dict(item))

    if diff.get('lost_dependency_edges'):
        _append_unique(blocked_reasons, 'lost_dependency_edge')
        for item in diff.get('lost_dependency_edges') or []:
            if isinstance(item, Mapping):
                lost_items.append({'type': 'dependency', **dict(item)})

    if diff.get('semantic_changes'):
        _append_unique(blocked_reasons, 'changed_preserved_record_meaning')
        for item in diff.get('semantic_changes') or []:
            if isinstance(item, Mapping):
                lost_items.append({'type': 'semantic_change', **dict(item)})

    if diff.get('top_level_semantic_changes'):
        _append_unique(blocked_reasons, 'changed_preserved_graph_meaning')
        for item in diff.get('top_level_semantic_changes') or []:
            if isinstance(item, Mapping):
                lost_items.append({'type': 'graph_semantic_change', **dict(item)})

    if diff.get('hidden_failure_visibility_losses'):
        _append_unique(blocked_reasons, 'hidden_failure_visibility_lost')
        for item in diff.get('hidden_failure_visibility_losses') or []:
            if isinstance(item, Mapping):
                lost_items.append({'type': 'hidden_failure_visibility', **dict(item)})

    base_review = base.get('global_semantic_closure_review')
    candidate_review = candidate.get('global_semantic_closure_review')
    if (
        isinstance(base_review, Mapping)
        and bool(base_review.get('semantic_review_required'))
        and not isinstance(candidate_review, Mapping)
    ):
        _append_unique(blocked_reasons, 'lost_review_obligation')
        lost_items.append({'type': 'review_obligation', 'id': 'global_semantic_closure_review'})

    status = 'blocked' if blocked_reasons else 'passed'
    proof = {
        'kind': GRAPH_REBASE_PRESERVATION_PROOF_KIND,
        'status': status,
        'base_graph_digest': stable_graph_digest(base),
        'candidate_graph_digest': stable_graph_digest(candidate),
        'intent_anchor_preserved': 'lost_required_intent_obligation' not in blocked_reasons,
        'intent_obligation_summary': {
            'base_required_ids': base_intent_ids,
            'candidate_required_ids': candidate_intent_ids,
            'preserved': sorted(set(base_intent_ids) & set(candidate_intent_ids)),
            'lost': sorted(set(base_intent_ids) - set(candidate_intent_ids)),
            'added': sorted(set(candidate_intent_ids) - set(base_intent_ids)),
        },
        'output_obligation_summary': {
            'base_required_ids': base_output_ids,
            'candidate_required_ids': candidate_output_ids,
            'preserved': sorted(set(base_output_ids) & set(candidate_output_ids)),
            'lost': sorted(set(base_output_ids) - set(candidate_output_ids)),
            'added': sorted(set(candidate_output_ids) - set(base_output_ids)),
        },
        'phase_lineage_summary': {
            'preserved': (diff.get('preserved_ids') or {}).get('phases') or [],
            'added': (diff.get('added_ids') or {}).get('phases') or [],
        },
        'branch_lineage_summary': {
            'preserved': (diff.get('preserved_ids') or {}).get('branches') or [],
            'added': (diff.get('added_ids') or {}).get('branches') or [],
        },
        'dependency_preservation_summary': {
            'rebound_edges': diff.get('rebound_dependency_edges') or [],
            'removed_edges': diff.get('removed_dependency_edges') or [],
            'lost_edges': diff.get('lost_dependency_edges') or [],
            'removed_without_preservation': [
                item for item in diff.get('removed_without_preservation') or []
                if isinstance(item, Mapping) and item.get('type') == 'dependency'
            ],
        },
        'semantic_preservation_summary': {
            'changed_records': diff.get('semantic_changes') or [],
            'changed_graph_contracts': diff.get('top_level_semantic_changes') or [],
            'hidden_failure_visibility_losses': diff.get('hidden_failure_visibility_losses') or [],
        },
        'artifact_ref_preservation_summary': {
            'base': sorted(set(base_refs)),
            'candidate': sorted(set(candidate_refs)),
            'carried': sorted(set(base_refs) & set(candidate_refs)),
            'lost': lost_refs,
            'added': sorted(set(candidate_refs) - set(base_refs)),
        },
        'review_obligation_summary': diff.get('review_obligation_changes') or {},
        'closure_review_summary': dict(closure_review or {}) if isinstance(closure_review, Mapping) else {},
        'intent_lens_review_summary': dict(intent_lens_review or {}) if isinstance(intent_lens_review, Mapping) else {},
        'artifact_registry_summary': dict(artifact_registry or {}) if isinstance(artifact_registry, Mapping) else {},
        'lost_items': lost_items,
        'blocked_reasons': blocked_reasons,
    }
    return _json_safe(proof)


def _proposal_has_learning_only_authority(source: Any, evidence_refs: Sequence[str]) -> bool:
    source_status = _status(source)
    if source_status in _LEARNING_SOURCES:
        return True
    if not evidence_refs:
        return False
    return all(_status(ref).startswith('accepted_learning') or _status(ref).startswith('self_learning') for ref in evidence_refs)


def _has_route_health_or_provider_authority(payload: Mapping[str, Any]) -> bool:
    authorization = (
        payload.get('graph_rebase_authorization')
        if isinstance(payload.get('graph_rebase_authorization'), Mapping)
        else {}
    )
    evidence_surface = {
        'source': payload.get('source'),
        'reason': payload.get('reason'),
        'evidence_refs': payload.get('evidence_refs'),
        'authority': payload.get('authority'),
        'authorization': {
            'authority': authorization.get('authority'),
            'reason': authorization.get('reason'),
            'evidence_refs': authorization.get('evidence_refs'),
        },
    }
    text = json.dumps(_json_safe(evidence_surface), sort_keys=True).lower()
    return any(token in text for token in _ROUTE_HEALTH_EVIDENCE_TOKENS)


def _partial_rebase_scope_violations(
    proposal: Mapping[str, Any],
    diff: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return meaningful candidate changes that escape a declared partial scope."""

    if _status(proposal.get('requested_rebase_class')) != 'partial_subtree_rebase':
        return []
    phase_ids = set(_clean_string_list(proposal.get('scope_phase_ids') or []))
    branch_ids = set(_clean_string_list(proposal.get('scope_branch_ids') or []))
    root_ids = set(_clean_string_list(proposal.get('scope_root_ids') or []))
    phase_ids.update(root_ids)
    branch_ids.update(root_ids)
    violations: list[dict[str, Any]] = []
    if proposal.get('preserve_outside_scope') is not True:
        violations.append({'reason': 'preserve_outside_scope_not_confirmed'})
    if not phase_ids and not branch_ids:
        violations.append({'reason': 'partial_scope_ids_missing'})
    if diff.get('top_level_semantic_changes'):
        violations.append({'reason': 'top_level_graph_semantics_changed'})

    removed_ids = diff.get('removed_ids') if isinstance(diff.get('removed_ids'), Mapping) else {}
    for removed_phase_id in _clean_string_list(removed_ids.get('phases') or []):
        if removed_phase_id not in phase_ids:
            violations.append(
                {
                    'reason': 'removed_phase_outside_declared_scope',
                    'phase_id': removed_phase_id,
                }
            )
    for removed_branch_id in _clean_string_list(removed_ids.get('branches') or []):
        if removed_branch_id not in branch_ids:
            violations.append(
                {
                    'reason': 'removed_branch_outside_declared_scope',
                    'branch_id': removed_branch_id,
                }
            )

    for operation in diff.get('meaningful_operations') or []:
        if not isinstance(operation, Mapping):
            continue
        op = _status(operation.get('op'))
        phase_refs = {
            _clean_text(value)
            for key, value in operation.items()
            if 'phase_id' in _status(key)
            for value in (
                _clean_string_list(value)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
                else [_clean_text(value)]
            )
            if _clean_text(value)
        }
        branch_refs = {
            _clean_text(value)
            for key, value in operation.items()
            if 'branch_id' in _status(key)
            for value in (
                _clean_string_list(value)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
                else [_clean_text(value)]
            )
            if _clean_text(value)
        }
        phase_id = _clean_text(operation.get('phase_id'))
        branch_id = _clean_text(operation.get('branch_id'))
        record_id = _clean_text(operation.get('record_id'))
        target_id = _clean_text(operation.get('target_id'))
        obligation_id = _clean_text(operation.get('obligation_id'))
        inside = True
        if 'dependency' in op:
            inside = target_id in phase_ids or target_id in branch_ids
        elif phase_refs or branch_refs:
            phases_inside = all(item in phase_ids for item in phase_refs)
            branches_inside = all(item in branch_ids for item in branch_refs)
            inside = phases_inside and branches_inside
        elif 'phase' in op:
            candidate_id = phase_id or record_id
            inside = candidate_id in phase_ids
        elif 'branch' in op:
            candidate_id = branch_id or record_id
            inside = candidate_id in branch_ids or phase_id in phase_ids
        elif 'obligation' in op:
            inside = (
                phase_id in phase_ids
                or branch_id in branch_ids
                or obligation_id in root_ids
                or record_id in root_ids
            )
        if not inside:
            violations.append({'reason': 'change_outside_declared_scope', 'operation': dict(operation)})
    return violations


def _partial_rebase_candidate_record_index(
    candidate_graph: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return candidate branch records and phase-to-branch identity truth."""

    records: dict[str, dict[str, Any]] = {}
    phase_to_branch: dict[str, str] = {}
    for collection in ('phases', 'downstream_branches'):
        for raw_record in _records(candidate_graph, collection):
            branch_id = _branch_id(raw_record)
            phase_id = _phase_id(raw_record)
            if not branch_id:
                continue
            existing = records.get(branch_id, {})
            source_records = [
                dict(item)
                for item in existing.get('_graph_rebase_source_records') or []
                if isinstance(item, Mapping)
            ]
            records[branch_id] = {
                **existing,
                **raw_record,
                '_graph_rebase_source_records': [*source_records, dict(raw_record)],
            }
            if phase_id:
                phase_to_branch[phase_id] = branch_id
    return records, phase_to_branch


def _partial_rebase_owed_branch_ids(
    proposal: Mapping[str, Any],
    diff: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Derive executable branch/phase/obligation scope from runtime diff truth."""

    candidate = (
        proposal.get('candidate_graph')
        if isinstance(proposal.get('candidate_graph'), Mapping)
        else {}
    )
    records, phase_to_branch = _partial_rebase_candidate_record_index(candidate)
    scope_phase_ids = set(_clean_string_list(proposal.get('scope_phase_ids') or []))
    scope_branch_ids = set(_clean_string_list(proposal.get('scope_branch_ids') or []))
    scope_root_ids = set(_clean_string_list(proposal.get('scope_root_ids') or []))
    scope_phase_ids.update(scope_root_ids)
    scope_branch_ids.update(scope_root_ids)
    owed_branch_ids: list[str] = []
    owed_phase_ids: list[str] = []
    owed_obligation_ids: list[str] = []

    def add_branch(raw_id: Any) -> None:
        record_id = _clean_text(raw_id)
        branch_id = record_id if record_id in records else phase_to_branch.get(record_id, '')
        if not branch_id or branch_id not in records:
            return
        record = records[branch_id]
        phase_id = _phase_id(record)
        if (
            branch_id not in scope_branch_ids
            and phase_id not in scope_phase_ids
            and branch_id not in scope_root_ids
            and phase_id not in scope_root_ids
        ):
            return
        if branch_id not in owed_branch_ids:
            owed_branch_ids.append(branch_id)
        if phase_id and phase_id not in owed_phase_ids:
            owed_phase_ids.append(phase_id)

    for operation in diff.get('meaningful_operations') or []:
        if not isinstance(operation, Mapping):
            continue
        op = _status(operation.get('op'))
        if op.startswith('remove_'):
            continue
        if op in {
            'add_branch',
            'change_branch_semantics',
            'split_branch',
            'merge_with_replacement',
            'supersede_with_replacement',
        }:
            add_branch(operation.get('branch_id'))
        if op in {
            'add_phase',
            'change_phase_semantics',
            'split_branch',
            'merge_with_replacement',
            'supersede_with_replacement',
        }:
            add_branch(operation.get('phase_id'))
        if op in {'add_dependency', 'rebind_dependency', 'remove_dependency'}:
            add_branch(operation.get('target_id'))
        if op in {'add_obligation', 'change_obligation_semantics'}:
            obligation_id = _clean_text(operation.get('obligation_id') or operation.get('record_id'))
            if obligation_id and obligation_id not in owed_obligation_ids:
                owed_obligation_ids.append(obligation_id)
            add_branch(operation.get('branch_id') or operation.get('phase_id'))

    return sorted(owed_branch_ids), sorted(owed_phase_ids), sorted(owed_obligation_ids)


def _partial_rebase_branch_local_binding_review(
    record: Mapping[str, Any],
    *,
    root_phase_ids: set[str],
    root_prompt: str = '',
    root_prompt_digest: str = '',
) -> dict[str, Any]:
    branch_id = _branch_id(record)
    phase_id = _phase_id(record)
    capability = _status(record.get('capability'))
    execution_contract = (
        record.get('execution_contract')
        if isinstance(record.get('execution_contract'), Mapping)
        else {}
    )
    source_records = [
        dict(item)
        for item in record.get('_graph_rebase_source_records') or []
        if isinstance(item, Mapping)
    ] or [dict(record)]
    input_refs = [
        dict(item)
        for item in (
            execution_contract.get('input_refs')
            or record.get('input_refs')
            or []
        )
        if isinstance(item, Mapping)
    ]
    root_prompt_fallback = bool(
        execution_contract.get('root_scoped') is True
        or execution_contract.get('allow_root_prompt') is True
        or _status(execution_contract.get('execution_scope'))
        in {'root', 'root_scoped', 'whole_request', 'original_prompt'}
    )
    for ref in input_refs:
        ref_kind = _status(ref.get('kind'))
        ref_phase_id = _clean_text(ref.get('phase_id') or ref.get('ref'))
        if ref_kind == 'user_prompt' or (
            ref_kind == 'phase_output' and ref_phase_id in root_phase_ids
        ):
            root_prompt_fallback = True
    for source_record in source_records:
        source_contract = (
            source_record.get('execution_contract')
            if isinstance(source_record.get('execution_contract'), Mapping)
            else {}
        )
        if (
            source_contract.get('root_scoped') is True
            or source_contract.get('allow_root_prompt') is True
            or _status(source_contract.get('execution_scope'))
            in {'root', 'root_scoped', 'whole_request', 'original_prompt'}
        ):
            root_prompt_fallback = True
        for ref in source_contract.get('input_refs') or source_record.get('input_refs') or []:
            if not isinstance(ref, Mapping):
                continue
            ref_kind = _status(ref.get('kind'))
            ref_phase_id = _clean_text(ref.get('phase_id') or ref.get('ref'))
            if ref_kind == 'user_prompt' or (
                ref_kind == 'phase_output' and ref_phase_id in root_phase_ids
            ):
                root_prompt_fallback = True

    content_payload = _clean_text(record.get('content_payload'))
    artifact_prompt = _clean_text(record.get('artifact_prompt'))
    batch_prompts = _clean_string_list(record.get('batch_prompts') or [])
    payload_sources = {
        _status(source_record.get(key))
        for source_record in source_records
        for key in ('content_payload_source', 'artifact_prompt_source')
    }
    forbidden_source_tokens = {
        'assistant_output',
        'assistant_output_claim',
        'current_phase_output',
        'original_prompt',
        'request_prompt',
        'root_prompt',
        'user_prompt',
    }
    if payload_sources & forbidden_source_tokens:
        root_prompt_fallback = True

    def prompt_carrier_values(value: Any) -> list[str]:
        if isinstance(value, Mapping):
            values: list[str] = []
            for child in value.values():
                values.extend(prompt_carrier_values(child))
            return values
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            values = []
            for child in value:
                values.extend(prompt_carrier_values(child))
            return values
        text = _clean_text(value)
        return [text] if text else []

    # These fields can become, wrap, constrain, or otherwise influence the
    # actual backend prompt during Late Fill.  Partial rebase must therefore
    # prove all of them branch-local, not merely the three primary payloads.
    prompt_carrier_keys = (
        'content_payload',
        'artifact_prompt',
        'batch_prompts',
        'phase_summary',
        'stage_direction',
        'instruct',
        'prompt',
        '_prompt_hint',
        'instructions',
        'review_criteria',
        'semantic_review_criteria',
        'controlled_attention_question',
    )
    local_prompt_values: list[str] = []
    for source_record in source_records:
        source_contract = (
            source_record.get('execution_contract')
            if isinstance(source_record.get('execution_contract'), Mapping)
            else {}
        )
        for key in prompt_carrier_keys:
            local_prompt_values.extend(prompt_carrier_values(source_record.get(key)))
            local_prompt_values.extend(prompt_carrier_values(source_contract.get(key)))
    for local_prompt in local_prompt_values:
        local_prompt_digest = stable_graph_rebase_prompt_digest(local_prompt)
        if root_prompt_digest and local_prompt_digest == root_prompt_digest:
            root_prompt_fallback = True
        if graph_rebase_prompt_contains_root(local_prompt, root_prompt):
            root_prompt_fallback = True

    dependency_refs = [
        item
        for item in input_refs
        if _status(item.get('kind')) in {'phase_output', 'runtime_evidence', 'selected_reference'}
        and _clean_text(item.get('phase_id') or item.get('ref')) not in root_phase_ids
    ]
    if capability == 'image_generation':
        local_binding = bool(artifact_prompt or batch_prompts)
    elif capability in {'chat', 'embedding', 'text_to_speech'}:
        local_binding = bool(content_payload)
    elif capability in {'speech_to_text', 'vision_analysis'}:
        local_binding = bool(dependency_refs)
    else:
        local_binding = bool(content_payload or artifact_prompt or batch_prompts)

    status = _status(record.get('status') or record.get('resolution'))
    executable_state = status not in {
        'cancelled',
        'discarded',
        'fulfilled',
        'omitted',
        'rejected',
        'reserved',
        'superseded',
        'waived',
    }
    blocked_reasons: list[str] = []
    if not capability:
        _append_unique(blocked_reasons, 'partial_rebase_branch_capability_missing')
    if not executable_state:
        _append_unique(blocked_reasons, 'partial_rebase_owed_branch_not_executable')
    if not local_binding:
        _append_unique(blocked_reasons, 'partial_rebase_branch_local_payload_missing')
    if root_prompt_fallback:
        _append_unique(blocked_reasons, 'partial_rebase_root_prompt_fallback_forbidden')
    return _json_safe(
        {
            'branch_id': branch_id,
            'phase_id': phase_id,
            'capability': capability,
            'status': 'passed' if not blocked_reasons else 'blocked',
            'local_binding_present': local_binding,
            'root_prompt_fallback': root_prompt_fallback,
            'input_refs': input_refs,
            'blocked_reasons': blocked_reasons,
        }
    )


def build_graph_rebase_execution_contract_proof(
    request_phase_graph: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    root_prompt: str = '',
) -> dict[str, Any]:
    """Prove an accepted partial candidate has exact branch-local execution inputs."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    proposal_payload = proposal if isinstance(proposal, Mapping) else {}
    candidate = (
        proposal_payload.get('candidate_graph')
        if isinstance(proposal_payload.get('candidate_graph'), Mapping)
        else {}
    )
    requested_class = _status(proposal_payload.get('requested_rebase_class'))
    diff = build_graph_rebase_diff(graph, candidate)
    owed_branch_ids, owed_phase_ids, owed_obligation_ids = _partial_rebase_owed_branch_ids(
        proposal_payload,
        diff,
    )
    candidate_records, _phase_to_branch = _partial_rebase_candidate_record_index(candidate)
    current_phase_id = _clean_text(graph.get('current_phase_id'))
    root_prompt_guard = (
        proposal_payload.get('root_prompt_guard')
        if isinstance(proposal_payload.get('root_prompt_guard'), Mapping)
        else {}
    )
    root_prompt_digest = _clean_text(root_prompt_guard.get('digest'))
    root_prompt_guard_valid = bool(
        root_prompt_guard.get('kind') == 'ollmo.graph_rebase_root_prompt_guard'
        and _status(root_prompt_guard.get('authority')) == 'runtime_request_truth'
        and root_prompt_digest
    )
    root_phase_ids = {current_phase_id} if current_phase_id else set()
    for phase in _records(graph, 'phases'):
        phase_id = _phase_id(phase)
        if phase_id and not _branch_id(phase):
            root_phase_ids.add(phase_id)
    branch_reviews = [
        _partial_rebase_branch_local_binding_review(
            candidate_records.get(branch_id, {}),
            root_phase_ids=root_phase_ids,
            root_prompt=root_prompt,
            root_prompt_digest=root_prompt_digest,
        )
        for branch_id in owed_branch_ids
    ]
    blocked_reasons: list[str] = []
    if requested_class != 'partial_subtree_rebase':
        _append_unique(blocked_reasons, 'apply_reviewed_partial_rebase_only')
    if requested_class == 'partial_subtree_rebase' and not root_prompt_guard_valid:
        _append_unique(blocked_reasons, 'partial_rebase_root_prompt_guard_missing')
    current_root_prompt_digest = stable_graph_rebase_prompt_digest(root_prompt)
    if requested_class == 'partial_subtree_rebase' and not current_root_prompt_digest:
        _append_unique(
            blocked_reasons,
            'partial_rebase_current_root_prompt_truth_unavailable',
        )
    if (
        requested_class == 'partial_subtree_rebase'
        and current_root_prompt_digest
        and root_prompt_digest != current_root_prompt_digest
    ):
        _append_unique(blocked_reasons, 'partial_rebase_root_prompt_guard_mismatch')
    for violation in _partial_rebase_scope_violations(proposal_payload, diff):
        _append_unique(
            blocked_reasons,
            _clean_text(violation.get('reason')) or 'partial_rebase_changes_outside_scope',
        )
    if not owed_branch_ids:
        _append_unique(blocked_reasons, 'partial_rebase_owed_branch_scope_empty')
    for review in branch_reviews:
        for reason in _clean_string_list(review.get('blocked_reasons') or []):
            _append_unique(blocked_reasons, reason)
    root_prompt_fallback_branch_ids = [
        _clean_text(review.get('branch_id'))
        for review in branch_reviews
        if review.get('root_prompt_fallback') is True
        and _clean_text(review.get('branch_id'))
    ]
    missing_local_payload_branch_ids = [
        _clean_text(review.get('branch_id'))
        for review in branch_reviews
        if review.get('local_binding_present') is not True
        and _clean_text(review.get('branch_id'))
    ]
    return _json_safe(
        {
            'kind': GRAPH_REBASE_EXECUTION_CONTRACT_PROOF_KIND,
            'status': 'passed' if not blocked_reasons else 'blocked',
            'requested_rebase_class': requested_class,
            'base_graph_digest': stable_graph_digest(graph),
            'candidate_graph_digest': stable_graph_digest(candidate),
            'diff_digest': _stable_digest(diff, prefix='graph-rebase-diff-'),
            'scope_digest': _stable_digest(
                {
                    'scope_root_ids': _clean_string_list(proposal_payload.get('scope_root_ids') or []),
                    'scope_phase_ids': _clean_string_list(proposal_payload.get('scope_phase_ids') or []),
                    'scope_branch_ids': _clean_string_list(proposal_payload.get('scope_branch_ids') or []),
                    'owed_branch_ids': owed_branch_ids,
                },
                prefix='graph-rebase-scope-',
            ),
            'owed_branch_ids': owed_branch_ids,
            'owed_phase_ids': owed_phase_ids,
            'owed_obligation_ids': owed_obligation_ids,
            'branch_reviews': branch_reviews,
            'missing_local_payload_branch_ids': missing_local_payload_branch_ids,
            'root_prompt_fallback_branch_ids': root_prompt_fallback_branch_ids,
            'root_prompt_guard_digest': root_prompt_digest,
            'root_prompt_guard_checked': bool(
                root_prompt_guard_valid
                and current_root_prompt_digest
                and root_prompt_digest == current_root_prompt_digest
            ),
            'blocked_reasons': blocked_reasons,
            'runtime_effect': 'none',
        }
    )


def validate_graph_rebase_proposal(
    proposal: Mapping[str, Any],
    *,
    request_phase_graph: Mapping[str, Any],
    closure_review: Optional[Mapping[str, Any]] = None,
    artifact_payload: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
    intent_lens_review: Optional[Mapping[str, Any]] = None,
    runtime_gate_reasons: Optional[Sequence[str]] = None,
    trusted_authorization: Optional[Mapping[str, Any]] = None,
    root_prompt: str = '',
) -> dict[str, Any]:
    """Validate a graph rebase proposal without applying it."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    proposal_payload = proposal if isinstance(proposal, Mapping) else {}
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    def add_check(name: str, status: str, reason: str) -> None:
        checks.append({'check': name, 'status': status, 'reason': reason})
        if status != 'passed':
            _append_unique(reasons, reason)

    if not proposal_payload:
        add_check('proposal_shape', 'rejected', 'proposal_missing')
    elif proposal_payload.get('kind') != GRAPH_REBASE_PROPOSAL_KIND:
        add_check('proposal_shape', 'rejected', 'proposal_kind_mismatch')
    else:
        add_check('proposal_shape', 'passed', 'proposal_schema_present')

    proposal_id = _clean_text(proposal_payload.get('proposal_id'))
    if proposal_id:
        add_check('proposal_id', 'passed', 'proposal_id_present')
    else:
        add_check('proposal_id', 'rejected', 'proposal_id_missing')

    if proposal_payload.get('allowed_use') == PROPOSAL_ALLOWED_USE:
        add_check('allowed_use', 'passed', 'proposal_only_until_validated')
    else:
        add_check('allowed_use', 'rejected', 'proposal_allowed_use_mismatch')
    if proposal_payload.get('forbidden_use') == PROPOSAL_FORBIDDEN_USE:
        add_check('forbidden_use', 'passed', 'direct_mutation_forbidden')
    else:
        add_check('forbidden_use', 'rejected', 'proposal_forbidden_use_mismatch')

    base_digest = stable_graph_digest(graph)
    if proposal_payload.get('base_graph_digest') == base_digest:
        add_check('base_graph_digest', 'passed', 'base_graph_digest_matches_runtime_truth')
    else:
        add_check('base_graph_digest', 'rejected', 'base_graph_digest_mismatch')

    candidate = proposal_payload.get('candidate_graph')
    if isinstance(candidate, Mapping) and (_records(candidate, 'phases') or _records(candidate, 'downstream_branches')):
        add_check('candidate_graph', 'passed', 'candidate_graph_present')
    else:
        add_check('candidate_graph', 'rejected', 'candidate_graph_missing_or_empty')
        candidate = {}

    duplicate_ids = _candidate_duplicate_id_summary(candidate)
    if duplicate_ids:
        add_check(
            'candidate_graph_unique_ids',
            'rejected',
            'candidate_graph_duplicate_record_id',
        )
    else:
        add_check(
            'candidate_graph_unique_ids',
            'passed',
            'candidate_graph_record_ids_unique',
        )
    dangling_dependency_edges = _candidate_dangling_dependency_edges(candidate, graph)
    if dangling_dependency_edges:
        add_check(
            'candidate_graph_dependency_closure',
            'rejected',
            'candidate_graph_dangling_dependency_source',
        )
    else:
        add_check(
            'candidate_graph_dependency_closure',
            'passed',
            'candidate_graph_dependencies_resolve',
        )
    projection_consistency_issues = _candidate_projection_consistency_issues(candidate)
    if projection_consistency_issues:
        add_check(
            'candidate_graph_projection_consistency',
            'rejected',
            'candidate_graph_derived_projection_inconsistent',
        )
    else:
        add_check(
            'candidate_graph_projection_consistency',
            'passed',
            'candidate_graph_derived_projection_consistent',
        )

    computed_candidate_digest = stable_graph_digest(candidate)
    if proposal_payload.get('candidate_graph_digest') == computed_candidate_digest:
        add_check('candidate_graph_digest', 'passed', 'candidate_graph_digest_matches_candidate')
    else:
        add_check('candidate_graph_digest', 'rejected', 'candidate_graph_digest_mismatch')

    forbidden_candidate_bookkeeping = sorted(
        key
        for key in _GRAPH_REBASE_VOLATILE_KEYS
        if isinstance(candidate, Mapping) and candidate.get(key) not in (None, '', [], {})
    )
    if forbidden_candidate_bookkeeping:
        add_check(
            'candidate_graph_bookkeeping',
            'rejected',
            'candidate_graph_contains_rebase_bookkeeping',
        )
    else:
        add_check(
            'candidate_graph_bookkeeping',
            'passed',
            'candidate_graph_rebase_bookkeeping_absent',
        )

    for runtime_gate_reason in _clean_string_list(runtime_gate_reasons or []):
        add_check('runtime_gate', 'blocked', runtime_gate_reason)

    source_status = _status(proposal_payload.get('source'))
    evidence_refs = _clean_string_list(proposal_payload.get('evidence_refs') or [])
    if _proposal_has_learning_only_authority(proposal_payload.get('source'), evidence_refs):
        add_check('runtime_evidence', 'rejected', 'accepted_learning_not_rebase_authority')
    elif _has_route_health_or_provider_authority(proposal_payload):
        add_check('runtime_evidence', 'rejected', 'backend_route_health_signal_is_not_rebase_authority')
    elif source_status not in _RUNTIME_REBASE_SOURCES:
        add_check('runtime_evidence', 'blocked', 'runtime_rebase_authority_missing')
    elif not evidence_refs and not closure_review:
        add_check('runtime_evidence', 'blocked', 'runtime_rebase_evidence_missing')
    else:
        add_check('runtime_evidence', 'passed', 'runtime_or_closure_evidence_present')

    if _records(graph, 'intent_obligations') and not _records(candidate, 'intent_obligations'):
        add_check('intent_obligations', 'rejected', 'candidate_graph_missing_current_intent_obligations')
    else:
        add_check('intent_obligations', 'passed', 'candidate_carries_intent_obligation_surface')

    diff = build_graph_rebase_diff(graph, candidate)
    if int(diff.get('meaningful_change_count') or 0) > 0:
        add_check('meaningful_candidate_diff', 'passed', 'candidate_graph_has_meaningful_change')
    else:
        add_check('meaningful_candidate_diff', 'blocked', 'candidate_graph_has_no_meaningful_change')
    primary_meaningful_operations = [
        item
        for item in (diff.get('meaningful_operations') or [])
        if isinstance(item, Mapping)
        and _status(item.get('op')) != 'change_derived_graph_projection'
    ]
    if diff.get('derived_projection_changes') and not primary_meaningful_operations:
        add_check(
            'derived_projection_authority',
            'blocked',
            'derived_projection_changed_without_structural_diff',
        )

    partial_scope_violations = _partial_rebase_scope_violations(proposal_payload, diff)
    if partial_scope_violations:
        add_check('partial_scope_containment', 'blocked', 'partial_rebase_changes_outside_scope')
    elif _status(proposal_payload.get('requested_rebase_class')) == 'partial_subtree_rebase':
        add_check('partial_scope_containment', 'passed', 'partial_rebase_changes_contained_in_scope')
    proof = build_graph_rebase_preservation_proof(
        graph,
        candidate,
        diff,
        closure_review=closure_review,
        intent_lens_review=intent_lens_review,
        artifact_registry=artifact_payload,
    )
    if proof.get('status') == 'passed':
        add_check('preservation_proof', 'passed', 'candidate_preserves_runtime_truth')
    else:
        add_check('preservation_proof', 'blocked', 'candidate_preservation_proof_failed')
        for reason in _clean_string_list(proof.get('blocked_reasons')):
            _append_unique(reasons, reason)

    hard_rejection = any(item.get('status') == 'rejected' for item in checks)
    blocked = any(item.get('status') == 'blocked' for item in checks)
    status = 'rejected' if hard_rejection else 'blocked' if blocked else 'accepted'
    authorization_payload = (
        dict(trusted_authorization)
        if isinstance(trusted_authorization, Mapping)
        else {}
    )
    execution_contract_proof = build_graph_rebase_execution_contract_proof(
        graph,
        proposal_payload,
        root_prompt=root_prompt,
    )
    accepted_learning_summary: dict[str, Any] = {}
    if isinstance(accepted_learning_hints, Mapping):
        accepted_learning_summary = {
            'authority': 'soft_hint_only',
            'runtime_effect': 'none',
            'hint_count': accepted_learning_hints.get('hint_count'),
            'allowed_use': 'orientation_only_not_rebase_authority',
        }
    review = {
        'kind': GRAPH_REBASE_REVIEW_KIND,
        'review_id': _stable_digest({'proposal_id': proposal_id, 'checks': checks}, prefix='graph-rebase-review-'),
        'proposal_id': proposal_id,
        'status': status,
        'authority': 'runtime_rebase_validation',
        'target_response_id': proposal_payload.get('target_response_id'),
        'target_frame_id': proposal_payload.get('target_frame_id'),
        'target_graph_id': proposal_payload.get('target_graph_id') or base_digest,
        'base_graph_digest': base_digest,
        'candidate_graph_digest': stable_graph_digest(candidate),
        'proposal_digest': _stable_digest(proposal_payload, prefix='proposal-'),
        'runtime_evidence_refs': evidence_refs,
        'blocked_reasons': reasons,
        'validation_checks': checks,
        'candidate_duplicate_ids': duplicate_ids,
        'candidate_dangling_dependency_edges': dangling_dependency_edges,
        'candidate_projection_consistency_issues': projection_consistency_issues,
        'diff': diff,
        'partial_scope_violations': partial_scope_violations,
        'preservation_proof': proof,
        'execution_contract_proof': execution_contract_proof,
        'accepted_successor_graph': copy.deepcopy(dict(candidate)) if status == 'accepted' else {},
        'allowed_runtime_action': (
            'create_partial_successor_rebase'
            if status == 'accepted'
            and execution_contract_proof.get('status') == 'passed'
            else 'stage_rebase_only'
            if status == 'accepted'
            else 'none'
        ),
        'runtime_effect': 'none',
        'graph_rebase_authorization': authorization_payload,
        'accepted_learning_boundary': accepted_learning_summary,
        'source_proposal': proposal_payload,
    }
    return _json_safe(review)


def _graph_rebase_authorization_allows(review: Mapping[str, Any], level: str) -> bool:
    authorization = review.get('graph_rebase_authorization')
    if not isinstance(authorization, Mapping):
        return False
    if _status(authorization.get('status')) not in _ACCEPTED_AUTHORIZATION_STATUSES:
        return False
    if _status(authorization.get('authority')) not in _AUTHORIZED_REBASE_AUTHORITIES:
        return False
    if authorization.get('kind') != 'ollmo.graph_rebase_authorization':
        return False
    if _status(authorization.get('source')) != 'runtime_operator_registry':
        return False
    if not _clean_text(authorization.get('registry_record_id')):
        return False
    allowed = _clean_string_list(authorization.get('allowed_autonomy'))
    if level not in {_status(item) for item in allowed}:
        return False
    if not _clean_string_list(authorization.get('evidence_refs')):
        return False
    auth_proposal_id = _clean_text(authorization.get('proposal_id'))
    if not auth_proposal_id or auth_proposal_id != _clean_text(review.get('proposal_id')):
        return False
    auth_candidate_digest = _clean_text(authorization.get('candidate_graph_digest'))
    if not auth_candidate_digest or auth_candidate_digest != _clean_text(
        review.get('candidate_graph_digest')
    ):
        return False
    source_proposal = (
        review.get('source_proposal')
        if isinstance(review.get('source_proposal'), Mapping)
        else {}
    )
    requested_class = _status(source_proposal.get('requested_rebase_class'))
    if requested_class != 'partial_subtree_rebase':
        return False
    if _status(authorization.get('requested_rebase_class')) != requested_class:
        return False
    return True


def _lifecycle_runtime_effect(*, status: str, autonomy_level: str) -> str:
    if status in {'blocked', 'rejected'}:
        return 'none'
    if autonomy_level == 'shadow':
        return 'shadow_no_mutation'
    if autonomy_level == 'stage':
        return 'staged_no_executable_mutation'
    if autonomy_level == 'apply_reviewed':
        return 'successor_rebase_required'
    return 'none'


def build_graph_rebase_lifecycle(
    *,
    request_phase_graph: Mapping[str, Any],
    rebase_review: Mapping[str, Any],
    autonomy_level: str = 'stage',
    trusted_authorization: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a backend-visible graph rebase lifecycle record without applying it."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    review = dict(rebase_review) if isinstance(rebase_review, Mapping) else {}
    authorization_payload = (
        dict(trusted_authorization)
        if isinstance(trusted_authorization, Mapping)
        else {}
    )
    review['graph_rebase_authorization'] = authorization_payload
    source_proposal = (
        review.get('source_proposal')
        if isinstance(review.get('source_proposal'), Mapping)
        else {}
    )
    execution_contract_proof = (
        review.get('execution_contract_proof')
        if isinstance(review.get('execution_contract_proof'), Mapping)
        else {}
    )
    level = normalize_graph_rebase_autonomy(autonomy_level)
    review_status = _status(review.get('status'))
    blocked_reasons = _clean_string_list(review.get('blocked_reasons'))
    proposal_id = _clean_text(review.get('proposal_id'))
    review_id = _clean_text(review.get('review_id'))
    candidate_digest = _clean_text(review.get('candidate_graph_digest'))
    before_digest = stable_graph_digest(graph)
    rebase_id = _stable_digest(
        {
            'proposal_id': proposal_id,
            'review_id': review_id,
            'candidate_graph_digest': candidate_digest,
        },
        prefix='graph-rebase-',
    )
    idempotency_key = _stable_digest(
        {
            'target_graph_id': review.get('target_graph_id') or before_digest,
            'proposal_id': proposal_id,
            'candidate_graph_digest': candidate_digest,
        },
        prefix='graph-rebase-idem-',
    )
    enforced_policy_review: dict[str, Any] = {}
    if review.get('kind') != GRAPH_REBASE_REVIEW_KIND or review_status != 'accepted':
        status = 'rejected' if review_status == 'rejected' else 'blocked'
        _append_unique(blocked_reasons, 'rebase_review_not_accepted')
    elif level == 'off':
        status = 'blocked'
        _append_unique(blocked_reasons, 'graph_rebase_autonomy_off')
    elif level == 'shadow':
        status = 'validated'
    elif level == 'stage':
        status = 'staged'
    elif level == 'apply_enforced':
        enforced_policy_review = build_enforced_policy_review(
            autonomy_level=level,
            lifecycle={
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'rebase_id': rebase_id,
                'proposal_id': proposal_id,
                'review_id': review_id,
                'risk_level': 'reviewed_rebase',
                'source_evidence_refs': _clean_string_list(review.get('runtime_evidence_refs')),
                'validation_review': review,
                'idempotency_key': idempotency_key,
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
    elif level == 'apply_reviewed' and _status(
        source_proposal.get('requested_rebase_class')
    ) != 'partial_subtree_rebase':
        status = 'blocked'
        _append_unique(blocked_reasons, 'apply_reviewed_partial_rebase_only')
    elif level == 'apply_reviewed' and _status(
        execution_contract_proof.get('status')
    ) != 'passed':
        status = 'blocked'
        _append_unique(blocked_reasons, 'apply_reviewed_execution_contract_proof_required')
        for reason in _clean_string_list(
            execution_contract_proof.get('blocked_reasons') or []
        ):
            _append_unique(blocked_reasons, reason)
    elif level == 'apply_reviewed' and not _graph_rebase_authorization_allows(review, level):
        status = 'blocked'
        _append_unique(blocked_reasons, 'apply_reviewed_requires_explicit_rebase_authorization')
    else:
        status = 'staged'

    if level != 'apply_enforced':
        enforced_policy_review = {}

    payload = {
        'kind': GRAPH_REBASE_LIFECYCLE_KIND,
        'rebase_id': rebase_id,
        'proposal_id': proposal_id,
        'review_id': review_id,
        'status': status,
        'autonomy_level': level,
        'risk_level': 'reviewed_rebase',
        'requested_rebase_class': _status(
            source_proposal.get('requested_rebase_class')
        ),
        'source_evidence_refs': _clean_string_list(review.get('runtime_evidence_refs')),
        'target_graph_id': review.get('target_graph_id') or before_digest,
        'before_graph_digest': before_digest,
        'base_graph_digest': review.get('base_graph_digest') or before_digest,
        'candidate_graph_digest': candidate_digest,
        'diff_digest': _stable_digest(review.get('diff') or {}, prefix='graph-rebase-diff-'),
        'preservation_proof_digest': _stable_digest(
            review.get('preservation_proof') or {},
            prefix='graph-rebase-proof-',
        ),
        'execution_contract_proof_digest': _stable_digest(
            execution_contract_proof,
            prefix='graph-rebase-execution-proof-',
        ),
        'execution_contract_proof': execution_contract_proof,
        'idempotency_key': idempotency_key,
        'preconditions': [
            'runtime_rebase_review_accepted',
            'graph_digest_matches_before_graph_digest',
            'preservation_proof_passed',
            'apply_reviewed_requires_explicit_authorization',
        ],
        'postconditions': [
            'successor_rebase_truth_visible',
            'parent_graph_not_mutated',
            'no_duplicate_rebase_for_idempotency_key',
        ],
        'validation_review': review,
        'accepted_successor_graph': review.get('accepted_successor_graph') or {},
        'graph_rebase_authorization': authorization_payload,
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


def _graph_has_successor_rebase(graph: Mapping[str, Any], idempotency_key: str, rebase_id: str) -> bool:
    for key in ('successor_rebase_requests', 'applied_graph_rebases'):
        for item in graph.get(key) or []:
            if not isinstance(item, Mapping):
                continue
            if idempotency_key and _clean_text(item.get('idempotency_key')) == idempotency_key:
                return True
            if rebase_id and _clean_text(item.get('rebase_id')) == rebase_id:
                return True
    for item in graph.get('graph_rebase_lifecycle') or []:
        if not isinstance(item, Mapping):
            continue
        lifecycle_status = _status(item.get('status'))
        runtime_effect = _status((item.get('outcome') or {}).get('runtime_effect'))
        if lifecycle_status not in {'applied', 'already_applied'} and runtime_effect not in {
            'successor_rebase_created',
            'idempotency_guard_no_duplicate_successor_rebase',
        }:
            continue
        if idempotency_key and _clean_text(item.get('idempotency_key')) == idempotency_key:
            return True
        if rebase_id and _clean_text(item.get('rebase_id')) == rebase_id:
            return True
    return False


def _with_lifecycle_record(graph: dict[str, Any], lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    graph['graph_rebase_lifecycle'] = _upsert_record(
        graph.get('graph_rebase_lifecycle') or [],
        lifecycle,
        'proposal_id',
        'rebase_id',
        'idempotency_key',
    )
    return graph


def _without_proposal_records(records: Sequence[Any], proposal_id: str) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in records or []
        if isinstance(item, Mapping)
        and (
            not proposal_id
            or _clean_text(item.get('proposal_id')) != proposal_id
        )
    ]


def _apply_reviewed_sink_blocked_reasons(
    graph: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    successor_graph: Mapping[str, Any],
    *,
    runtime_gate_reasons: Optional[Sequence[str]] = None,
    trusted_authorization: Optional[Mapping[str, Any]] = None,
    root_prompt: str = '',
) -> list[str]:
    """Revalidate every authority and digest at the successor-creation sink."""

    reasons: list[str] = []
    review = (
        lifecycle.get('validation_review')
        if isinstance(lifecycle.get('validation_review'), Mapping)
        else {}
    )
    if review.get('kind') != GRAPH_REBASE_REVIEW_KIND or _status(review.get('status')) != 'accepted':
        _append_unique(reasons, 'apply_reviewed_requires_accepted_validation_review')
        return reasons
    source_proposal = (
        review.get('source_proposal')
        if isinstance(review.get('source_proposal'), Mapping)
        else {}
    )
    if not source_proposal:
        _append_unique(reasons, 'apply_reviewed_source_proposal_missing')
        return reasons
    if not stable_graph_rebase_prompt_digest(root_prompt):
        _append_unique(
            reasons,
            'partial_rebase_current_root_prompt_truth_unavailable',
        )
        return reasons

    current_gate_reasons = _clean_string_list(runtime_gate_reasons or [])
    redraw_scope_review = (
        graph.get('redraw_scope_ladder_review')
        if isinstance(graph.get('redraw_scope_ladder_review'), Mapping)
        else {}
    )
    if _status(redraw_scope_review.get('selected_scope')) in _SMALLER_REDRAW_SCOPES:
        _append_unique(current_gate_reasons, 'smaller_redraw_scope_precedes_rebase')

    proof = review.get('preservation_proof') if isinstance(review.get('preservation_proof'), Mapping) else {}
    revalidated = validate_graph_rebase_proposal(
        source_proposal,
        request_phase_graph=graph,
        closure_review=(
            proof.get('closure_review_summary')
            if isinstance(proof.get('closure_review_summary'), Mapping)
            else None
        ),
        artifact_payload=(
            proof.get('artifact_registry_summary')
            if isinstance(proof.get('artifact_registry_summary'), Mapping)
            else None
        ),
        intent_lens_review=(
            proof.get('intent_lens_review_summary')
            if isinstance(proof.get('intent_lens_review_summary'), Mapping)
            else None
        ),
        runtime_gate_reasons=current_gate_reasons,
        trusted_authorization=trusted_authorization,
        root_prompt=root_prompt,
    )
    if _status(revalidated.get('status')) != 'accepted':
        _append_unique(reasons, 'apply_reviewed_sink_revalidation_failed')
        return reasons
    if not _graph_rebase_authorization_allows(revalidated, 'apply_reviewed'):
        _append_unique(reasons, 'apply_reviewed_requires_explicit_rebase_authorization')

    for key in (
        'proposal_id',
        'review_id',
        'base_graph_digest',
        'candidate_graph_digest',
        'proposal_digest',
    ):
        if _clean_text(review.get(key)) != _clean_text(revalidated.get(key)):
            _append_unique(reasons, 'apply_reviewed_validation_review_binding_mismatch')
            break
    if _json_safe(lifecycle.get('graph_rebase_authorization') or {}) != _json_safe(
        revalidated.get('graph_rebase_authorization') or {}
    ):
        _append_unique(reasons, 'apply_reviewed_authorization_binding_mismatch')

    current_digest = stable_graph_digest(graph)
    expected_candidate_digest = _clean_text(revalidated.get('candidate_graph_digest'))
    successor_digest = stable_graph_digest(successor_graph)
    if successor_digest != expected_candidate_digest:
        _append_unique(reasons, 'apply_reviewed_successor_graph_digest_mismatch')
    for key in ('before_graph_digest', 'base_graph_digest'):
        if _clean_text(lifecycle.get(key)) != current_digest:
            _append_unique(reasons, 'apply_reviewed_current_graph_digest_mismatch')
            break
    if _clean_text(lifecycle.get('candidate_graph_digest')) != expected_candidate_digest:
        _append_unique(reasons, 'apply_reviewed_candidate_graph_digest_mismatch')

    expected_lifecycle = build_graph_rebase_lifecycle(
        request_phase_graph=graph,
        rebase_review=revalidated,
        autonomy_level='apply_reviewed',
        trusted_authorization=trusted_authorization,
    )
    if _status(expected_lifecycle.get('status')) != 'staged':
        _append_unique(reasons, 'apply_reviewed_expected_lifecycle_not_staged')
    for key in (
        'rebase_id',
        'proposal_id',
        'review_id',
        'idempotency_key',
        'before_graph_digest',
        'base_graph_digest',
        'candidate_graph_digest',
        'diff_digest',
        'preservation_proof_digest',
        'execution_contract_proof_digest',
    ):
        if _clean_text(lifecycle.get(key)) != _clean_text(expected_lifecycle.get(key)):
            _append_unique(reasons, 'apply_reviewed_lifecycle_binding_mismatch')
            break
    return reasons


def apply_validated_graph_rebase(
    request_phase_graph: Mapping[str, Any],
    rebase_lifecycle: Mapping[str, Any],
    *,
    autonomy_level: str = 'stage',
    runtime_gate_reasons: Optional[Sequence[str]] = None,
    trusted_authorization: Optional[Mapping[str, Any]] = None,
    root_prompt: str = '',
) -> dict[str, Any]:
    """Apply reviewed graph rebase by creating successor truth, never parent mutation."""

    graph = copy.deepcopy(dict(request_phase_graph or {}))
    lifecycle = dict(rebase_lifecycle) if isinstance(rebase_lifecycle, Mapping) else {}
    level = normalize_graph_rebase_autonomy(autonomy_level or lifecycle.get('autonomy_level'))
    rebase_id = _clean_text(lifecycle.get('rebase_id'))
    proposal_id = _clean_text(lifecycle.get('proposal_id'))
    idempotency_key = _clean_text(lifecycle.get('idempotency_key'))
    base_status = _status(lifecycle.get('status'))
    blocked_reasons = _clean_string_list(lifecycle.get('blocked_reasons'))

    if lifecycle.get('kind') != GRAPH_REBASE_LIFECYCLE_KIND:
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': 'rejected',
                'autonomy_level': level,
                'blocked_reasons': [*blocked_reasons, 'rebase_lifecycle_kind_mismatch'],
                'outcome': {'status': 'rejected', 'runtime_effect': 'none'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': 'rejected',
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'blocked_reasons': result_lifecycle.get('blocked_reasons'),
                'runtime_effect': 'none',
                'graph': graph,
            }
        )

    if idempotency_key and _graph_has_successor_rebase(graph, idempotency_key, rebase_id):
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'already_applied',
                'autonomy_level': level,
                'after_graph_digest': stable_graph_digest(graph),
                'outcome': {
                    'status': 'already_applied',
                    'runtime_effect': 'idempotency_guard_no_duplicate_successor_rebase',
                },
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': 'already_applied',
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'idempotency_key': idempotency_key,
                'runtime_effect': 'idempotency_guard_no_duplicate_successor_rebase',
                'graph': graph,
            }
        )

    if base_status in {'rejected', 'blocked'}:
        graph['staged_graph_rebases'] = _without_proposal_records(
            graph.get('staged_graph_rebases') or [],
            proposal_id,
        )
        graph = _with_lifecycle_record(graph, lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': base_status,
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'blocked_reasons': blocked_reasons,
                'runtime_effect': 'none',
                'graph': graph,
            }
        )

    if level == 'apply_enforced':
        enforced_policy_review = build_enforced_policy_review(
            autonomy_level=level,
            lifecycle=lifecycle,
            request_phase_graph=graph,
        )
        if not enforced_policy_allows_application(enforced_policy_review):
            for reason in _clean_string_list(enforced_policy_review.get('blocked_reasons')):
                _append_unique(blocked_reasons, reason)
        result_lifecycle = _json_safe(
            {
                **lifecycle,
                'status': 'blocked',
                'autonomy_level': level,
                'blocked_reasons': blocked_reasons,
                'authority': 'runtime_enforced_policy_denied',
                'enforced_policy_review': enforced_policy_review,
                'enforced_policy_id': enforced_policy_review.get('policy_id'),
                'enforced_class': enforced_policy_review.get('enforced_class'),
                'policy_mode': enforced_policy_review.get('policy_mode'),
                'allowed_by_policy': enforced_policy_review.get('allowed'),
                'current_evidence_refs': enforced_policy_review.get('current_evidence_refs'),
                'forbidden_evidence_seen': enforced_policy_review.get('forbidden_evidence_seen'),
                'outcome': {'status': 'blocked', 'runtime_effect': 'none'},
            }
        )
        graph = _with_lifecycle_record(graph, result_lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': 'blocked',
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'blocked_reasons': blocked_reasons,
                'runtime_effect': 'none',
                'graph': graph,
            }
        )

    if level != 'apply_reviewed':
        if level == 'stage' and base_status not in {'blocked', 'rejected'}:
            graph['staged_graph_rebases'] = _upsert_record(
                graph.get('staged_graph_rebases') or [],
                lifecycle,
                'proposal_id',
                'rebase_id',
                'idempotency_key',
            )
        else:
            graph['staged_graph_rebases'] = _without_proposal_records(
                graph.get('staged_graph_rebases') or [],
                proposal_id,
            )
        graph = _with_lifecycle_record(graph, lifecycle)
        return _json_safe(
            {
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': base_status or 'staged',
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'runtime_effect': (lifecycle.get('outcome') or {}).get('runtime_effect') or 'none',
                'graph': graph,
            }
        )

    successor_graph = lifecycle.get('accepted_successor_graph')
    if not isinstance(successor_graph, Mapping) or not successor_graph:
        _append_unique(blocked_reasons, 'accepted_successor_graph_missing')
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
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': 'blocked',
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'blocked_reasons': blocked_reasons,
                'runtime_effect': 'none',
                'graph': graph,
            }
        )

    sink_blocked_reasons = _apply_reviewed_sink_blocked_reasons(
        graph,
        lifecycle,
        successor_graph,
        runtime_gate_reasons=runtime_gate_reasons,
        trusted_authorization=trusted_authorization,
        root_prompt=root_prompt,
    )
    if sink_blocked_reasons:
        for reason in sink_blocked_reasons:
            _append_unique(blocked_reasons, reason)
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
                'kind': GRAPH_REBASE_LIFECYCLE_KIND,
                'status': 'blocked',
                'proposal_id': proposal_id,
                'rebase_id': rebase_id,
                'blocked_reasons': blocked_reasons,
                'runtime_effect': 'none',
                'graph': graph,
            }
        )

    successor_request = _json_safe(
        {
            'kind': GRAPH_REBASE_SUCCESSOR_REQUEST_KIND,
            'status': 'pending',
            'runtime_effect': 'successor_rebase_created',
            'rebase_id': rebase_id,
            'proposal_id': proposal_id,
            'review_id': lifecycle.get('review_id'),
            'requested_rebase_class': lifecycle.get('requested_rebase_class'),
            'idempotency_key': idempotency_key,
            'parent_response_id': graph.get('response_id') or lifecycle.get('target_response_id'),
            'parent_frame_id': graph.get('frame_id') or lifecycle.get('target_frame_id'),
            'parent_graph_digest': lifecycle.get('before_graph_digest') or stable_graph_digest(graph),
            'candidate_graph_digest': lifecycle.get('candidate_graph_digest'),
            'lineage': {
                'relation': 'graph_rebase_successor',
                'parent_frozen': True,
                'parent_mutation': 'forbidden',
            },
            'diff_digest': lifecycle.get('diff_digest'),
            'preservation_proof_digest': lifecycle.get('preservation_proof_digest'),
            'execution_contract_proof_digest': lifecycle.get('execution_contract_proof_digest'),
            'execution_contract_proof': lifecycle.get('execution_contract_proof') or {},
            'graph_rebase_authorization': lifecycle.get('graph_rebase_authorization') or {},
            'successor_graph': copy.deepcopy(dict(successor_graph)),
        }
    )
    graph['successor_rebase_requests'] = _upsert_record(
        graph.get('successor_rebase_requests') or [],
        successor_request,
        'rebase_id',
        'idempotency_key',
    )
    applied_record = {
        'kind': GRAPH_REBASE_LIFECYCLE_KIND,
        'status': 'applied',
        'rebase_id': rebase_id,
        'proposal_id': proposal_id,
        'review_id': lifecycle.get('review_id'),
        'idempotency_key': idempotency_key,
        'runtime_effect': 'successor_rebase_created',
        'candidate_graph_digest': lifecycle.get('candidate_graph_digest'),
        'authority': 'runtime_reviewed_successor_rebase',
    }
    graph['applied_graph_rebases'] = _upsert_record(
        graph.get('applied_graph_rebases') or [],
        applied_record,
        'rebase_id',
        'idempotency_key',
    )
    result_lifecycle = _json_safe(
        {
            **lifecycle,
            'status': 'applied',
            'autonomy_level': level,
            'after_graph_digest': stable_graph_digest(graph),
            'outcome': {'status': 'applied', 'runtime_effect': 'successor_rebase_created'},
        }
    )
    graph = _with_lifecycle_record(graph, result_lifecycle)
    return _json_safe(
        {
            'kind': GRAPH_REBASE_LIFECYCLE_KIND,
            'status': 'applied',
            'proposal_id': proposal_id,
            'rebase_id': rebase_id,
            'idempotency_key': idempotency_key,
            'runtime_effect': 'successor_rebase_created',
            'successor_request': successor_request,
            'graph': graph,
        }
    )
