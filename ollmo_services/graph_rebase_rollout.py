"""Pure rollout evidence and promotion gates for graph rebase.

This module intentionally owns no persistence and no control-plane route.  Its
callers must hydrate canonical response/frame truth and trusted operator review
records before invoking :func:`build_graph_rebase_readiness_report`.  Keeping
the evaluator pure makes report generation repeatable, read-only, and usable by
tests, the future observer route, and operator tooling without granting any
graph-mutation authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any, Optional


GRAPH_REBASE_READINESS_REPORT_KIND = 'ollmo.graph_rebase_rollout_readiness'
GRAPH_REBASE_READINESS_POLICY_KIND = 'ollmo.graph_rebase_rollout_policy'
GRAPH_REBASE_READINESS_POLICY_ID = 'ollmo-graph-rebase-safe-partial-v1'
GRAPH_REBASE_READINESS_OBSERVATION_KIND = (
    'ollmo.graph_rebase_readiness_observation'
)

PARTIAL_REBASE_CLASS = 'partial_subtree_rebase'
FULL_REBASE_CLASS = 'full_successor_rebase'

ADJUDICATION_CLASSES = (
    'useful_proposal',
    'false_positive',
    'false_negative',
    'needs_investigation',
    'rejected_authorization',
)

CRITICAL_SAFETY_CATEGORIES = (
    'no_op',
    'lost_dependency_or_failure_visibility',
    'same_id_semantic_drift',
    'scope_escape',
    'digest_mismatch',
    'bookkeeping_smuggling',
    'missing_local_execution_contract',
    'stale_binding',
    'replay_mismatch',
    'parent_mutation',
    'root_prompt_fallback',
    'preservation_failure',
    'graph_integrity',
    'operator_false_positive',
    'operator_false_negative',
    'needs_investigation',
)

DEFAULT_GRAPH_REBASE_ROLLOUT_POLICY: dict[str, Any] = {
    'kind': GRAPH_REBASE_READINESS_POLICY_KIND,
    'policy_id': GRAPH_REBASE_READINESS_POLICY_ID,
    'policy_version': 1,
    'authority': 'runtime_rollout_observer_only',
    'runtime_effect': 'none',
    'shadow_to_stage': {
        'minimum_settled_candidate_opportunities': 20,
        'minimum_unique_workload_families': 5,
        'minimum_settled_not_proposed': 5,
        'minimum_qualifying_partial_proposals': 3,
        'minimum_partial_useful_adjudications': 3,
        'minimum_partial_replay_confirmations': 2,
        'maximum_false_positive_adjudications': 0,
        'maximum_false_negative_adjudications': 0,
        'maximum_open_investigations': 0,
        'maximum_unresolved_critical_safety_findings': 0,
    },
    'partial_stage_to_apply_reviewed': {
        'minimum_partial_stages': 3,
        'minimum_partial_useful_adjudications': 3,
        'minimum_partial_replay_confirmations': 2,
        'minimum_partial_local_execution_contract_proofs': 3,
        'maximum_false_positive_adjudications': 0,
        'maximum_false_negative_adjudications': 0,
        'maximum_open_investigations': 0,
        'maximum_unresolved_critical_safety_findings': 0,
    },
    'full_shadow_only': {
        'execution_allowed': False,
        'maximum_unresolved_critical_safety_findings': 0,
        'reason': 'full_successor_rebase_execution_is_outside_safe_partial_v1',
    },
}

_ACTIVE_STATUSES = {
    'active',
    'in_progress',
    'late_fill_pending',
    'late_fill_running',
    'pending',
    'queued',
    'running',
    'scheduled',
}
_SETTLED_STATUSES = {
    'blocked',
    'cancelled',
    'completed',
    'error',
    'failed',
    'fulfilled',
    'late_fill_completed',
    'repair_needed',
    'skipped',
    'superseded',
    'terminal',
    'waived',
}
_TERMINAL_REBASE_OUTCOME_STATUSES = {
    'already_applied',
    'blocked',
    'cancelled',
    'completed',
    'failed',
    'rejected',
    'succeeded',
    'superseded',
}
_PASSED_STATUSES = {'accepted', 'passed', 'satisfied', 'verified'}
_ACCEPTED_OPERATOR_STATUSES = {'accepted', 'applied', 'authorized', 'completed', 'recorded', 'staged'}

_PARTIAL_STAGE_PAIRING_FIELDS = (
    'response_id',
    'target_frame_id',
    'proposal_id',
    'proposal_digest',
    'runtime_review_id',
    'base_graph_digest',
    'candidate_graph_digest',
    'requested_rebase_class',
)

_CRITICAL_REASON_TOKENS: dict[str, tuple[str, ...]] = {
    'no_op': (
        'candidate_graph_has_no_meaningful_change',
        'identical_noop',
        'no_meaningful_change',
        'no_op',
        'noop',
    ),
    'lost_dependency_or_failure_visibility': (
        'hidden_failure_visibility_lost',
        'lost_dependency',
        'lost_failure_visibility',
        'removed_dependency_without_preservation',
    ),
    'same_id_semantic_drift': (
        'changed_preserved_graph_meaning',
        'changed_preserved_record_meaning',
        'same_id_semantic',
        'semantic_drift',
        'top_level_graph_semantics_changed',
    ),
    'scope_escape': (
        'change_outside_declared_scope',
        'changes_outside_scope',
        'partial_rebase_changes_outside_scope',
        'scope_escape',
        'widened_scope',
    ),
    'digest_mismatch': (
        'candidate_graph_digest_mismatch',
        'current_graph_digest_mismatch',
        'digest_mismatch',
        'successor_graph_digest_mismatch',
    ),
    'bookkeeping_smuggling': (
        'bookkeeping_smuggling',
        'candidate_graph_contains_rebase_bookkeeping',
        'rebase_bookkeeping_present',
    ),
    'missing_local_execution_contract': (
        'branch_local_execution_contract_missing',
        'execution_contract_missing',
        'missing_local_execution_contract',
        'local_execution_contract_missing',
    ),
    'stale_binding': (
        'authorization_binding_mismatch',
        'binding_mismatch',
        'lifecycle_binding_mismatch',
        'stale_binding',
        'stale_frame',
        'stale_parent',
        'validation_review_binding_mismatch',
    ),
    'replay_mismatch': (
        'idempotency_mismatch',
        'replay_digest_mismatch',
        'replay_mismatch',
        'replay_scope_mismatch',
    ),
    'parent_mutation': (
        'parent_graph_mutated',
        'parent_mutation',
        'parent_was_mutated',
    ),
    'root_prompt_fallback': (
        'assistant_prompt_fallback',
        'inherited_request_prompt',
        'replayed_root_prompt',
        'root_prompt_fallback',
        'root_prompt_replay',
    ),
    'preservation_failure': (
        'candidate_preservation_proof_failed',
        'lost_artifact_ref',
        'lost_required_intent_obligation',
        'lost_required_output_obligation',
        'preservation_proof_failed',
        'removed_without_preservation',
        'target_bound_repair_target_lost',
    ),
    'graph_integrity': (
        'candidate_graph_dangling_dependency_source',
        'candidate_graph_duplicate_record_id',
        'candidate_graph_orphan',
        'cycle_detected',
        'graph_integrity',
    ),
}

# Formal rebase records can embed a complete candidate graph (and reviews or
# lifecycles can embed the proposal again).  None of these payload carriers is
# consumed by the rollout evaluator: their frozen digests, validation proofs,
# diffs, bindings, and safety findings are.  Denying only these known bulky
# carriers keeps the projection forward-compatible with new safety fields while
# preventing a readiness scan from retaining whole response/artifact bodies.
_READINESS_BULK_EVIDENCE_KEYS = {
    'artifact_payload',
    'artifacts',
    'assistant_message',
    'base_graph',
    'candidate_graph',
    'content_payload',
    'current_state',
    'input',
    'messages',
    'output',
    'output_text',
    'outputs',
    'prompt',
    'request',
    'request_payload',
    'request_phase_graph',
    'response_frame',
    'route_payload',
    'runtime',
    'source_route_payload',
}

_READINESS_GRAPH_RECORD_KEYS = (
    ('graph_rebase_proposals', 'runtime_graph_rebase_proposals'),
    ('graph_rebase_reviews', 'runtime_graph_rebase_reviews'),
    ('graph_rebase_lifecycle', 'graph_rebase_lifecycle'),
    ('staged_graph_rebases', 'staged_graph_rebases'),
    ('successor_rebase_requests', 'successor_rebase_requests'),
    ('applied_graph_rebases', 'applied_graph_rebases'),
    ('graph_rebase_outcomes', 'graph_rebase_outcomes'),
    ('partial_rebase_outcomes', 'partial_rebase_outcomes'),
)


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _status(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_').replace(' ', '_')


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _clean_text(key): _json_safe(item)
            for key, item in value.items()
            if _clean_text(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(value: Any, *, prefix: str = '') -> str:
    serialized = json.dumps(_json_safe(value), sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:20]
    return f'{prefix}{digest}' if prefix else digest


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _count(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(_status(value) for value in values if _status(value))
    return dict(sorted(counts.items()))


def _deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _json_safe(base)
    if not isinstance(override, Mapping):
        return merged
    for raw_key, value in override.items():
        key = _clean_text(raw_key)
        if not key:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _json_safe(value)
    return merged


def _response_frame(payload: Mapping[str, Any]) -> dict[str, Any]:
    frame = payload.get('response_frame')
    return dict(frame) if isinstance(frame, Mapping) else {}


def _runtime(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        payload.get('runtime'),
        _response_frame(payload).get('runtime'),
        _mapping(payload.get('current_state')).get('runtime'),
        _mapping(_response_frame(payload).get('current_state')).get('runtime'),
    ]
    populated = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping) and candidate]
    if populated:
        return max(
            populated,
            key=lambda candidate: (
                2 if isinstance(candidate.get('request_phase_graph'), Mapping) else 0
            )
            + (1 if isinstance(candidate.get('developer_diagnostics'), Mapping) else 0),
        )
    return {}


def _graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(payload)
    candidates = [
        runtime.get('request_phase_graph'),
        payload.get('request_phase_graph'),
        _mapping(payload.get('planning')).get('request_phase_graph'),
        _mapping(_response_frame(payload).get('planning')).get('request_phase_graph'),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return {}


def _diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    frame = _response_frame(payload)
    runtime_candidates = [
        payload.get('runtime'),
        frame.get('runtime'),
        _mapping(payload.get('current_state')).get('runtime'),
        _mapping(frame.get('current_state')).get('runtime'),
    ]
    for runtime in runtime_candidates:
        if not isinstance(runtime, Mapping):
            continue
        diagnostics = runtime.get('developer_diagnostics')
        if isinstance(diagnostics, Mapping) and diagnostics:
            return dict(diagnostics)
    return {}


def _request(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        payload.get('request'),
        _response_frame(payload).get('request'),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return {}


def _response_id(payload: Mapping[str, Any]) -> str:
    graph = _graph(payload)
    frame = _response_frame(payload)
    for value in (
        payload.get('response_id'),
        payload.get('id'),
        frame.get('response_id'),
        graph.get('response_id'),
    ):
        if _clean_text(value):
            return _clean_text(value)
    return ''


def _frame_id(payload: Mapping[str, Any]) -> str:
    graph = _graph(payload)
    frame = _response_frame(payload)
    for value in (
        payload.get('frame_id'),
        frame.get('frame_id'),
        graph.get('frame_id'),
    ):
        if _clean_text(value):
            return _clean_text(value)
    return ''


def _candidate_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _diagnostics(payload).get('runtime_graph_rebase_candidate_review')
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _diagnostics(payload).get('response_time_graph_rebase_candidate')
    return dict(value) if isinstance(value, Mapping) else {}


def _late_fill_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = _response_frame(payload)
    candidates = [
        payload.get('late_fill'),
        frame.get('late_fill'),
        _mapping(payload.get('current_state')).get('late_fill'),
        _mapping(frame.get('current_state')).get('late_fill'),
        _runtime(payload).get('late_fill'),
    ]
    return [dict(item) for item in candidates if isinstance(item, Mapping) and item]


def _lifecycle_state(payload: Mapping[str, Any]) -> str:
    frame = _response_frame(payload)
    candidates = (
        payload.get('lifecycle_state'),
        _mapping(payload.get('current_state')).get('lifecycle_state'),
        _mapping(frame.get('current_state')).get('lifecycle_state'),
        frame.get('lifecycle_state'),
        payload.get('status'),
        _mapping(payload.get('current_state')).get('status'),
        frame.get('status'),
    )
    for value in candidates:
        if _status(value):
            return _status(value)
    return ''


def _active_late_fill(payload: Mapping[str, Any]) -> bool:
    if _lifecycle_state(payload) in {'late_fill_pending', 'late_fill_running'}:
        return True
    candidate = _candidate_review(payload)
    if _status(candidate.get('reason')) == 'active_late_fill_must_settle':
        return True
    for late_fill in _late_fill_records(payload):
        if _status(late_fill.get('status')) in _ACTIVE_STATUSES:
            return True
        for key in ('active_count', 'pending_count', 'queued_count', 'running_count'):
            try:
                if int(late_fill.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        for key in ('active_branches', 'pending_branches', 'queued_branches'):
            if _records(late_fill.get(key)):
                return True
    return False


def _settled_final(payload: Mapping[str, Any]) -> bool:
    if _active_late_fill(payload):
        return False
    state = _lifecycle_state(payload)
    if state in _SETTLED_STATUSES:
        return True
    if state in _ACTIVE_STATUSES:
        return False
    late_fill_records = _late_fill_records(payload)
    return bool(late_fill_records) and all(
        _status(item.get('status')) in {'completed', 'failed', 'blocked', 'cancelled'}
        for item in late_fill_records
    )


def _sequence_value(payload: Mapping[str, Any]) -> int:
    frame = _response_frame(payload)
    candidates = (
        payload.get('ledger_sequence'),
        payload.get('frame_sequence'),
        payload.get('sequence'),
        frame.get('ledger_sequence'),
        frame.get('frame_sequence'),
        frame.get('sequence'),
        frame.get('frame_version'),
    )
    for value in candidates:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _timestamp_value(payload: Mapping[str, Any]) -> str:
    frame = _response_frame(payload)
    for value in (
        payload.get('updated_at'),
        payload.get('completed_at'),
        payload.get('created_at'),
        frame.get('updated_at'),
        frame.get('created_at'),
    ):
        if _clean_text(value):
            return _clean_text(value)
    return ''


def _workload_family(payload: Mapping[str, Any]) -> tuple[str, str]:
    request = _request(payload)
    request_meta_candidates = [
        payload.get('request_meta'),
        request.get('request_meta'),
        _response_frame(payload).get('request_meta'),
    ]
    explicit_values = [
        payload.get('workload_family'),
        payload.get('prompt_family'),
        request.get('workload_family'),
        request.get('prompt_family'),
    ]
    for metadata in request_meta_candidates:
        if isinstance(metadata, Mapping):
            explicit_values.extend(
                [metadata.get('workload_family'), metadata.get('prompt_family')]
            )
    for value in explicit_values:
        if _clean_text(value):
            return _status(value), 'explicit_family'

    for value in (
        request.get('prompt'),
        request.get('input'),
        payload.get('prompt'),
        payload.get('input'),
    ):
        text = ' '.join(_clean_text(value).split())
        if text:
            return _stable_digest(text, prefix='prompt-family-'), 'prompt_digest'
    return '', 'unknown'


def _project_readiness_evidence_value(value: Any) -> Any:
    """Copy evidence while removing payload bodies the evaluator never reads."""

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = _clean_text(raw_key)
            if not key or _status(key) in _READINESS_BULK_EVIDENCE_KEYS:
                continue
            projected[key] = _project_readiness_evidence_value(item)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_project_readiness_evidence_value(item) for item in value]
    return _json_safe(value)


def _project_readiness_records(value: Any) -> list[dict[str, Any]]:
    return [
        projected
        for item in _records(value)
        for projected in [_project_readiness_evidence_value(item)]
        if isinstance(projected, dict)
    ]


def project_graph_rebase_readiness_observation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bounded projection consumed by the rollout evaluator.

    A canonical response frame may retain artifact bodies, model output, the
    complete request, and one or more full candidate graphs.  Readiness needs
    none of those bodies.  It needs exact response/frame identity and ordering,
    terminal-vs-active state, workload diversity, candidate decisions, and the
    formal records (including their proofs, diffs, bindings, and safety
    findings).  This projection keeps precisely that truth and can therefore be
    appended to a corpus while the hydrated source frame is immediately
    released.

    The projection is evidence-only.  It does not validate, authorize, stage,
    execute, persist, or mutate a rebase.
    """

    if not isinstance(payload, Mapping):
        return {
            'kind': GRAPH_REBASE_READINESS_OBSERVATION_KIND,
            'projection_status': 'invalid_source_payload',
            'runtime_effect': 'none',
        }

    response_id = _response_id(payload)
    frame_id = _frame_id(payload)
    lifecycle_state = _lifecycle_state(payload)
    sequence = _sequence_value(payload)
    timestamp = _timestamp_value(payload)
    workload_family, workload_family_source = _workload_family(payload)
    active_late_fill = _active_late_fill(payload)
    settled_final = _settled_final(payload)
    graph = _graph(payload)
    diagnostics = _diagnostics(payload)

    projected_graph: dict[str, Any] = {
        'kind': _clean_text(graph.get('kind')) or 'ollmo.request_phase_graph',
    }
    if response_id:
        projected_graph['response_id'] = response_id
    if frame_id:
        projected_graph['frame_id'] = frame_id
    scope_review = graph.get('redraw_scope_ladder_review')
    if isinstance(scope_review, Mapping) and scope_review:
        projected_graph['redraw_scope_ladder_review'] = (
            _project_readiness_evidence_value(scope_review)
        )

    projected_diagnostics: dict[str, Any] = {}
    candidate_review = diagnostics.get('runtime_graph_rebase_candidate_review')
    if isinstance(candidate_review, Mapping) and candidate_review:
        projected_diagnostics['runtime_graph_rebase_candidate_review'] = (
            _project_readiness_evidence_value(candidate_review)
        )
    candidate_context = diagnostics.get('response_time_graph_rebase_candidate')
    if isinstance(candidate_context, Mapping) and candidate_context:
        projected_diagnostics['response_time_graph_rebase_candidate'] = (
            _project_readiness_evidence_value(candidate_context)
        )

    for graph_key, diagnostic_key in _READINESS_GRAPH_RECORD_KEYS:
        if isinstance(graph.get(graph_key), Sequence) and not isinstance(
            graph.get(graph_key), (str, bytes, bytearray)
        ):
            projected_graph[graph_key] = _project_readiness_records(
                graph.get(graph_key)
            )
        if isinstance(diagnostics.get(diagnostic_key), Sequence) and not isinstance(
            diagnostics.get(diagnostic_key), (str, bytes, bytearray)
        ):
            projected_diagnostics[diagnostic_key] = _project_readiness_records(
                diagnostics.get(diagnostic_key)
            )

    relation = payload.get('frame_relation')
    if not isinstance(relation, Mapping):
        relation = _response_frame(payload).get('frame_relation')

    projection: dict[str, Any] = {
        'kind': GRAPH_REBASE_READINESS_OBSERVATION_KIND,
        'projection_status': 'projected',
        'runtime_effect': 'none',
        'response_id': response_id,
        'id': response_id,
        'frame_id': frame_id,
        'lifecycle_state': lifecycle_state,
        'ledger_sequence': sequence,
        'updated_at': timestamp,
        'workload_family': workload_family,
        'workload_family_source': workload_family_source,
        'readiness_state': {
            'active_late_fill': active_late_fill,
            'settled_final': settled_final,
        },
        'runtime': {
            'request_phase_graph': projected_graph,
            'developer_diagnostics': projected_diagnostics,
        },
    }
    if isinstance(relation, Mapping) and relation:
        projection['frame_relation'] = _project_readiness_evidence_value(relation)

    # Preserve the evaluator's active/final classification even when the
    # original reached it through a nested late-fill snapshot rather than the
    # top-level lifecycle field.  No branch payload is retained.
    if active_late_fill:
        projection['late_fill'] = {
            'status': 'running',
            'active_count': 1,
            'pending_count': 0,
        }
    elif settled_final:
        projection['late_fill'] = {
            'status': 'completed',
            'active_count': 0,
            'pending_count': 0,
        }

    return _json_safe(projection)


def _observation(payload: Mapping[str, Any], index: int) -> dict[str, Any]:
    response_id = _response_id(payload)
    frame_id = _frame_id(payload)
    workload_family, workload_family_source = _workload_family(payload)
    active_late_fill = _active_late_fill(payload)
    settled_final = _settled_final(payload)
    identity_payload = {
        'frame_id': frame_id,
        'lifecycle_state': _lifecycle_state(payload),
        'candidate_review': _candidate_review(payload),
        'graph': _graph(payload),
    }
    observation_key = response_id or _stable_digest(identity_payload, prefix='anonymous-response-')
    return {
        'payload': dict(payload),
        'input_index': index,
        'response_id': response_id or observation_key,
        'frame_id': frame_id,
        'observation_key': observation_key,
        'observation_digest': _stable_digest(identity_payload, prefix='observation-'),
        'lifecycle_state': _lifecycle_state(payload),
        'active_late_fill': active_late_fill,
        'settled_final': settled_final,
        'sequence': _sequence_value(payload),
        'timestamp': _timestamp_value(payload),
        'workload_family': workload_family,
        'workload_family_source': workload_family_source,
    }


def _observation_preference(item: Mapping[str, Any]) -> tuple[int, int, str, int]:
    state_rank = 3 if item.get('settled_final') else 1 if item.get('active_late_fill') else 0
    return (
        state_rank,
        int(item.get('sequence') or 0),
        _clean_text(item.get('timestamp')),
        int(item.get('input_index') or 0),
    )


def _dedupe_observations(
    response_payloads: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    invalid_count = 0
    for index, payload in enumerate(response_payloads):
        if not isinstance(payload, Mapping):
            invalid_count += 1
            continue
        observations.append(_observation(payload, index))

    selected_by_key: dict[str, dict[str, Any]] = {}
    for observation in observations:
        key = _clean_text(observation.get('observation_key'))
        current = selected_by_key.get(key)
        if current is None or _observation_preference(observation) >= _observation_preference(current):
            selected_by_key[key] = observation

    selected = sorted(selected_by_key.values(), key=lambda item: int(item.get('input_index') or 0))
    selected_keys = {_clean_text(item.get('observation_key')) for item in selected}
    settled_keys = {
        _clean_text(item.get('observation_key'))
        for item in selected
        if item.get('settled_final')
    }
    superseded_active = sum(
        1
        for item in observations
        if item.get('active_late_fill')
        and _clean_text(item.get('observation_key')) in settled_keys
        and item not in selected
    )
    diagnostics = {
        'input_observation_count': len(observations) + invalid_count,
        'valid_input_observation_count': len(observations),
        'invalid_observation_count': invalid_count,
        'unique_response_observation_count': len(selected_keys),
        'duplicate_response_observations_excluded': len(observations) - len(selected),
        'superseded_nonterminal_observations_excluded': superseded_active,
    }
    return selected, diagnostics


def _record_envelope(
    record: Mapping[str, Any],
    *,
    response_id: str,
    source: str,
    ordinal: int,
    proposal_id: str = '',
) -> dict[str, Any]:
    return {
        'record': dict(record),
        'response_id': response_id,
        'source': source,
        'ordinal': ordinal,
        'proposal_id': proposal_id or _clean_text(record.get('proposal_id')),
    }


def _collect_formal_records(
    settled_observations: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[dict[str, Any]]] = {
        'proposals': [],
        'reviews': [],
        'preservation_proofs': [],
        'lifecycles': [],
        'stages': [],
        'inline_authorizations': [],
        'successor_requests': [],
        'applied_rebases': [],
        'terminal_outcomes': [],
    }
    ordinal = 0

    def append_records(
        target: str,
        values: Any,
        *,
        response_id: str,
        source: str,
    ) -> None:
        nonlocal ordinal
        for record in _records(values):
            ordinal += 1
            collected[target].append(
                _record_envelope(
                    record,
                    response_id=response_id,
                    source=source,
                    ordinal=ordinal,
                )
            )

    for observation in settled_observations:
        payload = _mapping(observation.get('payload'))
        response_id = _clean_text(observation.get('response_id'))
        graph = _graph(payload)
        diagnostics = _diagnostics(payload)
        for target, graph_key, diagnostic_key in (
            ('proposals', 'graph_rebase_proposals', 'runtime_graph_rebase_proposals'),
            ('reviews', 'graph_rebase_reviews', 'runtime_graph_rebase_reviews'),
            ('lifecycles', 'graph_rebase_lifecycle', 'graph_rebase_lifecycle'),
            ('stages', 'staged_graph_rebases', 'staged_graph_rebases'),
            ('successor_requests', 'successor_rebase_requests', 'successor_rebase_requests'),
            ('applied_rebases', 'applied_graph_rebases', 'applied_graph_rebases'),
        ):
            append_records(
                target,
                graph.get(graph_key),
                response_id=response_id,
                source=f'request_phase_graph.{graph_key}',
            )
            append_records(
                target,
                diagnostics.get(diagnostic_key),
                response_id=response_id,
                source=f'developer_diagnostics.{diagnostic_key}',
            )

        for key in ('graph_rebase_outcomes', 'partial_rebase_outcomes'):
            append_records(
                'terminal_outcomes',
                graph.get(key),
                response_id=response_id,
                source=f'request_phase_graph.{key}',
            )
            append_records(
                'terminal_outcomes',
                diagnostics.get(key),
                response_id=response_id,
                source=f'developer_diagnostics.{key}',
            )

        relation = _mapping(
            payload.get('frame_relation')
            or _response_frame(payload).get('frame_relation')
        )
        relation_kind = _status(
            relation.get('relation')
            or relation.get('kind')
            or relation.get('type')
        )
        if 'rebase' in relation_kind and _lifecycle_state(payload) in _SETTLED_STATUSES:
            ordinal += 1
            collected['terminal_outcomes'].append(
                _record_envelope(
                    {
                        'kind': 'ollmo.graph_rebase_terminal_outcome',
                        'status': _lifecycle_state(payload),
                        'proposal_id': relation.get('proposal_id'),
                        'rebase_id': relation.get('rebase_id'),
                        'requested_rebase_class': relation.get('requested_rebase_class'),
                        'frame_relation': relation,
                    },
                    response_id=response_id,
                    source='response_frame.frame_relation',
                    ordinal=ordinal,
                )
            )

    # Proofs and inline authorizations are nested in proposal/review/lifecycle
    # truth.  They are extracted from raw records so projection duplicates stay
    # visible in raw_count, then deduplicated by frozen proposal bindings below.
    for source_name in ('proposals', 'reviews', 'lifecycles'):
        for envelope in list(collected[source_name]):
            record = _mapping(envelope.get('record'))
            proposal_id = _clean_text(
                record.get('proposal_id')
                or _mapping(record.get('source_proposal')).get('proposal_id')
                or envelope.get('proposal_id')
            )
            proof = record.get('preservation_proof')
            if not isinstance(proof, Mapping):
                proof = _mapping(record.get('validation_review')).get('preservation_proof')
            if isinstance(proof, Mapping) and proof:
                ordinal += 1
                collected['preservation_proofs'].append(
                    _record_envelope(
                        proof,
                        response_id=_clean_text(envelope.get('response_id')),
                        source=f"{envelope.get('source')}.preservation_proof",
                        ordinal=ordinal,
                        proposal_id=proposal_id,
                    )
                )
            authorization = record.get('graph_rebase_authorization')
            if not isinstance(authorization, Mapping):
                authorization = _mapping(record.get('validation_review')).get(
                    'graph_rebase_authorization'
                )
            if isinstance(authorization, Mapping) and authorization:
                ordinal += 1
                collected['inline_authorizations'].append(
                    _record_envelope(
                        authorization,
                        response_id=_clean_text(envelope.get('response_id')),
                        source=f"{envelope.get('source')}.graph_rebase_authorization",
                        ordinal=ordinal,
                        proposal_id=proposal_id,
                    )
                )

    # A terminal successor status is an outcome even if no dedicated outcome
    # list exists yet.
    for envelope in collected['successor_requests']:
        record = _mapping(envelope.get('record'))
        if _status(record.get('status')) not in _TERMINAL_REBASE_OUTCOME_STATUSES:
            continue
        ordinal += 1
        collected['terminal_outcomes'].append(
            _record_envelope(
                record,
                response_id=_clean_text(envelope.get('response_id')),
                source=f"{envelope.get('source')}.terminal_status",
                ordinal=ordinal,
            )
        )
    return collected


def _record_status_rank(record: Mapping[str, Any]) -> int:
    status = _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
    ranks = {
        'applied': 8,
        'already_applied': 8,
        'completed': 8,
        'succeeded': 8,
        'failed': 7,
        'cancelled': 7,
        'blocked': 7,
        'rejected': 7,
        'staged': 6,
        'authorized': 6,
        'validated': 5,
        'accepted': 5,
        'running': 3,
        'queued': 2,
        'pending': 1,
    }
    return ranks.get(status, 0)


def _record_key(kind: str, envelope: Mapping[str, Any]) -> str:
    record = _mapping(envelope.get('record'))
    proposal_id = _clean_text(envelope.get('proposal_id') or record.get('proposal_id'))
    candidate_digest = _clean_text(record.get('candidate_graph_digest'))
    if kind == 'proposals':
        value = _clean_text(record.get('proposal_id'))
    elif kind == 'reviews':
        value = _clean_text(record.get('review_id')) or proposal_id
    elif kind == 'preservation_proofs':
        binding = ':'.join(
            item
            for item in (
                proposal_id,
                _clean_text(record.get('base_graph_digest')),
                _clean_text(record.get('candidate_graph_digest')),
            )
            if item
        )
        value = binding or _stable_digest(record, prefix='proof-')
    elif kind in {'lifecycles', 'stages'}:
        value = _clean_text(record.get('rebase_id')) or ':'.join(
            item for item in (proposal_id, candidate_digest) if item
        )
    elif kind in {'successor_requests', 'applied_rebases'}:
        value = (
            _clean_text(record.get('idempotency_key'))
            or _clean_text(record.get('rebase_id'))
            or ':'.join(item for item in (proposal_id, candidate_digest) if item)
        )
    elif kind == 'inline_authorizations':
        value = ':'.join(
            item
            for item in (
                proposal_id,
                candidate_digest,
                _clean_text(record.get('authorization_id')),
                _stable_digest(record, prefix='authorization-'),
            )
            if item
        )
    elif kind == 'terminal_outcomes':
        value = ':'.join(
            item
            for item in (
                _clean_text(record.get('outcome_id')),
                _clean_text(record.get('idempotency_key')),
                _clean_text(record.get('rebase_id')),
                proposal_id,
                _status(record.get('status')),
            )
            if item
        )
    else:
        value = ''
    if value:
        return f'{kind}:{value}'
    return f'{kind}:{_stable_digest(record)}'


def _dedupe_record_envelopes(
    kind: str,
    envelopes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    for envelope in envelopes:
        key = _record_key(kind, envelope)
        current = selected.get(key)
        record = _mapping(envelope.get('record'))
        current_record = _mapping(current.get('record')) if current else {}
        if current is None or (
            _record_status_rank(record), int(envelope.get('ordinal') or 0)
        ) >= (
            _record_status_rank(current_record), int(current.get('ordinal') or 0)
        ):
            selected[key] = dict(envelope)
    result = sorted(selected.values(), key=lambda item: int(item.get('ordinal') or 0))
    return result, len(envelopes) - len(result)


def _nested_rebase_class(record: Mapping[str, Any]) -> str:
    candidates = [
        record.get('requested_rebase_class'),
        record.get('rebase_class'),
        record.get('selected_scope'),
        _mapping(record.get('source_proposal')).get('requested_rebase_class'),
        _mapping(record.get('validation_review')).get('requested_rebase_class'),
        _mapping(_mapping(record.get('validation_review')).get('source_proposal')).get(
            'requested_rebase_class'
        ),
        _mapping(record.get('frame_relation')).get('requested_rebase_class'),
    ]
    for value in candidates:
        normalized = _status(value)
        if normalized in {PARTIAL_REBASE_CLASS, FULL_REBASE_CLASS}:
            return normalized
    return ''


def _build_class_indexes(
    deduped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, str], dict[str, str]]:
    proposal_classes: dict[str, str] = {}
    rebase_classes: dict[str, str] = {}
    for envelope in deduped.get('proposals') or []:
        record = _mapping(envelope.get('record'))
        proposal_id = _clean_text(record.get('proposal_id'))
        rebase_class = _nested_rebase_class(record)
        if proposal_id and rebase_class:
            proposal_classes[proposal_id] = rebase_class
    for kind in ('reviews', 'lifecycles', 'stages', 'successor_requests', 'applied_rebases'):
        for envelope in deduped.get(kind) or []:
            record = _mapping(envelope.get('record'))
            proposal_id = _clean_text(record.get('proposal_id') or envelope.get('proposal_id'))
            rebase_class = _nested_rebase_class(record) or proposal_classes.get(proposal_id, '')
            if proposal_id and rebase_class:
                proposal_classes[proposal_id] = rebase_class
            rebase_id = _clean_text(record.get('rebase_id'))
            if rebase_id and rebase_class:
                rebase_classes[rebase_id] = rebase_class
    return proposal_classes, rebase_classes


def _envelope_rebase_class(
    envelope: Mapping[str, Any],
    proposal_classes: Mapping[str, str],
    rebase_classes: Mapping[str, str],
) -> str:
    record = _mapping(envelope.get('record'))
    direct = _nested_rebase_class(record)
    if direct:
        return direct
    proposal_id = _clean_text(record.get('proposal_id') or envelope.get('proposal_id'))
    if proposal_id and proposal_classes.get(proposal_id):
        return _status(proposal_classes.get(proposal_id))
    rebase_id = _clean_text(record.get('rebase_id'))
    if rebase_id and rebase_classes.get(rebase_id):
        return _status(rebase_classes.get(rebase_id))
    return 'unknown'


def _record_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    values = record.get('blocked_reasons')
    if isinstance(values, str):
        values = [values]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        reasons.extend(_status(item) for item in values if _status(item))
    if _status(record.get('reason')):
        reasons.append(_status(record.get('reason')))
    return list(dict.fromkeys(reasons))


def _record_summary(
    kind: str,
    raw: Sequence[Mapping[str, Any]],
    deduped: Sequence[Mapping[str, Any]],
    *,
    proposal_classes: Mapping[str, str],
    rebase_classes: Mapping[str, str],
) -> dict[str, Any]:
    statuses: list[str] = []
    classes: list[str] = []
    modes: list[str] = []
    effects: list[str] = []
    reasons: list[str] = []
    record_ids: list[str] = []
    for envelope in deduped:
        record = _mapping(envelope.get('record'))
        status = _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
        statuses.append(status or ('proposed' if kind == 'proposals' else 'unknown'))
        classes.append(_envelope_rebase_class(envelope, proposal_classes, rebase_classes))
        mode = _status(record.get('autonomy_level') or record.get('mode'))
        if mode:
            modes.append(mode)
        effect = _status(
            record.get('runtime_effect') or _mapping(record.get('outcome')).get('runtime_effect')
        )
        if effect:
            effects.append(effect)
        reasons.extend(_record_reasons(record))
        for key in (
            'proposal_id',
            'review_id',
            'rebase_id',
            'idempotency_key',
            'outcome_id',
            'authorization_id',
        ):
            value = _clean_text(record.get(key))
            if value:
                record_ids.append(value)
                break
    return {
        'raw_count': len(raw),
        'total': len(deduped),
        'duplicate_records_excluded': len(raw) - len(deduped),
        'by_status': _count(statuses),
        'by_class': _count(classes),
        'by_autonomy_mode': _count(modes),
        'by_runtime_effect': _count(effects),
        'by_reason': _count(reasons),
        'record_ids': list(dict.fromkeys(record_ids)),
    }


def _candidate_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses: list[str] = []
    reasons: list[str] = []
    smaller_scopes: list[str] = []
    selected_scopes: list[str] = []
    classes: list[str] = []
    response_ids: list[str] = []
    candidate_count = 0
    not_proposed_count = 0
    with_formal_proposal_count = 0
    for observation in observations:
        payload = _mapping(observation.get('payload'))
        review = _candidate_review(payload)
        context = _candidate_context(payload)
        if not review and not context:
            continue
        candidate_count += 1
        response_ids.append(_clean_text(observation.get('response_id')))
        status = _status(review.get('status') or context.get('status') or 'candidate')
        statuses.append(status)
        reason = _status(review.get('reason'))
        if reason:
            reasons.append(reason)
        if status == 'not_proposed':
            not_proposed_count += 1
        if _clean_text(review.get('proposal_id')):
            with_formal_proposal_count += 1
        smaller_scope = _status(review.get('smaller_scope'))
        if smaller_scope:
            smaller_scopes.append(smaller_scope)
        graph_scope = _mapping(_graph(payload).get('redraw_scope_ladder_review'))
        selected_scope = _status(review.get('selected_scope') or graph_scope.get('selected_scope'))
        if selected_scope:
            selected_scopes.append(selected_scope)
        requested_class = _status(review.get('requested_rebase_class'))
        if requested_class in {PARTIAL_REBASE_CLASS, FULL_REBASE_CLASS}:
            classes.append(requested_class)
        elif selected_scope in {PARTIAL_REBASE_CLASS, FULL_REBASE_CLASS}:
            classes.append(selected_scope)
        else:
            classes.append('unknown')
    return {
        'total': candidate_count,
        'not_proposed_count': not_proposed_count,
        'with_formal_proposal_count': with_formal_proposal_count,
        'by_status': _count(statuses),
        'by_reason': _count(reasons),
        'by_smaller_scope': _count(smaller_scopes),
        'by_selected_scope': _count(selected_scopes),
        'by_class': _count(classes),
        'response_ids': list(dict.fromkeys(item for item in response_ids if item)),
    }


def _trusted_record_key(record: Mapping[str, Any]) -> str:
    for key in (
        'record_id',
        'operator_review_id',
        'authorization_id',
        'stage_record_id',
        'id',
    ):
        value = _clean_text(record.get(key))
        if value:
            return f'{key}:{value}'
    return _stable_digest(record, prefix='trusted-review-')


def _dedupe_trusted_records(
    trusted_review_records: Optional[Iterable[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw: list[dict[str, Any]] = []
    invalid_count = 0
    for record in trusted_review_records or []:
        if isinstance(record, Mapping):
            raw.append(dict(record))
        else:
            invalid_count += 1
    selected: dict[str, dict[str, Any]] = {}
    for record in raw:
        key = _trusted_record_key(record)
        current = selected.get(key)
        if current is None or _record_status_rank(record) >= _record_status_rank(current):
            selected[key] = record
    return list(selected.values()), {
        'raw_trusted_review_record_count': len(raw) + invalid_count,
        'valid_trusted_review_record_count': len(raw),
        'invalid_trusted_review_record_count': invalid_count,
        'duplicate_trusted_review_records_excluded': len(raw) - len(selected),
    }


def _adjudication_classification(record: Mapping[str, Any]) -> str:
    value = _status(
        record.get('adjudication')
        or record.get('classification')
        or _mapping(record.get('outcome')).get('classification')
    )
    aliases = {
        'accepted_proposal': 'useful_proposal',
        'correct_positive': 'useful_proposal',
        'true_positive': 'useful_proposal',
        'useful': 'useful_proposal',
        'false_accept': 'false_positive',
        'missed_rebase': 'false_negative',
        'investigate': 'needs_investigation',
        'open_investigation': 'needs_investigation',
        'authorization_rejected': 'rejected_authorization',
    }
    value = aliases.get(value, value)
    action = _status(record.get('action') or record.get('review_action'))
    status = _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
    if action in {'authorize', 'authorize_partial', 'authorize_partial_rebase'} and status in {
        'blocked',
        'denied',
        'rejected',
    }:
        return 'rejected_authorization'
    return value if value in ADJUDICATION_CLASSES else ''


def _trusted_record_rebase_class(
    record: Mapping[str, Any],
    proposal_classes: Mapping[str, str],
) -> str:
    direct = _nested_rebase_class(record)
    if direct:
        return direct
    proposal_id = _clean_text(record.get('proposal_id'))
    return _status(proposal_classes.get(proposal_id)) or 'unknown'


def _trusted_record_is_bound(
    record: Mapping[str, Any],
    *,
    proposal_ids: set[str],
    response_ids: set[str],
) -> bool:
    classification = _adjudication_classification(record)
    proposal_id = _clean_text(record.get('proposal_id'))
    response_id = _clean_text(record.get('response_id') or record.get('target_response_id'))
    if classification == 'false_negative':
        return bool(response_id and response_id in response_ids)
    if proposal_id:
        return proposal_id in proposal_ids
    return False


def _operator_adjudication_summary(
    trusted_records: Sequence[Mapping[str, Any]],
    *,
    proposal_classes: Mapping[str, str],
    proposal_ids: set[str],
    response_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adjudications: list[dict[str, Any]] = []
    unclassified_count = 0
    for record_index, record in enumerate(trusted_records):
        classification = _adjudication_classification(record)
        if not classification:
            unclassified_count += 1
            continue
        adjudications.append(
            {
                'record': dict(record),
                'record_id': _trusted_record_key(record),
                '_record_index': record_index,
                'classification': classification,
                'rebase_class': _trusted_record_rebase_class(record, proposal_classes),
                'bound_to_settled_truth': _trusted_record_is_bound(
                    record,
                    proposal_ids=proposal_ids,
                    response_ids=response_ids,
                ),
            }
        )

    adjudications_by_raw_id = {
        _clean_text(_mapping(item.get('record')).get('record_id')): item
        for item in adjudications
        if _clean_text(_mapping(item.get('record')).get('record_id'))
    }
    resolved_by: dict[str, str] = {}
    for resolver in adjudications:
        record = _mapping(resolver.get('record'))
        false_negative_id = _clean_text(record.get('resolves_record_id'))
        if not false_negative_id:
            continue
        false_negative = adjudications_by_raw_id.get(false_negative_id)
        false_negative_record = _mapping(
            false_negative.get('record') if isinstance(false_negative, Mapping) else {}
        )
        if (
            not isinstance(false_negative, Mapping)
            or resolver.get('classification') != 'useful_proposal'
            or _status(record.get('action')) != 'adjudicate'
            or resolver.get('bound_to_settled_truth') is not True
            or not _replay_verified(record)
            or false_negative.get('classification') != 'false_negative'
            or false_negative.get('bound_to_settled_truth') is not True
            or int(false_negative.get('_record_index') or 0)
            >= int(resolver.get('_record_index') or 0)
            or resolver.get('rebase_class') != false_negative.get('rebase_class')
            or _clean_text(record.get('resolved_candidate_observation_id'))
            != _clean_text(false_negative_record.get('candidate_observation_id'))
            or _clean_text(record.get('resolved_response_id'))
            != _clean_text(false_negative_record.get('response_id'))
        ):
            continue
        resolver_id = _clean_text(record.get('record_id'))
        if not resolver_id or false_negative_id in resolved_by:
            continue
        resolved_by[false_negative_id] = resolver_id

    false_negative_ids: list[str] = []
    resolved_false_negative_ids: list[str] = []
    unresolved_false_negative_ids: list[str] = []
    for item in adjudications:
        record = _mapping(item.get('record'))
        if item.get('classification') == 'false_negative':
            raw_id = _clean_text(record.get('record_id')) or _clean_text(
                item.get('record_id')
            )
            false_negative_ids.append(raw_id)
            resolution_id = resolved_by.get(raw_id, '')
            item['resolved'] = bool(resolution_id)
            item['resolution_record_id'] = resolution_id
            if resolution_id:
                resolved_false_negative_ids.append(raw_id)
            else:
                unresolved_false_negative_ids.append(raw_id)
        item.pop('_record_index', None)

    by_classification = {key: 0 for key in ADJUDICATION_CLASSES}
    by_classification.update(_count(item['classification'] for item in adjudications))
    summary = {
        'total': len(adjudications),
        'bound_total': sum(1 for item in adjudications if item['bound_to_settled_truth']),
        'unbound_total': sum(1 for item in adjudications if not item['bound_to_settled_truth']),
        'unclassified_trusted_record_count': unclassified_count,
        'by_classification': by_classification,
        'by_class': _count(item['rebase_class'] for item in adjudications),
        'bound_by_classification': _count(
            item['classification']
            for item in adjudications
            if item['bound_to_settled_truth']
        ),
        'false_negative_count': len(false_negative_ids),
        'resolved_false_negative_count': len(resolved_false_negative_ids),
        'unresolved_false_negative_count': len(unresolved_false_negative_ids),
        'false_negative_record_ids': false_negative_ids,
        'resolved_false_negative_record_ids': resolved_false_negative_ids,
        'unresolved_false_negative_record_ids': unresolved_false_negative_ids,
        'false_negative_resolution_record_ids': list(resolved_by.values()),
    }
    return summary, adjudications


def _proof_passed(value: Any) -> bool:
    return isinstance(value, Mapping) and _status(value.get('status')) in _PASSED_STATUSES


def _local_execution_contract_proof(record: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        'local_execution_contract_proof',
        'branch_local_execution_contract_proof',
        'execution_contract_proof',
    ):
        value = record.get(key)
        if isinstance(value, Mapping) and value:
            return dict(value)
    validation_review = _mapping(record.get('validation_review'))
    source_proposal = _mapping(record.get('source_proposal'))
    for source in (validation_review, source_proposal):
        for key in (
            'local_execution_contract_proof',
            'branch_local_execution_contract_proof',
            'execution_contract_proof',
        ):
            value = source.get(key)
            if isinstance(value, Mapping) and value:
                return dict(value)
    return {}


def _replay_verified(record: Mapping[str, Any]) -> bool:
    if record.get('replay_verified') is True or record.get('replay_matches') is True:
        return True
    for key in ('replay_status', 'replay_verification_status', 'replay_result'):
        if _status(record.get(key)) in {'matched', 'passed', 'repeated', 'verified'}:
            return True
    return False


def _accepted_operator_action(record: Mapping[str, Any], actions: set[str]) -> bool:
    action = _status(record.get('action') or record.get('review_action'))
    status = _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
    return action in actions and status in _ACCEPTED_OPERATOR_STATUSES


def _runtime_partial_stage_evidence(
    envelope: Mapping[str, Any],
    *,
    proposal_classes: Mapping[str, str],
    rebase_classes: Mapping[str, str],
) -> dict[str, Any]:
    record = _mapping(envelope.get('record'))
    if (
        _envelope_rebase_class(envelope, proposal_classes, rebase_classes)
        != PARTIAL_REBASE_CLASS
    ):
        return {}
    outcome = _mapping(record.get('outcome'))
    if (
        _status(record.get('status') or outcome.get('status')) != 'staged'
        or _status(record.get('autonomy_level')) != 'stage'
        or _status(record.get('runtime_effect') or outcome.get('runtime_effect'))
        != 'staged_no_executable_mutation'
    ):
        return {}

    validation_review = _mapping(record.get('validation_review'))
    source_proposal = _mapping(validation_review.get('source_proposal'))
    binding = {
        'response_id': _clean_text(envelope.get('response_id')),
        'target_frame_id': _clean_text(
            validation_review.get('target_frame_id')
            or source_proposal.get('target_frame_id')
        ),
        'proposal_id': _clean_text(
            record.get('proposal_id') or envelope.get('proposal_id')
        ),
        'proposal_digest': _clean_text(validation_review.get('proposal_digest')),
        'runtime_review_id': _clean_text(
            record.get('review_id') or validation_review.get('review_id')
        ),
        'base_graph_digest': _clean_text(record.get('base_graph_digest')),
        'candidate_graph_digest': _clean_text(record.get('candidate_graph_digest')),
        'requested_rebase_class': PARTIAL_REBASE_CLASS,
    }
    runtime_identities = {
        'rebase_id': _clean_text(record.get('rebase_id')),
        'idempotency_key': _clean_text(record.get('idempotency_key')),
    }
    missing_fields = [
        key for key in _PARTIAL_STAGE_PAIRING_FIELDS if not binding.get(key)
    ]
    missing_fields.extend(
        key for key, value in runtime_identities.items() if not value
    )
    return {
        'evidence_id': (
            runtime_identities['rebase_id']
            or runtime_identities['idempotency_key']
            or _record_key('stages', envelope)
        ),
        'binding': binding,
        'runtime_identities': runtime_identities,
        'missing_fields': sorted(set(missing_fields)),
    }


def _trusted_partial_stage_evidence(
    record: Mapping[str, Any],
    *,
    proposal_classes: Mapping[str, str],
    proposal_ids: set[str],
    response_ids: set[str],
) -> dict[str, Any]:
    if not _accepted_operator_action(record, {'stage', 'stage_partial', 'stage_rebase'}):
        return {}
    if _trusted_record_rebase_class(record, proposal_classes) != PARTIAL_REBASE_CLASS:
        return {}
    if not _trusted_record_is_bound(
        record,
        proposal_ids=proposal_ids,
        response_ids=response_ids,
    ):
        return {}
    if _status(record.get('runtime_effect')) != 'staged_no_executable_mutation':
        return {}

    binding = {
        'response_id': _clean_text(
            record.get('response_id') or record.get('target_response_id')
        ),
        'target_frame_id': _clean_text(record.get('target_frame_id')),
        'proposal_id': _clean_text(record.get('proposal_id')),
        'proposal_digest': _clean_text(record.get('proposal_digest')),
        'runtime_review_id': _clean_text(
            record.get('runtime_review_id') or record.get('review_id')
        ),
        'base_graph_digest': _clean_text(record.get('base_graph_digest')),
        'candidate_graph_digest': _clean_text(record.get('candidate_graph_digest')),
        'requested_rebase_class': _status(record.get('requested_rebase_class')),
    }
    missing_fields = [
        key for key in _PARTIAL_STAGE_PAIRING_FIELDS if not binding.get(key)
    ]
    return {
        'evidence_id': _trusted_record_key(record),
        'binding': binding,
        'missing_fields': missing_fields,
    }


def _stage_evidence_index(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    complete: dict[str, dict[str, Any]] = {}
    incomplete: dict[str, dict[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, Mapping) or not item:
            continue
        evidence_id = _clean_text(item.get('evidence_id')) or _stable_digest(
            item,
            prefix='partial-stage-evidence-',
        )
        if item.get('missing_fields'):
            incomplete[evidence_id] = dict(item)
            continue
        binding = _mapping(item.get('binding'))
        binding_id = _stable_digest(binding, prefix='partial-stage-binding-')
        complete[binding_id] = dict(item)
    return complete, incomplete


def _paired_partial_stage_evidence(
    deduped: Mapping[str, Sequence[Mapping[str, Any]]],
    trusted_records: Sequence[Mapping[str, Any]],
    *,
    proposal_classes: Mapping[str, str],
    rebase_classes: Mapping[str, str],
    proposal_ids: set[str],
    response_ids: set[str],
) -> dict[str, Any]:
    runtime_evidence = [
        evidence
        for envelope in deduped.get('stages') or []
        for evidence in [
            _runtime_partial_stage_evidence(
                envelope,
                proposal_classes=proposal_classes,
                rebase_classes=rebase_classes,
            )
        ]
        if evidence
    ]
    trusted_evidence = [
        evidence
        for record in trusted_records
        for evidence in [
            _trusted_partial_stage_evidence(
                record,
                proposal_classes=proposal_classes,
                proposal_ids=proposal_ids,
                response_ids=response_ids,
            )
        ]
        if evidence
    ]
    runtime_complete, runtime_incomplete = _stage_evidence_index(runtime_evidence)
    trusted_complete, trusted_incomplete = _stage_evidence_index(trusted_evidence)
    paired_binding_ids = set(runtime_complete) & set(trusted_complete)
    runtime_orphan_ids = {
        _clean_text(item.get('evidence_id')) or binding_id
        for binding_id, item in runtime_complete.items()
        if binding_id not in paired_binding_ids
    } | set(runtime_incomplete)
    trusted_orphan_ids = {
        _clean_text(item.get('evidence_id')) or binding_id
        for binding_id, item in trusted_complete.items()
        if binding_id not in paired_binding_ids
    } | set(trusted_incomplete)
    paired_proposal_ids = {
        _clean_text(_mapping(runtime_complete[binding_id].get('binding')).get('proposal_id'))
        for binding_id in paired_binding_ids
        if _clean_text(
            _mapping(runtime_complete[binding_id].get('binding')).get('proposal_id')
        )
    }
    return {
        'exact_pair_count': len(paired_binding_ids),
        'trusted_registry_stage_count': len(trusted_complete) + len(trusted_incomplete),
        'durable_runtime_stage_count': len(runtime_complete) + len(runtime_incomplete),
        'trusted_registry_orphan_count': len(trusted_orphan_ids),
        'durable_runtime_orphan_count': len(runtime_orphan_ids),
        'pairing_fields': list(_PARTIAL_STAGE_PAIRING_FIELDS),
        'paired_stage_binding_ids': sorted(paired_binding_ids),
        'paired_proposal_ids': sorted(paired_proposal_ids),
        'trusted_registry_orphan_ids': sorted(trusted_orphan_ids),
        'durable_runtime_orphan_ids': sorted(runtime_orphan_ids),
    }


def _qualifying_evidence(
    deduped: Mapping[str, Sequence[Mapping[str, Any]]],
    trusted_records: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    *,
    proposal_classes: Mapping[str, str],
    rebase_classes: Mapping[str, str],
    proposal_ids: set[str],
    response_ids: set[str],
) -> dict[str, Any]:
    qualifying_proposal_ids: set[str] = set()
    partial_qualifying_ids: set[str] = set()
    full_qualifying_ids: set[str] = set()
    passed_proof_ids: set[str] = set()
    local_execution_proof_ids: set[str] = set()

    for envelope in deduped.get('reviews') or []:
        record = _mapping(envelope.get('record'))
        proposal_id = _clean_text(record.get('proposal_id') or envelope.get('proposal_id'))
        rebase_class = _envelope_rebase_class(envelope, proposal_classes, rebase_classes)
        proof = record.get('preservation_proof')
        if _proof_passed(proof) and proposal_id:
            passed_proof_ids.add(proposal_id)
        if _status(record.get('status')) != 'accepted' or not _proof_passed(proof):
            continue
        if proposal_id:
            qualifying_proposal_ids.add(proposal_id)
            if rebase_class == PARTIAL_REBASE_CLASS:
                partial_qualifying_ids.add(proposal_id)
            elif rebase_class == FULL_REBASE_CLASS:
                full_qualifying_ids.add(proposal_id)
        local_proof = _local_execution_contract_proof(record)
        if proposal_id and _proof_passed(local_proof):
            local_execution_proof_ids.add(proposal_id)

    for kind in ('proposals', 'lifecycles', 'stages'):
        for envelope in deduped.get(kind) or []:
            record = _mapping(envelope.get('record'))
            proposal_id = _clean_text(record.get('proposal_id') or envelope.get('proposal_id'))
            if proposal_id and _proof_passed(_local_execution_contract_proof(record)):
                local_execution_proof_ids.add(proposal_id)

    for record in trusted_records:
        proposal_id = _clean_text(record.get('proposal_id'))
        if proposal_id and _proof_passed(_local_execution_contract_proof(record)):
            local_execution_proof_ids.add(proposal_id)

    partial_stage_pairing = _paired_partial_stage_evidence(
        deduped,
        trusted_records,
        proposal_classes=proposal_classes,
        rebase_classes=rebase_classes,
        proposal_ids=proposal_ids,
        response_ids=response_ids,
    )
    partial_stage_ids = set(partial_stage_pairing.get('paired_proposal_ids') or [])

    accepted_partial_authorization_ids: set[str] = set()
    rejected_authorization_count = 0
    for record in trusted_records:
        action = _status(record.get('action') or record.get('review_action'))
        if action not in {'authorize', 'authorize_partial', 'authorize_partial_rebase'}:
            continue
        status = _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
        if status in _ACCEPTED_OPERATOR_STATUSES:
            if (
                _trusted_record_rebase_class(record, proposal_classes) == PARTIAL_REBASE_CLASS
                and _trusted_record_is_bound(
                    record,
                    proposal_ids=proposal_ids,
                    response_ids=response_ids,
                )
            ):
                accepted_partial_authorization_ids.add(
                    _clean_text(record.get('proposal_id')) or _trusted_record_key(record)
                )
        elif status in {'blocked', 'denied', 'rejected'}:
            rejected_authorization_count += 1

    partial_useful_ids = {
        _clean_text(_mapping(item.get('record')).get('proposal_id')) or _clean_text(item.get('record_id'))
        for item in adjudications
        if item.get('bound_to_settled_truth')
        and item.get('classification') == 'useful_proposal'
        and item.get('rebase_class') == PARTIAL_REBASE_CLASS
    }
    partial_replay_ids = {
        _clean_text(_mapping(item.get('record')).get('proposal_id')) or _clean_text(item.get('record_id'))
        for item in adjudications
        if item.get('bound_to_settled_truth')
        and item.get('rebase_class') == PARTIAL_REBASE_CLASS
        and _replay_verified(_mapping(item.get('record')))
    }
    return {
        'qualifying_proposal_count': len(qualifying_proposal_ids),
        'partial_proposal_count': len(partial_qualifying_ids),
        'full_proposal_count': len(full_qualifying_ids),
        'passed_preservation_proof_count': len(passed_proof_ids),
        'partial_stage_count': int(partial_stage_pairing.get('exact_pair_count') or 0),
        'partial_stage_pairing': partial_stage_pairing,
        'partial_useful_adjudication_count': len(partial_useful_ids),
        'partial_replay_confirmation_count': len(partial_replay_ids),
        'partial_local_execution_contract_proof_count': len(
            local_execution_proof_ids & (partial_qualifying_ids | partial_stage_ids)
        ),
        'accepted_partial_authorization_count': len(accepted_partial_authorization_ids),
        'rejected_authorization_count': rejected_authorization_count,
        'qualifying_proposal_ids': sorted(qualifying_proposal_ids),
        'partial_qualifying_proposal_ids': sorted(partial_qualifying_ids),
        'full_qualifying_proposal_ids': sorted(full_qualifying_ids),
    }


def _reason_category(reason: Any) -> str:
    token = _status(reason)
    if token in CRITICAL_SAFETY_CATEGORIES:
        return token
    for category, patterns in _CRITICAL_REASON_TOKENS.items():
        if any(pattern in token for pattern in patterns):
            return category
    return ''


def _explicit_safety_reasons(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for key in ('safety_findings', 'critical_findings', 'safety_violations'):
        value = record.get(key)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        for item in value:
            if isinstance(item, Mapping):
                reason = _status(item.get('reason') or item.get('category') or item.get('kind'))
                disposition = _status(item.get('disposition') or item.get('status'))
            else:
                reason = _status(item)
                disposition = ''
            if reason:
                findings.append((reason, disposition))
    return findings


def _build_safety_summary(
    settled_observations: Sequence[Mapping[str, Any]],
    deduped: Mapping[str, Sequence[Mapping[str, Any]]],
    trusted_records: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    *,
    proposal_classes: Mapping[str, str],
    rebase_classes: Mapping[str, str],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []

    def add_finding(
        *,
        category: str,
        reason: str,
        disposition: str,
        source: str,
        proposal_id: str = '',
        rebase_class: str = 'unknown',
        record_id: str = '',
    ) -> None:
        normalized_category = _reason_category(category) or _reason_category(reason)
        if not normalized_category:
            return
        normalized_disposition = _status(disposition)
        if normalized_disposition in {
            'blocked',
            'contained',
            'passed',
            'rejected',
            'resolved',
            'verified',
        }:
            normalized_disposition = 'contained'
        else:
            normalized_disposition = 'unresolved'
        findings.append(
            {
                'category': normalized_category,
                'reason': _status(reason) or normalized_category,
                'disposition': normalized_disposition,
                'source': source,
                'proposal_id': proposal_id,
                'rebase_class': rebase_class or 'unknown',
                'record_id': record_id,
            }
        )

    for observation in settled_observations:
        payload = _mapping(observation.get('payload'))
        candidate = _candidate_review(payload)
        reason = _status(candidate.get('reason'))
        category = _reason_category(reason)
        if category:
            add_finding(
                category=category,
                reason=reason,
                disposition='contained',
                source='runtime_graph_rebase_candidate_review',
                proposal_id=_clean_text(candidate.get('proposal_id')),
                rebase_class=_status(candidate.get('requested_rebase_class')) or 'unknown',
                record_id=_clean_text(observation.get('response_id')),
            )

    for kind in ('reviews', 'lifecycles', 'successor_requests', 'terminal_outcomes'):
        for envelope in deduped.get(kind) or []:
            record = _mapping(envelope.get('record'))
            status = _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
            disposition = 'contained' if status in {'blocked', 'rejected'} else 'unresolved'
            proposal_id = _clean_text(record.get('proposal_id') or envelope.get('proposal_id'))
            rebase_class = _envelope_rebase_class(envelope, proposal_classes, rebase_classes)
            record_id = _record_key(kind, envelope)
            reasons = _record_reasons(record)
            proof = record.get('preservation_proof')
            if isinstance(proof, Mapping):
                reasons.extend(_record_reasons(proof))
                if _status(proof.get('status')) not in _PASSED_STATUSES:
                    reasons.append('preservation_proof_failed')
            diff = record.get('diff')
            if isinstance(diff, Mapping):
                if diff.get('lost_dependency_edges'):
                    reasons.append('lost_dependency_edge')
                if diff.get('hidden_failure_visibility_losses'):
                    reasons.append('hidden_failure_visibility_lost')
                if diff.get('semantic_changes') or diff.get('top_level_semantic_changes'):
                    reasons.append('same_id_semantic_drift')
            if record.get('partial_scope_violations'):
                reasons.append('partial_rebase_changes_outside_scope')
            lineage = _mapping(record.get('lineage'))
            if lineage and _status(lineage.get('parent_mutation')) not in {'', 'forbidden'}:
                reasons.append('parent_graph_mutated')
            if record.get('parent_mutated') is True:
                reasons.append('parent_graph_mutated')
            for reason in list(dict.fromkeys(_status(item) for item in reasons if _status(item))):
                add_finding(
                    category=_reason_category(reason),
                    reason=reason,
                    disposition=disposition,
                    source=_clean_text(envelope.get('source')) or kind,
                    proposal_id=proposal_id,
                    rebase_class=rebase_class,
                    record_id=record_id,
                )
            for reason, explicit_disposition in _explicit_safety_reasons(record):
                add_finding(
                    category=_reason_category(reason),
                    reason=reason,
                    disposition=explicit_disposition or disposition,
                    source=_clean_text(envelope.get('source')) or kind,
                    proposal_id=proposal_id,
                    rebase_class=rebase_class,
                    record_id=record_id,
                )

    for record in trusted_records:
        proposal_id = _clean_text(record.get('proposal_id'))
        rebase_class = _trusted_record_rebase_class(record, proposal_classes)
        record_id = _trusted_record_key(record)
        for reason, explicit_disposition in _explicit_safety_reasons(record):
            add_finding(
                category=_reason_category(reason),
                reason=reason,
                disposition=explicit_disposition or 'unresolved',
                source='trusted_operator_review',
                proposal_id=proposal_id,
                rebase_class=rebase_class,
                record_id=record_id,
            )

    for item in adjudications:
        classification = _status(item.get('classification'))
        category = {
            'false_positive': 'operator_false_positive',
            'false_negative': 'operator_false_negative',
            'needs_investigation': 'needs_investigation',
        }.get(classification, '')
        if not category:
            continue
        record = _mapping(item.get('record'))
        add_finding(
            category=category,
            reason=classification,
            disposition=(
                'resolved'
                if classification == 'false_negative' and item.get('resolved') is True
                else 'unresolved'
            ),
            source='trusted_operator_adjudication',
            proposal_id=_clean_text(record.get('proposal_id')),
            rebase_class=_clean_text(item.get('rebase_class')) or 'unknown',
            record_id=_clean_text(item.get('record_id')),
        )

    unique_findings: dict[str, dict[str, Any]] = {}
    for finding in findings:
        key = _stable_digest(
            {
                'category': finding.get('category'),
                'reason': finding.get('reason'),
                'disposition': finding.get('disposition'),
                'proposal_id': finding.get('proposal_id'),
                'record_id': finding.get('record_id'),
            },
            prefix='safety-finding-',
        )
        unique_findings[key] = finding
    findings = list(unique_findings.values())
    unresolved = [item for item in findings if item.get('disposition') == 'unresolved']
    contained = [item for item in findings if item.get('disposition') == 'contained']
    unresolved_counts = {key: 0 for key in CRITICAL_SAFETY_CATEGORIES}
    unresolved_counts.update(_count(item['category'] for item in unresolved))
    contained_counts = {key: 0 for key in CRITICAL_SAFETY_CATEGORIES}
    contained_counts.update(_count(item['category'] for item in contained))
    unresolved_partial = [
        item
        for item in unresolved
        if item.get('rebase_class') in {PARTIAL_REBASE_CLASS, 'unknown'}
    ]
    unresolved_full = [
        item
        for item in unresolved
        if item.get('rebase_class') in {FULL_REBASE_CLASS, 'unknown'}
    ]
    return {
        'critical_categories': list(CRITICAL_SAFETY_CATEGORIES),
        'total_finding_count': len(findings),
        'unresolved_critical_finding_count': len(unresolved),
        'contained_critical_finding_count': len(contained),
        'unresolved_partial_or_unknown_finding_count': len(unresolved_partial),
        'unresolved_full_or_unknown_finding_count': len(unresolved_full),
        'unresolved_by_category': unresolved_counts,
        'contained_by_category': contained_counts,
        'zero_tolerance_satisfied': not unresolved,
        'findings': findings,
    }


def _policy_number(
    section: Mapping[str, Any],
    key: str,
    *,
    maximum: bool = False,
) -> int:
    try:
        value = int(section.get(key))
    except (TypeError, ValueError):
        return -1 if maximum else 2**31 - 1
    if value < 0:
        return -1 if maximum else 2**31 - 1
    return value


def _minimum_requirement(
    requirement_id: str,
    *,
    actual: int,
    threshold: int,
    rationale: str,
) -> dict[str, Any]:
    return {
        'requirement': requirement_id,
        'comparison': 'greater_than_or_equal',
        'actual': int(actual),
        'threshold': int(threshold),
        'met': int(actual) >= int(threshold),
        'rationale': rationale,
    }


def _maximum_requirement(
    requirement_id: str,
    *,
    actual: int,
    threshold: int,
    rationale: str,
) -> dict[str, Any]:
    return {
        'requirement': requirement_id,
        'comparison': 'less_than_or_equal',
        'actual': int(actual),
        'threshold': int(threshold),
        'met': int(actual) <= int(threshold),
        'rationale': rationale,
    }


def _gate_payload(
    *,
    decision_when_ready: str,
    decision_when_blocked: str,
    requirements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_requirements = [_json_safe(item) for item in requirements]
    unmet = [
        _clean_text(item.get('requirement'))
        for item in normalized_requirements
        if not bool(item.get('met')) and _clean_text(item.get('requirement'))
    ]
    ready = not unmet
    return {
        'ready': ready,
        'decision': decision_when_ready if ready else decision_when_blocked,
        'requirements': normalized_requirements,
        'unmet_requirements': unmet,
    }


def _build_gates(
    *,
    policy: Mapping[str, Any],
    corpus: Mapping[str, Any],
    settled_candidates: Mapping[str, Any],
    qualifying: Mapping[str, Any],
    adjudications: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    shadow_policy = _mapping(policy.get('shadow_to_stage'))
    partial_policy = _mapping(policy.get('partial_stage_to_apply_reviewed'))
    full_policy = _mapping(policy.get('full_shadow_only'))
    classifications = _mapping(adjudications.get('by_classification'))

    shadow_requirements = [
        _minimum_requirement(
            'minimum_settled_candidate_opportunities',
            actual=int(settled_candidates.get('total') or 0),
            threshold=_policy_number(
                shadow_policy, 'minimum_settled_candidate_opportunities'
            ),
            rationale='Volume is a diversity signal, not authority by itself.',
        ),
        _minimum_requirement(
            'minimum_unique_workload_families',
            actual=int(corpus.get('unique_workload_family_count') or 0),
            threshold=_policy_number(shadow_policy, 'minimum_unique_workload_families'),
            rationale='Repeated copies of one prompt family do not establish workload diversity.',
        ),
        _minimum_requirement(
            'minimum_settled_not_proposed',
            actual=int(settled_candidates.get('not_proposed_count') or 0),
            threshold=_policy_number(shadow_policy, 'minimum_settled_not_proposed'),
            rationale='Settled negative decisions expose false-positive behavior and smaller-scope precedence.',
        ),
        _minimum_requirement(
            'minimum_qualifying_partial_proposals',
            actual=int(qualifying.get('partial_proposal_count') or 0),
            threshold=_policy_number(shadow_policy, 'minimum_qualifying_partial_proposals'),
            rationale='Only accepted partial proposals with passed preservation proof qualify.',
        ),
        _minimum_requirement(
            'minimum_partial_useful_adjudications',
            actual=int(qualifying.get('partial_useful_adjudication_count') or 0),
            threshold=_policy_number(shadow_policy, 'minimum_partial_useful_adjudications'),
            rationale='Trusted operator usefulness review is distinct from runtime validation.',
        ),
        _minimum_requirement(
            'minimum_partial_replay_confirmations',
            actual=int(qualifying.get('partial_replay_confirmation_count') or 0),
            threshold=_policy_number(shadow_policy, 'minimum_partial_replay_confirmations'),
            rationale='Replay evidence must be explicit and exact-bound; duplicate snapshots do not count.',
        ),
        _maximum_requirement(
            'maximum_false_positive_adjudications',
            actual=int(classifications.get('false_positive') or 0),
            threshold=_policy_number(
                shadow_policy,
                'maximum_false_positive_adjudications',
                maximum=True,
            ),
            rationale='A false accepted rebase cannot be averaged away by corpus size.',
        ),
        _maximum_requirement(
            'maximum_false_negative_adjudications',
            actual=int(adjudications.get('unresolved_false_negative_count') or 0),
            threshold=_policy_number(
                shadow_policy,
                'maximum_false_negative_adjudications',
                maximum=True,
            ),
            rationale='Missed structurally necessary rebases remain unresolved rollout evidence.',
        ),
        _maximum_requirement(
            'maximum_open_investigations',
            actual=int(classifications.get('needs_investigation') or 0),
            threshold=_policy_number(
                shadow_policy,
                'maximum_open_investigations',
                maximum=True,
            ),
            rationale='Open safety investigations block promotion.',
        ),
        _maximum_requirement(
            'maximum_unresolved_critical_safety_findings',
            actual=int(safety.get('unresolved_critical_finding_count') or 0),
            threshold=_policy_number(
                shadow_policy,
                'maximum_unresolved_critical_safety_findings',
                maximum=True,
            ),
            rationale='Critical preservation, scope, binding, replay, and mutation findings are zero-tolerance.',
        ),
    ]
    shadow_gate = _gate_payload(
        decision_when_ready='ready_for_stage',
        decision_when_blocked='remain_shadow',
        requirements=shadow_requirements,
    )
    shadow_gate['evidence_denominators'] = {
        'settled_response_count': int(corpus.get('settled_final_response_count') or 0),
        'settled_candidate_opportunity_count': int(settled_candidates.get('total') or 0),
        'qualifying_partial_proposal_count': int(
            qualifying.get('partial_proposal_count') or 0
        ),
        'bound_operator_adjudication_count': int(adjudications.get('bound_total') or 0),
    }

    partial_requirements = [
        {
            'requirement': 'shadow_to_stage_gate_green',
            'comparison': 'is_true',
            'actual': bool(shadow_gate.get('ready')),
            'threshold': True,
            'met': bool(shadow_gate.get('ready')),
            'rationale': 'Partial reviewed authority cannot skip the prior rollout rung.',
        },
        _minimum_requirement(
            'minimum_partial_stages',
            actual=int(qualifying.get('partial_stage_count') or 0),
            threshold=_policy_number(partial_policy, 'minimum_partial_stages'),
            rationale=(
                'Only exact trusted-registry plus durable-runtime partial stage pairs '
                'qualify; either orphan remains diagnostic-only evidence.'
            ),
        ),
        _minimum_requirement(
            'minimum_partial_useful_adjudications',
            actual=int(qualifying.get('partial_useful_adjudication_count') or 0),
            threshold=_policy_number(partial_policy, 'minimum_partial_useful_adjudications'),
            rationale='Trusted usefulness adjudication must cover the partial class.',
        ),
        _minimum_requirement(
            'minimum_partial_replay_confirmations',
            actual=int(qualifying.get('partial_replay_confirmation_count') or 0),
            threshold=_policy_number(partial_policy, 'minimum_partial_replay_confirmations'),
            rationale='Exact partial staging must replay without scope or digest drift.',
        ),
        _minimum_requirement(
            'minimum_partial_local_execution_contract_proofs',
            actual=int(
                qualifying.get('partial_local_execution_contract_proof_count') or 0
            ),
            threshold=_policy_number(
                partial_policy, 'minimum_partial_local_execution_contract_proofs'
            ),
            rationale='Every executable partial branch needs a bounded local contract; root prompt fallback is forbidden.',
        ),
        _maximum_requirement(
            'maximum_false_positive_adjudications',
            actual=int(classifications.get('false_positive') or 0),
            threshold=_policy_number(
                partial_policy,
                'maximum_false_positive_adjudications',
                maximum=True,
            ),
            rationale='A false positive blocks reviewed execution.',
        ),
        _maximum_requirement(
            'maximum_false_negative_adjudications',
            actual=int(adjudications.get('unresolved_false_negative_count') or 0),
            threshold=_policy_number(
                partial_policy,
                'maximum_false_negative_adjudications',
                maximum=True,
            ),
            rationale='Unresolved missed rebases block reviewed execution.',
        ),
        _maximum_requirement(
            'maximum_open_investigations',
            actual=int(classifications.get('needs_investigation') or 0),
            threshold=_policy_number(
                partial_policy,
                'maximum_open_investigations',
                maximum=True,
            ),
            rationale='Open investigations block reviewed execution.',
        ),
        _maximum_requirement(
            'maximum_unresolved_critical_safety_findings',
            actual=int(safety.get('unresolved_partial_or_unknown_finding_count') or 0),
            threshold=_policy_number(
                partial_policy,
                'maximum_unresolved_critical_safety_findings',
                maximum=True,
            ),
            rationale='Partial or unclassified critical findings are zero-tolerance.',
        ),
    ]
    partial_gate = _gate_payload(
        decision_when_ready='ready_for_exact_partial_apply_reviewed_authorization',
        decision_when_blocked='keep_partial_non_executable',
        requirements=partial_requirements,
    )
    partial_gate['evidence_denominators'] = {
        'qualifying_partial_proposal_count': int(
            qualifying.get('partial_proposal_count') or 0
        ),
        'partial_stage_count': int(qualifying.get('partial_stage_count') or 0),
        'partial_useful_adjudication_count': int(
            qualifying.get('partial_useful_adjudication_count') or 0
        ),
    }

    full_safety_requirement = _maximum_requirement(
        'maximum_unresolved_critical_safety_findings',
        actual=int(safety.get('unresolved_full_or_unknown_finding_count') or 0),
        threshold=_policy_number(
            full_policy,
            'maximum_unresolved_critical_safety_findings',
            maximum=True,
        ),
        rationale='Full-class evidence remains visible even though execution is policy-blocked.',
    )
    full_reason = _clean_text(full_policy.get('reason')) or (
        'full_successor_rebase_execution_is_outside_safe_partial_v1'
    )
    full_gate = {
        'ready': False,
        'ready_for_execution': False,
        'decision': 'remain_shadow',
        'requirements': [
            full_safety_requirement,
            {
                'requirement': 'full_successor_execution_policy_enabled',
                'comparison': 'is_true',
                'actual': bool(full_policy.get('execution_allowed')),
                'threshold': True,
                'met': False,
                'rationale': full_reason,
            },
        ],
        'unmet_requirements': [full_reason],
        'evidence_denominators': {
            'qualifying_full_proposal_count': int(
                qualifying.get('full_proposal_count') or 0
            ),
        },
    }
    return {
        'shadow_to_stage': shadow_gate,
        'partial_stage_to_apply_reviewed': partial_gate,
        'full_shadow_only': full_gate,
    }


def _source_ledger_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_safe(value)
    if _clean_text(value):
        return {
            'kind': 'caller_supplied_ledger_identity',
            'identity': _clean_text(value),
        }
    return {
        'kind': 'caller_hydrated_response_truth',
        'identity': 'in_memory_payload_sequence',
    }


def _trusted_authorization_summary(
    trusted_records: Sequence[Mapping[str, Any]],
    proposal_classes: Mapping[str, str],
) -> dict[str, Any]:
    authorizations = [
        record
        for record in trusted_records
        if _status(record.get('action') or record.get('review_action'))
        in {'authorize', 'authorize_partial', 'authorize_partial_rebase'}
    ]
    return {
        'total': len(authorizations),
        'by_status': _count(
            _status(record.get('status') or _mapping(record.get('outcome')).get('status'))
            or 'unknown'
            for record in authorizations
        ),
        'by_class': _count(
            _trusted_record_rebase_class(record, proposal_classes)
            for record in authorizations
        ),
        'record_ids': [_trusted_record_key(record) for record in authorizations],
    }


def build_graph_rebase_readiness_report(
    response_payloads: Iterable[Mapping[str, Any]],
    *,
    trusted_review_records: Optional[Iterable[Mapping[str, Any]]] = None,
    policy: Optional[Mapping[str, Any]] = None,
    corpus_window: Optional[Mapping[str, Any]] = None,
    source_ledger_identity: Any = None,
) -> dict[str, Any]:
    """Evaluate already-hydrated graph-rebase truth without I/O or mutation.

    ``response_payloads`` may contain canonical response payloads, hydrated
    ``ollmo.response_frame`` mappings, or payloads containing a hydrated
    ``response_frame``.  The evaluator resolves no paths and reads no state of
    its own.  Records supplied through ``trusted_review_records`` are treated as
    trusted only because a future runtime-owned registry loader passes them via
    this separate argument; similarly shaped inline response dictionaries never
    become trusted operator authority here.
    """

    normalized_policy = _deep_merge(DEFAULT_GRAPH_REBASE_ROLLOUT_POLICY, policy)
    selected, observation_dedup = _dedupe_observations(response_payloads or [])
    settled = [item for item in selected if item.get('settled_final')]
    active = [item for item in selected if item.get('active_late_fill')]
    other_nonterminal = [
        item
        for item in selected
        if not item.get('settled_final') and not item.get('active_late_fill')
    ]

    raw_formal = _collect_formal_records(settled)
    deduped_formal: dict[str, list[dict[str, Any]]] = {}
    formal_duplicates: dict[str, int] = {}
    for kind, records in raw_formal.items():
        deduped_formal[kind], formal_duplicates[kind] = _dedupe_record_envelopes(
            kind,
            records,
        )
    proposal_classes, rebase_classes = _build_class_indexes(deduped_formal)
    proposal_ids = {
        _clean_text(_mapping(item.get('record')).get('proposal_id'))
        for item in deduped_formal.get('proposals') or []
        if _clean_text(_mapping(item.get('record')).get('proposal_id'))
    }
    response_ids = {
        _clean_text(item.get('response_id'))
        for item in settled
        if _clean_text(item.get('response_id'))
    }

    trusted_records, trusted_dedup = _dedupe_trusted_records(trusted_review_records)
    adjudication_summary, adjudications = _operator_adjudication_summary(
        trusted_records,
        proposal_classes=proposal_classes,
        proposal_ids=proposal_ids,
        response_ids=response_ids,
    )
    qualifying = _qualifying_evidence(
        deduped_formal,
        trusted_records,
        adjudications,
        proposal_classes=proposal_classes,
        rebase_classes=rebase_classes,
        proposal_ids=proposal_ids,
        response_ids=response_ids,
    )
    safety = _build_safety_summary(
        settled,
        deduped_formal,
        trusted_records,
        adjudications,
        proposal_classes=proposal_classes,
        rebase_classes=rebase_classes,
    )

    workload_families = [
        _clean_text(item.get('workload_family'))
        for item in settled
        if _clean_text(item.get('workload_family'))
    ]
    family_counts = _count(workload_families)
    timestamps = sorted(
        _clean_text(item.get('timestamp'))
        for item in settled
        if _clean_text(item.get('timestamp'))
    )
    window = _json_safe(corpus_window) if isinstance(corpus_window, Mapping) else {}
    if timestamps:
        window = {
            **window,
            'observed_at_min': timestamps[0],
            'observed_at_max': timestamps[-1],
        }
    corpus = {
        'source_ledger_identity': _source_ledger_payload(source_ledger_identity),
        'window': window,
        **observation_dedup,
        'settled_final_response_count': len(settled),
        'nonterminal_active_late_fill_response_count': len(active),
        'other_nonterminal_response_count': len(other_nonterminal),
        'unique_workload_family_count': len(family_counts),
        'unknown_workload_family_count': sum(
            1 for item in settled if not _clean_text(item.get('workload_family'))
        ),
        'repeated_workload_observation_count': sum(
            max(0, count - 1) for count in family_counts.values()
        ),
        'workload_family_counts': family_counts,
    }
    corpus['corpus_digest'] = _stable_digest(
        {
            'source_ledger_identity': corpus['source_ledger_identity'],
            'window': corpus['window'],
            'settled': [
                {
                    'response_id': item.get('response_id'),
                    'frame_id': item.get('frame_id'),
                    'observation_digest': item.get('observation_digest'),
                }
                for item in settled
            ],
            'trusted_record_ids': [_trusted_record_key(item) for item in trusted_records],
        },
        prefix='graph-rebase-corpus-',
    )

    settled_candidates = _candidate_summary(settled)
    active_candidates = _candidate_summary(active)
    active_candidates['superseded_by_settled_final_count'] = int(
        observation_dedup.get('superseded_nonterminal_observations_excluded') or 0
    )
    formal_evidence = {
        kind: _record_summary(
            kind,
            raw_formal[kind],
            deduped_formal[kind],
            proposal_classes=proposal_classes,
            rebase_classes=rebase_classes,
        )
        for kind in (
            'proposals',
            'reviews',
            'preservation_proofs',
            'lifecycles',
            'stages',
            'successor_requests',
            'applied_rebases',
            'terminal_outcomes',
        )
    }
    formal_evidence['authorizations'] = {
        'inline_observed_untrusted': _record_summary(
            'inline_authorizations',
            raw_formal['inline_authorizations'],
            deduped_formal['inline_authorizations'],
            proposal_classes=proposal_classes,
            rebase_classes=rebase_classes,
        ),
        'trusted_operator_records': _trusted_authorization_summary(
            trusted_records,
            proposal_classes,
        ),
        'authority_boundary': (
            'only_records_supplied_through_trusted_review_records_are_operator_authority'
        ),
    }
    gates = _build_gates(
        policy=normalized_policy,
        corpus=corpus,
        settled_candidates=settled_candidates,
        qualifying=qualifying,
        adjudications=adjudication_summary,
        safety=safety,
    )

    report = {
        'kind': GRAPH_REBASE_READINESS_REPORT_KIND,
        'schema_version': 1,
        'evaluation_mode': 'pure_already_hydrated_response_truth',
        'runtime_effect': 'none',
        'corpus': corpus,
        'deduplication': {
            'settled_response_key': 'response_id_or_anonymous_response_digest',
            'selected_observation_rule': (
                'settled_final_then_sequence_then_timestamp_then_last_input'
            ),
            'formal_record_keys': {
                'proposals': 'proposal_id',
                'reviews': 'review_id_or_proposal_id',
                'preservation_proofs': 'proposal_and_graph_digests_and_proof_digest',
                'lifecycles': 'rebase_id_or_proposal_candidate_binding',
                'stages': 'rebase_id_or_proposal_candidate_binding',
                'successor_requests': 'idempotency_key_or_rebase_id',
                'terminal_outcomes': 'outcome_or_rebase_binding_and_status',
            },
            'duplicate_response_observations_excluded': int(
                observation_dedup.get('duplicate_response_observations_excluded') or 0
            ),
            'superseded_nonterminal_observations_excluded': int(
                observation_dedup.get('superseded_nonterminal_observations_excluded') or 0
            ),
            'duplicate_formal_records_excluded': formal_duplicates,
            **trusted_dedup,
        },
        'candidate_opportunities': {
            'settled_final': settled_candidates,
            'nonterminal_active_late_fill': active_candidates,
            'settled_responses_without_candidate_opportunity': len(settled)
            - int(settled_candidates.get('total') or 0),
        },
        'formal_evidence': formal_evidence,
        'qualifying_evidence': qualifying,
        'safety': safety,
        'operator_adjudications': adjudication_summary,
        'policy': {
            **normalized_policy,
            'configuration_source': 'caller_override' if isinstance(policy, Mapping) else 'product_default',
            'thresholds_are_authority': False,
            'promotion_requires_explicit_operator_action': True,
        },
        'gates': gates,
    }
    report['report_digest'] = _stable_digest(report, prefix='graph-rebase-readiness-')
    return _json_safe(report)


def evaluate_graph_rebase_readiness(
    response_payloads: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility-named wrapper around the canonical report builder."""

    return build_graph_rebase_readiness_report(response_payloads, **kwargs)


def build_partial_graph_rebase_promotion_gate(
    readiness_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the partial execution gate to one immutable readiness report."""

    report = _mapping(readiness_report)
    gate = _mapping(_mapping(report.get('gates')).get('partial_stage_to_apply_reviewed'))
    report_digest = _clean_text(report.get('report_digest'))
    corpus_digest = _clean_text(_mapping(report.get('corpus')).get('corpus_digest'))
    ready = bool(
        report.get('kind') == GRAPH_REBASE_READINESS_REPORT_KIND
        and report_digest
        and corpus_digest
        and gate.get('ready') is True
        and _status(gate.get('decision'))
        == 'ready_for_exact_partial_apply_reviewed_authorization'
    )
    policy_digest = _stable_digest(
        _mapping(report.get('policy')),
        prefix='graph-rebase-rollout-policy-',
    )
    gate_id = _stable_digest(
        {
            'report_digest': report_digest,
            'corpus_digest': corpus_digest,
            'policy_digest': policy_digest,
            'gate': 'partial_stage_to_apply_reviewed',
        },
        prefix='graph-rebase-partial-gate-',
    )
    evidence_refs = [
        f'readiness:{report_digest}' if report_digest else '',
        f'corpus:{corpus_digest}' if corpus_digest else '',
    ]
    return _json_safe(
        {
            'kind': 'ollmo.graph_rebase_promotion_gate',
            'gate_id': gate_id,
            'gate': 'partial_stage_to_apply_reviewed',
            'status': 'ready' if ready else 'blocked',
            'decision': 'promote' if ready else 'keep_partial_non_executable',
            'evidence_refs': [item for item in evidence_refs if item],
            'policy_digest': policy_digest,
            'readiness_report_digest': report_digest,
            'corpus_digest': corpus_digest,
            'unmet_requirements': _records(gate.get('unmet_requirements'))
            if isinstance(gate.get('unmet_requirements'), Sequence)
            and not isinstance(gate.get('unmet_requirements'), (str, bytes, bytearray))
            and any(isinstance(item, Mapping) for item in gate.get('unmet_requirements') or [])
            else [
                _clean_text(item)
                for item in (gate.get('unmet_requirements') or [])
                if _clean_text(item)
            ],
            'runtime_effect': 'none',
        }
    )


__all__ = [
    'ADJUDICATION_CLASSES',
    'CRITICAL_SAFETY_CATEGORIES',
    'DEFAULT_GRAPH_REBASE_ROLLOUT_POLICY',
    'FULL_REBASE_CLASS',
    'GRAPH_REBASE_READINESS_POLICY_ID',
    'GRAPH_REBASE_READINESS_POLICY_KIND',
    'GRAPH_REBASE_READINESS_OBSERVATION_KIND',
    'GRAPH_REBASE_READINESS_REPORT_KIND',
    'PARTIAL_REBASE_CLASS',
    'build_partial_graph_rebase_promotion_gate',
    'build_graph_rebase_readiness_report',
    'evaluate_graph_rebase_readiness',
    'project_graph_rebase_readiness_observation',
]
