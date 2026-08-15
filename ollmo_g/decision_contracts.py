"""Derived Ghost decision contract helpers.

The decision contract is a read-model over existing graph truth. It gives Ghost
one coherent thinking surface without making Ghost the source of runtime truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ollmo_g.semantic_roles import semantic_role_for_lens
from ollmo_services.graph_repair import build_graph_repair_proposals

GHOST_DECISION_CONTRACT_VERSION = 11

_OPEN_OBLIGATION_STATUSES = {'pending', 'planned', 'active', 'deferred'}
_BLOCKED_OBLIGATION_STATUSES = {'blocked', 'failed', 'error'}
_CLOSED_OBLIGATION_STATUSES = {'fulfilled', 'completed', 'waived', 'superseded'}
_RECONSIDERABLE_DECISIONS = {'reserved', 'omitted', 'stale'}
_WAIVED_DECISIONS = {'waived'}
_SUPERSEDED_DECISIONS = {'superseded'}
_DETERMINISTIC_REVIEW_CRITERIA = {
    'consumes_declared_input_refs',
    'does_not_restart_root_request',
    'output_contract_matches_capability',
    'preparation_text_is_bounded_to_downstream_inputs',
    'runtime_artifact_exists_when_fulfilled',
    'runtime_evidence_text_exists_when_fulfilled',
    'runtime_status_reaches_fulfilled_blocked_failed_waived_or_pending',
    'runtime_status_reaches_fulfilled_blocked_failed_waived_superseded_or_pending',
    'runtime_text_artifact_exists_when_fulfilled',
    'runtime_text_artifact_revision_preservation_passed_when_required',
    'runtime_text_artifact_revision_write_proven_when_fulfilled',
    'runtime_text_exists_when_fulfilled',
    'uses_dependency_evidence',
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


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


def _status(value: Any, *, fallback: str = '') -> str:
    return _clean_text(value).lower().replace('-', '_') or fallback


def _normalized_review_criterion(value: Any) -> str:
    return ''.join(
        char if char.isalnum() else '_'
        for char in _clean_text(value).lower()
    ).strip('_')


def _review_criterion_is_deterministic(value: Any) -> bool:
    return _normalized_review_criterion(value) in _DETERMINISTIC_REVIEW_CRITERIA


def _semantic_review_criteria_for_task(task: Mapping[str, Any]) -> list[str]:
    explicit = _clean_string_list(task.get('semantic_review_criteria'))
    if explicit:
        return explicit
    return [
        criterion
        for criterion in _clean_string_list(task.get('review_criteria'))
        if not _review_criterion_is_deterministic(criterion)
    ]


def _semantic_review_lens_payload(record: Mapping[str, Any], *, source_kind: str = 'task') -> dict[str, Any]:
    capability = _clean_text(record.get('capability')).lower()
    output_type = _clean_text(record.get('output_type')).lower()
    role_text = ' '.join(
        _clean_text(record.get(key)).lower()
        for key in (
            'advisory_role',
            'role',
            'semantic_intent',
            'objective',
            'deliverable',
            'review_type',
            'check_kind',
            'source_category',
            'source_kind',
            'content_payload_source',
            'stage_direction',
        )
    )
    criteria = _clean_string_list(record.get('semantic_review_criteria')) or _clean_string_list(record.get('review_criteria'))
    evidence_requirements = _clean_string_list(record.get('evidence_requirements'))
    depends_on = _clean_string_list(record.get('depends_on'))
    status = _status(record.get('status'))
    repair_action = _clean_text(record.get('repair_action') or record.get('recovery_action') or record.get('recommended_transition') or record.get('decision_action')).lower()
    explicit_lens = _clean_text(record.get('semantic_review_lens')).lower()
    explicit_role = semantic_role_for_lens(explicit_lens)

    if explicit_role:
        lens = _clean_text(explicit_role.get('role_id')) or explicit_lens
    elif source_kind == 'aspiration_frame':
        lens = 'possibility_expander'
    elif 'risk' in role_text or 'safety' in role_text or 'compliance' in role_text:
        lens = 'risk_sentinel'
    elif 'simplif' in role_text or 'vereinfach' in role_text:
        lens = 'simplifier'
    elif 'doubt' in role_text or 'assumption' in role_text or 'socratic' in role_text:
        lens = 'doubt_challenger'
    elif 'planner' in role_text or 'promotion' in role_text:
        lens = 'planner'
    elif source_kind == 'commitment_frame':
        lens = 'transition_committer'
    elif repair_action.startswith('repair_') or status in _BLOCKED_OBLIGATION_STATUSES or 'repair' in role_text:
        lens = 'repairer'
    elif source_kind == 'semantic_quality_contract' or criteria or 'quality' in role_text or 'review' in role_text:
        lens = 'quality_reviewer'
    elif capability in {'vision_analysis', 'speech_to_text'} or 'evidence' in role_text:
        lens = 'evidence_verifier'
    elif capability in {'image_generation', 'text_to_speech'} or output_type in {'image', 'audio'} or record.get('requires_artifact') is True:
        lens = 'materializer'
    elif depends_on or 'integrat' in role_text or 'join' in role_text or 'compare' in role_text:
        lens = 'integrator'
    elif source_kind in {'global_semantic_closure', 'global_semantic_closure_proposal'}:
        lens = 'whole_turn_reviewer'
    else:
        lens = 'worker'
    role_definition = semantic_role_for_lens(lens) or {}

    definitions = {
        'planner': 'Success means the candidate or workload shape covers the current intent without executing unpromoted work.',
        'structural_planner': 'Success means the candidate or workload shape covers the current intent without executing unpromoted work.',
        'transition_committer': 'Success means choosing the right-sized sufficient next transition without forcing completion or under-scoping the work.',
        'repairer': 'Success means identifying the exact blocked edge and proposing dependency, branch-contract, clarification, waiver, supersession, or manual-review repair from evidence.',
        'quality_reviewer': 'Success means runtime evidence satisfies every stated review criterion; output existence alone is not enough.',
        'evidence_verifier': 'Success means deriving usable evidence from the referenced artifact or dependency, not returning a path-only or no-access claim.',
        'evidence_reasoner': 'Success means deriving usable evidence from the referenced artifact or dependency, not returning a path-only or no-access claim.',
        'materializer': 'Success means the required output is materialized as runtime evidence of the requested type and bound to the branch identity.',
        'integrator': 'Success means the branch uses its dependencies and current intent to synthesize the requested local result without replaying the root prompt.',
        'whole_turn_reviewer': 'Success means fulfilled-looking local branches also satisfy the whole current intent as one coherent graph.',
        'worker': 'Success means completing the declared branch-local deliverable cleanly against runtime evidence and review criteria.',
        'possibility_expander': 'Success means important possibilities remain visible without becoming unearned obligations.',
        'doubt_challenger': 'Success means hidden assumptions are surfaced before plausible text becomes false truth.',
        'risk_sentinel': 'Success means material risks are named early with proportionate mitigation paths.',
        'simplifier': 'Success means unnecessary complexity is removed without dropping promoted work.',
    }
    failure_modes_by_lens = {
        'planner': [
            'candidate_space_too_small',
            'unpromoted_work_treated_as_executable',
            'root_intent_collapsed_to_chat',
        ],
        'structural_planner': [
            'candidate_space_too_small',
            'unpromoted_work_treated_as_executable',
            'root_intent_collapsed_to_chat',
        ],
        'transition_committer': [
            'over_pending_after_enough_evidence',
            'minimal_transition_that_solves_the_wrong_problem',
            'force_completion_without_truth',
        ],
        'repairer': [
            'blind_retry_instead_of_edge_repair',
            'missing_dependency_hidden_as_text',
            'blocked_state_overwritten_without_evidence',
        ],
        'quality_reviewer': [
            'artifact_exists_but_criterion_unproven',
            'semantic_verdict_missing_or_unparseable',
            'review_uses_wrong_evidence_scope',
        ],
        'evidence_verifier': [
            'path_only_evidence',
            'no_access_claim',
            'evidence_from_wrong_branch',
        ],
        'evidence_reasoner': [
            'path_only_evidence',
            'no_access_claim',
            'evidence_from_wrong_branch',
        ],
        'materializer': [
            'visible_text_claim_without_runtime_artifact',
            'artifact_bound_to_wrong_branch',
            'wrong_output_type_materialized',
        ],
        'integrator': [
            'dependency_evidence_ignored',
            'root_prompt_restarted',
            'only_one_required_branch_used',
        ],
        'whole_turn_reviewer': [
            'local_success_global_mismatch',
            'final_synthesis_missing_required_evidence',
            'whole_intent_fit_unproven',
        ],
        'worker': [
            'declared_deliverable_missing',
            'branch_identity_lost',
            'review_criteria_not_checked',
        ],
        'possibility_expander': [
            'minimal_collapse',
            'missing_candidate_surface',
            'premature_narrowing',
        ],
        'doubt_challenger': [
            'unsupported_claim',
            'missing_evidence',
            'premature_freeze',
        ],
        'risk_sentinel': [
            'unsafe_output_path',
            'missing_privacy_check',
            'unreviewed_external_effect',
        ],
        'simplifier': [
            'overbuilt_graph',
            'duplicate_branch',
            'false_minimalism',
        ],
    }
    evidence_by_lens = {
        'planner': ['candidate_graph', 'promotion_review', 'workload_proposal_review'],
        'structural_planner': ['candidate_graph', 'promotion_review', 'workload_proposal_review'],
        'transition_committer': ['current_runtime_state', 'available_evidence_refs', 'allowed_transitions'],
        'repairer': ['blocked_check', 'dependency_or_branch_contract_evidence', 'repair_candidate'],
        'quality_reviewer': ['branch_output', 'runtime_evidence_for_review_criteria', 'semantic_review_verdict_when_promoted'],
        'evidence_verifier': ['source_artifact_ref', 'derived_evidence_text', 'matching_branch_id'],
        'evidence_reasoner': ['source_artifact_ref', 'derived_evidence_text', 'matching_branch_id'],
        'materializer': ['runtime_artifact_record', 'matching_branch_id', 'requested_output_type'],
        'integrator': ['dependency_outputs', 'current_user_intent', 'branch_review_criteria'],
        'whole_turn_reviewer': ['all_relevant_branch_outputs', 'current_user_intent', 'global_semantic_verdict'],
        'worker': ['branch_output', 'execution_contract', 'review_criteria'],
        'possibility_expander': ['candidate_space', 'underplanning_signals', 'reconsiderable_options'],
        'doubt_challenger': ['runtime_truth', 'counter_evidence', 'branch_status'],
        'risk_sentinel': ['risk_context', 'runtime_action_scope', 'mitigation_evidence'],
        'simplifier': ['promoted_obligations', 'duplicate_work_signals', 'waiver_or_supersession_evidence'],
    }
    return _json_safe(
        {
            'kind': 'ollmo.semantic_review_lens',
            'lens': lens,
            'semantic_role_id': _clean_text(role_definition.get('role_id')),
            'semantic_role_name': _clean_text(role_definition.get('name')),
            'semantic_role_orientation': _clean_text(role_definition.get('orientation')),
            'authority': 'advisory_read_model_only',
            'source_kind': source_kind,
            'success_definition': (
                _clean_text(record.get('success_definition'))
                or _clean_text(role_definition.get('success_definition'))
                or definitions.get(lens)
                or _clean_text(role_definition.get('summary'))
                or definitions['worker']
            ),
            'failure_modes': (
                _clean_string_list(record.get('failure_modes'))
                or _clean_string_list(role_definition.get('failure_modes'))
                or failure_modes_by_lens.get(lens)
                or failure_modes_by_lens['worker']
            ),
            'evidence_requirements': _clean_string_list(
                [
                    *evidence_requirements,
                    *_clean_string_list(role_definition.get('evidence_requirements')),
                    *evidence_by_lens.get(lens, evidence_by_lens['worker']),
                ]
            ),
            'focus_questions': _clean_string_list(
                [
                    *_clean_string_list(role_definition.get('focus_questions')),
                    f'What must be true for the {lens} lens to pass?',
                    'Which runtime evidence proves that truth?',
                    'Which failure mode should become repair, waiver, supersession, clarification, review, or freeze?',
                ]
            ),
            'non_authority_boundary': (
                _clean_text(role_definition.get('non_authority_boundary'))
                or 'review_lens_only_runtime_contracts_closure_decide_truth'
            ),
        }
    )


def _obligation_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            'obligation_id': _clean_text(record.get('obligation_id')),
            'phase_id': _clean_text(record.get('phase_id')),
            'branch_id': _clean_text(record.get('branch_id')),
            'capability': _clean_text(record.get('capability')),
            'output_type': _clean_text(record.get('output_type')),
            'status': _status(record.get('status'), fallback='pending'),
            'role': _clean_text(record.get('role')),
        }
    )


def _candidate_decision_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            'candidate_id': _clean_text(record.get('candidate_id')),
            'candidate_type': _clean_text(record.get('candidate_type')),
            'decision': _status(record.get('decision'), fallback='reserved'),
            'contract_ref': _clean_text(record.get('contract_ref')),
            'reason': _clean_text(record.get('reason')),
            'reconsiderable': record.get('reconsiderable') if isinstance(record.get('reconsiderable'), bool) else None,
            'reconsideration_policy': _clean_text(record.get('reconsideration_policy')),
            'execution_policy': _clean_text(record.get('execution_policy')),
            'authority': _clean_text(record.get('authority')),
        }
    )


def _task_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    lens = _semantic_review_lens_payload(record, source_kind=_clean_text(record.get('source_kind')) or 'task')
    return _json_safe(
        {
            'task_id': _clean_text(record.get('task_id')),
            'phase_id': _clean_text(record.get('phase_id')),
            'branch_id': _clean_text(record.get('branch_id')),
            'capability': _clean_text(record.get('capability')),
            'visibility': _clean_text(record.get('visibility')),
            'status': _status(record.get('status'), fallback='pending'),
            'semantic_intent': _clean_text(record.get('semantic_intent')),
            'objective': _clean_text(record.get('objective')),
            'deliverable': _clean_text(record.get('deliverable')),
            'advisory_role': _clean_text(record.get('advisory_role')),
            'semantic_review_lens': _clean_text(record.get('semantic_review_lens')) or _clean_text(lens.get('lens')),
            'success_definition': _clean_text(record.get('success_definition')) or _clean_text(lens.get('success_definition')),
            'failure_modes': _clean_string_list(record.get('failure_modes')) or _clean_string_list(lens.get('failure_modes')),
            'decision_notes': _clean_text(record.get('decision_notes')),
            'promotion_policy': _clean_text(record.get('promotion_policy')),
            'reconsideration_policy': _clean_text(record.get('reconsideration_policy')),
            'review_criteria': _clean_string_list(record.get('review_criteria')),
            'evidence_requirements': _clean_string_list(record.get('evidence_requirements')) or _clean_string_list(lens.get('evidence_requirements')),
            'reconsideration_triggers': _clean_string_list(record.get('reconsideration_triggers')),
            'depends_on': _clean_string_list(record.get('depends_on')),
            'semantic_review_lens_contract': lens,
        }
    )


def _reconsideration_items(promotion_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = promotion_review.get('decisions') if isinstance(promotion_review.get('decisions'), list) else []
    items: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        if decision.get('reconsiderable') is not True:
            continue
        payload = _candidate_decision_ref(decision)
        if payload:
            payload.setdefault('action', 'keep_available_for_future_promotion_review')
            items.append(payload)
    return items


def _supersession_items(
    output_obligations: Sequence[Mapping[str, Any]],
    promotion_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for obligation in output_obligations:
        if not isinstance(obligation, Mapping) or _status(obligation.get('status')) != 'superseded':
            continue
        payload = _obligation_ref(obligation)
        for key in ('superseded_by', 'superseded_by_candidate_id', 'superseded_by_obligation_id', 'supersession_reason'):
            value = _clean_text(obligation.get(key))
            if value:
                payload[key] = value
        if payload:
            payload['action'] = 'do_not_retry_superseded_obligation'
            items.append(payload)
    decisions = promotion_review.get('decisions') if isinstance(promotion_review.get('decisions'), list) else []
    for decision in decisions:
        if not isinstance(decision, Mapping) or _status(decision.get('decision')) != 'superseded':
            continue
        payload = _candidate_decision_ref(decision)
        for key in ('superseded_by', 'superseded_by_candidate_id', 'superseded_by_obligation_id'):
            value = _clean_text(decision.get(key))
            if value:
                payload[key] = value
        if payload and payload not in items:
            payload['action'] = 'preserve_supersession_truth'
            items.append(payload)
    return items


def _semantic_review_items(workload_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    items: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        review_criteria = _semantic_review_criteria_for_task(task)
        if not review_criteria:
            continue
        payload = _task_ref(task)
        payload['review_criteria'] = review_criteria
        payload['semantic_review_criteria'] = review_criteria
        payload['authority'] = 'advisory_until_promoted_semantic_review'
        items.append(payload)
    return items


def _supersession_candidate_items(workload_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    items: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        candidates = task.get('supersession_candidates') if isinstance(task.get('supersession_candidates'), list) else []
        if not candidates:
            continue
        payload = _task_ref(task)
        payload['supersession_candidates'] = _json_safe(candidates)
        payload['authority'] = 'advisory_until_closure_review_confirms_supersession'
        items.append(payload)
    return items


def _promotion_suggestion_items(workload_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    items: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        suggestions = task.get('promotion_suggestions') if isinstance(task.get('promotion_suggestions'), list) else []
        if not suggestions:
            continue
        payload = _task_ref(task)
        payload['promotion_suggestions'] = _json_safe(suggestions)
        payload['authority'] = 'advisory_until_promotion_review_confirms_current_relevance'
        payload['action'] = 'review_possible_promotion_without_executing_unpromoted_work'
        items.append(payload)
    return items


def _waiver_candidate_items(workload_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    items: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        candidates = task.get('waiver_candidates') if isinstance(task.get('waiver_candidates'), list) else []
        if not candidates:
            continue
        payload = _task_ref(task)
        payload['waiver_candidates'] = _json_safe(candidates)
        payload['authority'] = 'advisory_until_closure_review_confirms_explicit_release'
        payload['action'] = 'review_possible_waiver_without_hiding_missing_work'
        items.append(payload)
    return items


def _repair_items(workload_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    items: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        repair_action = _clean_text(task.get('repair_action') or task.get('recovery_action'))
        repair_candidates = task.get('repair_candidates') if isinstance(task.get('repair_candidates'), list) else []
        if not repair_action and not repair_candidates:
            continue
        payload = _task_ref(task)
        if repair_action:
            payload['repair_action'] = repair_action
        if repair_candidates:
            payload['repair_candidates'] = _json_safe(repair_candidates)
        items.append(payload)
    return items


def _block_resolution_signal(
    *,
    source: str,
    category: str,
    action: str,
    reason: str,
    payload: Mapping[str, Any],
    severity: str = 'advisory',
) -> dict[str, Any]:
    return _json_safe(
        {
            'kind': 'ollmo.block_resolution_signal',
            'source': source,
            'category': category,
            'severity': severity,
            'action': action,
            'reason': reason,
            'authority': 'read_model_only_not_runtime_truth',
            'resolution_policy': 'right_sized_verified_state_transition',
            'principle': 'the_solution_to_a_block_is_the_blocks_own_resolution',
            **dict(payload),
        }
    )


def _workload_task_ref_if_needed(task: Mapping[str, Any]) -> dict[str, Any]:
    payload = _task_ref(task)
    workload_task_id = _clean_text(task.get('workload_task_id') or task.get('task_id'))
    if workload_task_id:
        payload['workload_task_id'] = workload_task_id
    return _json_safe(payload)


def _block_resolution_reflex(
    *,
    obligations: Sequence[Mapping[str, Any]],
    promotion_review: Mapping[str, Any],
    workload_graph: Mapping[str, Any],
    reconsideration_items: Sequence[Mapping[str, Any]],
    supersession_items: Sequence[Mapping[str, Any]],
    waiver_candidate_items: Sequence[Mapping[str, Any]],
    supersession_candidate_items: Sequence[Mapping[str, Any]],
    repair_items: Sequence[Mapping[str, Any]],
    semantic_review_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []

    for obligation in obligations:
        if not isinstance(obligation, Mapping):
            continue
        status = _status(obligation.get('status'), fallback='pending')
        ref = _obligation_ref(obligation)
        if status in _BLOCKED_OBLIGATION_STATUSES:
            signals.append(
                _block_resolution_signal(
                    source='output_obligations',
                    category='blocked_obligation',
                    severity='blocking',
                    action='resolve_block_from_dependency_contract_waiver_supersession_or_truthful_freeze',
                    reason='blocked obligation remains real; do not rewrite intent or blind-retry around it',
                    payload=ref,
                )
            )
        elif status in _OPEN_OBLIGATION_STATUSES:
            signals.append(
                _block_resolution_signal(
                    source='output_obligations',
                    category='open_obligation',
                    severity='open',
                    action='continue_or_repair_promoted_work_before_freeze',
                    reason='promoted work remains owed until runtime truth closes it',
                    payload=ref,
                )
            )
        elif status == 'superseded':
            signals.append(
                _block_resolution_signal(
                    source='output_obligations',
                    category='superseded_obligation',
                    action='preserve_replacement_truth_and_do_not_retry',
                    reason='superseded work is closed by newer runtime truth or replacement edge',
                    payload=ref,
                )
            )
        elif status == 'waived':
            signals.append(
                _block_resolution_signal(
                    source='output_obligations',
                    category='waived_obligation',
                    action='preserve_explicit_release_evidence',
                    reason='waived work is closed only by explicit release evidence',
                    payload=ref,
                )
            )

    decisions = promotion_review.get('decisions') if isinstance(promotion_review.get('decisions'), list) else []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        decision_status = _status(decision.get('decision'), fallback='reserved')
        ref = _candidate_decision_ref(decision)
        if decision.get('reconsiderable') is True or decision_status in _RECONSIDERABLE_DECISIONS:
            signals.append(
                _block_resolution_signal(
                    source='promotion_review',
                    category='reconsiderable_candidate',
                    action='keep_visible_for_future_relevance_review_without_execution',
                    reason='reserved, omitted, or stale possibilities stay available but non-executable',
                    payload=ref,
                )
            )
        elif decision_status in _SUPERSEDED_DECISIONS:
            signals.append(
                _block_resolution_signal(
                    source='promotion_review',
                    category='superseded_candidate',
                    action='preserve_closed_supersession_truth',
                    reason='superseded candidates should not reappear as missing work',
                    payload=ref,
                )
            )
        elif decision_status in _WAIVED_DECISIONS:
            signals.append(
                _block_resolution_signal(
                    source='promotion_review',
                    category='waived_candidate',
                    action='preserve_waiver_without_hiding_promoted_work',
                    reason='waiver is a recorded release, not a silent omission',
                    payload=ref,
                )
            )

    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_status = _status(task.get('status'), fallback='pending')
        if task_status in _BLOCKED_OBLIGATION_STATUSES:
            signals.append(
                _block_resolution_signal(
                    source='workload_graph.tasks',
                    category='blocked_workload_task',
                    severity='blocking',
                    action='repair_task_dependency_or_branch_contract_before_retry',
                    reason='blocked branch work should resolve at its own dependency or contract edge',
                    payload=_workload_task_ref_if_needed(task),
                )
            )
        elif task_status in _OPEN_OBLIGATION_STATUSES:
            signals.append(
                _block_resolution_signal(
                    source='workload_graph.tasks',
                    category='open_workload_task',
                    severity='open',
                    action='continue_or_verify_branch_local_work',
                    reason='branch-local work remains open inside the larger request movement',
                    payload=_workload_task_ref_if_needed(task),
                )
            )

    for category, source, action, reason, records in (
        (
            'waiver_candidate',
            'decision_contract.waiver_candidates',
            'review_release_evidence_before_waiving',
            'waiver candidates are possible releases, not automatic closure',
            waiver_candidate_items,
        ),
        (
            'supersession_candidate',
            'decision_contract.supersession_candidates',
            'review_replacement_truth_before_superseding',
            'supersession candidates require newer runtime truth or replacement edge',
            supersession_candidate_items,
        ),
        (
            'repair_candidate',
            'decision_contract.repair_candidates',
            'prefer_targeted_repair_over_forced_completion',
            'repair should start at the right-sized verified failing edge',
            repair_items,
        ),
        (
            'semantic_review_candidate',
            'decision_contract.semantic_review_candidates',
            'treat_quality_as_review_work_not_prose_truth',
            'qualitative criteria need review evidence before truthful freeze',
            semantic_review_items,
        ),
    ):
        if records:
            signals.append(
                _block_resolution_signal(
                    source=source,
                    category=category,
                    action=action,
                    reason=reason,
                    payload={'record_count': len(records)},
                )
            )

    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for signal in signals:
        category = _clean_text(signal.get('category')) or 'unknown'
        severity = _clean_text(signal.get('severity')) or 'advisory'
        category_counts[category] = category_counts.get(category, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.block_resolution_reconsideration_reflex',
            'status': 'active' if signals else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'resolve_blocks_by_preserving_intent_and_applying_the_right_sized_verified_state_transition',
            'movement_cycle': [
                'observe_runtime_truth',
                'classify_open_blocked_reserved_waived_or_superseded_state',
                'preserve_intent_and_candidate_visibility',
                'choose_repair_waiver_supersession_wait_clarify_or_truthful_freeze',
                'never_force_completion_by_rewriting_intent',
            ],
            'signal_count': len(signals),
            'category_counts': category_counts,
            'severity_counts': severity_counts,
            'open_obligation_count': sum(
                1
                for item in obligations
                if isinstance(item, Mapping) and _status(item.get('status'), fallback='pending') in _OPEN_OBLIGATION_STATUSES
            ),
            'blocked_obligation_count': sum(
                1
                for item in obligations
                if isinstance(item, Mapping) and _status(item.get('status'), fallback='pending') in _BLOCKED_OBLIGATION_STATUSES
            ),
            'reconsiderable_candidate_count': len(reconsideration_items),
            'supersession_record_count': len(supersession_items),
            'signals': signals,
        }
    )


def _stable_contract_token(*values: Any, fallback: str = 'item') -> str:
    for value in values:
        text = _clean_text(value).lower()
        if not text:
            continue
        token = ''.join(ch if ch.isalnum() else '-' for ch in text)
        token = '-'.join(part for part in token.split('-') if part)
        if token:
            return token
    return fallback


def _active_reconsideration_decision(signal: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    category = _clean_text(signal.get('category')) or 'unknown'
    action = _clean_text(signal.get('action'))
    review_type_by_category = {
        'blocked_obligation': 'block_resolution_review',
        'blocked_workload_task': 'block_resolution_review',
        'open_obligation': 'continuation_or_repair_review',
        'open_workload_task': 'continuation_or_repair_review',
        'reconsiderable_candidate': 'promotion_relevance_review',
        'waived_obligation': 'waiver_evidence_review',
        'waived_candidate': 'waiver_evidence_review',
        'waiver_candidate': 'waiver_evidence_review',
        'superseded_obligation': 'supersession_truth_review',
        'superseded_candidate': 'supersession_truth_review',
        'supersession_candidate': 'supersession_truth_review',
        'repair_candidate': 'repair_contract_review',
        'semantic_review_candidate': 'semantic_quality_review',
    }
    recommended_action_by_category = {
        'blocked_obligation': 'resolve_block_from_dependency_contract_waiver_supersession_or_truthful_freeze',
        'blocked_workload_task': 'repair_task_dependency_or_branch_contract_before_retry',
        'open_obligation': 'continue_or_repair_promoted_work_before_freeze',
        'open_workload_task': 'continue_or_verify_branch_local_work',
        'reconsiderable_candidate': 'review_current_relevance_before_promotion',
        'waived_obligation': 'verify_explicit_release_evidence_before_freeze',
        'waived_candidate': 'verify_explicit_release_evidence_before_freeze',
        'waiver_candidate': 'review_release_evidence_before_waiving',
        'superseded_obligation': 'verify_replacement_truth_before_freeze',
        'superseded_candidate': 'verify_replacement_truth_before_freeze',
        'supersession_candidate': 'review_replacement_truth_before_superseding',
        'repair_candidate': 'promote_right_sized_valid_repair_contract_when_needed',
        'semantic_review_candidate': 'run_semantic_quality_review_before_claiming_quality_truth',
    }
    allowed_outcomes_by_category = {
        'blocked_obligation': ['repair_dependency_chain', 'repair_branch_contract', 'wait', 'clarify', 'waive_with_evidence', 'supersede_with_replacement_truth', 'truthful_freeze'],
        'blocked_workload_task': ['repair_dependency_chain', 'repair_branch_contract', 'wait', 'clarify', 'waive_with_evidence', 'supersede_with_replacement_truth', 'truthful_freeze'],
        'open_obligation': ['continue_branch_local_work', 'repair_dependency_chain', 'repair_branch_contract', 'waive_with_evidence', 'supersede_with_replacement_truth', 'truthful_freeze'],
        'open_workload_task': ['continue_branch_local_work', 'repair_dependency_chain', 'repair_branch_contract', 'waive_with_evidence', 'supersede_with_replacement_truth', 'truthful_freeze'],
        'reconsiderable_candidate': ['keep_reserved', 'promote_after_current_relevance_review', 'reject_as_stale', 'waive_with_evidence', 'supersede_with_replacement_truth'],
        'waived_obligation': ['preserve_waiver', 'reopen_if_waiver_lacks_evidence'],
        'waived_candidate': ['preserve_waiver', 'reopen_if_waiver_lacks_evidence'],
        'waiver_candidate': ['keep_advisory', 'waive_after_explicit_release_review', 'continue_if_not_released'],
        'superseded_obligation': ['preserve_supersession', 'reopen_if_replacement_truth_missing'],
        'superseded_candidate': ['preserve_supersession', 'reopen_if_replacement_truth_missing'],
        'supersession_candidate': ['keep_advisory', 'supersede_after_replacement_truth_review', 'continue_if_not_replaced'],
        'repair_candidate': ['promote_repair_contract', 'keep_advisory', 'manual_review'],
        'semantic_review_candidate': ['schedule_semantic_review', 'keep_pending_review', 'manual_review'],
    }
    token = _stable_contract_token(
        signal.get('branch_id'),
        signal.get('phase_id'),
        signal.get('obligation_id'),
        signal.get('task_id'),
        signal.get('candidate_id'),
        category,
        fallback=f'item-{index}',
    )
    return _json_safe(
        {
            'kind': 'ollmo.active_reconsideration_decision',
            'decision_id': f'active-reconsideration-{index}-{token}',
            'status': 'pending_review',
            'authority': 'advisory_read_model_only',
            'source': _clean_text(signal.get('source')),
            'source_category': category,
            'review_type': review_type_by_category.get(category, 'state_relevance_review'),
            'recommended_action': recommended_action_by_category.get(category) or action,
            'source_action': action,
            'allowed_outcomes': allowed_outcomes_by_category.get(category, ['keep_visible', 'manual_review']),
            'reason': _clean_text(signal.get('reason')),
            'resolution_policy': _clean_text(signal.get('resolution_policy')) or 'right_sized_verified_state_transition',
            'principle': _clean_text(signal.get('principle')) or 'the_solution_to_a_block_is_the_blocks_own_resolution',
            'candidate_id': _clean_text(signal.get('candidate_id')),
            'candidate_type': _clean_text(signal.get('candidate_type')),
            'decision': _clean_text(signal.get('decision')),
            'obligation_id': _clean_text(signal.get('obligation_id')),
            'task_id': _clean_text(signal.get('task_id') or signal.get('workload_task_id')),
            'workload_task_id': _clean_text(signal.get('workload_task_id') or signal.get('task_id')),
            'phase_id': _clean_text(signal.get('phase_id')),
            'branch_id': _clean_text(signal.get('branch_id')),
            'capability': _clean_text(signal.get('capability')),
            'output_type': _clean_text(signal.get('output_type')),
            'severity': _clean_text(signal.get('severity')),
        }
    )


def _active_reconsideration_review(block_resolution_reflex: Mapping[str, Any]) -> dict[str, Any]:
    signals = (
        block_resolution_reflex.get('signals')
        if isinstance(block_resolution_reflex, Mapping) and isinstance(block_resolution_reflex.get('signals'), list)
        else []
    )
    decisions = [
        _active_reconsideration_decision(signal, index=index)
        for index, signal in enumerate(signals, start=1)
        if isinstance(signal, Mapping)
    ]
    category_counts: dict[str, int] = {}
    recommended_next_actions: list[str] = []
    for decision in decisions:
        category = _clean_text(decision.get('source_category')) or 'unknown'
        category_counts[category] = category_counts.get(category, 0) + 1
        action = _clean_text(decision.get('recommended_action'))
        if action and action not in recommended_next_actions:
            recommended_next_actions.append(action)
    return _json_safe(
        {
            'kind': 'ollmo.active_reconsideration_review',
            'status': 'active' if decisions else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'convert_reflex_signals_into_reviewable_state_transitions_without_executing_them',
            'decision_count': len(decisions),
            'category_counts': category_counts,
            'recommended_next_actions': recommended_next_actions,
            'decisions': decisions,
        }
    )


def _semantic_quality_review(semantic_review_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    for index, item in enumerate(semantic_review_items, start=1):
        if not isinstance(item, Mapping):
            continue
        criteria = _clean_string_list(item.get('semantic_review_criteria')) or _clean_string_list(item.get('review_criteria'))
        if not criteria:
            continue
        token = _stable_contract_token(
            item.get('branch_id'),
            item.get('phase_id'),
            item.get('task_id'),
            fallback=f'item-{index}',
        )
        lens = _semantic_review_lens_payload(item, source_kind='semantic_quality_contract')
        contracts.append(
            _json_safe(
                {
                    'kind': 'ollmo.semantic_quality_contract',
                    'quality_review_id': f'semantic-quality-{index}-{token}',
                    'status': 'pending_semantic_review',
                    'authority': 'advisory_until_promoted_semantic_verifier',
                    'review_policy': 'quality_is_not_proven_by_output_existence',
                    'recommended_action': 'run_semantic_quality_review_before_truthful_freeze',
                    'task_id': _clean_text(item.get('task_id')),
                    'workload_task_id': _clean_text(item.get('workload_task_id') or item.get('task_id')),
                    'phase_id': _clean_text(item.get('phase_id')),
                    'branch_id': _clean_text(item.get('branch_id')),
                    'capability': _clean_text(item.get('capability')),
                    'semantic_intent': _clean_text(item.get('semantic_intent')),
                    'objective': _clean_text(item.get('objective')),
                    'deliverable': _clean_text(item.get('deliverable')),
                    'semantic_review_lens': _clean_text(lens.get('lens')),
                    'success_definition': _clean_text(lens.get('success_definition')),
                    'failure_modes': _clean_string_list(lens.get('failure_modes')),
                    'review_criteria': criteria,
                    'evidence_requirements': _clean_string_list(lens.get('evidence_requirements')),
                    'depends_on': _clean_string_list(item.get('depends_on')),
                    'verifier_scope': 'branch_local_then_parent_intent',
                    'semantic_review_lens_contract': lens,
                }
            )
        )
    return _json_safe(
        {
            'kind': 'ollmo.semantic_quality_review',
            'status': 'required' if contracts else 'not_required',
            'authority': 'advisory_until_promoted_semantic_verifier',
            'policy': 'separate_subjective_quality_review_from_runtime_artifact_truth',
            'contract_count': len(contracts),
            'contracts': contracts,
        }
    )


def _recursive_cycle_review(workload_graph: Mapping[str, Any]) -> dict[str, Any]:
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    cycle_tasks: list[dict[str, Any]] = []
    depth_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    max_depth = 0
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, Mapping):
            continue
        depth_value = task.get('decomposition_level', task.get('depth', 0))
        try:
            depth = max(0, int(depth_value or 0))
        except (TypeError, ValueError):
            depth = 0
        max_depth = max(max_depth, depth)
        status = _status(task.get('status'), fallback='pending')
        depth_key = str(depth)
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        lifecycle = task.get('lifecycle') if isinstance(task.get('lifecycle'), Mapping) else {}
        raw_stages = lifecycle.get('stages') if isinstance(lifecycle.get('stages'), list) else []
        stages: list[dict[str, Any]] = []
        for raw_stage in raw_stages:
            if not isinstance(raw_stage, Mapping):
                continue
            stage_name = _clean_text(raw_stage.get('stage'))
            stage_status = _status(raw_stage.get('status'), fallback='pending')
            if not stage_name:
                continue
            stages.append({'stage': stage_name, 'status': stage_status})
            stage_counts[f'{stage_name}:{stage_status}'] = stage_counts.get(f'{stage_name}:{stage_status}', 0) + 1
        if not stages:
            stages = [
                {'stage': 'prepare', 'status': 'pending'},
                {'stage': 'gather_evidence', 'status': 'pending'},
                {'stage': 'execute', 'status': status},
                {'stage': 'verify', 'status': 'pending'},
                {'stage': 'repair_or_freeze', 'status': 'pending'},
            ]
            for stage in stages:
                key = f"{stage['stage']}:{stage['status']}"
                stage_counts[key] = stage_counts.get(key, 0) + 1
        token = _stable_contract_token(
            task.get('branch_id'),
            task.get('phase_id'),
            task.get('task_id'),
            fallback=f'item-{index}',
        )
        lens = _semantic_review_lens_payload(task, source_kind='recursive_cycle_task')
        cycle_tasks.append(
            _json_safe(
                {
                    'kind': 'ollmo.recursive_cycle_task',
                    'cycle_task_id': f'recursive-cycle-{index}-{token}',
                    'task_id': _clean_text(task.get('task_id')),
                    'workload_task_id': _clean_text(task.get('workload_task_id') or task.get('task_id')),
                    'phase_id': _clean_text(task.get('phase_id')),
                    'branch_id': _clean_text(task.get('branch_id')),
                    'capability': _clean_text(task.get('capability')),
                    'status': status,
                    'decomposition_level': depth,
                    'cycle_policy': 'prepare_gather_execute_verify_repair_or_freeze',
                    'stages': stages,
                    'depends_on': _clean_string_list(task.get('depends_on')),
                    'parent_task_ids': _clean_string_list(task.get('parent_task_ids')),
                    'child_task_ids': _clean_string_list(task.get('child_task_ids')),
                    'semantic_intent': _clean_text(task.get('semantic_intent') or task.get('intent')),
                    'review_criteria': _clean_string_list(task.get('review_criteria')),
                    'semantic_review_lens': _clean_text(lens.get('lens')),
                    'success_definition': _clean_text(lens.get('success_definition')),
                    'failure_modes': _clean_string_list(lens.get('failure_modes')),
                    'evidence_requirements': _clean_string_list(lens.get('evidence_requirements')),
                    'semantic_review_lens_contract': lens,
                }
            )
        )
    return _json_safe(
        {
            'kind': 'ollmo.recursive_cycle_review',
            'status': 'active' if cycle_tasks else 'not_applicable',
            'authority': 'read_model_only_not_execution_engine',
            'policy': 'every_subtask_uses_the_same_cycle_without_a_hard_depth_cap',
            'cycle': ['prepare', 'gather_evidence', 'execute', 'verify', 'repair_or_freeze'],
            'task_count': len(cycle_tasks),
            'max_depth': max_depth,
            'depth_counts': depth_counts,
            'status_counts': status_counts,
            'stage_counts': stage_counts,
            'tasks': cycle_tasks,
        }
    )


def _candidate_refs(candidate_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = candidate_graph.get('candidates') if isinstance(candidate_graph.get('candidates'), list) else []
    refs: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        payload = _candidate_decision_ref(candidate)
        for key in ('source', 'status', 'capability', 'output_type', 'task_id', 'phase_id', 'branch_id'):
            value = candidate.get(key)
            if value not in (None, '', [], {}):
                payload[key] = _json_safe(value)
        if payload:
            refs.append(payload)
    return refs


def _orientation_source_ref(source: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            'candidate_id': _clean_text(source.get('candidate_id')),
            'obligation_id': _clean_text(source.get('obligation_id')),
            'task_id': _clean_text(source.get('task_id') or source.get('workload_task_id')),
            'workload_task_id': _clean_text(source.get('workload_task_id') or source.get('task_id')),
            'phase_id': _clean_text(source.get('phase_id')),
            'branch_id': _clean_text(source.get('branch_id')),
            'capability': _clean_text(source.get('capability')),
            'output_type': _clean_text(source.get('output_type')),
            'source': _clean_text(source.get('source')),
            'source_category': _clean_text(source.get('source_category') or source.get('category')),
        }
    )


def _aspiration_frame(
    *,
    source: Mapping[str, Any],
    index: int,
    action: str,
    reason: str,
    evidence_refs: Sequence[str],
    allowed_actions: Sequence[str],
    priority: str = 'normal',
) -> dict[str, Any]:
    token = _stable_contract_token(
        source.get('branch_id'),
        source.get('phase_id'),
        source.get('task_id'),
        source.get('workload_task_id'),
        source.get('obligation_id'),
        source.get('candidate_id'),
        action,
        fallback=f'item-{index}',
    )
    return _json_safe(
        {
            'kind': 'ollmo.aspiration_frame',
            'frame_id': f'aspiration-{index}-{token}',
            'status': 'attention_pending',
            'authority': 'advisory_read_model_only',
            'source_kind': 'aspiration_frame',
            'aspiration_action': action,
            'recommended_action': action,
            'allowed_actions': _clean_string_list(allowed_actions),
            'allowed_transitions': _clean_string_list(allowed_actions),
            'priority': priority,
            'reason': reason,
            'evidence_refs': _clean_string_list(evidence_refs),
            'source_ref': _orientation_source_ref(source),
            'candidate_id': _clean_text(source.get('candidate_id')),
            'obligation_id': _clean_text(source.get('obligation_id')),
            'task_id': _clean_text(source.get('task_id') or source.get('workload_task_id')),
            'workload_task_id': _clean_text(source.get('workload_task_id') or source.get('task_id')),
            'phase_id': _clean_text(source.get('phase_id')),
            'branch_id': _clean_text(source.get('branch_id')),
            'capability': _clean_text(source.get('capability')),
            'output_type': _clean_text(source.get('output_type')),
            'movement_policy': 'aspire_inquire_commit_without_overriding_runtime_truth',
            'non_authority_boundary': 'aspiration_only_runtime_contracts_closure_decide_truth',
        }
    )


def _aspiration_review(
    *,
    candidate_graph: Mapping[str, Any],
    promotion_review: Mapping[str, Any],
    workload_graph: Mapping[str, Any],
    workload_proposal_review: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    reconsideration_items: Sequence[Mapping[str, Any]],
    promotion_suggestion_items: Sequence[Mapping[str, Any]],
    semantic_review_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    index = 1
    coverage = (
        workload_proposal_review.get('coverage')
        if isinstance(workload_proposal_review.get('coverage'), Mapping)
        else {}
    )
    coverage_status = _status(coverage.get('status')) if isinstance(coverage, Mapping) else ''
    if coverage_status in {'missing', 'partial'}:
        missing_task_ids = _clean_string_list(coverage.get('missing_task_ids'))
        targets = missing_task_ids or ['workload_proposal_coverage']
        for target in targets:
            source = {
                'source': 'workload_proposal_review.coverage',
                'task_id': target if target != 'workload_proposal_coverage' else '',
                'status': coverage_status,
            }
            frames.append(
                _aspiration_frame(
                    source=source,
                    index=index,
                    action='review_underplanned_graph',
                    reason='workload proposal coverage indicates the graph may be too small for the current intent',
                    evidence_refs=['workload_proposal_review.coverage', coverage_status, target],
                    allowed_actions=['review_underplanned_graph', 'expand_candidate_space', 'raise_solution_bar'],
                    priority='high',
                )
            )
            index += 1

    for item in promotion_suggestion_items:
        if not isinstance(item, Mapping):
            continue
        frames.append(
            _aspiration_frame(
                source=item,
                index=index,
                action='expand_candidate_space',
                reason='workload task carries promotion suggestions that should stay visible before minimal collapse',
                evidence_refs=[
                    _clean_text(item.get('task_id') or item.get('workload_task_id')),
                    'decision_contract.promotion_suggestions',
                ],
                allowed_actions=['expand_candidate_space', 'review_underplanned_graph', 'preserve_possibility_space'],
                priority='normal',
            )
        )
        index += 1

    for item in reconsideration_items:
        if not isinstance(item, Mapping):
            continue
        frames.append(
            _aspiration_frame(
                source=item,
                index=index,
                action='preserve_possibility_space',
                reason='reconsiderable candidate remains possible but non-executable until relevance is reviewed',
                evidence_refs=[
                    _clean_text(item.get('candidate_id')),
                    _clean_text(item.get('decision')),
                ],
                allowed_actions=['preserve_possibility_space', 'expand_candidate_space', 'avoid_minimal_collapse'],
                priority='normal',
            )
        )
        index += 1

    for item in semantic_review_items:
        if not isinstance(item, Mapping):
            continue
        frames.append(
            _aspiration_frame(
                source=item,
                index=index,
                action='raise_solution_bar',
                reason='semantic review criteria mean existence alone is below the intended solution bar',
                evidence_refs=[
                    _clean_text(item.get('task_id') or item.get('workload_task_id')),
                    *_clean_string_list(item.get('review_criteria')),
                ],
                allowed_actions=['raise_solution_bar', 'review_underplanned_graph', 'preserve_possibility_space'],
                priority='normal',
            )
        )
        index += 1

    candidate_count = int(candidate_graph.get('candidate_count') or 0)
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    open_obligation_count = sum(
        1
        for item in obligations
        if isinstance(item, Mapping) and _status(item.get('status'), fallback='pending') in _OPEN_OBLIGATION_STATUSES
    )
    promoted_count = int((promotion_review.get('counts') or {}).get('promoted') or 0) if isinstance(promotion_review.get('counts'), Mapping) else 0
    if (
        not frames
        and open_obligation_count
        and candidate_count <= max(1, promoted_count)
        and len([task for task in tasks if isinstance(task, Mapping)]) == 0
    ):
        source = {
            'source': 'decision_contract.minimal_shape',
            'status': 'open',
        }
        frames.append(
            _aspiration_frame(
                source=source,
                index=index,
                action='avoid_minimal_collapse',
                reason='open promoted work has no visible workload-task possibility space; keep the solution bar inspectable',
                evidence_refs=['open_obligation_count', str(open_obligation_count), 'candidate_count', str(candidate_count)],
                allowed_actions=['avoid_minimal_collapse', 'review_underplanned_graph', 'raise_solution_bar'],
                priority='low',
            )
        )

    action_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for frame in frames:
        action = _clean_text(frame.get('aspiration_action')) or 'unknown'
        priority = _clean_text(frame.get('priority')) or 'normal'
        action_counts[action] = action_counts.get(action, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.aspiration_review',
            'status': 'active' if frames else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'keep_possibility_solution_ambition_and_non_minimal_planning_visible_without_creating_truth',
            'philosophy': 'great_faith_as_active_aspiration_not_gullibility',
            'frame_count': len(frames),
            'action_counts': action_counts,
            'priority_counts': priority_counts,
            'candidate_count': candidate_count,
            'candidate_refs': _candidate_refs(candidate_graph)[:12],
            'frames': frames,
        }
    )


def _commitment_frame(
    *,
    source: Mapping[str, Any],
    index: int,
    action: str,
    recommended_transition: str,
    reason: str,
    evidence_refs: Sequence[str],
    allowed_transitions: Sequence[str],
    priority: str = 'normal',
) -> dict[str, Any]:
    token = _stable_contract_token(
        source.get('branch_id'),
        source.get('phase_id'),
        source.get('task_id'),
        source.get('workload_task_id'),
        source.get('obligation_id'),
        source.get('candidate_id'),
        action,
        fallback=f'item-{index}',
    )
    return _json_safe(
        {
            'kind': 'ollmo.commitment_frame',
            'frame_id': f'commitment-{index}-{token}',
            'status': 'attention_pending',
            'authority': 'advisory_read_model_only',
            'source_kind': 'commitment_frame',
            'commitment_action': action,
            'recommended_action': action,
            'recommended_transition': recommended_transition,
            'allowed_transitions': _clean_string_list(allowed_transitions),
            'priority': priority,
            'reason': reason,
            'evidence_refs': _clean_string_list(evidence_refs),
            'source_ref': _orientation_source_ref(source),
            'candidate_id': _clean_text(source.get('candidate_id')),
            'obligation_id': _clean_text(source.get('obligation_id')),
            'task_id': _clean_text(source.get('task_id') or source.get('workload_task_id')),
            'workload_task_id': _clean_text(source.get('workload_task_id') or source.get('task_id')),
            'phase_id': _clean_text(source.get('phase_id')),
            'branch_id': _clean_text(source.get('branch_id')),
            'capability': _clean_text(source.get('capability')),
            'output_type': _clean_text(source.get('output_type')),
            'movement_policy': 'commit_to_the_right_sized_sufficient_transition_without_forcing_completion',
            'non_authority_boundary': 'commitment_only_runtime_contracts_closure_decide_truth',
        }
    )


def _commitment_review(
    *,
    active_reconsideration_review: Mapping[str, Any],
    semantic_quality_review: Mapping[str, Any],
    recursive_cycle_review: Mapping[str, Any],
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    index = 1
    active_decisions = (
        active_reconsideration_review.get('decisions')
        if isinstance(active_reconsideration_review, Mapping) and isinstance(active_reconsideration_review.get('decisions'), list)
        else []
    )
    for decision in active_decisions:
        if not isinstance(decision, Mapping):
            continue
        transition = _semantic_action_for_active_decision(decision)
        category = _clean_text(decision.get('source_category'))
        action = (
            'commit_to_right_sized_sufficient_transition'
            if category in {'blocked_obligation', 'blocked_workload_task', 'repair_candidate', 'semantic_review_candidate'}
            else transition
        )
        frames.append(
            _commitment_frame(
                source=decision,
                index=index,
                action=action,
                recommended_transition=transition,
                reason=(
                    _clean_text(decision.get('reason'))
                    or 'active reconsideration identified a bounded state transition that should not remain vague'
                ),
                evidence_refs=[
                    _clean_text(decision.get('decision_id')),
                    _clean_text(decision.get('source')),
                    category,
                ],
                allowed_transitions=decision.get('allowed_outcomes') if isinstance(decision.get('allowed_outcomes'), list) else [],
                priority='high' if _clean_text(decision.get('severity')) == 'blocking' else 'normal',
            )
        )
        index += 1

    quality_contracts = (
        semantic_quality_review.get('contracts')
        if isinstance(semantic_quality_review, Mapping) and isinstance(semantic_quality_review.get('contracts'), list)
        else []
    )
    for contract in quality_contracts:
        if not isinstance(contract, Mapping):
            continue
        frames.append(
            _commitment_frame(
                source=contract,
                index=index,
                action='commit_to_right_sized_sufficient_transition',
                recommended_transition='semantic_review',
                reason='quality truth needs an explicit semantic review transition instead of indefinite pending',
                evidence_refs=[
                    _clean_text(contract.get('quality_review_id')),
                    *_clean_string_list(contract.get('review_criteria')),
                ],
                allowed_transitions=['semantic_review', 'manual_review', 'truthful_freeze_after_review'],
                priority='high',
            )
        )
        index += 1

    recursive_tasks = (
        recursive_cycle_review.get('tasks')
        if isinstance(recursive_cycle_review, Mapping) and isinstance(recursive_cycle_review.get('tasks'), list)
        else []
    )
    for task in recursive_tasks:
        if not isinstance(task, Mapping):
            continue
        status = _status(task.get('status'), fallback='pending')
        if status in _BLOCKED_OBLIGATION_STATUSES:
            transition = 'repair_dependency_chain'
            priority = 'high'
        elif status in _OPEN_OBLIGATION_STATUSES:
            transition = 'continue_branch_local_work'
            priority = 'normal'
        else:
            continue
        frames.append(
            _commitment_frame(
                source=task,
                index=index,
                action=transition,
                recommended_transition=transition,
                reason='recursive task cycle has a next local movement and should not stay as abstract pending state',
                evidence_refs=[
                    _clean_text(task.get('cycle_task_id')),
                    _clean_text(task.get('cycle_policy')),
                    status,
                ],
                allowed_transitions=['continue_branch_local_work', 'repair_dependency_chain', 'repair_branch_contract', 'clarify', 'waive_with_evidence', 'supersede_with_replacement_truth', 'truthful_freeze_after_review'],
                priority=priority,
            )
        )
        index += 1

    action_counts: dict[str, int] = {}
    transition_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for frame in frames:
        action = _clean_text(frame.get('commitment_action')) or 'unknown'
        transition = _clean_text(frame.get('recommended_transition')) or action
        priority = _clean_text(frame.get('priority')) or 'normal'
        action_counts[action] = action_counts.get(action, 0) + 1
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.commitment_review',
            'status': 'active' if frames else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'propose_the_right_sized_sufficient_transition_without_force_completion',
            'philosophy': 'great_courage_as_bounded_commitment_not_aggression',
            'frame_count': len(frames),
            'action_counts': action_counts,
            'transition_counts': transition_counts,
            'priority_counts': priority_counts,
            'frames': frames,
        }
    )


def _learning_orientation(
    accepted_learning: Mapping[str, Any],
    *,
    target_areas: set[str],
) -> dict[str, Any]:
    if not isinstance(accepted_learning, Mapping):
        return {}
    hints = accepted_learning.get('hints') if isinstance(accepted_learning.get('hints'), list) else []
    matched: list[dict[str, Any]] = []
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        target_area = _clean_text(hint.get('target_area'))
        if target_areas and target_area not in target_areas:
            continue
        matched.append(
            _json_safe(
                {
                    'learning_id': _clean_text(hint.get('learning_id')),
                    'candidate_id': _clean_text(hint.get('candidate_id')),
                    'target_area': target_area,
                    'hint': _clean_text(hint.get('hint')),
                    'case_kinds': hint.get('case_kinds') if isinstance(hint.get('case_kinds'), Mapping) else {},
                    'authority': _clean_text(hint.get('authority')) or _clean_text(accepted_learning.get('authority')) or 'soft_hint',
                    'allowed_use': _clean_text(hint.get('allowed_use')) or 'soft_hint_only',
                }
            )
        )
    if not matched:
        return {}
    return _json_safe(
        {
            'kind': 'ollmo.semantic_decision_learning_orientation',
            'authority': _clean_text(accepted_learning.get('authority')) or 'soft_hint',
            'allowed_use': 'orientation_only_not_decision_proof',
            'hint_count': len(matched),
            'hints': matched,
        }
    )


def _semantic_role_orientation_review(semantic_role_profile: Mapping[str, Any]) -> dict[str, Any]:
    profile = semantic_role_profile if isinstance(semantic_role_profile, Mapping) else {}
    orientation = (
        profile.get('semantic_role_orientation')
        if isinstance(profile.get('semantic_role_orientation'), Mapping)
        else profile
    )
    mode = _clean_text(orientation.get('mode') or profile.get('mode'))
    if not mode:
        return _json_safe(
            {
                'kind': 'ollmo.semantic_role_orientation_review',
                'status': 'idle',
                'authority': 'advisory_read_model_only',
                'policy': 'semantic_roles_are_available_only_when_present_and_never_create_truth',
                'frame_count': 0,
                'frames': [],
            }
        )

    suggested_lenses = _clean_string_list(orientation.get('suggested_semantic_review_lenses')) or ['worker']
    attention_biases = _clean_string_list(orientation.get('attention_biases'))
    allowed_uses = _clean_string_list(orientation.get('allowed_uses')) or [
        'orient_candidate_review',
        'bias_controlled_attention_only',
        'suggest_review_lens_when_no_stronger_contract_lens_exists',
    ]
    forbidden_uses = _clean_string_list(orientation.get('forbidden_uses')) or [
        'do_not_override_semantic_review_lens',
        'do_not_change_work_graph_shape_without_promotion_review',
        'do_not_execute_waive_supersede_or_freeze',
    ]
    frames: list[dict[str, Any]] = []
    for index, lens in enumerate(suggested_lenses, start=1):
        token = _stable_contract_token(mode, lens, fallback=f'item-{index}')
        frames.append(
            _json_safe(
                {
                    'kind': 'ollmo.semantic_role_orientation_frame',
                    'frame_id': f'semantic-role-orientation-{index}-{token}',
                    'status': 'attention_available',
                    'authority': 'advisory_read_model_only',
                    'source_kind': 'semantic_role_orientation_frame',
                    'mode': mode,
                    'mode_source': _clean_text(orientation.get('mode_source') or profile.get('mode_source')),
                    'semantic_review_lens': lens,
                    'semantic_role_id': _clean_text((semantic_role_for_lens(lens) or {}).get('role_id')),
                    'recommended_action': _clean_text(orientation.get('recommended_action')) or 'orient_attention_only',
                    'allowed_transitions': [
                        'orient_attention_only',
                        'ignore_if_contract_conflicts',
                        'manual_review',
                    ],
                    'allowed_uses': allowed_uses,
                    'forbidden_uses': forbidden_uses,
                    'attention_biases': attention_biases,
                    'risk_if_over_applied': _clean_text(orientation.get('risk_if_over_applied')),
                    'evidence_refs': _clean_string_list(orientation.get('evidence_refs')) or ['semantic_role_profile', mode],
                    'reason': (
                        'semantic roles are global advisory orientation, not planner, graph, routing, '
                        'waiver, supersession, or freeze authority'
                    ),
                    'non_authority_boundary': _clean_text(orientation.get('non_authority_boundary'))
                    or 'semantic_role_orientation_only_contracts_lenses_runtime_and_closure_decide_truth',
                }
            )
        )

    lens_counts: dict[str, int] = {}
    for frame in frames:
        lens = _clean_text(frame.get('semantic_review_lens')) or 'unknown'
        lens_counts[lens] = lens_counts.get(lens, 0) + 1
    return _json_safe(
        {
            'kind': 'ollmo.semantic_role_orientation_review',
            'status': 'active',
            'authority': 'advisory_read_model_only',
            'policy': 'compile_semantic_roles_into_global_orientation_frames_without_parallel_control_authority',
            'mode': mode,
            'mode_source': _clean_text(orientation.get('mode_source') or profile.get('mode_source')),
            'frame_count': len(frames),
            'lens_counts': lens_counts,
            'frames': frames,
            'non_authority_boundary': 'semantic_role_orientation_cannot_override_contract_lens_runtime_or_closure_truth',
        }
    )


def _semantic_action_for_active_decision(decision: Mapping[str, Any]) -> str:
    category = _clean_text(decision.get('source_category'))
    recommended = _clean_text(decision.get('recommended_action'))
    review_type = _clean_text(decision.get('review_type'))
    if category in {'blocked_obligation', 'blocked_workload_task'}:
        if 'branch_contract' in recommended:
            return 'repair_branch_contract'
        return 'repair_dependency_chain'
    if category in {'open_obligation', 'open_workload_task'}:
        return 'continue_branch_local_work'
    if category == 'reconsiderable_candidate' or review_type == 'promotion_relevance_review':
        return 'keep_reserved_or_promote_after_relevance_review'
    if category in {'waived_obligation', 'waived_candidate', 'waiver_candidate'}:
        return 'waive_with_evidence'
    if category in {'superseded_obligation', 'superseded_candidate', 'supersession_candidate'}:
        return 'supersede_with_replacement_truth'
    if category == 'repair_candidate':
        if 'branch_contract' in recommended:
            return 'repair_branch_contract'
        return 'repair_dependency_chain'
    if category == 'semantic_review_candidate':
        return 'semantic_review'
    return 'manual_review'


def _semantic_decision_source_ref(item: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            'candidate_id': _clean_text(item.get('candidate_id')),
            'obligation_id': _clean_text(item.get('obligation_id')),
            'task_id': _clean_text(item.get('task_id') or item.get('workload_task_id')),
            'workload_task_id': _clean_text(item.get('workload_task_id') or item.get('task_id')),
            'phase_id': _clean_text(item.get('phase_id')),
            'branch_id': _clean_text(item.get('branch_id')),
            'capability': _clean_text(item.get('capability')),
            'output_type': _clean_text(item.get('output_type')),
            'semantic_review_lens': _clean_text(item.get('semantic_review_lens')),
        }
    )


def _semantic_decision_proposal(
    *,
    source: Mapping[str, Any],
    index: int,
    decision_action: str,
    reason: str,
    source_kind: str,
    confidence: float,
    allowed_transitions: Sequence[str],
    evidence_refs: Sequence[str],
    learning_orientation: Mapping[str, Any],
) -> dict[str, Any]:
    lens = _semantic_review_lens_payload(source, source_kind=source_kind)
    token = _stable_contract_token(
        source.get('branch_id'),
        source.get('phase_id'),
        source.get('task_id'),
        source.get('workload_task_id'),
        source.get('obligation_id'),
        source.get('candidate_id'),
        source_kind,
        fallback=f'item-{index}',
    )
    return _json_safe(
        {
            'kind': 'ollmo.semantic_decision_proposal',
            'proposal_id': f'semantic-decision-{index}-{token}',
            'status': 'advisory',
            'authority': 'advisory_read_model_only',
            'source_kind': source_kind,
            'decision_action': decision_action,
            'recommended_transition': decision_action,
            'allowed_transitions': _clean_string_list(allowed_transitions),
            'confidence': round(max(0.0, min(1.0, float(confidence))), 2),
            'reason': reason,
            'evidence_refs': _clean_string_list(evidence_refs),
            'source_ref': _semantic_decision_source_ref(source),
            'candidate_id': _clean_text(source.get('candidate_id')),
            'obligation_id': _clean_text(source.get('obligation_id')),
            'task_id': _clean_text(source.get('task_id') or source.get('workload_task_id')),
            'workload_task_id': _clean_text(source.get('workload_task_id') or source.get('task_id')),
            'phase_id': _clean_text(source.get('phase_id')),
            'branch_id': _clean_text(source.get('branch_id')),
            'capability': _clean_text(source.get('capability')),
            'output_type': _clean_text(source.get('output_type')),
            'review_type': _clean_text(source.get('review_type')),
            'quality_review_id': _clean_text(source.get('quality_review_id')),
            'semantic_review_lens': _clean_text(source.get('semantic_review_lens')) or _clean_text(lens.get('lens')),
            'success_definition': _clean_text(source.get('success_definition')) or _clean_text(lens.get('success_definition')),
            'failure_modes': _clean_string_list(source.get('failure_modes')) or _clean_string_list(lens.get('failure_modes')),
            'evidence_requirements': _clean_string_list(source.get('evidence_requirements')) or _clean_string_list(lens.get('evidence_requirements')),
            'semantic_review_lens_contract': lens,
            'learning_orientation': learning_orientation,
            'non_authority_boundary': 'proposal_only_runtime_contracts_closure_decide_truth',
        }
    )


def _semantic_decision_review(
    *,
    active_reconsideration_review: Mapping[str, Any],
    semantic_quality_review: Mapping[str, Any],
    recursive_cycle_review: Mapping[str, Any],
    aspiration_review: Mapping[str, Any],
    commitment_review: Mapping[str, Any],
    semantic_role_orientation_review: Mapping[str, Any],
    accepted_learning: Mapping[str, Any],
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    index = 1
    active_decisions = (
        active_reconsideration_review.get('decisions')
        if isinstance(active_reconsideration_review, Mapping) and isinstance(active_reconsideration_review.get('decisions'), list)
        else []
    )
    for decision in active_decisions:
        if not isinstance(decision, Mapping):
            continue
        decision_action = _semantic_action_for_active_decision(decision)
        target_areas = {
            'ghost_decision_contract_policy',
            'reconsideration_policy',
        }
        if decision_action == 'semantic_review':
            target_areas.add('semantic_review_policy')
        if decision_action.startswith('repair_'):
            target_areas.add('closure_review_policy')
        if decision_action == 'waive_with_evidence':
            target_areas.add('workload_decision_policy')
        if decision_action == 'supersede_with_replacement_truth':
            target_areas.add('supersession_policy')
        proposals.append(
            _semantic_decision_proposal(
                source=decision,
                index=index,
                decision_action=decision_action,
                source_kind='active_reconsideration_decision',
                reason=(
                    _clean_text(decision.get('reason'))
                    or 'active reconsideration marked this state transition for review'
                ),
                confidence=0.72 if _clean_text(decision.get('severity')) == 'blocking' else 0.62,
                allowed_transitions=decision.get('allowed_outcomes') if isinstance(decision.get('allowed_outcomes'), list) else [],
                evidence_refs=[
                    _clean_text(decision.get('decision_id')),
                    _clean_text(decision.get('source')),
                    _clean_text(decision.get('source_category')),
                ],
                learning_orientation=_learning_orientation(accepted_learning, target_areas=target_areas),
            )
        )
        index += 1

    quality_contracts = (
        semantic_quality_review.get('contracts')
        if isinstance(semantic_quality_review, Mapping) and isinstance(semantic_quality_review.get('contracts'), list)
        else []
    )
    for contract in quality_contracts:
        if not isinstance(contract, Mapping):
            continue
        proposals.append(
            _semantic_decision_proposal(
                source=contract,
                index=index,
                decision_action='semantic_review',
                source_kind='semantic_quality_contract',
                reason='semantic quality criteria require review evidence before quality truth can freeze',
                confidence=0.78,
                allowed_transitions=['semantic_review', 'manual_review', 'truthful_freeze_after_review'],
                evidence_refs=[
                    _clean_text(contract.get('quality_review_id')),
                    *_clean_string_list(contract.get('review_criteria')),
                ],
                learning_orientation=_learning_orientation(
                    accepted_learning,
                    target_areas={'semantic_review_policy', 'ghost_decision_contract_policy'},
                ),
            )
        )
        index += 1

    recursive_tasks = (
        recursive_cycle_review.get('tasks')
        if isinstance(recursive_cycle_review, Mapping) and isinstance(recursive_cycle_review.get('tasks'), list)
        else []
    )
    for task in recursive_tasks:
        if not isinstance(task, Mapping):
            continue
        status = _status(task.get('status'), fallback='pending')
        if status in _BLOCKED_OBLIGATION_STATUSES:
            action = 'repair_dependency_chain'
            confidence = 0.68
        elif status in _OPEN_OBLIGATION_STATUSES:
            action = 'continue_branch_local_work'
            confidence = 0.54
        else:
            continue
        proposals.append(
            _semantic_decision_proposal(
                source=task,
                index=index,
                decision_action=action,
                source_kind='recursive_cycle_task',
                reason='recursive task cycle remains open or blocked inside the larger request movement',
                confidence=confidence,
                allowed_transitions=['continue_branch_local_work', 'repair_dependency_chain', 'repair_branch_contract', 'semantic_review', 'truthful_freeze'],
                evidence_refs=[
                    _clean_text(task.get('cycle_task_id')),
                    _clean_text(task.get('cycle_policy')),
                    status,
                ],
                learning_orientation=_learning_orientation(
                    accepted_learning,
                    target_areas={'workload_decision_policy', 'closure_review_policy', 'ghost_decision_contract_policy'},
                ),
            )
        )
        index += 1

    aspiration_frames = (
        aspiration_review.get('frames')
        if isinstance(aspiration_review, Mapping) and isinstance(aspiration_review.get('frames'), list)
        else []
    )
    for frame in aspiration_frames:
        if not isinstance(frame, Mapping):
            continue
        action = _clean_text(frame.get('aspiration_action')) or 'preserve_possibility_space'
        proposals.append(
            _semantic_decision_proposal(
                source=frame,
                index=index,
                decision_action=action,
                source_kind='aspiration_frame',
                reason=(
                    _clean_text(frame.get('reason'))
                    or 'aspiration review marked a possibility or solution-bar risk for advisory review'
                ),
                confidence=0.66 if _clean_text(frame.get('priority')) == 'high' else 0.56,
                allowed_transitions=frame.get('allowed_actions') if isinstance(frame.get('allowed_actions'), list) else [],
                evidence_refs=[
                    _clean_text(frame.get('frame_id')),
                    *_clean_string_list(frame.get('evidence_refs')),
                ],
                learning_orientation=_learning_orientation(
                    accepted_learning,
                    target_areas={'aspiration_policy', 'ghost_decision_contract_policy', 'workload_decision_policy'},
                ),
            )
        )
        index += 1

    commitment_frames = (
        commitment_review.get('frames')
        if isinstance(commitment_review, Mapping) and isinstance(commitment_review.get('frames'), list)
        else []
    )
    for frame in commitment_frames:
        if not isinstance(frame, Mapping):
            continue
        action = _clean_text(frame.get('recommended_transition')) or _clean_text(frame.get('commitment_action')) or 'commit_to_right_sized_sufficient_transition'
        proposals.append(
            _semantic_decision_proposal(
                source=frame,
                index=index,
                decision_action=action,
                source_kind='commitment_frame',
                reason=(
                    _clean_text(frame.get('reason'))
                    or 'commitment review marked a bounded transition that should not remain abstract'
                ),
                confidence=0.74 if _clean_text(frame.get('priority')) == 'high' else 0.64,
                allowed_transitions=frame.get('allowed_transitions') if isinstance(frame.get('allowed_transitions'), list) else [],
                evidence_refs=[
                    _clean_text(frame.get('frame_id')),
                    *_clean_string_list(frame.get('evidence_refs')),
                ],
                learning_orientation=_learning_orientation(
                    accepted_learning,
                    target_areas={'commitment_policy', 'semantic_decision_policy', 'ghost_decision_contract_policy'},
                ),
            )
        )
        index += 1

    semantic_role_orientation_frames = (
        semantic_role_orientation_review.get('frames')
        if isinstance(semantic_role_orientation_review, Mapping)
        and isinstance(semantic_role_orientation_review.get('frames'), list)
        else []
    )
    for frame in semantic_role_orientation_frames:
        if not isinstance(frame, Mapping):
            continue
        proposals.append(
            _semantic_decision_proposal(
                source=frame,
                index=index,
                decision_action='orient_attention_only',
                source_kind='semantic_role_orientation_frame',
                reason=(
                    _clean_text(frame.get('reason'))
                    or 'semantic role profile orientation may guide attention but cannot change the graph or runtime truth'
                ),
                confidence=0.42,
                allowed_transitions=frame.get('allowed_transitions') if isinstance(frame.get('allowed_transitions'), list) else [],
                evidence_refs=[
                    _clean_text(frame.get('frame_id')),
                    *_clean_string_list(frame.get('evidence_refs')),
                ],
                learning_orientation=_learning_orientation(
                    accepted_learning,
                    target_areas={'ghost_decision_contract_policy', 'semantic_decision_policy'},
                ),
            )
        )
        index += 1

    action_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for proposal in proposals:
        action = _clean_text(proposal.get('decision_action')) or 'unknown'
        source = _clean_text(proposal.get('source_kind')) or 'unknown'
        action_counts[action] = action_counts.get(action, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.semantic_decision_review',
            'status': 'active' if proposals else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'propose_state_transitions_without_changing_runtime_truth',
            'proposal_count': len(proposals),
            'action_counts': action_counts,
            'source_counts': source_counts,
            'learning_policy': {
                key: accepted_learning.get(key)
                for key in ('status', 'authority', 'runtime_effect', 'hint_count', 'allowed_use')
                if isinstance(accepted_learning, Mapping) and accepted_learning.get(key) not in (None, '', [], {})
            },
            'proposals': proposals,
        }
    )


def _attention_priority_for_source(source: Mapping[str, Any], *, source_kind: str) -> str:
    severity = _clean_text(source.get('severity')).lower()
    if severity == 'blocking':
        return 'blocking'
    if source_kind in {'aspiration_frame', 'commitment_frame'}:
        priority = _clean_text(source.get('priority')).lower()
        if priority in {'blocking', 'high', 'normal', 'low'}:
            return priority
    if source_kind == 'semantic_quality_contract':
        return 'high'
    if source_kind == 'recursive_cycle_task' and _status(source.get('status')) in _BLOCKED_OBLIGATION_STATUSES:
        return 'high'
    if source_kind == 'semantic_decision_proposal':
        try:
            confidence = float(source.get('confidence') or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= 0.75:
            return 'high'
        if confidence <= 0.55:
            return 'low'
    if source_kind == 'accepted_learning_hint':
        return 'low'
    if _clean_text(source.get('source_category')) in {'open_obligation', 'open_workload_task'}:
        return 'high'
    return 'normal'


def _attention_scope_for_source(source: Mapping[str, Any], *, source_kind: str) -> str:
    if source_kind == 'active_reconsideration_decision':
        category = _clean_text(source.get('source_category'))
        if category == 'reconsiderable_candidate':
            return 'candidate_relevance'
        if category in {'waived_obligation', 'waived_candidate', 'waiver_candidate'}:
            return 'waiver_evidence'
        if category in {'superseded_obligation', 'superseded_candidate', 'supersession_candidate'}:
            return 'supersession_truth'
        if category in {'blocked_obligation', 'blocked_workload_task'}:
            return 'block_resolution'
        return 'runtime_state_transition'
    if source_kind == 'semantic_quality_contract':
        return 'semantic_quality'
    if source_kind == 'recursive_cycle_task':
        return 'recursive_subtask_cycle'
    if source_kind == 'semantic_decision_proposal':
        return 'semantic_transition_proposal'
    if source_kind == 'aspiration_frame':
        return 'aspiration_possibility'
    if source_kind == 'commitment_frame':
        return 'commitment_transition'
    if source_kind == 'semantic_role_orientation_frame':
        return 'semantic_role_orientation'
    if source_kind == 'accepted_learning_hint':
        return 'learning_orientation'
    return 'whole_turn_attention'


def _attention_question_for_source(source: Mapping[str, Any], *, source_kind: str) -> str:
    if source_kind == 'active_reconsideration_decision':
        category = _clean_text(source.get('source_category'))
        if category == 'reconsiderable_candidate':
            return 'Is this reserved possibility currently relevant enough to promote, or should it stay reserved?'
        if category in {'blocked_obligation', 'blocked_workload_task'}:
            return 'What is the right-sized truthful repair, wait, clarification, waiver, supersession, or freeze transition for this block?'
        if category in {'open_obligation', 'open_workload_task'}:
            return 'Should this promoted work continue, be repaired, be reviewed, or close by truthful waiver or supersession?'
        if category in {'waived_obligation', 'waived_candidate', 'waiver_candidate'}:
            return 'Does explicit release evidence support the waiver, or should the work remain open?'
        if category in {'superseded_obligation', 'superseded_candidate', 'supersession_candidate'}:
            return 'Does replacement runtime truth support supersession, or should the work be reopened?'
        return 'Which bounded state transition should be considered for this runtime signal?'
    if source_kind == 'semantic_quality_contract':
        return 'What semantic evidence is needed before quality can truthfully freeze?'
    if source_kind == 'recursive_cycle_task':
        return 'Which mini-cycle stage is next for this subtask: prepare, gather evidence, execute, verify, repair, or freeze?'
    if source_kind == 'semantic_decision_proposal':
        return 'Should this advisory transition proposal guide the next reviewed action, stay advisory, or be superseded by newer evidence?'
    if source_kind == 'aspiration_frame':
        return 'Does this request need a wider possibility space or higher solution bar before it collapses to minimal work?'
    if source_kind == 'commitment_frame':
        return 'What is the right-sized sufficient transition to commit to without forcing completion or under-scoping the work?'
    if source_kind == 'semantic_role_orientation_frame':
        return 'Should this semantic role hint orient attention here, or should the branch-local contract and review lens ignore it?'
    if source_kind == 'accepted_learning_hint':
        return 'Should this accepted learning orient attention for the current turn without overriding runtime truth?'
    return 'What should receive controlled model attention before the next state transition?'


def _controlled_attention_source_id(source: Mapping[str, Any]) -> str:
    for key in (
        'frame_id',
        'decision_id',
        'quality_review_id',
        'cycle_task_id',
        'proposal_id',
        'learning_id',
        'candidate_id',
        'obligation_id',
        'task_id',
        'workload_task_id',
        'branch_id',
        'phase_id',
    ):
        value = _clean_text(source.get(key))
        if value:
            return value
    return ''


def _controlled_attention_frame(
    *,
    source: Mapping[str, Any],
    index: int,
    source_kind: str,
    allowed_transitions: Sequence[str],
    evidence_refs: Sequence[str],
    reason: str,
    accepted_learning: Mapping[str, Any],
    target_areas: set[str],
) -> dict[str, Any]:
    source_id = _controlled_attention_source_id(source)
    scope = _attention_scope_for_source(source, source_kind=source_kind)
    lens = _semantic_review_lens_payload(source, source_kind=source_kind)
    token = _stable_contract_token(
        source.get('branch_id'),
        source.get('phase_id'),
        source.get('task_id'),
        source.get('workload_task_id'),
        source.get('obligation_id'),
        source.get('candidate_id'),
        source_id,
        scope,
        fallback=f'item-{index}',
    )
    return _json_safe(
        {
            'kind': 'ollmo.controlled_attention_frame',
            'frame_id': f'controlled-attention-{index}-{token}',
            'status': 'attention_pending',
            'authority': 'advisory_read_model_only',
            'scope': scope,
            'source_kind': source_kind,
            'source_id': source_id,
            'priority': _attention_priority_for_source(source, source_kind=source_kind),
            'attention_question': _attention_question_for_source(source, source_kind=source_kind),
            'allowed_transitions': _clean_string_list(allowed_transitions),
            'evidence_refs': _clean_string_list(evidence_refs),
            'reason': reason,
            'target_ref': _semantic_decision_source_ref(source),
            'candidate_id': _clean_text(source.get('candidate_id')),
            'obligation_id': _clean_text(source.get('obligation_id')),
            'task_id': _clean_text(source.get('task_id') or source.get('workload_task_id')),
            'workload_task_id': _clean_text(source.get('workload_task_id') or source.get('task_id')),
            'phase_id': _clean_text(source.get('phase_id')),
            'branch_id': _clean_text(source.get('branch_id')),
            'capability': _clean_text(source.get('capability')),
            'output_type': _clean_text(source.get('output_type')),
            'review_type': _clean_text(source.get('review_type')),
            'quality_review_id': _clean_text(source.get('quality_review_id')),
            'proposal_id': _clean_text(source.get('proposal_id')),
            'decision_action': _clean_text(source.get('decision_action') or source.get('recommended_action')),
            'semantic_review_lens': _clean_text(source.get('semantic_review_lens')) or _clean_text(lens.get('lens')),
            'success_definition': _clean_text(source.get('success_definition')) or _clean_text(lens.get('success_definition')),
            'failure_modes': _clean_string_list(source.get('failure_modes')) or _clean_string_list(lens.get('failure_modes')),
            'semantic_lens_evidence_requirements': _clean_string_list(source.get('evidence_requirements')) or _clean_string_list(lens.get('evidence_requirements')),
            'semantic_review_lens_contract': lens,
            'learning_orientation': _learning_orientation(accepted_learning, target_areas=target_areas),
            'movement_policy': 'possibility_relevance_contract_work_review_repair_or_freeze',
            'non_authority_boundary': 'attention_only_runtime_contracts_closure_decide_truth',
        }
    )


def _controlled_attention_review(
    *,
    active_reconsideration_review: Mapping[str, Any],
    semantic_quality_review: Mapping[str, Any],
    recursive_cycle_review: Mapping[str, Any],
    aspiration_review: Mapping[str, Any],
    commitment_review: Mapping[str, Any],
    semantic_role_orientation_review: Mapping[str, Any],
    semantic_decision_review: Mapping[str, Any],
    accepted_learning: Mapping[str, Any],
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    index = 1

    active_decisions = (
        active_reconsideration_review.get('decisions')
        if isinstance(active_reconsideration_review, Mapping) and isinstance(active_reconsideration_review.get('decisions'), list)
        else []
    )
    for decision in active_decisions:
        if not isinstance(decision, Mapping):
            continue
        decision_action = _semantic_action_for_active_decision(decision)
        target_areas = {'ghost_decision_contract_policy', 'reconsideration_policy'}
        if decision_action.startswith('repair_'):
            target_areas.add('closure_review_policy')
        if decision_action == 'semantic_review':
            target_areas.add('semantic_review_policy')
        if decision_action == 'waive_with_evidence':
            target_areas.add('workload_decision_policy')
        if decision_action == 'supersede_with_replacement_truth':
            target_areas.add('supersession_policy')
        frames.append(
            _controlled_attention_frame(
                source=decision,
                index=index,
                source_kind='active_reconsideration_decision',
                allowed_transitions=decision.get('allowed_outcomes') if isinstance(decision.get('allowed_outcomes'), list) else [],
                evidence_refs=[
                    _clean_text(decision.get('decision_id')),
                    _clean_text(decision.get('source')),
                    _clean_text(decision.get('source_category')),
                ],
                reason=(
                    _clean_text(decision.get('reason'))
                    or 'active reconsideration marked this state for scoped attention'
                ),
                accepted_learning=accepted_learning,
                target_areas=target_areas,
            )
        )
        index += 1

    quality_contracts = (
        semantic_quality_review.get('contracts')
        if isinstance(semantic_quality_review, Mapping) and isinstance(semantic_quality_review.get('contracts'), list)
        else []
    )
    for contract in quality_contracts:
        if not isinstance(contract, Mapping):
            continue
        frames.append(
            _controlled_attention_frame(
                source=contract,
                index=index,
                source_kind='semantic_quality_contract',
                allowed_transitions=['schedule_semantic_review', 'manual_review', 'truthful_freeze_after_review'],
                evidence_refs=[
                    _clean_text(contract.get('quality_review_id')),
                    *_clean_string_list(contract.get('review_criteria')),
                ],
                reason='semantic quality needs focused review evidence before it can become truth',
                accepted_learning=accepted_learning,
                target_areas={'semantic_review_policy', 'ghost_decision_contract_policy'},
            )
        )
        index += 1

    recursive_tasks = (
        recursive_cycle_review.get('tasks')
        if isinstance(recursive_cycle_review, Mapping) and isinstance(recursive_cycle_review.get('tasks'), list)
        else []
    )
    for task in recursive_tasks:
        if not isinstance(task, Mapping):
            continue
        status = _status(task.get('status'), fallback='pending')
        if status not in _OPEN_OBLIGATION_STATUSES and status not in _BLOCKED_OBLIGATION_STATUSES:
            continue
        frames.append(
            _controlled_attention_frame(
                source=task,
                index=index,
                source_kind='recursive_cycle_task',
                allowed_transitions=['prepare', 'gather_evidence', 'execute', 'verify', 'repair_dependency_chain', 'repair_branch_contract', 'semantic_review', 'truthful_freeze'],
                evidence_refs=[
                    _clean_text(task.get('cycle_task_id')),
                    _clean_text(task.get('cycle_policy')),
                    status,
                ],
                reason='recursive subtask remains in the same movement cycle as the whole request',
                accepted_learning=accepted_learning,
                target_areas={'workload_decision_policy', 'closure_review_policy', 'ghost_decision_contract_policy'},
            )
        )
        index += 1

    aspiration_frames = (
        aspiration_review.get('frames')
        if isinstance(aspiration_review, Mapping) and isinstance(aspiration_review.get('frames'), list)
        else []
    )
    for frame in aspiration_frames:
        if not isinstance(frame, Mapping):
            continue
        frames.append(
            _controlled_attention_frame(
                source=frame,
                index=index,
                source_kind='aspiration_frame',
                allowed_transitions=frame.get('allowed_actions') if isinstance(frame.get('allowed_actions'), list) else [],
                evidence_refs=[
                    _clean_text(frame.get('frame_id')),
                    *_clean_string_list(frame.get('evidence_refs')),
                ],
                reason=_clean_text(frame.get('reason')) or 'aspiration review needs scoped attention before minimal collapse',
                accepted_learning=accepted_learning,
                target_areas={'aspiration_policy', 'ghost_decision_contract_policy', 'workload_decision_policy'},
            )
        )
        index += 1

    commitment_frames = (
        commitment_review.get('frames')
        if isinstance(commitment_review, Mapping) and isinstance(commitment_review.get('frames'), list)
        else []
    )
    for frame in commitment_frames:
        if not isinstance(frame, Mapping):
            continue
        frames.append(
            _controlled_attention_frame(
                source=frame,
                index=index,
                source_kind='commitment_frame',
                allowed_transitions=frame.get('allowed_transitions') if isinstance(frame.get('allowed_transitions'), list) else [],
                evidence_refs=[
                    _clean_text(frame.get('frame_id')),
                    *_clean_string_list(frame.get('evidence_refs')),
                ],
                reason=_clean_text(frame.get('reason')) or 'commitment review needs scoped attention before the next transition',
                accepted_learning=accepted_learning,
                target_areas={'commitment_policy', 'semantic_decision_policy', 'ghost_decision_contract_policy'},
            )
        )
        index += 1

    semantic_role_orientation_frames = (
        semantic_role_orientation_review.get('frames')
        if isinstance(semantic_role_orientation_review, Mapping)
        and isinstance(semantic_role_orientation_review.get('frames'), list)
        else []
    )
    for frame in semantic_role_orientation_frames:
        if not isinstance(frame, Mapping):
            continue
        frames.append(
            _controlled_attention_frame(
                source=frame,
                index=index,
                source_kind='semantic_role_orientation_frame',
                allowed_transitions=frame.get('allowed_transitions') if isinstance(frame.get('allowed_transitions'), list) else [],
                evidence_refs=[
                    _clean_text(frame.get('frame_id')),
                    *_clean_string_list(frame.get('evidence_refs')),
                ],
                reason=_clean_text(frame.get('reason')) or 'semantic role hint may orient attention but must not steer execution directly',
                accepted_learning=accepted_learning,
                target_areas={'ghost_decision_contract_policy', 'semantic_decision_policy'},
            )
        )
        index += 1

    proposals = (
        semantic_decision_review.get('proposals')
        if isinstance(semantic_decision_review, Mapping) and isinstance(semantic_decision_review.get('proposals'), list)
        else []
    )
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        action = _clean_text(proposal.get('decision_action'))
        target_areas = {'semantic_decision_policy', 'ghost_decision_contract_policy'}
        if action == 'semantic_review':
            target_areas.add('semantic_review_policy')
        if action.startswith('repair_'):
            target_areas.add('closure_review_policy')
        if action == 'supersede_with_replacement_truth':
            target_areas.add('supersession_policy')
        frames.append(
            _controlled_attention_frame(
                source=proposal,
                index=index,
                source_kind='semantic_decision_proposal',
                allowed_transitions=proposal.get('allowed_transitions') if isinstance(proposal.get('allowed_transitions'), list) else [],
                evidence_refs=[
                    _clean_text(proposal.get('proposal_id')),
                    *_clean_string_list(proposal.get('evidence_refs')),
                ],
                reason=_clean_text(proposal.get('reason')) or 'semantic decision proposal needs reviewed attention before transition',
                accepted_learning=accepted_learning,
                target_areas=target_areas,
            )
        )
        index += 1

    hints = accepted_learning.get('hints') if isinstance(accepted_learning, Mapping) and isinstance(accepted_learning.get('hints'), list) else []
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        frames.append(
            _controlled_attention_frame(
                source=hint,
                index=index,
                source_kind='accepted_learning_hint',
                allowed_transitions=['orient_attention_only', 'ignore_if_current_truth_conflicts', 'manual_review'],
                evidence_refs=[
                    _clean_text(hint.get('learning_id')),
                    _clean_text(hint.get('target_area')),
                ],
                reason='accepted learning may orient attention but cannot override current runtime truth',
                accepted_learning=accepted_learning,
                target_areas={_clean_text(hint.get('target_area'))},
            )
        )
        index += 1

    priority_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    for frame in frames:
        priority = _clean_text(frame.get('priority')) or 'normal'
        scope = _clean_text(frame.get('scope')) or 'unknown'
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        scope_counts[scope] = scope_counts.get(scope, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.controlled_attention_review',
            'status': 'active' if frames else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'focus_model_attention_between_steps_without_granting_execution_authority',
            'attention_cycle': [
                'observe_current_runtime_truth',
                'select_scoped_attention_target',
                'ask_bounded_transition_question',
                'propose_only_allowed_transition',
                'let_runtime_contracts_closure_or_user_confirm_truth',
            ],
            'frame_count': len(frames),
            'priority_counts': priority_counts,
            'scope_counts': scope_counts,
            'frames': frames,
        }
    )


def _semantic_review_lens_review(
    *,
    active_reconsideration_decisions: Sequence[Mapping[str, Any]],
    semantic_quality_contracts: Sequence[Mapping[str, Any]],
    recursive_cycle_tasks: Sequence[Mapping[str, Any]],
    aspiration_frames: Sequence[Mapping[str, Any]],
    commitment_frames: Sequence[Mapping[str, Any]],
    semantic_role_orientation_frames: Sequence[Mapping[str, Any]],
    semantic_decision_proposals: Sequence[Mapping[str, Any]],
    controlled_attention_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sources: list[tuple[str, Mapping[str, Any]]] = []
    for source_kind, records in (
        ('active_reconsideration_decision', active_reconsideration_decisions),
        ('semantic_quality_contract', semantic_quality_contracts),
        ('recursive_cycle_task', recursive_cycle_tasks),
        ('aspiration_frame', aspiration_frames),
        ('commitment_frame', commitment_frames),
        ('semantic_role_orientation_frame', semantic_role_orientation_frames),
    ):
        for record in records:
            if isinstance(record, Mapping):
                sources.append((source_kind, record))

    lenses: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, (source_kind, record) in enumerate(sources, start=1):
        lens = _semantic_review_lens_payload(record, source_kind=source_kind)
        lens_name = _clean_text(lens.get('lens'))
        source_id = _controlled_attention_source_id(record) or _clean_text(record.get('quality_review_id')) or f'item-{index}'
        key = (
            lens_name,
            source_kind,
            source_id,
            _clean_text(record.get('branch_id')),
            _clean_text(record.get('phase_id')),
        )
        if key in seen:
            continue
        seen.add(key)
        lenses.append(
            _json_safe(
                {
                    **lens,
                    'lens_id': f'semantic-lens-{index}-{_stable_contract_token(source_id, lens_name, fallback=f"item-{index}")}',
                    'status': 'advisory',
                    'source_id': source_id,
                    'source_ref': _semantic_decision_source_ref(record),
                    'task_id': _clean_text(record.get('task_id') or record.get('workload_task_id')),
                    'workload_task_id': _clean_text(record.get('workload_task_id') or record.get('task_id')),
                    'phase_id': _clean_text(record.get('phase_id')),
                    'branch_id': _clean_text(record.get('branch_id')),
                    'capability': _clean_text(record.get('capability')),
                    'output_type': _clean_text(record.get('output_type')),
                }
            )
        )

    lens_counts: dict[str, int] = {}
    for lens in lenses:
        name = _clean_text(lens.get('lens')) or 'unknown'
        lens_counts[name] = lens_counts.get(name, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.semantic_review_lens_review',
            'status': 'active' if lenses else 'idle',
            'authority': 'advisory_read_model_only',
            'policy': 'use_internal_review_lenses_to_sharpen_success_definition_without_creating_runtime_truth',
            'lens_count': len(lenses),
            'lens_counts': lens_counts,
            'lenses': lenses,
            'non_authority_boundary': 'review_lenses_do_not_execute_fulfill_waive_supersede_or_freeze',
        }
    )


def _semantic_planning_contract(
    *,
    priorities: Sequence[str],
    coverage: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    reconsideration_items: Sequence[Mapping[str, Any]],
    promotion_suggestion_items: Sequence[Mapping[str, Any]],
    waiver_candidate_items: Sequence[Mapping[str, Any]],
    supersession_items: Sequence[Mapping[str, Any]],
    supersession_candidate_items: Sequence[Mapping[str, Any]],
    semantic_review_items: Sequence[Mapping[str, Any]],
    repair_items: Sequence[Mapping[str, Any]],
    accepted_learning: Mapping[str, Any],
    block_resolution_reflex: Mapping[str, Any],
    active_reconsideration_review: Mapping[str, Any],
    semantic_quality_review: Mapping[str, Any],
    recursive_cycle_review: Mapping[str, Any],
    aspiration_review: Mapping[str, Any],
    commitment_review: Mapping[str, Any],
    semantic_role_orientation_review: Mapping[str, Any],
    semantic_decision_review: Mapping[str, Any],
    controlled_attention_review: Mapping[str, Any],
    semantic_review_lens_review: Mapping[str, Any],
) -> dict[str, Any]:
    open_ids = _obligation_ids_for_statuses(obligations, _OPEN_OBLIGATION_STATUSES)
    blocked_ids = _obligation_ids_for_statuses(obligations, _BLOCKED_OBLIGATION_STATUSES)
    proposal_obligations: list[dict[str, Any]] = []

    coverage_status = _status(coverage.get('status')) if isinstance(coverage, Mapping) else ''
    missing_task_ids = (
        _clean_string_list(coverage.get('missing_task_ids'))
        if isinstance(coverage, Mapping)
        else []
    )
    if coverage_status in {'missing', 'partial'}:
        proposal_obligations.append(
            {
                'action': 'complete_semantic_proposals_for_existing_tasks',
                'target_task_ids': missing_task_ids,
                'reason': 'workload proposal coverage is incomplete for executable multi-task work',
            }
        )
    if open_ids:
        proposal_obligations.append(
            {
                'action': 'resolve_or_repair_open_promoted_obligations',
                'target_obligation_ids': open_ids,
                'reason': 'promoted work remains owed until fulfilled, blocked, waived, or superseded by runtime truth',
            }
        )
    if blocked_ids:
        proposal_obligations.append(
            {
                'action': 'repair_blocked_obligations_from_their_dependency_or_contract_edge',
                'target_obligation_ids': blocked_ids,
                'reason': 'blocked work needs dependency or branch-contract repair before backend retry',
            }
        )
    reflex_signal_count = (
        int(block_resolution_reflex.get('signal_count') or 0)
        if isinstance(block_resolution_reflex, Mapping)
        else 0
    )
    if reflex_signal_count:
        proposal_obligations.append(
            {
                'action': 'apply_block_resolution_reconsideration_reflex_between_steps',
                'record_count': reflex_signal_count,
                'reason': 'fluid state contains open, blocked, reserved, waived, superseded, repair, or review signals',
            }
        )
    active_reconsideration_count = (
        int(active_reconsideration_review.get('decision_count') or 0)
        if isinstance(active_reconsideration_review, Mapping)
        else 0
    )
    if active_reconsideration_count:
        proposal_obligations.append(
            {
                'action': 'review_active_reconsideration_decisions_before_changing_contract_state',
                'record_count': active_reconsideration_count,
                'reason': 'reflex signals need current relevance, waiver, supersession, repair, or continuation review before state changes',
            }
        )
    semantic_quality_count = (
        int(semantic_quality_review.get('contract_count') or 0)
        if isinstance(semantic_quality_review, Mapping)
        else 0
    )
    if semantic_quality_count:
        proposal_obligations.append(
            {
                'action': 'treat_semantic_quality_as_pending_review_work',
                'record_count': semantic_quality_count,
                'reason': 'subjective quality criteria need a review contract; output existence is insufficient',
            }
        )
    recursive_task_count = (
        int(recursive_cycle_review.get('task_count') or 0)
        if isinstance(recursive_cycle_review, Mapping)
        else 0
    )
    if recursive_task_count:
        proposal_obligations.append(
            {
                'action': 'apply_prepare_gather_execute_verify_repair_or_freeze_cycle_per_subtask',
                'record_count': recursive_task_count,
                'reason': 'depth n stays fluid by giving each subtask the same branch-local movement cycle',
            }
        )
    aspiration_frame_count = (
        int(aspiration_review.get('frame_count') or 0)
        if isinstance(aspiration_review, Mapping)
        else 0
    )
    if aspiration_frame_count:
        proposal_obligations.append(
            {
                'action': 'use_aspiration_review_to_keep_possibility_and_solution_bar_visible',
                'record_count': aspiration_frame_count,
                'reason': 'great faith is modeled as advisory aspiration, not proof or execution authority',
            }
        )
    commitment_frame_count = (
        int(commitment_review.get('frame_count') or 0)
        if isinstance(commitment_review, Mapping)
        else 0
    )
    if commitment_frame_count:
        proposal_obligations.append(
            {
                'action': 'use_commitment_review_to_choose_the_right_sized_sufficient_transition',
                'record_count': commitment_frame_count,
                'reason': 'great courage is modeled as right-sized bounded commitment, not minimalism or force completion',
            }
        )
    semantic_role_orientation_frame_count = (
        int(semantic_role_orientation_review.get('frame_count') or 0)
        if isinstance(semantic_role_orientation_review, Mapping)
        else 0
    )
    if semantic_role_orientation_frame_count:
        proposal_obligations.append(
            {
                'action': 'use_semantic_roles_as_advisory_orientation_only',
                'record_count': semantic_role_orientation_frame_count,
                'reason': 'legacy roles may orient attention but must not remain a parallel planner or routing control path',
            }
        )
    semantic_decision_count = (
        int(semantic_decision_review.get('proposal_count') or 0)
        if isinstance(semantic_decision_review, Mapping)
        else 0
    )
    if semantic_decision_count:
        proposal_obligations.append(
            {
                'action': 'use_semantic_decision_review_as_advisory_next_transition_input',
                'record_count': semantic_decision_count,
                'reason': 'semantic decision proposals combine reconsideration, quality, recursive cycle, and learning orientation without changing runtime truth',
            }
        )
    controlled_attention_count = (
        int(controlled_attention_review.get('frame_count') or 0)
        if isinstance(controlled_attention_review, Mapping)
        else 0
    )
    if controlled_attention_count:
        proposal_obligations.append(
            {
                'action': 'use_controlled_attention_frames_as_scoped_prompt_targets',
                'record_count': controlled_attention_count,
                'reason': 'model attention should focus on bounded transition questions between execution steps instead of replaying the root prompt',
            }
        )
    semantic_review_lens_count = (
        int(semantic_review_lens_review.get('lens_count') or 0)
        if isinstance(semantic_review_lens_review, Mapping)
        else 0
    )
    if semantic_review_lens_count:
        proposal_obligations.append(
            {
                'action': 'apply_semantic_review_lenses_to_expectation_success_and_evidence_checks',
                'record_count': semantic_review_lens_count,
                'reason': 'branches need role-specific review posture so success is checked against the right evidence',
            }
        )
    for action, records, reason in (
        (
            'keep_reconsiderable_candidates_reserved_for_later_review',
            reconsideration_items,
            'reconsiderable candidates are possible future work, not current executable work',
        ),
        (
            'review_advisory_promotion_suggestions_against_current_relevance',
            promotion_suggestion_items,
            'promotion suggestions need promotion review before creating owed work',
        ),
        (
            'review_advisory_waiver_candidates_against_explicit_release_evidence',
            waiver_candidate_items,
            'waiver candidates must not hide missing promoted work',
        ),
        (
            'preserve_superseded_records_as_closed_truth',
            supersession_items,
            'superseded obligations are closed by replacement truth and should not be retried',
        ),
        (
            'review_advisory_supersession_candidates_against_runtime_truth',
            supersession_candidate_items,
            'supersession candidates need closure truth before closing work',
        ),
        (
            'surface_semantic_review_work_without_claiming_it_is_proven',
            semantic_review_items,
            'subjective review criteria require semantic review evidence',
        ),
        (
            'prefer_targeted_repair_candidates_over_blind_retry',
            repair_items,
            'repair candidates describe the right-sized likely recovery edge',
        ),
    ):
        if records:
            proposal_obligations.append(
                {
                    'action': action,
                    'record_count': len(records),
                    'reason': reason,
                }
            )

    return _json_safe(
        {
            'kind': 'ollmo.ghost_semantic_planning_contract',
            'planning_contract_version': 1,
            'authority': 'advisory_read_model_only',
            'planning_cycle': [
                'orient_to_current_turn_and_runtime_truth',
                'enumerate_candidate_possibilities_without_executing_them',
                'assess_current_relevance_and_promotion_need',
                'actively_reconsider_reserved_open_blocked_waived_or_superseded_state_between_steps',
                'propose_branch_local_workload_annotations_for_existing_tasks',
                'attach_semantic_quality_review_when_presence_is_not_enough',
                'apply_the_same_prepare_gather_execute_verify_repair_or_freeze_cycle_per_subtask',
                'keep_possibility_and_solution_ambition_visible_through_aspiration_review',
                'commit_to_the_right_sized_sufficient_transition_through_commitment_review',
                'compile_legacy_semantic_roles_modes_into_advisory_orientation',
                'propose_advisory_semantic_decisions_with_reason_confidence_and_evidence_refs',
                'focus_model_attention_on_bounded_transition_questions_between_steps',
                'propose_evidence_review_repair_reconsideration_waiver_or_supersession_paths',
                'leave_execution_truth_to_runtime_contracts_and_closure_review',
            ],
            'proposal_requirements': [
                'bind_every_proposal_to_an_existing_task_phase_branch_or_obligation_id',
                'state_branch_local_semantic_intent_and_deliverable',
                'name_required_input_refs_or_evidence_before dependent work can run',
                'include deterministic_review_criteria_when_presence_is_not_enough',
                'mark_subjective_quality_checks_as_semantic_review_criteria',
                'preserve_reconsiderable_candidates_without_promoting_them',
                'treat_blocked_state_as_runtime_truth_to_resolve_not_override',
                'suggest_promotion_or_waiver_only_as_advisory_review_inputs',
                'prefer_repair_dependency_chain_or_repair_branch_contract_over_blind_retry_when_inputs_are_missing',
                'review_active_reconsideration_decisions_before_recommending_state_changes',
                'express_quality_as_semantic_review_contracts_not_as_prose_confidence',
                'apply_recursive_cycle_thinking_to_each_subtask_without_depth_caps',
                'use_aspiration_frames_to_avoid_minimal_collapse_without_creating_work',
                'use_commitment_frames_to_prevent_decision_paralysis_without_forcing_success',
                'treat_legacy_semantic_roles_modes_as_orientation_not_planner_or_graph_authority',
                'emit_semantic_decision_proposals_as_advisory_inputs_only',
                'answer_controlled_attention_frames_with_scoped_proposals_not_root_prompt_repetition',
                'use_semantic_review_lenses_to_state_success_evidence_and_failure_modes',
            ],
            'non_authority_boundaries': [
                'do_not_create_executable_topology_from_a_proposal',
                'do_not_execute_candidates_before_promotion',
                'do_not_claim_artifact_or_output_truth',
                'do_not_waive_promoted_work_without_explicit_runtime_review',
                'do_not_supersede_work_without_newer_runtime_truth_or_replacement_edge',
                'do_not_force_completion_when_the_block_itself_is_the_next_truthful_work',
                'do_not_use_accepted_learning_as_promotion_waiver_supersession_or_review_proof',
                'do_not_treat_controlled_attention_as_execution_permission',
                'do_not_treat_aspiration_as_evidence_or_execution_permission',
                'do_not_treat_commitment_as_force_completion_or_freeze_authority',
                'do_not_treat_semantic_roles_as_planner_timeout_branching_or_payload_authority',
            ],
            'task_proposal_fields': [
                'semantic_intent',
                'objective',
                'deliverable',
                'advisory_role',
                'evidence_requirements',
                'input_refs',
                'review_criteria',
                'semantic_review_criteria',
                'promotion_suggestions',
                'waiver_candidates',
                'reconsideration_triggers',
                'repair_candidates',
                'supersession_candidates',
                'learning_hint_refs',
                'execution_contract',
                'active_reconsideration_review',
                'semantic_quality_review',
                'recursive_cycle_review',
                'aspiration_review',
                'commitment_review',
                'semantic_role_orientation_review',
                'semantic_decision_review',
                'controlled_attention_review',
                'semantic_review_lens',
                'success_definition',
                'failure_modes',
                'semantic_review_lens_contract',
            ],
            'current_focus': list(priorities) or ['route_current_phase_against_runtime_truth'],
            'proposal_obligations': proposal_obligations,
            'block_resolution_reflex': block_resolution_reflex,
            'active_reconsideration_review': active_reconsideration_review,
            'semantic_quality_review': semantic_quality_review,
            'recursive_cycle_review': recursive_cycle_review,
            'aspiration_review': aspiration_review,
            'commitment_review': commitment_review,
            'semantic_role_orientation_review': semantic_role_orientation_review,
            'semantic_decision_review': semantic_decision_review,
            'controlled_attention_review': controlled_attention_review,
            'semantic_review_lens_review': semantic_review_lens_review,
            'accepted_learning_policy': {
                key: accepted_learning.get(key)
                for key in ('status', 'authority', 'runtime_effect', 'hint_count', 'allowed_use')
                if accepted_learning.get(key) not in (None, '', [], {})
            },
        }
    )


def _obligation_status_counts(output_obligations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obligation in output_obligations:
        if not isinstance(obligation, Mapping):
            continue
        status = _status(obligation.get('status'), fallback='pending')
        counts[status] = counts.get(status, 0) + 1
    return counts


def _obligation_ids_for_statuses(
    output_obligations: Sequence[Mapping[str, Any]],
    statuses: set[str],
) -> list[str]:
    values: list[str] = []
    for obligation in output_obligations:
        if not isinstance(obligation, Mapping):
            continue
        if _status(obligation.get('status'), fallback='pending') not in statuses:
            continue
        obligation_id = _clean_text(obligation.get('obligation_id'))
        if obligation_id and obligation_id not in values:
            values.append(obligation_id)
    return values


def _learning_hint_summary(accepted_learning_hints: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(accepted_learning_hints, Mapping):
        return {
            'status': 'not_available',
            'authority': 'soft_hint',
            'runtime_effect': 'none',
            'hint_count': 0,
            'allowed_use': 'none',
        }
    hints = accepted_learning_hints.get('hints') if isinstance(accepted_learning_hints.get('hints'), list) else []
    return _json_safe(
        {
            'status': _clean_text(accepted_learning_hints.get('status')) or 'unknown',
            'enabled': accepted_learning_hints.get('enabled') if isinstance(accepted_learning_hints.get('enabled'), bool) else None,
            'authority': _clean_text(accepted_learning_hints.get('authority')) or 'soft_hint',
            'runtime_effect': _clean_text(accepted_learning_hints.get('runtime_effect')) or 'none',
            'hint_count': int(accepted_learning_hints.get('hint_count') or 0),
            'allowed_use': 'orientation_only_not_promotion_authority',
            'hints': [
                {
                    'learning_id': _clean_text(item.get('learning_id')),
                    'candidate_id': _clean_text(item.get('candidate_id')),
                    'target_area': _clean_text(item.get('target_area')),
                    'hint': _clean_text(item.get('hint')),
                    'case_kinds': item.get('case_kinds') if isinstance(item.get('case_kinds'), Mapping) else {},
                    'authority': _clean_text(item.get('authority')),
                    'allowed_use': _clean_text(item.get('allowed_use')),
                }
                for item in hints
                if isinstance(item, Mapping)
            ],
        }
    )


def build_ghost_decision_contract(
    *,
    candidate_graph: Optional[Mapping[str, Any]] = None,
    promotion_review: Optional[Mapping[str, Any]] = None,
    workload_graph: Optional[Mapping[str, Any]] = None,
    workload_proposal_review: Optional[Mapping[str, Any]] = None,
    output_obligations: Optional[Sequence[Mapping[str, Any]]] = None,
    accepted_learning_hints: Optional[Mapping[str, Any]] = None,
    semantic_role_profile: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a compact role/decision contract derived from runtime truth."""

    candidate_graph = candidate_graph if isinstance(candidate_graph, Mapping) else {}
    promotion_review = promotion_review if isinstance(promotion_review, Mapping) else {}
    workload_graph = workload_graph if isinstance(workload_graph, Mapping) else {}
    workload_proposal_review = workload_proposal_review if isinstance(workload_proposal_review, Mapping) else {}
    obligations = [
        item for item in (output_obligations or []) if isinstance(item, Mapping)
    ]

    reconsideration_items = _reconsideration_items(promotion_review)
    supersession_items = _supersession_items(obligations, promotion_review)
    supersession_candidate_items = _supersession_candidate_items(workload_graph)
    promotion_suggestion_items = _promotion_suggestion_items(workload_graph)
    waiver_candidate_items = _waiver_candidate_items(workload_graph)
    semantic_review_items = _semantic_review_items(workload_graph)
    repair_items = _repair_items(workload_graph)
    graph_repair_proposals = build_graph_repair_proposals(
        request_phase_graph=workload_graph,
        decision_contract={'repair_candidates': repair_items},
        accepted_learning_hints=accepted_learning_hints,
    )
    counts = _obligation_status_counts(obligations)
    coverage = (
        workload_proposal_review.get('coverage')
        if isinstance(workload_proposal_review.get('coverage'), Mapping)
        else {}
    )
    priorities: list[str] = []
    if coverage and _status(coverage.get('status')) in {'missing', 'partial'}:
        priorities.append('complete_branch_local_workload_task_proposals')
    if _obligation_ids_for_statuses(obligations, _OPEN_OBLIGATION_STATUSES):
        priorities.append('continue_or_repair_open_promoted_obligations')
    if _obligation_ids_for_statuses(obligations, _BLOCKED_OBLIGATION_STATUSES):
        priorities.append('repair_blocked_obligations_before_retry')
    if reconsideration_items:
        priorities.append('keep_reconsiderable_candidates_available_without_executing_them')
    if supersession_items:
        priorities.append('preserve_superseded_obligations_as_closed_truth')
    if promotion_suggestion_items:
        priorities.append('review_advisory_promotion_suggestions_before_promoting_work')
    if waiver_candidate_items:
        priorities.append('review_advisory_waiver_candidates_without_hiding_missing_work')
    if supersession_candidate_items:
        priorities.append('review_advisory_supersession_candidates_against_runtime_truth')
    if semantic_review_items:
        priorities.append('surface_semantic_review_criteria_without_claiming_runtime_truth')
    if graph_repair_proposals:
        priorities.append('review_graph_repair_proposals_before_runtime_patch')
    accepted_learning = _learning_hint_summary(accepted_learning_hints)
    block_resolution_reflex = _block_resolution_reflex(
        obligations=obligations,
        promotion_review=promotion_review,
        workload_graph=workload_graph,
        reconsideration_items=reconsideration_items,
        supersession_items=supersession_items,
        waiver_candidate_items=waiver_candidate_items,
        supersession_candidate_items=supersession_candidate_items,
        repair_items=repair_items,
        semantic_review_items=semantic_review_items,
    )
    reflex_signals = (
        block_resolution_reflex.get('signals')
        if isinstance(block_resolution_reflex.get('signals'), list)
        else []
    )
    active_reconsideration = _active_reconsideration_review(block_resolution_reflex)
    active_reconsideration_decisions = (
        active_reconsideration.get('decisions')
        if isinstance(active_reconsideration.get('decisions'), list)
        else []
    )
    semantic_quality = _semantic_quality_review(semantic_review_items)
    semantic_quality_contracts = (
        semantic_quality.get('contracts')
        if isinstance(semantic_quality.get('contracts'), list)
        else []
    )
    recursive_cycle = _recursive_cycle_review(workload_graph)
    recursive_cycle_tasks = (
        recursive_cycle.get('tasks')
        if isinstance(recursive_cycle.get('tasks'), list)
        else []
    )
    aspiration = _aspiration_review(
        candidate_graph=candidate_graph,
        promotion_review=promotion_review,
        workload_graph=workload_graph,
        workload_proposal_review=workload_proposal_review,
        obligations=obligations,
        reconsideration_items=reconsideration_items,
        promotion_suggestion_items=promotion_suggestion_items,
        semantic_review_items=semantic_review_items,
    )
    aspiration_frames = (
        aspiration.get('frames')
        if isinstance(aspiration.get('frames'), list)
        else []
    )
    commitment = _commitment_review(
        active_reconsideration_review=active_reconsideration,
        semantic_quality_review=semantic_quality,
        recursive_cycle_review=recursive_cycle,
    )
    commitment_frames = (
        commitment.get('frames')
        if isinstance(commitment.get('frames'), list)
        else []
    )
    semantic_role_orientation = _semantic_role_orientation_review(
        semantic_role_profile if isinstance(semantic_role_profile, Mapping) else {}
    )
    semantic_role_orientation_frames = (
        semantic_role_orientation.get('frames')
        if isinstance(semantic_role_orientation.get('frames'), list)
        else []
    )
    semantic_decision = _semantic_decision_review(
        active_reconsideration_review=active_reconsideration,
        semantic_quality_review=semantic_quality,
        recursive_cycle_review=recursive_cycle,
        aspiration_review=aspiration,
        commitment_review=commitment,
        semantic_role_orientation_review=semantic_role_orientation,
        accepted_learning=accepted_learning,
    )
    semantic_decision_proposals = (
        semantic_decision.get('proposals')
        if isinstance(semantic_decision.get('proposals'), list)
        else []
    )
    controlled_attention = _controlled_attention_review(
        active_reconsideration_review=active_reconsideration,
        semantic_quality_review=semantic_quality,
        recursive_cycle_review=recursive_cycle,
        aspiration_review=aspiration,
        commitment_review=commitment,
        semantic_role_orientation_review=semantic_role_orientation,
        semantic_decision_review=semantic_decision,
        accepted_learning=accepted_learning,
    )
    controlled_attention_frames = (
        controlled_attention.get('frames')
        if isinstance(controlled_attention.get('frames'), list)
        else []
    )
    semantic_review_lens_review = _semantic_review_lens_review(
        active_reconsideration_decisions=active_reconsideration_decisions,
        semantic_quality_contracts=semantic_quality_contracts,
        recursive_cycle_tasks=recursive_cycle_tasks,
        aspiration_frames=aspiration_frames,
        commitment_frames=commitment_frames,
        semantic_role_orientation_frames=semantic_role_orientation_frames,
        semantic_decision_proposals=semantic_decision_proposals,
        controlled_attention_frames=controlled_attention_frames,
    )
    semantic_review_lenses = (
        semantic_review_lens_review.get('lenses')
        if isinstance(semantic_review_lens_review.get('lenses'), list)
        else []
    )
    if reflex_signals:
        priorities.append('apply_block_resolution_reconsideration_reflex_between_steps')
    if active_reconsideration_decisions:
        priorities.append('review_active_reconsideration_decisions_before_state_change')
    if semantic_quality_contracts:
        priorities.append('run_semantic_quality_review_before_claiming_quality_truth')
    if recursive_cycle_tasks:
        priorities.append('apply_recursive_mini_cycle_per_subtask')
    if aspiration_frames:
        priorities.append('use_aspiration_review_to_keep_possibility_and_solution_bar_visible')
    if commitment_frames:
        priorities.append('use_commitment_review_to_choose_the_right_sized_sufficient_transition')
    if semantic_role_orientation_frames:
        priorities.append('use_semantic_roles_as_advisory_orientation_only')
    if semantic_decision_proposals:
        priorities.append('review_semantic_decision_proposals_before_state_transition')
    if controlled_attention_frames:
        priorities.append('use_controlled_model_attention_between_execution_steps')
    if semantic_review_lenses:
        priorities.append('apply_semantic_review_lenses_to_branch_expectations')

    return _json_safe(
        {
            'kind': 'ollmo.ghost_decision_contract',
            'decision_contract_version': GHOST_DECISION_CONTRACT_VERSION,
            'authority_model': {
                'ghost': [
                    'interpret_current_intent',
                    'propose_candidates',
                    'annotate_existing_workload_tasks',
                    'suggest_reconsideration_repair_or_supersession',
                    'propose_graph_repair_candidates',
                    'propose_semantic_quality_and_recursive_cycle_reviews',
                    'preserve_aspiration_and_commitment_orientation',
                    'compile_legacy_semantic_roles_modes_into_advisory_orientation',
                    'propose_semantic_decision_review_inputs',
                    'focus_controlled_attention_between_execution_steps',
                ],
                'contract_layer': [
                    'normalize_candidates',
                    'validate_promotions',
                    'validate_graph_repair_proposals',
                    'reject_structural_mismatches',
                ],
                'runtime': [
                    'execute_only_promoted_ready_obligations',
                    'apply_only_validated_graph_repair_patches',
                    'prove_artifacts_and_outputs',
                    'freeze_runtime_truth',
                ],
                'closure_review': [
                    'classify_fulfilled_pending_blocked_waived_superseded',
                    'request_repair_or_semantic_review_when_needed',
                ],
            },
            'candidate_policy': {
                'candidate_execution_policy': 'non_executable_until_promoted',
                'reconsideration_policy': 'reserved_omitted_or_stale_candidates_remain_reconsiderable',
                'promotion_policy': 'current_evidence_required',
            },
            'obligation_policy': {
                'open_statuses': sorted(_OPEN_OBLIGATION_STATUSES),
                'blocked_statuses': sorted(_BLOCKED_OBLIGATION_STATUSES),
                'closed_statuses': sorted(_CLOSED_OBLIGATION_STATUSES),
                'superseded_policy': 'closed_replacement_truth_not_retry_work',
            },
            'candidate_count': int(candidate_graph.get('candidate_count') or 0),
            'promotion_counts': promotion_review.get('counts') if isinstance(promotion_review.get('counts'), Mapping) else {},
            'obligation_counts': counts,
            'open_obligation_ids': _obligation_ids_for_statuses(obligations, _OPEN_OBLIGATION_STATUSES),
            'blocked_obligation_ids': _obligation_ids_for_statuses(obligations, _BLOCKED_OBLIGATION_STATUSES),
            'closed_obligation_ids': _obligation_ids_for_statuses(obligations, _CLOSED_OBLIGATION_STATUSES),
            'reconsideration_candidates': reconsideration_items,
            'supersession_records': supersession_items,
            'promotion_suggestions': promotion_suggestion_items,
            'waiver_candidates': waiver_candidate_items,
            'supersession_candidates': supersession_candidate_items,
            'semantic_review_candidates': semantic_review_items,
            'repair_candidates': repair_items,
            'graph_repair_proposals': graph_repair_proposals,
            'block_resolution_reflex': block_resolution_reflex,
            'reconsideration_reflex_signals': reflex_signals,
            'active_reconsideration_review': active_reconsideration,
            'active_reconsideration_decisions': active_reconsideration_decisions,
            'semantic_quality_review': semantic_quality,
            'semantic_quality_contracts': semantic_quality_contracts,
            'recursive_cycle_review': recursive_cycle,
            'recursive_cycle_tasks': recursive_cycle_tasks,
            'aspiration_review': aspiration,
            'aspiration_frames': aspiration_frames,
            'commitment_review': commitment,
            'commitment_frames': commitment_frames,
            'semantic_role_orientation_review': semantic_role_orientation,
            'semantic_role_orientation_frames': semantic_role_orientation_frames,
            'semantic_decision_review': semantic_decision,
            'semantic_decision_proposals': semantic_decision_proposals,
            'controlled_attention_review': controlled_attention,
            'controlled_attention_frames': controlled_attention_frames,
            'semantic_review_lens_review': semantic_review_lens_review,
            'semantic_review_lenses': semantic_review_lenses,
            'workload_proposal_coverage': coverage,
            'accepted_learning': accepted_learning,
            'next_decision_priorities': priorities,
            'semantic_planning_contract': _semantic_planning_contract(
                priorities=priorities,
                coverage=coverage,
                obligations=obligations,
                reconsideration_items=reconsideration_items,
                promotion_suggestion_items=promotion_suggestion_items,
                waiver_candidate_items=waiver_candidate_items,
                supersession_items=supersession_items,
                supersession_candidate_items=supersession_candidate_items,
                semantic_review_items=semantic_review_items,
                repair_items=repair_items,
                accepted_learning=accepted_learning,
                block_resolution_reflex=block_resolution_reflex,
                active_reconsideration_review=active_reconsideration,
                semantic_quality_review=semantic_quality,
                recursive_cycle_review=recursive_cycle,
                aspiration_review=aspiration,
                commitment_review=commitment,
                semantic_role_orientation_review=semantic_role_orientation,
                semantic_decision_review=semantic_decision,
                controlled_attention_review=controlled_attention,
                semantic_review_lens_review=semantic_review_lens_review,
            ),
        }
    )
