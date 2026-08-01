import copy

import pytest

from ollmo_services.graph_rebase_rollout import (
    build_partial_graph_rebase_promotion_gate,
    build_graph_rebase_readiness_report,
)


@pytest.fixture
def current_shadow_baseline_payloads():
    reasons = [
        ('current_structural_closure_evidence_missing', '', 'site-with-images'),
        ('current_structural_closure_evidence_missing', '', 'site-with-images'),
        ('current_structural_closure_evidence_missing', '', 'multimodal-review'),
        ('current_structural_closure_evidence_missing', '', 'multimodal-review'),
        ('current_structural_closure_evidence_missing', '', 'artifact-rework'),
        (
            'smaller_redraw_scope_precedes_rebase',
            'repair_artifact_ref_identity',
            'artifact-rework',
        ),
    ]
    payloads = []
    for index, (reason, smaller_scope, prompt_family) in enumerate(reasons, start=1):
        candidate_review = {
            'kind': 'ollmo.runtime_graph_rebase_candidate_review',
            'status': 'not_proposed',
            'reason': reason,
            'base_graph_digest': f'base-{index}',
            'candidate_graph_digest': f'candidate-{index}',
            'runtime_effect': 'none',
        }
        if smaller_scope:
            candidate_review['smaller_scope'] = smaller_scope
        payloads.append(
            {
                'id': f'resp-baseline-{index}',
                'lifecycle_state': 'completed',
                'late_fill': {
                    'status': 'completed',
                    'active_count': 0,
                    'pending_count': 0,
                },
                'request_meta': {'prompt_family': prompt_family},
                'runtime': {
                    'request_phase_graph': {
                        'kind': 'ollmo.request_phase_graph',
                        'response_id': f'resp-baseline-{index}',
                        'frame_id': f'frame-baseline-{index}',
                        'graph_rebase_proposals': [],
                        'graph_rebase_reviews': [],
                    },
                    'developer_diagnostics': {
                        'runtime_graph_rebase_candidate_review': candidate_review,
                    },
                },
            }
        )
    return payloads


def _accepted_partial_payload(response_id='resp-partial', proposal_id='proposal-partial'):
    proposal = {
        'kind': 'ollmo.graph_rebase_proposal',
        'proposal_id': proposal_id,
        'requested_rebase_class': 'partial_subtree_rebase',
        'base_graph_digest': f'base-{proposal_id}',
        'candidate_graph_digest': f'candidate-{proposal_id}',
        'scope_root_ids': ['phase-review'],
    }
    proof = {
        'kind': 'ollmo.graph_rebase_preservation_proof',
        'status': 'passed',
        'base_graph_digest': proposal['base_graph_digest'],
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'blocked_reasons': [],
    }
    review = {
        'kind': 'ollmo.graph_rebase_review',
        'review_id': f'review-{proposal_id}',
        'proposal_id': proposal_id,
        'status': 'accepted',
        'source_proposal': proposal,
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'preservation_proof': proof,
        'blocked_reasons': [],
    }
    lifecycle = {
        'kind': 'ollmo.graph_rebase_lifecycle',
        'rebase_id': f'rebase-{proposal_id}',
        'proposal_id': proposal_id,
        'review_id': review['review_id'],
        'status': 'validated',
        'autonomy_level': 'shadow',
        'validation_review': review,
        'outcome': {
            'status': 'validated',
            'runtime_effect': 'shadow_no_mutation',
        },
    }
    return {
        'id': response_id,
        'lifecycle_state': 'completed',
        'request_meta': {'prompt_family': 'partial-review'},
        'late_fill': {'status': 'completed', 'active_count': 0, 'pending_count': 0},
        'runtime': {
            'request_phase_graph': {
                'kind': 'ollmo.request_phase_graph',
                'response_id': response_id,
                'frame_id': f'frame-{response_id}',
                'redraw_scope_ladder_review': {
                    'selected_scope': 'partial_subtree_rebase',
                },
                'graph_rebase_proposals': [proposal],
                'graph_rebase_reviews': [review],
                'graph_rebase_lifecycle': [lifecycle],
            },
            'developer_diagnostics': {
                'runtime_graph_rebase_candidate_review': {
                    'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                    'status': 'validated_by_runtime_review',
                    'proposal_id': proposal_id,
                    'requested_rebase_class': 'partial_subtree_rebase',
                },
                # These are duplicate projections of canonical graph records.
                'runtime_graph_rebase_proposals': [copy.deepcopy(proposal)],
                'runtime_graph_rebase_reviews': [copy.deepcopy(review)],
                'graph_rebase_lifecycle': [copy.deepcopy(lifecycle)],
            },
        },
    }


def _zero_volume_policy():
    return {
        'shadow_to_stage': {
            'minimum_settled_candidate_opportunities': 0,
            'minimum_unique_workload_families': 0,
            'minimum_settled_not_proposed': 0,
            'minimum_qualifying_partial_proposals': 0,
            'minimum_partial_useful_adjudications': 0,
            'minimum_partial_replay_confirmations': 0,
        },
        'partial_stage_to_apply_reviewed': {
            'minimum_partial_stages': 0,
            'minimum_partial_useful_adjudications': 0,
            'minimum_partial_replay_confirmations': 0,
            'minimum_partial_local_execution_contract_proofs': 0,
        },
    }


def _one_partial_stage_policy():
    policy = _zero_volume_policy()
    policy['partial_stage_to_apply_reviewed']['minimum_partial_stages'] = 1
    return policy


def _add_durable_partial_stage(payload):
    graph = payload['runtime']['request_phase_graph']
    proposal = graph['graph_rebase_proposals'][0]
    review = graph['graph_rebase_reviews'][0]
    lifecycle = graph['graph_rebase_lifecycle'][0]
    response_id = payload['id']
    target_frame_id = f'frame-target-{response_id}'
    proposal_digest = f'proposal-digest-{proposal["proposal_id"]}'
    proposal.update(
        {
            'target_response_id': response_id,
            'target_frame_id': target_frame_id,
        }
    )
    review.update(
        {
            'target_response_id': response_id,
            'target_frame_id': target_frame_id,
            'base_graph_digest': proposal['base_graph_digest'],
            'proposal_digest': proposal_digest,
        }
    )
    lifecycle.update(
        {
            'review_id': review['review_id'],
            'requested_rebase_class': 'partial_subtree_rebase',
            'base_graph_digest': proposal['base_graph_digest'],
            'before_graph_digest': proposal['base_graph_digest'],
            'candidate_graph_digest': proposal['candidate_graph_digest'],
            'idempotency_key': f'idempotency-{proposal["proposal_id"]}',
            'validation_review': review,
        }
    )
    staged = {
        **copy.deepcopy(lifecycle),
        'status': 'staged',
        'autonomy_level': 'stage',
        'outcome': {
            'status': 'staged',
            'runtime_effect': 'staged_no_executable_mutation',
        },
    }
    graph['staged_graph_rebases'] = [staged]
    return staged


def _trusted_partial_stage_record(payload, staged):
    graph = payload['runtime']['request_phase_graph']
    proposal = graph['graph_rebase_proposals'][0]
    review = graph['graph_rebase_reviews'][0]
    return {
        'kind': 'ollmo.graph_rebase_operator_record',
        'record_id': f'trusted-stage-{proposal["proposal_id"]}',
        'action': 'stage',
        'status': 'staged',
        'runtime_effect': 'staged_no_executable_mutation',
        'response_id': payload['id'],
        'frame_id': f'frame-parent-{payload["id"]}',
        'target_frame_id': proposal['target_frame_id'],
        'proposal_id': proposal['proposal_id'],
        'proposal_digest': review['proposal_digest'],
        'runtime_review_id': staged['review_id'],
        'base_graph_digest': staged['base_graph_digest'],
        'candidate_graph_digest': staged['candidate_graph_digest'],
        'requested_rebase_class': 'partial_subtree_rebase',
    }


def test_current_baseline_is_six_settled_not_proposed_and_zero_qualifying(
    current_shadow_baseline_payloads,
):
    report = build_graph_rebase_readiness_report(
        current_shadow_baseline_payloads,
        source_ledger_identity={
            'kind': 'ollmo.response_frame_ledger',
            'record_count': 685,
            'response_count': 263,
        },
        corpus_window={'label': 'audited-2026-07-19-baseline'},
    )

    assert report['corpus']['settled_final_response_count'] == 6
    assert report['corpus']['unique_workload_family_count'] == 3
    assert report['candidate_opportunities']['settled_final']['total'] == 6
    assert report['candidate_opportunities']['settled_final']['not_proposed_count'] == 6
    assert report['candidate_opportunities']['settled_final']['by_reason'] == {
        'current_structural_closure_evidence_missing': 5,
        'smaller_redraw_scope_precedes_rebase': 1,
    }
    assert report['candidate_opportunities']['settled_final']['by_smaller_scope'] == {
        'repair_artifact_ref_identity': 1,
    }
    assert report['formal_evidence']['proposals']['total'] == 0
    assert report['formal_evidence']['reviews']['total'] == 0
    assert report['formal_evidence']['preservation_proofs']['total'] == 0
    assert report['qualifying_evidence']['partial_proposal_count'] == 0
    assert report['gates']['shadow_to_stage']['ready'] is False
    assert 'minimum_qualifying_partial_proposals' in report['gates']['shadow_to_stage'][
        'unmet_requirements'
    ]
    assert report['gates']['full_shadow_only']['decision'] == 'remain_shadow'
    assert report['gates']['full_shadow_only']['ready_for_execution'] is False


def test_duplicate_payload_and_projection_replays_do_not_inflate_evidence():
    payload = _accepted_partial_payload()
    replay = copy.deepcopy(payload)
    replay['runtime']['request_phase_graph']['frame_id'] = 'frame-replayed-projection'

    report = build_graph_rebase_readiness_report([payload, replay])

    assert report['corpus']['input_observation_count'] == 2
    assert report['corpus']['settled_final_response_count'] == 1
    assert report['deduplication']['duplicate_response_observations_excluded'] == 1
    assert report['formal_evidence']['proposals']['raw_count'] == 2
    assert report['formal_evidence']['proposals']['total'] == 1
    assert report['formal_evidence']['reviews']['raw_count'] == 2
    assert report['formal_evidence']['reviews']['total'] == 1
    assert report['formal_evidence']['lifecycles']['total'] == 1
    assert report['qualifying_evidence']['partial_proposal_count'] == 1
    assert report['deduplication']['duplicate_formal_records_excluded']['proposals'] == 1
    assert report['deduplication']['duplicate_formal_records_excluded']['reviews'] == 1


def test_active_late_fill_candidate_is_visible_but_excluded_from_settled_denominator():
    payload = _accepted_partial_payload(response_id='resp-active')
    payload['lifecycle_state'] = 'late_fill_running'
    payload['late_fill'] = {'status': 'running', 'active_count': 1, 'pending_count': 0}
    graph = payload['runtime']['request_phase_graph']
    graph['graph_rebase_proposals'] = []
    graph['graph_rebase_reviews'] = []
    graph['graph_rebase_lifecycle'] = []
    diagnostics = payload['runtime']['developer_diagnostics']
    diagnostics['runtime_graph_rebase_proposals'] = []
    diagnostics['runtime_graph_rebase_reviews'] = []
    diagnostics['graph_rebase_lifecycle'] = []
    diagnostics['runtime_graph_rebase_candidate_review'] = {
        'kind': 'ollmo.runtime_graph_rebase_candidate_review',
        'status': 'not_proposed',
        'reason': 'active_late_fill_must_settle',
        'late_fill_status': 'running',
    }

    report = build_graph_rebase_readiness_report([payload])

    assert report['corpus']['settled_final_response_count'] == 0
    assert report['candidate_opportunities']['settled_final']['total'] == 0
    assert report['candidate_opportunities']['nonterminal_active_late_fill']['total'] == 1
    assert report['candidate_opportunities']['nonterminal_active_late_fill']['by_reason'] == {
        'active_late_fill_must_settle': 1,
    }


def test_unresolved_critical_finding_is_zero_tolerance_even_when_volume_gates_are_zero():
    payload = _accepted_partial_payload()
    trusted_records = [
        {
            'kind': 'ollmo.graph_rebase_operator_review',
            'record_id': 'operator-review-scope-escape',
            'action': 'adjudicate',
            'status': 'accepted',
            'proposal_id': 'proposal-partial',
            'requested_rebase_class': 'partial_subtree_rebase',
            'adjudication': 'false_positive',
            'safety_findings': ['partial_rebase_changes_outside_scope'],
            'reason': 'Observed scope escape in replay.',
        }
    ]

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=trusted_records,
        policy=_zero_volume_policy(),
    )

    assert report['safety']['unresolved_critical_finding_count'] >= 1
    assert report['safety']['unresolved_by_category']['scope_escape'] == 1
    assert report['safety']['zero_tolerance_satisfied'] is False
    assert report['operator_adjudications']['by_classification']['false_positive'] == 1
    assert report['gates']['shadow_to_stage']['ready'] is False
    assert 'maximum_unresolved_critical_safety_findings' in report['gates'][
        'shadow_to_stage'
    ]['unmet_requirements']
    assert report['gates']['partial_stage_to_apply_reviewed']['ready'] is False


def test_exact_duplicate_trusted_review_records_are_deduplicated():
    payload = _accepted_partial_payload()
    record = {
        'kind': 'ollmo.graph_rebase_operator_review',
        'record_id': 'operator-review-useful',
        'action': 'adjudicate',
        'status': 'accepted',
        'proposal_id': 'proposal-partial',
        'requested_rebase_class': 'partial_subtree_rebase',
        'adjudication': 'useful_proposal',
        'replay_verified': True,
    }

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[record, copy.deepcopy(record)],
    )

    assert report['operator_adjudications']['total'] == 1
    assert report['operator_adjudications']['by_classification']['useful_proposal'] == 1
    assert report['qualifying_evidence']['partial_replay_confirmation_count'] == 1
    assert report['deduplication']['duplicate_trusted_review_records_excluded'] == 1


def test_false_negative_blocks_until_later_bound_replay_verified_proposal_resolves_it():
    payload = _accepted_partial_payload(response_id='resp-remediated')
    false_negative = {
        'kind': 'ollmo.graph_rebase_operator_record',
        'record_id': 'operator-false-negative',
        'action': 'adjudicate',
        'status': 'recorded',
        'response_id': 'resp-remediated',
        'candidate_observation_id': 'candidate-observation-missed',
        'requested_rebase_class': 'partial_subtree_rebase',
        'adjudication': 'false_negative',
    }

    unresolved = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[false_negative],
        policy=_zero_volume_policy(),
    )

    assert unresolved['operator_adjudications']['by_classification']['false_negative'] == 1
    assert unresolved['operator_adjudications']['unresolved_false_negative_count'] == 1
    false_negative_requirement = next(
        item
        for item in unresolved['gates']['shadow_to_stage']['requirements']
        if item['requirement'] == 'maximum_false_negative_adjudications'
    )
    assert false_negative_requirement['actual'] == 1
    assert false_negative_requirement['met'] is False

    resolution = {
        'kind': 'ollmo.graph_rebase_operator_record',
        'record_id': 'operator-false-negative-resolution',
        'action': 'adjudicate',
        'status': 'recorded',
        'response_id': 'resp-remediated',
        'proposal_id': 'proposal-partial',
        'requested_rebase_class': 'partial_subtree_rebase',
        'adjudication': 'useful_proposal',
        'replay_verified': True,
        'resolves_record_id': false_negative['record_id'],
        'resolved_candidate_observation_id': false_negative[
            'candidate_observation_id'
        ],
        'resolved_response_id': false_negative['response_id'],
    }
    resolved = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[false_negative, resolution],
        policy=_zero_volume_policy(),
    )

    adjudications = resolved['operator_adjudications']
    assert adjudications['by_classification']['false_negative'] == 1
    assert adjudications['resolved_false_negative_count'] == 1
    assert adjudications['unresolved_false_negative_count'] == 0
    assert adjudications['resolved_false_negative_record_ids'] == [
        false_negative['record_id']
    ]
    false_negative_requirement = next(
        item
        for item in resolved['gates']['shadow_to_stage']['requirements']
        if item['requirement'] == 'maximum_false_negative_adjudications'
    )
    assert false_negative_requirement['actual'] == 0
    assert false_negative_requirement['met'] is True
    assert resolved['safety']['unresolved_by_category']['operator_false_negative'] == 0
    assert resolved['safety']['contained_by_category']['operator_false_negative'] == 1


def test_false_negative_resolution_without_replay_or_exact_binding_does_not_clear_gate():
    payload = _accepted_partial_payload(response_id='resp-remediated-invalid')
    false_negative = {
        'record_id': 'operator-false-negative-invalid',
        'action': 'adjudicate',
        'response_id': 'resp-remediated-invalid',
        'candidate_observation_id': 'candidate-observation-missed',
        'requested_rebase_class': 'partial_subtree_rebase',
        'adjudication': 'false_negative',
    }
    invalid_resolution = {
        'record_id': 'operator-false-negative-invalid-resolution',
        'action': 'adjudicate',
        'response_id': 'resp-remediated-invalid',
        'proposal_id': 'proposal-partial',
        'requested_rebase_class': 'partial_subtree_rebase',
        'adjudication': 'useful_proposal',
        'replay_verified': False,
        'resolves_record_id': false_negative['record_id'],
        'resolved_candidate_observation_id': 'different-candidate-observation',
        'resolved_response_id': false_negative['response_id'],
    }

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[false_negative, invalid_resolution],
        policy=_zero_volume_policy(),
    )

    assert report['operator_adjudications']['resolved_false_negative_count'] == 0
    assert report['operator_adjudications']['unresolved_false_negative_count'] == 1


def test_hydrated_response_frame_mapping_is_accepted_without_filesystem_resolution():
    payload = _accepted_partial_payload(response_id='resp-frame-mapping')
    frame = {
        'kind': 'ollmo.response_frame',
        'response_id': 'resp-frame-mapping',
        'status': 'completed',
        'current_state': {'lifecycle_state': 'completed'},
        'late_fill': {'status': 'completed', 'active_count': 0, 'pending_count': 0},
        'request': {'request_meta': {'prompt_family': 'partial-review'}},
        'runtime': copy.deepcopy(payload['runtime']),
    }

    report = build_graph_rebase_readiness_report([frame])

    assert report['corpus']['settled_final_response_count'] == 1
    assert report['corpus']['unique_workload_family_count'] == 1
    assert report['formal_evidence']['proposals']['total'] == 1
    assert report['qualifying_evidence']['partial_proposal_count'] == 1


def test_stage_successor_outcome_authorization_and_local_proof_are_separate_evidence():
    payload = _accepted_partial_payload(response_id='resp-rollout-records')
    graph = payload['runtime']['request_phase_graph']
    review = graph['graph_rebase_reviews'][0]
    review['local_execution_contract_proof'] = {
        'kind': 'ollmo.graph_rebase_local_execution_contract_proof',
        'status': 'passed',
    }
    lifecycle = graph['graph_rebase_lifecycle'][0]
    lifecycle['validation_review'] = review
    staged = _add_durable_partial_stage(payload)
    graph['successor_rebase_requests'] = [
        {
            'kind': 'ollmo.graph_rebase_successor_request',
            'status': 'completed',
            'proposal_id': 'proposal-partial',
            'rebase_id': 'rebase-proposal-partial',
            'idempotency_key': 'partial-idempotency-key',
            'requested_rebase_class': 'partial_subtree_rebase',
        }
    ]
    graph['applied_graph_rebases'] = [
        {
            'kind': 'ollmo.graph_rebase_lifecycle',
            'status': 'applied',
            'proposal_id': 'proposal-partial',
            'rebase_id': 'rebase-proposal-partial',
            'idempotency_key': 'partial-idempotency-key',
            'requested_rebase_class': 'partial_subtree_rebase',
        }
    ]
    trusted_records = [
        _trusted_partial_stage_record(payload, staged),
        {
            'kind': 'ollmo.graph_rebase_operator_review',
            'record_id': 'adjudicate-partial',
            'action': 'adjudicate',
            'status': 'accepted',
            'proposal_id': 'proposal-partial',
            'requested_rebase_class': 'partial_subtree_rebase',
            'adjudication': 'useful_proposal',
            'replay_verified': True,
        },
        {
            'kind': 'ollmo.graph_rebase_operator_review',
            'record_id': 'authorize-partial',
            'action': 'authorize_partial',
            'status': 'authorized',
            'proposal_id': 'proposal-partial',
            'requested_rebase_class': 'partial_subtree_rebase',
        },
    ]

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=trusted_records,
    )

    assert report['formal_evidence']['stages']['total'] == 1
    assert report['formal_evidence']['successor_requests']['total'] == 1
    assert report['formal_evidence']['applied_rebases']['total'] == 1
    assert report['formal_evidence']['terminal_outcomes']['total'] == 1
    assert report['formal_evidence']['authorizations']['trusted_operator_records']['total'] == 1
    assert report['qualifying_evidence']['partial_stage_count'] == 1
    assert report['qualifying_evidence']['partial_local_execution_contract_proof_count'] == 1
    assert report['qualifying_evidence']['accepted_partial_authorization_count'] == 1


def test_trusted_only_partial_stage_is_diagnostic_orphan_and_does_not_qualify():
    payload = _accepted_partial_payload(response_id='resp-trusted-stage-only')
    staged = _add_durable_partial_stage(payload)
    payload['runtime']['request_phase_graph']['staged_graph_rebases'] = []
    trusted_stage = _trusted_partial_stage_record(payload, staged)

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[trusted_stage],
        policy=_one_partial_stage_policy(),
    )

    pairing = report['qualifying_evidence']['partial_stage_pairing']
    assert report['qualifying_evidence']['partial_stage_count'] == 0
    assert pairing['exact_pair_count'] == 0
    assert pairing['trusted_registry_stage_count'] == 1
    assert pairing['durable_runtime_stage_count'] == 0
    assert pairing['trusted_registry_orphan_count'] == 1
    assert pairing['durable_runtime_orphan_count'] == 0
    assert 'minimum_partial_stages' in report['gates'][
        'partial_stage_to_apply_reviewed'
    ]['unmet_requirements']


def test_runtime_only_partial_stage_is_diagnostic_orphan_and_does_not_qualify():
    payload = _accepted_partial_payload(response_id='resp-runtime-stage-only')
    _add_durable_partial_stage(payload)

    report = build_graph_rebase_readiness_report(
        [payload],
        policy=_one_partial_stage_policy(),
    )

    pairing = report['qualifying_evidence']['partial_stage_pairing']
    assert report['qualifying_evidence']['partial_stage_count'] == 0
    assert pairing['exact_pair_count'] == 0
    assert pairing['trusted_registry_stage_count'] == 0
    assert pairing['durable_runtime_stage_count'] == 1
    assert pairing['trusted_registry_orphan_count'] == 0
    assert pairing['durable_runtime_orphan_count'] == 1
    assert 'minimum_partial_stages' in report['gates'][
        'partial_stage_to_apply_reviewed'
    ]['unmet_requirements']


def test_exact_trusted_and_runtime_partial_stage_pair_qualifies_once():
    payload = _accepted_partial_payload(response_id='resp-paired-stage')
    staged = _add_durable_partial_stage(payload)
    trusted_stage = _trusted_partial_stage_record(payload, staged)

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[trusted_stage],
        policy=_one_partial_stage_policy(),
    )

    pairing = report['qualifying_evidence']['partial_stage_pairing']
    assert report['qualifying_evidence']['partial_stage_count'] == 1
    assert pairing['exact_pair_count'] == 1
    assert pairing['trusted_registry_stage_count'] == 1
    assert pairing['durable_runtime_stage_count'] == 1
    assert pairing['trusted_registry_orphan_count'] == 0
    assert pairing['durable_runtime_orphan_count'] == 0
    stage_requirement = next(
        item
        for item in report['gates']['partial_stage_to_apply_reviewed']['requirements']
        if item['requirement'] == 'minimum_partial_stages'
    )
    assert stage_requirement['actual'] == 1
    assert stage_requirement['met'] is True


def test_partial_stage_pair_requires_every_exact_binding_field():
    payload = _accepted_partial_payload(response_id='resp-mismatched-stage')
    staged = _add_durable_partial_stage(payload)
    trusted_stage = _trusted_partial_stage_record(payload, staged)
    trusted_stage['candidate_graph_digest'] = 'candidate-digest-from-another-stage'

    report = build_graph_rebase_readiness_report(
        [payload],
        trusted_review_records=[trusted_stage],
        policy=_one_partial_stage_policy(),
    )

    pairing = report['qualifying_evidence']['partial_stage_pairing']
    assert report['qualifying_evidence']['partial_stage_count'] == 0
    assert pairing['exact_pair_count'] == 0
    assert pairing['trusted_registry_orphan_count'] == 1
    assert pairing['durable_runtime_orphan_count'] == 1
    assert 'candidate_graph_digest' in pairing['pairing_fields']


def test_partial_promotion_gate_is_blocked_for_current_shadow_baseline(
    current_shadow_baseline_payloads,
):
    report = build_graph_rebase_readiness_report(current_shadow_baseline_payloads)

    gate = build_partial_graph_rebase_promotion_gate(report)

    assert gate['kind'] == 'ollmo.graph_rebase_promotion_gate'
    assert gate['status'] == 'blocked'
    assert gate['decision'] == 'keep_partial_non_executable'
    assert gate['readiness_report_digest'] == report['report_digest']
    assert gate['corpus_digest'] == report['corpus']['corpus_digest']


def test_partial_promotion_gate_binds_one_green_readiness_report():
    report = {
        'kind': 'ollmo.graph_rebase_rollout_readiness',
        'report_digest': 'graph-rebase-readiness-green',
        'corpus': {'corpus_digest': 'graph-rebase-corpus-green'},
        'policy': {'policy_id': 'safe-partial-v1'},
        'gates': {
            'partial_stage_to_apply_reviewed': {
                'ready': True,
                'decision': 'ready_for_exact_partial_apply_reviewed_authorization',
                'unmet_requirements': [],
            }
        },
    }

    gate = build_partial_graph_rebase_promotion_gate(report)

    assert gate['status'] == 'ready'
    assert gate['decision'] == 'promote'
    assert gate['evidence_refs'] == [
        'readiness:graph-rebase-readiness-green',
        'corpus:graph-rebase-corpus-green',
    ]
