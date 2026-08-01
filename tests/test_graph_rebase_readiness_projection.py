import copy
import json

from ollmo_services.graph_rebase_rollout import (
    GRAPH_REBASE_READINESS_OBSERVATION_KIND,
    build_graph_rebase_readiness_report,
    project_graph_rebase_readiness_observation,
)


def _settled_payload_with_bulky_candidate_graph():
    candidate_graph = {
        'kind': 'ollmo.request_phase_graph',
        'phases': [
            {
                'phase_id': 'phase-partial',
                'branch_id': 'branch-partial',
                'content_payload': 'candidate-graph-body-' + ('x' * 250_000),
            }
        ],
        'outputs': [{'content': 'candidate-output-' + ('y' * 250_000)}],
    }
    proposal = {
        'kind': 'ollmo.graph_rebase_proposal',
        'proposal_id': 'proposal-projected-partial',
        'target_response_id': 'resp-projected-partial',
        'target_frame_id': 'frame-projected-partial',
        'requested_rebase_class': 'partial_subtree_rebase',
        'base_graph_digest': 'base-projected-partial',
        'candidate_graph_digest': 'candidate-projected-partial',
        'scope_root_ids': ['phase-partial'],
        'candidate_graph': candidate_graph,
    }
    proof = {
        'kind': 'ollmo.graph_rebase_preservation_proof',
        'status': 'passed',
        'base_graph_digest': proposal['base_graph_digest'],
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'blocked_reasons': [],
    }
    local_proof = {
        'kind': 'ollmo.graph_rebase_local_execution_contract_proof',
        'status': 'passed',
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'content_payload': 'local-contract-body-' + ('z' * 200_000),
    }
    review = {
        'kind': 'ollmo.graph_rebase_review',
        'review_id': 'review-projected-partial',
        'proposal_id': proposal['proposal_id'],
        'status': 'accepted',
        'requested_rebase_class': 'partial_subtree_rebase',
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'source_proposal': proposal,
        'preservation_proof': proof,
        'local_execution_contract_proof': local_proof,
        'diff': {
            'semantic_changes': [{'id': 'phase-partial'}],
            'lost_dependency_edges': [],
            'hidden_failure_visibility_losses': [],
        },
        'blocked_reasons': [],
        'graph_rebase_authorization': {
            'kind': 'ollmo.graph_rebase_authorization',
            'authorization_id': 'inline-untrusted-authorization',
            'status': 'accepted',
            'requested_rebase_class': 'partial_subtree_rebase',
        },
    }
    lifecycle = {
        'kind': 'ollmo.graph_rebase_lifecycle',
        'rebase_id': 'rebase-projected-partial',
        'proposal_id': proposal['proposal_id'],
        'status': 'validated',
        'autonomy_level': 'shadow',
        'requested_rebase_class': 'partial_subtree_rebase',
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'validation_review': review,
        'outcome': {
            'status': 'validated',
            'runtime_effect': 'shadow_no_mutation',
        },
    }
    staged = {
        **copy.deepcopy(lifecycle),
        'status': 'staged',
        'autonomy_level': 'stage',
        'outcome': {
            'status': 'staged',
            'runtime_effect': 'staged_no_executable_mutation',
        },
    }
    successor = {
        'kind': 'ollmo.graph_rebase_successor_request',
        'status': 'completed',
        'proposal_id': proposal['proposal_id'],
        'rebase_id': lifecycle['rebase_id'],
        'idempotency_key': 'successor-projected-partial',
        'requested_rebase_class': 'partial_subtree_rebase',
        'candidate_graph_digest': proposal['candidate_graph_digest'],
        'blocked_reasons': ['root_prompt_fallback'],
    }
    outcome = {
        'kind': 'ollmo.graph_rebase_terminal_outcome',
        'outcome_id': 'outcome-projected-partial',
        'status': 'blocked',
        'proposal_id': proposal['proposal_id'],
        'rebase_id': lifecycle['rebase_id'],
        'requested_rebase_class': 'partial_subtree_rebase',
        'reason': 'parent_graph_mutated',
        'parent_mutated': True,
    }
    graph = {
        'kind': 'ollmo.request_phase_graph',
        'response_id': 'resp-projected-partial',
        'frame_id': 'frame-projected-partial',
        'phases': [{'content_payload': 'parent-graph-body-' + ('p' * 250_000)}],
        'redraw_scope_ladder_review': {
            'review_id': 'scope-review-projected-partial',
            'selected_scope': 'partial_subtree_rebase',
        },
        'graph_rebase_proposals': [proposal],
        'graph_rebase_reviews': [review],
        'graph_rebase_lifecycle': [lifecycle],
        'staged_graph_rebases': [staged],
        'successor_rebase_requests': [successor],
        'applied_graph_rebases': [],
        'graph_rebase_outcomes': [outcome],
    }
    diagnostics = {
        'runtime_graph_rebase_candidate_review': {
            'kind': 'ollmo.runtime_graph_rebase_candidate_review',
            'status': 'validated_by_runtime_review',
            'proposal_id': proposal['proposal_id'],
            'requested_rebase_class': 'partial_subtree_rebase',
            'base_graph_digest': proposal['base_graph_digest'],
            'candidate_graph_digest': proposal['candidate_graph_digest'],
        },
        'response_time_graph_rebase_candidate': {
            'kind': 'ollmo.runtime_graph_rebase_candidate',
            'status': 'rederived_from_terminal_materialization_truth',
            'candidate_graph_digest': proposal['candidate_graph_digest'],
            'candidate_graph': candidate_graph,
        },
        'runtime_graph_rebase_proposals': [copy.deepcopy(proposal)],
        'runtime_graph_rebase_reviews': [copy.deepcopy(review)],
        'graph_rebase_lifecycle': [copy.deepcopy(lifecycle)],
        'staged_graph_rebases': [copy.deepcopy(staged)],
        'successor_rebase_requests': [copy.deepcopy(successor)],
        'applied_graph_rebases': [],
        'graph_rebase_outcomes': [copy.deepcopy(outcome)],
    }
    return {
        'id': 'resp-projected-partial',
        'response_id': 'resp-projected-partial',
        'frame_id': 'frame-projected-partial',
        'lifecycle_state': 'completed',
        'ledger_sequence': 731,
        'updated_at': '2026-07-19T14:31:00Z',
        'request': {'prompt': 'project this graph rebase evidence'},
        'artifacts': [{'content': 'artifact-body-' + ('a' * 300_000)}],
        'outputs': [{'content': 'response-output-' + ('o' * 300_000)}],
        'late_fill': {'status': 'completed', 'active_count': 0, 'pending_count': 0},
        'frame_relation': {
            'kind': 'graph_rebase_partial_successor',
            'proposal_id': proposal['proposal_id'],
            'rebase_id': lifecycle['rebase_id'],
            'requested_rebase_class': 'partial_subtree_rebase',
        },
        'runtime': {
            'request_phase_graph': graph,
            'developer_diagnostics': diagnostics,
        },
    }


def test_projection_is_bounded_and_preserves_readiness_and_safety_evidence():
    payload = _settled_payload_with_bulky_candidate_graph()

    projection = project_graph_rebase_readiness_observation(payload)

    serialized_source = json.dumps(payload, sort_keys=True)
    serialized_projection = json.dumps(projection, sort_keys=True)
    assert projection['kind'] == GRAPH_REBASE_READINESS_OBSERVATION_KIND
    assert projection['response_id'] == 'resp-projected-partial'
    assert projection['frame_id'] == 'frame-projected-partial'
    assert projection['ledger_sequence'] == 731
    assert projection['updated_at'] == '2026-07-19T14:31:00Z'
    assert projection['readiness_state'] == {
        'active_late_fill': False,
        'settled_final': True,
    }
    assert len(serialized_projection) < len(serialized_source) // 20
    for forbidden_body in (
        'artifact-body-',
        'candidate-graph-body-',
        'candidate-output-',
        'local-contract-body-',
        'parent-graph-body-',
        'response-output-',
    ):
        assert forbidden_body not in serialized_projection

    full_report = build_graph_rebase_readiness_report([payload])
    projected_report = build_graph_rebase_readiness_report([projection])
    for key in (
        'candidate_opportunities',
        'formal_evidence',
        'qualifying_evidence',
        'safety',
        'gates',
    ):
        assert projected_report[key] == full_report[key]
    assert projected_report['corpus']['settled_final_response_count'] == 1
    assert projected_report['corpus']['unique_workload_family_count'] == 1
    assert projected_report['formal_evidence']['authorizations'][
        'inline_observed_untrusted'
    ]['total'] == 1
    assert projected_report['safety']['unresolved_by_category'][
        'same_id_semantic_drift'
    ] == 1
    assert projected_report['safety']['unresolved_by_category'][
        'root_prompt_fallback'
    ] >= 1
    assert projected_report['safety']['contained_by_category'][
        'parent_mutation'
    ] == 1


def test_projection_preserves_active_candidate_without_retaining_candidate_graph():
    payload = {
        'id': 'resp-active-projection',
        'lifecycle_state': 'late_fill_running',
        'frame_sequence': 17,
        'created_at': '2026-07-19T14:35:00Z',
        'late_fill': {
            'status': 'running',
            'active_branches': [
                {'branch_id': 'branch-active', 'payload': 'active-body-' + ('q' * 250_000)}
            ],
        },
        'request_meta': {'prompt_family': 'multimodal-review'},
        'runtime': {
            'request_phase_graph': {
                'kind': 'ollmo.request_phase_graph',
                'response_id': 'resp-active-projection',
                'frame_id': 'frame-active-projection',
            },
            'developer_diagnostics': {
                'response_time_graph_rebase_candidate': {
                    'kind': 'ollmo.runtime_graph_rebase_candidate',
                    'status': 'rederived_from_terminal_materialization_truth',
                    'candidate_graph_digest': 'candidate-active-projection',
                    'candidate_graph': {
                        'phases': [{'content_payload': 'candidate-active-' + ('r' * 250_000)}]
                    },
                },
                'runtime_graph_rebase_candidate_review': {
                    'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                    'status': 'not_proposed',
                    'reason': 'active_late_fill_must_settle',
                },
            },
        },
    }

    projection = project_graph_rebase_readiness_observation(payload)
    report = build_graph_rebase_readiness_report([projection])

    assert projection['readiness_state'] == {
        'active_late_fill': True,
        'settled_final': False,
    }
    assert projection['workload_family'] == 'multimodal_review'
    assert ('candidate-active-' + ('r' * 100)) not in json.dumps(projection)
    assert report['corpus']['settled_final_response_count'] == 0
    assert report['corpus']['nonterminal_active_late_fill_response_count'] == 1
    assert report['candidate_opportunities']['nonterminal_active_late_fill'][
        'by_reason'
    ] == {'active_late_fill_must_settle': 1}
