import copy
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from ollmo_g.decision_contracts import build_ghost_decision_contract
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_services.graph_repair import (
    GRAPH_PATCH_LIFECYCLE_KIND,
    GRAPH_REPAIR_PROPOSAL_KIND,
    PROPOSAL_ALLOWED_USE,
    PROPOSAL_FORBIDDEN_USE,
    apply_validated_graph_patch,
    apply_validated_graph_repair_patch,
    build_graph_patch_lifecycle,
    build_graph_repair_proposal_from_repair_gap,
    build_graph_repair_proposals,
    build_graph_repair_proposals_from_runtime_evidence,
    describe_graph_repair_autonomy_from_env,
    graph_repair_autonomy_from_env,
    normalize_graph_repair_autonomy,
    validate_graph_repair_proposal,
)
from scripts.ollmo_run_monitor import (
    _artifact_checks,
    _collect_learning_healing,
    _render_human,
    _render_learning_healing_lines,
)


class GraphRepairSelfHealingTests(unittest.TestCase):
    def test_graph_repair_autonomy_product_default_and_fail_closed_overrides(self):
        self.assertEqual(normalize_graph_repair_autonomy(None), 'off')
        self.assertEqual(graph_repair_autonomy_from_env({}), 'apply_enforced')

        product_default = describe_graph_repair_autonomy_from_env({})
        self.assertEqual(product_default['autonomy_level'], 'apply_enforced')
        self.assertEqual(product_default['normalized'], 'apply_enforced')
        self.assertEqual(product_default['source'], 'product_default')
        self.assertFalse(product_default['configured'])

        explicit_off = describe_graph_repair_autonomy_from_env(
            {'OLLMO_GRAPH_REPAIR_AUTONOMY': 'off'}
        )
        self.assertEqual(graph_repair_autonomy_from_env({'OLLMO_GRAPH_REPAIR_AUTONOMY': 'off'}), 'off')
        self.assertEqual(explicit_off['autonomy_level'], 'off')
        self.assertEqual(explicit_off['source'], 'environment')
        self.assertTrue(explicit_off['configured'])
        self.assertFalse(explicit_off['invalid_value'])

        invalid = describe_graph_repair_autonomy_from_env(
            {'OLLMO_GRAPH_REPAIR_AUTONOMY': 'launch_the_missiles'}
        )
        self.assertEqual(invalid['autonomy_level'], 'off')
        self.assertEqual(invalid['source'], 'environment')
        self.assertTrue(invalid['configured'])
        self.assertTrue(invalid['invalid_value'])

    def _runtime_owner(self):
        def build_late_fill_state(artifact_gap, *, status, prior_state=None, extra=None):
            state = dict(prior_state or {})
            state.update(dict(artifact_gap or {}))
            state.update(dict(extra or {}))
            state['status'] = status
            return state

        def attach_late_fill_state(payload, late_fill_state):
            updated = dict(payload)
            updated['late_fill'] = dict(late_fill_state)
            runtime = dict(updated.get('runtime') or {})
            runtime['late_fill'] = dict(late_fill_state)
            updated['runtime'] = runtime
            return updated

        return ResponsesRequestRuntimeOwner(
            hooks={
                'normalize_capability': lambda value: str(value or '').strip().lower(),
                'build_late_fill_state': build_late_fill_state,
                'attach_late_fill_state': attach_late_fill_state,
            },
            capability_chat='chat',
            capability_embedding='embedding',
            capability_image_generation='image_generation',
            capability_speech_to_text='speech_to_text',
            request_timeout_error=TimeoutError,
            request_exception_error=Exception,
        )

    def _base_graph(self):
        return {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                }
            ],
            'downstream_branches': [],
            'output_obligations': [],
        }

    def _terminal_successor_sink_fixture(
        self,
        *,
        include_unrelated_pending: bool = False,
        parent_depth: int = 0,
    ):
        owner = self._runtime_owner()
        graph = self._base_graph()
        if include_unrelated_pending:
            graph.update(
                {
                    'downstream_branch_ids': ['unrelated-image'],
                    'downstream_branches': [
                        {
                            'phase_id': 'unrelated-image',
                            'branch_id': 'unrelated-image',
                            'obligation_id': 'obligation-unrelated-image',
                            'capability': 'image_generation',
                            'output_type': 'image',
                            'status': 'pending',
                            'artifact_prompt': 'Unrelated pre-existing pending work.',
                        }
                    ],
                    'output_obligations': [
                        {
                            'obligation_id': 'obligation-unrelated-image',
                            'phase_id': 'unrelated-image',
                            'branch_id': 'unrelated-image',
                            'capability': 'image_generation',
                            'output_type': 'image',
                            'status': 'pending',
                        }
                    ],
                }
            )
        parent_relation = {'kind': 'late_fill_successor'}
        if parent_depth:
            parent_relation.update(
                {
                    'kind': 'graph_patch_reopen_successor',
                    'successor_reopen_depth': parent_depth,
                }
            )
        late_fill = {
            'status': 'partial_failed',
            'final_materialization_contract_status': 'unmet',
            'materialization_contract_unmet': True,
            'pending_branches': [
                {
                    'phase_id': 'repair-image',
                    'branch_id': 'repair-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                    'artifact_prompt': 'Only the proven repair branch is owed.',
                }
            ],
        }
        if parent_depth:
            late_fill['successor_reopen_execution'] = {
                'status': 'completed',
                'successor_reopen_depth': parent_depth,
            }
        payload = {
            'id': 'resp-terminal-successor-sink-fixture',
            'response_id': 'resp-terminal-successor-sink-fixture',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'response_frame': {
                'response_id': 'resp-terminal-successor-sink-fixture',
                'frame_id': 'resp-terminal-successor-sink-fixture:frame-6',
                'frame_sequence': 6,
                'frame_relation': parent_relation,
            },
            'runtime': {'request_phase_graph': graph},
            'late_fill': late_fill,
        }
        reviewed = owner._attach_graph_patch_lifecycle(
            owner._attach_runtime_graph_repair_evidence(payload),
            graph_repair_autonomy='apply_safe',
        )
        candidate = next(
            copy.deepcopy(item)
            for item in reviewed['runtime']['request_phase_graph']['successor_reopen_requests']
            if item.get('owed_branch_ids') == ['repair-image']
        )
        return owner, reviewed, candidate

    def _proposal(self, **overrides):
        payload = {
            'kind': GRAPH_REPAIR_PROPOSAL_KIND,
            'proposal_id': 'graph-repair-test',
            'source': 'closure_review',
            'repair_type': 'additive_graph_patch',
            'target_graph_id': 'graph-test',
            'evidence_refs': ['closure_review:ghost_repair_feedback'],
            'patch': {
                'add_phases': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'pending',
                    }
                ]
            },
            'allowed_use': PROPOSAL_ALLOWED_USE,
            'forbidden_use': PROPOSAL_FORBIDDEN_USE,
        }
        payload.update(overrides)
        return payload

    def _closure_review(self):
        return {
            'status': 'repair_required',
            'ghost_repair_feedback': {'status': 'repair_required'},
        }

    def test_closure_repair_gap_builds_validated_additive_patch(self):
        graph = self._base_graph()
        repair_gap = {
            'trigger': 'ghost_repair_feedback',
            'ghost_repair_feedback': {'status': 'repair_required'},
            'repair_loop': {'status': 'promoted'},
            'pending_branches': [
                {
                    'phase_id': 'repair-image',
                    'branch_id': 'repair-image',
                    'obligation_id': 'obligation-repair-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'repair_action': 'rebuild_from_promoted_obligations',
                    'repair_contract_id': 'contract-image',
                    'repair_evidence': 'intent_graph_adequacy_missing_capability_obligation',
                }
            ],
        }

        proposal = build_graph_repair_proposal_from_repair_gap(
            request_phase_graph=graph,
            repair_gap=repair_gap,
        )
        self.assertEqual(proposal['kind'], GRAPH_REPAIR_PROPOSAL_KIND)
        self.assertEqual(proposal['allowed_use'], PROPOSAL_ALLOWED_USE)
        self.assertEqual(proposal['forbidden_use'], PROPOSAL_FORBIDDEN_USE)

        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )

        self.assertEqual(review['status'], 'accepted')
        application = apply_validated_graph_repair_patch(graph, review)
        self.assertEqual(application['status'], 'applied')
        patched_graph = application['graph']
        self.assertEqual(patched_graph['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(patched_graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(patched_graph['output_obligations'][0]['output_type'], 'image')
        self.assertTrue(patched_graph['continuation_required'])

    def test_validated_patch_application_is_idempotent(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(),
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )
        first = apply_validated_graph_repair_patch(graph, review)
        second = apply_validated_graph_repair_patch(first['graph'], review)

        self.assertEqual(first['status'], 'applied')
        self.assertEqual(second['status'], 'already_applied')
        self.assertEqual(second['graph']['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(len(second['graph']['phases']), 2)
        self.assertEqual(len(second['graph']['output_obligations']), 1)

    def test_validated_patch_adds_dependency_edges_idempotently(self):
        graph = self._base_graph()
        graph['phases'].append(
            {
                'phase_id': 'phase-2',
                'branch_id': 'phase-2',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
            }
        )
        proposal = self._proposal(
            patch={
                'add_dependencies': [
                    {
                        'target_phase_id': 'phase-2',
                        'source_phase_id': 'phase-1',
                    }
                ]
            }
        )
        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )

        first = apply_validated_graph_repair_patch(graph, review)
        second = apply_validated_graph_repair_patch(first['graph'], review)

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(first['status'], 'applied')
        self.assertEqual(first['applied_dependency_edges'], [{'target_id': 'phase-2', 'source_id': 'phase-1'}])
        phase_2 = next(phase for phase in first['graph']['phases'] if phase['phase_id'] == 'phase-2')
        self.assertEqual(phase_2['depends_on'], ['phase-1'])
        self.assertEqual(second['status'], 'already_applied')
        self.assertEqual(second.get('applied_dependency_edges', []), [])

    def test_intent_adequacy_dependency_gap_builds_safe_additive_repair_proposal(self):
        prompt = (
            'Create a small two-page website with index.html, suiten.html, shared styles.css, '
            'navigation between both pages, and exactly two generated local images linked from the pages.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        image_phase_id = next(
            branch['phase_id'] for branch in graph['downstream_branches']
            if branch.get('capability') == 'image_generation'
        )
        html_phase_id = next(
            branch['phase_id'] for branch in graph['downstream_branches']
            if branch.get('role') == 'text_artifact_output'
            and branch.get('text_artifact_source_name') == 'index'
        )
        repair_graph = dict(graph)
        repair_graph['phases'] = [dict(item) for item in graph.get('phases', [])]
        repair_graph['downstream_branches'] = [dict(item) for item in graph.get('downstream_branches', [])]
        repair_graph['output_obligations'] = [dict(item) for item in graph.get('output_obligations', [])]
        for collection in (repair_graph['phases'], repair_graph['downstream_branches'], repair_graph['output_obligations']):
            for record in collection:
                if html_phase_id in {
                    record.get('phase_id'),
                    record.get('branch_id'),
                    record.get('obligation_id'),
                }:
                    record['depends_on'] = ['phase-1']
        closure_review = {
            'status': 'pending',
            'intent_graph_adequacy': {
                'status': 'pending',
                'checks': [
                    {
                        'check_kind': 'intent_graph_adequacy',
                        'obligation_id': 'intent-obligation-dependency-index-local-image',
                        'intent_obligation_id': 'intent-obligation-dependency-index-local-image',
                        'status': 'pending',
                        'evidence': 'intent_graph_adequacy_missing_dependency_edge',
                        'repair_action': 'rebind_artifact_dependency',
                        'dependency_contract': 'local_visual_asset_binding',
                        'add_dependencies': [
                            {
                                'target_phase_id': html_phase_id,
                                'source_phase_id': image_phase_id,
                                'dependency_contract': 'local_visual_asset_binding',
                            }
                        ],
                    }
                ],
            },
        }

        proposals = build_graph_repair_proposals(
            request_phase_graph=repair_graph,
            closure_review=closure_review,
        )
        proposal = proposals[0]
        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=repair_graph,
            closure_review=closure_review,
            promotion_review={},
        )
        first = apply_validated_graph_repair_patch(repair_graph, review)
        second = apply_validated_graph_repair_patch(first['graph'], review)

        self.assertEqual(proposal['repair_type'], 'rebind_artifact_dependency')
        self.assertEqual(proposal['patch']['add_dependencies'][0]['target_phase_id'], html_phase_id)
        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(
            first['applied_dependency_edges'],
            [{'target_id': html_phase_id, 'source_id': image_phase_id}],
        )
        self.assertEqual(second['status'], 'already_applied')

    def test_graph_patch_lifecycle_shadow_records_without_executable_graph_mutation(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='repair_missing_materialization_contract',
                evidence_refs=['late_fill:materialization_contract_unmet'],
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )

        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='shadow',
        )
        result = apply_validated_graph_patch(graph, lifecycle, autonomy_level='shadow')

        self.assertEqual(normalize_graph_repair_autonomy(''), 'off')
        self.assertEqual(lifecycle['kind'], GRAPH_PATCH_LIFECYCLE_KIND)
        self.assertEqual(lifecycle['autonomy_level'], 'shadow')
        self.assertEqual(lifecycle['status'], 'validated')
        self.assertEqual(lifecycle['outcome']['runtime_effect'], 'shadow_no_mutation')
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(graph.get('downstream_branch_ids', []), [])
        self.assertEqual(result['graph'].get('downstream_branch_ids', []), [])
        self.assertEqual(result['graph']['graph_patch_lifecycle'][0]['status'], 'validated')

    def test_graph_patch_lifecycle_stage_records_without_executable_graph_mutation(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='repair_missing_materialization_contract',
                evidence_refs=['late_fill:materialization_contract_unmet'],
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )

        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='stage',
        )
        result = apply_validated_graph_patch(graph, lifecycle, autonomy_level='stage')

        self.assertEqual(lifecycle['status'], 'staged')
        self.assertEqual(lifecycle['outcome']['runtime_effect'], 'staged_no_executable_mutation')
        self.assertEqual(result['status'], 'staged')
        self.assertEqual(result['graph'].get('downstream_branch_ids', []), [])
        self.assertEqual(len(result['graph']['staged_graph_patches']), 1)

    def test_apply_safe_missing_materialization_patch_applies_once_with_lifecycle(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='repair_missing_materialization_contract',
                evidence_refs=['late_fill:materialization_contract_unmet'],
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )
        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='apply_safe',
        )

        first = apply_validated_graph_patch(graph, lifecycle, autonomy_level='apply_safe')
        second = apply_validated_graph_patch(first['graph'], lifecycle, autonomy_level='apply_safe')

        self.assertEqual(first['status'], 'applied')
        self.assertEqual(second['status'], 'already_applied')
        self.assertEqual(first['graph']['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(first['graph']['graph_patch_lifecycle'][0]['status'], 'applied')
        self.assertEqual(first['graph']['applied_graph_patches'][0]['idempotency_key'], lifecycle['idempotency_key'])
        self.assertEqual(second['graph']['downstream_branch_ids'], ['repair-image'])

    def test_apply_safe_missing_dependency_edge_applies_once_with_lifecycle(self):
        graph = self._base_graph()
        graph['phases'].append(
            {
                'phase_id': 'phase-2',
                'branch_id': 'phase-2',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
            }
        )
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='rebind_artifact_dependency',
                evidence_refs=['monitor:artifact_dependency:artifact://fake-image-ref'],
                patch={
                    'add_dependencies': [
                        {
                            'target_phase_id': 'phase-2',
                            'source_phase_id': 'phase-1',
                        }
                    ]
                },
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )
        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='apply_safe',
        )

        first = apply_validated_graph_patch(graph, lifecycle, autonomy_level='apply_safe')
        second = apply_validated_graph_patch(first['graph'], lifecycle, autonomy_level='apply_safe')

        self.assertEqual(lifecycle['repair_class'], 'missing_dependency_edge')
        self.assertEqual(first['status'], 'applied')
        self.assertEqual(first['applied_dependency_edges'], [{'target_id': 'phase-2', 'source_id': 'phase-1'}])
        self.assertEqual(second['status'], 'already_applied')

    def test_apply_safe_blocks_review_required_patch_class(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='branch_split',
                evidence_refs=['monitor:branch_split:phase-1'],
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )
        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='apply_safe',
        )
        result = apply_validated_graph_patch(graph, lifecycle, autonomy_level='apply_safe')

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(lifecycle['risk_level'], 'review_required')
        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('patch_class_requires_reviewed_autonomy', result['blocked_reasons'])
        self.assertEqual(result['graph'].get('downstream_branch_ids', []), [])

    def test_apply_reviewed_requires_explicit_runtime_review_authorization(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='branch_split',
                evidence_refs=['monitor:branch_split:phase-1'],
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )

        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='apply_reviewed',
        )
        result = apply_validated_graph_patch(graph, lifecycle, autonomy_level='apply_reviewed')

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(lifecycle['risk_level'], 'review_required')
        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertIn('apply_reviewed_requires_explicit_review_authorization', lifecycle['blocked_reasons'])
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('apply_reviewed_requires_explicit_review_authorization', result['blocked_reasons'])
        self.assertEqual(result['graph'].get('downstream_branch_ids', []), [])

    def test_apply_reviewed_with_authorization_can_apply_review_required_patch(self):
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                source='monitor_evidence',
                repair_type='branch_split',
                evidence_refs=['monitor:branch_split:phase-1'],
            ),
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )
        review = {
            **review,
            'graph_patch_authorization': {
                'status': 'accepted',
                'authority': 'runtime_review',
                'allowed_autonomy': ['apply_reviewed'],
                'evidence_refs': ['runtime_review:authorized-branch-split'],
            },
        }

        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=review,
            autonomy_level='apply_reviewed',
        )
        result = apply_validated_graph_patch(graph, lifecycle, autonomy_level='apply_reviewed')

        self.assertEqual(lifecycle['risk_level'], 'review_required')
        self.assertEqual(lifecycle['status'], 'staged')
        self.assertEqual(lifecycle['graph_patch_authorization']['status'], 'accepted')
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(result['graph']['downstream_branch_ids'], ['repair-image'])

    def test_accepted_learning_alone_cannot_create_executable_graph_truth(self):
        graph = self._base_graph()
        proposal = self._proposal(
            source='accepted_learning',
            evidence_refs=['accepted_learning:case-1'],
        )

        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
            accepted_learning_hints={
                'authority': 'soft_hint',
                'runtime_effect': 'none',
                'hint_count': 1,
            },
        )
        application = apply_validated_graph_repair_patch(graph, review)

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('accepted_learning_not_runtime_evidence', review['reasons'])
        self.assertEqual(application['status'], 'rejected')
        self.assertEqual(application['graph'].get('downstream_branches', []), [])

    def test_basic_intent_learning_orients_graph_repair_without_authority(self):
        graph = self._base_graph()
        proposals = build_graph_repair_proposals(
            request_phase_graph=graph,
            accepted_learning_hints={
                'status': 'active',
                'enabled': True,
                'authority': 'soft_hint',
                'runtime_effect': 'soft_hints_available',
                'hint_count': 1,
                'hints': [
                    {
                        'learning_id': 'accepted-basic-intent',
                        'target_area': 'ghost_intake_graph_policy',
                        'hint': 'Basic intent was underrepresented; propose bounded graph repair.',
                        'case_kinds': {'intent_graph_inadequacy': 2},
                    }
                ],
            },
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]['source'], 'accepted_learning')
        self.assertEqual(proposals[0]['repair_type'], 'advisory_orientation_only')
        review = validate_graph_repair_proposal(
            proposals[0],
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )
        self.assertEqual(review['status'], 'rejected')
        self.assertIn('accepted_learning_not_runtime_evidence', review['reasons'])

    def test_capability_output_mismatch_is_rejected(self):
        graph = self._base_graph()
        proposal = self._proposal(
            patch={
                'add_phases': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'text',
                        'status': 'pending',
                    }
                ]
            }
        )

        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('capability_output_contract_mismatch', review['reasons'])

    def test_reserved_or_deferred_intent_blocks_executable_patch(self):
        graph = self._base_graph()
        proposal = self._proposal(
            patch={
                'add_phases': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'contract_state': 'reserved',
                    }
                ]
            }
        )

        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('deferred_or_reserved_intent_conflict', review['reasons'])

    def test_backend_route_health_issue_is_diagnostic_not_graph_patch(self):
        graph = self._base_graph()
        proposal = self._proposal(provider_bans=['some-backend-family'])

        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'backend_route_health_signal_is_not_graph_patch_authority',
            review['reasons'],
        )

    def test_decision_contract_exposes_repair_proposals_as_advisory_only(self):
        contract = build_ghost_decision_contract(
            workload_graph={
                'tasks': [
                    {
                        'task_id': 'task-image',
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'repair_action': 'rebuild_from_promoted_obligations',
                    }
                ]
            },
            accepted_learning_hints={
                'status': 'enabled',
                'authority': 'soft_hint',
                'runtime_effect': 'none',
                'hint_count': 1,
                'hints': [],
            },
        )

        proposals = contract['graph_repair_proposals']
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]['kind'], GRAPH_REPAIR_PROPOSAL_KIND)
        self.assertEqual(proposals[0]['allowed_use'], PROPOSAL_ALLOWED_USE)
        self.assertEqual(proposals[0]['forbidden_use'], PROPOSAL_FORBIDDEN_USE)
        self.assertIn(
            'review_graph_repair_proposals_before_runtime_patch',
            contract['next_decision_priorities'],
        )

    def test_runtime_attach_records_validated_graph_repair_review(self):
        owner = self._runtime_owner()
        payload = {'runtime': {'request_phase_graph': self._base_graph()}}
        updated = owner._attach_repair_gap_to_request_phase_graph(
            payload,
            {
                'repair_action': 'rebuild_from_promoted_obligations',
                'repair_actions': ['rebuild_from_promoted_obligations'],
                'ghost_repair_feedback': {'status': 'repair_required'},
                'repair_loop': {'status': 'promoted'},
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'repair_action': 'rebuild_from_promoted_obligations',
                    }
                ],
            },
        )

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(graph['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(graph['graph_repair_reviews'][0]['status'], 'accepted')
        self.assertEqual(diagnostics['graph_repair_proposal_review']['status'], 'accepted')
        self.assertEqual(
            diagnostics['closure_repair_graph_patch']['graph_repair_review_status'],
            'accepted',
        )

    def test_runtime_attach_rejects_unvalidated_backend_route_health_patch(self):
        owner = self._runtime_owner()
        payload = {'runtime': {'request_phase_graph': self._base_graph()}}
        updated = owner._attach_repair_gap_to_request_phase_graph(
            payload,
            {
                'ghost_repair_feedback': {'status': 'repair_required'},
                'repair_loop': {'status': 'promoted'},
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'provider_bans': ['backend-family'],
                    }
                ],
            },
        )

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(graph.get('downstream_branch_ids', []), [])
        self.assertEqual(graph['graph_repair_reviews'][0]['status'], 'rejected')
        self.assertEqual(diagnostics['graph_repair_proposal_review']['status'], 'rejected')
        self.assertIn(
            'backend_route_health_signal_is_not_graph_patch_authority',
            diagnostics['graph_repair_proposal_review']['reasons'],
        )

    def test_runtime_graph_patch_lifecycle_shadow_and_stage_do_not_create_executable_work(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-shadow-stage-patch',
            'lifecycle_state': 'late_fill_running',
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        with_evidence = owner._attach_runtime_graph_repair_evidence(payload)

        shadowed = owner._attach_graph_patch_lifecycle(with_evidence, graph_repair_autonomy='shadow')
        staged = owner._attach_graph_patch_lifecycle(with_evidence, graph_repair_autonomy='stage')

        shadow_graph = shadowed['runtime']['request_phase_graph']
        stage_graph = staged['runtime']['request_phase_graph']
        self.assertEqual(shadow_graph.get('downstream_branch_ids', []), [])
        self.assertEqual(stage_graph.get('downstream_branch_ids', []), [])
        self.assertEqual(shadow_graph['graph_patch_lifecycle'][0]['status'], 'validated')
        self.assertEqual(stage_graph['graph_patch_lifecycle'][0]['status'], 'staged')
        self.assertEqual(len(stage_graph['staged_graph_patches']), 1)

    def test_runtime_graph_patch_lifecycle_apply_safe_adds_late_fill_visible_work(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-apply-safe-patch',
            'lifecycle_state': 'late_fill_running',
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        with_evidence = owner._attach_runtime_graph_repair_evidence(payload)

        updated = owner._attach_graph_patch_lifecycle(with_evidence, graph_repair_autonomy='apply_safe')

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(graph['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(graph['output_obligations'][0]['phase_id'], 'repair-image')
        self.assertEqual(graph['graph_patch_lifecycle'][0]['status'], 'applied')
        self.assertEqual(graph['applied_graph_patches'][0]['status'], 'applied')
        self.assertEqual(diagnostics['graph_patch_autonomy']['autonomy_level'], 'apply_safe')

    def test_apply_safe_gap_becomes_normal_late_fill_visible_runtime_truth(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-apply-safe-visible-gap',
            'lifecycle_state': 'late_fill_running',
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        with_evidence = owner._attach_runtime_graph_repair_evidence(payload)

        updated = owner._attach_graph_patch_lifecycle(with_evidence, graph_repair_autonomy='apply_safe')

        graph = updated['runtime']['request_phase_graph']
        self.assertTrue(updated['late_fill']['materialization_contract_unmet'])
        self.assertEqual(updated['late_fill']['final_materialization_contract_status'], 'unmet')
        self.assertEqual(graph['graph_repair_proposals'][0]['repair_type'], 'repair_missing_materialization_contract')
        self.assertEqual(graph['graph_repair_reviews'][0]['status'], 'accepted')
        self.assertEqual(graph['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(graph['output_obligations'][0]['status'], 'pending')
        self.assertEqual(graph['output_obligations'][0]['phase_id'], 'repair-image')
        self.assertEqual(graph['graph_patch_lifecycle'][0]['status'], 'applied')
        self.assertIn(
            'late_fill:materialization_contract_unmet',
            graph['graph_patch_lifecycle'][0]['source_evidence_refs'],
        )

    def test_completed_compatibility_status_is_not_a_pre_freeze_terminal_frame(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-pre-freeze-compat-completed',
            'status': 'completed',
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        with_evidence = owner._attach_runtime_graph_repair_evidence(payload)

        updated = owner._attach_graph_patch_lifecycle(
            with_evidence,
            graph_repair_autonomy='apply_safe',
        )

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']['graph_patch_autonomy']
        self.assertEqual(graph['graph_patch_lifecycle'][0]['status'], 'applied')
        self.assertEqual(graph['downstream_branch_ids'], ['repair-image'])
        self.assertFalse(graph.get('successor_reopen_requests'))
        self.assertEqual(diagnostics['response_state'], 'pre_freeze')
        self.assertEqual(diagnostics['response_state_source'], 'pre_freeze_runtime')
        self.assertFalse(diagnostics['terminal_apply_blocked'])

    def test_pre_freeze_applied_patch_is_returned_as_same_turn_late_fill_gap(self):
        def attach_late_fill_state(payload, state):
            updated = dict(payload)
            runtime = dict(updated.get('runtime') or {})
            updated['late_fill'] = dict(state)
            runtime['late_fill'] = dict(state)
            updated['runtime'] = runtime
            return updated

        closure_review_calls = []

        def build_graph_closure_review(*args, **kwargs):
            artifact_payload = kwargs.get('artifact_payload') or {}
            runtime = artifact_payload.get('runtime') if isinstance(artifact_payload.get('runtime'), dict) else {}
            graph = runtime.get('request_phase_graph') if isinstance(runtime.get('request_phase_graph'), dict) else {}
            closure_review_calls.append(graph)
            repair_branch = next(
                (
                    dict(item)
                    for item in (graph.get('downstream_branches') or [])
                    if isinstance(item, dict) and item.get('branch_id') == 'repair-image'
                ),
                None,
            )
            if repair_branch:
                return {
                    'kind': 'ollmo.graph_closure_review',
                    'status': 'pending',
                    'reason': 'one or more graph obligations remain open inside the same intent',
                    'pending_branch_count': 1,
                    'counts': {'fulfilled': 2, 'pending': 1, 'blocked': 0},
                    'checks': [
                        {
                            'check_kind': 'graph_phase',
                            'branch_id': 'repair-image',
                            'phase_id': 'repair-image',
                            'status': 'pending',
                        }
                    ],
                    'surface_state': {
                        'kind': 'ollmo.surface_state',
                        'status': 'pending',
                        'category_counts': {'late_fill_pending': 1},
                        'active_categories': ['late_fill_pending'],
                    },
                }
            return {
                'kind': 'ollmo.graph_closure_review',
                'status': 'fulfilled',
                'surface_state': {'kind': 'ollmo.surface_state', 'status': 'fulfilled'},
            }

        owner = ResponsesRequestRuntimeOwner(
            hooks={
                'normalize_capability': lambda value: str(value or '').strip().lower(),
                'extract_responses_prompt': lambda payload: str(payload.get('prompt') or ''),
                'extract_responses_current_turn_prompt': lambda payload: str(payload.get('prompt') or ''),
                'build_pre_freeze_closure_review_gap': lambda *args, **kwargs: None,
                'build_graph_closure_review': build_graph_closure_review,
                'build_late_fill_state': lambda gap, *, status, prior_state=None, extra=None: {
                    **dict(prior_state or {}),
                    **dict(gap),
                    **dict(extra or {}),
                    'status': status,
                },
                'attach_late_fill_state': attach_late_fill_state,
            },
            capability_chat='chat',
            capability_embedding='embedding',
            capability_image_generation='image_generation',
            capability_speech_to_text='speech_to_text',
            request_timeout_error=TimeoutError,
            request_exception_error=Exception,
        )
        graph = self._base_graph()
        graph['downstream_branches'] = [
            {
                'branch_id': 'completed-context',
                'phase_id': 'completed-context',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'completed',
            }
        ]
        graph['downstream_branch_ids'] = ['completed-context']
        graph['phases'].append(dict(graph['downstream_branches'][0]))
        review = validate_graph_repair_proposal(
            self._proposal(
                patch={
                    'add_phases': [
                        {
                            'phase_id': 'repair-image',
                            'branch_id': 'repair-image',
                            'capability': 'image_generation',
                            'output_type': 'image',
                            'status': 'pending',
                        },
                        {
                            'phase_id': 'repair-image-blocked',
                            'branch_id': 'repair-image-blocked',
                            'capability': 'image_generation',
                            'output_type': 'image',
                            'status': 'blocked',
                            'artifact_prompt': 'A second bounded image waiting for dependency evidence.',
                            'repair_action': 'repair_dependency_chain',
                            'repair_execution_policy': 'blocked_until_dependency_evidence',
                            'auto_execute': False,
                            'materialization_blocked': True,
                        },
                    ]
                },
            ),
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )
        self.assertEqual(review['status'], 'accepted')
        graph['graph_repair_reviews'] = [review]

        with patch.dict('os.environ', {'OLLMO_GRAPH_REPAIR_AUTONOMY': 'apply_safe'}, clear=False):
            updated, gap = owner.attach_pre_freeze_closure_review(
                {
                    'response_id': 'resp-runtime-same-turn-reseed',
                    'status': 'completed',
                    'output_text': 'The current response is complete.',
                },
                output_text='The current response is complete.',
                route_payload={'route_runtime': {'request_phase_graph': graph}},
                request_payload={'prompt': 'Return the current response.'},
            )

        self.assertIsNotNone(gap)
        self.assertEqual(gap['code'], 'graph_patch_late_fill')
        self.assertEqual(gap['trigger'], 'runtime_applied_graph_patch')
        self.assertEqual(gap['pending_branches'][0]['branch_id'], 'repair-image')
        self.assertEqual(updated['late_fill']['pending_branches'][0]['branch_id'], 'repair-image')
        patched_graph = updated['runtime']['request_phase_graph']
        self.assertIn('repair-image', patched_graph['downstream_branch_ids'])
        self.assertFalse(patched_graph.get('successor_reopen_requests'))
        reconciliation = updated['runtime']['developer_diagnostics']['graph_patch_late_fill_reconciliation']
        self.assertEqual(reconciliation['status'], 'applied')
        self.assertEqual(reconciliation['scheduled_branch_ids'], ['repair-image'])
        self.assertEqual(reconciliation['unscheduled_branch_ids'], ['repair-image-blocked'])
        self.assertEqual(updated['late_fill']['pending_branches'][0]['branch_id'], 'repair-image')
        self.assertEqual(updated['late_fill']['blocked_branches'][0]['branch_id'], 'repair-image-blocked')
        self.assertEqual(updated['late_fill']['blocked_branches'][0]['status'], 'blocked')
        self.assertEqual(len(closure_review_calls), 2)
        self.assertNotIn('repair-image', closure_review_calls[0].get('downstream_branch_ids', []))
        self.assertIn('repair-image', closure_review_calls[1]['downstream_branch_ids'])
        final_closure = updated['runtime']['graph_closure_review']
        self.assertEqual(final_closure['status'], 'pending')
        self.assertEqual(final_closure['pending_branch_count'], 1)
        self.assertEqual(final_closure['checks'][0]['branch_id'], 'repair-image')
        self.assertEqual(final_closure['checks'][0]['status'], 'pending')
        self.assertIn('late_fill_pending', final_closure['surface_state']['active_categories'])

    def test_reconcile_applied_patch_does_not_schedule_policy_blocked_branch(self):
        owner = self._runtime_owner()
        blocked_branch = {
            'branch_id': 'repair-image',
            'phase_id': 'repair-image',
            'obligation_id': 'obligation-repair-image',
            'capability': 'image_generation',
            'output_type': 'image',
            'status': 'blocked',
            'artifact_prompt': 'A concrete image that must wait for dependency evidence.',
            'repair_action': 'repair_dependency_chain',
            'repair_execution_policy': 'blocked_until_dependency_evidence',
            'auto_execute': False,
            'materialization_blocked': True,
        }
        graph = self._base_graph()
        graph['phases'].append(dict(blocked_branch))
        graph['downstream_branches'] = [dict(blocked_branch)]
        graph['downstream_branch_ids'] = ['repair-image']
        graph['output_obligations'] = [
            {
                'obligation_id': 'obligation-repair-image',
                'branch_id': 'repair-image',
                'phase_id': 'repair-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'required': True,
                'status': 'blocked',
                'repair_execution_policy': 'blocked_until_dependency_evidence',
                'auto_execute': False,
                'materialization_blocked': True,
            }
        ]
        payload = {
            'response_id': 'resp-runtime-policy-blocked-reseed',
            'runtime': {
                'request_phase_graph': graph,
                'developer_diagnostics': {
                    'graph_patch_lifecycle_results': [
                        {
                            'status': 'applied',
                            'patch_id': 'patch-policy-blocked-reseed',
                            'applied_branch_ids': ['repair-image'],
                            'applied_phase_ids': ['repair-image'],
                            'applied_obligation_ids': ['obligation-repair-image'],
                        }
                    ]
                },
            },
        }

        updated, gap = owner._reconcile_applied_graph_patch_late_fill_gap(payload, None)

        self.assertIsNone(gap)
        reconciliation = updated['runtime']['developer_diagnostics']['graph_patch_late_fill_reconciliation']
        self.assertEqual(reconciliation['status'], 'no_executable_branches')
        self.assertEqual(reconciliation['scheduled_branch_ids'], [])
        self.assertEqual(reconciliation['unscheduled_branch_ids'], ['repair-image'])
        self.assertEqual(
            reconciliation['unscheduled_branches'][0]['repair_execution_policy'],
            'blocked_until_dependency_evidence',
        )
        patched_branch = updated['runtime']['request_phase_graph']['downstream_branches'][0]
        self.assertEqual(patched_branch['status'], 'blocked')
        self.assertFalse(patched_branch['auto_execute'])

        for collection_name in ('phases', 'downstream_branches', 'output_obligations'):
            for item in graph[collection_name]:
                if item.get('branch_id') == 'repair-image' or item.get('phase_id') == 'repair-image':
                    item['input_refs'] = [{'path': '/tmp/concrete-dependency.png'}]
        resolved_payload = {
            'response_id': 'resp-runtime-policy-resolved-reseed',
            'runtime': {
                'request_phase_graph': graph,
                'developer_diagnostics': {
                    'graph_patch_lifecycle_results': [
                        {
                            'status': 'applied',
                            'patch_id': 'patch-policy-resolved-reseed',
                            'applied_branch_ids': ['repair-image'],
                            'applied_phase_ids': ['repair-image'],
                            'applied_obligation_ids': ['obligation-repair-image'],
                        }
                    ]
                },
            },
        }

        resolved, resolved_gap = owner._reconcile_applied_graph_patch_late_fill_gap(
            resolved_payload,
            None,
        )

        self.assertIsNotNone(resolved_gap)
        self.assertEqual(resolved_gap['pending_branches'][0]['branch_id'], 'repair-image')
        self.assertEqual(
            resolved_gap['pending_branches'][0]['repair_execution_policy'],
            'schedule_late_fill_branch',
        )
        self.assertTrue(resolved_gap['pending_branches'][0]['auto_execute'])
        self.assertFalse(resolved_gap['pending_branches'][0]['materialization_blocked'])
        self.assertEqual(
            resolved['runtime']['developer_diagnostics']['graph_patch_late_fill_reconciliation']['status'],
            'applied',
        )

    def test_pre_freeze_policy_blocked_patch_recomputes_closure_without_pending_late_fill(self):
        blocked_branch = {
            'branch_id': 'repair-image',
            'phase_id': 'repair-image',
            'obligation_id': 'obligation-repair-image',
            'capability': 'image_generation',
            'output_type': 'image',
            'status': 'blocked',
            'artifact_prompt': 'A bounded image branch waiting for dependency evidence.',
            'repair_action': 'repair_dependency_chain',
            'repair_execution_policy': 'blocked_until_dependency_evidence',
            'auto_execute': False,
            'materialization_blocked': True,
        }
        closure_calls = []

        def build_graph_closure_review(*args, **kwargs):
            artifact_payload = kwargs.get('artifact_payload') or {}
            runtime = artifact_payload.get('runtime') if isinstance(artifact_payload.get('runtime'), dict) else {}
            graph = runtime.get('request_phase_graph') if isinstance(runtime.get('request_phase_graph'), dict) else {}
            closure_calls.append(graph)
            if 'repair-image' in (graph.get('downstream_branch_ids') or []):
                return {
                    'kind': 'ollmo.graph_closure_review',
                    'status': 'blocked',
                    'reason': 'one or more graph obligations are blocked by runtime truth',
                    'pending_branch_count': 0,
                    'counts': {'fulfilled': 1, 'pending': 0, 'blocked': 1},
                    'checks': [
                        {
                            'check_kind': 'graph_phase',
                            'branch_id': 'repair-image',
                            'phase_id': 'repair-image',
                            'status': 'blocked',
                            'repair_execution_policy': 'blocked_until_dependency_evidence',
                        }
                    ],
                    'surface_state': {
                        'kind': 'ollmo.surface_state',
                        'status': 'blocked',
                        'category_counts': {'blocked': 1},
                        'active_categories': ['blocked'],
                    },
                }
            return {
                'kind': 'ollmo.graph_closure_review',
                'status': 'fulfilled',
                'surface_state': {'kind': 'ollmo.surface_state', 'status': 'fulfilled'},
            }

        def attach_late_fill_state(payload, state):
            updated = dict(payload)
            runtime = dict(updated.get('runtime') or {})
            updated['late_fill'] = dict(state)
            runtime['late_fill'] = dict(state)
            updated['runtime'] = runtime
            return updated

        owner = ResponsesRequestRuntimeOwner(
            hooks={
                'normalize_capability': lambda value: str(value or '').strip().lower(),
                'extract_responses_prompt': lambda payload: str(payload.get('prompt') or ''),
                'extract_responses_current_turn_prompt': lambda payload: str(payload.get('prompt') or ''),
                'build_pre_freeze_closure_review_gap': lambda *args, **kwargs: {
                    'code': 'closure_review_repair',
                    'trigger': 'ghost_repair_feedback',
                    'expected_capability': 'image_generation',
                    'active_capability': 'image_generation',
                    'pending_capabilities': ['image_generation'],
                    'pending_branches': [dict(blocked_branch)],
                },
                'build_graph_closure_review': build_graph_closure_review,
                'build_late_fill_state': lambda gap, *, status, prior_state=None, extra=None: {
                    **dict(prior_state or {}),
                    **dict(gap),
                    **dict(extra or {}),
                    'status': status,
                },
                'attach_late_fill_state': attach_late_fill_state,
            },
            capability_chat='chat',
            capability_embedding='embedding',
            capability_image_generation='image_generation',
            capability_speech_to_text='speech_to_text',
            request_timeout_error=TimeoutError,
            request_exception_error=Exception,
        )
        graph = self._base_graph()
        review = validate_graph_repair_proposal(
            self._proposal(
                proposal_id='graph-repair-policy-blocked',
                patch={'add_phases': [dict(blocked_branch)]},
            ),
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            promotion_review={'status': 'promoted'},
        )
        self.assertEqual(review['status'], 'accepted')
        graph['graph_repair_reviews'] = [review]

        with patch.dict('os.environ', {'OLLMO_GRAPH_REPAIR_AUTONOMY': 'apply_safe'}, clear=False):
            updated, gap = owner.attach_pre_freeze_closure_review(
                {
                    'response_id': 'resp-runtime-policy-blocked-full-path',
                    'status': 'completed',
                    'output_text': 'The current text response is complete.',
                },
                output_text='The current text response is complete.',
                route_payload={'route_runtime': {'request_phase_graph': graph}},
                request_payload={'prompt': 'Return the current response.'},
            )

        self.assertIsNone(gap)
        self.assertEqual(len(closure_calls), 2)
        self.assertEqual(updated['runtime']['graph_closure_review']['status'], 'blocked')
        self.assertEqual(updated['runtime']['graph_closure_review']['checks'][0]['status'], 'blocked')
        self.assertEqual(updated['late_fill']['status'], 'blocked')
        self.assertEqual(updated['late_fill']['pending_branches'], [])
        self.assertEqual(updated['late_fill']['blocked_branches'][0]['branch_id'], 'repair-image')
        self.assertEqual(updated['late_fill']['surface_state']['status'], 'blocked')
        self.assertIn('blocked', updated['late_fill']['surface_state']['active_categories'])
        reconciliation = updated['runtime']['developer_diagnostics']['graph_patch_late_fill_reconciliation']
        self.assertEqual(reconciliation['status'], 'no_executable_branches')
        self.assertEqual(reconciliation['scheduled_branch_ids'], [])
        self.assertEqual(reconciliation['unscheduled_branch_ids'], ['repair-image'])

    def test_runtime_graph_patch_lifecycle_invalid_autonomy_is_visible_and_safe_off(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-invalid-autonomy',
            'lifecycle_state': 'late_fill_running',
            'runtime': {'request_phase_graph': self._base_graph()},
        }

        updated = owner._attach_graph_patch_lifecycle(payload, graph_repair_autonomy='launch_the_missiles')

        diagnostics = updated['runtime']['developer_diagnostics']['graph_patch_autonomy']
        graph = updated['runtime']['request_phase_graph']
        self.assertEqual(diagnostics['raw_value'], 'launch_the_missiles')
        self.assertEqual(diagnostics['autonomy_level'], 'off')
        self.assertEqual(diagnostics['normalized'], 'off')
        self.assertTrue(diagnostics['invalid_value'])
        self.assertEqual(graph.get('downstream_branch_ids', []), [])

    def test_runtime_graph_patch_diagnostics_preserve_product_default_provenance(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-product-default-autonomy',
            'status': 'completed',
            'runtime': {'request_phase_graph': self._base_graph()},
        }

        with patch.dict('os.environ', {}, clear=True):
            updated = owner._attach_graph_patch_lifecycle(payload)

        diagnostics = updated['runtime']['developer_diagnostics']
        autonomy = diagnostics['graph_patch_autonomy']
        policy = diagnostics['enforced_policy']
        self.assertEqual(autonomy['autonomy_level'], 'apply_enforced')
        self.assertEqual(autonomy['source'], 'product_default')
        self.assertFalse(autonomy['configured'])
        self.assertEqual(autonomy['response_state'], 'pre_freeze')
        self.assertEqual(policy['mode'], 'safe_v1')
        self.assertEqual(policy['source'], 'product_default')
        self.assertFalse(policy['configured'])

    def test_runtime_graph_patch_lifecycle_apply_safe_blocks_terminal_frame_mutation(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-terminal-patch',
            'lifecycle_state': 'completed',
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        with_evidence = owner._attach_runtime_graph_repair_evidence(payload)

        updated = owner._attach_graph_patch_lifecycle(with_evidence, graph_repair_autonomy='apply_safe')

        graph = updated['runtime']['request_phase_graph']
        lifecycle_statuses = {item['status'] for item in graph['graph_patch_lifecycle']}
        blocked_reasons = {
            reason
            for item in graph['graph_patch_lifecycle']
            for reason in item.get('blocked_reasons', [])
        }
        self.assertEqual(graph.get('downstream_branch_ids', []), [])
        self.assertEqual(lifecycle_statuses, {'blocked'})
        self.assertIn('terminal_frame_requires_successor_reopen', blocked_reasons)

    def test_runtime_graph_patch_lifecycle_apply_safe_terminal_creates_successor_reopen_request(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-terminal-successor',
            'lifecycle_state': 'completed',
            'response_frame': {
                'frame_id': 'resp-runtime-terminal-successor:frame-1',
                'frame_sequence': 1,
                'frame_relation': {'kind': 'initial'},
            },
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        with_evidence = owner._attach_runtime_graph_repair_evidence(payload)

        updated = owner._attach_graph_patch_lifecycle(with_evidence, graph_repair_autonomy='apply_safe')

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(graph.get('downstream_branch_ids', []), [])
        self.assertEqual(graph['graph_patch_lifecycle'][0]['status'], 'blocked')
        self.assertIn(
            'terminal_frame_requires_successor_reopen',
            graph['graph_patch_lifecycle'][0]['blocked_reasons'],
        )

        successor_requests = graph['successor_reopen_requests']
        self.assertGreaterEqual(len(successor_requests), 1)
        successor = next(
            item
            for item in successor_requests
            if item.get('repair_class') == 'missing_materialization_branch'
        )
        successor_graph = successor['successor_request_phase_graph']
        self.assertEqual(successor['kind'], 'ollmo.graph_patch_successor_reopen_request')
        self.assertEqual(successor['status'], 'candidate')
        self.assertEqual(successor['runtime_effect'], 'successor_reopen_required')
        self.assertEqual(successor['parent_response_id'], 'resp-runtime-terminal-successor')
        self.assertEqual(successor['parent_frame_id'], 'resp-runtime-terminal-successor:frame-1')
        self.assertEqual(successor['parent_frame_sequence'], 1)
        self.assertEqual(successor['autonomy_level'], 'apply_safe')
        self.assertEqual(successor_graph['downstream_branch_ids'], ['repair-image'])
        self.assertEqual(successor_graph['output_obligations'][0]['status'], 'pending')
        self.assertEqual(successor_graph['output_obligations'][0]['phase_id'], 'repair-image')
        self.assertEqual(
            {
                item['patch_id']
                for item in diagnostics['graph_patch_successor_reopen_requests']
            },
            {item['patch_id'] for item in successor_requests},
        )

    def test_terminal_graph_patch_successor_queues_only_newly_applied_branch(self):
        owner = self._runtime_owner()
        graph = self._base_graph()
        graph.update(
            {
                'downstream_branch_ids': ['already-completed-image'],
                'downstream_branches': [
                    {
                        'phase_id': 'already-completed-image',
                        'branch_id': 'already-completed-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'fulfilled',
                    }
                ],
                'output_obligations': [
                    {
                        'obligation_id': 'obligation-already-completed-image',
                        'phase_id': 'already-completed-image',
                        'branch_id': 'already-completed-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'fulfilled',
                    }
                ],
            }
        )
        payload = {
            'id': 'resp-terminal-successor-queue',
            'response_id': 'resp-terminal-successor-queue',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'output_text': 'Frozen parent output.',
            'response_frame': {
                'response_id': 'resp-terminal-successor-queue',
                'frame_id': 'resp-terminal-successor-queue:frame-3',
                'frame_sequence': 3,
                'frame_relation': {'kind': 'late_fill_successor'},
            },
            'runtime': {
                'request_phase_graph': graph,
                'graph_closure_review': {
                    'kind': 'ollmo.graph_closure_review',
                    'status': 'repair_required',
                    'checks': [
                        {
                            'check_kind': 'materialization_contract',
                            'status': 'repair_required',
                            'repair_action': 'repair_missing_materialization_contract',
                            'phase_id': 'repair-image',
                            'branch_id': 'repair-image',
                            'evidence': 'missing materialization branch',
                        }
                    ],
                },
            },
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'pending',
                        'artifact_prompt': 'A bounded product-default successor image.',
                    }
                ],
                'completed_branches': [
                    {
                        'phase_id': 'already-completed-image',
                        'branch_id': 'already-completed-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'fulfilled',
                    }
                ],
            },
        }

        with patch.dict('os.environ', {}, clear=True):
            result = owner.prepare_terminal_graph_patch_successor(payload)

        self.assertEqual(result['status'], 'queued')
        self.assertEqual(result['autonomy']['autonomy_level'], 'apply_enforced')
        self.assertEqual(result['autonomy']['source'], 'product_default')
        self.assertEqual(result['artifact_gap']['pending_branch_scope'], 'exact_successor_reopen')
        self.assertEqual(
            [item['branch_id'] for item in result['artifact_gap']['pending_branches']],
            ['repair-image'],
        )
        candidate = result['successor_reopen_request']
        self.assertEqual(candidate['owed_branch_ids'], ['repair-image'])
        self.assertNotIn('already-completed-image', candidate['owed_branch_ids'])
        self.assertEqual(candidate['risk_level'], 'safe_additive')
        self.assertTrue(candidate['allowed_by_policy'])
        self.assertEqual(candidate['policy_mode'], 'safe_v1')
        self.assertTrue(candidate['enforced_policy_id'])
        self.assertTrue(candidate['successor_graph_digest'])
        successor_payload = result['response_payload']
        self.assertEqual(successor_payload['response_frame']['frame_id'], 'resp-terminal-successor-queue:frame-3')
        self.assertEqual(successor_payload['frame_relation']['kind'], 'graph_patch_reopen_successor')
        self.assertEqual(
            successor_payload['frame_relation']['parent_frame_id'],
            'resp-terminal-successor-queue:frame-3',
        )
        self.assertEqual(successor_payload['lifecycle_state'], 'late_fill_pending')
        self.assertEqual(
            [item['branch_id'] for item in successor_payload['late_fill']['pending_branches']],
            ['repair-image'],
        )
        self.assertEqual(
            [item['branch_id'] for item in successor_payload['late_fill']['completed_branches']],
            ['already-completed-image'],
        )
        with patch.dict('os.environ', {}, clear=True):
            repeated = owner.prepare_terminal_graph_patch_successor(successor_payload)
        self.assertNotEqual(repeated['status'], 'queued')
        self.assertEqual(
            successor_payload['runtime']['request_phase_graph']
            ['successor_reopen_requests'][0]['status'],
            'applied_to_successor',
        )

    def test_terminal_graph_patch_successor_explicit_off_fails_closed(self):
        owner = self._runtime_owner()
        payload = {
            'id': 'resp-terminal-successor-off',
            'response_id': 'resp-terminal-successor-off',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'response_frame': {
                'response_id': 'resp-terminal-successor-off',
                'frame_id': 'resp-terminal-successor-off:frame-1',
                'frame_sequence': 1,
            },
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }

        result = owner.prepare_terminal_graph_patch_successor(
            payload,
            graph_repair_autonomy='off',
        )

        self.assertEqual(result['status'], 'not_applicable')
        self.assertEqual(result['reason'], 'graph_repair_autonomy_off')
        self.assertNotIn('artifact_gap', result)

        with patch.dict(
            'os.environ',
            {'OLLMO_GRAPH_REPAIR_AUTONOMY': 'off'},
            clear=True,
        ):
            environment_off = owner.prepare_terminal_graph_patch_successor(payload)
        self.assertEqual(environment_off['status'], 'not_applicable')
        self.assertEqual(environment_off['reason'], 'graph_repair_autonomy_off')

        with patch.dict(
            'os.environ',
            {'OLLMO_GRAPH_REPAIR_AUTONOMY': 'invalid-autonomy'},
            clear=True,
        ):
            invalid_autonomy = owner.prepare_terminal_graph_patch_successor(payload)
        self.assertEqual(invalid_autonomy['status'], 'not_applicable')
        self.assertEqual(invalid_autonomy['reason'], 'graph_repair_autonomy_off')

        with patch.dict(
            'os.environ',
            {'OLLMO_APPLY_ENFORCED_POLICY': 'off'},
            clear=True,
        ):
            enforced_policy_off = owner.prepare_terminal_graph_patch_successor(payload)
        self.assertNotEqual(enforced_policy_off['status'], 'queued')
        self.assertNotIn('artifact_gap', enforced_policy_off)

        with patch.dict(
            'os.environ',
            {
                'OLLMO_GRAPH_REPAIR_AUTONOMY': 'apply_safe',
                'OLLMO_APPLY_ENFORCED_POLICY': 'off',
            },
            clear=True,
        ):
            explicit_apply_safe = owner.prepare_terminal_graph_patch_successor(payload)
        self.assertEqual(explicit_apply_safe['status'], 'queued')

    def test_terminal_graph_patch_successor_rejects_tampered_successor_graph(self):
        owner = self._runtime_owner()
        payload = {
            'id': 'resp-terminal-successor-tamper',
            'response_id': 'resp-terminal-successor-tamper',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'response_frame': {
                'response_id': 'resp-terminal-successor-tamper',
                'frame_id': 'resp-terminal-successor-tamper:frame-2',
                'frame_sequence': 2,
            },
            'runtime': {'request_phase_graph': self._base_graph()},
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                    }
                ],
            },
        }
        reviewed = owner._attach_graph_patch_lifecycle(
            owner._attach_runtime_graph_repair_evidence(payload),
            graph_repair_autonomy='apply_safe',
        )
        candidate = dict(
            reviewed['runtime']['request_phase_graph']['successor_reopen_requests'][0]
        )
        tampered_graph = dict(candidate['successor_request_phase_graph'])
        tampered_graph['continuation_required'] = False
        candidate['successor_request_phase_graph'] = tampered_graph

        result = owner._prepare_graph_patch_successor_reopen(reviewed, candidate)

        self.assertEqual(result['status'], 'blocked')
        self.assertIn('successor_graph_digest_mismatch', result['blocked_reasons'])

    def test_terminal_graph_patch_successor_rejects_scope_widening_and_tampered_keys(self):
        owner, reviewed, candidate = self._terminal_successor_sink_fixture(
            include_unrelated_pending=True,
        )
        widened = copy.deepcopy(candidate)
        widened['owed_branch_ids'] = ['repair-image', 'unrelated-image']

        widened_result = owner._prepare_graph_patch_successor_reopen(reviewed, widened)

        self.assertEqual(widened_result['status'], 'blocked')
        self.assertIn(
            'successor_owed_branch_scope_mismatch',
            widened_result['blocked_reasons'],
        )

        for field, blocked_reason in (
            ('successor_execution_key', 'successor_execution_key_mismatch'),
            ('successor_reopen_key', 'successor_reopen_key_mismatch'),
        ):
            with self.subTest(field=field):
                owner, reviewed, candidate = self._terminal_successor_sink_fixture()
                candidate[field] = f'tampered-{field}'

                result = owner._prepare_graph_patch_successor_reopen(reviewed, candidate)

                self.assertEqual(result['status'], 'blocked')
                self.assertIn(blocked_reason, result['blocked_reasons'])

    def test_terminal_graph_patch_successor_depth_is_exactly_parent_derived(self):
        owner, reviewed, candidate = self._terminal_successor_sink_fixture(parent_depth=6)
        self.assertEqual(candidate['successor_reopen_depth'], 7)
        exhausted = owner._prepare_graph_patch_successor_reopen(reviewed, candidate)
        self.assertEqual(exhausted['status'], 'blocked')
        self.assertIn('successor_reopen_depth_exhausted', exhausted['blocked_reasons'])

        bypass = copy.deepcopy(candidate)
        bypass['successor_reopen_depth'] = 1

        bypass_result = owner._prepare_graph_patch_successor_reopen(reviewed, bypass)

        self.assertEqual(bypass_result['status'], 'blocked')
        self.assertIn(
            'successor_reopen_depth_parent_mismatch',
            bypass_result['blocked_reasons'],
        )

    def test_terminal_graph_patch_successor_requires_exact_identity_and_execution_contract(self):
        identity_cases = (
            ('proposal_id', 'different-proposal', 'successor_proposal_id_lifecycle_mismatch'),
            ('patch_id', '', 'successor_patch_id_missing'),
            ('idempotency_key', '', 'successor_idempotency_key_missing'),
        )
        for field, tampered_value, blocked_reason in identity_cases:
            with self.subTest(field=field):
                owner, reviewed, candidate = self._terminal_successor_sink_fixture()
                candidate[field] = tampered_value
                candidate['patch_application'][field] = tampered_value

                result = owner._prepare_graph_patch_successor_reopen(reviewed, candidate)

                self.assertEqual(result['status'], 'blocked')
                self.assertIn(blocked_reason, result['blocked_reasons'])

        owner, reviewed, candidate = self._terminal_successor_sink_fixture()
        candidate['execution_contract'] = {}

        contract_result = owner._prepare_graph_patch_successor_reopen(reviewed, candidate)

        self.assertEqual(contract_result['status'], 'blocked')
        self.assertIn(
            'successor_execution_contract_mismatch',
            contract_result['blocked_reasons'],
        )

    def test_backend_runtime_records_advisory_surface_actionability_without_monitor_proposal(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-backend-advisory-surface',
            'lifecycle_state': 'completed',
            'runtime': {
                'request_phase_graph': self._base_graph(),
                'graph_closure_review': {
                    'status': 'fulfilled',
                    'surface_state': {
                        'kind': 'ollmo.surface_state',
                        'state': 'pending',
                        'category_counts': {
                            'controlled_attention_advisory': 1,
                            'aspiration_advisory': 1,
                            'commitment_advisory': 1,
                            'reconsiderable': 1,
                        },
                        'items': [
                            {'category': 'controlled_attention_advisory', 'status': 'active', 'phase_id': 'phase-1'},
                            {'category': 'aspiration_advisory', 'status': 'active', 'branch_id': 'branch-1'},
                            {'category': 'commitment_advisory', 'status': 'active', 'slot_id': 'slot-1'},
                            {'category': 'reconsiderable', 'status': 'active', 'wave_id': 'wave-1'},
                        ],
                    },
                },
            },
            'late_fill': {
                'status': 'completed',
                'final_materialization_contract_status': 'fulfilled',
                'materialization_contract_unmet': False,
            },
        }

        updated = owner._attach_runtime_graph_repair_evidence(payload)

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(diagnostics['surface_repair_actionability']['status'], 'advisory')
        self.assertEqual(diagnostics['runtime_graph_repair_proposals'], [])
        self.assertEqual(diagnostics['runtime_graph_repair_proposal_reviews'], [])
        self.assertFalse([
            proposal for proposal in graph.get('graph_repair_proposals', [])
            if proposal.get('repair_type') == 'reconcile_surface_state_or_reopen_contract'
        ])

    def test_backend_runtime_records_actionable_surface_repair_truth_without_monitor_dependency(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-backend-actionable-surface',
            'lifecycle_state': 'completed',
            'runtime': {
                'request_phase_graph': self._base_graph(),
                'graph_closure_review': {
                    'status': 'fulfilled',
                    'surface_state': {
                        'kind': 'ollmo.surface_state',
                        'state': 'review_pending',
                        'category_counts': {'semantic_review_pending': 1},
                        'items': [
                            {
                                'category': 'semantic_review_pending',
                                'status': 'pending',
                                'check_kind': 'branch_semantic_review',
                                'branch_id': 'branch-final-review',
                                'reason': 'promoted branch semantic review remains pending',
                            }
                        ],
                    },
                },
            },
            'late_fill': {
                'status': 'completed',
                'final_materialization_contract_status': 'fulfilled',
                'materialization_contract_unmet': False,
            },
        }

        updated = owner._attach_runtime_graph_repair_evidence(payload)

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        proposals = graph['graph_repair_proposals']
        reviews = graph['graph_repair_reviews']
        reconcile = next(
            proposal for proposal in proposals
            if proposal['repair_type'] == 'reconcile_surface_state_or_reopen_contract'
        )
        review = next(item for item in reviews if item['proposal_id'] == reconcile['proposal_id'])

        self.assertEqual(diagnostics['surface_repair_actionability']['status'], 'actionable')
        self.assertEqual(
            reconcile['runtime_monitor_evidence']['surface_actionability']['status'],
            'actionable',
        )
        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(diagnostics['runtime_graph_repair_proposals'][0]['proposal_id'], reconcile['proposal_id'])
        self.assertEqual(
            diagnostics['runtime_graph_repair_proposal_reviews'][0]['proposal_id'],
            reconcile['proposal_id'],
        )

    def test_runtime_monitor_evidence_maps_latest_failure_classes_to_proposals(self):
        graph = self._base_graph()
        closure_review = {'status': 'blocked'}
        late_fill = {
            'status': 'partial_failed',
            'final_materialization_contract_status': 'unmet',
            'materialization_contract_unmet': True,
            'pending_branches': [
                {
                    'phase_id': 'branch-text-artifact-1',
                    'branch_id': 'branch-text-artifact-1',
                    'capability': 'chat',
                    'output_type': 'text',
                }
            ],
        }
        monitor_report = {
            'response_id': 'resp-runtime-evidence',
            'lifecycle_state': 'repair_needed',
            'materialization_contract_unmet': True,
            'final_materialization_contract_status': 'unmet',
            'branch_counts': {'pending': 1, 'failed': 0, 'completed': 2},
            'artifacts': {
                'duplicate_artifact_refs': ['artifact:image_duplicate'],
                'missing_files': ['/Users/example/Projects/ollmo/artifacts/images/missing.png'],
                'html_image_links': [
                    {'src': 'artifact://fake-image-ref', 'exists': False},
                ],
                'registry_records': [
                    {'artifact_ref': 'artifact://fake-image-ref', 'kind': 'image'},
                ],
            },
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report=monitor_report,
        )
        repair_types = {proposal['repair_type'] for proposal in proposals}

        self.assertIn('repair_missing_materialization_contract', repair_types)
        self.assertIn('resume_or_repair_pending_branch', repair_types)
        self.assertIn('repair_artifact_ref_identity', repair_types)
        self.assertIn('rebind_artifact_dependency', repair_types)

        reviews = [
            validate_graph_repair_proposal(
                proposal,
                request_phase_graph=graph,
                closure_review=closure_review,
                promotion_review={},
            )
            for proposal in proposals
        ]
        self.assertTrue(reviews)
        self.assertTrue(all(review['status'] == 'accepted' for review in reviews))

    def test_runtime_monitor_evidence_maps_surface_mismatch_to_reconcile_proposal(self):
        graph = self._base_graph()
        closure_review = {
            'status': 'fulfilled',
            'surface_state': {
                'state': 'blocked',
                'category_counts': {'blocked': 1},
                'reason': 'surface still has blocked work',
            },
        }
        late_fill = {
            'status': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }
        monitor_report = {
            'response_id': 'resp-surface-mismatch',
            'lifecycle_state': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report=monitor_report,
        )

        proposal = next(
            item for item in proposals
            if item['repair_type'] == 'reconcile_surface_state_or_reopen_contract'
        )
        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=closure_review,
            promotion_review={},
        )
        first = apply_validated_graph_repair_patch(graph, review)
        second = apply_validated_graph_repair_patch(first['graph'], review)

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(first['status'], 'applied')
        self.assertEqual(second['status'], 'already_applied')

    def test_advisory_only_pending_surface_does_not_create_reconcile_repair(self):
        graph = self._base_graph()
        closure_review = {
            'status': 'fulfilled',
            'surface_state': {
                'kind': 'ollmo.surface_state',
                'state': 'pending',
                'category_counts': {
                    'controlled_attention_advisory': 4,
                    'aspiration_advisory': 2,
                    'commitment_advisory': 2,
                    'reconsiderable': 1,
                    'open': 1,
                    'completed': 3,
                },
                'items': [
                    {
                        'category': 'controlled_attention_advisory',
                        'status': 'active',
                        'review_type': 'block_resolution',
                    },
                    {
                        'category': 'aspiration_advisory',
                        'status': 'active',
                        'action': 'keep_possibility_visible',
                    },
                    {
                        'category': 'commitment_advisory',
                        'status': 'active',
                        'action': 'choose_right_sized_transition',
                    },
                    {
                        'category': 'open',
                        'status': 'pending',
                        'check_kind': 'controlled_attention_review',
                        'reason': 'attention frame is still visible but not promoted owed work',
                    },
                ],
                'reason': 'advisory movement frames remain visible',
            },
        }
        late_fill = {
            'status': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }
        monitor_report = {
            'response_id': 'resp-advisory-only-surface',
            'lifecycle_state': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report=monitor_report,
            accepted_learning_hints={
                'enabled': True,
                'authority': 'soft_hint',
                'runtime_effect': 'soft_hints_available',
                'hint_count': 1,
            },
        )

        self.assertNotIn(
            'reconcile_surface_state_or_reopen_contract',
            {proposal['repair_type'] for proposal in proposals},
        )

    def test_resolved_repair_metadata_surface_does_not_create_reconcile_repair(self):
        graph = self._base_graph()
        closure_review = {
            'status': 'fulfilled',
            'checks': [
                {
                    'status': 'fulfilled',
                    'check_kind': 'output_obligation',
                    'branch_id': 'branch-text_artifact-1',
                    'repair_action': 'manual_review',
                },
                {
                    'status': 'fulfilled',
                    'check_kind': 'output_obligation',
                    'branch_id': 'branch-text_artifact-2',
                    'repair_action': 'manual_review',
                },
            ],
            'surface_state': {
                'kind': 'ollmo.surface_state',
                'state': 'pending',
                'category_counts': {
                    'completed': 2,
                    'repair_pending': 2,
                },
                'items': [
                    {
                        'category': 'repair_pending',
                        'status': 'fulfilled',
                        'check_kind': 'output_obligation',
                        'branch_id': 'branch-text_artifact-1',
                        'repair_action': 'manual_review',
                    },
                    {
                        'category': 'repair_pending',
                        'status': 'fulfilled',
                        'check_kind': 'output_obligation',
                        'branch_id': 'branch-text_artifact-2',
                        'repair_action': 'manual_review',
                    },
                ],
                'reason': 'stale fulfilled repair metadata remained visible',
            },
        }
        late_fill = {
            'status': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report={'response_id': 'resp-resolved-repair-metadata-surface'},
        )

        self.assertNotIn(
            'reconcile_surface_state_or_reopen_contract',
            {proposal['repair_type'] for proposal in proposals},
        )

    def test_actionable_semantic_review_surface_still_creates_reconcile_repair(self):
        graph = self._base_graph()
        closure_review = {
            'status': 'fulfilled',
            'surface_state': {
                'kind': 'ollmo.surface_state',
                'state': 'review_pending',
                'category_counts': {
                    'semantic_review_pending': 1,
                    'controlled_attention_advisory': 2,
                },
                'items': [
                    {
                        'category': 'semantic_review_pending',
                        'status': 'pending',
                        'check_kind': 'branch_semantic_review',
                        'branch_id': 'branch-final-review',
                        'reason': 'promoted branch semantic review remains pending',
                    }
                ],
            },
        }
        late_fill = {
            'status': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report={'response_id': 'resp-actionable-semantic-surface'},
        )

        proposal = next(
            item for item in proposals
            if item['repair_type'] == 'reconcile_surface_state_or_reopen_contract'
        )
        self.assertEqual(
            proposal['runtime_monitor_evidence']['surface_actionability']['status'],
            'actionable',
        )

    def test_branch_wave_phase_slot_advisory_surface_stays_non_actionable(self):
        graph = self._base_graph()
        closure_review = {
            'status': 'fulfilled',
            'surface_state': {
                'kind': 'ollmo.surface_state',
                'state': 'pending',
                'category_counts': {
                    'controlled_attention_advisory': 1,
                    'aspiration_advisory': 1,
                    'commitment_advisory': 1,
                    'reconsiderable': 1,
                },
                'items': [
                    {'category': 'controlled_attention_advisory', 'status': 'active', 'phase_id': 'phase-1'},
                    {'category': 'aspiration_advisory', 'status': 'active', 'branch_id': 'branch-1'},
                    {'category': 'commitment_advisory', 'status': 'active', 'slot_id': 'slot-1'},
                    {'category': 'reconsiderable', 'status': 'active', 'wave_id': 'wave-1'},
                ],
            },
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill={'final_materialization_contract_status': 'fulfilled'},
            monitor_report={'response_id': 'resp-advisory-scale-surface'},
        )

        self.assertFalse([
            proposal for proposal in proposals
            if proposal['repair_type'] == 'reconcile_surface_state_or_reopen_contract'
        ])

    def test_fulfilled_contract_with_stale_open_surface_items_stays_advisory(self):
        graph = self._base_graph()
        closure_review = {
            'status': 'fulfilled',
            'surface_state': {
                'kind': 'ollmo.surface_state',
                'state': 'pending',
                'category_counts': {
                    'open': 4,
                    'completed': 20,
                    'controlled_attention_advisory': 4,
                },
                'items': [
                    {
                        'category': 'open',
                        'status': 'pending',
                        'branch_id': 'branch-image_generation-1',
                        'phase_id': 'phase-image_generation-1',
                        'reason': 'stale advisory movement item carried branch identity after fulfilled closure',
                    },
                    {
                        'category': 'open',
                        'status': 'active',
                        'slot_id': 'output-phase-1',
                        'reason': 'stale surface item is visible but has no promoted repair/check kind',
                    },
                    {
                        'category': 'controlled_attention_advisory',
                        'status': 'active',
                        'branch_id': 'branch-final-review',
                    },
                ],
            },
        }
        late_fill = {
            'status': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
        }

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report={'response_id': 'resp-stale-open-surface-items'},
        )

        self.assertNotIn(
            'reconcile_surface_state_or_reopen_contract',
            {proposal['repair_type'] for proposal in proposals},
        )

    def test_monitor_rendering_pairs_reviews_by_proposal_id_and_marks_unmatched(self):
        lines = _render_learning_healing_lines(
            {
                'graph_repair': {
                    'proposal_count': 2,
                    'review_count': 2,
                    'proposal_review_pairs': [
                        {
                            'proposal': {
                                'proposal_id': 'proposal-a',
                                'status': 'proposed',
                                'repair_type': 'repair_missing_materialization_contract',
                                'repair_gap_code': 'repair_missing_materialization_contract',
                            },
                            'reviews': [
                                {
                                    'review_id': 'review-a',
                                    'proposal_id': 'proposal-a',
                                    'status': 'accepted',
                                }
                            ],
                        },
                        {
                            'proposal': {
                                'proposal_id': 'proposal-b',
                                'status': 'proposed',
                                'repair_type': 'reconcile_surface_state_or_reopen_contract',
                                'repair_gap_code': 'reconcile_surface_state_or_reopen_contract',
                            },
                            'reviews': [],
                        },
                    ],
                    'unmatched_reviews': [
                        {
                            'review_id': 'review-orphan',
                            'proposal_id': 'proposal-orphan',
                            'status': 'rejected',
                            'reasons': 'deferred_or_reserved_intent_conflict',
                        }
                    ],
                }
            }
        )
        rendered = '\n'.join(lines)

        self.assertIn('matched review: id review-a; proposal proposal-a; status accepted.', rendered)
        self.assertIn(
            'unmatched review: id review-orphan; proposal proposal-orphan; status rejected;',
            rendered,
        )
        self.assertNotIn('  review: id review-orphan', rendered)

    def test_monitor_summary_synthesizes_repair_proposals_and_keeps_weak_freeze_visible(self):
        graph = self._base_graph()
        payload = {
            'response_id': 'resp-monitor-summary',
            'lifecycle_state': 'completed',
            'runtime': {
                'request_phase_graph': graph,
                'graph_closure_review': {
                    'status': 'blocked',
                    'surface_state': {'state': 'blocked', 'category_counts': {'blocked': 1}},
                },
            },
            'late_fill': {
                'status': 'partial_failed',
                'final_materialization_contract_status': 'unmet',
                'materialization_contract_unmet': True,
                'pending_branches': [
                    {
                        'phase_id': 'branch-text-artifact-1',
                        'branch_id': 'branch-text-artifact-1',
                        'capability': 'chat',
                        'output_type': 'text',
                    }
                ],
            },
        }

        info = _collect_learning_healing(
            Path('/Users/example/Projects/ollmo'),
            payload,
            {},
            lambda ref, **kwargs: {},
        )

        self.assertGreaterEqual(info['graph_repair']['proposal_count'], 2)
        self.assertTrue(info['freeze_guard']['repair_needed_visible'])
        self.assertTrue(info['freeze_guard']['weak_freeze_watch'])

    def test_monitor_summary_surfaces_apply_enforced_runtime_truth(self):
        payload = {
            'response_id': 'resp-monitor-enforced-authority',
            'runtime': {
                'accepted_learning_hints': {
                    'status': 'active',
                    'authority': 'soft_hint',
                    'runtime_effect': 'soft_hint_only',
                    'hint_count': 1,
                    'hints': [],
                },
                'developer_diagnostics': {
                    'enforced_policy': {
                        'mode': 'safe_v1',
                        'enabled': True,
                        'default_action': 'deny',
                    },
                    'graph_patch_autonomy': {
                        'autonomy_level': 'apply_enforced',
                        'source': 'environment',
                        'terminal_apply_blocked': True,
                    },
                    'graph_rebase_autonomy': {
                        'autonomy_level': 'apply_enforced',
                        'source': 'environment',
                    },
                    'graph_patch_enforced_policy_reviews': [
                        {
                            'review_id': 'patch-review-allowed',
                            'status': 'allowed',
                            'allowed': True,
                            'authority': 'runtime_enforced_policy',
                            'enforced_class': 'safe_additive_missing_branch',
                            'blocked_reasons': [],
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                        },
                        {
                            'review_id': 'patch-review-blocked',
                            'status': 'blocked',
                            'allowed': False,
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'safe_additive_dependency_repair',
                            'blocked_reasons': [
                                'forbidden_evidence_not_enforced_authority',
                                'current_runtime_evidence_required',
                            ],
                            'forbidden_evidence_seen': [
                                'accepted_learning',
                                'degraded_liveness_only',
                                'frontend',
                                'monitor_only',
                            ],
                            'current_evidence_refs': [
                                'accepted_learning:case-1',
                                'runtime:degraded_liveness_only',
                                'frontend:ui_label',
                                'monitor_only:summary',
                            ],
                        },
                    ],
                    'graph_rebase_enforced_policy_reviews': [
                        {
                            'review_id': 'rebase-review-full',
                            'status': 'blocked',
                            'allowed': False,
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'full_successor_rebase',
                            'blocked_reasons': ['full_successor_rebase_not_enforced_v1'],
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                        },
                        {
                            'review_id': 'rebase-review-partial',
                            'status': 'blocked',
                            'allowed': False,
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'partial_subtree_rebase_strict',
                            'blocked_reasons': ['partial_subtree_rebase_enforced_v1_audit_only'],
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                        },
                    ],
                    'redraw_scope_ladder_review': {
                        'review_id': 'redraw-selected',
                        'status': 'selected',
                        'selected_scope': 'add_missing_branch',
                        'selected_candidate': {'scope': 'add_missing_branch'},
                        'blocked_reasons': [],
                    },
                    'redraw_scope_ladder_reviews': [
                        {
                            'review_id': 'redraw-blocked',
                            'status': 'blocked',
                            'selected_scope': 'observe',
                            'selected_candidate': {'scope': 'observe'},
                            'blocked_reasons': [
                                'current_runtime_evidence_required',
                                'degraded_or_provider_signal_not_scope_authority',
                            ],
                        }
                    ],
                    'graph_patch_successor_reopen_requests': [
                        {
                            'patch_id': 'patch-open',
                            'proposal_id': 'proposal-open',
                            'idempotency_key': 'idem-open',
                            'runtime_effect': 'successor_reopen_required',
                        }
                    ],
                    'successor_rebase_requests': [
                        {
                            'rebase_id': 'rebase-open',
                            'proposal_id': 'proposal-rebase-open',
                            'idempotency_key': 'idem-rebase-open',
                            'runtime_effect': 'successor_rebase_created',
                        }
                    ],
                },
                'request_phase_graph': {
                    'graph_patch_lifecycle': [
                        {
                            'patch_id': 'patch-open',
                            'proposal_id': 'proposal-open',
                            'idempotency_key': 'idem-open',
                            'status': 'blocked',
                            'autonomy_level': 'apply_enforced',
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'safe_additive_dependency_repair',
                            'blocked_reasons': [
                                'forbidden_evidence_not_enforced_authority',
                                'current_runtime_evidence_required',
                                'terminal_frame_requires_successor_reopen',
                            ],
                            'source_evidence_refs': ['monitor_only:summary'],
                            'outcome': {'runtime_effect': 'terminal_frame_not_mutated'},
                        }
                    ],
                    'graph_rebase_lifecycle': [
                        {
                            'rebase_id': 'rebase-full',
                            'proposal_id': 'proposal-rebase-full',
                            'idempotency_key': 'idem-rebase-full',
                            'status': 'blocked',
                            'autonomy_level': 'apply_enforced',
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'full_successor_rebase',
                            'blocked_reasons': ['full_successor_rebase_not_enforced_v1'],
                        },
                        {
                            'rebase_id': 'rebase-partial',
                            'proposal_id': 'proposal-rebase-partial',
                            'idempotency_key': 'idem-rebase-partial',
                            'status': 'blocked',
                            'autonomy_level': 'apply_enforced',
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'partial_subtree_rebase_strict',
                            'blocked_reasons': ['partial_subtree_rebase_enforced_v1_audit_only'],
                        },
                    ],
                    'successor_reopen_requests': [
                        {
                            'patch_id': 'patch-open',
                            'proposal_id': 'proposal-open',
                            'idempotency_key': 'idem-open',
                            'runtime_effect': 'successor_reopen_required',
                        }
                    ],
                    'successor_rebase_requests': [
                        {
                            'rebase_id': 'rebase-open',
                            'proposal_id': 'proposal-rebase-open',
                            'idempotency_key': 'idem-rebase-open',
                            'runtime_effect': 'successor_rebase_created',
                        }
                    ],
                    'redraw_scope_ladder_review': {
                        'review_id': 'redraw-selected',
                        'status': 'selected',
                        'selected_scope': 'add_missing_branch',
                        'selected_candidate': {'scope': 'add_missing_branch'},
                        'blocked_reasons': [],
                    },
                },
            },
        }

        info = _collect_learning_healing(
            Path('/Users/example/Projects/ollmo'),
            payload,
            {},
            lambda ref, **kwargs: {},
        )
        rendered = '\n'.join(_render_learning_healing_lines(info))

        self.assertIn('Enforced policy: mode safe_v1, enabled true, default_action deny.', rendered)
        self.assertIn('Graph repair enforcement: autonomy apply_enforced, allowed 1, blocked 1', rendered)
        self.assertIn(
            'Graph rebase enforcement: autonomy apply_enforced, allowed 0, blocked 2, full successor rebase blocked yes, partial subtree rebase audit-only/blocked yes',
            rendered,
        )
        self.assertIn('Successor/reopen: 1 reopen, 1 rebase, parent frozen/unmutated true.', rendered)
        self.assertIn('Redraw scope ladder: selected add_missing_branch, observe', rendered)
        self.assertIn('Forbidden evidence rejected: learning_only, degraded/provider, frontend, monitor_only.', rendered)
        self.assertIn('Accepted learning remains soft_hint_only.', rendered)

    def test_monitor_summary_keeps_policy_off_neutral(self):
        payload = {
            'response_id': 'resp-monitor-enforced-off',
            'runtime': {
                'accepted_learning_hints': {
                    'status': 'active',
                    'authority': 'soft_hint',
                    'runtime_effect': 'soft_hint_only',
                    'hint_count': 0,
                    'hints': [],
                },
                'developer_diagnostics': {
                    'enforced_policy': {
                        'mode': 'off',
                        'enabled': False,
                        'default_action': 'deny',
                    },
                    'graph_patch_autonomy': {'autonomy_level': 'apply_enforced'},
                    'graph_patch_enforced_policy_reviews': [
                        {
                            'review_id': 'patch-review-off',
                            'status': 'blocked',
                            'allowed': False,
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'safe_additive_missing_branch',
                            'blocked_reasons': ['enforced_policy_off'],
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                        }
                    ],
                },
                'request_phase_graph': {
                    'graph_patch_lifecycle': [
                        {
                            'patch_id': 'patch-off',
                            'proposal_id': 'proposal-off',
                            'idempotency_key': 'idem-off',
                            'status': 'blocked',
                            'autonomy_level': 'apply_enforced',
                            'authority': 'runtime_enforced_policy_denied',
                            'enforced_class': 'safe_additive_missing_branch',
                            'blocked_reasons': ['enforced_policy_off'],
                        }
                    ]
                },
            },
        }

        info = _collect_learning_healing(
            Path('/Users/example/Projects/ollmo'),
            payload,
            {},
            lambda ref, **kwargs: {},
        )
        rendered = '\n'.join(_render_learning_healing_lines(info))

        self.assertIn('Enforced policy: mode off, enabled false, default_action deny.', rendered)
        self.assertIn('Graph repair enforcement: autonomy apply_enforced, allowed 0, blocked 1', rendered)
        self.assertNotIn('needs attention', rendered)

    def test_monitor_artifact_identity_summary_classifies_duplicate_sources(self):
        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / 'voice.wav'
            audio_path.write_bytes(b'RIFFtestWAVE')
            other_path = Path(tmp) / 'other.png'
            other_path.write_bytes(b'png')

            checks = _artifact_checks(
                {
                    'artifacts': [
                        {
                            'artifact_ref': 'artifact:final-audio',
                            'path': str(audio_path),
                            'kind': 'audio',
                        },
                        {
                            'artifact_ref': 'artifact:final-audio',
                            'path': str(audio_path),
                            'kind': 'audio',
                        },
                    ],
                    'outputs': [
                        {
                            'kind': 'audio',
                            'artifact_ref': 'artifact:output-audio',
                            'path': str(audio_path),
                        },
                        {
                            'kind': 'audio',
                            'artifact_ref': 'artifact:output-audio',
                            'path': str(audio_path),
                        },
                    ],
                },
                [
                    (
                        1,
                        {
                            'artifact': {
                                'artifact_ref': 'artifact:registry-conflict',
                                'path': str(audio_path),
                                'kind': 'audio',
                            }
                        },
                        1,
                    ),
                    (
                        2,
                        {
                            'artifact': {
                                'artifact_ref': 'artifact:registry-conflict',
                                'path': str(other_path),
                                'kind': 'image',
                            }
                        },
                        1,
                    ),
                ],
                None,
            )

        summary = checks['duplicate_ref_summary']

        self.assertEqual(summary['final_projection'][0]['classification'], 'canonicalizable')
        self.assertEqual(summary['final_projection'][0]['reason'], 'same ref, same path/type')
        self.assertEqual(summary['output'][0]['classification'], 'canonicalizable')
        self.assertEqual(summary['registry'][0]['classification'], 'conflict')
        self.assertEqual(summary['registry'][0]['reason'], 'same ref, different path/type')
        self.assertEqual(checks['audio_artifact_count'], 1)

    def test_monitor_report_renders_audio_and_duplicate_identity_wording(self):
        report = {
            'verdict': 'needs_attention',
            'response_id': 'resp-monitor-audio',
            'frame_sequence': 2,
            'lifecycle_state': 'repair_needed',
            'late_fill_status': 'partial_failed',
            'final_materialization_contract_status': 'unmet',
            'materialization_contract_unmet': True,
            'failed_branch_count': 0,
            'branch_counts': {'pending': 0, 'active': 0, 'failed': 0, 'completed': 1},
            'timing': {
                'start': None,
                'initial_chat_finished': None,
                'initial_chat_seconds': None,
                'image_wave_start': None,
                'image_wave_end': None,
                'image_wave_seconds': None,
                'terminal_output': None,
                'terminal_seconds': None,
                'hygiene_finished': None,
                'hygiene_seconds': None,
            },
            'sizes': {
                'state_total_bytes': 100,
                'state_add_approx_bytes': 10,
                'snapshots_total_bytes': 100,
                'snapshots_add_bytes': 10,
                'artifacts_total_bytes': 100,
                'artifacts_add_bytes': 10,
                'latest_response_line_bytes': 10,
                'new_snapshot_file_count': 1,
                'new_artifact_file_count': 1,
            },
            'artifacts': {
                'missing_files': [],
                'sha_mismatches': [],
                'html_issues': [],
                'css_issues': [],
                'html_image_links': [],
                'weak_viewport_tags': [],
                'audio_artifact_count': 1,
                'artifact_kind_counts': {'audio': 1},
                'artifact_file_count_by_suffix': {'wav': 1},
                'output_ref_counts': {'audio': 2},
                'duplicate_ref_summary': {
                    'final_projection': [
                        {
                            'artifact_ref': 'artifact:audio-a',
                            'count': 2,
                            'classification': 'canonicalizable',
                            'reason': 'same ref, same path/type',
                            'paths': ['/tmp/audio.wav'],
                            'kinds': ['audio'],
                            'sources': ['final_projection'],
                        }
                    ],
                    'registry': [
                        {
                            'artifact_ref': 'artifact:conflict',
                            'count': 2,
                            'classification': 'conflict',
                            'reason': 'same ref, different path/type',
                            'paths': ['/tmp/audio.wav', '/tmp/image.png'],
                            'kinds': ['audio', 'image'],
                            'sources': ['registry'],
                        }
                    ],
                    'output': [
                        {
                            'artifact_ref': 'artifact:output-audio',
                            'count': 2,
                            'classification': 'canonicalizable',
                            'reason': 'same ref, same path/type',
                            'paths': ['/tmp/audio.wav'],
                            'kinds': ['audio'],
                            'sources': ['output'],
                        }
                    ],
                    'canonicalizable': [],
                    'conflicts': [],
                },
            },
            'registry_parse_clean': True,
            'image_models': {},
            'image_branch_count': 0,
            'branch_role_counts': {'text_to_speech': 1},
            'branch_capability_counts': {'text_to_speech': 1},
            'output_obligation_counts': {'audio': 1},
            'text_artifact_branch_count': 0,
            'learning_healing': {},
            'notes': ['No repair/requeue/materialization failure events observed.'],
            'timing_diagnostics': {'phase_gaps': {}, 'scheduling_policy': {}, 'waves': [], 'branches': []},
        }

        rendered = _render_human(report)

        self.assertIn('Branch roles: text_to_speech=1.', rendered)
        self.assertIn('Output obligations: audio=1.', rendered)
        self.assertIn('Artifact kinds: audio=1.', rendered)
        self.assertIn('Artifact file suffixes: wav=1.', rendered)
        self.assertIn('Audio artifacts: 1 saved.', rendered)
        self.assertIn('HTML image links: not applicable.', rendered)
        self.assertIn('duplicate final projection yes; duplicate registry ref yes; duplicate output ref yes', rendered)
        self.assertIn('Artifact identity canonicalizable: artifact:audio-a', rendered)
        self.assertIn('Artifact identity conflicts: artifact:conflict', rendered)

    def test_monitor_evidence_source_without_concrete_evidence_is_rejected(self):
        graph = self._base_graph()
        proposal = self._proposal(
            source='monitor_evidence',
            evidence_refs=[],
            patch={
                'add_phases': [
                    {
                        'phase_id': 'repair-monitor',
                        'branch_id': 'repair-monitor',
                        'capability': 'chat',
                        'output_type': 'text',
                    }
                ]
            },
        )

        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review={},
            promotion_review={},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('runtime_evidence_missing', review['reasons'])


if __name__ == '__main__':
    unittest.main()
