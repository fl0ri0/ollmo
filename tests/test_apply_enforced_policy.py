import copy
import os
import unittest
from unittest.mock import patch

from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_services.enforced_policy import (
    ENFORCED_POLICY_REVIEW_KIND,
    build_enforced_policy_review,
    describe_enforced_policy,
    describe_enforced_policy_from_env,
    enforced_policy_allows_application,
    normalize_enforced_policy_mode,
)
from ollmo_services.graph_rebase import (
    build_graph_rebase_lifecycle,
    build_graph_rebase_proposal,
    validate_graph_rebase_proposal,
)
from ollmo_services.graph_repair import (
    GRAPH_REPAIR_PROPOSAL_KIND,
    PROPOSAL_ALLOWED_USE,
    PROPOSAL_FORBIDDEN_USE,
    apply_validated_graph_patch,
    build_graph_patch_lifecycle,
    validate_graph_repair_proposal,
)


class ApplyEnforcedPolicyTests(unittest.TestCase):
    def _runtime_owner(self):
        return ResponsesRequestRuntimeOwner(
            hooks={'normalize_capability': lambda value: str(value or '').strip().lower()},
            capability_chat='chat',
            capability_embedding='embedding',
            capability_image_generation='image_generation',
            capability_speech_to_text='speech_to_text',
            request_timeout_error=TimeoutError,
            request_exception_error=Exception,
        )

    def _base_graph(self):
        return {
            'graph_version': 1,
            'response_id': 'resp-enforced',
            'frame_id': 'frame-enforced',
            'lifecycle_state': 'late_fill_running',
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

    def _scope_review(self, selected_scope, *, evidence_refs=None, artifact_identity=None):
        return {
            'kind': 'ollmo.redraw_scope_ladder_review',
            'review_id': f'redraw-scope-{selected_scope}',
            'status': 'selected',
            'selected_scope': selected_scope,
            'selected_candidate': {
                'scope': selected_scope,
                'evidence_refs': evidence_refs or ['closure:intent_graph_adequacy'],
            },
            'forbidden_evidence_seen': False,
            'blocked_reasons': [],
            'artifact_identity': artifact_identity or {},
        }

    def _repair_proposal(self, *, patch=None, source='closure_review', evidence_refs=None, repair_type='repair_missing_materialization_contract'):
        return {
            'kind': GRAPH_REPAIR_PROPOSAL_KIND,
            'proposal_id': f'proposal-{repair_type}',
            'source': source,
            'repair_type': repair_type,
            'target_graph_id': 'graph-enforced',
            'evidence_refs': evidence_refs or ['closure:intent_graph_adequacy'],
            'patch': patch
            or {
                'add_phases': [
                    {
                        'phase_id': 'repair-image',
                        'branch_id': 'repair-image',
                        'obligation_id': 'obligation-repair-image',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'pending',
                        'placeholder_ref': 'pending-output-repair-image',
                        'output_slot_id': 'slot-repair-image',
                    }
                ]
            },
            'allowed_use': PROPOSAL_ALLOWED_USE,
            'forbidden_use': PROPOSAL_FORBIDDEN_USE,
        }

    def _accepted_repair_review(self, graph, *, proposal=None):
        return validate_graph_repair_proposal(
            proposal or self._repair_proposal(),
            request_phase_graph=graph,
            closure_review={'status': 'repair_required', 'ghost_repair_feedback': {'status': 'repair_required'}},
            promotion_review={'status': 'promoted'},
        )

    def _rebase_base_graph(self):
        graph = self._base_graph()
        graph['kind'] = 'ollmo.request_phase_graph'
        graph['intent_obligations'] = [
            {'obligation_id': 'intent-phase-1', 'phase_id': 'phase-1', 'required': True}
        ]
        graph['output_obligations'] = [
            {'obligation_id': 'obligation-phase-1', 'phase_id': 'phase-1', 'required': True}
        ]
        return graph

    def _rebase_candidate_graph(self):
        candidate = copy.deepcopy(self._rebase_base_graph())
        candidate['frame_id'] = 'frame-rebase-candidate'
        candidate['phases'].append(
            {
                'phase_id': 'phase-review',
                'branch_id': 'branch-review',
                'obligation_id': 'obligation-review',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
                'depends_on': ['phase-1'],
                'lineage': {'parent_phase_id': 'phase-1', 'relation': 'split_branch'},
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-review',
                'branch_id': 'branch-review',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {'parent_phase_id': 'phase-1', 'relation': 'split_branch'},
            }
        )
        candidate['output_obligations'].append(
            {
                'obligation_id': 'obligation-review',
                'phase_id': 'phase-review',
                'capability': 'chat',
                'output_type': 'text',
                'required': True,
                'lineage': {'parent_obligation_id': 'obligation-phase-1', 'relation': 'split_branch'},
            }
        )
        return candidate

    def _accepted_rebase_review(self, *, requested_rebase_class='full_successor_rebase'):
        base = self._rebase_base_graph()
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=self._rebase_candidate_graph(),
            target_response_id='resp-enforced',
            target_frame_id='frame-enforced',
            source='runtime_closure_review',
            reason='Runtime reviewed candidate.',
            intent_anchor={'prompt': 'preserve current graph and add review branch'},
            evidence_refs=['closure:intent_graph_adequacy'],
            requested_rebase_class=requested_rebase_class,
            scope_root_ids=['phase-1'] if requested_rebase_class == 'partial_subtree_rebase' else [],
            scope_phase_ids=['phase-1', 'phase-review']
            if requested_rebase_class == 'partial_subtree_rebase'
            else [],
            scope_branch_ids=['branch-review']
            if requested_rebase_class == 'partial_subtree_rebase'
            else [],
            preserve_outside_scope=True if requested_rebase_class == 'partial_subtree_rebase' else None,
            redraw_scope_review_ref='redraw-scope-rebase',
        )
        return validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

    def test_policy_mode_is_default_deny_and_invalid_values_are_visible(self):
        self.assertEqual(normalize_enforced_policy_mode(None), 'off')
        self.assertEqual(normalize_enforced_policy_mode('safe_v1'), 'safe_v1')
        self.assertEqual(normalize_enforced_policy_mode('anything-else'), 'off')

        description = describe_enforced_policy('anything-else')

        self.assertEqual(description['mode'], 'off')
        self.assertEqual(description['normalized'], 'off')
        self.assertEqual(description['raw_value'], 'anything-else')
        self.assertEqual(description['invalid_value'], 'anything-else')
        self.assertFalse(description['enabled'])

    def test_policy_product_default_and_fail_closed_overrides(self):
        product_default = describe_enforced_policy_from_env({})
        self.assertEqual(product_default['mode'], 'safe_v1')
        self.assertEqual(product_default['normalized'], 'safe_v1')
        self.assertEqual(product_default['source'], 'product_default')
        self.assertFalse(product_default['configured'])
        self.assertTrue(product_default['enabled'])

        explicit_off = describe_enforced_policy_from_env(
            {'OLLMO_APPLY_ENFORCED_POLICY': 'off'}
        )
        self.assertEqual(explicit_off['mode'], 'off')
        self.assertEqual(explicit_off['source'], 'environment')
        self.assertTrue(explicit_off['configured'])
        self.assertFalse(explicit_off['enabled'])

        invalid = describe_enforced_policy_from_env(
            {'OLLMO_APPLY_ENFORCED_POLICY': 'anything-else'}
        )
        self.assertEqual(invalid['mode'], 'off')
        self.assertEqual(invalid['source'], 'environment')
        self.assertTrue(invalid['configured'])
        self.assertEqual(invalid['invalid_value'], 'anything-else')
        self.assertFalse(invalid['enabled'])

    def test_apply_enforced_explicit_off_denies_safe_additive_patch(self):
        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = self._scope_review('add_missing_branch')
        review = self._accepted_repair_review(graph)

        with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'off'}):
            lifecycle = build_graph_patch_lifecycle(
                request_phase_graph=graph,
                proposal_review=review,
                autonomy_level='apply_enforced',
            )

        self.assertIn(lifecycle['status'], {'blocked', 'rejected'})
        self.assertEqual(lifecycle['enforced_policy_review']['kind'], ENFORCED_POLICY_REVIEW_KIND)
        self.assertIn('enforced_policy_off', lifecycle['blocked_reasons'])
        self.assertFalse(enforced_policy_allows_application(lifecycle['enforced_policy_review']))

    def test_apply_enforced_safe_v1_allows_missing_branch_and_preserves_lineage(self):
        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = self._scope_review('add_missing_branch')
        graph['output_slots'] = [
            {
                'slot_id': 'slot-repair-image',
                'branch_id': 'repair-image',
                'phase_id': 'repair-image',
                'placeholder_ref': 'pending-output-repair-image',
            }
        ]
        graph['ollmo'] = {
            'work_tree': {
                'nodes': [
                    {
                        'node_id': 'work-repair-image',
                        'branch_id': 'repair-image',
                        'phase_id': 'repair-image',
                        'placeholder_ref': 'pending-output-repair-image',
                    }
                ]
            }
        }
        review = self._accepted_repair_review(graph)

        with patch.dict(os.environ, {}, clear=True):
            lifecycle = build_graph_patch_lifecycle(
                request_phase_graph=graph,
                proposal_review=review,
                autonomy_level='apply_enforced',
            )
            result = apply_validated_graph_patch(
                graph,
                lifecycle,
                autonomy_level='apply_enforced',
            )

        self.assertEqual(lifecycle['status'], 'staged')
        self.assertTrue(lifecycle['enforced_policy_review']['allowed'])
        self.assertEqual(
            lifecycle['enforced_policy_review']['policy']['source'],
            'product_default',
        )
        self.assertFalse(lifecycle['enforced_policy_review']['policy']['configured'])
        self.assertEqual(lifecycle['authority'], 'runtime_enforced_policy')
        self.assertEqual(result['status'], 'applied')
        patched = result['graph']
        phase = next(item for item in patched['phases'] if item['phase_id'] == 'repair-image')
        self.assertEqual(phase['placeholder_ref'], 'pending-output-repair-image')
        self.assertEqual(patched['output_slots'][0]['placeholder_ref'], 'pending-output-repair-image')
        self.assertEqual(
            patched['ollmo']['work_tree']['nodes'][0]['placeholder_ref'],
            'pending-output-repair-image',
        )
        lifecycle_record = patched['graph_patch_lifecycle'][0]
        self.assertEqual(lifecycle_record['outcome']['runtime_effect'], 'graph_mutated')
        self.assertEqual(lifecycle_record['authority'], 'runtime_enforced_policy')
        self.assertEqual(
            lifecycle_record['placeholder_lineage']['placeholder_refs'],
            ['pending-output-repair-image'],
        )
        self.assertEqual(lifecycle_record['output_slot_lineage']['slot_ids'], ['slot-repair-image'])
        self.assertEqual(lifecycle_record['work_tree_lineage']['node_ids'], ['work-repair-image'])

        with patch.dict(os.environ, {}, clear=True):
            replay = apply_validated_graph_patch(
                patched,
                lifecycle,
                autonomy_level='apply_enforced',
            )
        self.assertEqual(replay['status'], 'already_applied')
        self.assertEqual(len(replay['graph']['phases']), len(patched['phases']))

    def test_apply_enforced_allows_dependency_repair_only_with_scope_gate(self):
        graph = self._base_graph()
        graph['phases'].append(
            {
                'phase_id': 'phase-html',
                'branch_id': 'phase-html',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
            }
        )
        proposal = self._repair_proposal(
            repair_type='rebind_artifact_dependency',
            patch={'add_dependencies': [{'target_phase_id': 'phase-html', 'source_phase_id': 'phase-1'}]},
        )
        graph['redraw_scope_ladder_review'] = self._scope_review('observe')
        review = self._accepted_repair_review(graph, proposal=proposal)

        with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'safe_v1'}):
            blocked = build_graph_patch_lifecycle(
                request_phase_graph=graph,
                proposal_review=review,
                autonomy_level='apply_enforced',
            )

        self.assertEqual(blocked['status'], 'blocked')
        self.assertIn('redraw_scope_not_smallest_allowed_for_enforced_class', blocked['blocked_reasons'])

        graph['redraw_scope_ladder_review'] = self._scope_review('repair_binding_dependency')
        review = self._accepted_repair_review(graph, proposal=proposal)
        with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'safe_v1'}):
            lifecycle = build_graph_patch_lifecycle(
                request_phase_graph=graph,
                proposal_review=review,
                autonomy_level='apply_enforced',
            )
            result = apply_validated_graph_patch(graph, lifecycle, autonomy_level='apply_enforced')

        self.assertEqual(lifecycle['status'], 'staged')
        self.assertEqual(lifecycle['enforced_class'], 'safe_additive_dependency_repair')
        self.assertEqual(result['status'], 'applied')
        phase_html = next(item for item in result['graph']['phases'] if item['phase_id'] == 'phase-html')
        self.assertEqual(phase_html['depends_on'], ['phase-1'])

    def test_apply_enforced_requires_safe_additive_risk_for_safe_policy_classes(self):
        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = self._scope_review('add_missing_branch')
        review = self._accepted_repair_review(graph)

        with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'safe_v1'}):
            lifecycle = build_graph_patch_lifecycle(
                request_phase_graph=graph,
                proposal_review=review,
                autonomy_level='apply_enforced',
            )
            malformed_lifecycle = {**lifecycle, 'risk_level': 'review_required'}
            result = apply_validated_graph_patch(
                graph,
                malformed_lifecycle,
                autonomy_level='apply_enforced',
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn('safe_additive_risk_level_required', result['blocked_reasons'])
        lifecycle_record = result['graph']['graph_patch_lifecycle'][0]
        self.assertFalse(lifecycle_record['enforced_policy_review']['allowed'])

    def test_learning_only_and_forbidden_evidence_cannot_enforce(self):
        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = self._scope_review('add_missing_branch')
        learning_proposal = self._repair_proposal(
            source='accepted_learning',
            evidence_refs=['accepted_learning:case-1'],
        )
        learning_review = validate_graph_repair_proposal(
            learning_proposal,
            request_phase_graph=graph,
            accepted_learning_hints={'hint_count': 1},
        )
        lifecycle = build_graph_patch_lifecycle(
            request_phase_graph=graph,
            proposal_review=learning_review,
            autonomy_level='apply_enforced',
        )
        self.assertIn(lifecycle['status'], {'blocked', 'rejected'})
        self.assertIn('proposal_review_not_accepted', lifecycle['blocked_reasons'])

        proposal = self._repair_proposal(evidence_refs=['runtime:degraded_liveness_only'])
        review = self._accepted_repair_review(graph, proposal=proposal)
        with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'safe_v1'}):
            blocked = build_graph_patch_lifecycle(
                request_phase_graph=graph,
                proposal_review=review,
                autonomy_level='apply_enforced',
            )
        self.assertEqual(blocked['status'], 'blocked')
        self.assertIn('forbidden_evidence_not_enforced_authority', blocked['blocked_reasons'])

    def test_duplicate_artifact_alias_canonicalization_is_allowed_but_conflicts_block(self):
        allowed = build_enforced_policy_review(
            autonomy_level='apply_enforced',
            lifecycle={
                'kind': 'ollmo.graph_patch_lifecycle',
                'repair_class': 'artifact_binding_repair_branch',
                'enforced_class': 'duplicate_artifact_alias_canonicalization',
                'source_evidence_refs': ['artifact_identity:duplicate_ref:artifact://a'],
                'idempotency_key': 'idem-artifact-alias',
                'outcome': {'runtime_effect': 'pending_application'},
            },
            request_phase_graph={
                'redraw_scope_ladder_review': self._scope_review(
                    'repair_artifact_ref_identity',
                    artifact_identity={
                        'duplicate_refs': ['artifact://a'],
                        'canonical_refs': ['file:///tmp/a.png'],
                        'final_projection_blocked': False,
                    },
                )
            },
            policy={'mode': 'safe_v1'},
        )
        self.assertTrue(allowed['allowed'])
        self.assertEqual(allowed['enforced_class'], 'duplicate_artifact_alias_canonicalization')

        conflict = build_enforced_policy_review(
            autonomy_level='apply_enforced',
            lifecycle={
                'kind': 'ollmo.graph_patch_lifecycle',
                'repair_class': 'artifact_binding_repair_branch',
                'enforced_class': 'duplicate_artifact_alias_canonicalization',
                'source_evidence_refs': ['artifact_identity:duplicate_ref:artifact://a'],
                'idempotency_key': 'idem-artifact-alias',
            },
            request_phase_graph={
                'redraw_scope_ladder_review': self._scope_review(
                    'repair_artifact_ref_identity',
                    artifact_identity={
                        'duplicate_refs': ['artifact://a', 'artifact://b'],
                        'conflicting_refs': ['artifact://b'],
                        'final_projection_blocked': True,
                    },
                )
            },
            policy={'mode': 'safe_v1'},
        )
        self.assertFalse(conflict['allowed'])
        self.assertIn('conflicting_duplicate_artifact_ref_not_enforced', conflict['blocked_reasons'])

    def test_apply_enforced_full_and_partial_rebase_remain_blocked_v1(self):
        base = self._rebase_base_graph()
        for requested_class, expected_reason in (
            ('full_successor_rebase', 'full_successor_rebase_not_enforced_v1'),
            ('partial_subtree_rebase', 'partial_subtree_rebase_enforced_v1_audit_only'),
        ):
            review = self._accepted_rebase_review(requested_rebase_class=requested_class)
            with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'safe_v1'}):
                lifecycle = build_graph_rebase_lifecycle(
                    request_phase_graph=base,
                    rebase_review=review,
                    autonomy_level='apply_enforced',
                )

            self.assertEqual(lifecycle['status'], 'blocked')
            self.assertEqual(lifecycle['enforced_policy_review']['kind'], ENFORCED_POLICY_REVIEW_KIND)
            self.assertFalse(lifecycle['enforced_policy_review']['allowed'])
            self.assertIn(expected_reason, lifecycle['blocked_reasons'])

    def test_runtime_diagnostics_include_enforced_policy_reviews(self):
        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = self._scope_review('add_missing_branch')
        graph['graph_repair_reviews'] = [self._accepted_repair_review(graph)]
        payload = {
            'response_id': 'resp-enforced-runtime-diag',
            'lifecycle_state': 'late_fill_running',
            'runtime': {'request_phase_graph': graph},
        }

        with patch.dict(os.environ, {'OLLMO_APPLY_ENFORCED_POLICY': 'safe_v1'}):
            updated = self._runtime_owner()._attach_graph_patch_lifecycle(
                payload,
                graph_repair_autonomy='apply_enforced',
            )

        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(diagnostics['enforced_policy']['mode'], 'safe_v1')
        self.assertEqual(diagnostics['graph_patch_enforced_policy_reviews'][0]['kind'], ENFORCED_POLICY_REVIEW_KIND)
        self.assertTrue(diagnostics['graph_patch_enforced_policy_reviews'][0]['allowed'])
        self.assertEqual(
            diagnostics['graph_patch_lifecycle'][0]['authority'],
            'runtime_enforced_policy',
        )


if __name__ == '__main__':
    unittest.main()
