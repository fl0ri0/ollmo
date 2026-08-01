import copy
import unittest

from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_services.redraw_scope import (
    REDRAW_SCOPE_REVIEW_KIND,
    build_redraw_scope_ladder_review,
    canonicalize_duplicate_artifact_refs,
)


class RedrawScopeLadderTests(unittest.TestCase):
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
            'kind': 'ollmo.request_phase_graph',
            'response_id': 'resp-scope',
            'prompt_intent': {
                'artifact_request': True,
                'local_visual_asset_requirement': True,
            },
            'intent_obligations': [
                {
                    'obligation_id': 'intent-html',
                    'kind': 'text_artifact',
                    'required': True,
                    'phase_id': 'phase-html',
                },
                {
                    'obligation_id': 'intent-image',
                    'kind': 'media_artifact',
                    'required': True,
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'phase_id': 'phase-image',
                },
                {
                    'obligation_id': 'intent-binding',
                    'kind': 'dependency',
                    'required': True,
                    'dependency_contract': 'local_visual_asset_binding',
                    'execution_dependency_required': True,
                    'target_phase_id': 'phase-html',
                    'source_phase_ids': ['phase-image'],
                },
            ],
            'output_obligations': [
                {
                    'obligation_id': 'obligation-html',
                    'phase_id': 'phase-html',
                    'capability': 'chat',
                    'output_type': 'text',
                    'target_path': 'artifacts/documents/index.html',
                    'required': True,
                }
            ],
            'phases': [
                {
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'pending',
                }
            ],
            'downstream_branches': [
                {
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'pending',
                }
            ],
        }

    def _closure_review(self, *, repair_action='add_missing_branch'):
        return {
            'kind': 'ollmo.graph_closure_review',
            'status': 'repair_required',
            'intent_graph_adequacy': {
                'status': 'pending',
                'checks': [
                    {
                        'check_kind': 'intent_graph_adequacy',
                        'status': 'pending',
                        'evidence': 'intent_graph_adequacy_missing_image_output',
                        'repair_action': repair_action,
                        'add_phases': [{'phase_id': 'phase-image', 'capability': 'image_generation'}],
                    }
                ],
                'intent_lens_review': {
                    'intent_attention_review': {'status': 'pending'},
                    'intent_commitment_review': {'status': 'pending'},
                },
            },
        }

    def test_reserved_slot_fill_precedes_new_branch(self):
        graph = self._base_graph()
        graph['candidate_graph'] = {
            'candidates': [
                {
                    'candidate_id': 'candidate-image',
                    'candidate_type': 'workload_task',
                    'status': 'reserved',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'phase_id': 'phase-reserved-image',
                    'branch_id': 'branch-reserved-image',
                    'obligation_id': 'obligation-reserved-image',
                    'reserved_reason': 'possibility_slot',
                }
            ],
            'promotion_review': {'status': 'pending'},
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )

        self.assertEqual(review['kind'], REDRAW_SCOPE_REVIEW_KIND)
        self.assertEqual(review['selected_scope'], 'promote_reserved_slot')
        self.assertEqual(review['selected_candidate']['target_ids'], ['candidate-image'])
        self.assertFalse(review['learning_orientation']['used_as_authority'])

    def test_reserved_workload_candidates_already_represented_by_active_graph_are_ignored(self):
        graph = self._base_graph()
        represented_identity_cases = (
            {'phase_id': 'phase-html', 'branch_id': 'branch-new-phase-match'},
            {'phase_id': 'phase-new-branch-match', 'branch_id': 'branch-html'},
            {
                'phase_id': 'phase-new-obligation-match',
                'branch_id': 'branch-new-obligation-match',
                'obligation_id': 'obligation-html',
            },
        )
        graph['candidate_graph'] = {
            'candidates': [
                {
                    'candidate_id': f'candidate-represented-{index}',
                    'candidate_type': 'workload_task',
                    'status': 'reserved',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'reserved_reason': 'possibility_slot',
                    **identity,
                }
                for index, identity in enumerate(represented_identity_cases, start=1)
            ],
            'promotion_review': {'status': 'pending'},
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )

        self.assertEqual(review['selected_scope'], 'add_missing_branch')
        reserved_scope = next(
            item for item in review['scopes_considered'] if item['scope'] == 'promote_reserved_slot'
        )
        self.assertFalse(reserved_scope['eligible'])
        self.assertEqual(reserved_scope.get('target_ids'), None)

    def test_additive_missing_branch_precedes_rebase_without_reserved_candidate(self):
        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=self._base_graph(),
            closure_review=self._closure_review(),
        )

        self.assertEqual(review['selected_scope'], 'add_missing_branch')
        self.assertIn('intent_graph_adequacy', review['selected_candidate']['evidence_refs'])
        self.assertNotIn('full_successor_rebase', review['blocked_reasons'])

    def test_binding_dependency_repair_precedes_rebase_for_existing_artifacts(self):
        graph = self._base_graph()
        graph['phases'].append(
            {
                'phase_id': 'phase-image',
                'branch_id': 'branch-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'fulfilled',
                'artifact_refs': ['artifact://image-one'],
            }
        )
        graph['downstream_branches'].append(
            {
                'phase_id': 'phase-image',
                'branch_id': 'branch-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'fulfilled',
            }
        )
        graph['output_obligations'].append(
            {
                'obligation_id': 'obligation-image',
                'phase_id': 'phase-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'artifact_refs': ['artifact://image-one'],
                'required': True,
            }
        )
        closure = self._closure_review(repair_action='rebind_artifact_dependency')

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'repair_binding_dependency')
        self.assertEqual(review['selected_candidate']['repair_type'], 'rebind_artifact_dependency')

    def test_nested_failed_branch_dependency_repair_precedes_generic_manual_review(self):
        graph = self._base_graph()
        graph['phases'][0]['depends_on'] = ['phase-image']
        graph['downstream_branches'][0]['depends_on'] = ['phase-image']
        graph['phases'].append(
            {
                'phase_id': 'phase-image',
                'branch_id': 'branch-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'fulfilled',
            }
        )
        graph['downstream_branches'].append(
            {
                'phase_id': 'phase-image',
                'branch_id': 'branch-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'fulfilled',
            }
        )
        closure = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'blocked',
            'checks': [
                {
                    'check_kind': 'output_obligation',
                    'status': 'blocked',
                    'evidence': 'late_fill_failed_branch',
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'depends_on': ['phase-image'],
                    'repair_action': 'manual_review',
                    'recovery_action': 'manual_review',
                    'blocked_by_dependency_input': True,
                    'recovery_context': {
                        'repair_required': True,
                        'retry_scope': 'dependency_chain',
                        'suggested_action': 'repair_dependency_chain',
                    },
                    'recovery_state': {
                        'status': 'candidate',
                        'repair_required': True,
                        'retry_scope': 'dependency_chain',
                        'suggested_action': 'repair_dependency_chain',
                    },
                }
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'repair_binding_dependency')
        self.assertEqual(review['selected_candidate']['repair_type'], 'rebind_artifact_dependency')

    def test_non_dependency_nested_recovery_does_not_become_binding_repair(self):
        closure = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'blocked',
            'checks': [
                {
                    'status': 'blocked',
                    'evidence': 'late_fill_failed_branch',
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'repair_action': 'manual_review',
                    'recovery_state': {
                        'status': 'candidate',
                        'suggested_action': 'retry_same_branch',
                    },
                }
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=self._base_graph(),
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'observe')

    def test_nested_dependency_repair_precedes_explicit_partial_rebase(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'partial_subtree_rebase',
            'scope_root_ids': ['phase-html'],
            'evidence_refs': ['closure:structural-conflict'],
        }
        closure = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'blocked',
            'checks': [
                {
                    'status': 'blocked',
                    'evidence': 'late_fill_failed_branch',
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'repair_action': 'manual_review',
                    'blocked_by_dependency_input': True,
                    'recovery_state': {
                        'status': 'candidate',
                        'suggested_action': 'repair_dependency_chain',
                    },
                }
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'repair_binding_dependency')

    def test_unbound_intent_adequacy_obligation_gap_remains_additive_scope(self):
        cases = (
            {
                'capability': 'image_generation',
                'output_type': 'image',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
            },
            {
                'capability': 'speech_to_text',
                'output_type': 'text',
                'evidence': 'intent_graph_adequacy_missing_capability_obligation',
            },
        )
        for case in cases:
            with self.subTest(evidence=case['evidence']):
                closure = {
                    'kind': 'ollmo.graph_closure_review',
                    'status': 'repair_required',
                    'intent_graph_adequacy': {
                        'status': 'pending',
                        'checks': [
                            {
                                'check_kind': 'intent_graph_adequacy',
                                'status': 'pending',
                                'repair_action': 'rebuild_from_promoted_obligations',
                                'expected_count': 1,
                                'actual_count': 0,
                                'missing_count': 1,
                                **case,
                            }
                        ],
                    },
                }

                review = build_redraw_scope_ladder_review(
                    response_payload={'response_id': 'resp-scope'},
                    request_phase_graph=self._base_graph(),
                    closure_review=closure,
                )

                self.assertEqual(review['selected_scope'], 'add_missing_branch')

    def test_unproven_unbound_adequacy_gap_does_not_suppress_partial_rebase(self):
        cases = (
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 1,
                'actual_count': 1,
                'missing_count': 0,
            },
            {
                'check_kind': 'semantic_review',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 1,
                'actual_count': 0,
                'missing_count': 1,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'reason': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 1,
                'actual_count': 0,
                'missing_count': 1,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'fulfilled',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 1,
                'actual_count': 0,
                'missing_count': 1,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 0,
                'actual_count': 0,
                'missing_count': True,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 0,
                'actual_count': 0,
                'missing_count': 1.5,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 0,
                'actual_count': 0,
                'missing_count': float('inf'),
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': True,
                'actual_count': False,
                'missing_count': 0,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': 0,
                'actual_count': -1,
                'missing_count': 0,
            },
            {
                'check_kind': 'intent_graph_adequacy',
                'status': 'pending',
                'evidence': 'intent_graph_adequacy_missing_output_obligation',
                'expected_count': '0',
                'actual_count': '0',
                'missing_count': '1' * 5000,
            },
        )
        for check in cases:
            with self.subTest(check=check):
                graph = self._base_graph()
                graph['redraw_scope_evidence'] = {
                    'status': 'repair_required',
                    'recommended_scope': 'partial_subtree_rebase',
                    'scope_root_ids': ['phase-html'],
                    'evidence_refs': ['closure:structural-conflict'],
                }
                closure = {
                    'kind': 'ollmo.graph_closure_review',
                    'status': 'repair_required',
                    'intent_graph_adequacy': {'status': 'pending', 'checks': [check]},
                }

                review = build_redraw_scope_ladder_review(
                    response_payload={'response_id': 'resp-scope'},
                    request_phase_graph=graph,
                    closure_review=closure,
                )

                self.assertEqual(review['selected_scope'], 'partial_subtree_rebase')

    def test_fulfilled_missing_branch_check_for_existing_phase_is_not_actionable(self):
        closure = {
            'status': 'repair_required',
            'intent_graph_adequacy': {
                'checks': [
                    {
                        'check_kind': 'intent_graph_adequacy',
                        'status': 'fulfilled',
                        'repair_action': 'add_missing_branch',
                        'evidence': 'intent_graph_adequacy_missing_output',
                        'add_phases': [{'phase_id': 'phase-html'}],
                    }
                ]
            },
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=self._base_graph(),
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'observe')
        missing_scope = next(
            item for item in review['scopes_considered']
            if item['scope'] == 'add_missing_branch'
        )
        self.assertFalse(missing_scope['eligible'])

    def test_open_missing_branch_check_for_existing_branch_is_not_actionable(self):
        closure = {
            'status': 'repair_required',
            'checks': [
                {
                    'check_kind': 'intent_graph_adequacy',
                    'status': 'pending',
                    'repair_action': 'add_missing_branch',
                    'branch_id': 'branch-html',
                }
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=self._base_graph(),
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'observe')

    def test_missing_branch_under_existing_phase_remains_additive_scope(self):
        closure = {
            'status': 'repair_required',
            'checks': [
                {
                    'check_kind': 'intent_graph_adequacy',
                    'status': 'pending',
                    'repair_action': 'add_missing_branch',
                    'add_phases': [
                        {
                            'phase_id': 'phase-html',
                            'branch_id': 'branch-new-under-existing-phase',
                        }
                    ],
                }
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=self._base_graph(),
            closure_review=closure,
        )

        self.assertEqual(review['selected_scope'], 'add_missing_branch')

    def test_correct_local_visual_binding_is_not_selected_for_repair(self):
        graph = self._base_graph()
        graph['phases'][0]['depends_on'] = ['phase-image']
        graph['downstream_branches'][0]['depends_on'] = ['phase-image']
        graph['phases'].append(
            {
                'phase_id': 'phase-image',
                'branch_id': 'branch-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'fulfilled',
                'artifact_refs': ['artifact://image-one'],
            }
        )
        graph['downstream_branches'].append(
            {
                'phase_id': 'phase-image',
                'branch_id': 'branch-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'fulfilled',
            }
        )
        graph['output_obligations'].append(
            {
                'obligation_id': 'obligation-image',
                'phase_id': 'phase-image',
                'capability': 'image_generation',
                'output_type': 'image',
                'artifact_refs': ['artifact://image-one'],
                'required': True,
            }
        )

        review = build_redraw_scope_ladder_review(
            response_payload={
                'response_id': 'resp-scope',
                'artifacts': [
                    {
                        'type': 'image',
                        'artifact_ref': 'artifact://image-one',
                        'branch_id': 'branch-image',
                    }
                ],
            },
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['selected_scope'], 'observe')
        binding_scope = next(
            item for item in review['scopes_considered']
            if item['scope'] == 'repair_binding_dependency'
        )
        self.assertFalse(binding_scope['eligible'])

    def test_duplicate_artifact_refs_are_identity_hygiene_before_rebase(self):
        payload = {
            'response_id': 'resp-duplicates',
            'artifacts': [
                {
                    'type': 'image',
                    'path': '/tmp/generated/a.png',
                    'artifact_ref': 'artifact:image-dup',
                    'branch_id': 'branch-image-1',
                    'batch_prompt': 'first image',
                },
                {
                    'type': 'image',
                    'path': '/tmp/generated/a.png',
                    'artifact_ref': 'artifact:image-dup',
                    'branch_id': 'branch-image-1',
                    'batch_prompt': 'first image alias',
                },
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload=payload,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )
        canonical = canonicalize_duplicate_artifact_refs(payload['artifacts'])

        self.assertEqual(review['selected_scope'], 'repair_artifact_ref_identity')
        self.assertTrue(review['artifact_identity']['canonicalization_required'])
        self.assertFalse(review['artifact_identity']['final_projection_blocked'])
        self.assertEqual(len(canonical['artifacts']), 1)
        self.assertEqual(canonical['artifacts'][0]['alias_artifact_refs'], ['artifact:image-dup'])
        self.assertIn('first image alias', canonical['artifacts'][0]['alias_metadata']['batch_prompts'])

    def test_conflicting_duplicate_artifact_refs_block_final_projection(self):
        payload = {
            'response_id': 'resp-duplicate-conflict',
            'artifacts': [
                {'type': 'image', 'path': '/tmp/generated/a.png', 'artifact_ref': 'artifact:image-dup'},
                {'type': 'image', 'path': '/tmp/generated/b.png', 'artifact_ref': 'artifact:image-dup'},
            ],
        }

        review = build_redraw_scope_ladder_review(
            response_payload=payload,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )
        canonical = canonicalize_duplicate_artifact_refs(payload['artifacts'])

        self.assertEqual(review['selected_scope'], 'repair_artifact_ref_identity')
        self.assertTrue(review['artifact_identity']['final_projection_blocked'])
        self.assertTrue(canonical['final_projection_blocked'])
        self.assertIn('conflicting_duplicate_artifact_ref', review['blocked_reasons'])

    def test_document_and_text_labels_are_one_identity_for_same_ref_and_path(self):
        artifacts = [
            {
                'type': 'document',
                'path': '/tmp/generated/result.md',
                'artifact_ref': 'artifact:result',
                'branch_id': 'branch-chat-1',
            },
            {
                'type': 'text',
                'path': '/tmp/generated/result.md',
                'artifact_ref': 'artifact:result',
                'phase_id': 'phase-chat-1',
            },
        ]

        canonical = canonicalize_duplicate_artifact_refs(artifacts)

        self.assertTrue(canonical['canonicalization_required'])
        self.assertFalse(canonical['final_projection_blocked'])
        self.assertEqual(len(canonical['artifacts']), 1)
        self.assertEqual(canonical['artifacts'][0]['type'], 'document')
        self.assertEqual(canonical['artifacts'][0]['artifact_ref'], 'artifact:result')

    def test_document_and_text_aliases_still_conflict_across_different_paths(self):
        canonical = canonicalize_duplicate_artifact_refs(
            [
                {
                    'type': 'document',
                    'path': '/tmp/generated/result-a.md',
                    'artifact_ref': 'artifact:result',
                },
                {
                    'type': 'text',
                    'path': '/tmp/generated/result-b.md',
                    'artifact_ref': 'artifact:result',
                },
            ]
        )

        self.assertTrue(canonical['final_projection_blocked'])
        self.assertEqual(canonical['conflicts'][0]['reason'], 'conflicting_duplicate_artifact_ref')

    def test_document_and_non_text_labels_still_conflict_for_same_ref_and_path(self):
        canonical = canonicalize_duplicate_artifact_refs(
            [
                {
                    'type': 'document',
                    'path': '/tmp/generated/result.md',
                    'artifact_ref': 'artifact:result',
                },
                {
                    'type': 'audio',
                    'path': '/tmp/generated/result.md',
                    'artifact_ref': 'artifact:result',
                },
            ]
        )

        self.assertTrue(canonical['final_projection_blocked'])
        self.assertEqual(canonical['conflicts'][0]['types'], ['audio', 'document'])

    def test_document_and_text_alias_requires_concrete_text_path_proof(self):
        cases = (
            (
                [
                    {'type': 'document', 'artifact_ref': 'artifact:result'},
                    {'type': 'text', 'artifact_ref': 'artifact:result'},
                ],
                'missing_path',
            ),
            (
                [
                    {
                        'type': 'document',
                        'path': '/tmp/generated/result.md',
                        'artifact_ref': 'artifact:result',
                    },
                    {'type': 'text', 'artifact_ref': 'artifact:result'},
                ],
                'partially_missing_path',
            ),
            (
                [
                    {
                        'type': 'document',
                        'path': '/tmp/generated/result.md',
                        'artifact_ref': 'artifact:result',
                    },
                    {
                        'type': 'text',
                        'path': '/tmp/generated/result.md',
                        'artifact_ref': 'artifact:result',
                    },
                    {
                        'path': '/tmp/generated/result.md',
                        'artifact_ref': 'artifact:result',
                    },
                ],
                'untyped_alias_record',
            ),
            (
                [
                    {
                        'type': 'document',
                        'path': '/tmp/generated/result.pdf',
                        'artifact_ref': 'artifact:result',
                    },
                    {
                        'type': 'text',
                        'path': '/tmp/generated/result.pdf',
                        'artifact_ref': 'artifact:result',
                    },
                ],
                'non_text_path',
            ),
        )

        for artifacts, label in cases:
            with self.subTest(label=label):
                canonical = canonicalize_duplicate_artifact_refs(artifacts)
                self.assertTrue(canonical['final_projection_blocked'])
                self.assertEqual(
                    canonical['conflicts'][0]['reason'],
                    'conflicting_duplicate_artifact_ref',
                )

    def test_partial_subtree_rebase_selected_before_full_rebase(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'partial_subtree_rebase',
            'scope_root_ids': ['phase-image', 'phase-html'],
            'reason': 'section_media_mapping_conflict',
            'evidence_refs': ['closure:section_media_mapping_conflict'],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['selected_scope'], 'partial_subtree_rebase')
        self.assertEqual(review['selected_candidate']['scope_root_ids'], ['phase-image', 'phase-html'])
        self.assertTrue(review['selected_candidate']['preserve_outside_scope'])

    def test_full_successor_rebase_is_last_reviewed_path(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'full_successor_rebase',
            'reason': 'global_graph_shape_conflicts_with_intent',
            'evidence_refs': ['closure:global_graph_shape_conflict'],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['selected_scope'], 'full_successor_rebase')
        self.assertEqual(review['selected_candidate']['runtime_action'], 'reviewed_graph_rebase_only')
        self.assertEqual(review['scope_ceiling'], 'full_successor_rebase')

    def test_smaller_missing_branch_scope_precedes_explicit_full_rebase(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'full_successor_rebase',
            'reason': 'global_graph_shape_conflicts_with_intent',
            'evidence_refs': ['closure:global_graph_shape_conflict'],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )

        self.assertEqual(review['selected_scope'], 'add_missing_branch')

    def test_repeated_scope_review_keeps_stable_base_graph_digest(self):
        graph = self._base_graph()
        first = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )
        graph['redraw_scope_ladder_review'] = first

        second = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )

        self.assertEqual(first['base_graph_digest'], second['base_graph_digest'])

    def test_learning_only_and_degraded_provider_evidence_cannot_select_executable_scope(self):
        learning_review = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-learning'},
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'fulfilled'},
            accepted_learning_hints=[
                {
                    'hint_id': 'hint-partial-rebase',
                    'suggested_scope': 'partial_subtree_rebase',
                    'authority': 'soft_hint',
                }
            ],
        )
        degraded_review = build_redraw_scope_ladder_review(
            response_payload={
                'response_id': 'resp-degraded',
                'runtime': {'route_health': {'status': 'degraded'}},
            },
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'fulfilled'},
            surface_state={'state': 'pending', 'category_counts': {'provider_degraded': 1}},
        )

        self.assertEqual(learning_review['selected_scope'], 'observe')
        self.assertTrue(learning_review['learning_orientation']['used'])
        self.assertFalse(learning_review['learning_orientation']['used_as_authority'])
        self.assertIn('current_runtime_evidence_required', learning_review['blocked_reasons'])
        self.assertEqual(degraded_review['selected_scope'], 'observe')
        self.assertIn('degraded_or_provider_signal_not_scope_authority', degraded_review['blocked_reasons'])

    def test_degraded_film_texture_prompt_is_not_provider_health_evidence(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'full_successor_rebase',
            'reason': 'global_graph_shape_conflicts_with_intent',
            'evidence_refs': ['closure:global_graph_shape_conflict'],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={
                'response_id': 'resp-degraded-film-texture',
                'prompt': 'Create a degraded film texture with dust and scratches.',
            },
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['selected_scope'], 'full_successor_rebase')
        self.assertEqual(review['status'], 'selected')
        self.assertEqual(review['forbidden_evidence_seen'], [])
        self.assertNotIn(
            'degraded_or_provider_signal_not_scope_authority',
            review['blocked_reasons'],
        )

    def test_structured_degraded_route_and_provider_health_block_requested_rebase(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'full_successor_rebase',
            'reason': 'global_graph_shape_conflicts_with_intent',
            'evidence_refs': ['closure:global_graph_shape_conflict'],
        }

        review = build_redraw_scope_ladder_review(
            response_payload={
                'response_id': 'resp-structured-provider-degradation',
                'runtime': {
                    'route_health': {'status': 'degraded'},
                    'provider_health': {'status': 'provider_degraded'},
                },
            },
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['selected_scope'], 'observe')
        self.assertEqual(review['status'], 'blocked')
        self.assertIn(
            'degraded_or_provider_signal_not_scope_authority',
            review['blocked_reasons'],
        )
        self.assertIn('route_health', review['forbidden_evidence_seen'])
        self.assertIn('provider_degraded', review['forbidden_evidence_seen'])

    def test_recovered_route_health_does_not_reuse_stale_scope_diagnostics(self):
        graph = self._base_graph()
        graph['redraw_scope_evidence'] = {
            'status': 'repair_required',
            'recommended_scope': 'partial_subtree_rebase',
            'scope_root_ids': ['phase-html'],
            'evidence_refs': ['closure:structural-conflict'],
        }
        first_payload = {
            'response_id': 'resp-scope-recovery',
            'runtime': {'route_health': {'status': 'degraded'}},
        }
        first = build_redraw_scope_ladder_review(
            response_payload=first_payload,
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )
        graph['redraw_scope_ladder_review'] = first
        recovered_payload = {
            'response_id': 'resp-scope-recovery',
            'runtime': {
                'request_phase_graph': graph,
                'developer_diagnostics': {
                    'redraw_scope_ladder_review': first,
                },
            },
        }

        recovered = build_redraw_scope_ladder_review(
            response_payload=recovered_payload,
            request_phase_graph=graph,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(first['selected_scope'], 'observe')
        self.assertEqual(recovered['selected_scope'], 'partial_subtree_rebase')
        self.assertEqual(recovered['forbidden_evidence_seen'], [])

    def test_runtime_attaches_scope_review_before_graph_repair_lifecycle(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-scope',
            'lifecycle_state': 'late_fill_running',
            'runtime': {
                'graph_closure_review': self._closure_review(),
                'request_phase_graph': self._base_graph(),
            },
        }

        updated = owner._attach_redraw_scope_ladder_review(payload)

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(graph['redraw_scope_ladder_review']['selected_scope'], 'add_missing_branch')
        self.assertEqual(
            diagnostics['redraw_scope_ladder_review']['kind'],
            REDRAW_SCOPE_REVIEW_KIND,
        )

    def test_graph_repair_proposal_consumes_scope_as_orientation_only(self):
        from ollmo_services.graph_repair import (
            build_graph_repair_proposals_from_runtime_evidence,
            validate_graph_repair_proposal,
        )

        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = build_redraw_scope_ladder_review(
            response_payload={'response_id': 'resp-scope-repair'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )

        proposals = build_graph_repair_proposals_from_runtime_evidence(
            response_frame={'response_id': 'resp-scope-repair'},
            request_phase_graph=graph,
            closure_review=self._closure_review(),
            late_fill={
                'materialization_contract_unmet': True,
                'final_materialization_contract_status': 'unmet',
            },
        )
        proposal = proposals[0]
        review = validate_graph_repair_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=self._closure_review(),
        )

        self.assertEqual(
            proposal['redraw_scope_orientation']['selected_scope'],
            'add_missing_branch',
        )
        self.assertEqual(
            proposal['redraw_scope_orientation']['allowed_use'],
            'orientation_only_not_patch_authority',
        )
        self.assertEqual(proposal['redraw_scope_orientation']['runtime_effect'], 'none')
        self.assertEqual(review['status'], 'accepted')

    def test_partial_rebase_scope_fields_survive_graph_rebase_proposal(self):
        from ollmo_services.graph_rebase import build_graph_rebase_proposal

        graph = self._base_graph()
        candidate = copy.deepcopy(graph)
        candidate['phases'].append(
            {
                'phase_id': 'phase-image-review',
                'branch_id': 'branch-image-review',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
                'lineage': {'parent_phase_id': 'phase-image', 'relation': 'split_branch'},
            }
        )

        proposal = build_graph_rebase_proposal(
            request_phase_graph=graph,
            candidate_graph=candidate,
            target_response_id='resp-scope',
            target_frame_id='frame-scope',
            source='runtime_closure_review',
            evidence_refs=['closure:partial_subtree_rebase'],
            requested_rebase_class='partial_subtree_rebase',
            scope_root_ids=['phase-image', 'phase-html'],
            preserve_outside_scope=True,
        )

        self.assertEqual(proposal['requested_rebase_class'], 'partial_subtree_rebase')
        self.assertEqual(proposal['scope_root_ids'], ['phase-image', 'phase-html'])
        self.assertTrue(proposal['preserve_outside_scope'])


if __name__ == '__main__':
    unittest.main()
