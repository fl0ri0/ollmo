import copy
import unittest
from unittest.mock import patch

from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_g.request_ir import build_request_ir
from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_services.graph_rebase import (
    GRAPH_REBASE_LIFECYCLE_KIND,
    GRAPH_REBASE_PROPOSAL_KIND,
    build_graph_rebase_execution_contract_proof,
    build_graph_rebase_diff,
    build_graph_rebase_lifecycle,
    build_graph_rebase_preservation_proof,
    build_graph_rebase_proposal,
    describe_graph_rebase_autonomy_from_env,
    graph_rebase_autonomy_from_env,
    normalize_graph_rebase_autonomy,
    apply_validated_graph_rebase,
    validate_graph_rebase_proposal,
)


class GraphRebaseReviewTests(unittest.TestCase):
    ROOT_PROMPT = 'Create the linked local image site from the original request.'

    def test_graph_rebase_autonomy_product_default_and_fail_closed_overrides(self):
        self.assertEqual(normalize_graph_rebase_autonomy(None), 'off')
        self.assertEqual(graph_rebase_autonomy_from_env({}), 'shadow')

        product_default = describe_graph_rebase_autonomy_from_env({})
        self.assertEqual(product_default['autonomy_level'], 'shadow')
        self.assertEqual(product_default['normalized'], 'shadow')
        self.assertEqual(product_default['source'], 'product_default')
        self.assertFalse(product_default['configured'])

        explicit_off = describe_graph_rebase_autonomy_from_env(
            {'OLLMO_GRAPH_REBASE_AUTONOMY': 'off'}
        )
        self.assertEqual(explicit_off['autonomy_level'], 'off')
        self.assertEqual(explicit_off['source'], 'environment')
        self.assertTrue(explicit_off['configured'])
        self.assertFalse(explicit_off['invalid_value'])

        invalid = describe_graph_rebase_autonomy_from_env(
            {'OLLMO_GRAPH_REBASE_AUTONOMY': 'full_auto_redraw_now'}
        )
        self.assertEqual(invalid['autonomy_level'], 'off')
        self.assertEqual(invalid['source'], 'environment')
        self.assertTrue(invalid['configured'])
        self.assertTrue(invalid['invalid_value'])

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
            'graph_version': 3,
            'kind': 'ollmo.request_phase_graph',
            'response_id': 'resp-base',
            'frame_id': 'frame-base',
            'lifecycle_state': 'completed',
            'current_phase_id': 'phase-1',
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'obligation_id': 'obligation-phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-image-1',
                    'branch_id': 'branch-image-1',
                    'obligation_id': 'obligation-image-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                    'artifact_refs': ['artifact://image-1'],
                    'depends_on': ['phase-1'],
                },
                {
                    'phase_id': 'phase-image-2',
                    'branch_id': 'branch-image-2',
                    'obligation_id': 'obligation-image-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                    'artifact_refs': ['artifact://image-2'],
                    'depends_on': ['phase-1'],
                },
                {
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'obligation_id': 'obligation-html',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'text_artifact_source_name': 'index',
                    'text_artifact_extension': 'html',
                    'status': 'pending',
                    'depends_on': ['phase-1'],
                    'target_path': 'artifacts/documents/index.html',
                },
                {
                    'phase_id': 'phase-css',
                    'branch_id': 'branch-css',
                    'obligation_id': 'obligation-css',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'text_artifact_source_name': 'styles',
                    'text_artifact_extension': 'css',
                    'status': 'pending',
                    'depends_on': ['phase-1'],
                    'target_path': 'artifacts/documents/styles.css',
                    'repair_contract_id': 'repair-styles',
                    'repair_target_path': 'artifacts/documents/styles.css',
                },
            ],
            'downstream_branches': [
                {
                    'phase_id': 'phase-image-1',
                    'branch_id': 'branch-image-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                },
                {
                    'phase_id': 'phase-image-2',
                    'branch_id': 'branch-image-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                },
                {
                    'phase_id': 'phase-html',
                    'branch_id': 'branch-html',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'text_artifact_source_name': 'index',
                    'text_artifact_extension': 'html',
                    'depends_on': ['phase-1'],
                },
                {
                    'phase_id': 'phase-css',
                    'branch_id': 'branch-css',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'text_artifact_source_name': 'styles',
                    'text_artifact_extension': 'css',
                    'depends_on': ['phase-1'],
                },
            ],
            'intent_obligations': [
                {
                    'obligation_id': 'intent-image-1',
                    'kind': 'media_artifact',
                    'required': True,
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'phase_id': 'phase-image-1',
                },
                {
                    'obligation_id': 'intent-image-2',
                    'kind': 'media_artifact',
                    'required': True,
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'phase_id': 'phase-image-2',
                },
                {
                    'obligation_id': 'intent-html',
                    'kind': 'text_artifact',
                    'required': True,
                    'target_name': 'index',
                    'target_extension': 'html',
                    'phase_id': 'phase-html',
                },
                {
                    'obligation_id': 'intent-css',
                    'kind': 'text_artifact',
                    'required': True,
                    'target_name': 'styles',
                    'target_extension': 'css',
                    'phase_id': 'phase-css',
                },
                {
                    'obligation_id': 'intent-local-image-binding',
                    'kind': 'dependency',
                    'required': True,
                    'dependency_contract': 'local_visual_asset_binding',
                    'execution_dependency_required': True,
                    'target_phase_id': 'phase-html',
                    'source_phase_ids': ['phase-image-1', 'phase-image-2'],
                },
            ],
            'output_obligations': [
                {
                    'obligation_id': 'obligation-image-1',
                    'phase_id': 'phase-image-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                    'artifact_refs': ['artifact://image-1'],
                },
                {
                    'obligation_id': 'obligation-image-2',
                    'phase_id': 'phase-image-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                    'artifact_refs': ['artifact://image-2'],
                },
                {
                    'obligation_id': 'obligation-html',
                    'phase_id': 'phase-html',
                    'capability': 'chat',
                    'output_type': 'text',
                    'required': True,
                    'target_path': 'artifacts/documents/index.html',
                },
                {
                    'obligation_id': 'obligation-css',
                    'phase_id': 'phase-css',
                    'capability': 'chat',
                    'output_type': 'text',
                    'required': True,
                    'target_path': 'artifacts/documents/styles.css',
                    'repair_contract_id': 'repair-styles',
                    'repair_target_path': 'artifacts/documents/styles.css',
                },
            ],
            'global_semantic_closure_review': {
                'status': 'pending',
                'semantic_review_required': True,
            },
        }

    def _candidate_graph(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['frame_id'] = 'frame-rebase-candidate'
        candidate['phases'] = [dict(item) for item in candidate['phases']]
        for phase in candidate['phases']:
            if phase['phase_id'] == 'phase-html':
                phase['depends_on'] = ['phase-1', 'phase-image-1', 'phase-image-2', 'phase-css']
            if phase['phase_id'] == 'phase-css':
                phase['depends_on'] = ['phase-1', 'phase-image-1', 'phase-image-2']
        candidate['phases'].append(
            {
                'phase_id': 'phase-html-review',
                'branch_id': 'branch-html-review',
                'obligation_id': 'obligation-html-review',
                'capability': 'chat',
                'output_type': 'text',
                'kind': 'review',
                'status': 'pending',
                'depends_on': ['phase-html'],
                'lineage': {'parent_phase_id': 'phase-html', 'relation': 'split_branch'},
            }
        )
        candidate['downstream_branches'] = [
            dict(item) for item in candidate['downstream_branches']
        ]
        for branch in candidate['downstream_branches']:
            if branch['phase_id'] == 'phase-html':
                branch['depends_on'] = ['phase-1', 'phase-image-1', 'phase-image-2', 'phase-css']
            if branch['phase_id'] == 'phase-css':
                branch['depends_on'] = ['phase-1', 'phase-image-1', 'phase-image-2']
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-html-review',
                'branch_id': 'branch-html-review',
                'capability': 'chat',
                'output_type': 'text',
                'kind': 'review',
                'depends_on': ['phase-html'],
                'lineage': {'parent_phase_id': 'phase-html', 'relation': 'split_branch'},
            }
        )
        candidate['output_obligations'].append(
            {
                'obligation_id': 'obligation-html-review',
                'phase_id': 'phase-html-review',
                'capability': 'chat',
                'output_type': 'text',
                'required': True,
                'lineage': {'parent_obligation_id': 'obligation-html', 'relation': 'split_branch'},
            }
        )
        return candidate

    def _proposal(self, *, candidate_graph=None, source='runtime_closure_review', evidence_refs=None, authorization=None):
        base_graph = self._base_graph()
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base_graph,
            candidate_graph=candidate_graph or self._candidate_graph(),
            target_response_id='resp-base',
            target_frame_id='frame-base',
            source=source,
            reason='Reviewed graph rebase candidate from runtime evidence.',
            intent_anchor={'prompt': 'create linked local image site'},
            evidence_refs=evidence_refs or ['closure:intent_graph_adequacy', 'semantic_review:verdict'],
        )
        if authorization is not None:
            proposal['graph_rebase_authorization'] = authorization
        return proposal

    def _accepted_review(self, *, authorization=None):
        proposal = self._proposal(authorization=authorization)
        return validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

    def _authorization(self, proposal_id='*', candidate_graph_digest='*'):
        return {
            'kind': 'ollmo.graph_rebase_authorization',
            'status': 'accepted',
            'authority': 'operator_review',
            'source': 'runtime_operator_registry',
            'registry_record_id': 'graph-rebase-operator-record-test',
            'allowed_autonomy': ['apply_reviewed'],
            'proposal_id': proposal_id,
            'candidate_graph_digest': candidate_graph_digest,
            'requested_rebase_class': 'partial_subtree_rebase',
            'evidence_refs': ['operator:reviewed-rebase'],
        }

    def _executable_partial_candidate_graph(self):
        candidate = copy.deepcopy(self._base_graph())
        local_payload = (
            'Review the completed HTML artifact for exact local image bindings. '
            'Return a concise branch-local verdict only.'
        )
        local_contract = {
            'kind': 'ollmo.execution_contract',
            'execution_scope': 'branch_local',
            'root_scoped': False,
            'allow_root_prompt': False,
            'input_refs': [
                {'kind': 'content_payload', 'source': 'runtime_partial_rebase_review'},
                {'kind': 'phase_output', 'phase_id': 'phase-html', 'role': 'dependency'},
            ],
        }
        branch = {
            'phase_id': 'phase-html-review',
            'branch_id': 'branch-html-review',
            'obligation_id': 'obligation-html-review',
            'capability': 'chat',
            'output_type': 'text',
            'kind': 'review',
            'status': 'pending',
            'depends_on': ['phase-html'],
            'content_payload': local_payload,
            'content_payload_source': 'runtime_partial_rebase_review',
            'input_refs': copy.deepcopy(local_contract['input_refs']),
            'execution_contract': copy.deepcopy(local_contract),
            'lineage': {'parent_phase_id': 'phase-html', 'relation': 'split_branch'},
        }
        candidate['phases'].append(copy.deepcopy(branch))
        candidate['downstream_branches'].append(copy.deepcopy(branch))
        candidate['output_obligations'].append(
            {
                'obligation_id': 'obligation-html-review',
                'phase_id': 'phase-html-review',
                'branch_id': 'branch-html-review',
                'capability': 'chat',
                'output_type': 'text',
                'required': True,
                'lineage': {
                    'parent_obligation_id': 'obligation-html',
                    'relation': 'split_branch',
                },
            }
        )
        return candidate

    def _executable_partial_proposal(self, *, authorization=None):
        proposal = build_graph_rebase_proposal(
            request_phase_graph=self._base_graph(),
            candidate_graph=self._executable_partial_candidate_graph(),
            target_response_id='resp-base',
            target_frame_id='frame-base',
            source='runtime_closure_review',
            reason='Review one bounded HTML subtree from current Closure evidence.',
            intent_anchor={'prompt': 'create linked local image site'},
            evidence_refs=['closure:partial_subtree_rebase'],
            requested_rebase_class='partial_subtree_rebase',
            scope_root_ids=['phase-html', 'phase-html-review', 'branch-html-review'],
            scope_phase_ids=['phase-html', 'phase-html-review'],
            scope_branch_ids=['branch-html', 'branch-html-review'],
            preserve_outside_scope=True,
            root_prompt=self.ROOT_PROMPT,
        )
        if authorization is not None:
            proposal['graph_rebase_authorization'] = copy.deepcopy(authorization)
        return proposal

    def _trusted_partial_authorization(self, proposal):
        return self._authorization(
            proposal_id=proposal['proposal_id'],
            candidate_graph_digest=proposal['candidate_graph_digest'],
        )

    def _real_builder_graph(self):
        prompt = 'Create an HTML page with one generated image.'
        return build_request_phase_graph(
            prompt,
            request_payload={'prompt': prompt, 'ghost_route': True},
            route_payload={'capability': 'chat'},
            response_payload={'output_text': 'Prepared the page copy.'},
        )

    def test_graph_rebase_diff_detects_preserve_add_rebind_supersede(self):
        diff = build_graph_rebase_diff(self._base_graph(), self._candidate_graph())
        operation_types = {item['op'] for item in diff['operations']}

        self.assertEqual(diff['kind'], 'ollmo.graph_rebase_diff')
        self.assertIn('preserve_phase', operation_types)
        self.assertIn('add_phase', operation_types)
        self.assertIn('add_dependency', operation_types)
        self.assertIn('split_branch', operation_types)
        self.assertEqual(diff['removed_without_preservation'], [])
        self.assertFalse({'delete_phase', 'delete_branch', 'delete_obligation'} & operation_types)

    def test_real_builder_schema_accepts_consistent_structural_projection_addition(self):
        prompt = 'Create an HTML page with one generated image.'
        base = build_request_phase_graph(
            prompt,
            request_payload={'prompt': prompt, 'ghost_route': True},
            route_payload={'capability': 'chat'},
            response_payload={'output_text': 'Prepared the page copy.'},
        )
        candidate = copy.deepcopy(base)
        new_phase = {
            'phase_id': 'phase-structural-review',
            'branch_id': 'branch-structural-review',
            'obligation_id': 'obligation-structural-review',
            'kind': 'review',
            'role': 'semantic_review',
            'capability': 'chat',
            'output_type': 'text',
            'status': 'pending',
            'depends_on': ['phase-1'],
        }
        new_branch = {
            'phase_id': 'phase-structural-review',
            'branch_id': 'branch-structural-review',
            'obligation_id': 'obligation-structural-review',
            'capability': 'chat',
            'output_type': 'text',
            'status': 'pending',
            'depends_on': ['phase-1'],
        }
        candidate['phases'].append(new_phase)
        candidate['downstream_branches'].append(new_branch)
        current_phase = next(
            item for item in candidate['phases'] if item['phase_id'] == 'phase-1'
        )
        current_phase['downstream_phase_ids'] = [
            *current_phase.get('downstream_phase_ids', []),
            'phase-structural-review',
        ]
        candidate['downstream_phase_ids'] = [
            *candidate.get('downstream_phase_ids', []),
            'phase-structural-review',
        ]
        candidate['downstream_branch_ids'] = [
            *candidate.get('downstream_branch_ids', []),
            'branch-structural-review',
        ]
        candidate['downstream_capabilities'] = list(
            dict.fromkeys([*candidate.get('downstream_capabilities', []), 'chat'])
        )
        rebuilt_ir = build_request_ir(
            intent_prompt=candidate['request_ir']['intent_anchor'],
            prompt_intent=candidate['prompt_intent'],
            phases=candidate['phases'],
            current_phase_id=candidate['current_phase_id'],
            graph_mode=candidate['mode'],
        )
        candidate['request_ir'] = rebuilt_ir
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
            candidate[key] = copy.deepcopy(rebuilt_ir.get(key) or ([] if key.endswith('s') else {}))
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:full-successor-structural-review'],
            requested_rebase_class='full_successor_rebase',
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(review['diff']['semantic_changes'], [])
        self.assertEqual(review['diff']['top_level_semantic_changes'], [])
        self.assertTrue(review['diff']['derived_projection_changes'])

    def test_real_builder_rejects_request_ir_final_output_projection_mismatch(self):
        base = self._real_builder_graph()
        candidate = copy.deepcopy(base)
        candidate['request_ir']['final_output_obligation_ids'] = [
            'obligation-phase-1'
        ]
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:final-output-projection-review'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_derived_projection_inconsistent',
            review['blocked_reasons'],
        )
        self.assertTrue(
            any(
                item.get('field') == 'request_ir.final_output_obligation_ids'
                and item.get('reason') == 'projection_mismatch'
                for item in review['candidate_projection_consistency_issues']
            )
        )

    def test_real_builder_rejects_unknown_workload_task_authority(self):
        base = self._real_builder_graph()
        candidate = copy.deepcopy(base)
        candidate['request_ir']['workload_graph']['tasks'][0]['authority'] = (
            'execute_without_review'
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:workload-projection-review'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_derived_projection_inconsistent',
            review['blocked_reasons'],
        )
        self.assertTrue(
            any(
                item.get('field') == 'request_ir.workload_graph.tasks'
                and item.get('reason') == 'unknown_task_projection_fields'
                and item.get('unknown_fields') == ['authority']
                for item in review['candidate_projection_consistency_issues']
            )
        )

    def test_real_builder_rejects_fabricated_output_candidate_artifact_ref(self):
        base = self._real_builder_graph()
        candidate = copy.deepcopy(base)
        fabricated_candidate = {
            'candidate_id': 'candidate-fabricated-artifact',
            'phase_id': 'phase-2',
            'branch_id': 'branch-image_generation-1',
            'capability': 'image_generation',
            'output_type': 'image',
            'artifact_refs': ['artifact://fabricated'],
        }
        candidate['request_ir']['output_candidates'] = [
            copy.deepcopy(fabricated_candidate)
        ]
        candidate['output_candidates'] = [copy.deepcopy(fabricated_candidate)]
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:output-candidate-projection-review'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_derived_projection_inconsistent',
            review['blocked_reasons'],
        )
        self.assertTrue(
            any(
                item.get('field') == 'request_ir.output_candidates'
                and item.get('reason') == 'unknown_candidate_projection_fields'
                and item.get('candidate_id') == 'candidate-fabricated-artifact'
                and item.get('unknown_fields') == ['artifact_refs']
                for item in review['candidate_projection_consistency_issues']
            )
        )

    def test_real_builder_rejects_orphan_phase_without_branch_or_obligation(self):
        base = self._real_builder_graph()
        candidate = copy.deepcopy(base)
        candidate['phases'].append(
            {
                'phase_id': 'phase-orphan',
                'branch_id': 'branch-orphan',
                'obligation_id': 'obligation-orphan',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
                'depends_on': ['phase-1'],
            }
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:orphan-phase-review'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        issue_reasons = {
            item.get('reason')
            for item in review['candidate_projection_consistency_issues']
            if item.get('phase_id') == 'phase-orphan'
        }
        self.assertIn('phase_branch_cardinality_mismatch', issue_reasons)
        self.assertIn('phase_obligation_cardinality_mismatch', issue_reasons)

    def test_graph_rebase_preservation_rejects_lost_intent_obligation(self):
        candidate = self._candidate_graph()
        candidate['intent_obligations'] = [
            item for item in candidate['intent_obligations']
            if item['obligation_id'] != 'intent-image-2'
        ]
        diff = build_graph_rebase_diff(self._base_graph(), candidate)
        proof = build_graph_rebase_preservation_proof(self._base_graph(), candidate, diff)

        self.assertIn(proof['status'], {'blocked', 'rejected'})
        self.assertIn('lost_required_intent_obligation', proof['blocked_reasons'])
        self.assertTrue(
            any(item.get('id') == 'intent-image-2' for item in proof['lost_items'])
        )

    def test_graph_rebase_preservation_rejects_lost_artifact_ref(self):
        candidate = self._candidate_graph()
        for phase in candidate['phases']:
            phase.pop('artifact_refs', None)
        for obligation in candidate['output_obligations']:
            obligation.pop('artifact_refs', None)
        diff = build_graph_rebase_diff(self._base_graph(), candidate)
        proof = build_graph_rebase_preservation_proof(self._base_graph(), candidate, diff)

        self.assertIn(proof['status'], {'blocked', 'rejected'})
        self.assertIn('lost_artifact_ref', proof['blocked_reasons'])
        self.assertIn('artifact://image-1', proof['artifact_ref_preservation_summary']['lost'])

    def test_graph_rebase_review_blocks_identical_noop_candidate(self):
        proposal = self._proposal(candidate_graph=self._base_graph())

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('candidate_graph_has_no_meaningful_change', review['blocked_reasons'])

    def test_graph_rebase_preservation_rejects_removed_dependency_edge(self):
        candidate = copy.deepcopy(self._base_graph())
        for phase in candidate['phases']:
            if phase['phase_id'] == 'phase-image-1':
                phase['depends_on'] = []
        for branch in candidate['downstream_branches']:
            if branch['branch_id'] == 'branch-image-1':
                branch['depends_on'] = []

        diff = build_graph_rebase_diff(self._base_graph(), candidate)
        proof = build_graph_rebase_preservation_proof(self._base_graph(), candidate, diff)

        self.assertEqual(proof['status'], 'blocked')
        self.assertIn('lost_dependency_edge', proof['blocked_reasons'])

    def test_graph_rebase_preservation_does_not_treat_an_unproven_replacement_edge_as_preserved(self):
        candidate = copy.deepcopy(self._base_graph())
        for phase in candidate['phases']:
            if phase['phase_id'] == 'phase-image-1':
                phase['depends_on'] = ['phase-css']
        for branch in candidate['downstream_branches']:
            if branch['branch_id'] == 'branch-image-1':
                branch['depends_on'] = ['phase-css']

        diff = build_graph_rebase_diff(self._base_graph(), candidate)
        proof = build_graph_rebase_preservation_proof(self._base_graph(), candidate, diff)

        self.assertTrue(diff['rebound_dependency_edges'])
        self.assertEqual(proof['status'], 'blocked')
        self.assertIn('lost_dependency_edge', proof['blocked_reasons'])

    def test_graph_rebase_preservation_rejects_same_id_semantic_change(self):
        candidate = copy.deepcopy(self._base_graph())
        for collection in ('phases', 'downstream_branches'):
            for record in candidate[collection]:
                if record.get('phase_id') == 'phase-image-1':
                    record['capability'] = 'chat'
                    record['output_type'] = 'text'
        for obligation in candidate['intent_obligations']:
            if obligation['obligation_id'] == 'intent-image-1':
                obligation['capability'] = 'chat'
                obligation['output_type'] = 'text'
        for obligation in candidate['output_obligations']:
            if obligation['obligation_id'] == 'obligation-image-1':
                obligation['capability'] = 'chat'
                obligation['output_type'] = 'text'

        diff = build_graph_rebase_diff(self._base_graph(), candidate)
        proof = build_graph_rebase_preservation_proof(self._base_graph(), candidate, diff)

        self.assertEqual(proof['status'], 'blocked')
        self.assertIn('changed_preserved_record_meaning', proof['blocked_reasons'])

    def test_graph_rebase_preserves_same_obligation_id_in_both_collections(self):
        base = self._base_graph()
        base['intent_obligations'].append(
            {
                'obligation_id': 'shared-obligation',
                'kind': 'media_artifact',
                'required': True,
                'capability': 'image_generation',
                'phase_id': 'phase-image-1',
            }
        )
        base['output_obligations'].append(
            {
                'obligation_id': 'shared-obligation',
                'required': True,
                'capability': 'image_generation',
                'phase_id': 'phase-image-1',
            }
        )
        candidate = copy.deepcopy(base)
        next(
            item
            for item in candidate['intent_obligations']
            if item['obligation_id'] == 'shared-obligation'
        )['capability'] = 'chat'
        candidate['phases'].append(
            {
                'phase_id': 'phase-review-shared',
                'branch_id': 'branch-review-shared',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-review-shared',
                'branch_id': 'branch-review-shared',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:shared-obligation-review'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('changed_preserved_record_meaning', review['blocked_reasons'])
        self.assertTrue(
            any(
                item.get('id') == 'intent_obligations:shared-obligation'
                for item in review['diff']['semantic_changes']
            )
        )

    def test_graph_rebase_rejects_duplicate_candidate_record_ids(self):
        candidate = self._candidate_graph()
        candidate['phases'].insert(
            0,
            {
                'phase_id': 'phase-1',
                'branch_id': 'phase-1-malicious-duplicate',
                'capability': 'image_generation',
                'output_type': 'image',
            },
        )
        proposal = self._proposal(candidate_graph=candidate)

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('candidate_graph_duplicate_record_id', review['blocked_reasons'])
        self.assertEqual(review['candidate_duplicate_ids']['phases'], ['phase-1'])

    def test_graph_rebase_rejects_dangling_candidate_dependency_source(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['phases'].append(
            {
                'phase_id': 'phase-dangling',
                'branch_id': 'branch-dangling',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-does-not-exist'],
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-dangling',
                'branch_id': 'branch-dangling',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-does-not-exist'],
            }
        )
        proposal = self._proposal(candidate_graph=candidate)

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_dangling_dependency_source',
            review['blocked_reasons'],
        )

    def test_graph_rebase_rejects_unknown_lineage_parent(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['phases'].append(
            {
                'phase_id': 'phase-unknown-lineage-parent',
                'branch_id': 'branch-unknown-lineage-parent',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {
                    'relation': 'split_branch',
                    'parent_phase_id': 'phase-does-not-exist',
                },
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-unknown-lineage-parent',
                'branch_id': 'branch-unknown-lineage-parent',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        proposal = self._proposal(candidate_graph=candidate)

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_dangling_dependency_source',
            review['blocked_reasons'],
        )
        self.assertIn(
            {
                'target_id': 'phase-unknown-lineage-parent',
                'source_id': 'phase-does-not-exist',
                'relation': 'phase_lineage:parent_phase_id',
            },
            review['candidate_dangling_dependency_edges'],
        )

    def test_graph_rebase_rejects_self_dependency_and_dependency_cycle(self):
        cases = {
            'self_dependency': [
                {
                    'phase_id': 'phase-self-dependent',
                    'branch_id': 'branch-self-dependent',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-self-dependent'],
                }
            ],
            'dependency_cycle': [
                {
                    'phase_id': 'phase-cycle-a',
                    'branch_id': 'branch-cycle-a',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-cycle-b'],
                },
                {
                    'phase_id': 'phase-cycle-b',
                    'branch_id': 'branch-cycle-b',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-cycle-a'],
                },
            ],
        }

        for expected_relation, added_phases in cases.items():
            with self.subTest(expected_relation=expected_relation):
                candidate = copy.deepcopy(self._base_graph())
                candidate['phases'].extend(copy.deepcopy(added_phases))
                candidate['downstream_branches'].extend(
                    copy.deepcopy(added_phases)
                )
                proposal = self._proposal(candidate_graph=candidate)

                review = validate_graph_rebase_proposal(
                    proposal,
                    request_phase_graph=self._base_graph(),
                    closure_review={'status': 'repair_required'},
                )

                self.assertEqual(review['status'], 'rejected')
                self.assertIn(
                    'candidate_graph_dangling_dependency_source',
                    review['blocked_reasons'],
                )
                self.assertTrue(
                    any(
                        item.get('relation') == expected_relation
                        for item in review['candidate_dangling_dependency_edges']
                    )
                )

    def test_graph_rebase_rejects_dangling_obligation_targets_and_obligation_dependencies(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['phases'].append(
            {
                'phase_id': 'phase-valid-addition',
                'branch_id': 'branch-valid-addition',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-valid-addition',
                'branch_id': 'branch-valid-addition',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate['intent_obligations'].append(
            {
                'obligation_id': 'intent-dangling-obligation-dependency',
                'phase_id': 'phase-valid-addition',
                'required': True,
                'depends_on_obligation_ids': ['obligation-does-not-exist'],
            }
        )
        candidate['output_obligations'].append(
            {
                'obligation_id': 'obligation-dangling-target',
                'phase_id': 'phase-does-not-exist',
                'required': True,
            }
        )
        proposal = self._proposal(candidate_graph=candidate)

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        relations = {
            item.get('relation')
            for item in review['candidate_dangling_dependency_edges']
        }
        self.assertIn('output_obligations:phase_id', relations)
        self.assertIn('intent_obligations:depends_on_obligation_ids', relations)

    def test_graph_rebase_preserves_hidden_failure_visibility(self):
        base = self._base_graph()
        failed_phase = next(
            item for item in base['phases'] if item['phase_id'] == 'phase-image-1'
        )
        failed_phase['status'] = 'failed'
        failed_phase['last_error'] = 'provider returned corrupt bytes'
        candidate = copy.deepcopy(base)
        candidate_phase = next(
            item for item in candidate['phases'] if item['phase_id'] == 'phase-image-1'
        )
        candidate_phase['status'] = 'pending'
        candidate_phase.pop('last_error')
        candidate['phases'].append(
            {
                'phase_id': 'phase-failure-review',
                'branch_id': 'branch-failure-review',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-failure-review',
                'branch_id': 'branch-failure-review',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:hidden-failure-review'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('hidden_failure_visibility_lost', review['blocked_reasons'])

    def test_graph_rebase_review_rejects_mismatched_candidate_digest(self):
        proposal = self._proposal()
        proposal['candidate_graph_digest'] = 'graph-declared-but-wrong'

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('candidate_graph_digest_mismatch', review['blocked_reasons'])

    def test_graph_rebase_review_rejects_bookkeeping_smuggled_after_digest(self):
        proposal = self._proposal()
        proposal['candidate_graph']['successor_rebase_requests'] = [
            {
                'kind': 'ollmo.graph_rebase_successor_request',
                'runtime_effect': 'successor_rebase_created',
            }
        ]

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_contains_rebase_bookkeeping',
            review['blocked_reasons'],
        )

    def test_partial_rebase_cannot_change_global_semantic_review_contract(self):
        base = self._base_graph()
        candidate = copy.deepcopy(base)
        candidate['global_semantic_closure_review']['semantic_review_required'] = False
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:semantic_review_contract'],
            requested_rebase_class='partial_subtree_rebase',
            scope_phase_ids=['phase-html'],
            scope_branch_ids=['branch-html'],
            preserve_outside_scope=True,
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('partial_rebase_changes_outside_scope', review['blocked_reasons'])
        self.assertIn('changed_preserved_graph_meaning', review['blocked_reasons'])
        self.assertEqual(
            review['diff']['top_level_semantic_changes'][0]['changed_fields'],
            ['global_semantic_closure_review'],
        )

    def test_partial_rebase_rejects_continuation_projection_without_structural_change(self):
        base = self._base_graph()
        base['continuation_required'] = True
        candidate = copy.deepcopy(base)
        candidate['continuation_required'] = False
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:continuation_contract'],
            requested_rebase_class='partial_subtree_rebase',
            scope_phase_ids=['phase-html'],
            scope_branch_ids=['branch-html'],
            preserve_outside_scope=True,
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn(
            'candidate_graph_derived_projection_inconsistent',
            review['blocked_reasons'],
        )
        self.assertIn(
            'derived_projection_changed_without_structural_diff',
            review['blocked_reasons'],
        )
        self.assertEqual(review['diff']['top_level_semantic_changes'], [])
        self.assertIn(
            'continuation_required',
            review['diff']['derived_projection_changes'][0]['changed_fields'],
        )

    def test_partial_rebase_scope_contains_superseded_parent(self):
        base = self._base_graph()
        base['phases'].append(
            {
                'phase_id': 'phase-outside-parent',
                'branch_id': 'branch-outside-parent',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        base['downstream_branches'].append(
            {
                'phase_id': 'phase-outside-parent',
                'branch_id': 'branch-outside-parent',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate = copy.deepcopy(base)
        candidate['phases'] = [
            item for item in candidate['phases']
            if item['phase_id'] != 'phase-outside-parent'
        ]
        candidate['downstream_branches'] = [
            item for item in candidate['downstream_branches']
            if item['branch_id'] != 'branch-outside-parent'
        ]
        candidate['phases'].append(
            {
                'phase_id': 'phase-inside-replacement',
                'branch_id': 'branch-inside-replacement',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {
                    'relation': 'supersedes',
                    'parent_phase_id': 'phase-outside-parent',
                },
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-inside-replacement',
                'branch_id': 'branch-inside-replacement',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {
                    'relation': 'split_branch',
                    'parent_branch_id': 'branch-outside-parent',
                },
            }
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:partial-supersession'],
            requested_rebase_class='partial_subtree_rebase',
            scope_phase_ids=['phase-inside-replacement'],
            scope_branch_ids=['branch-inside-replacement'],
            preserve_outside_scope=True,
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('partial_rebase_changes_outside_scope', review['blocked_reasons'])

    def test_partial_rebase_scope_contains_split_parent_branch(self):
        base = self._base_graph()
        base['downstream_branches'].append(
            {
                'phase_id': 'phase-html',
                'branch_id': 'branch-outside-parent-only',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate = copy.deepcopy(base)
        candidate['downstream_branches'] = [
            item for item in candidate['downstream_branches']
            if item['branch_id'] != 'branch-outside-parent-only'
        ]
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-html',
                'branch_id': 'branch-inside-child-only',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {
                    'relation': 'split_branch',
                    'parent_branch_id': 'branch-outside-parent-only',
                },
            }
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:partial-branch-split'],
            requested_rebase_class='partial_subtree_rebase',
            scope_phase_ids=['phase-html'],
            scope_branch_ids=['branch-inside-child-only'],
            preserve_outside_scope=True,
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('partial_rebase_changes_outside_scope', review['blocked_reasons'])

    def test_partial_merge_cannot_remove_phase_or_branch_outside_declared_scope(self):
        base = self._base_graph()
        base['phases'].append(
            {
                'phase_id': 'phase-outside-merge',
                'branch_id': 'branch-outside-merge',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        base['downstream_branches'].append(
            {
                'phase_id': 'phase-outside-merge',
                'branch_id': 'branch-outside-merge',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
            }
        )
        candidate = copy.deepcopy(base)
        candidate['phases'] = [
            item
            for item in candidate['phases']
            if item['phase_id'] != 'phase-outside-merge'
        ]
        candidate['downstream_branches'] = [
            item
            for item in candidate['downstream_branches']
            if item['branch_id'] != 'branch-outside-merge'
        ]
        candidate['phases'].append(
            {
                'phase_id': 'phase-inside-merge',
                'branch_id': 'branch-inside-merge',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {
                    'relation': 'merge_branches',
                    'parent_phase_id': 'phase-outside-merge',
                },
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-inside-merge',
                'branch_id': 'branch-inside-merge',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-1'],
                'lineage': {
                    'relation': 'merge_branches',
                    'parent_branch_id': 'branch-outside-merge',
                },
            }
        )
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:partial-merge-review'],
            requested_rebase_class='partial_subtree_rebase',
            scope_phase_ids=['phase-inside-merge'],
            scope_branch_ids=['branch-inside-merge'],
            preserve_outside_scope=True,
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('partial_rebase_changes_outside_scope', review['blocked_reasons'])
        violation_reasons = {
            item.get('reason') for item in review['partial_scope_violations']
        }
        self.assertIn('removed_phase_outside_declared_scope', violation_reasons)
        self.assertIn('removed_branch_outside_declared_scope', violation_reasons)
        self.assertGreaterEqual(
            sum(
                item.get('op') == 'merge_with_replacement'
                for item in review['diff']['meaningful_operations']
            ),
            2,
        )

    def test_graph_rebase_review_blocks_partial_candidate_changes_outside_scope(self):
        candidate = self._candidate_graph()
        proposal = build_graph_rebase_proposal(
            request_phase_graph=self._base_graph(),
            candidate_graph=candidate,
            target_response_id='resp-base',
            target_frame_id='frame-base',
            source='runtime_closure_review',
            reason='Reviewed partial graph rebase candidate from runtime evidence.',
            intent_anchor={'prompt': 'create linked local image site'},
            evidence_refs=['closure:intent_graph_adequacy', 'semantic_review:verdict'],
            requested_rebase_class='partial_subtree_rebase',
            scope_root_ids=['phase-html'],
            scope_phase_ids=['phase-html', 'phase-html-review'],
            scope_branch_ids=['branch-html', 'branch-html-review'],
            preserve_outside_scope=True,
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'blocked')
        self.assertIn('partial_rebase_changes_outside_scope', review['blocked_reasons'])

    def test_graph_rebase_review_rejects_learning_only_evidence(self):
        proposal = self._proposal(
            source='accepted_learning',
            evidence_refs=['accepted_learning:case-1'],
        )
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            accepted_learning_hints={'hint_count': 1},
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('accepted_learning_not_rebase_authority', review['blocked_reasons'])

    def test_graph_rebase_review_requires_authorization_for_apply_reviewed(self):
        proposal = self._executable_partial_proposal()
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
        )

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertIn('apply_reviewed_requires_explicit_rebase_authorization', lifecycle['blocked_reasons'])
        self.assertEqual(lifecycle['outcome']['runtime_effect'], 'none')

    def test_partial_execution_contract_proof_requires_branch_local_payload(self):
        proposal = build_graph_rebase_proposal(
            request_phase_graph=self._base_graph(),
            candidate_graph=self._candidate_graph(),
            source='runtime_closure_review',
            evidence_refs=['closure:partial_subtree_rebase'],
            requested_rebase_class='partial_subtree_rebase',
            scope_root_ids=['phase-html', 'phase-html-review', 'branch-html-review'],
            scope_phase_ids=['phase-html', 'phase-html-review'],
            scope_branch_ids=['branch-html', 'branch-html-review'],
            preserve_outside_scope=True,
        )

        proof = build_graph_rebase_execution_contract_proof(
            self._base_graph(),
            proposal,
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(proof['status'], 'blocked')
        self.assertIn('partial_rebase_branch_local_payload_missing', proof['blocked_reasons'])
        self.assertIn('branch-html-review', proof['owed_branch_ids'])

    def test_partial_execution_contract_proof_accepts_exact_local_payload(self):
        proposal = self._executable_partial_proposal()

        proof = build_graph_rebase_execution_contract_proof(
            self._base_graph(),
            proposal,
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(proof['status'], 'passed')
        self.assertEqual(proof['owed_branch_ids'], ['branch-html-review'])
        self.assertEqual(proof['root_prompt_fallback_branch_ids'], [])

    def test_partial_execution_contract_proof_requires_current_root_truth(self):
        proposal = self._executable_partial_proposal()

        proof = build_graph_rebase_execution_contract_proof(
            self._base_graph(),
            proposal,
        )

        self.assertEqual(proof['status'], 'blocked')
        self.assertIn(
            'partial_rebase_current_root_prompt_truth_unavailable',
            proof['blocked_reasons'],
        )
        self.assertFalse(proof['root_prompt_guard_checked'])

    def test_partial_execution_contract_proof_rejects_current_root_drift(self):
        proposal = self._executable_partial_proposal()

        proof = build_graph_rebase_execution_contract_proof(
            self._base_graph(),
            proposal,
            root_prompt='A different current root request.',
        )

        self.assertEqual(proof['status'], 'blocked')
        self.assertIn(
            'partial_rebase_root_prompt_guard_mismatch',
            proof['blocked_reasons'],
        )
        self.assertFalse(proof['root_prompt_guard_checked'])

    def test_partial_execution_contract_proof_checks_all_prompt_carriers(self):
        for carrier in ('phase_summary', 'stage_direction', 'instruct'):
            for source_collection in ('phases', 'downstream_branches'):
                with self.subTest(
                    carrier=carrier,
                    source_collection=source_collection,
                ):
                    candidate = self._executable_partial_candidate_graph()
                    for record in candidate[source_collection]:
                        if record.get('branch_id') == 'branch-html-review':
                            record[carrier] = self.ROOT_PROMPT.swapcase()
                    proposal = build_graph_rebase_proposal(
                        request_phase_graph=self._base_graph(),
                        candidate_graph=candidate,
                        source='runtime_closure_review',
                        evidence_refs=['closure:partial_subtree_rebase'],
                        requested_rebase_class='partial_subtree_rebase',
                        scope_root_ids=[
                            'phase-html',
                            'phase-html-review',
                            'branch-html-review',
                        ],
                        scope_phase_ids=['phase-html', 'phase-html-review'],
                        scope_branch_ids=['branch-html', 'branch-html-review'],
                        preserve_outside_scope=True,
                        root_prompt=self.ROOT_PROMPT,
                    )

                    proof = build_graph_rebase_execution_contract_proof(
                        self._base_graph(),
                        proposal,
                        root_prompt=self.ROOT_PROMPT,
                    )

                    self.assertEqual(proof['status'], 'blocked')
                    self.assertIn(
                        'partial_rebase_root_prompt_fallback_forbidden',
                        proof['blocked_reasons'],
                    )

    def test_partial_execution_contract_proof_rejects_root_prompt_relabelled_local(self):
        root_prompt = 'Create the complete linked local image site from the original request.'
        candidate = self._executable_partial_candidate_graph()
        for collection in ('phases', 'downstream_branches'):
            for record in candidate[collection]:
                if record.get('branch_id') == 'branch-html-review':
                    record['content_payload'] = root_prompt.swapcase()
                    record['content_payload_source'] = 'runtime_partial_rebase_review'
        proposal = build_graph_rebase_proposal(
            request_phase_graph=self._base_graph(),
            candidate_graph=candidate,
            source='runtime_closure_review',
            evidence_refs=['closure:partial_subtree_rebase'],
            requested_rebase_class='partial_subtree_rebase',
            scope_root_ids=['phase-html', 'phase-html-review', 'branch-html-review'],
            scope_phase_ids=['phase-html', 'phase-html-review'],
            scope_branch_ids=['branch-html', 'branch-html-review'],
            preserve_outside_scope=True,
            root_prompt=root_prompt,
        )

        proof = build_graph_rebase_execution_contract_proof(
            self._base_graph(),
            proposal,
            root_prompt=root_prompt,
        )

        self.assertEqual(proof['status'], 'blocked')
        self.assertIn(
            'partial_rebase_root_prompt_fallback_forbidden',
            proof['blocked_reasons'],
        )
        self.assertEqual(
            proof['root_prompt_fallback_branch_ids'],
            ['branch-html-review'],
        )
        self.assertTrue(proof['root_prompt_guard_checked'])

    def test_graph_rebase_apply_reviewed_rejects_wildcard_authorization(self):
        proposal = self._executable_partial_proposal()
        proposal['graph_rebase_authorization'] = self._authorization(
            proposal_id='*',
            candidate_graph_digest='*',
        )
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
        )

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertIn(
            'apply_reviewed_requires_explicit_rebase_authorization',
            lifecycle['blocked_reasons'],
        )

    def test_graph_rebase_validation_authority_cannot_self_authorize_apply(self):
        proposal = self._executable_partial_proposal()
        proposal['graph_rebase_authorization'] = {
            **self._authorization(
                proposal_id=proposal['proposal_id'],
                candidate_graph_digest=proposal['candidate_graph_digest'],
            ),
            'authority': 'runtime_rebase_validation',
        }
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
        )

        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertIn(
            'apply_reviewed_requires_explicit_rebase_authorization',
            lifecycle['blocked_reasons'],
        )

    def test_apply_reviewed_sink_rejects_forged_lifecycle_without_review_or_authorization(self):
        candidate = self._candidate_graph()
        forged_lifecycle = {
            'kind': GRAPH_REBASE_LIFECYCLE_KIND,
            'status': 'staged',
            'autonomy_level': 'apply_reviewed',
            'rebase_id': 'graph-rebase-forged',
            'proposal_id': 'graph-rebase-proposal-forged',
            'idempotency_key': 'graph-rebase-idem-forged',
            'before_graph_digest': self._proposal()['base_graph_digest'],
            'base_graph_digest': self._proposal()['base_graph_digest'],
            'candidate_graph_digest': self._proposal()['candidate_graph_digest'],
            'accepted_successor_graph': candidate,
        }

        result = apply_validated_graph_rebase(
            self._base_graph(),
            forged_lifecycle,
            autonomy_level='apply_reviewed',
        )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn(
            'apply_reviewed_requires_accepted_validation_review',
            result['blocked_reasons'],
        )
        self.assertEqual(result['graph'].get('successor_rebase_requests') or [], [])

    def test_apply_reviewed_sink_replay_respects_current_smaller_redraw_scope(self):
        proposal = self._executable_partial_proposal()
        authorization = self._trusted_partial_authorization(proposal)
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )
        control_result = apply_validated_graph_rebase(
            self._base_graph(),
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        current_graph = self._base_graph()
        current_graph['redraw_scope_ladder_review'] = {
            'kind': 'ollmo.redraw_scope_ladder_review',
            'status': 'selected',
            'selected_scope': 'add_missing_branch',
            'scopes_considered': [
                {
                    'scope': 'add_missing_branch',
                    'eligible': True,
                    'reason': 'missing_promoted_branch_or_obligation',
                }
            ],
        }

        result = apply_validated_graph_rebase(
            current_graph,
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(lifecycle['status'], 'staged')
        self.assertEqual(control_result['status'], 'applied')
        self.assertEqual(result['status'], 'blocked')
        self.assertIn(
            'apply_reviewed_sink_revalidation_failed',
            result['blocked_reasons'],
        )
        self.assertEqual(result['graph'].get('successor_rebase_requests') or [], [])

    def test_apply_reviewed_sink_requires_current_root_truth(self):
        proposal = self._executable_partial_proposal()
        authorization = self._trusted_partial_authorization(proposal)
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )

        result = apply_validated_graph_rebase(
            self._base_graph(),
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn(
            'partial_rebase_current_root_prompt_truth_unavailable',
            result['blocked_reasons'],
        )

    def test_graph_rebase_apply_reviewed_creates_successor_rebase_request(self):
        proposal = self._executable_partial_proposal()
        authorization = self._trusted_partial_authorization(proposal)
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )

        result = apply_validated_graph_rebase(
            self._base_graph(),
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(lifecycle['kind'], GRAPH_REBASE_LIFECYCLE_KIND)
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(result['runtime_effect'], 'successor_rebase_created')
        self.assertEqual(len(result['graph']['phases']), len(self._base_graph()['phases']))
        self.assertEqual(len(result['graph']['successor_rebase_requests']), 1)
        successor = result['graph']['successor_rebase_requests'][0]
        self.assertEqual(successor['kind'], 'ollmo.graph_rebase_successor_request')
        self.assertEqual(successor['lineage']['relation'], 'graph_rebase_successor')
        self.assertEqual(successor['requested_rebase_class'], 'partial_subtree_rebase')
        self.assertEqual(
            successor['successor_graph']['phases'],
            self._executable_partial_candidate_graph()['phases'],
        )

    def test_inline_proposal_authorization_is_not_trusted_at_apply_reviewed(self):
        proposal = self._executable_partial_proposal()
        proposal['graph_rebase_authorization'] = self._trusted_partial_authorization(proposal)
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
        )

        self.assertEqual(review['graph_rebase_authorization'], {})
        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertIn(
            'apply_reviewed_requires_explicit_rebase_authorization',
            lifecycle['blocked_reasons'],
        )

    def test_apply_reviewed_refuses_full_rebase_even_with_trusted_authorization(self):
        proposal = self._executable_partial_proposal()
        proposal['requested_rebase_class'] = 'full_successor_rebase'
        authorization = {
            **self._trusted_partial_authorization(proposal),
            'requested_rebase_class': 'full_successor_rebase',
        }
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            trusted_authorization=authorization,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )

        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertIn('apply_reviewed_partial_rebase_only', lifecycle['blocked_reasons'])

    def test_graph_rebase_apply_enforced_full_rebase_remains_policy_blocked(self):
        proposal = self._proposal()
        proposal['graph_rebase_authorization'] = self._authorization(
            proposal_id=proposal['proposal_id'],
            candidate_graph_digest=proposal['candidate_graph_digest'],
        )
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_enforced',
        )

        result = apply_validated_graph_rebase(
            self._base_graph(),
            lifecycle,
            autonomy_level='apply_enforced',
        )

        self.assertEqual(lifecycle['status'], 'blocked')
        self.assertNotIn('enforced_policy_off', lifecycle['blocked_reasons'])
        self.assertIn('full_successor_rebase_not_enforced_v1', lifecycle['blocked_reasons'])
        self.assertEqual(lifecycle['enforced_policy_review']['policy']['mode'], 'safe_v1')
        self.assertEqual(lifecycle['enforced_policy_review']['policy']['source'], 'product_default')
        self.assertFalse(lifecycle['enforced_policy_review']['allowed'])
        self.assertEqual(result['status'], 'blocked')

    def test_graph_rebase_replay_idempotency_prevents_duplicate_successor(self):
        proposal = self._executable_partial_proposal()
        authorization = self._trusted_partial_authorization(proposal)
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )

        first = apply_validated_graph_rebase(
            self._base_graph(),
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        second = apply_validated_graph_rebase(
            first['graph'],
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(first['status'], 'applied')
        self.assertEqual(second['status'], 'already_applied')
        self.assertEqual(len(second['graph']['successor_rebase_requests']), 1)

    def test_shadow_review_does_not_consume_later_apply_reviewed_idempotency(self):
        proposal = self._executable_partial_proposal()
        authorization = self._trusted_partial_authorization(proposal)
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
            closure_review={'status': 'repair_required'},
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )
        shadow_lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=self._base_graph(),
            rebase_review=review,
            autonomy_level='shadow',
        )
        shadow_result = apply_validated_graph_rebase(
            self._base_graph(),
            shadow_lifecycle,
            autonomy_level='shadow',
        )
        apply_lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=shadow_result['graph'],
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )

        applied = apply_validated_graph_rebase(
            shadow_result['graph'],
            apply_lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=self.ROOT_PROMPT,
        )

        self.assertEqual(shadow_result['status'], 'validated')
        self.assertEqual(applied['status'], 'applied')
        self.assertEqual(applied['runtime_effect'], 'successor_rebase_created')
        self.assertEqual(len(applied['graph']['successor_rebase_requests']), 1)

    def test_graph_rebase_rejects_degraded_or_provider_ban_evidence(self):
        proposal = self._proposal(
            source='runtime_closure_review',
            evidence_refs=['runtime:degraded_liveness_only', 'provider_family_ban:mlx'],
        )
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=self._base_graph(),
        )

        self.assertEqual(review['status'], 'rejected')
        self.assertIn('backend_route_health_signal_is_not_rebase_authority', review['blocked_reasons'])

    def test_graph_rebase_allows_ordinary_prompt_text_containing_degraded(self):
        base = self._base_graph()
        candidate = self._candidate_graph()
        proposal = build_graph_rebase_proposal(
            request_phase_graph=base,
            candidate_graph=candidate,
            source='runtime_closure_review',
            reason='Reviewed graph rebase candidate from runtime evidence.',
            intent_anchor={
                'prompt': (
                    'Create a degraded film texture without changing provider routing.'
                )
            },
            evidence_refs=['closure:intent_graph_adequacy'],
        )

        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=base,
            closure_review={'status': 'repair_required'},
        )

        self.assertEqual(review['status'], 'accepted')
        self.assertNotIn(
            'backend_route_health_signal_is_not_rebase_authority',
            review['blocked_reasons'],
        )

    def test_graph_rebase_preserves_target_bound_text_repair(self):
        candidate = self._candidate_graph()
        candidate['phases'] = [
            phase for phase in candidate['phases']
            if phase.get('phase_id') != 'phase-css'
        ]
        candidate['phases'].append(
            {
                'phase_id': 'phase-index-repair',
                'branch_id': 'branch-index-repair',
                'obligation_id': 'obligation-index-repair',
                'capability': 'chat',
                'output_type': 'text',
                'role': 'text_artifact_output',
                'text_artifact_source_name': 'index',
                'text_artifact_extension': 'html',
                'target_path': 'artifacts/documents/index.html',
                'lineage': {'parent_phase_id': 'phase-css', 'relation': 'supersedes'},
            }
        )
        diff = build_graph_rebase_diff(self._base_graph(), candidate)
        proof = build_graph_rebase_preservation_proof(self._base_graph(), candidate, diff)

        self.assertIn(proof['status'], {'blocked', 'rejected'})
        self.assertIn('target_bound_repair_target_lost', proof['blocked_reasons'])

    def test_runtime_graph_rebase_explicit_proposal_waits_for_active_late_fill(self):
        owner = self._runtime_owner()
        graph = self._base_graph()
        graph['graph_rebase_proposals'] = [self._proposal()]
        payload = {
            'response_id': 'resp-runtime-rebase-stage',
            'lifecycle_state': 'late_fill_running',
            'runtime': {
                'graph_closure_review': {'status': 'repair_required'},
                'request_phase_graph': graph,
            },
        }

        with_evidence = owner._attach_runtime_graph_rebase_evidence(payload)
        updated = owner._attach_graph_rebase_lifecycle(
            with_evidence,
            graph_rebase_autonomy='stage',
        )

        result_graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(result_graph['graph_rebase_reviews'][0]['status'], 'blocked')
        self.assertIn(
            'active_late_fill_must_settle',
            result_graph['graph_rebase_reviews'][0]['blocked_reasons'],
        )
        self.assertEqual(result_graph['graph_rebase_lifecycle'][0]['status'], 'blocked')
        self.assertEqual(
            result_graph['graph_rebase_lifecycle'][0]['outcome']['runtime_effect'],
            'none',
        )
        self.assertEqual(result_graph.get('staged_graph_rebases') or [], [])
        self.assertEqual(result_graph.get('successor_rebase_requests') or [], [])
        self.assertEqual(diagnostics['graph_rebase_autonomy']['autonomy_level'], 'stage')

    def test_runtime_graph_rebase_explicit_proposal_respects_smaller_scope(self):
        owner = self._runtime_owner()
        graph = self._base_graph()
        graph['graph_rebase_proposals'] = [self._proposal()]
        graph['redraw_scope_ladder_review'] = {
            'kind': 'ollmo.redraw_scope_ladder_review',
            'status': 'selected',
            'selected_scope': 'add_missing_branch',
            'scopes_considered': [
                {
                    'scope': 'add_missing_branch',
                    'eligible': True,
                    'reason': 'missing_promoted_branch_or_obligation',
                }
            ],
        }
        payload = {
            'response_id': 'resp-runtime-rebase-smaller-scope',
            'lifecycle_state': 'completed',
            'runtime': {
                'graph_closure_review': {'status': 'repair_required'},
                'request_phase_graph': graph,
            },
        }

        with_evidence = owner._attach_runtime_graph_rebase_evidence(payload)
        updated = owner._attach_graph_rebase_lifecycle(
            with_evidence,
            graph_rebase_autonomy='stage',
        )

        result_graph = updated['runtime']['request_phase_graph']
        review = result_graph['graph_rebase_reviews'][0]
        self.assertEqual(review['status'], 'blocked')
        self.assertIn('smaller_redraw_scope_precedes_rebase', review['blocked_reasons'])
        self.assertEqual(result_graph.get('staged_graph_rebases') or [], [])

    def test_runtime_graph_rebase_current_gate_replaces_stale_accepted_review(self):
        owner = self._runtime_owner()
        graph = self._base_graph()
        graph['graph_rebase_proposals'] = [self._proposal()]
        terminal_payload = {
            'response_id': 'resp-runtime-rebase-current-gate',
            'lifecycle_state': 'completed',
            'runtime': {
                'graph_closure_review': {'status': 'repair_required'},
                'request_phase_graph': graph,
            },
        }
        accepted = owner._attach_runtime_graph_rebase_evidence(terminal_payload)
        self.assertEqual(
            accepted['runtime']['request_phase_graph']['graph_rebase_reviews'][0]['status'],
            'accepted',
        )
        staged = owner._attach_graph_rebase_lifecycle(
            accepted,
            graph_rebase_autonomy='stage',
        )
        staged_graph = staged['runtime']['request_phase_graph']
        self.assertEqual(len(staged_graph['graph_rebase_lifecycle']), 1)
        self.assertEqual(staged_graph['graph_rebase_lifecycle'][0]['status'], 'staged')
        self.assertEqual(len(staged_graph['staged_graph_rebases']), 1)

        active_payload = copy.deepcopy(staged)
        active_payload['lifecycle_state'] = 'late_fill_running'
        active_payload['late_fill'] = {
            'status': 'running',
            'active_count': 1,
            'active_branches': [{'branch_id': 'branch-active'}],
        }
        active_payload['runtime'].setdefault('developer_diagnostics', {})[
            'runtime_graph_rebase_candidate_review'
        ] = {
            'kind': 'ollmo.runtime_graph_rebase_candidate_review',
            'status': 'validated_by_runtime_review',
            'proposal_id': 'stale-proposal-diagnostic',
        }

        gated = owner._attach_runtime_graph_rebase_evidence(active_payload)
        updated = owner._attach_graph_rebase_lifecycle(
            gated,
            graph_rebase_autonomy='stage',
        )

        result_graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(len(result_graph['graph_rebase_reviews']), 1)
        self.assertEqual(result_graph['graph_rebase_reviews'][0]['status'], 'blocked')
        self.assertIn(
            'active_late_fill_must_settle',
            result_graph['graph_rebase_reviews'][0]['blocked_reasons'],
        )
        self.assertEqual(len(result_graph['graph_rebase_lifecycle']), 1)
        self.assertEqual(
            result_graph['graph_rebase_lifecycle'][0]['proposal_id'],
            result_graph['graph_rebase_reviews'][0]['proposal_id'],
        )
        self.assertEqual(result_graph['graph_rebase_lifecycle'][0]['status'], 'blocked')
        self.assertIn(
            'active_late_fill_must_settle',
            result_graph['graph_rebase_lifecycle'][0]['blocked_reasons'],
        )
        self.assertEqual(result_graph.get('staged_graph_rebases') or [], [])
        self.assertEqual(result_graph.get('successor_rebase_requests') or [], [])
        self.assertNotIn(
            'runtime_graph_rebase_candidate_review',
            diagnostics,
        )
        self.assertNotIn('response_time_graph_rebase_candidate', diagnostics)

    def test_runtime_graph_rebase_product_default_shadow_validates_without_mutation(self):
        owner = self._runtime_owner()
        graph = self._base_graph()
        graph['graph_rebase_proposals'] = [self._proposal()]
        payload = {
            'response_id': 'resp-runtime-rebase-shadow',
            'lifecycle_state': 'late_fill_completed',
            'runtime': {
                'graph_closure_review': {'status': 'repair_required'},
                'request_phase_graph': graph,
            },
        }

        with patch.dict('os.environ', {}, clear=True):
            with_evidence = owner._attach_runtime_graph_rebase_evidence(payload)
            updated = owner._attach_graph_rebase_lifecycle(with_evidence)

        result_graph = updated['runtime']['request_phase_graph']
        lifecycle = result_graph['graph_rebase_lifecycle'][0]
        diagnostics = updated['runtime']['developer_diagnostics']['graph_rebase_autonomy']
        self.assertEqual(lifecycle['status'], 'validated')
        self.assertEqual(lifecycle['outcome']['runtime_effect'], 'shadow_no_mutation')
        self.assertEqual(result_graph['phases'], self._base_graph()['phases'])
        self.assertEqual(result_graph.get('staged_graph_rebases') or [], [])
        self.assertEqual(result_graph.get('successor_rebase_requests') or [], [])
        self.assertEqual(diagnostics['source'], 'product_default')

    def test_runtime_graph_rebase_apply_reviewed_does_not_trust_inline_authorization(self):
        owner = self._runtime_owner()
        graph = self._base_graph()
        proposal = self._executable_partial_proposal()
        proposal['graph_rebase_authorization'] = self._authorization(
            proposal_id=proposal['proposal_id'],
            candidate_graph_digest=proposal['candidate_graph_digest'],
        )
        graph['graph_rebase_proposals'] = [proposal]
        payload = {
            'response_id': 'resp-runtime-rebase-apply-reviewed',
            'lifecycle_state': 'completed',
            'response_frame': {'frame_id': 'frame-base', 'frame_sequence': 1},
            'request': {'prompt': self.ROOT_PROMPT},
            'runtime': {
                'graph_closure_review': {'status': 'repair_required'},
                'request_phase_graph': graph,
            },
        }

        with_evidence = owner._attach_runtime_graph_rebase_evidence(payload)
        updated = owner._attach_graph_rebase_lifecycle(
            with_evidence,
            graph_rebase_autonomy='apply_reviewed',
        )

        result_graph = updated['runtime']['request_phase_graph']
        self.assertEqual(len(result_graph['phases']), len(self._base_graph()['phases']))
        self.assertEqual(result_graph['graph_rebase_lifecycle'][0]['status'], 'blocked')
        self.assertIn(
            'apply_reviewed_requires_explicit_rebase_authorization',
            result_graph['graph_rebase_lifecycle'][0]['blocked_reasons'],
        )
        self.assertEqual(result_graph.get('successor_rebase_requests') or [], [])

    def test_runtime_graph_rebase_invalid_autonomy_is_visible_safe_off(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-rebase-invalid-autonomy',
            'runtime': {'request_phase_graph': self._base_graph()},
        }

        updated = owner._attach_graph_rebase_lifecycle(
            payload,
            graph_rebase_autonomy='full_auto_redraw_now',
        )

        diagnostics = updated['runtime']['developer_diagnostics']['graph_rebase_autonomy']
        result_graph = updated['runtime']['request_phase_graph']
        self.assertEqual(diagnostics['raw_value'], 'full_auto_redraw_now')
        self.assertEqual(diagnostics['autonomy_level'], 'off')
        self.assertTrue(diagnostics['invalid_value'])
        self.assertEqual(result_graph.get('successor_rebase_requests') or [], [])

    def test_runtime_graph_rebase_diagnostics_preserve_product_default_provenance(self):
        owner = self._runtime_owner()
        payload = {
            'response_id': 'resp-runtime-rebase-product-default',
            'runtime': {'request_phase_graph': self._base_graph()},
        }

        with patch.dict('os.environ', {}, clear=True):
            updated = owner._attach_graph_rebase_lifecycle(payload)

        diagnostics = updated['runtime']['developer_diagnostics']['graph_rebase_autonomy']
        result_graph = updated['runtime']['request_phase_graph']
        self.assertEqual(diagnostics['autonomy_level'], 'shadow')
        self.assertEqual(diagnostics['source'], 'product_default')
        self.assertFalse(diagnostics['configured'])
        self.assertEqual(result_graph.get('successor_rebase_requests') or [], [])


if __name__ == '__main__':
    unittest.main()
