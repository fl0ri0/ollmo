import copy
import os
import unittest
from unittest.mock import patch

from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner


class RuntimeGraphRebaseShadowProducerTests(unittest.TestCase):
    def _owner(self):
        return ResponsesRequestRuntimeOwner(
            hooks={
                'normalize_capability': lambda value: str(value or '').strip().lower(),
                'extract_responses_prompt': lambda payload: payload.get('prompt'),
                'extract_responses_current_turn_prompt': lambda payload: payload.get('prompt'),
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
            'kind': 'ollmo.request_phase_graph',
            'graph_version': 3,
            'response_id': 'resp-shadow-producer',
            'frame_id': 'frame-current',
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
                    'phase_id': 'phase-image',
                    'branch_id': 'branch-image',
                    'obligation_id': 'obligation-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                    'depends_on': ['phase-1'],
                },
            ],
            'downstream_branches': [
                {
                    'phase_id': 'phase-image',
                    'branch_id': 'branch-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                }
            ],
            'intent_obligations': [
                {
                    'obligation_id': 'intent-image',
                    'phase_id': 'phase-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                }
            ],
            'output_obligations': [
                {
                    'obligation_id': 'obligation-image',
                    'phase_id': 'phase-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                }
            ],
        }

    def _structured_candidate(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['frame_id'] = 'frame-response-time-candidate'
        candidate['phases'].append(
            {
                'phase_id': 'phase-image-review',
                'branch_id': 'branch-image-review',
                'obligation_id': 'obligation-image-review',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
                'depends_on': ['phase-image'],
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-image-review',
                'branch_id': 'branch-image-review',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-image'],
            }
        )
        candidate['output_obligations'].append(
            {
                'obligation_id': 'obligation-image-review',
                'phase_id': 'phase-image-review',
                'capability': 'chat',
                'output_type': 'text',
                'required': True,
            }
        )
        return candidate

    def _dependency_only_candidate(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['phases'][1]['depends_on'] = ['phase-1', 'phase-review-source']
        candidate['downstream_branches'][0]['depends_on'] = [
            'phase-1',
            'phase-review-source',
        ]
        return candidate

    def _closure_review(self, *, include_smaller_scope=False):
        checks = [
            {
                'check_kind': 'semantic_graph_shape',
                'status': 'repair_required',
                'repair_action': 'rebuild_from_promoted_obligations',
            }
        ]
        if include_smaller_scope:
            checks.append(
                {
                    'check_kind': 'intent_graph_adequacy',
                    'status': 'repair_required',
                    'repair_action': 'add_missing_branch',
                    'add_phases': [
                        {
                            'phase_id': 'phase-missing',
                            'branch_id': 'branch-missing',
                            'capability': 'chat',
                            'output_type': 'text',
                        }
                    ],
                }
            )
        return {
            'kind': 'ollmo.graph_closure_review',
            'status': 'repair_required',
            'checks': checks,
        }

    def _payload(self, candidate, *, late_fill_status='completed', include_smaller_scope=False):
        return {
            'id': 'resp-shadow-producer',
            'response_id': 'resp-shadow-producer',
            'lifecycle_state': 'late_fill_completed',
            'late_fill': {
                'status': late_fill_status,
                'pending_branches': [],
                'active_count': 0,
                'pending_count': 0,
            },
            'runtime': {
                'request_phase_graph': self._base_graph(),
                'graph_closure_review': self._closure_review(
                    include_smaller_scope=include_smaller_scope
                ),
                'developer_diagnostics': {
                    'response_time_graph_rebase_candidate': {
                        'kind': 'ollmo.runtime_graph_rebase_candidate',
                        'candidate_origin': 'response_time_request_phase_graph',
                        'candidate_graph': copy.deepcopy(candidate),
                    }
                },
            },
        }

    def test_fluid_graph_retains_only_meaningful_unselected_response_time_candidate(self):
        owner = self._owner()
        route_payload = {'route_runtime': {'request_phase_graph': self._base_graph()}}
        with patch(
            'ollmo_server.responses_request_runtime.build_request_phase_graph',
            return_value=self._structured_candidate(),
        ):
            updated, selected = owner._attach_fluid_request_phase_graph(
                {'response_id': 'resp-shadow-producer'},
                output_text='done',
                route_payload=route_payload,
                request_payload={'prompt': 'review the generated image'},
            )

        candidate = updated['runtime']['developer_diagnostics'][
            'response_time_graph_rebase_candidate'
        ]
        self.assertEqual(selected, self._base_graph())
        self.assertGreater(candidate['diff_summary']['meaningful_change_count'], 0)

        status_only = copy.deepcopy(self._base_graph())
        status_only['phases'][1]['status'] = 'completed'
        with patch(
            'ollmo_server.responses_request_runtime.build_request_phase_graph',
            return_value=status_only,
        ):
            status_updated, _ = owner._attach_fluid_request_phase_graph(
                {'response_id': 'resp-shadow-producer'},
                output_text='done',
                route_payload=route_payload,
                request_payload={'prompt': 'review the generated image'},
            )
        self.assertNotIn(
            'response_time_graph_rebase_candidate',
            status_updated['runtime']['developer_diagnostics'],
        )

    def test_structural_candidate_produces_non_mutating_product_default_shadow_truth(self):
        owner = self._owner()
        payload = owner._attach_redraw_scope_ladder_review(
            self._payload(self._structured_candidate())
        )
        payload = owner._attach_runtime_graph_repair_evidence(payload)
        self.assertIn(
            'response_time_graph_rebase_candidate',
            payload['runtime']['developer_diagnostics'],
        )

        with patch.dict(os.environ, {}, clear=True):
            with_evidence = owner._attach_runtime_graph_rebase_evidence(payload)
            updated = owner._attach_graph_rebase_lifecycle(with_evidence)

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(len(graph['graph_rebase_proposals']), 1)
        self.assertEqual(graph['graph_rebase_reviews'][0]['status'], 'accepted')
        self.assertEqual(graph['redraw_scope_ladder_review']['selected_scope'], 'partial_subtree_rebase')
        self.assertEqual(graph['graph_rebase_lifecycle'][0]['status'], 'validated')
        self.assertEqual(
            graph['graph_rebase_lifecycle'][0]['outcome']['runtime_effect'],
            'shadow_no_mutation',
        )
        self.assertEqual(
            [item['phase_id'] for item in graph['phases']],
            ['phase-1', 'phase-image'],
        )
        self.assertEqual(graph.get('staged_graph_rebases') or [], [])
        self.assertEqual(graph.get('successor_rebase_requests') or [], [])
        self.assertNotIn('graph_rebase_authorization', graph['graph_rebase_proposals'][0])
        self.assertNotIn('response_time_graph_rebase_candidate', diagnostics)
        self.assertEqual(
            diagnostics['runtime_graph_rebase_candidate_review']['status'],
            'validated_by_runtime_review',
        )

    def test_active_late_fill_consumes_candidate_context_without_proposal(self):
        owner = self._owner()
        payload = self._payload(self._structured_candidate(), late_fill_status='running')
        payload['lifecycle_state'] = 'late_fill_running'
        payload['late_fill']['active_count'] = 1

        updated = owner._attach_runtime_graph_rebase_evidence(payload)

        diagnostics = updated['runtime']['developer_diagnostics']
        graph = updated['runtime']['request_phase_graph']
        self.assertEqual(graph.get('graph_rebase_proposals') or [], [])
        self.assertNotIn('response_time_graph_rebase_candidate', diagnostics)
        self.assertEqual(
            diagnostics['runtime_graph_rebase_candidate_review']['reason'],
            'active_late_fill_must_settle',
        )

        terminal_payload = copy.deepcopy(updated)
        terminal_payload['lifecycle_state'] = 'late_fill_completed'
        terminal_payload['late_fill'] = {
            'status': 'completed',
            'active_count': 0,
            'pending_count': 0,
            'active_branches': [],
            'pending_branches': [],
        }
        with patch(
            'ollmo_server.responses_request_runtime.build_request_phase_graph',
            return_value=self._structured_candidate(),
        ), patch.dict(os.environ, {}, clear=True):
            terminal = owner.review_terminal_graph_rebase_after_late_fill(
                terminal_payload,
                request_payload={'prompt': 'review the generated image'},
                route_payload={},
            )

        terminal_graph = terminal['runtime']['request_phase_graph']
        terminal_diagnostics = terminal['runtime']['developer_diagnostics']
        self.assertEqual(len(terminal_graph['graph_rebase_proposals']), 1)
        self.assertEqual(terminal_graph['graph_rebase_reviews'][0]['status'], 'accepted')
        self.assertEqual(terminal_graph['graph_rebase_lifecycle'][0]['status'], 'validated')
        self.assertNotIn('response_time_graph_rebase_candidate', terminal_diagnostics)

    def test_terminal_noop_rederivation_replaces_stale_active_late_fill_diagnostic(self):
        owner = self._owner()
        payload = self._payload(self._structured_candidate(), late_fill_status='running')
        payload['lifecycle_state'] = 'late_fill_running'
        payload['late_fill']['active_count'] = 1
        active = owner._attach_runtime_graph_rebase_evidence(payload)
        terminal_payload = copy.deepcopy(active)
        terminal_payload['lifecycle_state'] = 'late_fill_completed'
        terminal_payload['late_fill'] = {
            'status': 'completed',
            'active_count': 0,
            'pending_count': 0,
            'active_branches': [],
            'pending_branches': [],
        }

        with patch(
            'ollmo_server.responses_request_runtime.build_request_phase_graph',
            return_value=self._base_graph(),
        ), patch.dict(os.environ, {}, clear=True):
            terminal = owner.review_terminal_graph_rebase_after_late_fill(
                terminal_payload,
                request_payload={'prompt': 'review the generated image'},
                route_payload={},
            )

        diagnostics = terminal['runtime']['developer_diagnostics']
        graph = terminal['runtime']['request_phase_graph']
        candidate_review = diagnostics['runtime_graph_rebase_candidate_review']
        self.assertEqual(candidate_review['status'], 'not_proposed')
        self.assertEqual(
            candidate_review['reason'],
            'terminal_candidate_has_no_meaningful_change',
        )
        self.assertNotIn('late_fill_status', candidate_review)
        self.assertEqual(graph.get('graph_rebase_proposals') or [], [])

    def test_terminal_completed_late_fill_overrides_stale_pending_lifecycle_label(self):
        owner = self._owner()
        payload = self._payload(self._structured_candidate())
        payload['lifecycle_state'] = 'late_fill_pending'
        payload['late_fill'] = {
            'status': 'completed',
            'active_count': 0,
            'pending_count': 0,
            'active_branches': [],
            'pending_branches': [],
        }

        with patch(
            'ollmo_server.responses_request_runtime.build_request_phase_graph',
            return_value=self._structured_candidate(),
        ), patch.dict(os.environ, {}, clear=True):
            terminal = owner.review_terminal_graph_rebase_after_late_fill(
                payload,
                request_payload={'prompt': 'review the generated image'},
                route_payload={},
            )

        graph = terminal['runtime']['request_phase_graph']
        diagnostics = terminal['runtime']['developer_diagnostics']
        self.assertEqual(graph['graph_rebase_reviews'][0]['status'], 'accepted')
        self.assertEqual(graph['graph_rebase_lifecycle'][0]['status'], 'validated')
        self.assertNotIn(
            'active_late_fill_must_settle',
            graph['graph_rebase_reviews'][0].get('blocked_reasons') or [],
        )
        self.assertNotEqual(
            diagnostics['runtime_graph_rebase_candidate_review'].get('reason'),
            'active_late_fill_must_settle',
        )

    def test_smaller_additive_scope_precedes_structural_candidate(self):
        owner = self._owner()
        payload = owner._attach_redraw_scope_ladder_review(
            self._payload(self._structured_candidate(), include_smaller_scope=True)
        )

        updated = owner._attach_runtime_graph_rebase_evidence(payload)

        diagnostics = updated['runtime']['developer_diagnostics']
        graph = updated['runtime']['request_phase_graph']
        self.assertEqual(graph['redraw_scope_ladder_review']['selected_scope'], 'add_missing_branch')
        self.assertEqual(graph.get('graph_rebase_proposals') or [], [])
        self.assertEqual(
            diagnostics['runtime_graph_rebase_candidate_review']['reason'],
            'smaller_redraw_scope_precedes_rebase',
        )

    def test_additive_only_candidate_does_not_escalate_to_rebase(self):
        owner = self._owner()
        payload = owner._attach_redraw_scope_ladder_review(
            self._payload(self._dependency_only_candidate())
        )

        updated = owner._attach_runtime_graph_rebase_evidence(payload)

        diagnostics = updated['runtime']['developer_diagnostics']
        graph = updated['runtime']['request_phase_graph']
        self.assertEqual(graph.get('graph_rebase_proposals') or [], [])
        self.assertEqual(
            diagnostics['runtime_graph_rebase_candidate_review']['reason'],
            'candidate_change_is_additive_repair_only',
        )

    def test_advisory_ghost_action_is_not_runtime_rebase_authority(self):
        owner = self._owner()
        payload = self._payload(self._structured_candidate())
        payload['runtime']['graph_closure_review'] = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'repair_required',
            'ghost_repair_feedback': {
                'kind': 'ollmo.ghost_repair_feedback',
                'status': 'repair_required',
                'items': [
                    {
                        'status': 'candidate',
                        'authority': 'advisory_only',
                        'repair_action': 'partial_subtree_rebase',
                    }
                ],
            },
        }

        updated = owner._attach_runtime_graph_rebase_evidence(payload)

        graph = updated['runtime']['request_phase_graph']
        diagnostics = updated['runtime']['developer_diagnostics']
        self.assertEqual(graph.get('graph_rebase_proposals') or [], [])
        self.assertEqual(
            diagnostics['runtime_graph_rebase_candidate_review']['reason'],
            'current_structural_closure_evidence_missing',
        )


if __name__ == '__main__':
    unittest.main()
