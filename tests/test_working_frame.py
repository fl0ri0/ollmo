import unittest

from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_orchestration.working_frame import build_working_frame


class WorkingFrameTests(unittest.TestCase):
    def test_build_working_frame_tracks_chain_loop_goals_and_journal(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-1',
                'prompt': 'Summarize the PDF and read it aloud.',
                'ghost_self_heal_attempted': True,
                'input_artifacts': [{'type': 'pdf', 'path': 'artifacts/inputs/notes.pdf'}],
                'voice': 'alloy',
            },
            route_payload={
                'instance_id': 'tts-1',
                'capability': 'text_to_speech',
                'route_source': 'self_heal',
                'route_reason': 'Recovered to a stable TTS route.',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'qwen-tts',
                    'backend': 'mlx',
                },
                'route_runtime': {
                    'semantic_role_profile': {
                        'mode': 'repair',
                        'loop': {'max_passes': 2, 'critic_passes': 1},
                    },
                    'execution_planner': {
                        'attempted': True,
                        'applied': True,
                        'reason': 'compound summarize-then-read request',
                    },
                    'control_hints': {'voice': 'alloy'},
                    'retry_failure': {
                        'failed_instance_id': 'chat-1',
                        'status_code': 503,
                        'error_message': 'previous route failed',
                    },
                },
            },
        )

        self.assertEqual(working_frame['kind'], 'ollmo.working_frame')
        self.assertEqual(working_frame['working_frame_version'], 4)
        self.assertEqual(working_frame['status'], 'repairing')
        self.assertEqual(working_frame['loop']['chain_id'], 'conv-1')
        self.assertEqual(working_frame['loop']['pass_index'], 2)
        self.assertFalse(working_frame['loop']['remaining_passes'] < 0)
        self.assertEqual(working_frame['goal_stack'][0]['kind'], 'respond')
        self.assertEqual(working_frame['goal_stack'][1]['kind'], 'integrate_input_artifact')
        self.assertIn('controls', working_frame)
        phases = [entry['phase'] for entry in working_frame['journal']]
        self.assertIn('critic', phases)
        self.assertIn('revise', phases)
        self.assertEqual(working_frame['work_tree']['kind'], 'ollmo.work_tree')
        self.assertTrue(working_frame['artifact_dossiers'])
        self.assertEqual(working_frame['possibility_space']['state'], 'open')
        self.assertIn('continue', working_frame['possibility_space']['review_paths'])
        self.assertIn('revise', working_frame['possibility_space']['review_paths'])
        self.assertEqual(working_frame['closure']['status'], 'open')
        self.assertTrue(working_frame['editability']['mutable'])

    def test_build_working_frame_marks_frozen_state_during_freeze(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-2',
                'prompt': 'Make a square album cover.',
                'width': 1024,
                'height': 1024,
            },
            response_payload={
                'id': 'resp_1',
                'status': 'completed',
                'instance_id': 'image-1',
                'model': 'flux',
                'backend': 'mlx',
                'capability': 'image_generation',
                'mode': 'image_generation',
                'output_text': 'Generated image.',
                'artifacts': [{'type': 'image', 'path': 'artifacts/images/out.png'}],
            },
            freeze=True,
        )

        self.assertEqual(working_frame['status'], 'frozen')
        self.assertEqual(working_frame['freeze']['status'], 'frozen')
        self.assertEqual(working_frame['possibility_space']['state'], 'closed')
        self.assertEqual(working_frame['closure']['status'], 'closed')
        self.assertEqual(working_frame['closure']['close_authority'], 'ollmo')
        self.assertFalse(working_frame['editability']['mutable'])
        self.assertEqual(working_frame['review']['status'], 'frozen')

    def test_build_working_frame_marks_freeze_ready_before_final_freeze(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-3',
                'prompt': 'Answer briefly.',
            },
            response_payload={
                'id': 'resp_ready',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gpt-oss:20b',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'Done.',
            },
            freeze=False,
        )

        self.assertEqual(working_frame['status'], 'completed')
        self.assertEqual(working_frame['closure']['status'], 'ready')
        self.assertTrue(working_frame['closure']['postable'])
        self.assertIn('freeze', working_frame['possibility_space']['review_paths'])

    def test_build_working_frame_keeps_deferred_outputs_fluid_before_freeze(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-4',
                'prompt': 'Translate that quote and read it aloud.',
                'reference_artifacts': [{'type': 'text', 'path': 'artifacts/texts/quote.txt'}],
            },
            response_payload={
                'id': 'resp_deferred_audio',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'qwen',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'Here is the translation and narration script.',
                'late_fill': {
                    'status': 'pending',
                    'expected_capability': 'text_to_speech',
                    'missing_artifact_type': 'audio',
                },
            },
            freeze=False,
        )

        self.assertEqual(working_frame['status'], 'active')
        self.assertEqual(working_frame['artifact_flow']['reference_routes'][0]['lifecycle'], 'carried_reference')
        self.assertEqual(working_frame['artifact_flow']['output_slots'][0]['status'], 'fulfilled')
        self.assertEqual(working_frame['artifact_flow']['output_slots'][0]['type'], 'text')
        self.assertEqual(working_frame['artifact_flow']['output_slots'][0]['child_slot_ids'], ['output-phase-2'])
        self.assertEqual(working_frame['artifact_flow']['output_slots'][1]['status'], 'pending')
        self.assertEqual(working_frame['artifact_flow']['output_slots'][1]['type'], 'audio')
        self.assertEqual(working_frame['artifact_flow']['output_slots'][1]['lifecycle'], 'deferred_output')
        self.assertEqual(working_frame['artifact_flow']['output_slots'][1]['parent_slot_id'], 'output-phase-1')
        self.assertEqual(working_frame['work_tree']['output_root_node_ids'], ['node-output-phase-1'])
        work_nodes = {item['node_id']: item for item in working_frame['work_tree']['nodes']}
        self.assertEqual(work_nodes['node-output-phase-1']['child_node_ids'], ['node-output-phase-2'])
        self.assertEqual(work_nodes['node-output-phase-2']['parent_node_id'], 'node-output-phase-1')
        dossier = next(iter(working_frame['artifact_dossiers'].values()))
        self.assertEqual(dossier['roles'], ['reference'])
        self.assertEqual(working_frame['review']['status'], 'pending_outputs')
        self.assertEqual(working_frame['closure']['status'], 'open')
        self.assertIn('output-phase-2', working_frame['review']['pending_output_slot_ids'])
        self.assertIn('output-phase-2', working_frame['review']['deferred_output_slot_ids'])
        self.assertIn('goal-reference-1', [item['goal_id'] for item in working_frame['goal_stack']])

    def test_build_working_frame_projects_intent_contract_from_graph_review(self):
        phase_graph = build_request_phase_graph(
            'Describe two tiny stage scenes and then generate two images.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
            response_payload={'output_text': 'Two image prompts are ready.'},
        )
        graph_review = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'pending',
            'reason': 'one image obligation remains open',
            'contract_source': 'request_ir.output_obligations',
            'obligation_count': 3,
            'counts': {'fulfilled': 2, 'pending': 1, 'deferred': 0, 'blocked': 0},
            'checks': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'fulfilled',
                    'evidence': 'current_phase_output_text',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                    'evidence': 'output_artifact_type:image',
                },
                {
                    'obligation_id': 'obligation-phase-3',
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-image_generation-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                    'evidence': 'pending_graph_branch',
                },
            ],
        }

        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-contract',
                'prompt': 'Describe two tiny stage scenes and then generate two images.',
                'ghost_route': True,
            },
            response_payload={
                'id': 'resp_contract',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gemma',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'Two image prompts are ready.',
                'runtime': {
                    'request_phase_graph': phase_graph,
                    'graph_closure_review': graph_review,
                },
                'late_fill': {
                    'status': 'pending',
                    'pending_branches': [
                        {
                            'branch_id': 'branch-image_generation-2',
                            'phase_id': 'phase-3',
                            'obligation_id': 'obligation-phase-3',
                            'capability': 'image_generation',
                            'output_type': 'image',
                        }
                    ],
                },
                'artifacts': [{'type': 'image', 'path': 'artifacts/images/one.png'}],
            },
            freeze=False,
        )

        intent_contract = working_frame['intent_contract']
        self.assertEqual(intent_contract['kind'], 'ollmo.intent_contract')
        self.assertEqual(intent_contract['source'], 'request_ir.output_obligations')
        self.assertEqual(intent_contract['status'], 'pending')
        self.assertEqual(intent_contract['counts']['fulfilled'], 2)
        self.assertEqual(intent_contract['pending_obligation_ids'], ['obligation-phase-3'])
        self.assertEqual(working_frame['review']['intent_contract_status'], 'pending')
        self.assertEqual(working_frame['review']['pending_obligation_ids'], ['obligation-phase-3'])
        self.assertEqual(working_frame['possibility_space']['pending_obligation_ids'], ['obligation-phase-3'])

    def test_build_working_frame_freezes_pending_obligations_as_truthful_partial_state(self):
        phase_graph = build_request_phase_graph(
            'Describe two tiny stage scenes and then generate two images.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
            response_payload={'output_text': 'Two image prompts are ready.'},
        )
        graph_review = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'pending',
            'reason': 'one image obligation remains open',
            'contract_source': 'request_ir.output_obligations',
            'obligation_count': 3,
            'counts': {'fulfilled': 2, 'pending': 1, 'deferred': 0, 'blocked': 0},
            'checks': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'fulfilled',
                    'evidence': 'current_phase_output_text',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                    'evidence': 'output_artifact_type:image',
                },
                {
                    'obligation_id': 'obligation-phase-3',
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-image_generation-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                    'evidence': 'pending_graph_branch',
                },
            ],
        }

        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-partial-freeze',
                'prompt': 'Describe two tiny stage scenes and then generate two images.',
                'ghost_route': True,
            },
            response_payload={
                'id': 'resp_partial_freeze',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gemma',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'Two image prompts are ready.',
                'runtime': {
                    'request_phase_graph': phase_graph,
                    'graph_closure_review': graph_review,
                },
                'late_fill': {
                    'status': 'pending',
                    'pending_branches': [
                        {
                            'branch_id': 'branch-image_generation-2',
                            'phase_id': 'phase-3',
                            'obligation_id': 'obligation-phase-3',
                            'capability': 'image_generation',
                            'output_type': 'image',
                        }
                    ],
                },
                'artifacts': [{'type': 'image', 'path': 'artifacts/images/one.png'}],
            },
            freeze=True,
        )

        self.assertEqual(working_frame['status'], 'frozen')
        self.assertEqual(working_frame['review']['status'], 'partial_frozen')
        self.assertFalse(working_frame['review']['freeze_ready'])
        self.assertEqual(working_frame['review']['pending_obligation_ids'], ['obligation-phase-3'])
        self.assertEqual(working_frame['closure']['status'], 'closed')
        self.assertTrue(working_frame['closure']['continuation_expected'])
        self.assertEqual(working_frame['closure']['pending_obligation_ids'], ['obligation-phase-3'])

    def test_build_working_frame_projects_explicit_waived_obligations(self):
        phase_graph = build_request_phase_graph(
            'Describe two tiny stage scenes and then generate two images.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
            response_payload={'output_text': 'One image prompt is enough.'},
        )
        graph_review = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'fulfilled',
            'reason': 'remaining image obligation was explicitly waived',
            'contract_source': 'request_ir.output_obligations',
            'obligation_count': 3,
            'counts': {'fulfilled': 2, 'pending': 0, 'deferred': 0, 'blocked': 0, 'waived': 1},
            'checks': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'fulfilled',
                    'evidence': 'current_phase_output_text',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                    'evidence': 'output_artifact_type:image',
                },
                {
                    'obligation_id': 'obligation-phase-3',
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-image_generation-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'waived',
                    'evidence': 'explicit_obligation_waiver',
                },
            ],
        }

        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-waived-contract',
                'prompt': 'Describe two tiny stage scenes and then generate two images.',
                'ghost_route': True,
            },
            response_payload={
                'id': 'resp_waived_contract',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gemma',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'One image prompt is enough.',
                'runtime': {
                    'request_phase_graph': phase_graph,
                    'graph_closure_review': graph_review,
                },
                'artifacts': [{'type': 'image', 'path': 'artifacts/images/one.png'}],
            },
            freeze=False,
        )

        intent_contract = working_frame['intent_contract']
        self.assertEqual(intent_contract['status'], 'fulfilled')
        self.assertEqual(intent_contract['counts']['waived'], 1)
        self.assertEqual(intent_contract.get('pending_obligation_ids', []), [])
        self.assertEqual(intent_contract['waived_obligation_ids'], ['obligation-phase-3'])
        self.assertEqual(working_frame['review']['waived_obligation_ids'], ['obligation-phase-3'])
        self.assertEqual(working_frame['possibility_space']['waived_obligation_ids'], ['obligation-phase-3'])

    def test_build_working_frame_projects_output_candidates_and_promotions(self):
        phase_graph = build_request_phase_graph(
            'Now use the reserved image direction and generate it.',
            request_payload={
                'ghost_route': True,
                'downstream_branches': [
                    {
                        'candidate_id': 'candidate-image-1',
                        'branch_id': 'branch-image-1',
                        'phase_id': 'phase-2',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'contract_state': 'promoted',
                        'promotion_reason': 'user confirmed generation',
                        'promotion_source': 'user_confirmation',
                    }
                ],
            },
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
            response_payload={'output_text': 'The reserved image prompt is ready.'},
        )

        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-candidate-promotion',
                'prompt': 'Now use the reserved image direction and generate it.',
                'ghost_route': True,
            },
            response_payload={
                'id': 'resp_candidate_promotion',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gemma',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'The reserved image prompt is ready.',
                'runtime': {'request_phase_graph': phase_graph},
            },
            freeze=False,
        )

        intent_contract = working_frame['intent_contract']
        self.assertEqual(intent_contract['candidate_count'], 1)
        self.assertEqual(intent_contract['promotion_count'], 1)
        self.assertEqual(intent_contract['candidate_graph']['kind'], 'ollmo.candidate_graph')
        self.assertEqual(intent_contract['promotion_review']['kind'], 'ollmo.promotion_review')
        self.assertGreaterEqual(intent_contract['general_promoted_count'], 2)
        self.assertEqual(intent_contract['candidate_output_ids'], ['candidate-image-1'])
        self.assertEqual(intent_contract['output_candidates'][0]['status'], 'promoted')
        self.assertEqual(intent_contract['promotions'][0]['obligation_id'], 'obligation-phase-2')
        self.assertEqual(intent_contract['pending_obligation_ids'], ['obligation-phase-2'])
        self.assertGreaterEqual(working_frame['possibility_space']['general_intent_candidate_count'], 1)
        self.assertGreaterEqual(working_frame['possibility_space']['general_promoted_candidate_count'], 2)
        output_goals = [item for item in working_frame['goal_stack'] if item['kind'] == 'materialize_output']
        self.assertIn('obligation_id', output_goals[0])

    def test_build_working_frame_projects_unpromoted_context_candidates(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-context-candidate',
                'prompt': 'Answer the current question only.',
                'history_candidates': [
                    {
                        'candidate_id': 'history-old-scene',
                        'summary': 'A previous generated stage scene may be relevant later.',
                        'candidate_status': 'reserved',
                        'source_message_id': 'msg_old_scene',
                    }
                ],
                'memory_candidates': [
                    {
                        'candidate_id': 'memory-user-prefers-german',
                        'memory_id': 'pref-language-de',
                        'summary': 'User often writes in German.',
                    }
                ],
            },
            response_payload={
                'id': 'resp_context_candidate',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gemma',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'Current answer.',
            },
            freeze=False,
        )

        context_contract = working_frame['context_contract']
        self.assertEqual(context_contract['kind'], 'ollmo.context_contract')
        self.assertEqual(context_contract['status'], 'candidate_only')
        self.assertEqual(context_contract['candidate_count'], 2)
        self.assertEqual(context_contract.get('promotion_count', 0), 0)
        candidates = {
            item['candidate_id']: item
            for item in context_contract['context_candidates']
        }
        self.assertEqual(candidates['history-old-scene']['status'], 'reserved')
        self.assertEqual(candidates['memory-user-prefers-german']['source_kind'], 'memory')
        self.assertNotIn('promoted_candidate_ids', context_contract)
        self.assertNotIn('active_reference_artifact_refs', context_contract)

    def test_build_working_frame_promotes_selected_reference_artifact_as_context(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-context-reference',
                'prompt': 'Use this previous image as visual reference.',
                'selected_reference_artifacts': [
                    {
                        'type': 'image',
                        'path': 'artifacts/images/previous.png',
                        'name': 'previous.png',
                        'source_response_id': 'resp_previous_image',
                    }
                ],
            },
            response_payload={
                'id': 'resp_context_reference',
                'status': 'completed',
                'instance_id': 'image-1',
                'model': 'flux',
                'backend': 'ollama',
                'capability': 'image_generation',
                'mode': 'image_generation',
                'output_text': '',
            },
            freeze=False,
        )

        context_contract = working_frame['context_contract']
        self.assertEqual(context_contract['status'], 'active')
        self.assertEqual(context_contract['candidate_count'], 1)
        self.assertEqual(context_contract['promotion_count'], 1)
        self.assertEqual(context_contract['candidate_graph']['kind'], 'ollmo.candidate_graph')
        self.assertEqual(context_contract['promotion_review']['kind'], 'ollmo.promotion_review')
        self.assertEqual(context_contract['general_promoted_count'], 1)
        self.assertEqual(context_contract['context_candidates'][0]['status'], 'promoted')
        self.assertEqual(context_contract['context_candidates'][0]['source_kind'], 'artifact')
        self.assertEqual(context_contract['promotions'][0]['target'], 'active_reference')
        self.assertEqual(
            context_contract['active_reference_artifact_refs'],
            [context_contract['context_candidates'][0]['artifact_ref']],
        )
        self.assertEqual(working_frame['artifact_flow']['reference_routes'][0]['lifecycle'], 'carried_reference')

    def test_build_working_frame_projects_context_strategy_as_context_state(self):
        working_frame = build_working_frame(
            request_payload={
                'conversation_id': 'conv-current-turn',
                'prompt': 'Tell me a new fact.',
            },
            route_payload={
                'instance_id': 'chat-1',
                'capability': 'chat',
                'route_runtime': {
                    'context_strategy': {
                        'mode': 'current_turn_only',
                        'reason': 'fresh turn with no referential backlink',
                        'applied': True,
                        'context_candidates': [
                            {
                                'candidate_id': 'context-message-old-1',
                                'source_kind': 'message',
                                'status': 'not_promoted',
                                'summary': 'older assistant answer',
                            },
                            {
                                'candidate_id': 'history-scan-deeper-pool',
                                'source_kind': 'history_scan',
                                'status': 'promoted',
                                'promotion_target': 'history_scan',
                                'promotion_reason': 'current turn asks for broader history search',
                                'scan_targets': ['chat_history', 'response_frame_ledger', 'artifact_registry'],
                            }
                        ],
                        'context_gate_review': {
                            'kind': 'ollmo.context_gate_review',
                            'status': 'checked',
                            'intake_boundary': 'current_turn',
                            'mode': 'current_turn_only',
                            'history_scan': {
                                'decision': 'promoted',
                                'executed': True,
                                'status': 'completed',
                                'matched_candidate_count': 0,
                                'promoted_candidate_count': 0,
                                'scanned': {
                                    'chat_history': 4,
                                    'response_frame_ledger': 1,
                                    'artifact_registry': 1,
                                },
                            },
                            'candidate_count': 2,
                            'promoted_candidate_count': 1,
                            'not_promoted_candidate_count': 1,
                        },
                    }
                },
            },
            response_payload={
                'id': 'resp_current_turn',
                'status': 'completed',
                'instance_id': 'chat-1',
                'model': 'gemma',
                'backend': 'ollama',
                'capability': 'chat',
                'mode': 'chat',
                'output_text': 'New fact.',
            },
            freeze=False,
        )

        context_contract = working_frame['context_contract']
        self.assertEqual(context_contract['status'], 'active')
        candidates = {
            item['candidate_id']: item
            for item in context_contract['context_candidates']
        }
        self.assertEqual(candidates['context-strategy-current_turn_only']['status'], 'not_promoted')
        self.assertEqual(candidates['context-message-old-1']['status'], 'not_promoted')
        self.assertEqual(candidates['history-scan-deeper-pool']['status'], 'promoted')
        self.assertEqual(
            working_frame['context_contract']['promoted_history_scan_candidate_ids'],
            ['history-scan-deeper-pool'],
        )
        self.assertEqual(
            candidates['history-scan-deeper-pool']['scan_targets'],
            ['chat_history', 'response_frame_ledger', 'artifact_registry'],
        )
        self.assertEqual(
            context_contract['context_gate_review']['history_scan']['decision'],
            'promoted',
        )
        self.assertEqual(
            context_contract['context_gate_review']['history_scan']['matched_candidate_count'],
            0,
        )
        self.assertEqual(working_frame['review']['context_contract_status'], 'active')


if __name__ == '__main__':
    unittest.main()
