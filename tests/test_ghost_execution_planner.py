import unittest
from unittest.mock import Mock

from ollmo_g.execution_planner import (
    _extract_latest_assistant_content,
    plan_compound_execution,
    split_visible_image_payload,
)
from ollmo_g.control_hints import infer_tts_instruct_from_prompt
from ollmo_g.request_ir import build_request_ir
from ollmo_g.request_phase_graph import (
    build_request_phase_graph,
    downstream_phase_records,
    next_executable_downstream_branches,
)
from ollmo_g.router import _sanitize_workload_task_proposals


class GhostExecutionPlannerTests(unittest.TestCase):
    def test_split_visible_image_payload_prefers_explicit_prompt_block(self):
        display_text = (
            "My dream place is a sheltered lagoon at twilight, ringed by pale stone bridges and silver trees.\n\n"
            "***\n\n"
            "### The Image Generation Prompt\n\n"
            "As an AI language model, I cannot physically generate the image for you.\n\n"
            "**Here is your prompt:**\n\n"
            "> A cinematic twilight lagoon with silver trees, pale stone bridges, emerald bioluminescent vines, "
            "misty mountains, ethereal fantasy lighting, hyper-real detail."
        )

        payload = split_visible_image_payload(display_text)

        self.assertEqual(
            payload['artifact_prompt'],
            'A cinematic twilight lagoon with silver trees, pale stone bridges, emerald bioluminescent vines, '
            'misty mountains, ethereal fantasy lighting, hyper-real detail.',
        )
        self.assertEqual(payload['artifact_prompt_source'], 'prompt_blockquote_section')
        self.assertIn('sheltered lagoon at twilight', payload['content_payload'])

    def test_split_visible_image_payload_extracts_quoted_prompt_section_after_heading(self):
        display_text = (
            "I understand the desire to see a new, visually rich scene.\n\n"
            "***\n\n"
            "### The Crystal Atrium of Whispering Roots\n\n"
            "Imagine a place deep beneath a canopy of ancient, forgotten forest.\n\n"
            "***\n\n"
            "### Prompt for Image Generation\n\n"
            "Since I cannot generate the image, I am providing a prompt tailored for an ethereal fantasy style.\n\n"
            "**Prompt:**\n\n"
            "\"An overgrown, subterranean crystal atrium with immense hanging amethyst and jade crystals, "
            "a glowing phosphorescent pool, volumetric rainbow light, humid ancient atmosphere, "
            "fantasy concept art, deep focus, cinematic lighting.\""
        )

        payload = split_visible_image_payload(display_text)

        self.assertEqual(
            payload['artifact_prompt'],
            'An overgrown, subterranean crystal atrium with immense hanging amethyst and jade crystals, '
            'a glowing phosphorescent pool, volumetric rainbow light, humid ancient atmosphere, '
            'fantasy concept art, deep focus, cinematic lighting.',
        )
        self.assertEqual(payload['artifact_prompt_source'], 'quoted_prompt_section')

    def test_split_visible_image_payload_extracts_inline_prompt_capsule(self):
        display_text = (
            'The place is a vast luminous archive where every corridor hums with memory.\n\n'
            '[Image prompt: A glowing impossible library of light with infinite shelves, floating data streams, '
            'and cathedral-scale perspective.]'
        )

        payload = split_visible_image_payload(display_text)

        self.assertEqual(
            payload['artifact_prompt'],
            'A glowing impossible library of light with infinite shelves, floating data streams, '
            'and cathedral-scale perspective.',
        )
        self.assertEqual(payload['artifact_prompt_source'], 'inline_prompt_capsule')
        self.assertEqual(
            payload['content_payload'],
            'The place is a vast luminous archive where every corridor hums with memory.',
        )

    def test_split_visible_image_payload_does_not_extract_unrelated_quoted_text(self):
        payload = split_visible_image_payload('Generate me the following text: "bla bla"')

        self.assertEqual(payload['artifact_prompt'], 'Generate me the following text: "bla bla"')
        self.assertEqual(payload['artifact_prompt_source'], 'full_display_text')

    def test_request_phase_graph_tracks_parallel_audio_and_image_follow_ups_after_text_preparation(self):
        graph = build_request_phase_graph(
            'Write a short mystical story, then read it aloud and also show it to me as an image.'
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech', 'image_generation'])
        self.assertEqual(len(graph['phases']), 3)
        self.assertEqual(graph['phases'][1]['depends_on'], ['phase-1'])
        self.assertEqual(graph['phases'][2]['depends_on'], ['phase-1'])

    def test_request_phase_graph_marks_german_joke_then_read_aloud_as_tts_chain(self):
        graph = build_request_phase_graph(
            'Hallo Olmo, erzähl mir mal einen Witz und lese ihn mir dann vor, bitte.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['text_preparation_before_audio_output'])
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(len(graph['downstream_branches']), 1)

    def test_request_phase_graph_treats_quoted_tts_passage_as_payload_not_image_work(self):
        prompt = (
            'Create exactly one English audio artifact using local text-to-speech. '
            'Read only the quoted passage. Do not create or plan an image. '
            '"Nothing was finished, but everything was finally moving."'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(
            [branch.get('capability') for branch in graph['downstream_branches']],
            ['text_to_speech'],
        )
        self.assertEqual(graph['prompt_intent']['required_intent_output_counts'], {'audio': 1})
        self.assertFalse(graph['prompt_intent']['requests_visual_output'])

    def test_request_phase_graph_owns_reported_answer_as_audio_intent_before_output_claims(self):
        prompts = (
            'Erzähl mir etwas über doch. gib mir die antwort als generiertes audio.',
            'Erzähl mir etwas über dich. gib mir die antwort als generiertes audio.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                self.assertEqual(graph['current_phase_capability'], 'chat')
                self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
                self.assertEqual(graph['mode'], 'carried_phase_chain')
                self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
                self.assertTrue(graph['prompt_intent']['requests_audio_output'])
                self.assertTrue(graph['prompt_intent']['direct_audio_materialization_request'])
                self.assertTrue(graph['prompt_intent']['text_preparation_before_audio_output'])
                self.assertFalse(graph['prompt_intent']['refined_from_output_claim'])
                self.assertEqual(graph.get('graph_refinements') or [], [])
                self.assertEqual(len(graph['downstream_branches']), 1)
                self.assertEqual(graph['downstream_branches'][0]['source'], 'request_phase_graph')
                self.assertEqual(graph['downstream_branches'][0]['depends_on'], ['phase-1'])

    def test_request_phase_graph_marks_german_describe_image_then_generate_as_image_chain(self):
        graph = build_request_phase_graph(
            'beschreibt mir mal ein lustiges bild von einem tier in einem humanen umfeld '
            'völlig fehl am platz und generieren',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 1)

    def test_request_phase_graph_preserves_post_image_text_continuation(self):
        graph = build_request_phase_graph(
            'Schreibe zuerst einen sehr kurzen Bildprompt fuer ein kleines rotes Segelboot im Nebel, '
            'generiere danach ein Bild davon, und schreibe nach der Bildgenerierung einen Satz, '
            'der das erzeugte Bild beschreibt.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(graph['downstream_branches'][0]['depends_on'], ['phase-1'])
        self.assertEqual(graph['downstream_branches'][1]['depends_on'], ['phase-2'])
        self.assertEqual(graph['downstream_branches'][2]['depends_on'], ['phase-3'])
        self.assertEqual(graph['downstream_branches'][2]['kind'], 'postprocess')
        self.assertEqual(graph['phases'][3]['role'], 'post_artifact_text_follow_up')

    def test_request_phase_graph_preserves_image_then_caption_continuation(self):
        graph = build_request_phase_graph(
            'Write one vivid image prompt for a glass lighthouse in a storm, generate the image, '
            'then write one caption after the image generation.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )
        self.assertEqual(graph['phases'][3]['role'], 'post_artifact_text_follow_up')

    def test_request_phase_graph_fans_out_single_generate_two_images_before_comparison(self):
        graph = build_request_phase_graph(
            'Write two separate image prompts for two alien gardens, generate two images from them, '
            'then compare the two generated images in one sentence.',
            request_payload={
                'ghost_route': True,
                'prompt': 'Write two separate image prompts for two alien gardens, generate two images from them, then compare them.',
            },
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'image_generation', 'vision_analysis', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-1'], ['phase-2'], ['phase-3'], ['phase-4', 'phase-5']],
        )
        self.assertEqual([item.get('queue_index') for item in graph['downstream_branches'][:2]], [1, 2])

    def test_request_phase_graph_detects_german_generated_image_comparison(self):
        graph = build_request_phase_graph(
            'Erstelle zwei unterschiedliche Bildideen als Bilder und vergleiche danach nur die tatsächlich generierten Bilder.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'image_generation', 'vision_analysis', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-1'], ['phase-2'], ['phase-3'], ['phase-4', 'phase-5']],
        )

    def test_request_phase_graph_forks_audio_image_before_final_artifact_sentence(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-1'], ['phase-3'], ['phase-2', 'phase-4']],
        )
        self.assertEqual(graph['downstream_branches'][2]['kind'], 'evidence')
        self.assertEqual(graph['downstream_branches'][3]['kind'], 'postprocess')
        self.assertEqual(graph['phases'][4]['role'], 'post_artifact_text_follow_up')

    def test_request_phase_graph_preserves_generated_poster_visual_evidence_caption(self):
        graph = build_request_phase_graph(
            'Write a one-line slogan for Ollmo, generate a poster image for it, '
            'analyze the generated poster image, then write a final caption that references '
            'the actual visual evidence.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )
        self.assertEqual(graph['downstream_branches'][2]['kind'], 'postprocess')

    def test_request_phase_graph_preserves_analyze_both_generated_images_before_comparison(self):
        graph = build_request_phase_graph(
            'Generate two images: first a calm blue moon garden, second a sharp orange desert machine. '
            'Analyze both generated images, then write one sentence comparing mood, color, and shape.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'image_generation', 'vision_analysis', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-1'], ['phase-2'], ['phase-3'], ['phase-4', 'phase-5']],
        )
        self.assertEqual([item.get('queue_index') for item in graph['downstream_branches'][:2]], [1, 2])

    def test_request_phase_graph_forks_audio_image_stt_and_vision_before_final_sentence(self):
        graph = build_request_phase_graph(
            'Write a short product launch line, turn it into audio, generate a matching image, '
            'analyze the generated image, transcribe the generated audio, then write one final '
            'sentence that uses both the transcript and visual evidence.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'image_generation', 'vision_analysis', 'speech_to_text', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-1'], ['phase-3'], ['phase-2'], ['phase-5', 'phase-4']],
        )
        self.assertEqual(graph['downstream_branches'][2]['kind'], 'evidence')
        self.assertEqual(graph['downstream_branches'][4]['kind'], 'postprocess')

    def test_request_phase_graph_forks_german_audio_poster_stt_and_vision_before_final_sentence(self):
        graph = build_request_phase_graph(
            'Schreibe einen kurzen Slogan, mache daraus Audio, generiere zusätzlich ein Posterbild '
            'zum Slogan, analysiere das Posterbild, transkribiere das Audio und schreibe am Ende '
            'einen Satz, der nur Audio-Transkript und Bildanalyse nutzt.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'image_generation', 'vision_analysis', 'speech_to_text', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-1'], ['phase-3'], ['phase-2'], ['phase-5', 'phase-4']],
        )
        self.assertEqual(graph['downstream_branches'][4]['kind'], 'postprocess')

    def test_request_phase_graph_exports_hierarchical_workload_for_mixed_media_join(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        workload_graph = graph['workload_graph']
        tasks_by_phase = {item['phase_id']: item for item in workload_graph['tasks']}

        self.assertEqual(
            workload_graph['task_ids'],
            ['task-phase-1', 'task-phase-2', 'task-phase-3', 'task-phase-4', 'task-phase-5'],
        )
        self.assertEqual(workload_graph['leaf_task_ids'], ['task-phase-5'])
        self.assertEqual(tasks_by_phase['phase-2']['parent_task_ids'], ['task-phase-1'])
        self.assertEqual(tasks_by_phase['phase-3']['parent_task_ids'], ['task-phase-1'])
        self.assertEqual(tasks_by_phase['phase-4']['parent_task_ids'], ['task-phase-3'])
        self.assertEqual(tasks_by_phase['phase-5']['parent_task_ids'], ['task-phase-2', 'task-phase-4'])
        self.assertEqual(tasks_by_phase['phase-4']['visibility'], 'evidence')
        self.assertEqual(
            tasks_by_phase['phase-4']['output_contract']['fulfillment_policy'],
            'runtime_evidence_text',
        )
        self.assertEqual(tasks_by_phase['phase-1']['child_task_ids'], ['task-phase-2', 'task-phase-3'])
        self.assertEqual(tasks_by_phase['phase-5']['decomposition_level'], 3)
        self.assertTrue(tasks_by_phase['phase-5']['lifecycle']['recursive'])
        self.assertEqual(
            tasks_by_phase['phase-5']['lifecycle']['cycle'],
            ['prepare', 'gather_evidence', 'execute', 'verify', 'repair_or_freeze'],
        )
        self.assertIn('uses_dependency_evidence', tasks_by_phase['phase-5']['review_criteria'])
        self.assertIn('does_not_restart_root_request', tasks_by_phase['phase-5']['review_criteria'])
        coverage = graph['workload_proposal_review']['coverage']
        self.assertEqual(coverage['status'], 'missing')
        self.assertIn('task-phase-5', coverage['missing_task_ids'])

    def test_request_phase_graph_accepts_bounded_ghost_workload_task_proposals(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={
                'route_source': 'ghost_carried',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'proposal-source-text',
                        'phase_id': 'phase-1',
                        'capability': 'chat',
                        'semantic_intent': 'Draft only the bounded slogan source text.',
                        'review_criteria': ['contains only the slogan payload'],
                        'input_refs': [{'kind': 'user_prompt', 'ref': 'intent_anchor'}],
                    },
                    {
                        'proposal_id': 'proposal-final-join',
                        'phase_id': 'phase-5',
                        'capability': 'chat',
                        'depends_on': ['phase-2', 'phase-4'],
                        'semantic_intent': 'Write one final sentence grounded in generated audio and visual evidence.',
                        'input_refs': [
                            {'kind': 'phase_output', 'phase_id': 'phase-2', 'role': 'audio artifact'},
                            {'kind': 'phase_output', 'phase_id': 'phase-4', 'role': 'vision evidence'},
                        ],
                        'review_criteria': [
                            'references the generated audio artifact',
                            'references the visual evidence',
                        ],
                    },
                ],
            },
        )

        review = graph['workload_proposal_review']
        tasks_by_phase = {item['phase_id']: item for item in graph['workload_graph']['tasks']}

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(review['accepted_count'], 2)
        self.assertEqual(review['rejected_count'], 0)
        self.assertEqual(review['coverage']['status'], 'partial')
        self.assertIn('task-phase-1', review['coverage']['accepted_task_ids'])
        self.assertIn('task-phase-5', review['coverage']['accepted_task_ids'])
        self.assertIn('task-phase-2', review['coverage']['missing_task_ids'])
        self.assertEqual(tasks_by_phase['phase-1']['semantic_intent'], 'Draft only the bounded slogan source text.')
        self.assertIn('contains only the slogan payload', tasks_by_phase['phase-1']['review_criteria'])
        self.assertEqual(
            tasks_by_phase['phase-5']['semantic_intent'],
            'Write one final sentence grounded in generated audio and visual evidence.',
        )
        self.assertIn('references the visual evidence', tasks_by_phase['phase-5']['review_criteria'])
        self.assertEqual(tasks_by_phase['phase-5']['parent_task_ids'], ['task-phase-2', 'task-phase-4'])

    def test_request_phase_graph_rejects_workload_task_proposal_that_changes_edges(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={
                'route_source': 'ghost_carried',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'proposal-bad-edge',
                        'phase_id': 'phase-5',
                        'capability': 'chat',
                        'depends_on': ['phase-1'],
                        'semantic_intent': 'Bypass artifact evidence and answer from the root prompt.',
                    }
                ],
            },
        )

        review = graph['workload_proposal_review']
        final_task = graph['workload_graph']['tasks'][-1]

        self.assertEqual(review['status'], 'rejected')
        self.assertEqual(review['accepted_count'], 0)
        self.assertEqual(review['rejected_count'], 1)
        self.assertEqual(review['rejections'][0]['reason'], 'depends_on_mismatch')
        self.assertNotIn('semantic_intent', final_task)
        self.assertEqual(final_task['parent_task_ids'], ['task-phase-2', 'task-phase-4'])

    def test_router_sanitizes_rich_workload_execution_contract_proposal(self):
        proposals = _sanitize_workload_task_proposals(
            [
                {
                    'proposal_id': 'rich-final',
                    'phase_id': 'phase-5',
                    'task_id': 'task-phase-5',
                    'capability': 'chat',
                    'decomposition_level': 42,
                    'semantic_intent': 'Write the final sentence from generated audio and visual evidence only.',
                    'advisory_role': 'semantic join planner',
                    'evidence_requirements': ['audio artifact must exist', 'vision evidence must exist'],
                    'semantic_review_criteria': ['final sentence must reference both generated branches'],
                    'reconsideration_triggers': ['if one generated branch is waived'],
                    'learning_hint_refs': ['accepted-policy-workload-decision'],
                    'promotion_suggestions': [
                        {
                            'candidate_id': 'candidate-final-join',
                            'promotion_reason': 'final branch is owed after both generated branches exist',
                        }
                    ],
                    'waiver_candidates': [
                        {
                            'obligation_id': 'obligation-phase-4',
                            'waiver_reason': 'visual branch was explicitly waived by the user',
                        }
                    ],
                    'repair_candidates': [
                        {
                            'task_id': 'task-phase-5',
                            'repair_action': 'repair_dependency_chain',
                            'reason': 'dependency evidence missing',
                        }
                    ],
                    'supersession_candidates': [
                        {
                            'obligation_id': 'obligation-phase-4',
                            'superseded_by_obligation_id': 'obligation-phase-5',
                            'supersession_reason': 'final join replaces intermediate prose',
                        }
                    ],
                    'execution_contract': {
                        'kind': 'ollmo.execution_contract',
                        'branch_id': 'branch-chat-1',
                        'phase_id': 'phase-5',
                        'capability': 'chat',
                        'output_type': 'text',
                        'workload_task_ref': {
                            'task_id': 'task-phase-5',
                            'phase_id': 'phase-5',
                            'branch_id': 'branch-chat-1',
                        },
                        'output_obligation_ref': {
                            'obligation_id': 'obligation-phase-5',
                            'phase_id': 'phase-5',
                            'branch_id': 'branch-chat-1',
                            'output_type': 'text',
                        },
                        'output_contract': {
                            'output_type': 'text',
                            'required': True,
                            'fulfillment_policy': 'runtime_text',
                        },
                        'depends_on': ['phase-2', 'phase-4'],
                        'input_refs': [
                            {'kind': 'phase_output', 'phase_id': 'phase-2', 'role': 'audio artifact'},
                            {'kind': 'phase_output', 'phase_id': 'phase-4', 'role': 'vision evidence'},
                        ],
                    },
                },
            ]
        )

        self.assertEqual(proposals[0]['execution_contract']['branch_id'], 'branch-chat-1')
        self.assertEqual(proposals[0]['execution_contract']['workload_task_ref']['task_id'], 'task-phase-5')
        self.assertEqual(proposals[0]['execution_contract']['output_contract']['fulfillment_policy'], 'runtime_text')
        self.assertEqual(proposals[0]['execution_contract']['depends_on'], ['phase-2', 'phase-4'])
        self.assertEqual(proposals[0]['decomposition_level'], 42)
        self.assertEqual(proposals[0]['advisory_role'], 'semantic join planner')
        self.assertEqual(proposals[0]['evidence_requirements'], ['audio artifact must exist', 'vision evidence must exist'])
        self.assertEqual(
            proposals[0]['semantic_review_criteria'],
            ['final sentence must reference both generated branches'],
        )
        self.assertEqual(proposals[0]['repair_candidates'][0]['repair_action'], 'repair_dependency_chain')
        self.assertEqual(proposals[0]['supersession_candidates'][0]['obligation_id'], 'obligation-phase-4')
        self.assertEqual(proposals[0]['promotion_suggestions'][0]['candidate_id'], 'candidate-final-join')
        self.assertEqual(proposals[0]['waiver_candidates'][0]['obligation_id'], 'obligation-phase-4')

    def test_request_phase_graph_accepts_rich_ghost_execution_contract_proposal(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={
                'route_source': 'ghost_carried',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'rich-final',
                        'phase_id': 'phase-5',
                        'task_id': 'task-phase-5',
                        'branch_id': 'branch-chat-1',
                        'capability': 'chat',
                        'decomposition_level': 42,
                        'semantic_intent': 'Write one final sentence from generated audio and visual evidence only.',
                        'objective': 'Join the two fulfilled artifact branches without restarting generation.',
                        'advisory_role': 'branch_local_semantic_planner',
                        'evidence_requirements': ['audio artifact exists', 'vision evidence text exists'],
                        'semantic_review_criteria': ['uses generated audio and visual evidence'],
                        'reconsideration_triggers': ['generated evidence changes'],
                        'learning_hint_refs': ['accepted-policy-workload-decision'],
                        'promotion_suggestions': [
                            {
                                'candidate_id': 'candidate-final-join',
                                'promotion_reason': 'both evidence branches are available',
                            }
                        ],
                        'waiver_candidates': [
                            {
                                'obligation_id': 'obligation-phase-3',
                                'waiver_reason': 'intermediate summary is no longer required after final join',
                            }
                        ],
                        'repair_candidates': [
                            {
                                'task_id': 'task-phase-5',
                                'repair_action': 'repair_dependency_chain',
                                'reason': 'missing evidence branch',
                            }
                        ],
                        'supersession_candidates': [
                            {
                                'obligation_id': 'obligation-phase-3',
                                'superseded_by_obligation_id': 'obligation-phase-5',
                                'supersession_reason': 'final text supersedes intermediate summary',
                            }
                        ],
                        'execution_contract': {
                            'kind': 'ollmo.execution_contract',
                            'branch_id': 'branch-chat-1',
                            'phase_id': 'phase-5',
                            'capability': 'chat',
                            'output_type': 'text',
                            'workload_task_ref': {
                                'task_id': 'task-phase-5',
                                'phase_id': 'phase-5',
                                'branch_id': 'branch-chat-1',
                            },
                            'output_obligation_ref': {
                                'obligation_id': 'obligation-phase-5',
                                'phase_id': 'phase-5',
                                'branch_id': 'branch-chat-1',
                                'output_type': 'text',
                            },
                            'output_contract': {
                                'output_type': 'text',
                                'required': True,
                                'fulfillment_policy': 'runtime_text',
                            },
                            'depends_on': ['phase-2', 'phase-4'],
                            'input_refs': [
                                {'kind': 'phase_output', 'phase_id': 'phase-2', 'role': 'audio artifact'},
                                {'kind': 'phase_output', 'phase_id': 'phase-4', 'role': 'vision evidence'},
                            ],
                        },
                        'review_criteria': ['mentions audio and visual evidence without reusing the root prompt'],
                    },
                ],
            },
        )

        review = graph['workload_proposal_review']
        tasks_by_phase = {item['phase_id']: item for item in graph['workload_graph']['tasks']}
        final_task = tasks_by_phase['phase-5']

        self.assertEqual(review['status'], 'accepted')
        self.assertEqual(review['accepted_count'], 1)
        self.assertEqual(final_task['execution_contract']['branch_id'], 'branch-chat-1')
        self.assertEqual(final_task['workload_task_ref']['task_id'], 'task-phase-5')
        self.assertEqual(final_task['output_obligation_ref']['obligation_id'], 'obligation-phase-5')
        self.assertEqual(final_task['decomposition_level'], 42)
        self.assertEqual(
            final_task['semantic_intent'],
            'Write one final sentence from generated audio and visual evidence only.',
        )
        self.assertIn(
            'mentions audio and visual evidence without reusing the root prompt',
            final_task['review_criteria'],
        )
        self.assertEqual(final_task['advisory_role'], 'branch_local_semantic_planner')
        self.assertEqual(final_task['evidence_requirements'], ['audio artifact exists', 'vision evidence text exists'])
        self.assertIn('uses generated audio and visual evidence', final_task['semantic_review_criteria'])
        self.assertEqual(final_task['promotion_suggestions'][0]['candidate_id'], 'candidate-final-join')
        self.assertEqual(final_task['waiver_candidates'][0]['obligation_id'], 'obligation-phase-3')
        self.assertEqual(final_task['repair_candidates'][0]['repair_action'], 'repair_dependency_chain')
        self.assertEqual(final_task['supersession_candidates'][0]['obligation_id'], 'obligation-phase-3')
        self.assertIn('task-phase-5', graph['workload_proposal_review']['coverage']['accepted_task_ids'])
        semantic_review_tasks = {
            item['task_id']: item
            for item in graph['decision_contract']['semantic_review_candidates']
        }
        self.assertIn('task-phase-5', semantic_review_tasks)
        self.assertEqual(graph['decision_contract']['promotion_suggestions'][0]['task_id'], 'task-phase-5')
        self.assertEqual(graph['decision_contract']['waiver_candidates'][0]['task_id'], 'task-phase-5')
        self.assertIn(
            'review_advisory_promotion_suggestions_before_promoting_work',
            graph['decision_contract']['semantic_planning_contract']['current_focus'],
        )
        self.assertEqual(graph['decision_contract']['repair_candidates'][0]['task_id'], 'task-phase-5')
        self.assertEqual(graph['decision_contract']['supersession_candidates'][0]['task_id'], 'task-phase-5')
        final_obligation = next(
            item for item in graph['output_obligations']
            if item['phase_id'] == 'phase-5'
        )
        self.assertEqual(final_obligation['advisory_role'], 'branch_local_semantic_planner')
        self.assertIn('uses generated audio and visual evidence', final_obligation['semantic_review_criteria'])
        self.assertEqual(final_obligation['promotion_suggestions'][0]['candidate_id'], 'candidate-final-join')
        self.assertEqual(final_obligation['waiver_candidates'][0]['obligation_id'], 'obligation-phase-3')

    def test_request_phase_graph_rejects_ghost_execution_contract_dependency_mismatch(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={
                'route_source': 'ghost_carried',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'bad-contract',
                        'phase_id': 'phase-5',
                        'capability': 'chat',
                        'execution_contract': {
                            'branch_id': 'branch-chat-1',
                            'phase_id': 'phase-5',
                            'capability': 'chat',
                            'output_type': 'text',
                            'depends_on': ['phase-1'],
                        },
                    },
                ],
            },
        )

        review = graph['workload_proposal_review']
        final_task = graph['workload_graph']['tasks'][-1]

        self.assertEqual(review['status'], 'rejected')
        self.assertEqual(review['rejected_count'], 1)
        self.assertEqual(review['rejections'][0]['reason'], 'execution_contract_depends_on_mismatch')
        self.assertNotIn('execution_contract', final_task)

    def test_accepted_workload_task_contract_projects_to_downstream_branch(self):
        graph = build_request_phase_graph(
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.',
            request_payload={'ghost_route': True},
            route_payload={
                'route_source': 'ghost_carried',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'rich-final',
                        'phase_id': 'phase-5',
                        'capability': 'chat',
                        'semantic_intent': 'Final branch must use only generated artifact evidence.',
                        'evidence_requirements': ['generated audio and image evidence'],
                        'semantic_review_criteria': ['does not answer from root prompt alone'],
                        'execution_contract': {
                            'branch_id': 'branch-chat-1',
                            'phase_id': 'phase-5',
                            'capability': 'chat',
                            'output_type': 'text',
                            'workload_task_ref': {
                                'task_id': 'task-phase-5',
                                'phase_id': 'phase-5',
                                'branch_id': 'branch-chat-1',
                            },
                            'output_obligation_ref': {
                                'obligation_id': 'obligation-phase-5',
                                'phase_id': 'phase-5',
                                'branch_id': 'branch-chat-1',
                                'output_type': 'text',
                            },
                            'output_contract': {
                                'output_type': 'text',
                                'required': True,
                                'fulfillment_policy': 'runtime_text',
                            },
                            'depends_on': ['phase-2', 'phase-4'],
                        },
                    },
                ],
            },
        )

        final_branch = next(
            item for item in graph['downstream_branches']
            if item['phase_id'] == 'phase-5'
        )

        self.assertEqual(final_branch['semantic_intent'], 'Final branch must use only generated artifact evidence.')
        self.assertEqual(final_branch['evidence_requirements'], ['generated audio and image evidence'])
        self.assertIn('does not answer from root prompt alone', final_branch['semantic_review_criteria'])
        self.assertEqual(final_branch['execution_contract']['branch_id'], 'branch-chat-1')
        self.assertEqual(final_branch['execution_contract']['workload_task_ref']['task_id'], 'task-phase-5')
        self.assertEqual(final_branch['output_contract']['fulfillment_policy'], 'runtime_text')

    def test_request_phase_graph_detects_turn_previous_text_into_audio_then_confirm(self):
        graph = build_request_phase_graph(
            'Write a tiny launch announcement, turn it into audio, then confirm the exact spoken text.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

    def test_request_phase_graph_depends_generated_audio_transcription_on_audio_branch(self):
        graph = build_request_phase_graph(
            'Schreibe einen kurzen Satz, mache daraus Audio, transkribiere danach das Audio '
            'und sag mir am Ende, ob die Transkription dem Satz entspricht.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'speech_to_text'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2']],
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'speech_to_text'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2']],
        )

    def test_request_phase_graph_preserves_post_audio_text_continuation(self):
        graph = build_request_phase_graph(
            'Schreibe zuerst einen sehr kurzen deutschen Satz, lies ihn danach als Audio vor, '
            'und schreibe nach der Audiogenerierung eine kurze Bestaetigung mit dem exakten gesprochenen Text.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'speech_to_text'],
        )
        self.assertEqual(graph['downstream_branches'][0]['depends_on'], ['phase-1'])
        self.assertEqual(graph['downstream_branches'][1]['depends_on'], ['phase-2'])
        self.assertEqual(graph['phases'][2]['output_type'], 'text')

    def test_request_phase_graph_speaks_only_selected_best_candidate(self):
        graph = build_request_phase_graph(
            'Skizziere drei Poster-Ideen für eine verlassene Mondbibliothek. '
            'Generiere aber noch kein Bild. Lies die beste Idee als kurzes Audio vor.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        promoted = [
            item for item in graph['downstream_branches']
            if item.get('contract_state') != 'reserved'
        ]

        self.assertEqual([item['capability'] for item in promoted], ['text_to_speech'])
        self.assertEqual(promoted[0]['selection_policy'], 'best_candidate_only')
        self.assertEqual(promoted[0]['candidate_selection_index'], 1)
        self.assertEqual(promoted[0]['content_payload_source'], 'selected_candidate_from_phase_output')

    def test_request_phase_graph_inserts_audio_evidence_before_quality_review(self):
        graph = build_request_phase_graph(
            'Schreibe eine ernste Sicherheitswarnung für eine Raumstation bei Sauerstoffverlust. '
            'Generiere daraus Audio und prüfe danach in Textform, ob die gesprochene Version eindringlich genug wirkt.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'speech_to_text', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )

    def test_request_phase_graph_promotes_selected_audio_reference_for_analysis(self):
        graph = build_request_phase_graph(
            'Hier hast du das Audio, jetzt kannst du die Analyse durchführen.',
            request_payload={
                'ghost_route': True,
                'selected_reference_artifacts': [
                    {'type': 'audio', 'path': '/tmp/generated-warning.wav'},
                ],
            },
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
        self.assertEqual(graph['downstream_capabilities'], [])

    def test_request_phase_graph_normalizes_german_create_typo_for_image_follow_up(self):
        graph = build_request_phase_graph(
            'Da hast du recht. Dann kreiiere sie zuerst, basierend auf diesem Workbench. '
            'Und dann das Poster.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])

    def test_request_phase_graph_preserves_arbitrary_dependent_phase_depth(self):
        graph = build_request_phase_graph(
            'Schreibe zuerst einen kurzen Bildprompt fuer einen goldenen Fuchs im Schnee, '
            'generiere ein Bild daraus, schreibe danach eine einzeilige Beschreibung des erzeugten Bildes, '
            'und lies diese Beschreibung anschliessend als Audio vor.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat', 'text_to_speech'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3'], ['phase-4']],
        )

    def test_request_phase_graph_preserves_audio_text_image_dependent_phase_depth(self):
        graph = build_request_phase_graph(
            'Schreibe einen kurzen Satz, lies ihn danach als Audio vor, schreibe nach der Audiogenerierung '
            'eine Bestaetigung, und generiere anschliessend ein Bild basierend auf dieser Bestaetigung.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['text_to_speech', 'chat', 'image_generation'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )

    def test_request_phase_graph_preserves_repeated_capability_in_dependent_chain(self):
        graph = build_request_phase_graph(
            'Schreibe zuerst einen kurzen Bildprompt fuer eine blaue Laterne im Regen, '
            'generiere ein Bild daraus, schreibe danach eine kurze Beschreibung des Bildes, '
            'und generiere anschliessend ein zweites Bild basierend auf dieser Beschreibung.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat', 'image_generation'],
        )
        self.assertEqual(
            [item['phase_id'] for item in graph['downstream_branches']],
            ['phase-2', 'phase-3', 'phase-4', 'phase-5'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3'], ['phase-4']],
        )

    def test_request_phase_graph_refines_from_explicit_assistant_pending_tts_claim(self):
        graph = build_request_phase_graph(
            'Tell me a joke.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            response_payload={
                'output_text': (
                    'Here is a joke. Phase 2: text_to_speech. '
                    'Status: pending. Trigger audio generation branch.'
                )
            },
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(graph['downstream_branches'][0]['source'], 'assistant_output_claim_refinement')
        self.assertEqual(graph['graph_refinements'][0]['capability'], 'text_to_speech')

    def test_request_phase_graph_preserves_duplicate_same_capability_branches(self):
        graph = build_request_phase_graph(
            'Describe two dream places and show them to me as two separate images.',
            request_payload={
                'prompt': 'Describe two dream places and show them to me as two separate images.',
                'downstream_branches': [
                    {
                        'branch_id': 'phase-image-1',
                        'phase_id': 'phase-image-1',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'depends_on': ['phase-1'],
                    },
                    {
                        'branch_id': 'phase-image-2',
                        'phase_id': 'phase-image-2',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'depends_on': ['phase-1'],
                    },
                ],
            },
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(graph['downstream_branch_ids'], ['phase-image-1', 'phase-image-2'])
        self.assertEqual(len(graph['downstream_branches']), 2)
        self.assertEqual(len(graph['phases']), 3)
        self.assertEqual(graph['phases'][1]['branch_id'], 'phase-image-1')
        self.assertEqual(graph['phases'][2]['branch_id'], 'phase-image-2')

    def test_request_phase_graph_infers_multiple_image_branches_from_natural_language_count(self):
        graph = build_request_phase_graph(
            'Beschreibe mir bitte zwei Traumorte in lebendigen Details und zeig sie mir als zwei Bilder in der Antwort.'
        )

        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 2)
        self.assertEqual(graph['downstream_branch_ids'], ['branch-image_generation-1', 'branch-image_generation-2'])
        self.assertEqual(len(graph['downstream_branches']), 2)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][1]['queue_index'], 2)

    def test_request_phase_graph_detects_german_image_variant_compound_as_multi_image_chain(self):
        graph = build_request_phase_graph(
            'Erzeuge zwei unterschiedliche Bildvarianten einer mobilen Forschungsstation in einer Wüste: '
            'eine realistische Nachtaufnahme und eine technische Schnittzeichnung. '
            'Warte mit dem Vergleich, bis beide Bilder wirklich erzeugt sind. '
            'Vergleiche danach in genau drei Sätzen, welche Variante besser für ein Sicherheitsbriefing geeignet ist.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 2)
        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches']],
            ['image_generation', 'image_generation', 'vision_analysis', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            len([branch for branch in graph['downstream_branches'] if branch['capability'] == 'image_generation']),
            2,
        )

    def test_request_phase_graph_uses_current_turn_for_history_payload_and_nenne_followup(self):
        current_prompt = (
            'Erzeuge ein Bild eines autonomen Polar-Forschungszugs in einem Schneesturm. '
            'Danach analysiere das generierte Bild und nenne im Chat drei sichtbare technische Risiken.'
        )
        history_prompt = (
            '[user]\nBaue index.html und style.css Artefakte.\n\n'
            '[assistant]\n```html\n<!doctype html><h1>Old</h1>\n```\n\n'
            '```css\nbody { color: blue; }\n```\n\n'
            f'[user]\n{current_prompt}'
        )
        request_payload = {
            'input': [
                {'role': 'user', 'content': 'Baue index.html und style.css Artefakte.'},
                {
                    'role': 'assistant',
                    'content': (
                        '```html\n<!doctype html><h1>Old</h1>\n```\n\n'
                        '```css\nbody { color: blue; }\n```'
                    ),
                },
                {'role': 'user', 'content': current_prompt},
            ],
            'ghost_route': True,
        }

        graph = build_request_phase_graph(
            history_prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['prompt'], current_prompt)
        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertFalse(graph['prompt_intent']['requests_text_artifact_output'])

    def test_request_phase_graph_does_not_recover_old_source_blocks_for_new_save_request(self):
        current_prompt = (
            'Erzeuge einen kurzen Warnslogan für den Polar-Forschungszug, '
            'speichere ihn als txt-Artefakt, erzeuge daraus Audio und zusätzlich ein Posterbild.'
        )
        request_payload = {
            'input': [
                {'role': 'user', 'content': 'Baue index.html und style.css Artefakte.'},
                {
                    'role': 'assistant',
                    'content': (
                        '```html\n<!doctype html><h1>Old</h1>\n```\n\n'
                        '```css\nbody { color: blue; }\n```'
                    ),
                },
                {'role': 'user', 'content': current_prompt},
            ],
            'ghost_route': True,
        }

        graph = build_request_phase_graph(
            '[user]\nold\n\n[assistant]\n```html\nold\n```\n\n```css\nold\n```\n\n[user]\n' + current_prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['prompt_intent']['text_artifact_extensions'], ['txt'])
        self.assertNotIn('html', graph['prompt_intent']['text_artifact_extensions'])
        self.assertNotIn('css', graph['prompt_intent']['text_artifact_extensions'])

    def test_warning_prompt_infers_serious_urgent_voice_instruct(self):
        instruct = infer_tts_instruct_from_prompt(
            'Erzeuge einen Warnslogan und mache daraus Audio für einen Notfall-Alarm.'
        )

        self.assertIn('serious', instruct)
        self.assertIn('urgent', instruct)
        self.assertIn('Avoid laughter', instruct)

    def test_request_phase_graph_exports_request_ir_output_obligations(self):
        graph = build_request_phase_graph(
            'Describe two tiny stage scenes and then generate images (2) of them.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        request_ir = graph['request_ir']
        obligations = request_ir['output_obligations']

        self.assertEqual(request_ir['kind'], 'ollmo.request_ir')
        self.assertEqual(request_ir['ir_version'], 1)
        self.assertEqual(graph['output_obligations'], obligations)
        self.assertEqual([item['output_type'] for item in obligations], ['text', 'image', 'image'])
        self.assertEqual(obligations[0]['role'], 'preparation_output')
        self.assertEqual(obligations[1]['role'], 'final_output')
        self.assertEqual(obligations[1]['branch_id'], 'branch-image_generation-1')
        self.assertEqual(obligations[2]['branch_id'], 'branch-image_generation-2')
        self.assertEqual(
            request_ir['final_output_obligation_ids'],
            ['obligation-phase-2', 'obligation-phase-3'],
        )
        workload_graph = request_ir['workload_graph']
        self.assertEqual(workload_graph['kind'], 'ollmo.workload_graph')
        self.assertEqual(workload_graph['workload_graph_version'], 1)
        self.assertEqual(graph['workload_graph'], workload_graph)
        self.assertEqual(request_ir['workload_task_ids'], ['task-phase-1', 'task-phase-2', 'task-phase-3'])
        self.assertEqual(request_ir['candidate_graph']['kind'], 'ollmo.candidate_graph')
        self.assertEqual(request_ir['candidate_graph']['type_counts']['output'], 3)
        self.assertEqual(request_ir['candidate_graph']['type_counts']['workload_task'], 3)
        self.assertEqual(request_ir['promotion_review']['kind'], 'ollmo.promotion_review')
        self.assertEqual(request_ir['promotion_review']['promoted_count'], 3)
        self.assertEqual(graph['candidate_graph'], request_ir['candidate_graph'])
        self.assertEqual(graph['promotion_review'], request_ir['promotion_review'])
        self.assertEqual(len(workload_graph['tasks']), len(graph['phases']))
        self.assertEqual(
            workload_graph['tasks'][0]['input_refs'],
            [{'kind': 'user_prompt', 'ref': 'intent_anchor'}],
        )
        self.assertEqual(workload_graph['tasks'][1]['parent_task_ids'], ['task-phase-1'])
        self.assertEqual(workload_graph['tasks'][1]['output_contract']['output_type'], 'image')

    def test_request_ir_preserves_superseded_promoted_obligation(self):
        request_ir = build_request_ir(
            intent_prompt='Write a text, replace the first image branch, and keep only the newer branch.',
            prompt_intent={},
            current_phase_id='phase-1',
            graph_mode='phase_chain',
            phases=[
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image-old',
                    'candidate_id': 'candidate-image-old',
                    'promoted_from_candidate_id': 'candidate-image-old',
                    'obligation_id': 'obligation-phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'superseded',
                    'superseded_by_obligation_id': 'obligation-phase-3',
                    'supersession_reason': 'newer image branch replaced this branch',
                },
                {
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-image-new',
                    'candidate_id': 'candidate-image-new',
                    'promoted_from_candidate_id': 'candidate-image-new',
                    'obligation_id': 'obligation-phase-3',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                },
            ],
        )

        obligations = {
            item['obligation_id']: item
            for item in request_ir['output_obligations']
        }
        self.assertEqual(obligations['obligation-phase-2']['status'], 'superseded')
        self.assertEqual(obligations['obligation-phase-2']['superseded_by_obligation_id'], 'obligation-phase-3')
        tasks_by_phase = {
            item['phase_id']: item
            for item in request_ir['workload_graph']['tasks']
        }
        self.assertEqual(tasks_by_phase['phase-2']['status'], 'superseded')
        self.assertEqual(tasks_by_phase['phase-2']['lifecycle']['stages'][1]['status'], 'superseded')
        self.assertEqual(request_ir['promotion_review']['superseded_count'], 2)
        superseded_decisions = [
            item['candidate_id']
            for item in request_ir['promotion_review']['decisions']
            if item.get('decision') == 'superseded'
        ]
        self.assertIn('candidate-image-old', superseded_decisions)

    def test_request_phase_graph_promotes_text_artifact_requests_to_required_branches(self):
        graph = build_request_phase_graph(
            'Baue ein kleines HTML+CSS Dashboard als Artefakte. Kein json output bitte.'
        )

        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(graph['prompt_intent']['text_artifact_extensions'], ['html', 'css'])
        artifact_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('role') == 'text_artifact_output'
        ]
        self.assertEqual(
            [(branch['text_artifact_extension'], branch['requires_artifact']) for branch in artifact_branches],
            [('html', True), ('css', True)],
        )
        artifact_obligations = [
            item for item in graph['request_ir']['output_obligations']
            if item.get('role') == 'text_artifact_output'
        ]
        self.assertEqual(
            [item.get('text_artifact_extension') for item in artifact_obligations],
            ['html', 'css'],
        )
        self.assertEqual(
            [item.get('fulfillment_policy') for item in artifact_obligations],
            ['runtime_text_artifact', 'runtime_text_artifact'],
        )

    def test_request_phase_graph_uses_reference_artifacts_for_markdown_artifact_request(self):
        prompt = (
            'Nutze das generierte Bild als Referenz und schreibe daraus ein kurzes '
            'Einsatzprotokoll als Markdown-Artefakt.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'input': prompt,
                'reference_artifacts': [
                    {'type': 'image', 'path': '/tmp/generated.png'},
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(graph['prompt_intent']['text_artifact_extensions'], ['md'])
        artifact_obligation = next(
            item for item in graph['request_ir']['output_obligations']
            if item.get('role') == 'text_artifact_output'
        )
        self.assertEqual(artifact_obligation['text_artifact_extension'], 'md')
        self.assertEqual(artifact_obligation['fulfillment_policy'], 'runtime_text_artifact')

    def test_request_phase_graph_uses_route_artifact_for_latest_image_markdown_artifact_request(self):
        prompt = (
            'Nutze das zuletzt generierte Bild als Referenz und schreibe daraus ein kurzes '
            'Einsatzprotokoll als Markdown-Artefakt.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'input': prompt},
            route_payload={
                'capability': 'vision_analysis',
                'route_source': 'ghost_carried',
                'route_artifact_path': '/tmp/generated-latest.png',
                'route_artifact_ref': '/tmp/generated-latest.png',
            },
        )

        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(graph['prompt_intent']['text_artifact_extensions'], ['md'])
        artifact_obligation = next(
            item for item in graph['request_ir']['output_obligations']
            if item.get('role') == 'text_artifact_output'
        )
        self.assertEqual(artifact_obligation['text_artifact_extension'], 'md')

    def test_request_phase_graph_promotes_each_part_markdown_artifacts(self):
        graph = build_request_phase_graph(
            'Erstelle ein dreiteiliges Lernmodul über “Wind als unsichtbare Architektur”: '
            'erst eine einfache Metapher, dann eine technische Erklärung, dann eine Mini-Übung. '
            'Speichere jedes Teil als eigenes Markdown-Artefakt. Prüfe danach, ob die drei Teile '
            'zusammen eine sinnvolle Lernprogression ergeben.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(graph['prompt_intent']['text_artifact_output_count'], 3)
        artifact_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('role') == 'text_artifact_output'
        ]
        self.assertEqual(len(artifact_branches), 3)
        self.assertEqual([branch['text_artifact_extension'] for branch in artifact_branches], ['md', 'md', 'md'])
        self.assertEqual(
            [branch['text_artifact_source_name'] for branch in artifact_branches],
            ['generated-md-part-1', 'generated-md-part-2', 'generated-md-part-3'],
        )

    def test_request_phase_graph_promotes_txt_artifact_audio_and_review_line(self):
        prompt = (
            'Erstelle einen kurzen Warnhinweis für Bergretter, speichere ihn als txt-Artefakt, '
            'erzeuge daraus Audio, und gib danach im Chat eine knappe Kontrollzeile aus, '
            'ob Text und Audio zusammenpassen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'input': prompt},
            route_payload={'capability': 'text_to_speech', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertIn('txt', graph['prompt_intent']['text_artifact_extensions'])
        promoted = [
            (item.get('capability'), item.get('output_type'), item.get('role'), item.get('text_artifact_extension'))
            for item in graph['request_ir']['output_obligations']
        ]
        self.assertIn(('chat', 'text', 'text_artifact_output', 'txt'), promoted)
        self.assertIn(('text_to_speech', 'audio', 'final_output', None), promoted)
        self.assertTrue(
            any(item[0] == 'chat' and item[2] == 'final_output' for item in promoted),
            promoted,
        )
        self.assertIn('speech_to_text', graph['downstream_capabilities'])

    def test_request_phase_graph_schedules_post_audio_quality_review(self):
        graph = build_request_phase_graph(
            'Schreibe eine ernsthafte Sicherheitswarnung für eine Lawinenstation in einem Satz. '
            'Generiere daraus Audio. Prüfe danach explizit, ob die gesprochene Version ernst '
            'und eindringlich wirkt. Falls nicht prüfbar, markiere es als nicht prüfbar.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches']],
            ['text_to_speech', 'speech_to_text', 'chat'],
        )
        self.assertEqual(
            [branch['depends_on'] for branch in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )
        self.assertEqual(graph['downstream_branches'][2]['role'], 'post_artifact_text_follow_up')

    def test_request_phase_graph_schedules_post_audio_review_when_action_precedes_danach(self):
        graph = build_request_phase_graph(
            'Schreibe eine ernste Sicherheitswarnung für eine Raumstation bei Sauerstoffverlust. '
            'Generiere daraus Audio und prüfe danach in Textform, ob die gesprochene Version '
            'eindringlich genug wirkt.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches']],
            ['text_to_speech', 'speech_to_text', 'chat'],
        )
        self.assertEqual(
            [branch['depends_on'] for branch in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )
        self.assertEqual(graph['downstream_branches'][2]['role'], 'post_artifact_text_follow_up')

    def test_request_phase_graph_keeps_reserved_candidates_out_of_output_obligations(self):
        graph = build_request_phase_graph(
            'Sketch a possible image direction, but do not generate it yet.',
            request_payload={
                'ghost_route': True,
                'downstream_branches': [
                    {
                        'candidate_id': 'candidate-image-1',
                        'branch_id': 'branch-image-possible-1',
                        'phase_id': 'phase-image-possible-1',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'required': False,
                        'contract_state': 'reserved',
                        'promotion_policy': 'requires_user_confirmation',
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        request_ir = graph['request_ir']
        self.assertEqual([item['output_type'] for item in request_ir['output_obligations']], ['text'])
        self.assertEqual(request_ir['output_candidates'][0]['candidate_id'], 'candidate-image-1')
        self.assertEqual(request_ir['output_candidates'][0]['status'], 'reserved')
        self.assertEqual(request_ir['output_candidates'][0]['promotion_policy'], 'requires_user_confirmation')
        self.assertEqual(graph['output_candidates'], request_ir['output_candidates'])
        self.assertEqual(downstream_phase_records(graph), [])

    def test_request_phase_graph_demotes_negated_direct_image_route_to_reserved_candidate(self):
        graph = build_request_phase_graph(
            'Skizziere eine mögliche Poster-Idee, aber generiere noch kein Bild. Merke sie nur als Option.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        request_ir = graph['request_ir']
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual([item['output_type'] for item in request_ir['output_obligations']], ['text'])
        self.assertEqual(request_ir['output_candidates'][0]['status'], 'reserved')
        self.assertEqual(request_ir['output_candidates'][0]['capability'], 'image_generation')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertFalse(graph['continuation_required'])
        self.assertEqual(downstream_phase_records(graph), [])

    def test_request_phase_graph_keeps_image_executable_when_prompt_only_disclaims_fulfillment(self):
        prompt = (
            'First generate exactly one local image artifact for the station. '
            'Then create exactly two local file artifacts: index.html and styles.css. '
            'Treat a prompt for an image as preparation only, not as the image artifact.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertIn('image_generation', graph['downstream_capabilities'])
        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches'] if branch.get('contract_state') == 'reserved'],
            [],
        )
        request_ir = graph['request_ir']
        self.assertIn(
            ('image_generation', 'image', 'runtime_artifact_or_branch_state'),
            [
                (
                    obligation['capability'],
                    obligation['output_type'],
                    obligation['fulfillment_policy'],
                )
                for obligation in request_ir['output_obligations']
            ],
        )
        self.assertEqual(
            sorted(
                obligation.get('text_artifact_extension')
                for obligation in request_ir['output_obligations']
                if obligation.get('fulfillment_policy') == 'runtime_text_artifact'
            ),
            ['css', 'html'],
        )

    def test_request_phase_graph_binds_page_artifacts_to_local_image_for_closure_rebind(self):
        prompt = (
            'Create a polished one-screen landing page for a fictional deep-sea research station called '
            'Aethelgard Abyss-7. Generate exactly one local image artifact first: a wide cinematic exterior view. '
            'Then create exactly two local file artifacts: 1. index.html 2. styles.css. '
            'Use the generated image as the hero background. '
            'Treat missing PNG/image artifact, missing index.html, or missing styles.css as incomplete.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'image_generation'
        ]
        text_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('role') == 'text_artifact_output'
        ]

        self.assertEqual(len(image_branches), 1)
        self.assertEqual(
            [(branch['text_artifact_extension'], branch['text_artifact_source_name']) for branch in text_branches],
            [('html', 'index'), ('css', 'styles')],
        )
        image_phase_ids = [branch['phase_id'] for branch in image_branches]
        self.assertEqual(
            [branch['depends_on'] for branch in text_branches],
            [image_phase_ids, image_phase_ids],
        )
        self.assertEqual([branch['content_payload_source'] for branch in text_branches], ['current_phase_output', 'current_phase_output'])
        self.assertTrue(all(branch.get('image_asset_binding_required') is True for branch in text_branches))
        self.assertEqual(
            {branch.get('dependency_contract') for branch in text_branches},
            {'local_visual_asset_binding'},
        )

    def test_request_phase_graph_keeps_multi_image_page_request_with_no_extra_artifacts_constraint(self):
        prompt = (
            'Create a polished, extended one-screen landing page for a fictional floating high-altitude '
            'aeroponics and atmospheric research station called Aetheria Zenith-9.\n\n'
            'First, identify and generate exactly three distinct image assets based on these visual requirements:\n'
            '1. A wide cinematic exterior view of a massive sci-fi station suspended inside turbulent clouds.\n'
            '2. An interior view of a zero-gravity greenhouse biome filled with bioluminescent plants.\n'
            '3. A high-tech atmospheric analysis laboratory with holographic telemetry displays.\n\n'
            'Then create exactly two local file artifacts:\n'
            '1. index.html\n'
            '2. styles.css\n\n'
            'The HTML must structurally integrate all three image assets into a cohesive landing-page layout. '
            'You may use semantic placeholders or logical image paths in the first HTML/CSS draft only. '
            'Before final closure, all placeholders and guessed image/CSS links must be rebound to the '
            'concrete saved local artifacts using correct relative links from index.html/styles.css.\n\n'
            'Do not create extra HTML or CSS artifacts beyond index.html and styles.css. '
            'If linked assets are unresolved after generation, repair the existing files instead of creating '
            'duplicate page files.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['explicit_visual_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['explicit_audio_defer_materialization'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        image_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'image_generation'
        ]
        text_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('role') == 'text_artifact_output'
        ]

        self.assertEqual(len(image_branches), 3)
        self.assertEqual(
            [(branch['text_artifact_extension'], branch['text_artifact_source_name']) for branch in text_branches],
            [('html', 'index'), ('css', 'styles')],
        )
        self.assertEqual(
            [
                (
                    item.get('capability'),
                    item.get('output_type'),
                    item.get('text_artifact_extension'),
                    item.get('text_artifact_source_name'),
                )
                for item in graph['request_ir']['output_obligations']
            ],
            [
                ('chat', 'text', None, None),
                ('image_generation', 'image', None, None),
                ('image_generation', 'image', None, None),
                ('image_generation', 'image', None, None),
                ('chat', 'text', 'html', 'index'),
                ('chat', 'text', 'css', 'styles'),
            ],
        )

    def test_request_phase_graph_keeps_audio_executable_when_script_only_disclaims_fulfillment(self):
        prompt = (
            'Write a short narration script, then generate exactly one audio artifact from it. '
            'Treat the script as preparation only, not as the audio artifact.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['explicit_audio_defer_materialization'])
        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_audio_output'])
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches'] if branch.get('contract_state') == 'reserved'],
            [],
        )
        self.assertIn(
            ('text_to_speech', 'audio', 'runtime_artifact_or_branch_state'),
            [
                (
                    obligation['capability'],
                    obligation['output_type'],
                    obligation['fulfillment_policy'],
                )
                for obligation in graph['request_ir']['output_obligations']
            ],
        )

    def test_request_phase_graph_keeps_text_file_artifact_request_when_placeholder_only_disclaims_fulfillment(self):
        prompt = (
            'Create index.html as a local file artifact. '
            'Treat placeholder prose as preparation only, not as the HTML artifact.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertEqual(graph['downstream_capabilities'], ['chat'])
        text_artifacts = [
            obligation
            for obligation in graph['request_ir']['output_obligations']
            if obligation.get('fulfillment_policy') == 'runtime_text_artifact'
        ]
        self.assertEqual(len(text_artifacts), 1)
        self.assertEqual(text_artifacts[0]['text_artifact_extension'], 'html')
        self.assertEqual(text_artifacts[0]['text_artifact_source_name'], 'index')

    def test_request_phase_graph_honors_german_defer_image_wording_on_chat_route(self):
        graph = build_request_phase_graph(
            'Skizziere eine Idee für ein animiertes Logo für “Cinder Garden”, aber generiere '
            'noch kein Bild und keine Animation. Erstelle nur eine kurze Textbeschreibung und '
            'halte die Bildidee als mögliche spätere Option fest.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertEqual(graph['downstream_branches'][0]['contract_state'], 'reserved')
        self.assertEqual(graph['downstream_branches'][0]['capability'], 'image_generation')
        self.assertEqual(downstream_phase_records(graph), [])

    def test_request_phase_graph_visual_defer_does_not_block_audio_in_later_sentence(self):
        graph = build_request_phase_graph(
            'Skizziere drei Poster-Ideen für eine verlassene Mondbibliothek. '
            'Generiere aber noch kein Bild. Lies die beste Idee als kurzes Audio vor.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertTrue(graph['prompt_intent']['explicit_visual_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['explicit_audio_defer_materialization'])
        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertIn('text_to_speech', graph['downstream_capabilities'])
        promoted = [branch for branch in graph['downstream_branches'] if branch.get('contract_state') != 'reserved']
        reserved = [branch for branch in graph['downstream_branches'] if branch.get('contract_state') == 'reserved']
        self.assertEqual([branch['capability'] for branch in promoted], ['text_to_speech'])
        self.assertEqual(reserved[0]['capability'], 'image_generation')

    def test_request_phase_graph_keeps_reserved_option_image_out_of_executable_audio_chain(self):
        graph = build_request_phase_graph(
            'Erstelle zuerst eine kurze Produktidee. Wenn daraus ein Bild sinnvoll wäre, halte es '
            'nur als reservierte Option fest. Erzeuge stattdessen ein Audio-Pitching und '
            'transkribiere es danach.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech', 'speech_to_text'])
        self.assertEqual(
            [item['capability'] for item in graph['downstream_branches'] if item.get('contract_state') != 'reserved'],
            ['text_to_speech', 'speech_to_text'],
        )
        self.assertEqual(
            [item['depends_on'] for item in graph['downstream_branches'] if item.get('contract_state') != 'reserved'],
            [['phase-1'], ['phase-2']],
        )
        self.assertEqual(graph['output_candidates'][0]['status'], 'reserved')
        self.assertEqual(graph['output_candidates'][0]['capability'], 'image_generation')

    def test_request_phase_graph_promotes_only_selected_image_candidate_before_analysis(self):
        graph = build_request_phase_graph(
            'Plane drei mögliche Bildideen als Kandidaten, generiere aber nur die zweite. '
            'Merke die erste und dritte nur als Optionen. Analysiere danach nur das tatsächlich '
            'generierte Bild.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        promoted = [item for item in graph['downstream_branches'] if item.get('contract_state') != 'reserved']
        self.assertEqual(
            [item['capability'] for item in promoted],
            ['image_generation', 'vision_analysis'],
        )
        self.assertEqual(
            [item['depends_on'] for item in promoted],
            [['phase-1'], ['phase-2']],
        )
        self.assertEqual(promoted[0]['queue_index'], 2)
        self.assertEqual(promoted[0]['candidate_selection_index'], 2)
        self.assertEqual(graph['output_candidates'][0]['status'], 'reserved')

    def test_request_phase_graph_promotes_only_best_image_candidate(self):
        graph = build_request_phase_graph(
            'Erzeuge drei Bildideen für ein Festival “Fog Machines & Fireflies”, aber generiere '
            'nur die beste davon als Bild. Danach erkläre kurz, warum die anderen zwei nicht '
            'ausgeführt wurden.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        promoted = [item for item in graph['downstream_branches'] if item.get('contract_state') != 'reserved']
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertEqual(
            [item['capability'] for item in promoted],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            [item['depends_on'] for item in promoted],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )
        self.assertEqual(
            len([item for item in promoted if item.get('capability') == 'image_generation']),
            1,
        )

    def test_request_phase_graph_promotes_candidate_to_pending_obligation(self):
        graph = build_request_phase_graph(
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
                        'promotion_reason': 'user asked to generate the reserved direction',
                        'promotion_source': 'user_confirmation',
                    }
                ],
            },
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        request_ir = graph['request_ir']
        self.assertEqual([item['output_type'] for item in request_ir['output_obligations']], ['text', 'image'])
        image_obligation = request_ir['output_obligations'][1]
        self.assertEqual(image_obligation['status'], 'pending')
        self.assertEqual(image_obligation['promoted_from_candidate_id'], 'candidate-image-1')
        self.assertEqual(request_ir['output_candidates'][0]['status'], 'promoted')
        self.assertEqual(request_ir['output_candidates'][0]['promoted_obligation_id'], 'obligation-phase-2')
        self.assertEqual(request_ir['promotions'][0]['candidate_id'], 'candidate-image-1')
        self.assertEqual(request_ir['promotions'][0]['obligation_id'], 'obligation-phase-2')
        self.assertEqual(len(downstream_phase_records(graph)), 1)

    def test_request_phase_graph_marks_prompt_written_multi_image_request_as_text_first_image_chain(self):
        graph = build_request_phase_graph(
            'Create me three totally different images. You write the prompts.'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 3)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][2]['queue_index'], 3)

    def test_request_phase_graph_counts_parenthetical_image_count_after_noun(self):
        graph = build_request_phase_graph(
            'imageine just something. and then generate images (2) of it.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 2)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 2)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][1]['queue_index'], 2)

    def test_request_phase_graph_refines_visible_image_action_json_into_branch(self):
        graph = build_request_phase_graph(
            'ok. generate it now',
            request_payload={'ghost_route': True, 'prompt': 'ok. generate it now'},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            response_payload={
                'output_text': (
                    '{\n'
                    '  "action": "image_generation",\n'
                    '  "action_input": "A small russet fox in a glowing autumn forest."\n'
                    '}'
                )
            },
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(graph['downstream_branches'][0]['source'], 'assistant_output_claim_refinement')
        self.assertEqual(graph['graph_refinements'][0]['capability'], 'image_generation')

    def test_request_phase_graph_marks_open_choice_multi_image_request_as_text_first_image_chain(self):
        graph = build_request_phase_graph(
            'hallo. freut mich wieder. hey. nochmals zum testen. mache mir nun 4 verschiedene bilder. '
            'thema "lustiges realistische tierselfies". du bist frei in der wahl der tier(e). '
            'zeige sie in ihrem natürlichen habitat. verwende verschiedene bildformate (aspect ratios)'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 4)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][3]['queue_index'], 4)

    def test_request_phase_graph_counts_distributive_each_image_prompt_from_subject_count(self):
        graph = build_request_phase_graph(
            'hey ollmo. ich habe eine aufgabe fur dich. denk dir bitte funf total unterschiedliche orte aus '
            'und beschreibe sie ausfuhrlich. dann generiere mir bitte je ein bild davon.'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 5)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 5)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][4]['queue_index'], 5)

    def test_request_phase_graph_marks_imagined_situations_image_request_as_text_first_multi_image_chain(self):
        graph = build_request_phase_graph(
            'hallo. stell dir bitte drei absurde situationen vor und mache davon bilder.'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 3)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][2]['queue_index'], 3)

    def test_request_phase_graph_marks_poem_paragraph_image_request_as_text_first_multi_image_chain(self):
        graph = build_request_phase_graph(
            'hallo. schreib mir mal ein gedicht mir 3 absätzen. dann mache für jeden ein bild dazu.'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 3)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][2]['queue_index'], 3)

    def test_request_phase_graph_marks_own_theme_multi_image_request_as_text_first_image_chain(self):
        graph = build_request_phase_graph(
            'Kannst du mir bitte drei Bilder erstellen, nach deinem eigenen Thema bitte. '
            'Mach sie ganz unterschiedlich. stelle sicher alle in der antwort angezeit werden.'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 3)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][2]['queue_index'], 3)

    def test_request_phase_graph_marks_source_transfer_describe_then_images_request_as_text_first_chain(self):
        graph = build_request_phase_graph(
            'hey. kannst du mir 3 total verschiedene orte oder situationen beschreiben '
            'und aus diesen dann 3 bilder generieren. danke'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 3)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][2]['queue_index'], 3)

    def test_request_phase_graph_marks_imagine_place_then_single_image_request_as_text_first_chain(self):
        graph = build_request_phase_graph(
            'stell dir einen ort irgendwo im kosmos vor, beschreibe ihn detailliert '
            'und erstell mir dann bitte ein bild davon'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 1)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)

    def test_request_phase_graph_does_not_count_later_damage_list_as_image_count(self):
        graph = build_request_phase_graph(
            'Erzeuge ein Bild einer alpinen Rettungsstation nach einem Sturm. '
            'Danach beschreibe im Chat drei sichtbare Schäden im Bild.'
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(
            [branch['capability'] for branch in graph['downstream_branches']],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(
            len([branch for branch in graph['downstream_branches'] if branch['capability'] == 'image_generation']),
            1,
        )

    def test_request_phase_graph_keeps_meta_runtime_example_prompt_as_single_phase_chat(self):
        graph = build_request_phase_graph(
            'Erklär mir jetzt denselben Ablauf nur an einem ganz konkreten Beispiel: '
            '"Schreib ein Gedicht mit 3 Absätzen und generiere danach für jeden Absatz ein Bild." '
            'Nenne nur echte Begriffe aus der Runtime.'
        )

        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertFalse(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['mode'], 'single_phase')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertEqual(graph['downstream_branches'], [])

    def test_request_phase_graph_defaults_multi_image_route_to_prepare_first_phase_chain(self):
        graph = build_request_phase_graph(
            'Generate six different images of strange dream locations.',
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 6)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 6)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)
        self.assertEqual(graph['downstream_branches'][5]['queue_index'], 6)

    def test_request_phase_graph_keeps_explicit_batch_prompts_as_direct_image_batch_contract(self):
        graph = build_request_phase_graph(
            'Generate two different images of strange dream locations.',
            request_payload={
                'ghost_route': True,
                'batch_prompts': ['wide dream lake', 'tower above clouds'],
            },
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 2)
        self.assertEqual(graph['current_phase_capability'], 'image_generation')
        self.assertEqual(graph['current_phase_resolution'], 'router_required')
        self.assertEqual(graph['mode'], 'single_phase')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertEqual(graph['downstream_branches'], [])

    def test_request_phase_graph_defaults_single_image_ghost_request_to_prepare_first_phase_chain(self):
        graph = build_request_phase_graph(
            'Generate an image of a red fox in snow.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual(len(graph['downstream_branches']), 1)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)

    def test_next_executable_downstream_branches_refuses_dependency_fallback(self):
        graph = build_request_phase_graph(
            'Prepare evidence and then compare it.',
            request_payload={
                'ghost_route': True,
                'downstream_branches': [
                    {
                        'branch_id': 'branch-final-compare',
                        'phase_id': 'phase-4',
                        'capability': 'chat',
                        'output_type': 'text',
                        'depends_on': ['phase-2', 'phase-3'],
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            response_payload={'output_text': 'Initial preparation is done.'},
        )

        pending_final = [graph['downstream_branches'][0]]
        self.assertEqual(
            next_executable_downstream_branches(graph, pending_branches=pending_final),
            [],
        )
        self.assertEqual(
            next_executable_downstream_branches(
                graph,
                pending_branches=pending_final,
                completed_branches=[{'phase_id': 'phase-2'}],
                failed_branches=[{'phase_id': 'phase-3'}],
            ),
            [],
        )
        self.assertEqual(
            next_executable_downstream_branches(
                graph,
                pending_branches=pending_final,
                completed_branches=[{'phase_id': 'phase-2'}, {'phase_id': 'phase-3'}],
            )[0]['phase_id'],
            'phase-4',
        )

    def test_request_phase_graph_defaults_single_audio_ghost_request_to_prepare_first_phase_chain(self):
        graph = build_request_phase_graph(
            'Read this exact line aloud in a calm voice: The stars remember us.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'text_to_speech', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(len(graph['downstream_branches']), 1)
        self.assertEqual(graph['downstream_branches'][0]['queue_index'], 1)

    def test_request_phase_graph_keeps_pure_transcription_separate_from_tts(self):
        graph = build_request_phase_graph(
            'Transkribiere diese Audiodatei.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'speech_to_text', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_speech_to_text_output'])
        self.assertFalse(graph['prompt_intent']['requests_audio_output'])
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['speech_to_text'])
        self.assertEqual(graph['downstream_branches'][0]['output_type'], 'text')

    def test_request_phase_graph_does_not_materialize_from_missing_audio_upload_mention(self):
        graph = build_request_phase_graph(
            'Sorry, die Audiodatei habe ich vergessen anzuhängen. Danke trotzdem.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['requests_audio_output'])
        self.assertFalse(graph['prompt_intent']['requests_speech_to_text_output'])
        self.assertEqual(graph['mode'], 'single_phase')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertEqual(graph['downstream_branches'], [])

    def test_request_phase_graph_keeps_voiceover_script_as_text_until_audio_requested(self):
        graph = build_request_phase_graph(
            'Write a clean voiceover script, no audio yet.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['requests_audio_output'])
        self.assertFalse(graph['prompt_intent']['text_preparation_before_audio_output'])
        self.assertEqual(graph['mode'], 'single_phase')
        self.assertEqual(graph['downstream_capabilities'], [])

    def test_request_phase_graph_keeps_natural_language_deferred_image_as_reserved_candidate(self):
        graph = build_request_phase_graph(
            'Skizziere eine mögliche Bildidee, aber generiere noch kein Bild.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertFalse(graph['continuation_required'])
        self.assertEqual(graph['downstream_branches'][0]['contract_state'], 'reserved')
        self.assertEqual(graph['downstream_branches'][0]['required'], False)
        self.assertEqual(graph['request_ir']['output_candidates'][0]['status'], 'reserved')
        self.assertEqual(downstream_phase_records(graph), [])

    def test_request_phase_graph_does_not_let_reserved_image_candidate_mask_audio_work(self):
        graph = build_request_phase_graph(
            'Skizziere eine mögliche Poster-Idee, generiere aber noch kein Bild, und lies die Idee als Audio vor.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertTrue(graph['continuation_required'])
        self.assertEqual(
            [item['capability'] for item in downstream_phase_records(graph)],
            ['text_to_speech'],
        )
        self.assertEqual(
            [item['status'] for item in graph['request_ir']['output_candidates']],
            ['reserved'],
        )

    def test_request_phase_graph_keeps_negated_audio_generation_out_of_stt_translation(self):
        graph = build_request_phase_graph(
            'Transkribiere diese Audiodatei und übersetze die Transkription danach ins Deutsche. '
            'Erzeuge kein neues Audio.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_speech_to_text_output'])
        self.assertFalse(graph['prompt_intent']['requests_audio_output'])
        self.assertEqual(graph['downstream_capabilities'], ['speech_to_text', 'chat'])
        self.assertEqual(
            [(branch['capability'], branch['output_type']) for branch in graph['downstream_branches']],
            [('speech_to_text', 'text'), ('chat', 'text')],
        )

    def test_request_phase_graph_promotes_current_audio_attachment_follow_up_to_stt(self):
        graph = build_request_phase_graph(
            'ok. funktioniert es jetzt? ich habe die datei per attachement angefügt.',
            request_payload={
                'ghost_route': True,
                'input_artifacts': [
                    {
                        'artifact_ref': 'artifact:audio_test',
                        'type': 'audio',
                        'kind': 'audio',
                        'path': '/tmp/audio-light-test.wav',
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['input_audio_artifact_promoted_to_stt'])
        self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertEqual(graph['downstream_branches'], [])

    def test_request_phase_graph_allows_tts_and_stt_in_same_turn(self):
        graph = build_request_phase_graph(
            'Generier mir ein Audio von XY und transkribiere mir diese Datei, '
            'übersetze danach in Deutsch.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertTrue(graph['prompt_intent']['requests_speech_to_text_output'])
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech', 'speech_to_text', 'chat'])
        self.assertEqual(
            [(branch['capability'], branch['output_type']) for branch in graph['downstream_branches']],
            [('text_to_speech', 'audio'), ('speech_to_text', 'text'), ('chat', 'text')],
        )
        self.assertEqual(
            [branch['depends_on'] for branch in graph['downstream_branches']],
            [['phase-1'], ['phase-2'], ['phase-3']],
        )

    def test_request_phase_graph_promotes_input_audio_stt_before_dependent_tts(self):
        graph = build_request_phase_graph(
            'Transkribiere diese angehängte Audiodatei und generiere danach ein Audio daraus.',
            request_payload={
                'ghost_route': True,
                'input_artifacts': [
                    {
                        'artifact_ref': 'artifact:audio_test',
                        'type': 'audio',
                        'kind': 'audio',
                        'path': '/tmp/audio-light-test.wav',
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(graph['phases'][1]['depends_on'], ['phase-1'])
        self.assertEqual(graph['downstream_branches'][0]['content_payload_source'], 'speech_to_text_branch_result')

    def test_request_phase_graph_keeps_independent_tts_payload_separate_from_stt(self):
        graph = build_request_phase_graph(
            'Generiere ein Audio mit dem Satz "Hallo Welt" und transkribiere die angehängte Audiodatei.',
            request_payload={
                'ghost_route': True,
                'input_artifacts': [
                    {
                        'artifact_ref': 'artifact:audio_test',
                        'type': 'audio',
                        'kind': 'audio',
                        'path': '/tmp/audio-light-test.wav',
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
        self.assertEqual(graph['phases'][1]['depends_on'], ['phase-1'])
        self.assertEqual(graph['downstream_branches'][0]['content_payload'], 'Hallo Welt')
        self.assertEqual(
            graph['downstream_branches'][0]['content_payload_source'],
            'current_turn_direct_spoken_clause',
        )

    def test_request_phase_graph_promotes_mixed_materialized_artifact_list(self):
        graph = build_request_phase_graph(
            'Plane vier Artefakte: 1 Textzusammenfassung, 1 Bildidee, 1 TTS-Audio, '
            '1 JSON-Checkliste. Materialisiere nur, was die Runtime wirklich kann.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertEqual(graph['mode'], 'carried_phase_chain')
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech', 'image_generation'])
        self.assertEqual(
            [(branch['capability'], branch['queue_index']) for branch in graph['downstream_branches']],
            [('text_to_speech', 1), ('image_generation', 1)],
        )

    def test_request_phase_graph_keeps_plain_text_artifacts_single_phase(self):
        graph = build_request_phase_graph(
            'Erstelle fünf kurze Textartefakte als reine Antwortabschnitte.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['requests_audio_output'])
        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertEqual(graph['mode'], 'single_phase')
        self.assertEqual(graph['downstream_capabilities'], [])

    def test_request_phase_graph_does_not_carry_stale_visual_follow_up_for_plain_text_turn(self):
        graph = build_request_phase_graph(
            'Petra is a testament to human ingenuity, built with brilliant engineering and trade acumen.',
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['mode'], 'single_phase')
        self.assertEqual(graph['downstream_capabilities'], [])
        self.assertEqual(graph['downstream_branches'], [])

    def test_plans_compound_tts_prompt_and_adds_default_instruct(self):
        execute_chat = Mock(
            return_value='{"apply": true, "planned_prompt": "Honor guides my blade.", "reason": "imagined speech"}'
        )
        payload = {
            'prompt': 'imagine something he would say and then generate an audio clip of it.',
        }
        route_info = {
            'capability': 'text_to_speech',
            'instance': {
                'instance_id': 'tts-1',
                'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                'backend': 'mlx',
                'capability': 'text_to_speech',
                'port': 11504,
                'tts_model_type': 'voice_design',
                'session_controls': {
                    'enabled': True,
                    'fields': {
                        'tts_instruct': {
                            'visible': True,
                            'required': True,
                        }
                    },
                },
            },
        }
        instances = [
            {
                'instance_id': 'chat-1',
                'model': 'gemma4:26b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11437,
                'readiness': 'ready',
                'activity': 'idle',
            }
        ]

        updated, meta = plan_compound_execution(
            payload,
            route_info=route_info,
            instances=instances,
            context_messages=[
                {'role': 'user', 'content': 'create me an image of a samurai 2:3'},
                {'role': 'assistant', 'content': 'Image generated.'},
                {'role': 'user', 'content': payload['prompt']},
            ],
            execute_chat_request=execute_chat,
            semantic_role_profile={
                'mode': 'repair',
                'summary': 'Advisory semantic role orientation for this route.',
                'semantic_role_ids': ['repairer', 'evidence_reasoner'],
                'semantic_role_orientation': {
                    'mode': 'repair',
                    'mode_source': 'request',
                    'suggested_semantic_review_lenses': ['repairer', 'evidence_reasoner'],
                    'authority': 'advisory_read_model_only',
                },
            },
        )

        self.assertEqual(updated['prompt'], 'Honor guides my blade.')
        self.assertEqual(updated['instruct'], 'Use a natural, conversational voice.')
        self.assertTrue(meta['applied'])
        self.assertIn('prompt', meta['applied_fields'])
        self.assertIn('instruct', meta['applied_fields'])
        self.assertEqual(meta['planner_instance_id'], 'chat-1')
        self.assertEqual(meta['semantic_role_mode'], 'repair')
        self.assertEqual(execute_chat.call_args.kwargs['timeout_override_sec'], 90)
        planner_prompt = execute_chat.call_args.kwargs['messages'][0]['content']
        self.assertNotIn('Current Ghost mode: repair.', planner_prompt)
        self.assertNotIn('Planner style: conservative.', planner_prompt)
        self.assertIn('Semantic role profile "repair"', planner_prompt)
        self.assertIn('advisory orientation only', planner_prompt)
        self.assertIn('execution contracts, runtime evidence, and closure truth first', planner_prompt)
        execute_chat.assert_called_once()

    def test_plans_logged_story_plus_listen_prompt_when_routed_to_tts(self):
        execute_chat = Mock(
            return_value='{"apply": true, "planned_prompt": "A bell keeper tells a quiet story beside the harbor.", "reason": "compound story to spoken content"}'
        )

        updated, meta = plan_compound_execution(
            {
                'prompt': 'now.. write some little story ... and then give me something i can listen to. can you? your free in the topic.',
            },
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {
                                'visible': True,
                                'required': True,
                            }
                        },
                    },
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[
                {
                    'role': 'user',
                    'content': 'now.. write some little story ... and then give me something i can listen to. can you? your free in the topic.',
                },
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], 'A bell keeper tells a quiet story beside the harbor.')
        self.assertEqual(updated['instruct'], 'Use a natural, conversational voice.')
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'compound_tts_prompt')
        execute_chat.assert_called_once()

    def test_marks_story_plus_listen_prompt_as_text_first_tts_follow_up_when_current_route_is_chat(self):
        execute_chat = Mock()

        updated, meta = plan_compound_execution(
            {
                'prompt': 'now.. write some little story ... and then give me something i can listen to. can you? your free in the topic.',
            },
            route_info={
                'capability': 'chat',
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                },
            },
            instances=[],
            context_messages=[
                {
                    'role': 'user',
                    'content': 'now.. write some little story ... and then give me something i can listen to. can you? your free in the topic.',
                },
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['request_meta']['capability_hint'], 'text_to_speech')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'request_phase_graph_follow_up')
        self.assertEqual(meta['deferred_capability'], 'text_to_speech')
        self.assertEqual(meta['deferred_capabilities'], ['text_to_speech'])
        execute_chat.assert_not_called()

    def test_marks_translation_plus_narration_script_prompt_as_text_first_without_tts_hint(self):
        execute_chat = Mock()

        updated, meta = plan_compound_execution(
            {
                'prompt': 'Translate that quote into natural English and give me a clean narration script.',
            },
            route_info={
                'capability': 'chat',
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                },
            },
            instances=[],
            context_messages=[
                {
                    'role': 'user',
                    'content': 'Translate that quote into natural English and give me a clean narration script.',
                },
            ],
            execute_chat_request=execute_chat,
        )

        self.assertNotIn('request_meta', updated)
        self.assertFalse(meta['attempted'])
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'not_a_compound_request')
        self.assertNotIn('deferred_capability', meta)
        execute_chat.assert_not_called()

    def test_marks_story_read_aloud_prompt_as_text_first_tts_follow_up_when_current_route_is_chat(self):
        execute_chat = Mock()
        prompt = (
            'Write a short mystical story in 2 short paragraphs, then read that exact story aloud. '
            'If audio needs a speaker or voice style, choose the default automatically.'
        )

        updated, meta = plan_compound_execution(
            {'prompt': prompt},
            route_info={
                'capability': 'chat',
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                },
            },
            instances=[],
            context_messages=[{'role': 'user', 'content': prompt}],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['request_meta']['capability_hint'], 'text_to_speech')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'request_phase_graph_follow_up')
        self.assertEqual(meta['deferred_capability'], 'text_to_speech')
        self.assertEqual(meta['deferred_capabilities'], ['text_to_speech'])
        execute_chat.assert_not_called()

    def test_marks_describe_then_show_prompt_as_text_first_image_follow_up_when_current_route_is_chat(self):
        execute_chat = Mock()
        prompt = 'Describe a place you would love to visit in vivid detail, then show it to me as an image.'

        updated, meta = plan_compound_execution(
            {'prompt': prompt},
            route_info={
                'capability': 'chat',
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:e4b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                },
            },
            instances=[],
            context_messages=[{'role': 'user', 'content': prompt}],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['request_meta']['capability_hint'], 'image_generation')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'request_phase_graph_follow_up')
        self.assertEqual(meta['deferred_capability'], 'image_generation')
        self.assertEqual(meta['deferred_capabilities'], ['image_generation'])
        execute_chat.assert_not_called()

    def test_request_phase_graph_tracks_describe_then_show_image_of_it_phrase_family(self):
        prompt = 'Can you imagine the place you were "born", describe what it was like, and then show an image of it to me?'

        graph = build_request_phase_graph(prompt)

        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])

    def test_generated_tts_content_waits_for_phase_output_instead_of_literal_controls(self):
        prompts = (
            'Write a poem titled "At Sunrise", then speak it aloud.',
            'Write a poem, then speak it aloud. Style: concise',
            'Summarize the source, then read the summary aloud: '
            '```text\nSource material only.\n```',
            'Read this passage "The harbor was quiet.", condense it to one sentence, '
            'then speak the result aloud.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                execute_chat = Mock()
                updated, meta = plan_compound_execution(
                    {'ghost_route': True, 'prompt': prompt},
                    route_info={
                        'capability': 'chat',
                        'route_source': 'ghost_carried',
                        'instance': {
                            'instance_id': 'chat-1',
                            'model': 'gemma4:e4b',
                            'backend': 'ollama',
                            'capability': 'chat',
                            'port': 11437,
                        },
                    },
                    instances=[],
                    context_messages=[{'role': 'user', 'content': prompt}],
                    execute_chat_request=execute_chat,
                )

                self.assertNotIn('content_payload', updated)
                self.assertIsNone(meta['planned_prompt'])
                execute_chat.assert_not_called()

    def test_ghost_carried_direct_reply_read_aloud_defers_tts_and_extracts_content_payload(self):
        execute_chat = Mock()
        referenced_reply = (
            'The fog rolling into the harbor carried a low metallic knock from the deepest channel.'
        )
        prompt = 'Read that text aloud using voice "Vivian".'

        updated, meta = plan_compound_execution(
            {'prompt': prompt},
            route_info={
                'capability': 'chat',
                'route_source': 'ghost_carried',
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:e4b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                },
            },
            instances=[],
            context_messages=[
                {'role': 'assistant', 'content': referenced_reply},
                {'role': 'user', 'content': prompt},
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['request_meta']['capability_hint'], 'text_to_speech')
        self.assertEqual(updated['content_payload'], referenced_reply)
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'request_phase_graph_follow_up')
        self.assertEqual(meta['deferred_capability'], 'text_to_speech')
        execute_chat.assert_not_called()

    def test_ghost_carried_direct_image_prompt_defers_image_generation(self):
        execute_chat = Mock()
        prompt = 'Describe a place you would love to visit in vivid detail, then show it to me as an image.'

        updated, meta = plan_compound_execution(
            {'prompt': prompt},
            route_info={
                'capability': 'chat',
                'route_source': 'ghost_carried',
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:e4b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                },
            },
            instances=[],
            context_messages=[{'role': 'user', 'content': prompt}],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['request_meta']['capability_hint'], 'image_generation')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'request_phase_graph_follow_up')
        self.assertEqual(meta['deferred_capability'], 'image_generation')
        self.assertEqual(meta['deferred_capabilities'], ['image_generation'])
        execute_chat.assert_not_called()

    def test_marks_screenshot_translate_read_aloud_prompt_as_text_first_tts_follow_up_when_current_route_is_vision(self):
        execute_chat = Mock()
        prompt = (
            'From this screenshot, extract the exact quoted text, translate it into natural English, '
            'then read your English version aloud. If the active TTS model needs a speaker or style, '
            'choose a sensible default automatically from what Ollmo already knows.'
        )

        updated, meta = plan_compound_execution(
            {'prompt': prompt},
            route_info={
                'capability': 'vision_analysis',
                'instance': {
                    'instance_id': 'vision-1',
                    'model': 'gemma4:e4b',
                    'backend': 'ollama',
                    'capability': 'vision_analysis',
                    'port': 11438,
                },
            },
            instances=[],
            context_messages=[{'role': 'user', 'content': prompt}],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['request_meta']['capability_hint'], 'text_to_speech')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'text_first_tts_follow_up')
        self.assertEqual(meta['deferred_capability'], 'text_to_speech')
        execute_chat.assert_not_called()

    def test_plans_compound_image_prompt(self):
        execute_chat = Mock(
            return_value='{"apply": true, "planned_prompt": "A cinematic poster of a squirrel superhero on a tree, bold typography, dramatic lighting.", "reason": "rewrote indirect image request"}'
        )
        payload = {
            'prompt': 'write a stronger prompt for this and then generate a poster of it',
        }
        route_info = {
            'capability': 'image_generation',
            'instance': {
                'instance_id': 'img-1',
                'model': 'x/flux2-klein:latest',
                'backend': 'ollama',
                'capability': 'image_generation',
                'port': 11436,
            },
        }
        instances = [
            {
                'instance_id': 'chat-1',
                'model': 'gemma4:26b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11437,
                'readiness': 'ready',
                'activity': 'idle',
            }
        ]

        updated, meta = plan_compound_execution(
            payload,
            route_info=route_info,
            instances=instances,
            context_messages=[
                {'role': 'user', 'content': 'paint me "a squirrel on a tree"'},
                {'role': 'assistant', 'content': 'Image generated.'},
                {'role': 'user', 'content': payload['prompt']},
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(
            updated['prompt'],
            'A cinematic poster of a squirrel superhero on a tree, bold typography, dramatic lighting.',
        )
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'compound_image_prompt')

    def test_plans_tts_detail_refinement_for_explicit_spoken_content(self):
        execute_chat = Mock(
            return_value='{"apply": true, "planned_prompt": null, "instruct": "Speak in English with natural English pronunciation. Use a calm, clearly male voice.", "lang_code": "en", "response_format": "mp3", "reason": "filled prompt-specified tts controls"}'
        )

        updated, meta = plan_compound_execution(
            {
                'prompt': 'please read this aloud in sound format: "Hello, what\'s going on today?" set the style language to english, male voice as mp3',
            },
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {
                                'visible': True,
                                'required': True,
                            }
                        },
                    },
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[
                {'role': 'user', 'content': 'please read this aloud in sound format: "Hello, what\'s going on today?" set the style language to english, male voice as mp3'},
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], 'please read this aloud in sound format: "Hello, what\'s going on today?" set the style language to english, male voice as mp3')
        self.assertEqual(updated['lang_code'], 'en')
        self.assertEqual(updated['response_format'], 'mp3')
        self.assertEqual(
            updated['instruct'],
            'Speak in English with natural English pronunciation. Use a calm, clearly male voice.',
        )
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'tts_detail_refinement')
        execute_chat.assert_called_once()

    def test_skips_planning_for_direct_prompt(self):
        execute_chat = Mock()

        updated, meta = plan_compound_execution(
            {'prompt': 'generate an image of a butterfly'},
            route_info={
                'capability': 'image_generation',
                'instance': {
                    'instance_id': 'img-1',
                    'model': 'x/flux2-klein:latest',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'port': 11436,
                },
            },
            instances=[],
            context_messages=[],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], 'generate an image of a butterfly')
        self.assertFalse(meta['attempted'])
        execute_chat.assert_not_called()

    def test_plans_contextual_image_edit_from_recent_image_state(self):
        execute_chat = Mock(
            return_value='{"apply": true, "planned_prompt": "A humanoid robot with glowing yellow eyes standing centered within a rocky archway. Keep the same robot, rocky archway, and prehistoric scene. Change only the robot armor to gold and the eyes to black.", "reason": "contextual image edit rewrite"}'
        )

        updated, meta = plan_compound_execution(
            {'prompt': 'give the robot a golden armor and black eyes'},
            route_info={
                'capability': 'image_generation',
                'instance': {
                    'instance_id': 'img-1',
                    'model': 'x/flux2-klein:latest',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'port': 11436,
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[
                {
                    'role': 'assistant',
                    'content': 'Image generated.',
                    'artifacts': [
                        {
                            'type': 'image',
                            'path': '/tmp/generated.png',
                            'image_state': {
                                'summary': 'A metallic humanoid robot standing centered within a rocky archway.',
                                'subject': 'A humanoid robot with glowing yellow eyes.',
                                'scene': 'A prehistoric rocky archway scene with mist and jungle foliage.',
                                'style': 'Vintage illustration with muted colors.',
                            },
                        }
                    ],
                },
                {'role': 'user', 'content': 'give the robot a golden armor and black eyes'},
            ],
            execute_chat_request=execute_chat,
        )

        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'contextual_image_edit')
        self.assertIn('Keep the same robot, rocky archway, and prehistoric scene', updated['prompt'])

    def test_skips_contextual_image_edit_planning_for_brand_new_image_request(self):
        execute_chat = Mock()

        updated, meta = plan_compound_execution(
            {'prompt': 'create a brand new image of a spaceship over the ocean'},
            route_info={
                'capability': 'image_generation',
                'instance': {
                    'instance_id': 'img-1',
                    'model': 'x/flux2-klein:latest',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'port': 11436,
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[
                {
                    'role': 'assistant',
                    'content': 'Image generated.',
                    'artifacts': [
                        {
                            'type': 'image',
                            'path': '/tmp/generated.png',
                            'image_state': {
                                'summary': 'A metallic humanoid robot standing centered within a rocky archway.',
                                'subject': 'A humanoid robot with glowing yellow eyes.',
                            },
                        }
                    ],
                },
                {'role': 'user', 'content': 'create a brand new image of a spaceship over the ocean'},
            ],
            execute_chat_request=execute_chat,
        )

        self.assertFalse(meta['attempted'])
        self.assertEqual(updated['prompt'], 'create a brand new image of a spaceship over the ocean')
        execute_chat.assert_not_called()

    def test_locally_extracts_pinned_reply_for_tts_reference_read(self):
        execute_chat = Mock()

        updated, meta = plan_compound_execution(
            {'prompt': 'read the summary of the last reply aloud in a calm male voice as mp3'},
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                    'tts_speakers': ['male', 'female'],
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {
                                'visible': True,
                                'required': True,
                            }
                        },
                    },
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[
                {
                    'role': 'assistant',
                    'content': 'Long earlier answer that should not be read.',
                },
	                {
	                    'role': 'system',
	                    'content': (
	                        'Selected prior message reference for this conversation turn. '
	                        'Treat it as bounded reference context only; the current user message remains the live instruction. '
	                        'Do not infer new tasks from this reference unless the current turn explicitly asks.\n\n'
	                        '[assistant]\nQuantum entanglement means two particles share linked states across distance.'
	                    ),
	                },
                {
                    'role': 'user',
                    'content': 'read the summary of the last reply aloud in a calm male voice as mp3',
                },
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(
            updated['prompt'],
            'Quantum entanglement means two particles share linked states across distance.',
        )
        self.assertEqual(updated['_prompt_hint'], 'read the summary of the last reply aloud in a calm male voice as mp3')
        self.assertEqual(updated['voice'], 'male')
        self.assertEqual(updated['response_format'], 'mp3')
        self.assertEqual(
            updated['instruct'],
            'Use a calm, clearly male voice.',
        )
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'tts_reference_read')
        execute_chat.assert_not_called()

    def test_locally_extracts_stage_direction_free_story_for_tts_reference_read(self):
        execute_chat = Mock()
        story_text = (
            'Elara lived at the edge of Whisperwood, where the moss glowed faintly at twilight.\n\n'
            'When she touched the polished stone, the air grew fragrant with rain and cinnamon.'
        )

        updated, meta = plan_compound_execution(
            {'prompt': 'read that reply aloud in a calm voice'},
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {
                                'visible': True,
                                'required': True,
                            }
                        },
                    },
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[
                {
                    'role': 'assistant',
                    'content': f'{story_text}\n\n***\n\n**(Reading the story aloud)**',
                },
                {
                    'role': 'user',
                    'content': 'read that reply aloud in a calm voice',
                },
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], story_text)
        self.assertEqual(updated['_prompt_hint'], 'read that reply aloud in a calm voice')
        self.assertEqual(updated['instruct'], 'Use a calm voice.')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'tts_reference_read')
        execute_chat.assert_not_called()

    def test_locally_extracts_story_without_trailing_self_note_for_tts_reference_read(self):
        execute_chat = Mock()
        story_text = (
            'The Weaver of Whispers sat at the loom of twilight, her threads spun from forgotten dreams.\n\n'
            'A single luminous thread drifted down through the veil and awakened the sleeping heart of the world.'
        )

        updated, meta = plan_compound_execution(
            {'prompt': 'read that reply aloud in a calm voice'},
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {
                                'visible': True,
                                'required': True,
                            }
                        },
                    },
                },
            },
            instances=[],
            context_messages=[
                {
                    'role': 'assistant',
                    'content': (
                        f'{story_text}\n\n***\n\n'
                        '*(Self-Correction/Note: As an AI text model, I cannot "read aloud" in the traditional sense. '
                        'I will present the text as if it were being read.)*'
                    ),
                },
                {
                    'role': 'user',
                    'content': 'read that reply aloud in a calm voice',
                },
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], story_text)
        self.assertEqual(updated['_prompt_hint'], 'read that reply aloud in a calm voice')
        self.assertEqual(updated['instruct'], 'Use a calm voice.')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'tts_reference_read')
        execute_chat.assert_not_called()

    def test_extract_latest_assistant_content_skips_short_handoff_ack_when_story_exists(self):
        story_text = (
            'The Last Firefly drifted over the moss with a trembling lantern glow.\n\n'
            'When Lily opened the jar, the forest answered with a thousand tiny stars.'
        )

        extracted = _extract_latest_assistant_content(
            [
                {'role': 'assistant', 'content': story_text},
                {'role': 'assistant', 'content': 'I can certainly prepare that story for you to hear.'},
                {'role': 'user', 'content': 'ok. please do it.'},
            ]
        )

        self.assertEqual(extracted, story_text)

    def test_locally_extracts_story_for_tts_reference_even_after_meta_acknowledgment(self):
        execute_chat = Mock()
        story_text = (
            'At the edge of Ember Bay, a brass lighthouse sang softly to the tide.\n\n'
            'Each note woke a lanternfish beneath the waves until the harbor glittered like a moving constellation.'
        )

        updated, meta = plan_compound_execution(
            {'prompt': 'would it be possible for you to read this for me in a male voice?'},
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                    'tts_speakers': ['male', 'female'],
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {
                                'visible': True,
                                'required': True,
                            }
                        },
                    },
                },
            },
            instances=[],
            context_messages=[
                {'role': 'assistant', 'content': story_text},
                {'role': 'assistant', 'content': 'I can certainly prepare that story for you to hear.'},
                {'role': 'user', 'content': 'would it be possible for you to read this for me in a male voice?'},
            ],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], story_text)
        self.assertEqual(updated['voice'], 'male')
        self.assertEqual(updated['instruct'], 'Use a clearly male voice.')
        self.assertTrue(meta['attempted'])
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['trigger'], 'tts_reference_read')
        execute_chat.assert_not_called()

    def test_falls_back_when_planner_output_is_not_json(self):
        execute_chat = Mock(return_value='not json at all')

        updated, meta = plan_compound_execution(
            {'prompt': 'imagine something he would say and then generate an audio clip of it.'},
            route_info={
                'capability': 'text_to_speech',
                'instance': {
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11504,
                    'tts_model_type': 'voice_design',
                },
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            ],
            context_messages=[],
            execute_chat_request=execute_chat,
        )

        self.assertEqual(updated['prompt'], 'imagine something he would say and then generate an audio clip of it.')
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'planner_output_not_json')


if __name__ == '__main__':
    unittest.main()
