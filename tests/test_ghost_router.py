import unittest
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import ollmo_g.router as ghost_router

from ollmo_g.intent import analyze_prompt_intent
from ollmo_g.memory import (
    build_ghost_memory,
    build_recent_learnings_from_events,
    build_recent_self_observations,
    build_self_healing_hints,
)
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_g.router import (
    build_embedding_route_audit,
    build_embedding_hints_from_vectors,
    build_embedding_route_candidates,
    build_route_hint,
    build_route_memory_scope,
    build_route_context,
    build_router_messages,
    infer_route_session_class,
    is_obvious_route_hint_fast_path,
    maybe_apply_embedding_route_bias,
    parse_router_output,
    select_embedding_instance,
    select_router_instance,
    validate_route_decision,
)
from ollmo_server.ghost_route_runtime import GhostRouteRuntimeOwner
from ollmo_services.chat_history import write_chat_history


class GhostRouterTests(unittest.TestCase):
    def test_build_router_messages_includes_runtime_policy_file(self):
        context = {'prompt': 'Generate audio of this sentence.'}
        with TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / 'GHOST.md'
            policy_path.write_text('Ghost runtime policy from file.', encoding='utf-8')
            ghost_router._GHOST_POLICY_CACHE['mtime_ns'] = None
            ghost_router._GHOST_POLICY_CACHE['text'] = None
            with patch.object(ghost_router, 'GHOST_POLICY_PATH', policy_path):
                messages = build_router_messages(context)

        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('Ghost runtime policy from file.', messages[0]['content'])

    def test_build_router_messages_does_not_truncate_runtime_policy_file(self):
        context = {'prompt': 'Generate audio of this sentence.'}
        tail_marker = 'TAIL_POLICY_MARKER_NO_HARD_CAP'
        with TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / 'GHOST.md'
            policy_path.write_text(
                'Ghost runtime policy start.\n'
                + ('policy body line\n' * 500)
                + tail_marker,
                encoding='utf-8',
            )
            ghost_router._GHOST_POLICY_CACHE['mtime_ns'] = None
            ghost_router._GHOST_POLICY_CACHE['text'] = None
            with patch.object(ghost_router, 'GHOST_POLICY_PATH', policy_path):
                messages = build_router_messages(context)

        self.assertIn('Ghost runtime policy start.', messages[0]['content'])
        self.assertIn(tail_marker, messages[0]['content'])
        self.assertNotIn('...[truncated]', messages[0]['content'])

    def test_build_router_messages_falls_back_when_policy_file_missing(self):
        context = {'prompt': 'Hello'}
        ghost_router._GHOST_POLICY_CACHE['mtime_ns'] = None
        ghost_router._GHOST_POLICY_CACHE['text'] = None
        with patch.object(ghost_router, 'GHOST_POLICY_PATH', Path('/nonexistent/ghost-policy.md')):
            messages = build_router_messages(context)

        self.assertIn('Trust the provided runtime manifest', messages[0]['content'])

    def test_build_router_messages_allows_bounded_workload_task_proposals(self):
        context = {
            'prompt': 'Create a slogan, generate audio, then write a final sentence.',
            'runtime': {
                'request_phase_graph': {
                    'workload_graph': {
                        'task_ids': ['task-phase-1', 'task-phase-2', 'task-phase-3'],
                    },
                },
            },
        }

        messages = build_router_messages(context)

        self.assertIn('workload_task_proposals', messages[0]['content'])
        self.assertIn('multiple tasks or dependency edges', messages[0]['content'])
        self.assertIn('semantic intent, input_refs', messages[0]['content'])
        self.assertIn('Use only existing task IDs or phase IDs', messages[0]['content'])
        self.assertIn('"workload_task_proposals":[{"task_id":"existing-task-id"', messages[1]['content'])
        self.assertIn('"evidence_requirements":[]', messages[1]['content'])
        self.assertIn('"repair_candidates":[]', messages[1]['content'])
        self.assertNotIn('Do not plan tasks', messages[0]['content'])

    def test_recent_user_capability_context_prefers_selected_reference_artifact(self):
        capability = ghost_router._recent_user_capability_context(
            {
                'selected_reference_artifact': {
                    'type': 'image',
                    'path': '/tmp/harbor.png',
                },
                'reference_artifacts': [
                    {
                        'type': 'image',
                        'path': '/tmp/harbor.png',
                    }
                ],
                'latest_artifacts': {
                    'image': {
                        'type': 'image',
                        'path': '/tmp/harbor.png',
                    }
                },
                'recent_messages': [],
            },
            current_prompt='make this image warmer',
        )

        self.assertEqual(capability, 'image_generation')

    def test_recent_user_capability_context_no_longer_guesses_from_prior_user_prompt_alone(self):
        capability = ghost_router._recent_user_capability_context(
            {
                'recent_messages': [
                    {'role': 'user', 'content': 'Generate an image of a floating lantern city.'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                    {'role': 'user', 'content': 'read that aloud'},
                ],
            },
            current_prompt='read that aloud',
        )

        self.assertIsNone(capability)

    def test_select_router_instance_prefers_structured_outputs_when_available(self):
        instances = [
            {
                'instance_id': 'coder-1',
                'model': 'qwen3-coder:latest',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11439,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': True},
            },
            {
                'instance_id': 'chat-1',
                'model': 'mlx-community/Apertus-8B-Instruct-2509-bf16',
                'backend': 'mlx',
                'capability': 'chat',
                'port': 11501,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': False},
            },
            {
                'instance_id': 'reasoning-1',
                'model': 'mlx-community/Qwen3.5-27B-4bit',
                'backend': 'mlx',
                'capability': 'chat',
                'port': 11508,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': False},
            },
        ]

        selected = select_router_instance(instances)

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'coder-1')

    def test_select_router_instance_uses_mlx_tie_break_after_runtime_signals(self):
        instances = [
            {
                'instance_id': 'chat-1',
                'model': 'gpt-oss:20b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11437,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': False},
            },
            {
                'instance_id': 'chat-mlx-1',
                'model': 'mlx-community/Qwen3.5-9B-MLX-4bit',
                'backend': 'mlx',
                'capability': 'chat',
                'port': 11508,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': False},
            },
        ]

        selected = select_router_instance(instances)

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'chat-mlx-1')

    def test_select_router_instance_prefers_smaller_same_family_candidate_when_fit_is_similar(self):
        instances = [
            {
                'instance_id': 'gemma-large-1',
                'model': 'gemma4:26b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11437,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': False, 'function_calling': True, 'tool_calling': True},
                'backend_metadata': {
                    'details': {
                        'family': 'gemma4',
                        'parameter_size': '25.8B',
                        'quantization_level': 'Q4_K_M',
                    },
                    'context_length': 262144,
                },
            },
            {
                'instance_id': 'gemma-small-1',
                'model': 'ggml-org/gemma-4-E4B-it-GGUF',
                'backend': 'llama_cpp',
                'capability': 'chat',
                'port': 11552,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                'features': {'structured_outputs': False, 'function_calling': False, 'tool_calling': False},
                'context_length': 32768,
                'size': 5 * 1024 * 1024 * 1024,
            },
        ]

        selected = select_router_instance(instances)

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'gemma-small-1')

    def test_select_embedding_instance_prefers_ready_ollama_embedding(self):
        instances = [
            {
                'instance_id': 'embed-mlx-1',
                'model': 'mlx-embed',
                'backend': 'mlx',
                'capability': 'chat',
                'provider_capabilities': ['chat', 'embedding'],
                'outputs': ['text', 'embedding'],
                'routing_summary': {
                    'backend_metadata': {
                        'native_endpoint_paths': ['/v1/embeddings'],
                    }
                },
                'port': 11501,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
            },
            {
                'instance_id': 'embed-ollama-1',
                'model': 'embeddinggemma:latest',
                'backend': 'ollama',
                'capability': 'embedding',
                'port': 11435,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
            },
        ]

        selected = select_embedding_instance(instances)

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'embed-ollama-1')

    def test_select_embedding_instance_accepts_mixed_capability_helper(self):
        instances = [
            {
                'instance_id': 'chat-embed-1',
                'model': 'mixed-helper',
                'backend': 'ollama',
                'capability': 'chat',
                'provider_capabilities': ['chat', 'embedding'],
                'outputs': ['text', 'embedding'],
                'port': 11435,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
            }
        ]

        selected = select_embedding_instance(instances)

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'chat-embed-1')
        self.assertEqual(selected['embedding_transport'], 'ollama_api_embed')

    def test_select_embedding_instance_accepts_openai_embeddings_contract(self):
        instances = [
            {
                'instance_id': 'embed-openai-1',
                'model': 'future-helper',
                'backend': 'mlx',
                'capability': 'chat',
                'provider_capabilities': ['chat', 'embedding'],
                'outputs': ['text', 'embedding'],
                'backend_package': 'mlx_lm',
                'backend_contract': 'openai.compat',
                'routing_summary': {
                    'backend_metadata': {
                        'native_endpoint_paths': ['/v1/embeddings'],
                    }
                },
                'port': 11512,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
            }
        ]

        selected = select_embedding_instance(instances)

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'embed-openai-1')
        self.assertEqual(selected['embedding_transport'], 'openai_embeddings')

    def test_select_embedding_instance_honors_stable_preferred_target(self):
        instances = [
            {
                'instance_id': 'embed-ollama-1',
                'model': 'embeddinggemma:latest',
                'backend': 'ollama',
                'capability': 'embedding',
                'port': 11435,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
            },
            {
                'instance_id': 'embed-mlx-1',
                'model': 'mlx-embed',
                'backend': 'mlx',
                'capability': 'chat',
                'provider_capabilities': ['chat', 'embedding'],
                'outputs': ['text', 'embedding'],
                'routing_summary': {
                    'backend_metadata': {
                        'native_endpoint_paths': ['/v1/embeddings'],
                    }
                },
                'port': 11501,
                'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
            },
        ]

        selected = select_embedding_instance(
            instances,
            preferred_target={'model': 'mlx-embed', 'backend': 'mlx', 'capability': 'embedding'},
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected['instance_id'], 'embed-mlx-1')

    def test_parse_router_output_accepts_fenced_json(self):
        payload, error = parse_router_output(
            '```json\n{"capability":"vision_analysis","instance_id":null,"reuse_last_artifact":true,'
            '"artifact_path":"/tmp/example.png","confidence":0.91,"reason":"latest image reference"}\n```'
        )

        self.assertIsNone(error)
        self.assertEqual(payload['capability'], 'vision_analysis')
        self.assertTrue(payload['reuse_last_artifact'])

    def test_validate_route_decision_rejects_image_artifact_for_chat(self):
        route, error = validate_route_decision(
            {
                'capability': 'chat',
                'instance_id': None,
                'reuse_last_artifact': True,
                'artifact_path': '/tmp/generated.png',
                'confidence': 0.9,
                'reason': 'bad route',
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gpt-oss:20b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                }
            ],
            recent_artifacts=[{'type': 'image', 'path': '/tmp/generated.png'}],
        )

        self.assertIsNone(route)
        self.assertIn('Image artifacts', error)

    def test_validate_route_decision_derives_capability_from_instance_id(self):
        route, error = validate_route_decision(
            {
                'instance_id': 'vision-1',
                'reuse_last_artifact': False,
                'artifact_path': None,
                'confidence': 0.82,
                'reason': 'picked the OCR instance directly',
            },
            instances=[
                {
                    'instance_id': 'vision-1',
                    'model': 'deepseek-ocr:latest',
                    'backend': 'ollama',
                    'capability': 'vision_analysis',
                    'port': 11435,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                }
            ],
            recent_artifacts=[],
        )

        self.assertIsNone(error)
        self.assertEqual(route['capability'], 'vision_analysis')

    def test_validate_route_decision_preserves_workload_task_proposals(self):
        route, error = validate_route_decision(
            {
                'capability': 'chat',
                'instance_id': None,
                'reuse_last_artifact': False,
                'artifact_path': None,
                'confidence': 0.82,
                'reason': 'route with bounded task annotations',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'proposal-final',
                        'phase_id': 'phase-3',
                        'capability': 'chat',
                        'semantic_intent': 'Write a final sentence from declared dependency evidence.',
                        'depends_on': ['phase-2'],
                        'input_refs': [{'kind': 'phase_output', 'phase_id': 'phase-2', 'role': 'dependency'}],
                        'review_criteria': ['uses dependency evidence'],
                        'evidence_requirements': ['phase-2 output exists'],
                        'semantic_review_criteria': ['final text uses the dependency output'],
                        'repair_candidates': [
                            {'task_id': 'task-phase-3', 'repair_action': 'repair_dependency_chain'}
                        ],
                        'unknown_field': 'drop me',
                    }
                ],
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gpt-oss:20b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                }
            ],
            recent_artifacts=[],
        )

        self.assertIsNone(error)
        self.assertEqual(route['workload_task_proposals'][0]['proposal_id'], 'proposal-final')
        self.assertEqual(route['workload_task_proposals'][0]['depends_on'], ['phase-2'])
        self.assertEqual(route['workload_task_proposals'][0]['input_refs'][0]['phase_id'], 'phase-2')
        self.assertEqual(route['workload_task_proposals'][0]['evidence_requirements'], ['phase-2 output exists'])
        self.assertEqual(route['workload_task_proposals'][0]['repair_candidates'][0]['repair_action'], 'repair_dependency_chain')
        self.assertNotIn('unknown_field', route['workload_task_proposals'][0])

    def test_validate_route_decision_accepts_vision_route_for_chat_primary_multimodal_instance(self):
        route, error = validate_route_decision(
            {
                'capability': 'vision_analysis',
                'instance_id': 'qwen-coding-1',
                'reuse_last_artifact': False,
                'artifact_path': None,
                'confidence': 0.88,
                'reason': 'vision-capable multimodal coding model',
            },
            instances=[
                {
                    'instance_id': 'qwen-coding-1',
                    'model': 'qwen3.5:35b-a3b-coding-nvfp4',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'provider_capabilities': ['chat', 'vision_analysis'],
                    'inputs': ['text', 'image'],
                    'outputs': ['text'],
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                }
            ],
            recent_artifacts=[],
        )

        self.assertIsNone(error)
        self.assertEqual(route['capability'], 'vision_analysis')
        self.assertEqual(route['instance_id'], 'qwen-coding-1')

    def test_parse_router_output_accepts_nested_route_object(self):
        payload, error = parse_router_output(
            '{"route":{"selected_capability":"image_generation","target_instance_id":"flux-1",'
            '"reuse_last_artifact":false,"confidence":0.77,"reason":"image request"}}'
        )

        self.assertIsNone(error)
        self.assertEqual(payload['capability'], 'image_generation')
        self.assertEqual(payload['instance_id'], 'flux-1')

    def test_build_route_hint_reuses_latest_image_for_follow_up(self):
        route = build_route_hint(
            {
                'prompt': 'describe this image',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'vision_analysis')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-image.png')

    def test_build_route_hint_detects_typoed_image_prompt(self):
        route = build_route_hint(
            {
                'prompt': 'generate me a igmae of "a butterfly"',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')

    def test_build_route_hint_detects_visual_poster_prompt_without_literal_image_word(self):
        route = build_route_hint(
            {
                'prompt': 'draw a butterfly poster in an art nouveau style',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')

    def test_build_route_hint_keeps_summary_request_as_chat(self):
        route = build_route_hint(
            {
                'prompt': 'generate a summary of this transcript',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_detects_implicit_tts_accessibility_prompt(self):
        route = build_route_hint(
            {
                'prompt': 'for my blind friend, output a file he can consume with a female voice saying hello world',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
                'memory': {},
            }
        )

        self.assertEqual(route['capability'], 'text_to_speech')
        self.assertEqual(route['reason'], 'text-to-speech cue')

    def test_build_route_hint_keeps_story_plus_listen_prompt_in_chat_until_text_is_ready(self):
        route = build_route_hint(
            {
                'prompt': 'now.. write some little story ... and then give me something i can listen to. can you? your free in the topic.',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
                'memory': {},
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'text preparation required before audio output')

    def test_answer_as_audio_delivery_uses_one_shared_prepare_first_intent(self):
        prompts = (
            'Erzähl mir etwas über doch. gib mir die antwort als generiertes audio.',
            'Erzähl mir etwas über dich. gib mir die antwort als generiertes audio.',
            'Liefere mir das Ergebnis als Audiodatei.',
            'Return the response as generated audio.',
            'Send me the reply as a voice clip.',
            'No, give me the answer as generated audio.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                intent = analyze_prompt_intent(prompt)
                route = build_route_hint(
                    {
                        'prompt': prompt,
                        'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                        'latest_artifacts': {},
                        'memory': {},
                    }
                )

                self.assertTrue(intent['requests_audio_output'])
                self.assertTrue(intent['direct_audio_materialization_request'])
                self.assertTrue(intent['has_audio_follow_up_request'])
                self.assertTrue(intent['text_preparation_before_audio_output'])
                self.assertIn(
                    'answer_as_audio_delivery_request',
                    intent['capability_cues']['text_to_speech'],
                )
                self.assertEqual(route['capability'], 'chat')
                self.assertEqual(route['reason'], 'text preparation required before audio output')

    def test_answer_as_audio_delivery_honors_negation_and_defer(self):
        cases = (
            ('Gib mir die Antwort als Text, nicht als generiertes Audio.', True, False),
            ('Gib mir die Antwort noch nicht als Audio.', False, True),
            ('Gib mir die Antwort später als Audio, jetzt nur als Text.', False, True),
            ('Give me the response as text, not as generated audio.', True, False),
            ('Do not give me the answer as generated audio.', True, False),
            ('Return the answer later as audio; text only for now.', False, True),
        )

        for prompt, negated, deferred in cases:
            with self.subTest(prompt=prompt):
                intent = analyze_prompt_intent(prompt)

                self.assertFalse(intent['requests_audio_output'])
                self.assertFalse(intent['direct_audio_materialization_request'])
                self.assertFalse(intent['has_audio_follow_up_request'])
                self.assertFalse(intent['text_preparation_before_audio_output'])
                self.assertEqual(intent['negated_audio_output_request'], negated)
                self.assertEqual(intent['explicit_audio_defer_materialization'], deferred or negated)

    def test_answer_as_audio_delivery_ignores_quoted_and_topical_phrases(self):
        prompts = (
            'Erkläre den Ausdruck „Gib mir die Antwort als generiertes Audio“.',
            'Explain the phrase "Give me the answer as generated audio".',
            'Vergleiche Antworttexte und generierte Audioformate auf begrifflicher Ebene.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                intent = analyze_prompt_intent(prompt)

                self.assertFalse(intent['requests_audio_output'])
                self.assertFalse(intent['direct_audio_materialization_request'])
                self.assertFalse(intent['text_preparation_before_audio_output'])
                self.assertNotIn(
                    'answer_as_audio_delivery_request',
                    intent['capability_cues']['text_to_speech'],
                )

    def test_build_route_hint_keeps_translation_plus_narration_script_prompt_in_chat(self):
        route = build_route_hint(
            {
                'prompt': 'Translate that quote into natural English and give me a clean narration script.',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
                'memory': {},
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_does_not_treat_male_voice_tts_prompt_as_image(self):
        route = build_route_hint(
            {
                'prompt': 'Read this aloud as mp3 in a calm male voice: "The train leaves at seven."',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
                'memory': {},
            }
        )

        self.assertEqual(route['capability'], 'text_to_speech')
        self.assertEqual(route['reason'], 'text-to-speech cue')

    def test_build_route_hint_detects_indirect_german_poster_prompt(self):
        route = build_route_hint(
            {
                'prompt': 'mach daraus ein poster mit art nouveau flair',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')

    def test_build_route_hint_detects_indirect_show_prompt_with_visual_modifiers_as_image(self):
        route = build_route_hint(
            {
                'prompt': 'that place you said you would like to be... can you show it to me with moonlight /at night), not at daytime please?',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')

    def test_build_route_hint_reuses_latest_image_for_indirect_german_poster_follow_up(self):
        route = build_route_hint(
            {
                'prompt': 'mach daraus ein poster mit art nouveau flair',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-image.png')
        self.assertEqual(route['reason'], 'image-generation prompt references the latest image artifact')

    def test_build_route_hint_keeps_image_edit_follow_up_on_generation(self):
        route = build_route_hint(
            {
                'prompt': 'the main character shall hold the weapons. not just randomly place them in the picture.',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'paint me "a squirrel on a tree"'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                    {'role': 'user', 'content': 'make it more powerful and cinematic'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-image.png')
        self.assertEqual(route['reason'], 'image-edit follow-up on latest image artifact')

    def test_build_route_hint_keeps_natural_robot_transform_follow_up_on_generation(self):
        route = build_route_hint(
            {
                'prompt': 'make the robot a alien entity',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': "dessin moi un image d'un robot in pre historic times"},
                    {'role': 'assistant', 'content': 'Image generated.'},
                    {'role': 'user', 'content': 'Can you make the robot golden armored with glowing blue eyes, while keeping the same prehistoric jungle scene?'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-image.png')
        self.assertEqual(route['reason'], 'image-edit follow-up on latest image artifact')

    def test_build_route_hint_keeps_pronoun_clothing_edit_follow_up_on_generation(self):
        route = build_route_hint(
            {
                'prompt': 'make her wear a red dress',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of an asian woman in a black dress'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-image.png')
        self.assertEqual(route['reason'], 'image-edit follow-up on latest image artifact')

    def test_build_route_hint_keeps_pronoun_color_edit_follow_up_on_generation(self):
        route = build_route_hint(
            {
                'prompt': 'make it gold',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of a ceremonial coin on velvet'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-image.png')
        self.assertEqual(route['reason'], 'image-edit follow-up on latest image artifact')

    def test_build_route_hint_keeps_abstract_coordination_follow_up_on_chat_after_image_history(self):
        route = build_route_hint(
            {
                'prompt': 'yeah. that i understand. but you coordinate and all. and make sure it actually works.',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of a robot diagram'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertIsNone(route['artifact_path'])
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_keeps_route_to_image_gen_control_turn_off_edit_reuse(self):
        route = build_route_hint(
            {
                'prompt': 'route to image gen.... ha ha',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of a robot diagram'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertIsNone(route['artifact_path'])
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_keeps_plain_text_gold_question_on_chat_after_image_history(self):
        route = build_route_hint(
            {
                'prompt': 'how to transmute copper to gold',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of an asian woman in a black dress'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertIsNone(route['artifact_path'])
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_keeps_plain_text_answer_request_on_chat_after_image_history(self):
        route = build_route_hint(
            {
                'prompt': 'please answer: how to transmute copper to gold, make a plan',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of an asian woman in a black dress'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertIsNone(route['artifact_path'])
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_uses_recent_user_context_for_short_tts_follow_up(self):
        route = build_route_hint(
            {
                'prompt': 'make it calmer with a männliche Stimme',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'Read this aloud: "Hallo Welt."'},
                    {'role': 'assistant', 'content': 'Audio generated.'},
                ],
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'text_to_speech')
        self.assertIn(route['reason'], {'text-to-speech cue', 'follow-up cue from recent user context'})

    def test_build_route_hint_does_not_inherit_prior_tts_for_self_contained_topic_break(self):
        route = build_route_hint(
            {
                'prompt': 'this time explain quantum entanglement in 5 bullets',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'Read this aloud: "Hallo Welt."'},
                    {'role': 'assistant', 'content': 'Audio generated.'},
                ],
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_keeps_missing_audio_upload_note_on_chat(self):
        route = build_route_hint(
            {
                'prompt': 'Sorry, die Audiodatei habe ich vergessen anzuhängen. Danke trotzdem.',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'Read this aloud: "Hallo Welt."'},
                    {'role': 'assistant', 'content': 'Audio generated.'},
                ],
                'latest_artifacts': {
                    'audio': {'type': 'audio', 'path': '/tmp/latest-audio.wav'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_does_not_inherit_prior_image_for_self_contained_topic_break(self):
        route = build_route_hint(
            {
                'prompt': 'this time explain quantum entanglement in 5 bullets',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of an asian woman in a black dress'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'default chat fallback')

    def test_build_route_hint_treats_explicit_sound_format_prompt_as_tts(self):
        route = build_route_hint(
            {
                'prompt': 'please read me aloud in sound format: "Hello, what\'s going on today?" set the style language to english, male voice',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'text_to_speech')
        self.assertEqual(route['reason'], 'text-to-speech cue')

    def test_build_route_hint_fresh_task_tts_does_not_inherit_prior_image_context(self):
        route = build_route_hint(
            {
                'prompt': 'new task: please read me aloud in sound format: "Hello, what\'s going on today?" set the style language to english, male voice',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of a ceremonial coin on velvet'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                    {'role': 'user', 'content': 'make it gold'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/latest-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'text_to_speech')
        self.assertEqual(route['reason'], 'text-to-speech cue')
        self.assertFalse(route['reuse_last_artifact'])

    def test_build_route_hint_fresh_task_show_prompt_stays_on_image_without_reusing_stale_artifact(self):
        route = build_route_hint(
            {
                'prompt': 'well then. now a completly new task. imagine another building, as mystical as this. and show it to me.',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'paint me a moonlit tower on the coast'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/stale-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertIsNone(route['artifact_path'])

    def test_build_route_hint_fresh_task_image_generation_does_not_reuse_stale_latest_image(self):
        route = build_route_hint(
            {
                'prompt': 'new task: generate an image of a Japanese village in the 16th century',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'recent_messages': [
                    {'role': 'user', 'content': 'describe this image'},
                    {'role': 'assistant', 'content': 'Detected text.'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/stale-image.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')
        self.assertFalse(route['reuse_last_artifact'])
        self.assertIsNone(route['artifact_path'])

    def test_build_route_hint_fresh_task_can_keep_explicit_selected_reference_anchor(self):
        route = build_route_hint(
            {
                'prompt': 'new task: describe this image',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/pinned-image.png'},
                },
                'selected_reference_artifact': {
                    'type': 'image',
                    'path': '/tmp/pinned-image.png',
                },
            }
        )

        self.assertEqual(route['capability'], 'vision_analysis')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/pinned-image.png')

    def test_infer_route_session_class_does_not_mark_fresh_task_with_stale_artifact_as_artifact_chain(self):
        session_class = infer_route_session_class(
            {
                'prompt': 'new task: generate an image of a Japanese village in the 16th century',
                'conversation_id': '__responses_workbench__',
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/stale-image.png'},
                },
            }
        )

        self.assertEqual(session_class, 'workbench_session')

    def test_infer_route_session_class_keeps_explicit_selected_reference_as_artifact_chain(self):
        session_class = infer_route_session_class(
            {
                'prompt': 'new task: describe this image',
                'conversation_id': '__responses_workbench__',
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/pinned-image.png'},
                },
                'selected_reference_artifact': {
                    'type': 'image',
                    'path': '/tmp/pinned-image.png',
                },
            }
        )

        self.assertEqual(session_class, 'artifact_chain')

    def test_infer_route_session_class_treats_fresh_root_workbench_as_threaded_not_inherited(self):
        session_class = infer_route_session_class(
            {
                'prompt': 'give me three bullets about Zurich',
                'conversation_id': '__responses_workbench__--fresh',
                'conversation_metadata': {
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'fresh_root': True,
                },
            }
        )

        self.assertEqual(session_class, 'threaded_session')

    def test_build_route_hint_keeps_direct_german_male_prompt_as_image(self):
        route = build_route_hint(
            {
                'prompt': 'male mir einen fuchs im schnee',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
            }
        )

        self.assertEqual(route['capability'], 'image_generation')
        self.assertEqual(route['reason'], 'image-generation cue')

    def test_build_route_hint_reuses_latest_text_for_german_tts_follow_up(self):
        route = build_route_hint(
            {
                'prompt': 'lies das vor aus diesem transkript',
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {
                    'text': {'type': 'text', 'path': '/tmp/latest-transcript.md'},
                },
            }
        )

        self.assertEqual(route['capability'], 'text_to_speech')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/latest-transcript.md')

    def test_is_obvious_route_hint_fast_path_accepts_simple_chat(self):
        context = {
            'prompt': 'expand this sentence a bit further please',
            'request_attachment': {'has_explicit_file': False, 'file_kind': None},
            'latest_artifacts': {},
        }
        route_hint = {
            'capability': 'chat',
            'confidence': 0.95,
            'reason': 'default chat fallback',
        }

        self.assertTrue(is_obvious_route_hint_fast_path(context, route_hint))

    def test_is_obvious_route_hint_fast_path_accepts_obvious_tts(self):
        context = {
            'prompt': 'generate me an audio of the following english sentence: "Hello world."',
            'request_attachment': {'has_explicit_file': False, 'file_kind': None},
            'latest_artifacts': {},
        }
        route_hint = {
            'capability': 'text_to_speech',
            'confidence': 0.9,
            'reason': 'text-to-speech cue',
        }

        self.assertTrue(is_obvious_route_hint_fast_path(context, route_hint))

    def test_is_obvious_route_hint_fast_path_rejects_artifact_reference_chat(self):
        context = {
            'prompt': 'continue from that transcript',
            'request_attachment': {'has_explicit_file': False, 'file_kind': None},
            'latest_artifacts': {'text': {'path': '/tmp/latest.md'}},
        }
        route_hint = {
            'capability': 'chat',
            'confidence': 0.95,
            'reason': 'default chat fallback',
        }

        self.assertFalse(is_obvious_route_hint_fast_path(context, route_hint))

    @patch('ollmo_g.router.read_chat_history', return_value={'instance_id': '__responses_workbench__', 'messages': []})
    def test_build_route_context_collects_recent_artifacts(self, _mock_read_chat_history):
        context = build_route_context(
            prompt='continue from that OCR',
            upload_filename='',
            file_path='',
            conversation_id='__responses_workbench__',
            messages=[
                {
                    'role': 'assistant',
                    'content': 'Audio generated.',
                    'saved_audio_path': '/tmp/voice.wav',
                    'timestamp': '2026-03-21T09:58:00Z',
                },
                {
                    'role': 'assistant',
                    'content': 'OCR output',
                    'saved_text_path': '/tmp/ocr.md',
                    'timestamp': '2026-03-21T10:01:00Z',
                },
                {
                    'role': 'assistant',
                    'content': 'Images generated.',
                    'artifacts': [
                        {
                            'type': 'image',
                            'path': '/tmp/one.png',
                            'image_state': {
                                'summary': 'A heroic squirrel on a tree, cinematic comic-book style.',
                                'subject': 'heroic squirrel on a tree',
                            },
                        },
                        {
                            'type': 'image',
                            'path': '/tmp/two.png',
                            'image_state': {
                                'summary': 'A squirrel wearing a cape beside a waterfall.',
                                'subject': 'squirrel beside a waterfall',
                            },
                        },
                    ],
                    'timestamp': '2026-03-21T10:02:00Z',
                },
            ],
            runtime_manifest={
                'capabilities': {
                    'chat': {
                        'default_instance_id': 'chat-1',
                        'count': 1,
                        'candidates': [
                            {
                                'instance_id': 'chat-1',
                                'model': 'gpt-oss:20b',
                                'backend': 'ollama',
                                'backend_package': 'ollama',
                                'backend_contract': 'ollama.api',
                                'provider_capabilities': ['chat'],
                                'session_controls_summary': {'visible_fields': ['temperature'], 'required_fields': []},
                                'routing_summary': {'backend_package': 'ollama'},
                            }
                        ],
                    },
                },
                'instances': [
                    {
                        'instance_id': 'chat-1',
                        'model': 'gpt-oss:20b',
                        'backend': 'ollama',
                        'capability': 'chat',
                        'backend_package': 'ollama',
                        'backend_contract': 'ollama.api',
                        'provider_capabilities': ['chat'],
                        'session_controls_summary': {'visible_fields': ['temperature'], 'required_fields': []},
                        'routing_summary': {'backend_package': 'ollama'},
                    }
                ],
            },
            ghost_payload={
                'recommendations': [],
                'issues': [],
                'recent_events': [
                    {
                        'timestamp': '2026-03-21T10:05:00Z',
                        'category': 'infer',
                        'action': 'request',
                        'status': 'ok',
                        'message': 'chat',
                        'instance_id': 'chat-1',
                        'model': 'gpt-oss:20b',
                        'backend': 'ollama',
                        'capability': 'chat',
                    }
                ],
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gpt-oss:20b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'backend_package': 'ollama',
                    'backend_contract': 'ollama.api',
                    'provider_capabilities': ['chat'],
                    'port': 11437,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                },
            ],
        )

        router_messages = build_router_messages(context)

        self.assertEqual(
            [item['path'] for item in context['recent_artifacts']],
            ['/tmp/one.png', '/tmp/two.png', '/tmp/ocr.md', '/tmp/voice.wav'],
        )
        self.assertEqual(context['latest_artifacts']['image']['path'], '/tmp/one.png')
        self.assertEqual(context['latest_artifacts']['text']['path'], '/tmp/ocr.md')
        self.assertEqual(context['latest_artifacts']['audio']['path'], '/tmp/voice.wav')
        self.assertEqual(context['recent_artifacts'][0]['image_state']['subject'], 'heroic squirrel on a tree')
        self.assertEqual(context['recent_artifacts'][1]['image_state']['subject'], 'squirrel beside a waterfall')
        self.assertEqual(context['runtime']['available_capabilities']['chat']['candidates'][0]['backend_package'], 'ollama')
        self.assertEqual(context['runtime']['available_instances'][0]['backend_contract'], 'ollama.api')
        self.assertEqual(context['instances'][0]['routing_summary']['backend_package'], 'ollama')
        self.assertNotIn('memory', context)
        self.assertEqual(router_messages[0]['role'], 'system')
        self.assertEqual(router_messages[1]['role'], 'user')
        self.assertNotIn('"memory"', router_messages[1]['content'])
        self.assertNotIn('hot_memory', router_messages[1]['content'])
        self.assertIn('"recent_messages"', router_messages[1]['content'])
        self.assertIn('"recent_artifacts"', router_messages[1]['content'])

    def test_build_route_context_preserves_dynamic_control_and_tts_truth(self):
        context = build_route_context(
            prompt='read this aloud in English with a calm male voice',
            upload_filename='',
            file_path='',
            conversation_id='conversation-tts',
            messages=[{'role': 'user', 'content': 'read this aloud in English with a calm male voice'}],
            runtime_manifest={
                'capabilities': {
                    'text_to_speech': {
                        'default_instance_id': 'tts-voice',
                        'count': 1,
                        'candidates': [
                            {
                                'instance_id': 'tts-voice',
                                'model': 'Qwen3-TTS-VoiceDesign',
                                'backend': 'mlx',
                                'backend_package': 'mlx_audio',
                                'backend_contract': 'mlx_audio.server',
                                'provider_capabilities': ['text_to_speech'],
                                'inputs': ['text'],
                                'outputs': ['audio'],
                                'dynamic_model_traits': {'snapshot_languages': ['en', 'de'], 'voice_clone': True},
                                'tts_model_type': 'voice_design',
                                'tts_languages': ['en', 'de'],
                                'tts_speakers': ['serena'],
                                'session_controls_summary': {
                                    'enabled': True,
                                    'visible_fields': ['tts_instruct', 'lang_code'],
                                    'required_fields': ['tts_instruct'],
                                    'labels': ['Style / Instruct', 'Language'],
                                    'field_types': {'tts_instruct': 'textarea', 'lang_code': 'select'},
                                    'field_options': {'lang_code': ['en', 'de']},
                                },
                                'routing_summary': {'session_controls': {'required_fields': ['tts_instruct']}},
                            }
                        ],
                    }
                },
                'instances': [
                    {
                        'instance_id': 'tts-voice',
                        'model': 'Qwen3-TTS-VoiceDesign',
                        'backend': 'mlx',
                        'capability': 'text_to_speech',
                        'backend_package': 'mlx_audio',
                        'backend_contract': 'mlx_audio.server',
                        'provider_capabilities': ['text_to_speech'],
                        'inputs': ['text'],
                        'outputs': ['audio'],
                        'dynamic_model_traits': {'snapshot_languages': ['en', 'de'], 'voice_clone': True},
                        'tts_model_type': 'voice_design',
                        'tts_languages': ['en', 'de'],
                        'tts_speakers': ['serena'],
                        'session_controls_summary': {
                            'enabled': True,
                            'visible_fields': ['tts_instruct', 'lang_code'],
                            'required_fields': ['tts_instruct'],
                            'labels': ['Style / Instruct', 'Language'],
                            'field_types': {'tts_instruct': 'textarea', 'lang_code': 'select'},
                            'field_options': {'lang_code': ['en', 'de']},
                        },
                        'routing_summary': {'session_controls': {'required_fields': ['tts_instruct']}},
                    }
                ],
            },
            ghost_payload={'recommendations': [], 'issues': [], 'recent_events': []},
            instances=[
                {
                    'instance_id': 'tts-voice',
                    'model': 'Qwen3-TTS-VoiceDesign',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'backend_package': 'mlx_audio',
                    'backend_contract': 'mlx_audio.server',
                    'provider_capabilities': ['text_to_speech'],
                    'port': 11509,
                    'snapshot_languages': ['en', 'de'],
                    'voice_clone': True,
                    'tts_model_type': 'voice_design',
                    'tts_languages': ['en', 'de'],
                    'tts_speakers': ['serena'],
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {'visible': True, 'required': True, 'label': 'Style / Instruct', 'kind': 'textarea'},
                            'lang_code': {'visible': True, 'label': 'Language', 'kind': 'select', 'options': ['en', 'de']},
                        },
                    },
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                }
            ],
        )

        candidate = context['runtime']['available_capabilities']['text_to_speech']['candidates'][0]
        instance = context['runtime']['available_instances'][0]
        normalized_instance = context['instances'][0]

        self.assertEqual(candidate['tts_model_type'], 'voice_design')
        self.assertEqual(candidate['tts_languages'], ['en', 'de'])
        self.assertEqual(candidate['session_controls_summary']['field_types']['lang_code'], 'select')
        self.assertEqual(candidate['session_controls_summary']['field_options']['lang_code'], ['en', 'de'])
        self.assertEqual(candidate['dynamic_model_traits']['snapshot_languages'], ['en', 'de'])
        self.assertEqual(instance['tts_speakers'], ['serena'])
        self.assertEqual(instance['session_controls_summary']['field_options']['lang_code'], ['en', 'de'])
        self.assertTrue(instance['dynamic_model_traits']['voice_clone'])
        self.assertEqual(normalized_instance['routing_summary']['tts_model_type'], 'voice_design')
        self.assertEqual(normalized_instance['routing_summary']['dynamic_model_traits']['snapshot_languages'], ['en', 'de'])
        self.assertEqual(normalized_instance['routing_summary']['session_controls']['field_types']['tts_instruct'], 'textarea')

    def test_build_ghost_memory_summarizes_older_messages_into_warm_memory(self):
        messages = []
        for index in range(1, 7):
            base_minute = (index - 1) * 2
            messages.extend(
                [
                    {'role': 'user', 'content': f'Request {index}', 'timestamp': f'2026-03-21T10:{base_minute:02d}:00Z'},
                    {'role': 'assistant', 'content': f'Answer {index}', 'timestamp': f'2026-03-21T10:{base_minute + 1:02d}:00Z', 'response_model': 'gemma4:26b'},
                ]
            )
        memory = build_ghost_memory(
            messages=messages,
            recent_artifacts=[],
            recent_events=[],
            conversation_id='conversation-1',
        )

        self.assertEqual(memory['hot_memory']['turn_count'], 8)
        self.assertEqual(memory['warm_memory']['older_turn_count'], 4)
        self.assertEqual(len(memory['warm_memory']['summary_lines']), 4)
        self.assertEqual(memory['expansion_order'], ['hot_memory', 'warm_memory', 'deep_memory'])
        self.assertNotIn('recent_user_chain', memory)

    @patch('ollmo_g.router.read_chat_history')
    def test_build_route_context_hydrates_recent_messages_from_durable_history(self, mock_read_chat_history):
        mock_read_chat_history.return_value = {
            'instance_id': '__responses_workbench__',
            'messages': [
                {'role': 'user', 'content': 'Read this aloud: "Hallo Welt."', 'timestamp': '2026-03-21T10:00:00Z'},
                {'role': 'assistant', 'content': 'Audio generated.', 'timestamp': '2026-03-21T10:01:00Z', 'response_model': 'tts-model'},
                {'role': 'user', 'content': 'Use a männliche Stimme', 'timestamp': '2026-03-21T10:02:00Z'},
                {'role': 'assistant', 'content': 'Audio generated.', 'timestamp': '2026-03-21T10:03:00Z', 'response_model': 'tts-model'},
                {'role': 'user', 'content': 'male mir einen fuchs im schnee', 'timestamp': '2026-03-21T10:04:00Z'},
                {'role': 'assistant', 'content': 'Image generated.', 'timestamp': '2026-03-21T10:05:00Z', 'response_model': 'flux'},
                {'role': 'user', 'content': 'mach daraus ein poster', 'timestamp': '2026-03-21T10:06:00Z'},
                {'role': 'assistant', 'content': 'Image generated.', 'timestamp': '2026-03-21T10:07:00Z', 'response_model': 'flux'},
                {'role': 'user', 'content': 'continue', 'timestamp': '2026-03-21T10:08:00Z'},
            ],
        }

        context = build_route_context(
            prompt='make it calmer',
            upload_filename='',
            file_path='',
            conversation_id='__responses_workbench__',
            messages=[
                {'role': 'user', 'content': 'make it calmer', 'timestamp': '2026-03-21T10:09:00Z'},
            ],
            runtime_manifest={'capabilities': {}, 'instances': []},
            ghost_payload={
                'recommendations': [],
                'issues': [],
                'recent_events': [],
                'self_observations': [{'kind': 'successful_pattern', 'reason': 'recent image chain success', 'count': 2}],
                'self_healing_hints': [{'kind': 'preserve_successful_execution', 'reason': 'Prefer stable successes.'}],
            },
            instances=[],
        )

        self.assertEqual(
            [item['content'] for item in context['recent_messages']],
            ['mach daraus ein poster', 'Image generated.'],
        )
        self.assertNotIn('memory', context)
        self.assertNotIn('ghost_self_healing_hints', context['runtime'])

    @patch('ollmo_g.router.read_chat_history')
    def test_build_route_context_omits_recent_messages_for_fresh_turns(self, mock_read_chat_history):
        mock_read_chat_history.return_value = {
            'instance_id': '__responses_workbench__',
            'messages': [
                {'role': 'user', 'content': 'male mir einen fuchs im schnee', 'timestamp': '2026-03-21T10:04:00Z'},
                {
                    'role': 'assistant',
                    'content': 'Image generated.',
                    'timestamp': '2026-03-21T10:05:00Z',
                    'response_model': 'flux',
                    'saved_image_path': '/tmp/fox.png',
                },
                {'role': 'user', 'content': 'mach daraus ein poster', 'timestamp': '2026-03-21T10:06:00Z'},
                {
                    'role': 'assistant',
                    'content': 'Image generated.',
                    'timestamp': '2026-03-21T10:07:00Z',
                    'response_model': 'flux',
                    'saved_image_path': '/tmp/poster.png',
                },
            ],
        }

        context = build_route_context(
            prompt='give me three bullets about Zurich',
            upload_filename='',
            file_path='',
            conversation_id='__responses_workbench__',
            messages=[
                {'role': 'user', 'content': 'give me three bullets about Zurich', 'timestamp': '2026-03-21T10:09:00Z'},
            ],
            runtime_manifest={'capabilities': {}, 'instances': []},
            ghost_payload={
                'recommendations': [],
                'issues': [],
                'recent_events': [],
            },
            instances=[],
        )

        self.assertEqual(context['recent_messages'], [])
        self.assertEqual(context['recent_artifacts'], [])
        self.assertEqual(context['latest_artifacts'], {})
        self.assertEqual(context['runtime']['intake_context']['history_binding'], 'current_turn_only')
        self.assertNotIn('memory', context)

    @patch('ollmo_g.router.read_chat_history')
    def test_inline_direct_tts_is_not_an_artifact_follow_up_or_thread_reference(
        self,
        mock_read_chat_history,
    ):
        mock_read_chat_history.return_value = {
            'instance_id': '__responses_workbench__',
            'messages': [
                {
                    'role': 'user',
                    'content': 'Draft an unrelated sentence.',
                    'timestamp': '2026-08-03T18:00:00Z',
                },
                {
                    'role': 'assistant',
                    'content': 'This is stale thread text.',
                    'saved_text_path': '/tmp/stale-thread-text.md',
                    'timestamp': '2026-08-03T18:01:00Z',
                },
            ],
        }
        prompts = (
            (
                'Create exactly one English audio artifact using local text-to-speech. '
                'Speak this text exactly: "At sunrise, the harbor slowly came alive."'
            ),
            (
                'Erstelle genau ein deutsches Audio-Artefakt mit lokaler Sprachsynthese. '
                'Sprich diesen Text genau: „Bei Sonnenaufgang erwachte der Hafen langsam.“'
            ),
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                context = build_route_context(
                    prompt=prompt,
                    upload_filename='',
                    file_path='',
                    conversation_id='__responses_workbench__',
                    messages=[{'role': 'user', 'content': prompt}],
                    runtime_manifest={'capabilities': {}, 'instances': []},
                    ghost_payload={'recommendations': [], 'issues': []},
                    instances=[],
                )
                route = build_route_hint(context)
                memory_scope = build_route_memory_scope(context, route_hint=route)

                self.assertEqual(route['capability'], 'text_to_speech')
                self.assertIsNone(memory_scope['prompt_class'])
                self.assertFalse(
                    context['runtime']['intake_context']['thread_context_requested']
                )
                self.assertEqual(
                    context['runtime']['intake_context']['history_binding'],
                    'current_turn_only',
                )
                self.assertEqual(context['recent_messages'], [])
                self.assertEqual(context['recent_artifacts'], [])
                self.assertEqual(context['latest_artifacts'], {})

    def test_write_then_read_aloud_remains_chat_preparation(self):
        prompt = 'Write a short poem about the harbor, then read it aloud.'
        intent = analyze_prompt_intent(prompt)
        route = build_route_hint(
            {
                'prompt': prompt,
                'request_attachment': {'has_explicit_file': False, 'file_kind': None},
                'latest_artifacts': {},
                'memory': {},
            }
        )

        self.assertTrue(intent['requests_audio_output'])
        self.assertTrue(intent['text_preparation_before_audio_output'])
        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'text preparation required before audio output')

    @patch('ollmo_g.router.read_chat_history')
    def test_ungrounded_tts_transform_remains_an_artifact_follow_up_with_context(
        self,
        mock_read_chat_history,
    ):
        prompt = 'Read this text aloud.'
        mock_read_chat_history.return_value = {
            'instance_id': '__responses_workbench__',
            'messages': [
                {'role': 'user', 'content': 'Write a short status update.'},
                {'role': 'assistant', 'content': 'The rollout is complete.'},
            ],
        }

        context = build_route_context(
            prompt=prompt,
            upload_filename='',
            file_path='',
            conversation_id='__responses_workbench__',
            messages=[{'role': 'user', 'content': prompt}],
            runtime_manifest={'capabilities': {}, 'instances': []},
            ghost_payload={'recommendations': [], 'issues': []},
            instances=[],
        )
        route = build_route_hint(context)
        memory_scope = build_route_memory_scope(context, route_hint=route)

        self.assertEqual(memory_scope['prompt_class'], 'artifact_follow_up')
        self.assertTrue(context['runtime']['intake_context']['thread_context_requested'])
        self.assertEqual(
            context['runtime']['intake_context']['history_binding'],
            'referential',
        )
        self.assertNotEqual(context['recent_messages'], [])

    @patch('ollmo_g.router.read_chat_history')
    def test_build_route_context_preserves_fresh_root_conversation_metadata(self, mock_read_chat_history):
        mock_read_chat_history.return_value = {
            'instance_id': '__responses_workbench__--fresh',
            'conversation_metadata': {
                'workspace': 'responses',
                'slot_id': 'responses-workbench',
                'fresh_root': True,
            },
            'messages': [],
        }

        context = build_route_context(
            prompt='give me three bullets about Zurich',
            upload_filename='',
            file_path='',
            conversation_id='__responses_workbench__--fresh',
            messages=[
                {'role': 'user', 'content': 'give me three bullets about Zurich', 'timestamp': '2026-03-21T10:09:00Z'},
            ],
            runtime_manifest={'capabilities': {}, 'instances': []},
            ghost_payload={
                'recommendations': [],
                'issues': [],
                'recent_events': [],
            },
            instances=[],
        )

        self.assertTrue(context['conversation_metadata']['fresh_root'])
        self.assertEqual(infer_route_session_class(context), 'threaded_session')

    def test_build_ghost_memory_collects_only_explicit_stable_preferences(self):
        memory = build_ghost_memory(
            messages=[
                {'role': 'user', 'content': 'Bitte antworte standardmäßig auf Deutsch.'},
                {'role': 'assistant', 'content': 'ok'},
                {'role': 'user', 'content': 'By default, use a calm male voice when reading aloud.'},
                {'role': 'assistant', 'content': 'ok'},
                {'role': 'user', 'content': 'I prefer audio replies by default.'},
                {'role': 'assistant', 'content': 'ok'},
                {'role': 'user', 'content': 'Use a male voice for this clip only.'},
            ],
            recent_artifacts=[],
            recent_events=[],
        )

        preferences = memory['deep_memory']['stable_user_preferences']

        self.assertEqual(preferences['languages'], ['de'])
        self.assertEqual(preferences['voice_descriptors'], ['male', 'calm'])
        self.assertEqual(preferences['preferred_modalities'], ['text_to_speech'])
        self.assertEqual(preferences['audio_formats'], [])

    def test_build_ghost_memory_does_not_infer_stable_preferences_from_ordinary_turns(self):
        memory = build_ghost_memory(
            messages=[
                {'role': 'user', 'content': 'Read this aloud in German with a calm male voice.'},
                {'role': 'assistant', 'content': 'ok'},
                {'role': 'user', 'content': 'Make me an image of a fox in snow.'},
            ],
            recent_artifacts=[],
            recent_events=[],
        )

        self.assertEqual(
            memory['deep_memory']['stable_user_preferences'],
            {
                'languages': [],
                'voice_descriptors': [],
                'preferred_modalities': [],
                'audio_formats': [],
            },
        )

    def test_build_recent_learnings_from_events_includes_successful_execution_patterns(self):
        learnings = build_recent_learnings_from_events(
            [
                {
                    'timestamp': '2026-03-21T10:05:00Z',
                    'category': 'infer',
                    'action': 'request',
                    'status': 'ok',
                    'message': 'text_to_speech',
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                },
                {
                    'timestamp': '2026-03-21T10:06:00Z',
                    'category': 'infer',
                    'action': 'request',
                    'status': 'ok',
                    'message': 'text_to_speech',
                    'instance_id': 'tts-1',
                    'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                },
            ]
        )

        self.assertEqual(len(learnings), 1)
        self.assertEqual(learnings[0]['kind'], 'successful_execution')
        self.assertEqual(learnings[0]['capability'], 'text_to_speech')
        self.assertEqual(learnings[0]['model'], 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16')
        self.assertEqual(learnings[0]['count'], 2)

    def test_build_recent_self_observations_balances_successes_failures_and_runtime_state(self):
        observations = build_recent_self_observations(
            [
                {
                    'timestamp': '2026-03-21T10:05:00Z',
                    'category': 'infer',
                    'action': 'request',
                    'status': 'ok',
                    'message': 'text_to_speech',
                    'instance_id': 'tts-1',
                    'model': 'tts-model',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                },
                {
                    'timestamp': '2026-03-21T10:06:00Z',
                    'category': 'infer',
                    'action': 'request',
                    'status': 'ok',
                    'message': 'text_to_speech',
                    'instance_id': 'tts-1',
                    'model': 'tts-model',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                },
                {
                    'timestamp': '2026-03-21T10:07:00Z',
                    'category': 'responses',
                    'action': 'route',
                    'status': 'ok',
                    'message': 'default chat fallback [supports image-aware chat]',
                    'instance_id': 'chat-1',
                    'model': 'chat-model',
                    'backend': 'ollama',
                    'capability': 'chat',
                },
            ],
            runtime_issues=['mlx-1 is degraded: sample issue'],
            log_lines=['2026-04-03 16:55:45,334 - INFO - Ghost router execution fallback: HTTPConnectionPool(host=\'localhost\', port=11437): Read timed out. (read timeout=35)'],
        )

        self.assertEqual(observations[0]['kind'], 'successful_pattern')
        self.assertEqual(observations[0]['capability'], 'text_to_speech')
        self.assertTrue(any(item.get('kind') == 'fallback_pattern' for item in observations))
        self.assertTrue(any(item.get('kind') == 'router_timeout' for item in observations))
        self.assertTrue(any(item.get('kind') == 'runtime_issue' for item in observations))

        hints = build_self_healing_hints(observations)
        hint_kinds = {item.get('kind') for item in hints}
        self.assertIn('preserve_successful_execution', hint_kinds)
        self.assertIn('degrade_gracefully_after_router_timeout', hint_kinds)

    def test_build_embedding_hints_from_vectors_ranks_dynamic_candidates(self):
        candidates = build_embedding_route_candidates(
            runtime_manifest={
                'capabilities': {
                    'chat': {
                        'aliases': ['chat'],
                        'default_instance_id': 'chat-1',
                        'candidates': [
                            {
                                'instance_id': 'chat-1',
                                'model': 'gpt-oss:20b',
                                'backend_package': 'ollama',
                                'backend_contract': 'ollama.api',
                                'session_controls_summary': {'required_fields': []},
                            }
                        ],
                    },
                    'vision_analysis': {
                        'aliases': ['vision', 'ocr'],
                        'default_instance_id': 'vision-1',
                        'candidates': [
                            {
                                'instance_id': 'vision-1',
                                'model': 'deepseek-ocr:latest',
                                'backend_package': 'mlx_vlm',
                                'backend_contract': 'mlx_vlm.server',
                                'session_controls_summary': {'required_fields': ['ocr_mode']},
                            }
                        ],
                    },
                    'embedding': {
                        'aliases': ['embedding'],
                        'default_instance_id': 'embed-1',
                        'candidates': [{'instance_id': 'embed-1', 'model': 'embeddinggemma:latest'}],
                    },
                }
            },
            instances=[
                {
                    'instance_id': 'chat-1',
                    'model': 'gpt-oss:20b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                    'inputs': ['text'],
                    'outputs': ['text'],
                    'backend_package': 'ollama',
                    'backend_contract': 'ollama.api',
                },
                {
                    'instance_id': 'vision-1',
                    'model': 'deepseek-ocr:latest',
                    'backend': 'ollama',
                    'capability': 'vision_analysis',
                    'port': 11438,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                    'inputs': ['text', 'image'],
                    'outputs': ['text'],
                    'backend_package': 'mlx_vlm',
                    'backend_contract': 'mlx_vlm.server',
                    'backend_metadata': {
                        'package_label': 'mlx-vlm',
                        'runtime_constraints': ['single_loaded_model'],
                    },
                    'runtime_status': {
                        'readiness': 'ready',
                        'activity': 'idle',
                        'backend_runtime': {'request_model_strategy': 'model_bound_at_launch'},
                    },
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'ocr_mode': {'visible': True, 'required': True, 'label': 'Document Mode'},
                        },
                    },
                },
                {
                    'instance_id': 'embed-1',
                    'model': 'embeddinggemma:latest',
                    'backend': 'ollama',
                    'capability': 'embedding',
                    'port': 11435,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                    'inputs': ['text'],
                    'outputs': ['embedding'],
                },
            ],
        )

        hints = build_embedding_hints_from_vectors(
            [1.0, 0.0],
            candidates,
            [
                [0.2, 0.8],
                [1.0, 0.0],
                [0.1, 0.9],
                [0.9, 0.1],
            ],
        )

        self.assertIsNotNone(hints)
        self.assertEqual(hints['top_capabilities'][0]['capability'], 'vision_analysis')
        self.assertEqual(hints['top_instances'][0]['instance_id'], 'vision-1')
        self.assertIn('package mlx_vlm', candidates[3]['text'])
        self.assertIn('required_controls ocr_mode', candidates[1]['text'])

    def test_build_embedding_route_candidates_mentions_dynamic_control_truth(self):
        candidates = build_embedding_route_candidates(
            runtime_manifest={
                'capabilities': {
                    'text_to_speech': {
                        'aliases': ['tts'],
                        'default_instance_id': 'tts-voice',
                        'candidates': [
                            {
                                'instance_id': 'tts-voice',
                                'model': 'Qwen3-TTS-VoiceDesign',
                                'backend_package': 'mlx_audio',
                                'backend_contract': 'mlx_audio.server',
                                'dynamic_model_traits': {'snapshot_languages': ['en', 'de'], 'voice_clone': True},
                                'tts_model_type': 'voice_design',
                                'tts_languages': ['en', 'de'],
                                'session_controls_summary': {
                                    'required_fields': ['tts_instruct'],
                                    'visible_fields': ['tts_instruct', 'lang_code'],
                                },
                            }
                        ],
                    }
                }
            },
            instances=[
                {
                    'instance_id': 'tts-voice',
                    'model': 'Qwen3-TTS-VoiceDesign',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'port': 11509,
                    'runtime_status': {'readiness': 'ready', 'activity': 'idle'},
                    'backend_package': 'mlx_audio',
                    'backend_contract': 'mlx_audio.server',
                    'snapshot_languages': ['en', 'de'],
                    'voice_clone': True,
                    'tts_model_type': 'voice_design',
                    'tts_languages': ['en', 'de'],
                    'tts_speakers': ['serena'],
                    'session_controls': {
                        'enabled': True,
                        'fields': {
                            'tts_instruct': {'visible': True, 'required': True, 'label': 'Style / Instruct', 'kind': 'textarea'},
                            'lang_code': {'visible': True, 'label': 'Language', 'kind': 'select', 'options': ['en', 'de']},
                        },
                    },
                }
            ],
        )

        capability_candidate = next(item for item in candidates if item['kind'] == 'capability')
        instance_candidate = next(item for item in candidates if item['kind'] == 'instance')

        self.assertIn('dynamic_traits snapshot_languages:[\'en\', \'de\']', capability_candidate['text'])
        self.assertIn('tts_models voice_design', capability_candidate['text'])
        self.assertIn('tts_languages en de', capability_candidate['text'])
        self.assertIn('dynamic_traits snapshot_languages:[\'en\', \'de\'] voice_clone:True', instance_candidate['text'])
        self.assertIn('control_types tts_instruct:textarea', instance_candidate['text'])
        self.assertIn('control_options lang_code:en/de', instance_candidate['text'])
        self.assertIn('tts_speakers serena', instance_candidate['text'])

    def test_maybe_apply_embedding_route_bias_rescues_image_follow_up_from_chat(self):
        context = {
            'prompt': 'make it moodier and more cinematic',
            'recent_messages': [
                {'role': 'user', 'content': 'generate an image of a fox in a snowy forest'},
                {'role': 'assistant', 'content': 'Image generated.'},
            ],
            'latest_artifacts': {
                'image': {'type': 'image', 'path': '/tmp/generated/fox.png'},
            },
            'selected_reference_artifact': {'type': 'image', 'path': '/tmp/generated/fox.png'},
            'request_attachment': {'has_explicit_file': False},
            'runtime': {
                'embedding_hints': {
                    'top_capabilities': [
                        {'capability': 'image_generation', 'score': 0.93, 'default_instance_id': 'flux-1'},
                        {'capability': 'chat', 'score': 0.71},
                    ],
                    'top_instances': [
                        {'instance_id': 'flux-1', 'score': 0.9},
                    ],
                }
            },
        }

        route = maybe_apply_embedding_route_bias(
            context,
            {
                'capability': 'chat',
                'confidence': 0.62,
                'reuse_last_artifact': False,
                'artifact_path': None,
                'reason': 'default chat fallback',
            },
        )

        self.assertIsNotNone(route)
        self.assertEqual(route['capability'], 'image_generation')
        self.assertTrue(route['reuse_last_artifact'])
        self.assertEqual(route['artifact_path'], '/tmp/generated/fox.png')

    def test_build_embedding_route_audit_marks_biased_alignment(self):
        context = {
            'prompt': 'make it moodier and more cinematic',
            'recent_messages': [
                {'role': 'user', 'content': 'generate an image of a fox in a snowy forest'},
            ],
            'latest_artifacts': {
                'image': {'type': 'image', 'path': '/tmp/generated/fox.png'},
            },
            'selected_reference_artifact': {'type': 'image', 'path': '/tmp/generated/fox.png'},
            'runtime': {
                'embedding_helper': {
                    'available': True,
                    'attached': True,
                    'instance_id': 'embeddinggemma:latest-1',
                    'model': 'embeddinggemma:latest',
                },
                'embedding_hints': {
                    'top_capabilities': [
                        {'capability': 'image_generation', 'score': 0.93, 'default_instance_id': 'flux-1'},
                        {'capability': 'chat', 'score': 0.71},
                    ],
                    'top_instances': [
                        {'instance_id': 'flux-1', 'score': 0.9},
                    ],
                },
            },
        }

        audit = build_embedding_route_audit(
            context,
            {'capability': 'chat', 'confidence': 0.62},
            {'capability': 'image_generation'},
            bias_applied=True,
        )

        self.assertEqual(audit['status'], 'biased_alignment')
        self.assertEqual(audit['prompt_class'], 'image_edit_follow_up')
        self.assertEqual(audit['embedding_capability'], 'image_generation')
        self.assertEqual(audit['final_capability'], 'image_generation')

    def test_build_recent_self_observations_keeps_router_timeout_scope_from_events(self):
        observations = build_recent_self_observations(
            [
                {
                    'timestamp': '2026-04-10T18:40:00Z',
                    'category': 'responses',
                    'action': 'router_runtime',
                    'status': 'timeout',
                    'prompt_class': 'artifact_follow_up',
                    'session_class': 'artifact_chain',
                    'message': 'Ghost router execution fallback: Read timed out',
                }
            ]
        )

        timeout_observation = next(item for item in observations if item.get('kind') == 'router_timeout')
        self.assertEqual(timeout_observation['prompt_class'], 'artifact_follow_up')
        self.assertEqual(timeout_observation['session_class'], 'artifact_chain')

    def test_build_route_memory_scope_reports_prompt_and_session_classes(self):
        scope = build_route_memory_scope(
            {
                'prompt': 'make it moodier and more cinematic',
                'conversation_id': 'responses-workbench',
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of a fox in a snowy forest'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/generated/fox.png'},
                },
                'selected_reference_artifact': {
                    'type': 'image',
                    'path': '/tmp/generated/fox.png',
                },
            },
            route_hint={'capability': 'chat', 'confidence': 0.6},
        )

        self.assertEqual(scope['prompt_class'], 'image_edit_follow_up')
        self.assertEqual(scope['session_class'], 'artifact_chain')
        self.assertEqual(scope['routing_preferences']['matched_policy_ids'], [])

    def test_build_route_hint_prefers_plain_chat_for_plain_text_prompt(self):
        route = build_route_hint(
            {
                'prompt': 'hello there',
                'recent_messages': [
                    {'role': 'user', 'content': 'generate an image of a fox in a snowy forest'},
                    {'role': 'assistant', 'content': 'Image generated.'},
                    {'role': 'user', 'content': 'hello there'},
                ],
                'latest_artifacts': {
                    'image': {'type': 'image', 'path': '/tmp/generated/fox.png'},
                },
            }
        )

        self.assertEqual(route['capability'], 'chat')
        self.assertEqual(route['reason'], 'default chat fallback')


class GhostContextStrategyTests(unittest.TestCase):
    def setUp(self):
        self.owner = GhostRouteRuntimeOwner(
            hooks={'estimate_route_context_tokens': lambda **_kwargs: 256},
            wrapper_capability_aliases={},
            max_recent_messages=8,
        )

    def test_legacy_mode_planner_timeout_bonus_is_ignored(self):
        owner = GhostRouteRuntimeOwner(
            hooks={'timeout_ms_to_seconds': lambda value: None},
            wrapper_capability_aliases={},
            max_recent_messages=8,
        )

        timeout_sec = owner._planner_timeout_seconds_for_payload(
            {},
            semantic_role_profile={'runtime_orientation': {'planner_timeout_bonus_sec': 600}},
        )

        self.assertIsNone(timeout_sec)

    def test_current_turn_only_strategy_prunes_stale_artifact_history(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {'role': 'user', 'content': 'Generate an image of a fox in a snowy forest.'},
            {
                'role': 'assistant',
                'content': 'Image generated.',
                'artifacts': [{'type': 'image', 'path': '/tmp/fox.png'}],
            },
            {'role': 'user', 'content': 'Explain request graphs conceptually.'},
        ]

        strategy = self.owner.choose_context_strategy(
            instance={},
            messages=messages,
            prompt='Explain request graphs conceptually.',
            has_file_context=False,
        )
        prepared = self.owner.apply_context_strategy(messages, strategy)

        self.assertEqual(strategy['mode'], 'current_turn_only')
        candidates = {
            item['candidate_id']: item
            for item in strategy['context_candidates']
        }
        message_candidates = [
            item for item in strategy['context_candidates']
            if item.get('source_kind') == 'message'
        ]
        self.assertEqual(message_candidates[0]['status'], 'not_promoted')
        self.assertEqual(message_candidates[1]['status'], 'not_promoted')
        self.assertEqual(candidates['history-scan-deeper-pool']['status'], 'not_promoted')
        self.assertEqual(
            candidates['history-scan-deeper-pool']['scan_targets'],
            ['chat_history', 'response_frame_ledger', 'artifact_registry'],
        )
        review = strategy['context_gate_review']
        self.assertEqual(review['kind'], 'ollmo.context_gate_review')
        self.assertEqual(review['history_scan']['decision'], 'not_promoted')
        self.assertFalse(review['history_scan']['executed'])
        self.assertEqual(review['not_promoted_candidate_count'], 3)
        self.assertEqual(strategy['promoted_candidate_ids'], [])
        self.assertEqual(
            [item['content'] for item in prepared],
            ['Ollmo runtime policy.', 'Explain request graphs conceptually.'],
        )

    def test_inline_direct_tts_uses_current_turn_only_but_ungrounded_tts_keeps_history(self):
        inline_prompts = (
            (
                'Create exactly one English audio artifact using local text-to-speech. '
                'Speak this text exactly: "At sunrise, the harbor slowly came alive."'
            ),
            (
                'Erstelle genau ein deutsches Audio-Artefakt mit lokaler Sprachsynthese. '
                'Sprich diesen Text genau: „Bei Sonnenaufgang erwachte der Hafen langsam.“'
            ),
        )

        for prompt in inline_prompts:
            with self.subTest(prompt=prompt):
                messages = [
                    {'role': 'system', 'content': 'Ollmo runtime policy.'},
                    {'role': 'user', 'content': 'Draft an unrelated sentence.'},
                    {'role': 'assistant', 'content': 'This is stale thread text.'},
                    {'role': 'user', 'content': prompt},
                ]
                strategy = self.owner.choose_context_strategy(
                    instance={},
                    messages=messages,
                    prompt=prompt,
                    has_file_context=False,
                )
                prepared = self.owner.apply_context_strategy(messages, strategy)

                self.assertEqual(strategy['mode'], 'current_turn_only')
                self.assertEqual(
                    [item['content'] for item in prepared],
                    ['Ollmo runtime policy.', prompt],
                )

        ungrounded_prompt = 'Read this text aloud.'
        ungrounded_messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {'role': 'user', 'content': 'Draft a short status update.'},
            {'role': 'assistant', 'content': 'The rollout is complete.'},
            {'role': 'user', 'content': ungrounded_prompt},
        ]
        ungrounded_strategy = self.owner.choose_context_strategy(
            instance={},
            messages=ungrounded_messages,
            prompt=ungrounded_prompt,
            has_file_context=False,
        )

        self.assertEqual(ungrounded_strategy['mode'], 'recent_history')

    def test_referential_strategy_keeps_recent_history(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {'role': 'user', 'content': 'Draft a short answer about graphs.'},
            {'role': 'assistant', 'content': 'Graphs connect obligations.'},
            {'role': 'user', 'content': 'Continue the previous answer and make it shorter.'},
        ]

        strategy = self.owner.choose_context_strategy(
            instance={},
            messages=messages,
            prompt='Continue the previous answer and make it shorter.',
            has_file_context=False,
        )
        prepared = self.owner.apply_context_strategy(messages, strategy)

        self.assertEqual(strategy['mode'], 'recent_history')
        self.assertEqual(
            [
                item['status']
                for item in strategy['context_candidates']
                if item.get('source_kind') == 'message'
            ],
            ['promoted', 'promoted'],
        )
        self.assertEqual(len(strategy['promoted_candidate_ids']), 2)
        scan_candidate = next(
            item for item in strategy['context_candidates']
            if item['candidate_id'] == 'history-scan-deeper-pool'
        )
        self.assertEqual(scan_candidate['status'], 'not_promoted')
        self.assertEqual(prepared, messages)

    def test_german_inflected_previous_turn_references_keep_recent_history(self):
        prompts = (
            'Die vorherige Antwort ist die Grundlage.',
            (
                'Beziehe dich ausdrücklich auf die Artefakte aus dem unmittelbar '
                'vorherigen Turn.'
            ),
            'In vorheriger Nachricht steht die Quelle.',
            'Vorheriges Ergebnis ist die Grundlage.',
            'Aus vorherigem Turn stammen die Eingaben.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                messages = [
                    {'role': 'system', 'content': 'Ollmo runtime policy.'},
                    {'role': 'user', 'content': 'Erzeuge die Ausgangsartefakte.'},
                    {'role': 'assistant', 'content': 'Die Ausgangsartefakte sind bereit.'},
                    {'role': 'user', 'content': prompt},
                ]

                strategy = self.owner.choose_context_strategy(
                    instance={},
                    messages=messages,
                    prompt=prompt,
                    has_file_context=False,
                )
                prepared = self.owner.apply_context_strategy(messages, strategy)

                self.assertEqual(strategy['mode'], 'recent_history')
                self.assertEqual(
                    strategy['context_gate_review']['recent_history_decision'],
                    'promoted',
                )
                self.assertEqual(
                    strategy['context_gate_review']['history_scan']['decision'],
                    'not_promoted',
                )
                self.assertFalse(
                    strategy['context_gate_review']['history_scan']['executed']
                )
                self.assertEqual(
                    [
                        item['status']
                        for item in strategy['context_candidates']
                        if item.get('source_kind') == 'message'
                    ],
                    ['promoted', 'promoted'],
                )
                self.assertEqual(prepared, messages)

    def test_german_vorher_word_stems_do_not_promote_history(self):
        prompts = (
            'Erstelle eine Vorhersage für das Wetter.',
            'Beschreibe eine vorhersehbare Folge.',
            'Vorhersehbarkeit ist ein Qualitätsmerkmal.',
            'Diese Definition gilt unabhängig.',
            'Das ist eine eigenständige Aufgabe.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                messages = [
                    {'role': 'system', 'content': 'Ollmo runtime policy.'},
                    {'role': 'user', 'content': 'Eine alte, unverbundene Frage.'},
                    {'role': 'assistant', 'content': 'Eine alte, unverbundene Antwort.'},
                    {'role': 'user', 'content': prompt},
                ]

                strategy = self.owner.choose_context_strategy(
                    instance={},
                    messages=messages,
                    prompt=prompt,
                    has_file_context=False,
                )
                prepared = self.owner.apply_context_strategy(messages, strategy)

                self.assertEqual(strategy['mode'], 'current_turn_only')
                self.assertEqual(
                    strategy['context_gate_review']['recent_history_decision'],
                    'not_promoted',
                )
                self.assertEqual(
                    strategy['context_gate_review']['history_scan']['decision'],
                    'not_promoted',
                )
                self.assertEqual(
                    [
                        item['status']
                        for item in strategy['context_candidates']
                        if item.get('source_kind') == 'message'
                    ],
                    ['not_promoted', 'not_promoted'],
                )
                self.assertEqual(
                    [item['content'] for item in prepared],
                    ['Ollmo runtime policy.', prompt],
                )

    def test_materialization_audit_prompt_keeps_thread_context(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {'role': 'user', 'content': 'Wurden die Bilder materialisiert? Prüfe.'},
        ]

        strategy = self.owner.choose_context_strategy(
            instance={},
            messages=messages,
            prompt='Wurden die Bilder materialisiert? Prüfe.',
            has_file_context=False,
        )

        self.assertEqual(strategy['mode'], 'recent_history')
        self.assertEqual(
            strategy['reason'],
            'recent chat history fits the current context budget',
        )

    def test_materialization_audit_long_unmatched_prompt_is_bounded(self):
        prompt = ('ordinary filler ' * 500) + 'artifacts'

        started_at = time.perf_counter()
        needs_readback = self.owner._prompt_needs_materialization_readback(prompt)
        elapsed = time.perf_counter() - started_at

        self.assertFalse(needs_readback)
        self.assertLess(elapsed, 0.25)

    def test_materialization_audit_recognizes_positive_term_pairs_in_either_order(self):
        prompts = (
            'Audit the artifacts now.',
            'The artifacts need an audit.',
            'Do the generated images exist?',
            'Sind die Bilder materialisiert und vorhanden?',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(
                    self.owner._prompt_needs_materialization_readback(prompt)
                )

    def test_materialization_audit_requires_both_term_classes_on_one_line(self):
        self.assertFalse(
            self.owner._prompt_needs_materialization_readback('artifacts\ncheck')
        )
        self.assertFalse(
            self.owner._prompt_needs_materialization_readback('audit\nimages')
        )
        self.assertTrue(
            self.owner._prompt_needs_materialization_readback(
                'unrelated first line\ncheck these artifacts'
            )
        )

    def test_materialization_audit_promotes_recent_durable_artifact_evidence(self):
        with TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir) / 'chat_history'
            write_chat_history(
                'artifact-audit-thread',
                [
                    {'role': 'user', 'content': 'Generate two images.'},
                    {
                        'role': 'assistant',
                        'content': 'Generated two images.',
                        'response_id': 'resp_images',
                        'artifacts': [
                            {
                                'type': 'image',
                                'path': '/tmp/image-a.png',
                                'artifact_ref': 'artifact:image_a',
                                'availability': 'available',
                            },
                            {
                                'type': 'image',
                                'path': '/tmp/image-b.png',
                                'artifact_ref': 'artifact:image_b',
                                'availability': 'available',
                            },
                        ],
                        'output_slots': [
                            {
                                'slot_id': 'output-phase-2',
                                'type': 'image',
                                'status': 'fulfilled',
                                'artifact_ref': 'artifact:image_a',
                            },
                            {
                                'slot_id': 'output-phase-3',
                                'type': 'image',
                                'status': 'fulfilled',
                                'artifact_ref': 'artifact:image_b',
                            },
                        ],
                    },
                ],
                history_dir=history_dir,
                model='gemma',
                backend='ollama',
                capability='chat',
            )
            messages = [
                {'role': 'system', 'content': 'Ollmo runtime policy.'},
                {'role': 'user', 'content': 'Wurden die Bilder in einer späteren Antwort materialisiert? Prüfe.'},
            ]

            strategy = self.owner.choose_context_strategy(
                instance={},
                messages=messages,
                prompt='Wurden die Bilder in einer späteren Antwort materialisiert? Prüfe.',
                has_file_context=False,
                conversation_id='artifact-audit-thread',
                history_dir=history_dir,
            )
            prepared = self.owner.apply_context_strategy(messages, strategy)

        durable_candidates = [
            item for item in strategy['context_candidates']
            if item.get('promotion_source') == 'durable_readback'
        ]
        self.assertEqual(len(durable_candidates), 1)
        self.assertIn('artifact:image_a', durable_candidates[0]['artifact_refs'])
        self.assertIn('artifact:image_b', durable_candidates[0]['artifact_refs'])
        injected = [
            item for item in prepared
            if str(item.get('content') or '').startswith('Ollmo promoted these prior-context matches')
        ]
        self.assertEqual(len(injected), 1)
        self.assertIn('artifact:image_a', injected[0]['content'])
        self.assertIn('fulfilled', injected[0]['content'])
        self.assertIn('bounded context, not hidden memory', injected[0]['content'])

    def test_current_turn_only_keeps_explicit_selected_reference(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {'role': 'user', 'content': 'Generate an image of a fox in a snowy forest.'},
            {
                'role': 'assistant',
                'content': 'Selected reference image: /tmp/fox.png',
                'selected_reference': True,
            },
            {'role': 'user', 'content': 'Explain request graphs conceptually.'},
        ]

        strategy = self.owner.choose_context_strategy(
            instance={},
            messages=messages,
            prompt='Explain request graphs conceptually.',
            has_file_context=False,
        )
        prepared = self.owner.apply_context_strategy(messages, strategy)

        self.assertEqual(strategy['mode'], 'current_turn_only')
        self.assertEqual(
            [
                item['status']
                for item in strategy['context_candidates']
                if item.get('source_kind') == 'message'
            ],
            ['not_promoted', 'promoted'],
        )
        selected_candidate = next(
            item for item in strategy['context_candidates']
            if item.get('promotion_target') == 'active_reference'
        )
        self.assertEqual(selected_candidate['status'], 'promoted')
        self.assertEqual(
            [item['content'] for item in prepared],
            [
                'Ollmo runtime policy.',
                'Selected reference image: /tmp/fox.png',
                'Explain request graphs conceptually.',
            ],
        )

    def test_current_turn_only_does_not_reuse_stale_selected_reference_context(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {
                'role': 'system',
                'content': (
                    'Selected prior message reference for this conversation turn. '
                    'Treat it as bounded reference context only; the current user message remains the live instruction.\n\n'
                    '[assistant]\nUse copper-and-gold plan.'
                ),
            },
            {'role': 'user', 'content': 'Explain request graphs conceptually.'},
        ]

        strategy = self.owner.choose_context_strategy(
            instance={},
            messages=messages,
            prompt='Explain request graphs conceptually.',
            has_file_context=False,
        )
        prepared = self.owner.apply_context_strategy(messages, strategy)

        self.assertEqual(strategy['mode'], 'current_turn_only')
        self.assertEqual(
            [item['content'] for item in prepared],
            ['Ollmo runtime policy.', 'Explain request graphs conceptually.'],
        )

    def test_compressed_history_omits_stale_selected_reference_context(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {
                'role': 'system',
                'content': (
                    'Selected prior message reference for this conversation turn. '
                    'Treat it as bounded reference context only; the current user message remains the live instruction.\n\n'
                    '[assistant]\nOld copper plan should not become hidden intent.'
                ),
            },
            {'role': 'user', 'content': 'Turn 1'},
            {'role': 'assistant', 'content': 'Answer 1'},
            {'role': 'user', 'content': 'Turn 2'},
            {'role': 'assistant', 'content': 'Answer 2'},
            {'role': 'user', 'content': 'Turn 3'},
            {'role': 'assistant', 'content': 'Answer 3'},
            {'role': 'user', 'content': 'Summarize the recent discussion.'},
        ]
        strategy = {'mode': 'compressed_history'}

        prepared = self.owner.apply_context_strategy(messages, strategy)

        summary = next(item for item in prepared if item.get('content', '').startswith('Conversation summary'))
        self.assertNotIn('Old copper plan should not become hidden intent', summary['content'])
        self.assertEqual(prepared[-1]['content'], 'Summarize the recent discussion.')

    def test_deep_history_scan_is_candidate_until_prompt_promotes_it(self):
        messages = [
            {'role': 'system', 'content': 'Ollmo runtime policy.'},
            {'role': 'user', 'content': 'We discussed several architecture options last month.'},
            {'role': 'assistant', 'content': 'I noted the options.'},
            {'role': 'user', 'content': 'Search the entire conversation history for that decision.'},
        ]

        strategy = self.owner.choose_context_strategy(
            instance={},
            messages=messages,
            prompt='Search the entire conversation history for that decision.',
            has_file_context=False,
        )

        candidates = {
            item['candidate_id']: item
            for item in strategy['context_candidates']
        }
        self.assertEqual(candidates['history-scan-deeper-pool']['status'], 'promoted')
        self.assertEqual(candidates['history-scan-deeper-pool']['promotion_target'], 'history_scan')
        self.assertEqual(candidates['history-scan-deeper-pool']['scan_status'], 'needed')
        self.assertEqual(
            candidates['history-scan-deeper-pool']['scan_targets'],
            ['chat_history', 'response_frame_ledger', 'artifact_registry'],
        )
        self.assertEqual(strategy['context_gate_review']['history_scan']['decision'], 'promoted')
        self.assertTrue(strategy['context_gate_review']['history_scan']['executed'])
        self.assertIn('history-scan-deeper-pool', strategy['promoted_candidate_ids'])

    def test_deep_history_scan_candidates_are_injected_only_after_promotion(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'chat_history'
            artifact_ledger = root / 'artifact_registry.jsonl'
            write_chat_history(
                'old-vision-thread',
                [
                    {'role': 'user', 'content': 'The music analogy uses the room between notes.'},
                    {'role': 'assistant', 'content': 'The room between notes is fluid IR before freeze.'},
                ],
                history_dir=history_dir,
                model='gemma',
                backend='ollama',
                capability='chat',
            )
            artifact_ledger.write_text(
                json.dumps(
                    {
                        'artifact_ref': 'artifact:music-ir-note',
                        'artifact': {
                            'type': 'document',
                            'path': '/tmp/music-ir-note.txt',
                            'artifact_ref': 'artifact:music-ir-note',
                        },
                        'metadata': {'summary': 'music analogy and room between notes'},
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )
            messages = [
                {'role': 'system', 'content': 'Ollmo runtime policy.'},
                {'role': 'user', 'content': 'Search the entire conversation history for the music analogy.'},
            ]

            strategy = self.owner.choose_context_strategy(
                instance={},
                messages=messages,
                prompt='Search the entire conversation history for the music analogy.',
                has_file_context=False,
                history_dir=history_dir,
                artifact_registry_ledger=artifact_ledger,
            )
            prepared = self.owner.apply_context_strategy(messages, strategy)

        injected = [
            item for item in prepared
            if str(item.get('content') or '').startswith('Ollmo promoted these prior-context matches')
        ]
        self.assertEqual(len(injected), 1)
        self.assertIn('room between notes', injected[0]['content'])
        self.assertIn('artifact:music-ir-note', injected[0]['content'])
        review = strategy['context_gate_review']
        self.assertEqual(review['history_scan']['status'], 'completed')
        self.assertGreaterEqual(review['history_scan']['matched']['chat_history'], 1)
        self.assertEqual(review['history_scan']['matched']['artifact_registry'], 1)
        self.assertGreaterEqual(review['history_scan']['promoted_candidate_count'], 2)
        promoted_scan_matches = [
            item for item in strategy['context_candidates']
            if item.get('promotion_source') == 'history_scan'
        ]
        self.assertGreaterEqual(len(promoted_scan_matches), 2)

    def test_deep_history_scan_with_no_matches_stays_audited_but_non_binding(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'chat_history'
            write_chat_history(
                'old-vision-thread',
                [
                    {'role': 'user', 'content': 'We discussed a moonlit castle motif.'},
                    {'role': 'assistant', 'content': 'The castle motif stayed fluid.'},
                ],
                history_dir=history_dir,
                model='gemma',
                backend='ollama',
                capability='chat',
            )
            messages = [
                {'role': 'system', 'content': 'Ollmo runtime policy.'},
                {'role': 'user', 'content': 'Search the entire conversation history for the quantum orchard semaphore.'},
            ]

            strategy = self.owner.choose_context_strategy(
                instance={},
                messages=messages,
                prompt='Search the entire conversation history for the quantum orchard semaphore.',
                has_file_context=False,
                history_dir=history_dir,
            )
            prepared = self.owner.apply_context_strategy(messages, strategy)

        self.assertNotIn(
            'Ollmo promoted these prior-context matches',
            '\n'.join(str(item.get('content') or '') for item in prepared),
        )
        self.assertEqual(strategy['context_gate_review']['history_scan']['decision'], 'promoted')
        self.assertEqual(strategy['context_gate_review']['history_scan']['status'], 'completed')
        self.assertEqual(strategy['context_gate_review']['history_scan']['matched_candidate_count'], 0)
        self.assertEqual(strategy['context_gate_review']['history_scan']['promoted_candidate_count'], 0)
        promoted_scan_matches = [
            item for item in strategy['context_candidates']
            if item.get('promotion_source') == 'history_scan'
        ]
        self.assertEqual(promoted_scan_matches, [])


class GhostRouteGraphConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.owner = GhostRouteRuntimeOwner(
            hooks={},
            wrapper_capability_aliases={},
            max_recent_messages=8,
        )

    def _draft_single_phase_chat_graph(self, prompt: str, prompt_analysis: dict[str, object]) -> dict[str, object]:
        graph = build_request_phase_graph(
            'hello',
            request_payload={
                'ghost_route': True,
                'prompt': 'hello',
            },
        )
        graph['prompt'] = prompt
        graph['prompt_intent'] = {
            'primary_capability': prompt_analysis.get('primary_capability'),
            'requests_audio_output': bool(prompt_analysis.get('requests_audio_output')),
            'requests_visual_output': bool(prompt_analysis.get('requests_visual_output')),
            'requested_audio_output_count': int(prompt_analysis.get('requested_audio_output_count') or 0),
            'counted_audio_output_obligation': bool(prompt_analysis.get('counted_audio_output_obligation')),
            'audio_output_count_exceeds_bound': bool(prompt_analysis.get('audio_output_count_exceeds_bound')),
            'requested_visual_output_count': int(prompt_analysis.get('requested_visual_output_count') or 0),
            'text_preparation_before_audio_output': bool(prompt_analysis.get('text_preparation_before_audio_output')),
            'text_preparation_before_visual_output': bool(prompt_analysis.get('text_preparation_before_visual_output')),
        }
        return graph

    def test_route_graph_consistency_enforces_missing_visual_follow_up_for_chat_route(self):
        prompt = 'Write a short harbor story, then show it to me as an image.'
        prompt_analysis = analyze_prompt_intent(prompt)
        draft_graph = self._draft_single_phase_chat_graph(prompt, prompt_analysis)

        result = self.owner._maybe_enforce_chat_route_graph_consistency(
            payload={
                'ghost_route': True,
                'prompt': prompt,
            },
            prompt=prompt,
            current_turn_prompt=prompt,
            prompt_analysis=prompt_analysis,
            draft_phase_graph=draft_graph,
            route_hint={
                'capability': 'chat',
                'confidence': 0.86,
                'reason': 'text preparation required before image output',
            },
        )

        self.assertEqual(result['phase_graph_source'], 'consistency_enforced')
        self.assertEqual(
            result['phase_graph']['current_phase_resolution'],
            'graph_resolved',
        )
        self.assertEqual(
            result['phase_graph']['downstream_capabilities'],
            ['image_generation'],
        )
        self.assertEqual(len(result['downstream_branches']), 1)
        self.assertEqual(result['downstream_branches'][0]['source'], 'ghost_route_graph_consistency_v1')
        self.assertEqual(result['diagnostics']['status'], 'accepted')
        self.assertEqual(result['diagnostics']['final_graph_source'], 'consistency_enforced')

    def test_route_graph_consistency_skips_explicit_deferal_prompt(self):
        prompt = 'First produce a draft brief only. Do not generate images yet.'
        prompt_analysis = analyze_prompt_intent(prompt)
        draft_graph = self._draft_single_phase_chat_graph(prompt, prompt_analysis)

        result = self.owner._maybe_enforce_chat_route_graph_consistency(
            payload={
                'ghost_route': True,
                'prompt': prompt,
            },
            prompt=prompt,
            current_turn_prompt=prompt,
            prompt_analysis=prompt_analysis,
            draft_phase_graph=draft_graph,
            route_hint={
                'capability': 'chat',
                'confidence': 0.96,
                'reason': 'explicit deferal suppresses current-turn materialization',
            },
        )

        self.assertEqual(result['phase_graph_source'], 'draft')
        self.assertEqual(result['phase_graph'], draft_graph)
        self.assertEqual(result['diagnostics']['status'], 'skipped')
        self.assertEqual(result['diagnostics']['reason'], 'explicit_defer_materialization')

    def test_route_graph_consistency_skips_stale_visual_hint_without_current_turn_visual_intent(self):
        prompt = 'Petra is a testament to human ingenuity, built with brilliant engineering and trade acumen.'
        prompt_analysis = analyze_prompt_intent(prompt)
        draft_graph = self._draft_single_phase_chat_graph(prompt, prompt_analysis)

        result = self.owner._maybe_enforce_chat_route_graph_consistency(
            payload={
                'ghost_route': True,
                'prompt': prompt,
            },
            prompt=prompt,
            current_turn_prompt=prompt,
            prompt_analysis=prompt_analysis,
            draft_phase_graph=draft_graph,
            route_hint={
                'capability': 'image_generation',
                'confidence': 0.58,
                'reason': 'current phase remains text-capable while downstream materialization phases depend on its output',
            },
        )

        self.assertEqual(result['phase_graph_source'], 'draft')
        self.assertEqual(result['phase_graph'], draft_graph)
        self.assertEqual(result['diagnostics']['status'], 'skipped')
        self.assertEqual(result['diagnostics']['reason'], 'no_downstream_obligation_signals')

    def test_chat_route_downstream_contract_preserves_counted_audio_cardinality(self):
        downstream = self.owner._derive_chat_route_downstream_contract(
            prompt_analysis={
                'requests_audio_output': True,
                'requested_audio_output_count': 2,
                'counted_audio_output_obligation': True,
                'audio_output_count_exceeds_bound': False,
            },
            route_hint={'capability': 'chat'},
        )

        self.assertEqual(
            downstream,
            [{'capability': 'text_to_speech', 'count': 2}],
        )

    def test_chat_route_downstream_contract_does_not_promote_audio_overflow(self):
        downstream = self.owner._derive_chat_route_downstream_contract(
            prompt_analysis={
                'requests_audio_output': True,
                'requested_audio_output_count': 0,
                'requested_audio_output_count_raw': 99,
                'audio_output_count_exceeds_bound': True,
            },
            route_hint={'capability': 'chat'},
        )

        self.assertEqual(downstream, [])

    def test_route_graph_consistency_skips_chat_graph_that_already_has_downstream(self):
        prompt = 'Generate two images of a harbor at sunrise.'
        prompt_analysis = analyze_prompt_intent(prompt)
        draft_graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'batch_prompts': ['harbor dawn wide shot', 'harbor dawn close shot'],
                'downstream_branches': [
                    {
                        'branch_id': 'phase-image-1',
                        'phase_id': 'phase-image-1',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'depends_on': ['phase-1'],
                        'prompt': 'harbor dawn wide shot',
                    },
                    {
                        'branch_id': 'phase-image-2',
                        'phase_id': 'phase-image-2',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'depends_on': ['phase-1'],
                        'prompt': 'harbor dawn close shot',
                    },
                ],
            },
        )

        result = self.owner._maybe_enforce_chat_route_graph_consistency(
            payload={
                'ghost_route': True,
                'prompt': prompt,
                'batch_prompts': ['harbor dawn wide shot', 'harbor dawn close shot'],
            },
            prompt=prompt,
            current_turn_prompt=prompt,
            prompt_analysis=prompt_analysis,
            draft_phase_graph=draft_graph,
            route_hint={
                'capability': 'image_generation',
                'confidence': 0.9,
                'reason': 'image-generation cue',
            },
        )

        self.assertEqual(result['phase_graph_source'], 'draft')
        self.assertEqual(result['phase_graph'], draft_graph)
        self.assertEqual(result['diagnostics']['status'], 'skipped')
        self.assertEqual(result['diagnostics']['reason'], 'draft_already_has_downstream')

    def test_downstream_contract_review_accepts_richer_repeated_branch_surface(self):
        prompt = 'Prepare two narrations, inspect an image, and summarize the result.'
        prompt_analysis = analyze_prompt_intent(prompt)
        draft_graph = self._draft_single_phase_chat_graph(prompt, prompt_analysis)

        result = self.owner._review_phase_graph_downstream_contract(
            prompt=prompt,
            current_turn_prompt=prompt,
            payload={
                'ghost_route': True,
                'prompt': prompt,
            },
            draft_phase_graph=draft_graph,
            downstream=[
                {'capability': 'text_to_speech', 'count': 2, 'review_criteria': ['audio_exists']},
                {'capability': 'vision_analysis', 'count': 1, 'depends_on': ['phase-2']},
                {'capability': 'chat', 'count': 1, 'depends_on': ['phase-3', 'phase-4']},
            ],
            branch_source='test_contract_review',
        )

        self.assertTrue(result['accepted'])
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(
            [item['capability'] for item in result['downstream_branches']],
            ['text_to_speech', 'text_to_speech', 'vision_analysis', 'chat'],
        )
        self.assertEqual(result['downstream_branches'][0]['output_type'], 'audio')
        self.assertEqual(result['downstream_branches'][2]['output_type'], 'text')
        self.assertEqual(result['downstream_branches'][3]['depends_on'], ['phase-3', 'phase-4'])
        self.assertEqual(
            result['candidate_phase_graph']['downstream_capabilities'],
            ['text_to_speech', 'vision_analysis', 'chat'],
        )

    def test_merge_request_meta_runtime_truth_preserves_route_workload_task_proposals(self):
        owner = GhostRouteRuntimeOwner(
            hooks={
                'timeout_ms_to_seconds': lambda value: None,
                'extract_responses_prompt': lambda payload: payload.get('prompt'),
                'extract_responses_current_turn_prompt': lambda payload: payload.get('prompt'),
            },
            wrapper_capability_aliases={},
            max_recent_messages=8,
        )
        prompt = (
            'Create a short slogan for Ollmo, generate an audio version, generate a poster image '
            'for the same slogan, then write one final sentence that references both generated artifacts.'
        )

        runtime = owner.merge_request_meta_runtime_truth(
            {},
            {'ghost_route': True, 'prompt': prompt},
            route_payload={
                'route_source': 'ghost_carried',
                'workload_task_proposals': [
                    {
                        'proposal_id': 'proposal-final',
                        'phase_id': 'phase-5',
                        'capability': 'chat',
                        'depends_on': ['phase-2', 'phase-4'],
                        'semantic_intent': 'Write the final sentence from generated audio and vision evidence.',
                    }
                ],
            },
        )

        graph = runtime['request_phase_graph']
        final_task = graph['workload_graph']['tasks'][-1]
        self.assertEqual(graph['workload_proposal_review']['status'], 'accepted')
        self.assertEqual(final_task['semantic_intent'], 'Write the final sentence from generated audio and vision evidence.')


if __name__ == '__main__':
    unittest.main()
