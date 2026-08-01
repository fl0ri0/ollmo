import hashlib
import json
import unittest
import tempfile
from pathlib import Path

from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_g.request_meta import extract_request_meta
from helpers.model_capabilities import normalize_capability
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_server.response_semantics_runtime import (
    ResponseSemanticsRuntimeOwner,
    _request_requires_current_source_for_transform,
    _route_phase_graph_has_artifact_consumer_edge,
    classify_phase_output_text,
    phase_output_acceptance_metadata,
)
from ollmo_services.responses import build_canonical_response_artifacts
from ollmo_services.tts_audio_integrity import TTS_AUDIO_INTEGRITY_POLICY_ID


def _normalize_capability_list(values):
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        token = str(value or '').strip().lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


class ResponseSemanticsRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.owner = ResponseSemanticsRuntimeOwner(
            hooks={
                'build_canonical_response_artifacts': build_canonical_response_artifacts,
                'normalize_capability_list': _normalize_capability_list,
                'extract_request_meta': extract_request_meta,
                'extract_responses_prompt': lambda payload: str(
                    (payload or {}).get('prompt') or (payload or {}).get('input') or ''
                ),
                'load_running_instances': lambda: [],
                'merge_instances_with_runtime_status': lambda instances, **kwargs: instances,
            }
        )
        self.late_fill_owner = object.__new__(LateFillRuntimeOwner)
        self.late_fill_owner.capability_image_generation = 'image_generation'
        self.late_fill_owner.capability_text_to_speech = 'text_to_speech'
        self.late_fill_owner.normalize_capability = normalize_capability
        self.late_fill_owner.branch_id = lambda branch: str(
            branch.get('branch_id') or branch.get('phase_id') or ''
        ).strip()
        self.late_fill_owner.branch_capability = lambda branch: str(branch.get('capability') or '').strip().lower() or None
        self.late_fill_owner.build_canonical_response_artifacts = build_canonical_response_artifacts

    @staticmethod
    def _structured_dependency_join_fixture(
        terminal_result,
        *,
        prompt=None,
        fenced=False,
    ):
        prompt = prompt or (
            'Erzeuge genau drei unterschiedliche lokale Bilder: '
            'A – ein roter Leuchtturm im Schnee bei Tag; '
            'B – eine grüne Bibliothek bei Nacht; '
            'C – ein blaues Gewächshaus im Regen. '
            'Analysiere danach jedes tatsächlich erzeugte Bild separat. '
            'Gib abschließend im Chat genau eine JSON-Liste mit den Feldern '
            'label, artifact_ref und visible_evidence für A, B und C aus. '
            'Ordne jede visible_evidence ausschließlich dem artifact_ref '
            'des gleich bezeichneten Bildes zu.'
        )
        image_paths = {
            'A': '/tmp/structured-join-a.png',
            'B': '/tmp/structured-join-b.png',
            'C': '/tmp/structured-join-c.png',
        }
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'phase_chain',
            'prompt': prompt,
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'intent_obligations': [
                {
                    'obligation_id': 'intent-images',
                    'kind': 'media_artifact',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                },
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_preparation',
                    'status': 'completed',
                },
                *[
                    {
                        'phase_id': f'phase-{index + 1}',
                        'branch_id': f'branch-image_generation-{index}',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'role': 'image_generation_follow_up',
                        'depends_on': ['phase-1'],
                        'queue_index': index,
                        'status': 'completed',
                    }
                    for index in range(1, 4)
                ],
                *[
                    {
                        'phase_id': f'phase-{index + 4}',
                        'branch_id': f'branch-vision_analysis-{index}',
                        'capability': 'vision_analysis',
                        'output_type': 'text',
                        'role': 'vision_analysis_follow_up',
                        'depends_on': [f'phase-{index + 1}'],
                        'queue_index': index,
                        'status': 'completed',
                    }
                    for index in range(1, 4)
                ],
                {
                    'phase_id': 'phase-8',
                    'branch_id': 'branch-chat-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'post_artifact_text_follow_up',
                    'depends_on': ['phase-5', 'phase-6', 'phase-7'],
                    'stage_direction': 'write_text_after_artifact_generation',
                    'status': 'completed',
                },
            ],
        }
        correct_rows = [
            {
                'label': label,
                'artifact_ref': image_paths[label],
                'visible_evidence': f'Visible evidence for {label}.',
            }
            for label in ('A', 'B', 'C')
        ]
        if isinstance(terminal_result, str):
            terminal_text = terminal_result
        else:
            terminal_text = json.dumps(terminal_result, ensure_ascii=False, indent=2)
        if fenced:
            terminal_text = f'```json\n{terminal_text}\n```'
        completed_branches = [
            {
                'phase_id': phase['phase_id'],
                'branch_id': phase.get('branch_id', phase['phase_id']),
                'capability': phase['capability'],
                'status': 'fulfilled',
            }
            for phase in phase_graph['phases'][1:]
        ]
        fill_results = [
            {
                'phase_id': f'phase-{index + 1}',
                'branch_id': f'branch-image_generation-{index}',
                'capability': 'image_generation',
                'saved_image_path': image_paths[label],
            }
            for index, label in enumerate(('A', 'B', 'C'), start=1)
        ]
        fill_results.extend(
            {
                'phase_id': f'phase-{index + 4}',
                'branch_id': f'branch-vision_analysis-{index}',
                'capability': 'vision_analysis',
                'result_text': json.dumps([correct_rows[index - 1]], ensure_ascii=False),
            }
            for index in range(1, 4)
        )
        fill_results.append(
            {
                'phase_id': 'phase-8',
                'branch_id': 'branch-chat-1',
                'capability': 'chat',
                'result_text': terminal_text,
                'execution_contract': {
                    'phase_id': 'phase-8',
                    'branch_id': 'branch-chat-1',
                    'capability': 'chat',
                    'role': 'post_artifact_text_follow_up',
                    'depends_on': ['phase-5', 'phase-6', 'phase-7'],
                    'stage_direction': 'write_text_after_artifact_generation',
                    'content_payload_source': (
                        'late_fill_results:branch-vision_analysis-1,'
                        'branch-vision_analysis-2,branch-vision_analysis-3'
                    ),
                },
            }
        )
        payload = {
            'id': 'resp-structured-dependency-join-test',
            # Deliberately correct: the guard must inspect the terminal fill result,
            # not accept this public/root surface as evidence for the terminal branch.
            'output_text': json.dumps(correct_rows, ensure_ascii=False, indent=2),
            'runtime': {'request_phase_graph': phase_graph},
            'artifacts': [
                {
                    'type': 'image',
                    'path': image_paths[label],
                    'artifact_ref': f'artifact:image-{label.lower()}',
                    'phase_id': f'phase-{index + 1}',
                    'branch_id': f'branch-image_generation-{index}',
                }
                for index, label in enumerate(('A', 'B', 'C'), start=1)
            ],
            'late_fill': {
                'status': 'completed',
                'completed_capabilities': ['image_generation', 'vision_analysis', 'chat'],
                'completed_branches': completed_branches,
                'pending_branches': [],
                'failed_branches': [],
                'fill_results': fill_results,
            },
        }
        return prompt, phase_graph, payload, image_paths, correct_rows

    @staticmethod
    def _mixed_structured_dependency_join_fixture(terminal_result=None):
        prompt = (
            'Erzeuge zwei reale lokale Artefakte: '
            'A – ein Bild eines Leuchtturms im Sturm; '
            'B – ein Audio des Satzes „Leuchtturm im Sturm“. '
            'Analysiere A und transkribiere B separat. '
            'Gib abschließend im Chat genau eine JSON-Liste mit den Feldern '
            'label, artifact_ref und evidence für A und B aus.'
        )
        image_path = '/tmp/mixed-structured-join-a.png'
        audio_path = '/tmp/mixed-structured-join-b.wav'
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'phase_chain',
            'prompt': prompt,
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'intent_obligations': [
                {
                    'obligation_id': 'intent-mixed-image',
                    'kind': 'media_artifact',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                },
                {
                    'obligation_id': 'intent-mixed-audio',
                    'kind': 'media_artifact',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'required': True,
                },
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_preparation',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'queue_index': 1,
                    'depends_on': ['phase-1'],
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-text_to_speech-1',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'queue_index': 1,
                    'depends_on': ['phase-1'],
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-4',
                    'branch_id': 'branch-vision_analysis-1',
                    'capability': 'vision_analysis',
                    'output_type': 'text',
                    'queue_index': 1,
                    'depends_on': ['phase-2'],
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-5',
                    'branch_id': 'branch-speech_to_text-1',
                    'capability': 'speech_to_text',
                    'output_type': 'text',
                    'queue_index': 1,
                    'depends_on': ['phase-3'],
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-6',
                    'branch_id': 'branch-chat-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'post_artifact_text_follow_up',
                    'stage_direction': 'write_text_after_artifact_generation',
                    'depends_on': ['phase-4', 'phase-5'],
                    'status': 'completed',
                },
            ],
        }
        correct_rows = [
            {
                'label': 'A',
                'artifact_ref': 'artifact:mixed-image-a',
                'evidence': 'A lighthouse is visible in storm conditions.',
            },
            {
                'label': 'B',
                'artifact_ref': 'artifact:mixed-audio-b',
                'evidence': 'Transcript: Leuchtturm im Sturm.',
            },
        ]
        result_rows = correct_rows if terminal_result is None else terminal_result
        terminal_text = (
            result_rows
            if isinstance(result_rows, str)
            else json.dumps(result_rows, ensure_ascii=False)
        )
        completed_branches = [
            {
                'phase_id': phase['phase_id'],
                'branch_id': phase.get('branch_id', phase['phase_id']),
                'capability': phase['capability'],
                'status': 'fulfilled',
            }
            for phase in phase_graph['phases'][1:]
        ]
        fill_results = [
            {
                'phase_id': 'phase-2',
                'branch_id': 'branch-image_generation-1',
                'capability': 'image_generation',
                'saved_image_path': image_path,
            },
            {
                'phase_id': 'phase-3',
                'branch_id': 'branch-text_to_speech-1',
                'capability': 'text_to_speech',
                'saved_audio_path': audio_path,
                'tts_audio_integrity_evidence': (
                    ResponseSemanticsRuntimeTests._tts_integrity_evidence(
                        source_text='Leuchtturm im Sturm.',
                        path=audio_path,
                    )
                ),
            },
            {
                'phase_id': 'phase-4',
                'branch_id': 'branch-vision_analysis-1',
                'capability': 'vision_analysis',
                'result_text': 'A lighthouse is visible in storm conditions.',
            },
            {
                'phase_id': 'phase-5',
                'branch_id': 'branch-speech_to_text-1',
                'capability': 'speech_to_text',
                'result_text': 'Leuchtturm im Sturm.',
            },
            {
                'phase_id': 'phase-6',
                'branch_id': 'branch-chat-1',
                'capability': 'chat',
                'result_text': terminal_text,
                'execution_contract': {
                    'phase_id': 'phase-6',
                    'branch_id': 'branch-chat-1',
                    'capability': 'chat',
                    'role': 'post_artifact_text_follow_up',
                    'stage_direction': 'write_text_after_artifact_generation',
                    'depends_on': ['phase-4', 'phase-5'],
                    'content_payload_source': (
                        'late_fill_results:branch-vision_analysis-1,'
                        'branch-speech_to_text-1'
                    ),
                },
            },
        ]
        payload = {
            'id': 'resp-mixed-structured-dependency-join-test',
            'output_text': terminal_text,
            'runtime': {'request_phase_graph': phase_graph},
            'artifacts': [
                {
                    'type': 'image',
                    'path': image_path,
                    'artifact_ref': 'artifact:mixed-image-a',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                },
                {
                    'type': 'audio',
                    'path': audio_path,
                    'artifact_ref': 'artifact:mixed-audio-b',
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-text_to_speech-1',
                },
            ],
            'late_fill': {
                'status': 'completed',
                'completed_capabilities': [
                    'image_generation',
                    'text_to_speech',
                    'vision_analysis',
                    'speech_to_text',
                    'chat',
                ],
                'completed_branches': completed_branches,
                'pending_branches': [],
                'failed_branches': [],
                'fill_results': fill_results,
            },
        }
        return prompt, phase_graph, payload, correct_rows

    @staticmethod
    def _preserved_visual_object_join_fixture(terminal_result=None):
        prompt = (
            'Beziehe dich auf das Observatorium-Bild und seine Bildanalyse. Bewahre Bild und '
            'Bildanalyse unverändert; erzeuge das Bild nicht neu und analysiere es nicht erneut. '
            'Erzeuge zwei getrennte Audiofassungen: einmal die ursprüngliche deutsche Erzählung '
            'und einmal eine getreue englische Übersetzung. Transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein neues JSON-Objekt aus, das die unveränderte '
            'Bildevidenz sowie beide Audio-artifact_refs und beide realen Transkripte eindeutig verbindet.'
        )
        references = [
            {
                'type': 'message',
                'message_id': 'msg-observatory',
                'source_response_id': 'resp-observatory',
                'content': (
                    '**Sichtbare Details:**\n'
                    'Die erhaltene Kuppel steht auf einem dunklen Hügel unter der Milchstraße.\n\n'
                    '**Deutsche Erzählung:**\nDie stille Kuppel blickte in die Nacht.'
                ),
            },
            {
                'type': 'image',
                'kind': 'image',
                'artifact_ref': 'artifact:image-observatory',
                'path': '/tmp/observatory.png',
                'source_message_id': 'msg-observatory',
                'source_response_id': 'resp-observatory',
            },
        ]
        request = {
            'prompt': prompt,
            'ghost_route': True,
            'reference_artifacts': references,
        }
        graph = build_request_phase_graph(
            prompt,
            request_payload=request,
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        branches = graph.get('downstream_branches') or []
        tts = [branch for branch in branches if branch.get('capability') == 'text_to_speech']
        stt = [branch for branch in branches if branch.get('capability') == 'speech_to_text']
        final_branch = next(
            branch
            for branch in branches
            if branch.get('dependency_contract') == 'structured_multi_evidence_join'
        )
        valid_object = {
            'preserved_visual_evidence': (
                'Die erhaltene Kuppel steht auf einem dunklen Hügel unter der Milchstraße.'
            ),
            'audio_variant_1_artifact_ref': 'artifact:audio-1',
            'audio_variant_1_transcript': 'Die stille Kuppel blickte in die Nacht.',
            'audio_variant_2_artifact_ref': 'artifact:audio-2',
            'audio_variant_2_transcript': 'The silent dome gazed into the night.',
        }
        object_result = valid_object if terminal_result is None else terminal_result
        terminal_text = (
            object_result
            if isinstance(object_result, str)
            else json.dumps(object_result, ensure_ascii=False)
        )
        fill_results = []
        artifacts = []
        for index, (tts_branch, stt_branch) in enumerate(zip(tts, stt), start=1):
            source_text = valid_object[
                f'audio_variant_{index}_transcript'
            ]
            audio = {
                'type': 'audio',
                'kind': 'audio',
                'path': f'/tmp/audio-{index}.wav',
                'artifact_ref': f'artifact:audio-{index}',
                'phase_id': tts_branch['phase_id'],
                'branch_id': tts_branch['branch_id'],
            }
            artifacts.append(audio)
            fill_results.extend(
                [
                    {
                        'phase_id': tts_branch['phase_id'],
                        'branch_id': tts_branch['branch_id'],
                        'capability': 'text_to_speech',
                        'saved_audio_path': audio['path'],
                        'artifacts': [audio],
                        'tts_audio_integrity_evidence': (
                            ResponseSemanticsRuntimeTests._tts_integrity_evidence(
                                source_text=source_text,
                                path=audio['path'],
                            )
                        ),
                    },
                    {
                        'phase_id': stt_branch['phase_id'],
                        'branch_id': stt_branch['branch_id'],
                        'capability': 'speech_to_text',
                        'result_text': valid_object[
                            f'audio_variant_{index}_transcript'
                        ],
                    },
                ]
            )
        fill_results.append(
            {
                'phase_id': final_branch['phase_id'],
                'branch_id': final_branch['branch_id'],
                'capability': 'chat',
                'result_text': terminal_text,
                'execution_contract': dict(final_branch),
            }
        )
        completed_branches = [
            {
                'phase_id': branch['phase_id'],
                'branch_id': branch['branch_id'],
                'capability': branch['capability'],
                'status': 'fulfilled',
            }
            for branch in branches
        ]
        payload = {
            'id': 'resp-preserved-visual-object-join',
            'output_text': terminal_text,
            'request': request,
            'runtime': {'request_phase_graph': graph},
            'artifacts': artifacts,
            'late_fill': {
                'status': 'completed',
                'completed_branches': completed_branches,
                'pending_branches': [],
                'failed_branches': [],
                'fill_results': fill_results,
            },
        }
        return request, graph, payload, valid_object

    def test_terminal_graph_rebase_callback_receives_final_payload_and_returns_reviewed_copy(self):
        calls = []

        def review(payload, *, request_payload, route_payload):
            calls.append((payload, request_payload, route_payload))
            return {**payload, 'terminal_graph_rebase_reviewed': True}

        self.late_fill_owner.review_terminal_graph_rebase = review
        payload = {'response_id': 'resp-terminal-rebase'}
        updated = self.late_fill_owner._review_terminal_graph_rebase_if_available(
            payload,
            request_payload={'prompt': 'review final graph'},
            route_payload={'route_source': 'ghost_carried'},
        )

        self.assertTrue(updated['terminal_graph_rebase_reviewed'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], payload)
        self.assertEqual(calls[0][1]['prompt'], 'review final graph')

    def test_truth_gate_annotates_soft_materialized_artifact_claims_as_text_only(self):
        payload = {
            'id': 'resp_text_only_artifact_claim',
            'output_text': (
                '### Materialisierte Artefakte\n\n'
                'Artefakt 1: Narratives Element -> fulfilled\n'
                'Artefakt 2: Codeblock -> fulfilled\n'
            ),
        }

        updated = self.owner.truth_gate_response_output_claims(payload)

        self.assertIn('Runtime truth: no separate local or downloadable artifacts', updated['output_text'])
        self.assertIn('Artefakt 1: Narratives Element', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'annotated')

    def test_truth_gate_leaves_normal_text_fulfillment_alone(self):
        payload = {
            'id': 'resp_normal_text',
            'output_text': 'Hier ist die gewünschte Checkliste mit fünf Punkten.',
        }

        updated = self.owner.truth_gate_response_output_claims(payload)

        self.assertEqual(updated, payload)

    def test_preserved_visual_object_join_requires_exact_object_and_runtime_bindings(self):
        request, graph, payload, valid_object = self._preserved_visual_object_join_fixture()
        checks = self.owner._structured_dependency_join_checks(
            request_payload=request,
            artifact_payload=payload,
            request_phase_graph=graph,
        )
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].get('status'), 'fulfilled')
        self.assertEqual(
            checks[0].get('evidence'),
            'structured_object_contract_satisfied',
        )

        invalid_cases = [
            (
                'array_instead_of_object',
                [valid_object],
                'json_object_required',
            ),
            (
                'missing_binding',
                {
                    key: value
                    for key, value in valid_object.items()
                    if key != 'audio_variant_2_transcript'
                },
                'missing_required_field',
            ),
            (
                'wrong_artifact_ref',
                {
                    **valid_object,
                    'audio_variant_2_artifact_ref': 'artifact:audio-1',
                },
                'artifact_ref_binding_mismatch',
            ),
            (
                'wrong_transcript',
                {
                    **valid_object,
                    'audio_variant_1_transcript': 'Invented transcript.',
                },
                'transcript_binding_mismatch',
            ),
            (
                'invented_visual_evidence',
                {
                    **valid_object,
                    'preserved_visual_evidence': 'A different image description.',
                },
                'preserved_visual_evidence_binding_mismatch',
            ),
            (
                'same_message_narration_is_not_visual_evidence',
                {
                    **valid_object,
                    'preserved_visual_evidence': (
                        'Die stille Kuppel blickte in die Nacht.'
                    ),
                },
                'preserved_visual_evidence_binding_mismatch',
            ),
        ]
        for label, terminal_result, expected_issue in invalid_cases:
            with self.subTest(label=label):
                request, graph, payload, _valid = (
                    self._preserved_visual_object_join_fixture(terminal_result)
                )
                checks = self.owner._structured_dependency_join_checks(
                    request_payload=request,
                    artifact_payload=payload,
                    request_phase_graph=graph,
                )
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0].get('status'), 'pending')
                self.assertIn(expected_issue, checks[0].get('issue_codes') or [])
                self.assertEqual(
                    checks[0].get('repair_action'),
                    'repair_branch_contract',
                )

        request, _graph, payload, _valid = (
            self._preserved_visual_object_join_fixture([valid_object])
        )
        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload=request,
            artifact_payload=payload,
        )
        integrated_check = next(
            item
            for item in review.get('checks') or []
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review.get('status'), 'pending')
        self.assertEqual(integrated_check.get('status'), 'pending')
        self.assertIn(
            'json_object_required',
            integrated_check.get('issue_codes') or [],
        )

    def test_simple_chat_does_not_mark_semantic_review_pending(self):
        prompt = 'Sag kurz hallo.'
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            response_payload={'output_text': 'Hallo.'},
        )

        decision_contract = phase_graph['decision_contract']
        review = self.owner.build_graph_closure_review(
            'Hallo.',
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload={
                'output_text': 'Hallo.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        self.assertEqual(decision_contract.get('semantic_review_candidates', []), [])
        self.assertEqual(decision_contract['semantic_quality_review']['status'], 'not_required')
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(review['semantic_review_required_count'], 0)
        self.assertNotIn('semantic_review_status', review)
        self.assertEqual(
            review['surface_state']['category_counts']['semantic_review_pending'],
            0,
        )

    def test_structured_dependency_join_rejects_d3r2_duplicate_labels_and_refs(self):
        malformed_rows = [
            {
                'label': 'A',
                'artifact_ref': '/tmp/structured-join-a.png',
                'visible_evidence': 'Visible evidence for A.',
            },
            {
                'label': 'A',
                'artifact_ref': '/tmp/structured-join-b.png',
                'visible_evidence': 'B evidence incorrectly labeled A.',
            },
            {
                'label': 'B',
                'artifact_ref': '/tmp/structured-join-b.png',
                'visible_evidence': 'Visible evidence for B.',
            },
            {
                'label': 'C',
                'artifact_ref': '/tmp/structured-join-b.png',
                'visible_evidence': 'B evidence incorrectly labeled C.',
            },
            {
                'label': 'C',
                'artifact_ref': '/tmp/structured-join-c.png',
                'visible_evidence': 'Visible evidence for C.',
            },
        ]
        prompt, _graph, payload, _paths, _correct_rows = self._structured_dependency_join_fixture(
            malformed_rows
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(check['status'], 'pending')
        self.assertEqual(check['branch_id'], 'branch-chat-1')
        self.assertEqual(check['phase_id'], 'phase-8')
        self.assertEqual(check['role'], 'post_artifact_text_follow_up')
        self.assertEqual(check['depends_on'], ['phase-5', 'phase-6', 'phase-7'])
        self.assertEqual(check['repair_action'], 'repair_branch_contract')
        self.assertTrue(
            {
                'exact_count_mismatch',
                'duplicate_label',
                'duplicate_artifact_ref',
                'label_ref_dependency_mismatch',
            }.issubset(set(check['issue_codes']))
        )

    def test_structured_dependency_join_accepts_exact_fenced_abc_result(self):
        prompt, _graph, payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        payload['late_fill']['fill_results'][-1]['result_text'] = (
            f'```json\n{json.dumps(correct_rows, ensure_ascii=False, indent=2)}\n```'
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(check['status'], 'fulfilled')
        self.assertEqual(check['expected_labels'], ['A', 'B', 'C'])
        self.assertEqual(
            check['required_fields'],
            ['label', 'artifact_ref', 'visible_evidence'],
        )
        self.assertNotIn('issue_codes', check)

    def test_structured_dependency_join_accepts_mixed_vision_and_stt_lineages(self):
        prompt, graph, payload, _correct_rows = (
            self._mixed_structured_dependency_join_fixture()
        )
        reordered_dependencies = ['phase-5', 'phase-4']
        graph['phases'][-1]['depends_on'] = reordered_dependencies
        payload['late_fill']['fill_results'][-1]['execution_contract'][
            'depends_on'
        ] = reordered_dependencies

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(check['status'], 'fulfilled')
        self.assertEqual(check['expected_labels'], ['A', 'B'])
        self.assertNotIn('issue_codes', check)

    def test_structured_dependency_join_rejects_mixed_cross_modal_identity_defects(self):
        cases = ('swapped_refs', 'wrong_stt_producer')
        for case_name in cases:
            with self.subTest(case=case_name):
                prompt, graph, payload, correct_rows = (
                    self._mixed_structured_dependency_join_fixture()
                )
                expected_issue = 'label_ref_dependency_mismatch'
                if case_name == 'swapped_refs':
                    malformed_rows = [dict(row) for row in correct_rows]
                    malformed_rows[0]['artifact_ref'], malformed_rows[1]['artifact_ref'] = (
                        malformed_rows[1]['artifact_ref'],
                        malformed_rows[0]['artifact_ref'],
                    )
                    payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(
                        malformed_rows,
                        ensure_ascii=False,
                    )
                    payload['output_text'] = payload['late_fill']['fill_results'][-1]['result_text']
                else:
                    graph['phases'][4]['depends_on'] = ['phase-2']
                    expected_issue = 'dependency_lineage_unresolved'

                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )
                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                self.assertEqual(review['status'], 'pending')
                self.assertEqual(check['status'], 'pending')
                self.assertIn(expected_issue, check['issue_codes'])

    def test_structured_dependency_join_supports_direct_media_producers_but_not_unknown_dependencies(self):
        cases = ('direct_image_and_audio', 'unknown_chat_dependency')
        for case_name in cases:
            with self.subTest(case=case_name):
                prompt, graph, payload, _correct_rows = (
                    self._mixed_structured_dependency_join_fixture()
                )
                terminal_dependencies = (
                    ['phase-2', 'phase-3']
                    if case_name == 'direct_image_and_audio'
                    else ['phase-2', 'phase-1']
                )
                graph['phases'][-1]['depends_on'] = terminal_dependencies
                payload['late_fill']['fill_results'][-1]['execution_contract'][
                    'depends_on'
                ] = terminal_dependencies

                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )
                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                if case_name == 'direct_image_and_audio':
                    self.assertEqual(review['status'], 'fulfilled')
                    self.assertEqual(check['status'], 'fulfilled')
                    self.assertNotIn('issue_codes', check)
                else:
                    self.assertEqual(review['status'], 'pending')
                    self.assertEqual(check['status'], 'pending')
                    self.assertIn('dependency_lineage_unresolved', check['issue_codes'])

    def test_structured_dependency_join_accepts_and_enforces_lowercase_and_numeric_labels(self):
        cases = (
            (
                'lowercase',
                ('a', 'b', 'c'),
                'Erzeuge drei lokale Bilder: a – Leuchtturm; b – Bibliothek; '
                'c – Gewächshaus. Analysiere jedes Bild separat. Gib im Chat genau '
                'eine JSON-Liste mit den Feldern label, artifact_ref und visible_evidence '
                'für a, b und c aus.',
            ),
            (
                'numeric',
                ('1', '2', '3'),
                'Erzeuge drei lokale Bilder: 1 – Leuchtturm; 2 – Bibliothek; '
                '3 – Gewächshaus. Analysiere jedes Bild separat. Gib im Chat genau '
                'eine JSON-Liste mit den Feldern label, artifact_ref und visible_evidence '
                'für 1, 2 und 3 aus.',
            ),
        )
        for case_name, labels, prompt in cases:
            with self.subTest(case=case_name, result='valid'):
                _prompt, _graph, payload, _paths, original_rows = (
                    self._structured_dependency_join_fixture([], prompt=prompt)
                )
                valid_rows = [dict(row) for row in original_rows]
                for row, label in zip(valid_rows, labels):
                    row['label'] = label
                terminal_text = json.dumps(valid_rows, ensure_ascii=False)
                payload['late_fill']['fill_results'][-1]['result_text'] = terminal_text
                payload['output_text'] = terminal_text

                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )
                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                self.assertEqual(review['status'], 'fulfilled')
                self.assertEqual(check['status'], 'fulfilled')
                self.assertEqual(check['expected_labels'], list(labels))

            with self.subTest(case=case_name, result='malformed'):
                _prompt, _graph, payload, _paths, original_rows = (
                    self._structured_dependency_join_fixture([], prompt=prompt)
                )
                malformed_rows = [dict(row) for row in original_rows]
                for row, label in zip(malformed_rows, labels):
                    row['label'] = label
                malformed_rows[1]['label'] = labels[0]
                malformed_rows[1]['artifact_ref'] = malformed_rows[0]['artifact_ref']
                terminal_text = json.dumps(malformed_rows, ensure_ascii=False)
                payload['late_fill']['fill_results'][-1]['result_text'] = terminal_text
                payload['output_text'] = terminal_text

                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )
                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                self.assertEqual(review['status'], 'pending')
                self.assertEqual(check['status'], 'pending')
                self.assertTrue(
                    {'duplicate_label', 'duplicate_artifact_ref', 'missing_label'}
                    .issubset(set(check['issue_codes']))
                )

    def test_structured_dependency_join_rejects_unique_but_swapped_refs(self):
        prompt, _graph, payload, image_paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        swapped_rows = [dict(row) for row in correct_rows]
        swapped_rows[0]['artifact_ref'] = image_paths['B']
        swapped_rows[1]['artifact_ref'] = image_paths['A']
        payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(
            swapped_rows,
            ensure_ascii=False,
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(check['status'], 'pending')
        self.assertEqual(check['repair_action'], 'rebind_dependency_evidence')
        self.assertIn('label_ref_dependency_mismatch', check['issue_codes'])
        self.assertNotIn('exact_count_mismatch', check['issue_codes'])
        self.assertNotIn('duplicate_artifact_ref', check['issue_codes'])

    def test_structured_dependency_join_validates_json_list_and_requested_fields(self):
        prompt, _graph, _payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        missing_field_rows = [dict(row) for row in correct_rows]
        del missing_field_rows[1]['visible_evidence']
        cases = (
            ('not-json', '{not-json', 'json_parse_error'),
            ('json-object', json.dumps(correct_rows[0]), 'json_list_required'),
            (
                'missing-requested-field',
                json.dumps(missing_field_rows),
                'missing_required_field',
            ),
        )
        for case_name, terminal_text, expected_issue in cases:
            with self.subTest(case=case_name):
                _prompt, _graph, payload, _paths, _rows = self._structured_dependency_join_fixture(
                    terminal_text,
                    prompt=prompt,
                )
                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )
                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                self.assertEqual(review['status'], 'pending')
                self.assertEqual(check['status'], 'pending')
                self.assertEqual(check['repair_action'], 'repair_branch_contract')
                self.assertIn(expected_issue, check['issue_codes'])

    def test_structured_dependency_join_does_not_fall_back_to_root_text_without_terminal_result(self):
        prompt, _graph, payload, _paths, _correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        payload['late_fill']['fill_results'].pop()

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(check['status'], 'pending')
        self.assertIn('terminal_result_missing', check['issue_codes'])
        self.assertEqual(check['terminal_result_source'], 'late_fill.fill_results')
        self.assertEqual(check['repair_action'], 'repair_branch_contract')

    def test_structured_dependency_join_rejects_invalid_values_and_nonstandard_json(self):
        prompt, _graph, _payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        whitespace_refs = [dict(row) for row in correct_rows]
        for row in whitespace_refs:
            row['artifact_ref'] = '   '
        nonscalar_evidence = [dict(row) for row in correct_rows]
        nonscalar_evidence[1]['visible_evidence'] = {'text': 'not a scalar'}
        nan_evidence = [dict(row) for row in correct_rows]
        nan_evidence[1]['visible_evidence'] = float('nan')
        duplicate_key_json = (
            '[{"label":"A","label":"B","artifact_ref":"/tmp/structured-join-a.png",'
            '"visible_evidence":"A"},'
            '{"label":"B","artifact_ref":"/tmp/structured-join-b.png","visible_evidence":"B"},'
            '{"label":"C","artifact_ref":"/tmp/structured-join-c.png","visible_evidence":"C"}]'
        )
        unexpected_field_rows = [dict(row) for row in correct_rows]
        unexpected_field_rows[0]['confidence'] = 1.0
        cases = (
            (
                'whitespace-refs',
                json.dumps(whitespace_refs),
                'invalid_required_field_value',
            ),
            (
                'nonscalar-evidence',
                json.dumps(nonscalar_evidence),
                'invalid_required_field_value',
            ),
            (
                'nan-evidence',
                json.dumps(nan_evidence),
                'json_parse_error',
            ),
            ('duplicate-json-key', duplicate_key_json, 'duplicate_json_key'),
            (
                'unexpected-field',
                json.dumps(unexpected_field_rows),
                'unexpected_field',
            ),
        )
        for case_name, terminal_text, expected_issue in cases:
            with self.subTest(case=case_name):
                _prompt, _graph, payload, _paths, _rows = self._structured_dependency_join_fixture(
                    terminal_text,
                    prompt=prompt,
                )
                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )
                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                self.assertEqual(review['status'], 'pending')
                self.assertEqual(check['repair_action'], 'repair_branch_contract')
                self.assertIn(expected_issue, check['issue_codes'])

    def test_structured_dependency_join_rejects_ambiguous_lineage_instead_of_fail_open(self):
        prompt, graph, payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        graph['phases'][4]['depends_on'] = ['phase-2', 'phase-3']
        payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(correct_rows)

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'pending')
        self.assertIn('dependency_lineage_ambiguous', check['issue_codes'])
        self.assertEqual(check['repair_action'], 'repair_branch_contract')

    def test_structured_dependency_join_rejects_missing_or_duplicate_queue_identity(self):
        cases = (
            ('missing', 'dependency_label_identity_unresolved'),
            ('duplicate', 'dependency_label_identity_ambiguous'),
        )
        for case_name, expected_issue in cases:
            with self.subTest(case=case_name):
                prompt, graph, payload, _paths, correct_rows = self._structured_dependency_join_fixture(
                    [],
                )
                if case_name == 'missing':
                    graph['phases'][1].pop('queue_index')
                    graph['phases'][4].pop('queue_index')
                    graph['phases'][1]['branch_id'] = 'branch-image_generation-a'
                    graph['phases'][4]['branch_id'] = 'branch-vision_analysis-a'
                else:
                    graph['phases'][5]['queue_index'] = 1
                payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(correct_rows)

                review = self.owner.build_graph_closure_review(
                    payload['output_text'],
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    artifact_payload=payload,
                )

                check = next(
                    item for item in review['checks']
                    if item.get('check_kind') == 'structured_dependency_join'
                )
                self.assertEqual(review['status'], 'pending')
                self.assertIn(expected_issue, check['issue_codes'])
                self.assertEqual(check['repair_action'], 'repair_branch_contract')

    def test_structured_dependency_join_maps_labels_by_queue_index_not_dependency_order(self):
        prompt, graph, payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        reordered_dependencies = ['phase-6', 'phase-5', 'phase-7']
        graph['phases'][-1]['depends_on'] = reordered_dependencies
        terminal_result = payload['late_fill']['fill_results'][-1]
        terminal_result['execution_contract']['depends_on'] = reordered_dependencies
        terminal_result['result_text'] = json.dumps(correct_rows)

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(check['status'], 'fulfilled')
        self.assertEqual(check['depends_on'], reordered_dependencies)

    def test_structured_dependency_join_rejects_stale_nonterminal_artifact_alias(self):
        prompt, _graph, payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        stale_rows = [dict(row) for row in correct_rows]
        stale_rows[0]['artifact_ref'] = 'artifact:stale-image-a'
        payload['artifacts'].append(
            {
                'type': 'image',
                'path': '/tmp/stale-structured-join-a.png',
                'artifact_ref': 'artifact:stale-image-a',
                'phase_id': 'phase-2',
                'branch_id': 'branch-image_generation-1',
                'status': 'repair_needed',
            }
        )
        payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(stale_rows)

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'pending')
        self.assertIn('label_ref_dependency_mismatch', check['issue_codes'])
        self.assertEqual(check['repair_action'], 'rebind_dependency_evidence')

    def test_structured_dependency_join_accepts_backticked_requested_fields(self):
        plain_prompt, _graph, _payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        prompt = plain_prompt.replace(
            'label, artifact_ref und visible_evidence',
            '`label`, `artifact_ref` und `visible_evidence`',
        )
        _prompt, _graph, payload, _paths, _rows = self._structured_dependency_join_fixture(
            correct_rows,
            prompt=prompt,
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(check['status'], 'fulfilled')

    def test_structured_dependency_join_accepts_statusless_canonical_artifact_aliases(self):
        prompt, _graph, payload, _paths, correct_rows = self._structured_dependency_join_fixture(
            [],
        )
        canonical_ref_rows = [dict(row) for row in correct_rows]
        for label, row in zip(('A', 'B', 'C'), canonical_ref_rows):
            row['artifact_ref'] = f'artifact:image-{label.lower()}'
        payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(
            canonical_ref_rows
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(check['status'], 'fulfilled')

    def test_structured_dependency_join_accepts_mixed_current_response_artifact_aliases(self):
        prompt, _graph, payload, _paths, correct_rows = (
            self._structured_dependency_join_fixture([])
        )
        canonical_images = [
            item
            for item in build_canonical_response_artifacts(payload)
            if item.get('type') == 'image'
        ]
        canonical_by_phase = {
            item.get('phase_id'): item
            for item in canonical_images
        }
        mixed_alias_rows = [dict(row) for row in correct_rows]
        mixed_alias_rows[0]['artifact_ref'] = canonical_by_phase['phase-2']['artifact_ref']
        mixed_alias_rows[1]['artifact_ref'] = canonical_by_phase['phase-3']['artifact_ref']
        mixed_alias_rows[2]['artifact_ref'] = payload['artifacts'][2]['artifact_ref']
        payload['late_fill']['fill_results'][-1]['result_text'] = json.dumps(
            mixed_alias_rows,
            ensure_ascii=False,
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'structured_dependency_join'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(check['status'], 'fulfilled')
        self.assertNotIn('issue_codes', check)

    def test_structured_dependency_join_gate_ignores_generic_post_artifact_chat(self):
        prompt = 'Vergleiche die drei erzeugten Bilder in einem kurzen Absatz.'
        _prompt, _graph, payload, _paths, _correct_rows = self._structured_dependency_join_fixture(
            'A ist heller als B, während C die stärkste Regenstimmung zeigt.',
            prompt=prompt,
        )

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'structured_dependency_join'
            ]
        )
        self.assertEqual(review['status'], 'fulfilled')

    def test_closure_review_does_not_spend_text_artifact_pool_on_chat_output(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'mode': 'phase_chain',
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'preparation_output',
                    'status': 'completed',
                    'repair_action': 'manual_review',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-text_artifact-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'status': 'planned',
                    'repair_action': 'manual_review',
                },
                {
                    'obligation_id': 'obligation-phase-3',
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-text_artifact-2',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'status': 'planned',
                    'repair_action': 'manual_review',
                },
            ],
            'phases': [
                {'phase_id': 'phase-1', 'branch_id': 'phase-1', 'capability': 'chat', 'status': 'completed'},
                {'phase_id': 'phase-2', 'branch_id': 'branch-text_artifact-1', 'capability': 'chat', 'status': 'planned'},
                {'phase_id': 'phase-3', 'branch_id': 'branch-text_artifact-2', 'capability': 'chat', 'status': 'planned'},
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Prepared landing page assets.',
            request_payload={'ghost_route': True, 'prompt': 'Create index.html and styles.css.'},
            artifact_payload={
                'output_text': 'Prepared landing page assets.',
                'runtime': {'request_phase_graph': phase_graph},
                'artifacts': [
                    {'type': 'text', 'path': '/tmp/index.html', 'text_artifact_source_name': 'index'},
                    {'type': 'text', 'path': '/tmp/styles.css', 'text_artifact_source_name': 'styles'},
                ],
                'saved_text_artifacts': [
                    {'path': '/tmp/index.html', 'source_name': 'index'},
                    {'path': '/tmp/styles.css', 'source_name': 'styles'},
                ],
                'late_fill': {
                    'status': 'completed',
                    'completed_capabilities': ['chat'],
                    'completed_branches': [
                        {
                            'branch_id': 'branch-text_artifact-1',
                            'phase_id': 'phase-2',
                            'capability': 'chat',
                            'output_type': 'text',
                            'status': 'fulfilled',
                        },
                        {
                            'branch_id': 'branch-text_artifact-2',
                            'phase_id': 'phase-3',
                            'capability': 'chat',
                            'output_type': 'text',
                            'status': 'fulfilled',
                        },
                    ],
                    'final_materialization_contract_status': 'fulfilled',
                },
            },
        )

        checks_by_branch = {item.get('branch_id'): item for item in review['checks']}

        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(review['counts']['pending'], 0)
        self.assertEqual(checks_by_branch['branch-text_artifact-1']['status'], 'fulfilled')
        self.assertEqual(checks_by_branch['branch-text_artifact-2']['status'], 'fulfilled')
        self.assertEqual(
            review['surface_state']['category_counts']['repair_pending'],
            0,
        )

    def test_closure_review_keeps_completed_sibling_when_repeated_capability_branch_fails(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'preparation_output',
                    'status': 'completed',
                },
                {
                    'obligation_id': 'obligation-phase-6',
                    'phase_id': 'phase-6',
                    'branch_id': 'branch-chat-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'final_output',
                    'depends_on': ['phase-4', 'phase-5'],
                    'status': 'pending',
                },
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-6',
                    'branch_id': 'branch-chat-1',
                    'capability': 'chat',
                    'depends_on': ['phase-4', 'phase-5'],
                    'status': 'pending',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Prepared scene text.',
            request_payload={'ghost_route': True, 'prompt': 'Prepare a scene, then compare evidence.'},
            artifact_payload={
                'output_text': 'Prepared scene text.',
                'runtime': {'request_phase_graph': phase_graph},
                'late_fill': {
                    'status': 'failed',
                    'failed_capabilities': ['chat'],
                    'failed_branches': [
                        {
                            'branch_id': 'branch-chat-1',
                            'phase_id': 'phase-6',
                            'capability': 'chat',
                            'status': 'failed',
                        },
                    ],
                },
            },
        )

        checks_by_branch = {item.get('branch_id'): item for item in review['checks']}

        self.assertEqual(checks_by_branch['phase-1']['status'], 'fulfilled')
        self.assertNotEqual(
            checks_by_branch['phase-1']['evidence'],
            'late_fill_failed_capability',
        )
        self.assertEqual(checks_by_branch['branch-chat-1']['status'], 'blocked')
        self.assertEqual(
            checks_by_branch['branch-chat-1']['evidence'],
            'late_fill_failed_branch',
        )
        self.assertEqual(review['counts']['fulfilled'], 1)
        self.assertEqual(review['counts']['blocked'], 1)

    def test_fulfilled_closure_does_not_carry_pre_fill_continuation_or_gap(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'continuation_required': True,
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'preparation_output',
                    'status': 'completed',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'role': 'final_output',
                    'depends_on': ['phase-1'],
                    'status': 'planned',
                },
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                    'status': 'planned',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Prepared scene text.',
            request_payload={'ghost_route': True, 'prompt': 'Prepare a scene, then create an image.'},
            artifact_payload={
                'output_text': 'Prepared scene text.',
                'runtime': {'request_phase_graph': phase_graph},
                'saved_image_path': '/tmp/generated-scene.png',
                'artifacts': [
                    {
                        'type': 'image',
                        'path': '/tmp/generated-scene.png',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-image_generation-1',
                    },
                ],
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        {
                            'branch_id': 'branch-image_generation-1',
                            'phase_id': 'phase-2',
                            'capability': 'image_generation',
                            'output_type': 'image',
                            'status': 'fulfilled',
                        },
                    ],
                    'final_materialization_contract_status': 'fulfilled',
                },
            },
            artifact_gap={
                'code': 'deferred_follow_up_fill',
                'trigger': 'execution_planner_deferred_follow_up',
            },
        )

        self.assertEqual(review['status'], 'fulfilled')
        self.assertFalse(review['continuation_required'])
        self.assertNotIn('closure_gap_code', review)
        self.assertNotIn('closure_gap_trigger', review)

    def test_extract_batch_image_prompts_keeps_inline_markdown_labeled_third_prompt_before_html(self):
        output_text = (
            '**Image Prompts**\n\n'
            '**Image 1 (Hero):** Cinematic exterior of Nocturne Sanctum in bright alpine daylight.\n\n'
            '**Image 2 (Interior 1):** Luxurious lounge with anthracite furniture and warm amber light.\n\n'
            '**Image 3 (Interior 2):** Serene alpine bedroom with soft white textiles and gold accents.\n\n'
            '***\n\n'
            '**index.html**\n\n'
            '```html\n'
            '<!doctype html><html><body><div class="hero-image" style="background-image: url(\'image1.jpg\');"></div></body></html>\n'
            '```\n\n'
            '**styles.css**\n\n'
            '```css\n'
            '.hero-image { background-image: url("image1.jpg"); }\n'
            '```'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=3)

        self.assertEqual(
            prompts,
            [
                'Cinematic exterior of Nocturne Sanctum in bright alpine daylight.',
                'Luxurious lounge with anthracite furniture and warm amber light.',
                'Serene alpine bedroom with soft white textiles and gold accents.',
            ],
        )

    def test_extract_batch_image_prompts_preserves_sequential_bold_alpha_labels_before_analysis_section(self):
        output_text = (
            '**Image Generation Prompts:**\n\n'
            '*   **A:** A photorealistic red lighthouse on a snowy coast during a bright clear day, '
            'crisp white snow and blue sky.\n'
            '*   **B:** An atmospheric green-toned library at night; dark emerald and mossy hues, '
            'towering bookshelves and pools of warm reading light.\n'
            '*   **C:** A glass greenhouse with a distinct blue tint during a heavy rainstorm, '
            'rain streaking across the panes and lush plants visible inside.\n\n'
            '**Analysis Instruction:**\n'
            'Perform an independent visual analysis of each generated image, checking subject, '
            'color, weather, time of day, and lighting.\n\n'
            '**Final Output Specification:**\n'
            'Return one JSON list with entries A, B, and C.'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=3)

        self.assertEqual(
            prompts,
            [
                'A photorealistic red lighthouse on a snowy coast during a bright clear day, '
                'crisp white snow and blue sky.',
                'An atmospheric green-toned library at night; dark emerald and mossy hues, '
                'towering bookshelves and pools of warm reading light.',
                'A glass greenhouse with a distinct blue tint during a heavy rainstorm, '
                'rain streaking across the panes and lush plants visible inside.',
            ],
        )

    def test_extract_batch_image_prompts_preserves_plain_leading_alpha_sequence_before_consumer_paragraphs(self):
        output_text = (
            'A: A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
            'high resolution, cinematic photography.\n'
            'B: An interior of a library with green accents at night, soft moonlight filtering '
            'through windows, illuminating dark wooden bookshelves, atmospheric and moody.\n'
            'C: A glass greenhouse with blue-tinted panes during a heavy rainstorm, visible '
            'raindrops streaking down the glass, lush indoor plants, cold and rainy atmosphere.\n\n'
            'For each generated image (A, B, and C), perform a separate visual analysis focusing '
            'on identifying specific colors, lighting states, and weather elements.\n\n'
            'Output exactly one JSON list containing three objects with the fields label, '
            'artifact_ref, and visible_evidence for A, B, and C.'
        )

        prompts = self.owner.extract_batch_image_prompts(
            output_text,
            expected_count=3,
            allow_plain_alpha_sequence=True,
        )

        self.assertEqual(
            prompts,
            [
                'A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
                'high resolution, cinematic photography.',
                'An interior of a library with green accents at night, soft moonlight filtering '
                'through windows, illuminating dark wooden bookshelves, atmospheric and moody.',
                'A glass greenhouse with blue-tinted panes during a heavy rainstorm, visible '
                'raindrops streaking down the glass, lush indoor plants, cold and rainy atmosphere.',
            ],
        )

    def test_extract_batch_image_prompts_plain_alpha_sequence_fails_closed_when_not_exact(self):
        invalid_outputs = {
            'missing_b': (
                'A: A red lighthouse in bright snow by day.\n'
                'C: A blue greenhouse in heavy rain at night.'
            ),
            'duplicate_b': (
                'A: A red lighthouse in bright snow by day.\n'
                'B: A green library with warm lamps at night.\n'
                'B: A blue greenhouse in heavy rain at night.'
            ),
            'too_few': (
                'A: A red lighthouse in bright snow by day.\n'
                'B: A green library with warm lamps at night.'
            ),
            'extra_consecutive_label': (
                'A: A red lighthouse in bright snow by day.\n'
                'B: A green library with warm lamps at night.\n'
                'C: A blue greenhouse in heavy rain at night.\n'
                'D: A golden observatory beneath a starry sky.'
            ),
            'nonleading_sequence': (
                'Prepared image options follow.\n'
                'A: A red lighthouse in bright snow by day.\n'
                'B: A green library with warm lamps at night.\n'
                'C: A blue greenhouse in heavy rain at night.'
            ),
        }

        for name, output_text in invalid_outputs.items():
            with self.subTest(name=name):
                self.assertEqual(
                    self.owner.extract_batch_image_prompts(
                        output_text,
                        expected_count=3,
                        allow_plain_alpha_sequence=True,
                    ),
                    [],
                )

    def test_extract_batch_image_prompts_does_not_enable_plain_alpha_without_prepare_authority(self):
        output_text = (
            'A: Explain why the first review criterion is important.\n'
            'B: Explain why the second review criterion is important.'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=2)

        self.assertLess(len(prompts), 2)

    def test_build_response_semantic_phase_payload_uses_graph_counted_plain_alpha_sequence(self):
        output_text = (
            'A: A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
            'high resolution, cinematic photography.\n'
            'B: An interior of a library with green accents at night, soft moonlight filtering '
            'through windows, illuminating dark wooden bookshelves, atmospheric and moody.\n'
            'C: A glass greenhouse with blue-tinted panes during a heavy rainstorm, visible '
            'raindrops streaking down the glass, lush indoor plants, cold and rainy atmosphere.\n\n'
            'For each generated image (A, B, and C), perform a separate visual analysis.\n\n'
            'Output exactly one JSON list with label, artifact_ref, and visible_evidence.'
        )
        image_branches = [
            {
                'phase_id': f'phase-{index + 1}',
                'branch_id': f'branch-image_generation-{index}',
                'capability': 'image_generation',
                'output_type': 'image',
                'depends_on': ['phase-1'],
                'queue_index': index,
            }
            for index in range(1, 4)
        ]
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'phase_chain',
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'current_phase_resolution': 'graph_resolved',
            'prompt_intent': {'requested_visual_output_count': 3},
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'kind': 'prepare',
                    'role': 'text_preparation',
                    'capability': 'chat',
                    'status': 'completed',
                },
                *image_branches,
            ],
            'downstream_phase_ids': [item['phase_id'] for item in image_branches],
            'downstream_branches': image_branches,
            'downstream_capabilities': ['image_generation'],
            'is_multi_phase': True,
            'continuation_required': True,
        }

        payload = self.owner.build_response_semantic_phase_payload(
            output_text=output_text,
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload={'prompt': 'Generate three labeled images.'},
            capability='chat',
        )

        self.assertEqual(
            payload['batch_prompts'],
            [
                'A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
                'high resolution, cinematic photography.',
                'An interior of a library with green accents at night, soft moonlight filtering '
                'through windows, illuminating dark wooden bookshelves, atmospheric and moody.',
                'A glass greenhouse with blue-tinted panes during a heavy rainstorm, visible '
                'raindrops streaking down the glass, lush indoor plants, cold and rainy atmosphere.',
            ],
        )
        self.assertEqual(payload['batch_prompt_expected_count'], 3)
        self.assertEqual(payload['batch_prompts_source'], 'semantic_prepare_phase_output')
        self.assertEqual(payload['batch_prompt_source_phase_id'], 'phase-1')

    def test_build_response_semantic_phase_payload_rejects_plain_alpha_without_exact_image_slots(self):
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'phase_chain',
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'current_phase_resolution': 'graph_resolved',
            'prompt_intent': {'requested_visual_output_count': 2},
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'kind': 'prepare',
                    'role': 'text_preparation',
                    'capability': 'chat',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image-alpha',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                    'queue_index': 1,
                },
                {
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-image-beta',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                    'queue_index': 3,
                },
            ],
            'downstream_capabilities': ['image_generation'],
            'is_multi_phase': True,
            'continuation_required': True,
        }

        payload = self.owner.build_response_semantic_phase_payload(
            output_text=(
                'A: Explain why the first review criterion is important.\n'
                'B: Explain why the second review criterion is important.'
            ),
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload={'prompt': 'Compare two review criteria.'},
            capability='chat',
        )

        self.assertNotIn('batch_prompts', payload)

    def test_build_response_semantic_phase_payload_rejects_untyped_plain_alpha_for_mixed_direct_consumers(self):
        image_branches = [
            {
                'phase_id': f'phase-image-{index}',
                'branch_id': f'branch-image-{index}',
                'capability': 'image_generation',
                'output_type': 'image',
                'depends_on': ['phase-1'],
                'queue_index': index,
            }
            for index in range(1, 3)
        ]
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'phase_chain',
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'current_phase_resolution': 'graph_resolved',
            'prompt_intent': {'requested_visual_output_count': 2},
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'kind': 'prepare',
                    'role': 'text_preparation',
                    'capability': 'chat',
                    'status': 'completed',
                },
                *image_branches,
                {
                    'phase_id': 'phase-audio',
                    'branch_id': 'branch-audio',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'depends_on': ['phase-1'],
                    'queue_index': 3,
                },
            ],
            'downstream_capabilities': ['image_generation', 'text_to_speech'],
            'is_multi_phase': True,
            'continuation_required': True,
        }

        payload = self.owner.build_response_semantic_phase_payload(
            output_text=(
                'A: Silence folding into crimson geometry.\n'
                'B: Time resting beneath translucent blue glass.'
            ),
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload={'prompt': 'Create two images and one spoken line.'},
            capability='chat',
        )

        self.assertNotIn('batch_prompts', payload)

    def test_extract_batch_image_prompts_does_not_promote_unrelated_bold_alpha_review_options(self):
        output_text = (
            '### Image Generation Manifest\n\n'
            '- `@Red_Fox` | "Forest watch" | A red fox portrait in snowy woodland, '
            'soft daylight and crisp photographic detail.\n'
            '- `@Blue_Wing` | "Garden flight" | A blue butterfly macro photograph in a '
            'summer garden, shallow depth of field and natural light.\n\n'
            '**Review Options:**\n\n'
            '* **A:** Explain why the first review criterion is important.\n'
            '* **B:** Explain why the second review criterion is important.\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=2)

        self.assertEqual(
            prompts,
            [
                'Red Fox, A red fox portrait in snowy woodland, soft daylight and crisp photographic detail.',
                'Blue Wing, A blue butterfly macro photograph in a summer garden, shallow depth of field and natural light.',
            ],
        )
        self.assertFalse(any('review criterion' in prompt for prompt in prompts))

    def test_late_fill_batch_normalization_recovers_sequential_bold_alpha_prompts_from_corrupted_batch(self):
        artifact_prompt = (
            '*   **A:** A photorealistic red lighthouse on a snowy coast during a bright clear day, '
            'crisp white snow and blue sky.\n'
            '*   **B:** An atmospheric green-toned library at night; dark emerald and mossy hues, '
            'towering bookshelves and pools of warm reading light.\n'
            '*   **C:** A glass greenhouse with a distinct blue tint during a heavy rainstorm, '
            'rain streaking across the panes and lush plants visible inside.\n\n'
            '**Analysis Instruction:**\n'
            'Perform an independent visual analysis of each generated image, checking subject, '
            'color, weather, time of day, and lighting.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                'A:** A photorealistic red lighthouse on a snowy coast during a bright clear day, '
                'crisp white snow and blue sky.',
                'C:** A glass greenhouse with a distinct blue tint during a heavy rainstorm, '
                'rain streaking across the panes and lush plants visible inside.',
                'Perform an independent visual analysis of each generated image, checking subject, '
                'color, weather, time of day, and lighting.',
            ],
            expected_count=3,
            artifact_prompt=artifact_prompt,
        )

        self.assertEqual(
            prompts,
            [
                'A photorealistic red lighthouse on a snowy coast during a bright clear day, '
                'crisp white snow and blue sky.',
                'An atmospheric green-toned library at night; dark emerald and mossy hues, '
                'towering bookshelves and pools of warm reading light.',
                'A glass greenhouse with a distinct blue tint during a heavy rainstorm, '
                'rain streaking across the panes and lush plants visible inside.',
            ],
        )

    def test_late_fill_batch_normalization_keeps_healthy_batch_over_unrelated_bold_alpha_context(self):
        healthy_prompts = [
            'A red fox portrait in snowy woodland, soft daylight and crisp photographic detail.',
            'A blue butterfly macro photograph in a summer garden, shallow depth of field and natural light.',
        ]
        artifact_prompt = (
            '* **A:** Explain why the first review criterion is important.\n'
            '* **B:** Explain why the second review criterion is important.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            healthy_prompts,
            expected_count=2,
            artifact_prompt=artifact_prompt,
        )

        self.assertEqual(prompts, healthy_prompts)

    def test_late_fill_batch_normalization_recovers_collapsed_plain_alpha_sequence(self):
        prepared_text = (
            'A: A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
            'high resolution, cinematic photography.\n'
            'B: An interior of a library with green accents at night, soft moonlight filtering '
            'through windows, illuminating dark wooden bookshelves, atmospheric and moody.\n'
            'C: A glass greenhouse with blue-tinted panes during a heavy rainstorm, visible '
            'raindrops streaking down the glass, lush indoor plants, cold and rainy atmosphere.\n\n'
            'For each generated image (A, B, and C), perform a separate visual analysis.\n\n'
            'Output exactly one JSON list with label, artifact_ref, and visible_evidence.'
        )
        corrupt_batch = [
            (
                'A: A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
                'high resolution, cinematic photography. B: An interior of a library with green '
                'accents at night, soft moonlight filtering through windows, illuminating dark '
                'wooden bookshelves, atmospheric and moody. C: A glass greenhouse with blue-tinted '
                'panes during a heavy rainstorm, visible raindrops streaking down the glass, lush '
                'indoor plants, cold and rainy atmosphere.'
            ),
            'For each generated image (A, B, and C), perform a separate visual analysis.',
            'Output exactly one JSON list with label, artifact_ref, and visible_evidence.',
        ]

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            corrupt_batch,
            expected_count=3,
            content_payload=prepared_text,
        )

        self.assertEqual(
            prompts,
            [
                'A vibrant red lighthouse surrounded by thick white snow under bright daylight, '
                'high resolution, cinematic photography.',
                'An interior of a library with green accents at night, soft moonlight filtering '
                'through windows, illuminating dark wooden bookshelves, atmospheric and moody.',
                'A glass greenhouse with blue-tinted panes during a heavy rainstorm, visible '
                'raindrops streaking down the glass, lush indoor plants, cold and rainy atmosphere.',
            ],
        )

    def test_late_fill_batch_normalization_rejects_foreign_plain_alpha_recovery_source(self):
        corrupt_batch = [
            (
                'A: A red fox portrait in snowy woodland. '
                'B: A blue butterfly macro in a summer garden. '
                'C: A green tree frog beneath tropical leaves.'
            ),
            'For each generated image, inspect it and report visible evidence.',
            'Output exactly one JSON list with label and visible_evidence.',
        ]
        foreign_context = (
            'A: A bronze compass on a white table in soft daylight.\n'
            'B: A silver compass on a black table at night.\n'
            'C: A copper compass on a green table in rain.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            corrupt_batch,
            expected_count=3,
            content_payload=foreign_context,
        )

        self.assertFalse(any('compass' in prompt.lower() for prompt in prompts))

    def test_late_fill_batch_normalization_preserves_slots_when_plain_alpha_provenance_fails(self):
        raw_batch = [
            (
                'A: A red fox portrait in snowy woodland. '
                'B: A blue butterfly macro in a summer garden. '
                'C: A green tree frog beneath tropical leaves.'
            ),
            '.grid.reverse { direction: rtl; }',
            'A snowy owl flying over a frozen lake in crisp morning light.',
        ]
        foreign_context = (
            'A: A bronze compass on a white table in soft daylight.\n'
            'B: A silver compass on a black table at night.\n'
            'C: A copper compass on a green table in rain.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            raw_batch,
            expected_count=3,
            content_payload=foreign_context,
        )

        self.assertEqual(prompts, raw_batch)

    def test_late_fill_batch_normalization_preserves_empty_and_invalid_slot_positions(self):
        batches = {
            'empty_middle_slot': [
                'A cinematic red lighthouse in snow under bright daylight.',
                '',
                'A blue glass greenhouse during a heavy rainstorm.',
            ],
            'invalid_middle_slot': [
                'A cinematic red lighthouse in snow under bright daylight.',
                '.grid.reverse { direction: rtl; }',
                'A blue glass greenhouse during a heavy rainstorm.',
            ],
        }

        for name, raw_batch in batches.items():
            with self.subTest(name=name):
                prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
                    raw_batch,
                    expected_count=3,
                )

                self.assertEqual(prompts, raw_batch)

    def test_late_fill_batch_normalization_keeps_healthy_tail_prompts_over_composite_alpha_context(self):
        healthy_batch = [
            (
                'A: A red fox portrait in snowy woodland. '
                'B: A blue butterfly macro in a summer garden. '
                'C: A green tree frog beneath tropical leaves.'
            ),
            'A snowy owl flying over a frozen lake in crisp morning light.',
            'A grey wolf standing beneath pine trees in soft evening fog.',
        ]
        matching_alpha_context = (
            'A: A red fox portrait in snowy woodland.\n'
            'B: A blue butterfly macro in a summer garden.\n'
            'C: A green tree frog beneath tropical leaves.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            healthy_batch,
            expected_count=3,
            content_payload=matching_alpha_context,
        )

        self.assertEqual(prompts, healthy_batch)

    def test_late_fill_batch_normalization_keeps_healthy_batch_over_plain_alpha_context(self):
        healthy_prompts = [
            'A red fox portrait in snowy woodland with crisp photographic detail.',
            'A blue butterfly macro in a summer garden with shallow depth of field.',
            'A green tree frog beneath tropical leaves in soft rainy daylight.',
        ]
        unrelated_context = (
            'A: A bronze compass on a white table in soft daylight.\n'
            'B: A silver compass on a black table at night.\n'
            'C: A copper compass on a green table in rain.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            healthy_prompts,
            expected_count=3,
            content_payload=unrelated_context,
        )

        self.assertEqual(prompts, healthy_prompts)

    def test_extract_batch_image_prompts_keeps_structural_hero_asset_beyond_counted_set(self):
        manifest_items = [
            '1. **hero-bg.jpg**: A panoramic sunny park hero background with happy dogs running, '
            'vibrant social landing-page photography.',
        ]
        manifest_items.extend(
            f'{index + 1}. **selfie-{index}.jpg**: Ultra realistic animal selfie number {index}, '
            'curiously close to a wide-angle camera lens with cinematic lighting.'
            for index in range(1, 17)
        )
        output_text = (
            '### image_generation_manifest (for downstream phase)\n\n'
            '*Note to Image Generation Phase: Generate these image assets for the landing page.*\n\n'
            + '\n'.join(manifest_items)
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=16)

        self.assertEqual(len(prompts), 17)
        self.assertEqual(
            prompts[0],
            'A panoramic sunny park hero background with happy dogs running, '
            'vibrant social landing-page photography.',
        )
        self.assertEqual(
            prompts[-1],
            'Ultra realistic animal selfie number 16, curiously close to a wide-angle camera lens '
            'with cinematic lighting.',
        )
        self.assertNotIn('hero-bg.jpg', prompts[0])

    def test_extract_batch_image_prompts_caps_non_structural_surplus_at_expected_count(self):
        output_text = '**Image Generation Prompts**\n\n' + '\n'.join(
            f'{index}. Ultra realistic animal portrait number {index}, direct camera gaze, '
            'cinematic lighting and detailed fur texture.'
            for index in range(1, 18)
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=16)

        self.assertEqual(len(prompts), 16)
        self.assertIn('number 16', prompts[-1])
        self.assertTrue(all('number 17' not in prompt for prompt in prompts))

    def test_extract_batch_image_prompts_uses_numbered_target_section_before_css(self):
        prompt_items = [
            (
                '1. **petsie-1.jpg**: A Golden Retriever taking a wide-angle selfie in the '
                'middle of Times Square, NYC, neon lights reflecting in eyes, paw visible at '
                'edge of frame, funny tongue out.'
            ),
            (
                '2. **petsie-2.jpg**: A Pug taking a selfie with the Eiffel Tower in the '
                'background, Paris, sunlight, funny squished face expression, extreme close-up.'
            ),
            (
                '3. **petsie-3.jpg**: A Siamese Cat taking a selfie in Shibuya Crossing, Tokyo, '
                'heavy neon pink and blue lighting, wide-angle distortion, intense eyes.'
            ),
            (
                '4. **<b>petsie-4.jpg</b>**: A Capybara taking a selfie at a colorful Rio de '
                'Janeiro Carnival, feathers and glitter in background, extreme close-up, zen '
                'but funny expression.'
            ),
        ]
        output_text = (
            '### Image Generation Prompts (Target: 4 Images)\n'
            '**Style Note for all prompts:** Extreme wide-angle "GoPro-style" animal selfie, '
            'distorted lens effect, high energy, funny facial expression, vibrant cinematic lighting.\n\n'
            + '\n'.join(prompt_items)
            + '\n\n***\n\n'
            '### index.html\n'
            '```html\n'
            '<!doctype html><html><body><section class="hero"></section></body></html>\n'
            '```\n\n'
            '### styles.css\n'
            '```css\n'
            '.btn-primary:hover { transform: scale(1.05); box-shadow: 0 0 30px var(--primary); }\n'
            '.section-title { font-size: 3rem; text-align: center; margin-bottom: 50px; }\n'
            '```'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=4)

        self.assertEqual(len(prompts), 4)
        self.assertIn('Golden Retriever', prompts[0])
        self.assertIn('Pug', prompts[1])
        self.assertIn('Siamese Cat', prompts[2])
        self.assertIn('Capybara', prompts[3])
        self.assertFalse(
            any(
                token in prompt
                for prompt in prompts
                for token in (
                    'Image Generation Prompts',
                    'Style Note',
                    'font-size',
                    'transform:',
                    'box-shadow',
                    'index.html',
                    'styles.css',
                    '<b>',
                    '</b>',
                )
            )
        )

    def test_extract_batch_image_prompts_uses_embedded_visual_prompt_field_only(self):
        prompt_bodies = [
            "Extreme macro shot of a Ring-tailed Lemur's eye, hyper-detailed iris textures, lush jungle bokeh.",
            'Underwater photography, Sea Turtle swimming through coral reef, light rays through turquoise water.',
            'Anaconda coiled on a dark rainforest floor, neon-tinted jungle lighting, wet scales reflecting flora.',
        ]
        manifest_items = []
        for index, prompt_body in enumerate(prompt_bodies, start=1):
            manifest_items.append(
                f'{index}. **Asset ID: snout_{index:02d}**\n'
                f'    - **@username**: `@Animal{index:02d}`\n'
                f'    - **Caption**: "Caption {index}"\n'
                f'    - **Visual Prompt**: {prompt_body}'
            )
        output_text = (
            '### Image Generation Manifest (Batch 3)\n'
            '*Target Style: High-end, cinematic, diverse species, extreme photography.*\n\n'
            + '\n'.join(manifest_items)
            + '\n\n---\n\n'
            '### File Materialization Blueprint\n'
            '#### CSS Specification (`styles.css`)\n'
            '- `.gallery { display: grid; }`\n'
            '#### HTML Specification (`index.html`)\n'
            '- `<main class="gallery-grid">`'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=3)

        self.assertEqual(prompts, prompt_bodies)
        for prompt in prompts:
            self.assertNotIn('Asset ID', prompt)
            self.assertNotIn('@username', prompt)
            self.assertNotIn('Caption', prompt)
            self.assertNotIn('snout_', prompt)
            self.assertNotIn('Visual Prompt', prompt)
            self.assertNotIn('styles.css', prompt)

    def test_extract_batch_image_prompts_uses_manifesto_table_prompt_column_without_image_heading(self):
        prompt_bodies = [
            'A serene Capybara floating in a Brazilian river, warm golden hour sunset-glow, soft ripples, cinematic lighting.',
            'An Axolotl underwater in Mexico, bioluminescent pinks and blues, neon-underwater lighting, tiny floating particles.',
            "Extreme macro shot of a Fennec Fox's face in the Sahara, focus on eyes/ears, fine sand textures, desert sun harsh shadows.",
            'A Puffin on an Icelandic cliff, moody Atlantic ocean background, dark basalt rocks, misty spray, cold color temperature.',
            'A Sloth in Costa Rican rainforest, dappled forest light through leaves, soft focus, lush green textures, peaceful atmosphere.',
            'An Octopus deep in Japanese waters, bioluminescent deep-sea lighting, dark indigo background, glowing suction cup details.',
            'A wide-angle lens shot of a Ring-tailed Lemur leaping through Madagascar canopy, motion blur, vibrant jungle greenery.',
            'A Toucan on a tropical branch, hyper-saturated tropical colors, vivid orange and yellow, sharp focus, bright sunlight.',
            "Ultra-macro shot of a Chameleon's skin/eye, extreme texture detail, shifting iridescent colors, macro photography style.",
            'A Penguin on an Antarctic glacier, high-contrast blue and white, crisp frozen textures, bright arctic daylight.',
            'A Komodo Dragon in Indonesia, cinematic heat haze effect, sun-baked earth tones, heavy atmosphere, low angle shot.',
            'A Red Panda in the Himalayas, soft bokeh mountain background, misty clouds, warm fur textures against cool blue mountains.',
        ]
        table_rows = [
            '| ID | @Username | Location/Caption | Visual Style Prompt (for Image Gen) |',
            '|:---|:---|:---|:---|',
        ]
        for index, prompt_body in enumerate(prompt_bodies, start=1):
            table_rows.append(
                f'| **{index:02d}** | `@Animal{index:02d}` | "Caption {index}" | *{prompt_body}* |'
            )
        output_text = (
            '### Phase Graph: Petsie Global Snout Gallery Production\n\n'
            '**Execution Strategy:**\n'
            '1. **Phase 1: Asset Generation (Image Branch)**\n'
            '   - Execute 12 parallel image generation tasks based on the `Influencer Manifesto`.\n\n'
            '---\n\n'
            '### Influencer Manifesto: The Global 12\n'
            '*Each entry contains the exact visual prompt for downstream generation and metadata.*\n\n'
            + '\n'.join(table_rows)
            + '\n\n---\n\n'
            '### Implementation Blueprint\n\n'
            '#### Structural Requirements (`index.html`)\n'
            '- Image container with `overflow: hidden` and hover scale effect (`transform: scale(1.05)`).\n'
            '- **Responsive Breakpoints**: Mobile: 1 column. Tablet: 2 columns. Desktop: 3-4 columns.'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=12)

        self.assertEqual(prompts, prompt_bodies)
        for prompt in prompts:
            self.assertNotIn('Phase Graph', prompt)
            self.assertNotIn('Influencer Manifesto', prompt)
            self.assertNotIn('@Animal', prompt)
            self.assertNotIn('Caption', prompt)
            self.assertNotIn('hidden', prompt)
            self.assertNotIn('transform:', prompt)
            self.assertNotIn('Responsive Breakpoints', prompt)
            self.assertNotIn('|', prompt)

    def test_extract_batch_image_prompts_composes_table_row_context_for_style_only_cells(self):
        output_text = (
            '### Content Payload: Image Generation Metadata (The 4 Influencers)\n\n'
            '| ID | @username | Caption | Visual Style / Prompt Instruction |\n'
            '| :--- | :--- | :--- | :--- |\n'
            '| **01** | `@Axo_Lotl` | "Living my best pink life." | Underwater, soft pink bioluminescent glow, macro focus. |\n'
            '| **02** | `@Capy_Chill` | "Zen master of the Pantanal." | Golden hour, warm sunset lighting, cinematic wide angle. |\n'
            '| **03** | `@Snow_Leap` | "Cold, but make it fashion." | Extreme macro, frosty texture, blue-toned cold atmosphere. |\n'
            '| **04** | `@Birdie_Boss` | "Legs for days, eyes on the prize." | High-contrast savanna sun, sharp eye detail, wide angle. |\n\n'
            '---\n\n'
            '### Content Payload: styles.css\n'
            '```css\n'
            '.grid { display: grid; }\n'
            '```'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=4)

        self.assertEqual(
            prompts,
            [
                'Axo Lotl, Underwater, soft pink bioluminescent glow, macro focus.',
                'Capy Chill, Golden hour, warm sunset lighting, cinematic wide angle.',
                'Snow Leap, Extreme macro, frosty texture, blue-toned cold atmosphere.',
                'Birdie Boss, High-contrast savanna sun, sharp eye detail, wide angle.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('|', prompt)
            self.assertNotIn('styles.css', prompt)
            self.assertNotIn('display:', prompt)

    def test_extract_batch_image_prompts_normalizes_social_manifest_pipe_records(self):
        output_text = (
            '### Influencer Manifesto: Global Trending\n\n'
            '1. `@Sloth_Slowmo` | "Nap time in Costa Rica" | Soft bokeh, dappled sunlight through leaves, cinematic jungle atmosphere.\n'
            '2. `@Toucan_Toni` | "Morning breakfast in the Amazon" | Extreme macro, vibrant tropical colors, shallow depth of field, raindrops on beak.\n'
            '3. `@Orchid_Orangutan` | "Swingin through Borneo" | Soft forest light, motion blur on edges, heavy foliage depth.\n'
            '4. `@Lemur_Lens` | "Big eyes, big dreams, Madagascar" | Soft focus background, cute eye-reflections, warm sunbeams. **Category: Deep Blue Wonders\n\n'
            '### File Content (`styles.css`)\n'
            '```css\n'
            '.grid { display: grid; }\n'
            '```'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=4)

        self.assertEqual(
            prompts,
            [
                'Sloth Slowmo, Soft bokeh, dappled sunlight through leaves, cinematic jungle atmosphere.',
                'Toucan Toni, Extreme macro, vibrant tropical colors, shallow depth of field, raindrops on beak.',
                'Orchid Orangutan, Soft forest light, motion blur on edges, heavy foliage depth.',
                'Lemur Lens, Soft focus background, cute eye-reflections, warm sunbeams.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('|', prompt)
            self.assertNotIn('`', prompt)
            self.assertNotIn('"', prompt)
            self.assertNotIn('Nap time', prompt)
            self.assertNotIn('Morning breakfast', prompt)
            self.assertNotIn('Category', prompt)
            self.assertNotIn('styles.css', prompt)

    def test_extract_batch_image_prompts_strips_unlabeled_social_caption_copy(self):
        output_text = (
            '### Image Generation Prompts\n\n'
            '1. pandalove cn, Fluff levels are off the charts! 🐼, '
            'High-key lighting, soft focus, misty bamboo forest, adorable red panda face.\n'
            "2. flamingo flame, Pink isn't just a color, it's a lifestyle. <0xF0><0x9F><0xA6><0xA9>, "
            'Vibrant saturation, tropical sun, turquoise water reflections, bright pinks.\n'
            '3. Golden Retriever, tongue out, bright sunny park background, wide-angle lens look.\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=3)

        self.assertEqual(
            prompts,
            [
                'Pandalove Cn, High-key lighting, soft focus, misty bamboo forest, adorable red panda face.',
                'Flamingo Flame, Vibrant saturation, tropical sun, turquoise water reflections, bright pinks.',
                'Golden Retriever, tongue out, bright sunny park background, wide-angle lens look.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('<0x', prompt)
            self.assertNotIn('Fluff levels', prompt)
            self.assertNotIn("isn't just", prompt)

    def test_extract_batch_image_prompts_uses_explicit_prompt_field_in_labeled_social_manifest_records(self):
        output_text = (
            '### Image Generation Prompts\n\n'
            '1. ID: 01** | **User:** @Luna_Paris | **Caption:** "Midnight in Montmartre ✨" | '
            '**Style:** Paris Sunset Glow | **Prompt:** Extreme close-up of a fluffy white cat face, '
            'amber eyes reflecting the Eiffel Tower, warm pink sunset bokeh.\n'
            '2. ID: 02** | **User:** @Rex_NYC | **Caption:** "Concrete Jungle Dreams 🍎" | '
            '**Style:** Gritty Street Photography | **Prompt:** A rugged Bulldog wearing a tiny bandana, '
            'New York City street background, motion blur of yellow taxis.\n'
            '3. ID: 03** | **User:** @Pip_Amazon | **Caption:** "Tropical Fever! 🦜" | '
            '**Style:** Tropical Vibrance | **Prompt:** Brightly colored Scarlet Macaw, extreme macro of feathers, '
            'lush green jungle leaves in background.\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=3)

        self.assertEqual(
            prompts,
            [
                'Extreme close-up of a fluffy white cat face, amber eyes reflecting the Eiffel Tower, warm pink sunset bokeh.',
                'A rugged Bulldog wearing a tiny bandana, New York City street background, motion blur of yellow taxis.',
                'Brightly colored Scarlet Macaw, extreme macro of feathers, lush green jungle leaves in background.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('|', prompt)
            self.assertNotIn('Caption', prompt)
            self.assertNotIn('Style:', prompt)
            self.assertNotIn('User:', prompt)
            self.assertNotIn('ID:', prompt)

    def test_extract_batch_image_prompts_strips_social_handle_title_prefix_labels(self):
        output_text = (
            '### Image Generation Prompts\n\n'
            '1. @Quack_Master**: Low angle shot of a white duck walking through vibrant Dutch tulip fields, '
            'bright spring sunlight, sharp focus.\n'
            '2. Chic Poodle**: Soft aesthetic, poodle in a Paris cafe, dreamy bokeh of cafe lights, '
            'elegant lighting, warm tones.\n'
            '3. Lion King Vibes**: Majestic lion face, Kenya savannah at sunset, golden backlight, '
            'dust motes, intense gaze.\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=3)

        self.assertEqual(
            prompts,
            [
                'Low angle shot of a white duck walking through vibrant Dutch tulip fields, bright spring sunlight, sharp focus.',
                'Soft aesthetic, poodle in a Paris cafe, dreamy bokeh of cafe lights, elegant lighting, warm tones.',
                'Majestic lion face, Kenya savannah at sunset, golden backlight, dust motes, intense gaze.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('**:', prompt)
            self.assertNotIn('Quack_Master', prompt)
            self.assertNotIn('Chic Poodle**', prompt)

    def test_extract_batch_image_prompts_composes_filename_social_asset_rows(self):
        output_text = (
            '### assets/images/ (Reference List for 2-D Assets)\n\n'
            '1. `shiba_tokyo.jpg`: @shiba_zen | "Neon lights & puppy bites" | **Macro**\n'
            '2. `siamese_paris.jpg`: @paris_paws | "Crepes and couture" | **Sunset-glow**\n'
            '3. `axolotl_mexico.jpg`: @axie_pink | "Hydration is key" | **Underwater-distortion**\n'
            '4. `tiger_india.jpg`: @stripes_royal | "King of the jungle walk" | **Cinematic-shadows**\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=4)

        self.assertEqual(
            prompts,
            [
                'Shiba Tokyo, Macro, social media portrait',
                'Siamese Paris, Sunset-glow',
                'Axolotl Mexico, Underwater-distortion',
                'Tiger India, Cinematic-shadows',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('|', prompt)
            self.assertNotIn('Neon lights & puppy bites', prompt)
            self.assertNotIn('Crepes and couture', prompt)
            self.assertNotIn('Hydration is key', prompt)
            self.assertNotIn('King of the jungle walk', prompt)

    def test_extract_batch_image_prompts_keeps_social_prefix_manifest_rows_without_keyword_gate(self):
        output_text = (
            '### Image Generation Assets: "Global Trending" 6-Set\n\n'
            '1. **@FennecEars**: A Fennec Fox peering through Sahara sand dunes; '
            '**Style: Heat-haze, golden hour, warm tones.**\n'
            '2. **@LemurLeap**: A Ring-tailed Lemur jumping between Madagascar branches; '
            '**Style: Action freeze-frame, high shutter speed.**\n'
            '3. **@SnowLeopardGhost**: A Snow Leopard blending into Himalayan rocks; '
            '**Style: Misty atmosphere, desaturated cool tones.**\n'
            '4. **@PlatypusPuzzled**: A Platypus surfacing in an Australian creek; '
            '**Style: Surface ripple distortion, soft focus.**\n'
            '5. **@FlamingoFlash**: A flock of Flamingos in a Florida wetland; '
            '**Style: Pastel aesthetic, high brightness.**\n'
            '6. **@TigerTough**: A Bengal Tiger walking through tall grass; India; '
            '**Style: Dramatic shadows, intense eye contact.**\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=6)

        self.assertEqual(len(prompts), 6)
        self.assertIn('A Fennec Fox peering through Sahara sand dunes', prompts[0])
        self.assertIn('A Ring-tailed Lemur jumping between Madagascar branches', prompts[1])
        self.assertIn('A Snow Leopard blending into Himalayan rocks', prompts[2])
        self.assertIn('A Platypus surfacing in an Australian creek', prompts[3])
        self.assertIn('A flock of Flamingos in a Florida wetland', prompts[4])
        self.assertIn('A Bengal Tiger walking through tall grass', prompts[5])
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('**Style', prompt)

    def test_extract_batch_image_prompts_uses_html_image_cards_before_css(self):
        output_text = (
            '<section class="global-feed">\n'
            '  <div class="gallery-card">\n'
            '    <img src="assets/golden.jpg" alt="Golden Retriever in Malibu" class="gallery-image">\n'
            '    <div class="card-overlay">\n'
            '      <p class="username">@golden_wave</p>\n'
            '      <p class="caption">Chasing sunset waves.</p>\n'
            '      <span class="style-tag">Golden Hour Glow</span>\n'
            '    </div>\n'
            '  </div>\n\n'
            '  <div class="gallery-card">\n'
            '    <img src="assets/sloth.jpg" alt="Sloth in Costa Rica" class="gallery-image">\n'
            '    <div class="card-overlay">\n'
            '      <p class="username">@slow_mo_life</p>\n'
            '      <p class="caption">Do not rush the process.</p>\n'
            '      <span class="style-tag">Canopy Green</span>\n'
            '    </div>\n'
            '  </div>\n\n'
            '  <div class="gallery-card">\n'
            '    <img src="assets/flamingo.jpg" alt="Flamingo in Caribbean" class="gallery-image">\n'
            '    <div class="card-overlay">\n'
            '      <p class="username">@pink_flamingo</p>\n'
            '      <p class="caption">Always standing tall. <0xF0><0x9F><0xA6><0xA9></p>\n'
            '      <span class="style-tag">Tropical Bright</span>\n'
            '    </div>\n'
            '  </div>\n\n'
            '  <div class="gallery-card">\n'
            '    <img src="assets/orca.jpg" alt="Orca in Norway" class="gallery-image">\n'
            '    <div class="card-overlay">\n'
            '      <p class="username">@fjord_whale</p>\n'
            '      <p class="caption">Breaching through the frost.</p>\n'
            '      <span class="style-tag">Cold Blue Monochrome</span>\n'
            '    </div>\n'
            '  </div>\n'
            '</section>\n\n'
            '.cta-button:hover { transform: scale(1.05) rotate(-2deg); box-shadow: 0 0 30px rgba(204, 255, 0, 0.4); }\n\n'
            '@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }\n\n'
            '.gallery-card:hover .card-overlay { opacity: 1; transform: translateY(0); }\n\n'
            '.gallery-card:hover .gallery-image { transform: scale(1.1); }\n'
        )

        prompts = self.owner.extract_batch_image_prompts(output_text, expected_count=4)

        self.assertEqual(len(prompts), 4)
        self.assertIn('Golden Retriever in Malibu', prompts[0])
        self.assertIn('Chasing sunset waves', prompts[0])
        self.assertIn('Sloth in Costa Rica', prompts[1])
        self.assertIn('Flamingo in Caribbean', prompts[2])
        self.assertIn('Tropical Bright', prompts[2])
        self.assertIn('Orca in Norway', prompts[3])
        for prompt in prompts:
            self.assertNotIn('<0x', prompt)
            self.assertNotIn('transform:', prompt)
            self.assertNotIn('box-shadow', prompt)
            self.assertNotIn('@keyframes', prompt)
            self.assertNotIn('{', prompt)

    def test_late_fill_image_prompt_units_keep_inline_markdown_labeled_third_prompt_before_html(self):
        content = (
            '**Image 1 (Hero):** Cinematic exterior of Nocturne Sanctum in bright alpine daylight.\n\n'
            '**Image 2 (Interior 1):** Luxurious lounge with anthracite furniture and warm amber light.\n\n'
            '**Image 3 (Interior 2):** Serene alpine bedroom with soft white textiles and gold accents.\n\n'
            '**index.html**\n'
            '```html\n'
            '<img src="image1.jpg">\n'
            '```'
        )

        prompts = LateFillRuntimeOwner._extract_late_fill_image_prompt_units(content)

        self.assertEqual(
            prompts,
            [
                'Cinematic exterior of Nocturne Sanctum in bright alpine daylight.',
                'Luxurious lounge with anthracite furniture and warm amber light.',
                'Serene alpine bedroom with soft white textiles and gold accents.',
            ],
        )

    def test_late_fill_normalization_rejects_layout_polluted_batch_prompts(self):
        layout_prompts = [
            (
                'Hero Section:** * **Background:** Full-width implementation of the '
                '**Workshop Image** (Image 2) with a dark overlay. * **Headline:** '
                '"Handwerk mit Leidenschaft." * **Subheadline:** Traditional craft.'
            ),
            (
                'About Us / Craftsmanship Section:** * **Layout:** Two-column split. '
                '* **Visual (Left):** Close-up feature using the **Oak Wood Image** '
                '(Image 1). * **Text (Right):** Qualität, die man fühlt.'
            ),
            (
                'Portfolio / Featured Work Section:** * **Layout:** Large feature block '
                'or gallery. * **Visual:** High-quality display of the **Minimalist Shelf '
                'Image** (Image 3). * **Text Overlay:** Nordic Minimal.'
            ),
        ]
        artifact_prompt = (
            '**Image 1 (Texture/Detail):** Extreme macro photography of freshly planed '
            'oak wood, highly detailed grain texture, scattered curly wooden shavings, '
            'warm natural lighting, professional woodworking aesthetic.\n\n'
            "**Image 2 (Atmosphere):** Cinematic shot of a traditional carpenter's "
            'workshop, intense backlighting streaming through large windows, visible '
            'dust motes dancing in sunbeams, silhouettes of hand tools and workbenches.\n\n'
            '**Image 3 (Product/Lifestyle):** High-end lifestyle photography of a '
            'minimalist wooden shelf made of light oak, placed in a bright contemporary '
            'living room, Scandinavian interior design style.'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            layout_prompts,
            expected_count=3,
            artifact_prompt=artifact_prompt,
        )

        self.assertEqual(len(prompts), 3)
        self.assertIn('freshly planed oak wood', prompts[0])
        self.assertIn("traditional carpenter's workshop", prompts[1])
        self.assertIn('minimalist wooden shelf', prompts[2])
        self.assertFalse(
            any(
                token in prompt
                for prompt in prompts
                for token in ('Hero Section', 'Headline', 'Layout:', 'Text Overlay')
            )
        )

    def test_extract_batch_image_prompts_uses_following_paragraph_for_prompt_title_lines(self):
        content = (
            '### Image Generation Prompts\n\n'
            '**Prompt 1: Hero Image (vessel_exterior)**\n'
            'An immense cathedral-like archive vessel drifting beside a shattered moon.\n\n'
            '**Prompt 2: Memory Atrium Image (atrium_interior)**\n'
            'Interior of a vast vertical archive atrium filled with floating holographic books.\n\n'
            '**Prompt 3: Restoration Laboratory Image (lab_interior)**\n'
            'A high-tech preservation laboratory where damaged cultural records are reconstructed.\n\n'
            '***\n\n'
            '### HTML Materialization Payload\n'
            '**Structure & Elements:** hero image `hero.png`, atrium image `atrium.png`, lab image `lab.png`.\n\n'
            '***\n\n'
            '### CSS Design Specification\n'
            '**Visual Theme:** Dark Cinematic Editorial.'
        )

        prompts = self.owner.extract_batch_image_prompts(content, expected_count=3)

        self.assertEqual(
            prompts,
            [
                'An immense cathedral-like archive vessel drifting beside a shattered moon.',
                'Interior of a vast vertical archive atrium filled with floating holographic books.',
                'A high-tech preservation laboratory where damaged cultural records are reconstructed.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('Hero Image', prompt)
            self.assertNotIn('HTML Materialization Payload', prompt)
            self.assertNotIn('CSS Design Specification', prompt)

    def test_extract_batch_image_prompts_uses_artifact_heading_bodies_before_text_artifacts(self):
        content = (
            '### Artifact 1: Image Generation Prompt (Teapot)\n'
            'A high-end minimalist product photograph of a sleek matte ceramic teapot on a light stone '
            'surface, illuminated by soft warm morning sunlight.\n\n'
            '### Artifact 2: Image Generation Prompt (Tea Leaves)\n'
            'A macro photographic close-up of high-quality loose tea leaves scattered on a minimalist '
            'light-colored textured surface, with intricate organic detail and natural morning light.\n\n'
            '### Artifact 3: index.html\n'
            '```html\n<img src="teapot.png"><img src="leaves.png">\n```\n\n'
            '### Artifact 4: styles.css\n'
            '```css\nimg { width: 100%; }\n```'
        )

        prompts = self.owner.extract_batch_image_prompts(content, expected_count=2)

        self.assertEqual(
            prompts,
            [
                'A high-end minimalist product photograph of a sleek matte ceramic teapot on a light stone '
                'surface, illuminated by soft warm morning sunlight.',
                'A macro photographic close-up of high-quality loose tea leaves scattered on a minimalist '
                'light-colored textured surface, with intricate organic detail and natural morning light.',
            ],
        )
        self.assertFalse(any('Image Generation Prompt' in prompt for prompt in prompts))
        self.assertFalse(any('index.html' in prompt or 'styles.css' in prompt for prompt in prompts))

    def test_late_fill_batch_normalization_replaces_artifact_heading_only_prompts_with_bodies(self):
        content = (
            '### Artifact 1: Image Generation Prompt (Teapot)\n'
            'A high-end minimalist product photograph of a sleek matte ceramic teapot on a light stone '
            'surface, illuminated by soft warm morning sunlight.\n\n'
            '### Artifact 2: Image Generation Prompt (Tea Leaves)\n'
            'A macro photographic close-up of high-quality loose tea leaves scattered on a minimalist '
            'light-colored textured surface, with intricate organic detail and natural morning light.\n\n'
            '### Artifact 3: index.html\n'
            '```html\n<img src="teapot.png"><img src="leaves.png">\n```'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                'Image Generation Prompt (Teapot)',
                'Image Generation Prompt (Tea Leaves)',
            ],
            expected_count=2,
            artifact_prompt='Image Generation Prompt (Teapot)',
            content_payload=content,
        )

        self.assertEqual(
            prompts,
            [
                'A high-end minimalist product photograph of a sleek matte ceramic teapot on a light stone '
                'surface, illuminated by soft warm morning sunlight.',
                'A macro photographic close-up of high-quality loose tea leaves scattered on a minimalist '
                'light-colored textured surface, with intricate organic detail and natural morning light.',
            ],
        )

    def test_artifact_image_prompt_body_stops_before_following_text_or_audio_artifact(self):
        content = (
            '### Artifact 1: Image Generation Prompt (Poster)\n'
            'A crisp editorial poster photograph with cobalt shapes and warm studio lighting.\n\n'
            '### Artefakt 2: Bildgenerierungs-Prompt (Detail)\n'
            'A macro detail photograph of textured cobalt paper under the same warm studio lighting.\n\n'
            '### Artifact 3: narration.txt\n'
            'Speak slowly: cobalt forms become a quiet landscape.\n\n'
            '### Artifact 4: Audio Generation\n'
            'Render narration.txt as speech.'
        )

        prompts = self.owner.extract_batch_image_prompts(content, expected_count=2)

        self.assertEqual(
            prompts,
            [
                'A crisp editorial poster photograph with cobalt shapes and warm studio lighting.',
                'A macro detail photograph of textured cobalt paper under the same warm studio lighting.',
            ],
        )
        self.assertFalse(any('narration.txt' in prompt or 'Speak slowly' in prompt for prompt in prompts))

    def test_late_fill_batch_normalization_recovers_target_section_before_css(self):
        content = (
            '### Image Generation Prompts (Target: 4 Images)\n'
            '**Style Note for all prompts:** Extreme wide-angle animal selfie, vibrant cinematic lighting.\n\n'
            '1. **petsie-1.jpg**: A Golden Retriever taking a wide-angle selfie in Times Square.\n'
            '2. **petsie-2.jpg**: A Pug taking a selfie with the Eiffel Tower in the background.\n'
            '3. **petsie-3.jpg**: A Siamese Cat taking a selfie in Shibuya Crossing.\n'
            '4. **<b>petsie-4.jpg</b>**: A Capybara taking a selfie at a colorful carnival.\n\n'
            '***\n\n'
            '### styles.css\n'
            '```css\n'
            '.btn-primary:hover { transform: scale(1.05); box-shadow: 0 0 30px var(--primary); }\n'
            '.section-title { font-size: 3rem; margin-bottom: 50px; }\n'
            '```'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                '### Image Generation Prompts (Target: 4 Images) **Style Note for all prompts:** '
                'Extreme wide-angle animal selfie.',
                '.btn-primary:hover { transform: scale(1.05); box-shadow: 0 0 30px var(--primary); }',
                '.section-title { font-size: 3rem; margin-bottom: 50px; }',
            ],
            expected_count=4,
            content_payload=content,
        )

        self.assertEqual(len(prompts), 4)
        self.assertIn('Golden Retriever', prompts[0])
        self.assertIn('Pug', prompts[1])
        self.assertIn('Siamese Cat', prompts[2])
        self.assertIn('Capybara', prompts[3])
        self.assertFalse(any('font-size' in prompt or 'transform:' in prompt for prompt in prompts))

    def test_late_fill_batch_normalization_recovers_html_image_cards_when_batch_is_css(self):
        content = (
            '<div class="gallery-card">\n'
            '  <img src="assets/golden.jpg" alt="Golden Retriever in Malibu">\n'
            '  <p class="username">@golden_wave</p>\n'
            '  <p class="caption">Chasing sunset waves.</p>\n'
            '</div>\n\n'
            '<div class="gallery-card">\n'
            '  <img src="assets/sloth.jpg" alt="Sloth in Costa Rica">\n'
            '  <p class="username">@slow_mo_life</p>\n'
            '  <p class="caption">Do not rush the process.</p>\n'
            '</div>\n\n'
            '<div class="gallery-card">\n'
            '  <img src="assets/flamingo.jpg" alt="Flamingo in Caribbean">\n'
            '  <p class="username">@pink_flamingo</p>\n'
            '  <p class="caption">Always standing tall. <0xF0><0x9F><0xA6><0xA9></p>\n'
            '</div>\n\n'
            '<div class="gallery-card">\n'
            '  <img src="assets/orca.jpg" alt="Orca in Norway">\n'
            '  <p class="username">@fjord_whale</p>\n'
            '  <p class="caption">Breaching through the frost.</p>\n'
            '</div>\n'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                '.cta-button:hover { transform: scale(1.05); box-shadow: 0 0 30px red; }',
                '@keyframes pulse { 0% { opacity: 1; } 100% { opacity: 0; } }',
                '.gallery-card:hover .card-overlay { opacity: 1; transform: translateY(0); }',
                '.gallery-card:hover .gallery-image { transform: scale(1.1); }',
            ],
            expected_count=4,
            content_payload=content,
        )

        self.assertEqual(len(prompts), 4)
        self.assertIn('Golden Retriever in Malibu', prompts[0])
        self.assertIn('Sloth in Costa Rica', prompts[1])
        self.assertIn('Flamingo in Caribbean', prompts[2])
        self.assertIn('Orca in Norway', prompts[3])
        self.assertFalse(any('transform:' in prompt or '<0x' in prompt for prompt in prompts))

    def test_late_fill_batch_normalization_uses_explicit_prompt_field_in_labeled_social_manifest_records(self):
        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                'ID: 01** | **User:** @Luna_Paris | **Caption:** "Midnight in Montmartre ✨" | '
                '**Style:** Paris Sunset Glow | **Prompt:** Extreme close-up of a fluffy white cat face, '
                'amber eyes reflecting the Eiffel Tower, warm pink sunset bokeh.',
                'ID: 02** | **User:** @Rex_NYC | **Caption:** "Concrete Jungle Dreams 🍎" | '
                '**Style:** Gritty Street Photography | **Prompt:** A rugged Bulldog wearing a tiny bandana, '
                'New York City street background, motion blur of yellow taxis.',
            ],
            expected_count=2,
        )

        self.assertEqual(
            prompts,
            [
                'Extreme close-up of a fluffy white cat face, amber eyes reflecting the Eiffel Tower, warm pink sunset bokeh.',
                'A rugged Bulldog wearing a tiny bandana, New York City street background, motion blur of yellow taxis.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('Caption', prompt)
            self.assertNotIn('User', prompt)

    def test_late_fill_batch_normalization_strips_social_handle_title_prefix_labels(self):
        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                '@Quack_Master**: Low angle shot of a white duck walking through vibrant Dutch tulip fields, '
                'bright spring sunlight, sharp focus.',
                'Chic Poodle**: Soft aesthetic, poodle in a Paris cafe, dreamy bokeh of cafe lights, '
                'elegant lighting, warm tones.',
            ],
            expected_count=2,
        )

        self.assertEqual(
            prompts,
            [
                'Low angle shot of a white duck walking through vibrant Dutch tulip fields, bright spring sunlight, sharp focus.',
                'Soft aesthetic, poodle in a Paris cafe, dreamy bokeh of cafe lights, elegant lighting, warm tones.',
            ],
        )
        for prompt in prompts:
            self.assertNotIn('@Quack_Master', prompt)
            self.assertNotIn('Chic Poodle**', prompt)

    def test_late_fill_batch_normalization_recovers_filename_social_asset_rows(self):
        content_payload = (
            '### assets/images/ (Reference List for 2-D Assets)\n\n'
            '1. `shiba_tokyo.jpg`: @shiba_zen | "Neon lights & puppy bites" | **Macro**\n'
            '2. `siamese_paris.jpg`: @paris_paws | "Crepes and couture" | **Sunset-glow**\n'
            '3. `axolotl_mexico.jpg`: @axie_pink | "Hydration is key" | **Underwater-distortion**\n'
            '4. `tiger_india.jpg`: @stripes_royal | "King of the jungle walk" | **Cinematic-shadows**\n'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                'shiba zen, Neon lights & puppy bites',
                'paris paws, Crepes and couture',
                'axie pink, Hydration is key',
            ],
            expected_count=4,
            content_payload=content_payload,
        )

        self.assertEqual(
            prompts,
            [
                'Shiba Tokyo, Macro, social media portrait',
                'Siamese Paris, Sunset-glow',
                'Axolotl Mexico, Underwater-distortion',
                'Tiger India, Cinematic-shadows',
            ],
        )
        self.assertFalse(any('Crepes and couture' in prompt for prompt in prompts))
        self.assertFalse(any('Neon lights & puppy bites' in prompt for prompt in prompts))

    def test_late_fill_batch_normalization_recovers_social_prefix_manifest_rows(self):
        content_payload = (
            '### Image Generation Assets: "Global Trending" 4-Set\n\n'
            '1. **@CapyVibes**: Extreme macro shot of a Capybara face; Brazil Amazon; '
            '**Style: Sunset-glow, soft bokeh.**\n'
            '2. **@FennecEars**: A Fennec Fox peering through Sahara sand dunes; '
            '**Style: Heat-haze, golden hour, warm tones.**\n'
            '3. **@SlothSlowmo**: A Sloth hanging upside down in Costa Rica; '
            '**Style: Dreamy, soft-focus, low contrast.**\n'
            '4. **@TigerTough**: A Bengal Tiger walking through tall grass; India; '
            '**Style: Dramatic shadows, intense eye contact.**\n'
        )

        prompts = LateFillRuntimeOwner._normalize_late_fill_image_batch_prompts(
            [
                'Extreme macro shot of a Capybara face; Brazil Amazon; **Style: Sunset-glow, soft bokeh.',
                'A Fennec Fox peering through Sahara sand dunes; **Style: Heat-haze, golden hour, warm tones.',
            ],
            expected_count=4,
            content_payload=content_payload,
        )

        self.assertEqual(len(prompts), 4)
        self.assertIn('A Fennec Fox peering through Sahara sand dunes', prompts[1])
        self.assertIn('A Sloth hanging upside down in Costa Rica', prompts[2])
        self.assertIn('A Bengal Tiger walking through tall grass', prompts[3])
        for prompt in prompts:
            self.assertNotIn('@', prompt)
            self.assertNotIn('**Style', prompt)

    def test_composed_page_repair_avoids_single_template_card_container(self):
        content = (
            '<main id="feed">\n'
            '  <div class="image-grid" id="trending">\n'
            '    <!-- Repeatable Asset Block Template: To be populated by 24 assets from Manifest -->\n'
            '    <article class="grid-item">\n'
            '      <div class="image-wrapper">\n'
            '        <img src="../images/existing.png" alt="Luna Paris" loading="lazy">\n'
            '        <div class="image-overlay">\n'
            '          <div class="user-tag">@Luna_Paris\n'
            '          <p class="caption">Midnight in Montmartre</p>\n'
            '        </div>\n'
            '      </div>\n'
            '    </article>\n'
            '  </div>\n'
            '</main>'
        )

        updated, inserted, container_tag = LateFillRuntimeOwner._insert_html_into_existing_image_container(
            content,
            ['../images/missing-02.png', '../images/missing-03.png'],
        )

        self.assertFalse(inserted)
        self.assertEqual(container_tag, '')
        self.assertEqual(updated, content)

    def test_composed_page_repair_still_uses_healthy_multi_image_gallery_container(self):
        content = (
            '<main id="feed">\n'
            '  <section class="feed gallery">\n'
            '    <!-- Sample of the 12 images integrated into a feed -->\n'
            '    <article><img src="../images/one.png" alt="Post 1"></article>\n'
            '    <article><img src="../images/two.png" alt="Post 2"></article>\n'
            '    <article><img src="../images/three.png" alt="Post 3"></article>\n'
            '    <!-- Additional posts would populate here up to 12 -->\n'
            '  </section>\n'
            '</main>'
        )

        updated, inserted, container_tag = LateFillRuntimeOwner._insert_html_into_existing_image_container(
            content,
            ['../images/four.png', '../images/five.png'],
        )

        self.assertTrue(inserted)
        self.assertEqual(container_tag, 'section')
        self.assertIn('../images/four.png', updated)
        self.assertIn('../images/five.png', updated)
        self.assertNotIn('ollmo-generated-media', updated)

    def test_composed_page_repair_avoids_content_rich_repeat_gallery_container(self):
        content = (
            '<main id="feed">\n'
            '  <section class="image-grid gallery">\n'
            '    <!-- The 24 Assets will be injected here -->\n'
            '    <!-- Asset 1 -->\n'
            '    <div class="gallery-card">\n'
            '      <div class="image-wrapper"><img src="../images/one.png" alt="@Panda_Prime"></div>\n'
            '      <div class="card-info"><span class="username">@Panda_Prime</span>'
            '<p class="caption">Bamboo breakfast hits different.</p></div>\n'
            '    </div>\n'
            '    <!-- Asset 2 -->\n'
            '    <div class="gallery-card">\n'
            '      <div class="image-wrapper"><img src="../images/two.png" alt="@Quack_Master"></div>\n'
            '      <div class="card-info"><span class="username">@Quack_Master</span>'
            '<p class="caption">Tulip fields and morning dew.</p></div>\n'
            '    </div>\n'
            '    <!-- ... (Repeated for all 24 assets following the same pattern) ... -->\n'
            '    <!-- For brevity in this preparation phase, the template below covers the structural logic used for all 24 cards -->\n'
            '  </section>\n'
            '</main>'
        )

        updated, inserted, container_tag = LateFillRuntimeOwner._insert_html_into_existing_image_container(
            content,
            ['../images/three.png', '../images/four.png', '../images/five.png'],
        )

        self.assertFalse(inserted)
        self.assertEqual(container_tag, '')
        self.assertEqual(updated, content)

        flat_card_content = (
            '<main><section class="image-grid gallery">'
            '<div class="gallery-card"><img src="../images/one.png" alt="@Panda_Prime">'
            '<span class="username">@Panda_Prime</span><p class="caption">copy</p></div>'
            '<div class="gallery-card"><img src="../images/two.png" alt="@Quack_Master">'
            '<span class="username">@Quack_Master</span><p class="caption">copy</p></div>'
            '<!-- ... (Repeated for all 24 assets following the same pattern) ... -->'
            '<!-- For brevity in this preparation phase, the template below covers the structural logic used for all 24 cards -->'
            '</section></main>'
        )

        flat_updated, flat_inserted, flat_container_tag = (
            LateFillRuntimeOwner._insert_html_into_existing_image_container(
                flat_card_content,
                ['../images/three.png', '../images/four.png', '../images/five.png'],
            )
        )

        self.assertFalse(flat_inserted)
        self.assertEqual(flat_container_tag, '')
        self.assertEqual(flat_updated, flat_card_content)

    def test_composed_page_repair_refuses_detached_section_for_unfinished_template_gallery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents = root / 'documents'
            images = root / 'images'
            documents.mkdir()
            images.mkdir()
            html_path = documents / 'index.html'
            image_paths = [images / f'snout-{index:02d}.png' for index in range(1, 5)]
            for image_path in image_paths:
                image_path.write_bytes(b'png')
            original_html = (
                '<!doctype html><html><body><main id="feed">\n'
                '  <section class="gallery">\n'
                '    <!-- The 4 assets will be rendered here via the following pattern -->\n'
                '    <!-- Item 1 to 4 template -->\n'
                f'    <article class="gallery-card"><img src="../images/{image_paths[0].name}" alt="Post 1"></article>\n'
                '    <!-- ... repeat for all 4 assets ... -->\n'
                '  </section>\n'
                '</main></body></html>'
            )
            html_path.write_text(original_html, encoding='utf-8')
            payload = {
                'id': 'resp_unfinished_template_gallery',
                'artifacts': [
                    {
                        'type': 'text',
                        'path': str(html_path),
                        'artifact_ref': 'artifact:index',
                        'branch_id': 'branch-text_artifact-1',
                        'phase_id': 'phase-5',
                        'text_artifact_extension': 'html',
                        'text_artifact_source_name': 'index',
                    },
                    *[
                        {
                            'type': 'image',
                            'path': str(image_path),
                            'artifact_ref': f'artifact:image-{index}',
                            'branch_id': f'branch-image_generation-{index}',
                            'phase_id': f'phase-{index}',
                        }
                        for index, image_path in enumerate(image_paths, start=1)
                    ],
                ],
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        *[
                            {
                                'branch_id': f'branch-image_generation-{index}',
                                'phase_id': f'phase-{index}',
                                'capability': 'image_generation',
                                'output_type': 'image',
                                'status': 'fulfilled',
                            }
                            for index in range(1, 5)
                        ],
                        {
                            'branch_id': 'branch-text_artifact-1',
                            'phase_id': 'phase-5',
                            'capability': 'chat',
                            'output_type': 'text',
                            'status': 'fulfilled',
                        },
                    ],
                    'fill_results': [
                        *[
                            {
                                'branch_id': f'branch-image_generation-{index}',
                                'phase_id': f'phase-{index}',
                                'capability': 'image_generation',
                                'saved_image_path': str(image_path),
                            }
                            for index, image_path in enumerate(image_paths, start=1)
                        ],
                        {
                            'branch_id': 'branch-text_artifact-1',
                            'phase_id': 'phase-5',
                            'capability': 'chat',
                            'saved_text_path': str(html_path),
                            'text_artifact_extension': 'html',
                            'text_artifact_source_name': 'index',
                        },
                    ],
                },
            }

            updated = self.late_fill_owner._repair_terminal_composed_page_image_representation(payload)

            self.assertEqual(updated, payload)
            self.assertEqual(html_path.read_text(encoding='utf-8'), original_html)
            self.assertNotIn('ollmo-generated-media', html_path.read_text(encoding='utf-8'))

    def test_composed_page_repair_expands_manifest_backed_unfinished_template_gallery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents = root / 'documents'
            images = root / 'images'
            documents.mkdir()
            images.mkdir()
            html_path = documents / 'index.html'
            image_paths = [images / f'snout-{index:02d}.png' for index in range(1, 5)]
            for image_path in image_paths:
                image_path.write_bytes(b'png')
            original_html = (
                '<!doctype html><html><body><main id="feed">\n'
                '  <section id="trending" class="grid-container">\n'
                '    <div class="image-grid">\n'
                '      <!-- Example of a single card structure (to be repeated 4x) -->\n'
                '      <div class="feed-card"><div class="card-image-wrapper">'
                f'<img src="../images/{image_paths[0].name}" alt="@CapyVibes" loading="lazy">'
                '<div class="card-overlay"></div></div><div class="card-content">'
                '<span class="username">@CapyVibes</span><p class="caption">Amazon humidity hits different.</p>'
                '</div></div>\n'
                '      <!-- ... Repeat for all 4 assets ... -->\n'
                '    </div>\n'
                '  </section>\n'
                '</main></body></html>'
            )
            html_path.write_text(original_html, encoding='utf-8')
            content_payload = (
                '### Image Generation Assets: "Global Trending" 4-Set\n\n'
                '1. **@CapyVibes**: Extreme macro shot of a Capybara face; Brazil Amazon; '
                '**Style: Sunset-glow, soft bokeh.**\n'
                '2. **@FennecEars**: A Fennec Fox peering through Sahara sand dunes; '
                '**Style: Heat-haze, golden hour, warm tones.**\n'
                '3. **@SlothSlowmo**: A Sloth hanging upside down in Costa Rica; '
                '**Style: Dreamy, soft-focus, low contrast.**\n'
                '4. **@TigerTough**: A Bengal Tiger walking through tall grass; India; '
                '**Style: Dramatic shadows, intense eye contact.**\n'
            )
            payload = {
                'id': 'resp_manifest_backed_unfinished_template_gallery',
                'artifacts': [
                    {
                        'type': 'text',
                        'path': str(html_path),
                        'artifact_ref': 'artifact:index',
                        'branch_id': 'branch-text_artifact-1',
                        'phase_id': 'phase-6',
                        'text_artifact_extension': 'html',
                        'text_artifact_source_name': 'index',
                    },
                    *[
                        {
                            'type': 'image',
                            'path': str(image_path),
                            'artifact_ref': f'artifact:image-{index}',
                            'branch_id': f'branch-image_generation-{index}',
                            'phase_id': f'phase-{index}',
                        }
                        for index, image_path in enumerate(image_paths, start=1)
                    ],
                ],
                'late_fill': {
                    'status': 'completed',
                    'content_payload': content_payload,
                    'completed_branches': [
                        *[
                            {
                                'branch_id': f'branch-image_generation-{index}',
                                'phase_id': f'phase-{index}',
                                'capability': 'image_generation',
                                'output_type': 'image',
                                'status': 'fulfilled',
                            }
                            for index in range(1, 5)
                        ],
                        {
                            'branch_id': 'branch-text_artifact-1',
                            'phase_id': 'phase-6',
                            'capability': 'chat',
                            'output_type': 'text',
                            'status': 'fulfilled',
                        },
                    ],
                    'fill_results': [
                        *[
                            {
                                'branch_id': f'branch-image_generation-{index}',
                                'phase_id': f'phase-{index}',
                                'capability': 'image_generation',
                                'saved_image_path': str(image_path),
                            }
                            for index, image_path in enumerate(image_paths, start=1)
                        ],
                        {
                            'branch_id': 'branch-text_artifact-1',
                            'phase_id': 'phase-6',
                            'capability': 'chat',
                            'saved_text_path': str(html_path),
                            'text_artifact_extension': 'html',
                            'text_artifact_source_name': 'index',
                        },
                    ],
                },
            }

            updated = self.late_fill_owner._repair_terminal_composed_page_image_representation(payload)
            html = html_path.read_text(encoding='utf-8')

            self.assertNotEqual(updated, payload)
            self.assertNotIn('ollmo-generated-media', html)
            self.assertEqual(html.count('data-ollmo-repair="manifest-backed-gallery-expansion"'), 4)
            for image_path in image_paths:
                self.assertIn(f'../images/{image_path.name}', html)
            self.assertIn('@FennecEars', html)
            self.assertIn('@SlothSlowmo', html)
            self.assertIn('@TigerTough', html)

    def test_truth_gate_rewrites_visible_control_json(self):
        payload = {
            'id': 'resp_control_json_leak',
            'output_text': (
                '{\n'
                '  "request_ir": {"output_obligations": []},\n'
                '  "candidate_graph": {"type_counts": {"output": 1}}\n'
                '}'
            ),
        }

        updated = self.owner.truth_gate_response_output_claims(payload)

        self.assertNotIn('request_ir', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['kind'], 'control_json_boundary')
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'repair_required')
        self.assertEqual(
            updated['runtime']['phase_output_acceptance']['runtime_effect'],
            'materialization_blocked',
        )

    def test_truth_gate_unwraps_single_control_json_content_payload(self):
        payload = {
            'id': 'resp_control_json_content_payload',
            'output_text': (
                '```json\n'
                '{\n'
                '  "request_ir": {\n'
                '    "output_obligations": [\n'
                '      {\n'
                '        "output_type": "text",\n'
                '        "content_payload": "Bitte lade ein Bild hoch oder referenziere eines."\n'
                '      }\n'
                '    ]\n'
                '  }\n'
                '}\n'
                '```'
            ),
        }

        updated = self.owner.truth_gate_response_output_claims(payload)

        self.assertEqual(updated['output_text'], 'Bitte lade ein Bild hoch oder referenziere eines.')
        self.assertNotIn('request_ir', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['kind'], 'control_json_boundary')
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'unwrapped')
        self.assertEqual(
            updated['runtime']['phase_output_acceptance']['runtime_effect'],
            'canonical_phase_text',
        )

    def test_phase_output_acceptance_rejects_process_prose_in_user_facing_response(self):
        source = json.dumps(
            {
                'decision_contract': {'status': 'ready'},
                'request_phase_graph': {'current_phase_id': 'phase-1'},
                'request_ir': {'output_obligations': []},
                'user_facing_response': 'First I will prepare the response and then run TTS.',
            }
        )

        classified = classify_phase_output_text(source)
        metadata = phase_output_acceptance_metadata([classified])

        self.assertEqual(classified['status'], 'repair_required')
        self.assertEqual(classified['accepted_text'], '')
        self.assertEqual(metadata['runtime_effect'], 'materialization_blocked')
        self.assertNotIn(source, json.dumps(metadata))

    def test_phase_output_acceptance_unwraps_only_one_safe_content_payload(self):
        source = json.dumps(
            {
                'request_ir': {
                    'output_obligations': [
                        {'content_payload': 'Nur dieser Satz wird gesprochen.'},
                    ]
                },
                'user_facing_response': 'Process narration must not win.',
            }
        )

        classified = classify_phase_output_text(source)

        self.assertEqual(classified['status'], 'unwrapped')
        self.assertEqual(classified['accepted_text'], 'Nur dieser Satz wird gesprochen.')
        self.assertEqual(
            classified['accepted_sha256'],
            hashlib.sha256('Nur dieser Satz wird gesprochen.'.encode('utf-8')).hexdigest(),
        )

    def test_phase_output_acceptance_rejects_ambiguous_content_payloads(self):
        source = json.dumps(
            {
                'request_ir': {
                    'output_obligations': [
                        {'content_payload': 'Erste Fassung.'},
                        {'content_payload': 'Zweite Fassung.'},
                    ]
                }
            }
        )

        classified = classify_phase_output_text(source)

        self.assertEqual(classified['status'], 'repair_required')
        self.assertEqual(classified['accepted_text'], '')

    def test_phase_output_acceptance_rejects_duplicate_identical_content_payload_occurrences(self):
        source = json.dumps(
            {
                'request_ir': {
                    'output_obligations': [
                        {'content_payload': 'Identischer Satz.'},
                        {'content_payload': 'Identischer Satz.'},
                    ]
                }
            }
        )

        classified = classify_phase_output_text(source)

        self.assertEqual(classified['status'], 'repair_required')
        self.assertEqual(classified['accepted_text'], '')

    def test_phase_output_acceptance_rejects_empty_preparation_text(self):
        classified = classify_phase_output_text('   ')

        self.assertEqual(classified['status'], 'repair_required')
        self.assertEqual(classified['accepted_text'], '')
        self.assertEqual(classified['source_bytes'], 0)

    def test_phase_output_acceptance_rejects_truncated_control_envelope(self):
        source = '{"decision_contract":{"status":"ready"},"request_ir":{"output_obligations":['

        classified = classify_phase_output_text(source)

        self.assertEqual(classified['status'], 'repair_required')
        self.assertEqual(classified['accepted_text'], '')
        self.assertIn('truncated', classified['reason'])

    def test_truth_gate_allows_explicit_control_diagnostic_without_materializer(self):
        source = json.dumps({'request_ir': {'output_obligations': []}})

        updated = self.owner.truth_gate_response_output_claims(
            {'id': 'resp_control_diagnostic', 'output_text': source, 'capability': 'chat'},
            request_payload={'prompt': 'Show the request_ir as planner JSON for debugging.'},
        )

        self.assertEqual(updated['output_text'], source)
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_rewrites_fenced_visible_control_json(self):
        payload = {
            'id': 'resp_fenced_control_json_leak',
            'output_text': (
                '```json\n'
                '{\n'
                '  "request_phase_graph": {"mode": "single_phase"},\n'
                '  "request_ir": {"output_obligations": []}\n'
                '}\n'
                '```'
            ),
        }

        updated = self.owner.truth_gate_response_output_claims(payload)

        self.assertNotIn('request_phase_graph', updated['output_text'])
        self.assertNotIn('request_ir', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['kind'], 'control_json_boundary')

    def test_truth_gate_rewrites_ungrounded_this_artifact_request_to_clarification(self):
        payload = {
            'id': 'resp_ungrounded_this_artifact',
            'output_text': (
                'Since you are referring to the previous control center, '
                'I have bundled it as an HTML artifact.'
            ),
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={'prompt': 'Generate me this html file as artifact'},
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['kind'], 'ungrounded_text_artifact_reference')
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_does_not_treat_loose_reference_history_as_current_source(self):
        payload = {
            'id': 'resp_loose_history_this_artifact',
            'output_text': 'Here is this HTML file as an artifact.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Generate me this html file as artifact',
                'reference_artifacts': [
                    {'type': 'message', 'artifact_ref': 'artifact:old-html', 'content': '<html></html>'}
                ],
            },
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_preserves_all_typed_predecessor_references_for_r5_follow_up(self):
        prompt = (
            'Beziehe dich ausdrücklich auf das Observatorium-Bild, seine Bildanalyse, die deutsche '
            'Erzählung und das Audio aus dem unmittelbar vorherigen Turn. Bewahre Bild und Bildanalyse '
            'unverändert; erzeuge das Bild nicht neu und analysiere es nicht erneut. Ersetze den bisherigen '
            'einzelnen Audiozweig durch zwei getrennte Audiofassungen: einmal die ursprüngliche deutsche '
            'Erzählung und einmal eine getreue englische Übersetzung. Transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein neues JSON-Objekt aus, das die unveränderte Bildevidenz '
            'sowie beide Audio-artifact_refs und beide realen Transkripte eindeutig verbindet.'
        )
        references = [
            {
                'type': 'message',
                'message_role': 'assistant',
                'message_id': 'msg_r5_root',
                'source_response_id': 'resp_r5_root',
                'content': 'Prior grounded image analysis and German narration.',
            },
            {
                'type': 'image',
                'artifact_ref': 'artifact:image_r5_root',
                'path': '/artifacts/images/r5-root.png',
                'source_message_id': 'msg_r5_root',
                'source_response_id': 'resp_r5_root',
            },
            {
                'type': 'audio',
                'artifact_ref': 'artifact:audio_r5_root',
                'path': '/artifacts/audio/r5-root.mp3',
                'source_message_id': 'msg_r5_root',
                'source_response_id': 'resp_r5_root',
            },
            {
                'type': 'text',
                'artifact_ref': 'artifact:text_r5_root',
                'path': '/artifacts/transcripts/r5-root.md',
                'source_message_id': 'msg_r5_root',
                'source_response_id': 'resp_r5_root',
            },
        ]

        def compatibility_selected_reference_sanitizer(raw):
            raw_items = raw if isinstance(raw, list) else [raw]
            message_reference = None
            artifact_reference = None
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'message':
                    message_reference = dict(item)
                else:
                    artifact_reference = dict(item)
            return [
                item
                for item in (message_reference, artifact_reference)
                if isinstance(item, dict)
            ]

        self.owner.hooks['sanitize_selected_reference_artifacts'] = (
            compatibility_selected_reference_sanitizer
        )
        self.owner.hooks['get_response_lookup_record'] = lambda response_id: {
            'id': response_id,
            'message_id': 'msg_r5_root',
            'lifecycle_state': 'completed',
            'response_payload': {
                'lifecycle_state': 'completed',
                'output_text': references[0]['content'],
                'artifacts': [dict(item) for item in references[1:]],
                'response_frame': {
                    'request': {'conversation_id': 'conv_r5'},
                },
            },
        }
        try:
            updated = self.owner.truth_gate_response_output_claims(
                {
                    'id': 'resp_r5_follow_truth_sources',
                    'output_text': 'Prepared the requested final evidence join.',
                },
                request_payload={
                    'ghost_route': True,
                    'conversation_id': 'conv_r5',
                    'prompt': prompt,
                    'ghost_messages': [
                        {
                            'role': 'assistant',
                            'response_id': 'resp_r5_root',
                            'message_id': 'msg_r5_root',
                            'content': references[0]['content'],
                            'artifacts': [dict(item) for item in references[1:]],
                        }
                    ],
                    'reference_artifacts': references,
                },
            )
        finally:
            self.owner.hooks.pop('sanitize_selected_reference_artifacts', None)
            self.owner.hooks.pop('get_response_lookup_record', None)

        self.assertEqual(
            updated['output_text'],
            'Prepared the requested final evidence join.',
        )
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_accepts_repair_needed_predecessor_image_prompt_bundle(self):
        prompt = (
            'can you please create the images and link them to the site properly? '
            'thank you.'
        )
        image_prompts = [
            f'Concrete community garden image prompt {index}.'
            for index in range(1, 6)
        ]
        references = [
            {
                'type': 'message',
                'message_role': 'assistant',
                'message_id': 'msg-site-root',
                'source_message_id': 'msg-site-root',
                'source_response_id': 'resp-site-root',
                'artifact_ref': 'msg-site-root',
                'content': 'Artifact generated.',
            },
            {
                'type': 'text',
                'artifact_ref': 'artifact:site-root',
                'path': '/artifacts/documents/community-garden.html',
                'source_message_id': 'msg-site-root',
                'source_response_id': 'resp-site-root',
            },
            {
                'type': 'text',
                'artifact_ref': 'artifact:site-stream-root',
                'path': '/artifacts/documents/community-garden-stream.html',
                'source_message_id': 'msg-site-root',
                'source_response_id': 'resp-site-root',
            },
        ]

        def compatibility_selected_reference_sanitizer(raw):
            raw_items = raw if isinstance(raw, list) else [raw]
            message_reference = None
            artifact_reference = None
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'message':
                    message_reference = dict(item)
                else:
                    artifact_reference = dict(item)
            return [
                item
                for item in (message_reference, artifact_reference)
                if isinstance(item, dict)
            ]

        predecessor_payload = {
            'lifecycle_state': 'repair_needed',
            'output_text': references[0]['content'],
            'artifacts': [dict(item) for item in references[1:]],
            'late_fill': {'batch_prompts': image_prompts},
            'response_frame': {
                'request': {'conversation_id': 'conv-site-repair'},
            },
        }
        self.owner.hooks['sanitize_selected_reference_artifacts'] = (
            compatibility_selected_reference_sanitizer
        )
        self.owner.hooks['get_response_lookup_record'] = lambda response_id: {
            'id': response_id,
            'message_id': 'msg-site-root',
            'lifecycle_state': 'repair_needed',
            'response_payload': predecessor_payload,
        }
        try:
            updated = self.owner.truth_gate_response_output_claims(
                {
                    'id': 'resp-site-follow-up',
                    'output_text': 'The five image specifications are ready.',
                },
                request_payload={
                    'ghost_route': True,
                    'conversation_id': 'conv-site-repair',
                    'prompt': prompt,
                    'ghost_messages': [
                        {
                            'role': 'assistant',
                            'response_id': 'resp-site-root',
                            'message_id': 'msg-site-root',
                            'content': references[0]['content'],
                            'artifacts': [dict(item) for item in references[1:]],
                        }
                    ],
                    'reference_artifacts': references,
                    'current_predecessor_context': {
                        'status': 'authorized',
                        'authorization': (
                            'canonical_same_conversation_predecessor'
                        ),
                        'source_response_id': 'resp-site-root',
                        'source_message_id': 'msg-site-root',
                        'batch_prompts': image_prompts,
                        'text_artifact_refs': [
                            'artifact:site-root',
                            'artifact:site-stream-root',
                        ],
                    },
                },
            )
        finally:
            self.owner.hooks.pop('sanitize_selected_reference_artifacts', None)
            self.owner.hooks.pop('get_response_lookup_record', None)

        self.assertEqual(
            updated['output_text'],
            'The five image specifications are ready.',
        )
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_rewrites_ungrounded_daraus_html_audio_to_clarification(self):
        payload = {
            'id': 'resp_ungrounded_daraus_transform',
            'output_text': '<!DOCTYPE html><html><body><h1>Invented source</h1></body></html>',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={'prompt': 'Mach daraus ein HTML und ein Audio.'},
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertNotIn('<!DOCTYPE html>', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['kind'], 'ungrounded_text_artifact_reference')
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_allows_same_turn_generated_scene_source_for_multimodal_transform(self):
        prompt = (
            'Schreibe einen deutschen Szenentext mit genau zwanzig Wörtern über einen '
            'Leuchtturm im Sturm. Erzeuge daraus parallel ein Bild und ein Audio. '
            'Analysiere danach das tatsächlich erzeugte Bild und transkribiere das '
            'tatsächlich erzeugte Audio. Vergleiche abschließend im Chat anhand genau '
            'dieser beiden realen Evidenzzweige, ob Leuchtturm, Sturm und Nacht in '
            'beiden vorkommen. Bildanalyse darf nur vom Bild, Transkription nur vom Audio, '
            'der Schluss nur von beiden Evidenzzweigen abhängen.'
        )
        prepared_scene = (
            'Nachts trotzt der alte Leuchtturm unbeirrt peitschendem Sturm, während schwarze '
            'Wellen donnern und sein Licht mutig durch dichten Regen schneidet.'
        )
        payload = {
            'id': 'resp_same_turn_generated_multimodal_source',
            'output_text': prepared_scene,
        }
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload=request_payload,
        )

        self.assertEqual(updated['output_text'], prepared_scene)
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_graph_proven_same_turn_image_for_vision(self):
        prompt = (
            'Erzeuge ein lokales Bild eines kleinen Observatoriums bei klarem Nachthimmel '
            'und analysiere danach nur sichtbare Details dieses Bildes. Schreibe außerdem '
            'eine deutsche Erzählung aus genau zwei kurzen Sätzen, erzeuge daraus ein Audio '
            'und transkribiere das tatsächlich erzeugte Audio. Gib abschließend ein JSON-Objekt '
            'aus, das den Bild-artifact_ref, die sichtbare Bildevidenz, den Audio-artifact_ref '
            'und das reale Transkript getrennt bindet.'
        )
        payload = {
            'id': 'resp_same_turn_image_vision_source',
            'output_text': 'Unter der klaren Nachtkuppel steht ein kleines Observatorium.',
        }
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )
        route_payload = {'route_runtime': {'request_phase_graph': phase_graph}}

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            route_payload=route_payload,
            request_payload=request_payload,
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_graph_proven_same_turn_audio_for_transcription(self):
        prompt = (
            'Erzeuge ein Audio mit einem kurzen Warnton und transkribiere danach '
            'dieses Audio.'
        )
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )
        payload = {
            'id': 'resp_same_turn_audio_stt_source',
            'output_text': 'Ein kurzer Warnton wird vorbereitet.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload=request_payload,
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_graph_grounding_requires_promoted_exact_dependency(self):
        prompt = (
            'Erzeuge ein lokales Bild eines kleinen Observatoriums und analysiere danach '
            'nur sichtbare Details dieses Bildes.'
        )
        request_payload = {'ghost_route': True, 'prompt': prompt}
        base_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )

        def mutate_second_image_producer(graph):
            producer = next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            )
            duplicate_phase = json.loads(json.dumps(producer))
            duplicate_phase.update(
                {
                    'phase_id': 'phase-second-image',
                    'branch_id': 'branch-second-image',
                    'obligation_id': 'obligation-second-image',
                }
            )
            graph['phases'].append(duplicate_phase)
            producer_candidate = next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'image_generation'
            )
            duplicate_candidate = json.loads(json.dumps(producer_candidate))
            duplicate_candidate.update(
                {
                    'candidate_id': 'candidate-output-second-image',
                    'phase_id': 'phase-second-image',
                    'branch_id': 'branch-second-image',
                    'contract_ref': 'obligation-second-image',
                    'obligation_id': 'obligation-second-image',
                }
            )
            graph['candidate_graph']['candidates'].append(duplicate_candidate)
            consumer = next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'vision_analysis'
            )
            consumer['depends_on'].append('phase-second-image')
            consumer['input_refs'].append(
                {
                    'kind': 'phase_output',
                    'phase_id': 'phase-second-image',
                    'role': 'dependency',
                }
            )
            consumer_candidate = next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'vision_analysis'
            )
            consumer_candidate['depends_on'].append('phase-second-image')

        def mutate_dependency_cycle(graph):
            producer = next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            )
            consumer = next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'vision_analysis'
            )
            producer['depends_on'] = [consumer['phase_id']]
            producer['input_refs'] = [
                {
                    'kind': 'phase_output',
                    'phase_id': consumer['phase_id'],
                    'role': 'dependency',
                }
            ]
            producer_candidate = next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'image_generation'
            )
            producer_candidate['depends_on'] = [consumer['phase_id']]

        def add_dangling_context_input(graph):
            consumer = next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'vision_analysis'
            )
            consumer['input_refs'].append(
                {
                    'kind': 'phase_output',
                    'phase_id': 'phase-missing',
                    'role': 'context',
                }
            )

        mutations = {
            'producer_reserved': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            ).update({'resolution': 'reserved_until_promoted'}),
            'consumer_not_promoted': lambda graph: next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'vision_analysis'
            ).update({'status': 'candidate', 'execution_policy': 'non_executable_until_promoted'}),
            'consumer_input_ref_missing': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'vision_analysis'
            ).update({'input_refs': []}),
            'producer_input_ref_missing': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            ).update({'input_refs': []}),
            'consumer_candidate_dependency_mismatch': lambda graph: next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'vision_analysis'
            ).update({'depends_on': ['phase-1']}),
            'consumer_dangling_dependency': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'vision_analysis'
            ).update(
                {
                    'depends_on': ['phase-missing'],
                    'input_refs': [
                        {
                            'kind': 'phase_output',
                            'phase_id': 'phase-missing',
                            'role': 'dependency',
                        }
                    ],
                }
            ),
            'consumer_dangling_context_input': add_dangling_context_input,
            'dependency_cycle': mutate_dependency_cycle,
            'phases_not_list': lambda graph: graph.update({'phases': 1}),
            'candidates_not_list': lambda graph: graph['candidate_graph'].update(
                {'candidates': 1}
            ),
            'producer_failed': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            ).update({'status': 'failed'}),
            'producer_superseded': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            ).update({'status': 'superseded'}),
            'producer_optional': lambda graph: next(
                phase
                for phase in graph['phases']
                if phase.get('capability') == 'image_generation'
            )['output_contract'].update({'required': False}),
            'candidate_contract_ref_missing': lambda graph: next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'image_generation'
            ).update({'contract_ref': ''}),
            'candidate_obligation_mismatch': lambda graph: next(
                candidate
                for candidate in graph['candidate_graph']['candidates']
                if candidate.get('candidate_type') == 'output'
                and candidate.get('capability') == 'vision_analysis'
            ).update({'obligation_id': 'obligation-wrong'}),
            'consumer_input_role_context': lambda graph: next(
                input_ref
                for phase in graph['phases']
                if phase.get('capability') == 'vision_analysis'
                for input_ref in phase.get('input_refs') or []
                if input_ref.get('phase_id')
            ).update({'role': 'context'}),
            'consumer_has_two_image_producers': mutate_second_image_producer,
        }
        for token in (
            'possible',
            'draft',
            'not_promoted',
            'not-promoted',
            'unpromoted',
            'discarded',
        ):
            mutations[f'producer_resolution_{token}'] = (
                lambda graph, value=token: next(
                    phase
                    for phase in graph['phases']
                    if phase.get('capability') == 'image_generation'
                ).update({'resolution': value})
            )
        for token in (
            'reserved',
            'failed',
            'waived',
            'superseded',
            'optional',
            'discarded',
        ):
            mutations[f'producer_output_contract_{token}'] = (
                lambda graph, value=token: next(
                    phase
                    for phase in graph['phases']
                    if phase.get('capability') == 'image_generation'
                )['output_contract'].update({'status': value})
            )
        for token in ('reserved', 'possible', 'draft', 'discarded', 'rejected'):
            mutations[f'candidate_contract_state_{token}'] = (
                lambda graph, value=token: next(
                    candidate
                    for candidate in graph['candidate_graph']['candidates']
                    if candidate.get('candidate_type') == 'output'
                    and candidate.get('capability') == 'image_generation'
                ).update({'contract_state': value})
            )
        for token in ('audio', 'text', 'file'):
            mutations[f'producer_output_type_{token}'] = (
                lambda graph, value=token: next(
                    phase
                    for phase in graph['phases']
                    if phase.get('capability') == 'image_generation'
                ).update({'output_type': value})
            )
        for token in ('postprocess', 'prepare', 'evidence'):
            mutations[f'producer_kind_{token}'] = (
                lambda graph, value=token: next(
                    phase
                    for phase in graph['phases']
                    if phase.get('capability') == 'image_generation'
                ).update({'kind': value})
            )

        for label, mutate in mutations.items():
            with self.subTest(label=label):
                phase_graph = json.loads(json.dumps(base_graph))
                mutate(phase_graph)
                updated = self.owner.truth_gate_response_output_claims(
                    {
                        'id': f'resp_unproven_same_turn_image_{label}',
                        'output_text': 'Invented image analysis source.',
                    },
                    route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
                    request_payload=request_payload,
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_graph_grounding_rejects_competing_external_image(self):
        prompt = (
            'Erzeuge ein neues Bild einer Sternwarte und analysiere danach dieses alte Bild.'
        )
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )

        updated = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_competing_external_image_graph',
                'output_text': 'Invented analysis of an old image.',
            },
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload=request_payload,
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(
            updated['runtime']['truth_guard']['status'],
            'clarification_required',
        )

    def test_truth_gate_graph_grounding_binds_current_reference_and_rejects_external_sources(self):
        prompts = (
            'Generate an image and analyze this image. Then convert this image from a URL into audio.',
            'Generate an image and analyze this image, then from this remote image make audio.',
            'Generate an image and analyze this image, then from this image supplied by Alice make audio.',
            'Generate an image and analyze this image, then from this image at ./source.png make audio.',
            'Generate an image and analyze this remote image.',
            'Generate an image and analyze this image from https://example.test/source.png.',
            'Generate an image and analyze this image from ./external.png.',
            'Generate an image and analyze this image from ../external.png.',
            'Generate an image and analyze this image from source.png.',
            'Generate an image and analyze this image provided by Alice.',
            'Generate an image and analyze this image stored in S3.',
            'Generate an image and analyze this image at C:\\Temp\\source.png.',
            'Generate an image and analyze this generated image and this remote image.',
            'Generate an image and analyze this image, then inspect this remote image.',
            'Create an audio recording and transcribe this audio and this other audio.',
            'Generate an image and analyze this image from data:image/png;base64,AAAA.',
            'Generate an image and analyze this image from artifact://old-image.',
            'Generate an image and analyze this image from ftp://example.test/id.',
            'Generate an image and analyze this image from ipfs://bafy.',
            'Generate an image and analyze this image from gs://bucket/key.',
            'Generate an image and analyze this image from smb://host/share.',
            'Generate an image and analyze this image from blob:deadbeef.',
            'Generate an image and analyze this image from urn:uuid:1234.',
            'Generate an image and analyze this image from ~/old-image.',
            'Generate an image and analyze this image from \\\\server\\share\\old-image.',
            'Generate an image and analyze this image from host.example:old-image.',
            'Generate an image and analyze this image from user@host.example:old-image.',
            'Generate an image and analyze this image from $HOME/old-image.',
            'Generate an image and analyze this image. Then inspect this remote image.',
            'Generate an image and analyze this image. Review this other image.',
            'Generate an image. Inspect this remote image.',
            'Create an audio recording and transcribe this audio. Then review this other audio.',
            'Create an audio recording and transcribe this remote audio.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                request_payload = {'ghost_route': True, 'prompt': prompt}
                phase_graph = build_request_phase_graph(
                    prompt,
                    request_payload=request_payload,
                    route_payload={'capability': 'chat'},
                )
                updated = self.owner.truth_gate_response_output_claims(
                    {
                        'id': 'resp_external_graph_source',
                        'output_text': 'Invented external source result.',
                    },
                    route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
                    request_payload=request_payload,
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_graph_grounding_accepts_bounded_same_turn_variants(self):
        prompts = (
            'Create an audio recording and transcribe it.',
            'For this test, create an image of a lighthouse and analyze this image.',
            'For this experiment, create an image of a lighthouse and analyze this image.',
            'As a quick test, create an image of a lighthouse and analyze this image.',
            'As part of this test, create an image of a lighthouse and analyze this image.',
            'Step 1: create an image of a lighthouse and analyze this image.',
            'To begin, create an image of a lighthouse and analyze this image.',
            'Für diesen Versuch erzeuge ein Bild und analysiere dieses Bild.',
            'Im ersten Schritt erzeuge ein Bild und analysiere dieses Bild.',
            'Generate an image and analyze this image using a different visibility metric.',
            'Generate an image and analyze this image using rubric:contrast.',
            'Generate an image and analyze this image using style:cinematic.',
            'Generate an image and analyze this image with note:bright.',
            'Generate an image and analyze this image using metric:visibility.',
            'Generate an image. Then analyze this image.',
            'Create an audio recording. Then transcribe this audio.',
            'Generate an image, analyze this image, and describe it.',
            'Create an audio recording, transcribe this audio, and review it.',
            'Erzeuge ein lokales Bild und analysiere dieses lokal erzeugte Bild.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                request_payload = {'ghost_route': True, 'prompt': prompt}
                phase_graph = build_request_phase_graph(
                    prompt,
                    request_payload=request_payload,
                    route_payload={'capability': 'chat'},
                )
                payload = {
                    'id': 'resp_bounded_same_turn_variant',
                    'output_text': 'Prepared content for the proven same-turn graph.',
                }
                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
                    request_payload=request_payload,
                )

                self.assertEqual(updated['output_text'], payload['output_text'])
                self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_ignores_non_actionable_or_inline_textual_consumer_mentions(self):
        prompts = (
            'Review this image prompt: a lighthouse at dawn. Then generate it.',
            'Review this image concept: a red lighthouse; then generate an image from it.',
            'Review this audio script: Hello world. Then create an audio file from it.',
            'Describe this image idea: a fox in snow. Then generate it.',
            'Review this HTML: <p>Hello</p>. Then improve the page.',
            'This image prompt: a red lighthouse in snow. Generate an image from it.',
            'This audio script: Hello from Zurich. Generate audio from it.',
            'Script: Hello from Zurich. Turn this into audio.',
            'Bildprompt: ein roter Leuchtturm im Schnee. Erzeuge daraus ein Bild.',
            'Bildkonzept: ein roter Leuchtturm im Schnee. Erzeuge daraus ein Bild.',
            'Audioskript: Hallo Zürich. Erzeuge daraus ein Audio.',
            'Skript: Hallo Zürich. Erzeuge daraus ein Audio.',
            'Dokument: <p>Hallo</p>. Erzeuge daraus eine HTML-Datei.',
            'Explain why "Review this image" is ambiguous.',
            'Explain the instruction `Inspect this image` without executing it.',
            'If a user says “Review this audio”, what should the assistant do?',
            'Give an example sentence: Review this image.',
            'Do not review this image.',
            'Never inspect this audio.',
            'Do not, in this image, analyze the visible objects.',
            "Don't, using this image, analyze the visible objects.",
            'Never, with this audio, transcribe every spoken word.',
            'Bitte nicht, in diesem Bild, die Objekte analysieren.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_non_actionable_consumer_mention',
                    'output_text': 'Bounded response to the textual instruction.',
                }
                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertEqual(updated['output_text'], payload['output_text'])
                self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_deictic_consumer_requires_local_artifact_binding(self):
        prompts = (
            'Review this plan, then generate an image.',
            'Describe this approach and make an HTML file.',
            'Review this JSON schema and create an image.',
            'Inspect this configuration, then create a website.',
            'Listen, I think this plan should create an image.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(
                    _request_requires_current_source_for_transform(
                        prompt,
                        {'prompt': prompt},
                        None,
                    )
                )

    def test_truth_gate_guards_reverse_and_plural_missing_artifact_references(self):
        prompts = (
            'In this image, analyze the visible objects.',
            'On this image, inspect the visible objects.',
            'For this audio, transcribe every spoken word.',
            'From this audio, transcribe every spoken word.',
            'Analysiere die sichtbaren Objekte in diesem Bild.',
            'Transkribiere alle Wörter aus diesem Audio.',
            'Generate three images, then analyze them.',
            'Erzeuge drei Bilder und analysiere sie.',
            'Generate an image of a lighthouse in a violent winter storm with dark clouds, '
            'white spray, sharp rocks, and a distant rescue boat, then analyze it.',
            'Paint an image of a lighthouse and analyze it.',
            'Illustrate a lighthouse and inspect this image.',
            'Create a photo and analyze it.',
            'Erzeuge ein Foto und analysiere es.',
            'Male ein Bild von einem Leuchtturm und analysiere es.',
            'Erzeuge ein Audio und transkribiere es.',
            'Erzeuge eine Sprachaufnahme und transkribiere sie.',
            'Create a recording and transcribe it.',
            'Synthesize a voice clip and transcribe it.',
            'Produce an audio warning and transcribe it.',
            'Use this image to create audio.',
            'Use this audio to create an image.',
            'With this audio, create an image.',
            'Given this image, create audio.',
            'Mit diesem Audio erzeuge ein Bild.',
            'Aus diesem Bild erzeuge ein Audio.',
            'This image prompt: a red lighthouse. Transcribe this audio.',
            'This audio script: Hello Zurich. Analyze this image.',
            'This HTML: <p>Hello</p>. Convert this image into audio.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                updated = self.owner.truth_gate_response_output_claims(
                    {
                        'id': 'resp_missing_reverse_or_plural_source',
                        'output_text': 'Invented evidence from an absent source.',
                    },
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

        graph_prompt = 'Generate three images and analyze these images.'
        graph_request = {'ghost_route': True, 'prompt': graph_prompt}
        phase_graph = build_request_phase_graph(
            graph_prompt,
            request_payload=graph_request,
            route_payload={'capability': 'chat'},
        )
        graph_grounded = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_graph_grounded_plural_pronoun',
                'output_text': 'Prepared for all three graph-grounded images.',
            },
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload=graph_request,
        )

        self.assertEqual(
            graph_grounded['output_text'],
            'Prepared for all three graph-grounded images.',
        )
        self.assertNotIn('truth_guard', graph_grounded.get('runtime') or {})

    def test_truth_gate_direct_source_must_match_deictic_modality(self):
        cases = (
            ('Convert this image into audio.', '/tmp/unrelated.wav', True),
            ('Convert this image into audio.', '/tmp/page.html', True),
            ('Convert this image into audio.', '/tmp/source.png', False),
            ('Convert this audio into an image.', '/tmp/source.png', True),
            ('Convert this audio into an image.', '/tmp/source.wav', False),
        )
        for prompt, source_path, clarification_expected in cases:
            with self.subTest(prompt=prompt, source_path=source_path):
                payload = {
                    'id': 'resp_typed_direct_source',
                    'output_text': 'Prepared from the selected source.',
                }
                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={
                        'prompt': prompt,
                        'selected_reference_artifacts': [{'path': source_path}],
                    },
                )

                if clarification_expected:
                    self.assertIn('need the source/content', updated['output_text'])
                    self.assertEqual(
                        updated['runtime']['truth_guard']['status'],
                        'clarification_required',
                    )
                else:
                    self.assertEqual(updated['output_text'], payload['output_text'])
                    self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_direct_source_rejects_wrong_text_modality_and_conflicting_metadata(self):
        cases = (
            ('Convert this HTML into audio.', {'path': '/tmp/source.png'}, True),
            ('Convert this HTML into audio.', {'path': '/tmp/source.html'}, False),
            ('Convert this code into audio.', {'type': 'audio', 'path': '/tmp/source.wav'}, True),
            ('Convert this document into audio.', {'type': 'document', 'content': 'Source'}, False),
            ('Convert this image into audio.', {'type': 'audio', 'path': '/tmp/source.png'}, True),
            ('Convert this audio into an image.', {'type': 'image', 'path': '/tmp/source.wav'}, True),
        )
        for prompt, source_record, clarification_expected in cases:
            with self.subTest(prompt=prompt, source_record=source_record):
                payload = {
                    'id': 'resp_typed_direct_source_record',
                    'output_text': 'Prepared from the selected source.',
                }
                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={
                        'prompt': prompt,
                        'selected_reference_artifacts': [source_record],
                    },
                )

                if clarification_expected:
                    self.assertIn('need the source/content', updated['output_text'])
                    self.assertEqual(
                        updated['runtime']['truth_guard']['status'],
                        'clarification_required',
                    )
                else:
                    self.assertEqual(updated['output_text'], payload['output_text'])
                    self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_direct_source_requires_each_declared_modality_and_occurrence(self):
        missing_source_prompts = (
            'Convert this image and this audio into HTML.',
            'Convert this image and audio into HTML.',
            'Convert this image and this other image into HTML.',
        )
        for prompt in missing_source_prompts:
            with self.subTest(prompt=prompt):
                updated = self.owner.truth_gate_response_output_claims(
                    {
                        'id': 'resp_incomplete_direct_source_set',
                        'output_text': 'Invented content for an absent source.',
                    },
                    request_payload={
                        'prompt': prompt,
                        'input_artifacts': [
                            {'type': 'image', 'path': '/tmp/source-one.png'},
                        ],
                    },
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

        complete = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_complete_direct_source_set',
                'output_text': 'Prepared from both selected sources.',
            },
            request_payload={
                'prompt': 'Convert this image and this audio into HTML.',
                'input_artifacts': [
                    {'type': 'image', 'path': '/tmp/source.png'},
                    {'type': 'audio', 'path': '/tmp/source.wav'},
                ],
            },
        )

        self.assertEqual(complete['output_text'], 'Prepared from both selected sources.')
        self.assertNotIn('truth_guard', complete.get('runtime') or {})

    def test_truth_gate_plural_references_require_multiple_sources(self):
        cases = (
            ('Convert these images into HTML.', {'type': 'image', 'path': '/tmp/one.png'}),
            ('Review these images.', {'type': 'image', 'path': '/tmp/one.png'}),
            ('Inspect those images.', {'type': 'image', 'path': '/tmp/one.png'}),
            ('Transcribe these audio files.', {'type': 'audio', 'path': '/tmp/one.wav'}),
            ('Review those audio recordings.', {'type': 'audio', 'path': '/tmp/one.wav'}),
        )
        for prompt, source_record in cases:
            with self.subTest(prompt=prompt):
                updated = self.owner.truth_gate_response_output_claims(
                    {
                        'id': 'resp_plural_source_shortfall',
                        'output_text': 'Invented result for missing plural inputs.',
                    },
                    request_payload={
                        'prompt': prompt,
                        'input_artifacts': [source_record],
                    },
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

        complete = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_plural_source_complete',
                'output_text': 'Prepared from both images.',
            },
            request_payload={
                'prompt': 'Convert these images into HTML.',
                'input_artifacts': [
                    {'type': 'image', 'path': '/tmp/one.png'},
                    {'type': 'image', 'path': '/tmp/two.png'},
                ],
            },
        )

        self.assertEqual(complete['output_text'], 'Prepared from both images.')
        self.assertNotIn('truth_guard', complete.get('runtime') or {})

    def test_truth_gate_plural_sources_deduplicate_identity_across_payloads(self):
        source = {
            'type': 'image',
            'path': '/tmp/shared-source.png',
            'artifact_ref': 'artifact:shared-source',
        }
        cases = (
            ([source, dict(source)], []),
            ([source], [dict(source)]),
            (
                [
                    dict(source),
                    {
                        'type': 'image',
                        'path': '/tmp/shared-source.png',
                        'artifact_ref': 'artifact:different-alias',
                    },
                ],
                [],
            ),
        )
        for request_sources, response_sources in cases:
            with self.subTest(
                request_sources=request_sources,
                response_sources=response_sources,
            ):
                payload = {
                    'id': 'resp_duplicate_plural_source',
                    'output_text': 'Invented result for a duplicated source.',
                }
                if response_sources:
                    payload['input_artifacts'] = response_sources
                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={
                        'prompt': 'Analyze these images.',
                        'input_artifacts': request_sources,
                    },
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_shared_noun_deictics_require_each_direct_source(self):
        cases = (
            ('Analyze this and that image.', 'image', '/tmp/one.png'),
            ('Transcribe this and that audio.', 'audio', '/tmp/one.wav'),
        )
        for prompt, source_type, source_path in cases:
            with self.subTest(prompt=prompt):
                rejected = self.owner.truth_gate_response_output_claims(
                    {
                        'id': 'resp_shared_noun_source_shortfall',
                        'output_text': 'Invented evidence for the absent second source.',
                    },
                    request_payload={
                        'prompt': prompt,
                        'input_artifacts': [
                            {'type': source_type, 'path': source_path},
                        ],
                    },
                )

                self.assertIn('need the source/content', rejected['output_text'])
                self.assertEqual(
                    rejected['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_uses_sanitized_selected_reference_truth(self):
        self.owner.hooks['sanitize_selected_reference_artifacts'] = lambda raw: []
        invalid_sources = (
            {'type': 'image', 'content': 'not an image'},
            {'type': 'image', 'path': '/definitely/does/not/exist.png'},
            {'type': 'audio', 'content': 'not audio'},
        )
        try:
            for source_record in invalid_sources:
                prompt = (
                    'Convert this audio into an image.'
                    if source_record.get('type') == 'audio'
                    else 'Convert this image into audio.'
                )
                with self.subTest(source_record=source_record):
                    updated = self.owner.truth_gate_response_output_claims(
                        {
                            'id': 'resp_unsanitized_selected_source',
                            'output_text': 'Invented output from an unavailable source.',
                        },
                        request_payload={
                            'prompt': prompt,
                            'selected_reference_artifacts': [source_record],
                        },
                    )

                    self.assertIn('need the source/content', updated['output_text'])
                    self.assertEqual(
                        updated['runtime']['truth_guard']['status'],
                        'clarification_required',
                    )
            invalid_input = self.owner.truth_gate_response_output_claims(
                {
                    'id': 'resp_unsanitized_input_source',
                    'output_text': 'Invented output from an unavailable input artifact.',
                },
                request_payload={
                    'prompt': 'Convert this image into audio.',
                    'input_artifacts': [
                        {'type': 'image', 'path': '/does/not/exist.png'},
                    ],
                },
            )
            self.assertIn('need the source/content', invalid_input['output_text'])
            self.assertEqual(
                invalid_input['runtime']['truth_guard']['status'],
                'clarification_required',
            )
        finally:
            self.owner.hooks.pop('sanitize_selected_reference_artifacts', None)

    def test_truth_gate_graph_grounding_requires_one_edge_per_prompt_pair(self):
        prompt = (
            'Generate an image and analyze this image. '
            'Generate another image and analyze this image.'
        )
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )
        valid_payload = {
            'id': 'resp_two_proven_image_pairs',
            'output_text': 'Prepared content for both proven image branches.',
        }

        valid = self.owner.truth_gate_response_output_claims(
            valid_payload,
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload=request_payload,
        )

        self.assertEqual(valid['output_text'], valid_payload['output_text'])
        self.assertNotIn('truth_guard', valid.get('runtime') or {})

        incomplete_graph = json.loads(json.dumps(phase_graph))
        second_consumer = next(
            phase
            for phase in incomplete_graph['phases']
            if phase.get('phase_id') == 'phase-5'
        )
        second_consumer.update({'capability': 'chat', 'output_type': 'text'})
        second_consumer['output_contract']['output_type'] = 'text'
        second_candidate = next(
            candidate
            for candidate in incomplete_graph['candidate_graph']['candidates']
            if candidate.get('candidate_type') == 'output'
            and candidate.get('phase_id') == 'phase-5'
        )
        second_candidate.update({'capability': 'chat', 'output_type': 'text'})

        rejected = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_one_of_two_image_pairs_proven',
                'output_text': 'Invented content for the unproven image branch.',
            },
            route_payload={'route_runtime': {'request_phase_graph': incomplete_graph}},
            request_payload=request_payload,
        )

        self.assertIn('need the source/content', rejected['output_text'])
        self.assertEqual(
            rejected['runtime']['truth_guard']['status'],
            'clarification_required',
        )

        rebound_graph = json.loads(json.dumps(phase_graph))
        first_producer = next(
            phase
            for phase in rebound_graph['phases']
            if phase.get('capability') == 'image_generation'
        )
        vision_consumers = [
            phase
            for phase in rebound_graph['phases']
            if phase.get('capability') == 'vision_analysis'
        ]
        rebound_consumer = vision_consumers[-1]
        rebound_consumer['depends_on'] = [first_producer['phase_id']]
        rebound_consumer['input_refs'] = [
            {
                'kind': 'phase_output',
                'phase_id': first_producer['phase_id'],
                'role': 'dependency',
            }
        ]
        rebound_candidate = next(
            candidate
            for candidate in rebound_graph['candidate_graph']['candidates']
            if candidate.get('candidate_type') == 'output'
            and candidate.get('phase_id') == rebound_consumer['phase_id']
        )
        rebound_candidate['depends_on'] = [first_producer['phase_id']]

        rebound_rejected = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_two_consumers_one_producer',
                'output_text': 'Invented content for the unconsumed second image.',
            },
            route_payload={'route_runtime': {'request_phase_graph': rebound_graph}},
            request_payload=request_payload,
        )

        self.assertIn('need the source/content', rebound_rejected['output_text'])
        self.assertEqual(
            rebound_rejected['runtime']['truth_guard']['status'],
            'clarification_required',
        )

    def test_truth_gate_graph_grounding_requires_declared_plural_cardinality(self):
        prompt = 'Generate three images and analyze these images.'
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )
        valid_payload = {
            'id': 'resp_three_proven_image_edges',
            'output_text': 'Prepared content for all three proven image branches.',
        }

        valid = self.owner.truth_gate_response_output_claims(
            valid_payload,
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload=request_payload,
        )

        self.assertEqual(valid['output_text'], valid_payload['output_text'])
        self.assertNotIn('truth_guard', valid.get('runtime') or {})

        incomplete_graph = json.loads(json.dumps(phase_graph))
        vision_phases = [
            phase
            for phase in incomplete_graph['phases']
            if phase.get('capability') == 'vision_analysis'
        ]
        missing_consumer = vision_phases[-1]
        missing_consumer.update({'capability': 'chat', 'output_type': 'text'})
        missing_candidate = next(
            candidate
            for candidate in incomplete_graph['candidate_graph']['candidates']
            if candidate.get('candidate_type') == 'output'
            and candidate.get('phase_id') == missing_consumer['phase_id']
        )
        missing_candidate.update({'capability': 'chat', 'output_type': 'text'})

        rejected = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_two_of_three_image_edges',
                'output_text': 'Invented content for the missing third image edge.',
            },
            route_payload={'route_runtime': {'request_phase_graph': incomplete_graph}},
            request_payload=request_payload,
        )

        self.assertIn('need the source/content', rejected['output_text'])
        self.assertEqual(
            rejected['runtime']['truth_guard']['status'],
            'clarification_required',
        )

    def test_graph_grounding_handles_deep_acyclic_topology_iteratively(self):
        prompt = 'Generate an image and analyze this image.'
        request_payload = {'ghost_route': True, 'prompt': prompt}
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={'capability': 'chat'},
        )
        deep_chain = []
        for index in range(1200):
            phase_id = f'phase-deep-{index}'
            dependency_id = f'phase-deep-{index - 1}' if index else ''
            deep_chain.append(
                {
                    'phase_id': phase_id,
                    'capability': 'chat',
                    'depends_on': [dependency_id] if dependency_id else [],
                    'input_refs': (
                        [
                            {
                                'kind': 'phase_output',
                                'phase_id': dependency_id,
                                'role': 'dependency',
                            }
                        ]
                        if dependency_id
                        else [{'kind': 'user_prompt', 'ref': 'intent_anchor'}]
                    ),
                }
            )
        phase_graph['phases'].extend(reversed(deep_chain))

        self.assertTrue(
            _route_phase_graph_has_artifact_consumer_edge(
                {'route_runtime': {'request_phase_graph': phase_graph}},
                'image_generation',
            )
        )

    def test_truth_gate_allows_same_turn_generated_slogan_source_for_audio_transform(self):
        payload = {
            'id': 'resp_same_turn_generated_slogan_source',
            'output_text': 'Warnung: Sofort umkehren!',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Schreibe einen kurzen Warnslogan. Erzeuge daraus ein Audio.',
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_same_turn_generated_english_scene_source_for_audio_transform(self):
        payload = {
            'id': 'resp_same_turn_generated_english_scene_source',
            'output_text': 'The lighthouse beam cuts through rain above the black sea.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Write a short lighthouse scene. Turn it into audio.',
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_binds_show_pronoun_to_same_turn_generated_description(self):
        prompt = (
            'Describe a place of your dream in vivid detail, '
            'then show it to me as an image.'
        )
        payload = {
            'id': 'resp_same_turn_description_to_image',
            'output_text': 'A moonlit coastal village wrapped around a sheltered bay.',
        }

        grounded = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={'prompt': prompt},
        )
        missing = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_missing_show_source',
                'output_text': 'Invented image source.',
            },
            request_payload={'prompt': 'Show it to me as an image.'},
        )

        self.assertEqual(grounded['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', grounded.get('runtime') or {})
        self.assertIn('need the source/content', missing['output_text'])
        self.assertEqual(
            missing['runtime']['truth_guard']['status'],
            'clarification_required',
        )

    def test_truth_gate_rejects_text_source_declared_after_deictic_transform(self):
        payload = {
            'id': 'resp_late_same_turn_source_declaration',
            'output_text': 'Invented audio source.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Erzeuge daraus Audio. Schreibe danach einen Szenentext.',
            },
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_rejects_negated_same_turn_text_source_declaration(self):
        payload = {
            'id': 'resp_negated_same_turn_source_declaration',
            'output_text': 'Invented audio source.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Schreibe keinen Szenentext. Erzeuge daraus Audio.',
            },
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_ignores_fully_quoted_meta_instruction(self):
        payload = {
            'id': 'resp_quoted_same_turn_source_example',
            'output_text': 'Invented audio source.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': (
                    'Erkläre den Ablauf mit diesem Beispiel: '
                    '"Schreibe einen Szenentext. Erzeuge daraus Audio."'
                ),
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_guards_real_command_after_typographic_quote(self):
        prompts = (
            '„Nur ein Beispiel: Analysiere dieses Bild.“ Analysiere jetzt dieses Bild.',
            '‚Nur ein Beispiel: Analysiere dieses Bild.‘ Analysiere jetzt dieses Bild.',
            '“Example only: analyze this image.” Analyze this image now.',
            '«Exemple: analyse this image.» Analyze this image now.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                updated = self.owner.truth_gate_response_output_claims(
                    {
                        'id': 'resp_real_command_after_quote',
                        'output_text': 'Invented evidence from an absent image.',
                    },
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_rejects_meta_instruction_as_same_turn_source_declaration(self):
        payload = {
            'id': 'resp_meta_same_turn_source_instruction',
            'output_text': 'Invented audio source.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Explain how to write a story, then turn that old file into audio.',
            },
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_rejects_competing_explicit_source_after_generated_text(self):
        for prompt in (
            'Write a story. Then turn that old file into audio.',
            'Schreibe einen Szenentext. Verwandle danach diese alte Datei in Audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_competing_explicit_source',
                    'output_text': 'Invented audio source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_rejects_competing_source_in_second_transform(self):
        payload = {
            'id': 'resp_competing_second_transform_source',
            'output_text': 'Invented combined result.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': (
                    'Write a story. Turn it into audio. '
                    'Then turn that old file into HTML.'
                ),
            },
        )

        self.assertIn('need the source/content', updated['output_text'])
        self.assertEqual(updated['runtime']['truth_guard']['status'], 'clarification_required')

    def test_truth_gate_rejects_single_quoted_and_backtick_source_examples(self):
        for prompt in (
            "The phrase 'Write a story.' Turn it into audio.",
            'The phrase `Write a story.` Turn it into audio.',
            'The phrase ‘Write a story.’ Turn it into audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_quoted_source_variant',
                    'output_text': 'Invented audio source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_allows_affirmative_dont_hesitate_source_command(self):
        payload = {
            'id': 'resp_affirmative_dont_hesitate_source',
            'output_text': 'The lighthouse keeps watch through the storm.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': "Don't hesitate to write a story. Turn it into audio.",
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_described_scene_as_same_turn_source(self):
        for prompt in (
            'Describe a lighthouse scene. Turn it into audio.',
            'Beschreibe eine Leuchtturmszene. Erzeuge daraus Audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_described_scene_source',
                    'output_text': 'Ein Leuchtturm steht nachts im peitschenden Sturm.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertEqual(updated['output_text'], payload['output_text'])
                self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_coordinated_source_and_transform_target(self):
        for prompt in (
            'Generate a slogan and an audio artifact from it.',
            'Erzeuge einen Szenentext und Audio daraus.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_coordinated_source_and_target',
                    'output_text': 'Licht hält stand.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertEqual(updated['output_text'], payload['output_text'])
                self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_explicit_reference_to_generated_source_kind(self):
        payload = {
            'id': 'resp_explicit_generated_story_reference',
            'output_text': 'The lighthouse keeps watch through the storm.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Write a story. Turn that story into audio.',
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_rejects_declarative_source_mentions_as_generation_authority(self):
        for prompt in (
            'I will write a story tomorrow. Turn it into audio.',
            'Alice will write a story tomorrow. Turn it into audio.',
            'I plan to write a story. Turn it into audio.',
            'I want to write a story. Turn it into audio.',
            'Can I write a story? Turn it into audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_declarative_source_mention',
                    'output_text': 'Invented audio source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_allows_you_directed_source_generation_question(self):
        payload = {
            'id': 'resp_you_directed_source_question',
            'output_text': 'The lighthouse keeps watch through the storm.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={'prompt': 'Can you write a story? Turn it into audio.'},
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_rejects_arbitrarily_modified_external_source_noun(self):
        for prompt in (
            'Write a story. Turn that referenced file into audio.',
            'Write a story. Turn that missing local file into audio.',
            'Schreibe einen Szenentext. Verwandle diese referenzierte Datei in Audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_modified_external_source_noun',
                    'output_text': 'Invented audio source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_rejects_fenced_and_blockquoted_source_examples(self):
        for prompt in (
            'The instruction is:\n```\nWrite a story.\n```\nTurn it into audio.',
            'The instruction is:\n> Write a story.\nTurn it into audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_multiline_quoted_source',
                    'output_text': 'Invented audio source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_allows_non_source_negation_inside_generation_constraint(self):
        for prompt in (
            'Schreibe einen kurzen, nicht kitschigen Szenentext. Erzeuge daraus Audio.',
            'Write, not merely outline, a story. Turn it into audio.',
            'Write no less than a vivid story. Turn it into audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_non_source_negation_constraint',
                    'output_text': 'The lighthouse keeps watch through the storm.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertEqual(updated['output_text'], payload['output_text'])
                self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_german_you_directed_modal_source_generation(self):
        payload = {
            'id': 'resp_german_modal_source_generation',
            'output_text': 'Ein Leuchtturm steht nachts im peitschenden Sturm.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Kannst du einen Szenentext schreiben? Erzeuge daraus Audio.',
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_rejects_ungrounded_image_only_transform(self):
        for prompt in (
            'Mach daraus ein Bild.',
            'Turn it into an image.',
            'Create a picture from that.',
            'Erzeuge hieraus eine Grafik.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_ungrounded_image_only_transform',
                    'output_text': 'Invented image source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_allows_same_turn_generated_scene_for_image_only_transform(self):
        payload = {
            'id': 'resp_generated_scene_image_transform',
            'output_text': 'A lighthouse stands through the storm.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Write a lighthouse scene. Turn it into an image.',
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_rejects_competing_content_source_kinds_and_parenthetical_it(self):
        for prompt in (
            'Write a story. Turn that report into audio.',
            'Write a story. Turn that article into audio.',
            'Write a story. Turn that email into audio.',
            'Write a story. Turn that message into audio.',
            'Schreibe eine Geschichte. Verwandle diesen Bericht in Audio.',
            'Write a story. Turn that poem into audio.',
            'Write a story. Turn that narration into audio.',
            'Write a story. Turn that summary into audio.',
            'Write a story. Turn that announcement into audio.',
            'Write a story. Turn that warning into audio.',
            'Write a story. Turn it (the missing report) into audio.',
        ):
            with self.subTest(prompt=prompt):
                payload = {
                    'id': 'resp_competing_content_source_kind',
                    'output_text': 'Invented transformed source.',
                }

                updated = self.owner.truth_gate_response_output_claims(
                    payload,
                    request_payload={'prompt': prompt},
                )

                self.assertIn('need the source/content', updated['output_text'])
                self.assertEqual(
                    updated['runtime']['truth_guard']['status'],
                    'clarification_required',
                )

    def test_truth_gate_allows_explicit_reference_to_same_turn_generated_report(self):
        payload = {
            'id': 'resp_same_turn_generated_report',
            'output_text': 'The storm report warns all vessels to remain in port.',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={'prompt': 'Write a report. Turn that report into audio.'},
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_truth_gate_allows_daraus_transform_with_selected_source(self):
        payload = {
            'id': 'resp_grounded_daraus_transform',
            'output_text': '<!DOCTYPE html><html><body><p>Quelle</p></body></html>',
        }

        updated = self.owner.truth_gate_response_output_claims(
            payload,
            request_payload={
                'prompt': 'Mach daraus ein HTML.',
                'selected_reference_artifacts': [{'content': '<p>Quelle</p>'}],
            },
        )

        self.assertEqual(updated['output_text'], payload['output_text'])
        self.assertNotIn('truth_guard', updated.get('runtime') or {})

    def test_late_fill_blocks_materialization_from_truth_guard_clarification(self):
        error = LateFillRuntimeOwner.repair_branch_execution_error(
            self.late_fill_owner,
            {
                'branch_id': 'branch-text_to_speech-1',
                'phase_id': 'phase-2',
                'capability': 'text_to_speech',
                'depends_on': ['phase-1'],
            },
            current_payload={
                'output_text': 'I need the source/content for the referenced file before I can create that artifact.',
                'runtime': {
                    'truth_guard': {
                        'kind': 'ungrounded_text_artifact_reference',
                        'status': 'clarification_required',
                    }
                },
            },
        )

        self.assertIsNotNone(error)
        self.assertEqual(error['code'], 'UPSTREAM_CLARIFICATION_REQUIRED')
        self.assertEqual(error['stage'], 'truth_guard_gate')
        self.assertFalse(error['retryable'])

    def test_late_fill_blocks_materialization_from_repair_required_control_payload(self):
        control_payload = json.dumps(
            {
                'decision_contract': {'status': 'ready'},
                'request_ir': {'output_obligations': []},
            }
        )
        error = LateFillRuntimeOwner.repair_branch_execution_error(
            self.late_fill_owner,
            {
                'branch_id': 'branch-text_to_speech-1',
                'phase_id': 'phase-2',
                'capability': 'text_to_speech',
                'depends_on': ['phase-1'],
            },
            current_payload={
                'output_text': control_payload,
                'runtime': {
                    'truth_guard': {
                        'kind': 'control_json_boundary',
                        'status': 'repair_required',
                    }
                },
            },
        )

        self.assertIsNotNone(error)
        self.assertEqual(error['code'], 'UPSTREAM_CONTROL_PAYLOAD_REJECTED')
        self.assertEqual(error['reason'], 'control_envelope_not_speakable')
        self.assertFalse(error['retryable'])

    def test_late_fill_blocks_legacy_control_payload_without_truth_guard(self):
        control_payload = json.dumps(
            {
                'request_phase_graph': {'current_phase_id': 'phase-1'},
                'request_ir': {'output_obligations': []},
            }
        )
        error = LateFillRuntimeOwner.repair_branch_execution_error(
            self.late_fill_owner,
            {
                'branch_id': 'branch-text_to_speech-legacy',
                'phase_id': 'phase-2',
                'capability': 'text_to_speech',
                'depends_on': ['phase-1'],
            },
            current_payload={'output_text': control_payload},
        )

        self.assertEqual(error['code'], 'UPSTREAM_CONTROL_PAYLOAD_REJECTED')

    def test_tts_final_sink_rejects_control_envelope_before_backend_invocation(self):
        control_payload = json.dumps(
            {
                'decision_contract': {'status': 'ready'},
                'request_ir': {'output_obligations': []},
            }
        )
        calls = []
        self.late_fill_owner.invoke_internal_api_json_route = lambda **kwargs: calls.append(kwargs)

        with self.assertRaisesRegex(RuntimeError, 'control_envelope_not_speakable'):
            LateFillRuntimeOwner.execute_prepared_late_fill_branch(
                self.late_fill_owner,
                {
                    'capability': 'text_to_speech',
                    'infer_payload': {'prompt': control_payload},
                    'effective_data': {'content_payload': control_payload},
                },
            )

        self.assertEqual(calls, [])

    def test_speech_to_text_gap_is_not_fulfilled_by_unrelated_text_artifact(self):
        gap = {
            'expected_capability': 'speech_to_text',
            'expected_branch_id': 'branch-speech_to_text-1',
            'missing_artifact_type': 'text',
        }
        payload = {
            'id': 'resp_warning_text_artifact',
            'response_capability': 'chat',
            'saved_text_path': '/tmp/artifacts/documents/generated-txt.txt',
            'artifacts': [
                {'type': 'text', 'path': '/tmp/artifacts/documents/generated-txt.txt'},
            ],
        }

        self.assertFalse(self.owner.artifact_gap_is_already_fulfilled(gap, payload))

    def test_speech_to_text_gap_is_fulfilled_by_matching_stt_fill_result(self):
        gap = {
            'expected_capability': 'speech_to_text',
            'expected_branch_id': 'branch-speech_to_text-1',
            'missing_artifact_type': 'text',
        }
        payload = {
            'id': 'resp_transcript_done',
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-speech_to_text-1',
                        'phase_id': 'phase-3',
                        'capability': 'speech_to_text',
                        'saved_text_path': '/tmp/artifacts/transcripts/generated-audio.md',
                    },
                ],
            },
        }

        self.assertTrue(self.owner.artifact_gap_is_already_fulfilled(gap, payload))

    def test_reserved_later_image_and_audio_phases_do_not_become_pending_late_fill(self):
        prompt = (
            'Plane eine kleine lokale KI-Demo mit genau drei Phasen: Text, Bild, Audio. '
            'Erzeuge nur die erste Phase als sichtbares Ergebnis. '
            'Halte Bild und Audio als mögliche spätere Schritte zurück und erkläre knapp, '
            'was noch nicht ausgeführt wurde.'
        )
        stale_branch = {
            'branch_id': 'branch-image_generation-1',
            'phase_id': 'phase-2',
            'capability': 'image_generation',
            'output_type': 'image',
            'artifact_prompt': 'Vorbereitet, aber noch nicht ausgeführt.',
        }
        stale_route_payload = {
            'route_runtime': {
                'request_phase_graph': {
                    'kind': 'ollmo.request_phase_graph',
                    'prompt': prompt,
                    'current_phase_id': 'phase-1',
                    'current_phase_capability': 'chat',
                    'continuation_required': True,
                    'downstream_capabilities': ['image_generation'],
                    'downstream_branch_ids': ['branch-image_generation-1'],
                    'downstream_branches': [stale_branch],
                },
                'execution_planner': {
                    'deferred_branch': stale_branch,
                    'deferred_branches': [stale_branch],
                    'deferred_capability': 'image_generation',
                    'deferred_capabilities': ['image_generation'],
                    'reason': 'request_phase_graph_follow_up',
                },
            },
        }

        branches = self.owner.extract_pending_deferred_branches(
            route_payload=stale_route_payload,
        )
        gap = self.owner.build_planner_deferred_follow_up_gap_spec(
            'Phase 1 ist vorbereitet. Bild und Audio bleiben mögliche spätere Schritte.',
            route_payload=stale_route_payload,
        )

        self.assertEqual(branches, [])
        self.assertIsNone(gap)

    def test_executable_later_image_phase_still_becomes_pending_late_fill(self):
        route_payload = {
            'route_runtime': {
                'request_phase_graph': {
                    'kind': 'ollmo.request_phase_graph',
                    'prompt': 'Schreibe einen kurzen Szenentext und generiere danach ein Bild davon.',
                    'current_phase_id': 'phase-1',
                    'current_phase_capability': 'chat',
                    'continuation_required': True,
                    'prompt_intent': {
                        'requests_visual_output': True,
                        'requested_visual_output_count': 1,
                    },
                    'downstream_capabilities': ['image_generation'],
                },
                'execution_planner': {
                    'deferred_branches': [
                        {
                            'branch_id': 'branch-image_generation-1',
                            'phase_id': 'phase-2',
                            'capability': 'image_generation',
                            'output_type': 'image',
                            'artifact_prompt': 'A small local AI demo on a desk.',
                        }
                    ],
                    'deferred_capability': 'image_generation',
                    'deferred_capabilities': ['image_generation'],
                },
            },
        }

        branches = self.owner.extract_pending_deferred_branches(route_payload=route_payload)

        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]['capability'], 'image_generation')
        self.assertEqual(branches[0]['branch_id'], 'branch-image_generation-1')

    def test_selected_tts_late_fill_focuses_to_one_candidate(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'best_candidate_only',
                'candidate_selection_index': 1,
            },
            {
                'content_payload': (
                    'Hier sind drei Ideen:\n\n'
                    '1. Stille der Ewigkeit: Eine zerbrochene Glaskuppel über einer Mondbibliothek.\n'
                    '2. Verlorenes Wissen: Dunkle Regale in einem Mondkrater.\n'
                    '3. Archiv der Menschheit: Ein Retro-NASA-Signet im Staub.'
                ),
            },
            capability='text_to_speech',
        )

        self.assertIn('Stille der Ewigkeit', payload['content_payload'])
        self.assertNotIn('Verlorenes Wissen', payload['content_payload'])
        self.assertEqual(payload['content_payload_source'], 'selected_candidate_from_phase_output')
        self.assertEqual(payload['selection_policy_applied'], 'best_candidate_only')

    def test_numbered_audio_prepare_contract_keeps_two_tts_bodies_distinct(self):
        tts_branches = [
            {
                'branch_id': f'branch-text_to_speech-{index}',
                'phase_id': f'phase-{index + 1}',
                'capability': 'text_to_speech',
                'output_type': 'audio',
                'depends_on': ['phase-1'],
                'queue_index': index,
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': index,
                'candidate_selection_count': 2,
                'audio_variant_index': index,
                'lang_code': 'de' if index == 1 else 'en',
                'audio_variant_role': (
                    'original_narration' if index == 1 else 'faithful_translation'
                ),
                'audio_variant_contract_source': 'explicit_language_role_sequence',
            }
            for index in (1, 2)
        ]
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'current_phase_resolution': 'graph_resolved',
            'prompt_intent': {
                'requests_audio_output': True,
                'counted_audio_output_obligation': True,
                'requested_audio_output_count': 2,
                'audio_output_count_exceeds_bound': False,
            },
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'kind': 'prepare',
                    'role': 'text_preparation',
                },
                *tts_branches,
            ],
            'downstream_branches': tts_branches,
            'downstream_capabilities': ['text_to_speech'],
        }
        prepare_contract = {
            'phase_graph': phase_graph,
            'current_phase': phase_graph['phases'][0],
            'downstream_capabilities': ['text_to_speech'],
            'reason': 'prepare two audio variants',
        }

        system_message = self.owner.build_prepare_phase_system_message(prepare_contract)
        semantic_payload = self.owner.build_response_semantic_phase_payload(
            output_text='1. Ruhige deutsche Fassung.\n2. Energetic English version.',
            route_payload={'route_runtime': {'request_phase_graph': phase_graph}},
            request_payload={'prompt': 'Erzeuge zwei Audiofassungen.'},
            capability='chat',
        )

        self.assertIsNotNone(system_message)
        self.assertIn('exactly 2 numbered, directly speakable bodies', system_message['content'])
        self.assertIn('1. through 2.', system_message['content'])
        self.assertIn(
            'Numbered body 1 must be in German and fulfill the original narration role.',
            system_message['content'],
        )
        self.assertIn(
            'Numbered body 2 must be in English and fulfill the faithful translation role.',
            system_message['content'],
        )
        self.assertNotIn('one clean reusable body', system_message['content'])
        self.assertEqual(
            semantic_payload['content_payload'],
            '1. Ruhige deutsche Fassung.\n2. Energetic English version.',
        )

    def test_selected_tts_late_fill_uses_second_distinct_numbered_candidate(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 2,
                'candidate_selection_count': 2,
            },
            {
                'content_payload': (
                    '1. Ruhige deutsche Fassung.\n'
                    '2. Energetic English version.'
                ),
            },
            capability='text_to_speech',
        )

        self.assertEqual(payload['content_payload'], 'Energetic English version.')
        self.assertEqual(payload['content_payload_source'], 'selected_candidate_from_phase_output')
        self.assertEqual(payload['selection_policy_applied'], 'selected_candidate_only')

    def test_selected_tts_late_fill_extracts_labeled_audio_variants_from_wrapped_output(self):
        wrapped_output = (
            'Hier ist die aktualisierte Ausgabe.\n\n'
            '**Bildanalyse (Unverändert):**\n'
            'Die erhaltene Kuppel steht unter der Milchstraße.\n\n'
            '**Audio-Erzeugung und Transkription:**\n'
            'Die folgenden Fassungen werden getrennt materialisiert.\n\n'
            '**Audio-Version 1 (Deutsch):**\n'
            '*   **Inhalt:** Die stille Kuppel blickte in die Nacht.\n'
            '*   **Transkript des erzeugten Audios:** "Nicht als Quelle verwenden."\n\n'
            '**Audio-Version 2 (Englisch):**\n'
            '*   **Inhalt:** The silent dome gazed into the night.\n'
            '*   **Transkript des erzeugten Audios:** "Also not a source."\n\n'
            '**JSON-Objekt:**\n'
            '```json\n{"audio_variant_1_transcript":"Invented"}\n```'
        )

        selected_payloads = []
        for index in (1, 2):
            payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
                {
                    'capability': 'text_to_speech',
                    'selection_policy': 'selected_candidate_only',
                    'candidate_selection_index': index,
                    'candidate_selection_count': 2,
                    'audio_variant_index': index,
                    'audio_variant_role': (
                        'original_narration' if index == 1 else 'faithful_translation'
                    ),
                    'lang_code': 'de' if index == 1 else 'en',
                    'stage_direction': f'materialize_requested_audio_variant_{index}',
                },
                {'content_payload': wrapped_output},
                capability='text_to_speech',
            )
            selected_payloads.append(payload)

        self.assertEqual(
            [payload.get('content_payload') for payload in selected_payloads],
            [
                'Die stille Kuppel blickte in die Nacht.',
                'The silent dome gazed into the night.',
            ],
        )
        self.assertEqual(
            [payload.get('candidate_extraction_source') for payload in selected_payloads],
            ['labeled_audio_variant_sections', 'labeled_audio_variant_sections'],
        )
        for payload in selected_payloads:
            self.assertEqual(
                payload.get('content_payload_source'),
                'selected_candidate_from_phase_output',
            )
            self.assertNotIn('Transkript', payload.get('content_payload') or '')
            self.assertNotIn('transcript', (payload.get('content_payload') or '').lower())
            self.assertNotIn('json', (payload.get('content_payload') or '').lower())

    def test_selected_tts_late_fill_fails_closed_for_duplicate_labeled_audio_index(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 1,
                'candidate_selection_count': 2,
                'audio_variant_index': 1,
                'stage_direction': 'materialize_requested_audio_variant_1',
            },
            {
                'content_payload': (
                    '**Audio-Version 1:**\n**Inhalt:** Erste Fassung.\n\n'
                    '**Audio-Version 1:**\n**Inhalt:** Zweite erste Fassung.\n\n'
                    '**Audio-Version 2:**\n**Inhalt:** Second version.'
                )
            },
            capability='text_to_speech',
        )

        self.assertEqual(payload.get('branch_contract_error'), 'selected_candidate_unavailable')
        self.assertEqual(payload.get('candidate_extraction_issue'), 'duplicate_audio_variant_index')
        self.assertTrue(payload.get('materialization_blocked'))

    def test_selected_tts_late_fill_fails_closed_for_multiple_speakable_fields_in_one_slot(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 1,
                'candidate_selection_count': 2,
                'audio_variant_index': 1,
                'stage_direction': 'materialize_requested_audio_variant_1',
            },
            {
                'content_payload': (
                    '**Audio-Version 1:**\n'
                    '**Inhalt:** Erste Fassung.\n'
                    '**Sprechtext:** Abweichende erste Fassung.\n\n'
                    '**Audio-Version 2:**\n**Inhalt:** Second version.'
                )
            },
            capability='text_to_speech',
        )

        self.assertEqual(payload.get('branch_contract_error'), 'selected_candidate_unavailable')
        self.assertEqual(payload.get('candidate_extraction_issue'), 'ambiguous_audio_variant_body')
        self.assertTrue(payload.get('materialization_blocked'))

    def test_selected_tts_late_fill_rejects_non_contiguous_numbered_candidates(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 1,
                'candidate_selection_count': 2,
            },
            {'content_payload': '1. Erste Fassung.\n3. Third version.'},
            capability='text_to_speech',
        )

        self.assertEqual(payload.get('branch_contract_error'), 'selected_candidate_unavailable')
        self.assertEqual(payload.get('candidate_extraction_issue'), 'non_contiguous_candidate_indexes')
        self.assertTrue(payload.get('materialization_blocked'))

    def test_selected_tts_late_fill_excludes_trailing_transcript_and_json_from_legacy_candidate(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 2,
                'candidate_selection_count': 2,
            },
            {
                'content_payload': (
                    '1. Erste Fassung.\n'
                    '2. Second version.\n\n'
                    '**Transkript:** invented evidence\n\n'
                    '**JSON-Objekt:** {"status":"prepared"}'
                )
            },
            capability='text_to_speech',
        )

        self.assertEqual(payload.get('content_payload'), 'Second version.')
        self.assertNotIn('Transkript', payload.get('content_payload') or '')
        self.assertNotIn('JSON', payload.get('content_payload') or '')

    def test_single_tts_late_fill_focuses_unique_labeled_narration_from_wrapped_output(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {'capability': 'text_to_speech'},
            {
                'content_payload': (
                    '**Sichtbare Details:**\n'
                    'Die Kuppel steht unter den Sternen.\n\n'
                    '**Deutsche Erzählung (zwei kurze Sätze):**\n'
                    'Unter dem Sternenhimmel ruhte die kleine Kuppel. '
                    'Die Milchstraße leuchtete über dem Hügel.\n\n'
                    '**Audio und Transkript:**\n'
                    'Transkript: Dieser behauptete Text ist keine Audioquelle.\n\n'
                    '**JSON-Objekt:**\n```json\n{"status":"prepared"}\n```'
                )
            },
            capability='text_to_speech',
        )

        self.assertEqual(
            payload.get('content_payload'),
            'Unter dem Sternenhimmel ruhte die kleine Kuppel. '
            'Die Milchstraße leuchtete über dem Hügel.',
        )
        self.assertEqual(payload.get('content_payload_source'), 'focused_labeled_speakable_text')
        self.assertEqual(payload.get('candidate_extraction_source'), 'labeled_speakable_section')

    def test_single_tts_late_fill_stops_labeled_narration_at_compact_json_sibling(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {'capability': 'text_to_speech'},
            {
                'content_payload': (
                    '**Deutsche Erzählung:**\n'
                    'Die Kuppel ruht unter den Sternen.\n'
                    '**JSON-Objekt:** {"status":"prepared"}'
                )
            },
            capability='text_to_speech',
        )

        self.assertEqual(
            payload.get('content_payload'),
            'Die Kuppel ruht unter den Sternen.',
        )

    def test_single_tts_late_fill_does_not_treat_javascript_as_speakable_script(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {'capability': 'text_to_speech'},
            {
                'content_payload': (
                    '**JavaScript:** window.renderAudioPreview();\n\n'
                    '**Deutsche Erzählung:** Die Kuppel ruht unter den Sternen.'
                )
            },
            capability='text_to_speech',
        )

        self.assertEqual(
            payload.get('content_payload'),
            'Die Kuppel ruht unter den Sternen.',
        )
        self.assertNotIn('branch_contract_error', payload)

    def test_counted_tts_late_fill_rejects_duplicate_candidate_bodies(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 1,
                'candidate_selection_count': 2,
            },
            {'content_payload': '1. Identischer Text.\n2. Identischer Text.'},
            capability='text_to_speech',
        )

        self.assertEqual(
            payload['branch_contract_error'],
            'selected_candidate_not_distinct',
        )
        self.assertTrue(payload['materialization_blocked'])
        self.assertNotIn('selection_policy_applied', payload)

    def test_counted_tts_late_fill_requires_exact_candidate_body_count(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 1,
                'candidate_selection_count': 2,
            },
            {'content_payload': '1. Only one prepared body.'},
            capability='text_to_speech',
        )

        self.assertEqual(
            payload['branch_contract_error'],
            'selected_candidate_unavailable',
        )
        self.assertTrue(payload['materialization_blocked'])

    def test_selected_tts_late_fill_fails_closed_when_candidate_is_missing(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': 2,
            },
            {'content_payload': '1. Only one prepared body.'},
            capability='text_to_speech',
        )

        self.assertEqual(payload['branch_contract_error'], 'selected_candidate_unavailable')
        self.assertTrue(payload['materialization_blocked'])
        self.assertEqual(payload['repair_action'], 'repair_branch_contract')
        self.assertNotIn('selection_policy_applied', payload)

    def test_selected_tts_late_fill_rejects_missing_or_invalid_selection_index(self):
        for raw_index in (None, 0, -2, 'bad', True, 2.0):
            with self.subTest(raw_index=raw_index):
                branch = {
                    'capability': 'text_to_speech',
                    'selection_policy': 'selected_candidate_only',
                }
                if raw_index is not None:
                    branch['candidate_selection_index'] = raw_index
                payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
                    branch,
                    {'content_payload': '1. First body.\n2. Second body.'},
                    capability='text_to_speech',
                )

                self.assertEqual(
                    payload['branch_contract_error'],
                    'selected_candidate_unavailable',
                )
                self.assertTrue(payload['materialization_blocked'])
                self.assertNotIn('selection_policy_applied', payload)

    def test_best_tts_candidate_without_index_keeps_deliberate_first_candidate_default(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'best_candidate_only',
            },
            {'content_payload': '1. First body.\n2. Second body.'},
            capability='text_to_speech',
        )

        self.assertEqual(payload['content_payload'], 'First body.')
        self.assertEqual(payload['selection_policy_applied'], 'best_candidate_only')

    def test_selected_tts_late_fill_accepts_bounded_decimal_string_index(self):
        payload = self.late_fill_owner.focus_late_fill_branch_gap_payload(
            {
                'capability': 'text_to_speech',
                'selection_policy': 'selected_candidate_only',
                'candidate_selection_index': '2',
            },
            {'content_payload': '1. First body.\n2. Second body.'},
            capability='text_to_speech',
        )

        self.assertEqual(payload['content_payload'], 'Second body.')
        self.assertEqual(payload['selection_policy_applied'], 'selected_candidate_only')

    def test_late_fill_plan_rejects_missing_selected_tts_candidate_before_routing(self):
        with self.assertRaisesRegex(RuntimeError, 'Selected TTS candidate contract is invalid'):
            self.late_fill_owner.prepare_late_fill_branch_plan(
                expected_capability='text_to_speech',
                artifact_gap={},
                current_payload={},
                request_payload={},
                assistant_message='',
                source_route_payload=None,
                failed_instance_id=None,
                build_deferred_follow_up_gap_for_capability=lambda gap, **kwargs: dict(gap),
                prepare_late_fill_request_payload=lambda *args, **kwargs: {
                    'branch_contract_error': 'selected_candidate_unavailable',
                },
                resolve_late_fill_route=lambda *args, **kwargs: self.fail(
                    'route resolution must not run after a selected-candidate contract error'
                ),
            )

    def test_expected_material_output_count_preserves_counted_audio_cardinality(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'prompt_intent': {
                'requests_audio_output': True,
                'counted_audio_output_obligation': True,
                'requested_audio_output_count': 2,
                'audio_output_count_exceeds_bound': False,
            },
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'chat'},
            ],
        }

        expected = self.owner._expected_material_output_counts(
            request_payload={'prompt': 'Erzeuge zwei Audiofassungen.'},
            request_phase_graph=phase_graph,
        )

        self.assertEqual(expected, {'audio': 2})

    def test_audio_cardinality_overflow_remains_canonically_blocked_in_closure(self):
        prompt = 'Erzeuge 99 getrennte Audiofassungen aus diesem Satz.'
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        review = self.owner.build_graph_closure_review(
            'Bitte reduziere die Anzahl auf höchstens sechs.',
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload={
                'output_text': 'Bitte reduziere die Anzahl auf höchstens sechs.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        cardinality_check = next(
            item
            for item in review['checks']
            if item.get('check_kind') == 'intent_cardinality_guard'
        )
        self.assertEqual(review['status'], 'blocked')
        self.assertTrue(review['continuation_required'])
        self.assertEqual(cardinality_check['status'], 'blocked')
        self.assertEqual(cardinality_check['requested_audio_output_count_raw'], 99)
        self.assertEqual(cardinality_check['requested_audio_output_count_max'], 6)
        self.assertEqual(cardinality_check['missing_count'], 1)
        self.assertTrue(cardinality_check['materialization_blocked'])
        self.assertTrue(cardinality_check['needs_external_input'])
        feedback = review['ghost_repair_feedback']
        self.assertEqual(feedback['repair_loop']['next_actions'], ['repair_branch_contract'])
        self.assertEqual(feedback['repair_loop']['executable_contract_count'], 0)
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 1)

    def test_path_only_vision_result_is_dependency_repair_not_completed_phase(self):
        graph = build_request_phase_graph(
            'Generate two images and then compare both generated images.',
            request_payload={'ghost_route': True},
            route_payload={'route_source': 'ghost_carried'},
        )
        overlaid = self.late_fill_owner.request_phase_graph_for_late_fill(
            route_payload={'route_runtime': {'request_phase_graph': graph}},
            artifact_payload={
                'late_fill': {
                    'completed_branches': [
                        {'branch_id': 'branch-vision_analysis-2', 'phase_id': 'phase-5', 'capability': 'vision_analysis'},
                    ],
                    'fill_results': [
                        {
                            'branch_id': 'branch-vision_analysis-2',
                            'phase_id': 'phase-5',
                            'capability': 'vision_analysis',
                            'result_text': '/Users/example/Projects/ollmo/artifacts/images/generated.png',
                        },
                    ],
                },
            },
        )

        phases = {item['phase_id']: item for item in overlaid['phases']}
        self.assertEqual(phases['phase-5']['status'], 'failed')

    def test_late_fill_uses_current_artifact_graph_before_stale_route_graph(self):
        stale_route_graph = {
            'current_phase_id': 'phase-1',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'chat', 'status': 'completed'},
            ],
            'downstream_branches': [],
        }
        current_artifact_graph = {
            'current_phase_id': 'phase-1',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'chat', 'status': 'completed'},
                {
                    'phase_id': 'repair-image',
                    'branch_id': 'repair-image',
                    'capability': 'image_generation',
                    'status': 'pending',
                    'depends_on': ['phase-1'],
                },
            ],
            'downstream_branches': [
                {
                    'phase_id': 'repair-image',
                    'branch_id': 'repair-image',
                    'capability': 'image_generation',
                    'status': 'pending',
                    'depends_on': ['phase-1'],
                }
            ],
        }

        resolved = self.late_fill_owner.request_phase_graph_for_late_fill(
            route_payload={'route_runtime': {'request_phase_graph': stale_route_graph}},
            artifact_payload={'runtime': {'request_phase_graph': current_artifact_graph}},
        )

        self.assertEqual(resolved['downstream_branches'][0]['branch_id'], 'repair-image')
        self.assertEqual(resolved['phases'][1]['depends_on'], ['phase-1'])

    def test_no_access_dependency_answer_becomes_repair_error(self):
        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            {
                'branch_id': 'branch-chat-1',
                'phase_id': 'phase-4',
                'capability': 'chat',
                'depends_on': ['phase-3'],
            },
            {
                'capability': 'chat',
                'output_text': 'Da ich keinen direkten Zugriff auf das generierte Audio habe, kann ich es nicht beurteilen.',
            },
        )

        self.assertIsNotNone(error)
        self.assertEqual(error['code'], 'DEPENDENCY_CHAIN_REPAIR_REQUIRED')

    def test_tts_source_evidence_retains_exact_focused_payload_and_digest(self):
        evidence = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {
                'content_payload': 'Die stille Kuppel blickte in die Nacht.',
                'content_payload_source': 'selected_candidate_from_phase_output',
            }
        )

        self.assertEqual(
            evidence.get('tts_source_text'),
            'Die stille Kuppel blickte in die Nacht.',
        )
        self.assertEqual(
            evidence.get('tts_source_text_source'),
            'selected_candidate_from_phase_output',
        )
        self.assertEqual(len(evidence.get('tts_source_text_sha256') or ''), 64)

    def test_tts_source_evidence_prefers_exact_final_infer_prompt(self):
        evidence = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {
                'content_payload': 'Nicht mehr autorisierte Zwischenfassung.',
                'content_payload_source': 'selected_candidate_from_phase_output',
            },
            infer_payload={
                'prompt': 'Die exakte, an das TTS-Backend gesendete Fassung.',
            },
        )

        self.assertEqual(
            evidence.get('tts_source_text'),
            'Die exakte, an das TTS-Backend gesendete Fassung.',
        )
        self.assertEqual(evidence.get('source_authority'), 'final_infer_prompt')

    def test_tts_source_evidence_hashes_final_infer_prompt_bytes_without_stripping(self):
        final_prompt = '  Die exakte Backend-Fassung.\n'
        evidence = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': 'stale'},
            infer_payload={'prompt': final_prompt},
        )

        self.assertEqual(evidence.get('tts_source_text'), final_prompt)
        self.assertEqual(
            evidence.get('tts_source_text_sha256'),
            hashlib.sha256(final_prompt.encode('utf-8')).hexdigest(),
        )

    @staticmethod
    def _tts_integrity_evidence(
        *,
        source_text,
        path='/tmp/integrity.wav',
        status='passed',
        reason_code='TTS_AUDIO_INTEGRITY_PASSED',
    ):
        return {
            'kind': 'ollmo.tts_audio_integrity_evidence',
            'version': 1,
            'policy_id': TTS_AUDIO_INTEGRITY_POLICY_ID,
            'authority': 'runtime_deterministic_audio_verification',
            'status': status,
            'reason_code': reason_code,
            'materialization_eligible': status == 'passed',
            'artifact_path': path,
            'source_sha256': hashlib.sha256(
                source_text.encode('utf-8')
            ).hexdigest(),
        }

    def test_tts_audio_integrity_gate_accepts_passed_bound_evidence(self):
        source_text = 'Die vollständige Audiofassung.'
        source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': source_text},
            infer_payload={'prompt': source_text},
        )
        branch = {
            'branch_id': 'branch-text_to_speech-1',
            'phase_id': 'phase-2',
            'capability': 'text_to_speech',
        }
        result = {
            'capability': 'text_to_speech',
            'saved_audio_path': '/tmp/integrity.wav',
            'tts_semantic_source': source,
            'tts_audio_integrity_evidence': self._tts_integrity_evidence(
                source_text=source_text,
            ),
        }

        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            result,
        )

        self.assertIsNone(error)

    def test_tts_audio_integrity_gate_preserves_failed_file_as_diagnostic_only(self):
        source_text = 'Diese Audiofassung muss vollständig gesprochen werden.'
        source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': source_text},
            infer_payload={'prompt': source_text},
        )
        branch = {
            'branch_id': 'branch-text_to_speech-1',
            'phase_id': 'phase-2',
            'capability': 'text_to_speech',
        }
        result = {
            'capability': 'text_to_speech',
            'saved_audio_path': '/tmp/truncated.wav',
            'tts_semantic_source': source,
            'tts_audio_integrity_evidence': self._tts_integrity_evidence(
                source_text=source_text,
                path='/tmp/truncated.wav',
                status='failed',
                reason_code='TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
            ),
        }

        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            result,
        )

        self.assertEqual(
            error.get('code'),
            'TTS_AUDIO_INTEGRITY_REPAIR_REQUIRED',
        )
        self.assertEqual(
            error.get('reason_code'),
            'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
        )
        self.assertTrue(error.get('materialization_blocked'))
        self.assertEqual(
            error.get('diagnostic_artifact', {}).get('path'),
            '/tmp/truncated.wav',
        )
        self.assertFalse(
            error.get('audio_integrity_evidence', {}).get(
                'materialization_eligible'
            )
        )

    def test_direct_audio_closure_blocks_failed_integrity_despite_saved_wav(self):
        source_text = 'Dies ist die vollständige Audiofassung.'
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'single_phase',
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'text_to_speech',
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'status': 'completed',
                }
            ],
            'output_obligations': [
                {
                    'obligation_id': 'obligation-audio-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'status': 'completed',
                    'required': True,
                }
            ],
        }
        review = self.owner.build_graph_closure_review(
            '',
            request_payload={'prompt': 'Gib die Antwort als Audio aus.'},
            artifact_payload={
                'saved_audio_path': '/tmp/truncated.wav',
                'artifacts': [
                    {'type': 'audio', 'path': '/tmp/truncated.wav'},
                ],
                'tts_audio_integrity_evidence': self._tts_integrity_evidence(
                    source_text=source_text,
                    path='/tmp/truncated.wav',
                    status='failed',
                    reason_code='TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
                ),
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        audio_check = next(
            item
            for item in review['checks']
            if item.get('obligation_id') == 'obligation-audio-1'
        )
        self.assertEqual(review['status'], 'blocked')
        self.assertEqual(audio_check['status'], 'blocked')
        self.assertEqual(
            audio_check['evidence'],
            'tts_audio_integrity_failed',
        )
        self.assertEqual(
            audio_check['repair_action'],
            'retry_same_branch',
        )
        self.assertTrue(audio_check['materialization_blocked'])

    def test_direct_audio_closure_accepts_passed_integrity(self):
        source_text = 'Dies ist die vollständige Audiofassung.'
        phase_graph = {
            'kind': 'ollmo.request_phase_graph',
            'mode': 'single_phase',
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'text_to_speech',
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'status': 'completed',
                }
            ],
            'output_obligations': [
                {
                    'obligation_id': 'obligation-audio-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'status': 'completed',
                    'required': True,
                }
            ],
        }
        review = self.owner.build_graph_closure_review(
            '',
            request_payload={'prompt': 'Gib die Antwort als Audio aus.'},
            artifact_payload={
                'saved_audio_path': '/tmp/complete.wav',
                'artifacts': [
                    {'type': 'audio', 'path': '/tmp/complete.wav'},
                ],
                'tts_audio_integrity_evidence': self._tts_integrity_evidence(
                    source_text=source_text,
                    path='/tmp/complete.wav',
                ),
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        audio_check = next(
            item
            for item in review['checks']
            if item.get('obligation_id') == 'obligation-audio-1'
        )
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(audio_check['status'], 'fulfilled')
        self.assertEqual(audio_check['audio_integrity_status'], 'passed')

    def test_execute_prepared_tts_branch_attaches_exact_final_prompt_source(self):
        final_prompt = '  Exakter Text für das TTS-Backend.\n'
        from tests.fake_backends.fixtures import tiny_wav_bytes

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'exact-source.wav'
            audio_path.write_bytes(tiny_wav_bytes())
            self.late_fill_owner.invoke_internal_api_json_route = lambda **kwargs: (
                {
                    'mode': 'text_to_speech',
                    'saved_audio_path': str(audio_path),
                },
                200,
            )
            self.late_fill_owner.filter_responses_infer_result = (
                lambda result, **kwargs: dict(result)
            )

            result = self.late_fill_owner.execute_prepared_late_fill_branch(
                {
                    'capability': 'text_to_speech',
                    'effective_data': {
                        'content_payload': 'Veraltete Zwischenfassung.',
                    },
                    'infer_payload': {'prompt': final_prompt},
                    'execution_contract': {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                    },
                }
            )

        source = result['infer_result']['tts_semantic_source']
        self.assertEqual(source['tts_source_text'], final_prompt)
        self.assertEqual(source['source_authority'], 'final_infer_prompt')
        self.assertEqual(source['branch_id'], 'branch-text_to_speech-1')
        self.assertEqual(source['phase_id'], 'phase-2')
        integrity = result['infer_result']['tts_audio_integrity_evidence']
        self.assertEqual(integrity['status'], 'passed')
        self.assertEqual(
            integrity['source_sha256'],
            source['tts_source_text_sha256'],
        )

    def test_standalone_stt_remains_outside_tts_semantic_policy(self):
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-2',
            'capability': 'speech_to_text',
            'depends_on': ['phase-1'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-chat-1',
                        'phase_id': 'phase-1',
                        'capability': 'chat',
                        'result_text': 'Transcribe the supplied recording.',
                    }
                ]
            }
        }
        infer_result = {
            'capability': 'speech_to_text',
            'result_text': 'Independent input recording transcript.',
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            infer_result,
            current_payload=current_payload,
        )
        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            infer_result,
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'not_applicable')
        self.assertEqual(evidence.get('reason_code'), 'NO_DIRECT_TTS_PRODUCER_RESULT')
        self.assertIsNone(error)

    def test_tts_stt_semantic_evidence_uses_only_declared_producer_pair(self):
        first_source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': 'Die deutsche Fassung.'},
            infer_payload={'prompt': 'Die deutsche Fassung.'},
        )
        second_source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': 'The English version.'},
            infer_payload={'prompt': 'The English version.'},
        )
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': first_source,
                    },
                    {
                        'branch_id': 'branch-text_to_speech-2',
                        'phase_id': 'phase-3',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': second_source,
                    },
                ]
            }
        }
        branch = {
            'branch_id': 'branch-speech_to_text-2',
            'phase_id': 'phase-5',
            'capability': 'speech_to_text',
            'depends_on': ['phase-3'],
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {
                'capability': 'speech_to_text',
                'result_text': 'the english version',
            },
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'matched')
        self.assertEqual(evidence.get('producer_phase_id'), 'phase-3')

    def test_tts_stt_semantic_evidence_uses_execution_contract_dependency_alias(self):
        source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {
                'branch_id': 'branch-text_to_speech-1',
                'phase_id': 'phase-2',
                'content_payload': 'Die Kuppel ruht.',
            },
            infer_payload={'prompt': 'Die Kuppel ruht.'},
        )
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'execution_contract': {'dependencies': ['phase-2']},
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': source,
                    }
                ]
            }
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {'capability': 'speech_to_text', 'result_text': 'Die Kuppel ruht.'},
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'matched')
        self.assertEqual(evidence.get('producer_phase_id'), 'phase-2')

    def test_declared_tts_dependency_without_fill_result_blocks_stt(self):
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'runtime': {
                'request_phase_graph': {
                    'phases': [
                        {
                            'phase_id': 'phase-2',
                            'branch_id': 'branch-text_to_speech-1',
                            'capability': 'text_to_speech',
                        },
                        branch,
                    ]
                }
            },
            'late_fill': {'fill_results': []},
        }
        infer_result = {
            'capability': 'speech_to_text',
            'result_text': 'An otherwise plausible transcript.',
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            infer_result,
            current_payload=current_payload,
        )
        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            infer_result,
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'unavailable')
        self.assertEqual(
            evidence.get('reason_code'),
            'TTS_PRODUCER_RESULT_UNAVAILABLE',
        )
        self.assertEqual(error.get('code'), 'DEPENDENCY_CHAIN_REPAIR_REQUIRED')

    def test_tts_stt_semantic_evidence_requires_source_digest(self):
        source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': 'Die Kuppel ruht.'},
            infer_payload={'prompt': 'Die Kuppel ruht.'},
        )
        source.pop('tts_source_text_sha256')
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': source,
                    }
                ]
            }
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {'capability': 'speech_to_text', 'result_text': 'Die Kuppel ruht.'},
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'unavailable')
        self.assertEqual(
            evidence.get('reason_code'),
            'TTS_SOURCE_EVIDENCE_DIGEST_UNAVAILABLE',
        )

    def test_tts_stt_semantic_evidence_requires_final_infer_prompt_authority(self):
        source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {'content_payload': 'Nur eine Zwischenfassung.'}
        )
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': source,
                    }
                ]
            }
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {
                'capability': 'speech_to_text',
                'result_text': 'Nur eine Zwischenfassung.',
            },
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'unavailable')
        self.assertEqual(
            evidence.get('reason_code'),
            'TTS_SOURCE_EVIDENCE_CONTRACT_INVALID',
        )

    def test_tts_stt_semantic_evidence_rejects_source_binding_mismatch(self):
        source = self.late_fill_owner.tts_source_evidence_from_effective_data(
            {
                'branch_id': 'branch-text_to_speech-other',
                'phase_id': 'phase-99',
                'content_payload': 'Die Kuppel ruht.',
            },
            infer_payload={'prompt': 'Die Kuppel ruht.'},
        )
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': source,
                    }
                ]
            }
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {'capability': 'speech_to_text', 'result_text': 'Die Kuppel ruht.'},
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'unavailable')
        self.assertEqual(
            evidence.get('reason_code'),
            'TTS_SOURCE_EVIDENCE_BINDING_MISMATCH',
        )

    def test_tts_stt_fidelity_rejects_inserted_negation_and_quarter_truncation(self):
        negated = self.late_fill_owner._tts_stt_similarity_metrics(
            'Die Kuppel leuchtet heute hell über dem stillen Tal.',
            'Die Kuppel leuchtet heute nicht hell über dem stillen Tal.',
        )
        truncated = self.late_fill_owner._tts_stt_similarity_metrics(
            'Die Kuppel leuchtet heute hell über dem stillen Tal bei Nacht.',
            'Die Kuppel leuchtet heute hell über dem stillen Tal.',
        )

        self.assertFalse(negated.get('semantic_match'))
        self.assertFalse(negated.get('negation_consistent'))
        self.assertFalse(truncated.get('semantic_match'))

    def test_tts_stt_semantic_evidence_accepts_harmless_transcript_normalization(self):
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': (
                            self.late_fill_owner.tts_source_evidence_from_effective_data(
                                {
                                    'branch_id': 'branch-text_to_speech-1',
                                    'phase_id': 'phase-2',
                                    'content_payload': 'Die stille Kuppel blickte in die Nacht.',
                                },
                                infer_payload={
                                    'prompt': 'Die stille Kuppel blickte in die Nacht.',
                                },
                            )
                        ),
                    }
                ]
            }
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {'capability': 'speech_to_text', 'result_text': 'die stille kuppel blickte in die nacht'},
            current_payload=current_payload,
        )
        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            {
                'capability': 'speech_to_text',
                'result_text': 'die stille kuppel blickte in die nacht',
                'tts_stt_semantic_evidence': evidence,
            },
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'matched')
        self.assertTrue(evidence.get('semantic_match'))
        self.assertIsNone(error)

    def test_tts_stt_semantic_evidence_blocks_observed_unrelated_transcript(self):
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'tts_semantic_source': (
                            self.late_fill_owner.tts_source_evidence_from_effective_data(
                                {
                                    'branch_id': 'branch-text_to_speech-1',
                                    'phase_id': 'phase-2',
                                    'content_payload': (
                                        'Unter dem prachtvollen Sternenhimmel ruhte das kleine, '
                                        'holzverkleidete Gehöft auf dem dunklen Hügel. Dort konnte man '
                                        'die majestätische Milchstraße über der friedlichen Landschaft bestaunen.'
                                    ),
                                },
                                infer_payload={
                                    'prompt': (
                                        'Unter dem prachtvollen Sternenhimmel ruhte das kleine, '
                                        'holzverkleidete Gehöft auf dem dunklen Hügel. Dort konnte man '
                                        'die majestätische Milchstraße über der friedlichen Landschaft bestaunen.'
                                    ),
                                },
                            )
                        ),
                    }
                ]
            }
        }
        infer_result = {
            'capability': 'speech_to_text',
            'result_text': 'o Thank you. Thank you. Thank you.',
        }
        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            infer_result,
            current_payload=current_payload,
        )
        infer_result['tts_stt_semantic_evidence'] = evidence
        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            infer_result,
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'mismatched')
        self.assertFalse(evidence.get('semantic_match'))
        self.assertEqual(error.get('code'), 'DEPENDENCY_CHAIN_REPAIR_REQUIRED')
        self.assertEqual(error.get('reason_code'), 'TTS_STT_SEMANTIC_MISMATCH')
        self.assertEqual(error.get('repair_action'), 'repair_dependency_chain')
        self.assertEqual(error.get('stage'), 'semantic_evidence_gate')
        self.assertFalse(error.get('retryable'))
        self.assertTrue(error.get('materialization_blocked'))

    def test_tts_stt_semantic_evidence_blocks_missing_producer_source_truth(self):
        branch = {
            'branch_id': 'branch-speech_to_text-1',
            'phase_id': 'phase-3',
            'capability': 'speech_to_text',
            'depends_on': ['phase-2'],
        }
        current_payload = {
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-2',
                        'capability': 'text_to_speech',
                        'saved_audio_path': '/tmp/audio.wav',
                    }
                ]
            }
        }

        evidence = self.late_fill_owner.tts_stt_semantic_evidence_for_branch_result(
            branch,
            {'capability': 'speech_to_text', 'result_text': 'Some transcript.'},
            current_payload=current_payload,
        )
        error = self.late_fill_owner.dependency_evidence_error_for_branch_result(
            branch,
            {
                'capability': 'speech_to_text',
                'result_text': 'Some transcript.',
                'tts_stt_semantic_evidence': evidence,
            },
            current_payload=current_payload,
        )

        self.assertEqual(evidence.get('status'), 'unavailable')
        self.assertEqual(evidence.get('reason_code'), 'TTS_SOURCE_EVIDENCE_UNAVAILABLE')
        self.assertEqual(error.get('code'), 'DEPENDENCY_CHAIN_REPAIR_REQUIRED')
        self.assertEqual(error.get('repair_action'), 'repair_dependency_chain')

    def test_closure_review_requires_saved_text_artifact_for_text_artifact_output(self):
        graph = build_request_phase_graph('Create an index.html artifact with a hello page.')
        payload = {
            'id': 'resp_missing_text_artifact',
            'output_text': '<!doctype html><h1>Hello</h1>',
            'output': [{'type': 'message', 'content': '<!doctype html><h1>Hello</h1>'}],
            'runtime': {'request_phase_graph': graph},
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': 'Create an index.html artifact with a hello page.'},
            artifact_payload=payload,
        )

        self.assertEqual(review['status'], 'pending')
        artifact_checks = [
            item for item in review['checks']
            if item.get('role') == 'text_artifact_output'
        ]
        self.assertEqual(len(artifact_checks), 1)
        self.assertIn(artifact_checks[0]['status'], {'pending', 'planned'})
        self.assertNotEqual(artifact_checks[0]['evidence'], 'current_phase_output_text')

    def test_closure_review_does_not_fulfill_text_artifact_from_completed_branch_without_file(self):
        graph = build_request_phase_graph('Create an index.html artifact with a hello page.')
        payload = {
            'id': 'resp_completed_branch_without_file',
            'output_text': '<!doctype html><h1>Hello</h1>',
            'runtime': {'request_phase_graph': graph},
            'late_fill': {
                'status': 'completed',
                'completed_capabilities': ['chat'],
                'completed_branches': [
                    {
                        'branch_id': 'branch-text_artifact-1',
                        'phase_id': 'phase-2',
                        'capability': 'chat',
                    }
                ],
                'fill_results': [
                    {
                        'branch_id': 'branch-text_artifact-1',
                        'phase_id': 'phase-2',
                        'capability': 'chat',
                        'result_text': '<!doctype html><h1>Hello</h1>',
                    }
                ],
            },
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': 'Create an index.html artifact with a hello page.'},
            artifact_payload=payload,
        )

        artifact_check = next(
            item for item in review['checks']
            if item.get('role') == 'text_artifact_output'
        )
        self.assertEqual(review['status'], 'pending')
        self.assertIn(artifact_check['status'], {'pending', 'planned'})

    def test_completion_gap_supports_text_artifact_chat_branch(self):
        graph = build_request_phase_graph('Create an index.html artifact with a hello page.')

        gap = self.owner.build_artifact_completion_gap_spec(
            'I prepared the artifact content.',
            route_payload={
                'route_runtime': {
                    'request_phase_graph': graph,
                    'execution_planner': {'reason': 'request_phase_graph_follow_up'},
                }
            },
            artifact_payload={'runtime': {'request_phase_graph': graph}},
        )

        self.assertIsNotNone(gap)
        self.assertEqual(gap['expected_capability'], 'chat')
        self.assertEqual(gap['missing_artifact_type'], 'text')
        self.assertEqual(gap['stage_direction'], 'materialize_requested_text_artifact')
        self.assertEqual(gap['text_artifact_extension'], 'html')

    def test_completion_gap_skips_fulfilled_text_artifact_branch_by_extension(self):
        prompt = 'Create index.html and styles.css artifacts for an alpine rescue station dashboard.'
        graph = build_request_phase_graph(prompt)
        payload = {
            'id': 'resp_html_done_css_pending',
            'output_text': '```html\n<!doctype html><h1>Alpine Rescue</h1>\n```',
            'saved_text_path': '/tmp/artifacts/documents/generated-html.html',
            'runtime': {'request_phase_graph': graph},
        }

        gap = self.owner.build_artifact_completion_gap_spec(
            payload['output_text'],
            route_payload={
                'route_runtime': {
                    'request_phase_graph': graph,
                    'execution_planner': {'reason': 'request_phase_graph_follow_up'},
                }
            },
            artifact_payload=payload,
        )

        self.assertIsNotNone(gap)
        self.assertEqual(gap['expected_branch_id'], 'branch-text_artifact-2')
        self.assertEqual(gap['missing_artifact_type'], 'text')
        self.assertEqual(gap['text_artifact_extension'], 'css')

    def test_closure_review_requests_rebind_for_unresolved_linked_artifact_placeholders(self):
        prompt = (
            'Generate exactly one local image artifact first. '
            'Then create index.html and styles.css and use the generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_unbound_page_links',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': '/tmp/aethelgard-abyss-7.png'},
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': '<!doctype html><link rel="stylesheet" href="styles.css"><main class="hero"></main>',
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.hero { background-image: url("placeholder.png"); }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        binding_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'linked_artifact_binding'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(len(binding_checks), 1)
        self.assertEqual(binding_checks[0]['status'], 'pending')
        self.assertEqual(binding_checks[0]['repair_action'], 'rebind_dependency_evidence')
        self.assertEqual(binding_checks[0]['text_artifact_extension'], 'css')
        self.assertEqual(binding_checks[0]['text_artifact_target_path'], '/tmp/styles.css')
        self.assertEqual(binding_checks[0]['artifact_request']['target_path'], '/tmp/styles.css')
        self.assertIn('/tmp/aethelgard-abyss-7.png', binding_checks[0]['content_payload'])
        self.assertIn('placeholder', binding_checks[0]['content_payload'])

    def test_closure_review_displays_ollmo_relative_artifact_paths_for_link_rebind(self):
        prompt = (
            'Generate exactly one local image artifact first. '
            'Then create index.html and styles.css and use the generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        workspace_root = Path.cwd().resolve()
        image_path = workspace_root / 'artifacts' / 'images' / 'aethelgard-abyss-7.png'
        index_path = workspace_root / 'artifacts' / 'documents' / 'index.html'
        styles_path = workspace_root / 'artifacts' / 'documents' / 'styles.css'
        payload = {
            'id': 'resp_unbound_page_links_relative_paths',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': str(image_path)},
                {
                    'type': 'text',
                    'path': str(index_path),
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': '<!doctype html><link rel="stylesheet" href="styles.css"><main class="hero"></main>',
                },
                {
                    'type': 'text',
                    'path': str(styles_path),
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.hero { background-image: url("placeholder.png"); }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        binding_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'linked_artifact_binding'
        ]
        self.assertEqual(len(binding_checks), 1)
        content_payload = binding_checks[0]['content_payload']
        self.assertIn('Target text artifact: artifacts/documents/styles.css', content_payload)
        self.assertIn('artifacts/images/aethelgard-abyss-7.png', content_payload)
        self.assertNotIn(str(workspace_root), content_payload)

    def test_closure_review_accepts_linked_artifacts_with_real_runtime_paths(self):
        prompt = (
            'Generate exactly one local image artifact first. '
            'Then create index.html and styles.css and use the generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_bound_page_links',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': '/tmp/aethelgard-abyss-7.png'},
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': '<!doctype html><link rel="stylesheet" href="styles.css"><main class="hero"></main>',
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.hero { background-image: url("aethelgard-abyss-7.png"); }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'linked_artifact_binding'
            ]
        )

    def test_closure_review_ignores_navigation_hash_when_runtime_links_are_bound(self):
        prompt = (
            'Generate exactly one local image artifact first. '
            'Then create index.html and styles.css and use the generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_bound_page_links_with_hash_nav',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': '/tmp/aethelgard-abyss-7.png'},
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><link rel="stylesheet" href="styles.css">'
                        '<nav><a href="#" class="brand">Aethelgard</a>'
                        '<a href="#story">Story</a></nav><main class="hero"></main>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.hero { background-image: url("aethelgard-abyss-7.png"); }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'linked_artifact_binding'
            ]
        )

    def test_closure_review_ignores_example_copy_when_runtime_links_are_bound(self):
        prompt = (
            'Generate exactly one local image artifact first. '
            'Then create index.html and styles.css and use the generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_bound_page_links_with_example_copy',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': '/tmp/aethelgard-abyss-7.png'},
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><link rel="stylesheet" href="styles.css">'
                        '<main class="hero"><p>Example field notes from the expedition.</p></main>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.hero { background-image: url("aethelgard-abyss-7.png"); }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'linked_artifact_binding'
            ]
        )

    def test_closure_review_ignores_hero_image_class_when_runtime_links_are_bound(self):
        prompt = (
            'Generate exactly one local image artifact first. '
            'Then create index.html and styles.css and use the generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_bound_page_links_with_hero_image_class',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': '/tmp/aethelgard-abyss-7.png'},
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><link rel="stylesheet" href="styles.css">'
                        '<main class="hero"><div class="hero-image"></div></main>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': (
                        '.hero-image { background-image: url("aethelgard-abyss-7.png"); } '
                        '.hero:hover .hero-image { transform: scale(1.02); }'
                    ),
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'linked_artifact_binding'
            ]
        )

    def test_closure_review_blocks_hero_placeholder_when_images_only_appear_later(self):
        prompt = (
            'Generate exactly three local image artifacts first. '
            'Then create index.html and styles.css and use a generated image as the hero background.'
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_hero_placeholder_with_later_images',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {'type': 'image', 'path': '/tmp/hero.png'},
                {'type': 'image', 'path': '/tmp/gallery-1.png'},
                {'type': 'image', 'path': '/tmp/gallery-2.png'},
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><html><head><link rel="stylesheet" href="styles.css"></head>'
                        '<body><header class="hero"><h1>Mon Repos</h1></header>'
                        '<main><img src="hero.png"><img src="gallery-1.png"><img src="gallery-2.png"></main>'
                        '</body></html>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': (
                        ".hero { min-height: 100vh; background: url('https://images.unsplash.com/example.jpg') center/cover; } "
                        '.gallery { display: grid; }'
                    ),
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        binding_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'linked_artifact_binding'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(len(binding_checks), 1)
        check = binding_checks[0]
        self.assertEqual(check['text_artifact_extension'], 'css')
        self.assertEqual(check['text_artifact_target_path'], '/tmp/styles.css')
        self.assertIn('Hero section does not use a concrete generated image artifact path', check['content_payload'])
        self.assertIn('/tmp/hero.png', check['content_payload'])

    def test_closure_review_blocks_html_css_selector_binding_drift(self):
        prompt = 'Create index.html and styles.css for a modern property page.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        html = (
            '<!doctype html><html><head><link rel="stylesheet" href="styles.css"></head>'
            '<body><section class="hero-image-container hero-bg hero-overlay content-split">'
            '<div class="split-wrapper split-image split-text reveal-img footer-links footer-bottom">'
            'Copy</div></section></body></html>'
        )
        css = (
            '.hero-image { min-height: 80vh; } '
            '.hero-image .overlay { opacity: .6; } '
            '.content-section { padding: 4rem; } '
            '.split-layout { display: grid; } '
            '.content-text { color: #222; } '
            '.content-image { width: 100%; } '
            '.feature-card { border: 1px solid #ddd; } '
            '.final-cta { text-align: center; }'
        )

        review = self.owner.build_graph_closure_review(
            'Artifacts generated.',
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload={
                'output_text': 'Artifacts generated.',
                'runtime': {'request_phase_graph': graph},
                'artifacts': [
                    {
                        'type': 'text',
                        'path': '/tmp/index.html',
                        'name': 'index',
                        'mime_type': 'text/html',
                        'content': html,
                        'branch_id': 'branch-text_artifact-1',
                        'phase_id': 'phase-2',
                    },
                    {
                        'type': 'text',
                        'path': '/tmp/styles.css',
                        'name': 'styles',
                        'mime_type': 'text/css',
                        'content': css,
                        'branch_id': 'branch-text_artifact-2',
                        'phase_id': 'phase-3',
                    },
                ],
            },
        )

        selector_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'html_css_selector_binding'
        ]
        self.assertEqual(len(selector_checks), 1)
        check = selector_checks[0]
        self.assertEqual(check['status'], 'pending')
        self.assertEqual(check['content_payload_source'], 'closure_html_css_selector_binding_review')
        self.assertEqual(check['text_artifact_extension'], 'css')
        self.assertIn('hero-image-container', check['html_class_tokens_missing_css_selectors'])
        self.assertIn('hero-image', check['css_class_selectors_missing_html_usage'])
        self.assertIn('Current saved HTML file content', check['content_payload'])
        self.assertIn('Current saved CSS target file content', check['content_payload'])

    def test_closure_review_allows_partially_styled_matching_html_css_pair(self):
        prompt = 'Create index.html and styles.css with a hero image.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        html = (
            '<!doctype html><html><head><link rel="stylesheet" href="styles.css"></head>'
            '<body><section class="hero hero-image overlay content-section split-layout">'
            '<div class="content-text content-image feature-card utility-gap">Copy</div>'
            '</section></body></html>'
        )
        css = (
            '.hero { min-height: 80vh; } '
            '.hero-image { background-size: cover; } '
            '.overlay { opacity: .6; } '
            '.content-section { padding: 4rem; } '
            '.split-layout { display: grid; } '
            '.content-text { color: #222; } '
            '.content-image { width: 100%; } '
            '.feature-card { border: 1px solid #ddd; }'
        )

        review = self.owner.build_graph_closure_review(
            'Artifacts generated.',
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload={
                'output_text': 'Artifacts generated.',
                'runtime': {'request_phase_graph': graph},
                'artifacts': [
                    {
                        'type': 'text',
                        'path': '/tmp/index.html',
                        'name': 'index',
                        'mime_type': 'text/html',
                        'content': html,
                    },
                    {
                        'type': 'text',
                        'path': '/tmp/styles.css',
                        'name': 'styles',
                        'mime_type': 'text/css',
                        'content': css,
                    },
                ],
            },
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'html_css_selector_binding'
            ]
        )

    def test_deterministic_syntax_repair_normalizes_malformed_html_closing_tag(self):
        content = '<!doctype html><html><body><section><p>Copy</p></承section></body></html>'

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('</section>', repaired)
        self.assertNotIn('</承section>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_closing_tag' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_stray_formatting_close_inside_open_tag(self):
        content = (
            '<!doctype html><html><body><nav><ul>'
            '<li><</strong>a href="#capabilities">Architektur</a></li>'
            '</ul></nav></body></html>'
        )

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('<a href="#capabilities">Architektur</a>', repaired)
        self.assertNotIn('</strong>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_stray_formatting_close_in_open_tag' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_partial_open_tag_before_valid_same_tag(self):
        content = (
            '<!doctype html><html><body><section><div class="grid-cards">'
            '<div class="card"><h3>One</h3><p>Copy</p></div>'
            '<div class</strong><div class="card"><h3>Two</h3><p>Copy</p></div>'
            '</div></section></body></html>'
        )

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('<div class="card"><h3>Two</h3>', repaired)
        self.assertNotIn('<div class</strong>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_partial_open_tag_with_stray_formatting_close' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_stray_regex_prefix_before_img(self):
        content = (
            '<!doctype html><html><body><section>'
            '<.*img src="../images/hero.png" alt="Hero image">'
            '</section></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('stray prefix `.*`', '\n'.join(before_issues))
        self.assertIn('<img src="../images/hero.png" alt="Hero image">', repaired)
        self.assertNotIn('<.*img', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_opening_tag_prefix' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_punctuation_pseudotag_before_anchor(self):
        content = (
            '<!doctype html><html><body><nav><ul>'
            '<li><)<a href="#feed">Feed</a></li>'
            '</ul></nav></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('stray punctuation pseudo-tag `<)` before <a>', '\n'.join(before_issues))
        self.assertIn('<li><a href="#feed">Feed</a></li>', repaired)
        self.assertNotIn('<)<a', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_stray_punctuation_pseudotag_removed' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_quoted_link_and_duplicate_angle_tag(self):
        content = (
            '<!doctype html><html><head>'
            '\'link\' rel="stylesheet" href="styles.css">'
            '</head><body><section><<div class="icon">✨</div></section></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        joined_issues = '\n'.join(before_issues)
        self.assertIn('quoted instead of angle-bracketed', joined_issues)
        self.assertIn('duplicate opening angle bracket', joined_issues)
        self.assertIn('<link rel="stylesheet" href="styles.css">', repaired)
        self.assertIn('<div class="icon">✨</div>', repaired)
        self.assertNotIn('\'link\' rel=', repaired)
        self.assertNotIn('<<div', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_quoted_known_opening_tag' for item in repairs)
        )
        self.assertTrue(
            any(item.get('kind') == 'html_duplicate_opening_angle_known_tag' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_malformed_stylesheet_link_attrs(self):
        content = (
            '<!doctype html><html><head>'
            '<link rel="stylesheet="href="styles.css">'
            '</head><body><main class="page"></main></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('malformed rel/href attributes', '\n'.join(before_issues))
        self.assertIn('<link rel="stylesheet" href="styles.css">', repaired)
        self.assertNotIn('rel="stylesheet="href=', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_stylesheet_link_attributes' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_invalid_stylesheet_link_rel_for_css_href(self):
        content = (
            '<!doctype html><html><head>'
            '<link rel="padding-box" href="https://fonts.googleapis.com/css2?family=Inter&display=swap">'
            '</head><body><main class="page"></main></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('invalid or missing rel attribute', '\n'.join(before_issues))
        self.assertIn(
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter&display=swap">',
            repaired,
        )
        self.assertNotIn('rel="padding-box"', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_invalid_stylesheet_link_rel' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_hero_header_landmark_containment_reports_syntax_issue(self):
        content = (
            '<!doctype html><html><body>'
            '<header class="hero"><div class="hero-content"><h1>Petsie</h1></div>'
            '<main id="gallery"><section>Cards</section></main>'
            '<footer>Footer</footer></header>'
            '</body></html>'
        )

        issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )

        self.assertIn(
            'HTML <main> landmark is nested inside hero/header opened',
            '\n'.join(issues),
        )

    def test_deterministic_syntax_repair_closes_hero_header_before_main(self):
        content = (
            '<!doctype html><html><body>'
            '<header class="hero"><div class="hero-content"><h1>Petsie</h1></div>'
            '<main id="gallery"><section>Cards</section></main>'
            '<footer>Footer</footer></header>'
            '</body></html>'
        )

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('</header>\n<main id="gallery">', repaired)
        self.assertLess(repaired.index('</header>\n<main'), repaired.index('<footer>Footer</footer>'))
        self.assertEqual(repaired.count('<header class="hero">'), 1)
        self.assertEqual(repaired.count('</header>'), 1)
        self.assertTrue(
            any(item.get('kind') == 'html_hero_header_landmark_containment_closed' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_header_followed_by_sibling_main_remains_clean(self):
        content = (
            '<!doctype html><html><body>'
            '<header class="hero"><div class="hero-content"><h1>Petsie</h1></div></header>'
            '<main id="gallery"><section>Cards</section></main>'
            '<footer>Footer</footer>'
            '</body></html>'
        )

        issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )

        self.assertEqual(issues, [])

    def test_deterministic_syntax_repair_normalizes_malformed_class_attribute_assignment(self):
        content = (
            '<!doctype html><html><body><section>'
            '<div class: gallery-item"><h3>One</h3><p>Copy</p></div>'
            '</section></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertTrue(before_issues)
        self.assertIn('class attribute uses `:` instead of `=`', before_issues[0])
        self.assertIn('<div class="gallery-item">', repaired)
        self.assertNotIn('class: gallery-item"', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_class_attribute_assignment' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_bare_duplicate_class_attribute_token(self):
        content = (
            '<!doctype html><html><body><header>'
            '<div class            class="nav-container"><a href="#work">Work</a></div>'
            '</header></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('bare duplicate token', '\n'.join(before_issues))
        self.assertIn('<div class="nav-container">', repaired)
        self.assertNotIn('class            class=', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_bare_duplicate_attribute_token' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_malformed_anchor_opening_tag(self):
        content = (
            '<!doctype html><html><body><nav><ul>'
            '<li><and href="#capabilities">Essenz</a></li>'
            '</ul></nav></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertTrue(before_issues)
        self.assertIn('stray closing tag </a>', '\n'.join(before_issues))
        self.assertIn('<li><a href="#capabilities">Essenz</a></li>', repaired)
        self.assertNotIn('<and href', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_anchor_opening_tag' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_obsolete_malformed_anchor_close(self):
        content = (
            '<!doctype html><html><body><nav><ul>'
            '<li><and href="#capabilities">Essenz</a></and></li>'
            '</ul></nav></body></html>'
        )

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('<li><a href="#capabilities">Essenz</a></li>', repaired)
        self.assertNotIn('</and>', repaired)
        self.assertTrue(
            any(
                item.get('kind') == 'html_malformed_anchor_opening_tag'
                and item.get('removed_obsolete_close') is True
                for item in repairs
            )
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_balanced_unsupported_href_element(self):
        content = (
            '<!doctype html><html><body><nav><ul>'
            '<li><far href="#details-1">Raum</far></li>'
            '</ul></nav></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertTrue(before_issues)
        self.assertIn('unsupported href element <far>', '\n'.join(before_issues))
        self.assertIn('<li><a href="#details-1">Raum</a></li>', repaired)
        self.assertNotIn('<far href', repaired)
        self.assertTrue(
            any(
                item.get('kind') == 'html_unsupported_href_element_rewritten_to_anchor'
                and item.get('from_tag') == 'far'
                for item in repairs
            )
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_sanity_allows_standard_and_svg_href_elements(self):
        content = (
            '<!doctype html><html><head>'
            '<link rel="stylesheet" href="styles.css">'
            '</head><body>'
            '<a href="#details">Details</a>'
            '<svg><use href="#icon-close"></use></svg>'
            '</body></html>'
        )

        self.assertEqual(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                content,
            ),
            [],
        )

    def test_deterministic_syntax_repair_unwraps_unsupported_navigation_anchor_wrapper(self):
        content = (
            '<!doctype html><html><body><header><nav><ul>'
            '<li><icon><a href="index.html" class="active">Suites</a></icon></li>'
            '</ul></nav></header></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('unsupported navigation anchor wrapper <icon>', '\n'.join(before_issues))
        self.assertIn('<li><a href="index.html" class="active">Suites</a></li>', repaired)
        self.assertNotIn('<icon>', repaired)
        self.assertTrue(
            any(
                item.get('kind') == 'html_unsupported_navigation_anchor_wrapper_unwrapped'
                and item.get('from_tag') == 'icon'
                for item in repairs
            )
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_malformed_formatted_anchor_fragment(self):
        content = (
            '<!doctype html><html><body><nav><ul>'
            '<li><<em>a href="#gallery">Trending</em></li>'
            '</ul></nav><main id="gallery">Feed</main></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('malformed formatted anchor fragment', '\n'.join(before_issues))
        self.assertIn('<li><a href="#gallery">Trending</a></li>', repaired)
        self.assertNotIn('<<em>a href', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_formatted_anchor_fragment' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_unwraps_literal_unsupported_tag(self):
        content = (
            '<!doctype html><html><body><header><nav><ul>'
            '<li><unsupported>Trending</unsupported></li>'
            '</ul></nav></header></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('literal unsupported placeholder tag <unsupported>', '\n'.join(before_issues))
        self.assertIn('<li>Trending</li>', repaired)
        self.assertNotIn('<unsupported>', repaired)
        self.assertTrue(
            any(
                item.get('kind') == 'html_literal_unsupported_tag_unwrapped'
                and item.get('from_tag') == 'unsupported'
                for item in repairs
            )
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_sanity_allows_non_nav_custom_anchor_wrapper(self):
        content = (
            '<!doctype html><html><body>'
            '<my-link><a href="index.html">Home</a></my-link>'
            '</body></html>'
        )

        self.assertEqual(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                content,
            ),
            [],
        )

    def test_deterministic_syntax_repair_removes_stray_unknown_closing_tag(self):
        content = (
            '<!doctype html><html><body>'
            '<section id="cta"><div><h2>Contact</h2></div></abs><footer>Fine</footer></section>'
            '</body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('stray closing tag </abs>', '\n'.join(before_issues))
        self.assertNotIn('</abs>', repaired)
        self.assertIn('<footer>Fine</footer></section>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_stray_unknown_closing_tag_removed' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_known_suffix_closing_tag(self):
        content = (
            '<!doctype html><html><body>'
            '<section><h2>Contact</h2></unsection>'
            '</body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('stray closing tag </unsection>', '\n'.join(before_issues))
        self.assertIn('<section><h2>Contact</h2></section>', repaired)
        self.assertNotIn('</unsection>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_closing_tag_known_suffix' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_stray_known_closing_tag(self):
        content = (
            '<!doctype html><html><body>'
            '<section><div class="feature-card"><h3>Title</h3><p>Copy</p></li></div></section>'
            '</body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('stray closing tag </li>', '\n'.join(before_issues))
        self.assertNotIn('</li>', repaired)
        self.assertIn('<div class="feature-card"><h3>Title</h3><p>Copy</p></div>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_stray_known_closing_tag_removed' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_malformed_class_only_opening_tag(self):
        content = (
            '<!doctype html><html><body><section>'
            '<div class="feature-card">'
            '<                    card-icon">04</div>'
            '<h3>Sublime Serenity</h3>'
            '<p>Climate-controlled environments.</p>'
            '</div></section></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('malformed class-only opening tag', '\n'.join(before_issues))
        self.assertIn('<div class="card-icon">04</div>', repaired)
        self.assertNotIn('<                    card-icon">04</div>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_malformed_class_only_opening_tag' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_unknown_class_wrapper_tag(self):
        content = (
            '<!doctype html><html><body>'
            '<article><int class="post-content"><p>Hello</p></int></article>'
            '<my-card class="post-content"><p>Allowed custom element</p></my-card>'
            '</body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertIn('unknown class/id wrapper tag <int>', '\n'.join(before_issues))
        self.assertIn('<div class="post-content"><p>Hello</p></div>', repaired)
        self.assertIn('<my-card class="post-content"><p>Allowed custom element</p></my-card>', repaired)
        self.assertNotIn('<int class="post-content">', repaired)
        self.assertTrue(
            any(
                item.get('kind') == 'html_unknown_class_wrapper_tag_rewritten_to_div'
                and item.get('from_tag') == 'int'
                for item in repairs
            )
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_corrects_known_css_property_typos(self):
        content = (
            '.section-title { font-scale: 2rem; } '
            '.cta-section h2 { font-mask: var(--font-heading); } '
            '.hero-overlay h1 { font-width: 900; } '
            '.logo { letter-length: -1px; } '
            '.lede { text-allign: center; } '
            '.page { scroll-template: smooth; } '
            '.hero { align-s_items: center; }'
        )

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('font-size: 2rem', repaired)
        self.assertIn('font-family: var(--font-heading)', repaired)
        self.assertIn('font-weight: 900', repaired)
        self.assertIn('letter-spacing: -1px', repaired)
        self.assertIn('text-align: center', repaired)
        self.assertIn('scroll-behavior: smooth', repaired)
        self.assertIn('align-items: center', repaired)
        self.assertNotIn('font-scale', repaired)
        self.assertNotIn('font-mask', repaired)
        self.assertNotIn('font-width', repaired)
        self.assertNotIn('letter-length', repaired)
        self.assertNotIn('text-allign', repaired)
        self.assertNotIn('scroll-template', repaired)
        self.assertNotIn('align-s_items', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_known_property_typo' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_invalid_css_property_value(self):
        content = '* { box-sizing: border-width; } .card { display: grid; }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS property `box-sizing` value `border-width`', '\n'.join(before_issues))
        self.assertIn('box-sizing: border-box', repaired)
        self.assertNotIn('box-sizing: border-width', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_invalid_property_value' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_sanity_flags_html_entity_fragment_inside_css_function(self):
        content = '.surface { background: rgba(255, 255, 25 &# 0, 0.02); }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('HTML entity fragment inside a CSS function', '\n'.join(before_issues))
        self.assertEqual(content, repaired)
        self.assertEqual([], repairs)

    def test_deterministic_syntax_repair_normalizes_malformed_css_declaration_name(self):
        content = 'nav { display: flex; justifyهم justify-content: center; }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS declaration name `justifyهم justify-content`', '\n'.join(before_issues))
        self.assertIn('justify-content: center', repaired)
        self.assertNotIn('justifyهم', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_malformed_declaration_name' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_removes_formatting_tag_inside_css_declaration_name(self):
        content = '.nav-links a { text-</strong>decoration: none; }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS declaration name `text-</strong>decoration`', '\n'.join(before_issues))
        self.assertIn('text-decoration: none', repaired)
        self.assertNotIn('</strong>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_formatting_tag_in_declaration_name' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_does_not_guess_ambiguous_css_property_names(self):
        content = 'nav li { foo bar: 10px; margin-length: 30px; margin-left: 30px; }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        issue_text = '\n'.join(before_issues)
        self.assertIn('CSS declaration name `foo bar`', issue_text)
        self.assertIn('CSS property `margin-length`', issue_text)
        self.assertEqual(repaired, content)
        self.assertEqual(repairs, [])

    def test_deterministic_syntax_sanity_ignores_css_punctuation_inside_functions_and_strings(self):
        content = (
            "@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400&display=swap'); "
            '/* fallback content context */ '
            ':root { --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); } '
            '.fallback { color: red; /* fallback content context */ display: block; }'
        )

        self.assertEqual(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                content,
            ),
            [],
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )
        self.assertEqual(repaired, content)
        self.assertEqual(repairs, [])

    def test_deterministic_syntax_repair_normalizes_duplicated_css_var_function(self):
        content = ':root { --gold: #c5a059; } .nav a { color: var\tvar(--gold); }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS token `var\tvar(`', '\n'.join(before_issues))
        self.assertIn('color: var(--gold)', repaired)
        self.assertNotIn('var\tvar(', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_invalid_duplicated_var_function_token' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_slash_css_var_reference(self):
        content = ':root { --transition-smooth: all 0.3s ease; } .card { transition: var/transition-smooth; }'

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS token `var/transition-smooth`', '\n'.join(before_issues))
        self.assertIn('transition: var(--transition-smooth)', repaired)
        self.assertNotIn('var/transition-smooth', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_invalid_var_slash_reference_token' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_invalid_background_token(self):
        content = ".hero { background: url('../images/hero.png') center/cover no-format; }"

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS background token `no-format`', '\n'.join(before_issues))
        self.assertIn("background: url('../images/hero.png') center/cover no-repeat", repaired)
        self.assertNotIn('no-format', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_invalid_background_shorthand_token' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_no_array_background_token(self):
        content = ".hero { background: url('../images/hero.png') center/cover no-array; }"

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('CSS background token `no-array`', '\n'.join(before_issues))
        self.assertIn("background: url('../images/hero.png') center/cover no-repeat", repaired)
        self.assertNotIn('no-array', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_invalid_background_shorthand_token' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_close_css_custom_property_near_miss(self):
        content = (
            ':root { --anthracite: #1a1a1a; --gold: #c5a059; } '
            '.btn-primary:hover { background: var(--anthracary); } '
            '.content-text h2 { color: var(--anthrit); }'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('variable reference `--anthracary`', '\n'.join(before_issues))
        self.assertIn('variable reference `--anthrit`', '\n'.join(before_issues))
        self.assertIn('background: var(--anthracite)', repaired)
        self.assertIn('color: var(--anthracite)', repaired)
        self.assertNotIn('--anthracary', repaired)
        self.assertNotIn('--anthrit', repaired)
        self.assertGreaterEqual(
            sum(
                1
                for item in repairs
                if item.get('kind') == 'css_custom_property_var_reference_near_miss'
            ),
            2,
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_css_custom_property_var_typo(self):
        content = (
            ':root { --accent-color: #ff4d00; } '
            '.cta-button { color: var(---accent-color); } '
            '.cta-button:hover { box-shadow: 0 0 30px var(---accent-color, #ff4d00); }'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('variable reference `---accent-color`', '\n'.join(before_issues))
        self.assertIn('color: var(--accent-color)', repaired)
        self.assertIn('var(--accent-color, #ff4d00)', repaired)
        self.assertNotIn('var(---accent-color', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_custom_property_var_reference_typo' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_keeps_intentional_triple_hyphen_custom_property(self):
        content = (
            ':root { ---accent-color: #ff4d00; } '
            '.cta-button { color: var(---accent-color); }'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertFalse(before_issues)
        self.assertEqual(content, repaired)
        self.assertEqual([], repairs)

    def test_deterministic_syntax_repair_normalizes_invalid_css_var_function_typo(self):
        content = (
            'body { background-color: varint(--color-bg); } '
            '.logo { color: varint(--color-primary); }'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertTrue(before_issues)
        self.assertIn('background-color: var(--color-bg)', repaired)
        self.assertIn('color: var(--color-primary)', repaired)
        self.assertNotIn('varint(', repaired)
        self.assertTrue(
            any(item.get('kind') == 'css_invalid_varint_function_token' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_closes_repeated_card_sibling_div(self):
        content = (
            '<!doctype html><html><body><section><div class="feature-grid">'
            '<div class="feature-card"><h3>One</h3><p>Copy</p>'
            '<div class="feature-card"><h3>Two</h3><p>Copy</p></div>'
            '</div></section></body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertTrue(before_issues)
        self.assertIn('</div>\n<div class="feature-card">', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_missing_repeated_sibling_div_close' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_closes_intervening_tag_before_parent_close(self):
        content = (
            '<!doctype html><html><body>'
            '<section class="final-cta"><div><h2>Bereit?</h2></div>'
            '<footer class="footer"><p>Footer</p></footer>'
            '</body></html>'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'html',
            content,
        )

        self.assertTrue(before_issues)
        self.assertIn('</section>\n</body>', repaired)
        self.assertTrue(
            any(item.get('kind') == 'html_missing_intervening_close_before_parent' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'html',
                repaired,
            )
        )

    def test_deterministic_syntax_repair_normalizes_simple_json_trailing_comma(self):
        content = '{ "name": "ollmo", "items": [1, 2,], }'

        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'json',
            content,
        )

        self.assertIn('"name": "ollmo"', repaired)
        self.assertTrue(
            any(item.get('kind') == 'json_remove_trailing_commas' for item in repairs)
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'json',
                repaired,
            )
        )

    def test_syntax_sanity_flags_unterminated_known_html_opening_tag(self):
        content = (
            '<!doctype html><html><body><section>'
            '<div classAnreise\n<p>Einfach über den See.</p></div>'
            '</section></body></html>'
        )

        issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )

        self.assertIn('unterminated opening tag <div>', '\n'.join(issues))

    def test_syntax_sanity_flags_unbalanced_html_attribute_quote(self):
        content = (
            '<!doctype html><html><body><section>'
            '<div class quite-card"><p>Nachhaltigkeit</p></div>'
            '</section></body></html>'
        )

        issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )

        self.assertIn('unbalanced quote in attributes', '\n'.join(issues))

    def test_syntax_sanity_flags_raw_stray_open_angle_text_fragment(self):
        content = (
            '<!doctype html><html><body><div>'
            '< رؤية <span class="username">Neon-Nico</span>'
            '</div></body></html>'
        )

        issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'html',
            content,
        )

        self.assertIn('stray opening angle bracket', '\n'.join(issues))

    def test_deterministic_syntax_repair_normalizes_observed_css_property_typos(self):
        content = (
            '.cta-button { letter-lag: 1px; } '
            '.section-title { margin-blop: 1rem; }'
        )

        before_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            'css',
            content,
        )
        repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
            'css',
            content,
        )

        self.assertIn('letter-lag', '\n'.join(before_issues))
        self.assertIn('margin-blop', '\n'.join(before_issues))
        self.assertIn('letter-spacing: 1px', repaired)
        self.assertIn('margin-bottom: 1rem', repaired)
        self.assertTrue(
            all(
                item.get('kind') == 'css_known_property_typo'
                for item in repairs
            )
        )
        self.assertFalse(
            ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
                'css',
                repaired,
            )
        )

    def test_closure_review_blocks_malformed_html_css_syntax_before_freeze(self):
        prompt = 'Create index.html and styles.css artifacts for a sci-fi landing page.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_malformed_page_artifacts',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><html><head>'
                        '<link rel="stylesheet" href="styles.css"></app></head>'
                        '<body><nav><li><lar>Tech</a></li></nav>'
                        '<div class="show</strong>case-text">Copy</div></body></html>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.section-title { font-scale: 2rem; }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        syntax_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'text_artifact_syntax_sanity'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(len(syntax_checks), 2)
        payload_text = '\n'.join(item['content_payload'] for item in syntax_checks)
        self.assertIn('stray closing tag </app>', payload_text)
        self.assertIn('HTML attribute contains markup-like closing tag', payload_text)
        self.assertIn('font-scale', payload_text)
        self.assertTrue(all(item['repair_action'] == 'retry_same_branch' for item in syntax_checks))
        syntax_by_extension = {item['text_artifact_extension']: item for item in syntax_checks}
        self.assertEqual(syntax_by_extension['html']['text_artifact_target_path'], '/tmp/index.html')
        self.assertEqual(syntax_by_extension['html']['artifact_request']['target_path'], '/tmp/index.html')
        self.assertEqual(syntax_by_extension['css']['text_artifact_target_path'], '/tmp/styles.css')
        self.assertEqual(syntax_by_extension['css']['artifact_request']['target_path'], '/tmp/styles.css')

    def test_syntax_repair_feedback_does_not_inherit_image_dependent_chat_edges(self):
        checks = self.owner._text_artifact_syntax_sanity_checks(
            artifact_payload={
                'artifacts': [
                    {
                        'type': 'text',
                        'path': '/tmp/index.html',
                        'name': 'index',
                        'mime_type': 'text/html',
                        'content': '<!doctype html><html><head></app></head><body></body></html>',
                    },
                ],
            },
        )
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['repair_scope'], 'syntax_only')
        self.assertEqual(checks[0]['resource_class'], 'text_io')
        self.assertEqual(checks[0]['dependency_policy'], 'target_artifact_snapshot_only')

        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='syntax defect',
            checks=checks,
            request_phase_graph={
                'current_phase_id': 'phase-chat-prepare',
                'current_phase_capability': 'chat',
                'downstream_branches': [
                    {
                        'branch_id': 'branch-chat-final',
                        'phase_id': 'branch-chat-final',
                        'capability': 'chat',
                        'depends_on': ['branch-image-1', 'branch-image-2'],
                    },
                ],
                'phases': [],
                'output_obligations': [],
                'prompt_intent': {'downstream_follow_up_capabilities': ['image_generation']},
            },
        )

        self.assertIsNotNone(feedback)
        item = feedback['items'][0]
        self.assertNotIn('depends_on', item)
        self.assertEqual(item['repair_scope'], 'syntax_only')
        self.assertEqual(item['resource_class'], 'text_io')
        self.assertEqual(item['dependency_policy'], 'target_artifact_snapshot_only')
        contract = feedback['repair_loop']['promoted_contracts'][0]
        self.assertEqual(contract['dependency_policy'], 'target_artifact_snapshot_only')

    def test_late_fill_shape_allows_syntax_repair_next_to_images_only(self):
        branches = [
            {
                'branch_id': 'branch-image-1',
                'phase_id': 'branch-image-1',
                'capability': 'image_generation',
                'output_type': 'image',
            },
            {
                'branch_id': 'repair-chat-syntax',
                'phase_id': 'repair-chat-syntax',
                'capability': 'chat',
                'output_type': 'text',
                'role': 'text_artifact_syntax_repair',
                'repair_scope': 'syntax_only',
                'resource_class': 'text_io',
                'dependency_policy': 'target_artifact_snapshot_only',
                'text_artifact_target_path': '/tmp/index.html',
                'artifact_request': {'target_path': '/tmp/index.html', 'source': 'closure_syntax_repair'},
            },
            {
                'branch_id': 'branch-chat-helper',
                'phase_id': 'branch-chat-helper',
                'capability': 'chat',
                'output_type': 'text',
                'role': 'semantic_review_helper',
            },
        ]

        shaped, policy = self.late_fill_owner.shape_active_late_fill_branches(
            branches,
            artifact_gap={
                'repair_scope': 'syntax_only',
                'resource_class': 'text_io',
                'dependency_policy': 'target_artifact_snapshot_only',
            },
        )

        self.assertEqual([branch['branch_id'] for branch in shaped], ['branch-image-1', 'repair-chat-syntax'])
        self.assertEqual(policy['gpu_heavy_guard'], 'deferred_non_image_branches')
        self.assertEqual(policy['deferred_branch_count'], 1)

    def test_closure_review_blocks_balanced_unsupported_href_element(self):
        prompt = 'Create index.html as a local artifact for a brutalist landing page.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_balanced_unsupported_href_element',
            'output_text': 'Artifact generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><html><body><nav><ul>'
                        '<li><far href="#details-1">Raum</far></li>'
                        '</ul></nav><main id="details-1">Raum</main></body></html>'
                    ),
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        syntax_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'text_artifact_syntax_sanity'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(len(syntax_checks), 1)
        self.assertIn('unsupported href element <far>', syntax_checks[0]['content_payload'])
        self.assertEqual(syntax_checks[0]['repair_action'], 'retry_same_branch')
        self.assertEqual(syntax_checks[0]['text_artifact_target_path'], '/tmp/index.html')

    def test_closure_review_blocks_malformed_class_attr_and_font_width_typo(self):
        prompt = 'Create index.html and styles.css artifacts for a playful landing page.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_malformed_attr_page_artifacts',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><html><head>'
                        '<link rel="stylesheet" href="styles.css"></head>'
                        '<body><section><div class: gallery-item">Copy</div></section></body></html>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.hero-overlay h1 { font-width: 900; }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        syntax_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'text_artifact_syntax_sanity'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(len(syntax_checks), 2)
        payload_text = '\n'.join(item['content_payload'] for item in syntax_checks)
        self.assertIn('class attribute uses `:` instead of `=`', payload_text)
        self.assertIn('font-width', payload_text)
        self.assertTrue(all(item['repair_action'] == 'retry_same_branch' for item in syntax_checks))

    def test_closure_review_blocks_nav_wrapper_and_malformed_css_declaration_name(self):
        prompt = 'Create index.html and styles.css artifacts for a boutique hotel page.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_nav_wrapper_and_malformed_css',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><html><head>'
                        '<link rel="stylesheet" href="styles.css"></head>'
                        '<body><nav><ul><li><icon><a href="suites.html">Suites</a></icon></li>'
                        '</ul></nav></body></html>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': 'nav { display: flex; justifyهم justify-content: center; }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        syntax_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'text_artifact_syntax_sanity'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(len(syntax_checks), 2)
        payload_text = '\n'.join(item['content_payload'] for item in syntax_checks)
        self.assertIn('unsupported navigation anchor wrapper <icon>', payload_text)
        self.assertIn('CSS declaration name `justifyهم justify-content`', payload_text)
        self.assertTrue(all(item['repair_action'] == 'retry_same_branch' for item in syntax_checks))

    def test_closure_review_accepts_basic_valid_html_css_syntax(self):
        prompt = 'Create index.html and styles.css artifacts for a sci-fi landing page.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_valid_page_artifacts',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': (
                        '<!doctype html><html><head>'
                        '<link rel="stylesheet" href="styles.css"></head>'
                        '<body><nav><a href="#labs">Labs</a></nav>'
                        '<main class="showcase-text">Copy</main></body></html>'
                    ),
                },
                {
                    'type': 'text',
                    'path': '/tmp/styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                    'content': '.section-title { font-size: 2rem; } .showcase-text { display: block; }',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'text_artifact_syntax_sanity'
            ]
        )

    def test_closure_review_groups_duplicate_text_artifacts_by_runtime_source_name(self):
        prompt = 'Create index.html and styles.css, and link the stylesheet from the HTML.'
        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        payload = {
            'id': 'resp_duplicate_css_records',
            'output_text': 'Artifacts generated.',
            'runtime': {'request_phase_graph': graph},
            'artifacts': [
                {
                    'type': 'text',
                    'path': '/tmp/20260516_styles.css',
                    'name': 'styles',
                    'mime_type': 'text/css',
                },
                {
                    'type': 'text',
                    'path': '/tmp/index.html',
                    'name': 'index',
                    'mime_type': 'text/html',
                    'content': '<!doctype html><link rel="stylesheet" href="20260516_styles.css">',
                },
                {
                    'type': 'text',
                    'path': '/tmp/20260516_styles.css',
                    'name': 'index',
                    'mime_type': 'text/css',
                },
            ],
            'late_fill': {
                'fill_results': [
                    {
                        'branch_id': 'branch-text_artifact-2',
                        'phase_id': 'phase-4',
                        'capability': 'chat',
                        'saved_text_path': '/tmp/20260516_styles.css',
                        'text_artifact_extension': 'css',
                        'text_artifact_source_name': 'styles',
                    }
                ]
            },
        }

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload=payload,
        )

        self.assertFalse(
            [
                item for item in review['checks']
                if item.get('check_kind') == 'linked_artifact_binding'
            ]
        )

    def test_closure_review_surfaces_text_only_artifact_claim_as_open_check(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'mode': 'single_phase',
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'planned',
                    'role': 'final_output',
                    'required': True,
                }
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'planned',
                }
            ],
        }
        payload = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_text_only_artifact_claim',
                'output_text': (
                    '### Materialisierte Artefakte\n\n'
                    'Artefakt 1: Narratives Element -> fulfilled\n'
                ),
            }
        )
        runtime = dict(payload.get('runtime') or {})
        runtime['request_phase_graph'] = phase_graph
        payload['runtime'] = runtime

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': 'Plane fünf Artefakte.'},
            artifact_payload=payload,
        )

        self.assertEqual(review['status'], 'pending')
        truth_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'truth_guard'
        ]
        self.assertEqual(len(truth_checks), 1)
        self.assertEqual(truth_checks[0]['evidence'], 'text_only_artifact_claim_guard')
        self.assertEqual(review['ghost_repair_feedback']['status'], 'repair_required')
        self.assertEqual(review['ghost_repair_feedback']['target'], 'request_ir_patch')

    def test_truth_guard_records_claimed_tts_capability_from_audio_artifact_claim(self):
        payload = {
            'id': 'resp_text_only_tts_claim',
            'output_text': (
                'Artefakt 4 (Audio): Nicht materialisiert. Die Ausführung erfordert einen '
                '`text_to_speech`-Übergang. (Status: deferred)'
            ),
        }

        updated = self.owner.truth_gate_response_output_claims(payload)

        truth_guard = updated['runtime']['truth_guard']
        self.assertEqual(truth_guard['claimed_capabilities'], ['text_to_speech'])

    def test_closure_review_flags_available_tts_claim_as_unmaterialized(self):
        self.owner.hooks['load_running_instances'] = lambda: [
            {
                'instance_id': 'tts-ready-1',
                'capability': 'text_to_speech',
                'supported_capabilities': ['text_to_speech'],
                'readiness': 'ready',
                'process_alive': True,
                'port_listening': True,
            }
        ]
        phase_graph = {
            'current_phase_id': 'phase-1',
            'mode': 'single_phase',
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'planned',
                    'role': 'final_output',
                    'required': True,
                }
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'planned',
                }
            ],
        }
        payload = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_text_only_tts_claim',
                'output_text': (
                    '### Materialisierte Artefakte\n\n'
                    'Artefakt 4 (Audio): Nicht materialisiert. Die Ausführung erfordert einen '
                    '`text_to_speech`-Übergang. (Status: deferred)'
                ),
            }
        )
        runtime = dict(payload.get('runtime') or {})
        runtime['request_phase_graph'] = phase_graph
        payload['runtime'] = runtime

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': 'Plane fünf Artefakte.'},
            artifact_payload=payload,
        )

        capability_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'truth_guard_capability'
        ]
        self.assertEqual(len(capability_checks), 1)
        self.assertEqual(capability_checks[0]['capability'], 'text_to_speech')
        self.assertEqual(capability_checks[0]['status'], 'pending')
        self.assertEqual(capability_checks[0]['evidence'], 'runtime_capability_available_but_unmaterialized')
        self.assertEqual(capability_checks[0]['available_instance_ids'], ['tts-ready-1'])

    def test_closure_review_blocks_unavailable_tts_claim(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'mode': 'single_phase',
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'planned',
                    'role': 'final_output',
                    'required': True,
                }
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'planned',
                }
            ],
        }
        payload = self.owner.truth_gate_response_output_claims(
            {
                'id': 'resp_text_only_tts_claim',
                'output_text': (
                    '### Materialisierte Artefakte\n\n'
                    'Artefakt 4 (Audio): Nicht materialisiert. Die Ausführung erfordert einen '
                    '`text_to_speech`-Übergang. (Status: deferred)'
                ),
            }
        )
        runtime = dict(payload.get('runtime') or {})
        runtime['request_phase_graph'] = phase_graph
        payload['runtime'] = runtime

        review = self.owner.build_graph_closure_review(
            payload['output_text'],
            request_payload={'ghost_route': True, 'prompt': 'Plane fünf Artefakte.'},
            artifact_payload=payload,
        )

        capability_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'truth_guard_capability'
        ]
        self.assertEqual(len(capability_checks), 1)
        self.assertEqual(capability_checks[0]['status'], 'blocked')
        self.assertEqual(capability_checks[0]['evidence'], 'runtime_capability_unavailable')

    def test_closure_review_emits_repair_feedback_for_missing_stt_obligation(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'mode': 'single_phase',
            'prompt': 'Transkribiere diese Audiodatei.',
            'prompt_intent': {
                'primary_capability': 'speech_to_text',
                'requests_speech_to_text_output': True,
            },
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                }
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Ich kann die Datei transkribieren.',
            request_payload={'ghost_route': True, 'prompt': 'Transkribiere diese Audiodatei.'},
            artifact_payload={
                'output_text': 'Ich kann die Datei transkribieren.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        self.assertEqual(review['status'], 'pending')
        missing = [
            item for item in review['checks']
            if item.get('evidence') == 'intent_graph_adequacy_missing_capability_obligation'
        ]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]['capability'], 'speech_to_text')
        feedback = review['ghost_repair_feedback']
        self.assertEqual(feedback['status'], 'repair_required')
        self.assertEqual(feedback['target'], 'request_ir_patch')
        self.assertEqual(feedback['patch_scope'], 'current_working_frame_request_phase_graph')
        self.assertTrue(feedback['preserve_request_id'])
        self.assertEqual(feedback['repair_mode'], 'bounded_graph_patch')
        self.assertIn('do not answer prose', feedback['instruction'])
        self.assertEqual(feedback['items'][0]['capability'], 'speech_to_text')
        self.assertEqual(feedback['items'][0]['repair_action'], 'rebuild_from_promoted_obligations')
        self.assertEqual(feedback['repair_loop']['status'], 'promoted')
        self.assertFalse(feedback['repair_loop']['auto_execute'])
        self.assertFalse(feedback['repair_loop']['requires_promotion'])
        self.assertEqual(feedback['repair_loop']['promoted_contract_count'], 1)
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 1)
        self.assertEqual(feedback['repair_rebuild_contracts'][0]['status'], 'promoted')
        self.assertEqual(
            feedback['repair_rebuild_contracts'][0]['execution_policy'],
            'blocked_until_dependency_evidence',
        )
        self.assertEqual(feedback['items'][0]['repair_contract']['status'], 'promoted')
        self.assertEqual(feedback['repair_loop']['next_actions'], ['rebuild_from_promoted_obligations'])

    def test_graph_closure_review_marks_dependency_repair_action(self):
        execution_contract = {
            'kind': 'ollmo.execution_contract',
            'branch_id': 'branch-image-1',
            'phase_id': 'phase-2',
            'capability': 'image_generation',
            'depends_on': ['phase-1'],
            'output_contract': {'output_type': 'image', 'required': True},
        }
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'decision_contract': {
                'semantic_decision_proposals': [
                    {
                        'kind': 'ollmo.semantic_decision_proposal',
                        'proposal_id': 'semantic-decision-image-1',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-image-1',
                        'obligation_id': 'obligation-phase-2',
                        'decision_action': 'continue_branch_local_work',
                        'reason': 'the branch-local dependency failure remains runtime-owned',
                        'authority': 'advisory_read_model_only',
                    }
                ],
            },
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'depends_on': ['phase-1'],
                    'execution_contract': execution_contract,
                },
            ],
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'chat', 'status': 'completed'},
                {'phase_id': 'phase-2', 'branch_id': 'branch-image-1', 'capability': 'image_generation', 'depends_on': ['phase-1']},
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Image prompt ready.',
            request_payload={'ghost_route': True, 'prompt': 'Create an image after writing the prompt.'},
            artifact_payload={
                'output_text': 'Image prompt ready.',
                'runtime': {'request_phase_graph': phase_graph},
                'late_fill': {
                    'status': 'failed',
                    'failed_branches': [
                        {
                            'branch_id': 'branch-image-1',
                            'phase_id': 'phase-2',
                            'capability': 'image_generation',
                            'depends_on': ['phase-1'],
                            'execution_contract': execution_contract,
                            'recovery_context': {
                                'suggested_action': 'repair_dependency_chain',
                                'repair_required': True,
                                'blocked_by_dependency_input': True,
                            },
                            'recovery_state': {
                                'suggested_action': 'repair_dependency_chain',
                                'repair_required': True,
                                'blocked_by_dependency_input': True,
                            },
                        }
                    ],
                },
            },
        )

        image_check = next(item for item in review['checks'] if item.get('branch_id') == 'branch-image-1')
        self.assertEqual(image_check['status'], 'blocked')
        self.assertEqual(image_check['semantic_decision_action'], 'continue_branch_local_work')
        self.assertEqual(image_check['repair_action'], 'repair_dependency_chain')
        self.assertEqual(image_check['execution_contract'], execution_contract)
        feedback_item = next(item for item in review['ghost_repair_feedback']['items'] if item.get('branch_id') == 'branch-image-1')
        self.assertEqual(feedback_item['repair_action'], 'repair_dependency_chain')
        self.assertEqual(feedback_item['execution_contract'], execution_contract)

    def test_dependency_repair_contract_runs_when_block_resolution_evidence_exists(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='final comparison lacked dependency evidence before repair',
            request_phase_graph={
                'current_phase_id': 'phase-4',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'branch-final-comparison',
                    'phase_id': 'phase-5',
                    'obligation_id': 'obligation-phase-5',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1', 'branch-speech_to_text-1'],
                    'repair_action': 'repair_dependency_chain',
                    'blocked_by_dependency_input': True,
                    'content_payload': (
                        'phase-1: Original postcard text.\n\n'
                        'branch-speech_to_text-1: Transcribed postcard text.'
                    ),
                    'content_payload_source': 'late_fill_results:phase-1,branch-speech_to_text-1',
                    'review_criteria': ['compare transcript against original text'],
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'repair_dependency_chain')
        self.assertEqual(contract['execution_policy'], 'schedule_late_fill_branch')
        self.assertTrue(contract['auto_execute'])
        self.assertEqual(feedback['repair_loop']['status'], 'promoted')
        self.assertTrue(feedback['repair_loop']['auto_execute'])
        self.assertEqual(feedback['repair_loop']['executable_contract_count'], 1)
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 0)
        self.assertEqual(feedback['items'][0]['repair_execution_policy'], 'schedule_late_fill_branch')

    def test_dependency_repair_contract_stays_blocked_with_only_planned_input_refs(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='dependency evidence is still absent',
            request_phase_graph={
                'current_phase_id': 'phase-4',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'branch-final-comparison',
                    'phase_id': 'phase-5',
                    'obligation_id': 'obligation-phase-5',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1', 'branch-speech_to_text-1'],
                    'input_refs': [
                        {'kind': 'phase_output', 'phase_id': 'phase-1', 'role': 'dependency'},
                        {'kind': 'phase_output', 'phase_id': 'branch-speech_to_text-1', 'role': 'dependency'},
                    ],
                    'repair_action': 'repair_dependency_chain',
                    'blocked_by_dependency_input': True,
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'repair_dependency_chain')
        self.assertEqual(contract['execution_policy'], 'blocked_until_dependency_evidence')
        self.assertFalse(contract['auto_execute'])
        self.assertTrue(contract['materialization_blocked'])
        self.assertEqual(contract['blocked_scope'], 'target_materialization')
        self.assertEqual(contract['blocked_prerequisite'], 'dependency_evidence')
        self.assertTrue(contract['repair_work_available'])
        self.assertEqual(contract['repair_work_policy'], 'repair_dependency_chain_before_materialization')
        self.assertFalse(contract['needs_external_input'])
        self.assertFalse(feedback['repair_loop']['auto_execute'])
        self.assertEqual(feedback['repair_loop']['executable_contract_count'], 0)
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 1)
        self.assertTrue(feedback['repair_loop']['repair_work_available'])
        self.assertEqual(feedback['repair_loop']['repair_work_available_count'], 1)
        self.assertEqual(feedback['repair_loop']['needs_external_input_count'], 0)
        self.assertEqual(feedback['items'][0]['repair_execution_policy'], 'blocked_until_dependency_evidence')
        self.assertTrue(feedback['items'][0]['repair_work_available'])

    def test_dependency_rebind_contract_runs_when_concrete_evidence_exists(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='dependency evidence exists but needs rebinding',
            request_phase_graph={
                'current_phase_id': 'phase-4',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'branch-final-comparison',
                    'phase_id': 'phase-5',
                    'obligation_id': 'obligation-phase-5',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1', 'branch-speech_to_text-1'],
                    'repair_action': 'rebind_dependency_evidence',
                    'blocked_by_dependency_input': True,
                    'input_refs': [
                        {
                            'kind': 'artifact',
                            'artifact_ref': 'artifact:text_transcript_1',
                            'role': 'dependency',
                        }
                    ],
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'rebind_dependency_evidence')
        self.assertEqual(contract['execution_policy'], 'schedule_late_fill_branch')
        self.assertTrue(contract['auto_execute'])
        self.assertTrue(feedback['repair_loop']['auto_execute'])
        self.assertEqual(feedback['repair_loop']['executable_contract_count'], 1)
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 0)

    def test_dependency_rebind_contract_blocks_without_concrete_evidence(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='dependency evidence is requested but absent',
            request_phase_graph={
                'current_phase_id': 'phase-4',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'branch-final-comparison',
                    'phase_id': 'phase-5',
                    'obligation_id': 'obligation-phase-5',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1', 'branch-speech_to_text-1'],
                    'repair_action': 'rebind_dependency_evidence',
                    'blocked_by_dependency_input': True,
                    'input_refs': [
                        {'kind': 'phase_output', 'phase_id': 'branch-speech_to_text-1', 'role': 'dependency'}
                    ],
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'rebind_dependency_evidence')
        self.assertEqual(contract['execution_policy'], 'blocked_until_dependency_evidence')
        self.assertFalse(contract['auto_execute'])
        self.assertTrue(contract['materialization_blocked'])
        self.assertEqual(contract['blocked_prerequisite'], 'dependency_evidence')
        self.assertTrue(contract['repair_work_available'])
        self.assertEqual(contract['repair_work_policy'], 'rebind_dependency_evidence_before_materialization')
        self.assertFalse(contract['needs_external_input'])
        self.assertFalse(feedback['repair_loop']['auto_execute'])
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 1)
        self.assertTrue(feedback['repair_loop']['repair_work_available'])

    def test_branch_contract_repair_runs_when_execution_contract_exists(self):
        execution_contract = {
            'kind': 'ollmo.execution_contract',
            'phase_id': 'phase-2',
            'branch_id': 'branch-image_generation-1',
            'capability': 'image_generation',
            'output_type': 'image',
        }
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='branch contract exists and can be materialized',
            request_phase_graph={
                'current_phase_id': 'phase-1',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'branch-image_generation-1',
                    'phase_id': 'phase-2',
                    'obligation_id': 'obligation-phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'repair_action': 'repair_branch_contract',
                    'blocked_by_branch_contract': True,
                    'execution_contract': execution_contract,
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'repair_branch_contract')
        self.assertEqual(contract['execution_policy'], 'schedule_late_fill_branch')
        self.assertTrue(contract['auto_execute'])
        self.assertEqual(contract['execution_contract'], execution_contract)

    def test_branch_contract_repair_blocks_without_execution_contract(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='branch contract is still absent',
            request_phase_graph={
                'current_phase_id': 'phase-1',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'branch-image_generation-1',
                    'phase_id': 'phase-2',
                    'obligation_id': 'obligation-phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'repair_action': 'repair_branch_contract',
                    'blocked_by_branch_contract': True,
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'repair_branch_contract')
        self.assertEqual(contract['execution_policy'], 'blocked_until_branch_contract')
        self.assertFalse(contract['auto_execute'])
        self.assertTrue(contract['materialization_blocked'])
        self.assertEqual(contract['blocked_scope'], 'target_materialization')
        self.assertEqual(contract['blocked_prerequisite'], 'branch_execution_contract')
        self.assertTrue(contract['repair_work_available'])
        self.assertEqual(contract['repair_work_policy'], 'build_branch_contract_before_materialization')
        self.assertFalse(contract['needs_external_input'])

    def test_promoted_obligation_rebuild_runs_when_concrete_image_branch_exists(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='closure promoted a missing image branch',
            request_phase_graph={
                'current_phase_id': 'phase-1',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'repair-missing-obligation-image',
                    'phase_id': 'repair-missing-obligation-image',
                    'obligation_id': 'obligation-image-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'repair_action': 'rebuild_from_promoted_obligations',
                    'artifact_prompt': 'A luminous deep-sea research station under black water.',
                    'artifact_prompt_source': 'closure_promoted_obligation',
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'rebuild_from_promoted_obligations')
        self.assertEqual(contract['execution_policy'], 'schedule_late_fill_branch')
        self.assertTrue(contract['auto_execute'])
        self.assertFalse(contract['materialization_blocked'])
        self.assertTrue(contract['repair_work_available'])
        self.assertFalse(contract['needs_external_input'])
        self.assertEqual(
            contract['artifact_prompt'],
            'A luminous deep-sea research station under black water.',
        )
        self.assertTrue(feedback['repair_loop']['auto_execute'])
        self.assertEqual(feedback['repair_loop']['executable_contract_count'], 1)
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 0)

    def test_counted_image_adequacy_feedback_expands_to_distinct_repair_contracts(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='closure found three missing promoted image obligations',
            request_phase_graph={
                'current_phase_id': 'phase-1',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
                'prompt_intent': {
                    'counted_visual_output_obligation': True,
                    'requested_visual_output_count': 3,
                    'downstream_follow_up_capabilities': ['image_generation'],
                },
            },
            checks=[
                {
                    'check_kind': 'intent_graph_adequacy',
                    'status': 'pending',
                    'obligation_id': 'missing-obligation-image',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'expected_count': 3,
                    'actual_count': 0,
                    'missing_count': 3,
                    'evidence': 'intent_graph_adequacy_missing_output_obligation',
                }
            ],
        )

        self.assertIsNotNone(feedback)
        self.assertEqual(len(feedback['items']), 3)
        self.assertEqual(len(feedback['repair_rebuild_contracts']), 3)
        self.assertEqual(
            [item.get('branch_id') for item in feedback['items']],
            [
                'repair-missing-obligation-image-1',
                'repair-missing-obligation-image-2',
                'repair-missing-obligation-image-3',
            ],
        )
        self.assertEqual([item.get('missing_count') for item in feedback['items']], [1, 1, 1])
        self.assertEqual([item.get('total_missing_count') for item in feedback['items']], [3, 3, 3])
        self.assertEqual(
            [item.get('repair_occurrence_index') for item in feedback['items']],
            [1, 2, 3],
        )
        self.assertEqual(
            [contract.get('branch_id') for contract in feedback['repair_rebuild_contracts']],
            [
                'repair-missing-obligation-image-1',
                'repair-missing-obligation-image-2',
                'repair-missing-obligation-image-3',
            ],
        )
        self.assertEqual(
            len({contract.get('contract_id') for contract in feedback['repair_rebuild_contracts']}),
            3,
        )
        self.assertEqual(feedback['repair_loop']['promoted_contract_count'], 3)
        self.assertEqual(feedback['repair_loop']['executable_contract_count'], 3)

    def test_promoted_obligation_rebuild_blocks_when_only_action_label_exists(self):
        feedback = self.owner.build_ghost_repair_feedback(
            review_status='pending',
            reason='closure knows graph underplanned but no concrete branch exists yet',
            request_phase_graph={
                'current_phase_id': 'phase-1',
                'current_phase_capability': 'chat',
                'mode': 'phase_chain',
            },
            checks=[
                {
                    'check_kind': 'graph_obligation',
                    'status': 'blocked',
                    'branch_id': 'repair-promoted-obligations',
                    'phase_id': 'repair-promoted-obligations',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'repair_action': 'rebuild_from_promoted_obligations',
                }
            ],
        )

        contract = feedback['repair_rebuild_contracts'][0]
        self.assertEqual(contract['repair_action'], 'rebuild_from_promoted_obligations')
        self.assertEqual(contract['execution_policy'], 'blocked_until_promoted_obligation_branch')
        self.assertFalse(contract['auto_execute'])
        self.assertTrue(contract['materialization_blocked'])
        self.assertEqual(contract['blocked_scope'], 'target_materialization')
        self.assertEqual(contract['blocked_prerequisite'], 'promoted_obligation_branch')
        self.assertTrue(contract['repair_work_available'])
        self.assertEqual(
            contract['repair_work_policy'],
            'rebuild_promoted_obligation_branch_before_materialization',
        )
        self.assertFalse(contract['needs_external_input'])
        self.assertFalse(feedback['repair_loop']['auto_execute'])
        self.assertEqual(feedback['repair_loop']['blocked_contract_count'], 1)

    def test_graph_closure_review_marks_contract_repair_action(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'output_obligations': [
                {
                    'obligation_id': 'missing-image-contract',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                    'role': 'visual_output',
                },
            ],
            'phases': [{'phase_id': 'phase-1', 'capability': 'chat', 'status': 'completed'}],
        }

        review = self.owner.build_graph_closure_review(
            'Image prompt ready.',
            request_payload={'ghost_route': True, 'prompt': 'Create an image.'},
            artifact_payload={
                'output_text': 'Image prompt ready.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        image_check = next(item for item in review['checks'] if item.get('capability') == 'image_generation')
        self.assertEqual(image_check['repair_action'], 'repair_branch_contract')
        self.assertEqual(review['ghost_repair_feedback']['items'][0]['repair_action'], 'repair_branch_contract')
        self.assertEqual(
            review['ghost_repair_feedback']['repair_loop']['next_actions'],
            ['repair_branch_contract'],
        )

    def test_closure_review_uses_decision_contract_repair_candidate_for_open_check(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'decision_contract': {
                'kind': 'ollmo.ghost_decision_contract',
                'decision_contract_version': 1,
                'next_decision_priorities': ['continue_or_repair_open_promoted_obligations'],
                'semantic_planning_contract': {
                    'kind': 'ollmo.ghost_semantic_planning_contract',
                    'authority': 'advisory_read_model_only',
                    'current_focus': ['continue_or_repair_open_promoted_obligations'],
                },
                'promotion_suggestions': [
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'promotion_suggestions': [
                            {
                                'candidate_id': 'candidate-final-review',
                                'promotion_reason': 'final review is owed after dependency evidence exists',
                            }
                        ],
                    }
                ],
                'waiver_candidates': [
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'waiver_candidates': [
                            {
                                'obligation_id': 'obligation-phase-2',
                                'waiver_reason': 'user explicitly releases the final review',
                            }
                        ],
                    }
                ],
                'repair_candidates': [
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'capability': 'chat',
                        'evidence_requirements': ['phase-1 dependency evidence'],
                        'repair_candidates': [
                            {
                                'task_id': 'task-phase-2',
                                'repair_action': 'repair_dependency_chain',
                                'reason': 'decision contract says final review needs dependency evidence',
                            }
                        ],
                    }
                ],
                'reconsideration_candidates': [
                    {
                        'candidate_id': 'candidate-extra-image',
                        'decision': 'reserved',
                        'reconsiderable': True,
                    }
                ],
                'block_resolution_reflex': {
                    'kind': 'ollmo.block_resolution_reconsideration_reflex',
                    'status': 'active',
                    'signal_count': 1,
                    'authority': 'advisory_read_model_only',
                },
                'reconsideration_reflex_signals': [
                    {
                        'kind': 'ollmo.block_resolution_signal',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'obligation_id': 'obligation-phase-2',
                        'category': 'open_obligation',
                        'action': 'continue_or_repair_promoted_work_before_freeze',
                        'resolution_policy': 'right_sized_verified_state_transition',
                        'principle': 'the_solution_to_a_block_is_the_blocks_own_resolution',
                        'authority': 'read_model_only_not_runtime_truth',
                    }
                ],
                'active_reconsideration_review': {
                    'kind': 'ollmo.active_reconsideration_review',
                    'status': 'active',
                    'authority': 'advisory_read_model_only',
                    'decision_count': 1,
                },
                'active_reconsideration_decisions': [
                    {
                        'kind': 'ollmo.active_reconsideration_decision',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'obligation_id': 'obligation-phase-2',
                        'source_category': 'open_obligation',
                        'review_type': 'continuation_or_repair_review',
                        'recommended_action': 'continue_or_repair_promoted_work_before_freeze',
                        'authority': 'advisory_read_model_only',
                    }
                ],
                'semantic_quality_review': {
                    'kind': 'ollmo.semantic_quality_review',
                    'status': 'required',
                    'contract_count': 1,
                    'authority': 'advisory_until_promoted_semantic_verifier',
                },
                'semantic_quality_contracts': [
                    {
                        'kind': 'ollmo.semantic_quality_contract',
                        'quality_review_id': 'semantic-quality-final-review',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'status': 'pending_semantic_review',
                        'semantic_review_lens': 'integrator',
                        'success_definition': 'Final review must integrate dependency evidence before it can pass.',
                        'failure_modes': ['dependency_evidence_ignored'],
                        'evidence_requirements': ['phase-1 dependency evidence'],
                        'review_criteria': ['final review uses dependency evidence'],
                    }
                ],
                'semantic_review_lens_review': {
                    'kind': 'ollmo.semantic_review_lens_review',
                    'status': 'active',
                    'authority': 'advisory_read_model_only',
                    'lens_count': 1,
                },
                'semantic_review_lenses': [
                    {
                        'kind': 'ollmo.semantic_review_lens',
                        'lens_id': 'semantic-lens-final-review',
                        'lens': 'integrator',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'authority': 'advisory_read_model_only',
                        'success_definition': 'Final review must integrate dependency evidence before it can pass.',
                        'failure_modes': ['dependency_evidence_ignored'],
                        'evidence_requirements': ['phase-1 dependency evidence'],
                    }
                ],
                'recursive_cycle_review': {
                    'kind': 'ollmo.recursive_cycle_review',
                    'status': 'active',
                    'task_count': 1,
                },
                'recursive_cycle_tasks': [
                    {
                        'kind': 'ollmo.recursive_cycle_task',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'status': 'pending',
                        'cycle_policy': 'prepare_gather_execute_verify_repair_or_freeze',
                    }
                ],
                'aspiration_review': {
                    'kind': 'ollmo.aspiration_review',
                    'status': 'active',
                    'authority': 'advisory_read_model_only',
                    'frame_count': 1,
                },
                'aspiration_frames': [
                    {
                        'kind': 'ollmo.aspiration_frame',
                        'frame_id': 'aspiration-final-review',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'obligation_id': 'obligation-phase-2',
                        'source_kind': 'aspiration_frame',
                        'aspiration_action': 'raise_solution_bar',
                        'allowed_actions': ['raise_solution_bar', 'preserve_possibility_space'],
                        'reason': 'final review should not collapse to dependency-free prose',
                        'authority': 'advisory_read_model_only',
                    }
                ],
                'commitment_review': {
                    'kind': 'ollmo.commitment_review',
                    'status': 'active',
                    'authority': 'advisory_read_model_only',
                    'frame_count': 1,
                },
                'commitment_frames': [
                    {
                        'kind': 'ollmo.commitment_frame',
                        'frame_id': 'commitment-final-review',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'obligation_id': 'obligation-phase-2',
                        'source_kind': 'commitment_frame',
                        'commitment_action': 'commit_to_right_sized_sufficient_transition',
                        'recommended_transition': 'repair_dependency_chain',
                        'allowed_transitions': ['repair_dependency_chain', 'semantic_review', 'truthful_freeze_after_review'],
                        'reason': 'repair dependency evidence instead of waiting indefinitely',
                        'authority': 'advisory_read_model_only',
                    }
                ],
                'semantic_decision_review': {
                    'kind': 'ollmo.semantic_decision_review',
                    'status': 'active',
                    'authority': 'advisory_read_model_only',
                    'proposal_count': 1,
                },
                'semantic_decision_proposals': [
                    {
                        'kind': 'ollmo.semantic_decision_proposal',
                        'proposal_id': 'semantic-decision-final-review',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'obligation_id': 'obligation-phase-2',
                        'decision_action': 'repair_dependency_chain',
                        'confidence': 0.72,
                        'reason': 'semantic decision review says dependency evidence should be repaired',
                        'authority': 'advisory_read_model_only',
                    }
                ],
                'controlled_attention_review': {
                    'kind': 'ollmo.controlled_attention_review',
                    'status': 'active',
                    'authority': 'advisory_read_model_only',
                    'frame_count': 1,
                },
                'controlled_attention_frames': [
                    {
                        'kind': 'ollmo.controlled_attention_frame',
                        'frame_id': 'controlled-attention-final-review',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'task_id': 'task-phase-2',
                        'obligation_id': 'obligation-phase-2',
                        'scope': 'block_resolution',
                        'priority': 'high',
                        'attention_question': 'Should this final review repair dependency evidence before continuing?',
                        'allowed_transitions': ['repair_dependency_chain', 'semantic_review', 'truthful_freeze'],
                        'non_authority_boundary': 'attention_only_runtime_contracts_closure_decide_truth',
                    }
                ],
                'accepted_learning': {
                    'status': 'active',
                    'authority': 'soft_hint',
                    'runtime_effect': 'soft_hints_available',
                    'hint_count': 1,
                    'allowed_use': 'orientation_only_not_promotion_authority',
                },
            },
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1'],
                    'status': 'pending',
                },
            ],
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'chat', 'status': 'completed'},
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'depends_on': ['phase-1'],
                    'status': 'pending',
                },
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Prepared dependency text.',
            request_payload={'ghost_route': True, 'prompt': 'Prepare text, then write a final review.'},
            artifact_payload={
                'output_text': 'Prepared dependency text.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        final_check = next(item for item in review['checks'] if item.get('branch_id') == 'branch-final-review')
        feedback_item = next(
            item for item in review['ghost_repair_feedback']['items']
            if item.get('branch_id') == 'branch-final-review'
        )

        self.assertEqual(review['status'], 'pending')
        self.assertEqual(review['decision_contract_review']['status'], 'available')
        self.assertEqual(review['decision_contract_review']['promotion_suggestion_count'], 1)
        self.assertEqual(review['decision_contract_review']['waiver_candidate_count'], 1)
        self.assertEqual(review['decision_contract_review']['repair_candidate_count'], 1)
        self.assertEqual(review['decision_contract_review']['block_resolution_signal_count'], 1)
        self.assertEqual(review['decision_contract_review']['active_reconsideration_decision_count'], 1)
        self.assertEqual(review['decision_contract_review']['semantic_quality_contract_count'], 1)
        self.assertEqual(review['decision_contract_review']['semantic_review_lens_count'], 1)
        self.assertEqual(review['decision_contract_review']['recursive_cycle_task_count'], 1)
        self.assertEqual(review['decision_contract_review']['aspiration_frame_count'], 1)
        self.assertEqual(review['decision_contract_review']['commitment_frame_count'], 1)
        self.assertEqual(review['decision_contract_review']['semantic_decision_proposal_count'], 1)
        self.assertEqual(review['decision_contract_review']['controlled_attention_frame_count'], 1)
        self.assertEqual(final_check['repair_action'], 'repair_dependency_chain')
        self.assertEqual(
            final_check['block_resolution_action'],
            'continue_or_repair_promoted_work_before_freeze',
        )
        self.assertEqual(
            final_check['reconsideration_reflex']['principle'],
            'the_solution_to_a_block_is_the_blocks_own_resolution',
        )
        self.assertEqual(final_check['evidence_requirements'], ['phase-1 dependency evidence'])
        self.assertEqual(final_check['active_reconsideration_review_type'], 'continuation_or_repair_review')
        self.assertEqual(
            final_check['active_reconsideration_action'],
            'continue_or_repair_promoted_work_before_freeze',
        )
        self.assertEqual(final_check['semantic_quality_review_id'], 'semantic-quality-final-review')
        self.assertEqual(final_check['semantic_quality_status'], 'pending_semantic_review')
        self.assertEqual(final_check['semantic_review_lens'], 'integrator')
        self.assertEqual(final_check['success_definition'], 'Final review must integrate dependency evidence before it can pass.')
        self.assertIn('dependency_evidence_ignored', final_check['failure_modes'])
        self.assertIn('phase-1 dependency evidence', final_check['semantic_lens_evidence_requirements'])
        self.assertEqual(final_check['recursive_cycle_state']['cycle_policy'], 'prepare_gather_execute_verify_repair_or_freeze')
        self.assertEqual(final_check['aspiration_frame_id'], 'aspiration-final-review')
        self.assertEqual(final_check['aspiration_action'], 'raise_solution_bar')
        self.assertEqual(final_check['commitment_frame_id'], 'commitment-final-review')
        self.assertEqual(final_check['commitment_recommended_transition'], 'repair_dependency_chain')
        self.assertEqual(final_check['semantic_decision_action'], 'repair_dependency_chain')
        self.assertEqual(final_check['semantic_decision_confidence'], 0.72)
        self.assertEqual(final_check['controlled_attention_frame_id'], 'controlled-attention-final-review')
        self.assertEqual(final_check['controlled_attention_scope'], 'block_resolution')
        self.assertEqual(final_check['controlled_attention_priority'], 'high')
        self.assertEqual(final_check['promotion_suggestions'][0]['candidate_id'], 'candidate-final-review')
        self.assertEqual(final_check['waiver_candidates'][0]['obligation_id'], 'obligation-phase-2')
        self.assertEqual(
            final_check['decision_contract_repair_candidates'][0]['reason'],
            'decision contract says final review needs dependency evidence',
        )
        self.assertEqual(feedback_item['repair_action'], 'repair_dependency_chain')
        self.assertEqual(
            feedback_item['block_resolution_action'],
            'continue_or_repair_promoted_work_before_freeze',
        )
        self.assertEqual(
            feedback_item['active_reconsideration_action'],
            'continue_or_repair_promoted_work_before_freeze',
        )
        self.assertEqual(feedback_item['semantic_quality_review_id'], 'semantic-quality-final-review')
        self.assertEqual(feedback_item['semantic_review_lens'], 'integrator')
        self.assertEqual(feedback_item['success_definition'], 'Final review must integrate dependency evidence before it can pass.')
        self.assertEqual(feedback_item['aspiration_frame_id'], 'aspiration-final-review')
        self.assertEqual(feedback_item['commitment_frame_id'], 'commitment-final-review')
        self.assertEqual(feedback_item['semantic_decision_action'], 'repair_dependency_chain')
        self.assertEqual(feedback_item['controlled_attention_frame_id'], 'controlled-attention-final-review')
        self.assertEqual(
            feedback_item['semantic_decision_reason'],
            'semantic decision review says dependency evidence should be repaired',
        )
        self.assertEqual(feedback_item['evidence_requirements'], ['phase-1 dependency evidence'])
        self.assertEqual(feedback_item['promotion_suggestions'][0]['candidate_id'], 'candidate-final-review')
        self.assertEqual(feedback_item['waiver_candidates'][0]['obligation_id'], 'obligation-phase-2')
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['reconsideration_candidates'][0]['candidate_id'],
            'candidate-extra-image',
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['block_resolution_reflex']['signal_count'],
            1,
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['active_reconsideration_review']['decision_count'],
            1,
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['semantic_decision_review']['proposal_count'],
            1,
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['aspiration_review']['frame_count'],
            1,
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['commitment_review']['frame_count'],
            1,
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['controlled_attention_review']['frame_count'],
            1,
        )
        self.assertEqual(
            review['surface_state']['category_counts']['open'],
            1,
        )
        self.assertEqual(
            review['surface_state']['category_counts']['repair_pending'],
            1,
        )
        self.assertEqual(
            review['surface_state']['category_counts']['semantic_review_pending'],
            1,
        )
        self.assertEqual(
            review['surface_state']['category_counts']['controlled_attention_advisory'],
            1,
        )
        self.assertEqual(
            review['surface_state']['category_counts']['aspiration_advisory'],
            1,
        )
        self.assertEqual(
            review['surface_state']['category_counts']['commitment_advisory'],
            1,
        )
        self.assertEqual(
            review['ghost_repair_feedback']['decision_contract_guidance']['accepted_learning']['allowed_use'],
            'orientation_only_not_promotion_authority',
        )

    def test_closure_review_keeps_decision_contract_supersession_candidate_advisory(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'decision_contract': {
                'kind': 'ollmo.ghost_decision_contract',
                'decision_contract_version': 1,
                'supersession_candidates': [
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-image-old',
                        'capability': 'image_generation',
                        'supersession_candidates': [
                            {
                                'obligation_id': 'obligation-phase-2',
                                'superseded_by_obligation_id': 'obligation-phase-3',
                                'supersession_reason': 'possible replacement exists but closure has not confirmed it',
                            }
                        ],
                    }
                ],
            },
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image-old',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                },
            ],
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'chat', 'status': 'completed'},
                {'phase_id': 'phase-2', 'branch_id': 'branch-image-old', 'capability': 'image_generation', 'status': 'pending'},
            ],
        }

        review = self.owner.build_graph_closure_review(
            'Image prompt prepared.',
            request_payload={'ghost_route': True, 'prompt': 'Generate an image.'},
            artifact_payload={
                'output_text': 'Image prompt prepared.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        image_check = next(item for item in review['checks'] if item.get('branch_id') == 'branch-image-old')
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(image_check['status'], 'pending')
        self.assertNotEqual(image_check.get('evidence'), 'explicit_obligation_superseded')
        self.assertTrue(image_check['supersession_review_required'])
        self.assertEqual(image_check['supersession_review_authority'], 'closure_review_required')
        self.assertEqual(
            image_check['decision_contract_supersession_candidates'][0]['superseded_by_obligation_id'],
            'obligation-phase-3',
        )
        self.assertEqual(
            review['ghost_repair_feedback']['items'][0]['decision_contract_supersession_candidates'][0]['obligation_id'],
            'obligation-phase-2',
        )

    def test_closure_review_uses_workload_review_criteria_for_dependency_repair(self):
        phase_graph = {
            'current_phase_id': 'phase-3',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'text_to_speech', 'output_type': 'audio', 'status': 'completed'},
                {'phase_id': 'phase-2', 'capability': 'vision_analysis', 'output_type': 'text', 'status': 'completed'},
                {
                    'phase_id': 'phase-3',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1', 'phase-2'],
                    'status': 'completed',
                },
            ],
            'workload_graph': {
                'tasks': [
                    {'task_id': 'task-phase-1', 'phase_id': 'phase-1', 'branch_id': 'phase-1'},
                    {'task_id': 'task-phase-2', 'phase_id': 'phase-2', 'branch_id': 'phase-2'},
                    {
                        'task_id': 'task-phase-3',
                        'phase_id': 'phase-3',
                        'branch_id': 'branch-final-review',
                        'capability': 'chat',
                        'depends_on': ['phase-1', 'phase-2'],
                        'review_criteria': ['uses_dependency_evidence', 'does_not_restart_root_request'],
                        'input_refs': [
                            {'kind': 'phase_output', 'phase_id': 'phase-1', 'role': 'audio'},
                            {'kind': 'phase_output', 'phase_id': 'phase-2', 'role': 'vision evidence'},
                        ],
                    },
                ],
            },
        }

        review = self.owner.build_graph_closure_review(
            'Here is a final comparison, but it was not bound to branch evidence.',
            request_payload={'ghost_route': True, 'prompt': 'Use the generated audio and image analysis.'},
            artifact_payload={
                'output_text': 'Here is a final comparison, but it was not bound to branch evidence.',
                'tts_audio_integrity_evidence': self._tts_integrity_evidence(
                    source_text='Bound audio dependency fixture.',
                ),
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        final_check = next(item for item in review['checks'] if item.get('branch_id') == 'branch-final-review')
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(final_check['status'], 'pending')
        self.assertEqual(final_check['evidence'], 'review_criteria_unverified')
        self.assertEqual(final_check['review_criteria_status'], 'repair_required')
        self.assertEqual(final_check['repair_action'], 'repair_dependency_chain')
        self.assertEqual(final_check['input_refs'][0]['phase_id'], 'phase-1')
        feedback_item = next(
            item for item in review['ghost_repair_feedback']['items']
            if item.get('branch_id') == 'branch-final-review'
        )
        self.assertEqual(feedback_item['repair_action'], 'repair_dependency_chain')
        self.assertEqual(feedback_item['review_criteria'][0], 'uses_dependency_evidence')
        self.assertEqual(
            review['ghost_repair_feedback']['repair_loop']['next_actions'],
            ['repair_dependency_chain'],
        )

    def test_closure_review_promotes_global_semantic_review_for_whole_intent_fit(self):
        phase_graph = {
            'current_phase_id': 'phase-2',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'image_generation', 'output_type': 'image', 'status': 'completed'},
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1'],
                    'status': 'completed',
                },
            ],
            'workload_graph': {
                'tasks': [
                    {'task_id': 'task-phase-1', 'phase_id': 'phase-1', 'branch_id': 'phase-1'},
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'capability': 'chat',
                        'depends_on': ['phase-1'],
                        'review_criteria': [
                            'final comparison is concise and visually grounded',
                        ],
                        'semantic_review_lens': 'integrator',
                        'success_definition': 'Final comparison must integrate visual evidence.',
                        'content_payload_source': 'late_fill_results:image_generation',
                    },
                ],
            },
        }

        review = self.owner.build_graph_closure_review(
            'The generated visual is ready for a compact deployment plan.',
            request_payload={'ghost_route': True, 'prompt': 'Write the final deployment plan from available evidence.'},
            artifact_payload={
                'output_text': 'The generated visual is ready for a compact deployment plan.',
                'runtime': {'request_phase_graph': phase_graph},
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        {'branch_id': 'phase-1', 'phase_id': 'phase-1', 'capability': 'image_generation'},
                    ],
                },
            },
        )

        final_check = next(item for item in review['checks'] if item.get('branch_id') == 'branch-final-review')
        branch_review_check = next(item for item in review['checks'] if item.get('check_kind') == 'branch_semantic_review')
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(review['global_semantic_closure_review']['status'], 'waiting_on_local_closure')
        self.assertEqual(review['global_semantic_closure_review']['proposal_count'], 0)
        self.assertEqual(review['semantic_review_status'], 'required_advisory')
        self.assertEqual(review['semantic_review_required_count'], 2)
        self.assertEqual(final_check['review_criteria_status'], 'semantic_review_required')
        self.assertTrue(final_check['semantic_review_required'])
        self.assertEqual(final_check['semantic_review_action'], 'semantic_review')
        self.assertEqual(
            final_check['semantic_review_criteria'],
            ['final comparison is concise and visually grounded'],
        )
        self.assertEqual(final_check['branch_semantic_review_status'], 'pending')
        self.assertEqual(final_check['semantic_review_lens'], 'integrator')
        self.assertEqual(final_check['success_definition'], 'Final comparison must integrate visual evidence.')
        self.assertEqual(branch_review_check['status'], 'pending')
        self.assertEqual(branch_review_check['repair_action'], 'semantic_review')
        self.assertEqual(branch_review_check['stage_direction'], 'run_branch_semantic_review')
        self.assertEqual(branch_review_check['branch_semantic_review_source_branch_id'], 'branch-final-review')
        self.assertEqual(branch_review_check['semantic_review_lens'], 'integrator')
        self.assertEqual(branch_review_check['success_definition'], 'Final comparison must integrate visual evidence.')
        self.assertIn('semantic_review_lens', branch_review_check['content_payload'])
        self.assertIn('success_definition', branch_review_check['content_payload'])
        feedback_item = next(
            item for item in review['ghost_repair_feedback']['items']
            if item.get('check_kind') == 'branch_semantic_review'
        )
        self.assertEqual(feedback_item['repair_action'], 'semantic_review')
        self.assertEqual(
            review['ghost_repair_feedback']['repair_loop']['next_actions'],
            ['semantic_review'],
        )
        self.assertTrue(review['ghost_repair_feedback']['repair_loop']['auto_execute'])

    def test_global_semantic_review_completion_allows_truthful_freeze(self):
        phase_graph = {
            'current_phase_id': 'phase-2',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'image_generation', 'output_type': 'image', 'status': 'completed'},
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1'],
                    'status': 'completed',
                },
            ],
            'workload_graph': {
                'tasks': [
                    {'task_id': 'task-phase-1', 'phase_id': 'phase-1', 'branch_id': 'phase-1'},
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'capability': 'chat',
                        'depends_on': ['phase-1'],
                        'review_criteria': ['final comparison is concise and visually grounded'],
                    },
                ],
            },
        }

        review = self.owner.build_graph_closure_review(
            'The generated visual is ready for a compact deployment plan.',
            request_payload={'ghost_route': True, 'prompt': 'Write the final deployment plan from available evidence.'},
            artifact_payload={
                'output_text': 'The generated visual is ready for a compact deployment plan.',
                'runtime': {'request_phase_graph': phase_graph},
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        {'branch_id': 'phase-1', 'phase_id': 'phase-1', 'capability': 'image_generation'},
                        {
                            'branch_id': 'branch-semantic-review-branch-final-review',
                            'phase_id': 'phase-semantic-review-branch-final-review',
                            'capability': 'chat',
                            'result_text': (
                                '{'
                                '"kind":"ollmo.semantic_review_verdict",'
                                '"verdict":"passed",'
                                '"overall_status":"fulfilled",'
                                '"whole_intent_fit":"The final branch uses the generated visual evidence.",'
                                '"criterion_results":[{"criterion":"final comparison is concise and visually grounded","status":"passed","evidence_refs":["phase-1","branch-final-review"]}],'
                                '"evidence_refs":["phase-1","branch-final-review"],'
                                '"defects":[],'
                                '"confidence":0.9,'
                                '"recommended_transition":"truthful_freeze"'
                                '}'
                            ),
                        },
                        {
                            'branch_id': 'branch-global-semantic-closure-review',
                            'phase_id': 'phase-global-semantic-closure-review',
                            'capability': 'chat',
                            'result_text': (
                                '{'
                                '"kind":"ollmo.semantic_review_verdict",'
                                '"verdict":"passed",'
                                '"overall_status":"fulfilled",'
                                '"whole_intent_fit":"The final deployment plan is grounded in the generated visual evidence.",'
                                '"criterion_results":[{"criterion":"final comparison is concise and visually grounded","status":"passed","evidence_refs":["phase-1","branch-final-review"]}],'
                                '"evidence_refs":["phase-1","branch-final-review"],'
                                '"defects":[],'
                                '"confidence":0.9,'
                                '"recommended_transition":"truthful_freeze"'
                                '}'
                            ),
                        },
                    ],
                },
            },
        )

        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(review['global_semantic_closure_review']['status'], 'fulfilled')
        self.assertEqual(
            review['global_semantic_closure_review']['semantic_review_verdict']['verdict'],
            'passed',
        )
        self.assertNotIn('ghost_repair_feedback', review)

    def test_global_semantic_review_failed_verdict_blocks_freeze(self):
        phase_graph = {
            'current_phase_id': 'phase-2',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'image_generation', 'output_type': 'image', 'status': 'completed'},
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1'],
                    'status': 'completed',
                },
            ],
            'workload_graph': {
                'tasks': [
                    {'task_id': 'task-phase-1', 'phase_id': 'phase-1', 'branch_id': 'phase-1'},
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'capability': 'chat',
                        'depends_on': ['phase-1'],
                        'review_criteria': ['final comparison is concise and visually grounded'],
                    },
                ],
            },
        }

        review = self.owner.build_graph_closure_review(
            'The generated visual is ready for a compact deployment plan.',
            request_payload={'ghost_route': True, 'prompt': 'Write the final deployment plan from available evidence.'},
            artifact_payload={
                'output_text': 'The generated visual is ready for a compact deployment plan.',
                'runtime': {'request_phase_graph': phase_graph},
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        {'branch_id': 'phase-1', 'phase_id': 'phase-1', 'capability': 'image_generation'},
                        {
                            'branch_id': 'branch-semantic-review-branch-final-review',
                            'phase_id': 'phase-semantic-review-branch-final-review',
                            'capability': 'chat',
                            'result_text': (
                                '{'
                                '"kind":"ollmo.semantic_review_verdict",'
                                '"verdict":"passed",'
                                '"overall_status":"fulfilled",'
                                '"whole_intent_fit":"The final branch uses the generated visual evidence.",'
                                '"criterion_results":[{"criterion":"final comparison is concise and visually grounded","status":"passed","evidence_refs":["phase-1","branch-final-review"]}],'
                                '"evidence_refs":["phase-1","branch-final-review"],'
                                '"defects":[],'
                                '"confidence":0.9,'
                                '"recommended_transition":"truthful_freeze"'
                                '}'
                            ),
                        },
                        {
                            'branch_id': 'branch-global-semantic-closure-review',
                            'phase_id': 'phase-global-semantic-closure-review',
                            'capability': 'chat',
                            'result_text': (
                                '{'
                                '"kind":"ollmo.semantic_review_verdict",'
                                '"verdict":"failed",'
                                '"overall_status":"blocked",'
                                '"whole_intent_fit":"The final text does not use the generated image evidence.",'
                                '"criterion_results":[{"criterion":"final comparison is concise and visually grounded","status":"failed","evidence_refs":["branch-final-review"]}],'
                                '"evidence_refs":["branch-final-review"],'
                                '"defects":["generated image evidence was not used"],'
                                '"confidence":0.86,'
                                '"recommended_transition":"repair_dependency_chain"'
                                '}'
                            ),
                        },
                    ],
                },
            },
        )

        global_check = next(item for item in review['checks'] if item.get('check_kind') == 'global_semantic_closure')
        self.assertEqual(review['status'], 'blocked')
        self.assertEqual(review['global_semantic_closure_review']['status'], 'blocked')
        self.assertEqual(
            review['global_semantic_closure_review']['semantic_review_verdict']['verdict'],
            'failed',
        )
        self.assertEqual(global_check['repair_action'], 'repair_dependency_chain')
        self.assertEqual(global_check['semantic_review_verdict_status'], 'blocked')
        self.assertIn('ghost_repair_feedback', review)

    def test_global_semantic_review_unparseable_output_keeps_manual_review_open(self):
        phase_graph = {
            'current_phase_id': 'phase-2',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'phases': [
                {'phase_id': 'phase-1', 'capability': 'image_generation', 'output_type': 'image', 'status': 'completed'},
                {
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-final-review',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-1'],
                    'status': 'completed',
                },
            ],
            'workload_graph': {
                'tasks': [
                    {'task_id': 'task-phase-1', 'phase_id': 'phase-1', 'branch_id': 'phase-1'},
                    {
                        'task_id': 'task-phase-2',
                        'phase_id': 'phase-2',
                        'branch_id': 'branch-final-review',
                        'capability': 'chat',
                        'depends_on': ['phase-1'],
                        'review_criteria': ['final comparison is concise and visually grounded'],
                    },
                ],
            },
        }

        review = self.owner.build_graph_closure_review(
            'The generated visual is ready for a compact deployment plan.',
            request_payload={'ghost_route': True, 'prompt': 'Write the final deployment plan from available evidence.'},
            artifact_payload={
                'output_text': 'The generated visual is ready for a compact deployment plan.',
                'runtime': {'request_phase_graph': phase_graph},
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        {'branch_id': 'phase-1', 'phase_id': 'phase-1', 'capability': 'image_generation'},
                        {
                            'branch_id': 'branch-semantic-review-branch-final-review',
                            'phase_id': 'phase-semantic-review-branch-final-review',
                            'capability': 'chat',
                            'result_text': (
                                '{'
                                '"kind":"ollmo.semantic_review_verdict",'
                                '"verdict":"passed",'
                                '"overall_status":"fulfilled",'
                                '"whole_intent_fit":"The final branch uses the generated visual evidence.",'
                                '"criterion_results":[{"criterion":"final comparison is concise and visually grounded","status":"passed","evidence_refs":["phase-1","branch-final-review"]}],'
                                '"evidence_refs":["phase-1","branch-final-review"],'
                                '"defects":[],'
                                '"confidence":0.9,'
                                '"recommended_transition":"truthful_freeze"'
                                '}'
                            ),
                        },
                        {
                            'branch_id': 'branch-global-semantic-closure-review',
                            'phase_id': 'phase-global-semantic-closure-review',
                            'capability': 'chat',
                            'result_text': 'The review looks okay overall.',
                        },
                    ],
                },
            },
        )

        global_check = next(item for item in review['checks'] if item.get('check_kind') == 'global_semantic_closure')
        self.assertEqual(review['status'], 'pending')
        self.assertEqual(review['global_semantic_closure_review']['status'], 'pending')
        self.assertEqual(
            review['global_semantic_closure_review']['semantic_review_verdict']['parse_status'],
            'missing_structured_verdict',
        )
        self.assertEqual(global_check['repair_action'], 'manual_review')
        self.assertFalse(review['ghost_repair_feedback']['repair_loop']['auto_execute'])

    def test_closure_review_treats_superseded_obligation_as_closed(self):
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'output_obligations': [
                {
                    'obligation_id': 'obligation-phase-1',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                    'role': 'preparation_output',
                },
                {
                    'obligation_id': 'obligation-phase-2',
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image-old',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'superseded',
                    'role': 'final_output',
                    'superseded_by_obligation_id': 'obligation-phase-3',
                    'supersession_reason': 'newer branch replaced the image obligation',
                    'review_criteria': ['image should fit the alpine rescue tone'],
                },
            ],
            'phases': [
                {'phase_id': 'phase-1', 'branch_id': 'phase-1', 'capability': 'chat', 'output_type': 'text', 'status': 'completed'},
                {'phase_id': 'phase-2', 'branch_id': 'branch-image-old', 'capability': 'image_generation', 'output_type': 'image', 'status': 'superseded'},
            ],
        }

        review = self.owner.build_graph_closure_review(
            'The old image branch has been replaced by the newer branch.',
            request_payload={'ghost_route': True, 'prompt': 'Review the existing branch plan.'},
            artifact_payload={
                'output_text': 'The old image branch has been replaced by the newer branch.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        checks = {item['obligation_id']: item for item in review['checks'] if item.get('obligation_id')}
        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(review['counts']['superseded'], 1)
        self.assertEqual(checks['obligation-phase-2']['status'], 'superseded')
        self.assertEqual(checks['obligation-phase-2']['evidence'], 'explicit_obligation_superseded')
        self.assertEqual(checks['obligation-phase-2']['review_criteria_status'], 'not_required')
        self.assertEqual(checks['obligation-phase-2']['superseded_by_obligation_id'], 'obligation-phase-3')
        self.assertNotIn('ghost_repair_feedback', review)

    def test_intent_adequacy_does_not_promote_reserved_image_candidate(self):
        prompt = (
            'Skizziere eine mögliche Poster-Idee, generiere aber noch kein Bild, '
            'und lies die Idee als Audio vor.'
        )
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=phase_graph,
        )

        self.assertEqual(review['expected_output_counts'], {'audio': 1})
        self.assertNotIn('image', review['expected_output_counts'])
        image_checks = [
            item for item in review['checks']
            if item.get('capability') == 'image_generation'
        ]
        self.assertEqual(image_checks, [])

    def test_intent_adequacy_current_preservation_overrides_stale_visual_graph_cues(self):
        prompt = (
            'Beziehe dich ausdrücklich auf das Observatorium-Bild, seine Bildanalyse, '
            'die deutsche Erzählung und das Audio aus dem unmittelbar vorherigen Turn. '
            'Bewahre Bild und Bildanalyse unverändert; erzeuge das Bild nicht neu und '
            'analysiere es nicht erneut. Ersetze den bisherigen einzelnen Audiozweig '
            'durch zwei getrennte Audiofassungen: einmal die ursprüngliche deutsche '
            'Erzählung und einmal eine getreue englische Übersetzung. Transkribiere '
            'beide tatsächlich erzeugten Audios separat und gib ein neues JSON-Objekt '
            'aus, das die unveränderte Bildevidenz sowie beide Audio-artifact_refs und '
            'beide realen Transkripte eindeutig verbindet.'
        )
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'prompt': prompt,
            # Simulate positive visual intent carried by an older graph snapshot.
            'prompt_intent': {
                'primary_capability': 'image_generation',
                'requests_visual_output': True,
                'requested_visual_output_count': 1,
                'counted_visual_output_obligation': True,
                'has_visual_follow_up_request': True,
                'text_preparation_before_visual_output': True,
                'visual_artifact_execution_suppressed_by_preservation': False,
                'visual_analysis_execution_suppressed_by_preservation': False,
                'requests_audio_output': True,
                'requested_audio_output_count': 2,
                'requests_speech_to_text_output': True,
                'downstream_follow_up_capabilities': [
                    'image_generation',
                    'vision_analysis',
                    'text_to_speech',
                    'speech_to_text',
                ],
            },
            'intent_obligations': [
                {
                    'obligation_id': 'stale-image-output',
                    'kind': 'media_artifact',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'required': True,
                },
                {
                    'obligation_id': 'stale-image-analysis',
                    'kind': 'evidence_branch',
                    'capability': 'vision_analysis',
                    'output_type': 'text',
                    'required': True,
                },
                *[
                    {
                        'obligation_id': f'audio-output-{index}',
                        'kind': 'media_artifact',
                        'phase_id': f'phase-tts-{index}',
                        'capability': 'text_to_speech',
                        'output_type': 'audio',
                        'required': True,
                    }
                    for index in (1, 2)
                ],
                *[
                    {
                        'obligation_id': f'transcript-output-{index}',
                        'kind': 'evidence_branch',
                        'phase_id': f'phase-stt-{index}',
                        'capability': 'speech_to_text',
                        'output_type': 'text',
                        'required': True,
                    }
                    for index in (1, 2)
                ],
                {
                    'obligation_id': 'final-json-join',
                    'kind': 'evidence_branch',
                    'phase_id': 'phase-final',
                    'capability': 'chat',
                    'output_type': 'text',
                    'required': True,
                },
            ],
            'output_obligations': [
                *[
                    {
                        'obligation_id': f'output-audio-{index}',
                        'phase_id': f'phase-tts-{index}',
                        'branch_id': f'branch-tts-{index}',
                        'capability': 'text_to_speech',
                        'output_type': 'audio',
                        'required': True,
                    }
                    for index in (1, 2)
                ],
                *[
                    {
                        'obligation_id': f'output-transcript-{index}',
                        'phase_id': f'phase-stt-{index}',
                        'branch_id': f'branch-stt-{index}',
                        'capability': 'speech_to_text',
                        'output_type': 'text',
                        'required': True,
                    }
                    for index in (1, 2)
                ],
                {
                    'obligation_id': 'output-final-json',
                    'phase_id': 'phase-final',
                    'branch_id': 'branch-final',
                    'capability': 'chat',
                    'output_type': 'text',
                    'required': True,
                },
            ],
        }

        merged_intent = self.owner._merged_prompt_intent_for_review(prompt, phase_graph)

        self.assertTrue(merged_intent['visual_artifact_execution_suppressed_by_preservation'])
        self.assertTrue(merged_intent['visual_analysis_execution_suppressed_by_preservation'])
        self.assertFalse(merged_intent['requests_visual_output'])
        self.assertEqual(merged_intent['requested_visual_output_count'], 0)
        self.assertNotIn('primary_capability', merged_intent)
        self.assertNotIn('image', merged_intent['required_intent_output_counts'])
        self.assertNotIn(
            'image_generation',
            merged_intent.get('downstream_follow_up_capabilities') or [],
        )
        self.assertNotIn(
            'vision_analysis',
            merged_intent.get('downstream_follow_up_capabilities') or [],
        )

        for reference_artifacts in (
            [
                {
                    'artifact_ref': 'artifact:image_predecessor',
                    'type': 'image',
                    'path': '/tmp/predecessor-observatory.png',
                    'source_response_id': 'resp_predecessor',
                }
            ],
            [],
        ):
            with self.subTest(has_preserved_reference=bool(reference_artifacts)):
                review = self.owner.build_intent_graph_adequacy_review(
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'reference_artifacts': reference_artifacts,
                    },
                    request_phase_graph=phase_graph,
                )

                self.assertEqual(review['status'], 'fulfilled')
                self.assertEqual(review['expected_output_counts'], {'audio': 2})
                self.assertEqual(
                    review['expected_capability_counts'],
                    {
                        'chat': 1,
                        'speech_to_text': 2,
                        'text_to_speech': 2,
                    },
                )
                self.assertFalse(
                    any(
                        item.get('capability') in {'image_generation', 'vision_analysis'}
                        for item in review['checks']
                    )
                )
                self.assertFalse(
                    any(
                        item.get('obligation_id') == 'missing-obligation-image'
                        for item in review['checks']
                    )
                )

    def test_dependency_join_adequacy_projects_required_ledger_counts(self):
        prompt = (
            'Schreibe einen deutschen Szenentext mit genau zwanzig Wörtern über einen Leuchtturm im Sturm. '
            'Erzeuge daraus parallel ein Bild und ein Audio. Analysiere danach das tatsächlich erzeugte Bild '
            'und transkribiere das tatsächlich erzeugte Audio. Vergleiche abschließend im Chat anhand genau '
            'dieser beiden realen Evidenzzweigen, ob Leuchtturm, Sturm und Nacht in beiden vorkommen. '
            'Bildanalyse darf nur vom Bild, Transkription nur vom Audio, der Schluss nur von beiden '
            'Evidenzzweigen abhängen.'
        )
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        phase_graph['prompt_intent'] = dict(phase_graph['prompt_intent'])
        phase_graph['prompt_intent'].update(
            {
                'requests_visual_output': False,
                'requested_visual_output_count': 0,
                'text_preparation_before_visual_output': False,
                'text_preparation_before_audio_output': False,
            }
        )
        for key in (
            'downstream_follow_up_capabilities',
            'required_intent_obligation_count',
            'required_intent_obligation_kinds',
            'required_intent_capabilities',
            'required_intent_capability_counts',
            'required_intent_output_counts',
        ):
            phase_graph['prompt_intent'].pop(key, None)

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=phase_graph,
        )

        self.assertEqual(review['status'], 'fulfilled')
        self.assertEqual(review['expected_output_counts'], {'image': 1, 'audio': 1})
        self.assertEqual(
            review['expected_capability_counts'],
            {
                'image_generation': 1,
                'text_to_speech': 1,
                'vision_analysis': 1,
                'speech_to_text': 1,
                'chat': 1,
            },
        )
        self.assertEqual(
            review['graph_capability_counts'],
            {
                'image_generation': 1,
                'text_to_speech': 1,
                'vision_analysis': 1,
                'speech_to_text': 1,
                'chat': 1,
            },
        )

        terminal_phase_id = next(
            item.get('phase_id')
            for item in phase_graph['intent_obligations']
            if item.get('kind') == 'evidence_branch' and item.get('capability') == 'chat'
        )
        broken_graph = json.loads(json.dumps(phase_graph))
        for owner in (broken_graph, broken_graph.get('request_ir') or {}):
            for key in ('phases', 'downstream_branches', 'output_obligations'):
                if not isinstance(owner.get(key), list):
                    continue
                owner[key] = [
                    item
                    for item in owner[key]
                    if item.get('phase_id') != terminal_phase_id
                ]
        broken_review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=broken_graph,
        )
        missing_chat = next(
            item
            for item in broken_review['checks']
            if item.get('evidence') == 'intent_graph_adequacy_missing_capability_obligation'
            and item.get('capability') == 'chat'
        )
        self.assertEqual(broken_review['status'], 'pending')
        self.assertEqual(broken_review['graph_capability_counts'].get('chat', 0), 0)
        self.assertEqual(missing_chat['missing_count'], 1)

    def test_intent_adequacy_does_not_project_reserved_or_deferred_ledger_entries(self):
        prompt = 'Skizziere eine Poster-Idee, aber generiere das Bild noch nicht.'
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'prompt': prompt,
            'prompt_intent': {
                'explicit_visual_defer_materialization': True,
                'requests_visual_output': False,
                'requested_visual_output_count': 0,
            },
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'required': True,
                },
            ],
            'output_obligations': [],
            'intent_obligations': [
                {
                    'obligation_id': 'intent-image-reserved',
                    'kind': 'media_artifact',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'count': 1,
                    'required': True,
                    'promotion_policy': 'reserved_only',
                },
                {
                    'obligation_id': 'intent-vision-deferred',
                    'kind': 'evidence_branch',
                    'capability': 'vision_analysis',
                    'output_type': 'text',
                    'count': 1,
                    'required': True,
                    'contract_state': 'deferred',
                    'promotion_policy': 'review_before_promotion',
                },
            ],
        }

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=phase_graph,
        )

        self.assertEqual(review['expected_output_counts'], {})
        self.assertEqual(review['expected_capability_counts'], {})
        self.assertNotIn('image_generation', review['graph_capability_counts'])
        self.assertFalse(
            any(
                item.get('capability') in {'image_generation', 'vision_analysis'}
                for item in review['checks']
            )
        )

    def test_intent_adequacy_keeps_exact_count_when_legacy_graph_only_has_reserved_images(self):
        prompt = (
            'Erstelle genau drei Bilder und zwei Dateien: index.html und styles.css. '
            'Keine Platzhalter, erfundenen Dateipfade oder externen Bilder.'
        )
        phase_graph = {
            'current_phase_id': 'phase-1',
            'current_phase_capability': 'chat',
            'mode': 'phase_chain',
            'prompt_intent': {
                'requests_visual_output': True,
                'counted_visual_output_obligation': True,
                'requested_visual_output_count': 3,
                'explicit_visual_defer_materialization': False,
                'downstream_follow_up_capabilities': ['image_generation'],
            },
            'candidate_graph': {
                'candidates': [
                    {
                        'candidate_id': 'candidate-image_generation-reserved-1',
                        'branch_id': 'branch-image_generation-reserved-1',
                        'phase_id': 'phase-image_generation-reserved-1',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'status': 'reserved',
                    },
                    {
                        'candidate_id': 'candidate-workload-task-phase-image-generation-reserved-1',
                        'branch_id': 'branch-image_generation-reserved-1',
                        'phase_id': 'phase-image_generation-reserved-1',
                        'capability': 'image_generation',
                        'status': 'reserved',
                    },
                ],
            },
            'output_candidates': [
                {
                    'candidate_id': 'candidate-image_generation-reserved-1',
                    'branch_id': 'branch-image_generation-reserved-1',
                    'phase_id': 'phase-image_generation-reserved-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'reserved',
                    'required': False,
                },
            ],
            'downstream_branches': [
                {
                    'branch_id': 'branch-image_generation-reserved-1',
                    'phase_id': 'phase-image_generation-reserved-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'contract_state': 'reserved',
                    'required': False,
                },
            ],
            'phases': [
                {
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                },
                {
                    'phase_id': 'phase-image_generation-reserved-1',
                    'branch_id': 'branch-image_generation-reserved-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'contract_state': 'reserved',
                    'required': False,
                },
            ],
            'output_obligations': [],
        }

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=phase_graph,
        )

        image_check = next(
            item for item in review['checks']
            if item.get('check_kind') == 'intent_graph_adequacy'
            and item.get('output_type') == 'image'
        )
        self.assertEqual(review['expected_output_counts']['image'], 3)
        self.assertEqual(image_check['expected_count'], 3)
        self.assertEqual(image_check['actual_count'], 0)
        self.assertEqual(image_check['missing_count'], 3)

    def test_intent_adequacy_reports_missing_ledger_dependency_edge_as_repairable(self):
        prompt = (
            'Create a small two-page website with index.html, suiten.html, shared styles.css, '
            'navigation between both pages, and exactly two generated local images linked from the pages.'
        )
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        broken_graph = dict(phase_graph)
        broken_graph['phases'] = [dict(item) for item in phase_graph.get('phases', [])]
        broken_graph['downstream_branches'] = [
            dict(item) for item in phase_graph.get('downstream_branches', [])
        ]
        for collection in (broken_graph['phases'], broken_graph['downstream_branches']):
            for record in collection:
                if (
                    record.get('role') == 'text_artifact_output'
                    and record.get('text_artifact_extension') == 'html'
                ):
                    record['depends_on'] = ['phase-1']
                    record.pop('dependency_contract', None)

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=broken_graph,
        )

        dependency_checks = [
            item for item in review['checks']
            if item.get('evidence') == 'intent_graph_adequacy_missing_dependency_edge'
        ]
        self.assertEqual(review['status'], 'pending')
        self.assertTrue(dependency_checks)
        self.assertTrue(
            all(item.get('repair_action') == 'rebind_artifact_dependency' for item in dependency_checks)
        )
        self.assertTrue(
            all(item.get('add_dependencies') for item in dependency_checks)
        )
        commitment_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'intent_commitment_review'
        ]
        self.assertTrue(commitment_checks)
        self.assertTrue(
            all(item.get('repair_action') == 'rebind_artifact_dependency' for item in commitment_checks)
        )
        self.assertEqual(
            review['intent_lens_review']['commitment_review']['status'],
            'pending',
        )

    def test_intent_lens_review_surfaces_attention_and_aspiration_for_fit_promises(self):
        prompt = (
            'Create a polished landing page for a boutique lake hotel. '
            'Create local HTML, CSS, and image artifacts. '
            'Generate four distinct local images, and make the text in each section exactly match its image. '
            'The HTML, CSS, and images must fit together as one coherent local artifact set.'
        )
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=phase_graph,
        )

        self.assertEqual(review['status'], 'fulfilled')
        lens = review['intent_lens_review']
        self.assertEqual(lens['attention_review']['status'], 'fulfilled')
        self.assertEqual(lens['commitment_review']['status'], 'fulfilled')
        self.assertEqual(lens['aspiration_review']['status'], 'fulfilled')
        self.assertTrue(lens['semantic_fit_requested'])
        aspiration_checks = [
            item for item in review['checks']
            if item.get('check_kind') == 'intent_aspiration_review'
        ]
        self.assertEqual(len(aspiration_checks), 1)
        self.assertEqual(aspiration_checks[0]['status'], 'fulfilled')
        self.assertTrue(aspiration_checks[0]['semantic_review_required'])
        self.assertIn(
            'whole_current_intent_fit_between_text_media_and_artifacts',
            aspiration_checks[0]['semantic_review_criteria'],
        )

    def test_intent_lens_promotes_review_for_explicit_image_mapping_and_tone_constraints(self):
        prompt = (
            'Erstelle eine einseitige Landing Page mit genau vier Bilder, index.html und styles.css. '
            'Die Gestaltung soll brutalistisch, direkt und menschlich sein, nicht dystopisch. '
            'Jede Bild-Section muss in Überschrift und Text konkret auf ihr Bild eingehen. '
            'Alle Bilder und Dateien müssen als lokales Artefakt-Set zusammenpassen.'
        )
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        review = self.owner.build_intent_graph_adequacy_review(
            request_payload={'ghost_route': True, 'prompt': prompt},
            request_phase_graph=phase_graph,
        )

        lens = review['intent_lens_review']
        self.assertTrue(lens['semantic_fit_requested'])
        self.assertEqual(lens['aspiration_review']['status'], 'fulfilled')
        aspiration_check = next(
            item
            for item in review['checks']
            if item.get('check_kind') == 'intent_aspiration_review'
        )
        self.assertTrue(aspiration_check['semantic_review_required'])
        self.assertIn(
            'explicit_visual_and_tone_constraints_match_prompt',
            aspiration_check['semantic_review_criteria'],
        )

    def test_closure_repair_feedback_excludes_reserved_image_candidate(self):
        prompt = (
            'Skizziere eine mögliche Poster-Idee, generiere aber noch kein Bild, '
            'und lies die Idee als Audio vor.'
        )
        phase_graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        review = self.owner.build_graph_closure_review(
            'Eine Poster-Idee für eine nachhaltige Zukunft.',
            request_payload={'ghost_route': True, 'prompt': prompt},
            artifact_payload={
                'output_text': 'Eine Poster-Idee für eine nachhaltige Zukunft.',
                'runtime': {'request_phase_graph': phase_graph},
            },
        )

        self.assertEqual(review['status'], 'pending')
        feedback = review['ghost_repair_feedback']
        self.assertEqual(
            [item.get('capability') for item in feedback['items']],
            ['text_to_speech'],
        )


if __name__ == '__main__':
    unittest.main()
