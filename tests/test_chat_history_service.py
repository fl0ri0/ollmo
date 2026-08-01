import json
import tempfile
import unittest
from pathlib import Path

from ollmo_services.chat_history import (
    LINEAGE_LOG_FILE_NAME,
    delete_chat_history,
    list_chat_history_index,
    read_chat_history,
    resolve_chat_history_slot,
    rotate_chat_history,
    write_chat_history,
)


class ChatHistoryServiceTests(unittest.TestCase):
    def test_write_and_read_chat_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            payload = write_chat_history(
                'gpt-oss:20b-1',
                [
                    {'role': 'user', 'content': 'Hi', 'timestamp': '2026-03-09T20:00:00Z'},
                    {'role': 'assistant', 'content': 'Hello', 'timestamp': '2026-03-09T20:00:01Z'},
                    {'role': 'assistant', 'content': 'Thinking', 'isLoading': True},
                ],
                history_dir=history_dir,
                model='gpt-oss:20b',
                backend='ollama',
                capability='chat',
            )

            self.assertEqual(len(payload['messages']), 2)
            stored = read_chat_history('gpt-oss:20b-1', history_dir=history_dir)
            self.assertEqual(stored['model'], 'gpt-oss:20b')
            self.assertEqual(stored['messages'][0]['content'], 'Hi')
            self.assertEqual(len(stored['messages']), 2)

    def test_write_and_read_chat_history_preserves_artifact_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-batch',
                [
                    {
                        'role': 'assistant',
                        'content': 'Generated 2 images.',
                        'artifacts': [
                            {
                                'type': 'image',
                                'path': '/tmp/artifacts/images/one.png',
                                'image_data_url': 'data:image/png;base64,ZmFrZQ==',
                                'batch_index': 1,
                                'prompt': 'storm over Zurich',
                                'image_state': {
                                    'summary': 'A stormy city skyline over Zurich.',
                                    'subject': 'Zurich skyline',
                                },
                            },
                            {
                                'type': 'image',
                                'path': '/tmp/artifacts/images/two.png',
                                'batch_index': 2,
                                'prompt': 'foggy neon diner',
                            },
                        ],
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-batch', history_dir=history_dir)
            self.assertEqual(len(stored['messages']), 1)
            self.assertEqual(len(stored['messages'][0]['artifacts']), 2)
            self.assertEqual(stored['messages'][0]['artifacts'][0]['batch_index'], 1)
            self.assertEqual(stored['messages'][0]['artifacts'][0]['image_state']['subject'], 'Zurich skyline')
            self.assertNotIn('image_data_url', stored['messages'][0]['artifacts'][0])
            self.assertEqual(stored['messages'][0]['artifacts'][1]['prompt'], 'foggy neon diner')

    def test_write_and_read_chat_history_dedupes_same_path_artifacts_with_different_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-duplicate-image-ref',
                [
                    {
                        'role': 'assistant',
                        'content': 'One image was generated.',
                        'artifacts': [
                            {
                                'type': 'image',
                                'artifact_ref': 'artifact:image_first_ref',
                                'path': '/tmp/artifacts/images/out.png',
                            },
                            {
                                'type': 'image',
                                'artifact_ref': 'artifact:image_second_ref',
                                'path': '/tmp/artifacts/images/out.png',
                            },
                        ],
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-duplicate-image-ref', history_dir=history_dir)
            self.assertEqual(len(stored['messages'][0]['artifacts']), 1)
            self.assertEqual(stored['messages'][0]['artifacts'][0]['path'], '/tmp/artifacts/images/out.png')

    def test_write_and_read_chat_history_dedupes_same_path_artifacts_with_partial_mime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-duplicate-audio-mime',
                [
                    {
                        'role': 'assistant',
                        'content': 'Audio generated.',
                        'artifacts': [
                            {
                                'type': 'audio',
                                'artifact_ref': 'artifact:audio_first_ref',
                                'path': '/tmp/artifacts/audio/out.wav',
                                'mime_type': 'audio/wav',
                            },
                            {
                                'type': 'audio',
                                'artifact_ref': 'artifact:audio_second_ref',
                                'path': '/tmp/artifacts/audio/out.wav',
                            },
                        ],
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-duplicate-audio-mime', history_dir=history_dir)
            artifacts = stored['messages'][0]['artifacts']
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]['path'], '/tmp/artifacts/audio/out.wav')
            self.assertEqual(artifacts[0]['mime_type'], 'audio/wav')

    def test_write_and_read_chat_history_preserves_response_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-provenance',
                [
                    {
                        'role': 'assistant',
                        'content': 'Audio generated.',
                        'response_model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                        'response_backend': 'mlx',
                        'response_instance_id': 'mlx-community__Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16-mlx-11506',
                        'route_source': 'heuristic',
                        'route_reason': 'text-to-speech cue',
                        'context_mode': 'recent_history',
                        'context_reason': 'recent chat history fits the current context budget',
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-provenance', history_dir=history_dir)
            self.assertEqual(stored['messages'][0]['response_model'], 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16')
            self.assertEqual(stored['messages'][0]['response_backend'], 'mlx')
            self.assertEqual(stored['messages'][0]['route_source'], 'heuristic')
            self.assertEqual(stored['messages'][0]['route_reason'], 'text-to-speech cue')
            self.assertEqual(stored['messages'][0]['context_mode'], 'recent_history')
            self.assertEqual(stored['messages'][0]['context_reason'], 'recent chat history fits the current context budget')

    def test_write_and_read_chat_history_preserves_canonical_outputs_for_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-canonical-outputs',
                [
                    {
                        'role': 'assistant',
                        'content': '',
                        'outputs': [
                            {
                                'slot_id': 'output-phase-1',
                                'branch_id': 'phase-1',
                                'phase_id': 'phase-1',
                                'type': 'text',
                                'status': 'fulfilled',
                                'lifecycle': 'materialized_output',
                                'value': 'A quiet valley opened in silver dawn.',
                                'child_slot_ids': ['output-phase-image-1'],
                            },
                            {
                                'slot_id': 'output-phase-image-1',
                                'branch_id': 'phase-image-1',
                                'phase_id': 'phase-image-1',
                                'type': 'image',
                                'status': 'fulfilled',
                                'lifecycle': 'materialized_output',
                                'artifact_ref': 'artifact:image_valley',
                                'parent_slot_id': 'output-phase-1',
                                'artifacts': [
                                    {
                                        'type': 'image',
                                        'artifact_ref': 'artifact:image_valley',
                                        'path': '/tmp/artifacts/images/valley.png',
                                    }
                                ],
                            },
                        ],
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-canonical-outputs', history_dir=history_dir)
            message = stored['messages'][0]
            self.assertEqual(message['outputs'][0]['type'], 'text')
            self.assertEqual(message['outputs'][0]['value'], 'A quiet valley opened in silver dawn.')
            self.assertEqual(message['outputs'][1]['type'], 'image')
            self.assertEqual(message['outputs'][1]['artifact_ref'], 'artifact:image_valley')
            self.assertEqual(message['outputs'][1]['artifacts'][0]['path'], '/tmp/artifacts/images/valley.png')

    def test_read_chat_history_rehydrates_late_fill_outputs_from_response_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'chat_history'
            frames_dir = root / 'response_frames'
            frames_dir.mkdir(parents=True)
            response_id = 'resp_two_late_images'
            write_chat_history(
                'responses-workbench',
                [
                    {
                        'role': 'assistant',
                        'content': 'Generated 2 images.',
                        'response_id': response_id,
                    },
                ],
                history_dir=history_dir,
            )
            frame = {
                'kind': 'ollmo.response_frame',
                'response_id': response_id,
                'artifacts': {
                    'dossiers': {
                        'artifact:image_one': {
                            'artifact': {
                                'type': 'image',
                                'kind': 'image',
                                'path': '/tmp/artifacts/images/one.png',
                                'artifact_ref': 'artifact:image_one',
                                'artifact_id': 'image_one',
                                'origin': 'assistant_output',
                            },
                            'metadata': {'availability': 'available'},
                        },
                        'artifact:image_two': {
                            'artifact': {
                                'type': 'image',
                                'kind': 'image',
                                'path': '/tmp/artifacts/images/two.png',
                                'artifact_ref': 'artifact:image_two',
                                'artifact_id': 'image_two',
                                'origin': 'assistant_output',
                            },
                            'metadata': {'availability': 'available'},
                        },
                    },
                },
                'planning': {
                    'artifact_flow': {
                        'output_slots': [
                            {'slot_id': 'output-text', 'type': 'text', 'status': 'fulfilled'},
                            {
                                'slot_id': 'output-image-1',
                                'type': 'image',
                                'status': 'fulfilled',
                                'artifact_ref': 'artifact:image_one',
                            },
                            {
                                'slot_id': 'output-image-2',
                                'type': 'image',
                                'status': 'fulfilled',
                                'artifact_ref': 'artifact:image_two',
                            },
                        ],
                    },
                },
                'output': {
                    'outputs': [
                        {'slot_id': 'output-text', 'type': 'text', 'status': 'fulfilled', 'value': 'Two image prompts.'},
                        {
                            'slot_id': 'output-image-1',
                            'type': 'image',
                            'status': 'fulfilled',
                            'artifact_ref': 'artifact:image_one',
                            'artifacts': [
                                {
                                    'type': 'image',
                                    'path': '/tmp/artifacts/images/one.png',
                                    'artifact_ref': 'artifact:image_one',
                                }
                            ],
                        },
                        {
                            'slot_id': 'output-image-2',
                            'type': 'image',
                            'status': 'fulfilled',
                            'artifact_ref': 'artifact:image_two',
                            'artifacts': [
                                {
                                    'type': 'image',
                                    'path': '/tmp/artifacts/images/two.png',
                                    'artifact_ref': 'artifact:image_two',
                                }
                            ],
                        },
                    ],
                },
            }
            (frames_dir / 'responses.jsonl').write_text(json.dumps(frame) + '\n', encoding='utf-8')

            stored = read_chat_history('responses-workbench', history_dir=history_dir)
            message = stored['messages'][0]
            self.assertEqual(
                sorted(artifact['path'] for artifact in message['artifacts']),
                ['/tmp/artifacts/images/one.png', '/tmp/artifacts/images/two.png'],
            )
            self.assertEqual(len(message['output_slots']), 3)
            self.assertEqual(len(message['outputs']), 3)
            self.assertEqual(
                sorted(
                    artifact['path']
                    for output in message['outputs']
                    for artifact in output.get('artifacts', [])
                ),
                ['/tmp/artifacts/images/one.png', '/tmp/artifacts/images/two.png'],
            )

    def test_write_and_read_chat_history_preserves_image_route_debug_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-image-debug',
                [
                    {
                        'role': 'assistant',
                        'content': 'Image generated from the reference image.',
                        'saved_image_path': '/tmp/generated/cat.png',
                        'response_model': 'x/flux2-klein:latest',
                        'response_backend': 'ollama',
                        'response_instance_id': 'flux-1',
                        'route_source': 'embedding_tiebreak',
                        'route_reason': 'image-generation prompt references the latest image artifact',
                        'route_artifact_path': '/tmp/generated/cat-with-chicken.png',
                        'route_reuse_last_artifact': True,
                        'reference_image_count': 1,
                        'reference_image_kind': 'image',
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-image-debug', history_dir=history_dir)
            message = stored['messages'][0]
            self.assertEqual(message['route_artifact_path'], '/tmp/generated/cat-with-chicken.png')
            self.assertEqual(message['route_reuse_last_artifact'], True)
            self.assertEqual(message['reference_image_count'], 1)
            self.assertEqual(message['reference_image_kind'], 'image')

    def test_write_and_read_chat_history_preserves_request_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-request-snapshot',
                [
                    {
                        'role': 'assistant',
                        'content': 'Audio generated.',
                        'saved_audio_path': '/tmp/sample.wav',
                        'request_snapshot': {
                            'request_id': 'msg-123',
                            'created_at': '2026-04-05T07:55:00Z',
                            'conversation_id': '__responses_workbench__',
                            'transport': 'responses',
                            'prompt_text': 'Read this in a calm radio voice.',
                            'target': {
                                'instance_id': 'qwen-tts-1',
                                'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                                'backend': 'mlx',
                                'capability': 'text_to_speech',
                            },
                            'session_controls': {
                                'voice': 'alloy',
                                'speed': '1.1',
                            },
                            'input_artifacts': [
                                {
                                    'type': 'audio',
                                    'path': '/tmp/artifacts/inputs/audio/request.wav',
                                    'source_path': '/tmp/local/request.wav',
                                    'mime_type': 'audio/wav',
                                },
                            ],
                            'settings': {
                                'ttsVoice': 'alloy',
                                'ttsSpeed': 1.1,
                                'pdfSynthesize': False,
                                'imageWidth': None,
                            },
                            'selected_reference_artifacts': [
                                {
                                    'type': 'message',
                                    'content': 'Use the same dramatic style as before.',
                                    'message_role': 'user',
                                },
                                {
                                    'type': 'audio',
                                    'path': '/tmp/reference-voice.wav',
                                    'prompt': 'previous voice sample',
                                },
                            ],
                        },
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-request-snapshot', history_dir=history_dir)
            snapshot = stored['messages'][0]['request_snapshot']
            self.assertEqual(snapshot['request_id'], 'msg-123')
            self.assertEqual(snapshot['target']['backend'], 'mlx')
            self.assertEqual(snapshot['session_controls']['voice'], 'alloy')
            self.assertEqual(snapshot['settings']['ttsVoice'], 'alloy')
            self.assertEqual(snapshot['settings']['pdfSynthesize'], False)
            self.assertIsNone(snapshot['settings']['imageWidth'])
            self.assertEqual(snapshot['input_artifacts'][0]['source_path'], '/tmp/local/request.wav')
            self.assertEqual(snapshot['reference_artifacts'][0]['type'], 'message')
            self.assertEqual(snapshot['reference_artifacts'][1]['path'], '/tmp/reference-voice.wav')

    def test_write_and_read_chat_history_does_not_mirror_selected_reference_artifacts_into_request_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-selected-reference-snapshot',
                [
                    {
                        'role': 'user',
                        'content': 'Use the earlier image as the active reference.',
                        'artifacts': [
                            {
                                'type': 'image',
                                'path': '/tmp/artifacts/inputs/image/selected-reference-copy.png',
                                'source_path': '/tmp/generated/reference-source.png',
                            },
                        ],
                        'request_snapshot': {
                            'request_id': 'msg-selected-ref',
                            'selected_reference_artifacts': [
                                {
                                    'type': 'image',
                                    'path': '/tmp/generated/reference-source.png',
                                    'seed': 777,
                                },
                            ],
                        },
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-selected-reference-snapshot', history_dir=history_dir)
            message = stored['messages'][0]
            snapshot = message['request_snapshot']
            self.assertEqual(snapshot['reference_artifacts'][0]['path'], '/tmp/generated/reference-source.png')
            self.assertNotIn('input_artifacts', snapshot)
            self.assertEqual(message['artifacts'][0]['source_path'], '/tmp/generated/reference-source.png')

    def test_write_and_read_chat_history_prefers_request_snapshot_input_artifact_identity_for_user_uploads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-user-upload-identity',
                [
                    {
                        'role': 'user',
                        'content': 'Extract the quote from this screenshot.',
                        'artifacts': [
                            {
                                'type': 'image',
                                'path': '/tmp/artifacts/inputs/image/screenshot.png',
                                'name': 'screenshot.png',
                            },
                        ],
                        'request_snapshot': {
                            'request_id': 'msg-upload-identity',
                            'input_artifacts': [
                                {
                                    'type': 'image',
                                    'path': '/tmp/artifacts/inputs/image/screenshot.png',
                                    'name': 'screenshot.png',
                                    'artifact_id': 'image_request_snapshot_id',
                                    'artifact_ref': 'artifact:image_request_snapshot_id',
                                },
                            ],
                        },
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-user-upload-identity', history_dir=history_dir)
            message = stored['messages'][0]
            self.assertEqual(len(message['artifacts']), 1)
            self.assertEqual(message['artifacts'][0]['artifact_id'], 'image_request_snapshot_id')
            self.assertEqual(message['artifacts'][0]['artifact_ref'], 'artifact:image_request_snapshot_id')
            self.assertEqual(message['request_snapshot']['input_artifacts'][0]['artifact_id'], 'image_request_snapshot_id')

    def test_write_and_read_chat_history_preserves_compact_selected_message_reference_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'responses-selected-message-reference',
                [
                    {
                        'role': 'user',
                        'content': 'Make the next answer shorter.',
                        'request_snapshot': {
                            'request_id': 'msg-selected-message',
                            'selected_reference_artifacts': [
                                {
                                    'type': 'message',
                                    'message_id': 'msg_reference',
                                    'message_role': 'assistant',
                                    'response_model': 'gemma4:26b',
                                    'response_instance_id': 'gemma4:26b-1',
                                    'timestamp': '2026-04-11T14:40:00Z',
                                    'content': 'A longer earlier reply that should compact to identity fields.',
                                },
                            ],
                        },
                    },
                ],
                history_dir=history_dir,
            )

            stored = read_chat_history('responses-selected-message-reference', history_dir=history_dir)
            selected_reference = stored['messages'][0]['request_snapshot']['reference_artifacts'][0]
            self.assertEqual(selected_reference['type'], 'message')
            self.assertEqual(selected_reference['artifact_ref'], 'message:message_msg_reference')
            self.assertEqual(selected_reference['message_id'], 'msg_reference')
            self.assertEqual(selected_reference['response_model'], 'gemma4:26b')
            self.assertEqual(selected_reference['response_instance_id'], 'gemma4:26b-1')
            self.assertEqual(selected_reference['timestamp'], '2026-04-11T14:40:00Z')
            self.assertNotIn('content', selected_reference)

    def test_delete_chat_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history('qwen-1', [{'role': 'user', 'content': 'x'}], history_dir=history_dir)
            self.assertTrue(delete_chat_history('qwen-1', history_dir=history_dir))
            self.assertFalse(delete_chat_history('qwen-1', history_dir=history_dir))

    def test_write_and_read_chat_history_preserves_conversation_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                '__responses_workbench__--test',
                [{'role': 'user', 'content': 'hello'}],
                history_dir=history_dir,
                conversation_metadata={
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                    'parent_conversation_id': '__responses_workbench__',
                    'root_conversation_id': '__responses_workbench__',
                    'created_at': '2026-04-04T17:00:00Z',
                    'display_title': 'Hello',
                    'preview_text': 'hello',
                    'message_count': 1,
                    'last_message_at': '2026-04-04T17:00:01Z',
                },
            )

            stored = read_chat_history('__responses_workbench__--test', history_dir=history_dir)
            self.assertEqual(stored['conversation_metadata']['workspace'], 'responses')
            self.assertEqual(stored['conversation_metadata']['slot_id'], 'responses-workbench')
            self.assertEqual(stored['conversation_metadata']['parent_conversation_id'], '__responses_workbench__')
            self.assertEqual(stored['conversation_metadata']['root_conversation_id'], '__responses_workbench__')
            self.assertEqual(stored['conversation_metadata']['display_title'], 'Hello')
            self.assertEqual(stored['conversation_metadata']['preview_text'], 'hello')
            self.assertEqual(stored['conversation_metadata']['message_count'], 1)
            self.assertEqual(stored['conversation_metadata']['last_message_at'], '2026-04-04T17:00:01Z')

    def test_rotate_chat_history_creates_successor_and_lineage_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                'gpt-oss:20b-1',
                [{'role': 'user', 'content': 'hello'}],
                history_dir=history_dir,
                model='gpt-oss:20b',
                backend='ollama',
                capability='chat',
            )

            successor = rotate_chat_history(
                'gpt-oss:20b-1',
                history_dir=history_dir,
                workspace='instance',
                slot_id='instance:gpt-oss:20b-1',
                source_instance_id='gpt-oss:20b-1',
                label='gpt-oss:20b-1',
            )

            self.assertTrue(successor['instance_id'].startswith('__instance_chat__--gpt-oss_20b-1--'))
            self.assertEqual(successor['messages'], [])
            self.assertEqual(successor['model'], 'gpt-oss:20b')
            self.assertEqual(successor['backend'], 'ollama')
            self.assertEqual(successor['capability'], 'chat')
            self.assertEqual(successor['conversation_metadata']['parent_conversation_id'], 'gpt-oss:20b-1')
            self.assertEqual(successor['conversation_metadata']['root_conversation_id'], 'gpt-oss:20b-1')
            self.assertEqual(read_chat_history('gpt-oss:20b-1', history_dir=history_dir)['messages'][0]['content'], 'hello')

            lineage_path = history_dir / LINEAGE_LOG_FILE_NAME
            self.assertTrue(lineage_path.exists())
            entries = [line for line in lineage_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(len(entries), 1)
            self.assertIn('"from_conversation_id": "gpt-oss:20b-1"', entries[0])
            self.assertIn(successor['instance_id'], entries[0])

    def test_rotate_chat_history_can_start_fresh_root_for_responses_workbench(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                '__responses_workbench__',
                [{'role': 'user', 'content': 'old workbench chat'}],
                history_dir=history_dir,
                conversation_metadata={
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                },
            )

            successor = rotate_chat_history(
                '__responses_workbench__',
                history_dir=history_dir,
                workspace='responses',
                slot_id='responses-workbench',
                label='responses-workbench',
                fresh_root=True,
            )

            metadata = successor['conversation_metadata']
            self.assertEqual(metadata['workspace'], 'responses')
            self.assertEqual(metadata['slot_id'], 'responses-workbench')
            self.assertTrue(metadata['fresh_root'])
            self.assertNotIn('parent_conversation_id', metadata)
            self.assertEqual(metadata['root_conversation_id'], successor['instance_id'])
            self.assertTrue(successor['rotation_event']['fresh_root'])
            self.assertNotIn('parent_conversation_id', successor['rotation_event'])

    def test_resolve_chat_history_slot_returns_latest_rotated_successor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                '__responses_workbench__',
                [{'role': 'user', 'content': 'older base'}],
                history_dir=history_dir,
                conversation_metadata={
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                },
            )

            successor = rotate_chat_history(
                '__responses_workbench__',
                history_dir=history_dir,
                workspace='responses',
                slot_id='responses-workbench',
                label='responses-workbench',
            )
            write_chat_history(
                successor['instance_id'],
                [{'role': 'assistant', 'content': 'latest successor message'}],
                history_dir=history_dir,
                conversation_metadata=successor.get('conversation_metadata'),
            )

            resolved = resolve_chat_history_slot(
                workspace='responses',
                slot_id='responses-workbench',
                history_dir=history_dir,
                fallback_instance_id='__responses_workbench__',
            )

            self.assertEqual(resolved['instance_id'], successor['instance_id'])
            self.assertEqual(resolved['messages'][0]['content'], 'latest successor message')
            self.assertEqual(
                resolved['slot_history_ids'],
                ['__responses_workbench__', successor['instance_id']],
            )

    def test_list_chat_history_index_includes_saved_chats_from_non_running_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            write_chat_history(
                '__responses_workbench__--saved',
                [{'role': 'user', 'content': 'hello from ollmo', 'timestamp': '2026-04-08T10:00:00Z'}],
                history_dir=history_dir,
                conversation_metadata={
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                },
            )
            write_chat_history(
                '__instance_chat__--retired-model--20260408T090000Z-abcdef12',
                [{'role': 'user', 'content': 'offline model chat', 'timestamp': '2026-04-08T09:00:00Z'}],
                history_dir=history_dir,
                model='retired/model:latest',
                backend='ollama',
                capability='chat',
                conversation_metadata={
                    'workspace': 'instance',
                    'slot_id': 'instance:retired-model-1',
                    'source_instance_id': 'retired-model-1',
                    'label': 'retired-model-1',
                },
            )
            write_chat_history(
                '__instance_chat__--empty-model--20260408T080000Z-deadbeef',
                [],
                history_dir=history_dir,
                model='empty/model:latest',
                backend='ollama',
                capability='chat',
                conversation_metadata={
                    'workspace': 'instance',
                    'slot_id': 'instance:empty-model-1',
                    'source_instance_id': 'empty-model-1',
                    'label': 'empty-model-1',
                },
            )

            items = list_chat_history_index(history_dir=history_dir)

            item_ids = [item['instance_id'] for item in items]
            self.assertIn('__responses_workbench__--saved', item_ids)
            self.assertIn('__instance_chat__--retired-model--20260408T090000Z-abcdef12', item_ids)
            self.assertNotIn('__instance_chat__--empty-model--20260408T080000Z-deadbeef', item_ids)
            retired_item = next(item for item in items if item['instance_id'] == '__instance_chat__--retired-model--20260408T090000Z-abcdef12')
            self.assertEqual(retired_item['model'], 'retired/model:latest')
            self.assertEqual(retired_item['conversation_metadata']['display_title'], 'offline model chat')
            self.assertEqual(retired_item['conversation_metadata']['message_count'], 1)


if __name__ == '__main__':
    unittest.main()
