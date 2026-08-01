import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_webserver import _RESPONSE_LOOKUP, app


class ChatHistoryApiTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        _RESPONSE_LOOKUP.clear()

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_round_trip(self, _mock_history_dir):
        response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': 'gpt-oss:20b-1',
                'model': 'gpt-oss:20b',
                'backend': 'ollama',
                'capability': 'chat',
                'conversation_metadata': {
                    'workspace': 'instance',
                    'slot_id': 'instance:gpt-oss:20b-1',
                    'source_instance_id': 'gpt-oss:20b-1',
                    'label': 'gpt-oss:20b-1',
                    'display_title': 'Hello',
                    'preview_text': 'hi',
                    'message_count': 2,
                    'last_message_at': '2026-03-09T20:00:01Z',
                },
                'messages': [
                    {'role': 'user', 'content': 'hello', 'timestamp': '2026-03-09T20:00:00Z'},
                    {'role': 'assistant', 'content': 'hi', 'timestamp': '2026-03-09T20:00:01Z'},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)

        get_response = self.client.get('/api/chat_history?instance_id=gpt-oss:20b-1')
        self.assertEqual(get_response.status_code, 200)
        payload = get_response.get_json()
        self.assertEqual(payload['messages'][1]['content'], 'hi')
        self.assertEqual(payload['conversation_metadata']['display_title'], 'Hello')
        self.assertEqual(payload['conversation_metadata']['preview_text'], 'hi')
        self.assertEqual(payload['conversation_metadata']['message_count'], 2)
        self.assertEqual(payload['conversation_metadata']['last_message_at'], '2026-03-09T20:00:01Z')

        delete_response = self.client.delete('/api/chat_history?instance_id=gpt-oss:20b-1')
        self.assertEqual(delete_response.status_code, 200)
        self.assertTrue(delete_response.get_json()['deleted'])

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_round_trip_for_responses_workbench(self, mock_history_dir):
        response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'make an audio sample',
                        'timestamp': '2026-03-21T13:00:00Z',
                    },
                    {
                        'role': 'assistant',
                        'content': 'Audio generated.',
                        'timestamp': '2026-03-21T13:00:05Z',
                        'saved_audio_path': '/tmp/sample.wav',
                        'response_model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                        'response_backend': 'mlx',
                        'route_source': 'heuristic',
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()['model'])
        self.assertTrue((mock_history_dir / 'responses_workbench.json').exists())

        get_response = self.client.get('/api/chat_history?instance_id=__responses_workbench__')
        self.assertEqual(get_response.status_code, 200)
        payload = get_response.get_json()
        self.assertEqual(payload['instance_id'], '__responses_workbench__')
        self.assertIsNone(payload['backend'])
        self.assertEqual(payload['messages'][1]['saved_audio_path'], '/tmp/sample.wav')
        self.assertEqual(payload['messages'][1]['response_model'], 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16')
        self.assertEqual(payload['messages'][1]['route_source'], 'heuristic')

        delete_response = self.client.delete('/api/chat_history?instance_id=__responses_workbench__')
        self.assertEqual(delete_response.status_code, 200)
        self.assertTrue(delete_response.get_json()['deleted'])
        self.assertFalse((mock_history_dir / 'responses_workbench.json').exists())

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_post_reconciles_reduced_response_projection_before_write(self, _mock_history_dir):
        response_id = 'resp_multi_artifact_history_post'
        _RESPONSE_LOOKUP[response_id] = {
            'id': response_id,
            'message_id': 'msg_multi_artifact_history_post',
            'instance_id': 'chat-1',
            'model_name': 'gemma4:26b',
            'backend': 'ollama',
            'capability': 'chat',
            'mode': 'chat',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'expires_at_ts': time.time() + 3600,
            'response_payload': {
                'id': response_id,
                'status': 'completed',
                'lifecycle_state': 'completed',
                'output_text': 'Generated image, audio, and document artifacts.',
                'artifacts': [
                    {'type': 'image', 'artifact_ref': 'artifact:image-1', 'path': '/tmp/generated/image-1.png'},
                    {'type': 'audio', 'artifact_ref': 'artifact:audio-1', 'path': '/tmp/generated/audio-1.wav'},
                    {'type': 'document', 'artifact_ref': 'artifact:doc-1', 'path': '/tmp/generated/report.md'},
                ],
                'outputs': [
                    {'type': 'text', 'status': 'fulfilled', 'value': 'Generated image, audio, and document artifacts.'},
                    {'type': 'image', 'status': 'fulfilled', 'artifact_ref': 'artifact:image-1'},
                    {'type': 'audio', 'status': 'fulfilled', 'artifact_ref': 'artifact:audio-1'},
                    {'type': 'document', 'status': 'fulfilled', 'artifact_ref': 'artifact:doc-1'},
                ],
                'output_slots': [
                    {'type': 'text', 'status': 'fulfilled', 'slot_id': 'slot-text-1'},
                    {'type': 'image', 'status': 'fulfilled', 'slot_id': 'slot-image-1', 'artifact_ref': 'artifact:image-1'},
                    {'type': 'audio', 'status': 'fulfilled', 'slot_id': 'slot-audio-1', 'artifact_ref': 'artifact:audio-1'},
                    {'type': 'document', 'status': 'fulfilled', 'slot_id': 'slot-doc-1', 'artifact_ref': 'artifact:doc-1'},
                ],
                'output_branches': [
                    {'type': 'image', 'status': 'completed', 'branch_id': 'branch-image-1', 'artifact_ref': 'artifact:image-1'},
                    {'type': 'audio', 'status': 'completed', 'branch_id': 'branch-audio-1', 'artifact_ref': 'artifact:audio-1'},
                    {'type': 'document', 'status': 'completed', 'branch_id': 'branch-doc-1', 'artifact_ref': 'artifact:doc-1'},
                ],
            },
        }

        response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__--projection-post-test',
                'messages': [
                    {'role': 'user', 'content': 'make mixed artifacts'},
                    {
                        'role': 'assistant',
                        'content': 'Generated image, audio, and document artifacts.',
                        'response_id': response_id,
                        'artifacts': [
                            {'type': 'image', 'artifact_ref': 'artifact:image-1', 'path': '/tmp/generated/image-1.png'},
                        ],
                        'outputs': [
                            {'type': 'text', 'status': 'fulfilled', 'value': 'Generated image, audio, and document artifacts.'},
                            {'type': 'image', 'status': 'fulfilled', 'artifact_ref': 'artifact:image-1'},
                            {'type': 'audio', 'status': 'fulfilled', 'artifact_ref': 'artifact:audio-1'},
                            {'type': 'document', 'status': 'fulfilled', 'artifact_ref': 'artifact:doc-1'},
                        ],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        written_message = response.get_json()['messages'][1]
        self.assertEqual(
            sorted(artifact['type'] for artifact in written_message['artifacts']),
            ['audio', 'document', 'image'],
        )
        self.assertEqual(len(written_message['outputs']), 4)
        self.assertEqual(len(written_message['output_slots']), 4)
        self.assertGreaterEqual(len(written_message['output_branches']), 3)
        self.assertTrue(
            {'image', 'audio', 'document'}.issubset(
                {branch.get('type') for branch in written_message['output_branches']}
            )
        )

        get_response = self.client.get('/api/chat_history?instance_id=__responses_workbench__--projection-post-test')
        self.assertEqual(get_response.status_code, 200)
        hydrated_message = get_response.get_json()['messages'][1]
        self.assertEqual(
            sorted(artifact['type'] for artifact in hydrated_message['artifacts']),
            ['audio', 'document', 'image'],
        )

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_get_synthesizes_missing_assistant_from_user_request_snapshot_response_id(self, _mock_history_dir):
        response_id = 'resp_missing_assistant_history'
        post_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__--missing-assistant',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'build a local landing page',
                        'timestamp': '2026-06-13T09:40:31Z',
                        'request_snapshot': {
                            'request_id': 'msg-user-missing-assistant',
                            'conversation_id': '__responses_workbench__--missing-assistant',
                            'response_id': response_id,
                            'transport': 'responses',
                            'prompt_text': 'build a local landing page',
                        },
                    },
                ],
            },
        )

        self.assertEqual(post_response.status_code, 200)
        self.assertEqual(len(post_response.get_json()['messages']), 1)
        _RESPONSE_LOOKUP[response_id] = {
            'id': response_id,
            'message_id': 'msg_missing_assistant_history',
            'instance_id': 'chat-1',
            'model_name': 'gemma4:26b',
            'backend': 'ollama',
            'capability': 'chat',
            'mode': 'chat',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'expires_at_ts': time.time() + 3600,
            'response_payload': {
                'id': response_id,
                'status': 'completed',
                'lifecycle_state': 'completed',
                'output_text': 'Artifacts generated.',
                'artifacts': [
                    {'type': 'image', 'artifact_ref': 'artifact:image-1', 'path': '/tmp/generated/image-1.png'},
                    {'type': 'text', 'artifact_ref': 'artifact:index', 'path': '/tmp/generated/index.html'},
                    {'type': 'text', 'artifact_ref': 'artifact:styles', 'path': '/tmp/generated/styles.css'},
                ],
                'outputs': [
                    {'type': 'image', 'status': 'fulfilled', 'artifact_ref': 'artifact:image-1'},
                    {'type': 'text', 'status': 'fulfilled', 'artifact_ref': 'artifact:index'},
                    {'type': 'text', 'status': 'fulfilled', 'artifact_ref': 'artifact:styles'},
                ],
                'output_slots': [
                    {'type': 'image', 'status': 'fulfilled', 'slot_id': 'slot-image-1', 'artifact_ref': 'artifact:image-1'},
                    {'type': 'text', 'status': 'fulfilled', 'slot_id': 'slot-index', 'artifact_ref': 'artifact:index'},
                    {'type': 'text', 'status': 'fulfilled', 'slot_id': 'slot-styles', 'artifact_ref': 'artifact:styles'},
                ],
                'output_branches': [
                    {'type': 'image', 'status': 'completed', 'branch_id': 'branch-image-1', 'artifact_ref': 'artifact:image-1'},
                    {'type': 'text', 'status': 'completed', 'branch_id': 'branch-index', 'artifact_ref': 'artifact:index'},
                    {'type': 'text', 'status': 'completed', 'branch_id': 'branch-styles', 'artifact_ref': 'artifact:styles'},
                ],
            },
        }
        get_response = self.client.get('/api/chat_history?instance_id=__responses_workbench__--missing-assistant')

        self.assertEqual(get_response.status_code, 200)
        payload = get_response.get_json()
        self.assertEqual(len(payload['messages']), 2)
        assistant = payload['messages'][1]
        self.assertEqual(assistant['role'], 'assistant')
        self.assertEqual(assistant['response_id'], response_id)
        self.assertEqual(assistant['content'], 'Artifacts generated.')
        self.assertEqual(assistant['output_text'], 'Artifacts generated.')
        self.assertEqual(len(assistant['artifacts']), 3)
        self.assertEqual(len(assistant['outputs']), 3)
        self.assertEqual(len(assistant['output_slots']), 3)
        self.assertEqual(len(assistant['output_branches']), 3)
        self.assertEqual(assistant['request_snapshot']['response_id'], response_id)

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_round_trip_preserves_request_snapshot(self, _mock_history_dir):
        response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'make an audio sample',
                        'timestamp': '2026-04-05T07:58:00Z',
                        'request_snapshot': {
                            'request_id': 'msg-456',
                            'created_at': '2026-04-05T07:58:00Z',
                            'conversation_id': '__responses_workbench__',
                            'transport': 'responses',
                            'prompt_text': 'make an audio sample',
                            'target': {
                                'instance_id': 'mlx-qwen-tts',
                                'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                                'backend': 'mlx',
                                'capability': 'text_to_speech',
                            },
                            'session_controls': {
                                'voice': 'alloy',
                                'speed': '1.0',
                            },
                            'settings': {
                                'ttsVoice': 'alloy',
                                'ttsSpeed': 1.0,
                                'pdfSynthesize': False,
                            },
                            'selected_reference_artifacts': [
                                {
                                    'type': 'audio',
                                    'path': '/tmp/reference-voice.wav',
                                    'prompt': 'prior take',
                                },
                            ],
                        },
                    },
                    {
                        'role': 'assistant',
                        'content': 'Audio generated.',
                        'timestamp': '2026-04-05T07:58:05Z',
                        'saved_audio_path': '/tmp/sample.wav',
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)

        get_response = self.client.get('/api/chat_history?instance_id=__responses_workbench__')
        self.assertEqual(get_response.status_code, 200)
        payload = get_response.get_json()
        snapshot = payload['messages'][0]['request_snapshot']
        self.assertEqual(snapshot['request_id'], 'msg-456')
        self.assertEqual(snapshot['target']['capability'], 'text_to_speech')
        self.assertEqual(snapshot['session_controls']['voice'], 'alloy')
        self.assertEqual(snapshot['settings']['pdfSynthesize'], False)
        self.assertEqual(snapshot['reference_artifacts'][0]['path'], '/tmp/reference-voice.wav')

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_rotate_keeps_previous_file_and_returns_successor(self, mock_history_dir):
        seed_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'read this aloud',
                        'timestamp': '2026-04-04T17:10:00Z',
                    },
                ],
            },
        )
        self.assertEqual(seed_response.status_code, 200)
        self.assertTrue((mock_history_dir / 'responses_workbench.json').exists())

        rotate_response = self.client.post(
            '/api/chat_history/rotate',
            json={
                'current_instance_id': '__responses_workbench__',
                'workspace': 'responses',
                'slot_id': 'responses-workbench',
                'label': 'responses-workbench',
            },
        )
        self.assertEqual(rotate_response.status_code, 200)
        payload = rotate_response.get_json()
        self.assertTrue(payload['instance_id'].startswith('__responses_workbench__--'))
        self.assertEqual(payload['messages'], [])
        self.assertEqual(payload['conversation_metadata']['workspace'], 'responses')
        self.assertEqual(payload['conversation_metadata']['parent_conversation_id'], '__responses_workbench__')
        self.assertEqual(payload['conversation_metadata']['root_conversation_id'], '__responses_workbench__')
        self.assertEqual(payload['rotation_event']['from_conversation_id'], '__responses_workbench__')
        self.assertEqual(payload['rotation_event']['to_conversation_id'], payload['instance_id'])
        self.assertTrue((mock_history_dir / 'responses_workbench.json').exists())
        successor_get = self.client.get(f"/api/chat_history?instance_id={payload['instance_id']}")
        self.assertEqual(successor_get.status_code, 200)
        self.assertEqual(successor_get.get_json()['instance_id'], payload['instance_id'])

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_slot_endpoint_returns_latest_responses_successor(self, _mock_history_dir):
        seed_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__',
                'conversation_metadata': {
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                },
                'messages': [
                    {'role': 'user', 'content': 'base chat'},
                ],
            },
        )
        self.assertEqual(seed_response.status_code, 200)

        rotate_response = self.client.post(
            '/api/chat_history/rotate',
            json={
                'current_instance_id': '__responses_workbench__',
                'workspace': 'responses',
                'slot_id': 'responses-workbench',
                'label': 'responses-workbench',
            },
        )
        self.assertEqual(rotate_response.status_code, 200)
        successor_id = rotate_response.get_json()['instance_id']

        write_successor_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': successor_id,
                'conversation_metadata': rotate_response.get_json()['conversation_metadata'],
                'messages': [
                    {'role': 'assistant', 'content': 'latest successor reply'},
                ],
            },
        )
        self.assertEqual(write_successor_response.status_code, 200)

        get_response = self.client.get(
            '/api/chat_history/slot',
            query_string={
                'workspace': 'responses',
                'slot_id': 'responses-workbench',
                'fallback_instance_id': '__responses_workbench__',
            },
        )
        self.assertEqual(get_response.status_code, 200)
        payload = get_response.get_json()
        self.assertEqual(payload['instance_id'], successor_id)
        self.assertEqual(payload['messages'][0]['content'], 'latest successor reply')

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_rotate_supports_fresh_root_for_responses_workbench(self, _mock_history_dir):
        seed_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__',
                'conversation_metadata': {
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                },
                'messages': [
                    {'role': 'user', 'content': 'base chat'},
                ],
            },
        )
        self.assertEqual(seed_response.status_code, 200)

        rotate_response = self.client.post(
            '/api/chat_history/rotate',
            json={
                'current_instance_id': '__responses_workbench__',
                'workspace': 'responses',
                'slot_id': 'responses-workbench',
                'label': 'responses-workbench',
                'fresh_root': True,
            },
        )
        self.assertEqual(rotate_response.status_code, 200)
        payload = rotate_response.get_json()
        self.assertTrue(payload['conversation_metadata']['fresh_root'])
        self.assertEqual(payload['conversation_metadata']['root_conversation_id'], payload['instance_id'])
        self.assertNotIn('parent_conversation_id', payload['conversation_metadata'])
        self.assertTrue(payload['rotation_event']['fresh_root'])
        self.assertNotIn('parent_conversation_id', payload['rotation_event'])

    @patch('ollmo_webserver.CHAT_HISTORY_DIR', new_callable=lambda: Path(tempfile.mkdtemp()))
    def test_chat_history_index_endpoint_lists_all_saved_chats(self, _mock_history_dir):
        first_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__responses_workbench__--saved',
                'conversation_metadata': {
                    'workspace': 'responses',
                    'slot_id': 'responses-workbench',
                    'label': 'responses-workbench',
                },
                'messages': [
                    {'role': 'user', 'content': 'hello from ollmo', 'timestamp': '2026-04-08T10:00:00Z'},
                ],
            },
        )
        self.assertEqual(first_response.status_code, 200)

        second_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '__instance_chat__--retired-model--20260408T090000Z-abcdef12',
                'model': 'retired/model:latest',
                'backend': 'ollama',
                'capability': 'chat',
                'conversation_metadata': {
                    'workspace': 'instance',
                    'slot_id': 'instance:retired-model-1',
                    'source_instance_id': 'retired-model-1',
                    'label': 'retired-model-1',
                },
                'messages': [
                    {'role': 'user', 'content': 'offline model chat', 'timestamp': '2026-04-08T09:00:00Z'},
                ],
            },
        )
        self.assertEqual(second_response.status_code, 200)

        index_response = self.client.get('/api/chat_history/index')
        self.assertEqual(index_response.status_code, 200)
        items = index_response.get_json()['items']
        item_ids = [item['instance_id'] for item in items]
        self.assertIn('__responses_workbench__--saved', item_ids)
        self.assertIn('__instance_chat__--retired-model--20260408T090000Z-abcdef12', item_ids)
        retired_item = next(item for item in items if item['instance_id'] == '__instance_chat__--retired-model--20260408T090000Z-abcdef12')
        self.assertEqual(retired_item['model'], 'retired/model:latest')
        self.assertEqual(retired_item['conversation_metadata']['display_title'], 'offline model chat')

    def test_chat_history_routes_reject_traversal_shaped_identifiers(self):
        response = self.client.get('/api/chat_history?instance_id=../secret')
        self.assertEqual(response.status_code, 400)
        self.assertIn('invalid path segments', response.get_json()['error'])

        upsert_response = self.client.post(
            '/api/chat_history',
            json={
                'instance_id': '../secret',
                'messages': [],
            },
        )
        self.assertEqual(upsert_response.status_code, 400)
        self.assertIn('invalid path segments', upsert_response.get_json()['error'])

        slot_response = self.client.get(
            '/api/chat_history/slot',
            query_string={
                'workspace': 'responses',
                'slot_id': '../secret',
                'fallback_instance_id': '__responses_workbench__',
            },
        )
        self.assertEqual(slot_response.status_code, 400)
        self.assertIn('invalid path segments', slot_response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
