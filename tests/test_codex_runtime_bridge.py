import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_integrations.codex.execution import (
    CodexAccessState,
    CodexAccessStatus,
    CodexDiscovery,
    CodexDiscoverySource,
    CodexExecutionInput,
    CodexExecutionResult,
    CodexExecutionState,
    CodexInputHandoff,
)
from ollmo_webserver import _resolve_ghost_auto_route, app


_OLLMO_DOWNSTREAM_EXECUTION_MARKER = '[OLLMO_DOWNSTREAM_EXECUTION_V1]'


def _discovery():
    return CodexDiscovery(
        available=True,
        source=CodexDiscoverySource.CHATGPT_APP_SYSTEM,
        executable=Path('/Applications/ChatGPT.app/Contents/Resources/codex'),
        version='codex-cli test',
    )


def _access(status=CodexAccessState.AVAILABLE):
    return CodexAccessStatus(
        status=status,
        discovery=_discovery(),
        auth_method='chatgpt' if status is CodexAccessState.AVAILABLE else None,
        exit_code=0 if status is CodexAccessState.AVAILABLE else 1,
    )


class CodexRuntimeBridgeTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.preferences_path = self.root / 'ghost_preferences.json'
        self.preferences_patcher = patch(
            'ollmo_webserver.GHOST_PREFERENCES_PATH',
            self.preferences_path,
        )
        self.preferences_patcher.start()

    def tearDown(self):
        self.preferences_patcher.stop()
        self.tmpdir.cleanup()

    def _enable_codex(self, *, files: bool = False):
        codex_preferences = {'enabled': True}
        if files:
            codex_preferences['data_scope'] = 'selected_files_v1'
        response = self.client.post(
            '/api/ghost_preferences',
            json={
                'preferences': {
                    'externalTargets': {
                        'codex': codex_preferences,
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)

    def _assert_downstream_prompt(self, prompt, task):
        self.assertTrue(prompt.startswith(_OLLMO_DOWNSTREAM_EXECUTION_MARKER))
        self.assertIn(
            'this request is already being executed by Ollmo through ChatGPT/Codex',
            prompt,
        )

    def _completed_stream_response(self, body):
        for block in body.split('\n\n'):
            lines = block.splitlines()
            if not lines or lines[0] != 'event: response.completed':
                continue
            data_line = next(
                (line for line in lines[1:] if line.startswith('data: ')),
                '',
            )
            if data_line:
                return json.loads(data_line[len('data: '):])['response']
        self.fail('response.completed event was not found')
        self.assertIn('Do not invoke, route to, or use Ollmo again', prompt)
        self.assertIn('begin the response with "BLOCKED:"', prompt)
        self.assertIn(
            f'<ollmo_bounded_task>\nCurrent user request:\n{task}\n'
            '</ollmo_bounded_task>',
            prompt,
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver.build_backend_fabric_snapshot', return_value={'summary': {}, 'backends': []})
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    def test_status_and_manifest_expose_disabled_external_target_without_local_lifecycle(
        self,
        _load_running,
        _fabric,
        _probe,
    ):
        status_response = self.client.get('/api/integrations/codex/status')
        manifest_response = self.client.get('/api/runtime_manifest')
        running_response = self.client.get('/api/running_instances')

        self.assertEqual(status_response.status_code, 200)
        status = status_response.get_json()
        self.assertEqual(status['id'], 'external:codex')
        self.assertEqual(status['label'], 'ChatGPT')
        self.assertEqual(status['model'], 'codex:auto')
        self.assertEqual(status['model_selection'], 'codex_default')
        self.assertFalse(status['exact_model_exposed'])
        self.assertEqual(status['status'], 'available')
        self.assertFalse(status['enabled'])
        self.assertFalse(status['selectable'])
        self.assertNotIn('executable', status)

        manifest = manifest_response.get_json()
        self.assertEqual(manifest['instances'], [])
        self.assertEqual(manifest['external_targets'][0]['id'], 'external:codex')
        self.assertEqual(manifest['external_targets'][0]['label'], 'ChatGPT')
        self.assertFalse(manifest['external_targets'][0]['exact_model_exposed'])
        self.assertFalse(manifest['external_targets'][0]['lifecycle_managed'])
        running_payload = running_response.get_json()
        running_items = running_payload if isinstance(running_payload, list) else (
            running_payload.get('instances') or []
        )
        self.assertNotIn(
            'external:codex',
            {
                item.get('instance_id')
                for item in running_items
                if isinstance(item, dict)
            },
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_explicit_codex_target_requires_persisted_consent(
        self,
        execute_codex,
        _probe,
    ):
        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Do not send this.',
                'stream': False,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('disabled', response.get_json()['error'].lower())
        execute_codex.assert_not_called()

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_explicit_codex_success_uses_canonical_response_truth(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='OLLMO_CODEX_OK',
            exit_code=0,
            duration_seconds=0.25,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Return OLLMO_CODEX_OK.',
                'stream': False,
                'response_id': 'resp_codex_success',
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['instance_id'], 'external:codex')
        self.assertEqual(payload['backend'], 'codex_cli')
        self.assertEqual(payload['model'], 'codex:auto')
        self.assertEqual(payload['output_text'], 'OLLMO_CODEX_OK')
        self.assertEqual(payload['outputs'][0]['value'], 'OLLMO_CODEX_OK')
        self.assertEqual(payload['lifecycle_state'], 'completed')
        self.assertEqual(payload['artifacts'], [])
        self.assertEqual(
            payload['runtime']['external_execution']['provider'],
            'codex_cli',
        )
        self.assertEqual(
            payload['runtime']['external_execution']['model_selection'],
            'codex_default',
        )
        self.assertFalse(
            payload['runtime']['external_execution']['exact_model_exposed'],
        )
        self.assertIn('response_frame', payload)
        execute_codex.assert_called_once()
        self._assert_downstream_prompt(
            execute_codex.call_args.args[0],
            'Return OLLMO_CODEX_OK.',
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_explicit_block_is_canonical_and_never_materialized(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        blocked_text = (
            'bLoCkEd: $OLLMO_HOME is unavailable in this downstream session.\n'
            'The bounded task cannot be completed safely.'
        )
        blocked_reason = (
            '$OLLMO_HOME is unavailable in this downstream session.\n'
            'The bounded task cannot be completed safely.'
        )
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text=blocked_text,
            exit_code=0,
        )

        with (
            patch(
                'ollmo_server.responses_request_runtime.phase_output_is_graph_preparation'
            ) as phase_acceptance,
            patch(
                'ollmo_server.responses_request_runtime.ResponsesRequestRuntimeOwner.'
                'attach_pre_freeze_closure_review'
            ) as pre_freeze,
            patch(
                'ollmo_server.responses_request_runtime.ResponsesRequestRuntimeOwner.'
                'apply_direct_artifact_materialization_closure'
            ) as direct_closure,
            patch('ollmo_webserver._schedule_response_late_fill') as late_fill,
        ):
            response = self.client.post(
                '/api/responses',
                json={
                    'instance_id': 'external:codex',
                    'prompt': 'Read $OLLMO_HOME and create an artifact.',
                    'stream': False,
                    'response_id': 'resp_codex_blocked',
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'completed')
        self.assertEqual(payload['lifecycle_state'], 'blocked')
        self.assertEqual(payload['output_text'], blocked_text)
        self.assertEqual(payload['artifacts'], [])
        self.assertNotIn('late_fill', payload)
        self.assertEqual(payload['outputs'][0]['status'], 'blocked')
        self.assertEqual(payload['outputs'][0]['lifecycle'], 'blocked_output')
        self.assertEqual(payload['outputs'][0]['value'], blocked_text)
        self.assertEqual(payload['outputs'][0]['blocked_reason'], blocked_reason)
        self.assertEqual(payload['surface_state']['status'], 'blocked')
        self.assertEqual(payload['surface_state']['category_counts'], {'blocked': 1})
        self.assertEqual(
            payload['runtime']['external_execution']['status'],
            'blocked',
        )
        self.assertEqual(
            payload['runtime']['external_execution']['invocation_status'],
            'completed',
        )
        self.assertEqual(
            payload['runtime']['external_execution']['blocked_reason'],
            blocked_reason,
        )
        phase_acceptance.assert_not_called()
        pre_freeze.assert_not_called()
        direct_closure.assert_not_called()
        late_fill.assert_not_called()

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_receives_ollmo_bounded_referential_context(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='CONTEXT_OK',
            exit_code=0,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'input': [
                    {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': 'Name this plan Atlas.'}],
                    },
                    {
                        'type': 'message',
                        'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': 'The plan is Atlas.'}],
                    },
                    {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': 'Continue from that answer.'}],
                    },
                ],
                'stream': False,
            },
        )

        self.assertEqual(response.status_code, 200)
        forwarded_prompt = execute_codex.call_args.args[0]
        self._assert_downstream_prompt(
            forwarded_prompt,
            'Continue from that answer.',
        )
        self.assertIn('Prior conversation context promoted by Ollmo', forwarded_prompt)
        self.assertIn('[user]\nName this plan Atlas.', forwarded_prompt)
        self.assertIn('[assistant]\nThe plan is Atlas.', forwarded_prompt)
        self.assertIn('Current user request:\nContinue from that answer.', forwarded_prompt)
        payload = response.get_json()
        self.assertEqual(
            payload['runtime']['context_strategy']['mode'],
            'recent_history',
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_marker_precedes_selected_message_reference_context(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='REFERENCE_OK',
            exit_code=0,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Use the selected reply as a reference.',
                'selected_reference_artifacts': [
                    {
                        'type': 'message',
                        'message_role': 'assistant',
                        'content': 'The bounded reference reply.',
                    }
                ],
                'stream': False,
            },
        )

        self.assertEqual(response.status_code, 200)
        execute_codex.assert_called_once()
        forwarded_prompt = execute_codex.call_args.args[0]
        self.assertTrue(forwarded_prompt.startswith(_OLLMO_DOWNSTREAM_EXECUTION_MARKER))
        self.assertIn('<ollmo_bounded_task>', forwarded_prompt)
        self.assertIn('Selected prior assistant reply reference', forwarded_prompt)
        self.assertIn('[assistant]\nThe bounded reference reply.', forwarded_prompt)
        self.assertIn(
            'Current user request:\nUse the selected reply as a reference.',
            forwarded_prompt,
        )
        self.assertTrue(forwarded_prompt.endswith('</ollmo_bounded_task>'))

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_stream_projects_the_canonical_terminal_response(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='STREAMED_CODEX_OK',
            exit_code=0,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'input': [
                    {
                        'type': 'message',
                        'role': 'user',
                        'content': [
                            {
                                'type': 'input_text',
                                'text': 'Return STREAMED_CODEX_OK.',
                            },
                        ],
                    },
                ],
                'stream': True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/event-stream')
        body = response.get_data(as_text=True)
        self.assertIn('event: response.completed', body)
        self.assertIn('STREAMED_CODEX_OK', body)
        execute_codex.assert_called_once()
        self._assert_downstream_prompt(
            execute_codex.call_args.args[0],
            'Return STREAMED_CODEX_OK.',
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_stream_preserves_explicit_block_without_late_fill(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        blocked_text = (
            'BLOCKED: $OLLMO_HOME cannot be inspected by the downstream provider.'
        )
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text=blocked_text,
            exit_code=0,
        )

        with (
            patch(
                'ollmo_server.responses_request_runtime.phase_output_is_graph_preparation'
            ) as phase_acceptance,
            patch(
                'ollmo_server.responses_request_runtime.ResponsesRequestRuntimeOwner.'
                'attach_pre_freeze_closure_review'
            ) as pre_freeze,
            patch(
                'ollmo_server.responses_request_runtime.ResponsesRequestRuntimeOwner.'
                'apply_direct_artifact_materialization_closure'
            ) as direct_closure,
            patch('ollmo_webserver._schedule_response_late_fill') as late_fill,
        ):
            response = self.client.post(
                '/api/responses',
                json={
                    'instance_id': 'external:codex',
                    'prompt': 'Inspect $OLLMO_HOME.',
                    'stream': True,
                },
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/event-stream')
        payload = self._completed_stream_response(body)
        self.assertEqual(payload['lifecycle_state'], 'blocked')
        self.assertEqual(payload['output_text'], blocked_text)
        self.assertEqual(payload['outputs'][0]['status'], 'blocked')
        self.assertEqual(payload['outputs'][0]['value'], blocked_text)
        self.assertEqual(payload.get('artifacts', []), [])
        self.assertEqual(payload['surface_state']['status'], 'blocked')
        self.assertNotIn('late_fill', payload)
        phase_acceptance.assert_not_called()
        pre_freeze.assert_not_called()
        direct_closure.assert_not_called()
        late_fill.assert_not_called()

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_timeout_is_failed_canonical_response(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.TIMED_OUT,
            discovery=_discovery(),
            exit_code=None,
            diagnostic='Codex timed out.',
            duration_seconds=5.0,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Wait forever.',
                'stream': False,
            },
        )

        self.assertEqual(response.status_code, 504)
        payload = response.get_json()
        self.assertEqual(payload['lifecycle_state'], 'failed')
        self.assertEqual(payload['error_ref']['code'], 'CODEX_TIMEOUT')
        self.assertEqual(payload['outputs'][0]['status'], 'blocked')
        self.assertIn('recovery_hint', payload)
        self.assertIn('response_frame', payload)

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_legacy_text_consent_blocks_file_before_execution(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        selected_file = self.root / 'legacy-consent.txt'
        selected_file.write_text('must remain local', encoding='utf-8')

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Describe this file.',
                'file_path': str(selected_file),
                'stream': False,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            'Enable sharing of explicitly selected files',
            response.get_json()['error'],
        )
        execute_codex.assert_not_called()

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_versioned_file_consent_allows_responses_file_handoff(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex(files=True)
        selected_file = self.root / 'selected-notes.txt'
        selected_bytes = b'current-turn notes\n'
        selected_file.write_bytes(selected_bytes)
        handoff = CodexInputHandoff(
            name=selected_file.name,
            kind='text',
            byte_size=len(selected_bytes),
            sha256=hashlib.sha256(selected_bytes).hexdigest(),
            source='local_path',
            native_image=False,
        )
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='FILE_HANDOFF_OK',
            exit_code=0,
            input_handoff=(handoff,),
        )

        persisted = self.client.get('/api/ghost_preferences').get_json()
        persisted_codex = persisted['preferences']['external_targets']['codex']
        self.assertTrue(persisted_codex['enabled'])
        self.assertTrue(persisted_codex['files_enabled'])
        self.assertEqual(persisted_codex['data_scope'], 'selected_files_v1')

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Use the selected notes.',
                'file_path': str(selected_file),
                'file_name': selected_file.name,
                'file_kind': 'text',
                'stream': False,
                'response_id': 'resp_codex_file_success',
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['output_text'], 'FILE_HANDOFF_OK')
        self.assertEqual(payload['lifecycle_state'], 'completed')
        self.assertEqual(
            payload['runtime']['external_execution']['input_handoff'],
            [handoff.as_dict()],
        )
        self.assertEqual(
            payload['runtime']['external_execution']['input_count'],
            1,
        )
        execute_codex.assert_called_once()
        self._assert_downstream_prompt(
            execute_codex.call_args.args[0],
            'Use the selected notes.',
        )
        self.assertEqual(
            execute_codex.call_args.kwargs['inputs'],
            [
                CodexExecutionInput(
                    path=str(selected_file),
                    display_name=selected_file.name,
                    kind='text',
                    source='local_path',
                )
            ],
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    def test_codex_accepts_multiple_explicit_local_paths_as_one_turn(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex(files=True)
        first = self.root / 'first.txt'
        second = self.root / 'second.pdf'
        first.write_text('first', encoding='utf-8')
        second.write_bytes(b'%PDF-test')
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='MULTI_FILE_OK',
            exit_code=0,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'instance_id': 'external:codex',
                'prompt': 'Use both selected files.',
                'file_paths_json': json.dumps([str(first), str(second)]),
                'stream': False,
            },
        )

        self.assertEqual(response.status_code, 200)
        execute_codex.assert_called_once()
        self._assert_downstream_prompt(
            execute_codex.call_args.args[0],
            'Use both selected files.',
        )
        self.assertEqual(
            execute_codex.call_args.kwargs['inputs'],
            [
                CodexExecutionInput(
                    path=str(first),
                    display_name=first.name,
                    source='local_path',
                ),
                CodexExecutionInput(
                    path=str(second),
                    display_name=second.name,
                    source='local_path',
                ),
            ],
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver.read_events', return_value=[])
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_ghost_can_select_enabled_codex_without_inventing_local_instance(
        self,
        _merge,
        _load,
        _events,
        _probe,
    ):
        self._enable_codex()
        payload = {
            'prompt': 'Explain this in one sentence.',
            'ghost_route': True,
            'ghost_preferences': {
                'primary_mode': 'lock',
                'primary_target': {
                    'model': 'codex:auto',
                    'backend': 'codex_cli',
                    'capability': 'chat',
                },
            },
        }

        with app.test_request_context(
            '/api/ghost_route_preview',
            method='POST',
            json=payload,
        ):
            route_info, error = _resolve_ghost_auto_route(payload)

        self.assertIsNone(error)
        self.assertEqual(route_info['instance_id'], 'external:codex')
        self.assertEqual(route_info['instance']['target_kind'], 'external')
        self.assertFalse(route_info['instance']['lifecycle_managed'])

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    @patch('ollmo_webserver.read_events', return_value=[])
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_ghost_executes_enabled_codex_through_canonical_responses(
        self,
        _merge,
        _load,
        _events,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='GHOST_CODEX_OK',
            exit_code=0,
        )

        response = self.client.post(
            '/api/responses',
            json={
                'prompt': 'Return GHOST_CODEX_OK.',
                'ghost_route': True,
                'stream': False,
                'ghost_preferences': {
                    'primary_mode': 'lock',
                    'primary_target': {
                        'model': 'codex:auto',
                        'backend': 'codex_cli',
                        'capability': 'chat',
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['instance_id'], 'external:codex')
        self.assertEqual(payload['output_text'], 'GHOST_CODEX_OK')
        self.assertEqual(payload['route_source'], 'ghost_carried')
        self.assertEqual(payload['lifecycle_state'], 'completed')

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    @patch('ollmo_webserver.read_events', return_value=[])
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_ghost_executes_versioned_file_turn_through_codex(
        self,
        _merge,
        _load,
        _events,
        execute_codex,
        _probe,
    ):
        self._enable_codex(files=True)
        selected_file = self.root / 'ghost-selected.txt'
        selected_file.write_text('ghost file context', encoding='utf-8')
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text='GHOST_FILE_OK',
            exit_code=0,
            input_handoff=(
                CodexInputHandoff(
                    name=selected_file.name,
                    kind='text',
                    byte_size=selected_file.stat().st_size,
                    sha256=hashlib.sha256(selected_file.read_bytes()).hexdigest(),
                    source='local_path',
                ),
            ),
        )

        response = self.client.post(
            '/api/responses',
            json={
                'prompt': 'Summarize the selected text file in one sentence.',
                'file_path': str(selected_file),
                'file_name': selected_file.name,
                'file_kind': 'text',
                'ghost_route': True,
                'stream': False,
                'ghost_preferences': {
                    'primary_mode': 'lock',
                    'primary_target': {
                        'model': 'codex:auto',
                        'backend': 'codex_cli',
                        'capability': 'chat',
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['instance_id'], 'external:codex')
        self.assertEqual(payload['output_text'], 'GHOST_FILE_OK')
        self.assertEqual(payload['route_source'], 'ghost_carried')
        self.assertEqual(payload['lifecycle_state'], 'completed')
        execute_codex.assert_called_once()
        self._assert_downstream_prompt(
            execute_codex.call_args.args[0],
            'Summarize the selected text file in one sentence.',
        )
        self.assertEqual(
            execute_codex.call_args.kwargs['inputs'],
            [
                CodexExecutionInput(
                    path=str(selected_file),
                    display_name=selected_file.name,
                    kind='text',
                    source='local_path',
                )
            ],
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_ghost_does_not_offer_codex_for_file_input_without_file_consent(
        self,
        _merge,
        _load,
        _probe,
    ):
        self._enable_codex()
        payload = {
            'prompt': 'Describe this file.',
            'file_path': '/tmp/private.txt',
            'ghost_route': True,
        }

        with app.test_request_context(
            '/api/ghost_route_preview',
            method='POST',
            json=payload,
        ):
            route_info, error = _resolve_ghost_auto_route(payload)

        self.assertIsNone(route_info)
        self.assertIn('No running instances', error)


if __name__ == '__main__':
    unittest.main()
