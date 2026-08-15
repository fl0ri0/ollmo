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
from ollmo_webserver import (
    _LATE_FILL_RUNTIME,
    _complete_response_late_fill,
    _prepare_late_fill_branch_plan,
    _RESPONSE_LATE_FILL_IN_FLIGHT,
    _RESPONSE_LOOKUP,
    _resolve_ghost_auto_route,
    app,
)
from ollmo_services.chat_history import read_chat_history


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
        self.runtime_registry_path = self.root / 'model_ports.json'
        self.runtime_status_path = self.root / 'runtime_status.json'
        self.response_frames_dir = self.root / 'response_frames'
        self.chat_history_dir = self.root / 'chat_history'
        self.preferences_path = self.root / 'ghost_preferences.json'
        self.runtime_registry_path.write_text('[]\n', encoding='utf-8')
        self.runtime_status_path.write_text(
            json.dumps({'schema_version': 1, 'updated_at': 'test', 'instances': {}}),
            encoding='utf-8',
        )
        self.response_frames_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)

        def read_isolated_chat_history(instance_id, *args, **kwargs):
            if kwargs.get('history_dir') is None:
                kwargs['history_dir'] = self.chat_history_dir
            return read_chat_history(instance_id, *args, **kwargs)

        self.runtime_patchers = [
            patch('ollmo_webserver.CONFIG_FILE_NAME', str(self.runtime_registry_path)),
            patch('ollmo_webserver.RUNTIME_STATUS_PATH', self.runtime_status_path),
            patch('ollmo_webserver.RESPONSE_FRAMES_DIR', self.response_frames_dir),
            patch('ollmo_webserver.CHAT_HISTORY_DIR', self.chat_history_dir),
            patch('ollmo_webserver.GHOST_PREFERENCES_PATH', self.preferences_path),
            patch('ollmo_g.router.read_chat_history', side_effect=read_isolated_chat_history),
            patch(
                'ollmo_server.ghost_route_runtime.read_chat_history',
                side_effect=read_isolated_chat_history,
            ),
            patch('ollmo_core.registry.DEFAULT_REGISTRY_PATH', self.runtime_registry_path),
            patch('ollmo_core.status.DEFAULT_RUNTIME_STATUS_PATH', self.runtime_status_path),
        ]
        for patcher in self.runtime_patchers:
            patcher.start()
        _RESPONSE_LOOKUP.clear()
        _RESPONSE_LATE_FILL_IN_FLIGHT.clear()

    def tearDown(self):
        _RESPONSE_LOOKUP.clear()
        _RESPONSE_LATE_FILL_IN_FLIGHT.clear()
        for patcher in reversed(self.runtime_patchers):
            patcher.stop()
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
        self.assertEqual(prompt.count(_OLLMO_DOWNSTREAM_EXECUTION_MARKER), 1)
        self.assertIn(
            'this request is already being executed by Ollmo through ChatGPT/Codex',
            prompt,
        )
        self.assertIn('Do not invoke, route to, or use Ollmo again', prompt)
        self.assertIn('begin the response with "BLOCKED:"', prompt)
        self.assertEqual(prompt.count('<ollmo_bounded_task>'), 1)
        self.assertEqual(prompt.count('</ollmo_bounded_task>'), 1)
        self.assertIn(
            f'<ollmo_bounded_task>\nCurrent user request:\n{task}\n'
            '</ollmo_bounded_task>',
            prompt,
        )

    def _bounded_task(self, prompt):
        start = prompt.index('<ollmo_bounded_task>') + len('<ollmo_bounded_task>')
        end = prompt.index('</ollmo_bounded_task>', start)
        return prompt[start:end].strip()

    def _assert_prepare_only_downstream_prompt(self, forwarded_prompt, root_prompt):
        bounded_task = self._bounded_task(forwarded_prompt)
        self.assertEqual(forwarded_prompt.count(_OLLMO_DOWNSTREAM_EXECUTION_MARKER), 1)
        self.assertEqual(forwarded_prompt.count('<ollmo_bounded_task>'), 1)
        self.assertIn('<ollmo_promoted_context>', forwarded_prompt)
        self.assertIn(root_prompt, forwarded_prompt)
        self.assertNotIn(root_prompt, bounded_task)
        self.assertIn('Ollmo phase contract: prepare-only.', bounded_task)
        self.assertIn('only the current text-preparation phase', bounded_task)
        self.assertIn('Do not perform filesystem writes', bounded_task)
        self.assertIn('Ollmo retains authority for downstream execution', bounded_task)
        self.assertNotIn('Correct errors directly in the files.', bounded_task)

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

    @staticmethod
    def _multi_phase_bundle_prompt():
        return (
            'Create a premium local landing page for Mon Repos. Create exactly three '
            'images and two files: a lakeside villa exterior, two complementary '
            'interior shots, index.html, and styles.css. All image paths must '
            'correctly point to the locally stored images at the end. Correct errors '
            'directly in the files.'
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
    def test_explicit_codex_artifact_requests_use_prepare_only_bounded_task(
        self,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        cases = [
            (
                'mixed_bundle',
                self._multi_phase_bundle_prompt(),
                (
                    '1. A lakeside villa at blue hour.\n'
                    '2. A warm salon with lake light.\n'
                    '3. Carved stone in soft raking light.'
                ),
                'Downstream phases will handle image generation and chat separately',
                5,
            ),
            (
                'file_only',
                'Create index.html with a concise premium landing page and save it '
                'as a local artifact.',
                '<main><h1>Mon Repos</h1><p>Stillness beside the lake.</p></main>',
                'Downstream phases will handle chat separately',
                1,
            ),
        ]

        with patch('ollmo_webserver._schedule_response_late_fill') as late_fill:
            for label, prompt, output_text, downstream_clause, pending_count in cases:
                with self.subTest(label=label):
                    execute_codex.reset_mock()
                    late_fill.reset_mock()
                    execute_codex.return_value = CodexExecutionResult(
                        status=CodexExecutionState.COMPLETED,
                        discovery=_discovery(),
                        output_text=output_text,
                        exit_code=0,
                    )
                    response = self.client.post(
                        '/api/responses',
                        json={
                            'instance_id': 'external:codex',
                            'prompt': prompt,
                            'stream': False,
                            'response_id': f'resp_explicit_codex_prepare_{label}',
                        },
                    )

                    self.assertEqual(response.status_code, 200)
                    execute_codex.assert_called_once()
                    self.assertNotIn('inputs', execute_codex.call_args.kwargs)
                    self._assert_prepare_only_downstream_prompt(
                        execute_codex.call_args.args[0],
                        prompt,
                    )
                    self.assertIn(
                        downstream_clause,
                        self._bounded_task(execute_codex.call_args.args[0]),
                    )
                    late_fill.assert_called_once()
                    self.assertEqual(
                        len(
                            late_fill.call_args.kwargs['artifact_gap'].get(
                                'pending_branches'
                            )
                            or []
                        ),
                        pending_count,
                    )
                    self.assertEqual(
                        response.get_json()['lifecycle_state'],
                        'late_fill_pending',
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
        self.assertIn('<ollmo_promoted_context>', forwarded_prompt)
        self.assertNotIn(
            'The bounded reference reply.',
            self._bounded_task(forwarded_prompt),
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

    @patch('ollmo_webserver._execute_codex_external_text')
    def test_graph_owned_external_chat_phase_is_bounded_and_preserves_block_truth(
        self,
        execute_codex,
    ):
        root_prompt = (
            'Create a premium local two-page site and save index.html, '
            'configurator.html, and styles.css.'
        )
        execution_contract = {
            'kind': 'ollmo.execution_contract',
            'branch_id': 'branch-chat-file-1',
            'phase_id': 'phase-2',
            'capability': 'chat',
            'output_type': 'text_artifact',
        }
        target = {
            'id': 'external:codex',
            'instance_id': 'external:codex',
            'target_kind': 'external',
            'backend': 'codex_cli',
            'model': 'codex:auto',
            'capability': 'chat',
            'supported_capabilities': ['chat'],
            'available': True,
            'enabled': True,
            'selectable': True,
            'readiness': 'ready',
            'activity': 'idle',
            'files_enabled': False,
        }
        with (
            patch(
                'ollmo_webserver._external_targets_payload',
                return_value=[target],
            ),
            app.test_request_context('/api/responses', method='POST'),
        ):
            plan = _prepare_late_fill_branch_plan(
                expected_capability='chat',
                artifact_gap={
                    'trigger': 'execution_planner_deferred_follow_up',
                    'stage_direction': 'materialize_requested_text_artifact',
                    'requires_artifact': True,
                    'branch_id': 'branch-chat-file-1',
                    'phase_id': 'phase-2',
                    'text_artifact_extension': 'html',
                    'text_artifact_source_name': 'index',
                    'text_artifact_target_path': str(self.root / 'index.html'),
                    'text_artifact_requests': [
                        {
                            'extension': 'html',
                            'source_name': 'index',
                            'target_path': str(self.root / 'index.html'),
                        },
                        {
                            'extension': 'html',
                            'source_name': 'configurator',
                            'target_path': str(
                                self.root / 'configurator.html'
                            ),
                        },
                        {
                            'extension': 'css',
                            'source_name': 'styles',
                            'target_path': str(self.root / 'styles.css'),
                        },
                    ],
                    'artifact_request': {
                        'extension': 'html',
                        'source_name': 'index',
                        'target_path': str(self.root / 'index.html'),
                    },
                    'execution_contract': execution_contract,
                },
                current_payload={},
                request_payload={
                    'ghost_route': True,
                    'prompt': root_prompt,
                },
                assistant_message='The preparation phase is complete.',
                source_route_payload={
                    'instance_id': 'external:codex',
                    'instance': target,
                    'route_source': 'ghost_carried',
                    'route_runtime': {
                        'external_execution': {'status': 'completed'},
                    },
                },
                failed_instance_id=None,
            )
        bounded_task = plan['external_chat_phase']['bounded_task_prompt']
        self.assertNotIn(root_prompt, bounded_task)
        self.assertIn('Write the complete file payloads', bounded_task)
        self.assertEqual(
            plan['route_info']['route_runtime']['selection_policy'],
            'selected_external_provider_for_graph_chat_phase',
        )

        revision_path = self.root / 'revision.html'
        revision_source = '<!DOCTYPE html><html><body>Original</body></html>'
        revision_path.write_text(revision_source, encoding='utf-8')
        revision_root_prompt = (
            'Update revision.html so the heading reads Stillwater; then create '
            'an image for a later branch.'
        )
        revision_contract = {
            **execution_contract,
            'branch_id': 'branch-chat-revision-1',
            'phase_id': 'phase-3',
        }
        with (
            patch(
                'ollmo_webserver._external_targets_payload',
                return_value=[target],
            ),
            app.test_request_context('/api/responses', method='POST'),
        ):
            revision_plan = _prepare_late_fill_branch_plan(
                expected_capability='chat',
                artifact_gap={
                    'trigger': 'execution_planner_deferred_follow_up',
                    'stage_direction': 'materialize_requested_text_artifact',
                    'requires_artifact': True,
                    'branch_id': 'branch-chat-revision-1',
                    'phase_id': 'phase-3',
                    'text_artifact_extension': 'html',
                    'text_artifact_source_name': 'revision',
                    'text_artifact_target_path': str(revision_path),
                    'text_artifact_revision_required': True,
                    'text_artifact_revision_preservation_required': True,
                    'content_payload': revision_source,
                    'content_payload_source': (
                        'canonical_predecessor_text_artifact_snapshot'
                    ),
                    'artifact_request': {
                        'extension': 'html',
                        'source_name': 'revision',
                        'target_path': str(revision_path),
                    },
                    'execution_contract': revision_contract,
                },
                current_payload={},
                request_payload={
                    'ghost_route': True,
                    'prompt': revision_root_prompt,
                },
                assistant_message='The edit delta is prepared.',
                source_route_payload={
                    'instance_id': 'external:codex',
                    'instance': target,
                    'route_source': 'ghost_carried',
                    'route_runtime': {
                        'external_execution': {'status': 'completed'},
                    },
                },
                failed_instance_id=None,
            )
        revision_task = revision_plan['external_chat_phase'][
            'bounded_task_prompt'
        ]
        self.assertNotIn(revision_root_prompt, revision_task)
        self.assertIn('Ollmo promoted context', revision_task)
        self.assertIn(revision_source, revision_task)

        cases = (
            (
                '```html\n<!DOCTYPE html><html><body>Stillwater</body></html>\n```\n'
                '```html\n<!DOCTYPE html><html><body>Configurator</body></html>\n```\n'
                '```css\nbody { color: #223344; }\n```',
                False,
            ),
            ('BLOCKED: The bounded file body cannot be completed.', True),
        )
        with patch.object(
            _LATE_FILL_RUNTIME,
            'invoke_internal_api_json_route',
            side_effect=AssertionError('external chat phase used local /api/infer'),
        ):
            for provider_output, blocked in cases:
                with self.subTest(blocked=blocked):
                    execute_codex.reset_mock()
                    execute_codex.return_value = CodexExecutionResult(
                        status=CodexExecutionState.COMPLETED,
                        discovery=_discovery(),
                        output_text=provider_output,
                        exit_code=0,
                    )

                    result = _LATE_FILL_RUNTIME.execute_prepared_late_fill_branch(
                        plan
                    )

                    execute_codex.assert_called_once()
                    forwarded_prompt = execute_codex.call_args.args[0]
                    self.assertEqual(
                        forwarded_prompt.count(_OLLMO_DOWNSTREAM_EXECUTION_MARKER),
                        1,
                    )
                    self.assertIn('<ollmo_promoted_context>', forwarded_prompt)
                    self.assertIn(root_prompt, forwarded_prompt)
                    self.assertEqual(
                        self._bounded_task(forwarded_prompt),
                        bounded_task,
                    )
                    self.assertNotIn(root_prompt, self._bounded_task(forwarded_prompt))
                    if blocked:
                        self.assertEqual(
                            result['infer_result']['external_provider_block'][
                                'code'
                            ],
                            'EXTERNAL_PROVIDER_BLOCKED',
                        )
                        self.assertNotIn('output_text', result['infer_result'])
                        self.assertNotIn('content', result['infer_result'])
                        self.assertEqual(
                            (self.root / 'index.html').read_text(
                                encoding='utf-8'
                            ),
                            '<!DOCTYPE html><html><body>Stillwater</body></html>',
                        )
                        self.assertEqual(
                            (self.root / 'configurator.html').read_text(
                                encoding='utf-8'
                            ),
                            '<!DOCTYPE html><html><body>Configurator</body></html>',
                        )
                        self.assertEqual(
                            (self.root / 'styles.css').read_text(
                                encoding='utf-8'
                            ),
                            'body { color: #223344; }',
                        )
                    else:
                        self.assertEqual(
                            result['infer_result']['output_text'],
                            provider_output,
                        )
                        self.assertEqual(
                            result['infer_result']['saved_text_path'],
                            str(self.root / 'index.html'),
                        )
                        self.assertEqual(
                            {
                                str(
                                    item.get('path')
                                    or item.get('saved_text_path')
                                    or ''
                                )
                                for item in result['infer_result'][
                                    'saved_text_artifacts'
                                ]
                            },
                            {
                                str(self.root / 'index.html'),
                                str(self.root / 'configurator.html'),
                                str(self.root / 'styles.css'),
                            },
                        )
                        self.assertEqual(
                            (self.root / 'index.html').read_text(
                                encoding='utf-8'
                            ),
                            '<!DOCTYPE html><html><body>Stillwater</body></html>',
                        )
                        self.assertEqual(
                            (self.root / 'configurator.html').read_text(
                                encoding='utf-8'
                            ),
                            '<!DOCTYPE html><html><body>Configurator</body></html>',
                        )
                        self.assertEqual(
                            (self.root / 'styles.css').read_text(
                                encoding='utf-8'
                            ),
                            'body { color: #223344; }',
                        )
                        self.assertEqual(
                            result['infer_result']['execution_contract'][
                                'branch_id'
                            ],
                            execution_contract['branch_id'],
                        )
                        self.assertEqual(
                            result['infer_result']['execution_contract'][
                                'phase_id'
                            ],
                            execution_contract['phase_id'],
                        )
                    self.assertEqual(
                        result['route_info']['route_runtime'][
                            'external_execution'
                        ]['target_id'],
                        'external:codex',
                    )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    @patch('ollmo_webserver.read_events', return_value=[])
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_ghost_codex_executes_only_graph_owned_preparation_phase(
        self,
        _merge,
        _load,
        _events,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        prompt = self._multi_phase_bundle_prompt()
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text=(
                '1. A wide cinematic view of the lakeside villa at blue hour.\n'
                '2. A warm salon interior with carved timber and quiet lake light.\n'
                '3. A macro study of hand-carved local stone in soft raking light.'
            ),
            exit_code=0,
        )

        with patch('ollmo_webserver._schedule_response_late_fill') as late_fill:
            response = self.client.post(
                '/api/responses',
                json={
                    'prompt': prompt,
                    'ghost_route': True,
                    'stream': False,
                    'response_id': 'resp_codex_prepare_only',
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
        execute_codex.assert_called_once()
        self.assertNotIn('inputs', execute_codex.call_args.kwargs)
        forwarded_prompt = execute_codex.call_args.args[0]
        self._assert_prepare_only_downstream_prompt(forwarded_prompt, prompt)
        late_fill.assert_called_once()

        payload = response.get_json()
        self.assertEqual(payload['instance_id'], 'external:codex')
        self.assertEqual(payload['lifecycle_state'], 'late_fill_pending')
        phase_graph = late_fill.call_args.kwargs['response_payload']['runtime'][
            'request_phase_graph'
        ]
        self.assertEqual(phase_graph['current_phase_id'], 'phase-1')
        self.assertEqual(phase_graph['current_phase_capability'], 'chat')
        self.assertEqual(
            [
                branch['capability']
                for branch in phase_graph['downstream_branches']
            ].count('image_generation'),
            3,
        )

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    @patch('ollmo_webserver.read_events', return_value=[])
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_pre_freeze_codex_materializes_three_text_artifacts_in_one_wave(
        self,
        _merge,
        _load,
        _events,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        response_id = 'resp_codex_pre_freeze_three_text_artifacts'
        root_prompt = (
            'Create a premium local two-page site and save index.html, '
            'configurator.html, and styles.css.'
        )
        index_body = (
            '<!doctype html><html><head>'
            '<link rel="stylesheet" href="styles.css">'
            '</head><body><main><h1>Atelier</h1>'
            '<a href="configurator.html">Configure</a>'
            '</main></body></html>'
        )
        styles_body = 'body { color: #112233; }'
        configurator_body = (
            '<!doctype html><html><head>'
            '<link rel="stylesheet" href="styles.css">'
            '</head><body><main><h1>Configurator</h1>'
            '<a href="index.html">Atelier</a>'
            '</main></body></html>'
        )
        execute_codex.side_effect = [
            CodexExecutionResult(
                status=CodexExecutionState.COMPLETED,
                discovery=_discovery(),
                output_text='The requested file structure is prepared for materialization.',
                exit_code=0,
            ),
            CodexExecutionResult(
                status=CodexExecutionState.COMPLETED,
                discovery=_discovery(),
                output_text=f'```html\n{index_body}\n```',
                exit_code=0,
            ),
            CodexExecutionResult(
                status=CodexExecutionState.COMPLETED,
                discovery=_discovery(),
                output_text=(
                    f'```html\n{index_body}\n```\n'
                    f'```css\n{styles_body}\n```\n'
                    f'```html\n{configurator_body}\n```'
                ),
                exit_code=0,
            ),
        ]

        with patch('ollmo_webserver._schedule_response_late_fill') as schedule:
            response = self.client.post(
                '/api/responses',
                json={
                    'instance_id': 'external:codex',
                    'prompt': root_prompt,
                    'stream': False,
                    'response_id': response_id,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['lifecycle_state'], 'late_fill_pending')
        execute_codex.assert_called_once()
        schedule.assert_called_once()
        scheduled = dict(schedule.call_args.kwargs)
        artifact_gap = scheduled['artifact_gap']
        self.assertEqual(artifact_gap['trigger'], 'pre_freeze_closure_review')
        pending_branches = artifact_gap['pending_branches']
        self.assertEqual(len(pending_branches), 3)
        original_branch_ids = {
            str(branch.get('branch_id') or '') for branch in pending_branches
        }
        self.assertEqual(len(original_branch_ids), 3)
        self.assertEqual(
            {
                (
                    str(branch.get('text_artifact_extension') or ''),
                    str(branch.get('text_artifact_source_name') or ''),
                )
                for branch in pending_branches
            },
            {
                ('html', 'index'),
                ('html', 'configurator'),
                ('css', 'styles'),
            },
        )
        self.assertTrue(
            all(
                branch.get('capability') == 'chat'
                and branch.get('stage_direction')
                == 'materialize_requested_text_artifact'
                and branch.get('requires_artifact') is True
                and not str(branch.get('text_artifact_target_path') or '').strip()
                and not str(
                    ((branch.get('artifact_request') or {}).get('target_path'))
                    or ''
                ).strip()
                for branch in pending_branches
            )
        )
        scheduled_runtime = scheduled['response_payload']['runtime']
        self.assertEqual(
            scheduled_runtime['external_execution']['target_id'],
            'external:codex',
        )
        self.assertEqual(
            scheduled_runtime['external_execution']['status'],
            'completed',
        )
        self.assertEqual(
            scheduled['response_payload']['working_frame']['target'][
                'instance_id'
            ],
            'external:codex',
        )

        documents_dir = self.root / 'documents'
        original_prepare = _LATE_FILL_RUNTIME.prepare_late_fill_branch_plan
        prepared_calls = []
        prepared_plans = []

        def prepare_branch_plan(**kwargs):
            prepared_calls.append(dict(kwargs))
            plan = original_prepare(**kwargs)
            prepared_plans.append(plan)
            return plan

        with (
            patch(
                'ollmo_server.late_fill_runtime.ARTIFACT_OUTPUTS_DOCUMENTS_DIR',
                documents_dir,
            ),
            patch.object(
                _LATE_FILL_RUNTIME,
                'load_running_instances',
                return_value=[],
            ),
            patch.object(
                _LATE_FILL_RUNTIME,
                'merge_instances_with_runtime_status',
                return_value=[],
            ),
            patch.object(
                _LATE_FILL_RUNTIME,
                'invoke_internal_api_json_route',
                side_effect=AssertionError(
                    'external chat materialization used local /api/infer'
                ),
            ) as local_infer,
            patch.object(
                _LATE_FILL_RUNTIME,
                'prepare_late_fill_branch_plan',
                side_effect=prepare_branch_plan,
            ) as prepare_branch,
        ):
            _complete_response_late_fill(**scheduled)

        self.assertEqual(prepare_branch.call_count, 2)
        prepare_call = prepared_calls[0]
        self.assertTrue(
            prepare_call['artifact_gap']['coalesced_text_artifact_wave']
        )
        self.assertEqual(
            {
                (
                    str(item.get('extension') or ''),
                    str(item.get('source_name') or ''),
                )
                for item in prepare_call['artifact_gap'][
                    'text_artifact_requests'
                ]
            },
            {
                ('html', 'index'),
                ('html', 'configurator'),
                ('css', 'styles'),
            },
        )
        self.assertIsNone(prepare_call['failed_instance_id'])
        self.assertEqual(prepare_call['excluded_instance_ids'], [])
        self.assertEqual(
            prepared_plans[0]['route_info']['route_runtime'][
                'selection_policy'
            ],
            'selected_external_provider_for_graph_chat_phase',
        )
        retry_prepare_call = prepared_calls[1]
        retry_recovery = retry_prepare_call['artifact_gap'][
            'coalesced_text_artifact_recovery'
        ]
        self.assertEqual(retry_recovery['attempt_number'], 2)
        self.assertEqual(retry_recovery['maximum_attempts'], 2)
        self.assertEqual(
            set(retry_recovery['member_branch_ids']),
            original_branch_ids,
        )
        self.assertEqual(
            prepared_plans[1]['route_info']['route_runtime'][
                'selection_policy'
            ],
            'selected_external_provider_for_graph_chat_phase',
        )
        local_infer.assert_not_called()
        self.assertEqual(execute_codex.call_count, 3)
        materialization_prompt = execute_codex.call_args_list[2].args[0]
        self.assertEqual(
            materialization_prompt.count(_OLLMO_DOWNSTREAM_EXECUTION_MARKER),
            1,
        )
        bounded_task = self._bounded_task(materialization_prompt)
        self.assertNotIn(root_prompt, bounded_task)
        self.assertIn('`index` as html', bounded_task)
        self.assertIn('`configurator` as html', bounded_task)
        self.assertIn('`styles` as css', bounded_task)
        self.assertIn('bounded complete-set recovery attempt', bounded_task)

        final_payload = _RESPONSE_LOOKUP[response_id]['response_payload']
        late_fill = final_payload['late_fill']
        self.assertEqual(
            late_fill['status'],
            'completed',
            msg=json.dumps(
                {
                    key: late_fill.get(key)
                    for key in (
                        'status',
                        'error',
                        'pending_branches',
                        'completed_branches',
                        'failed_branches',
                        'fill_results',
                    )
                },
                indent=2,
                sort_keys=True,
            ),
        )
        self.assertEqual(
            {
                str(branch.get('branch_id') or '')
                for branch in late_fill['completed_branches']
            },
            original_branch_ids,
        )
        self.assertEqual(
            {
                str(result.get('branch_id') or '')
                for result in late_fill['fill_results']
            },
            original_branch_ids,
        )
        self.assertFalse(
            any(
                str(record.get('branch_id') or '').startswith(
                    'coalesced-text-artifacts-'
                )
                for record in [
                    *late_fill['completed_branches'],
                    *late_fill['fill_results'],
                ]
            )
        )
        expected_content = {
            ('html', 'index'): (
                '<!doctype html><html><head>'
                '<link rel="stylesheet" href="styles.css">'
                '</head><body><main><h1>Atelier</h1>'
                '<a href="configurator.html">Configure</a>'
                '</main></body></html>'
            ),
            ('html', 'configurator'): (
                '<!doctype html><html><head>'
                '<link rel="stylesheet" href="styles.css">'
                '</head><body><main><h1>Configurator</h1>'
                '<a href="index.html">Atelier</a>'
                '</main></body></html>'
            ),
            ('css', 'styles'): styles_body,
        }
        saved_paths = set()
        for fill_result in late_fill['fill_results']:
            self.assertEqual(fill_result['fill_instance_id'], 'external:codex')
            self.assertEqual(
                fill_result['selection_policy'],
                'selected_external_provider_for_graph_chat_phase',
            )
            identity = (
                str(fill_result.get('text_artifact_extension') or ''),
                str(fill_result.get('text_artifact_source_name') or ''),
            )
            saved_path = Path(fill_result['saved_text_path'])
            saved_paths.add(saved_path)
            self.assertTrue(
                saved_path.resolve().is_relative_to(documents_dir.resolve())
            )
            self.assertEqual(saved_path.suffix, f'.{identity[0]}')
            self.assertEqual(
                saved_path.read_text(encoding='utf-8').strip(),
                expected_content[identity],
            )
        self.assertEqual(len(saved_paths), 3)
        self.assertEqual(
            {path.name for path in saved_paths},
            {'index.html', 'configurator.html', 'styles.css'},
        )
        self.assertEqual(len({path.parent for path in saved_paths}), 1)
        bundle_directories = [
            path for path in documents_dir.iterdir() if path.is_dir()
        ]
        self.assertEqual(len(bundle_directories), 1)
        self.assertEqual(
            {path.name for path in bundle_directories[0].iterdir()},
            {'index.html', 'configurator.html', 'styles.css'},
        )
        cohort_history = late_fill[
            'coalesced_text_artifact_recovery_history'
        ]
        self.assertEqual(len(cohort_history), 1)
        self.assertEqual(cohort_history[0]['status'], 'completed')
        self.assertEqual(
            set(cohort_history[0]['member_branch_ids']),
            original_branch_ids,
        )

    def test_external_text_bundle_path_collision_fails_before_writes(self):
        documents_dir = self.root / 'documents'
        plan = {
            'branch_id': 'coalesced-collision-check',
            'instance': {
                'instance_id': 'external:codex',
                'model': 'codex:auto',
                'target_kind': 'external',
            },
            'branch': {
                'branch_id': 'coalesced-collision-check',
                'capability': 'chat',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
            },
            'effective_data': {
                'stage_direction': 'materialize_requested_text_artifact',
                'text_artifact_requests': [
                    {'extension': 'html', 'source_name': 'a b'},
                    {'extension': 'html', 'source_name': 'a_b'},
                ],
            },
        }
        provider_result = {
            'output_text': (
                '```html\n<!doctype html><html><body>A</body></html>\n```\n'
                '```html\n<!doctype html><html><body>B</body></html>\n```'
            )
        }
        with patch(
            'ollmo_server.late_fill_runtime.ARTIFACT_OUTPUTS_DOCUMENTS_DIR',
            documents_dir,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'TEXT_ARTIFACT_PATH_COLLISION',
            ):
                _LATE_FILL_RUNTIME._materialize_external_chat_text_artifact_outputs(
                    plan,
                    provider_result,
                )
        self.assertFalse(documents_dir.exists())

    @patch(
        'ollmo_integrations.codex.runtime_target.probe_codex_access',
        return_value=_access(),
    )
    @patch('ollmo_webserver._execute_codex_external_text')
    @patch('ollmo_webserver.read_events', return_value=[])
    @patch('ollmo_webserver.load_running_instances', return_value=[])
    @patch('ollmo_webserver.merge_instances_with_runtime_status', return_value=[])
    def test_ghost_codex_prepare_block_remains_canonical_without_downstream_work(
        self,
        _merge,
        _load,
        _events,
        execute_codex,
        _probe,
    ):
        self._enable_codex()
        prompt = self._multi_phase_bundle_prompt()
        blocked_text = 'BLOCKED: workspace is intentionally unavailable.'
        execute_codex.return_value = CodexExecutionResult(
            status=CodexExecutionState.COMPLETED,
            discovery=_discovery(),
            output_text=blocked_text,
            exit_code=0,
        )

        with (
            patch('ollmo_webserver._schedule_response_late_fill') as late_fill,
            patch(
                'ollmo_server.responses_request_runtime.ResponsesRequestRuntimeOwner.'
                'apply_direct_artifact_materialization_closure'
            ) as direct_closure,
        ):
            response = self.client.post(
                '/api/responses',
                json={
                    'prompt': prompt,
                    'ghost_route': True,
                    'stream': False,
                    'response_id': 'resp_codex_prepare_blocked',
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
        bounded_task = self._bounded_task(execute_codex.call_args.args[0])
        self.assertIn('Ollmo phase contract: prepare-only.', bounded_task)
        payload = response.get_json()
        self.assertEqual(payload['lifecycle_state'], 'blocked')
        self.assertEqual(payload['surface_state']['status'], 'blocked')
        self.assertEqual(payload.get('artifacts', []), [])
        self.assertEqual(len(payload['outputs']), 1)
        self.assertEqual(payload['outputs'][0]['status'], 'blocked')
        self.assertEqual(payload['outputs'][0]['value'], blocked_text)
        self.assertNotIn('late_fill', payload)
        late_fill.assert_not_called()
        direct_closure.assert_not_called()

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
