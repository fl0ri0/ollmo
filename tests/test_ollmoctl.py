import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from helpers import ollmoctl


class OllmoctlTests(unittest.TestCase):
    class _FakeResponse:
        def __init__(self, payload):
            self._body = json.dumps(payload).encode('utf-8')
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._body

    @staticmethod
    def _graph_rebase_truth(
        *,
        response_id='resp-rebase-cli',
        frame_id='resp-rebase-cli:frame-1',
        frame_sequence=1,
        requested_class='partial_subtree_rebase',
        proposal_count=1,
        no_proposal_candidate=False,
    ):
        graph = {
            'kind': 'ollmo.request_phase_graph',
            'response_id': response_id,
            'phases': [
                {
                    'phase_id': 'phase-root',
                    'branch_id': 'branch-root',
                    'capability': 'chat',
                    'output_type': 'text',
                }
            ],
            'downstream_branches': [],
            'intent_obligations': [
                {'obligation_id': 'intent-root', 'phase_id': 'phase-root'}
            ],
            'output_obligations': [
                {'obligation_id': 'output-root', 'phase_id': 'phase-root'}
            ],
            'redraw_scope_ladder_review': {
                'selected_scope': requested_class,
            },
        }
        base_digest = ollmoctl.stable_graph_digest(graph)
        proposals = []
        reviews = []
        candidate_digest = ''
        for index in range(proposal_count):
            candidate = copy.deepcopy(graph)
            candidate.pop('redraw_scope_ladder_review', None)
            candidate['phases'].append(
                {
                    'phase_id': f'phase-replacement-{index}',
                    'branch_id': f'branch-replacement-{index}',
                    'capability': 'chat',
                    'output_type': 'text',
                    'depends_on': ['phase-root'],
                }
            )
            candidate_digest = ollmoctl.stable_graph_digest(candidate)
            proposal_id = f'proposal-cli-{index}'
            proposal = {
                'kind': 'ollmo.graph_rebase_proposal',
                'proposal_id': proposal_id,
                'requested_rebase_class': requested_class,
                'base_graph_digest': base_digest,
                'candidate_graph_digest': candidate_digest,
                'candidate_graph': candidate,
                'scope_root_ids': ['phase-root'],
                'scope_phase_ids': ['phase-root', f'phase-replacement-{index}'],
                'scope_branch_ids': ['branch-root', f'branch-replacement-{index}'],
                'preserve_outside_scope': True,
            }
            proposals.append(proposal)
            reviews.append(
                {
                    'kind': 'ollmo.graph_rebase_review',
                    'review_id': f'review-cli-{index}',
                    'proposal_id': proposal_id,
                    'status': 'accepted',
                    'base_graph_digest': base_digest,
                    'candidate_graph_digest': candidate_digest,
                    'diff': {'meaningful_change_count': 1},
                    'preservation_proof': {'status': 'passed'},
                    'execution_contract_proof': {'status': 'passed'},
                    'allowed_runtime_action': 'create_partial_successor_rebase',
                }
            )
        diagnostics = {}
        if no_proposal_candidate:
            proposals = []
            reviews = []
            diagnostics['runtime_graph_rebase_candidate_review'] = {
                'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                'status': 'not_proposed',
                'reason': 'current_structural_closure_evidence_missing',
                'base_graph_digest': base_digest,
                'candidate_graph_digest': candidate_digest,
                'runtime_effect': 'none',
            }
        graph['graph_rebase_proposals'] = proposals
        graph['graph_rebase_reviews'] = reviews
        return {
            'id': response_id,
            'response_id': response_id,
            'status': 'completed',
            'lifecycle_state': 'completed',
            'response_frame': {
                'kind': 'ollmo.response_frame',
                'response_id': response_id,
                'frame_id': frame_id,
                'frame_sequence': frame_sequence,
                'lifecycle_state': 'completed',
                'frame_relation': {'kind': 'initial'},
            },
            'runtime': {
                'request_phase_graph': graph,
                'developer_diagnostics': diagnostics,
            },
        }

    @staticmethod
    def _graph_rebase_readiness(*, shadow_ready=True, partial_ready=True):
        return {
            'kind': 'ollmo.graph_rebase_readiness_report',
            'runtime_effect': 'none',
            'report_digest': 'readiness-cli-1',
            'corpus': {
                'settled_final_response_count': 20,
                'unique_workload_family_count': 7,
            },
            'candidate_opportunities': {
                'settled_final': {'total': 20, 'not_proposed_count': 20},
            },
            'gates': {
                'shadow_to_stage': {
                    'ready': shadow_ready,
                    'decision': 'ready_for_stage' if shadow_ready else 'remain_shadow',
                    'requirements': [],
                },
                'partial_stage_to_apply_reviewed': {
                    'ready': partial_ready,
                    'decision': (
                        'ready_for_exact_partial_apply_reviewed_authorization'
                        if partial_ready
                        else 'keep_partial_non_executable'
                    ),
                    'requirements': [],
                },
            },
            'observer': {'index_ok': True, 'load_error_count': 0},
        }

    @patch('helpers.ollmoctl._build_runtime_doctor_payload')
    def test_doctor_runtime_returns_json_snapshot(self, mock_build_runtime_doctor_payload):
        mock_build_runtime_doctor_payload.return_value = {
            'ok': True,
            'issues': [],
            'ollama': {
                'configured_path': {'path': '/opt/homebrew/bin/ollama', 'resolved': '/opt/homebrew/Cellar/ollama/0.18.0/bin/ollama'},
                'opt_path': {'path': '/opt/homebrew/opt/ollama/bin/ollama', 'resolved': '/opt/homebrew/Cellar/ollama/0.18.0/bin/ollama'},
                'which': ['/opt/homebrew/bin/ollama'],
                'same_binary': True,
                'brew_service': {'status': 'stopped'},
                'ownership_conflict': False,
            },
            'mlx': {
                'python': {'path': '/opt/mlx/venv/bin/python', 'exists': True},
                'python_version': '3.12.12',
                'packages': {'mlx': '0.31.1', 'mlx-lm': '0.31.1', 'mlx-vlm': None, 'mlx-whisper': None, 'mlx-audio': None},
            },
            'mlx_servers': {
                'mlx_lm': {'python_resolved': True, 'runtime_module_available': True},
                'mlx_vlm': {'python_resolved': True, 'runtime_module_available': True},
                'mlx_audio': {'python_resolved': True, 'runtime_module_available': True},
                'mlx_whisper': {'python_resolved': True, 'runtime_module_available': True, 'server_script_present': True},
            },
            'runtime': {'reachable': True, 'count': 0, 'instances': []},
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'doctor', 'runtime',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['ollama']['brew_service']['status'], 'stopped')

    @patch('helpers.ollmoctl._request_json')
    def test_ghost_returns_json_payload(self, mock_request_json):
        mock_request_json.return_value = {
            'identity': {'name': 'ollmo-ghost'},
            'summary': 'runtime summary',
            'markdown': '# Ollmo Ghost\n',
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'ghost',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['identity']['name'], 'ollmo-ghost')

    @patch('helpers.ollmoctl._request_json')
    def test_ghost_human_output_prints_markdown(self, mock_request_json):
        mock_request_json.return_value = {
            'summary': 'runtime summary',
            'markdown': '# Ollmo Ghost\n\n- hello\n',
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'ghost',
            ])

        self.assertEqual(exit_code, 0)
        text = buf.getvalue()
        self.assertIn('# Ollmo Ghost', text)
        self.assertIn('- hello', text)

    @patch('helpers.ollmoctl._request_json')
    def test_read_commands_do_not_recover_control_plane_by_default(self, mock_request_json):
        mock_request_json.side_effect = [
            {'models': []},
            [],
            {
                'identity': {'name': 'ollmo-ghost'},
                'summary': 'runtime summary',
                'markdown': '# Ollmo Ghost\n',
            },
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(ollmoctl.main(['models', 'list', '--json']), 0)
            self.assertEqual(ollmoctl.main(['instances', 'list', '--json']), 0)
            self.assertEqual(ollmoctl.main(['ghost', '--json']), 0)

        self.assertEqual(
            [
                call.kwargs.get('allow_runtime_recovery')
                for call in mock_request_json.call_args_list
            ],
            [False, False, False],
        )

    @patch('helpers.ollmoctl._request_json')
    def test_read_commands_can_opt_into_control_plane_recovery(self, mock_request_json):
        mock_request_json.side_effect = [
            {'models': []},
            [],
            {
                'identity': {'name': 'ollmo-ghost'},
                'summary': 'runtime summary',
                'markdown': '# Ollmo Ghost\n',
            },
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(
                ollmoctl.main(['--recover-control-plane', 'models', 'list', '--json']),
                0,
            )
            self.assertEqual(
                ollmoctl.main(['--recover-control-plane', 'instances', 'list', '--json']),
                0,
            )
            self.assertEqual(
                ollmoctl.main(['--recover-control-plane', 'ghost', '--json']),
                0,
            )

        self.assertEqual(
            [
                call.kwargs.get('allow_runtime_recovery')
                for call in mock_request_json.call_args_list
            ],
            [True, True, True],
        )

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_readiness_is_passive_and_returns_raw_json(self, mock_request_json):
        mock_request_json.return_value = self._graph_rebase_readiness()

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main(['graph-rebase', 'readiness', '--json'])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(buf.getvalue())['report_digest'], 'readiness-cli-1')
        self.assertEqual(mock_request_json.call_args.args[2], '/api/graph_rebase/readiness')
        self.assertIs(mock_request_json.call_args.kwargs['allow_runtime_recovery'], False)
        self.assertNotIn('headers', mock_request_json.call_args.kwargs)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_inspect_derives_exact_latest_cas_without_recovery(self, mock_request_json):
        mock_request_json.return_value = self._graph_rebase_truth()

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'inspect', 'resp-rebase-cli', '--json',
            ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        cas = payload['selected_proposal']['cas']
        self.assertEqual(cas['expected_response_id'], 'resp-rebase-cli')
        self.assertEqual(cas['expected_frame_id'], 'resp-rebase-cli:frame-1')
        self.assertEqual(cas['expected_frame_sequence'], 1)
        self.assertEqual(cas['expected_proposal_id'], 'proposal-cli-0')
        self.assertTrue(payload['selected_proposal']['binding_valid'])
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/responses/resp-rebase-cli')
        self.assertEqual(called.kwargs['query'], {'view': 'truth'})
        self.assertIs(called.kwargs['allow_runtime_recovery'], False)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_inspect_rejects_ambiguous_proposal_selection(self, mock_request_json):
        mock_request_json.return_value = self._graph_rebase_truth(proposal_count=2)

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'inspect', 'resp-rebase-cli', '--json',
            ])

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertIn('Multiple current graph-rebase proposals', payload['error'])

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_inspect_rejects_misrouted_response_truth(self, mock_request_json):
        mock_request_json.return_value = self._graph_rebase_truth(
            response_id='resp-other',
            frame_id='resp-other:frame-1',
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'inspect', 'resp-rebase-cli', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('does not match', json.loads(buf.getvalue())['error'])
        self.assertEqual(mock_request_json.call_count, 1)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_inspect_rejects_noncanonical_frame_sequences(
        self,
        mock_request_json,
    ):
        invalid_sequences = (
            True,
            0,
            -1,
            1.0,
            float('inf'),
            float('nan'),
            '1',
            '01',
            '+1',
            '1.0',
        )
        for sequence in invalid_sequences:
            with self.subTest(sequence=sequence):
                mock_request_json.return_value = self._graph_rebase_truth(
                    frame_sequence=sequence
                )
                buf = io.StringIO()
                with redirect_stdout(buf):
                    exit_code = ollmoctl.main([
                        'graph-rebase', 'inspect', 'resp-rebase-cli', '--json',
                    ])
                self.assertEqual(exit_code, 1)
                self.assertIn(
                    'positive JSON integer',
                    json.loads(buf.getvalue())['error'],
                )

    @patch.dict(
        'os.environ',
        {
            'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'test-cli-token-0000000000000000000000000000',
            'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'cli-test-operator',
        },
    )
    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_operator_fails_closed_on_root_frame_lifecycle_mismatch(
        self,
        mock_request_json,
    ):
        truth = self._graph_rebase_truth(no_proposal_candidate=True)
        truth['response_frame']['current_state'] = {
            'lifecycle_state': 'late_fill_running',
        }
        mock_request_json.side_effect = [copy.deepcopy(truth), copy.deepcopy(truth)]
        commands = (
            ['graph-rebase', 'inspect', 'resp-rebase-cli', '--json'],
            [
                'graph-rebase', 'adjudicate', 'resp-rebase-cli',
                '--adjudication', 'false_negative',
                '--rebase-class', 'partial_subtree_rebase',
                '--reason', 'This must not pass inconsistent lifecycle truth.',
                '--evidence-ref', 'operator:lifecycle-integrity',
                '--yes', '--json',
            ],
        )

        for command in commands:
            with self.subTest(command=command[1]):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    exit_code = ollmoctl.main(command)
                self.assertEqual(exit_code, 1)
                self.assertIn('lifecycle', json.loads(buf.getvalue())['error'].lower())

        self.assertEqual(mock_request_json.call_count, 2)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_mutation_rejects_remote_target_before_any_request(self, mock_request_json):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://example.test:5001',
                'graph-rebase', 'stage', 'resp-rebase-cli',
                '--reason', 'Must remain local.',
                '--evidence-ref', 'operator:local-boundary',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('loopback', json.loads(buf.getvalue())['error'])
        mock_request_json.assert_not_called()

    def test_graph_rebase_credentialed_redirect_handler_fails_closed(self):
        handler = ollmoctl._RejectRedirectHandler()
        request = Request('http://127.0.0.1:5001/operator')
        with self.assertRaises(HTTPError) as raised:
            handler.http_error_302(request, None, 302, 'Found', Message())
        self.assertEqual(raised.exception.code, 302)
        self.assertIn('forbidden', str(raised.exception).lower())

    @patch('helpers.ollmoctl.build_opener')
    def test_graph_rebase_credentialed_transport_disables_environment_proxy(
        self,
        mock_build_opener,
    ):
        mock_build_opener.return_value.open.side_effect = URLError('expected test stop')
        with self.assertRaises(ollmoctl.CliError):
            ollmoctl._request(
                'POST',
                'http://127.0.0.1:5001',
                '/operator',
                headers={'Authorization': 'Bearer redacted-test-token'},
                allow_redirects=False,
            )

        handlers = mock_build_opener.call_args.args
        proxy_handlers = [
            handler for handler in handlers if isinstance(handler, ollmoctl.ProxyHandler)
        ]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies, {})
        self.assertTrue(
            any(isinstance(handler, ollmoctl._RejectRedirectHandler) for handler in handlers)
        )

    @patch.dict(
        'os.environ',
        {
            'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'test-cli-token-0000000000000000000000000000',
            'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'cli-test-operator',
        },
    )
    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_adjudicate_sends_headers_and_derived_binding_without_secret_output(
        self,
        mock_request_json,
    ):
        mock_request_json.side_effect = [
            self._graph_rebase_truth(),
            {
                'status': 'recorded',
                'runtime_effect': 'none',
                'operator_record': {'record_id': 'operator-record-cli-1'},
            },
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'adjudicate', 'resp-rebase-cli',
                '--adjudication', 'useful_proposal',
                '--reason', 'The bounded proposal is useful.',
                '--evidence-ref', 'operator:cli-review',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 0)
        output = buf.getvalue()
        self.assertNotIn('test-cli-token', output)
        post = mock_request_json.call_args_list[1]
        self.assertEqual(post.args[0], 'POST')
        self.assertEqual(
            post.kwargs['payload']['expected_proposal_id'],
            'proposal-cli-0',
        )
        self.assertEqual(
            post.kwargs['headers']['Authorization'],
            'Bearer test-cli-token-0000000000000000000000000000',
        )
        self.assertEqual(
            post.kwargs['headers']['X-Ollmo-Graph-Rebase-Operator'],
            'cli-test-operator',
        )
        self.assertIs(post.kwargs['allow_runtime_recovery'], False)
        self.assertIs(post.kwargs['allow_redirects'], False)

    @patch.dict(
        'os.environ',
        {
            'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'secret-newline-token-00000000000000000000\n',
            'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'cli-test-operator',
        },
    )
    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_invalid_header_secret_is_redacted(self, mock_request_json):
        mock_request_json.return_value = self._graph_rebase_truth()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = ollmoctl.main([
                'graph-rebase', 'adjudicate', 'resp-rebase-cli',
                '--adjudication', 'useful_proposal',
                '--reason', 'Safe header validation.',
                '--evidence-ref', 'operator:header-validation',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 1)
        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn('secret-newline-token', rendered)
        self.assertIn('single-line ASCII', rendered)
        self.assertEqual(mock_request_json.call_count, 1)

    @patch.dict(
        'os.environ',
        {
            'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'test-cli-token-0000000000000000000000000000',
            'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'cli-test-operator',
        },
    )
    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_false_negative_uses_sentinel_and_explicit_class(
        self,
        mock_request_json,
    ):
        mock_request_json.side_effect = [
            self._graph_rebase_truth(no_proposal_candidate=True),
            {'status': 'recorded', 'operator_record': {'record_id': 'false-negative-1'}},
        ]

        with redirect_stdout(io.StringIO()):
            exit_code = ollmoctl.main([
                'graph-rebase', 'adjudicate', 'resp-rebase-cli',
                '--adjudication', 'false_negative',
                '--rebase-class', 'partial_subtree_rebase',
                '--reason', 'Runtime missed a necessary bounded structural replacement.',
                '--evidence-ref', 'operator:false-negative',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 0)
        post_payload = mock_request_json.call_args_list[1].kwargs['payload']
        self.assertEqual(post_payload['expected_proposal_id'], 'no_formal_proposal')
        self.assertEqual(
            post_payload['expected_requested_rebase_class'],
            'partial_subtree_rebase',
        )

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_false_negative_requires_explicit_class(self, mock_request_json):
        mock_request_json.return_value = self._graph_rebase_truth(
            no_proposal_candidate=True
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'adjudicate', 'resp-rebase-cli',
                '--adjudication', 'false_negative',
                '--reason', 'Runtime missed a necessary bounded structural replacement.',
                '--evidence-ref', 'operator:false-negative',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('--rebase-class', json.loads(buf.getvalue())['error'])
        self.assertEqual(mock_request_json.call_count, 1)

    @patch.dict(
        'os.environ',
        {
            'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'test-cli-token-0000000000000000000000000000',
            'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'cli-test-operator',
        },
    )
    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_stage_refreshes_successor_frame_before_returning(
        self,
        mock_request_json,
    ):
        first = self._graph_rebase_truth()
        staged = self._graph_rebase_truth(
            frame_id='resp-rebase-cli:frame-2',
            frame_sequence=2,
        )
        staged['response_frame']['frame_relation'] = {
            'kind': 'graph_rebase_stage_successor',
            'parent_frame_id': 'resp-rebase-cli:frame-1',
        }
        mock_request_json.side_effect = [
            first,
            self._graph_rebase_readiness(),
            {'status': 'staged', 'runtime_effect': 'staged_no_executable_mutation'},
            staged,
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'stage', 'resp-rebase-cli',
                '--reason', 'Stage the exact reviewed proposal without execution.',
                '--evidence-ref', 'operator:stage',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 0)
        output = json.loads(buf.getvalue())
        self.assertEqual(
            output['latest_inspection']['frame']['frame_id'],
            'resp-rebase-cli:frame-2',
        )
        post = mock_request_json.call_args_list[2]
        self.assertEqual(
            post.kwargs['payload']['expected_frame_id'],
            'resp-rebase-cli:frame-1',
        )
        self.assertTrue(
            all(
                call.kwargs.get('allow_runtime_recovery') is False
                for call in mock_request_json.call_args_list
            )
        )
        self.assertIs(post.kwargs['allow_redirects'], False)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_stage_rejects_failed_local_proof_before_readiness_or_post(
        self,
        mock_request_json,
    ):
        truth = self._graph_rebase_truth()
        truth['runtime']['request_phase_graph']['graph_rebase_reviews'][0][
            'preservation_proof'
        ] = {'status': 'failed'}
        mock_request_json.return_value = truth

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'stage', 'resp-rebase-cli',
                '--reason', 'This proof is not green.',
                '--evidence-ref', 'operator:proof-check',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('do not permit', json.loads(buf.getvalue())['error'])
        self.assertEqual(mock_request_json.call_count, 1)

    def test_graph_rebase_confirmation_shows_scope_and_gate(self):
        class TTYInput(io.StringIO):
            def isatty(self):
                return True

        stderr = io.StringIO()
        with patch('sys.stdin', TTYInput('yes\n')), redirect_stderr(stderr):
            ollmoctl._confirm_graph_rebase_action(
                action='authorize_partial',
                cas={
                    'expected_response_id': 'resp-rebase-cli',
                    'expected_frame_id': 'resp-rebase-cli:frame-2',
                    'expected_frame_sequence': 2,
                    'expected_proposal_id': 'proposal-cli-0',
                    'expected_requested_rebase_class': 'partial_subtree_rebase',
                },
                assume_yes=False,
                scope={
                    'scope_root_ids': ['phase-root'],
                    'scope_phase_ids': ['phase-replacement-0'],
                    'scope_branch_ids': ['branch-replacement-0'],
                    'preserve_outside_scope': True,
                },
                gate_name='partial_stage_to_apply_reviewed',
                gate={'decision': 'ready_for_exact_partial_apply_reviewed_authorization'},
                readiness_report_digest='readiness-cli-1',
            )

        rendered = stderr.getvalue()
        self.assertIn('scope_branches=branch-replacement-0', rendered)
        self.assertIn('preserve_outside_scope=True', rendered)
        self.assertIn('gate=partial_stage_to_apply_reviewed', rendered)
        self.assertIn('report=readiness-cli-1', rendered)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_authorize_partial_requires_execute_before_any_request(
        self,
        mock_request_json,
    ):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'authorize-partial', 'resp-rebase-cli',
                '--reason', 'Authorize exact staged partial work.',
                '--evidence-ref', 'operator:authorization',
                '--yes', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('--execute', json.loads(buf.getvalue())['error'])
        mock_request_json.assert_not_called()

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_authorize_partial_rejects_full_proposal_before_post(
        self,
        mock_request_json,
    ):
        mock_request_json.return_value = self._graph_rebase_truth(
            requested_class='full_successor_rebase'
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'authorize-partial', 'resp-rebase-cli',
                '--reason', 'This must remain blocked.',
                '--evidence-ref', 'operator:full-boundary',
                '--execute', '--yes', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('cannot authorize a full successor', json.loads(buf.getvalue())['error'])
        self.assertEqual(mock_request_json.call_count, 1)

    @patch.dict(
        'os.environ',
        {
            'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'test-cli-token-0000000000000000000000000000',
            'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'cli-test-operator',
        },
    )
    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_authorize_partial_success_uses_exact_safe_transport(
        self,
        mock_request_json,
    ):
        mock_request_json.side_effect = [
            self._graph_rebase_truth(),
            self._graph_rebase_readiness(),
            {
                'status': 'scheduled',
                'runtime_effect': 'branch_local_partial_successor_queued',
                'response_frame': {'frame_id': 'resp-rebase-cli:frame-2'},
            },
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                'graph-rebase', 'authorize-partial', 'resp-rebase-cli',
                '--reason', 'Authorize the exact reviewed partial successor.',
                '--evidence-ref', 'operator:partial-authorization',
                '--execute', '--yes', '--json',
            ])

        self.assertEqual(exit_code, 0)
        output = json.loads(buf.getvalue())
        self.assertEqual(output['action'], 'authorize_partial')
        post = mock_request_json.call_args_list[2]
        self.assertEqual(post.kwargs['payload']['action'], 'authorize_partial')
        self.assertEqual(
            post.kwargs['payload']['expected_requested_rebase_class'],
            'partial_subtree_rebase',
        )
        self.assertIs(post.kwargs['allow_runtime_recovery'], False)
        self.assertIs(post.kwargs['allow_redirects'], False)

    @patch('helpers.ollmoctl._request_json')
    def test_graph_rebase_commands_reject_control_plane_recovery(self, mock_request_json):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--recover-control-plane',
                'graph-rebase', 'readiness', '--json',
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn('never recover', json.loads(buf.getvalue())['error'])
        mock_request_json.assert_not_called()

    @patch('helpers.ollmoctl._attempt_local_control_plane_recovery')
    @patch('helpers.ollmoctl.urlopen')
    def test_request_json_auto_recovers_local_control_plane_once(self, mock_urlopen, mock_recovery):
        mock_recovery.return_value = True
        mock_urlopen.side_effect = [
            URLError(ConnectionRefusedError(61, 'Connection refused')),
            self._FakeResponse({'models': []}),
        ]

        payload = ollmoctl._request_json(
            'GET',
            'http://127.0.0.1:5001',
            '/api/available_models',
            allow_runtime_recovery=True,
        )

        self.assertEqual(payload, {'models': []})
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_recovery.assert_called_once_with('http://127.0.0.1:5001')

    @patch('helpers.ollmoctl._attempt_local_control_plane_recovery')
    @patch('helpers.ollmoctl.urlopen')
    def test_request_json_keeps_diagnostics_read_only_when_recovery_disabled(self, mock_urlopen, mock_recovery):
        mock_urlopen.side_effect = URLError(ConnectionRefusedError(61, 'Connection refused'))

        with self.assertRaises(ollmoctl.CliError) as ctx:
            ollmoctl._request_json(
                'GET',
                'http://127.0.0.1:5001',
                '/api/running_instances',
                allow_runtime_recovery=False,
            )

        self.assertIn('Could not reach Ollmo', str(ctx.exception))
        mock_recovery.assert_not_called()

    @patch('helpers.ollmoctl._collect_running_instances_snapshot')
    @patch('helpers.ollmoctl._collect_mlx_server_runtime_checks')
    @patch('helpers.ollmoctl._collect_mlx_runtime_versions')
    @patch('helpers.ollmoctl._brew_service_status')
    @patch('helpers.ollmoctl._which_all')
    @patch('helpers.ollmoctl._path_snapshot')
    def test_build_runtime_doctor_payload_flags_brew_service_conflict(
        self,
        mock_path_snapshot,
        mock_which_all,
        mock_brew_service_status,
        mock_collect_mlx_runtime_versions,
        mock_collect_mlx_server_runtime_checks,
        mock_collect_running_instances_snapshot,
    ):
        mock_path_snapshot.side_effect = lambda path: {'path': path, 'exists': True, 'resolved': '/opt/homebrew/Cellar/ollama/0.18.0/bin/ollama'}
        mock_which_all.return_value = ['/opt/homebrew/bin/ollama']
        mock_brew_service_status.return_value = {'status': 'started', 'raw_line': 'ollama started dev ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist'}
        mock_collect_mlx_runtime_versions.return_value = {
            'python': {'path': '/opt/mlx/venv/bin/python', 'exists': True},
            'python_version': '3.12.12',
            'packages': {'mlx': '0.31.1', 'mlx-lm': '0.31.1', 'mlx-vlm': None, 'mlx-whisper': None, 'mlx-audio': None},
            'error': None,
        }
        mock_collect_mlx_server_runtime_checks.return_value = {
            'mlx_lm': {'python_resolved': True, 'runtime_module_available': True},
            'mlx_vlm': {'python_resolved': True, 'runtime_module_available': True},
            'mlx_audio': {'python_resolved': True, 'runtime_module_available': True},
            'mlx_whisper': {'python_resolved': True, 'runtime_module_available': True, 'server_script_present': True},
        }
        mock_collect_running_instances_snapshot.return_value = {
            'reachable': True,
            'error': None,
            'count': 1,
            'instances': [
                {
                    'instance_id': 'flux-1',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'port': 11435,
                    'readiness': 'ready',
                    'activity': 'idle',
                    'last_error': None,
                }
            ],
        }

        payload = ollmoctl._build_runtime_doctor_payload('http://127.0.0.1:5001', 20)

        self.assertFalse(payload['ok'])
        self.assertTrue(payload['ollama']['ownership_conflict'])
        self.assertTrue(any('brew services stop ollama' in item for item in payload['issues']))

    @patch('helpers.ollmoctl._collect_running_instances_snapshot')
    @patch('helpers.ollmoctl._collect_mlx_server_runtime_checks')
    @patch('helpers.ollmoctl._collect_mlx_runtime_versions')
    @patch('helpers.ollmoctl._brew_service_status')
    @patch('helpers.ollmoctl._which_all')
    @patch('helpers.ollmoctl._path_snapshot')
    def test_build_runtime_doctor_payload_reports_package_specific_mlx_runtime_gaps(
        self,
        mock_path_snapshot,
        mock_which_all,
        mock_brew_service_status,
        mock_collect_mlx_runtime_versions,
        mock_collect_mlx_server_runtime_checks,
        mock_collect_running_instances_snapshot,
    ):
        mock_path_snapshot.side_effect = lambda path: {'path': path, 'exists': True, 'resolved': str(path)}
        mock_which_all.return_value = ['/opt/homebrew/bin/ollama']
        mock_brew_service_status.return_value = {'status': 'stopped'}
        mock_collect_mlx_runtime_versions.return_value = {
            'python': {'path': '/opt/mlx/venv/bin/python', 'exists': True},
            'python_version': '3.12.12',
            'packages': {'mlx': '0.31.1', 'mlx-lm': '0.31.1', 'mlx-vlm': None, 'mlx-whisper': None, 'mlx-audio': None},
            'error': None,
        }
        mock_collect_mlx_server_runtime_checks.return_value = {
            'mlx_lm': {'python_resolved': True, 'runtime_module_available': True},
            'mlx_vlm': {'python_resolved': False, 'runtime_module_available': False, 'python_error': 'missing mlx_vlm'},
            'mlx_audio': {'python_resolved': True, 'runtime_module_available': False, 'required_runtime_module': 'mlx_audio.server', 'python_path': '/opt/mlx-audio/venv/bin/python'},
            'mlx_whisper': {'python_resolved': True, 'runtime_module_available': True, 'server_script_present': False, 'server_script_path': '/tmp/missing.py'},
        }
        mock_collect_running_instances_snapshot.return_value = {
            'reachable': True,
            'error': None,
            'count': 0,
            'instances': [],
        }

        payload = ollmoctl._build_runtime_doctor_payload('http://127.0.0.1:5001', 20)

        self.assertFalse(payload['ok'])
        self.assertIn('mlx_servers', payload)
        self.assertTrue(any('MLX runtime unavailable for mlx_vlm' in item for item in payload['issues']))
        self.assertTrue(any('MLX runtime module unavailable for mlx_audio' in item for item in payload['issues']))
        self.assertTrue(any('missing shim script' in item for item in payload['issues']))

    @patch('helpers.ollmoctl._collect_running_instances_snapshot')
    @patch('helpers.ollmoctl._collect_mlx_server_runtime_checks')
    @patch('helpers.ollmoctl._collect_mlx_runtime_versions')
    @patch('helpers.ollmoctl._brew_service_status')
    @patch('helpers.ollmoctl._which_all')
    @patch('helpers.ollmoctl._path_snapshot')
    def test_build_runtime_doctor_payload_reports_whisper_dependency_import_failure(
        self,
        mock_path_snapshot,
        mock_which_all,
        mock_brew_service_status,
        mock_collect_mlx_runtime_versions,
        mock_collect_mlx_server_runtime_checks,
        mock_collect_running_instances_snapshot,
    ):
        mock_path_snapshot.side_effect = lambda path: {'path': path, 'exists': True, 'resolved': str(path)}
        mock_which_all.return_value = ['/opt/homebrew/bin/ollama']
        mock_brew_service_status.return_value = {'status': 'stopped'}
        mock_collect_mlx_runtime_versions.return_value = {
            'python': {'path': '/opt/mlx/venv/bin/python', 'exists': True},
            'python_version': '3.12.12',
            'packages': {'mlx': '0.31.1', 'mlx-lm': '0.31.1', 'mlx-vlm': '0.6.5', 'mlx-whisper': '0.4.3', 'mlx-audio': '0.4.5'},
            'error': None,
        }
        mock_collect_mlx_server_runtime_checks.return_value = {
            'mlx_lm': {'python_resolved': True, 'runtime_module_available': True, 'runtime_dependencies_ready': True},
            'mlx_vlm': {'python_resolved': True, 'runtime_module_available': True, 'runtime_dependencies_ready': True},
            'mlx_audio': {'python_resolved': True, 'runtime_module_available': True, 'runtime_dependencies_ready': True},
            'mlx_whisper': {
                'python_resolved': True,
                'runtime_module_available': True,
                'runtime_dependencies_ready': False,
                'runtime_dependency_error': 'ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.',
                'server_script_present': True,
            },
        }
        mock_collect_running_instances_snapshot.return_value = {
            'reachable': True,
            'error': None,
            'count': 0,
            'instances': [],
        }

        payload = ollmoctl._build_runtime_doctor_payload('http://127.0.0.1:5001', 20)

        self.assertFalse(payload['ok'])
        self.assertTrue(
            any('Numba needs NumPy' in item for item in payload['issues'])
        )

    @patch('helpers.ollmoctl._request_json')
    def test_models_list_filters_and_returns_json(self, mock_request_json):
        mock_request_json.return_value = {
            'models': [
                {'model': 'mlx-community/Qwen3.5-27B-4bit', 'backend': 'mlx', 'capability': 'chat', 'runnable': True},
                {'model': 'mlx-community/broken', 'backend': 'mlx', 'capability': 'chat', 'runnable': False},
            ]
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'models', 'list',
                '--backend', 'mlx',
                '--runnable-only',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['count'], 1)
        self.assertEqual(payload['models'][0]['model'], 'mlx-community/Qwen3.5-27B-4bit')

    @patch('helpers.ollmoctl._request_json')
    def test_models_list_human_output_includes_feature_contract_summary(self, mock_request_json):
        mock_request_json.return_value = {
            'models': [
                {
                    'model': 'mlx-community/Qwen3.5-27B-4bit',
                    'backend': 'mlx',
                    'capability': 'chat',
                    'runnable': True,
                    'inputs': ['text', 'image'],
                    'outputs': ['text'],
                    'features': {
                        'tool_calling': True,
                        'function_calling': True,
                        'vision_input': True,
                    },
                },
            ]
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'models', 'list',
            ])

        self.assertEqual(exit_code, 0)
        text = buf.getvalue()
        self.assertIn('in=text,image', text)
        self.assertIn('out=text', text)
        self.assertIn('feat=tool_calling,function_calling', text)

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_routes_chat_for_chat_capability_without_file(self, mock_lookup, mock_request_json):
        mock_lookup.return_value = {
            'instance_id': 'chat-1',
            'capability': 'chat',
        }
        mock_request_json.return_value = {'output_text': 'hello'}

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'chat-1', 'hello world',
                '--instructions', 'be concise',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/responses')
        self.assertEqual(called.kwargs['payload']['instructions'], 'be concise')
        self.assertEqual(called.kwargs['payload']['input'], 'hello world')
        self.assertEqual(called.kwargs['payload']['instance_id'], 'chat-1')

    def test_response_truth_uses_canonical_lifecycle_over_nested_late_fill(self):
        payload = {
            'id': 'resp_failed',
            'status': 'completed',
            'lifecycle_state': 'late_fill_failed',
            'late_fill': {'status': 'running'},
        }

        truth = ollmoctl._build_response_truth_summary(payload)

        self.assertFalse(truth['has_open_continuation'])
        self.assertTrue(truth['is_terminal'])
        self.assertEqual(truth['canonical_lifecycle_source'], 'lifecycle_state')

    def test_response_truth_uses_status_semantics_lifecycle_before_legacy_fallback(self):
        payload = {
            'id': 'resp_failed_semantics',
            'status': 'completed',
            'status_semantics': {
                'canonical_lifecycle_state': 'late_fill_failed',
                'has_open_continuation': False,
                'is_terminal': True,
            },
            'late_fill': {'status': 'running'},
        }

        truth = ollmoctl._build_response_truth_summary(payload)

        self.assertFalse(truth['has_open_continuation'])
        self.assertTrue(truth['is_terminal'])
        self.assertEqual(
            truth['canonical_lifecycle_source'],
            'status_semantics.canonical_lifecycle_state',
        )

    def test_response_truth_marks_active_lifecycle_despite_completed_status(self):
        payload = {
            'id': 'resp_running',
            'status': 'completed',
            'lifecycle_state': 'late_fill_running',
        }

        truth = ollmoctl._build_response_truth_summary(payload)

        self.assertTrue(truth['has_open_continuation'])
        self.assertFalse(truth['is_terminal'])

    def test_response_truth_marks_partial_cancelled_lifecycle_terminal(self):
        truth = ollmoctl._build_response_truth_summary(
            {
                'id': 'resp_partial_cancelled',
                'status': 'completed',
                'lifecycle_state': 'partial_cancelled',
            }
        )

        self.assertFalse(truth['has_open_continuation'])
        self.assertFalse(truth['has_actionable_repair'])
        self.assertTrue(truth['is_terminal'])

    def test_response_truth_legacy_late_fill_fallback_when_no_canonical_lifecycle(self):
        payload = {
            'id': 'resp_legacy_running',
            'status': 'completed',
            'late_fill': {'status': 'running'},
        }

        truth = ollmoctl._build_response_truth_summary(payload)

        self.assertTrue(truth['has_open_continuation'])
        self.assertEqual(truth['canonical_lifecycle_source'], 'legacy_fallback')

    @patch('helpers.ollmoctl._request_json')
    def test_responses_get_human_output_surfaces_canonical_response_truth(self, mock_request_json):
        mock_request_json.return_value = {
            'id': 'resp_lookup',
            'status': 'completed',
            'lifecycle_state': 'late_fill_failed',
            'status_semantics': {
                'canonical_lifecycle_state': 'late_fill_failed',
                'has_open_continuation': False,
                'is_terminal': True,
                'has_actionable_repair': True,
            },
            'late_fill': {'status': 'running'},
            'outputs': [
                {'type': 'text', 'source': 'promoted_output_slot', 'compatibility_derived': False},
                {'type': 'text', 'source': 'compatibility_derived', 'compatibility_derived': True},
            ],
            'durability': {'source': 'response_frame_ledger', 'recovered': True},
            'response_frame': {
                'frame_id': 'resp_lookup:frame-2',
                'frame_sequence': 2,
                'frame_relation': {
                    'kind': 'late_fill_successor',
                    'parent_frame_id': 'resp_lookup:frame-1',
                    'parent_frame_sequence': 1,
                },
                'planning': {
                    'artifact_flow': {
                        'work_tree': {
                            'work_tree_source': 'runtime_owned',
                            'authoritative': True,
                            'compatibility_derived': False,
                        }
                    }
                },
            },
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'responses', 'get', 'resp_lookup',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/responses/resp_lookup?view=truth')
        self.assertEqual(called.kwargs.get('allow_runtime_recovery'), False)
        text = buf.getvalue()
        self.assertIn('lifecycle_state: late_fill_failed (canonical via lifecycle_state)', text)
        self.assertIn('open_continuation: no', text)
        self.assertIn('actionable_repair: yes', text)
        self.assertIn('terminal: yes', text)
        self.assertIn('response_frame: resp_lookup:frame-2 seq=2 relation=late_fill_successor parent=resp_lookup:frame-1', text)
        self.assertIn('outputs: count=2 canonical=1 compatibility_derived=1', text)
        self.assertIn('work_tree: source=runtime_owned authoritative=yes compatibility_derived=no', text)
        self.assertIn('durability: source=response_frame_ledger recovered=yes', text)

    @patch('helpers.ollmoctl._request_json')
    def test_responses_get_truth_json_returns_normalized_summary(self, mock_request_json):
        mock_request_json.return_value = {
            'id': 'resp_lookup',
            'status': 'completed',
            'lifecycle_state': 'late_fill_running',
            'response_frame': {
                'frame_id': 'resp_lookup:frame-1',
                'frame_sequence': 1,
                'frame_relation': {'kind': 'initial'},
            },
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'responses', 'get', 'resp_lookup',
                '--truth-json',
            ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['response_id'], 'resp_lookup')
        self.assertTrue(payload['has_open_continuation'])
        self.assertEqual(payload['response_frame']['frame_relation'], 'initial')
        self.assertEqual(mock_request_json.call_args.args[2], '/api/responses/resp_lookup?view=truth')
        self.assertEqual(mock_request_json.call_args.kwargs.get('allow_runtime_recovery'), False)

    @patch('helpers.ollmoctl._request_json')
    def test_responses_get_can_opt_into_control_plane_recovery(self, mock_request_json):
        mock_request_json.return_value = {
            'id': 'resp_lookup',
            'status': 'completed',
            'lifecycle_state': 'completed',
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'responses', 'get', 'resp_lookup',
                '--recover-control-plane',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/responses/resp_lookup?view=truth')
        self.assertEqual(called.kwargs.get('allow_runtime_recovery'), True)

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_truth_json_prints_normalized_response_truth(self, mock_lookup, mock_request_json):
        mock_lookup.return_value = {
            'instance_id': 'chat-1',
            'capability': 'chat',
        }
        mock_request_json.side_effect = [
            {
                'response_id': 'resp_send',
                'status': 'completed',
                'lifecycle_state': 'completed',
            },
            {
                'id': 'resp_send',
                'status': 'completed',
                'lifecycle_state': 'completed',
                'response_frame': {
                    'frame_id': 'resp_send:frame-2',
                    'frame_sequence': 2,
                    'frame_relation': {
                        'kind': 'late_fill_successor',
                        'parent_frame_id': 'resp_send:frame-1',
                    },
                    'planning': {
                        'artifact_flow': {
                            'work_tree': {
                                'work_tree_source': 'runtime_owned',
                                'authoritative': True,
                                'compatibility_derived': False,
                            },
                        },
                    },
                },
            },
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'chat-1', 'hello world',
                '--truth-json',
            ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['response_id'], 'resp_send')
        self.assertFalse(payload['has_open_continuation'])
        self.assertTrue(payload['is_terminal'])
        self.assertEqual(payload['response_frame']['frame_id'], 'resp_send:frame-2')
        self.assertEqual(payload['response_frame']['frame_sequence'], 2)
        self.assertEqual(
            payload['response_frame']['frame_relation'],
            'late_fill_successor',
        )
        self.assertEqual(payload['work_tree']['work_tree_source'], 'runtime_owned')
        self.assertTrue(payload['work_tree']['authoritative'])
        self.assertEqual(mock_request_json.call_count, 2)
        post, truth_get = mock_request_json.call_args_list
        self.assertEqual(post.args[0], 'POST')
        self.assertEqual(post.args[2], '/api/responses')
        self.assertEqual(truth_get.args[0], 'GET')
        self.assertEqual(truth_get.args[1], 'http://127.0.0.1:5001')
        self.assertEqual(
            truth_get.args[2],
            '/api/responses/resp_send?view=truth',
        )
        self.assertIs(truth_get.kwargs['allow_runtime_recovery'], False)
        self.assertNotIn('headers', truth_get.kwargs)
        self.assertNotIn('allow_redirects', truth_get.kwargs)

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_truth_json_rejects_mismatched_response_identity(
        self,
        mock_lookup,
        mock_request_json,
    ):
        mock_lookup.return_value = {'instance_id': 'chat-1', 'capability': 'chat'}
        mock_request_json.side_effect = [
            {'id': 'resp_expected', 'status': 'completed'},
            {'id': 'resp_other', 'status': 'completed'},
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'chat-1', 'hello', '--truth-json',
            ])

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['status_code'], 409)
        self.assertEqual(
            payload['details']['expected_response_id'],
            'resp_expected',
        )
        self.assertEqual(payload['details']['actual_response_id'], 'resp_other')

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_truth_json_rejects_frame_regression(
        self,
        mock_lookup,
        mock_request_json,
    ):
        mock_lookup.return_value = {'instance_id': 'chat-1', 'capability': 'chat'}
        mock_request_json.side_effect = [
            {
                'id': 'resp_frame_binding',
                'response_frame': {'frame_id': 'frame-2', 'frame_sequence': 2},
            },
            {
                'id': 'resp_frame_binding',
                'response_frame': {'frame_id': 'frame-1', 'frame_sequence': 1},
            },
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'chat-1', 'hello', '--truth-json',
            ])

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['status_code'], 409)
        self.assertEqual(payload['details']['post_frame_sequence'], 2)
        self.assertEqual(payload['details']['truth_frame_sequence'], 1)

    def test_response_truth_prefers_frozen_frame_lifecycle_and_outputs(self):
        truth = ollmoctl._build_response_truth_summary(
            {
                'id': 'resp_frozen_authority',
                'lifecycle_state': 'completed',
                'outputs': [
                    {'source': 'compatibility_derived', 'compatibility_derived': True},
                ],
                'response_frame': {
                    'current_state': {'lifecycle_state': 'late_fill_failed'},
                    'output': {
                        'outputs': [
                            {'source': 'promoted_output_slot'},
                            {'type': 'text'},
                        ],
                    },
                },
            }
        )

        self.assertEqual(truth['lifecycle_state'], 'late_fill_failed')
        self.assertEqual(
            truth['canonical_lifecycle_source'],
            'response_frame.current_state.lifecycle_state',
        )
        self.assertEqual(truth['outputs']['count'], 2)
        self.assertEqual(truth['outputs']['canonical_count'], 1)
        self.assertEqual(truth['outputs']['compatibility_derived_count'], 0)
        self.assertEqual(truth['outputs']['unknown_provenance_count'], 1)

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_json_does_not_fetch_canonical_truth(self, mock_lookup, mock_request_json):
        mock_lookup.return_value = {
            'instance_id': 'chat-1',
            'capability': 'chat',
        }
        mock_request_json.return_value = {
            'id': 'resp_send_bounded',
            'status': 'completed',
            'wire_projection': {'bounded': True},
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'chat-1', 'hello world',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(buf.getvalue())['wire_projection']['bounded'])
        self.assertEqual(mock_request_json.call_count, 1)
        self.assertEqual(mock_request_json.call_args.args[0], 'POST')
        self.assertEqual(mock_request_json.call_args.args[2], '/api/responses')

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_human_output_does_not_fetch_canonical_truth(self, mock_lookup, mock_request_json):
        mock_lookup.return_value = {
            'instance_id': 'chat-1',
            'capability': 'chat',
        }
        mock_request_json.return_value = {
            'id': 'resp_send_human',
            'output_text': 'bounded answer',
            'wire_projection': {'bounded': True},
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'chat-1', 'hello world',
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(buf.getvalue().strip(), 'bounded answer')
        self.assertEqual(mock_request_json.call_count, 1)
        self.assertEqual(mock_request_json.call_args.args[0], 'POST')
        self.assertEqual(mock_request_json.call_args.args[2], '/api/responses')

    @patch('helpers.ollmoctl._request_json')
    def test_instance_lookup_accepts_exact_model_name(self, mock_request_json):
        mock_request_json.return_value = [
            {
                'instance_id': 'mlx-community__Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16-mlx-11504',
                'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                'capability': 'text_to_speech',
            }
        ]

        resolved = ollmoctl._coerce_instance_lookup(
            'http://127.0.0.1:5001',
            'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
        )

        self.assertEqual(
            resolved['instance_id'],
            'mlx-community__Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16-mlx-11504',
        )

    @patch('helpers.ollmoctl._request_json')
    def test_instance_lookup_rejects_ambiguous_selector(self, mock_request_json):
        mock_request_json.return_value = [
            {'instance_id': 'foo-1', 'model': 'foo-model', 'capability': 'chat'},
            {'instance_id': 'foo-2', 'model': 'foo-model-2', 'capability': 'chat'},
        ]

        with self.assertRaises(ollmoctl.CliError) as ctx:
            ollmoctl._coerce_instance_lookup('http://127.0.0.1:5001', 'foo')

        self.assertIn('ambiguous', str(ctx.exception))

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_routes_infer_for_non_chat_capability(self, mock_lookup, mock_request_json):
        mock_lookup.return_value = {
            'instance_id': 'whisper-1',
            'capability': 'speech_to_text',
        }
        mock_request_json.return_value = {'output_text': 'transcript'}

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as handle:
            handle.write(b'fake-audio')
            file_path = handle.name
        self.addCleanup(lambda: Path(file_path).unlink(missing_ok=True))

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send', 'whisper-1',
                '--file', file_path,
                '--language', 'de',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/responses')
        self.assertEqual(called.kwargs['payload']['instance_id'], 'whisper-1')
        self.assertEqual(called.kwargs['payload']['file_path'], str(Path(file_path).resolve()))
        self.assertEqual(called.kwargs['payload']['language'], 'de')
        self.assertEqual(called.kwargs['payload']['infer_timeout_sec'], 3585)

    @patch('helpers.ollmoctl._request_json')
    def test_start_sends_expected_payload(self, mock_request_json):
        mock_request_json.return_value = {'status': 'started', 'instance': {'instance_id': 'mlx-1'}}

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'start',
                '--model', 'mlx-community/Qwen3.5-27B-4bit',
                '--backend', 'mlx',
                '--capability', 'chat',
                '--preferred-port', '11502',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/start_model')
        self.assertEqual(called.kwargs['payload']['model'], 'mlx-community/Qwen3.5-27B-4bit')
        self.assertEqual(called.kwargs['payload']['backend'], 'mlx')
        self.assertEqual(called.kwargs['payload']['preferred_port'], 11502)

    @patch('helpers.ollmoctl._request_json')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_send_uses_resolved_instance_id_when_selector_is_model_name(self, mock_lookup, mock_request_json):
        mock_lookup.return_value = {
            'instance_id': 'mlx-community__Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16-mlx-11504',
            'capability': 'text_to_speech',
        }
        mock_request_json.return_value = {'content': 'Audio generated.'}

        with redirect_stdout(io.StringIO()):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'send',
                'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                'hello',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(
            called.kwargs['payload']['instance_id'],
            'mlx-community__Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16-mlx-11504',
        )

    def test_response_truth_rejects_malformed_semantics_and_zero_frame_sequence(self):
        truth = ollmoctl._build_response_truth_summary(
            {
                'id': 'resp_malformed_semantics',
                'status': 'pending',
                'lifecycle_state': {'unexpected': 'mapping'},
                'status_semantics': {
                    'canonical_lifecycle_state': 'late_fill_running',
                    'has_open_continuation': 'false',
                    'is_terminal': 'false',
                },
                'response_frame': {
                    'frame_id': 'frame-zero',
                    'frame_sequence': 0,
                },
            }
        )

        self.assertEqual(truth['lifecycle_state'], 'late_fill_running')
        self.assertEqual(
            truth['canonical_lifecycle_source'],
            'status_semantics.canonical_lifecycle_state',
        )
        self.assertTrue(truth['has_open_continuation'])
        self.assertFalse(truth['is_terminal'])
        self.assertIsNone(
            ollmoctl._complete_response_frame_identity(
                {
                    'response_frame': {
                        'frame_id': 'frame-zero',
                        'frame_sequence': 0,
                    }
                }
            )
        )

    def test_response_truth_recovers_message_identity_from_frozen_output(self):
        truth = ollmoctl._build_response_truth_summary(
            {
                'id': 'resp_message_identity',
                'status': 'completed',
                'lifecycle_state': 'completed',
                'response_frame': {
                    'current_state': {
                        'output': [
                            {
                                'id': 'msg_frozen_identity',
                                'type': 'message',
                            }
                        ]
                    }
                },
            }
        )

        self.assertEqual(truth['message_id'], 'msg_frozen_identity')

    def test_send_and_responses_get_output_flags_are_mutually_exclusive(self):
        for argv in (
            ['send', 'chat-1', 'hello', '--json', '--truth-json'],
            [
                'responses',
                'get',
                'resp_flags',
                '--json',
                '--truth-json',
            ],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    ollmoctl.main(argv)
                self.assertEqual(raised.exception.code, 2)

    @patch('helpers.ollmoctl._request_bytes')
    def test_artifact_download_writes_file(self, mock_request_bytes):
        mock_request_bytes.return_value = (b'hello artifact', {'Content-Type': 'text/plain'})

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'artifact.txt'
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = ollmoctl.main([
                    '--base-url', 'http://127.0.0.1:5001',
                    'artifact', 'download',
                    '/tmp/example.txt',
                    '--output', str(target),
                    '--json',
                ])

            self.assertEqual(exit_code, 0)
            self.assertEqual(target.read_bytes(), b'hello artifact')
            payload = json.loads(buf.getvalue())
            self.assertEqual(Path(payload['saved_to']).resolve(), target.resolve())

    @patch('helpers.ollmoctl._request_json')
    def test_events_list_routes_to_event_history(self, mock_request_json):
        mock_request_json.return_value = {
            'items': [
                {'timestamp': '2026-03-14T20:30:00Z', 'category': 'infer', 'action': 'request', 'status': 'started', 'message': 'Infer request accepted.'}
            ],
            'count': 1,
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = ollmoctl.main([
                '--base-url', 'http://127.0.0.1:5001',
                'events', 'list',
                '--category', 'infer',
                '--status', 'started',
                '--json',
            ])

        self.assertEqual(exit_code, 0)
        called = mock_request_json.call_args
        self.assertEqual(called.args[2], '/api/event_history')
        self.assertEqual(called.kwargs['query']['category'], 'infer')
        self.assertEqual(called.kwargs['query']['status'], 'started')

    @patch('helpers.ollmoctl._fetch_event_items')
    def test_events_tail_emits_new_items(self, mock_fetch_event_items):
        mock_fetch_event_items.side_effect = [
            [],
            [{'id': 'event-1', 'timestamp': '2026-03-14T20:30:00Z', 'category': 'infer', 'action': 'request', 'status': 'started', 'message': 'Infer request accepted.'}],
        ]

        with patch('time.sleep', return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = ollmoctl.main([
                    '--base-url', 'http://127.0.0.1:5001',
                    'events', 'tail',
                    '--category', 'infer',
                    '--count', '1',
                    '--json',
                ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['id'], 'event-1')

    @patch('helpers.ollmoctl._fetch_event_items')
    @patch('helpers.ollmoctl._coerce_instance_lookup')
    def test_wait_returns_latest_terminal_event(self, mock_lookup, mock_fetch_event_items):
        mock_lookup.return_value = {'instance_id': 'whisper-1'}
        mock_fetch_event_items.side_effect = [
            [{'id': 'event-1', 'timestamp': '2026-03-14T20:30:00Z', 'instance_id': 'whisper-1', 'category': 'infer', 'action': 'request', 'status': 'started'}],
            [{'id': 'event-2', 'timestamp': '2026-03-14T20:31:00Z', 'instance_id': 'whisper-1', 'category': 'infer', 'action': 'request', 'status': 'ok', 'message': 'speech_to_text'}],
        ]

        with patch('time.sleep', return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = ollmoctl.main([
                    '--base-url', 'http://127.0.0.1:5001',
                    'wait', 'whisper-1',
                    '--json',
                ])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['instance_id'], 'whisper-1')
        self.assertEqual(payload['event']['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
