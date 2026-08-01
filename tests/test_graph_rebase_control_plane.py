import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import ollmo_webserver
from ollmo_services.graph_rebase import stable_graph_rebase_prompt_digest
from ollmo_services.graph_rebase_operator import GraphRebaseOperatorRegistryError
from ollmo_services.graph_rebase_readiness_registry import (
    load_graph_rebase_readiness_observations,
    load_graph_rebase_readiness_registry,
    sync_graph_rebase_readiness_epoch,
)
from ollmo_services.response_frames import (
    ResponseFrameParentCASMismatch,
    load_latest_response_state,
    persist_response_frame,
)


class GraphRebaseControlPlaneTests(unittest.TestCase):
    def setUp(self):
        self.operator_token = 'test-graph-rebase-operator-token-000000000000'
        self._prior_operator_token = ollmo_webserver.app.config.get(
            'GRAPH_REBASE_OPERATOR_TOKEN'
        )
        self._prior_operator_identity = ollmo_webserver.app.config.get(
            'GRAPH_REBASE_OPERATOR_IDENTITY'
        )
        ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_TOKEN'] = self.operator_token
        ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_IDENTITY'] = (
            'control-plane-test-operator'
        )
        ollmo_webserver.app.config['TESTING'] = True
        self.client = ollmo_webserver.app.test_client()
        self.expected = {
            'expected_response_id': 'resp-rebase-control',
            'expected_frame_id': 'frame-rebase-parent',
            'expected_proposal_id': 'proposal-partial-1',
            'expected_base_graph_digest': 'graph-base-1',
            'expected_candidate_graph_digest': 'graph-candidate-1',
            'expected_requested_rebase_class': 'partial_subtree_rebase',
            'expected_frame_sequence': 7,
        }
        self.response_payload = {
            'id': self.expected['expected_response_id'],
            'response_id': self.expected['expected_response_id'],
            'status': 'completed',
            'request': {'prompt': 'Create the reviewed graph from this root request.'},
            'response_frame': {
                'frame_id': self.expected['expected_frame_id'],
                'frame_sequence': self.expected['expected_frame_sequence'],
            },
            'runtime': {
                'request_phase_graph': {
                    'graph_digest': self.expected['expected_base_graph_digest'],
                    'graph_rebase_proposals': [
                        {
                            'proposal_id': self.expected['expected_proposal_id'],
                            'base_graph_digest': self.expected[
                                'expected_base_graph_digest'
                            ],
                            'candidate_graph_digest': self.expected[
                                'expected_candidate_graph_digest'
                            ],
                            'requested_rebase_class': self.expected[
                                'expected_requested_rebase_class'
                            ],
                        }
                    ],
                }
            },
        }

    def tearDown(self):
        ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_TOKEN'] = (
            self._prior_operator_token
        )
        ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_IDENTITY'] = (
            self._prior_operator_identity
        )

    def _post(self, action, **overrides):
        payload = {
            'action': action,
            'adjudication': 'useful_proposal'
            if action == 'adjudicate'
            else 'accepted',
            'reason': f'operator reason for {action}',
            'evidence_refs': ['frame:frame-rebase-parent', 'review:human-1'],
            **self.expected,
            **overrides,
        }
        return self.client.post(
            f"/api/responses/{self.expected['expected_response_id']}/graph_rebase/operator",
            json=payload,
            headers={
                'X-Ollmo-Graph-Rebase-Operator-Token': self.operator_token,
                'X-Ollmo-Graph-Rebase-Operator': 'control-plane-test-operator',
            },
        )

    @staticmethod
    def _readiness_report(*, shadow_ready=False, partial_ready=False):
        return {
            'kind': 'ollmo.graph_rebase_readiness_report',
            'schema_version': 1,
            'runtime_effect': 'none',
            'report_digest': 'readiness-report-1',
            'corpus': {
                'corpus_digest': 'readiness-corpus-1',
                'settled_final_response_count': 3,
            },
            'gates': {
                'shadow_to_stage': {
                    'ready': shadow_ready,
                    'decision': 'ready_for_stage'
                    if shadow_ready
                    else 'keep_shadow',
                },
                'partial_stage_to_apply_reviewed': {
                    'ready': partial_ready,
                    'decision': 'ready_for_exact_partial_apply_reviewed_authorization'
                    if partial_ready
                    else 'keep_partial_non_executable',
                },
            },
        }

    def test_readiness_get_returns_canonical_mocked_corpus_without_mutation(self):
        report = self._readiness_report()
        observer = {
            'kind': 'ollmo.graph_rebase_readiness_observer',
            'runtime_effect': 'none',
            'hydrated_response_count': 3,
            'trusted_operator_record_count': 2,
            'load_error_count': 0,
            'load_errors': [],
            'index_ok': True,
        }
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            return_value=(report, observer),
        ) as mock_corpus_helper, patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
        ) as mock_registry_mutation, patch.object(
            ollmo_webserver,
            '_persist_graph_rebase_stage_successor',
        ) as mock_stage_mutation, patch.object(
            ollmo_webserver._RESPONSES_REQUEST_RUNTIME,
            'prepare_terminal_partial_graph_rebase_successor',
        ) as mock_prepare, patch.object(
            ollmo_webserver._LATE_FILL_RUNTIME,
            'persist_and_schedule_partial_graph_rebase_successor',
        ) as mock_handoff:
            response = self.client.get('/api/graph_rebase/readiness')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['kind'], report['kind'])
        self.assertEqual(payload['report_digest'], report['report_digest'])
        self.assertEqual(payload['corpus'], report['corpus'])
        self.assertEqual(payload['observer'], observer)
        self.assertEqual(payload['runtime_effect'], 'none')
        mock_corpus_helper.assert_called_once_with()
        mock_registry_mutation.assert_not_called()
        mock_stage_mutation.assert_not_called()
        mock_prepare.assert_not_called()
        mock_handoff.assert_not_called()

    def test_readiness_get_fails_closed_when_evidence_registry_is_corrupt(self):
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            side_effect=ollmo_webserver.GraphRebaseReadinessRegistryError(
                'readiness_registry_corrupt_line',
                'Readiness registry line is corrupt.',
                details={'line_number': 2},
            ),
        ):
            response = self.client.get('/api/graph_rebase/readiness')

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertEqual(
            payload['error_detail']['code'],
            'readiness_registry_corrupt_line',
        )
        self.assertEqual(payload['error_detail']['details']['line_number'], 2)

    def test_runtime_readiness_overlays_registry_with_current_epoch_ids(self):
        historical = {
            'response_id': 'resp-historical',
            'frame_id': 'frame-historical',
        }
        stale_current = {
            'response_id': 'resp-current',
            'frame_id': 'frame-stale',
        }
        current = {
            'response_id': 'resp-current',
            'frame_id': 'frame-current',
        }
        registry_state = {
            'ok': True,
            'registry_path': 'state/graph_rebase/readiness_observations.jsonl',
            'registry_sha256': 'a' * 64,
            'unique_response_count': 2,
            'records': [
                {
                    'response_id': 'resp-historical',
                    'source_epoch': {'source_epoch_id': 'epoch-1'},
                    'observation': historical,
                },
                {
                    'response_id': 'resp-current',
                    'source_epoch': {'source_epoch_id': 'epoch-1'},
                    'observation': stale_current,
                },
            ],
        }
        index_state = {
            'ok': True,
            'response_map_digest': 'b' * 64,
            'ledger_line_count': 4,
            'ledger_size_bytes': 4096,
            'responses': {
                'resp-current': {},
                'resp-current-without-readiness': {},
            },
        }
        selection = {
            'selected_response_ids': ['resp-current'],
            'scan_error_count': 0,
            'scan_errors': [],
        }
        current_payload = {'id': 'resp-current'}
        expected_report = self._readiness_report()
        current_epoch_tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(current_epoch_tmpdir.cleanup)
        current_frames_dir = (
            Path(current_epoch_tmpdir.name) / 'state' / 'response_frames'
        )
        current_frames_dir.mkdir(parents=True, exist_ok=True)
        (current_frames_dir / 'current_index.json').write_text(
            json.dumps(index_state),
            encoding='utf-8',
        )
        with patch.object(
            ollmo_webserver,
            'RESPONSE_FRAMES_DIR',
            current_frames_dir,
        ), patch.object(
            ollmo_webserver,
            '_load_graph_rebase_readiness_registry',
            return_value=registry_state,
        ), patch.object(
            ollmo_webserver,
            '_load_response_frame_index',
            return_value=index_state,
        ), patch.object(
            ollmo_webserver,
            '_select_graph_rebase_observation_response_ids',
            return_value=selection,
        ), patch.object(
            ollmo_webserver,
            '_load_latest_response_observation_state',
            return_value={'ok': True, 'response_payload': current_payload},
        ), patch.object(
            ollmo_webserver,
            '_project_graph_rebase_readiness_observation',
            return_value=current,
        ), patch.object(
            ollmo_webserver,
            '_load_graph_rebase_operator_records',
            return_value=[],
        ), patch.object(
            ollmo_webserver,
            '_build_graph_rebase_readiness_report',
            return_value=expected_report,
        ) as mock_report:
            report, observer = ollmo_webserver._graph_rebase_runtime_readiness()

        self.assertEqual(report, expected_report)
        observations = mock_report.call_args.args[0]
        self.assertEqual(observations, [historical, current])
        self.assertNotIn(stale_current, observations)
        self.assertEqual(observer['registry_record_count'], 2)
        self.assertEqual(
            observer['registry_record_count_excluded_by_current_overlay'],
            1,
        )
        self.assertEqual(observer['current_epoch_observation_count'], 1)
        self.assertEqual(observer['combined_observation_count'], 2)
        self.assertEqual(
            observer['multi_epoch_source_identity']['kind'],
            'ollmo.graph_rebase_readiness_multi_epoch_source',
        )

    def test_settled_relevant_frame_is_registered_after_durable_append(self):
        projection = {
            'response_id': 'resp-current',
            'frame_id': 'frame-current',
            'ledger_sequence': 8,
            'runtime': {
                'developer_diagnostics': {
                    'runtime_graph_rebase_candidate_review': {
                        'status': 'not_proposed',
                    }
                }
            },
            'readiness_state': {
                'settled_final': True,
                'active_late_fill': False,
            },
        }
        verified_epoch = {
            'ok': True,
            'index_state': {
                'responses': {
                    'resp-current': {
                        'latest_frame_id': 'frame-current',
                        'latest_frame_sequence': 8,
                    }
                }
            },
            'source_frame_sha256_by_response': {
                'resp-current': 'c' * 64,
            },
        }
        with patch.object(
            ollmo_webserver,
            '_project_graph_rebase_readiness_observation',
            return_value=projection,
        ), patch.object(
            ollmo_webserver,
            '_verify_response_frame_epoch',
            return_value=verified_epoch,
        ), patch.object(
            ollmo_webserver,
            '_select_graph_rebase_observation_response_ids',
            return_value={
                'selected_response_ids': ['resp-current'],
                'scan_error_count': 0,
            },
        ), patch.object(
            ollmo_webserver,
            '_load_latest_response_observation_state',
            return_value={
                'ok': True,
                'response_payload': projection,
            },
        ), patch.object(
            ollmo_webserver,
            '_build_graph_rebase_source_epoch_identity',
            return_value={'source_epoch_id': 'epoch-current'},
        ), patch.object(
            ollmo_webserver,
            '_append_graph_rebase_readiness_observation',
            return_value={
                'ok': True,
                'status': 'appended',
                'appended_record_count': 1,
                'already_present_count': 0,
                'record_count': 38,
                'registry_sha256': 'd' * 64,
            },
        ) as mock_append:
            diagnostic = (
                ollmo_webserver._register_durable_graph_rebase_readiness_observation(
                    {'id': 'resp-current'}
                )
            )

        self.assertEqual(diagnostic['status'], 'appended')
        self.assertEqual(diagnostic['appended_record_count'], 1)
        mock_append.assert_called_once_with(
            projection,
            source_frame='c' * 64,
            source_epoch={'source_epoch_id': 'epoch-current'},
            verified_epoch=verified_epoch,
            frames_dir=ollmo_webserver.RESPONSE_FRAMES_DIR,
            registry_path=ollmo_webserver.GRAPH_REBASE_READINESS_REGISTRY_PATH,
        )

    def test_settled_ordinary_frame_does_not_scan_whole_epoch(self):
        projection = {
            'response_id': 'resp-ordinary',
            'frame_id': 'frame-ordinary',
            'ledger_sequence': 9,
            'runtime': {
                'request_phase_graph': {
                    'kind': 'ollmo.request_phase_graph',
                }
            },
            'readiness_state': {
                'settled_final': True,
                'active_late_fill': False,
            },
        }
        with patch.object(
            ollmo_webserver,
            '_project_graph_rebase_readiness_observation',
            return_value=projection,
        ), patch.object(
            ollmo_webserver,
            '_verify_response_frame_epoch',
        ) as mock_verify:
            diagnostic = (
                ollmo_webserver._register_durable_graph_rebase_readiness_observation(
                    {'id': 'resp-ordinary'}
                )
            )

        self.assertEqual(diagnostic['status'], 'not_relevant')
        mock_verify.assert_not_called()

    def test_alternate_frame_epoch_uses_isolated_readiness_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            ollmo_webserver,
            'RESPONSE_FRAMES_DIR',
            Path(tmpdir) / 'state' / 'response_frames',
        ):
            registry_path = (
                ollmo_webserver._effective_graph_rebase_readiness_registry_path()
            )

        self.assertEqual(
            registry_path,
            Path(tmpdir)
            / 'state'
            / 'graph_rebase'
            / 'readiness_observations.jsonl',
        )

    def test_missing_ledger_and_index_are_a_valid_empty_post_reset_epoch(self):
        historical = {
            'response_id': 'resp-historical-only',
            'frame_id': 'frame-historical-only',
        }
        registry_state = {
            'ok': True,
            'registry_path': 'state/graph_rebase/readiness_observations.jsonl',
            'registry_sha256': 'e' * 64,
            'unique_response_count': 1,
            'records': [
                {
                    'response_id': 'resp-historical-only',
                    'source_epoch': {'source_epoch_id': 'epoch-historical'},
                    'observation': historical,
                }
            ],
        }
        expected_report = self._readiness_report()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            ollmo_webserver,
            'RESPONSE_FRAMES_DIR',
            Path(tmpdir) / 'state' / 'response_frames',
        ), patch.object(
            ollmo_webserver,
            '_load_graph_rebase_readiness_registry',
            return_value=registry_state,
        ), patch.object(
            ollmo_webserver,
            '_load_response_frame_index',
        ) as mock_load_index, patch.object(
            ollmo_webserver,
            '_select_graph_rebase_observation_response_ids',
        ) as mock_select, patch.object(
            ollmo_webserver,
            '_load_graph_rebase_operator_records',
            return_value=[],
        ), patch.object(
            ollmo_webserver,
            '_build_graph_rebase_readiness_report',
            return_value=expected_report,
        ) as mock_report:
            report, observer = ollmo_webserver._graph_rebase_runtime_readiness()

        self.assertEqual(report, expected_report)
        self.assertEqual(mock_report.call_args.args[0], [historical])
        self.assertEqual(observer['current_epoch_status'], 'empty')
        self.assertEqual(observer['current_epoch_observation_count'], 0)
        self.assertEqual(observer['load_error_count'], 0)
        self.assertTrue(observer['index_ok'])
        mock_load_index.assert_not_called()
        mock_select.assert_not_called()

    def test_online_registry_append_matches_later_epoch_sync_projection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'state' / 'response_frames'
            registry_path = (
                root
                / 'state'
                / 'graph_rebase'
                / 'readiness_observations.jsonl'
            )
            response_id = 'resp-online-sync-idempotent'
            oversized_marker = 'raw-only-oversized-' * 1500
            persist_response_frame(
                {
                    'kind': 'ollmo.response_frame',
                    'frame_version': 9,
                    'response_id': response_id,
                    'status': 'completed',
                    'current_state': {
                        'id': response_id,
                        'status': 'completed',
                        'lifecycle_state': 'completed',
                    },
                    'request': {
                        'prompt': 'Test bounded graph-rebase online retention.',
                        'workload_family': 'online-sync-idempotence',
                    },
                    'runtime': {
                        'request_phase_graph': {
                            'kind': 'ollmo.request_phase_graph',
                            'graph_rebase_proposals': [
                                {
                                    'kind': 'ollmo.graph_rebase_proposal',
                                    'proposal_id': 'proposal-online-sync',
                                    'payload': oversized_marker,
                                }
                            ],
                        },
                        'developer_diagnostics': {
                            'runtime_graph_rebase_candidate_review': {
                                'status': 'not_proposed',
                                'reason': 'test-bounded-projection',
                            }
                        },
                    },
                },
                frames_dir=frames_dir,
            )
            full_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            self.assertTrue(full_state['ok'])

            with patch.object(
                ollmo_webserver,
                'RESPONSE_FRAMES_DIR',
                frames_dir,
            ), patch.object(
                ollmo_webserver,
                'GRAPH_REBASE_READINESS_REGISTRY_PATH',
                registry_path,
            ):
                diagnostic = (
                    ollmo_webserver._register_durable_graph_rebase_readiness_observation(
                        full_state['response_payload']
                    )
                )

            self.assertEqual(diagnostic['status'], 'appended')
            self.assertEqual(
                load_graph_rebase_readiness_registry(registry_path)['record_count'],
                1,
            )
            serialized_observation = json.dumps(
                load_graph_rebase_readiness_observations(registry_path)[0],
                ensure_ascii=False,
                sort_keys=True,
            )
            self.assertLess(len(serialized_observation), 5_000)
            self.assertNotIn(oversized_marker, serialized_observation)

            persist_response_frame(
                {
                    'kind': 'ollmo.response_frame',
                    'frame_version': 9,
                    'response_id': 'resp-unrelated-after-online',
                    'status': 'completed',
                    'current_state': {
                        'id': 'resp-unrelated-after-online',
                        'status': 'completed',
                        'lifecycle_state': 'completed',
                    },
                    'request': {'prompt': 'Unrelated later response.'},
                },
                frames_dir=frames_dir,
            )
            repeated = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
            )

            self.assertTrue(repeated['ok'])
            self.assertEqual(repeated['status'], 'unchanged')
            self.assertEqual(repeated['appended_record_count'], 0)
            self.assertEqual(repeated['already_present_count'], 1)
            self.assertEqual(
                load_graph_rebase_readiness_registry(registry_path)['record_count'],
                1,
            )

    def test_operator_mutations_require_explicit_credential_and_identity(self):
        path = (
            f"/api/responses/{self.expected['expected_response_id']}"
            '/graph_rebase/operator'
        )
        request_payload = {
            'action': 'adjudicate',
            'adjudication': 'useful_proposal',
            'reason': 'Exact review.',
            'evidence_refs': ['operator:review'],
            **self.expected,
        }
        prior_token = ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_TOKEN']
        ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_TOKEN'] = ''
        try:
            unconfigured = self.client.post(path, json=request_payload)
        finally:
            ollmo_webserver.app.config['GRAPH_REBASE_OPERATOR_TOKEN'] = prior_token
        self.assertEqual(unconfigured.status_code, 503)
        self.assertEqual(
            unconfigured.get_json()['error_detail']['code'],
            'graph_rebase_operator_credential_not_configured',
        )

        unauthorized = self.client.post(
            path,
            json=request_payload,
            headers={
                'X-Ollmo-Graph-Rebase-Operator-Token': 'wrong-token',
                'X-Ollmo-Graph-Rebase-Operator': 'control-plane-test-operator',
            },
        )
        self.assertEqual(unauthorized.status_code, 401)

        missing_identity = self.client.post(
            path,
            json=request_payload,
            headers={
                'X-Ollmo-Graph-Rebase-Operator-Token': self.operator_token,
            },
        )
        self.assertEqual(missing_identity.status_code, 400)
        self.assertEqual(
            missing_identity.get_json()['error_detail']['code'],
            'graph_rebase_operator_identity_invalid',
        )

        wrong_identity = self.client.post(
            path,
            json=request_payload,
            headers={
                'X-Ollmo-Graph-Rebase-Operator-Token': self.operator_token,
                'X-Ollmo-Graph-Rebase-Operator': 'different-local-caller',
            },
        )
        self.assertEqual(wrong_identity.status_code, 401)
        self.assertEqual(
            wrong_identity.get_json()['error_detail']['code'],
            'graph_rebase_operator_identity_mismatch',
        )

    def test_adjudicate_forwards_every_exact_cas_binding_to_registry(self):
        operator_record = {
            'kind': 'ollmo.graph_rebase_operator_record',
            'record_id': 'operator-review-1',
            'action': 'adjudicate',
            'adjudication': 'useful_proposal',
        }
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
            return_value=(self.response_payload, None, 200),
        ), patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
            return_value=operator_record,
        ) as mock_record:
            response = self._post(
                'adjudicate',
                resolves_record_id='operator-false-negative-1',
            )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'recorded')
        self.assertEqual(payload['runtime_effect'], 'none')
        self.assertEqual(payload['operator_record'], operator_record)
        mock_record.assert_called_once()
        args, kwargs = mock_record.call_args
        self.assertEqual(args, (self.response_payload,))
        self.assertEqual(kwargs['action'], 'adjudicate')
        self.assertEqual(kwargs['adjudication'], 'useful_proposal')
        self.assertEqual(kwargs['operator_identity'], 'control-plane-test-operator')
        self.assertEqual(
            kwargs['resolves_record_id'],
            'operator-false-negative-1',
        )
        for key, value in self.expected.items():
            self.assertEqual(kwargs[key], value)
        self.assertEqual(
            kwargs['evidence_refs'],
            ['frame:frame-rebase-parent', 'review:human-1'],
        )
        self.assertNotIn('trusted_partial_promotion_gate', self.expected)
        self.assertIsNone(kwargs['trusted_partial_promotion_gate'])

    def test_operator_endpoint_rejects_noncanonical_frame_sequences_before_lookup(self):
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
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
        ) as mock_payload:
            for sequence in invalid_sequences:
                with self.subTest(sequence=sequence):
                    response = self._post(
                        'adjudicate',
                        expected_frame_sequence=sequence,
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(
                        'positive JSON integer',
                        response.get_json()['error'],
                    )

        mock_payload.assert_not_called()

    def test_stage_red_readiness_gate_blocks_before_registry_mutation(self):
        report = self._readiness_report(shadow_ready=False)
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
            return_value=(self.response_payload, None, 200),
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            return_value=(report, {'runtime_effect': 'none'}),
        ), patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
        ) as mock_record, patch.object(
            ollmo_webserver,
            '_persist_graph_rebase_stage_successor',
        ) as mock_persist:
            response = self._post('stage')

        self.assertEqual(response.status_code, 409)
        self.assertIn('not ready', response.get_json()['error'])
        self.assertFalse(response.get_json()['gate']['ready'])
        mock_record.assert_not_called()
        mock_persist.assert_not_called()

    def test_authorize_partial_red_gate_blocks_before_registry_mutation(self):
        report = self._readiness_report(partial_ready=False)
        promotion_gate = {
            'kind': 'ollmo.graph_rebase_promotion_gate',
            'status': 'blocked',
            'decision': 'keep_partial_non_executable',
        }
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
            return_value=(self.response_payload, None, 200),
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            return_value=(report, {'runtime_effect': 'none'}),
        ), patch.object(
            ollmo_webserver,
            '_build_partial_graph_rebase_promotion_gate',
            return_value=promotion_gate,
        ) as mock_gate, patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
        ) as mock_record, patch.object(
            ollmo_webserver._RESPONSES_REQUEST_RUNTIME,
            'prepare_terminal_partial_graph_rebase_successor',
        ) as mock_prepare:
            response = self._post('authorize_partial')

        self.assertEqual(response.status_code, 409)
        self.assertIn('not ready', response.get_json()['error'])
        self.assertEqual(response.get_json()['gate'], promotion_gate)
        mock_gate.assert_called_once_with(report)
        mock_record.assert_not_called()
        mock_prepare.assert_not_called()

    def test_authorize_partial_explicit_off_blocks_before_registry_mutation(self):
        with patch.dict(
            ollmo_webserver.os.environ,
            {'OLLMO_GRAPH_REBASE_AUTONOMY': 'off'},
            clear=True,
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
        ) as mock_readiness, patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
        ) as mock_payload, patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
        ) as mock_record, patch.object(
            ollmo_webserver._RESPONSES_REQUEST_RUNTIME,
            'prepare_terminal_partial_graph_rebase_successor',
        ) as mock_prepare:
            response = self._post('authorize_partial')

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertEqual(
            payload['error_detail']['code'],
            'graph_rebase_autonomy_off',
        )
        self.assertEqual(payload['autonomy']['autonomy_level'], 'off')
        mock_readiness.assert_not_called()
        mock_payload.assert_not_called()
        mock_record.assert_not_called()
        mock_prepare.assert_not_called()

    def test_stage_explicit_off_blocks_before_registry_or_frame_mutation(self):
        with patch.dict(
            ollmo_webserver.os.environ,
            {'OLLMO_GRAPH_REBASE_AUTONOMY': 'off'},
            clear=True,
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
        ) as mock_readiness, patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
        ) as mock_payload, patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
        ) as mock_record, patch.object(
            ollmo_webserver,
            '_persist_graph_rebase_stage_successor',
        ) as mock_persist:
            response = self._post('stage')

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertEqual(
            payload['error_detail']['code'],
            'graph_rebase_autonomy_off',
        )
        self.assertEqual(payload['autonomy']['autonomy_level'], 'off')
        mock_readiness.assert_not_called()
        mock_payload.assert_not_called()
        mock_record.assert_not_called()
        mock_persist.assert_not_called()

    def test_stage_atomic_parent_cas_race_fails_closed(self):
        lifecycle = {
            'status': 'staged',
            'proposal_id': self.expected['expected_proposal_id'],
            'rebase_id': 'rebase-stage-cas',
        }
        application = {
            'status': 'staged',
            'graph': copy.deepcopy(
                self.response_payload['runtime']['request_phase_graph']
            ),
        }
        mismatch = ResponseFrameParentCASMismatch(
            response_id=self.expected['expected_response_id'],
            expected_parent_frame_id=self.expected['expected_frame_id'],
            current_parent_frame_id='frame-won-by-another-successor',
            expected_parent_frame_sequence=self.expected['expected_frame_sequence'],
            current_parent_frame_sequence=self.expected['expected_frame_sequence'] + 1,
        )
        with patch.object(
            ollmo_webserver,
            '_validate_graph_rebase_proposal',
            return_value={'status': 'accepted'},
        ), patch.object(
            ollmo_webserver,
            '_build_graph_rebase_lifecycle',
            return_value=lifecycle,
        ), patch.object(
            ollmo_webserver,
            '_apply_validated_graph_rebase',
            return_value=application,
        ), patch.object(
            ollmo_webserver,
            '_load_latest_response_state',
            return_value={
                'ok': True,
                'response_frame': copy.deepcopy(
                    self.response_payload['response_frame']
                ),
            },
        ), patch.object(
            ollmo_webserver,
            '_finalize_response_frame_payload',
            side_effect=mismatch,
        ) as mock_finalize:
            result = ollmo_webserver._persist_graph_rebase_stage_successor(
                self.response_payload,
                proposal_id=self.expected['expected_proposal_id'],
                operator_record={'record_id': 'operator-stage-cas'},
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['blocked_reasons'], ['response_frame_parent_stale'])
        self.assertEqual(
            result['current_parent_frame_id'],
            'frame-won-by-another-successor',
        )
        mock_finalize.assert_called_once()
        self.assertEqual(
            mock_finalize.call_args.kwargs['expected_parent_frame_id'],
            self.expected['expected_frame_id'],
        )
        self.assertEqual(
            mock_finalize.call_args.kwargs['expected_parent_frame_sequence'],
            self.expected['expected_frame_sequence'],
        )
        self.assertEqual(
            mock_finalize.call_args.kwargs['request_payload'],
            self.response_payload['request'],
        )

    def test_stage_projects_durable_audit_only_successor(self):
        report = self._readiness_report(shadow_ready=True)
        operator_record = {
            'kind': 'ollmo.graph_rebase_operator_record',
            'record_id': 'operator-stage-1',
            'action': 'stage',
            'status': 'staged',
        }
        staged = {
            'status': 'staged',
            'response_payload': {
                'response_id': self.expected['expected_response_id'],
                'response_frame': {
                    'frame_id': 'frame-stage-successor',
                    'frame_relation': {
                        'kind': 'graph_rebase_stage_successor',
                        'parent_frame_id': self.expected['expected_frame_id'],
                    },
                },
            },
            'lifecycle': {
                'status': 'staged',
                'runtime_effect': 'staged_no_executable_mutation',
            },
        }
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
            return_value=(self.response_payload, None, 200),
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            return_value=(report, {'runtime_effect': 'none'}),
        ), patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
            return_value=operator_record,
        ) as mock_record, patch.object(
            ollmo_webserver,
            '_persist_graph_rebase_stage_successor',
            return_value=staged,
        ) as mock_persist, patch.object(
            ollmo_webserver._LATE_FILL_RUNTIME,
            'persist_and_schedule_partial_graph_rebase_successor',
        ) as mock_handoff:
            response = self._post('stage')

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'staged')
        self.assertEqual(payload['runtime_effect'], 'staged_no_executable_mutation')
        self.assertEqual(
            payload['response_frame']['frame_relation']['kind'],
            'graph_rebase_stage_successor',
        )
        mock_record.assert_called_once()
        mock_persist.assert_called_once_with(
            self.response_payload,
            proposal_id=self.expected['expected_proposal_id'],
            operator_record=operator_record,
        )
        mock_handoff.assert_not_called()

    def test_authorize_partial_rejoins_trusted_authorization_then_hands_off(self):
        report = self._readiness_report(partial_ready=True)
        promotion_gate = {
            'kind': 'ollmo.graph_rebase_promotion_gate',
            'gate_id': 'partial-gate-1',
            'gate': 'partial_stage_to_apply_reviewed',
            'status': 'ready',
            'decision': 'promote',
        }
        operator_record = {
            'kind': 'ollmo.graph_rebase_operator_record',
            'record_id': 'operator-authorize-1',
            'action': 'authorize_partial',
            'status': 'accepted',
        }
        trusted_authorization = {
            'kind': 'ollmo.graph_rebase_authorization',
            'registry_record_id': operator_record['record_id'],
            'response_id': self.expected['expected_response_id'],
            'frame_id': self.expected['expected_frame_id'],
            'proposal_id': self.expected['expected_proposal_id'],
            'requested_rebase_class': 'partial_subtree_rebase',
        }
        prepared = {
            'status': 'queued',
            'execution': {
                'execution_key': 'partial-rebase-execution-1',
                'proposal_id': self.expected['expected_proposal_id'],
            },
            'response_payload': {'response_id': self.expected['expected_response_id']},
            'artifact_gap': {'trigger': 'graph_rebase_partial_successor'},
        }
        successor_frame = {
            'frame_id': 'frame-partial-successor',
            'frame_relation': {
                'kind': 'graph_rebase_partial_successor',
                'parent_frame_id': self.expected['expected_frame_id'],
                'execution_key': 'partial-rebase-execution-1',
            },
        }
        handoff = {
            'status': 'queued',
            'execution': prepared['execution'],
            'response_payload': {
                'response_id': self.expected['expected_response_id'],
                'response_frame': successor_frame,
            },
            'scheduled': True,
        }
        route_payload = {'route_source': 'durable-parent-route'}
        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
            return_value=(self.response_payload, None, 200),
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            return_value=(report, {'runtime_effect': 'none'}),
        ), patch.object(
            ollmo_webserver,
            '_build_partial_graph_rebase_promotion_gate',
            return_value=promotion_gate,
        ), patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
            return_value=operator_record,
        ) as mock_record, patch.object(
            ollmo_webserver,
            '_find_trusted_graph_rebase_authorization',
            return_value=trusted_authorization,
        ) as mock_join, patch.object(
            ollmo_webserver._RESPONSES_REQUEST_RUNTIME,
            'prepare_terminal_partial_graph_rebase_successor',
            return_value=prepared,
        ) as mock_prepare, patch.object(
            ollmo_webserver,
            '_get_response_lookup_record',
            return_value={'route_payload': route_payload},
        ), patch.object(
            ollmo_webserver._LATE_FILL_RUNTIME,
            'persist_and_schedule_partial_graph_rebase_successor',
            return_value=handoff,
        ) as mock_handoff:
            response = self._post('authorize_partial')

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'queued')
        self.assertEqual(
            payload['runtime_effect'],
            'branch_local_partial_successor_queued',
        )
        self.assertTrue(payload['scheduled'])
        self.assertEqual(payload['execution'], prepared['execution'])
        self.assertEqual(payload['response_frame'], successor_frame)
        mock_record.assert_called_once()
        self.assertEqual(
            mock_record.call_args.kwargs['trusted_partial_promotion_gate'],
            promotion_gate,
        )
        mock_join.assert_called_once_with(
            response_id=self.expected['expected_response_id'],
            frame_id=self.expected['expected_frame_id'],
            proposal_id=self.expected['expected_proposal_id'],
            base_graph_digest=self.expected['expected_base_graph_digest'],
            candidate_graph_digest=self.expected['expected_candidate_graph_digest'],
            requested_rebase_class='partial_subtree_rebase',
            registry_path=ollmo_webserver.GRAPH_REBASE_OPERATOR_REGISTRY_PATH,
        )
        mock_prepare.assert_called_once_with(
            self.response_payload,
            proposal_id=self.expected['expected_proposal_id'],
            trusted_authorization=trusted_authorization,
            graph_rebase_autonomy='apply_reviewed',
        )
        mock_handoff.assert_called_once_with(
            prepared,
            source_route_payload=route_payload,
        )

    def test_authorize_partial_never_crosses_full_rebase_registry_boundary(self):
        report = self._readiness_report(partial_ready=True)
        promotion_gate = {
            'kind': 'ollmo.graph_rebase_promotion_gate',
            'status': 'ready',
            'decision': 'promote',
        }

        def reject_full(*_args, **kwargs):
            self.assertEqual(
                kwargs['expected_requested_rebase_class'],
                'full_successor_rebase',
            )
            raise GraphRebaseOperatorRegistryError(
                'full_successor_rebase_authorization_forbidden',
                status_code=403,
            )

        with patch.object(
            ollmo_webserver,
            '_graph_rebase_payload_for_operator',
            return_value=(self.response_payload, None, 200),
        ), patch.object(
            ollmo_webserver,
            '_graph_rebase_runtime_readiness',
            return_value=(report, {'runtime_effect': 'none'}),
        ), patch.object(
            ollmo_webserver,
            '_build_partial_graph_rebase_promotion_gate',
            return_value=promotion_gate,
        ), patch.object(
            ollmo_webserver,
            '_record_graph_rebase_operator_action',
            side_effect=reject_full,
        ) as mock_record, patch.object(
            ollmo_webserver,
            '_find_trusted_graph_rebase_authorization',
        ) as mock_join, patch.object(
            ollmo_webserver._RESPONSES_REQUEST_RUNTIME,
            'prepare_terminal_partial_graph_rebase_successor',
        ) as mock_prepare, patch.object(
            ollmo_webserver._LATE_FILL_RUNTIME,
            'persist_and_schedule_partial_graph_rebase_successor',
        ) as mock_handoff:
            response = self._post(
                'authorize_partial',
                expected_requested_rebase_class='full_successor_rebase',
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.get_json()['error_detail']['code'],
            'full_successor_rebase_authorization_forbidden',
        )
        mock_record.assert_called_once()
        mock_join.assert_not_called()
        mock_prepare.assert_not_called()
        mock_handoff.assert_not_called()

    def test_operator_payload_uses_only_exact_durable_projections(self):
        frame = {
            'kind': 'ollmo.response_frame',
            'response_id': 'resp-durable-operator',
            'frame_id': 'frame-durable-operator-2',
            'frame_sequence': 2,
        }
        full_state = {
            'ok': True,
            'errors': [],
            'response_frame': copy.deepcopy(frame),
            'response_payload': {
                'id': 'resp-durable-operator',
                'response_id': 'resp-durable-operator',
                'runtime': {'request_phase_graph': {'kind': 'ollmo.request_phase_graph'}},
            },
        }
        observation_state = {
            'ok': True,
            'response_frame': copy.deepcopy(frame),
            'response_payload': {
                'request': {'prompt': 'Exact inherited durable root.'},
            },
        }

        with patch.object(
            ollmo_webserver,
            '_load_latest_response_state',
            return_value=full_state,
        ) as load_full, patch.object(
            ollmo_webserver,
            '_load_latest_response_observation_state',
            return_value=observation_state,
        ) as load_observation, patch.object(
            ollmo_webserver,
            '_get_response_lookup_record',
        ) as lookup:
            payload, error, status = ollmo_webserver._graph_rebase_payload_for_operator(
                'resp-durable-operator'
            )

        self.assertEqual(status, 200)
        self.assertIsNone(error)
        self.assertEqual(payload['response_frame'], frame)
        self.assertEqual(
            payload['request'],
            {'prompt': 'Exact inherited durable root.'},
        )
        load_full.assert_called_once()
        load_observation.assert_called_once()
        lookup.assert_not_called()

    def test_operator_payload_rejects_durable_projection_frame_drift(self):
        full_state = {
            'ok': True,
            'errors': [],
            'response_frame': {
                'frame_id': 'frame-durable-operator-2',
                'frame_sequence': 2,
            },
            'response_payload': {'id': 'resp-durable-operator'},
        }
        observation_state = {
            'ok': True,
            'response_frame': {
                'frame_id': 'frame-durable-operator-1',
                'frame_sequence': 1,
            },
            'response_payload': {'request': {'prompt': 'Stale root.'}},
        }

        with patch.object(
            ollmo_webserver,
            '_load_latest_response_state',
            return_value=full_state,
        ), patch.object(
            ollmo_webserver,
            '_load_latest_response_observation_state',
            return_value=observation_state,
        ):
            payload, error, status = ollmo_webserver._graph_rebase_payload_for_operator(
                'resp-durable-operator'
            )

        self.assertIsNone(payload)
        self.assertEqual(status, 409)
        self.assertEqual(
            error['code'],
            'graph_rebase_operator_durable_truth_binding_mismatch',
        )


class PartialGraphRebaseDurableHandoffTests(unittest.TestCase):
    def setUp(self):
        self.owner = ollmo_webserver._LATE_FILL_RUNTIME
        self.response_id = 'resp-partial-handoff'
        self.parent_frame_id = 'frame-parent-handoff'
        self.execution_key = 'partial-execution-handoff-1'
        self.root_prompt = 'Create the exact parent request and its bounded successor.'
        self.execution = {
            'kind': 'ollmo.graph_rebase_partial_successor_execution',
            'execution_key': self.execution_key,
            'response_id': self.response_id,
            'parent_frame_id': self.parent_frame_id,
            'parent_frame_sequence': 4,
            'proposal_id': 'proposal-partial-handoff',
            'authorization_record_id': 'operator-authorization-handoff',
            'scheduled_branch_ids': ['branch-partial-new'],
        }
        self.relation = {
            'kind': 'graph_rebase_partial_successor',
            'response_id': self.response_id,
            'parent_frame_id': self.parent_frame_id,
            'execution_key': self.execution_key,
            'proposal_id': self.execution['proposal_id'],
        }
        self.successor_payload = {
            'id': self.response_id,
            'response_id': self.response_id,
            'status': 'completed',
            'frame_relation': copy.deepcopy(self.relation),
            'late_fill': {
                'status': 'pending',
                'partial_rebase_execution': copy.deepcopy(self.execution),
                'pending_branches': [
                    {
                        'branch_id': 'branch-partial-new',
                        'capability': 'chat',
                        'content_payload': 'bounded branch-local work',
                    }
                ],
            },
            'runtime': {
                'request_phase_graph': {
                    'successor_rebase_executions': [copy.deepcopy(self.execution)]
                }
            },
        }
        self.prepared = {
            'status': 'queued',
            'response_payload': copy.deepcopy(self.successor_payload),
            'artifact_gap': {
                'trigger': 'graph_rebase_partial_successor',
                'branch_id': 'branch-partial-new',
                'capability': 'chat',
                'content_payload': 'bounded branch-local work',
                'execution_contract': {
                    'forbidden_root_prompt_digest': (
                        stable_graph_rebase_prompt_digest(self.root_prompt)
                    ),
                },
            },
            'execution': copy.deepcopy(self.execution),
        }
        self.parent_payload = {
            'id': self.response_id,
            'response_id': self.response_id,
            'status': 'completed',
            'request': {'prompt': self.root_prompt},
            'response_frame': {
                'frame_id': self.parent_frame_id,
                'frame_sequence': 4,
                'request': {'prompt': self.root_prompt},
            },
        }

    def _parent_state(self):
        return {
            'ok': True,
            'response_frame': copy.deepcopy(self.parent_payload['response_frame']),
            'response_payload': copy.deepcopy(self.parent_payload),
        }

    def _parent_observation_state(self):
        return {
            'ok': True,
            'response_frame': {
                'frame_id': self.parent_frame_id,
                'frame_sequence': 4,
            },
            'response_payload': {
                'request': {'prompt': self.root_prompt},
            },
        }

    def _finalized_successor(self):
        payload = copy.deepcopy(self.successor_payload)
        payload['response_frame'] = {
            'frame_id': 'frame-successor-handoff',
            'frame_sequence': 5,
            'frame_relation': copy.deepcopy(self.relation),
        }
        return payload

    def test_persist_is_verified_before_root_guarded_schedule(self):
        events = []
        scheduled_calls = []
        finalized = self._finalized_successor()

        def finalize(
            payload,
            *,
            request_payload,
            persist,
            expected_parent_frame_id,
            expected_parent_frame_sequence,
        ):
            events.append('finalize')
            self.assertTrue(persist)
            self.assertEqual(request_payload, {'prompt': self.root_prompt})
            self.assertEqual(expected_parent_frame_id, self.parent_frame_id)
            self.assertEqual(expected_parent_frame_sequence, 4)
            self.assertEqual(payload['frame_relation'], self.relation)
            self.assertEqual(
                payload['late_fill']['partial_rebase_execution'],
                self.execution,
            )
            return copy.deepcopy(finalized)

        load_count = 0

        def load_latest(response_id):
            nonlocal load_count
            load_count += 1
            self.assertEqual(response_id, self.response_id)
            if load_count == 1:
                events.append('load_parent')
                return self._parent_state()
            events.append('verify_durable')
            return {
                'ok': True,
                'response_frame': copy.deepcopy(finalized['response_frame']),
                'response_payload': copy.deepcopy(finalized),
            }

        def touch(*args, **kwargs):
            events.append('touch_lookup')
            self.assertEqual(args[0], self.response_id)
            self.assertEqual(kwargs['response_payload'], finalized)

        def schedule(**kwargs):
            events.append('schedule')
            scheduled_calls.append(copy.deepcopy(kwargs))
            self.assertEqual(
                kwargs['request_payload'],
                {'prompt': self.root_prompt},
            )
            self.assertEqual(kwargs['assistant_message'], '')
            frame_relation = kwargs['response_payload']['response_frame'][
                'frame_relation'
            ]
            self.assertEqual(frame_relation, self.relation)
            self.assertEqual(
                kwargs['response_payload']['late_fill'][
                    'partial_rebase_execution'
                ],
                self.execution,
            )
            return True

        source_route = {'route_source': 'durable-parent-route'}
        with patch.object(
            self.owner,
            'finalize_response_frame_payload',
            side_effect=finalize,
        ), patch.object(
            self.owner,
            'load_latest_response_state',
            side_effect=load_latest,
        ), patch.object(
            self.owner,
            'load_latest_response_observation_state',
            return_value=self._parent_observation_state(),
        ), patch.object(
            self.owner,
            'touch_response_lookup',
            side_effect=touch,
        ), patch.object(
            self.owner,
            'schedule_response_late_fill',
            side_effect=schedule,
        ), patch.object(
            self.owner,
            'get_response_lookup_record',
        ) as lookup, patch.object(self.owner, 'log_unified_event'):
            result = self.owner.persist_and_schedule_partial_graph_rebase_successor(
                self.prepared,
                source_route_payload=source_route,
            )

        self.assertEqual(result['status'], 'queued')
        self.assertTrue(result['scheduled'])
        self.assertEqual(result['execution_key'], self.execution_key)
        self.assertEqual(result['execution'], self.execution)
        self.assertEqual(
            events,
            ['load_parent', 'finalize', 'verify_durable', 'touch_lookup', 'schedule'],
        )
        self.assertEqual(len(scheduled_calls), 1)
        self.assertEqual(scheduled_calls[0]['source_route_payload'], source_route)
        self.assertEqual(
            result['response_payload']['response_frame']['frame_relation'],
            self.relation,
        )
        lookup.assert_not_called()

    def test_duplicate_durable_execution_is_idempotent_and_never_rescheduled(self):
        durable_successor = self._finalized_successor()
        with patch.object(
            self.owner,
            'finalize_response_frame_payload',
        ) as mock_finalize, patch.object(
            self.owner,
            'load_latest_response_state',
            return_value={
                'ok': True,
                'response_frame': copy.deepcopy(durable_successor['response_frame']),
                'response_payload': copy.deepcopy(durable_successor),
            },
        ) as mock_load, patch.object(
            self.owner,
            'touch_response_lookup',
        ) as mock_touch, patch.object(
            self.owner,
            'schedule_response_late_fill',
        ) as mock_schedule:
            result = self.owner.persist_and_schedule_partial_graph_rebase_successor(
                self.prepared
            )

        self.assertEqual(result['status'], 'already_recorded')
        self.assertFalse(result['scheduled'])
        self.assertEqual(result['execution_key'], self.execution_key)
        self.assertEqual(result['response_payload'], durable_successor)
        mock_finalize.assert_not_called()
        mock_load.assert_called_once_with(self.response_id)
        mock_touch.assert_not_called()
        mock_schedule.assert_not_called()

    def test_atomic_parent_cas_race_never_schedules_partial_successor(self):
        mismatch = ResponseFrameParentCASMismatch(
            response_id=self.response_id,
            expected_parent_frame_id=self.parent_frame_id,
            current_parent_frame_id='frame-won-by-another-successor',
            expected_parent_frame_sequence=4,
            current_parent_frame_sequence=5,
        )
        with patch.object(
            self.owner,
            'finalize_response_frame_payload',
            side_effect=mismatch,
        ) as mock_finalize, patch.object(
            self.owner,
            'load_latest_response_state',
            return_value=self._parent_state(),
        ) as mock_load, patch.object(
            self.owner,
            'load_latest_response_observation_state',
            return_value=self._parent_observation_state(),
        ), patch.object(
            self.owner,
            'touch_response_lookup',
        ) as mock_touch, patch.object(
            self.owner,
            'schedule_response_late_fill',
        ) as mock_schedule:
            result = self.owner.persist_and_schedule_partial_graph_rebase_successor(
                self.prepared
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['blocked_reasons'], ['response_frame_parent_stale'])
        self.assertEqual(
            result['current_parent_frame_id'],
            'frame-won-by-another-successor',
        )
        mock_finalize.assert_called_once()
        self.assertEqual(
            mock_finalize.call_args.kwargs['expected_parent_frame_id'],
            self.parent_frame_id,
        )
        self.assertEqual(
            mock_finalize.call_args.kwargs['expected_parent_frame_sequence'],
            4,
        )
        mock_load.assert_called_once_with(self.response_id)
        mock_touch.assert_not_called()
        mock_schedule.assert_not_called()

    def test_durable_root_guard_drift_never_persists_or_schedules(self):
        observation_state = self._parent_observation_state()
        observation_state['response_payload']['request']['prompt'] = (
            'A different durable current root.'
        )
        with patch.object(
            self.owner,
            'load_latest_response_state',
            return_value=self._parent_state(),
        ), patch.object(
            self.owner,
            'load_latest_response_observation_state',
            return_value=observation_state,
        ), patch.object(
            self.owner,
            'finalize_response_frame_payload',
        ) as mock_finalize, patch.object(
            self.owner,
            'schedule_response_late_fill',
        ) as mock_schedule:
            result = self.owner.persist_and_schedule_partial_graph_rebase_successor(
                self.prepared
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(
            result['blocked_reasons'],
            ['partial_rebase_root_prompt_guard_mismatch'],
        )
        mock_finalize.assert_not_called()
        mock_schedule.assert_not_called()

    def test_failed_durable_execution_verification_never_schedules(self):
        finalized = self._finalized_successor()
        durable_without_execution = copy.deepcopy(finalized)
        durable_without_execution['late_fill'].pop('partial_rebase_execution')
        durable_without_execution['runtime']['request_phase_graph'][
            'successor_rebase_executions'
        ] = []
        with patch.object(
            self.owner,
            'finalize_response_frame_payload',
            return_value=finalized,
        ) as mock_finalize, patch.object(
            self.owner,
            'load_latest_response_state',
            side_effect=[
                self._parent_state(),
                {
                    'ok': True,
                    'response_frame': copy.deepcopy(finalized['response_frame']),
                    'response_payload': durable_without_execution,
                },
            ],
        ) as mock_load, patch.object(
            self.owner,
            'load_latest_response_observation_state',
            return_value=self._parent_observation_state(),
        ), patch.object(
            self.owner,
            'touch_response_lookup',
        ) as mock_touch, patch.object(
            self.owner,
            'schedule_response_late_fill',
        ) as mock_schedule, patch.object(self.owner, 'log_unified_event'):
            result = self.owner.persist_and_schedule_partial_graph_rebase_successor(
                self.prepared
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(
            result['blocked_reasons'],
            ['partial_rebase_successor_not_durable'],
        )
        mock_finalize.assert_called_once()
        self.assertEqual(mock_load.call_count, 2)
        mock_touch.assert_not_called()
        mock_schedule.assert_not_called()


if __name__ == '__main__':
    unittest.main()
