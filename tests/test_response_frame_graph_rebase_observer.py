import json
import tempfile
import unittest
from pathlib import Path

from ollmo_services.response_frames import (
    load_latest_response_observation_state,
    load_response_frame_index,
    persist_response_frame,
    select_graph_rebase_observation_response_ids,
)


class GraphRebaseResponseFrameObserverTests(unittest.TestCase):
    @staticmethod
    def _frame(
        response_id: str,
        *,
        runtime: dict | None = None,
        request: dict | None = None,
        working_frame: dict | None = None,
        frame_relation: dict | None = None,
    ) -> dict:
        frame = {
            'frame_version': 9,
            'kind': 'ollmo.response_frame',
            'response_id': response_id,
            'status': 'completed',
            'object': 'response',
            'current_state': {
                'id': response_id,
                'status': 'completed',
                'lifecycle_state': 'completed',
            },
        }
        if runtime:
            frame['runtime'] = runtime
        if request:
            frame['request'] = request
        if working_frame:
            frame['working_frame'] = working_frame
        if frame_relation:
            frame['frame_relation'] = frame_relation
        return frame

    def test_selector_ignores_marker_words_in_prompts_and_generic_advisory_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                self._frame(
                    'resp_marker_words_only',
                    request={
                        'prompt': 'Explain the token graph_rebase_stage_successor.',
                    },
                    runtime={
                        'developer_diagnostics': {
                            'semantic_role_profile': {
                                'role_name': 'graph_rebase_proposals',
                                'evidence_refs': [
                                    'response_time_graph_rebase_candidate',
                                    'runtime_graph_rebase_reviews',
                                ],
                            }
                        }
                    },
                ),
                frames_dir=frames_dir,
            )

            selection = select_graph_rebase_observation_response_ids(
                frames_dir=frames_dir,
            )

        self.assertEqual(selection['selected_response_ids'], [])
        self.assertEqual(selection['selected_response_count'], 0)
        self.assertEqual(selection['scan_error_count'], 0)

    def test_selector_finds_structural_candidate_proposal_and_successor_relation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                self._frame(
                    'resp_candidate',
                    runtime={
                        'developer_diagnostics': {
                            'response_time_graph_rebase_candidate': {
                                'candidate': True,
                                'reason': 'additive_repair_insufficient',
                            }
                        }
                    },
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                self._frame(
                    'resp_proposal',
                    runtime={
                        'request_phase_graph': {
                            'kind': 'ollmo.request_phase_graph',
                            'graph_rebase_proposals': [
                                {
                                    'proposal_id': 'graph-rebase-proposal-1',
                                    'rebase_class': 'partial',
                                }
                            ],
                        }
                    },
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                self._frame(
                    'resp_successor_relation',
                    frame_relation={
                        'kind': 'graph_rebase_partial_successor',
                        'parent_response_id': 'resp_parent',
                        'parent_frame_id': 'resp_parent:frame-1',
                        'parent_frame_sequence': 1,
                    },
                ),
                frames_dir=frames_dir,
            )

            selection = select_graph_rebase_observation_response_ids(
                frames_dir=frames_dir,
            )

        self.assertEqual(
            set(selection['selected_response_ids']),
            {'resp_candidate', 'resp_proposal', 'resp_successor_relation'},
        )
        self.assertEqual(selection['selected_response_count'], 3)
        self.assertEqual(selection['scan_error_count'], 0)

    def test_bounded_loader_recovers_prompt_from_working_frame_content_snapshot(self):
        prompt = 'Create three branch-local image artifacts and preserve their dependencies.'
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                self._frame(
                    'resp_prompt_snapshot',
                    request={'request_meta': {'conversation_id': 'conv-observer'}},
                    working_frame={
                        'kind': 'ollmo.working_frame',
                        'status': 'frozen',
                        'request': {
                            'prompt': prompt,
                            'context_candidates': [
                                {'candidate_id': f'context-{index}', 'text': 'x' * 200}
                                for index in range(10)
                            ],
                        },
                    },
                    runtime={
                        'request_phase_graph': {
                            'kind': 'ollmo.request_phase_graph',
                            'graph_rebase_proposals': [
                                {
                                    'proposal_id': 'graph-rebase-proposal-prompt',
                                    'rebase_class': 'partial',
                                }
                            ],
                            'large_unrelated_graph_field': 'not projected ' * 10_000,
                        },
                        'developer_diagnostics': {
                            'runtime_graph_rebase_reviews': [
                                {'proposal_id': 'graph-rebase-proposal-prompt', 'status': 'accepted'}
                            ],
                            'large_unrelated_diagnostic_field': 'not projected ' * 10_000,
                        },
                        'large_unrelated_runtime_field': 'not projected ' * 10_000,
                    },
                ),
                frames_dir=frames_dir,
            )
            index = load_response_frame_index(frames_dir=frames_dir)
            manifest = index['responses']['resp_prompt_snapshot']['effective_snapshot_manifest']

            observed = load_latest_response_observation_state(
                'resp_prompt_snapshot',
                frames_dir=frames_dir,
                index_state=index,
            )

        self.assertIn('working_frame.request.content', manifest)
        self.assertTrue(observed['ok'])
        self.assertTrue(observed['bounded_observation'])
        payload = observed['response_payload']
        self.assertEqual(payload['request']['prompt'], prompt)
        self.assertEqual(
            payload['request']['request_meta']['conversation_id'],
            'conv-observer',
        )
        graph = payload['runtime']['request_phase_graph']
        diagnostics = payload['runtime']['developer_diagnostics']
        self.assertEqual(
            graph['graph_rebase_proposals'][0]['proposal_id'],
            'graph-rebase-proposal-prompt',
        )
        self.assertEqual(
            diagnostics['runtime_graph_rebase_reviews'][0]['status'],
            'accepted',
        )
        self.assertNotIn('large_unrelated_graph_field', graph)
        self.assertNotIn('large_unrelated_diagnostic_field', diagnostics)
        self.assertNotIn('large_unrelated_runtime_field', payload['runtime'])
        self.assertNotIn('working_frame', payload)

    def test_bounded_loader_rejects_stale_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            ledger_path = persist_response_frame(
                self._frame(
                    'resp_stale_observer_index',
                    runtime={
                        'developer_diagnostics': {
                            'response_time_graph_rebase_candidate': {'candidate': True},
                        }
                    },
                ),
                frames_dir=frames_dir,
            )
            stale_index = load_response_frame_index(frames_dir=frames_dir)
            with ledger_path.open('ab') as handle:
                handle.write(b'{}\n')

            observed = load_latest_response_observation_state(
                'resp_stale_observer_index',
                frames_dir=frames_dir,
                index_state=stale_index,
            )

        self.assertFalse(observed['ok'])
        self.assertEqual(observed['status_code'], 409)
        self.assertEqual(observed['error']['code'], 'response_frame_index_stale')

    def test_selector_and_bounded_loader_reject_tampered_snapshot_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                self._frame(
                    'resp_tampered_observer_snapshot',
                    runtime={
                        'developer_diagnostics': {
                            'response_time_graph_rebase_candidate': {'candidate': True},
                        }
                    },
                ),
                frames_dir=frames_dir,
            )
            index = load_response_frame_index(frames_dir=frames_dir)
            runtime_ref = index['responses']['resp_tampered_observer_snapshot'][
                'effective_snapshot_manifest'
            ]['runtime']
            snapshot_path = frames_dir / runtime_ref['path']
            snapshot_path.write_text(
                json.dumps(
                    {
                        'developer_diagnostics': {
                            'response_time_graph_rebase_candidate': {'candidate': False},
                        }
                    },
                    sort_keys=True,
                )
                + '\n',
                encoding='utf-8',
            )

            selection = select_graph_rebase_observation_response_ids(
                frames_dir=frames_dir,
                index_state=index,
            )
            observed = load_latest_response_observation_state(
                'resp_tampered_observer_snapshot',
                frames_dir=frames_dir,
                index_state=index,
            )

        self.assertEqual(selection['selected_response_ids'], [])
        self.assertGreaterEqual(selection['scan_error_count'], 1)
        self.assertIn(
            'response_frame_snapshot_digest_mismatch',
            {item['code'] for item in selection['scan_errors']},
        )
        self.assertFalse(observed['ok'])
        self.assertEqual(observed['status_code'], 409)
        self.assertEqual(
            observed['error']['code'],
            'response_frame_snapshot_digest_mismatch',
        )


if __name__ == '__main__':
    unittest.main()
