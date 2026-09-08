import tempfile
import unittest
from pathlib import Path

from ollmo_services.frame_planning import build_artifact_flow_plan
from ollmo_services.response_artifact_bundles import bundle_response_artifacts
from ollmo_services.responses import filter_public_response_artifacts
from ollmo_webserver import _LATE_FILL_RUNTIME


class CurrentResponseArtifactAuthorityTests(unittest.TestCase):
    @staticmethod
    def _text_branch_spec(branch_id, source_name='app', extension='js'):
        branch = {
            'branch_id': branch_id,
            'phase_id': f'phase-{branch_id}',
            'capability': 'chat',
            'output_type': 'text',
            'stage_direction': 'materialize_requested_text_artifact',
            'requires_artifact': True,
            'text_artifact_extension': extension,
            'text_artifact_source_name': source_name,
            'artifact_request': {
                'extension': extension,
                'source_name': source_name,
                'source': 'explicit_filename',
            },
        }
        return {
            'branch_id': branch_id,
            'phase_id': branch['phase_id'],
            'capability': 'chat',
            'reservation_group': 'chat',
            'branch': branch,
            'prepare_args': {
                'expected_capability': 'chat',
                'artifact_gap': dict(branch),
            },
        }

    def test_duplicate_logical_text_branches_coalesce_to_one_request(self):
        coalesced = _LATE_FILL_RUNTIME._coalesce_required_text_artifact_branch_specs(
            [
                self._text_branch_spec('branch-app-primary'),
                self._text_branch_spec('branch-app-retry'),
            ]
        )
        self.assertEqual(len(coalesced), 1)
        self.assertEqual(len(coalesced[0]['text_artifact_requests']), 1)
        self.assertEqual(
            [item['branch_id'] for item in coalesced[0]['coalesced_text_artifact_branches']],
            ['branch-app-primary', 'branch-app-retry'],
        )

    def test_fulfilled_repair_contract_is_pruned_while_unrelated_repair_remains(self):
        image_contract = {
            'contract_id': 'contract-image',
            'branch_id': 'branch-image-1',
            'phase_id': 'phase-image-1',
            'capability': 'image_generation',
            'output_type': 'image',
            'repair_action': 'retry_same_branch',
            'auto_execute': True,
            'repair_work_available': True,
        }
        styles_contract = {
            'contract_id': 'contract-styles',
            'branch_id': 'branch-styles',
            'phase_id': 'phase-styles',
            'capability': 'chat',
            'output_type': 'text',
            'repair_action': 'rebind_dependency_evidence',
            'auto_execute': True,
            'repair_work_available': True,
        }
        late_fill = {
            'status': 'partial_failed',
            'completed_branches': [
                {
                    'branch_id': 'branch-image-1',
                    'phase_id': 'phase-image-1',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'fulfilled',
                }
            ],
            'repair_action': 'retry_same_branch',
            'repair_actions': ['retry_same_branch', 'rebind_dependency_evidence'],
            'repair_rebuild_contracts': [image_contract, styles_contract],
            'repair_loop': {
                'status': 'promoted',
                'promoted_contracts': [image_contract, styles_contract],
                'promoted_contract_count': 2,
                'executable_contract_count': 2,
                'repair_work_available': True,
                'repair_work_available_count': 2,
                'next_actions': ['retry_same_branch', 'rebind_dependency_evidence'],
            },
            'ghost_repair_feedback': {
                'status': 'repair_required',
                'items': [image_contract, styles_contract],
                'repair_rebuild_contracts': [image_contract, styles_contract],
            },
        }

        reconciled = _LATE_FILL_RUNTIME._reconcile_terminal_satisfied_repair_loop(late_fill)

        self.assertEqual(
            [item['contract_id'] for item in reconciled['repair_loop']['promoted_contracts']],
            ['contract-styles'],
        )
        self.assertEqual(
            [item['contract_id'] for item in reconciled['ghost_repair_feedback']['items']],
            ['contract-styles'],
        )
        self.assertEqual(reconciled['repair_actions'], ['rebind_dependency_evidence'])
        self.assertEqual(reconciled['repair_loop']['resolved_contract_count'], 1)

    def test_canonical_text_evidence_ignores_stale_linked_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current = root / '20260904_styles.css'
            stale = root / 'styles.css'
            current.write_text('body { color: green; }', encoding='utf-8')
            stale.write_text('body { color: red; }', encoding='utf-8')
            payload = {
                'id': 'resp_current_styles',
                'artifacts': [
                    {
                        'type': 'text',
                        'path': str(stale),
                        'name': 'styles',
                        'origin': 'linked_public_dependency',
                    },
                    {
                        'type': 'text',
                        'path': str(current),
                        'name': 'styles',
                        'origin': 'assistant_output',
                        'source_response_id': 'resp_current_styles',
                        'branch_id': 'branch-styles',
                    },
                ],
            }

            result = _LATE_FILL_RUNTIME._canonical_text_artifact_saved_result(
                payload,
                extension='css',
                source_name='styles',
            )

            self.assertEqual(result['saved_text_path'], str(current))
            self.assertIn('color: green', result['content'])

    def test_canonical_text_evidence_keeps_current_conflict_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / '20260904_a_styles.css'
            second = root / '20260904_b_styles.css'
            first.write_text('body { color: green; }', encoding='utf-8')
            second.write_text('body { color: blue; }', encoding='utf-8')
            payload = {
                'id': 'resp_conflicting_styles',
                'artifacts': [
                    {
                        'type': 'text',
                        'path': str(path),
                        'name': 'styles',
                        'origin': 'assistant_output',
                        'source_response_id': 'resp_conflicting_styles',
                        'branch_id': branch_id,
                    }
                    for path, branch_id in (
                        (first, 'branch-styles-a'),
                        (second, 'branch-styles-b'),
                    )
                ],
            }

            result = _LATE_FILL_RUNTIME._canonical_text_artifact_saved_result(
                payload,
                extension='css',
                source_name='styles',
            )

            self.assertEqual(result, {})

    def test_graph_owned_text_file_has_one_public_output_slot(self):
        current_path = '/tmp/20260904_styles.css'
        stale_path = '/tmp/styles.css'
        flow = build_artifact_flow_plan(
            [],
            [
                {
                    'type': 'text',
                    'path': stale_path,
                    'name': 'styles',
                    'origin': 'linked_public_dependency',
                    'artifact_ref': 'artifact:stale-styles',
                },
                {
                    'type': 'text',
                    'path': current_path,
                    'name': 'styles',
                    'origin': 'assistant_output',
                    'source_response_id': 'resp_one_styles_slot',
                    'branch_id': 'branch-styles',
                    'artifact_ref': 'artifact:current-styles',
                },
            ],
            request_payload={'prompt': 'Create styles.css.'},
            response_payload={
                'id': 'resp_one_styles_slot',
                'status': 'completed',
                'output_text': 'Artifacts generated.',
                'late_fill': {
                    'status': 'completed',
                    'completed_branches': [
                        {
                            'branch_id': 'repair-styles',
                            'phase_id': 'repair-styles',
                            'capability': 'chat',
                            'output_type': 'text',
                            'status': 'fulfilled',
                            'requires_artifact': True,
                            'saved_text_path': current_path,
                            'text_artifact_extension': 'css',
                            'text_artifact_source_name': 'styles',
                        }
                    ],
                },
            },
        )

        slots = flow['output_slots']
        self.assertEqual(
            [slot.get('artifact_ref') for slot in slots].count('artifact:current-styles'),
            1,
        )
        self.assertNotIn('artifact:stale-styles', [slot.get('artifact_ref') for slot in slots])

    def test_public_closure_and_bundle_prefer_current_named_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents = root / 'documents'
            documents.mkdir()
            index = documents / '20260904_index.html'
            current_css = documents / '20260904_styles.css'
            stale_css = documents / '20260903_styles.css'
            helper = documents / 'helper.js'
            index.write_text(
                f'<!doctype html><link rel="stylesheet" href="{stale_css.name}">'
                '<script src="helper.js"></script>',
                encoding='utf-8',
            )
            current_css.write_text('body { color: green; }', encoding='utf-8')
            stale_css.write_text('body { color: red; }', encoding='utf-8')
            helper.write_text('window.ready = true;', encoding='utf-8')
            artifacts = [
                {
                    'type': 'text',
                    'path': str(index),
                    'name': 'index',
                    'origin': 'assistant_output',
                    'source_response_id': 'resp_current_bundle',
                    'branch_id': 'branch-index',
                    'artifact_ref': 'artifact:index',
                },
                {
                    'type': 'text',
                    'path': str(current_css),
                    'name': 'styles',
                    'origin': 'assistant_output',
                    'source_response_id': 'resp_current_bundle',
                    'branch_id': 'branch-styles',
                    'artifact_ref': 'artifact:styles',
                },
                {
                    'type': 'text',
                    'path': str(stale_css),
                    'name': 'styles',
                    'origin': 'linked_public_dependency',
                    'artifact_ref': 'artifact:stale-styles',
                },
            ]
            outputs = [
                {
                    'type': 'text',
                    'status': 'fulfilled',
                    'source': 'promoted_output_slot',
                    'artifact_ref': 'artifact:index',
                },
                {
                    'type': 'text',
                    'status': 'fulfilled',
                    'source': 'promoted_output_slot',
                    'artifact_ref': 'artifact:styles',
                },
            ]
            payload = {
                'id': 'resp_current_bundle',
                'outputs': outputs,
                'artifacts': artifacts,
                'output_slots': outputs,
            }

            public = filter_public_response_artifacts(payload, artifacts, outputs=outputs)
            public_paths = {item.get('path') for item in public}
            self.assertIn(str(current_css), public_paths)
            self.assertIn(str(helper.resolve()), {str(Path(path).resolve()) for path in public_paths})
            self.assertNotIn(str(stale_css), public_paths)

            bundle = bundle_response_artifacts(
                payload,
                bundle_root=root / 'bundles',
                created_at='2026-09-04T20:00:00Z',
            )
            copied_sources = {
                str(Path(item['source_path']).resolve())
                for item in bundle['copied_artifacts']
            }
            self.assertIn(str(current_css.resolve()), copied_sources)
            self.assertIn(str(helper.resolve()), copied_sources)
            self.assertNotIn(str(stale_css.resolve()), copied_sources)
            self.assertEqual(bundle['link_check']['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
