import copy
import json
import tempfile
import unittest
from pathlib import Path

from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_server.responses_runtime import late_fill_has_actionable_repair_work
from ollmo_services.responses import build_canonical_response_artifacts
from ollmo_webserver import _normalize_late_fill_branches


class WebArtifactBindingGuardTests(unittest.TestCase):
    def _owner(self):
        owner = object.__new__(LateFillRuntimeOwner)
        owner.build_canonical_response_artifacts = build_canonical_response_artifacts
        owner.normalize_late_fill_branches = _normalize_late_fill_branches
        owner.branch_id = lambda item: str(item.get('branch_id') or item.get('phase_id') or '').strip()
        return owner

    def _payload(self, root: Path, *, missing_price: bool):
        html_path = root / 'configurator.html'
        json_path = root / 'pricing.json'
        html_path.write_text(
            '<script>\n'
            "fetch('pricing.json').then(r => r.json()).then(pricing => {\n"
            '  pricing.dial.forEach(item => {\n'
            '    const value = item.price;\n'
            '  });\n'
            '});\n'
            '</script>\n',
            encoding='utf-8',
        )
        dial = [{'name': 'Blue', 'price': 200}]
        if missing_price:
            dial.append({'name': 'Pearl', 'string': 300})
        json_path.write_text(json.dumps({'dial': dial}), encoding='utf-8')
        records = [
            {
                'type': 'text',
                'kind': 'text',
                'extension': 'html',
                'source_name': 'configurator',
                'path': str(html_path),
            },
            {
                'type': 'text',
                'kind': 'text',
                'extension': 'json',
                'source_name': 'pricing',
                'path': str(json_path),
            },
        ]
        return {'artifacts': records, 'saved_text_artifacts': records}

    def _projection_owner(self):
        owner = object.__new__(ResponsesRequestRuntimeOwner)
        owner.hooks = {
            'normalize_capability': lambda value: str(value or '').strip().lower() or None,
        }
        owner.capability_chat = 'chat'
        owner.capability_embedding = 'embedding'
        owner.capability_image_generation = 'image_generation'
        owner.capability_speech_to_text = 'speech_to_text'

        def attach_repair_gap(response_payload, repair_gap):
            updated = copy.deepcopy(response_payload)
            runtime = updated.setdefault('runtime', {})
            graph = runtime.setdefault('request_phase_graph', {})
            graph['downstream_branches'] = [
                dict(branch)
                for branch in repair_gap.get('pending_branches') or []
                if isinstance(branch, dict)
            ]
            graph['downstream_branch_ids'] = [
                str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
                for branch in graph['downstream_branches']
                if str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
            ]
            return updated

        owner._attach_repair_gap_to_request_phase_graph = attach_repair_gap
        return owner

    @staticmethod
    def _terminal_json_repair_state():
        target_path = '/tmp/ollmo-tests/pricing.json'
        open_check = {
            'check_kind': 'web_runtime_binding',
            'status': 'pending',
            'evidence': 'static_json_consumer_contract_mismatch',
            'branch_id': 'branch-pricing',
            'phase_id': 'phase-pricing',
            'repair_action': 'retry_same_branch',
            'recovery_action': 'retry_same_branch',
            'requires_artifact': True,
            'stage_direction': 'materialize_requested_text_artifact',
            'content_payload': 'Repair pricing.json so material.price exists.',
            'content_payload_source': 'terminal_web_runtime_binding_review',
            'text_artifact_extension': 'json',
            'text_artifact_source_name': 'pricing',
            'text_artifact_target_path': target_path,
            'artifact_request': {
                'extension': 'json',
                'source_name': 'pricing',
                'source': 'closure_web_binding_repair',
                'target_path': target_path,
            },
        }
        demoted_branch = {
            **open_check,
            'status': 'repair_needed',
            'capability': 'chat',
            'output_type': 'text',
            'materialization_contract_unmet': True,
            'materialization_contract_open_checks': [dict(open_check)],
        }
        return open_check, demoted_branch

    @staticmethod
    def _fulfilled_text_branch(branch_id, source_name, extension, target_path):
        return {
            'branch_id': branch_id,
            'phase_id': branch_id.replace('branch-', 'phase-', 1),
            'status': 'fulfilled',
            'capability': 'chat',
            'output_type': 'text',
            'role': 'text_artifact_output',
            'text_artifact_extension': extension,
            'text_artifact_source_name': source_name,
            'text_artifact_target_path': str(target_path),
            'artifact_request': {
                'extension': extension,
                'source_name': source_name,
                'target_path': str(target_path),
            },
        }

    def test_missing_json_consumer_field_blocks_linked_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._payload(Path(directory), missing_price=True)
            owner = self._owner()
            checks = owner._terminal_web_binding_contract_open_checks(payload)

            check = next(
                item
                for item in checks
                if item['evidence'] == 'static_json_consumer_contract_mismatch'
            )
            self.assertEqual(check['text_artifact_target_path'], str(Path(directory) / 'pricing.json'))
            self.assertEqual(check['repair_action'], 'retry_same_branch')
            self.assertIn('configurator.html', check['content_payload'])
            self.assertFalse(owner._terminal_linked_artifact_contract_is_fulfilled(payload))

    def test_valid_sibling_fetch_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._payload(Path(directory), missing_price=False)
            owner = self._owner()

            self.assertEqual(owner._terminal_web_binding_contract_open_checks(payload), [])
            self.assertTrue(owner._terminal_linked_artifact_contract_is_fulfilled(payload))

    def test_awaited_json_binding_requires_the_consumed_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html_path = root / 'configurator.html'
            json_path = root / 'pricing.json'
            html_path.write_text(
                '<script>async function init() {'
                "const response = await fetch('pricing.json');"
                'const data = await response.json();'
                'data.woods.forEach(wood => { const value = wood.price; });'
                '}</script>',
                encoding='utf-8',
            )
            json_path.write_text(json.dumps({'pricing': {}}), encoding='utf-8')
            records = [
                {
                    'type': 'text',
                    'extension': 'html',
                    'source_name': 'configurator',
                    'path': str(html_path),
                },
                {
                    'type': 'text',
                    'extension': 'json',
                    'source_name': 'pricing',
                    'path': str(json_path),
                },
            ]
            payload = {'artifacts': records, 'saved_text_artifacts': records}
            owner = self._owner()

            checks = owner._terminal_web_binding_contract_open_checks(payload)

            self.assertEqual(len(checks), 1)
            self.assertEqual(checks[0]['evidence'], 'static_json_consumer_contract_mismatch')
            self.assertEqual(checks[0]['text_artifact_target_path'], str(json_path))
            self.assertIn('`woods`', checks[0]['reason'])

            json_path.write_text(
                json.dumps({'woods': [{'name': 'Oak', 'price': 0}]}),
                encoding='utf-8',
            )
            self.assertEqual(owner._terminal_web_binding_contract_open_checks(payload), [])

    def test_reused_response_variable_does_not_bind_across_static_fetches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html_path = root / 'configurator.html'
            first_json = root / 'first.json'
            second_json = root / 'second.json'
            html_path.write_text(
                '<script>async function init() {'
                "{ const response = await fetch('first.json');"
                'const data = await response.json(); }'
                "{ const response = await fetch('second.json');"
                'const data = await response.json();'
                'data.items.forEach(item => { const value = item.price; }); }'
                '}</script>',
                encoding='utf-8',
            )
            first_json.write_text('{"metadata": {}}', encoding='utf-8')
            second_json.write_text(
                json.dumps({'items': [{'price': 12}]}),
                encoding='utf-8',
            )
            records = [
                {
                    'type': 'text',
                    'extension': 'html',
                    'source_name': 'configurator',
                    'path': str(html_path),
                },
                {
                    'type': 'text',
                    'extension': 'json',
                    'source_name': 'first',
                    'path': str(first_json),
                },
                {
                    'type': 'text',
                    'extension': 'json',
                    'source_name': 'second',
                    'path': str(second_json),
                },
            ]
            payload = {'artifacts': records, 'saved_text_artifacts': records}

            self.assertEqual(
                self._owner()._terminal_web_binding_contract_open_checks(payload),
                [],
            )

    def test_current_binding_defect_demotes_the_exact_saved_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._payload(root, missing_price=True)
            settings_path = root / 'settings.json'
            settings_path.write_text(json.dumps({'theme': 'light'}), encoding='utf-8')
            settings_record = {
                'type': 'text',
                'kind': 'text',
                'extension': 'json',
                'source_name': 'settings',
                'path': str(settings_path),
            }
            payload['artifacts'] = [*payload['artifacts'], settings_record]
            payload['saved_text_artifacts'] = [*payload['saved_text_artifacts'], settings_record]

            def fulfilled_branch(source_name, extension):
                target_path = root / f'{source_name}.{extension}'
                return {
                    'branch_id': f'branch-{source_name}',
                    'phase_id': f'phase-{source_name}',
                    'status': 'fulfilled',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'text_artifact_output',
                    'text_artifact_extension': extension,
                    'text_artifact_source_name': source_name,
                    'text_artifact_target_path': str(target_path),
                    'artifact_request': {
                        'extension': extension,
                        'source_name': source_name,
                        'target_path': str(target_path),
                    },
                }

            configurator_branch = fulfilled_branch('configurator', 'html')
            pricing_branch = fulfilled_branch('pricing', 'json')
            settings_branch = fulfilled_branch('settings', 'json')
            payload['late_fill'] = {
                'status': 'completed',
                'completed_branches': [
                    configurator_branch,
                    pricing_branch,
                    settings_branch,
                ],
                'pending_branches': [],
                'active_branches': [],
                'failed_branches': [],
            }
            owner = self._owner()

            checks = owner._terminal_web_binding_contract_open_checks(payload)
            check = next(
                item
                for item in checks
                if item['evidence'] == 'static_json_consumer_contract_mismatch'
            )
            self.assertEqual(check['branch_id'], 'branch-pricing')
            filtered = owner._filter_terminal_materialization_open_checks(
                checks,
                payload,
                payload['late_fill'],
            )
            self.assertEqual(len(filtered), 1)
            updated = owner._demote_terminal_materialization_branches_with_open_checks(
                payload['late_fill'],
                filtered,
            )
            self.assertEqual(
                [
                    (branch['branch_id'], branch['status'])
                    for branch in updated['completed_branches']
                ],
                [
                    ('branch-configurator', 'fulfilled'),
                    ('branch-settings', 'fulfilled'),
                ],
            )
            self.assertEqual(len(updated['pending_branches']), 1)
            self.assertEqual(updated['pending_branches'][0]['branch_id'], 'branch-pricing')
            self.assertEqual(updated['pending_branches'][0]['repair_action'], 'retry_same_branch')

    def test_branch_hint_prefers_normalized_target_for_same_named_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_target = root / 'first' / 'pricing.json'
            second_target = root / 'second' / 'pricing.json'
            late_fill = {
                'completed_branches': [
                    self._fulfilled_text_branch(
                        'branch-first-pricing',
                        'pricing',
                        'json',
                        first_target,
                    ),
                    self._fulfilled_text_branch(
                        'branch-second-pricing',
                        'pricing',
                        'json',
                        second_target,
                    ),
                ],
            }

            branch = self._owner()._terminal_text_branch_hint_for_dependency(
                late_fill=late_fill,
                extension='json',
                source_name='pricing',
                target_path=str(root / 'second' / '..' / 'second' / 'pricing.json'),
            )

            self.assertEqual(branch['branch_id'], 'branch-second-pricing')

    def test_demoter_rejects_stale_id_for_explicit_unequal_same_named_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_branch = self._fulfilled_text_branch(
                'branch-first-pricing',
                'pricing',
                'json',
                root / 'first' / 'pricing.json',
            )
            second_target = root / 'second' / 'pricing.json'
            second_branch = self._fulfilled_text_branch(
                'branch-second-pricing',
                'pricing',
                'json',
                second_target,
            )
            stale_id_check = {
                'check_kind': 'web_runtime_binding',
                'status': 'pending',
                'evidence': 'static_json_consumer_contract_mismatch',
                'branch_id': 'branch-first-pricing',
                'phase_id': 'phase-first-pricing',
                'repair_action': 'retry_same_branch',
                'text_artifact_extension': 'json',
                'text_artifact_source_name': 'pricing',
                'text_artifact_target_path': str(second_target),
                'artifact_request': {
                    'extension': 'json',
                    'source_name': 'pricing',
                    'target_path': str(second_target),
                },
            }
            late_fill = {
                'status': 'completed',
                'completed_branches': [first_branch, second_branch],
                'pending_branches': [],
            }

            updated = self._owner()._demote_terminal_materialization_branches_with_open_checks(
                late_fill,
                [stale_id_check],
            )

            self.assertEqual(
                [branch['branch_id'] for branch in updated['completed_branches']],
                ['branch-first-pricing'],
            )
            self.assertEqual(
                [branch['branch_id'] for branch in updated['pending_branches']],
                ['branch-second-pricing'],
            )

    def test_terminal_only_json_repair_is_projected_from_current_demoted_branch(self):
        open_check, demoted_branch = self._terminal_json_repair_state()
        payload = {
            'id': 'resp-terminal-json-repair',
            'late_fill': {
                'status': 'partial_failed',
                'materialization_contract_unmet': True,
                'materialization_contract_open_checks': [open_check],
                'materialization_contract_demoted_branches': [demoted_branch],
                'materialization_contract_current_demoted_branches': [demoted_branch],
                'pending_branches': [demoted_branch],
                'completed_branches': [],
                'failed_branches': [],
                'cancelled_branches': [],
            },
            'runtime': {
                'graph_closure_review': {
                    'status': 'fulfilled',
                    'checks': [],
                },
                'request_phase_graph': {
                    'downstream_branches': [],
                },
            },
        }

        projected = self._projection_owner().project_terminal_closure_repair(payload)

        self.assertEqual(projected['status'], 'queued')
        self.assertEqual(len(projected['branch_ids']), 1)
        self.assertTrue(
            projected['branch_ids'][0].startswith(
                'branch-repair-branch-pricing-'
            )
        )
        pending = projected['artifact_gap']['pending_branches']
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['repair_owner_branch_id'], 'branch-pricing')
        self.assertEqual(pending[0]['text_artifact_target_path'], open_check['text_artifact_target_path'])
        self.assertEqual(pending[0]['content_payload_source'], 'terminal_web_runtime_binding_review')

    def test_terminal_materialization_projection_uses_all_current_demotions_beyond_compact_diagnostics(self):
        open_checks = []
        demoted_branches = []
        for index in range(9):
            branch_id = f'branch-pricing-{index}'
            phase_id = f'phase-pricing-{index}'
            target_path = f'/tmp/ollmo-tests/pricing-{index}.json'
            open_check = {
                'check_kind': 'web_runtime_binding',
                'status': 'pending',
                'evidence': 'static_json_consumer_contract_mismatch',
                'branch_id': branch_id,
                'phase_id': phase_id,
                'repair_action': 'retry_same_branch',
                'recovery_action': 'retry_same_branch',
                'requires_artifact': True,
                'stage_direction': 'materialize_requested_text_artifact',
                'content_payload': f'Repair pricing-{index}.json so material.price exists.',
                'content_payload_source': 'terminal_web_runtime_binding_review',
                'text_artifact_extension': 'json',
                'text_artifact_source_name': f'pricing-{index}',
                'text_artifact_target_path': target_path,
                'artifact_request': {
                    'extension': 'json',
                    'source_name': f'pricing-{index}',
                    'source': 'closure_web_binding_repair',
                    'target_path': target_path,
                },
            }
            open_checks.append(open_check)
            demoted_branches.append(
                {
                    **open_check,
                    'status': 'repair_needed',
                    'capability': 'chat',
                    'output_type': 'text',
                    'materialization_contract_unmet': True,
                }
            )
        payload = {
            'id': 'resp-terminal-json-repair-nine-targets',
            'late_fill': {
                'status': 'partial_failed',
                'materialization_contract_unmet': True,
                # Public diagnostics remain deliberately compact.
                'materialization_contract_open_checks': open_checks[:8],
                'materialization_contract_demoted_branches': list(demoted_branches),
                # Execution authority must retain every exact current target.
                'materialization_contract_current_demoted_branches': list(
                    demoted_branches
                ),
                'pending_branches': list(demoted_branches),
                'completed_branches': [],
                'failed_branches': [],
                'cancelled_branches': [],
            },
            'runtime': {
                'graph_closure_review': {'status': 'fulfilled', 'checks': []},
                'request_phase_graph': {'downstream_branches': []},
            },
        }

        projected = self._projection_owner().project_terminal_closure_repair(payload)

        self.assertEqual(projected['status'], 'queued')
        self.assertEqual(len(projected['branch_ids']), 9)
        pending = projected['artifact_gap']['pending_branches']
        self.assertEqual(len(pending), 9)
        self.assertEqual(
            [branch['repair_owner_branch_id'] for branch in pending],
            [f'branch-pricing-{index}' for index in range(9)],
        )
        self.assertEqual(
            [branch['text_artifact_target_path'] for branch in pending],
            [f'/tmp/ollmo-tests/pricing-{index}.json' for index in range(9)],
        )

    def test_terminal_materialization_repair_stops_after_one_successor_generation(self):
        open_check, demoted_branch = self._terminal_json_repair_state()
        initial = {
            'id': 'resp-terminal-json-repair-initial',
            'late_fill': {
                'status': 'partial_failed',
                'materialization_contract_open_checks': [open_check],
                'materialization_contract_demoted_branches': [demoted_branch],
                'materialization_contract_current_demoted_branches': [demoted_branch],
                'pending_branches': [demoted_branch],
                'completed_branches': [],
                'failed_branches': [],
                'cancelled_branches': [],
            },
            'runtime': {
                'graph_closure_review': {'status': 'fulfilled', 'checks': []},
                'request_phase_graph': {'downstream_branches': []},
            },
        }
        first_projection = self._projection_owner().project_terminal_closure_repair(initial)
        first_repair = dict(first_projection['artifact_gap']['pending_branches'][0])
        self.assertEqual(
            first_repair['promotion_source'],
            'terminal_materialization_contract',
        )

        repeated_check = {
            **open_check,
            'branch_id': first_repair['branch_id'],
            'phase_id': first_repair['phase_id'],
        }
        repeated_branch = {
            **first_repair,
            'status': 'repair_needed',
            'materialization_contract_unmet': True,
        }
        repeated = copy.deepcopy(initial)
        repeated['id'] = 'resp-terminal-json-repair-repeated'
        repeated['late_fill']['materialization_contract_open_checks'] = [repeated_check]
        repeated['late_fill']['materialization_contract_demoted_branches'] = [repeated_branch]
        repeated['late_fill']['materialization_contract_current_demoted_branches'] = [
            repeated_branch
        ]
        repeated['late_fill']['pending_branches'] = [repeated_branch]

        second_projection = self._projection_owner().project_terminal_closure_repair(repeated)

        self.assertEqual(second_projection['status'], 'not_applicable')
        self.assertEqual(
            second_projection['reason'],
            'no_terminal_closure_repair_gap',
        )

    def test_graph_feedback_and_terminal_json_repair_are_merged_once(self):
        open_check, demoted_branch = self._terminal_json_repair_state()
        css_target = '/tmp/ollmo-tests/styles.css'
        css_contract = {
            'kind': 'ollmo.repair_rebuild_contract',
            'contract_id': 'repair-contract-branch-css',
            'status': 'promoted',
            'promotion_source': 'graph_closure_review',
            'repair_action': 'retry_same_branch',
            'execution_policy': 'schedule_late_fill_branch',
            'auto_execute': True,
            'repair_work_available': True,
            'needs_external_input': False,
            'branch_id': 'branch-css',
            'phase_id': 'branch-css',
            'capability': 'chat',
            'output_type': 'text',
            'content_payload': 'Repair styles.css against the linked HTML class vocabulary.',
            'content_payload_source': 'closure_html_css_selector_binding_review',
            'stage_direction': 'materialize_requested_text_artifact',
            'requires_artifact': True,
            'text_artifact_extension': 'css',
            'text_artifact_source_name': 'styles',
            'text_artifact_target_path': css_target,
            'artifact_request': {
                'extension': 'css',
                'source_name': 'styles',
                'target_path': css_target,
            },
        }
        css_item = {
            **css_contract,
            'check_kind': 'html_css_selector_binding',
            'repair_contract': dict(css_contract),
            'repair_contract_id': css_contract['contract_id'],
            'repair_execution_policy': 'schedule_late_fill_branch',
        }
        feedback = {
            'status': 'repair_required',
            'items': [css_item],
            'repair_rebuild_contracts': [css_contract],
            'repair_loop': {
                'status': 'promoted',
                'auto_execute': True,
                'repair_work_available': True,
                'repair_work_available_count': 1,
                'promoted_contracts': [css_contract, copy.deepcopy(css_contract)],
            },
        }
        payload = {
            'id': 'resp-graph-and-terminal-repair',
            'late_fill': {
                'status': 'partial_failed',
                'materialization_contract_unmet': True,
                'materialization_contract_open_checks': [open_check],
                'materialization_contract_demoted_branches': [demoted_branch],
                'materialization_contract_current_demoted_branches': [demoted_branch],
                'pending_branches': [demoted_branch],
                'completed_branches': [],
                'failed_branches': [],
                'cancelled_branches': [],
            },
            'runtime': {
                'graph_closure_review': {
                    'status': 'pending',
                    'checks': [css_item],
                    'ghost_repair_feedback': feedback,
                },
                'request_phase_graph': {
                    'downstream_branches': [],
                },
            },
        }

        projected = self._projection_owner().project_terminal_closure_repair(payload)

        self.assertEqual(projected['status'], 'queued')
        self.assertEqual(len(projected['branch_ids']), 2)
        self.assertEqual(projected['branch_ids'][0], 'branch-css')
        self.assertTrue(
            projected['branch_ids'][1].startswith(
                'branch-repair-branch-pricing-'
            )
        )
        pending = projected['artifact_gap']['pending_branches']
        self.assertEqual(
            [branch['text_artifact_target_path'] for branch in pending],
            [css_target, open_check['text_artifact_target_path']],
        )
        promoted_contracts = projected['artifact_gap']['repair_loop'][
            'promoted_contracts'
        ]
        self.assertEqual(len(promoted_contracts), 2)
        self.assertEqual(
            promoted_contracts[0]['contract_id'],
            css_contract['contract_id'],
        )
        self.assertEqual(
            promoted_contracts[0]['content_payload'],
            css_contract['content_payload'],
        )
        self.assertEqual(
            promoted_contracts[1]['contract_id'],
            pending[1]['repair_contract_id'],
        )
        self.assertEqual(
            promoted_contracts[1]['branch_id'],
            pending[1]['branch_id'],
        )

        completed_late_fill = {
            **projected['artifact_gap'],
            'status': 'completed',
            'final_materialization_contract_status': 'fulfilled',
            'pending_branches': [],
            'active_branches': [],
            'failed_branches': [],
            'cancelled_branches': [],
            'recovery_candidates': [],
            'completed_branches': [
                {**dict(branch), 'status': 'fulfilled'}
                for branch in pending
            ],
        }
        reconciled = LateFillRuntimeOwner._reconcile_terminal_satisfied_repair_loop(
            completed_late_fill
        )

        self.assertEqual(reconciled['repair_loop']['status'], 'completed')
        self.assertFalse(reconciled['repair_loop']['repair_work_available'])
        self.assertEqual(reconciled['repair_loop']['resolved_contract_count'], 2)
        self.assertNotIn('repair_action', reconciled)
        self.assertNotIn('repair_actions', reconciled)
        self.assertFalse(late_fill_has_actionable_repair_work(reconciled))

    def test_repeated_inline_identifier_separator_blocks_web_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._payload(root, missing_price=False)
            html_path = root / 'configurator.html'
            html_path.write_text(
                html_path.read_text(encoding='utf-8').replace(
                    '</script>',
                    'const label = dial・・dialSelect;\n</script>',
                ),
                encoding='utf-8',
            )
            owner = self._owner()

            checks = owner._terminal_web_binding_contract_open_checks(payload)
            self.assertTrue(any(item['evidence'] == 'inline_javascript_binding_mismatch' for item in checks))

    def test_repeated_identifier_separator_in_standalone_js_blocks_web_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._payload(root, missing_price=False)
            script_path = root / 'configurator.js'
            script_path.write_text(
                'const label = dial・・dialSelect.options[0].text;\n',
                encoding='utf-8',
            )
            payload['artifacts'].append(
                {
                    'type': 'text',
                    'kind': 'text',
                    'extension': 'js',
                    'source_name': 'configurator',
                    'path': str(script_path),
                }
            )
            owner = self._owner()

            checks = owner._terminal_web_binding_contract_open_checks(payload)
            check = next(
                item
                for item in checks
                if item['evidence'] == 'inline_javascript_binding_mismatch'
            )
            self.assertEqual(check['text_artifact_target_path'], str(script_path))

    def test_guard_ignores_valid_unicode_identifier_and_nonlocal_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._payload(root, missing_price=True)
            html_path = root / 'configurator.html'
            html_path.write_text(
                '<script>const caféMenu = {}; caféMenu.open = () => true;'
                "fetch('https://example.com/pricing.json');"
                "fetch('/api/pricing.json');</script>",
                encoding='utf-8',
            )
            owner = self._owner()

            self.assertEqual(owner._terminal_web_binding_contract_open_checks(payload), [])


if __name__ == '__main__':
    unittest.main()
