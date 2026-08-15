import json
import tempfile
import unittest
from pathlib import Path

from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_services.responses import build_canonical_response_artifacts


class WebArtifactBindingGuardTests(unittest.TestCase):
    def _owner(self):
        owner = object.__new__(LateFillRuntimeOwner)
        owner.build_canonical_response_artifacts = build_canonical_response_artifacts
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
            pricing_branch = {
                'branch_id': 'branch-pricing',
                'phase_id': 'phase-pricing',
                'status': 'fulfilled',
                'capability': 'chat',
                'output_type': 'text',
                'role': 'text_artifact_output',
                'text_artifact_extension': 'json',
                'text_artifact_source_name': 'pricing',
                'text_artifact_target_path': str(root / 'pricing.json'),
                'artifact_request': {
                    'extension': 'json',
                    'source_name': 'pricing',
                    'target_path': str(root / 'pricing.json'),
                },
            }
            payload['late_fill'] = {
                'status': 'completed',
                'completed_branches': [pricing_branch],
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
            self.assertEqual(updated['completed_branches'], [])
            self.assertEqual(updated['pending_branches'][0]['branch_id'], 'branch-pricing')
            self.assertEqual(updated['pending_branches'][0]['repair_action'], 'retry_same_branch')

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
