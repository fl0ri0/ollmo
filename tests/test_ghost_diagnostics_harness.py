import unittest

from ollmo_g.diagnostics_harness import (
    build_fixture_request_payload,
    evaluate_fixture_result,
    load_diagnostic_fixtures,
)


class GhostDiagnosticsHarnessTests(unittest.TestCase):
    def test_load_diagnostic_fixtures_includes_capability_hint_case(self):
        fixtures = load_diagnostic_fixtures()
        case_ids = {fixture['case_id'] for fixture in fixtures}

        self.assertIn('tts_capability_hint_summary_read_aloud', case_ids)
        self.assertIn('text_to_speech_direct', case_ids)

    def test_build_fixture_request_payload_promotes_meta_and_developer_flags(self):
        fixture = next(
            item
            for item in load_diagnostic_fixtures()
            if item['case_id'] == 'tts_capability_hint_summary_read_aloud'
        )

        payload = build_fixture_request_payload(fixture)

        self.assertTrue(payload['ghost_route'])
        self.assertEqual(payload['capability_hint'], 'text_to_speech')
        self.assertTrue(payload['developer_flags']['embedding_signals_enabled'])
        self.assertEqual(payload['developer_flags']['planner_timeout_ms'], 12000)

    def test_evaluate_fixture_result_accepts_matching_route_metadata_and_control_hints(self):
        fixtures = {fixture['case_id']: fixture for fixture in load_diagnostic_fixtures()}
        failures = evaluate_fixture_result(
            fixtures['tts_capability_hint_summary_read_aloud'],
            route_payload={
                'capability': 'text_to_speech',
                'request_meta': {
                    'capability_hint': 'text_to_speech',
                },
                'runtime': {
                    'developer_diagnostics': {
                        'routing_contract': 'ghost_primary',
                    }
                },
            },
        )
        control_failures = evaluate_fixture_result(
            fixtures['text_to_speech_direct'],
            route_payload={
                'capability': 'text_to_speech',
                'runtime': {
                    'developer_diagnostics': {
                        'routing_contract': 'ghost_primary',
                        'embedding_signals_enabled': True,
                    }
                },
            },
            control_hints={'response_format': 'mp3'},
        )

        self.assertEqual(failures, [])
        self.assertEqual(control_failures, [])


if __name__ == '__main__':
    unittest.main()
