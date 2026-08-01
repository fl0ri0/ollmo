import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_g.payload import build_ghost_payload
from ollmo_core.runtime_liveness import (
    format_runtime_timestamp,
    runtime_instance_is_selectable,
    runtime_instance_score,
)


class RuntimeLivenessTests(unittest.TestCase):
    def test_degraded_and_busy_are_advisory_when_process_and_port_are_live(self):
        ready_idle = {
            'instance_id': 'ready-idle',
            'capability': 'chat',
            'readiness': 'ready',
            'activity': 'idle',
            'process_alive': True,
            'port_listening': True,
        }
        degraded_busy = {
            'instance_id': 'degraded-busy',
            'capability': 'chat',
            'readiness': 'degraded',
            'activity': 'busy',
            'process_alive': True,
            'port_listening': True,
            'last_error': 'previous timeout',
        }

        self.assertTrue(runtime_instance_is_selectable(degraded_busy, capability='chat'))
        self.assertGreater(
            runtime_instance_score(ready_idle, capability='chat'),
            runtime_instance_score(degraded_busy, capability='chat'),
        )

    def test_dead_process_or_port_is_hard_unavailable(self):
        self.assertFalse(
            runtime_instance_is_selectable(
                {
                    'instance_id': 'dead-process',
                    'capability': 'chat',
                    'readiness': 'ready',
                    'process_alive': False,
                    'port_listening': True,
                },
                capability='chat',
            )
        )
        self.assertFalse(
            runtime_instance_is_selectable(
                {
                    'instance_id': 'dead-port',
                    'capability': 'chat',
                    'readiness': 'ready',
                    'process_alive': True,
                    'port_listening': False,
                },
                capability='chat',
            )
        )

    def test_fresh_failure_cooldown_makes_instance_temporarily_unselectable(self):
        now = dt.datetime(2026, 5, 23, 0, 0, tzinfo=dt.timezone.utc)
        cooldown_until = format_runtime_timestamp(now + dt.timedelta(seconds=120))

        self.assertFalse(
            runtime_instance_is_selectable(
                {
                    'instance_id': 'helper-cooling-down',
                    'capability': 'vision_analysis',
                    'readiness': 'degraded',
                    'process_alive': True,
                    'port_listening': True,
                    'cooldown_until': cooldown_until,
                    'cooldown_capability': 'vision_analysis',
                },
                capability='vision_analysis',
                now=now,
            )
        )

    def test_ghost_payload_does_not_turn_live_degraded_into_runtime_issue(self):
        payload = build_ghost_payload(
            [
                {
                    'instance_id': 'live-degraded-chat',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'degraded',
                    'process_alive': True,
                    'port_listening': True,
                    'runtime_status': {
                        'readiness': 'degraded',
                        'activity': 'idle',
                        'process_alive': True,
                        'port_listening': True,
                    },
                }
            ],
            recent_events=[],
            runtime_log_path=Path('/tmp/ollmo-missing-runtime.log'),
            response_frame_ledger_path=Path('/tmp/ollmo-missing-responses.jsonl'),
            accepted_learning_policy_path=Path('/tmp/ollmo-missing-accepted-policy.json'),
        )

        self.assertEqual(payload['capabilities']['chat']['default_instance_id'], 'live-degraded-chat')
        self.assertFalse(any('degraded' in str(issue).lower() for issue in payload['issues']))
        self.assertFalse(any('unavailable' in str(issue).lower() for issue in payload['issues']))

    def test_ghost_payload_reports_hard_unavailable_live_truth(self):
        payload = build_ghost_payload(
            [
                {
                    'instance_id': 'dead-port-chat',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11437,
                    'readiness': 'ready',
                    'process_alive': True,
                    'port_listening': False,
                    'runtime_status': {
                        'readiness': 'ready',
                        'activity': 'idle',
                        'process_alive': True,
                        'port_listening': False,
                    },
                }
            ],
            recent_events=[],
            runtime_log_path=Path('/tmp/ollmo-missing-runtime.log'),
            response_frame_ledger_path=Path('/tmp/ollmo-missing-responses.jsonl'),
            accepted_learning_policy_path=Path('/tmp/ollmo-missing-accepted-policy.json'),
        )

        self.assertIsNone(payload['capabilities']['chat']['default_instance_id'])
        self.assertTrue(any('unavailable' in str(issue).lower() for issue in payload['issues']))

    def test_ghost_payload_reclassifies_legacy_degraded_failed_event_as_advisory(self):
        payload = build_ghost_payload(
            [
                {
                    'instance_id': 'ready-chat',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'readiness': 'ready',
                    'process_alive': True,
                    'port_listening': True,
                }
            ],
            recent_events=[
                {
                    'timestamp': '2026-06-06T19:00:00Z',
                    'category': 'runtime',
                    'action': 'status_transition',
                    'status': 'failed',
                    'readiness': 'degraded',
                    'instance_id': 'ready-chat',
                    'message': 'Runtime status changed: ready -> degraded',
                }
            ],
            runtime_log_path=Path('/tmp/ollmo-missing-runtime.log'),
            response_frame_ledger_path=Path('/tmp/ollmo-missing-responses.jsonl'),
            accepted_learning_policy_path=Path('/tmp/ollmo-missing-accepted-policy.json'),
        )

        event = payload['recent_events'][0]
        self.assertEqual(event['status'], 'warning')
        self.assertEqual(event['severity'], 'advisory')
        self.assertEqual(
            event['runtime_truth_note'],
            'legacy_degraded_failed_event_reclassified_as_advisory',
        )

    def test_ghost_payload_can_skip_offline_self_learning_report_but_keep_accepted_hints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            accepted_policy_path = root / 'accepted_policy_snapshot.json'
            accepted_policy_path.write_text(
                json.dumps(
                    {
                        'kind': 'ollmo.accepted_learning_policy_snapshot',
                        'enabled': True,
                        'authority': 'soft_hint',
                        'status': 'enabled',
                        'runtime_effect': 'readable_policy_input',
                        'accepted_learnings': [
                            {
                                'kind': 'ollmo.accepted_learning',
                                'learning_id': 'accepted-test-learning',
                                'candidate_id': 'policy-improvement-artifact_fulfillment_policy',
                                'target_area': 'artifact_fulfillment_policy',
                                'bounded_hint': 'Keep artifact fulfillment bounded by runtime truth.',
                                'case_kinds': {'open_output_slots': 1},
                                'status': 'accepted',
                                'allowed_use': 'soft_hint_only',
                                'forbidden_use': 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth',
                            }
                        ],
                    }
                ),
                encoding='utf-8',
            )

            with patch(
                'ollmo_g.payload.build_self_learning_report',
                side_effect=AssertionError('offline report should not be read in hot routing payload'),
            ):
                payload = build_ghost_payload(
                    [],
                    recent_events=[],
                    runtime_log_path=root / 'missing.log',
                    response_frame_ledger_path=root / 'large-responses.jsonl',
                    accepted_learning_policy_path=accepted_policy_path,
                    include_self_learning_report=False,
                )

        self.assertEqual(payload['self_learning']['status'], 'omitted')
        self.assertEqual(
            payload['self_learning']['reason'],
            'offline_self_learning_report_omitted_for_hot_routing_path',
        )
        self.assertEqual(payload['accepted_learning_hints']['hint_count'], 1)
        self.assertEqual(payload['accepted_learning_hints']['hints'][0]['allowed_use'], 'soft_hint_only')


if __name__ == '__main__':
    unittest.main()
