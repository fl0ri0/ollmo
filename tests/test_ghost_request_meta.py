import unittest

from ollmo_g.request_meta import (
    MAX_PLANNER_TIMEOUT_MS,
    MIN_PLANNER_TIMEOUT_MS,
    apply_request_meta_to_route_context,
    normalize_request_meta,
)


class GhostRequestMetaTests(unittest.TestCase):
    def test_normalize_request_meta_defaults_to_embedding_helper_and_no_planner_override(self):
        request_meta = normalize_request_meta({})

        self.assertTrue(request_meta['developer_flags']['embedding_signals_enabled'])
        self.assertIsNone(request_meta['developer_flags']['planner_timeout_ms'])
        self.assertEqual(request_meta['developer_flags']['accepted_learning_authority'], 'soft_hint')

    def test_normalize_request_meta_ignores_retired_routing_flags_and_clamps_planner_timeout(self):
        request_meta = normalize_request_meta(
            {
                'role': 'diagnostics',
                'ghost_mode': 'repair',
                'developer_flags': (
                    '{"heuristics_enabled":"false","semantic_router_required":"true",'
                    '"router_timeout_ms":"10","planner_timeout_ms":"999999999"}'
                ),
                'executionProfile': '{"dry_run": true}',
            }
        )

        self.assertEqual(request_meta['ghost_mode'], 'repair')
        self.assertEqual(request_meta['ghost_mode_source'], 'request')
        self.assertIsNone(request_meta['capability_hint'])
        self.assertIsNone(request_meta['capability_hint_source'])
        self.assertTrue(request_meta['developer_flags']['embedding_signals_enabled'])
        self.assertEqual(request_meta['developer_flags']['planner_timeout_ms'], MAX_PLANNER_TIMEOUT_MS)

    def test_normalize_request_meta_uses_embedding_signals_clamps_timeout_and_ignores_drc_flags(self):
        request_meta = normalize_request_meta(
            {
                'developer_flags': {
                    'embedding_signals_enabled': False,
                    'ghost_plan_refinement_mode': 'compare',
                    'semantic_handoff_review_mode': 'on',
                    'planner_timeout_ms': '10',
                }
            }
        )

        self.assertFalse(request_meta['developer_flags']['embedding_signals_enabled'])
        self.assertNotIn('ghost_plan_refinement_mode', request_meta['developer_flags'])
        self.assertNotIn('semantic_handoff_review_mode', request_meta['developer_flags'])
        self.assertEqual(request_meta['developer_flags']['planner_timeout_ms'], MIN_PLANNER_TIMEOUT_MS)
        self.assertEqual(request_meta['developer_flags']['accepted_learning_authority'], 'soft_hint')

    def test_normalize_request_meta_accepts_bounded_learning_authority_levels(self):
        request_meta = normalize_request_meta(
            {
                'request_meta': {
                    'developer_flags': {
                        'acceptedLearningAuthority': 'preferred',
                    },
                },
            }
        )
        invalid = normalize_request_meta(
            {
                'developer_flags': {
                    'accepted_learning_authority': 'silent-auto-mutate',
                },
            }
        )

        self.assertEqual(request_meta['developer_flags']['accepted_learning_authority'], 'preferred')
        self.assertEqual(invalid['developer_flags']['accepted_learning_authority'], 'soft_hint')

    def test_normalize_request_meta_accepts_nested_request_meta_envelope(self):
        request_meta = normalize_request_meta(
            {
                'request_meta': {
                    'ghostMode': 'improviser',
                    'capability_hint': 'text_to_speech',
                    'developer_flags': {
                        'embedding_signals_enabled': False,
                    },
                }
            }
        )

        self.assertEqual(request_meta['ghost_mode'], 'improviser')
        self.assertEqual(request_meta['capability_hint'], 'text_to_speech')
        self.assertEqual(request_meta['capability_hint_source'], 'request')
        self.assertFalse(request_meta['developer_flags']['embedding_signals_enabled'])

    def test_apply_request_meta_to_route_context_attaches_compact_meta(self):
        route_context = {
            'recent_messages': [
                {'role': 'user', 'content': 'prefer german'},
            ],
            'runtime': {},
        }

        updated = apply_request_meta_to_route_context(
            route_context,
            normalize_request_meta(
                {
                    'developer_flags': {
                        'embedding_signals_enabled': False,
                    }
                }
            ),
        )

        self.assertIn('request_meta', updated)
        self.assertEqual(updated['recent_messages'], [{'role': 'user', 'content': 'prefer german'}])
        self.assertFalse(updated['runtime']['request_meta']['developer_flags']['embedding_signals_enabled'])


if __name__ == '__main__':
    unittest.main()
