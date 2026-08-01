import unittest

from ollmo_server.substrate_hygiene_runtime import (
    PostResponseSubstrateHygieneRuntimeOwner,
    collect_response_substrate_instance_ids,
)


class _Response:
    def raise_for_status(self):
        return None


class PostResponseSubstrateHygieneRuntimeTests(unittest.TestCase):
    def _owner(self, *, instances, posts, events, policy='conservative'):
        return PostResponseSubstrateHygieneRuntimeOwner(
            policy=policy,
            load_running_instances=lambda: list(instances),
            merge_instances_with_runtime_status=lambda raw_instances, **_kwargs: raw_instances,
            log_unified_event=lambda **event: events.append(event),
            requests_post=lambda *args, **kwargs: posts.append((args, kwargs)) or _Response(),
            run_async=False,
        )

    def test_collects_instance_ids_from_response_and_late_fill_truth(self):
        payload = {
            'instance_id': 'chat-1',
            'late_fill': {
                'fill_results': [{'fill_instance_id': 'image-1'}],
                'completed_branches': [{'execution_instance_id': 'tts-1'}],
            },
        }

        self.assertEqual(
            collect_response_substrate_instance_ids(payload),
            ['chat-1', 'image-1', 'tts-1'],
        )

    def test_disabled_policy_does_not_unload(self):
        posts = []
        events = []
        owner = self._owner(instances=[], posts=posts, events=events, policy='off')

        result = owner.schedule_post_response_substrate_hygiene({'id': 'resp_1', 'instance_id': 'chat-1'})

        self.assertEqual(result['status'], 'disabled')
        self.assertEqual(posts, [])
        self.assertEqual(events, [])

    def test_unloads_idle_ollama_instance_with_keep_alive_zero(self):
        posts = []
        events = []
        owner = self._owner(
            instances=[
                {
                    'instance_id': 'gemma4:26b-1',
                    'backend': 'ollama',
                    'model': 'gemma4:26b',
                    'port': 11434,
                    'activity': 'idle',
                }
            ],
            posts=posts,
            events=events,
        )

        result = owner.schedule_post_response_substrate_hygiene(
            {'id': 'resp_1', 'instance_id': 'gemma4:26b-1'},
            reason='test_terminal',
        )

        self.assertEqual(result['results'][0]['status'], 'ok')
        self.assertEqual(posts[0][0][0], 'http://127.0.0.1:11434/api/generate')
        self.assertEqual(
            posts[0][1]['json'],
            {'model': 'gemma4:26b', 'prompt': '', 'stream': False, 'keep_alive': 0},
        )
        self.assertEqual(events[0]['action'], 'post_response_substrate_hygiene')
        self.assertEqual(events[0]['reason'], 'test_terminal')

    def test_skips_busy_instance(self):
        posts = []
        events = []
        owner = self._owner(
            instances=[
                {
                    'instance_id': 'image-1',
                    'backend': 'ollama',
                    'model': 'x/z-image-turbo:latest',
                    'port': 11435,
                    'activity': 'busy',
                }
            ],
            posts=posts,
            events=events,
        )

        result = owner.schedule_post_response_substrate_hygiene({'id': 'resp_1', 'instance_id': 'image-1'})

        self.assertEqual(result['results'][0]['status'], 'skipped')
        self.assertEqual(result['results'][0]['skip_reason'], 'instance_busy')
        self.assertEqual(posts, [])

    def test_uses_mlx_unload_url_when_advertised(self):
        posts = []
        events = []
        owner = self._owner(
            instances=[
                {
                    'instance_id': 'vlm-1',
                    'backend': 'mlx',
                    'model': 'mlx-vlm-model',
                    'backend_runtime': {
                        'supports_unload': True,
                        'unload_url': 'http://127.0.0.1:11501/unload',
                    },
                }
            ],
            posts=posts,
            events=events,
        )

        result = owner.schedule_post_response_substrate_hygiene({'id': 'resp_1', 'instance_id': 'vlm-1'})

        self.assertEqual(result['results'][0]['status'], 'ok')
        self.assertEqual(posts[0][0][0], 'http://127.0.0.1:11501/unload')
        self.assertEqual(posts[0][1]['json'], {})


if __name__ == '__main__':
    unittest.main()
