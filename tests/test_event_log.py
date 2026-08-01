import tempfile
import unittest
from pathlib import Path

from ollmo_services.events import log_event, read_events


class EventLogTests(unittest.TestCase):
    def test_append_and_read_events_with_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'events.jsonl'
            log_event(
                category='runtime',
                action='start_model',
                status='ok',
                path=path,
                model='gpt-oss:20b',
                message='Started gpt-oss:20b',
            )
            log_event(
                category='chat',
                action='request',
                status='failed',
                path=path,
                model='qwen3.5:27b',
                message='Timeout Port 11435',
            )

            items = read_events(path=path, limit=10)
            runtime_items = read_events(path=path, limit=10, category='runtime')
            failed_items = read_events(path=path, limit=10, status='failed')

            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]['category'], 'chat')
            self.assertEqual(len(runtime_items), 1)
            self.assertEqual(runtime_items[0]['action'], 'start_model')
            self.assertEqual(len(failed_items), 1)
            self.assertEqual(failed_items[0]['message'], 'Timeout Port 11435')


if __name__ == '__main__':
    unittest.main()
