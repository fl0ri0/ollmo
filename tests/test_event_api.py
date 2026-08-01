import unittest
from unittest.mock import patch

from ollmo_webserver import app


class EventApiTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    @patch('ollmo_webserver.read_events')
    def test_event_history_route_uses_event_log_reader(self, mock_read_events):
        mock_read_events.return_value = [
            {'id': 'event-1', 'category': 'chat', 'action': 'request', 'status': 'ok'}
        ]

        response = self.client.get('/api/event_history?category=chat&limit=10')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['count'], 1)
        self.assertEqual(payload['items'][0]['id'], 'event-1')
        mock_read_events.assert_called_once()


if __name__ == '__main__':
    unittest.main()
