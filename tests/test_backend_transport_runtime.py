import unittest
from unittest.mock import patch

from ollmo_server.backend_transport_runtime import BackendTransportRuntimeOwner


class BackendTransportRuntimeTests(unittest.TestCase):
    def test_ollama_chat_uses_default_transport_timeout(self):
        owner = BackendTransportRuntimeOwner(
            hooks={'requests_module': object()},
            capability_chat='chat',
            request_timeout_error=TimeoutError,
            request_connection_error=ConnectionError,
            request_exception_error=Exception,
        )
        captured = {}

        def fake_ollama_chat_with_options(**kwargs):
            captured.update(kwargs)
            return {'content': 'ok'}

        messages = [{'role': 'user', 'content': 'hello'}]
        with patch(
            'ollmo_server.backend_transport_runtime.ollama_chat_with_options',
            side_effect=fake_ollama_chat_with_options,
        ):
            result = owner.ollama_chat(11434, 'gemma4:26b', messages)

        self.assertEqual(result, {'content': 'ok'})
        self.assertEqual(captured['port'], 11434)
        self.assertEqual(captured['model_name'], 'gemma4:26b')
        self.assertEqual(captured['messages'], messages)
        self.assertEqual(captured['timeout_sec'], 180)
        self.assertFalse(captured['allow_port_fallback'])


if __name__ == '__main__':
    unittest.main()
