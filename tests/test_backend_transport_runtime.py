import unittest
from unittest.mock import patch

from ollmo_server.backend_transport_runtime import BackendTransportRuntimeOwner


class BackendTransportRuntimeTests(unittest.TestCase):
    @staticmethod
    def _owner():
        return BackendTransportRuntimeOwner(
            hooks={'requests_module': object()},
            capability_chat='chat',
            request_timeout_error=TimeoutError,
            request_connection_error=ConnectionError,
            request_exception_error=Exception,
        )

    def test_ollama_chat_uses_default_transport_timeout(self):
        owner = self._owner()
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

    def test_mlx_audio_speech_forwards_adaptive_max_tokens(self):
        owner = self._owner()

        with patch(
            'ollmo_server.backend_transport_runtime.mlx_audio_speech',
            return_value={'audio_bytes': b'RIFFfakewav'},
        ) as mock_speech:
            result = owner.mlx_audio_speech(
                11502,
                'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
                'Ein kurzer deutscher Sprechtext.',
                lang_code='auto',
                max_tokens=144,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.05,
                timeout_sec=1200,
            )

        self.assertEqual(result, {'audio_bytes': b'RIFFfakewav'})
        _args, kwargs = mock_speech.call_args
        self.assertEqual(kwargs['lang_code'], 'auto')
        self.assertEqual(kwargs['max_tokens'], 144)
        self.assertEqual(kwargs['temperature'], 0.9)
        self.assertEqual(kwargs['top_p'], 1.0)
        self.assertEqual(kwargs['top_k'], 50)
        self.assertEqual(kwargs['repetition_penalty'], 1.05)
        self.assertEqual(kwargs['timeout_sec'], 1200)


if __name__ == '__main__':
    unittest.main()
