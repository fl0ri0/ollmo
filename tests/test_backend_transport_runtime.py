import unittest
from unittest.mock import Mock, patch

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

    @staticmethod
    def _request_owner(post):
        return BackendTransportRuntimeOwner(
            hooks={
                'requests_module': object(),
                'chat_timeout_seconds': lambda *_args: 180,
                'normalize_chat_messages_for_backend': lambda messages, **_kwargs: messages,
                'requests_post': post,
            },
            capability_chat='chat',
            request_timeout_error=TimeoutError,
            request_connection_error=ConnectionError,
            request_exception_error=Exception,
        )

    def test_mlx_reasoning_effort_is_explicit_and_reasoning_is_not_public_content(self):
        for effort in ('low', 'medium', 'xhigh'):
            with self.subTest(effort=effort):
                response = Mock()
                response.json.return_value = {
                    'choices': [{'message': {'content': '', 'reasoning': 'private'}}],
                }
                post = Mock(return_value=response)
                owner = self._request_owner(post)

                result = owner.execute_chat_backend_request(
                    target_port=11502,
                    model_name='mlx-qwen',
                    backend='mlx',
                    capability='chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    reasoning_effort=effort,
                )

                payload = post.call_args.kwargs['json']
                self.assertEqual(result, '')
                self.assertTrue(payload['enable_thinking'])
                self.assertEqual(payload['reasoning_effort'], effort)

    def test_mlx_reasoning_is_disabled_when_effort_is_omitted_or_off(self):
        for effort in (None, 'off', 'none', 'disabled'):
            with self.subTest(effort=effort):
                response = Mock()
                response.json.return_value = {
                    'choices': [{'message': {'content': 'answer', 'reasoning': 'private'}}],
                }
                post = Mock(return_value=response)
                owner = self._request_owner(post)

                result = owner.execute_chat_backend_request(
                    target_port=11502,
                    model_name='mlx-qwen',
                    backend='mlx',
                    capability='chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    reasoning_effort=effort,
                )

                payload = post.call_args.kwargs['json']
                self.assertEqual(result, 'answer')
                self.assertFalse(payload['enable_thinking'])
                self.assertNotIn('reasoning_effort', payload)

    def test_llama_cpp_and_ollama_never_receive_reasoning_fields(self):
        for backend in ('llama_cpp', 'ollama'):
            with self.subTest(backend=backend):
                response = Mock()
                response.json.return_value = (
                    {'choices': [{'message': {'content': 'answer'}}]}
                    if backend == 'llama_cpp'
                    else {'message': {'content': 'answer'}}
                )
                post = Mock(return_value=response)
                owner = self._request_owner(post)

                result = owner.execute_chat_backend_request(
                    target_port=11502,
                    model_name='model',
                    backend=backend,
                    capability='chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    reasoning_effort='xhigh',
                )

                payload = post.call_args.kwargs['json']
                self.assertEqual(result, 'answer')
                self.assertNotIn('enable_thinking', payload)
                self.assertNotIn('reasoning_effort', payload)

    def test_mlx_stream_uses_the_same_reasoning_contract(self):
        response = Mock()
        post = Mock(return_value=response)
        owner = self._request_owner(post)

        result = owner.open_openai_chat_stream(
            backend='mlx',
            target_port=11502,
            request_model_override=None,
            model_name='mlx-qwen',
            messages=[{'role': 'user', 'content': 'hello'}],
            timeout_sec=600,
            reasoning_effort='xhigh',
        )

        self.assertIs(result, response)
        payload = post.call_args.kwargs['json']
        self.assertTrue(payload['enable_thinking'])
        self.assertEqual(payload['reasoning_effort'], 'xhigh')


if __name__ == '__main__':
    unittest.main()
