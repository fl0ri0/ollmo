import tempfile
import unittest
from pathlib import Path
from typing import Optional

from ollmo_core.transports import (
    ollama_generate,
    mlx_audio_speech,
    persist_audio_bytes_locally,
)


class FakeResponse:
    def __init__(
        self,
        *,
        content: bytes,
        headers: Optional[dict[str, str]] = None,
        status_code: int = 200,
        json_payload=None,
        json_error: Optional[Exception] = None,
    ):
        self.content = content
        self.headers = headers or {}
        self.status_code = status_code
        self._json_payload = json_payload
        self._json_error = json_error
        self.text = content.decode('utf-8', errors='replace') if isinstance(content, bytes) else str(content or '')

    def raise_for_status(self):
        return None

    def json(self):
        if self._json_error:
            raise self._json_error
        if self._json_payload is not None:
            return self._json_payload
        return {}


class FakeRequests:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({'url': url, 'json': json, 'timeout': timeout})
        return self.response


class TransportAudioTests(unittest.TestCase):
    def test_mlx_audio_speech_defaults_blank_format_to_wav(self):
        fake_requests = FakeRequests(
            FakeResponse(content=b'RIFFfakewav', headers={}),
        )

        result = mlx_audio_speech(
            11505,
            'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            'Hello from Ollmo.',
            fake_requests,
            response_format=None,
            timeout_sec=1200,
        )

        self.assertEqual(fake_requests.calls[0]['json']['response_format'], 'wav')
        self.assertNotIn('max_tokens', fake_requests.calls[0]['json'])
        self.assertEqual(result['content_type'], 'audio/wav')
        self.assertEqual(result['audio_bytes'], b'RIFFfakewav')

    def test_mlx_audio_speech_allows_explicit_max_tokens_override(self):
        fake_requests = FakeRequests(
            FakeResponse(content=b'RIFFfakewav', headers={}),
        )

        mlx_audio_speech(
            11505,
            'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            'Hello from Ollmo.',
            fake_requests,
            response_format='wav',
            max_tokens=2048,
            timeout_sec=1200,
        )

        self.assertEqual(fake_requests.calls[0]['json']['max_tokens'], 2048)

    def test_persist_audio_bytes_locally_uses_mp3_extension_for_mpeg_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            out_path = persist_audio_bytes_locally(
                b'ID3\x04\x00\x00fake-mp3',
                model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
                output_dir=output_dir,
                response_format=None,
                content_type='audio/mpeg',
            )

            self.assertIsNotNone(out_path)
            self.assertTrue(str(out_path).endswith('.mp3'))
            self.assertTrue(Path(out_path).exists())

    def test_ollama_generate_reports_empty_non_json_response(self):
        fake_requests = FakeRequests(
            FakeResponse(content=b'', json_error=ValueError('empty json')),
        )

        with self.assertRaisesRegex(RuntimeError, 'non-JSON response.*<empty body>'):
            ollama_generate(
                11436,
                'x/flux2-klein:latest',
                'An elephant in a store.',
                fake_requests,
                TimeoutError,
                ConnectionError,
                RuntimeError,
                max_retries=1,
                allow_port_fallback=False,
            )


if __name__ == '__main__':
    unittest.main()
