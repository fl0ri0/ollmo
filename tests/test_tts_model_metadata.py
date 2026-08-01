import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from helpers.tts_model_metadata import read_snapshot_model_metadata, read_tts_model_metadata


class TtsModelMetadataTests(unittest.TestCase):
    def test_reads_qwen3_tts_speakers_and_languages_from_snapshot_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'config.json').write_text(
                json.dumps(
                    {
                        'tts_model_type': 'custom_voice',
                        'talker_config': {
                            'spk_id': {'serena': 1, 'vivian': 2},
                            'codec_language_id': {'english': 10, 'german': 11, 'beijing_dialect': 12},
                        },
                    }
                ),
                encoding='utf-8',
            )

            payload = read_tts_model_metadata(
                'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                str(model_dir),
            )

        self.assertEqual(payload['tts_model_type'], 'custom_voice')
        self.assertEqual(payload['tts_speakers'], ['serena', 'vivian'])
        self.assertEqual(payload['tts_languages'], ['auto', 'english', 'german'])

    def test_reads_kitten_tts_speakers_from_aliases_and_snapshot_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'config.json').write_text(
                json.dumps(
                    {
                        'model_type': 'kitten_tts',
                        'voices_path': 'voices.npz',
                        'voice_aliases': {
                            'Bella': 'expr-voice-2-f',
                            'Jasper': 'expr-voice-2-m',
                        },
                    }
                ),
                encoding='utf-8',
            )
            with zipfile.ZipFile(model_dir / 'voices.npz', 'w') as archive:
                archive.writestr('expr-voice-2-f.npy', b'fake')
                archive.writestr('expr-voice-2-m.npy', b'fake')

            payload = read_tts_model_metadata(
                'mlx-community/kitten-tts-mini-0.8-bf16',
                str(model_dir),
            )

        self.assertEqual(payload['tts_model_type'], 'kitten_tts')
        self.assertEqual(payload['tts_speakers'], ['Bella', 'Jasper'])

    def test_reads_model_card_front_matter_and_voice_list_for_tts_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'config.json').write_text(
                json.dumps({'model_type': 'voxtral_tts'}),
                encoding='utf-8',
            )
            (model_dir / 'README.md').write_text(
                '---\n'
                'language:\n'
                '  - en\n'
                '  - de\n'
                'tags:\n'
                '  - text-to-speech\n'
                '  - speech generation\n'
                'pipeline_tag: text-to-speech\n'
                '---\n\n'
                '# Demo\n\n'
                '## Available Voices\n\n'
                '**English:** `casual_male`, `casual_female`\n'
                '**German:** `de_male`, `de_female`\n',
                encoding='utf-8',
            )

            payload = read_snapshot_model_metadata(
                'mlx-community/Voxtral-4B-TTS-2603-mlx-bf16',
                str(model_dir),
            )

        self.assertEqual(payload['snapshot_pipeline_tag'], 'text-to-speech')
        self.assertEqual(payload['snapshot_languages'], ['en', 'de'])
        self.assertEqual(payload['provider_capabilities'], ['text_to_speech'])
        self.assertEqual(payload['tts_languages'], ['auto', 'en', 'de'])
        self.assertEqual(
            payload['tts_speakers'],
            ['casual_male', 'casual_female', 'de_male', 'de_female'],
        )
        self.assertEqual(payload['tts_response_formats'], ['mp3', 'wav', 'flac'])

    def test_reads_multimodal_pipeline_tag_as_vision_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'config.json').write_text(
                json.dumps({'model_type': 'gemma4'}),
                encoding='utf-8',
            )
            (model_dir / 'README.md').write_text(
                '---\n'
                'pipeline_tag: image-text-to-text\n'
                'tags:\n'
                '  - mlx\n'
                '---\n',
                encoding='utf-8',
            )

            payload = read_snapshot_model_metadata(
                'mlx-community/gemma-4-31b-it-bf16',
                str(model_dir),
            )

        self.assertEqual(payload['snapshot_pipeline_tag'], 'image-text-to-text')
        self.assertEqual(payload['provider_capabilities'], ['vision_analysis'])
        self.assertEqual(payload['inputs'], ['text', 'image'])
        self.assertEqual(payload['outputs'], ['text'])

    def test_reads_vision_snapshot_tags_and_config_as_vision_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / 'config.json').write_text(
                json.dumps(
                    {
                        'model_type': 'qwen3_5',
                        'image_token_id': 248056,
                    }
                ),
                encoding='utf-8',
            )
            (model_dir / 'README.md').write_text(
                '---\n'
                'tags:\n'
                '  - mlx\n'
                '  - vision-language-model\n'
                '---\n',
                encoding='utf-8',
            )

            payload = read_snapshot_model_metadata(
                'mlx-community/Qwen3.5-9B-MLX-4bit',
                str(model_dir),
            )

        self.assertEqual(payload['provider_capabilities'], ['vision_analysis'])
        self.assertEqual(payload['inputs'], ['text', 'image'])
        self.assertEqual(payload['outputs'], ['text'])


if __name__ == '__main__':
    unittest.main()
