import tempfile
import unittest
from pathlib import Path

from helpers.session_controls import build_session_controls
from ollmo_core.lifecycle import list_running_instances


class SessionControlsTests(unittest.TestCase):
    def test_chat_schema_exposes_sampling_controls(self):
        schema = build_session_controls(
            {
                'model': 'gpt-oss:20b',
                'backend': 'ollama',
                'capability': 'chat',
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertIn('temperature', schema['fields'])
        self.assertIn('top_p', schema['fields'])
        self.assertIn('Leave blank for the model default', schema['fields']['temperature']['description'])
        self.assertIn('probability mass', schema['fields']['top_p']['description'])

    def test_generic_vision_analysis_schema_exposes_ocr_controls(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
                'backend': 'mlx',
                'capability': 'vision_analysis',
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertIn('pdf_max_pages', schema['fields'])
        self.assertEqual(schema['fields']['ocr_meta']['label'], 'OCR / Document Mode')
        self.assertIn('vision-analysis model', schema['hint'])

    def test_glm_ocr_schema_exposes_model_aware_document_modes(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/GLM-OCR-bf16',
                'backend': 'mlx',
                'capability': 'vision_analysis',
            }
        )

        self.assertEqual(schema['fields']['ocr_mode']['options'], ['auto', 'text', 'table', 'formula', 'extract'])
        self.assertIn('GLM-OCR', schema['hint'])
        self.assertEqual(schema['fields']['ocr_mode']['label'], 'Document Mode')
        self.assertIn('GLM-OCR parsing preset', schema['fields']['ocr_mode']['description'])
        self.assertEqual(schema['fields']['pdf_max_pages']['label'], 'Max Page Budget')
        self.assertIn('Leave blank to process all PDF pages', schema['fields']['pdf_max_pages']['description'])

    def test_deepseek_ocr2_schema_exposes_model_aware_document_modes(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/DeepSeek-OCR-2-bf16',
                'backend': 'mlx',
                'capability': 'vision_analysis',
            }
        )

        self.assertEqual(schema['fields']['ocr_mode']['options'], ['auto', 'markdown', 'free_ocr', 'extract'])
        self.assertIn('DeepSeek-OCR-2', schema['hint'])

    def test_tts_schema_uses_snapshot_metadata(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                'backend': 'mlx',
                'capability': 'text_to_speech',
                'tts_model_type': 'custom_voice',
                'tts_speakers': ['serena', 'vivian'],
                'tts_languages': ['auto', 'english', 'german'],
                'tts_response_formats': ['mp3', 'wav', 'flac'],
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertEqual(schema['hint'], 'Speaker-based TTS controls for this CustomVoice model.')
        self.assertEqual(schema['fields']['tts_voice']['options'], ['serena', 'vivian'])
        self.assertTrue(schema['fields']['tts_voice']['default_first_option'])
        self.assertIn('tts_response_format', schema['fields'])
        self.assertEqual(schema['fields']['tts_response_format']['options'], ['mp3', 'wav', 'flac'])
        self.assertIn('tts_instruct', schema['fields'])
        self.assertNotIn('required', schema['fields']['tts_instruct'])

    def test_whisper_shim_schema_mentions_local_shim(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/whisper-large-v3-mlx',
                'backend': 'mlx',
                'backend_package': 'mlx_whisper_shim',
                'capability': 'speech_to_text',
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertEqual(schema['fields']['stt_meta']['label'], 'Whisper Shim Input')
        self.assertIn('local MLX Whisper shim', schema['hint'])
        self.assertEqual(schema['fields']['stt_language']['label'], 'Language')
        self.assertIn('auto-detect', schema['fields']['stt_language']['description'])
        self.assertEqual(schema['fields']['stt_task']['label'], 'Task')

    def test_realtime_stt_schema_uses_metadata_driven_hint(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/Voxtral-Realtime-bf16',
                'backend': 'mlx',
                'backend_package': 'mlx_audio',
                'capability': 'speech_to_text',
                'stt_realtime': True,
                'stt_languages': ['en', 'de'],
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertEqual(schema['fields']['stt_meta']['label'], 'Realtime STT Input')
        self.assertIn('realtime', schema['hint'].lower())
        self.assertEqual(schema['fields']['stt_language']['options'], ['en', 'de'])

    def test_image_generation_schema_exposes_root_level_size_controls(self):
        schema = build_session_controls(
            {
                'model': 'x/flux2-klein:latest',
                'backend': 'ollama',
                'capability': 'image_generation',
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertIn('image_aspect_ratio', schema['fields'])
        self.assertIn('image_width', schema['fields'])
        self.assertIn('image_height', schema['fields'])
        self.assertIn('image_count', schema['fields'])
        self.assertEqual(schema['fields']['image_aspect_ratio']['options'][0], 'auto')
        self.assertIn('3:2', schema['fields']['image_aspect_ratio']['options'])
        self.assertIn('2:3', schema['fields']['image_aspect_ratio']['options'])
        self.assertIn('multiple of 16', schema['fields']['image_width']['description'])
        self.assertIn('base image', schema['fields']['image_meta']['description'])

    def test_kitten_tts_schema_exposes_required_speaker_with_default(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/kitten-tts-mini-0.8-bf16',
                'backend': 'mlx',
                'capability': 'text_to_speech',
                'tts_model_type': 'kitten_tts',
                'tts_speakers': ['Bella', 'Jasper'],
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertEqual(schema['hint'], 'Speaker-based TTS controls for this Kitten model.')
        self.assertEqual(schema['fields']['tts_voice']['options'], ['Bella', 'Jasper'])
        self.assertTrue(schema['fields']['tts_voice']['default_first_option'])
        self.assertTrue(schema['fields']['tts_voice']['required'])
        self.assertIn('Kitten TTS models require', schema['fields']['tts_voice']['required_message'])
        self.assertIn('built-in Kitten speakers', schema['fields']['tts_voice']['description'])

    def test_voicedesign_tts_schema_marks_instruct_required(self):
        schema = build_session_controls(
            {
                'model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
                'backend': 'mlx',
                'capability': 'text_to_speech',
                'tts_model_type': 'voice_design',
            }
        )

        self.assertTrue(schema['enabled'])
        self.assertEqual(schema['hint'], 'Natural-language voice design controls for this VoiceDesign model.')
        self.assertTrue(schema['fields']['tts_instruct']['required'])
        self.assertIn('VoiceDesign models require', schema['fields']['tts_instruct']['required_message'])
        self.assertEqual(schema['fields']['tts_instruct']['label'], 'Style / Instruct')
        self.assertIn('Describe the target voice', schema['fields']['tts_instruct']['description'])

    def test_list_running_instances_includes_session_controls(self):
        original_list_runtime_entries = list_running_instances.__globals__['list_runtime_entries']
        original_read_snapshot_model_metadata = list_running_instances.__globals__['read_snapshot_model_metadata']
        try:
            list_running_instances.__globals__['list_runtime_entries'] = lambda **kwargs: [
                {
                    'instance_id': 'glm-ocr-1',
                    'model': 'mlx-community/GLM-OCR-bf16',
                    'backend': 'mlx',
                    'capability': 'vision_analysis',
                    'port': 11501,
                }
            ]
            list_running_instances.__globals__['read_snapshot_model_metadata'] = lambda *args, **kwargs: {}

            with tempfile.TemporaryDirectory() as tmpdir:
                registry_path = Path(tmpdir) / 'model_ports.json'
                registry_path.write_text('[]\n', encoding='utf-8')
                payload = list_running_instances(config_path=str(registry_path))
        finally:
            list_running_instances.__globals__['list_runtime_entries'] = original_list_runtime_entries
            list_running_instances.__globals__['read_snapshot_model_metadata'] = original_read_snapshot_model_metadata

        self.assertEqual(len(payload), 1)
        self.assertIn('session_controls', payload[0])
        self.assertIn('pdf_dpi', payload[0]['session_controls']['fields'])


if __name__ == '__main__':
    unittest.main()
