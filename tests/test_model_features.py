import unittest
import tempfile
from pathlib import Path

from helpers.model_capabilities import (
    build_feature_contract,
    build_registry_metadata,
    infer_capability,
)


class ModelFeaturesTests(unittest.TestCase):
    def test_chat_contract_defaults_to_text_io_and_conservative_features(self):
        payload = build_feature_contract('gpt-oss:20b', 'ollama', 'chat')

        self.assertEqual(payload['inputs'], ['text'])
        self.assertEqual(payload['outputs'], ['text'])
        self.assertFalse(payload['features']['tool_calling'])
        self.assertFalse(payload['features']['vision_input'])
        self.assertFalse(payload['features']['image_output'])

    def test_vision_contract_exposes_image_input(self):
        payload = build_feature_contract('mlx-community/GLM-OCR-bf16', 'mlx', 'vision_analysis')

        self.assertEqual(payload['inputs'], ['text', 'image'])
        self.assertEqual(payload['outputs'], ['text'])
        self.assertTrue(payload['features']['vision_input'])
        self.assertFalse(payload['features']['audio_input'])

    def test_registry_metadata_respects_explicit_feature_overrides(self):
        payload = build_registry_metadata(
            'gpt-oss:20b',
            'ollama',
            'chat',
            metadata={
                'features': {
                    'tool_calling': True,
                    'function_calling': True,
                    'structured_outputs': True,
                },
                'inputs': ['text', 'image'],
            },
        )

        self.assertTrue(payload['features']['tool_calling'])
        self.assertTrue(payload['features']['function_calling'])
        self.assertTrue(payload['features']['structured_outputs'])
        self.assertEqual(payload['inputs'], ['text', 'image'])
        self.assertTrue(payload['features']['vision_input'])
        self.assertEqual(payload['feature_sources']['tool_calling'], 'explicit_metadata')

    def test_local_template_evidence_enables_tool_calling_and_vision_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'chat_template.jinja').write_text(
                '{% if tools %}<tools>{% endif %}<|vision_start|><|image_pad|><|vision_end|>',
                encoding='utf-8',
            )

            payload = build_feature_contract(
                'mlx-community/Qwen3.5-27B-4bit',
                'mlx',
                'chat',
                metadata={'model_path': str(path)},
            )

        self.assertTrue(payload['features']['tool_calling'])
        self.assertTrue(payload['features']['function_calling'])
        self.assertTrue(payload['features']['vision_input'])
        self.assertIn('image', payload['inputs'])
        self.assertEqual(payload['feature_sources']['tool_calling'], 'local_template')
        self.assertEqual(payload['feature_sources']['vision_input'], 'local_template')

    def test_curated_override_marks_qwen3_coder_as_tool_capable(self):
        payload = build_feature_contract('qwen3-coder:latest', 'ollama', 'chat')

        self.assertTrue(payload['features']['tool_calling'])
        self.assertTrue(payload['features']['function_calling'])
        self.assertEqual(payload['feature_sources']['tool_calling'], 'curated_override')

    def test_backend_metadata_capabilities_enable_tool_and_vision_flags(self):
        payload = build_feature_contract(
            'qwen2.5-vl:latest',
            'ollama',
            'chat',
            metadata={
                'backend_metadata': {
                    'source': 'ollama_api_show',
                    'capabilities': ['vision', 'tools'],
                }
            },
        )

        self.assertTrue(payload['features']['tool_calling'])
        self.assertTrue(payload['features']['function_calling'])
        self.assertTrue(payload['features']['vision_input'])
        self.assertIn('image', payload['inputs'])
        self.assertEqual(payload['feature_sources']['tool_calling'], 'backend_metadata')
        self.assertEqual(payload['feature_sources']['vision_input'], 'backend_metadata')

    def test_provider_completion_plus_vision_prefers_chat_as_primary_capability(self):
        capability = infer_capability(
            'qwen3.5:35b-a3b-coding-nvfp4',
            'ollama',
            metadata={
                'backend_metadata': {
                    'source': 'ollama_api_show',
                    'capabilities': ['completion', 'vision', 'thinking', 'tools'],
                }
            },
        )

        self.assertEqual(capability, 'chat')

    def test_infer_capability_accepts_multimodal_pipeline_tag(self):
        capability = infer_capability(
            'mlx-community/gemma-4-31b-it-bf16',
            'mlx',
            metadata={
                'snapshot_pipeline_tag': 'image-text-to-text',
                'provider_capabilities': ['vision_analysis'],
                'inputs': ['text', 'image'],
                'outputs': ['text'],
            },
        )

        self.assertEqual(capability, 'vision_analysis')

    def test_registry_metadata_marks_multimodal_completion_model_as_text_capable(self):
        payload = build_registry_metadata(
            'qwen3.5:35b-a3b-coding-nvfp4',
            'ollama',
            None,
            metadata={
                'backend_metadata': {
                    'source': 'ollama_api_show',
                    'capabilities': ['completion', 'vision', 'thinking', 'tools'],
                }
            },
        )

        self.assertEqual(payload['capability'], 'chat')
        self.assertTrue(payload['text_capable'])
        self.assertIn('chat', payload['supported_capabilities'])
        self.assertIn('vision_analysis', payload['supported_capabilities'])
        self.assertIn('chat', payload['provider_capabilities'])
        self.assertIn('vision_analysis', payload['provider_capabilities'])
        self.assertTrue(payload['features']['tool_calling'])
        self.assertTrue(payload['features']['vision_input'])

    def test_registry_metadata_preserves_ollama_backend_summary(self):
        payload = build_registry_metadata(
            'gpt-oss:20b',
            'ollama',
            'chat',
            metadata={
                'backend_metadata': {
                    'source': 'ollama_api_show',
                    'capabilities': ['tools'],
                    'context_length': 32768,
                }
            },
        )

        self.assertIn('backend_metadata', payload)
        self.assertEqual(payload['backend_metadata']['source'], 'ollama_api_show')
        self.assertTrue(payload['features']['tool_calling'])
        self.assertEqual(payload['feature_sources']['tool_calling'], 'backend_metadata')

    def test_mlx_vlm_backend_metadata_marks_vision_input_as_backend_derived(self):
        payload = build_registry_metadata(
            'mlx-community/DeepSeek-OCR-2-bf16',
            'mlx',
            'vision_analysis',
            metadata={
                'backend_metadata': {
                    'source': 'mlx_package_contract',
                    'capabilities': ['vision'],
                }
            },
        )

        self.assertTrue(payload['features']['vision_input'])
        self.assertEqual(payload['feature_sources']['vision_input'], 'backend_metadata')

    def test_snapshot_pipeline_tag_drives_tts_capability_and_audio_output(self):
        payload = build_registry_metadata(
            'mlx-community/Voxtral-4B-TTS-2603-mlx-bf16',
            'mlx',
            None,
            metadata={
                'snapshot_pipeline_tag': 'text-to-speech',
                'provider_capabilities': ['text_to_speech'],
                'inputs': ['text'],
                'outputs': ['audio'],
            },
        )

        self.assertEqual(payload['capability'], 'text_to_speech')
        self.assertTrue(payload['features']['audio_output'])
        self.assertEqual(payload['outputs'], ['audio'])

    def test_provider_capabilities_override_broad_mlx_audio_package_capabilities(self):
        payload = build_registry_metadata(
            'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            'mlx',
            'speech_to_text',
            metadata={
                'provider_capabilities': ['text_to_speech'],
                'backend_metadata': {
                    'backend_package': 'mlx_audio',
                    'backend_contract': 'mlx_audio.server',
                    'capabilities': ['text_to_speech', 'speech_to_text', 'speech_to_speech'],
                },
                'inputs': ['text'],
                'outputs': ['audio'],
            },
        )

        self.assertEqual(payload['capability'], 'text_to_speech')
        self.assertIn('text_to_speech', payload['supported_capabilities'])
        self.assertEqual(payload['provider_capabilities'][0], 'text_to_speech')


if __name__ == '__main__':
    unittest.main()
