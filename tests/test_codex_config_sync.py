import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_integrations.codex.config_sync import build_provider_blocks, sync_codex_config


class CodexConfigSyncTests(unittest.TestCase):
    def test_build_provider_blocks_uses_control_plane_adapter_when_instance_id_exists(self):
        instances = [
            {
                "instance_id": "sample/model:1",
                "model": "sample-model",
                "port": 11436,
                "backend": "ollama",
            }
        ]

        with patch.dict("os.environ", {"OLLMO_WEB_BASE": "http://127.0.0.1:5001"}):
            blocks, count = build_provider_blocks(instances)

        self.assertEqual(count, 1)
        self.assertIn('[model_providers.local-11436]', blocks)
        self.assertIn(
            'base_url = "http://127.0.0.1:5001/api/local_provider/sample%2Fmodel%3A1/v1"',
            blocks,
        )
        self.assertIn('wire_api = "responses"', blocks)

    def test_build_provider_blocks_falls_back_to_raw_port_when_instance_id_missing(self):
        instances = [
            {
                "model": "sample-model",
                "port": 11437,
                "backend": "ollama",
            }
        ]

        blocks, count = build_provider_blocks(instances)

        self.assertEqual(count, 1)
        self.assertIn('base_url = "http://127.0.0.1:11437/v1"', blocks)

    def test_sync_codex_config_rewrites_local_sections_with_adapter_urls(self):
        original = (
            'model = "gpt-5.4"\n\n'
            '[model_providers.openai]\n'
            'name = "cloud"\n\n'
            '[model_providers.local-11435]\n'
            'name = "old local"\n'
            'base_url = "http://127.0.0.1:11435/v1"\n'
            'wire_api = "responses"\n'
        )
        instances = [
            {
                "instance_id": "sample:model",
                "model": "sample-model",
                "port": 11439,
                "backend": "ollama",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(original, encoding="utf-8")

            with patch.dict("os.environ", {"OLLMO_WEB_BASE": "http://127.0.0.1:5001"}):
                changed = sync_codex_config(instances, config_path=config_path)

            self.assertTrue(changed)
            updated = config_path.read_text(encoding="utf-8")
            self.assertIn('[model_providers.openai]', updated)
            self.assertNotIn('[model_providers.local-11435]', updated)
            self.assertIn('[model_providers.local-11439]', updated)
            self.assertIn(
                'base_url = "http://127.0.0.1:5001/api/local_provider/sample%3Amodel/v1"',
                updated,
            )


if __name__ == "__main__":
    unittest.main()
