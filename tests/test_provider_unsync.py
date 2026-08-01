import tempfile
import unittest
from pathlib import Path

from ollmo_integrations.codex.provider_unsync import unsync_codex_config


class ProviderUnsyncTests(unittest.TestCase):
    def test_codex_unsync_removes_only_ollmo_local_provider_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.toml'
            config_path.write_text(
                '[model_providers.local-11435]\n'
                'name = "chat"\n'
                '\n'
                '[model_providers.openai]\n'
                'name = "cloud"\n',
                encoding='utf-8',
            )

            changed = unsync_codex_config(config_path=config_path)

            self.assertTrue(changed)
            updated = config_path.read_text(encoding='utf-8')
            self.assertNotIn('[model_providers.local-11435]', updated)
            self.assertIn('[model_providers.openai]', updated)

if __name__ == '__main__':
    unittest.main()
