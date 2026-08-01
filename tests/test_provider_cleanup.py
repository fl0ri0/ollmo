import unittest

from ollmo_integrations.codex import provider_cleanup as cleanup_model_providers


class ProviderCleanupTests(unittest.TestCase):
    def test_codex_cleanup_can_remove_only_explicit_stopped_ports(self):
        original = (
            '[model_providers.local-11435]\n'
            'name = "chat"\n'
            '\n'
            '[model_providers.local-11436]\n'
            'name = "image"\n'
            '\n'
            '[model_providers.openai]\n'
            'name = "cloud"\n'
        )

        updated, removed = cleanup_model_providers.purge_inactive_sections(
            original,
            set(),
            explicit_remove_ports={11435},
        )

        self.assertEqual(removed, ["local-11435"])
        self.assertNotIn("[model_providers.local-11435]", updated)
        self.assertIn("[model_providers.local-11436]", updated)
        self.assertIn("[model_providers.openai]", updated)

if __name__ == "__main__":
    unittest.main()
