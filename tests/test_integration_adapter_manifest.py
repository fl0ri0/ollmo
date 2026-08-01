import unittest

from ollmo_integrations.adapter_manifest import (
    build_adapter_manifest,
    get_adapter_manifest,
    list_adapter_manifests,
)
from ollmo_integrations.registry import integration_module, named_module


class IntegrationAdapterManifestTests(unittest.TestCase):
    def test_lists_current_external_client_adapter_manifests(self):
        integration_ids = [manifest.integration_id for manifest in list_adapter_manifests()]

        self.assertEqual(integration_ids, ['codex'])

    def test_build_adapter_manifest_is_serializable_and_scoped(self):
        manifest = build_adapter_manifest(['codex'])

        self.assertEqual(manifest['kind'], 'ollmo.integration_adapter_manifest')
        self.assertEqual(len(manifest['adapters']), 1)
        adapter = manifest['adapters'][0]
        self.assertEqual(adapter['integration_id'], 'codex')
        self.assertEqual(adapter['status'], 'available')
        self.assertIn(['python3', 'scripts/sync_model_providers.py'], adapter['sync_commands'])
        self.assertIn(['python3', 'scripts/unsync_model_providers.py', '--integration', 'codex'], adapter['unsync_commands'])
        self.assertIn('~/.codex/config.toml', adapter['managed_paths'])

    def test_get_adapter_manifest_rejects_unknown_id(self):
        with self.assertRaises(KeyError):
            get_adapter_manifest('missing-client')

    def test_registry_can_resolve_adapter_manifest_module(self):
        module = named_module('adapter_manifest')

        self.assertTrue(callable(module.build_adapter_manifest))

    def test_registry_can_resolve_unsync_modules(self):
        module = named_module('provider_unsync')
        codex_unsync_module = integration_module('codex', 'unsync')

        self.assertTrue(callable(module.unsync_integrations))
        self.assertTrue(callable(codex_unsync_module.unsync_codex_config))

    def test_registry_keeps_codex_projection_and_execution_as_separate_roles(self):
        projection_module = integration_module('codex', 'sync')
        execution_module = integration_module('codex', 'execution')

        self.assertTrue(callable(projection_module.sync_codex_config))
        self.assertTrue(callable(execution_module.execute_codex_text))
        self.assertIsNot(projection_module, execution_module)


if __name__ == '__main__':
    unittest.main()
