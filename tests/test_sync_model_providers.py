import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ollmo_integrations import provider_sync as sync_model_providers


class SyncModelProvidersTests(unittest.TestCase):
    @patch('ollmo_integrations.provider_sync.sync_downstream_integrations')
    @patch('ollmo_integrations.provider_sync.load_instances')
    def test_main_delegates_to_codex_downstream_sync(self, mock_load_instances, mock_sync):
        mock_load_instances.return_value = [{'instance_id': 'qwen3-coder:latest-1', 'model': 'qwen3-coder:latest', 'port': 11435}]
        mock_sync.return_value = SimpleNamespace(codex_changed=True)

        sync_model_providers.main()

        mock_sync.assert_called_once_with(mock_load_instances.return_value)


if __name__ == '__main__':
    unittest.main()
