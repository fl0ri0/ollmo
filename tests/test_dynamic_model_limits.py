import unittest
from unittest.mock import patch

from ollmo_runtime import ollama_model_manager


class DynamicModelLimitsTests(unittest.TestCase):
    @patch("ollmo_runtime.ollama_model_manager.list_local_model_entries")
    @patch("ollmo_runtime.ollama_model_manager.ensure_default_server_running")
    def test_model_manager_does_not_invent_limits(self, mock_ensure_default_server_running, mock_list_local_model_entries):
        mock_ensure_default_server_running.return_value = True
        mock_list_local_model_entries.return_value = [
            {"name": "gpt-oss:latest", "capability": "chat"},
            {"name": "qwen3-coder:latest", "capability": "chat"},
        ]

        payload = ollama_model_manager.get_available_models(include_limits=True)

        self.assertEqual(
            payload,
            [
                {"name": "gpt-oss:latest", "capability": "chat"},
                {"name": "qwen3-coder:latest", "capability": "chat"},
            ],
        )

if __name__ == "__main__":
    unittest.main()
