import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ollmo_runtime.ollama_model_manager import (
    build_ollama_env,
    list_local_model_entries,
    safe_log_fragment,
    start_model,
    wait_for_embedding_model_loaded,
)


class SafeLogFragmentTests(unittest.TestCase):
    def test_replaces_slash_and_colon(self):
        self.assertEqual(
            safe_log_fragment("x/flux2-klein:latest-1"),
            "x_flux2-klein_latest-1",
        )

    def test_empty_fallback(self):
        self.assertEqual(safe_log_fragment(""), "model")

    @patch("ollmo_runtime.ollama_model_manager.OLLAMA_LIBRARY_DIR_CANDIDATES", [Path("/opt/homebrew/lib")])
    @patch("pathlib.Path.exists", return_value=True)
    def test_build_ollama_env_adds_library_paths_and_host(self, _mock_exists):
        with patch.dict(
            os.environ,
            {
                'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'must-not-reach-model',
                'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'must-not-reach-model',
            },
            clear=True,
        ):
            env = build_ollama_env(port=11436)

        self.assertEqual(env["OLLAMA_HOST"], "127.0.0.1:11436")
        self.assertEqual(env["OLLAMA_LIBRARY_PATH"], "/opt/homebrew/lib")
        self.assertEqual(env["DYLD_LIBRARY_PATH"], "/opt/homebrew/lib")
        self.assertEqual(env["DYLD_FALLBACK_LIBRARY_PATH"], "/opt/homebrew/lib")
        self.assertNotIn('OLLMO_GRAPH_REBASE_OPERATOR_TOKEN', env)
        self.assertNotIn('OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY', env)

    @patch("ollmo_runtime.ollama_model_manager._fetch_model_metadata")
    @patch("ollmo_runtime.ollama_model_manager.subprocess.run")
    def test_list_local_model_entries_uses_provider_embedding_capability(
        self,
        mock_run,
        mock_fetch_model_metadata,
    ):
        mock_run.return_value = Mock(
            stdout="NAME ID SIZE MODIFIED\nembeddinggemma:latest abc 621 MB now\n",
        )
        mock_fetch_model_metadata.return_value = {"capabilities": ["embedding"]}

        entries = list_local_model_entries()

        self.assertEqual(entries[0]["name"], "embeddinggemma:latest")
        self.assertEqual(entries[0]["capability"], "embedding")
        self.assertEqual(entries[0]["provider_capabilities"], ["embedding"])

    @patch("ollmo_runtime.ollama_model_manager.requests.post")
    @patch("ollmo_runtime.ollama_model_manager.time.sleep")
    def test_wait_for_embedding_model_loaded_accepts_embed_response(self, _mock_sleep, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_post.return_value = mock_response

        ready = wait_for_embedding_model_loaded(11435, "embeddinggemma:latest", timeout=1)

        self.assertTrue(ready)
        self.assertIn("/api/embed", mock_post.call_args.args[0])

    @patch("pathlib.Path.mkdir")
    @patch("ollmo_runtime.ollama_model_manager.open", create=True)
    @patch("ollmo_runtime.ollama_model_manager.wait_for_embedding_model_loaded", return_value=True)
    @patch("ollmo_runtime.ollama_model_manager.is_port_listening", return_value=True)
    @patch("ollmo_runtime.ollama_model_manager.find_free_port", return_value=11435)
    @patch("ollmo_runtime.ollama_model_manager.allocate_instance_id", return_value="embeddinggemma:latest-1")
    @patch("ollmo_runtime.ollama_model_manager.subprocess.Popen")
    def test_start_model_persists_resolved_embedding_capability(
        self,
        mock_popen,
        _mock_allocate_instance_id,
        _mock_find_free_port,
        _mock_is_port_listening,
        _mock_wait_for_embedding_model_loaded,
        mock_open,
        _mock_path_mkdir,
    ):
        serve_process = Mock(pid=4321)
        mock_popen.return_value = serve_process
        log_handle = Mock()
        log_handle.closed = False
        mock_open.return_value = log_handle

        instance = start_model(
            "embeddinggemma:latest",
            used_ports=set(),
            existing_instances=[],
            capability="embedding",
            start_source="api_start_model",
        )

        self.assertEqual(instance["capability"], "embedding")
        self.assertEqual(instance["outputs"], ["embedding"])


if __name__ == "__main__":
    unittest.main()
