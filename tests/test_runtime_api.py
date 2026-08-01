import unittest
from unittest.mock import patch

import ollmo_webserver
from ollmo_core.lifecycle import StartModelRequestError, StopResult
from ollmo_webserver import app


class RuntimeApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    @patch("ollmo_webserver.list_available_models")
    def test_available_models_route_returns_runtime_service_payload(
        self,
        mock_list_available_models,
        mock_load_running_instances,
        mock_merge_instances,
        mock_build_backend_fabric,
    ):
        mock_list_available_models.return_value = [
            {
                "name": "qwen3-coder:latest",
                "model": "qwen3-coder:latest",
                "backend": "ollama",
                "capability": "chat",
            }
        ]
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_build_backend_fabric.return_value = {"summary": {"runtime_runnable_backend_count": 1}, "backends": []}

        response = self.client.get("/api/available_models?with_limits=true")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["models"]), 1)
        self.assertEqual(payload["models"][0]["model"], "qwen3-coder:latest")
        self.assertIn("backend_fabric", payload)
        self.assertIn("features", payload["models"][0])
        self.assertIn("feature_sources", payload["models"][0])
        self.assertIn("inputs", payload["models"][0])
        self.assertIn("outputs", payload["models"][0])
        mock_list_available_models.assert_called_once_with(include_limits=True)

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    @patch("ollmo_webserver.list_available_models")
    def test_available_models_route_preserves_embedding_capability(
        self,
        mock_list_available_models,
        mock_load_running_instances,
        mock_merge_instances,
        mock_build_backend_fabric,
    ):
        mock_list_available_models.return_value = [
            {
                "name": "embeddinggemma:latest",
                "model": "embeddinggemma:latest",
                "backend": "ollama",
                "capability": "embedding",
                "provider_capabilities": ["embedding"],
            }
        ]
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_build_backend_fabric.return_value = {"summary": {}, "backends": []}

        response = self.client.get("/api/available_models")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["models"][0]["capability"], "embedding")
        self.assertEqual(payload["models"][0]["outputs"], ["embedding"])

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    @patch("ollmo_webserver.list_available_models")
    def test_available_models_route_exposes_text_capable_multimodal_truth(
        self,
        mock_list_available_models,
        mock_load_running_instances,
        mock_merge_instances,
        mock_build_backend_fabric,
    ):
        mock_list_available_models.return_value = [
            {
                "name": "qwen3.5:35b-a3b-coding-nvfp4",
                "model": "qwen3.5:35b-a3b-coding-nvfp4",
                "backend": "ollama",
                "backend_metadata": {
                    "source": "ollama_api_show",
                    "capabilities": ["completion", "vision", "thinking", "tools"],
                },
            }
        ]
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_build_backend_fabric.return_value = {"summary": {}, "backends": []}

        response = self.client.get("/api/available_models")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()["models"][0]
        self.assertEqual(payload["capability"], "chat")
        self.assertTrue(payload["text_capable"])
        self.assertIn("chat", payload["supported_capabilities"])
        self.assertIn("vision_analysis", payload["supported_capabilities"])
        self.assertIn("chat", payload["provider_capabilities"])
        self.assertIn("vision_analysis", payload["provider_capabilities"])

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.list_available_models")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_backend_fabric_route_can_include_catalog(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_list_available_models,
        mock_build_backend_fabric,
    ):
        mock_load_running_instances.return_value = [{"instance_id": "gemma4:26b-1"}]
        mock_merge_instances.return_value = [{"instance_id": "gemma4:26b-1"}]
        mock_list_available_models.return_value = [{"model": "gemma4:26b", "backend": "ollama"}]
        mock_build_backend_fabric.return_value = {
            "summary": {"runtime_runnable_backend_count": 1},
            "backends": [{"backend_id": "ollama", "runtime_state": "runnable"}],
        }

        response = self.client.get("/api/backend_fabric?with_catalog=true")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["summary"]["runtime_runnable_backend_count"], 1)
        self.assertEqual(payload["backends"][0]["backend_id"], "ollama")
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "cached")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], False)
        mock_list_available_models.assert_called_once_with(include_limits=False)
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], False)

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.list_available_models")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_backend_fabric_route_refresh_is_explicit(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_list_available_models,
        mock_build_backend_fabric,
    ):
        mock_load_running_instances.return_value = [{"instance_id": "gemma4:26b-1"}]
        mock_merge_instances.return_value = [{"instance_id": "gemma4:26b-1"}]
        mock_list_available_models.return_value = []
        mock_build_backend_fabric.return_value = {"summary": {}, "backends": []}

        response = self.client.get("/api/backend_fabric?refresh=true")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "refreshed")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], True)
        mock_list_available_models.assert_not_called()
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], True)

    @patch("ollmo_webserver.cleanup_runtime_hygiene")
    @patch("ollmo_webserver.remove_instance_status")
    @patch("ollmo_webserver.stop_instance")
    def test_stop_model_route_returns_stopped_payload(
        self,
        mock_stop_instance,
        mock_remove_instance_status,
        mock_cleanup_runtime_hygiene,
    ):
        mock_stop_instance.return_value = (
            StopResult(state="stopped", message="done", details={"backend": "ollama"}),
            {"instance_id": "qwen3-coder:latest-1", "backend": "ollama"},
        )
        mock_remove_instance_status.return_value = {
            "instance_id": "qwen3-coder:latest-1",
            "readiness": "ready",
        }
        mock_cleanup_runtime_hygiene.return_value = {
            "live_instance_count": 0,
            "runtime_status_count": 0,
            "archived_count": 1,
            "archived_paths": ["/tmp/archive.log"],
        }

        response = self.client.post(
            "/api/stop_model",
            json={"instance_id": "qwen3-coder:latest-1"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "stopped")
        self.assertEqual(payload["instance"]["instance_id"], "qwen3-coder:latest-1")
        mock_stop_instance.assert_called_once_with("qwen3-coder:latest-1")
        mock_remove_instance_status.assert_called_once()
        mock_cleanup_runtime_hygiene.assert_called_once()

    @patch("ollmo_webserver.stop_instance")
    def test_stop_model_route_propagates_runtime_request_error(self, mock_stop_instance):
        mock_stop_instance.side_effect = StartModelRequestError("MLX support is not available.", status_code=501)

        response = self.client.post(
            "/api/stop_model",
            json={"instance_id": "mlx-1"},
        )

        self.assertEqual(response.status_code, 501)
        payload = response.get_json()
        self.assertIn("MLX support", payload["error"])

    @patch("ollmo_webserver.stop_instance")
    def test_stop_model_route_rejects_traversal_shaped_instance_id(self, mock_stop_instance):
        response = self.client.post(
            "/api/stop_model",
            json={"instance_id": "../secret"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid path segments", response.get_json()["error"])
        mock_stop_instance.assert_not_called()

    @patch("ollmo_webserver._log_unified_event")
    def test_runtime_status_transition_logs_degraded_as_advisory_warning(self, mock_log_unified_event):
        previous_testing = app.config.get("TESTING")
        app.config["TESTING"] = False
        try:
            ollmo_webserver._log_runtime_status_transition(
                {"readiness": "ready", "instance_id": "chat-1"},
                {
                    "readiness": "degraded",
                    "instance_id": "chat-1",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "capability": "chat",
                    "port": 11437,
                },
            )
        finally:
            app.config["TESTING"] = previous_testing

        mock_log_unified_event.assert_called_once()
        kwargs = mock_log_unified_event.call_args.kwargs
        self.assertEqual(kwargs["status"], "warning")
        self.assertEqual(kwargs["severity"], "advisory")
        self.assertEqual(kwargs["runtime_truth_note"], "degraded_readiness_is_advisory_until_live_truth_fails")

    @patch("ollmo_webserver._log_unified_event")
    def test_runtime_status_transition_keeps_unreachable_failed(self, mock_log_unified_event):
        previous_testing = app.config.get("TESTING")
        app.config["TESTING"] = False
        try:
            ollmo_webserver._log_runtime_status_transition(
                {"readiness": "ready", "instance_id": "chat-1"},
                {"readiness": "unreachable", "instance_id": "chat-1"},
            )
        finally:
            app.config["TESTING"] = previous_testing

        mock_log_unified_event.assert_called_once()
        self.assertEqual(mock_log_unified_event.call_args.kwargs["status"], "failed")

    @patch("ollmo_webserver.pull_model")
    def test_pull_model_route_passes_backend(self, mock_pull_model):
        mock_pull_model.return_value = (True, "ok")

        response = self.client.post(
            "/api/pull_model",
            json={"model": "mlx-community/Qwen3.5-27B-4bit", "backend": "mlx"},
        )

        self.assertEqual(response.status_code, 200)
        mock_pull_model.assert_called_once_with("mlx-community/Qwen3.5-27B-4bit", "mlx")

    @patch("ollmo_webserver.pull_model")
    def test_pull_model_route_passes_llama_cpp_backend(self, mock_pull_model):
        mock_pull_model.return_value = (True, "ok")

        response = self.client.post(
            "/api/pull_model",
            json={"model": "ggml-org/gemma-4-26B-A4B-it-GGUF", "backend": "llama.cpp"},
        )

        self.assertEqual(response.status_code, 200)
        mock_pull_model.assert_called_once_with("ggml-org/gemma-4-26B-A4B-it-GGUF", "llama_cpp")

    @patch("ollmo_webserver.remove_model")
    def test_remove_model_route_passes_backend(self, mock_remove_model):
        mock_remove_model.return_value = (True, "ok")

        response = self.client.post(
            "/api/remove_model",
            json={"model": "mlx-community/Qwen3.5-27B-4bit", "backend": "mlx"},
        )

        self.assertEqual(response.status_code, 200)
        mock_remove_model.assert_called_once_with("mlx-community/Qwen3.5-27B-4bit", "mlx")

    @patch("ollmo_webserver.remove_model")
    def test_remove_model_route_passes_llama_cpp_source_details(self, mock_remove_model):
        mock_remove_model.return_value = (True, "ok")

        response = self.client.post(
            "/api/remove_model",
            json={
                "model": "ggml-org/gemma-4-E4B-it-GGUF",
                "backend": "llama.cpp",
                "model_source": "hf_repo",
                "hf_repo": "ggml-org/gemma-4-E4B-it-GGUF",
                "hf_file": "gemma-4-e4b-it-q4_k_m.gguf",
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_remove_model.assert_called_once_with(
            "ggml-org/gemma-4-E4B-it-GGUF",
            "llama_cpp",
            model_source="hf_repo",
            hf_repo="ggml-org/gemma-4-E4B-it-GGUF",
            hf_file="gemma-4-e4b-it-q4_k_m.gguf",
        )


if __name__ == "__main__":
    unittest.main()
