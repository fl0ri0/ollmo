import unittest
from unittest.mock import patch

from ollmo_core.lifecycle import StartModelRequestError
from ollmo_webserver import app


class StartModelApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    @patch("ollmo_webserver.start_instance")
    def test_start_chat_model(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "qwen3-coder:latest-1",
            "model": "qwen3-coder:latest",
            "port": 11435,
            "pid": 12345,
            "backend": "ollama",
            "capability": "chat",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "qwen3-coder:latest",
                "backend": "ollama",
                "capability": "chat",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "started")
        self.assertEqual(payload["instance"]["capability"], "chat")
        self.assertEqual(payload["instance"]["backend"], "ollama")
        mock_start_instance.assert_called_once_with(
            "qwen3-coder:latest",
            "ollama",
            "chat",
            model_path=None,
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_start_chat_model_reuses_existing_usable_instance(
        self,
        mock_load_running_instances,
        mock_merge_instances_with_runtime_status,
        mock_start_instance,
    ):
        existing = {
            "instance_id": "qwen3-coder:latest-1",
            "model": "qwen3-coder:latest",
            "port": 11435,
            "pid": 12345,
            "backend": "ollama",
            "capability": "chat",
            "readiness": "degraded",
            "activity": "busy",
            "process_alive": True,
            "port_listening": True,
            "runtime_status": {
                "readiness": "degraded",
                "activity": "busy",
                "process_alive": True,
                "port_listening": True,
            },
        }
        mock_load_running_instances.return_value = [existing]
        mock_merge_instances_with_runtime_status.return_value = [existing]

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "qwen3-coder:latest",
                "backend": "ollama",
                "capability": "chat",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "reused")
        self.assertTrue(payload["reused"])
        self.assertEqual(payload["instance"]["instance_id"], "qwen3-coder:latest-1")
        self.assertEqual(payload["start_source"], "api_start_model")
        self.assertEqual(payload["start_audit"]["status"], "reused")
        mock_start_instance.assert_not_called()

    @patch("ollmo_webserver.start_instance")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_frontend_button_force_start_bypasses_existing_instance_reuse(
        self,
        mock_load_running_instances,
        mock_merge_instances_with_runtime_status,
        mock_start_instance,
    ):
        existing = {
            "instance_id": "qwen3-coder:latest-1",
            "model": "qwen3-coder:latest",
            "port": 11435,
            "pid": 12345,
            "backend": "ollama",
            "capability": "chat",
            "readiness": "ready",
            "activity": "idle",
            "process_alive": True,
            "port_listening": True,
        }
        mock_load_running_instances.return_value = [existing]
        mock_merge_instances_with_runtime_status.return_value = [existing]
        mock_start_instance.return_value = {
            "instance_id": "qwen3-coder:latest-2",
            "model": "qwen3-coder:latest",
            "port": 11436,
            "pid": 23456,
            "backend": "ollama",
            "capability": "chat",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "qwen3-coder:latest",
                "backend": "ollama",
                "capability": "chat",
                "start_source": "frontend_button",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "started")
        self.assertFalse(payload.get("reused", False))
        self.assertEqual(payload["instance"]["instance_id"], "qwen3-coder:latest-2")
        self.assertEqual(payload["start_source"], "frontend_button")
        self.assertEqual(payload["start_audit"]["start_source"], "frontend_button")
        mock_load_running_instances.assert_not_called()
        mock_merge_instances_with_runtime_status.assert_not_called()
        mock_start_instance.assert_called_once_with(
            "qwen3-coder:latest",
            "ollama",
            "chat",
            model_path=None,
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="frontend_button",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_model_rejects_route_selection_start_source(self, mock_start_instance):
        response = self.client.post(
            "/api/start_model",
            json={
                "model": "qwen3-coder:latest",
                "backend": "ollama",
                "capability": "chat",
                "start_source": "ghost_route",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["policy_violation"]["start_source"], "ghost_route")
        self.assertEqual(payload["policy_violation"]["reason"], "forbidden_route_source")
        mock_start_instance.assert_not_called()

    @patch("ollmo_webserver.start_instance")
    def test_start_image_generation_model(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "x_flux2-klein-1",
            "model": "x/flux2-klein",
            "port": 11438,
            "pid": 23456,
            "backend": "ollama",
            "capability": "image_generation",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "x/flux2-klein",
                "backend": "ollama",
                "capability": "image_generation",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance"]["capability"], "image_generation")
        self.assertEqual(payload["instance"]["backend"], "ollama")
        mock_start_instance.assert_called_once_with(
            "x/flux2-klein",
            "ollama",
            "image_generation",
            model_path=None,
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_image_generation_model_without_tag_or_namespace(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "x_flux2-klein_latest-1",
            "model": "x/flux2-klein:latest",
            "port": 11438,
            "pid": 23456,
            "backend": "ollama",
            "capability": "image_generation",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "flux2-klein",
                "backend": "ollama",
                "capability": "image_generation",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance"]["model"], "x/flux2-klein:latest")
        mock_start_instance.assert_called_once_with(
            "flux2-klein",
            "ollama",
            "image_generation",
            model_path=None,
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_speech_model(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "openai__whisper-large-v3-mlx-11501",
            "model": "openai/whisper-large-v3",
            "port": 11501,
            "backend": "mlx",
            "capability": "speech_to_text",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "openai/whisper-large-v3",
                "backend": "mlx",
                "capability": "speech_to_text",
                "model_path": "/tmp/whisper",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance"]["capability"], "speech_to_text")
        self.assertEqual(payload["instance"]["backend"], "mlx")
        mock_start_instance.assert_called_once_with(
            "openai/whisper-large-v3",
            "mlx",
            "speech_to_text",
            model_path="/tmp/whisper",
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_mlx_vlm_model(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "mlx-community__Qwen2.5-VL-3B-Instruct-4bit-mlx-11520",
            "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "port": 11520,
            "backend": "mlx",
            "capability": "vision_analysis",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                "backend": "mlx",
                "capability": "vision_analysis",
                "model_path": "/tmp/qwen-vl",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance"]["capability"], "vision_analysis")
        self.assertEqual(payload["instance"]["backend"], "mlx")
        mock_start_instance.assert_called_once_with(
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "mlx",
            "vision_analysis",
            model_path="/tmp/qwen-vl",
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_text_to_speech_model(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "mlx-community__Qwen3-TTS-12Hz-0.6B-Base-bf16-mlx-11504",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "port": 11504,
            "backend": "mlx",
            "capability": "text_to_speech",
            "request_model": "/tmp/qwen3-tts",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                "backend": "mlx",
                "capability": "text_to_speech",
                "model_path": "/tmp/qwen3-tts",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance"]["capability"], "text_to_speech")
        self.assertEqual(payload["instance"]["backend"], "mlx")
        mock_start_instance.assert_called_once_with(
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "mlx",
            "text_to_speech",
            model_path="/tmp/qwen3-tts",
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_llama_cpp_model(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "gemma-3-1b-it-q4-llama_cpp-11551",
            "model": "gemma-3-1b-it-q4",
            "port": 11551,
            "backend": "llama_cpp",
            "capability": "chat",
            "request_model": "gemma-3-1b-it-q4",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "gemma-3-1b-it-q4",
                "backend": "llama.cpp",
                "capability": "chat",
                "model_path": "/tmp/gemma.gguf",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance"]["backend"], "llama_cpp")
        mock_start_instance.assert_called_once_with(
            "gemma-3-1b-it-q4",
            "llama_cpp",
            "chat",
            model_path="/tmp/gemma.gguf",
            preferred_port=None,
            hf_file=None,
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_llama_cpp_model_with_hf_file(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "gemma-4-26b-a4b-llama_cpp-11551",
            "model": "ggml-org/gemma-4-26B-A4B-it-GGUF",
            "port": 11551,
            "backend": "llama_cpp",
            "capability": "chat",
            "request_model": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "ggml-org/gemma-4-26B-A4B-it-GGUF",
                "backend": "llama.cpp",
                "capability": "chat",
                "hf_file": "gemma-4-26b-a4b-it-q4_k_m.gguf",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_start_instance.assert_called_once_with(
            "ggml-org/gemma-4-26B-A4B-it-GGUF",
            "llama_cpp",
            "chat",
            model_path=None,
            preferred_port=None,
            hf_file="gemma-4-26b-a4b-it-q4_k_m.gguf",
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_mlx_vlm_model_with_launch_defaults(self, mock_start_instance):
        mock_start_instance.return_value = {
            "instance_id": "mlx-community__Qwen2.5-VL-3B-Instruct-4bit-mlx-11520",
            "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "port": 11520,
            "backend": "mlx",
            "capability": "vision_analysis",
        }

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                "backend": "mlx",
                "capability": "vision_analysis",
                "model_path": "/tmp/qwen-vl",
                "launch_defaults": {
                    "kv_bits": 3.0,
                    "kv_quant_scheme": "turboquant",
                    "max_kv_size": 8192,
                },
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_start_instance.assert_called_once_with(
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "mlx",
            "vision_analysis",
            model_path="/tmp/qwen-vl",
            preferred_port=None,
            hf_file=None,
            launch_defaults={
                "kv_bits": 3.0,
                "kv_quant_scheme": "turboquant",
                "max_kv_size": 8192,
            },
            start_source="api_start_model",
        )

    @patch("ollmo_webserver.start_instance")
    def test_start_speech_model_runtime_error_returns_400(self, mock_start_instance):
        mock_start_instance.side_effect = StartModelRequestError(
            "No safetensors found in model path",
            status_code=400,
        )

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "mlx-community/whisper-large-v3-mlx",
                "backend": "mlx",
                "capability": "speech_to_text",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("No safetensors", payload["error"])

    @patch("ollmo_webserver.start_instance")
    def test_wrong_ollama_model_name_returns_400(self, mock_start_instance):
        mock_start_instance.side_effect = StartModelRequestError(
            "Model 'totally-unknown-model' is not available in 'ollama list'.",
            status_code=400,
        )

        response = self.client.post(
            "/api/start_model",
            json={
                "model": "totally-unknown-model",
                "backend": "ollama",
                "capability": "image_generation",
                "force_start": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("ollama list", payload["error"])


if __name__ == "__main__":
    unittest.main()
