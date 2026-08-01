import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers.model_capabilities import infer_capability, normalize_backend
from ollmo_core.lifecycle import (
    RuntimeRequestError,
    pull_model,
    remove_model,
    start_instance,
    stop_instance,
    validate_ollama_model_name,
)


class RuntimeCoreTests(unittest.TestCase):
    def test_normalize_backend_maps_llama_cpp_aliases(self):
        self.assertEqual(normalize_backend('llama.cpp'), 'llama_cpp')
        self.assertEqual(normalize_backend('llama-cpp'), 'llama_cpp')
        self.assertEqual(normalize_backend('llama_cpp'), 'llama_cpp')

    def test_infer_capability_identifies_qwen_tts_model(self):
        self.assertEqual(
            infer_capability("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", "mlx"),
            "text_to_speech",
        )

    def test_infer_capability_prefers_embedding_provider_metadata(self):
        self.assertEqual(
            infer_capability(
                "embeddinggemma:latest",
                "ollama",
                metadata={"capabilities": ["embedding"]},
            ),
            "embedding",
        )

    @patch("ollmo_core.lifecycle.manager_get_available_models")
    def test_validate_ollama_model_name_resolves_namespace_and_latest(self, mock_get_available_models):
        mock_get_available_models.return_value = ["x/flux2-klein:latest"]
        self.assertEqual(validate_ollama_model_name("flux2-klein"), "x/flux2-klein:latest")

    @patch("ollmo_core.lifecycle.manager_get_available_models")
    def test_validate_ollama_model_name_returns_400_for_unknown_model(self, mock_get_available_models):
        mock_get_available_models.return_value = ["qwen3-coder:latest"]
        with self.assertRaises(RuntimeRequestError) as ctx:
            validate_ollama_model_name("missing-model")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("ollama list", str(ctx.exception))

    @patch("ollmo_core.lifecycle.stop_mlx_instance")
    @patch("ollmo_core.lifecycle.read_registry_entries")
    def test_stop_instance_dispatches_to_mlx_backend(self, mock_read_registry_entries, mock_stop_mlx_instance):
        mock_read_registry_entries.return_value = [
            {
                "instance_id": "mlx-1",
                "backend": "mlx",
                "port": 11501,
            }
        ]
        mock_stop_mlx_instance.return_value = (True, {"instance_id": "mlx-1", "backend": "mlx"})

        result, instance = stop_instance("mlx-1")

        self.assertEqual(result.state, "stopped")
        self.assertEqual(instance["instance_id"], "mlx-1")
        mock_stop_mlx_instance.assert_called_once_with("mlx-1")

    @patch("ollmo_core.lifecycle.mlx_pull_hf_model")
    def test_pull_model_dispatches_to_mlx_hf_backend(self, mock_mlx_pull):
        mock_mlx_pull.return_value = (True, "downloaded")

        success, message = pull_model("mlx-community/Qwen3.5-27B-4bit", "mlx")

        self.assertTrue(success)
        self.assertEqual(message, "downloaded")
        mock_mlx_pull.assert_called_once_with("mlx-community/Qwen3.5-27B-4bit")

    @patch("ollmo_core.lifecycle.mlx_remove_hf_model")
    def test_remove_model_dispatches_to_mlx_hf_backend(self, mock_mlx_remove):
        mock_mlx_remove.return_value = (True, "removed")

        success, message = remove_model("mlx-community/Qwen3.5-27B-4bit", "mlx")

        self.assertTrue(success)
        self.assertEqual(message, "removed")
        mock_mlx_remove.assert_called_once_with("mlx-community/Qwen3.5-27B-4bit")

    @patch("ollmo_core.lifecycle.pull_llama_cpp_model")
    def test_pull_model_dispatches_to_llama_cpp_backend(self, mock_pull_llama_cpp_model):
        mock_pull_llama_cpp_model.return_value = (True, "downloaded")

        success, message = pull_model('ggml-org/gemma-4-26B-A4B-it-GGUF', 'llama_cpp')

        self.assertTrue(success)
        self.assertEqual(message, "downloaded")
        mock_pull_llama_cpp_model.assert_called_once_with('ggml-org/gemma-4-26B-A4B-it-GGUF')

    @patch("ollmo_core.lifecycle.remove_llama_cpp_model")
    def test_remove_model_dispatches_to_llama_cpp_backend(self, mock_remove_llama_cpp):
        mock_remove_llama_cpp.return_value = (True, "removed")

        success, message = remove_model(
            'ggml-org/gemma-3-1b-it-GGUF',
            'llama_cpp',
            model_source='hf_repo',
            hf_repo='ggml-org/gemma-3-1b-it-GGUF',
        )

        self.assertTrue(success)
        self.assertEqual(message, "removed")
        mock_remove_llama_cpp.assert_called_once_with(
            'ggml-org/gemma-3-1b-it-GGUF',
            model_source='hf_repo',
            model_path=None,
            hf_repo='ggml-org/gemma-3-1b-it-GGUF',
            hf_file=None,
        )

    @patch("ollmo_core.lifecycle.start_mlx_model")
    def test_start_instance_supports_mlx_text_to_speech(self, mock_start_mlx_model):
        mock_start_mlx_model.return_value = {
            "instance_id": "mlx-community__Qwen3-TTS-12Hz-0.6B-Base-bf16-mlx-11504",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "port": 11504,
            "backend": "mlx",
            "capability": "text_to_speech",
            "request_model": "/tmp/qwen3-tts",
            "mlx_server": "mlx_audio",
        }

        instance = start_instance(
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "mlx",
            "text_to_speech",
            model_path="/tmp/qwen3-tts",
            start_source="api_start_model",
        )

        self.assertEqual(instance["capability"], "text_to_speech")
        self.assertEqual(instance["backend"], "mlx")
        mock_start_mlx_model.assert_called_once_with(
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "/tmp/qwen3-tts",
            None,
            capability="text_to_speech",
            launch_defaults=None,
            start_source="api_start_model",
        )

    @patch("ollmo_core.lifecycle.start_mlx_model")
    def test_start_instance_supports_mlx_launch_defaults(self, mock_start_mlx_model):
        mock_start_mlx_model.return_value = {
            "instance_id": "mlx-community__Qwen2.5-VL-3B-Instruct-4bit-mlx-11520",
            "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "port": 11520,
            "backend": "mlx",
            "capability": "vision_analysis",
            "mlx_server": "mlx_vlm",
        }

        instance = start_instance(
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "mlx",
            "vision_analysis",
            model_path="/tmp/qwen-vl",
            launch_defaults={"kv_bits": 3.0, "kv_quant_scheme": "turboquant"},
            start_source="api_start_model",
        )

        self.assertEqual(instance["backend"], "mlx")
        mock_start_mlx_model.assert_called_once_with(
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "/tmp/qwen-vl",
            None,
            capability="vision_analysis",
            launch_defaults={"kv_bits": 3.0, "kv_quant_scheme": "turboquant"},
            start_source="api_start_model",
        )

    @patch("ollmo_core.lifecycle._start_ollama_instance")
    @patch("ollmo_core.lifecycle.manager_get_available_models")
    def test_start_instance_uses_available_ollama_embedding_capability_when_not_explicit(
        self,
        mock_get_available_models,
        mock_start_ollama_instance,
    ):
        mock_get_available_models.return_value = [
            {"name": "embeddinggemma:latest", "capability": "embedding"},
        ]
        mock_start_ollama_instance.return_value = {
            "instance_id": "embeddinggemma:latest-1",
            "model": "embeddinggemma:latest",
            "port": 11435,
            "backend": "ollama",
            "capability": "embedding",
        }

        instance = start_instance("embeddinggemma:latest", "ollama", None, start_source="api_start_model")

        self.assertEqual(instance["capability"], "embedding")
        mock_start_ollama_instance.assert_called_once_with(
            "embeddinggemma:latest",
            "embedding",
            start_source="api_start_model",
        )

    @patch("ollmo_core.lifecycle.start_llama_cpp_instance")
    def test_start_instance_supports_llama_cpp_chat(self, mock_start_llama_cpp_instance):
        mock_start_llama_cpp_instance.return_value = {
            'instance_id': 'gemma-3-1b-it-q4-llama_cpp-11551',
            'model': 'gemma-3-1b-it-q4',
            'port': 11551,
            'backend': 'llama_cpp',
            'capability': 'chat',
            'request_model': 'gemma-3-1b-it-q4',
            'backend_package': 'llama_cpp',
            'backend_contract': 'llama.cpp.server',
        }

        instance = start_instance(
            'gemma-3-1b-it-q4',
            'llama.cpp',
            'chat',
            model_path='/tmp/gemma.gguf',
            start_source="api_start_model",
        )

        self.assertEqual(instance['backend'], 'llama_cpp')
        self.assertEqual(instance['capability'], 'chat')
        mock_start_llama_cpp_instance.assert_called_once_with(
            'gemma-3-1b-it-q4',
            model_path='/tmp/gemma.gguf',
            preferred_port=None,
            capability='chat',
            hf_file=None,
            start_source="api_start_model",
        )

    @patch("ollmo_core.lifecycle.start_llama_cpp_instance")
    def test_start_instance_supports_llama_cpp_hf_file(self, mock_start_llama_cpp_instance):
        mock_start_llama_cpp_instance.return_value = {
            'instance_id': 'gemma-4-26b-a4b-llama_cpp-11551',
            'model': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
            'port': 11551,
            'backend': 'llama_cpp',
            'capability': 'chat',
        }

        instance = start_instance(
            'ggml-org/gemma-4-26B-A4B-it-GGUF',
            'llama_cpp',
            'chat',
            hf_file='gemma-4-26b-a4b-it-q4_k_m.gguf',
            start_source="api_start_model",
        )

        self.assertEqual(instance["backend"], "llama_cpp")
        mock_start_llama_cpp_instance.assert_called_once_with(
            'ggml-org/gemma-4-26B-A4B-it-GGUF',
            model_path=None,
            preferred_port=None,
            capability='chat',
            hf_file='gemma-4-26b-a4b-it-q4_k_m.gguf',
            start_source="api_start_model",
        )

    @patch("ollmo_core.lifecycle.stop_llama_cpp_instance")
    @patch("ollmo_core.lifecycle.read_registry_entries")
    def test_stop_instance_dispatches_to_llama_cpp_backend(self, mock_read_registry_entries, mock_stop_llama_cpp):
        mock_read_registry_entries.return_value = [
            {
                'instance_id': 'gemma-3-1b-it-q4-llama_cpp-11551',
                'backend': 'llama_cpp',
                'port': 11551,
            }
        ]
        mock_stop_llama_cpp.return_value = (
            True,
            {'instance_id': 'gemma-3-1b-it-q4-llama_cpp-11551', 'backend': 'llama_cpp'},
        )

        result, instance = stop_instance('gemma-3-1b-it-q4-llama_cpp-11551')

        self.assertEqual(result.state, 'stopped')
        self.assertEqual(instance['instance_id'], 'gemma-3-1b-it-q4-llama_cpp-11551')
        mock_stop_llama_cpp.assert_called_once_with('gemma-3-1b-it-q4-llama_cpp-11551')

    def test_start_instance_serializes_parallel_calls(self):
        barrier = threading.Barrier(2)
        state_lock = threading.Lock()
        active_calls = 0
        max_active_calls = 0
        results = []
        errors = []

        def fake_start(_model_name, _capability, **_kwargs):
            nonlocal active_calls, max_active_calls
            with state_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            time.sleep(0.1)
            with state_lock:
                active_calls -= 1
            return {
                "instance_id": "fake-instance",
                "model": "qwen3-coder:latest",
                "port": 11436,
                "pid": 12345,
                "backend": "ollama",
                "capability": "chat",
            }

        def worker():
            try:
                barrier.wait(timeout=2.0)
                result = start_instance(
                    "qwen3-coder:latest",
                    "ollama",
                    "chat",
                    start_source="api_start_model",
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "start-instance.lock"
            with patch("ollmo_core.lifecycle.START_INSTANCE_LOCK_PATH", lock_path):
                with patch("ollmo_core.lifecycle._start_ollama_instance", side_effect=fake_start):
                    threads = [threading.Thread(target=worker) for _ in range(2)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(max_active_calls, 1)


if __name__ == "__main__":
    unittest.main()
