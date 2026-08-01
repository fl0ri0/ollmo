import unittest
from unittest.mock import Mock, patch

from flask import Response

from ollmo_webserver import _chat_timeout_seconds, app


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_heavy_ollama_chat_timeout_is_extended(self):
        self.assertEqual(_chat_timeout_seconds("qwen3.5:27b", "ollama", "chat"), 600)
        self.assertEqual(_chat_timeout_seconds("qwen3-coder:latest", "ollama", "chat"), 600)
        self.assertEqual(_chat_timeout_seconds("gemma4:26b", "ollama", "chat"), 600)

    def test_default_ollama_chat_timeout_stays_standard(self):
        self.assertEqual(_chat_timeout_seconds("llama3:8b", "ollama", "chat"), 180)

    def test_mlx_chat_timeout_is_extended(self):
        self.assertEqual(_chat_timeout_seconds("mlx-community/Qwen3.5-27B-4bit", "mlx", "chat"), 900)
        self.assertEqual(_chat_timeout_seconds("mlx-community/Apertus-8B-Instruct-2509-bf16", "mlx", "chat"), 600)

    def test_llama_cpp_chat_timeout_is_extended(self):
        self.assertEqual(_chat_timeout_seconds("gemma-3-27b-it-q4", "llama_cpp", "chat"), 600)
        self.assertEqual(_chat_timeout_seconds("gemma-3-1b-it-q4", "llama_cpp", "chat"), 600)

    @patch("ollmo_webserver.load_running_instances")
    def test_chat_route_rejects_text_to_speech_capability(self, mock_running_instances):
        mock_running_instances.return_value = [
            {
                "instance_id": "tts-1",
                "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                "backend": "mlx",
                "capability": "text_to_speech",
                "port": 11504,
            }
        ]

        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "tts-1",
                "messages": [{"role": "user", "content": "Hallo"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("text_to_speech", response.get_json()["error"])

    @patch("ollmo_webserver.load_running_instances")
    def test_chat_route_rejects_embedding_capability(self, mock_running_instances):
        mock_running_instances.return_value = [
            {
                "instance_id": "embed-1",
                "model": "embeddinggemma:latest",
                "backend": "ollama",
                "capability": "embedding",
                "port": 11435,
            }
        ]

        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "embed-1",
                "messages": [{"role": "user", "content": "Hallo"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("embedding", response.get_json()["error"])

    @patch("ollmo_webserver.is_port_listening", return_value=True)
    @patch("ollmo_webserver.requests.post")
    @patch("ollmo_webserver.load_running_instances")
    def test_chat_route_uses_extended_timeout_for_heavy_model(
        self,
        mock_running_instances,
        mock_post,
        _mock_is_port_listening,
    ):
        mock_running_instances.return_value = [
            {
                "instance_id": "qwen-1",
                "model": "qwen3.5:27b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "hello"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "qwen-1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 321,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["content"], "hello")
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 600)
        self.assertEqual(mock_post.call_args.kwargs["json"]["options"]["num_predict"], 321)

    @patch("ollmo_webserver.is_port_listening", return_value=True)
    @patch("ollmo_webserver.requests.post")
    @patch("ollmo_webserver.load_running_instances")
    def test_mlx_chat_falls_back_to_reasoning_when_content_empty(
        self,
        mock_running_instances,
        mock_post,
        _mock_is_port_listening,
    ):
        mock_running_instances.return_value = [
            {
                "instance_id": "mlx-qwen-1",
                "model": "mlx-community/Qwen3.5-27B-4bit",
                "request_model": "/tmp/qwen3.5",
                "backend": "mlx",
                "capability": "chat",
                "port": 11502,
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "Hello from reasoning fallback.",
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "mlx-qwen-1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 654,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["content"], "Hello from reasoning fallback.")
        self.assertFalse(mock_post.call_args.kwargs["json"].get("enable_thinking"))
        self.assertEqual(mock_post.call_args.kwargs["json"]["max_tokens"], 654)
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 900)

    @patch("ollmo_webserver.is_port_listening", return_value=True)
    @patch("ollmo_webserver.requests.post")
    @patch("ollmo_webserver.load_running_instances")
    def test_llama_cpp_chat_uses_openai_compatible_transport(
        self,
        mock_running_instances,
        mock_post,
        _mock_is_port_listening,
    ):
        mock_running_instances.return_value = [
            {
                "instance_id": "gemma-3-1b-it-q4-llama_cpp-11551",
                "model": "gemma-3-1b-it-q4",
                "request_model": "gemma-3-1b-it-q4",
                "backend": "llama_cpp",
                "capability": "chat",
                "port": 11551,
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from llama.cpp.",
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "gemma-3-1b-it-q4-llama_cpp-11551",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["content"], "Hello from llama.cpp.")
        self.assertIn("/v1/chat/completions", mock_post.call_args.args[0])
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "gemma-3-1b-it-q4")

    @patch("ollmo_webserver.is_port_listening", return_value=True)
    @patch("ollmo_webserver.requests.post")
    @patch("ollmo_webserver.load_running_instances")
    def test_llama_cpp_chat_normalizes_multimodal_content_parts(
        self,
        mock_running_instances,
        mock_post,
        _mock_is_port_listening,
    ):
        mock_running_instances.return_value = [
            {
                "instance_id": "gemma-4-vision-llama_cpp-11552",
                "model": "ggml-org/gemma-4-E4B-it-GGUF",
                "request_model": "ggml-org/gemma-4-E4B-it-GGUF",
                "backend": "llama_cpp",
                "capability": "chat",
                "port": 11552,
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I see a photo.",
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "gemma-4-vision-llama_cpp-11552",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is in this image?"},
                            {"type": "input_image", "image_url": "data:image/png;base64,ZmFrZQ=="},
                        ],
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["content"], "I see a photo.")
        payload_messages = mock_post.call_args.kwargs["json"]["messages"]
        self.assertEqual(payload_messages[0]["content"][0], {"type": "text", "text": "What is in this image?"})
        self.assertEqual(
            payload_messages[0]["content"][1],
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
        )

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._execute_chat_backend_request")
    def test_local_provider_responses_adapter_returns_openai_shape(self, mock_execute, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "mlx-qwen-1",
            "model": "mlx-community/Qwen3.5-27B-4bit",
            "backend": "mlx",
            "capability": "chat",
            "port": 11502,
            "request_model": "/tmp/qwen3.5",
        }
        mock_execute.return_value = "Hello from adapter."

        response = self.client.post(
            "/api/local_provider/mlx-qwen-1/v1/responses",
            json={
                "input": "hello",
                "model": "mlx-community/Qwen3.5-27B-4bit",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output_text"], "Hello from adapter.")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "Hello from adapter.")

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._stream_chat_backend_as_responses")
    def test_local_provider_responses_adapter_streams_sse_events(self, mock_stream, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "gpt-oss:20b-1",
            "model": "gpt-oss:20b",
            "backend": "ollama",
            "capability": "chat",
            "port": 11435,
        }
        mock_stream.return_value = Response(
            "event: response.created\ndata: {}\n\n"
            "event: response.output_text.delta\ndata: {\"delta\":\"Hello streamed.\"}\n\n"
            "event: response.completed\ndata: {\"response\":{}}\n\n",
            mimetype="text/event-stream",
        )

        response = self.client.post(
            "/api/local_provider/gpt-oss:20b-1/v1/responses",
            json={
                "input": "hello",
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        body = response.get_data(as_text=True)
        self.assertIn("event: response.created", body)
        self.assertIn("event: response.output_text.delta", body)
        self.assertIn("Hello streamed.", body)
        self.assertIn("event: response.completed", body)

    @patch("ollmo_webserver._lookup_instance")
    def test_local_provider_responses_adapter_rejects_traversal_shaped_instance_id(self, mock_lookup):
        response = self.client.post(
            "/api/local_provider/..%2Fsecret/v1/responses",
            json={"input": "hello"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid path segments", response.get_json()["error"])
        mock_lookup.assert_not_called()

    @patch("ollmo_webserver._execute_chat_backend_request")
    def test_chat_route_rejects_traversal_shaped_instance_id(self, mock_execute):
        response = self.client.post(
            "/api/chat",
            json={
                "instance_id": "../secret",
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gemma4:26b",
                "backend": "ollama",
                "port": 11437,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid path segments", response.get_json()["error"])
        mock_execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
