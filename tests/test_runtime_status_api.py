import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_webserver import app


class RuntimeStatusApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_running_instances_route_includes_runtime_status(self, mock_load_running_instances, mock_merge_instances):
        mock_load_running_instances.return_value = [
            {
                "instance_id": "gpt-oss:20b-1",
                "model": "gpt-oss:20b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
            }
        ]
        mock_merge_instances.return_value = [
            {
                "instance_id": "gpt-oss:20b-1",
                "model": "gpt-oss:20b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
                "backend_metadata": {"source": "ollama_api_show", "capabilities": ["tools"]},
                "features": {"tool_calling": False},
                "inputs": ["text"],
                "outputs": ["text"],
                "readiness": "ready",
                "activity": "idle",
                "backend_runtime": {"source": "ollama_api_ps", "context_length": 32768},
                "runtime_status": {
                    "readiness": "ready",
                    "activity": "idle",
                    "backend_runtime": {"source": "ollama_api_ps", "context_length": 32768},
                },
            }
        ]

        response = self.client.get("/api/running_instances")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload[0]["readiness"], "ready")
        self.assertEqual(payload[0]["runtime_status"]["activity"], "idle")
        self.assertIn("features", payload[0])
        self.assertIn("inputs", payload[0])
        self.assertIn("outputs", payload[0])
        self.assertEqual(payload[0]["backend_metadata"]["source"], "ollama_api_show")
        self.assertEqual(payload[0]["backend_runtime"]["source"], "ollama_api_ps")
        self.assertEqual(response.headers["X-Ollmo-Truth-Mode"], "cached")
        self.assertEqual(response.headers["X-Ollmo-Refresh-Performed"], "false")
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], False)

    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_running_instances_route_refresh_is_explicit(self, mock_load_running_instances, mock_merge_instances):
        mock_load_running_instances.return_value = [{"instance_id": "gpt-oss:20b-1"}]
        mock_merge_instances.return_value = [{"instance_id": "gpt-oss:20b-1"}]

        response = self.client.get("/api/running_instances?refresh=true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Ollmo-Truth-Mode"], "refreshed")
        self.assertEqual(response.headers["X-Ollmo-Refresh-Performed"], "true")
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], True)

    @patch("ollmo_webserver.refresh_runtime_status_entries")
    @patch("ollmo_webserver.load_running_instances")
    def test_runtime_status_route_default_reads_cached_status_without_refresh_or_write(
        self,
        mock_load_running_instances,
        mock_refresh,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "runtime_status.json"
            status_payload = {
                "schema_version": 1,
                "updated_at": "2026-05-25T18:00:00Z",
                "instances": {
                    "qwen-1": {
                        "instance_id": "qwen-1",
                        "readiness": "ready",
                        "activity": "idle",
                    }
                },
            }
            status_path.write_text(json.dumps(status_payload, sort_keys=True), encoding="utf-8")
            before = status_path.read_text(encoding="utf-8")

            with patch("ollmo_webserver.RUNTIME_STATUS_PATH", status_path):
                response = self.client.get("/api/runtime_status")

            after = status_path.read_text(encoding="utf-8")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["instance_id"], "qwen-1")
        self.assertEqual(response.headers["X-Ollmo-Truth-Mode"], "cached")
        self.assertEqual(response.headers["X-Ollmo-Refresh-Performed"], "false")
        self.assertEqual(before, after)
        mock_load_running_instances.assert_not_called()
        mock_refresh.assert_not_called()

    @patch("ollmo_webserver.refresh_runtime_status_entries")
    @patch("ollmo_webserver.load_running_instances")
    def test_runtime_status_route_returns_single_instance_from_cached_status(self, mock_load_running_instances, mock_refresh):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "runtime_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "updated_at": "2026-05-25T18:00:00Z",
                        "instances": {
                            "qwen-1": {
                                "instance_id": "qwen-1",
                                "readiness": "ready",
                                "activity": "idle",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("ollmo_webserver.RUNTIME_STATUS_PATH", status_path):
                response = self.client.get("/api/runtime_status?instance_id=qwen-1")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance_id"], "qwen-1")
        self.assertEqual(payload["readiness"], "ready")
        self.assertEqual(response.headers["X-Ollmo-Truth-Mode"], "cached")
        self.assertEqual(response.headers["X-Ollmo-Refresh-Performed"], "false")
        mock_load_running_instances.assert_not_called()
        mock_refresh.assert_not_called()

    @patch("ollmo_webserver.refresh_runtime_status_entries")
    @patch("ollmo_webserver.load_running_instances")
    def test_runtime_status_route_refresh_is_explicit(self, mock_load_running_instances, mock_refresh):
        mock_load_running_instances.return_value = [{"instance_id": "qwen-1"}]
        mock_refresh.return_value = {
            "qwen-1": {
                "instance_id": "qwen-1",
                "readiness": "ready",
                "activity": "idle",
            }
        }

        response = self.client.get("/api/runtime_status?refresh=true&instance_id=qwen-1")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["instance_id"], "qwen-1")
        self.assertEqual(response.headers["X-Ollmo-Truth-Mode"], "refreshed")
        self.assertEqual(response.headers["X-Ollmo-Refresh-Performed"], "true")
        mock_load_running_instances.assert_called_once()
        mock_refresh.assert_called_once()

    @patch("ollmo_webserver.refresh_runtime_status_entries")
    @patch("ollmo_webserver.load_running_instances")
    def test_runtime_status_route_rejects_traversal_shaped_instance_id(self, mock_load_running_instances, mock_refresh):
        response = self.client.get("/api/runtime_status?instance_id=../secret")

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid path segments", response.get_json()["error"])
        mock_load_running_instances.assert_not_called()
        mock_refresh.assert_not_called()

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_runtime_manifest_exposes_canonical_and_direct_wrapper_routes(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_build_backend_fabric,
    ):
        mock_load_running_instances.return_value = [
            {
                "instance_id": "gpt-oss:20b-1",
                "model": "gpt-oss:20b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
            },
            {
                "instance_id": "x/flux2-klein:latest-1",
                "model": "x/flux2-klein:latest",
                "backend": "ollama",
                "capability": "image_generation",
                "port": 11436,
            },
        ]
        mock_merge_instances.return_value = [
            {
                "instance_id": "gpt-oss:20b-1",
                "model": "gpt-oss:20b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
                "backend_package": "ollama",
                "backend_contract": "ollama.api",
                "provider_capabilities": ["chat"],
                "backend_metadata": {"source": "ollama_api_show", "capabilities": ["tools"]},
                "session_controls": {
                    "enabled": True,
                    "hint": "Sampling controls for this chat model.",
                    "fields": {
                        "temperature": {"visible": True, "label": "Temperature"},
                    },
                },
                "runtime_status": {
                    "readiness": "ready",
                    "activity": "idle",
                    "backend_runtime": {"source": "ollama_api_ps", "context_length": 32768},
                },
            },
            {
                "instance_id": "x/flux2-klein:latest-1",
                "model": "x/flux2-klein:latest",
                "backend": "ollama",
                "capability": "image_generation",
                "port": 11436,
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            },
        ]
        mock_build_backend_fabric.return_value = {
            "summary": {"runtime_runnable_backend_count": 2},
            "backends": [{"backend_id": "ollama", "runtime_state": "runnable"}],
        }

        response = self.client.get("/api/runtime_manifest")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["service"]["name"], "ollmo")
        self.assertEqual(payload["service"]["canonical_responses"]["path"], "/api/responses")
        self.assertTrue(payload["service"]["canonical_responses"]["requires_instance_id"])
        self.assertEqual(payload["aliases"]["image"], "image_generation")
        self.assertEqual(payload["capabilities"]["chat"]["default_instance_id"], "gpt-oss:20b-1")
        self.assertEqual(payload["capabilities"]["image_generation"]["default_instance_id"], "x/flux2-klein:latest-1")
        self.assertEqual(payload["count"], 2)
        self.assertIn("backend_fabric", payload)
        self.assertEqual(payload["instances"][0]["backend_package"], "ollama")
        self.assertEqual(payload["instances"][0]["backend_contract"], "ollama.api")
        self.assertEqual(payload["instances"][0]["provider_capabilities"], ["chat"])
        self.assertIn("features", payload["instances"][0])
        self.assertIn("feature_sources", payload["instances"][0])
        self.assertIn("inputs", payload["instances"][0])
        self.assertIn("outputs", payload["instances"][0])
        self.assertIn("backend_metadata", payload["instances"][0])
        self.assertIn("backend_runtime", payload["instances"][0])
        self.assertIn("routing_summary", payload["instances"][0])
        self.assertIn("session_controls_summary", payload["instances"][0])
        self.assertEqual(payload["instances"][0]["routing_summary"]["backend_package"], "ollama")
        self.assertEqual(payload["capabilities"]["chat"]["candidates"][0]["backend_package"], "ollama")
        self.assertIn("routing_summary", payload["capabilities"]["chat"]["candidates"][0])
        direct_paths = {item["instance_id"]: item["direct_responses"]["path"] for item in payload["instances"]}
        self.assertEqual(
            direct_paths["x/flux2-klein:latest-1"],
            "/api/local_provider/x%2Fflux2-klein%3Alatest-1/v1/responses",
        )
        self.assertEqual(
            direct_paths["gpt-oss:20b-1"],
            "/api/local_provider/gpt-oss%3A20b-1/v1/responses",
        )
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "cached")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], False)
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], False)

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_runtime_manifest_refresh_is_explicit(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_build_backend_fabric,
    ):
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_build_backend_fabric.return_value = {"summary": {}, "backends": []}

        response = self.client.get("/api/runtime_manifest?refresh=true")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "refreshed")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], True)
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], True)

    @patch("ollmo_webserver.build_backend_fabric_snapshot")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_routing_table_alias_route_returns_same_manifest_shape(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_build_backend_fabric,
    ):
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_build_backend_fabric.return_value = {"summary": {}, "backends": []}

        response = self.client.get("/api/routing_table")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["instances"], [])
        self.assertEqual(payload["aliases"]["tts"], "text_to_speech")
        self.assertEqual(payload["capabilities"]["chat"]["count"], 0)

    @patch("ollmo_webserver.read_events")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    @patch("ollmo_g.payload._read_recent_log_lines")
    def test_ghost_route_returns_runtime_intelligence_payload(
        self,
        mock_read_log_lines,
        mock_load_running_instances,
        mock_merge_instances,
        mock_read_events,
    ):
        mock_read_log_lines.return_value = [
            "2026-04-03 16:55:45,334 - INFO - Ghost router execution fallback: HTTPConnectionPool(host='localhost', port=11437): Read timed out. (read timeout=35)"
        ]
        mock_load_running_instances.return_value = [
            {
                "instance_id": "gpt-oss:20b-1",
                "model": "gpt-oss:20b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
            }
        ]
        mock_merge_instances.return_value = [
            {
                "instance_id": "gpt-oss:20b-1",
                "model": "gpt-oss:20b",
                "backend": "ollama",
                "capability": "chat",
                "port": 11435,
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            }
        ]
        mock_read_events.return_value = [
            {
                "timestamp": "2026-03-21T09:30:00Z",
                "category": "runtime",
                "action": "request",
                "status": "ok",
                "instance_id": "gpt-oss:20b-1",
                "message": "Started",
            }
        ]

        response = self.client.get("/api/ghost")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["identity"]["name"], "ollmo-ghost")
        self.assertEqual(payload["identity"]["role"], "self-describing local runtime intelligence")
        self.assertEqual(payload["capabilities"]["chat"]["default_instance_id"], "gpt-oss:20b-1")
        self.assertEqual(payload["recommendations"][0]["instance_id"], "gpt-oss:20b-1")
        self.assertIn("self_healing_hints", payload)
        self.assertTrue(any("router" in str(item.get("reason") or "").lower() for item in payload["self_healing_hints"]))
        self.assertIn("Ollmo Ghost", payload["markdown"])
        self.assertEqual(payload["paths"]["guide"], "GHOST.md")
        self.assertEqual(payload["paths"]["response_frame_ledger"], "state/response_frames/responses.jsonl")
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "cached")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], False)
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], False)

    @patch("ollmo_webserver.read_events")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    @patch("ollmo_g.payload._read_recent_log_lines")
    def test_ghost_route_refresh_is_explicit(
        self,
        mock_read_log_lines,
        mock_load_running_instances,
        mock_merge_instances,
        mock_read_events,
    ):
        mock_read_log_lines.return_value = []
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_read_events.return_value = []

        response = self.client.get("/api/ghost?refresh=true")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "refreshed")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], True)
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], True)

    @patch("ollmo_webserver.read_events")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_ghost_route_can_return_markdown(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_read_events,
    ):
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = []
        mock_read_events.return_value = []

        response = self.client.get("/api/ghost?format=md")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/markdown", response.content_type)
        body = response.get_data(as_text=True)
        self.assertIn("# Ollmo Ghost", body)
        self.assertIn("No running instances", body)
        self.assertEqual(mock_merge_instances.call_args.kwargs["refresh"], False)

    @patch("ollmo_webserver._build_ghost_route_preview_payload")
    @patch("ollmo_webserver._prepare_effective_request_data")
    @patch("ollmo_webserver._resolve_ghost_auto_route")
    def test_ghost_route_preview_default_is_cached_observer(
        self,
        mock_resolve_ghost_auto_route,
        mock_prepare_effective_request_data,
        mock_build_preview_payload,
    ):
        mock_resolve_ghost_auto_route.return_value = ({"instance": {"instance_id": "qwen-1"}}, None)
        mock_prepare_effective_request_data.return_value = (
            {"prompt": "hello"},
            {"instance": {"instance_id": "qwen-1"}},
            None,
            None,
        )
        mock_build_preview_payload.return_value = {"route": {"instance_id": "qwen-1"}}

        response = self.client.post("/api/ghost_route_preview", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "cached")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], False)
        self.assertEqual(mock_resolve_ghost_auto_route.call_args.kwargs["refresh_runtime_status"], False)

    @patch("ollmo_webserver._build_ghost_route_preview_payload")
    @patch("ollmo_webserver._prepare_effective_request_data")
    @patch("ollmo_webserver._resolve_ghost_auto_route")
    def test_ghost_route_preview_refresh_is_explicit(
        self,
        mock_resolve_ghost_auto_route,
        mock_prepare_effective_request_data,
        mock_build_preview_payload,
    ):
        mock_resolve_ghost_auto_route.return_value = ({"instance": {"instance_id": "qwen-1"}}, None)
        mock_prepare_effective_request_data.return_value = (
            {"prompt": "hello", "refresh": True},
            {"instance": {"instance_id": "qwen-1"}},
            None,
            None,
        )
        mock_build_preview_payload.return_value = {"route": {"instance_id": "qwen-1"}}

        response = self.client.post("/api/ghost_route_preview", json={"prompt": "hello", "refresh": True})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runtime_truth"]["truth_mode"], "refreshed")
        self.assertEqual(payload["runtime_truth"]["refresh_performed"], True)
        self.assertEqual(mock_resolve_ghost_auto_route.call_args.kwargs["refresh_runtime_status"], True)


if __name__ == "__main__":
    unittest.main()
