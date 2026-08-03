import io
import tempfile
import unittest
from pathlib import Path

import ollmo_webserver
from flask import has_app_context
from requests.exceptions import ConnectionError, RequestException, Timeout
from unittest.mock import Mock
from unittest.mock import patch

from ollmo_webserver import (
    _GENERATED_IMAGE_STATE_CACHE,
    _GENERATED_IMAGE_STATE_ENRICHMENT_IN_FLIGHT,
    _build_image_state_for_generated_image,
    _find_cached_pdf_insight,
    _invoke_internal_api_json_route,
    _pick_image_state_helper_instance,
    _schedule_generated_image_payload_enrichment,
    app,
)
from ollmo_services.artifact_registry import (
    find_artifact_registry_record,
    find_generated_image_provenance,
    persist_generated_image_provenance,
)
from ollmo_core.transports import persist_input_file_locally


class InferApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        _GENERATED_IMAGE_STATE_CACHE.clear()
        _GENERATED_IMAGE_STATE_ENRICHMENT_IN_FLIGHT.clear()
        if hasattr(ollmo_webserver._GENERATED_IMAGE_POSTPROCESS, "helper_error_cooldowns"):
            ollmo_webserver._GENERATED_IMAGE_POSTPROCESS.helper_error_cooldowns.clear()
        self._artifact_inputs_tmpdir = tempfile.TemporaryDirectory()
        self._artifact_inputs_root = Path(self._artifact_inputs_tmpdir.name) / "artifacts" / "inputs"
        self._persist_input_file_locally_patcher = patch(
            "ollmo_webserver._persist_input_file_locally",
            side_effect=self._persist_input_file_locally_to_temp,
        )
        self._persist_input_file_locally_patcher.start()

    def tearDown(self):
        self._persist_input_file_locally_patcher.stop()
        self._artifact_inputs_tmpdir.cleanup()

    def _persist_input_file_locally_to_temp(
        self,
        source_path: Path,
        *,
        source_name: str,
        file_kind: str,
        output_root: Path | None = None,
    ) -> str | None:
        return persist_input_file_locally(
            source_path,
            source_name=source_name,
            file_kind=file_kind,
            output_root=self._artifact_inputs_root,
        )

    @patch("ollmo_webserver._lookup_instance")
    def test_infer_route_rejects_traversal_shaped_instance_id(self, mock_lookup):
        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "../secret",
                "prompt": "hello",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid path segments", response.get_json()["error"])
        mock_lookup.assert_not_called()

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch.object(ollmo_webserver._GENERATED_IMAGE_POSTPROCESS, "schedule_generated_image_payload_enrichment")
    @patch("ollmo_webserver._persist_image_data_url_locally")
    @patch("ollmo_webserver._ollama_generate")
    def test_image_generation_infer(self, mock_generate, mock_persist, mock_schedule_enrichment, mock_lookup, mock_activity, mock_success):
        mock_lookup.return_value = {
            "instance_id": "flux-1",
            "port": 11435,
            "model": "x/flux2-klein:latest",
            "backend": "ollama",
            "capability": "image_generation",
        }
        mock_generate.return_value = {"response": "Image done", "images": ["aGVsbG8="], "seed": 123}
        mock_persist.return_value = "/tmp/artifacts/images/flux.png"
        mock_schedule_enrichment.side_effect = lambda payload: dict(payload)
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        with tempfile.TemporaryDirectory() as tmpdir:
            provenance_path = Path(tmpdir) / "artifact_registry.jsonl"
            with (
                patch("ollmo_webserver.ARTIFACT_REGISTRY_LEDGER", provenance_path),
                patch("ollmo_webserver._ollama_openai_image_generation", return_value=None),
            ):
                response = self.client.post(
                    "/api/infer",
                    json={
                        "instance_id": "flux-1",
                        "prompt": "a cat in watercolor",
                        "width": 1024,
                        "height": 768,
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["mode"], "image_generation")
            self.assertTrue(payload["image_data_url"].startswith("data:image/png;base64,"))
            self.assertEqual(payload["saved_image_path"], "/tmp/artifacts/images/flux.png")
            self.assertEqual(payload["seed"], 123)
            self.assertNotIn("image_state", payload)
            provenance = find_generated_image_provenance(
                "/tmp/artifacts/images/flux.png",
                ledger_path=provenance_path,
            )
            self.assertIsNotNone(provenance)
            self.assertEqual(provenance["source"]["instance_id"], "flux-1")
            self.assertEqual(provenance["source"]["backend"], "ollama")
            self.assertEqual(provenance["request"]["prompt_text"], "a cat in watercolor")
            self.assertEqual(provenance["request"]["width"], 1024)
            self.assertEqual(provenance["request"]["height"], 768)
            self.assertEqual(provenance["output"]["saved_image_path"], "/tmp/artifacts/images/flux.png")
            self.assertEqual(provenance["output"]["seed"], 123)
            mock_generate.assert_called_once()
            _, kwargs = mock_generate.call_args
            self.assertEqual(kwargs["options"], {"width": 1024, "height": 768})
            mock_persist.assert_called_once()
            mock_schedule_enrichment.assert_called_once()
            scheduled_payload = mock_schedule_enrichment.call_args.args[0]
            self.assertEqual(scheduled_payload["saved_image_path"], "/tmp/artifacts/images/flux.png")
            mock_activity.assert_called_once()
            mock_success.assert_called_once()

    @patch("ollmo_webserver._execute_infer_request")
    def test_invoke_internal_api_json_route_establishes_app_context(self, mock_execute_infer):
        def fake_execute(payload, upload=None):
            self.assertTrue(has_app_context())
            return {"ok": True}, 200

        mock_execute_infer.side_effect = fake_execute

        payload, status_code = _invoke_internal_api_json_route(payload={"instance_id": "flux-1"})

        self.assertEqual(status_code, 200)
        self.assertTrue(payload["ok"])

    @patch("ollmo_server.infer_postprocess.threading.Thread")
    def test_schedule_generated_image_payload_enrichment_only_starts_one_worker_per_path(self, mock_thread):
        worker = Mock()
        mock_thread.return_value = worker
        target_dir = Path.cwd() / "artifacts" / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target_dir, suffix=".png", delete=False) as tmp:
            image_path = tmp.name
        payload = {
            "mode": "image_generation",
            "saved_image_path": image_path,
        }

        try:
            first = _schedule_generated_image_payload_enrichment(payload)
            second = _schedule_generated_image_payload_enrichment(payload)

            self.assertEqual(first["saved_image_path"], image_path)
            self.assertEqual(second["saved_image_path"], image_path)
            mock_thread.assert_called_once()
            worker.start.assert_called_once()
        finally:
            Path(image_path).unlink(missing_ok=True)

    def test_pick_image_state_helper_instance_prefers_multimodal_chat_over_ocr(self):
        selected = _pick_image_state_helper_instance(
            [
                {
                    "instance_id": "ocr-1",
                    "model": "mlx-community/GLM-OCR-bf16",
                    "backend": "mlx",
                    "backend_package": "mlx_vlm",
                    "capability": "vision_analysis",
                    "port": 11501,
                    "inputs": ["text", "image"],
                    "provider_capabilities": ["vision_analysis", "chat"],
                    "features": {"vision_input": True},
                    "runtime_status": {"readiness": "ready", "activity": "idle"},
                    "session_controls_summary": {"required_fields": ["ocr_mode"]},
                },
                {
                    "instance_id": "vision-chat-1",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "backend_package": "ollama",
                    "capability": "chat",
                    "port": 11437,
                    "inputs": ["text", "image"],
                    "provider_capabilities": ["chat", "vision_analysis"],
                    "features": {"vision_input": True, "structured_outputs": True},
                    "runtime_status": {"readiness": "ready", "activity": "idle"},
                },
            ]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["instance_id"], "vision-chat-1")

    def test_pick_image_state_helper_instance_prefers_multimodal_chat_over_mlx_vlm(self):
        selected = _pick_image_state_helper_instance(
            [
                {
                    "instance_id": "mlx-vlm-1",
                    "model": "mlx-community/gemma-4-e4b-it-8bit",
                    "backend": "mlx",
                    "backend_package": "mlx_vlm",
                    "capability": "vision_analysis",
                    "port": 11501,
                    "inputs": ["text", "image"],
                    "provider_capabilities": ["vision_analysis", "chat"],
                    "features": {"vision_input": True},
                    "runtime_status": {"readiness": "ready", "activity": "idle"},
                },
                {
                    "instance_id": "vision-chat-1",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "backend_package": "ollama",
                    "capability": "chat",
                    "port": 11437,
                    "inputs": ["text", "image"],
                    "provider_capabilities": ["chat", "vision_analysis"],
                    "features": {"vision_input": True, "structured_outputs": True},
                    "runtime_status": {"readiness": "ready", "activity": "idle"},
                },
            ]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["instance_id"], "vision-chat-1")

    def test_pick_image_state_helper_instance_skips_unhealthy_helpers(self):
        selected = _pick_image_state_helper_instance(
            [
                {
                    "instance_id": "dead-vision-chat-1",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "backend_package": "ollama",
                    "capability": "chat",
                    "port": 11437,
                    "inputs": ["text", "image"],
                    "provider_capabilities": ["chat", "vision_analysis"],
                    "features": {"vision_input": True, "structured_outputs": True},
                    "runtime_status": {
                        "readiness": "ready",
                        "activity": "idle",
                        "port_listening": False,
                    },
                },
                {
                    "instance_id": "vision-chat-2",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "backend_package": "ollama",
                    "capability": "chat",
                    "port": 11438,
                    "inputs": ["text", "image"],
                    "provider_capabilities": ["chat", "vision_analysis"],
                    "features": {"vision_input": True, "structured_outputs": True},
                    "runtime_status": {"readiness": "ready", "activity": "idle"},
                },
            ]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["instance_id"], "vision-chat-2")

    @patch("ollmo_webserver._invoke_internal_api_json_route")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_build_image_state_for_generated_image_schedules_generated_image_state_enrichment_helper_terminal(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_invoke,
    ):
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = [
            {
                "instance_id": "ocr-1",
                "model": "mlx-community/GLM-OCR-bf16",
                "backend": "mlx",
                "backend_package": "mlx_vlm",
                "capability": "vision_analysis",
                "port": 11501,
                "inputs": ["text", "image"],
                "provider_capabilities": ["vision_analysis", "chat"],
                "features": {"vision_input": True},
                "runtime_status": {"readiness": "ready", "activity": "idle"},
                "session_controls_summary": {"required_fields": ["ocr_mode"]},
            },
            {
                "instance_id": "vision-chat-1",
                "model": "gemma4:26b",
                "backend": "ollama",
                "backend_package": "ollama",
                "capability": "chat",
                "port": 11437,
                "inputs": ["text", "image"],
                "provider_capabilities": ["chat", "vision_analysis"],
                "features": {"vision_input": True, "structured_outputs": True},
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            },
        ]
        mock_invoke.return_value = (
            {
                "content": (
                    '{"summary":"Two armored knights crossing swords on a battlefield.",'
                    '"subject":"two armored knights","scene":"battlefield",'
                    '"style":"cinematic fantasy","key_elements":["knights","swords"]}'
                )
            },
            200,
        )

        with patch.object(
            ollmo_webserver._GENERATED_IMAGE_POSTPROCESS,
            "schedule_post_response_substrate_hygiene",
        ) as mock_schedule_hygiene:
            payload = _build_image_state_for_generated_image("/tmp/generated.png")

        self.assertEqual(payload["subject"], "two armored knights")
        called = mock_invoke.call_args.kwargs["payload"]
        self.assertEqual(called["instance_id"], "vision-chat-1")
        self.assertEqual(called["capability"], "chat")
        self.assertEqual(called["reference_artifacts"][0]["path"], "/tmp/generated.png")
        self.assertEqual(called["infer_timeout_sec"], 8)
        self.assertTrue(called["internal_fast_timeout"])
        mock_schedule_hygiene.assert_called_once()
        hygiene_payload = mock_schedule_hygiene.call_args.args[0]
        self.assertEqual(hygiene_payload["instance_id"], "vision-chat-1")
        self.assertEqual(hygiene_payload["mode"], "generated_image_state_enrichment")
        self.assertEqual(
            mock_schedule_hygiene.call_args.kwargs["reason"],
            "generated_image_state_enrichment_helper_terminal",
        )

    @patch("ollmo_webserver._invoke_internal_api_json_route")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_build_image_state_for_generated_image_tries_next_helper_after_exception(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_invoke,
    ):
        mock_load_running_instances.return_value = []
        mock_merge_instances.return_value = [
            {
                "instance_id": "vision-chat-z",
                "model": "gemma4:26b",
                "backend": "ollama",
                "backend_package": "ollama",
                "capability": "chat",
                "port": 11437,
                "inputs": ["text", "image"],
                "provider_capabilities": ["chat", "vision_analysis"],
                "features": {"vision_input": True, "structured_outputs": True},
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            },
            {
                "instance_id": "vision-chat-a",
                "model": "gemma4:26b",
                "backend": "ollama",
                "backend_package": "ollama",
                "capability": "chat",
                "port": 11438,
                "inputs": ["text", "image"],
                "provider_capabilities": ["chat", "vision_analysis"],
                "features": {"vision_input": True, "structured_outputs": True},
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            },
        ]
        mock_invoke.side_effect = [
            Timeout("timed out"),
            (
                {
                    "content": (
                        '{"summary":"A silver temple above misty water.",'
                        '"subject":"temple","scene":"misty water",'
                        '"style":"cinematic","key_elements":["temple","mist"]}'
                    )
                },
                200,
            ),
        ]

        with patch.object(
            ollmo_webserver._GENERATED_IMAGE_POSTPROCESS,
            "schedule_post_response_substrate_hygiene",
        ) as mock_schedule_hygiene:
            payload = _build_image_state_for_generated_image("/tmp/generated.png")

        self.assertEqual(payload["subject"], "temple")
        self.assertEqual(mock_invoke.call_count, 2)
        first_call = mock_invoke.call_args_list[0].kwargs["payload"]
        second_call = mock_invoke.call_args_list[1].kwargs["payload"]
        self.assertEqual(first_call["instance_id"], "vision-chat-z")
        self.assertEqual(second_call["instance_id"], "vision-chat-a")
        self.assertEqual(mock_schedule_hygiene.call_count, 2)
        self.assertEqual(
            [
                call.args[0]["instance_id"]
                for call in mock_schedule_hygiene.call_args_list
            ],
            ["vision-chat-z", "vision-chat-a"],
        )
        self.assertEqual(
            {
                call.kwargs["reason"]
                for call in mock_schedule_hygiene.call_args_list
            },
            {"generated_image_state_enrichment_helper_terminal"},
        )

    @patch("ollmo_webserver._invoke_internal_api_json_route")
    @patch("ollmo_webserver.merge_instances_with_runtime_status")
    @patch("ollmo_webserver.load_running_instances")
    def test_build_image_state_for_generated_image_uses_backend_agnostic_helper_cooldown(
        self,
        mock_load_running_instances,
        mock_merge_instances,
        mock_invoke,
    ):
        mock_load_running_instances.return_value = []
        helper_candidates = [
            {
                "instance_id": "vision-chat-z",
                "model": "gemma4:26b",
                "backend": "ollama",
                "backend_package": "ollama",
                "capability": "chat",
                "port": 11437,
                "inputs": ["text", "image"],
                "provider_capabilities": ["chat", "vision_analysis"],
                "features": {"vision_input": True, "structured_outputs": True},
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            },
            {
                "instance_id": "vision-chat-a",
                "model": "gemma4:26b",
                "backend": "ollama",
                "backend_package": "ollama",
                "capability": "chat",
                "port": 11438,
                "inputs": ["text", "image"],
                "provider_capabilities": ["chat", "vision_analysis"],
                "features": {"vision_input": True, "structured_outputs": True},
                "runtime_status": {"readiness": "ready", "activity": "idle"},
            },
        ]
        mock_merge_instances.return_value = helper_candidates
        success_payload = (
            {
                "content": (
                    '{"summary":"A silver temple above misty water.",'
                    '"subject":"temple","scene":"misty water",'
                    '"style":"cinematic","key_elements":["temple","mist"]}'
                )
            },
            200,
        )
        mock_invoke.side_effect = [Timeout("timed out"), success_payload]

        first_payload = _build_image_state_for_generated_image("/tmp/generated.png")

        self.assertEqual(first_payload["subject"], "temple")
        cooldowns = ollmo_webserver._GENERATED_IMAGE_POSTPROCESS.helper_error_cooldowns
        self.assertIn("vision-chat-z", cooldowns)

        mock_invoke.reset_mock()
        mock_invoke.side_effect = None
        mock_invoke.return_value = success_payload
        second_payload = _build_image_state_for_generated_image("/tmp/generated.png")

        self.assertEqual(second_payload["subject"], "temple")
        self.assertEqual(mock_invoke.call_count, 1)
        called = mock_invoke.call_args.kwargs["payload"]
        self.assertEqual(called["instance_id"], "vision-chat-a")

    def test_blocking_generated_image_enrichment_persists_image_state_onto_original_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'generated.png'
            image_path.write_bytes(b'png')
            ledger_path = Path(tmpdir) / 'artifact_registry.jsonl'
            provenance = {
                'kind': 'ollmo.generated_image_provenance',
                'provenance_id': 'generated_image_deadbeef',
                'created_at': '2026-04-14T00:00:00Z',
                'image_path': str(image_path),
                'source': {
                    'response_id': 'resp_1',
                    'model': 'flux',
                    'instance_id': 'flux-1',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'mode': 'image_generation',
                },
                'request': {
                    'prompt_text': 'dream temple at dusk',
                    'prompt_preview': 'dream temple at dusk',
                },
                'output': {
                    'saved_image_path': str(image_path),
                    'artifact_id': 'image_deadbeef',
                    'artifact_ref': 'artifact:image_deadbeef',
                },
            }
            persist_generated_image_provenance(
                provenance,
                ledger_path=ledger_path,
            )

            with (
                patch('ollmo_webserver.ARTIFACT_REGISTRY_LEDGER', ledger_path),
                patch.object(
                    ollmo_webserver._GENERATED_IMAGE_POSTPROCESS,
                    'build_image_state_for_generated_image',
                    return_value={
                        'summary': 'dream temple above silver water',
                        'subject': 'temple',
                    },
                ),
            ):
                payload = ollmo_webserver._enrich_generated_image_payload(
                    {
                        'mode': 'image_generation',
                        'saved_image_path': str(image_path),
                    },
                    blocking=True,
                )

            self.assertEqual(payload['image_state']['subject'], 'temple')
            self.assertEqual(payload['image_state_enrichment']['status'], 'completed')
            registry_record = find_artifact_registry_record(
                str(image_path),
                ledger_path=ledger_path,
            )
            self.assertIsNotNone(registry_record)
            self.assertEqual(
                registry_record['enrichments']['image_state']['subject'],
                'temple',
            )
            self.assertEqual(
                registry_record['enrichments']['image_state_enrichment']['mode'],
                'background_analysis',
            )

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._persist_image_data_url_locally")
    @patch("ollmo_webserver._ollama_openai_image_generation")
    @patch("ollmo_webserver._ollama_generate")
    def test_image_generation_uses_openai_fallback_when_no_inline_image(
        self,
        mock_generate,
        mock_openai_fallback,
        mock_persist,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "flux-1",
            "port": 11435,
            "model": "x/flux2-klein:latest",
            "backend": "ollama",
            "capability": "image_generation",
        }
        mock_generate.return_value = {"response": "", "done": True}
        mock_openai_fallback.return_value = "data:image/png;base64,ZmFrZQ=="
        mock_persist.return_value = "/tmp/artifacts/images/fallback.png"

        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "flux-1",
                "prompt": "a puppy",
                "width": 1280,
                "height": 720,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "image_generation")
        self.assertEqual(payload["image_data_url"], "data:image/png;base64,ZmFrZQ==")
        self.assertEqual(payload["content"], "Image generated.")
        self.assertEqual(payload["saved_image_path"], "/tmp/artifacts/images/fallback.png")
        mock_openai_fallback.assert_called_once_with(
            11435,
            "x/flux2-klein:latest",
            "a puppy",
            width=1280,
            height=720,
        )
        mock_persist.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    def test_image_generation_rejects_single_dimension_without_the_other(self, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "flux-1",
            "port": 11435,
            "model": "x/flux2-klein:latest",
            "backend": "ollama",
            "capability": "image_generation",
        }

        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "flux-1",
                "prompt": "a puppy",
                "width": 1280,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Width and Height together", response.get_json()["error"])

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._persist_transcript_text_locally")
    @patch("ollmo_webserver._whisper_transcribe")
    def test_speech_to_text_with_audio_upload(self, mock_transcribe, mock_persist_transcript, mock_lookup, mock_activity, mock_success):
        mock_lookup.return_value = {
            "instance_id": "whisper-1",
            "port": 11501,
            "model": "mlx-community/whisper-large-v3-mlx",
            "backend": "mlx",
            "capability": "speech_to_text",
        }
        mock_transcribe.return_value = {"text": "hello world"}
        mock_persist_transcript.return_value = "/tmp/artifacts/transcripts/hello.md"
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "whisper-1",
                "file": (io.BytesIO(b"RIFFfakewav"), "sample.wav"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "speech_to_text")
        self.assertEqual(payload["content"], "hello world")
        self.assertEqual(payload["saved_text_path"], "/tmp/artifacts/transcripts/hello.md")
        mock_transcribe.assert_called_once()
        mock_persist_transcript.assert_called_once()
        mock_activity.assert_called_once()
        mock_success.assert_called_once()

    @patch("ollmo_webserver.record_instance_failure")
    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._whisper_transcribe")
    def test_speech_to_text_missing_audio_is_contract_error_not_backend_failure(
        self,
        mock_transcribe,
        mock_lookup,
        mock_activity,
        mock_success,
        mock_failure,
    ):
        mock_lookup.return_value = {
            "instance_id": "whisper-1",
            "port": 11501,
            "model": "mlx-community/whisper-large-v3-mlx",
            "backend": "mlx",
            "capability": "speech_to_text",
        }
        mock_activity.side_effect = [
            ({"readiness": "ready", "activity": "idle"}, {"readiness": "ready", "activity": "busy"}),
            ({"readiness": "ready", "activity": "busy"}, {"readiness": "ready", "activity": "idle"}),
        ]

        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "whisper-1",
                "prompt": "Analysiere das zuletzt erzeugte Audio.",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("speech_to_text requires an audio file", response.get_json()["error"])
        mock_transcribe.assert_not_called()
        mock_success.assert_not_called()
        mock_failure.assert_not_called()
        self.assertEqual(mock_activity.call_count, 2)
        self.assertEqual(mock_activity.call_args_list[0].kwargs["activity"], "busy")
        self.assertEqual(mock_activity.call_args_list[1].kwargs["activity"], "idle")

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._persist_audio_bytes_locally")
    @patch("ollmo_webserver._mlx_audio_speech")
    def test_text_to_speech_infer_returns_saved_audio_artifact(
        self,
        mock_tts,
        mock_persist_audio,
        mock_lookup,
        mock_activity,
        mock_success,
    ):
        mock_lookup.return_value = {
            "instance_id": "tts-1",
            "port": 11504,
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "request_model": "/tmp/qwen3-tts",
            "backend": "mlx",
            "capability": "text_to_speech",
        }
        mock_tts.return_value = {
            "audio_bytes": b"RIFFfakewav",
            "content_type": "audio/wav",
            "result": {"bytes": 11},
        }
        mock_persist_audio.return_value = "/tmp/artifacts/audio/qwen3-tts.wav"
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "tts-1",
                "prompt": "Guten Tag aus Ollmo.",
                "voice": "Chelsie",
                "instruct": "Warm, calm, elegant German narration.",
                "speed": "0.95",
                "pitch": "1.10",
                "lang_code": "de",
                "response_format": "wav",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "text_to_speech")
        self.assertEqual(payload["saved_audio_path"], "/tmp/artifacts/audio/qwen3-tts.wav")
        self.assertEqual(payload["tts_generation_budget"]["max_tokens"], 256)
        self.assertEqual(
            payload["tts_sampling_profile"]["policy_id"],
            "qwen3_tts_model_native_sampling_v1",
        )
        mock_tts.assert_called_once()
        args, kwargs = mock_tts.call_args
        self.assertEqual(args[1], "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        self.assertEqual(kwargs["voice"], "Chelsie")
        self.assertEqual(kwargs["instruct"], "Warm, calm, elegant German narration.")
        self.assertEqual(kwargs["speed"], 0.95)
        self.assertEqual(kwargs["pitch"], 1.1)
        self.assertEqual(kwargs["lang_code"], "german")
        self.assertEqual(kwargs["max_tokens"], 256)
        self.assertEqual(kwargs["temperature"], 0.9)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(kwargs["top_k"], 50)
        self.assertEqual(kwargs["repetition_penalty"], 1.05)
        mock_persist_audio.assert_called_once()
        mock_activity.assert_called_once()
        mock_success.assert_called_once()

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._persist_audio_bytes_locally")
    @patch("ollmo_webserver._mlx_audio_speech")
    def test_text_to_speech_infer_derives_language_from_text_when_auto(
        self,
        mock_tts,
        mock_persist_audio,
        mock_lookup,
        mock_activity,
        mock_success,
    ):
        mock_lookup.return_value = {
            "instance_id": "tts-language-1",
            "port": 11504,
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            "request_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            "backend": "mlx",
            "capability": "text_to_speech",
            "tts_model_type": "voice_design",
            "tts_languages": ["auto", "english", "german", "french"],
        }
        mock_tts.return_value = {
            "audio_bytes": b"RIFFfakewav",
            "content_type": "audio/wav",
            "result": {"bytes": 11},
        }
        mock_persist_audio.return_value = "/tmp/artifacts/audio/german.wav"
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "tts-language-1",
                "prompt": (
                    "Es war einmal ein kleiner Fuchs, der in einem alten Wald lebte. "
                    "Plötzlich entdeckte er ein silbernes Licht."
                ),
                "instruct": "Warm, calm German narration.",
                "lang_code": "auto",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["lang_code"], "german")
        self.assertEqual(payload["lang_code_source"], "inferred_from_text")
        mock_tts.assert_called_once()
        _args, kwargs = mock_tts.call_args
        self.assertEqual(kwargs["lang_code"], "german")

    @patch("ollmo_webserver._lookup_instance")
    def test_text_to_speech_customvoice_rejects_unsupported_speaker(self, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "tts-2",
            "port": 11502,
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            "request_model": "/tmp/qwen3-tts-customvoice",
            "backend": "mlx",
            "capability": "text_to_speech",
        }

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "tts-2",
                "prompt": "Hallo Welt.",
                "voice": "Karl",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("Speaker 'Karl'", payload["error"])
        self.assertIn("VoiceDesign", payload["error"])

    @patch("ollmo_webserver._lookup_instance")
    def test_text_to_speech_customvoice_requires_speaker(self, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "tts-3",
            "port": 11502,
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            "request_model": "/tmp/qwen3-tts-customvoice",
            "backend": "mlx",
            "capability": "text_to_speech",
        }

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "tts-3",
                "prompt": "Hallo Welt.",
                "voice": "",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("requires a speaker", payload["error"])
        self.assertIn("VoiceDesign", payload["error"])

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._mlx_chat_completions")
    def test_mlx_vlm_vision_analysis_with_image_upload(self, mock_mlx_chat, mock_lookup, mock_activity, mock_success):
        mock_lookup.return_value = {
            "instance_id": "vlm-1",
            "port": 11520,
            "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "request_model": "/tmp/qwen-vl",
            "backend": "mlx",
            "capability": "vision_analysis",
        }
        mock_mlx_chat.return_value = {"content": "A dark biomechanical structure."}
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "vlm-1",
                "prompt": "Describe this image",
                "file": (io.BytesIO(b"fakepng"), "sample.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis")
        self.assertEqual(payload["content"], "A dark biomechanical structure.")
        self.assertEqual(len(payload["input_artifacts"]), 1)
        persisted_input = Path(payload["input_artifacts"][0]["path"]).resolve()
        self.assertTrue(persisted_input.is_relative_to(self._artifact_inputs_root.resolve()))
        self.assertTrue(persisted_input.exists())
        mock_mlx_chat.assert_called_once()
        called_args = mock_mlx_chat.call_args.args
        self.assertEqual(called_args[0], 11520)
        self.assertEqual(called_args[1], "/tmp/qwen-vl")
        self.assertTrue(isinstance(called_args[2], list))
        mock_activity.assert_called_once()
        mock_success.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    def test_speech_to_text_rejects_non_audio_upload(self, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "whisper-1",
            "port": 11501,
            "model": "mlx-community/whisper-large-v3-mlx",
            "backend": "mlx",
            "capability": "speech_to_text",
        }

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "whisper-1",
                "file": (io.BytesIO(b"not-audio"), "notes.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("Expected an audio file", payload["error"])

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_chat")
    def test_chat_infer_plain_prompt(self, mock_chat, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "qwen-1",
            "port": 11436,
            "model": "qwen3.5:27b",
            "backend": "ollama",
            "capability": "chat",
        }
        mock_chat.return_value = {"content": "Hi there"}

        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "qwen-1",
                "prompt": "say hi",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "chat")
        self.assertEqual(payload["content"], "Hi there")
        mock_chat.assert_called_once_with(
            11436,
            "qwen3.5:27b",
            [{"role": "user", "content": "say hi"}],
        )

    @patch("ollmo_webserver._persist_text_artifact_locally")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_chat")
    def test_chat_infer_persists_structured_text_artifact_request(
        self,
        mock_chat,
        mock_lookup,
        mock_persist_text_artifact,
    ):
        mock_lookup.return_value = {
            "instance_id": "qwen-1",
            "port": 11436,
            "model": "qwen3.5:27b",
            "backend": "ollama",
            "capability": "chat",
        }
        mock_chat.return_value = {"content": "```markdown\n# Einsatzprotokoll\n\n- Risiko: Vereisung.\n```"}
        mock_persist_text_artifact.return_value = "/tmp/artifacts/documents/einsatzprotokoll.md"

        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "qwen-1",
                "prompt": "Materialize only the requested md artifact.",
                "text_artifact_extension": "md",
                "text_artifact_source_name": "einsatzprotokoll",
                "text_artifact_source": "runtime_contract",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "chat")
        self.assertEqual(payload["saved_text_path"], "/tmp/artifacts/documents/einsatzprotokoll.md")
        self.assertEqual(payload["text_artifact_request"]["extension"], "md")
        mock_persist_text_artifact.assert_called_once()
        self.assertEqual(mock_persist_text_artifact.call_args.args[0], "# Einsatzprotokoll\n\n- Risiko: Vereisung.")
        self.assertEqual(mock_persist_text_artifact.call_args.kwargs["source_name"], "einsatzprotokoll")

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._persist_request_input_artifacts", return_value=[])
    @patch("ollmo_webserver._ollama_generate")
    @patch("ollmo_webserver._save_local_path_to_temp")
    @patch("ollmo_webserver._resolve_saved_downloadable_artifact_path")
    def test_chat_infer_selected_reference_image_and_message_use_file_context_and_prompt_anchor(
        self,
        mock_resolve_saved_artifact,
        mock_save_local_path,
        mock_generate,
        _mock_persist_input_artifacts,
        mock_lookup,
        mock_activity,
        mock_success,
    ):
        mock_lookup.return_value = {
            "instance_id": "chat-vision-1",
            "port": 11438,
            "model": "gemma4:e4b",
            "backend": "ollama",
            "capability": "chat",
            "features": {"vision_input": True},
            "inputs": ["text", "image"],
            "supported_capabilities": ["chat", "vision_analysis"],
            "provider_capabilities": ["chat", "vision_analysis"],
        }
        mock_generate.return_value = {"response": "The cat looks cautious."}
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as source_handle:
            source_handle.write(b"fakepng")
            selected_path = Path(source_handle.name)
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as temp_handle:
            temp_handle.write(b"fakepng")
            temp_path = Path(temp_handle.name)
        mock_resolve_saved_artifact.return_value = selected_path
        mock_save_local_path.return_value = (selected_path, temp_path)

        try:
            response = self.client.post(
                "/api/infer",
                json={
                    "instance_id": "chat-vision-1",
                    "prompt": "describe",
                    "selected_reference_artifact": [
                        {
                            "type": "message",
                            "content": "Use the earlier hooded-cat framing.",
                            "message_role": "assistant",
                        },
                        {
                            "type": "image",
                            "path": str(selected_path),
                        },
                    ],
                },
            )
        finally:
            selected_path.unlink(missing_ok=True)
            temp_path.unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "chat_with_image")
        mock_save_local_path.assert_called_once_with(str(selected_path))
        _mock_persist_input_artifacts.assert_not_called()
        mock_generate.assert_called_once()
        prompt = mock_generate.call_args.args[2]
        self.assertIn("Selected prior assistant reply reference for this conversation turn.", prompt)
        self.assertIn("bounded reference context only", prompt)
        self.assertIn("Use the earlier hooded-cat framing.", prompt)
        self.assertIn("Current user request:\ndescribe", prompt)
        self.assertTrue(mock_generate.call_args.kwargs["images"])

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._persist_request_input_artifacts", return_value=[])
    @patch("ollmo_webserver._ollama_generate")
    @patch("ollmo_webserver._save_local_path_to_temp")
    def test_chat_infer_artifact_binding_file_path_does_not_repersist_input_artifact(
        self,
        mock_save_local_path,
        mock_generate,
        mock_persist_input_artifacts,
        mock_lookup,
        mock_activity,
        mock_success,
    ):
        mock_lookup.return_value = {
            "instance_id": "chat-vision-1",
            "port": 11438,
            "model": "gemma4:e4b",
            "backend": "ollama",
            "capability": "chat",
            "features": {"vision_input": True},
            "inputs": ["text", "image"],
            "supported_capabilities": ["chat", "vision_analysis"],
            "provider_capabilities": ["chat", "vision_analysis"],
        }
        mock_generate.return_value = {"response": "The image shows a notebook."}
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as source_handle:
            source_handle.write(b"fakepng")
            source_path = Path(source_handle.name)
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as temp_handle:
            temp_handle.write(b"fakepng")
            temp_path = Path(temp_handle.name)
        mock_save_local_path.return_value = (source_path, temp_path)

        try:
            response = self.client.post(
                "/api/infer",
                json={
                    "instance_id": "chat-vision-1",
                    "prompt": "describe",
                    "file_path": str(source_path),
                    "artifact_bindings": [
                        {
                            "binding_kind": "route_reuse",
                            "artifact_ref": "artifact:image_latest",
                            "resolved_path": str(source_path),
                        }
                    ],
                },
            )
        finally:
            source_path.unlink(missing_ok=True)
            temp_path.unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["mode"], "chat_with_image")
        mock_save_local_path.assert_called_once_with(str(source_path))
        mock_persist_input_artifacts.assert_not_called()
        mock_generate.assert_called_once()

    @patch("ollmo_webserver.record_instance_success")
    @patch("ollmo_webserver.record_instance_activity")
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._find_artifact_registry_record")
    @patch("ollmo_webserver._persist_request_input_artifacts", return_value=[])
    @patch("ollmo_webserver._ollama_generate")
    @patch("ollmo_webserver._save_local_path_to_temp")
    def test_chat_infer_registry_known_file_path_does_not_repersist_input_artifact(
        self,
        mock_save_local_path,
        mock_generate,
        mock_persist_input_artifacts,
        mock_find_registry_record,
        mock_lookup,
        mock_activity,
        mock_success,
    ):
        mock_lookup.return_value = {
            "instance_id": "chat-vision-1",
            "port": 11438,
            "model": "gemma4:e4b",
            "backend": "ollama",
            "capability": "chat",
            "features": {"vision_input": True},
            "inputs": ["text", "image"],
            "supported_capabilities": ["chat", "vision_analysis"],
            "provider_capabilities": ["chat", "vision_analysis"],
        }
        mock_generate.return_value = {"response": "The image shows a notebook."}
        mock_activity.return_value = ({}, {"readiness": "ready"})
        mock_success.return_value = ({"readiness": "ready"}, {"readiness": "ready"})

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as source_handle:
            source_handle.write(b"fakepng")
            source_path = Path(source_handle.name)
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as temp_handle:
            temp_handle.write(b"fakepng")
            temp_path = Path(temp_handle.name)
        mock_save_local_path.return_value = (source_path, temp_path)
        mock_find_registry_record.return_value = {
            "roles": ["output"],
            "artifact": {
                "type": "image",
                "path": str(source_path),
                "origin": "assistant_output",
            },
        }

        try:
            response = self.client.post(
                "/api/infer",
                json={
                    "instance_id": "chat-vision-1",
                    "prompt": "describe",
                    "file_path": str(source_path),
                },
            )
        finally:
            source_path.unlink(missing_ok=True)
            temp_path.unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        mock_find_registry_record.assert_called_once()
        mock_persist_input_artifacts.assert_not_called()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_chat")
    def test_chat_infer_text_file_appended_to_prompt(self, mock_chat, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "coder-1",
            "port": 11437,
            "model": "qwen3-coder:latest",
            "backend": "ollama",
            "capability": "chat",
        }
        mock_chat.return_value = {"content": "Got it"}

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "coder-1",
                "prompt": "Summarize this",
                "file": (io.BytesIO(b"line1\nline2"), "notes.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["mode"], "chat")
        called_messages = mock_chat.call_args[0][2]
        self.assertEqual(called_messages[0]["role"], "user")
        self.assertIn("Summarize this", called_messages[0]["content"])
        self.assertIn("[Attached file content]", called_messages[0]["content"])
        self.assertIn("line1", called_messages[0]["content"])

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_chat")
    def test_chat_infer_local_text_path_appended_to_prompt(self, mock_chat, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "coder-1",
            "port": 11437,
            "model": "qwen3-coder:latest",
            "backend": "ollama",
            "capability": "chat",
        }
        mock_chat.return_value = {"content": "Done"}
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("alpha\nbeta")
            local_path = handle.name
        try:
            response = self.client.post(
                "/api/infer",
                json={
                    "instance_id": "coder-1",
                    "prompt": "Summarize this",
                    "file_path": local_path,
                },
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["mode"], "chat")
        called_messages = mock_chat.call_args[0][2]
        self.assertEqual(called_messages[0]["role"], "user")
        self.assertIn("Summarize this", called_messages[0]["content"])
        self.assertIn("[Attached file content]", called_messages[0]["content"])
        self.assertIn("alpha", called_messages[0]["content"])

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_chat")
    def test_chat_infer_large_text_file_adds_explicit_truncation_note(self, mock_chat, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "coder-1",
            "port": 11437,
            "model": "qwen3-coder:latest",
            "backend": "ollama",
            "capability": "chat",
        }
        mock_chat.return_value = {"content": "Done"}
        oversized_text = b"a" * 250100

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "coder-1",
                "prompt": "Review this file",
                "file": (io.BytesIO(oversized_text), "huge.py"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(any("truncated to 250000 of 250100 bytes" in warning for warning in payload["warnings"]))
        called_messages = mock_chat.call_args[0][2]
        self.assertIn("truncated to first 250000 of 250100 bytes", called_messages[0]["content"])
        self.assertIn("[Attached file content truncated", called_messages[0]["content"])

    @patch("ollmo_webserver._lookup_instance")
    def test_infer_rejects_upload_and_file_path_together(self, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "coder-1",
            "port": 11437,
            "model": "qwen3-coder:latest",
            "backend": "ollama",
            "capability": "chat",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("hello")
            local_path = handle.name
        try:
            response = self.client.post(
                "/api/infer",
                data={
                    "instance_id": "coder-1",
                    "prompt": "x",
                    "file_path": local_path,
                    "file": (io.BytesIO(b"hello"), "inline.txt"),
                },
                content_type="multipart/form-data",
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("either 'file' or 'file_path'", payload["error"])

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._persist_text_markdown_locally")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_text_layer(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_pdf,
        mock_persist_markdown,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = "Revenue grew by 20 percent."
        mock_render_pdf.return_value = ([], 0, [])
        mock_generate.return_value = {"response": "Umsatz stieg um 20 Prozent."}
        mock_persist_markdown.side_effect = [
            "/tmp/ocr_exports/report-source.md",
            "/tmp/ocr_exports/report-result.md",
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Decipher and translate to German.",
                "file": (io.BytesIO(b"%PDF-1.4 fake"), "report.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_pdf_text")
        self.assertEqual(payload["pdf_source"], "text_layer")
        self.assertIn("20 Prozent", payload["content"])
        self.assertEqual(payload["saved_source_text_path"], "/tmp/ocr_exports/report-source.md")
        self.assertEqual(payload["saved_text_path"], "/tmp/ocr_exports/report-result.md")
        self.assertFalse(payload["content_truncated"])
        called_prompt = mock_generate.call_args[0][2]
        self.assertIn("[PDF extracted text]", called_prompt)
        mock_render_pdf.assert_called_once()
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._persist_text_markdown_locally")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_pages_with_synthesis(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_persist_markdown,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_single_pdf_page.return_value = "ZmFrZQ=="
        mock_render_pdf.return_value = (["ZmFrZTE=", "ZmFrZTI="], 12, ["processed first pages only"])
        mock_persist_markdown.return_value = "/tmp/ocr_exports/scan.md"
        mock_generate.side_effect = [
            {"response": "Page 1 text"},
            {"response": "Page 2 text"},
            {"response": "Gesamtzusammenfassung auf Deutsch"},
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Summarize and translate to German.",
                "pdf_synthesize": "true",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_pdf_scan")
        self.assertEqual(payload["pdf_source"], "rendered_pages")
        self.assertEqual(payload["pdf_total_pages"], 12)
        self.assertEqual(payload["pdf_processed_pages"], 2)
        self.assertEqual(payload["content"], "Gesamtzusammenfassung auf Deutsch")
        self.assertEqual(payload["saved_text_path"], "/tmp/ocr_exports/scan.md")
        self.assertIn("processed first pages only", payload["warnings"])
        self.assertEqual(mock_generate.call_count, 3)
        self.assertEqual(mock_generate.call_args_list[0].kwargs["options"], {"num_predict": 8192})
        self.assertEqual(mock_generate.call_args_list[1].kwargs["options"], {"num_predict": 8192})
        mock_persist_markdown.assert_called_once()
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver.MAX_PDF_INLINE_RESPONSE_CHARS", 40)
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._persist_text_markdown_locally")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_text_layer_persists_full_source_and_flags_inline_truncation(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_pdf,
        mock_persist_markdown,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = "Original full extracted PDF text."
        mock_render_pdf.return_value = ([], 0, [])
        mock_generate.return_value = {"response": "A" * 120}
        mock_persist_markdown.side_effect = [
            "/tmp/ocr_exports/source.md",
            "/tmp/ocr_exports/result.md",
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Summarize this PDF.",
                "file": (io.BytesIO(b"%PDF-1.4 fake"), "report.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["content_truncated"])
        self.assertEqual(payload["saved_source_text_path"], "/tmp/ocr_exports/source.md")
        self.assertEqual(payload["saved_text_path"], "/tmp/ocr_exports/result.md")
        self.assertEqual(payload["full_content_chars"], 120)
        self.assertLess(payload["inline_content_chars"], payload["full_content_chars"])
        self.assertIn("UI output truncated", payload["content"])
        self.assertTrue(any("saved markdown artifact" in warning for warning in payload["warnings"]))
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver.MAX_PDF_INLINE_RESPONSE_CHARS", 60)
    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._persist_text_markdown_locally")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_flags_inline_truncation_but_saves_full_artifact(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_persist_markdown,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.return_value = {
            "response": " ".join(f"token{i:03d}" for i in range(80))
        }
        mock_persist_markdown.return_value = "/tmp/ocr_exports/full-scan.md"

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Extract text.",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["content_truncated"])
        self.assertEqual(payload["saved_text_path"], "/tmp/ocr_exports/full-scan.md")
        self.assertGreater(payload["full_content_chars"], payload["inline_content_chars"])
        self.assertIn("UI output truncated", payload["content"])
        self.assertTrue(any("saved markdown artifact" in warning for warning in payload["warnings"]))
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_returns_partial_result_when_one_page_times_out(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE=", "ZmFrZTI="], 2, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.side_effect = [
            Timeout("page 1 timeout"),
            Timeout("page 1 emergency timeout"),
            {"response": "Page 2 text"},
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Analyze this PDF",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_pdf_scan")
        self.assertEqual(payload["pdf_processed_pages"], 1)
        self.assertIn("[Page 2]", payload["content"])
        self.assertTrue(any("Page 1" in warning for warning in payload["warnings"]))
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_all_pages_timeout_returns_504(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.side_effect = [Timeout("page timeout"), Timeout("emergency timeout")]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Analyze this PDF",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 504)
        payload = response.get_json()
        self.assertIn("no OCR text was returned", payload["error"])
        self.assertTrue(any(("timeout" in warning.lower()) or ("timed out" in warning.lower()) for warning in payload["warnings"]))
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_uses_emergency_generate_when_generate_empty(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.side_effect = [
            {"response": ""},
            {"response": "Recovered OCR from emergency generate"},
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Analyze this PDF",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_pdf_scan")
        self.assertIn("Recovered OCR", payload["content"])
        self.assertEqual(payload["pdf_processed_pages"], 1)
        self.assertEqual(mock_generate.call_count, 2)
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_rejects_prompt_echo_and_uses_fallback(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.side_effect = [
            {
                "response": (
                    "User request/context:\nFor each page:\n"
                    "1) Verbatim transcription (preserve line breaks and structure).\n"
                    "2) Mark uncertain readings as [unclear].\n"
                    "3) List visual annotations."
                )
            },
            {"response": "Actual OCR text from fallback."},
        ]

        response_obj = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "For each page: verbatim transcription.",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response_obj.status_code, 200)
        payload = response_obj.get_json()
        self.assertIn("Actual OCR text from fallback", payload["content"])
        self.assertNotIn("User request/context", payload["content"])
        self.assertEqual(mock_generate.call_count, 2)
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_strips_deepseek_grounding_markup(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.return_value = {
            "response": (
                "<|ref|>text<|/ref|><|det|>[[10, 10, 20, 20]]<|/det|>\n"
                "Dear John,\n"
                "Enclosed is the correspondence."
            )
        }

        response_obj = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Extract text.",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response_obj.status_code, 200)
        payload = response_obj.get_json()
        self.assertIn("Dear John", payload["content"])
        self.assertNotIn("<|ref|>", payload["content"])
        self.assertNotIn("<|det|>", payload["content"])
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_rejects_low_quality_repetition_and_uses_fallback(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.side_effect = [
            {"response": ("On the Atlantic Ocean KEY WEST, FLORIDA\n" * 80)},
            {"response": "Recovered OCR from emergency generate"},
        ]

        response_obj = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Extract text.",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response_obj.status_code, 200)
        payload = response_obj.get_json()
        self.assertIn("Recovered OCR from emergency generate", payload["content"])
        self.assertEqual(mock_generate.call_count, 2)
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_replaces_repeated_phrase_noise_with_unclear(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        mock_generate.return_value = {
            "response": (
                "Her densities are acquired only through pressures of varying intensities, "
                "not because of their crystal forms, " * 60
            )
        }

        response_obj = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Extract text.",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response_obj.status_code, 200)
        payload = response_obj.get_json()
        self.assertIn("[Page 1]", payload["content"])
        self.assertIn("[unclear]", payload["content"])
        self.assertNotIn("not because of their crystal forms, not because", payload["content"])
        self.assertEqual(mock_generate.call_count, 1)
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._render_single_pdf_page_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_pdf_scan_uses_emergency_generate_when_generate_http_500(
        self,
        mock_generate,
        mock_extract_pdf_text,
        mock_render_single_pdf_page,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZTE="], 1, [])
        mock_render_single_pdf_page.return_value = None
        error = RequestException("boom")
        response = Mock()
        response.status_code = 500
        response.json.return_value = {"error": "model internal error"}
        error.response = response
        mock_generate.side_effect = [error, {"response": "Recovered after generate 500"}]

        response_obj = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Analyze this PDF",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response_obj.status_code, 200)
        payload = response_obj.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_pdf_scan")
        self.assertIn("Recovered after generate 500", payload["content"])
        self.assertEqual(payload["pdf_processed_pages"], 1)
        self.assertEqual(mock_generate.call_count, 2)
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._append_infer_history")
    @patch("ollmo_webserver._render_pdf_pages_to_base64")
    @patch("ollmo_webserver._extract_pdf_text_content")
    @patch("ollmo_webserver._ollama_chat")
    def test_chat_pdf_scan_requires_vision_model(
        self,
        mock_chat,
        mock_extract_pdf_text,
        mock_render_pdf,
        mock_append_history,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "chat-1",
            "port": 11436,
            "model": "qwen3.5:27b",
            "backend": "ollama",
            "capability": "chat",
        }
        mock_extract_pdf_text.return_value = ""
        mock_render_pdf.return_value = (["ZmFrZQ=="], 5, ["scan pdf"])

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "chat-1",
                "prompt": "Analyze this pdf",
                "file": (io.BytesIO(b"%PDF-1.4 fake scan"), "scan.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("vision_analysis", payload["error"])
        mock_chat.assert_not_called()
        mock_append_history.assert_called_once()

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._find_cached_pdf_insight")
    @patch("ollmo_webserver._hash_file_sha256")
    @patch("ollmo_webserver._ollama_generate")
    def test_pdf_cached_result_is_reused(
        self,
        mock_generate,
        mock_hash,
        mock_find_cached,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_hash.return_value = "abc123"
        mock_find_cached.return_value = {
            "id": "infer-1",
            "mode": "vision_analysis_pdf_scan",
            "content": "Cached OCR summary",
            "warnings": ["from cache"],
            "pdf_source": "rendered_pages",
            "pdf_total_pages": 12,
            "pdf_processed_pages": 8,
        }

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Summarize this",
                "file": (io.BytesIO(b"%PDF-1.4 fake"), "cached.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["cached"])
        self.assertEqual(payload["cache_id"], "infer-1")
        self.assertEqual(payload["content"], "Cached OCR summary")
        mock_generate.assert_not_called()

    @patch("ollmo_webserver._read_infer_history")
    def test_infer_history_endpoint_filters(self, mock_read_history):
        mock_read_history.return_value = [
            {"id": "1", "file_kind": "pdf", "mode": "vision_analysis_pdf_scan", "capability": "vision_analysis"},
            {"id": "2", "file_kind": "image", "mode": "vision_analysis", "capability": "vision_analysis"},
        ]

        response = self.client.get("/api/infer_history?file_kind=pdf&limit=10")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["id"], "1")

    @patch("ollmo_webserver._read_infer_history")
    def test_find_cached_pdf_insight_skips_prompt_echo_entries(self, mock_read_history):
        mock_read_history.return_value = [
            {
                "id": "bad-echo",
                "status": "ok",
                "file_kind": "pdf",
                "file_sha256": "abc123",
                "model": "deepseek-ocr:latest",
                "backend": "ollama",
                "capability": "vision_analysis",
                "prompt": "For each page: verbatim transcription.",
                "mode": "vision_analysis_pdf_scan",
                "pdf_processed_pages": 4,
                "content": "User request/context:\nFor each page:\n1) Verbatim transcription...",
            },
            {
                "id": "good-entry",
                "status": "ok",
                "file_kind": "pdf",
                "file_sha256": "abc123",
                "model": "deepseek-ocr:latest",
                "backend": "ollama",
                "capability": "vision_analysis",
                "prompt": "For each page: verbatim transcription.",
                "mode": "vision_analysis_pdf_scan",
                "pdf_processed_pages": 4,
                "content": "[Page 1]\nActual OCR content",
            },
        ]

        cached = _find_cached_pdf_insight(
            file_sha256="abc123",
            model_name="deepseek-ocr:latest",
            backend="ollama",
            capability="vision_analysis",
            prompt="For each page: verbatim transcription.",
        )

        self.assertIsNotNone(cached)
        self.assertEqual(cached["id"], "good-entry")

    @patch("ollmo_webserver._open_path_in_file_manager")
    @patch("ollmo_webserver._resolve_generated_image_path")
    def test_open_saved_image_success(self, mock_resolve_path, mock_open_path):
        mock_resolve_path.return_value = Path("/tmp/artifacts/images/cat.png")

        response = self.client.post(
            "/api/open_saved_image",
            json={"path": "/tmp/artifacts/images/cat.png"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "opened")
        mock_open_path.assert_called_once_with(Path("/tmp/artifacts/images/cat.png"))

    @patch("ollmo_webserver._resolve_generated_image_path")
    def test_open_saved_image_rejects_invalid_path(self, mock_resolve_path):
        mock_resolve_path.return_value = None

        response = self.client.post(
            "/api/open_saved_image",
            json={"path": "/Users/example/Desktop/secret.txt"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("artifacts/images", payload["error"])

    @patch("ollmo_webserver._open_path_in_file_manager")
    @patch("ollmo_webserver._resolve_saved_openable_artifact_path")
    def test_open_saved_artifact_success(self, mock_resolve_path, mock_open_path):
        mock_resolve_path.return_value = Path("/tmp/artifacts/ocr/result.md")

        response = self.client.post(
            "/api/open_saved_artifact",
            json={"path": "/tmp/artifacts/ocr/result.md"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "opened")
        mock_open_path.assert_called_once_with(Path("/tmp/artifacts/ocr/result.md"))

    @patch("ollmo_webserver._resolve_saved_openable_artifact_path")
    def test_open_saved_artifact_rejects_invalid_path(self, mock_resolve_path):
        mock_resolve_path.return_value = None

        response = self.client.post(
            "/api/open_saved_artifact",
            json={"path": "/Users/example/Desktop/secret.txt"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("artifacts/audio", payload["error"])
        self.assertIn("artifacts/ocr", payload["error"])

    @patch("ollmo_webserver._resolve_saved_downloadable_artifact_path")
    def test_download_saved_artifact_success(self, mock_resolve_path):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp.write(b"# OCR\n\nhello\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(lambda: tmp_path.unlink(missing_ok=True))
        mock_resolve_path.return_value = tmp_path

        response = self.client.get(
            "/api/download_saved_artifact",
            query_string={"path": str(tmp_path)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"# OCR\n\nhello\n")
        response.close()

    @patch("ollmo_webserver._resolve_saved_downloadable_artifact_path")
    def test_download_saved_artifact_rejects_invalid_path(self, mock_resolve_path):
        mock_resolve_path.return_value = None

        response = self.client.get(
            "/api/download_saved_artifact",
            query_string={"path": "/Users/example/Desktop/secret.txt"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("artifacts/audio", payload["error"])
        self.assertIn("artifacts/ocr", payload["error"])

    @patch("ollmo_webserver._resolve_saved_viewable_artifact_path")
    def test_view_saved_artifact_success(self, mock_resolve_path):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake image bytes")
            tmp_path = Path(tmp.name)
        self.addCleanup(lambda: tmp_path.unlink(missing_ok=True))
        mock_resolve_path.return_value = tmp_path

        response = self.client.get(
            "/api/view_saved_artifact",
            query_string={"path": str(tmp_path)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"fake image bytes")
        content_disposition = response.headers.get("Content-Disposition", "")
        self.assertNotIn("attachment", content_disposition.lower())
        response.close()

    @patch("ollmo_webserver._resolve_saved_viewable_artifact_path")
    def test_view_saved_html_artifact_sets_csp_without_unsafe_eval(self, mock_resolve_path):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(b"<!doctype html><title>Artifact</title>")
            tmp_path = Path(tmp.name)
        self.addCleanup(lambda: tmp_path.unlink(missing_ok=True))
        mock_resolve_path.return_value = tmp_path

        response = self.client.get(
            "/api/view_saved_artifact",
            query_string={"path": str(tmp_path)},
        )

        self.assertEqual(response.status_code, 200)
        csp = response.headers.get("Content-Security-Policy", "")
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-eval", csp)
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        response.close()

    @patch("ollmo_webserver._resolve_saved_viewable_artifact_path")
    def test_view_saved_artifact_rejects_invalid_path(self, mock_resolve_path):
        mock_resolve_path.return_value = None

        response = self.client.get(
            "/api/view_saved_artifact",
            query_string={"path": "/Users/example/Desktop/secret.txt"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("artifacts/audio", payload["error"])
        self.assertIn("artifacts/ocr", payload["error"])

    @patch("ollmo_webserver._resolve_saved_downloadable_artifact_path")
    def test_delete_saved_artifact_success(self, mock_resolve_path):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake image bytes")
            tmp_path = Path(tmp.name)
        self.addCleanup(lambda: tmp_path.unlink(missing_ok=True))
        mock_resolve_path.return_value = tmp_path

        response = self.client.post(
            "/api/delete_saved_artifact",
            json={"path": str(tmp_path)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "deleted")
        self.assertFalse(tmp_path.exists())

    @patch("ollmo_webserver._resolve_saved_downloadable_artifact_path")
    def test_delete_saved_artifact_rejects_invalid_path(self, mock_resolve_path):
        mock_resolve_path.return_value = None

        response = self.client.post(
            "/api/delete_saved_artifact",
            json={"path": "/Users/example/Desktop/secret.txt"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("artifacts/audio", payload["error"])
        self.assertIn("artifacts/ocr", payload["error"])

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_deepseek_image_ocr_mode_uses_grounding_prompt(
        self,
        mock_generate,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_generate.return_value = {
            "response": "<|ref|>text<|/ref|><|det|>[[10,10,20,20]]<|/det|>\nDear John,"
        }

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "ocr these",
                "file": (io.BytesIO(b"fake"), "scan.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_ocr_image")
        self.assertIn("Dear John", payload["content"])
        self.assertNotIn("<|ref|>", payload["content"])
        first_prompt = mock_generate.call_args_list[0].args[2]
        self.assertIn("<|grounding|>Convert the document to markdown.", first_prompt)

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_deepseek_image_ocr_mode_uses_emergency_fallback(
        self,
        mock_generate,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_generate.side_effect = [
            {"response": ""},
            {"response": "Recovered OCR text"},
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "ocr these",
                "file": (io.BytesIO(b"fake"), "scan.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_ocr_image")
        self.assertIn("Recovered OCR text", payload["content"])
        self.assertEqual(mock_generate.call_count, 2)
        first_prompt = mock_generate.call_args_list[0].args[2]
        second_prompt = mock_generate.call_args_list[1].args[2]
        self.assertIn("<|grounding|>Convert the document to markdown.", first_prompt)
        self.assertIn("Free OCR.", second_prompt)

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_deepseek_image_ocr_rejects_low_quality_repetition(
        self,
        mock_generate,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_generate.side_effect = [
            {"response": ("تخفيضات\n" * 120)},
            {"response": "Recovered OCR text"},
        ]

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "ocr",
                "file": (io.BytesIO(b"fake"), "scan.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_ocr_image")
        self.assertIn("Recovered OCR text", payload["content"])
        self.assertEqual(mock_generate.call_count, 2)

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_analysis_deepseek_image_ocr_replaces_numeric_spam_with_unclear(
        self,
        mock_generate,
        mock_lookup,
    ):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_generate.return_value = {
            "response": "## " + ("1 " * 240)
        }

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "ocr",
                "file": (io.BytesIO(b"fake"), "scan.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "vision_analysis_ocr_image")
        self.assertEqual(payload["content"], "[unclear]")
        self.assertEqual(mock_generate.call_count, 1)

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._ollama_generate")
    def test_vision_infer_connection_error_returns_503(self, mock_generate, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_generate.side_effect = ConnectionError("Remote end closed connection without response")

        response = self.client.post(
            "/api/infer",
            data={
                "instance_id": "ocr-1",
                "prompt": "Extract text",
                "file": (io.BytesIO(b"fake"), "scan.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertIn("interrupted", payload["error"])

    @patch("ollmo_webserver._lookup_instance")
    @patch("ollmo_webserver._acquire_infer_slot")
    def test_infer_duplicate_request_returns_409(self, mock_acquire_slot, mock_lookup):
        mock_lookup.return_value = {
            "instance_id": "ocr-1",
            "port": 11437,
            "model": "deepseek-ocr:latest",
            "backend": "ollama",
            "capability": "vision_analysis",
        }
        mock_acquire_slot.return_value = False

        response = self.client.post(
            "/api/infer",
            json={
                "instance_id": "ocr-1",
                "prompt": "Extract text",
            },
        )

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertIn("identical request", payload["error"])

    def test_expand_local_paths_expands_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "sub"
            nested.mkdir(parents=True, exist_ok=True)
            (root / "one.txt").write_text("hello", encoding="utf-8")
            (nested / "two.png").write_bytes(b"\x89PNG")

            response = self.client.post(
                "/api/expand_local_paths",
                json={"paths": [str(root)]},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 2)
        self.assertFalse(payload["truncated"])
        self.assertEqual(len(payload["paths"]), 2)
        self.assertTrue(any(item.endswith("one.txt") for item in payload["paths"]))
        self.assertTrue(any(item.endswith("two.png") for item in payload["paths"]))

    def test_expand_local_paths_rejects_empty_payload(self):
        response = self.client.post(
            "/api/expand_local_paths",
            json={},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("paths", payload["error"])


if __name__ == "__main__":
    unittest.main()
