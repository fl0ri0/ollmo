import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_services.response_frames import build_response_frame
from ollmo_services.settings_artifacts import (
    build_settings_artifact,
    list_settings_artifacts,
    load_settings_artifact,
    persist_settings_artifact,
)
from ollmo_webserver import app


def _sample_response_frame():
    return build_response_frame(
        {
            "id": "resp_settings",
            "object": "response",
            "status": "completed",
            "model": "flux",
            "instance_id": "image-1",
            "backend": "mlx",
            "capability": "image_generation",
            "mode": "image_generation",
            "output_text": "Generated image.",
            "artifacts": [{"type": "image", "path": "artifacts/images/out.png"}],
        },
        request_payload={
            "input": "make a reusable cover",
            "width": "1024",
            "height": "768",
            "seed": "1234",
            "selected_reference_artifacts": [
                {
                    "type": "message",
                    "message_id": "msg_reference",
                    "message_role": "assistant",
                    "response_model": "gemma4:26b",
                    "response_instance_id": "gemma4:26b-1",
                }
            ],
        },
    )


class SettingsArtifactTests(unittest.TestCase):
    def test_build_settings_artifact_from_response_frame(self):
        frame = _sample_response_frame()

        artifact = build_settings_artifact(
            frame,
            label="cover preset",
            created_at="2026-04-10T13:10:00Z",
        )

        self.assertEqual(artifact["kind"], "ollmo.settings_artifact")
        self.assertEqual(artifact["artifact_version"], 1)
        self.assertEqual(artifact["label"], "cover preset")
        self.assertEqual(artifact["source"]["response_id"], "resp_settings")
        self.assertEqual(artifact["target"]["instance_id"], "image-1")
        self.assertEqual(artifact["controls"]["image"]["width"], 1024)
        self.assertEqual(artifact["request_overrides"]["width"], 1024)
        self.assertEqual(artifact["request_overrides"]["height"], 768)
        self.assertEqual(artifact["request_overrides"]["seed"], 1234)
        self.assertNotIn("references", artifact["request_overrides"])
        self.assertEqual(artifact["references"]["selected"][0]["ref"], "message:message_msg_reference")
        self.assertEqual(artifact["references"]["selected"][0]["message_id"], "msg_reference")
        self.assertEqual(artifact["references"]["selected"][0]["message_role"], "assistant")
        self.assertEqual(artifact["references"]["selected"][0]["response_model"], "gemma4:26b")
        self.assertEqual(artifact["references"]["selected"][0]["response_instance_id"], "gemma4:26b-1")

    def test_persist_list_and_load_settings_artifact(self):
        frame = _sample_response_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = persist_settings_artifact(
                frame,
                label="image preset",
                artifacts_dir=Path(tmpdir),
            )

            path = Path(artifact["path"])
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["artifact_id"], artifact["artifact_id"])

            listed = list_settings_artifacts(artifacts_dir=Path(tmpdir))
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["artifact_id"], artifact["artifact_id"])

            loaded = load_settings_artifact(artifact["artifact_id"], artifacts_dir=Path(tmpdir))
            self.assertEqual(loaded["request_overrides"]["width"], 1024)

    def test_build_settings_artifact_rejects_empty_controls(self):
        with self.assertRaises(ValueError):
            build_settings_artifact({"kind": "ollmo.control_snapshot", "values": {}})

    def test_settings_artifact_api_promotes_lists_and_loads(self):
        app.config["TESTING"] = True
        client = app.test_client()
        frame = _sample_response_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ollmo_webserver.SETTINGS_ARTIFACTS_DIR", Path(tmpdir)):
                post_response = client.post(
                    "/api/settings_artifacts",
                    json={"response_frame": frame, "label": "api preset"},
                )
                self.assertEqual(post_response.status_code, 200)
                post_payload = post_response.get_json()
                artifact = post_payload["artifact"]
                self.assertEqual(artifact["label"], "api preset")
                self.assertEqual(artifact["request_overrides"]["seed"], 1234)

                list_response = client.get("/api/settings_artifacts")
                self.assertEqual(list_response.status_code, 200)
                list_payload = list_response.get_json()
                self.assertEqual(len(list_payload["artifacts"]), 1)

                get_response = client.get(f"/api/settings_artifacts/{artifact['artifact_id']}")
                self.assertEqual(get_response.status_code, 200)
                get_payload = get_response.get_json()
                self.assertEqual(get_payload["artifact"]["artifact_id"], artifact["artifact_id"])

    def test_settings_artifact_api_rejects_payload_without_controls(self):
        app.config["TESTING"] = True
        client = app.test_client()
        response = client.post(
            "/api/settings_artifacts",
            json={"response_frame": {"kind": "ollmo.response_frame", "response_id": "resp_empty"}},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("controls", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
