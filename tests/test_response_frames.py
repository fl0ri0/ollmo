import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ollmo_services.response_frames as response_frames_module
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_services.control_snapshots import build_control_snapshot
from ollmo_services.response_frames import (
    _read_snapshot_ref_payload,
    attach_response_frame,
    build_response_frame,
    inspect_response_frame_recovery_cache,
    load_response_frame_index,
    load_latest_response_observation_state,
    load_latest_response_state,
    load_latest_response_wire_state,
    persist_response_frame,
    response_payload_from_frame,
)
from ollmo_services.responses import (
    build_canonical_outputs,
    build_canonical_response_artifacts,
    merge_canonical_response_artifacts,
    select_public_output_text,
)


class ResponseFrameTests(unittest.TestCase):
    def test_terminal_error_identity_and_recovery_survive_wire_persistence(self):
        response_id = "resp_external_failure_truth"
        source_payload = {
            "id": response_id,
            "status": "failed",
            "lifecycle_state": "failed",
            "error": "Codex execution ended with status 'failed'.",
            "error_detail": {
                "message": "Codex execution ended with status 'failed'.",
                "diagnostic": "The external process exited with status 1.",
            },
            "error_ref": {
                "code": "CODEX_EXECUTION_FAILED",
                "stage": "external_execution",
            },
            "recovery_hint": "Verify the existing Codex login and retry.",
            "outputs": [
                {
                    "slot_id": "output-1",
                    "type": "text",
                    "status": "blocked",
                }
            ],
        }

        frame = build_response_frame(
            source_payload,
            request_payload={"prompt": "Return a short answer."},
        )

        self.assertEqual(
            frame["current_state"]["error_ref"]["code"],
            "CODEX_EXECUTION_FAILED",
        )
        self.assertEqual(
            frame["current_state"]["recovery_hint"],
            "Verify the existing Codex login and retry.",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(frame, frames_dir=frames_dir)
            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(wire_state["ok"])
        wire_payload = wire_state["response_payload"]
        self.assertEqual(
            wire_payload["error_ref"]["code"],
            "CODEX_EXECUTION_FAILED",
        )
        self.assertEqual(
            wire_payload["recovery_hint"],
            "Verify the existing Codex login and retry.",
        )

    def test_multiple_saved_text_artifacts_are_canonicalized(self):
        artifacts = build_canonical_response_artifacts(
            {
                "id": "resp_multi_text",
                "saved_text_path": "/tmp/artifacts/documents/index.html",
                "saved_text_artifacts": [
                    {
                        "path": "/tmp/artifacts/documents/index.html",
                        "text_artifact_request": {"source_name": "index", "extension": "html"},
                    },
                    {
                        "path": "/tmp/artifacts/documents/styles.css",
                        "text_artifact_request": {"source_name": "styles", "extension": "css"},
                    },
                ],
            }
        )

        self.assertEqual([item["type"] for item in artifacts], ["text", "text"])
        self.assertEqual(
            [item["path"] for item in artifacts],
            ["/tmp/artifacts/documents/index.html", "/tmp/artifacts/documents/styles.css"],
        )

    def test_late_fill_text_source_upgrades_top_level_saved_text_path(self):
        artifacts = build_canonical_response_artifacts(
            {
                "id": "resp_text_path_stale_provenance",
                "saved_text_path": "/tmp/artifacts/documents/index.html",
                "provenance_id": "generated_image_stale",
                "late_fill": {
                    "fill_results": [
                        {
                            "branch_id": "repair-chat",
                            "phase_id": "repair-chat",
                            "capability": "chat",
                            "output_type": "text",
                            "saved_text_path": "/tmp/artifacts/documents/index.html",
                            "text_artifact_source_name": "index",
                        },
                        {
                            "branch_id": "branch-text_artifact-1",
                            "phase_id": "phase-6",
                            "capability": "chat",
                            "output_type": "text",
                            "saved_text_path": "/tmp/artifacts/documents/index.html",
                            "text_artifact_source_name": "index",
                        }
                    ]
                },
            }
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["branch_id"], "branch-text_artifact-1")
        self.assertEqual(artifacts[0]["phase_id"], "phase-6")
        self.assertNotIn("text_generated_image", artifacts[0]["artifact_ref"])

    def test_late_fill_media_sources_preserve_branch_local_artifact_identity(self):
        artifacts = build_canonical_response_artifacts(
            {
                "id": "resp_branch_local_media_identity",
                "saved_image_path": "/tmp/artifacts/images/scene.png",
                "saved_audio_path": "/tmp/artifacts/audio/narration.wav",
                "provenance_id": "generated_image_global_compatibility",
                "late_fill": {
                    "fill_results": [
                        {
                            "branch_id": "branch-image_generation-1",
                            "phase_id": "phase-2",
                            "capability": "image_generation",
                            "saved_image_path": "/tmp/artifacts/images/scene.png",
                            "artifact_id": "image_registry_scene",
                            "artifact_ref": "artifact:image_registry_scene",
                            "provenance_id": "generated_image_scene",
                        },
                        {
                            "branch_id": "branch-text_to_speech-1",
                            "phase_id": "phase-3",
                            "capability": "text_to_speech",
                            "saved_audio_path": "/tmp/artifacts/audio/narration.wav",
                            "artifact_id": "audio_registry_narration",
                            "artifact_ref": "artifact:audio_registry_narration",
                            "provenance_id": "generated_audio_narration",
                        },
                    ]
                },
            }
        )

        by_branch = {
            artifact.get("branch_id"): artifact
            for artifact in artifacts
            if artifact.get("branch_id")
        }
        self.assertEqual(
            by_branch["branch-image_generation-1"]["artifact_ref"],
            "artifact:image_registry_scene",
        )
        self.assertEqual(
            by_branch["branch-text_to_speech-1"]["artifact_ref"],
            "artifact:audio_registry_narration",
        )
        self.assertEqual(
            by_branch["branch-text_to_speech-1"]["provenance_id"],
            "generated_audio_narration",
        )

    def test_branch_owned_audio_does_not_inherit_root_image_provenance_in_either_order(self):
        image_result = {
            "branch_id": "branch-image_generation-1",
            "phase_id": "phase-2",
            "capability": "image_generation",
            "saved_image_path": "/tmp/artifacts/images/scene.png",
            "artifact_id": "image_registry_scene",
            "artifact_ref": "artifact:image_registry_scene",
            "provenance_id": "generated_image_scene",
        }
        audio_result = {
            "branch_id": "branch-text_to_speech-1",
            "phase_id": "phase-3",
            "capability": "text_to_speech",
            "saved_audio_path": "/tmp/artifacts/audio/narration.wav",
            "artifact_id": "audio_registry_narration",
            "artifact_ref": "artifact:audio_registry_narration",
        }

        for fill_results in (
            [image_result, audio_result],
            [audio_result, image_result],
        ):
            payload = {
                "id": "resp_branch_owned_media_provenance",
                "saved_image_path": image_result["saved_image_path"],
                "saved_audio_path": audio_result["saved_audio_path"],
                "provenance_id": "generated_image_global_compatibility",
                "artifacts": [
                    {
                        "type": "image",
                        "path": image_result["saved_image_path"],
                        "artifact_id": image_result["artifact_id"],
                        "artifact_ref": image_result["artifact_ref"],
                        "provenance_id": image_result["provenance_id"],
                    },
                    {
                        "type": "audio",
                        "path": audio_result["saved_audio_path"],
                        "artifact_id": audio_result["artifact_id"],
                        "artifact_ref": audio_result["artifact_ref"],
                    },
                ],
                "late_fill": {"fill_results": fill_results},
            }
            artifacts = merge_canonical_response_artifacts(
                payload["artifacts"],
                build_canonical_response_artifacts(payload),
            )
            by_path = {artifact["path"]: artifact for artifact in artifacts}
            self.assertEqual(
                by_path[audio_result["saved_audio_path"]]["artifact_ref"],
                "artifact:audio_registry_narration",
            )
            self.assertNotIn(
                "provenance_id",
                by_path[audio_result["saved_audio_path"]],
            )
            self.assertEqual(
                by_path[image_result["saved_image_path"]]["provenance_id"],
                "generated_image_scene",
            )

    @patch('ollmo_services.artifact_dossiers.find_artifact_registry_record')
    @patch('ollmo_services.artifact_dossiers.find_artifact_registry_record_by_artifact_ref')
    def test_mixed_media_frame_keeps_distinct_refs_and_branch_producer_dossiers(
        self,
        mock_find_registry_by_ref,
        mock_find_registry_by_path,
    ):
        mock_find_registry_by_ref.return_value = None
        mock_find_registry_by_path.return_value = None
        image_path = "/tmp/artifacts/images/scene.png"
        audio_path = "/tmp/artifacts/audio/narration.wav"
        image_result = {
            "branch_id": "branch-image_generation-1",
            "phase_id": "phase-2",
            "capability": "image_generation",
            "fill_instance_id": "image-1",
            "fill_model": "flux",
            "fill_backend": "ollama",
            "fill_mode": "image_generation",
            "saved_image_path": image_path,
            "artifact_id": "image_registry_scene",
            "artifact_ref": "artifact:image_registry_scene",
            "provenance_id": "generated_image_scene",
        }
        audio_result = {
            "branch_id": "branch-text_to_speech-1",
            "phase_id": "phase-3",
            "capability": "text_to_speech",
            "fill_instance_id": "tts-1",
            "fill_model": "qwen3-tts",
            "fill_backend": "mlx",
            "fill_mode": "text_to_speech",
            "saved_audio_path": audio_path,
            "artifact_id": "audio_registry_narration",
            "artifact_ref": "artifact:audio_registry_narration",
            "provenance_id": "generated_audio_narration",
        }
        payload = {
            "id": "resp_mixed_media_frame_identity",
            "status": "completed",
            "model": "gemma-root",
            "instance_id": "chat-root",
            "backend": "ollama",
            "capability": "chat",
            "mode": "chat",
            "output_text": "Artifacts generated.",
            "saved_image_path": image_path,
            "saved_audio_path": audio_path,
            "provenance_id": "generated_image_global_compatibility",
            "artifacts": build_canonical_response_artifacts(
                {
                    "id": "resp_mixed_media_frame_identity",
                    "late_fill": {"fill_results": [image_result, audio_result]},
                }
            ),
            "late_fill": {
                "status": "completed",
                "completed_branches": [image_result, audio_result],
                "fill_results": [image_result, audio_result],
            },
        }

        frame = build_response_frame(payload, request_payload={"prompt": "Create image and audio."})

        by_path = {artifact["path"]: artifact for artifact in frame["artifacts"]["output"]}
        self.assertEqual(by_path[image_path]["artifact_ref"], "artifact:image_registry_scene")
        self.assertEqual(by_path[audio_path]["artifact_ref"], "artifact:audio_registry_narration")
        self.assertNotIn("generated_image", by_path[audio_path]["artifact_ref"])
        audio_dossier = frame["artifacts"]["dossiers"]["artifact:audio_registry_narration"]
        self.assertEqual(audio_dossier["metadata"]["response_model"], "qwen3-tts")
        self.assertEqual(audio_dossier["metadata"]["backend"], "mlx")
        self.assertEqual(audio_dossier["metadata"]["capability"], "text_to_speech")

    def test_canonical_artifact_merge_upgrades_repair_binding_to_public_branch(self):
        merged = merge_canonical_response_artifacts(
            [
                {
                    "type": "text",
                    "artifact_ref": "artifact:text_index",
                    "ref": "artifact:text_index",
                    "path": "/tmp/artifacts/documents/index.html",
                    "branch_id": "repair-chat",
                    "phase_id": "repair-chat",
                    "content": "<!doctype html><h1>Fixed</h1>",
                },
            ],
            [
                {
                    "type": "text",
                    "artifact_ref": "artifact:text_index",
                    "ref": "artifact:text_index",
                    "path": "/tmp/artifacts/documents/index.html",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-4",
                },
            ],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["branch_id"], "branch-text_artifact-1")
        self.assertEqual(merged[0]["phase_id"], "phase-4")
        self.assertEqual(merged[0]["content"], "<!doctype html><h1>Fixed</h1>")

    def test_saved_text_artifact_attaches_to_matching_text_artifact_branch(self):
        prompt = "Create index.html and styles.css artifacts for an alpine rescue station dashboard."
        graph = build_request_phase_graph(prompt)
        payload = {
            "id": "resp_html_done_css_pending",
            "status": "completed",
            "output_text": "```html\n<!doctype html><h1>Alpine Rescue</h1>\n```",
            "saved_text_path": "/tmp/artifacts/documents/generated-html.html",
            "runtime": {"request_phase_graph": graph},
        }
        payload["artifacts"] = build_canonical_response_artifacts(payload)

        frame = build_response_frame(payload, request_payload={"input": prompt})
        output_slots = frame["planning"]["artifact_flow"]["output_slots"]
        root_slot = next(slot for slot in output_slots if slot.get("phase_id") == "phase-1")
        html_slot = next(slot for slot in output_slots if slot.get("branch_id") == "branch-text_artifact-1")
        css_slot = next(slot for slot in output_slots if slot.get("branch_id") == "branch-text_artifact-2")

        self.assertEqual(root_slot["status"], "fulfilled")
        self.assertNotIn("artifact_ref", root_slot)
        self.assertEqual(html_slot["status"], "fulfilled")
        self.assertEqual(html_slot["artifact_ref"], payload["artifacts"][0]["artifact_ref"])
        self.assertEqual(html_slot["artifact_path"], payload["artifacts"][0]["path"])
        self.assertEqual(css_slot["status"], "pending")

    def test_completed_text_artifact_branch_without_file_stays_pending(self):
        prompt = "Create an index.html artifact with a hello page."
        graph = build_request_phase_graph(prompt)
        payload = {
            "id": "resp_completed_text_branch_without_file",
            "status": "completed",
            "output_text": "```html\n<!doctype html><h1>Hello</h1>\n```",
            "runtime": {"request_phase_graph": graph},
            "late_fill": {
                "completed_branches": [
                    {
                        "branch_id": "branch-text_artifact-1",
                        "phase_id": "phase-2",
                        "capability": "chat",
                    }
                ],
            },
        }
        payload["artifacts"] = build_canonical_response_artifacts(payload)

        frame = build_response_frame(payload, request_payload={"input": prompt})
        output_slots = frame["planning"]["artifact_flow"]["output_slots"]
        text_artifact_slot = next(
            slot for slot in output_slots if slot.get("branch_id") == "branch-text_artifact-1"
        )

        self.assertEqual(text_artifact_slot["status"], "pending")
        self.assertNotIn("artifact_path", text_artifact_slot)

    def test_saved_text_artifact_does_not_attach_to_speech_to_text_branch(self):
        prompt = (
            "Erstelle einen kurzen Warnhinweis, speichere ihn als txt-Artefakt, "
            "erzeuge daraus Audio, und gib danach aus, ob Text und Audio zusammenpassen."
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={"ghost_route": True, "input": prompt},
            route_payload={"capability": "text_to_speech", "route_source": "ghost_carried"},
        )
        payload = {
            "id": "resp_txt_audio_review",
            "status": "completed",
            "output_text": "ACHTUNG: Wetterumschwung im Sektor Nord.",
            "saved_text_path": "/tmp/artifacts/documents/generated-txt.txt",
            "saved_audio_path": "/tmp/artifacts/audio/generated-warning.wav",
            "runtime": {"request_phase_graph": graph},
            "late_fill": {
                "completed_branches": [
                    {"branch_id": "branch-text_to_speech-1", "phase_id": "phase-2", "capability": "text_to_speech"},
                    {"branch_id": "branch-speech_to_text-1", "phase_id": "phase-3", "capability": "speech_to_text"},
                    {"branch_id": "branch-text_artifact-1", "phase_id": "phase-5", "capability": "chat"},
                ],
            },
        }
        payload["artifacts"] = build_canonical_response_artifacts(payload)

        frame = build_response_frame(payload, request_payload={"input": prompt})
        output_slots = frame["planning"]["artifact_flow"]["output_slots"]
        stt_slot = next(slot for slot in output_slots if slot.get("branch_id") == "branch-speech_to_text-1")
        text_artifact_slot = next(slot for slot in output_slots if slot.get("branch_id") == "branch-text_artifact-1")

        self.assertEqual(stt_slot["status"], "fulfilled")
        self.assertNotIn("artifact_path", stt_slot)
        self.assertEqual(text_artifact_slot["status"], "fulfilled")
        self.assertEqual(text_artifact_slot["artifact_ref"], payload["artifacts"][0]["artifact_ref"])
        self.assertEqual(text_artifact_slot["artifact_path"], payload["artifacts"][0]["path"])

    def test_terminal_output_branches_reconcile_with_fulfilled_output_slots(self):
        prompt = "Schreibe einen Satz. Lies ihn als Audio vor."
        graph = build_request_phase_graph(
            prompt,
            request_payload={"ghost_route": True, "prompt": prompt},
            route_payload={"capability": "chat", "route_source": "ghost_carried"},
        )
        payload = {
            "id": "resp_terminal_branch_projection",
            "status": "completed",
            "output_text": "Hallo lokale KI.",
            "saved_audio_path": "/tmp/artifacts/audio/generated.wav",
            "runtime": {"request_phase_graph": graph},
            "late_fill": {
                "status": "completed",
                "completed_branches": [
                    {
                        "branch_id": "branch-text_to_speech-1",
                        "phase_id": "phase-2",
                        "capability": "text_to_speech",
                    }
                ],
            },
            "output_branches": [
                {
                    "slot_id": "output-phase-2",
                    "branch_id": "branch-text_to_speech-1",
                    "phase_id": "phase-2",
                    "type": "audio",
                    "status": "pending",
                    "lifecycle": "deferred_output",
                    "placeholder_ref": "pending-output-branch-text_to_speech-1",
                }
            ],
        }
        payload["artifacts"] = build_canonical_response_artifacts(payload)

        frame = build_response_frame(payload, request_payload={"prompt": prompt})

        audio_slot = next(
            slot
            for slot in frame["planning"]["artifact_flow"]["output_slots"]
            if slot.get("branch_id") == "branch-text_to_speech-1"
        )
        audio_branch = next(
            branch
            for branch in frame["current_state"]["output_branches"]
            if branch.get("branch_id") == "branch-text_to_speech-1"
        )
        self.assertEqual(audio_slot["status"], "fulfilled")
        self.assertNotIn("placeholder_ref", audio_slot)
        self.assertEqual(audio_branch["status"], "fulfilled")
        self.assertEqual(audio_branch["lifecycle"], "materialized_output")
        self.assertEqual(audio_branch["artifact_ref"], audio_slot["artifact_ref"])
        self.assertNotIn("placeholder_ref", audio_branch)

    def test_direct_target_response_fulfills_matching_unique_follow_up_branch(self):
        graph = {
            "kind": "ollmo.request_phase_graph",
            "current_phase_id": "phase-1",
            "phases": [
                {
                    "phase_id": "phase-1",
                    "capability": "text_artifact",
                    "output_type": "document",
                    "status": "planned",
                    "obligation_id": "obligation-phase-1",
                },
                {
                    "phase_id": "phase-2",
                    "branch_id": "branch-speech_to_text-1",
                    "capability": "speech_to_text",
                    "output_type": "text",
                    "status": "planned",
                    "obligation_id": "obligation-phase-2",
                },
            ],
        }
        payload = {
            "id": "resp_direct_stt_projection",
            "status": "completed",
            "mode": "speech_to_text",
            "capability": "speech_to_text",
            "output_text": "Lokale KI bietet viel Datenschutz.",
            "saved_text_path": "/tmp/artifacts/transcripts/generated.md",
            "runtime": {"request_phase_graph": graph},
        }
        payload["artifacts"] = build_canonical_response_artifacts(payload)

        frame = build_response_frame(
            payload,
            request_payload={"prompt": "Transkribiere das zuletzt erzeugte Audio."},
        )

        stt_slot = next(
            slot
            for slot in frame["planning"]["artifact_flow"]["output_slots"]
            if slot.get("branch_id") == "branch-speech_to_text-1"
        )
        stt_branch = next(
            branch
            for branch in frame["current_state"]["output_branches"]
            if branch.get("branch_id") == "branch-speech_to_text-1"
        )
        self.assertEqual(stt_slot["status"], "fulfilled")
        self.assertEqual(stt_slot["value"], "Lokale KI bietet viel Datenschutz.")
        self.assertNotIn("placeholder_ref", stt_slot)
        self.assertEqual(stt_branch["status"], "fulfilled")
        self.assertEqual(stt_branch["lifecycle"], "materialized_output")
        self.assertNotIn("placeholder_ref", stt_branch)

    def test_truth_guard_clarification_blocks_deferred_public_branches(self):
        prompt = "Mach daraus ein HTML und ein Audio."
        clarification = (
            "I need the source/content for the referenced file before I can create that artifact. "
            "Please provide or select the HTML/code/content to materialize."
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={"ghost_route": True, "prompt": prompt},
            route_payload={"capability": "text_to_speech", "route_source": "ghost_carried"},
        )
        payload = {
            "id": "resp_truth_guard_projection",
            "status": "completed",
            "output_text": clarification,
            "runtime": {
                "request_phase_graph": graph,
                "truth_guard": {
                    "kind": "ungrounded_text_artifact_reference",
                    "status": "clarification_required",
                    "reason": "artifact request used a demonstrative reference without a current or selected source",
                },
            },
        }

        frame = build_response_frame(payload, request_payload={"prompt": prompt})

        self.assertEqual(frame["current_state"]["output_text"], clarification)
        self.assertEqual(frame["current_state"]["outputs"][0]["value"], clarification)
        audio_branch = next(
            branch
            for branch in frame["current_state"]["output_branches"]
            if branch.get("follow_up_capability") == "text_to_speech"
        )
        self.assertEqual(audio_branch["status"], "blocked")
        self.assertEqual(audio_branch["lifecycle"], "blocked_output")
        self.assertEqual(audio_branch["error_ref"]["code"], "UPSTREAM_CLARIFICATION_REQUIRED")
        self.assertNotIn("placeholder_ref", audio_branch)

    def test_truth_guard_repair_required_blocks_tts_branch_contract(self):
        prompt = "Liefere deine Antwort als Audio."
        repair_notice = (
            "I could not obtain substantive user-facing text for the requested audio."
        )
        graph = build_request_phase_graph(
            prompt,
            request_payload={"ghost_route": True, "prompt": prompt},
            route_payload={"capability": "chat", "route_source": "ghost_carried"},
        )
        payload = {
            "id": "resp_phase_output_repair_projection",
            "status": "completed",
            "lifecycle_state": "repair_needed",
            "output_text": repair_notice,
            "runtime": {
                "request_phase_graph": graph,
                "truth_guard": {
                    "kind": "ollmo.phase_output_acceptance_guard",
                    "status": "repair_required",
                    "reason": "control_envelope_not_speakable",
                },
            },
        }

        frame = build_response_frame(payload, request_payload={"prompt": prompt})

        self.assertEqual(frame["current_state"]["output_text"], repair_notice)
        audio_branch = next(
            branch
            for branch in frame["current_state"]["output_branches"]
            if branch.get("follow_up_capability") == "text_to_speech"
        )
        self.assertEqual(audio_branch["status"], "blocked")
        self.assertEqual(audio_branch["lifecycle"], "blocked_output")
        self.assertEqual(
            audio_branch["error_ref"]["code"],
            "UPSTREAM_REPAIR_REQUIRED",
        )
        self.assertEqual(
            audio_branch["recovery_context"]["suggested_action"],
            "repair_branch_contract",
        )
        self.assertNotIn("placeholder_ref", audio_branch)

    def test_document_artifact_slots_do_not_copy_root_text_value(self):
        outputs = build_canonical_outputs(
            {
                "output_text": "```html\n<h1>Hello</h1>\n```\n\n```css\nbody { color: red; }\n```",
            },
            output_slots=[
                {
                    "slot_id": "output-1",
                    "type": "document",
                    "status": "fulfilled",
                    "artifact_ref": "artifact:html",
                },
                {
                    "slot_id": "output-2",
                    "type": "document",
                    "status": "fulfilled",
                    "artifact_ref": "artifact:css",
                },
            ],
            artifacts=[
                {"type": "text", "path": "/tmp/index.html", "artifact_ref": "artifact:html"},
                {"type": "text", "path": "/tmp/styles.css", "artifact_ref": "artifact:css"},
            ],
        )

        self.assertEqual(len(outputs), 2)
        self.assertNotIn("value", outputs[0])
        self.assertNotIn("value", outputs[1])
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:html")
        self.assertEqual(outputs[1]["artifact_ref"], "artifact:css")
        self.assertNotIn("artifacts", outputs[0])
        self.assertNotIn("artifacts", outputs[1])

    def test_artifact_backed_text_output_hydrates_source_placeholder_from_saved_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.html"
            path.write_text("<!doctype html><h1>Fixed page</h1>", encoding="utf-8")

            outputs = build_canonical_outputs(
                {
                    "output_text": "Artifacts generated.",
                },
                output_slots=[
                    {
                        "slot_id": "output-html",
                        "branch_id": "branch-text_artifact-1",
                        "phase_id": "phase-2",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:index",
                        "value": "I need the source/content for the referenced file before I can create that artifact.",
                    },
                ],
                artifacts=[
                    {
                        "type": "text",
                        "path": str(path),
                        "artifact_ref": "artifact:index",
                    }
                ],
            )

            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0]["value"], "<!doctype html><h1>Fixed page</h1>")

    def test_generic_artifact_status_text_is_suppressed_when_real_artifact_outputs_exist(self):
        outputs = build_canonical_outputs(
            {},
            output_slots=[
                {
                    "slot_id": "output-text",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "value": "Artifacts generated.",
                },
                {
                    "slot_id": "output-image",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                    "type": "image",
                    "status": "fulfilled",
                    "artifact_ref": "artifact:image",
                },
            ],
            artifacts=[
                {
                    "type": "image",
                    "path": "/tmp/artifacts/images/image.png",
                    "artifact_ref": "artifact:image",
                }
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["type"], "image")
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:image")

    def test_generated_image_artifacts_preserve_registry_identity(self):
        artifacts = build_canonical_response_artifacts(
            {
                "id": "resp_image_identity",
                "saved_image_path": "/tmp/artifacts/images/generated.png",
                "artifact_id": "image_canonical_registry",
                "artifact_ref": "artifact:image_canonical_registry",
                "provenance_id": "generated_image_old_provider_id",
            }
        )

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["artifact_id"], "image_canonical_registry")
        self.assertEqual(artifacts[0]["artifact_ref"], "artifact:image_canonical_registry")

    def test_canonical_outputs_prefer_assigned_artifact_ref_over_slot_ref(self):
        outputs = build_canonical_outputs(
            {},
            output_slots=[
                {
                    "slot_id": "output-phase-2",
                    "type": "image",
                    "status": "fulfilled",
                    "artifact_ref": "artifact:image_generated_provider_id",
                }
            ],
            artifacts=[
                {
                    "type": "image",
                    "path": "/tmp/artifacts/images/generated.png",
                    "artifact_ref": "artifact:image_canonical_registry",
                }
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:image_canonical_registry")
        self.assertNotIn("artifacts", outputs[0])

    def test_response_frame_canonicalizes_duplicate_artifact_aliases_in_final_projection(self):
        frame = build_response_frame(
            {
                "id": "resp_duplicate_alias_projection",
                "status": "completed",
                "artifacts": [
                    {
                        "type": "image",
                        "path": "/tmp/generated/hero.png",
                        "artifact_ref": "artifact:hero",
                        "branch_id": "branch-image-1",
                        "batch_prompt": "hero image",
                    },
                    {
                        "type": "image",
                        "path": "/tmp/generated/hero.png",
                        "artifact_ref": "artifact:hero",
                        "branch_id": "branch-image-1",
                        "batch_prompt": "hero image alias",
                    },
                ],
            },
            request_payload={"prompt": "Create a page with a hero image."},
        )

        self.assertTrue(frame["output"]["artifact_identity"]["canonicalization_required"])
        self.assertFalse(frame["output"]["artifact_identity"]["final_projection_blocked"])
        self.assertEqual(len(frame["artifacts"]["output"]), 1)
        artifact_outputs = [
            output for output in frame["output"]["outputs"] if output.get("artifact_ref") == "artifact:hero"
        ]
        self.assertEqual(len(artifact_outputs), 1)
        self.assertIn(
            "hero image alias",
            frame["artifacts"]["output"][0]["alias_metadata"]["batch_prompts"],
        )

    def test_response_frame_keeps_conflicting_duplicate_artifact_ref_repair_needed(self):
        frame = build_response_frame(
            {
                "id": "resp_duplicate_conflict_projection",
                "status": "completed",
                "artifacts": [
                    {
                        "type": "image",
                        "path": "/tmp/generated/hero-a.png",
                        "artifact_ref": "artifact:hero",
                    },
                    {
                        "type": "image",
                        "path": "/tmp/generated/hero-b.png",
                        "artifact_ref": "artifact:hero",
                    },
                ],
            },
            request_payload={"prompt": "Create a page with a hero image."},
        )

        self.assertTrue(frame["output"]["artifact_identity"]["final_projection_blocked"])
        artifact_outputs = [
            output for output in frame["output"]["outputs"] if output.get("artifact_ref") == "artifact:hero"
        ]
        self.assertEqual(len(artifact_outputs), 1)
        self.assertEqual(artifact_outputs[0]["status"], "repair_needed")
        self.assertEqual(
            artifact_outputs[0]["blocked_reason"],
            "conflicting_duplicate_artifact_ref",
        )

    def test_response_frame_canonicalizes_document_and_text_alias_for_same_artifact_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.md"
            result_path.write_text("# Result\n", encoding="utf-8")
            frame = build_response_frame(
                {
                    "id": "resp_document_text_alias_projection",
                    "status": "completed",
                    "artifacts": [
                        {
                            "type": "text",
                            "path": str(result_path),
                            "artifact_ref": "artifact:result",
                            "branch_id": "branch-chat-1",
                        },
                        {
                            "type": "document",
                            "path": str(result_path),
                            "artifact_ref": "artifact:result",
                            "phase_id": "phase-chat-1",
                        },
                    ],
                    "output_slots": [
                        {
                            "slot_id": "output-chat-1",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-chat-1",
                            "type": "document",
                            "status": "fulfilled",
                            "lifecycle": "materialized_output",
                            "artifact_ref": "artifact:result",
                        }
                    ],
                },
                request_payload={"prompt": "Return the completed result."},
            )

            self.assertTrue(frame["output"]["artifact_identity"]["canonicalization_required"])
            self.assertFalse(frame["output"]["artifact_identity"]["final_projection_blocked"])
            self.assertEqual(len(frame["artifacts"]["output"]), 1)
            self.assertEqual(frame["artifacts"]["output"][0]["type"], "text")
            result_outputs = [
                output for output in frame["output"]["outputs"]
                if output.get("artifact_ref") == "artifact:result"
            ]
            self.assertEqual(len(result_outputs), 1)
            self.assertEqual(result_outputs[0]["type"], "document")
            self.assertEqual(result_outputs[0]["status"], "fulfilled")
            self.assertNotIn("blocked_reason", result_outputs[0])

    def test_response_frame_rebinds_duplicate_slot_refs_from_late_fill_branch_paths(self):
        stale_ref = "artifact:image_generated_provider_duplicate"
        image_paths = [
            "/tmp/generated/branch-8.png",
            "/tmp/generated/branch-9.png",
        ]
        frame = build_response_frame(
            {
                "id": "resp_duplicate_slot_rebind_from_late_fill",
                "status": "completed",
                "output_slots": [
                    {
                        "slot_id": "output-phase-9",
                        "branch_id": "branch-image_generation-8",
                        "phase_id": "phase-9",
                        "type": "image",
                        "status": "fulfilled",
                        "lifecycle": "materialized_output",
                        "artifact_ref": stale_ref,
                    },
                    {
                        "slot_id": "output-phase-10",
                        "branch_id": "branch-image_generation-9",
                        "phase_id": "phase-10",
                        "type": "image",
                        "status": "fulfilled",
                        "lifecycle": "materialized_output",
                        "artifact_ref": stale_ref,
                    },
                ],
                "artifacts": [
                    {
                        "type": "image",
                        "path": image_paths[0],
                        "artifact_ref": stale_ref,
                    },
                    {
                        "type": "image",
                        "path": image_paths[1],
                        "artifact_ref": stale_ref,
                    },
                ],
                "late_fill": {
                    "status": "completed",
                    "fill_results": [
                        {
                            "branch_id": "branch-image_generation-8",
                            "phase_id": "phase-9",
                            "capability": "image_generation",
                            "output_type": "image",
                            "status": "fulfilled",
                            "saved_image_path": image_paths[0],
                        },
                        {
                            "branch_id": "branch-image_generation-9",
                            "phase_id": "phase-10",
                            "capability": "image_generation",
                            "output_type": "image",
                            "status": "fulfilled",
                            "saved_image_path": image_paths[1],
                        },
                    ],
                },
            },
            request_payload={"prompt": "Generate two distinct image artifacts."},
        )

        image_slots = [
            slot
            for slot in frame["planning"]["artifact_flow"]["output_slots"]
            if slot.get("type") == "image"
        ]
        image_branches = [
            branch
            for branch in frame["current_state"]["output_branches"]
            if branch.get("type") == "image"
        ]
        image_outputs = [
            output
            for output in frame["output"]["outputs"]
            if output.get("type") == "image"
        ]

        self.assertEqual(len(image_slots), 2)
        self.assertEqual({slot.get("artifact_path") for slot in image_slots}, set(image_paths))
        self.assertEqual(len({slot.get("artifact_ref") for slot in image_slots}), 2)
        self.assertNotIn(stale_ref, {slot.get("artifact_ref") for slot in image_slots})
        self.assertEqual({branch.get("artifact_path") for branch in image_branches}, set(image_paths))
        self.assertEqual(len({branch.get("artifact_ref") for branch in image_branches}), 2)
        self.assertEqual(len({output.get("artifact_ref") for output in image_outputs}), 2)
        self.assertTrue(all(output.get("status") == "fulfilled" for output in image_outputs))
        self.assertFalse(frame["output"].get("artifact_identity", {}).get("final_projection_blocked"))

    def test_response_frame_public_artifacts_exclude_repair_intermediate_outputs(self):
        frame = build_response_frame(
            {
                "id": "resp_public_artifact_surface_hygiene",
                "status": "completed",
                "output_slots": [
                    {
                        "slot_id": "output-html",
                        "branch_id": "branch-text_artifact-1",
                        "phase_id": "phase-2",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:index",
                    },
                    {
                        "slot_id": "output-css",
                        "branch_id": "branch-text_artifact-2",
                        "phase_id": "phase-3",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:styles",
                    },
                    {
                        "slot_id": "output-repair",
                        "branch_id": "branch-repair-chat-1",
                        "phase_id": "phase-4",
                        "type": "text",
                        "status": "repair_needed",
                        "artifact_ref": "artifact:text_generated_image_bad",
                    },
                ],
                "artifacts": [
                    {"type": "text", "path": "/tmp/final/index.html", "artifact_ref": "artifact:index"},
                    {"type": "text", "path": "/tmp/final/styles.css", "artifact_ref": "artifact:styles"},
                    {
                        "type": "text",
                        "path": "/tmp/intermediate/generated-image-index.html",
                        "artifact_ref": "artifact:text_generated_image_bad",
                    },
                ],
            },
            request_payload={"prompt": "Create index.html and styles.css."},
        )

        public_refs = {
            artifact.get("artifact_ref")
            for artifact in frame["artifacts"]["output"]
        }
        self.assertEqual(public_refs, {"artifact:index", "artifact:styles"})
        dossier_refs = set(frame["artifacts"]["dossiers"].keys())
        self.assertIn("artifact:text_generated_image_bad", dossier_refs)

        recovered = response_payload_from_frame(frame)
        recovered_refs = {
            artifact.get("artifact_ref")
            for artifact in recovered["artifacts"]
        }
        self.assertEqual(recovered_refs, {"artifact:index", "artifact:styles"})

    def test_response_payload_from_frame_recanonicalizes_stale_stored_outputs(self):
        frame = {
            "response_id": "resp_hydrate_stale_outputs",
            "object": "response",
            "status": "completed",
            "current_state": {
                "id": "resp_hydrate_stale_outputs",
                "output_text": "Artifacts generated.",
                "outputs": [
                    {
                        "slot_id": "output-phase-1",
                        "branch_id": "phase-1",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:old-index",
                        "value": "Artifacts generated.",
                    },
                    {
                        "slot_id": "output-phase-7",
                        "branch_id": "branch-text_artifact-1",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:old-index",
                    },
                    {
                        "slot_id": "output-phase-7",
                        "branch_id": "branch-text_artifact-1",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:new-index",
                    },
                ],
            },
            "output": {
                "text": "Artifacts generated.",
                "outputs": [
                    {
                        "slot_id": "output-phase-1",
                        "branch_id": "phase-1",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:old-index",
                        "value": "Artifacts generated.",
                    },
                    {
                        "slot_id": "output-phase-7",
                        "branch_id": "branch-text_artifact-1",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:old-index",
                    },
                    {
                        "slot_id": "output-phase-7",
                        "branch_id": "branch-text_artifact-1",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:new-index",
                    },
                ],
            },
            "artifacts": {
                "output": [
                    {"type": "text", "path": "/tmp/final/old-index.html", "artifact_ref": "artifact:old-index"},
                ],
            },
            "planning": {
                "artifact_flow": {
                    "output_slots": [
                        {
                            "slot_id": "output-phase-1",
                            "branch_id": "phase-1",
                            "phase_id": "phase-1",
                            "type": "text",
                            "status": "fulfilled",
                        },
                        {
                            "slot_id": "output-phase-7",
                            "branch_id": "branch-text_artifact-1",
                            "phase_id": "phase-7",
                            "type": "text",
                            "status": "fulfilled",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "chat",
                            "artifact_ref": "artifact:old-index",
                        },
                        {
                            "slot_id": "output-phase-6",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-6",
                            "type": "text",
                            "status": "fulfilled",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "chat",
                        },
                        {
                            "slot_id": "output-phase-7",
                            "branch_id": "branch-text_artifact-1",
                            "phase_id": "phase-7",
                            "type": "text",
                            "status": "fulfilled",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "chat",
                            "artifact_ref": "artifact:new-index",
                        },
                    ]
                }
            },
        }

        recovered = response_payload_from_frame(frame)

        self.assertEqual(
            [output["slot_id"] for output in recovered["outputs"]],
            ["output-phase-7", "output-phase-6"],
        )
        self.assertEqual(recovered["outputs"][0]["artifact_ref"], "artifact:new-index")

    def test_response_payload_from_frame_preserves_distinct_repeated_chat_phase_text(self):
        source_text = "Ein einsamer Leuchtturm steht fest im tosenden Sturm."
        terminal_text = "Bild und Audio enthalten Leuchtturm und Sturm."
        source_output = {
            "slot_id": "output-phase-1",
            "branch_id": "phase-1",
            "phase_id": "phase-1",
            "type": "text",
            "status": "fulfilled",
            "value": source_text,
        }
        terminal_output = {
            "slot_id": "output-phase-6",
            "branch_id": "branch-chat-1",
            "phase_id": "phase-6",
            "type": "text",
            "status": "fulfilled",
            "value": terminal_text,
        }
        frame = {
            "response_id": "resp_repeated_chat_phase_text",
            "object": "response",
            "status": "completed",
            "current_state": {
                "id": "resp_repeated_chat_phase_text",
                "status": "completed",
                "output_text": terminal_text,
                "outputs": [source_output, terminal_output],
                "late_fill": {
                    "status": "completed",
                    "fill_results": [
                        {
                            "slot_id": "output-phase-6",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-6",
                            "capability": "chat",
                            "output_type": "text",
                            "result_text": terminal_text,
                        }
                    ],
                },
            },
            "output": {
                "text": terminal_text,
                "outputs": [source_output, terminal_output],
            },
            "planning": {
                "artifact_flow": {
                    "output_slots": [
                        {
                            "slot_id": "output-phase-1",
                            "branch_id": "phase-1",
                            "phase_id": "phase-1",
                            "type": "text",
                            "status": "fulfilled",
                        },
                        {
                            "slot_id": "output-phase-6",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-6",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "chat",
                            "type": "text",
                            "status": "fulfilled",
                        },
                    ]
                }
            },
        }

        recovered = response_payload_from_frame(frame)
        outputs_by_phase = {
            output["phase_id"]: output
            for output in recovered["outputs"]
        }

        self.assertEqual(recovered["output_text"], terminal_text)
        self.assertEqual(outputs_by_phase["phase-1"]["value"], source_text)
        self.assertEqual(outputs_by_phase["phase-6"]["value"], terminal_text)

    def test_response_payload_from_frame_prefers_canonical_output_over_stale_planning_slot_text(self):
        source_text = "Ein einsamer Leuchtturm steht fest im tosenden Sturm."
        terminal_text = "Bild und Audio enthalten Leuchtturm und Sturm."
        source_output = {
            "slot_id": "output-phase-1",
            "branch_id": "phase-1",
            "phase_id": "phase-1",
            "type": "text",
            "status": "fulfilled",
            "value": source_text,
        }
        terminal_output = {
            "slot_id": "output-phase-6",
            "branch_id": "branch-chat-1",
            "phase_id": "phase-6",
            "type": "text",
            "status": "fulfilled",
            "value": terminal_text,
        }
        frame = {
            "response_id": "resp_stale_planning_slot_text",
            "object": "response",
            "status": "completed",
            "current_state": {
                "id": "resp_stale_planning_slot_text",
                "status": "completed",
                "output_text": terminal_text,
                "outputs": [source_output, terminal_output],
            },
            "output": {
                "text": terminal_text,
                "outputs": [source_output, terminal_output],
            },
            "planning": {
                "artifact_flow": {
                    "output_slots": [
                        {
                            "slot_id": "output-phase-1",
                            "branch_id": "phase-1",
                            "phase_id": "phase-1",
                            "type": "text",
                            "status": "fulfilled",
                            "value": terminal_text,
                        },
                        {
                            "slot_id": "output-phase-6",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-6",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "chat",
                            "type": "text",
                            "status": "fulfilled",
                            "value": terminal_text,
                        },
                    ]
                }
            },
        }

        recovered = response_payload_from_frame(frame)
        outputs_by_phase = {
            output["phase_id"]: output
            for output in recovered["outputs"]
        }

        self.assertEqual(outputs_by_phase["phase-1"]["value"], source_text)
        self.assertEqual(outputs_by_phase["phase-6"]["value"], terminal_text)

    def test_response_frame_public_artifacts_include_linked_html_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / "artifacts" / "documents"
            image_dir = root / "artifacts" / "images"
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            index_path = document_dir / "index.html"
            styles_path = document_dir / "styles.css"
            repair_path = document_dir / "repair-index.html"
            exterior_path = image_dir / "exterior.png"
            interior_path = image_dir / "interior.png"
            detail_path = image_dir / "detail.png"
            index_path.write_text(
                '<!doctype html>'
                '<link rel="stylesheet" href="styles.css">'
                '<section style="background-image: url(../images/exterior.png)"></section>'
                '<img src="../images/interior.png">'
                '<img src="../images/detail.png">',
                encoding="utf-8",
            )
            styles_path.write_text("body { color: #111; }", encoding="utf-8")
            repair_path.write_text("<!doctype html><p>repair scratch</p>", encoding="utf-8")
            exterior_path.write_bytes(b"exterior")
            interior_path.write_bytes(b"interior")
            detail_path.write_bytes(b"detail")

            frame = build_response_frame(
                {
                    "id": "resp_public_linked_deps",
                    "status": "completed",
                    "output_slots": [
                        {
                            "slot_id": "output-html",
                            "branch_id": "branch-text_artifact-1",
                            "phase_id": "phase-2",
                            "type": "text",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:index",
                        },
                        {
                            "slot_id": "output-css",
                            "branch_id": "branch-text_artifact-2",
                            "phase_id": "phase-3",
                            "type": "text",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:styles",
                        },
                        {
                            "slot_id": "output-image-1",
                            "branch_id": "branch-image_generation-1",
                            "phase_id": "phase-4",
                            "type": "image",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:image_slot_1",
                        },
                        {
                            "slot_id": "output-repair",
                            "branch_id": "repair-chat",
                            "phase_id": "repair-chat",
                            "type": "text",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:text_generated_image_bad",
                        },
                    ],
                    "artifacts": [
                        {"type": "text", "path": str(index_path), "artifact_ref": "artifact:index"},
                        {"type": "text", "path": str(styles_path), "artifact_ref": "artifact:styles"},
                        {"type": "image", "path": str(detail_path), "artifact_ref": "artifact:image_slot_1"},
                        {
                            "type": "text",
                            "path": str(repair_path),
                            "artifact_ref": "artifact:text_generated_image_bad",
                        },
                    ],
                },
                request_payload={"prompt": "Create a one-page landhouse site with three images."},
            )

            output_artifacts = frame["artifacts"]["output"]
            public_paths = {Path(artifact["path"]).name for artifact in output_artifacts}
            public_refs = {artifact.get("artifact_ref") for artifact in output_artifacts}
            self.assertEqual(
                public_paths,
                {"index.html", "styles.css", "detail.png", "exterior.png", "interior.png"},
            )
            self.assertNotIn("artifact:text_generated_image_bad", public_refs)

            recovered = response_payload_from_frame(frame)
            recovered_paths = {Path(artifact["path"]).name for artifact in recovered["artifacts"]}
            self.assertEqual(
                recovered_paths,
                {"index.html", "styles.css", "detail.png", "exterior.png", "interior.png"},
            )

    def test_response_frame_promotes_late_fill_saved_text_artifacts_with_images(self):
        graph = {
            "kind": "ollmo.request_phase_graph",
            "current_phase_id": "phase-1",
            "phases": [
                {
                    "phase_id": "phase-1",
                    "capability": "chat",
                    "output_type": "document",
                    "status": "completed",
                },
                {
                    "phase_id": "phase-2",
                    "branch_id": "branch-text_artifact-1",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "planned",
                    "requires_artifact": True,
                    "role": "text_artifact_output",
                    "fulfillment_policy": "runtime_text_artifact",
                    "text_artifact_extension": "html",
                    "text_artifact_source_name": "index",
                    "artifact_request": {"extension": "html", "source_name": "index"},
                },
                {
                    "phase_id": "phase-3",
                    "branch_id": "branch-text_artifact-2",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "planned",
                    "requires_artifact": True,
                    "role": "text_artifact_output",
                    "fulfillment_policy": "runtime_text_artifact",
                    "text_artifact_extension": "css",
                    "text_artifact_source_name": "styles",
                    "artifact_request": {"extension": "css", "source_name": "styles"},
                },
                {
                    "phase_id": "phase-4",
                    "branch_id": "branch-image_generation-1",
                    "capability": "image_generation",
                    "output_type": "image",
                    "status": "planned",
                },
                {
                    "phase_id": "phase-5",
                    "branch_id": "branch-image_generation-2",
                    "capability": "image_generation",
                    "output_type": "image",
                    "status": "planned",
                },
                {
                    "phase_id": "phase-6",
                    "branch_id": "branch-image_generation-3",
                    "capability": "image_generation",
                    "output_type": "image",
                    "status": "planned",
                },
            ],
        }
        index_path = "/tmp/landing/index.html"
        styles_path = "/tmp/landing/styles.css"

        frame = build_response_frame(
            {
                "id": "resp_late_fill_text_artifacts_with_images",
                "object": "response",
                "status": "completed",
                "capability": "chat",
                "output_text": "<!doctype html><html><head></head><body>Landing page</body></html>",
                "runtime": {"request_phase_graph": graph},
                "artifacts": [
                    {"type": "image", "path": "/tmp/landing/hero.png", "artifact_ref": "artifact:hero"},
                    {"type": "image", "path": "/tmp/landing/card.png", "artifact_ref": "artifact:card"},
                    {"type": "image", "path": "/tmp/landing/detail.png", "artifact_ref": "artifact:detail"},
                ],
                "late_fill": {
                    "status": "partial_failed",
                    "completed_branches": [
                        {
                            "branch_id": "branch-text_artifact-1",
                            "phase_id": "phase-2",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                            "saved_text_path": index_path,
                            "text_artifact_extension": "html",
                            "text_artifact_source_name": "index",
                            "artifact_request": {"extension": "html", "source_name": "index"},
                        },
                        {
                            "branch_id": "branch-text_artifact-2",
                            "phase_id": "phase-3",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                            "saved_text_path": styles_path,
                            "text_artifact_extension": "css",
                            "text_artifact_source_name": "styles",
                            "artifact_request": {"extension": "css", "source_name": "styles"},
                        },
                        {
                            "branch_id": "branch-image_generation-1",
                            "phase_id": "phase-4",
                            "capability": "image_generation",
                            "output_type": "image",
                            "status": "fulfilled",
                            "saved_image_path": "/tmp/landing/hero.png",
                        },
                        {
                            "branch_id": "branch-image_generation-2",
                            "phase_id": "phase-5",
                            "capability": "image_generation",
                            "output_type": "image",
                            "status": "fulfilled",
                            "saved_image_path": "/tmp/landing/card.png",
                        },
                        {
                            "branch_id": "branch-image_generation-3",
                            "phase_id": "phase-6",
                            "capability": "image_generation",
                            "output_type": "image",
                            "status": "fulfilled",
                            "saved_image_path": "/tmp/landing/detail.png",
                        },
                    ],
                    "fill_results": [
                        {
                            "branch_id": "branch-text_artifact-1",
                            "phase_id": "phase-2",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                            "saved_text_path": index_path,
                            "text_artifact_extension": "html",
                            "text_artifact_source_name": "index",
                            "artifact_request": {"extension": "html", "source_name": "index"},
                        },
                        {
                            "branch_id": "branch-text_artifact-2",
                            "phase_id": "phase-3",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                            "saved_text_path": styles_path,
                            "text_artifact_extension": "css",
                            "text_artifact_source_name": "styles",
                            "artifact_request": {"extension": "css", "source_name": "styles"},
                        },
                    ],
                },
            },
            request_payload={"prompt": "Create a landing page with index.html, styles.css, and three images."},
        )

        public_paths = {artifact.get("path") for artifact in frame["artifacts"]["output"]}
        self.assertEqual(
            public_paths,
            {
                index_path,
                styles_path,
                "/tmp/landing/hero.png",
                "/tmp/landing/card.png",
                "/tmp/landing/detail.png",
            },
        )
        text_slots = [
            slot
            for slot in frame["planning"]["artifact_flow"]["output_slots"]
            if slot.get("branch_id") in {"branch-text_artifact-1", "branch-text_artifact-2"}
        ]
        self.assertEqual(len(text_slots), 2)
        self.assertTrue(all(slot.get("status") == "fulfilled" for slot in text_slots))
        self.assertEqual({slot.get("artifact_path") for slot in text_slots}, {index_path, styles_path})
        self.assertTrue(all(str(slot.get("artifact_ref") or "").startswith("artifact:text_") for slot in text_slots))
        self.assertTrue(
            all("text_generated_image" not in str(slot.get("artifact_ref") or "") for slot in text_slots)
        )
        text_branches = [
            branch
            for branch in frame["current_state"]["output_branches"]
            if branch.get("branch_id") in {"branch-text_artifact-1", "branch-text_artifact-2"}
        ]
        self.assertEqual({branch.get("artifact_path") for branch in text_branches}, {index_path, styles_path})

    def test_response_frame_recovers_late_fill_saved_images_for_final_projection(self):
        graph = {
            "kind": "ollmo.request_phase_graph",
            "current_phase_id": "phase-1",
            "phases": [
                {
                    "phase_id": "phase-1",
                    "capability": "chat",
                    "output_type": "document",
                    "status": "completed",
                },
                *[
                    {
                        "phase_id": f"phase-{index + 1}",
                        "branch_id": f"branch-image_generation-{index}",
                        "capability": "image_generation",
                        "output_type": "image",
                        "status": "planned",
                    }
                    for index in range(1, 5)
                ],
                {
                    "phase_id": "phase-6",
                    "branch_id": "branch-text_artifact-1",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "planned",
                    "role": "text_artifact_output",
                    "fulfillment_policy": "runtime_text_artifact",
                    "text_artifact_extension": "html",
                    "text_artifact_source_name": "index",
                    "artifact_request": {"extension": "html", "source_name": "index"},
                },
                {
                    "phase_id": "phase-7",
                    "branch_id": "branch-text_artifact-2",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "planned",
                    "role": "text_artifact_output",
                    "fulfillment_policy": "runtime_text_artifact",
                    "text_artifact_extension": "css",
                    "text_artifact_source_name": "styles",
                    "artifact_request": {"extension": "css", "source_name": "styles"},
                },
            ],
        }
        image_paths = [
            "/tmp/petsie/selfie-1.png",
            "/tmp/petsie/selfie-2.png",
            "/tmp/petsie/selfie-3.png",
            "/tmp/petsie/selfie-4.png",
        ]
        late_fill_results = [
            {
                "branch_id": f"branch-image_generation-{index}",
                "phase_id": f"phase-{index + 1}",
                "capability": "image_generation",
                "output_type": "image",
                "status": "fulfilled",
                "saved_image_path": image_path,
            }
            for index, image_path in enumerate(image_paths, start=1)
        ]
        late_fill_results.extend(
            [
                {
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-6",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "fulfilled",
                    "saved_text_path": "/tmp/petsie/index.html",
                    "text_artifact_extension": "html",
                    "text_artifact_source_name": "index",
                    "artifact_request": {"extension": "html", "source_name": "index"},
                },
                {
                    "branch_id": "branch-text_artifact-2",
                    "phase_id": "phase-7",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "fulfilled",
                    "saved_text_path": "/tmp/petsie/styles.css",
                    "text_artifact_extension": "css",
                    "text_artifact_source_name": "styles",
                    "artifact_request": {"extension": "css", "source_name": "styles"},
                },
            ]
        )

        frame = build_response_frame(
            {
                "id": "resp_late_fill_saved_images_final_projection",
                "object": "response",
                "status": "completed",
                "capability": "chat",
                "output_text": "Artifacts generated.",
                "runtime": {"request_phase_graph": graph},
                "saved_image_path": image_paths[-1],
                "provenance_id": "generated_image_stale_top_level",
                "artifacts": [
                    {
                        "type": "image",
                        "path": image_paths[-1],
                        "artifact_ref": "artifact:image_stale_top_level",
                    }
                ],
                "late_fill": {
                    "status": "completed",
                    "completed_branches": late_fill_results,
                    "fill_results": late_fill_results,
                },
            },
            request_payload={"prompt": "Create a Petsie landing page with exactly four generated images."},
        )

        image_artifacts = [
            artifact
            for artifact in frame["artifacts"]["output"]
            if artifact.get("type") == "image"
        ]
        image_outputs = [
            output
            for output in frame["output"]["outputs"]
            if output.get("type") == "image"
        ]

        self.assertEqual({artifact.get("path") for artifact in image_artifacts}, set(image_paths))
        self.assertEqual(len({output.get("artifact_ref") for output in image_outputs}), 4)
        self.assertTrue(all(output.get("artifact_ref") for output in image_outputs))

    def test_canonical_outputs_use_late_fill_text_for_downstream_text_slot(self):
        outputs = build_canonical_outputs(
            {
                "output_text": "Phase one preparation.",
                "late_fill": {
                    "fill_results": [
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-3",
                            "capability": "chat",
                            "content_payload": "Final caption.",
                        }
                    ]
                },
            },
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-phase-3"],
                },
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                },
            ],
        )

        self.assertEqual(outputs[0]["value"], "Phase one preparation.")
        self.assertEqual(outputs[1]["value"], "Final caption.")
        self.assertEqual(outputs[0]["source"], "promoted_output_slot")
        self.assertFalse(outputs[0]["compatibility_derived"])

    def test_live_canonical_outputs_prefer_fresh_late_fill_text_over_stale_existing_output(self):
        outputs = build_canonical_outputs(
            {
                "output_text": "Phase one preparation.",
                "outputs": [
                    {
                        "slot_id": "output-phase-3",
                        "branch_id": "branch-chat-1",
                        "phase_id": "phase-3",
                        "type": "text",
                        "status": "fulfilled",
                        "value": "Stale draft caption.",
                    }
                ],
                "late_fill": {
                    "fill_results": [
                        {
                            "slot_id": "output-phase-3",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-3",
                            "capability": "chat",
                            "result_text": "Fresh terminal caption.",
                        }
                    ]
                },
            },
            output_slots=[
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                }
            ],
        )

        self.assertEqual(outputs[0]["value"], "Fresh terminal caption.")

    def test_response_frame_public_text_prefers_final_downstream_output(self):
        frame = build_response_frame(
            {
                "id": "resp_final_downstream_text",
                "object": "response",
                "status": "completed",
                "model": "gemma4:26b",
                "backend": "ollama",
                "capability": "chat",
                "output_text": "A yellow notebook image prompt.",
                "output": [
                    {
                        "id": "msg_phase_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "A yellow notebook image prompt.",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "work_tree": {
                    "kind": "ollmo.work_tree",
                    "work_tree_source": "runtime_owned",
                    "authoritative": True,
                    "status": "tracked",
                    "root_node_id": "node-request",
                    "node_order": ["node-request", "node-phase-1", "node-phase-4"],
                    "output_root_node_ids": ["node-phase-1"],
                    "nodes": [
                        {
                            "node_id": "node-request",
                            "kind": "request",
                            "role": "request_root",
                            "status": "completed",
                            "child_node_ids": ["node-phase-1"],
                        },
                        {
                            "node_id": "node-phase-1",
                            "kind": "output",
                            "slot_id": "output-phase-1",
                            "type": "text",
                            "status": "fulfilled",
                            "phase_id": "phase-1",
                            "branch_id": "phase-1",
                            "value": "A yellow notebook image prompt.",
                            "parent_node_id": "node-request",
                            "child_node_ids": ["node-phase-4"],
                        },
                        {
                            "node_id": "node-phase-4",
                            "kind": "output",
                            "slot_id": "output-phase-4",
                            "type": "text",
                            "status": "fulfilled",
                            "phase_id": "phase-4",
                            "branch_id": "branch-chat-1",
                            "value": "Das Bild zeigt ein gelbes Notizbuch mit schwarzer Aufschrift Plan A.",
                            "parent_node_id": "node-phase-1",
                        },
                    ],
                },
                "late_fill": {
                    "status": "completed",
                    "fill_results": [
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-4",
                            "capability": "chat",
                            "result_text": "Das Bild zeigt ein gelbes Notizbuch mit schwarzer Aufschrift Plan A.",
                        }
                    ],
                },
            },
            request_payload={
                "prompt": "Erstelle zuerst ein Bild von einem gelben Notizbuch. Analysiere danach das Bild.",
            },
        )

        final_text = "Das Bild zeigt ein gelbes Notizbuch mit schwarzer Aufschrift Plan A."
        self.assertEqual(frame["output"]["text"], final_text)
        self.assertEqual(frame["current_state"]["output_text"], final_text)
        self.assertEqual(frame["current_state"]["output"][0]["content"][0]["text"], final_text)
        self.assertEqual(frame["output"]["outputs"][-1]["value"], final_text)

    def test_response_frame_public_text_prefers_terminal_chat_join_backed_by_text_artifact(self):
        final_rows = [
            {"label": "A", "artifact_ref": "artifact:image-a", "visible_evidence": "red lighthouse in snow"},
            {"label": "B", "artifact_ref": "artifact:image-b", "visible_evidence": "green library at night"},
            {"label": "C", "artifact_ref": "artifact:image-c", "visible_evidence": "blue greenhouse in rain"},
        ]
        final_text = json.dumps(final_rows, ensure_ascii=False, indent=2)
        prior_vision_text = json.dumps([final_rows[-1]], ensure_ascii=False, indent=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "joined-evidence.json"
            final_path.write_text(final_text, encoding="utf-8")
            frame = build_response_frame(
                {
                    "id": "resp_terminal_chat_join_text_artifact",
                    "object": "response",
                    "status": "completed",
                    "model": "local-chat",
                    "backend": "local",
                    "capability": "chat",
                    "output_text": prior_vision_text,
                    "output_slots": [
                        {
                            "slot_id": "output-phase-7",
                            "branch_id": "branch-vision_analysis-3",
                            "phase_id": "phase-7",
                            "type": "text",
                            "status": "fulfilled",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "vision_analysis",
                        },
                        {
                            "slot_id": "output-phase-8",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                            "type": "text",
                            "status": "fulfilled",
                            "parent_slot_id": "output-phase-1",
                            "follow_up_capability": "chat",
                            "artifact_ref": "artifact:joined-json",
                        },
                    ],
                    "artifacts": [
                        {
                            "type": "text",
                            "path": str(final_path),
                            "artifact_ref": "artifact:joined-json",
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                            "extension": "json",
                            "mime_type": "application/json",
                        }
                    ],
                    "late_fill": {
                        "status": "completed",
                        "final_materialization_contract_status": "fulfilled",
                        "pending_branches": [],
                        "active_branches": [],
                        "completed_branches": [
                            {
                                "branch_id": "branch-vision_analysis-3",
                                "phase_id": "phase-7",
                                "capability": "vision_analysis",
                                "status": "fulfilled",
                            },
                            {
                                "branch_id": "branch-chat-1",
                                "phase_id": "phase-8",
                                "capability": "chat",
                                "status": "fulfilled",
                            },
                        ],
                        "fill_results": [
                            {
                                "branch_id": "branch-vision_analysis-3",
                                "phase_id": "phase-7",
                                "capability": "vision_analysis",
                                "result_text": prior_vision_text,
                            },
                            {
                                "branch_id": "branch-chat-1",
                                "phase_id": "phase-8",
                                "capability": "chat",
                                "result_text": final_text,
                                "saved_text_path": str(final_path),
                                "execution_contract": {
                                    "role": "post_artifact_text_follow_up",
                                    "depends_on": ["phase-5", "phase-6", "phase-7"],
                                },
                            },
                        ],
                    },
                },
                request_payload={"prompt": "Return one final JSON list for A, B, and C."},
            )

        self.assertEqual(frame["output"]["text"], final_text)
        self.assertEqual(frame["current_state"]["output_text"], final_text)
        self.assertEqual(frame["current_state"]["output"][0]["content"][0]["text"], final_text)
        self.assertEqual(frame["output"]["outputs"][-1]["artifact_ref"], "artifact:joined-json")
        self.assertEqual(frame["output"]["outputs"][-1]["value"], final_text)

    def test_public_text_recovers_terminal_chat_role_from_phase_graph(self):
        outputs = [
            {
                "slot_id": "output-phase-7",
                "branch_id": "branch-vision_analysis-3",
                "phase_id": "phase-7",
                "type": "text",
                "status": "fulfilled",
                "value": "C-only evidence",
            },
            {
                "slot_id": "output-phase-8",
                "branch_id": "branch-chat-1",
                "phase_id": "phase-8",
                "type": "text",
                "status": "fulfilled",
                "artifact_ref": "artifact:joined-json",
                "value": "A/B/C joined evidence",
            },
        ]
        payload = {
            "runtime": {
                "request_phase_graph": {
                    "phases": [
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                            "capability": "chat",
                            "role": "post_artifact_text_follow_up",
                        }
                    ]
                }
            }
        }

        self.assertEqual(
            select_public_output_text(payload, outputs),
            "A/B/C joined evidence",
        )

    def test_public_text_keeps_generic_artifact_backed_chat_materialization_hidden(self):
        outputs = [
            {
                "slot_id": "output-phase-7",
                "branch_id": "branch-vision_analysis-3",
                "phase_id": "phase-7",
                "type": "text",
                "status": "fulfilled",
                "value": "Public evidence summary",
            },
            {
                "slot_id": "output-phase-8",
                "branch_id": "branch-chat-1",
                "phase_id": "phase-8",
                "type": "text",
                "status": "fulfilled",
                "follow_up_capability": "chat",
                "artifact_ref": "artifact:internal-document",
                "value": "Internal document materialization",
            },
        ]
        payload = {
            "late_fill": {
                "fill_results": [
                    {
                        "branch_id": "branch-chat-1",
                        "phase_id": "phase-8",
                        "capability": "chat",
                        "role": "document_output",
                    }
                ]
            }
        }

        self.assertEqual(
            select_public_output_text(payload, outputs),
            "Public evidence summary",
        )

    def test_response_frame_reserves_explicit_terminal_chat_artifact_from_prior_text_slots(self):
        preparation_text = "Prepared three image prompts for the downstream branches."
        prior_vision_text = "Visible evidence for image C only."
        final_text = json.dumps(
            [
                {"label": "A", "artifact_ref": "artifact:image-a", "visible_evidence": "red lighthouse"},
                {"label": "B", "artifact_ref": "artifact:image-b", "visible_evidence": "green library"},
                {"label": "C", "artifact_ref": "artifact:image-c", "visible_evidence": "blue greenhouse"},
            ],
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "joined-evidence.json"
            final_path.write_text(final_text, encoding="utf-8")
            phase_graph = {
                "kind": "ollmo.request_phase_graph",
                "graph_version": 2,
                "mode": "phase_chain",
                "current_phase_id": "phase-1",
                "current_phase_capability": "chat",
                "phases": [
                    {
                        "phase_id": "phase-1",
                        "capability": "chat",
                        "output_type": "text",
                        "status": "completed",
                        "role": "text_preparation",
                    },
                    {
                        "phase_id": "phase-7",
                        "branch_id": "branch-vision_analysis-3",
                        "capability": "vision_analysis",
                        "output_type": "text",
                        "status": "completed",
                        "role": "vision_analysis_follow_up",
                        "depends_on": ["phase-4"],
                    },
                    {
                        "phase_id": "phase-8",
                        "branch_id": "branch-chat-1",
                        "capability": "chat",
                        "output_type": "text",
                        "status": "completed",
                        "role": "post_artifact_text_follow_up",
                        "depends_on": ["phase-5", "phase-6", "phase-7"],
                    },
                ],
                "downstream_branches": [
                    {
                        "phase_id": "phase-7",
                        "branch_id": "branch-vision_analysis-3",
                        "capability": "vision_analysis",
                        "output_type": "text",
                        "status": "completed",
                        "role": "vision_analysis_follow_up",
                        "depends_on": ["phase-4"],
                    },
                    {
                        "phase_id": "phase-8",
                        "branch_id": "branch-chat-1",
                        "capability": "chat",
                        "output_type": "text",
                        "status": "completed",
                        "role": "post_artifact_text_follow_up",
                        "depends_on": ["phase-5", "phase-6", "phase-7"],
                    }
                ],
            }
            frame = build_response_frame(
                {
                    "id": "resp_terminal_chat_artifact_owner",
                    "object": "response",
                    "status": "completed",
                    "capability": "chat",
                    "output_text": preparation_text,
                    "runtime": {"request_phase_graph": phase_graph},
                    "late_fill": {
                        "status": "completed",
                        "completed_branches": [
                            {
                                "phase_id": "phase-7",
                                "branch_id": "branch-vision_analysis-3",
                                "capability": "vision_analysis",
                                "output_type": "text",
                                "status": "fulfilled",
                            },
                            {
                                "phase_id": "phase-8",
                                "branch_id": "branch-chat-1",
                                "capability": "chat",
                                "output_type": "text",
                                "status": "fulfilled",
                            }
                        ],
                        "fill_results": [
                            {
                                "phase_id": "phase-7",
                                "branch_id": "branch-vision_analysis-3",
                                "capability": "vision_analysis",
                                "output_type": "text",
                                "result_text": prior_vision_text,
                                "content_payload": prior_vision_text,
                                "role": "vision_analysis_follow_up",
                            },
                            {
                                "phase_id": "phase-8",
                                "branch_id": "branch-chat-1",
                                "capability": "chat",
                                "output_type": "text",
                                "result_text": final_text,
                                "content_payload": final_text,
                                "saved_text_path": str(final_path),
                                "role": "post_artifact_text_follow_up",
                            }
                        ],
                    },
                    "artifacts": [
                        {
                            "type": "text",
                            "path": str(final_path),
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                        }
                    ],
                },
                request_payload={"prompt": "Return one final JSON join for A, B, and C."},
            )

        outputs = frame["output"]["outputs"]
        root_output = next(item for item in outputs if item.get("phase_id") == "phase-1")
        prior_vision_output = next(item for item in outputs if item.get("phase_id") == "phase-7")
        terminal_output = next(item for item in outputs if item.get("phase_id") == "phase-8")
        terminal_ref = terminal_output["artifact_ref"]
        self.assertNotIn("artifact_ref", root_output)
        self.assertNotIn("artifact_ref", prior_vision_output)
        self.assertEqual(root_output["value"], preparation_text)
        self.assertEqual(prior_vision_output["value"], prior_vision_text)
        self.assertEqual(terminal_output["value"], final_text)
        self.assertEqual(
            [item.get("artifact_ref") for item in outputs].count(terminal_ref),
            1,
            outputs,
        )
        for projection in (
            frame["planning"]["artifact_flow"]["output_slots"],
            frame["current_state"]["output_slots"],
            frame["current_state"]["outputs"],
        ):
            self.assertEqual(
                [item.get("artifact_ref") for item in projection].count(terminal_ref),
                1,
                projection,
            )
            prior_projection = next(item for item in projection if item.get("phase_id") == "phase-7")
            self.assertNotIn("artifact_ref", prior_projection)

    def test_response_frame_reprojects_stale_non_owner_ref_to_exact_late_fill_owner_once(self):
        prior_vision_text = "Visible evidence for the third image only."
        final_text = '[{"label":"A"},{"label":"B"},{"label":"C"}]'
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "joined-evidence.json"
            final_path.write_text(final_text, encoding="utf-8")
            phase_graph = {
                "kind": "ollmo.request_phase_graph",
                "graph_version": 2,
                "mode": "phase_chain",
                "current_phase_id": "phase-1",
                "current_phase_capability": "chat",
                "phases": [
                    {
                        "phase_id": "phase-1",
                        "capability": "chat",
                        "output_type": "text",
                        "status": "completed",
                    },
                    {
                        "phase_id": "phase-5",
                        "branch_id": "branch-vision_analysis-1",
                        "capability": "vision_analysis",
                        "output_type": "text",
                        "status": "completed",
                        "depends_on": ["phase-2"],
                    },
                    {
                        "phase_id": "phase-8",
                        "branch_id": "branch-chat-1",
                        "capability": "chat",
                        "output_type": "text",
                        "status": "completed",
                        "depends_on": ["phase-5"],
                    },
                ],
            }
            payload = {
                "id": "resp_stale_bound_ref_reprojection",
                "object": "response",
                "status": "completed",
                "capability": "chat",
                "output_text": "Prepared the downstream work.",
                "runtime": {"request_phase_graph": phase_graph},
                # These durable slots intentionally reproduce the stale D3R2
                # projection: both branches claim the one phase-8 artifact.
                "output_slots": [
                    {
                        "slot_id": "output-phase-5",
                        "branch_id": "branch-vision_analysis-1",
                        "phase_id": "phase-5",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:joined-json",
                        "artifact_path": str(final_path),
                        "value": prior_vision_text,
                    },
                    {
                        "slot_id": "output-phase-8",
                        "branch_id": "branch-chat-1",
                        "phase_id": "phase-8",
                        "type": "text",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:joined-json",
                        "artifact_path": str(final_path),
                        "value": final_text,
                    },
                    {
                        "slot_id": "output-phase-legacy-audio",
                        "branch_id": "branch-legacy-audio",
                        "phase_id": "phase-legacy-audio",
                        "type": "audio",
                        "status": "fulfilled",
                        "artifact_ref": "artifact:legacy-unbound",
                        "artifact_path": "/tmp/legacy-unbound.wav",
                    },
                ],
                "artifacts": [
                    {
                        "type": "text",
                        "path": str(final_path),
                        "artifact_ref": "artifact:joined-json",
                        "branch_id": "branch-chat-1",
                        "phase_id": "phase-8",
                    },
                    {
                        "type": "audio",
                        "path": "/tmp/legacy-unbound.wav",
                        "artifact_ref": "artifact:legacy-unbound",
                    },
                ],
                "late_fill": {
                    "status": "completed",
                    "completed_branches": [
                        {
                            "branch_id": "branch-vision_analysis-1",
                            "phase_id": "phase-5",
                            "capability": "vision_analysis",
                            "output_type": "text",
                            "status": "fulfilled",
                        },
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                        },
                    ],
                    "fill_results": [
                        {
                            "branch_id": "branch-vision_analysis-1",
                            "phase_id": "phase-5",
                            "capability": "vision_analysis",
                            "output_type": "text",
                            "result_text": prior_vision_text,
                        },
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                            "capability": "chat",
                            "output_type": "text",
                            "result_text": final_text,
                            "saved_text_path": str(final_path),
                        },
                    ],
                },
            }

            frame = build_response_frame(
                payload,
                request_payload={"prompt": "Return one final JSON join for A, B, and C."},
            )
            canonical_bound_payload = json.loads(json.dumps(payload))
            canonical_bound_payload["late_fill"]["fill_results"][1].pop("saved_text_path")
            canonical_bound_frame = build_response_frame(
                canonical_bound_payload,
                request_payload={"prompt": "Return one final JSON join for A, B, and C."},
            )
            replayed_frame = build_response_frame(
                response_payload_from_frame(frame),
                request_payload={"prompt": "Return one final JSON join for A, B, and C."},
            )

        for projected_frame in (frame, canonical_bound_frame, replayed_frame):
            for projection, exposes_artifact_path in (
                (projected_frame["planning"]["artifact_flow"]["output_slots"], True),
                (projected_frame["output"]["outputs"], False),
                (projected_frame["current_state"]["output_slots"], True),
                (projected_frame["current_state"]["outputs"], False),
            ):
                self.assertEqual(
                    [item.get("artifact_ref") for item in projection].count("artifact:joined-json"),
                    1,
                    projection,
                )
                non_owner = next(item for item in projection if item.get("phase_id") == "phase-5")
                owner = next(item for item in projection if item.get("phase_id") == "phase-8")
                self.assertNotIn("artifact_ref", non_owner)
                self.assertNotIn("artifact_path", non_owner)
                self.assertEqual(owner["artifact_ref"], "artifact:joined-json")
                if exposes_artifact_path:
                    self.assertEqual(owner["artifact_path"], str(final_path))
                legacy = next(
                    item for item in projection if item.get("phase_id") == "phase-legacy-audio"
                )
                self.assertEqual(legacy["artifact_ref"], "artifact:legacy-unbound")
                if exposes_artifact_path:
                    self.assertEqual(legacy["artifact_path"], "/tmp/legacy-unbound.wav")

    def test_response_frame_public_text_suppresses_completed_repair_prompt(self):
        prompt = "Create exactly two local file artifacts: index.html and styles.css for a landing page."
        graph = build_request_phase_graph(
            prompt,
            request_payload={"ghost_route": True, "prompt": prompt},
            route_payload={"capability": "chat", "route_source": "ghost_carried"},
        )
        repair_prompt = (
            "Target text artifact: artifacts/documents/index.html\n"
            "Deterministic syntax sanity issues:\n"
            "- HTML has stray closing tag </p>.\n\n"
            "Current saved target file content:\n"
            "<!doctype html><html><body><section><p>Copy</p></body></html>\n\n"
            "Update only the target text artifact."
        )

        frame = build_response_frame(
            {
                "id": "resp_completed_repair_prompt_hidden",
                "object": "response",
                "status": "completed",
                "model": "gemma4:26b",
                "backend": "ollama",
                "capability": "chat",
                "output_text": repair_prompt,
                "runtime": {"request_phase_graph": graph},
                "artifacts": [
                    {
                        "type": "text",
                        "path": "/tmp/index.html",
                        "artifact_ref": "artifact:index",
                        "text_artifact_extension": "html",
                        "text_artifact_source_name": "index",
                    },
                    {
                        "type": "text",
                        "path": "/tmp/styles.css",
                        "artifact_ref": "artifact:styles",
                        "text_artifact_extension": "css",
                        "text_artifact_source_name": "styles",
                    },
                ],
                "late_fill": {
                    "status": "completed",
                    "final_materialization_contract_status": "fulfilled",
                    "pending_branches": [],
                    "active_branches": [],
                    "completed_branches": [
                        {
                            "branch_id": "repair-chat",
                            "phase_id": "repair-chat",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                            "content_payload": repair_prompt,
                            "text_artifact_extension": "html",
                            "text_artifact_source_name": "index",
                        }
                    ],
                    "fill_results": [
                        {
                            "branch_id": "repair-chat",
                            "phase_id": "repair-chat",
                            "capability": "chat",
                            "content_payload": repair_prompt,
                            "saved_text_path": "/tmp/index.html",
                            "text_artifact_extension": "html",
                            "text_artifact_source_name": "index",
                        }
                    ],
                },
            },
            request_payload={"prompt": prompt},
        )

        self.assertEqual(frame["output"]["text"], "Artifacts generated.")
        self.assertEqual(frame["current_state"]["output_text"], "Artifacts generated.")
        self.assertNotIn("Target text artifact:", json.dumps(frame["output"]["outputs"]))

    def test_canonical_outputs_hydrate_artifact_backed_text_over_repair_prompt_value(self):
        repair_prompt = (
            "Target text artifact: artifacts/documents/styles.css\n"
            "Linked HTML artifact: artifacts/documents/index.html\n"
            "Update only the target text artifact."
        )
        css_content = "body { color: #111; }"

        outputs = build_canonical_outputs(
            {"output_text": repair_prompt},
            output_slots=[
                {
                    "slot_id": "output-phase-6",
                    "branch_id": "branch-text_artifact-2",
                    "phase_id": "branch-text_artifact-2",
                    "type": "text",
                    "status": "fulfilled",
                    "artifact_ref": "artifact:styles",
                    "value": repair_prompt,
                }
            ],
            artifacts=[
                {
                    "type": "text",
                    "path": "/tmp/artifacts/documents/styles.css",
                    "artifact_ref": "artifact:styles",
                    "content": css_content,
                }
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:styles")
        self.assertEqual(outputs[0]["value"], css_content)
        self.assertNotIn("Target text artifact:", json.dumps(outputs))

    def test_response_frame_public_text_suppresses_terminal_branch_status_summary(self):
        prompt = "Create exactly four images plus index.html and styles.css for a playful landing page."
        graph = build_request_phase_graph(
            prompt,
            request_payload={"ghost_route": True, "prompt": prompt},
            route_payload={"capability": "chat", "route_source": "ghost_carried"},
        )
        branch_summary = (
            "branch-image_generation-5: Image generated.\n\n"
            "branch-image_generation-6: Image generated.\n\n"
            "branch-image_generation-7: Image generated.\n\n"
            "branch-image_generation-8: Image generated."
        )

        frame = build_response_frame(
            {
                "id": "resp_completed_branch_summary_hidden",
                "object": "response",
                "status": "completed",
                "model": "gemma4:26b",
                "backend": "ollama",
                "capability": "chat",
                "output_text": branch_summary,
                "runtime": {"request_phase_graph": graph},
                "artifacts": [
                    {"type": "image", "path": "/tmp/one.png", "artifact_ref": "artifact:one"},
                    {"type": "image", "path": "/tmp/two.png", "artifact_ref": "artifact:two"},
                    {
                        "type": "text",
                        "path": "/tmp/index.html",
                        "artifact_ref": "artifact:index",
                        "text_artifact_extension": "html",
                        "text_artifact_source_name": "index",
                    },
                ],
                "late_fill": {
                    "status": "completed",
                    "final_materialization_contract_status": "fulfilled",
                    "pending_branches": [],
                    "active_branches": [],
                    "completed_branches": [
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-6",
                            "capability": "chat",
                            "output_type": "text",
                            "status": "fulfilled",
                            "content_payload": branch_summary,
                        }
                    ],
                    "fill_results": [
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-6",
                            "capability": "chat",
                            "result_text": branch_summary,
                        }
                    ],
                },
            },
            request_payload={"prompt": prompt},
        )

        self.assertEqual(frame["output"]["text"], "Artifacts generated.")
        self.assertEqual(frame["current_state"]["output_text"], "Artifacts generated.")
        self.assertNotIn("branch-image_generation-5", json.dumps(frame["output"]["outputs"]))

    def test_canonical_outputs_do_not_copy_root_text_into_downstream_text_slot(self):
        outputs = build_canonical_outputs(
            {"output_text": "Phase one preparation."},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-phase-3"],
                },
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                },
            ],
        )

        self.assertEqual(outputs[0]["value"], "Phase one preparation.")
        self.assertNotIn("value", outputs[1])

    def test_canonical_outputs_do_not_bind_root_text_to_text_artifact_by_type(self):
        outputs = build_canonical_outputs(
            {"output_text": "Prepare a landing page with HTML and CSS artifacts."},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-phase-7", "output-phase-8"],
                },
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                },
                {
                    "slot_id": "output-phase-8",
                    "branch_id": "branch-text_artifact-2",
                    "phase_id": "phase-8",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/index.html",
                    "text_artifact_request": {
                        "source_name": "index",
                        "extension": "html",
                    },
                },
                {
                    "artifact_ref": "artifact:styles",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/styles.css",
                    "text_artifact_request": {
                        "source_name": "styles",
                        "extension": "css",
                    },
                },
            ],
        )

        self.assertEqual(outputs[0]["value"], "Prepare a landing page with HTML and CSS artifacts.")
        self.assertNotIn("artifact_ref", outputs[0])
        self.assertEqual(outputs[1]["artifact_ref"], "artifact:index")
        self.assertEqual(outputs[2]["artifact_ref"], "artifact:styles")

    def test_canonical_outputs_hide_generic_chat_summary_when_text_artifact_exists(self):
        outputs = build_canonical_outputs(
            {"output_text": "Artifacts generated."},
            output_slots=[
                {
                    "slot_id": "output-phase-6",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-6",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "value": "Created the requested files.",
                },
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/index.html",
                    "text_artifact_request": {
                        "source_name": "index",
                        "extension": "html",
                    },
                },
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:index")

    def test_canonical_outputs_hide_planner_text_when_artifact_bundle_exists(self):
        outputs = build_canonical_outputs(
            {
                "output_text": (
                    "**Image Generation Prompts**\n\n"
                    "1. A joyful hero dog portrait.\n\n"
                    "```html\n<!doctype html><html><body><img src=\"hero.png\"></body></html>\n```\n\n"
                    "```css\nbody { color: #111; }\n```"
                ),
            },
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "document",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-phase-2", "output-phase-3", "output-phase-4"],
                },
                {
                    "slot_id": "output-phase-2",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                    "type": "image",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "image_generation",
                    "artifact_ref": "artifact:hero",
                },
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:index",
                },
                {
                    "slot_id": "output-phase-4",
                    "branch_id": "branch-text_artifact-2",
                    "phase_id": "phase-4",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:styles",
                },
            ],
            artifacts=[
                {"artifact_ref": "artifact:hero", "type": "image", "path": "/tmp/artifacts/images/hero.png"},
                {"artifact_ref": "artifact:index", "type": "text", "path": "/tmp/artifacts/documents/index.html"},
                {"artifact_ref": "artifact:styles", "type": "text", "path": "/tmp/artifacts/documents/styles.css"},
            ],
        )

        self.assertEqual(
            [output["slot_id"] for output in outputs],
            ["output-phase-2", "output-phase-3", "output-phase-4"],
        )
        self.assertNotIn("Image Generation Prompts", json.dumps(outputs))
        self.assertEqual({output["artifact_ref"] for output in outputs}, {"artifact:hero", "artifact:index", "artifact:styles"})

    def test_canonical_outputs_hide_duplicate_artifact_completion_status_text(self):
        outputs = build_canonical_outputs(
            {"output_text": "All requested image generation artifacts have been successfully generated."},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "document",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-phase-2", "output-phase-3", "output-phase-4"],
                },
                {
                    "slot_id": "output-phase-2",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                    "type": "image",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "image_generation",
                    "artifact_ref": "artifact:image",
                },
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "value": "All requested image generation artifacts have been successfully generated.",
                },
                {
                    "slot_id": "output-phase-4",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-4",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:index",
                },
            ],
            artifacts=[
                {"artifact_ref": "artifact:image", "type": "image", "path": "/tmp/artifacts/images/hero.png"},
                {"artifact_ref": "artifact:index", "type": "text", "path": "/tmp/artifacts/documents/index.html"},
            ],
        )

        self.assertEqual([output["slot_id"] for output in outputs], ["output-phase-2", "output-phase-4"])
        self.assertEqual({output["artifact_ref"] for output in outputs}, {"artifact:image", "artifact:index"})

    def test_canonical_outputs_compact_duplicate_output_slots_to_latest(self):
        outputs = build_canonical_outputs(
            {"output_text": "Artifacts generated."},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-phase-7", "output-phase-6"],
                },
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:old-index",
                },
                {
                    "slot_id": "output-phase-6",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-6",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                },
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:new-index",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:old-index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/old-index.html",
                },
            ],
        )

        self.assertEqual([output["slot_id"] for output in outputs], ["output-phase-7", "output-phase-6"])
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:new-index")

    def test_canonical_outputs_keep_missing_explicit_text_ref_without_type_fallback(self):
        outputs = build_canonical_outputs(
            {"output_text": "Artifacts generated."},
            output_slots=[
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:new-index",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:old-index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/old-index.html",
                },
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:new-index")

    def test_canonical_outputs_recover_stale_text_ref_from_matching_late_fill_artifact(self):
        outputs = build_canonical_outputs(
            {"output_text": "Artifacts generated."},
            output_slots=[
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:stale-index",
                    "artifact_path": "/tmp/artifacts/documents/index.html",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:canonical-index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/index.html",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                },
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:canonical-index")

    def test_canonical_outputs_hide_fulfilled_repair_text_materialization_output(self):
        outputs = build_canonical_outputs(
            {"output_text": "Artifacts generated."},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "branch_id": "phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "child_slot_ids": ["output-repair-chat", "output-phase-7"],
                },
                {
                    "slot_id": "output-repair-chat",
                    "branch_id": "repair-chat",
                    "phase_id": "repair-chat",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:stale-repair-index",
                    "value": "Target text artifact: /tmp/artifacts/documents/index.html\nUpdate only the target text artifact.",
                },
                {
                    "slot_id": "output-phase-7",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:stale-index",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/index.html",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-7",
                },
            ],
        )

        self.assertEqual([output["slot_id"] for output in outputs], ["output-phase-7"])
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:index")

    def test_canonical_outputs_hide_sectioned_artifact_handoff_text(self):
        handoff_text = (
            "IMAGE_PROMPTS:\n"
            "1. A professional pet portrait.\n\n"
            "WEB_CONTENT_SPECIFICATION:\n"
            "Target Files: index.html, styles.css, dog.png\n"
        )
        outputs = build_canonical_outputs(
            {"output_text": handoff_text},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                },
                {
                    "slot_id": "output-phase-2",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                    "type": "image",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "image_generation",
                    "artifact_ref": "artifact:image",
                },
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:index",
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:image",
                    "type": "image",
                    "path": "/tmp/artifacts/images/dog.png",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                },
                {
                    "artifact_ref": "artifact:index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/index.html",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-3",
                },
            ],
        )

        self.assertEqual([output["slot_id"] for output in outputs], ["output-phase-2", "output-phase-3"])
        self.assertNotIn("IMAGE_PROMPTS", json.dumps(outputs))

    def test_canonical_outputs_hide_artifact_availability_status_text(self):
        status_text = (
            "The image generation artifacts are available. "
            "Branch-image_generation-2, branch-image_generation-3, and branch-image_generation-4 "
            "all indicate successful image generation. Branch-image_generation-1 indicates that "
            "the image request was completed, but no inline image payload was returned."
        )
        outputs = build_canonical_outputs(
            {"output_text": status_text},
            output_slots=[
                {
                    "slot_id": "output-phase-1",
                    "phase_id": "phase-1",
                    "type": "text",
                    "status": "fulfilled",
                },
                {
                    "slot_id": "output-phase-2",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                    "type": "image",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "image_generation",
                    "artifact_ref": "artifact:image",
                },
                {
                    "slot_id": "output-phase-3",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-3",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "artifact_ref": "artifact:index",
                },
                {
                    "slot_id": "output-phase-4",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-4",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_slot_id": "output-phase-1",
                    "follow_up_capability": "chat",
                    "value": status_text,
                },
            ],
            artifacts=[
                {
                    "artifact_ref": "artifact:image",
                    "type": "image",
                    "path": "/tmp/artifacts/images/hero.png",
                    "branch_id": "branch-image_generation-1",
                    "phase_id": "phase-2",
                },
                {
                    "artifact_ref": "artifact:index",
                    "type": "text",
                    "path": "/tmp/artifacts/documents/index.html",
                    "branch_id": "branch-text_artifact-1",
                    "phase_id": "phase-3",
                },
            ],
        )

        self.assertEqual([output["slot_id"] for output in outputs], ["output-phase-2", "output-phase-3"])
        self.assertNotIn("image generation artifacts are available", json.dumps(outputs).lower())

    def test_canonical_outputs_mark_text_fallback_as_compatibility_derived(self):
        outputs = build_canonical_outputs({"output_text": "Legacy text only."})

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["value"], "Legacy text only.")
        self.assertEqual(outputs[0]["source"], "compatibility_derived")
        self.assertTrue(outputs[0]["compatibility_derived"])

    def test_canonical_outputs_mark_stray_artifact_fallback_as_compatibility_derived(self):
        outputs = build_canonical_outputs(
            {},
            artifacts=[
                {
                    "type": "image",
                    "path": "/tmp/stray.png",
                    "artifact_ref": "artifact:stray",
                }
            ],
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["artifact_ref"], "artifact:stray")
        self.assertEqual(outputs[0]["source"], "compatibility_derived")
        self.assertTrue(outputs[0]["compatibility_derived"])

    def test_canonical_outputs_absent_without_promoted_or_compatibility_surface(self):
        self.assertEqual(build_canonical_outputs({}), [])

    def test_build_response_frame_captures_target_route_artifacts_and_request(self):
        frame = build_response_frame(
            {
                "id": "resp_123",
                "object": "response",
                "status": "completed",
                "model": "gpt-oss:20b",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Hello.",
                "output": [{"id": "msg_123", "type": "message"}],
                "route_source": "router",
                "route_reason": "semantic routing chose chat",
                "route_confidence": 0.87,
                "runtime": {"context_strategy": {"mode": "recent_history"}},
                "artifacts": [{"type": "text", "path": "artifacts/documents/one.md"}],
            },
            request_payload={
                "input": "hello",
                "conversation_id": "responses-workbench",
                "ghost_route": True,
            },
        )

        self.assertEqual(frame["frame_version"], 9)
        self.assertEqual(frame["kind"], "ollmo.response_frame")
        self.assertEqual(frame["response_id"], "resp_123")
        self.assertEqual(frame["target"]["instance_id"], "chat-1")
        self.assertEqual(frame["target"]["model"], "gpt-oss:20b")
        self.assertEqual(frame["route"]["route_source"], "router")
        self.assertEqual(frame["runtime"]["context_strategy"]["mode"], "recent_history")
        self.assertEqual(frame["request"]["input"], "hello")
        self.assertEqual(frame["artifacts"]["output"][0]["path"], "artifacts/documents/one.md")
        dossier = next(iter(frame["artifacts"]["dossiers"].values()))
        self.assertEqual(dossier["roles"], ["output"])
        self.assertEqual(dossier["artifact"]["path"], "artifacts/documents/one.md")
        self.assertEqual(frame["planning"]["artifact_flow"]["output_slots"][0]["status"], "fulfilled")
        self.assertEqual(frame["planning"]["work_tree"]["kind"], "ollmo.work_tree")
        self.assertEqual(frame["working_frame"]["kind"], "ollmo.working_frame")
        self.assertEqual(frame["working_frame"]["status"], "frozen")
        self.assertEqual(frame["working_frame"]["closure"]["status"], "closed")
        self.assertEqual(frame["output"]["text"], "Hello.")
        self.assertEqual(frame["output"]["outputs"][0]["type"], "text")
        self.assertEqual(frame["output"]["outputs"][0]["value"], "Hello.")
        self.assertNotIn("controls", frame)

    def test_build_response_frame_exposes_intent_contract_in_planning(self):
        phase_graph = build_request_phase_graph(
            "Write a short prompt and generate an image from it.",
            request_payload={"ghost_route": True},
            route_payload={"capability": "image_generation", "route_source": "ghost_carried"},
            response_payload={"output_text": "A prompt is ready."},
        )
        frame = build_response_frame(
            {
                "id": "resp_contract_frame",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "A prompt is ready.",
                "runtime": {
                    "request_phase_graph": phase_graph,
                    "graph_closure_review": {
                        "kind": "ollmo.graph_closure_review",
                        "status": "pending",
                        "reason": "image output remains open",
                        "contract_source": "request_ir.output_obligations",
                        "obligation_count": 2,
                        "counts": {"fulfilled": 1, "pending": 1, "deferred": 0, "blocked": 0},
                        "checks": [
                            {
                                "obligation_id": "obligation-phase-1",
                                "phase_id": "phase-1",
                                "capability": "chat",
                                "output_type": "text",
                                "status": "fulfilled",
                                "evidence": "current_phase_output_text",
                            },
                            {
                                "obligation_id": "obligation-phase-2",
                                "phase_id": "phase-2",
                                "branch_id": "branch-image_generation-1",
                                "capability": "image_generation",
                                "output_type": "image",
                                "status": "pending",
                                "evidence": "pending_graph_branch",
                            },
                        ],
                    },
                },
                "late_fill": {
                    "status": "pending",
                    "expected_capability": "image_generation",
                    "missing_artifact_type": "image",
                    "pending_branches": [
                        {
                            "branch_id": "branch-image_generation-1",
                            "phase_id": "phase-2",
                            "obligation_id": "obligation-phase-2",
                            "capability": "image_generation",
                            "output_type": "image",
                        }
                    ],
                },
            },
            request_payload={
                "prompt": "Write a short prompt and generate an image from it.",
                "ghost_route": True,
            },
        )

        contract = frame["planning"]["intent_contract"]
        self.assertEqual(contract["kind"], "ollmo.intent_contract")
        self.assertEqual(contract["source"], "request_ir.output_obligations")
        self.assertEqual(contract["status"], "pending")
        self.assertEqual(contract["pending_obligation_ids"], ["obligation-phase-2"])
        self.assertEqual(frame["working_frame"]["intent_contract"], contract)

    def test_build_response_frame_exposes_context_contract_in_planning(self):
        frame = build_response_frame(
            {
                "id": "resp_context_contract",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Current answer.",
            },
            request_payload={
                "prompt": "Answer this without using the older image unless needed.",
                "context_candidates": [
                    {
                        "candidate_id": "ctx-old-image",
                        "source_kind": "artifact",
                        "summary": "Older image from the thread may be relevant.",
                        "status": "not_promoted",
                    }
                ],
                "selected_reference_artifacts": [
                    {
                        "type": "text",
                        "path": "artifacts/texts/current-reference.txt",
                        "name": "current-reference.txt",
                    }
                ],
            },
        )

        contract = frame["planning"]["context_contract"]
        self.assertEqual(contract["kind"], "ollmo.context_contract")
        self.assertEqual(contract["status"], "active")
        self.assertEqual(contract["candidate_count"], 2)
        self.assertEqual(contract["promotion_count"], 1)
        self.assertEqual(contract["context_candidates"][0]["candidate_id"], "ctx-old-image")
        self.assertEqual(contract["context_candidates"][0]["status"], "not_promoted")
        self.assertEqual(contract["promotions"][0]["target"], "active_reference")
        self.assertEqual(frame["working_frame"]["context_contract"], contract)

    def test_build_response_frame_marks_explicit_waived_output_slots_ready(self):
        phase_graph = build_request_phase_graph(
            "Write a short prompt and generate an image from it.",
            request_payload={"ghost_route": True},
            route_payload={"capability": "image_generation", "route_source": "ghost_carried"},
            response_payload={"output_text": "A prompt is ready; image output is not needed."},
        )
        for phase in phase_graph["phases"]:
            if phase.get("phase_id") == "phase-2":
                phase["status"] = "not_needed_verified"
        for obligation in phase_graph["request_ir"]["output_obligations"]:
            if obligation.get("obligation_id") == "obligation-phase-2":
                obligation["status"] = "not_needed_verified"
        phase_graph["output_obligations"] = phase_graph["request_ir"]["output_obligations"]

        frame = build_response_frame(
            {
                "id": "resp_waived_slot",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "A prompt is ready; image output is not needed.",
                "runtime": {"request_phase_graph": phase_graph},
            },
            request_payload={
                "prompt": "Write a short prompt and generate an image from it.",
                "ghost_route": True,
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        image_slot = next(slot for slot in artifact_flow["output_slots"] if slot["type"] == "image")
        self.assertEqual(image_slot["status"], "waived")
        self.assertEqual(image_slot["lifecycle"], "waived_output")
        self.assertEqual(artifact_flow["review"]["status"], "ready")
        self.assertEqual(artifact_flow["review"].get("pending_output_slot_ids", []), [])
        self.assertEqual(artifact_flow["review"]["waived_output_slot_ids"], [image_slot["slot_id"]])
        self.assertEqual(frame["working_frame"]["review"]["waived_output_slot_ids"], [image_slot["slot_id"]])
        self.assertEqual(frame["working_frame"]["possibility_space"]["waived_output_slot_ids"], [image_slot["slot_id"]])

    def test_build_response_frame_marks_superseded_output_slots_ready(self):
        phase_graph = build_request_phase_graph(
            "Write a short prompt, replace the planned image branch, and keep the newer branch.",
            request_payload={"ghost_route": True},
            route_payload={"capability": "image_generation", "route_source": "ghost_carried"},
            response_payload={"output_text": "A prompt is ready; the first image branch was replaced."},
        )
        for phase in phase_graph["phases"]:
            if phase.get("phase_id") == "phase-2":
                phase["status"] = "superseded"
                phase["superseded_by_obligation_id"] = "obligation-phase-3"
                phase["supersession_reason"] = "newer branch replaced this image obligation"
        for obligation in phase_graph["request_ir"]["output_obligations"]:
            if obligation.get("obligation_id") == "obligation-phase-2":
                obligation["status"] = "superseded"
                obligation["superseded_by_obligation_id"] = "obligation-phase-3"
                obligation["supersession_reason"] = "newer branch replaced this image obligation"
        phase_graph["output_obligations"] = phase_graph["request_ir"]["output_obligations"]

        frame = build_response_frame(
            {
                "id": "resp_superseded_slot",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "A prompt is ready; the first image branch was replaced.",
                "runtime": {"request_phase_graph": phase_graph},
            },
            request_payload={
                "prompt": "Write a short prompt, replace the planned image branch, and keep the newer branch.",
                "ghost_route": True,
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        image_slot = next(slot for slot in artifact_flow["output_slots"] if slot["type"] == "image")
        self.assertEqual(image_slot["status"], "superseded")
        self.assertEqual(image_slot["lifecycle"], "superseded_output")
        self.assertEqual(image_slot["superseded_by_obligation_id"], "obligation-phase-3")
        self.assertEqual(artifact_flow["review"]["status"], "ready")
        self.assertEqual(artifact_flow["review"].get("pending_output_slot_ids", []), [])
        self.assertEqual(artifact_flow["review"]["superseded_output_slot_ids"], [image_slot["slot_id"]])
        self.assertEqual(frame["working_frame"]["review"]["superseded_output_slot_ids"], [image_slot["slot_id"]])
        self.assertEqual(frame["working_frame"]["possibility_space"]["superseded_output_slot_ids"], [image_slot["slot_id"]])
        self.assertEqual(frame["working_frame"]["intent_contract"]["superseded_obligation_ids"], ["obligation-phase-2"])

    def test_build_response_frame_exposes_reserved_candidate_without_pending_slot(self):
        phase_graph = build_request_phase_graph(
            "Sketch a possible image direction, but do not generate it yet.",
            request_payload={
                "ghost_route": True,
                "downstream_branches": [
                    {
                        "candidate_id": "candidate-image-1",
                        "branch_id": "branch-image-possible-1",
                        "phase_id": "phase-image-possible-1",
                        "capability": "image_generation",
                        "output_type": "image",
                        "required": False,
                        "contract_state": "reserved",
                    }
                ],
            },
            route_payload={"capability": "chat", "route_source": "ghost_carried"},
            response_payload={"output_text": "A possible image direction is noted."},
        )

        frame = build_response_frame(
            {
                "id": "resp_reserved_candidate",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "A possible image direction is noted.",
                "runtime": {"request_phase_graph": phase_graph},
            },
            request_payload={
                "prompt": "Sketch a possible image direction, but do not generate it yet.",
                "ghost_route": True,
            },
        )

        contract = frame["planning"]["intent_contract"]
        self.assertEqual(contract["candidate_count"], 1)
        self.assertEqual(contract["candidate_output_ids"], ["candidate-image-1"])
        self.assertEqual(contract["output_candidates"][0]["status"], "reserved")
        self.assertEqual([slot["type"] for slot in frame["planning"]["artifact_flow"]["output_slots"]], ["text"])
        self.assertEqual(contract.get("pending_obligation_ids", []), [])

    def test_build_control_snapshot_captures_non_default_controls_and_compact_references(self):
        snapshot = build_control_snapshot(
            {
                "temperature": "0.4",
                "topP": "0.8",
                "maxTokens": "777",
                "voice": "alloy",
                "lang_code": "de",
                "response_format": "mp3",
                "speed": "1.2",
                "pitch": "0.8",
                "width": "1024",
                "height": "768",
                "seed": "42",
                "ocr_mode": "layout",
                "pdf_max_pages": "3",
                "selected_reference_artifacts": [
                    {
                        "type": "message",
                        "message_id": "msg_1",
                        "content": "previous assistant reply",
                        "response_model": "gpt-oss:20b",
                    },
                    {
                        "type": "image",
                        "path": "artifacts/images/reference.png",
                        "seed": 99,
                    },
                ],
            },
            {
                "instance_id": "image-1",
                "model": "flux",
                "backend": "mlx",
                "capability": "image_generation",
                "mode": "image_generation",
                "reference_image_count": 1,
                "reference_image_kind": "image",
            },
        )

        self.assertEqual(snapshot["kind"], "ollmo.control_snapshot")
        self.assertEqual(snapshot["target"]["instance_id"], "image-1")
        self.assertEqual(snapshot["values"]["generation"]["temperature"], 0.4)
        self.assertEqual(snapshot["values"]["generation"]["top_p"], 0.8)
        self.assertEqual(snapshot["values"]["generation"]["max_tokens"], 777)
        self.assertEqual(snapshot["values"]["audio"]["voice"], "alloy")
        self.assertEqual(snapshot["values"]["audio"]["lang_code"], "de")
        self.assertEqual(snapshot["values"]["audio"]["response_format"], "mp3")
        self.assertEqual(snapshot["values"]["audio"]["speed"], 1.2)
        self.assertEqual(snapshot["values"]["audio"]["pitch"], 0.8)
        self.assertEqual(snapshot["values"]["image"]["width"], 1024)
        self.assertEqual(snapshot["values"]["image"]["height"], 768)
        self.assertEqual(snapshot["values"]["image"]["seed"], 42)
        self.assertEqual(snapshot["values"]["document"]["ocr_mode"], "layout")
        self.assertEqual(snapshot["values"]["document"]["pdf_max_pages"], 3)
        selected = snapshot["values"]["references"]["selected"]
        self.assertEqual(selected[0]["ref"], "message:message_msg_1")
        self.assertIn("content_sha256", selected[0])
        self.assertNotIn("content", selected[0])
        self.assertEqual(selected[1]["path"], "artifacts/images/reference.png")
        self.assertEqual(snapshot["values"]["references"]["image_reference"]["count"], 1)
        self.assertTrue(snapshot["replay"]["promotable"])

    def test_build_control_snapshot_filters_default_only_controls(self):
        snapshot = build_control_snapshot(
            {
                "task": "transcribe",
                "ocr_mode": "auto",
                "pdf_dpi": "300",
                "pdf_page_timeout_sec": "180",
                "pdf_synthesize": "false",
                "reuse_cached": "false",
                "lang_code": "auto",
                "speed": "1.0",
                "pitch": 1.0,
                "image_count": "1",
            },
            {"capability": "vision"},
        )

        self.assertEqual(snapshot, {})

    def test_build_response_frame_tracks_multi_artifact_input_routes_and_placeholders(self):
        frame = build_response_frame(
            {
                "id": "resp_multi",
                "object": "response",
                "status": "completed",
                "model": "flux",
                "instance_id": "image-1",
                "backend": "ollama",
                "capability": "image_generation",
                "mode": "image_generation",
                "output_text": "",
                "input_artifacts": [
                    {"type": "image", "path": "artifacts/inputs/reference.png"},
                    {"type": "audio", "path": "artifacts/inputs/instruction.wav"},
                ],
            },
            request_payload={
                "input": "use the image and spoken instruction",
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        self.assertEqual(artifact_flow["status"], "tracked")
        self.assertEqual(len(artifact_flow["input_routes"]), 2)
        self.assertEqual(artifact_flow["input_routes"][0]["routing_hint"]["capability"], "image_generation")
        self.assertEqual(artifact_flow["input_routes"][1]["routing_hint"]["capability"], "stt")
        self.assertEqual(artifact_flow["output_slots"][0]["type"], "image")
        self.assertEqual(artifact_flow["output_slots"][0]["status"], "pending")
        self.assertEqual(artifact_flow["output_slots"][0]["lifecycle"], "emerging_output")
        self.assertEqual(artifact_flow["review"]["status"], "pending_outputs")

    def test_build_response_frame_marks_late_fill_outputs_as_deferred(self):
        frame = build_response_frame(
            {
                "id": "resp_late_fill_audio",
                "object": "response",
                "status": "completed",
                "model": "qwen",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Narration script ready.",
                "late_fill": {
                    "status": "pending",
                    "expected_capability": "text_to_speech",
                    "missing_artifact_type": "audio",
                },
            },
            request_payload={
                "input": "translate this and read it aloud",
                "reference_artifacts": [{"type": "text", "path": "artifacts/texts/source.txt"}],
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        self.assertEqual(artifact_flow["reference_routes"][0]["lifecycle"], "carried_reference")
        self.assertEqual(artifact_flow["output_slots"][0]["type"], "text")
        self.assertEqual(artifact_flow["output_slots"][0]["status"], "fulfilled")
        self.assertEqual(artifact_flow["output_slots"][0]["child_slot_ids"], ["output-phase-2"])
        self.assertEqual(artifact_flow["output_slots"][1]["type"], "audio")
        self.assertEqual(artifact_flow["output_slots"][1]["status"], "pending")
        self.assertEqual(artifact_flow["output_slots"][1]["lifecycle"], "deferred_output")
        self.assertEqual(artifact_flow["output_slots"][1]["parent_slot_id"], "output-phase-1")
        self.assertEqual(artifact_flow["output_slots"][1]["phase_id"], "phase-2")
        self.assertIn("output-phase-2", artifact_flow["review"]["deferred_output_slot_ids"])
        outputs = frame["output"]["outputs"]
        self.assertEqual(outputs[0]["type"], "text")
        self.assertEqual(outputs[0]["value"], "Narration script ready.")
        self.assertEqual(outputs[1]["type"], "audio")
        self.assertEqual(outputs[1]["status"], "pending")
        self.assertEqual(outputs[1]["slot_id"], "output-phase-2")

    def test_build_response_frame_keeps_slot_only_pending_late_fill_outputs_truthful(self):
        frame = build_response_frame(
            {
                "id": "resp_slot_only_pending_image",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Image generated.",
                "late_fill": {
                    "status": "pending",
                    "expected_capability": "image_generation",
                    "missing_artifact_type": "image",
                },
            },
            request_payload={
                "input": "show me a moonlit cove at night as an image",
            },
        )

        outputs = frame["output"]["outputs"]
        self.assertEqual(outputs[0]["type"], "text")
        self.assertEqual(outputs[0]["value"], "Image generated.")
        self.assertEqual(outputs[1]["type"], "image")
        self.assertEqual(outputs[1]["status"], "pending")
        self.assertEqual(outputs[1]["slot_id"], "output-phase-2")
        self.assertEqual(outputs[1]["branch_id"], "branch-image_generation-1")

    def test_build_response_frame_marks_failed_direct_image_output_blocked(self):
        frame = build_response_frame(
            {
                "id": "resp_failed_image",
                "object": "response",
                "status": "failed",
                "model": "x/flux2-klein:latest",
                "instance_id": "x/flux2-klein:latest-1",
                "backend": "ollama",
                "capability": "image_generation",
                "mode": "image_generation",
                "output_text": "",
                "error": "Ollama generate returned non-JSON response.",
                "error_detail": {
                    "message": "Ollama generate returned non-JSON response.",
                },
            },
            request_payload={
                "input": "a red cube on a white table",
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        slot = artifact_flow["output_slots"][0]
        self.assertEqual(slot["type"], "image")
        self.assertEqual(slot["status"], "blocked")
        self.assertEqual(slot["lifecycle"], "blocked_output")
        self.assertEqual(slot["blocked_reason"], "Ollama generate returned non-JSON response.")
        self.assertNotIn("placeholder_ref", slot)
        self.assertEqual(artifact_flow["review"]["status"], "blocked")
        self.assertEqual(artifact_flow["review"]["blocked_output_slot_ids"], ["output-1"])
        self.assertEqual(artifact_flow["review"].get("pending_output_slot_ids", []), [])

        work_tree = frame["planning"]["work_tree"]
        self.assertEqual(work_tree["status"], "blocked")
        self.assertIn("node-output-1", work_tree["blocked_node_ids"])
        self.assertEqual(work_tree.get("pending_node_ids", []), [])

        output = frame["output"]["outputs"][0]
        self.assertEqual(output["type"], "image")
        self.assertEqual(output["status"], "blocked")
        self.assertEqual(output["blocked_reason"], "Ollama generate returned non-JSON response.")

    def test_build_response_frame_keeps_diagnostic_audio_out_of_fulfilled_outputs(self):
        frame = build_response_frame(
            {
                "id": "resp_failed_tts_integrity",
                "object": "response",
                "status": "completed",
                "model": "fake-tts",
                "instance_id": "tts-1",
                "backend": "mlx",
                "capability": "text_to_speech",
                "mode": "text_to_speech",
                "saved_audio_path": "/tmp/diagnostic-silent.wav",
                "artifacts": [
                    {
                        "type": "audio",
                        "path": "/tmp/diagnostic-silent.wav",
                        "artifact_ref": "artifact:diagnostic-silent-audio",
                        "status": "failed",
                        "diagnostic_only": True,
                        "materialization_eligible": False,
                        "integrity_reason_code": "TTS_AUDIO_NO_ACTIVE_SIGNAL",
                    }
                ],
            },
            request_payload={
                "prompt": "Read this complete sentence aloud.",
                "capability": "text_to_speech",
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        slot = artifact_flow["output_slots"][0]
        self.assertEqual(slot["type"], "audio")
        self.assertEqual(slot["status"], "blocked")
        self.assertEqual(slot["lifecycle"], "diagnostic_artifact")
        self.assertEqual(slot["blocked_reason"], "TTS_AUDIO_NO_ACTIVE_SIGNAL")
        self.assertEqual(
            slot["artifact_ref"],
            "artifact:diagnostic-silent-audio",
        )
        self.assertEqual(artifact_flow["review"]["status"], "blocked")
        self.assertEqual(
            artifact_flow["work_tree"]["blocked_node_ids"],
            ["node-output-1"],
        )
        self.assertFalse(
            any(
                output.get("type") == "audio"
                and output.get("status") == "fulfilled"
                for output in frame["output"]["outputs"]
            )
        )

    def test_build_response_frame_persists_request_phase_graph_and_multiple_downstream_slots(self):
        frame = build_response_frame(
            {
                "id": "resp_story_multi",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Moonlight pooled around the old observatory as the story began.",
                "runtime": {
                    "request_phase_graph": {
                        "graph_version": 2,
                        "kind": "ollmo.request_phase_graph",
                        "mode": "phase_chain",
                        "current_phase_id": "phase-1",
                        "current_phase_capability": "chat",
                        "current_phase_resolution": "graph_resolved",
                        "downstream_phase_ids": ["phase-2", "phase-3"],
                        "downstream_capabilities": ["text_to_speech", "image_generation"],
                        "is_multi_phase": True,
                        "continuation_required": True,
                        "phases": [
                            {"phase_id": "phase-1", "capability": "chat", "status": "completed"},
                            {"phase_id": "phase-2", "capability": "text_to_speech", "status": "pending"},
                            {"phase_id": "phase-3", "capability": "image_generation", "status": "pending"},
                        ],
                    }
                },
            },
            request_payload={
                "prompt": "Write a short mystical story, then read it aloud and also show it to me as an image.",
            },
        )

        request_phase_graph = frame["planning"]["request_phase_graph"]
        artifact_flow = frame["planning"]["artifact_flow"]
        work_tree = frame["planning"]["work_tree"]
        self.assertEqual(request_phase_graph["downstream_capabilities"], ["text_to_speech", "image_generation"])
        self.assertEqual(frame["working_frame"]["request_phase_graph"]["current_phase_capability"], "chat")
        self.assertEqual(work_tree["root_node_id"], "node-request")
        self.assertEqual(work_tree["output_root_node_ids"], ["node-output-phase-1"])
        work_nodes = {item["node_id"]: item for item in work_tree["nodes"]}
        self.assertEqual(
            work_nodes["node-output-phase-1"]["child_node_ids"],
            ["node-output-phase-2", "node-output-phase-3"],
        )
        self.assertEqual(work_nodes["node-output-phase-2"]["parent_node_id"], "node-output-phase-1")
        self.assertEqual(work_nodes["node-output-phase-3"]["parent_node_id"], "node-output-phase-1")
        self.assertEqual(artifact_flow["output_slots"][0]["type"], "text")
        self.assertEqual(
            artifact_flow["output_slots"][0]["child_slot_ids"],
            ["output-phase-2", "output-phase-3"],
        )
        self.assertEqual(artifact_flow["output_slots"][1]["type"], "audio")
        self.assertEqual(artifact_flow["output_slots"][1]["parent_slot_id"], "output-phase-1")
        self.assertEqual(artifact_flow["output_slots"][1]["phase_id"], "phase-2")
        self.assertEqual(artifact_flow["output_slots"][2]["type"], "image")
        self.assertEqual(artifact_flow["output_slots"][2]["parent_slot_id"], "output-phase-1")
        self.assertEqual(artifact_flow["output_slots"][2]["phase_id"], "phase-3")
        self.assertIn("output-phase-2", artifact_flow["review"]["deferred_output_slot_ids"])
        self.assertIn("output-phase-3", artifact_flow["review"]["deferred_output_slot_ids"])

    def test_build_response_frame_derives_output_slots_from_runtime_owned_work_tree(self):
        work_tree = {
            "kind": "ollmo.work_tree",
            "status": "tracked",
            "root_node_id": "node-request",
            "node_order": ["node-request", "node-output-runtime-image"],
            "nodes": [
                {
                    "node_id": "node-request",
                    "kind": "request",
                    "role": "request_root",
                    "status": "completed",
                    "summary": "make the promised image",
                },
                {
                    "node_id": "node-output-runtime-image",
                    "kind": "output",
                    "role": "materialized_output",
                    "slot_id": "output-runtime-image",
                    "type": "image",
                    "status": "fulfilled",
                    "lifecycle": "materialized_output",
                    "artifact_ref": "artifact:image:runtime",
                    "branch_id": "branch-runtime-image",
                    "phase_id": "phase-image",
                    "parent_node_id": "node-request",
                },
            ],
            "output_root_node_ids": ["node-output-runtime-image"],
            "pending_node_ids": [],
            "blocked_node_ids": [],
        }
        frame = build_response_frame(
            {
                "id": "resp_runtime_work_tree",
                "object": "response",
                "status": "completed",
                "output_text": "Compatibility text must not own the slot tree.",
                "outputs": [{"slot_id": "stale-slot", "type": "text", "status": "fulfilled"}],
                "artifacts": [
                    {
                        "type": "image",
                        "path": "/tmp/runtime.png",
                        "artifact_ref": "artifact:image:runtime",
                    }
                ],
                "work_tree": work_tree,
            },
            request_payload={"prompt": "Describe a place and create its image."},
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        self.assertEqual(artifact_flow["work_tree_source"], "runtime_owned")
        self.assertTrue(artifact_flow["authoritative"])
        self.assertFalse(artifact_flow["compatibility_derived"])
        self.assertTrue(frame["planning"]["work_tree"]["authoritative"])
        self.assertEqual(frame["planning"]["work_tree"]["work_tree_source"], "runtime_owned")
        self.assertEqual(frame["planning"]["work_tree"]["node_order"], work_tree["node_order"])
        work_nodes = {item["node_id"]: item for item in frame["planning"]["work_tree"]["nodes"]}
        self.assertEqual(work_nodes["node-output-runtime-image"]["slot_id"], "output-runtime-image")
        self.assertEqual(work_nodes["node-output-runtime-image"]["branch_id"], "branch-runtime-image")
        self.assertEqual(len(artifact_flow["output_slots"]), 1)
        self.assertEqual(artifact_flow["output_slots"][0]["slot_id"], "output-runtime-image")
        self.assertEqual(artifact_flow["output_slots"][0]["branch_id"], "branch-runtime-image")

    def test_runtime_owned_work_tree_does_not_give_bound_artifact_to_prior_text_node(self):
        final_path = "/tmp/runtime-owned-final.json"
        phase_graph = {
            "kind": "ollmo.request_phase_graph",
            "graph_version": 2,
            "mode": "phase_chain",
            "current_phase_id": "phase-1",
            "current_phase_capability": "chat",
            "phases": [
                {"phase_id": "phase-1", "capability": "chat", "status": "completed"},
                {
                    "phase_id": "phase-5",
                    "branch_id": "branch-vision_analysis-1",
                    "capability": "vision_analysis",
                    "output_type": "text",
                    "status": "completed",
                },
                {
                    "phase_id": "phase-8",
                    "branch_id": "branch-chat-1",
                    "capability": "chat",
                    "output_type": "text",
                    "status": "completed",
                },
            ],
        }
        work_tree = {
            "kind": "ollmo.work_tree",
            "work_tree_source": "runtime_owned",
            "authoritative": True,
            "status": "tracked",
            "root_node_id": "node-request",
            "node_order": ["node-request", "node-phase-5", "node-phase-8"],
            "output_root_node_ids": ["node-phase-5", "node-phase-8"],
            "nodes": [
                {
                    "node_id": "node-request",
                    "kind": "request",
                    "role": "request_root",
                    "status": "completed",
                },
                {
                    "node_id": "node-phase-5",
                    "kind": "output",
                    "slot_id": "output-phase-5",
                    "branch_id": "branch-vision_analysis-1",
                    "phase_id": "phase-5",
                    "type": "text",
                    "status": "pending",
                    "parent_node_id": "node-request",
                },
                {
                    "node_id": "node-phase-8",
                    "kind": "output",
                    "slot_id": "output-phase-8",
                    "branch_id": "branch-chat-1",
                    "phase_id": "phase-8",
                    "type": "text",
                    "status": "pending",
                    "parent_node_id": "node-request",
                },
            ],
            "pending_node_ids": ["node-phase-5", "node-phase-8"],
            "blocked_node_ids": [],
        }
        frame = build_response_frame(
            {
                "id": "resp_runtime_owned_bound_artifact",
                "object": "response",
                "status": "completed",
                "capability": "chat",
                "output_text": "Prepared the final join.",
                "runtime": {"request_phase_graph": phase_graph},
                "work_tree": work_tree,
                "artifacts": [
                    {
                        "type": "text",
                        "path": final_path,
                        "artifact_ref": "artifact:runtime-final",
                        "branch_id": "branch-chat-1",
                        "phase_id": "phase-8",
                    }
                ],
                "late_fill": {
                    "status": "completed",
                    "fill_results": [
                        {
                            "branch_id": "branch-vision_analysis-1",
                            "phase_id": "phase-5",
                            "capability": "vision_analysis",
                            "output_type": "text",
                            "result_text": "Prior vision evidence.",
                        },
                        {
                            "branch_id": "branch-chat-1",
                            "phase_id": "phase-8",
                            "capability": "chat",
                            "output_type": "text",
                            "result_text": "Final joined evidence.",
                            "saved_text_path": final_path,
                        },
                    ],
                },
            },
            request_payload={"prompt": "Analyze, then produce one final JSON join."},
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        self.assertEqual(artifact_flow["work_tree_source"], "runtime_owned")
        work_nodes = {
            node["phase_id"]: node
            for node in artifact_flow["work_tree"]["nodes"]
            if node.get("phase_id")
        }
        slots = {slot["phase_id"]: slot for slot in artifact_flow["output_slots"]}
        self.assertNotIn("artifact_ref", work_nodes["phase-5"])
        self.assertNotIn("artifact_path", work_nodes["phase-5"])
        self.assertEqual(work_nodes["phase-8"]["artifact_ref"], "artifact:runtime-final")
        self.assertEqual(work_nodes["phase-8"]["artifact_path"], final_path)
        self.assertNotIn("artifact_ref", slots["phase-5"])
        self.assertEqual(slots["phase-8"]["artifact_ref"], "artifact:runtime-final")
        self.assertEqual(
            [slot.get("artifact_ref") for slot in artifact_flow["output_slots"]].count(
                "artifact:runtime-final"
            ),
            1,
        )

    def test_build_response_frame_marks_derived_work_tree_as_non_authoritative(self):
        frame = build_response_frame(
            {
                "id": "resp_derived_work_tree_marker",
                "object": "response",
                "status": "completed",
                "output_text": "Compatibility text only.",
            },
            request_payload={"prompt": "say hello"},
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        work_tree = frame["planning"]["work_tree"]
        self.assertEqual(artifact_flow["work_tree_source"], "derived_planning_snapshot")
        self.assertFalse(artifact_flow["authoritative"])
        self.assertTrue(artifact_flow["compatibility_derived"])
        self.assertEqual(work_tree["work_tree_source"], "derived_planning_snapshot")
        self.assertFalse(work_tree["authoritative"])
        self.assertTrue(work_tree["compatibility_derived"])
        self.assertEqual(artifact_flow.get("output_slots", []), [])

    def test_build_response_frame_preserves_runtime_work_tree_across_late_fill_successor(self):
        work_tree = {
            "kind": "ollmo.work_tree",
            "status": "pending",
            "root_node_id": "node-request",
            "node_order": ["node-request", "node-output-phase-1", "node-output-phase-2"],
            "nodes": [
                {"node_id": "node-request", "kind": "request", "role": "request_root", "status": "active"},
                {
                    "node_id": "node-output-phase-1",
                    "kind": "output",
                    "slot_id": "output-phase-1",
                    "type": "text",
                    "status": "fulfilled",
                    "parent_node_id": "node-request",
                    "child_node_ids": ["node-output-phase-2"],
                },
                {
                    "node_id": "node-output-phase-2",
                    "kind": "output",
                    "slot_id": "output-phase-2",
                    "type": "image",
                    "status": "pending",
                    "branch_id": "branch-image",
                    "phase_id": "phase-2",
                    "parent_node_id": "node-output-phase-1",
                },
            ],
            "output_root_node_ids": ["node-output-phase-1"],
            "pending_node_ids": ["node-output-phase-2"],
            "blocked_node_ids": [],
        }
        initial_frame = build_response_frame(
            {
                "id": "resp_work_tree_successor",
                "object": "response",
                "status": "completed",
                "output_text": "Initial text.",
                "work_tree": work_tree,
            },
            request_payload={"prompt": "Write a scene, then make an image."},
        )
        successor_payload = response_payload_from_frame(initial_frame)
        successor_payload["late_fill"] = {
            "status": "completed",
            "completed_branches": [
                {
                    "branch_id": "branch-image",
                    "phase_id": "phase-2",
                    "capability": "image_generation",
                    "output_type": "image",
                }
            ],
        }
        successor_payload["artifacts"] = [{"type": "image", "path": "/tmp/successor.png"}]

        successor_frame = build_response_frame(
            successor_payload,
            request_payload={"prompt": "Write a scene, then make an image."},
        )

        self.assertEqual(successor_frame["planning"]["artifact_flow"]["work_tree_source"], "runtime_owned")
        self.assertTrue(successor_frame["planning"]["artifact_flow"]["authoritative"])
        self.assertTrue(successor_frame["planning"]["work_tree"]["authoritative"])
        self.assertEqual(successor_frame["planning"]["work_tree"]["node_order"], work_tree["node_order"])
        successor_nodes = {item["node_id"]: item for item in successor_frame["planning"]["work_tree"]["nodes"]}
        self.assertEqual(successor_nodes["node-output-phase-2"]["branch_id"], "branch-image")
        self.assertEqual(successor_nodes["node-output-phase-2"]["status"], "fulfilled")
        self.assertEqual(successor_nodes["node-output-phase-2"]["artifact_path"], "/tmp/successor.png")
        self.assertEqual(
            successor_frame["planning"]["artifact_flow"]["output_slots"][1]["branch_id"],
            "branch-image",
        )

    def test_build_response_frame_does_not_regenerate_runtime_work_tree_from_prose_fallbacks(self):
        empty_work_tree = {
            "kind": "ollmo.work_tree",
            "status": "empty",
            "root_node_id": "node-request",
            "node_order": ["node-request"],
            "nodes": [
                {"node_id": "node-request", "kind": "request", "role": "request_root", "status": "completed"}
            ],
            "output_root_node_ids": [],
            "pending_node_ids": [],
            "blocked_node_ids": [],
        }
        frame = build_response_frame(
            {
                "id": "resp_empty_runtime_tree",
                "object": "response",
                "status": "completed",
                "output_text": "Compatibility-only text.",
                "work_tree": empty_work_tree,
            },
            request_payload={"prompt": "hello"},
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        self.assertEqual(artifact_flow["work_tree_source"], "runtime_owned")
        self.assertTrue(artifact_flow["authoritative"])
        self.assertEqual(artifact_flow.get("output_slots", []), [])
        self.assertEqual(frame["planning"]["work_tree"]["node_order"], empty_work_tree["node_order"])
        self.assertEqual(frame["planning"]["work_tree"]["nodes"], empty_work_tree["nodes"])

    def test_build_response_frame_projects_explicit_late_fill_branches_into_child_slots(self):
        frame = build_response_frame(
            {
                "id": "resp_two_images",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Two dream places unfolded.",
                "late_fill": {
                    "status": "completed",
                    "expected_capability": "image_generation",
                    "pending_branches": [],
                    "completed_branches": [
                        {
                            "branch_id": "phase-image-1",
                            "phase_id": "phase-image-1",
                            "capability": "image_generation",
                            "output_type": "image",
                        },
                        {
                            "branch_id": "phase-image-2",
                            "phase_id": "phase-image-2",
                            "capability": "image_generation",
                            "output_type": "image",
                        },
                    ],
                    "fill_results": [
                        {"branch_id": "phase-image-1", "phase_id": "phase-image-1", "capability": "image_generation"},
                        {"branch_id": "phase-image-2", "phase_id": "phase-image-2", "capability": "image_generation"},
                    ],
                },
                "artifacts": [
                    {"type": "image", "path": "artifacts/images/one.png"},
                    {"type": "image", "path": "artifacts/images/two.png"},
                ],
            },
            request_payload={
                "prompt": "Describe two dream places and show them to me as two separate images.",
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        work_tree = frame["planning"]["work_tree"]
        self.assertEqual(artifact_flow["output_slots"][0]["type"], "text")
        self.assertEqual(
            artifact_flow["output_slots"][0]["child_slot_ids"],
            ["output-phase-image-1", "output-phase-image-2"],
        )
        self.assertEqual(work_tree["output_root_node_ids"], ["node-output-phase-1"])
        work_nodes = {item["node_id"]: item for item in work_tree["nodes"]}
        self.assertEqual(
            work_nodes["node-output-phase-1"]["child_node_ids"],
            ["node-output-phase-image-1", "node-output-phase-image-2"],
        )
        self.assertEqual(artifact_flow["output_slots"][1]["branch_id"], "phase-image-1")
        self.assertEqual(artifact_flow["output_slots"][1]["parent_slot_id"], "output-phase-1")
        self.assertEqual(artifact_flow["output_slots"][1]["status"], "fulfilled")
        self.assertEqual(artifact_flow["output_slots"][2]["branch_id"], "phase-image-2")
        self.assertEqual(artifact_flow["output_slots"][2]["parent_slot_id"], "output-phase-1")
        self.assertEqual(artifact_flow["output_slots"][2]["status"], "fulfilled")

    def test_build_response_frame_keeps_same_capability_branches_distinct_when_only_one_sibling_is_done(self):
        frame = build_response_frame(
            {
                "id": "resp_two_images_partial",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Two dream places unfolded.",
                "runtime": {
                    "request_phase_graph": {
                        "graph_version": 2,
                        "kind": "ollmo.request_phase_graph",
                        "mode": "phase_chain",
                        "current_phase_id": "phase-1",
                        "current_phase_capability": "chat",
                        "current_phase_resolution": "graph_resolved",
                        "downstream_phase_ids": ["phase-image-1", "phase-image-2"],
                        "downstream_branch_ids": ["phase-image-1", "phase-image-2"],
                        "downstream_branches": [
                            {
                                "branch_id": "phase-image-1",
                                "phase_id": "phase-image-1",
                                "capability": "image_generation",
                                "output_type": "image",
                                "depends_on": ["phase-1"],
                            },
                            {
                                "branch_id": "phase-image-2",
                                "phase_id": "phase-image-2",
                                "capability": "image_generation",
                                "output_type": "image",
                                "depends_on": ["phase-1"],
                            },
                        ],
                        "phases": [
                            {"phase_id": "phase-1", "capability": "chat", "status": "completed"},
                            {"phase_id": "phase-image-1", "branch_id": "phase-image-1", "capability": "image_generation", "status": "pending"},
                            {"phase_id": "phase-image-2", "branch_id": "phase-image-2", "capability": "image_generation", "status": "pending"},
                        ],
                    }
                },
                "late_fill": {
                    "status": "running",
                    "expected_capability": "image_generation",
                    "completed_branches": [
                        {
                            "branch_id": "phase-image-1",
                            "phase_id": "phase-image-1",
                            "capability": "image_generation",
                            "output_type": "image",
                        }
                    ],
                    "pending_branches": [
                        {
                            "branch_id": "phase-image-2",
                            "phase_id": "phase-image-2",
                            "capability": "image_generation",
                            "output_type": "image",
                        }
                    ],
                },
                "artifacts": [
                    {"type": "image", "path": "artifacts/images/one.png"},
                ],
            },
            request_payload={
                "prompt": "Describe two dream places and show them to me as two separate images.",
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        image_slots = [slot for slot in artifact_flow["output_slots"] if slot["type"] == "image"]
        self.assertEqual(len(image_slots), 2)
        self.assertEqual(image_slots[0]["branch_id"], "phase-image-1")
        self.assertEqual(image_slots[0]["status"], "fulfilled")
        self.assertEqual(image_slots[1]["branch_id"], "phase-image-2")
        self.assertEqual(image_slots[1]["status"], "pending")
        self.assertEqual(image_slots[1]["lifecycle"], "deferred_output")

    def test_build_response_frame_rebuilds_outputs_from_fulfilled_slots_even_when_payload_outputs_are_stale(self):
        frame = build_response_frame(
            {
                "id": "resp_outputs_rebuilt_from_slots",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Two dream places unfolded.",
                "outputs": [
                    {"type": "text", "status": "fulfilled", "value": "Two dream places unfolded."},
                    {"type": "image", "status": "pending", "slot_id": "output-phase-image-1"},
                    {"type": "image", "status": "pending", "slot_id": "output-phase-image-2"},
                ],
                "runtime": {
                    "request_phase_graph": {
                        "graph_version": 3,
                        "kind": "ollmo.request_phase_graph",
                        "mode": "phase_chain",
                        "current_phase_id": "phase-1",
                        "current_phase_capability": "chat",
                        "current_phase_resolution": "graph_resolved",
                        "downstream_phase_ids": ["phase-image-1", "phase-image-2"],
                        "downstream_branch_ids": ["phase-image-1", "phase-image-2"],
                        "downstream_branches": [
                            {
                                "branch_id": "phase-image-1",
                                "phase_id": "phase-image-1",
                                "capability": "image_generation",
                                "output_type": "image",
                                "depends_on": ["phase-1"],
                            },
                            {
                                "branch_id": "phase-image-2",
                                "phase_id": "phase-image-2",
                                "capability": "image_generation",
                                "output_type": "image",
                                "depends_on": ["phase-1"],
                            },
                        ],
                        "phases": [
                            {"phase_id": "phase-1", "capability": "chat", "status": "completed"},
                            {"phase_id": "phase-image-1", "branch_id": "phase-image-1", "capability": "image_generation", "status": "pending"},
                            {"phase_id": "phase-image-2", "branch_id": "phase-image-2", "capability": "image_generation", "status": "pending"},
                        ],
                    }
                },
                "late_fill": {
                    "status": "completed",
                    "expected_capability": "image_generation",
                    "completed_branches": [
                        {
                            "branch_id": "phase-image-1",
                            "phase_id": "phase-image-1",
                            "capability": "image_generation",
                            "output_type": "image",
                        },
                        {
                            "branch_id": "phase-image-2",
                            "phase_id": "phase-image-2",
                            "capability": "image_generation",
                            "output_type": "image",
                        },
                    ],
                    "fill_results": [
                        {"branch_id": "phase-image-1", "phase_id": "phase-image-1", "capability": "image_generation"},
                        {"branch_id": "phase-image-2", "phase_id": "phase-image-2", "capability": "image_generation"},
                    ],
                },
                "artifacts": [
                    {"type": "image", "path": "artifacts/images/one.png"},
                    {"type": "image", "path": "artifacts/images/two.png"},
                ],
            },
            request_payload={
                "prompt": "Describe two dream places and show them to me as two separate images.",
            },
        )

        outputs = frame["output"]["outputs"]
        self.assertEqual(frame["output"]["item_count"], len(outputs))
        self.assertEqual(outputs[1]["status"], "fulfilled")
        self.assertEqual(outputs[2]["status"], "fulfilled")
        self.assertIn("artifact_ref", outputs[1])
        self.assertIn("artifact_ref", outputs[2])
        self.assertNotEqual(outputs[1]["artifact_ref"], outputs[2]["artifact_ref"])
        self.assertNotIn("artifacts", outputs[1])
        self.assertNotIn("artifacts", outputs[2])

    def test_build_response_frame_does_not_add_legacy_duplicate_follow_up_slot_when_graph_branch_exists(self):
        frame = build_response_frame(
            {
                "id": "resp_single_image_no_duplicate_slot",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "A single image prompt was prepared.",
                "runtime": {
                    "request_phase_graph": {
                        "graph_version": 3,
                        "kind": "ollmo.request_phase_graph",
                        "mode": "phase_chain",
                        "current_phase_id": "phase-1",
                        "current_phase_capability": "chat",
                        "current_phase_resolution": "graph_resolved",
                        "downstream_phase_ids": ["phase-image-1"],
                        "downstream_branch_ids": ["phase-image-1"],
                        "downstream_branches": [
                            {
                                "branch_id": "phase-image-1",
                                "phase_id": "phase-image-1",
                                "capability": "image_generation",
                                "output_type": "image",
                                "depends_on": ["phase-1"],
                            }
                        ],
                        "phases": [
                            {"phase_id": "phase-1", "capability": "chat", "status": "completed"},
                            {"phase_id": "phase-image-1", "branch_id": "phase-image-1", "capability": "image_generation", "status": "pending"},
                        ],
                    }
                },
                "late_fill": {
                    "status": "pending",
                    "expected_capability": "image_generation",
                },
            },
            request_payload={
                "prompt": "Describe a dream place and then show it as an image.",
            },
        )

        output_slots = frame["planning"]["artifact_flow"]["output_slots"]
        self.assertEqual(len(output_slots), 2)
        self.assertEqual(output_slots[0]["slot_id"], "output-phase-1")
        self.assertEqual(output_slots[1]["slot_id"], "output-phase-image-1")

    def test_build_response_frame_preserves_two_completed_image_branches_without_duplicate_slots(self):
        frame = build_response_frame(
            {
                "id": "resp_two_image_branches",
                "object": "response",
                "status": "completed",
                "model": "gemma",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "Two image prompts were prepared.",
                "artifacts": [
                    {"type": "image", "path": "artifacts/images/one.png", "artifact_ref": "artifact:image_one"},
                    {"type": "image", "path": "artifacts/images/two.png", "artifact_ref": "artifact:image_two"},
                ],
                "runtime": {
                    "request_phase_graph": {
                        "graph_version": 3,
                        "kind": "ollmo.request_phase_graph",
                        "mode": "carried_phase_chain",
                        "current_phase_id": "phase-1",
                        "current_phase_capability": "chat",
                        "current_phase_resolution": "graph_resolved",
                        "downstream_phase_ids": ["phase-2", "phase-3"],
                        "downstream_branch_ids": ["branch-image_generation-1", "branch-image_generation-2"],
                        "downstream_branches": [
                            {
                                "branch_id": "branch-image_generation-1",
                                "phase_id": "phase-2",
                                "capability": "image_generation",
                                "output_type": "image",
                            },
                            {
                                "branch_id": "branch-image_generation-2",
                                "phase_id": "phase-3",
                                "capability": "image_generation",
                                "output_type": "image",
                            },
                        ],
                        "phases": [
                            {"phase_id": "phase-1", "capability": "chat", "status": "completed"},
                            {
                                "phase_id": "phase-2",
                                "branch_id": "branch-image_generation-1",
                                "capability": "image_generation",
                                "status": "pending",
                            },
                            {
                                "phase_id": "phase-3",
                                "branch_id": "branch-image_generation-2",
                                "capability": "image_generation",
                                "status": "pending",
                            },
                        ],
                    }
                },
                "late_fill": {
                    "status": "completed",
                    "expected_capability": "image_generation",
                    "completed_branches": [
                        {
                            "branch_id": "branch-image_generation-1",
                            "phase_id": "phase-2",
                            "capability": "image_generation",
                            "output_type": "image",
                        },
                        {
                            "branch_id": "branch-image_generation-2",
                            "phase_id": "phase-3",
                            "capability": "image_generation",
                            "output_type": "image",
                        },
                    ],
                },
            },
            request_payload={
                "prompt": "Please create two more images and show both.",
            },
        )

        artifact_flow = frame["planning"]["artifact_flow"]
        image_slots = [slot for slot in artifact_flow["output_slots"] if slot["type"] == "image"]
        self.assertEqual(len(image_slots), 2)
        self.assertEqual(
            {slot["branch_id"] for slot in image_slots},
            {"branch-image_generation-1", "branch-image_generation-2"},
        )
        self.assertEqual(
            [artifact["path"] for artifact in frame["artifacts"]["output"]],
            ["artifacts/images/one.png", "artifacts/images/two.png"],
        )
        work_tree_image_nodes = [
            node for node in artifact_flow["work_tree"]["nodes"]
            if node.get("type") == "image" and node.get("status") == "fulfilled"
        ]
        self.assertEqual(len(work_tree_image_nodes), 2)

    def test_build_response_frame_omits_inline_image_bytes_from_durable_payload(self):
        frame = build_response_frame(
            {
                "id": "resp_image_bytes",
                "object": "response",
                "status": "completed",
                "model": "flux",
                "instance_id": "image-1",
                "backend": "ollama",
                "capability": "image_generation",
                "mode": "image_generation",
                "output_text": "Image generated.",
                "artifacts": [
                    {
                        "type": "image",
                        "path": "artifacts/images/out.png",
                        "image_data_url": "data:image/png;base64,ZmFrZQ==",
                    }
                ],
                "input_artifacts": [
                    {
                        "type": "image",
                        "path": "artifacts/inputs/ref.png",
                        "image_data_url": "data:image/png;base64,ZmFrZQ==",
                    }
                ],
            },
            request_payload={
                "input": "make it cinematic",
                "selected_reference_artifacts": [
                    {
                        "type": "image",
                        "path": "artifacts/inputs/ref.png",
                        "image_data_url": "data:image/png;base64,ZmFrZQ==",
                    }
                ],
            },
        )

        self.assertEqual(frame["artifacts"]["output"][0]["path"], "artifacts/images/out.png")
        self.assertEqual(frame["artifacts"]["input"][0]["path"], "artifacts/inputs/ref.png")
        self.assertNotIn("image_data_url", frame["artifacts"]["output"][0])
        self.assertNotIn("image_data_url", frame["artifacts"]["input"][0])
        self.assertNotIn("image_data_url", json.dumps(frame))

    def test_build_response_frame_attaches_control_snapshot_when_meaningful(self):
        frame = build_response_frame(
            {
                "id": "resp_controls",
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
                "input": "make a square cover",
                "width": "1024",
                "height": "1024",
                "seed": "1234",
            },
        )

        self.assertEqual(frame["controls"]["kind"], "ollmo.control_snapshot")
        self.assertEqual(frame["controls"]["target"]["capability"], "image_generation")
        self.assertEqual(frame["controls"]["values"]["image"]["width"], 1024)
        self.assertEqual(frame["controls"]["values"]["image"]["height"], 1024)
        self.assertEqual(frame["controls"]["values"]["image"]["seed"], 1234)

    def test_build_response_frame_exposes_artifact_dossier_with_provenance_and_enrichment(self):
        frame = build_response_frame(
            {
                "id": "resp_dossier_image",
                "object": "response",
                "status": "completed",
                "model": "flux",
                "instance_id": "image-1",
                "backend": "mlx",
                "capability": "image_generation",
                "mode": "image_generation",
                "output_text": "Image generated.",
                "saved_image_path": "artifacts/images/out.png",
                "provenance_id": "prov_image_1",
                "seed": 1234,
                "image_state": {"subject": "dream temple"},
                "image_state_enrichment": {"status": "completed", "mode": "background_analysis"},
                "artifacts": [
                    {
                        "type": "image",
                        "path": "artifacts/images/out.png",
                        "provenance_id": "prov_image_1",
                        "seed": 1234,
                    }
                ],
            },
            request_payload={"input": "show me the place as an image"},
        )

        dossiers = frame["artifacts"]["dossiers"]
        dossier = next(iter(dossiers.values()))
        self.assertEqual(dossier["type"], "image")
        self.assertEqual(dossier["roles"], ["output"])
        self.assertEqual(dossier["provenance"]["provenance_id"], "prov_image_1")
        self.assertEqual(dossier["metadata"]["seed"], 1234)
        self.assertEqual(dossier["enrichments"]["image_state"]["subject"], "dream temple")
        self.assertEqual(dossier["enrichments"]["image_state_enrichment"]["status"], "completed")

    def test_build_response_frame_freezes_provided_working_frame_snapshot(self):
        frame = build_response_frame(
            {
                "id": "resp_working",
                "object": "response",
                "status": "completed",
                "model": "gpt-oss:20b",
                "instance_id": "chat-1",
                "backend": "ollama",
                "capability": "chat",
                "mode": "chat",
                "output_text": "done",
                "runtime": {
                    "working_frame": {
                        "kind": "ollmo.working_frame",
                        "status": "frozen",
                        "loop": {"chain_id": "conv-1", "pass_index": 1},
                    }
                },
                "working_frame": {
                    "kind": "ollmo.working_frame",
                    "status": "frozen",
                    "loop": {"chain_id": "conv-1", "pass_index": 1},
                },
            },
            request_payload={"conversation_id": "conv-1", "prompt": "finish"},
        )

        self.assertEqual(frame["working_frame"]["loop"]["chain_id"], "conv-1")
        self.assertNotIn("runtime", frame)

    def test_attach_response_frame_returns_payload_copy(self):
        payload = {
            "id": "resp_abc",
            "object": "response",
            "status": "completed",
            "model": "flux",
            "instance_id": "flux-1",
            "backend": "mlx",
            "capability": "image_generation",
            "mode": "image_generation",
            "output_text": "Generated image.",
            "output": [],
        }

        framed = attach_response_frame(payload, request_payload={"prompt": "a fox"})

        self.assertNotIn("response_frame", payload)
        self.assertEqual(framed["response_frame"]["request"]["prompt"], "a fox")
        self.assertEqual(framed["response_frame"]["target"]["capability"], "image_generation")

    def test_build_response_frame_compacts_duplicate_ghost_preview_orchestration_truth(self):
        decision_contract = {
            "kind": "ollmo.decision_contract",
            "semantic_planning_contract": {
                "evidence": "repeated semantic planning evidence " * 3_000,
            },
        }
        phase_graph = {
            "kind": "ollmo.request_phase_graph",
            "request_ir": {"decision_contract": decision_contract},
            "decision_contract": decision_contract,
        }
        working_frame = {
            "kind": "ollmo.working_frame",
            "status": "fluid",
            "request_phase_graph": phase_graph,
            "intent_contract": {"decision_contract": decision_contract},
        }
        ghost_preview = {
            "instance": {
                "instance_id": "chat-ghost-1",
                "model": "local-chat",
                "backend": "mlx",
                "capability": "chat",
                "supported_capabilities": ["chat", "vision_analysis"],
                "oversized-" + ("x" * 512): "client metadata " * 10_000,
            },
            "route": {
                "source": "router",
                "reason": "semantic route selected local chat",
                "confidence": 0.97,
            },
            "request_meta": {
                "ghost_mode": "improviser",
                "capability_hint": "chat",
                "developer_flags": {"planner_timeout_ms": 12_000},
            },
            "runtime": {
                "developer_diagnostics": {
                    "routing_contract": "ghost_primary",
                    "routing_policy": "ghost_first",
                    "planner_timeout_ms": 12_000,
                    "route_graph_consistency": {
                        "status": "accepted",
                        "final_graph_source": "consistency_enforced",
                    },
                    "oversized-" + ("y" * 512): "unbounded diagnostic " * 10_000,
                    "graph_closure_review": {"decision_contract": decision_contract},
                },
                "working_frame": working_frame,
                "request_phase_graph": phase_graph,
                "execution_planner": {"phase_graph": phase_graph},
            },
            "working_frame": working_frame,
        }
        raw_preview_size = len(json.dumps(ghost_preview).encode("utf-8"))

        frame = build_response_frame(
            {
                "id": "resp_compact_ghost_preview",
                "object": "response",
                "status": "completed",
                "output_text": "done",
            },
            request_payload={
                "prompt": "finish the routed request",
                "ghost_route": True,
                "ghost_preview": ghost_preview,
            },
        )

        compact_preview = frame["request"]["ghost_preview"]
        compact_runtime = compact_preview["runtime"]
        diagnostics = compact_runtime["developer_diagnostics"]
        compact_preview_size = len(json.dumps(compact_preview).encode("utf-8"))
        self.assertEqual(compact_preview["instance"]["instance_id"], "chat-ghost-1")
        self.assertEqual(compact_preview["route"]["source"], "router")
        self.assertEqual(compact_preview["request_meta"]["ghost_mode"], "improviser")
        self.assertEqual(diagnostics["routing_contract"], "ghost_primary")
        self.assertEqual(diagnostics["route_graph_consistency"]["status"], "accepted")
        self.assertTrue(all(len(key) <= 128 for key in compact_preview["instance"]))
        self.assertTrue(all(len(key) <= 128 for key in diagnostics))
        self.assertNotIn("graph_closure_review", diagnostics)
        self.assertNotIn("working_frame", compact_preview)
        self.assertNotIn("working_frame", compact_runtime)
        self.assertNotIn("request_phase_graph", compact_runtime)
        self.assertNotIn("execution_planner", compact_runtime)
        self.assertTrue(compact_preview["compaction"]["duplicate_orchestration_truth_omitted"])
        content_refs = {
            item["json_path"]: item
            for item in compact_preview["compaction"]["omitted_content_refs"]
        }
        expected_runtime_digest = hashlib.sha256(
            json.dumps(
                ghost_preview["runtime"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            content_refs["ghost_preview.runtime"]["sha256"],
            expected_runtime_digest,
        )
        self.assertIn("ghost_preview.working_frame", content_refs)
        self.assertLess(compact_preview_size, 16_384)
        self.assertLess(compact_preview_size, raw_preview_size // 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            target = persist_response_frame(frame, frames_dir=frames_dir)
            persisted_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            recovered = load_latest_response_state(
                "resp_compact_ghost_preview",
                frames_dir=frames_dir,
            )

        persisted_refs = {
            item["json_path"]: item
            for item in persisted_frame["request"]["ghost_preview"]["compaction"]["omitted_content_refs"]
        }
        recovered_refs = {
            item["json_path"]: item
            for item in recovered["response_frame"]["request"]["ghost_preview"]["compaction"]["omitted_content_refs"]
        }
        self.assertEqual(persisted_refs["ghost_preview.runtime"]["sha256"], expected_runtime_digest)
        self.assertEqual(recovered_refs["ghost_preview.runtime"]["sha256"], expected_runtime_digest)

    def test_ghost_preview_compaction_reports_only_actual_omissions(self):
        compact_preview = response_frames_module._compact_request_ghost_preview(
            {
                "custom_scalar": "retained audit value",
                "instance": {"instance_id": "route-1", "capability": "chat"},
                "route": {"source": "router", "confidence": 0.75},
                "request_meta": {"ghost_mode": "assistant"},
                "runtime": {
                    "developer_diagnostics": {"routing_contract": "ghost_primary"},
                },
            }
        )

        self.assertEqual(compact_preview["custom_scalar"], "retained audit value")
        self.assertEqual(compact_preview["compaction"]["omitted_preview_key_count"], 0)
        self.assertNotIn("omitted_preview_keys", compact_preview["compaction"])
        self.assertFalse(compact_preview["compaction"]["duplicate_orchestration_truth_omitted"])
        self.assertFalse(compact_preview["compaction"]["preview_detail_omitted"])

        nested_preview = response_frames_module._compact_request_ghost_preview(
            {
                "instance": {
                    "instance_id": "route-2",
                    "features": {"vision_input": True},
                },
                "route": {
                    "source": "router",
                    "candidates": [{"instance_id": "candidate-1"}],
                },
                "request_meta": {
                    "ghost_mode": "assistant",
                    "nested_contract": {"status": "advisory"},
                },
            }
        )
        nested_compaction = nested_preview["compaction"]
        self.assertEqual(nested_compaction["omitted_nested_detail_count"], 3)
        self.assertEqual(
            nested_compaction["omitted_nested_detail_paths"],
            [
                "ghost_preview.instance.features",
                "ghost_preview.request_meta.nested_contract",
                "ghost_preview.route.candidates",
            ],
        )
        self.assertTrue(nested_compaction["preview_detail_omitted"])
        self.assertFalse(nested_compaction["duplicate_orchestration_truth_omitted"])
        self.assertEqual(
            {item["json_path"] for item in nested_compaction["omitted_content_refs"]},
            {
                "ghost_preview.instance",
                "ghost_preview.request_meta",
                "ghost_preview.route",
            },
        )

    def test_persist_response_frame_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = persist_response_frame(
                {"frame_version": 1, "kind": "ollmo.response_frame", "response_id": "resp_1"},
                frames_dir=Path(tmpdir),
            )

            self.assertTrue(target.exists())
            lines = target.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["response_id"], "resp_1")

    def test_persist_response_frame_fsyncs_ledger_before_index_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ordering = []

            def record_fsync(_file_descriptor):
                ordering.append("ledger_fsync")

            def record_index(*_args, **_kwargs):
                ordering.append("index_commit")

            with patch.object(
                response_frames_module.os,
                "fsync",
                side_effect=record_fsync,
            ), patch.object(
                response_frames_module,
                "_write_response_frame_index",
                side_effect=record_index,
            ):
                persist_response_frame(
                    {
                        "frame_version": 1,
                        "kind": "ollmo.response_frame",
                        "response_id": "resp_fsync_order",
                    },
                    frames_dir=Path(tmpdir),
                )

            self.assertIn("ledger_fsync", ordering)
            self.assertIn("index_commit", ordering)
            self.assertLess(
                ordering.index("ledger_fsync"),
                ordering.index("index_commit"),
            )

    def test_persist_response_frame_externalizes_large_runtime_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            large_graph = {
                "nodes": [
                    {"id": f"node-{index}", "summary": "planner truth " * 120}
                    for index in range(80)
                ],
                "edges": [
                    {"from": f"node-{index}", "to": f"node-{index + 1}"}
                    for index in range(79)
                ],
            }
            frame = build_response_frame(
                {
                    "id": "resp_compact_ledger",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Compact ledger keeps refs.",
                    "runtime": {
                        "request_phase_graph": large_graph,
                        "execution_planner": {"graph": large_graph},
                        "graph_closure_review": {"graph": large_graph},
                        "developer_diagnostics": {
                            "planner_timeout_ms": 60000,
                            "graph_closure_review": {"graph": large_graph},
                        },
                    },
                    "late_fill": {
                        "status": "running",
                        "pending_branches": [{"branch_id": "branch-image-1", "capability": "image_generation"}],
                        "controlled_attention_review": {"graph": large_graph},
                        "semantic_decision_review": {"graph": large_graph},
                    },
                },
                request_payload={"prompt": "answer then generate image"},
            )
            raw_frame_size = len(json.dumps(frame, ensure_ascii=False, sort_keys=True).encode("utf-8"))

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_line = target.read_text(encoding="utf-8").splitlines()[0]
            ledger_frame = json.loads(ledger_line)
            index_entry = load_response_frame_index(frames_dir=frames_dir)["responses"]["resp_compact_ledger"]
            recovered = load_latest_response_state("resp_compact_ledger", frames_dir=frames_dir)
            runtime_ref = ledger_frame["runtime_snapshot_ref"]
            current_runtime_ref = ledger_frame["current_state"]["runtime_snapshot_ref"]
            planning_graph_ref = ledger_frame["planning"]["request_phase_graph_snapshot_ref"]
            artifact_flow_graph_ref = ledger_frame["planning"]["artifact_flow"]["request_phase_graph_snapshot_ref"]
            late_fill_ref = ledger_frame["late_fill_snapshot_ref"]
            current_late_fill_ref = ledger_frame["current_state"]["late_fill_snapshot_ref"]
            runtime_snapshot_path = frames_dir / runtime_ref["path"]
            runtime_snapshot_exists = runtime_snapshot_path.exists()
            runtime_snapshot_payload = json.loads(runtime_snapshot_path.read_text(encoding="utf-8"))
            runtime_snapshot_payload_size = runtime_snapshot_path.stat().st_size - 1
            snapshot_refs = ledger_frame["external_snapshots"]["items"]
            unique_snapshot_paths = {
                ref["path"]
                for ref in snapshot_refs.values()
                if isinstance(ref, dict) and ref.get("path")
            }

        self.assertLess(len(ledger_line.encode("utf-8")), raw_frame_size // 2)
        self.assertEqual(ledger_frame["snapshot_policy"]["ledger_payload"], "compact")
        self.assertEqual(ledger_frame["snapshot_policy"]["dedupe_strategy"], "content_sha256")
        self.assertEqual(ledger_frame["snapshot_policy"]["snapshot_ref_count"], len(snapshot_refs))
        self.assertEqual(ledger_frame["snapshot_policy"]["unique_snapshot_count"], len(unique_snapshot_paths))
        self.assertIn("runtime_snapshot_ref", ledger_frame)
        self.assertIn("working_frame_snapshot_ref", ledger_frame)
        self.assertIn("request_phase_graph_snapshot_ref", ledger_frame["planning"])
        self.assertIn("request_phase_graph_snapshot_ref", ledger_frame["planning"]["artifact_flow"])
        self.assertEqual(runtime_ref["path"], current_runtime_ref["path"])
        self.assertEqual(planning_graph_ref["path"], artifact_flow_graph_ref["path"])
        self.assertEqual(late_fill_ref["path"], current_late_fill_ref["path"])
        self.assertLess(len(unique_snapshot_paths), len(snapshot_refs))
        self.assertTrue(runtime_ref["content_addressed"])
        self.assertIn("request_phase_graph_snapshot_ref", runtime_snapshot_payload)
        self.assertIn("execution_planner", runtime_snapshot_payload)
        self.assertIn("graph_snapshot_ref", runtime_snapshot_payload["execution_planner"])
        self.assertIn("graph_closure_review_snapshot_ref", runtime_snapshot_payload)
        self.assertNotIn("request_phase_graph", runtime_snapshot_payload)
        self.assertNotIn("graph", runtime_snapshot_payload["execution_planner"])
        self.assertNotIn("graph_closure_review", runtime_snapshot_payload)
        self.assertIn("runtime.execution_planner.graph", snapshot_refs)
        self.assertNotIn("request_phase_graph", ledger_frame["planning"])
        self.assertNotIn("request_phase_graph", ledger_frame["planning"]["artifact_flow"])
        self.assertIn("review_snapshot_ref", ledger_frame["late_fill"])
        self.assertNotIn("controlled_attention_review", ledger_frame["late_fill"])
        self.assertTrue(runtime_snapshot_exists)
        self.assertEqual(runtime_snapshot_payload_size, runtime_ref["size_bytes"])
        self.assertIn("byte_offset", index_entry)
        self.assertIn("ledger_size_bytes", index_entry)
        self.assertTrue(recovered["ok"])
        self.assertTrue(recovered["index_used"])
        self.assertFalse(recovered["ledger_fallback_used"])
        self.assertEqual(recovered["response_payload"]["late_fill"]["status"], "running")
        recovered_runtime = recovered["response_payload"]["runtime"]
        self.assertEqual(recovered_runtime["request_phase_graph"]["nodes"], large_graph["nodes"])
        self.assertEqual(recovered_runtime["request_phase_graph"]["edges"], large_graph["edges"])
        self.assertEqual(recovered_runtime["graph_closure_review"]["graph"], large_graph)
        self.assertEqual(recovered_runtime["developer_diagnostics"]["planner_timeout_ms"], 60000)
        self.assertEqual(
            recovered_runtime["developer_diagnostics"]["graph_closure_review"]["graph"],
            large_graph,
        )
        self.assertEqual(
            recovered["response_payload"]["response_frame"]["runtime_snapshot_ref"]["sha256"],
            runtime_ref["sha256"],
        )

    def test_persist_response_frame_sidecars_legacy_full_ghost_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            repeated_contract = {
                "kind": "ollmo.decision_contract",
                "evidence": "legacy repeated decision evidence " * 4_000,
            }
            phase_graph = {
                "kind": "ollmo.request_phase_graph",
                "request_ir": {"decision_contract": repeated_contract},
                "decision_contract": repeated_contract,
            }
            legacy_preview = {
                "instance": {
                    "instance_id": "legacy-route-1",
                    "model": "legacy-model",
                    "backend": "mlx",
                    "capability": "chat",
                },
                "route": {"source": "legacy_client", "confidence": 0.8},
                "request_meta": {"ghost_mode": "assistant", "capability_hint": "chat"},
                "runtime": {
                    "developer_diagnostics": {"routing_contract": "ghost_primary"},
                    "request_phase_graph": phase_graph,
                    "execution_planner": {"phase_graph": phase_graph},
                },
                "working_frame": {"request_phase_graph": phase_graph},
            }
            frame = {
                "frame_version": 9,
                "kind": "ollmo.response_frame",
                "response_id": "resp_legacy_full_ghost_preview",
                "status": "completed",
                "object": "response",
                "request": {"ghost_preview": legacy_preview},
                "current_state": {
                    "id": "resp_legacy_full_ghost_preview",
                    "status": "completed",
                },
            }
            raw_frame_size = len(json.dumps(frame).encode("utf-8"))

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_line = target.read_text(encoding="utf-8").splitlines()[0]
            ledger_frame = json.loads(ledger_line)
            request_frame = ledger_frame["request"]
            preview_ref = request_frame["ghost_preview_snapshot_ref"]
            snapshot_preview = _read_snapshot_ref_payload(preview_ref, frames_dir=frames_dir)

        self.assertEqual(request_frame["ghost_preview"]["instance"]["instance_id"], "legacy-route-1")
        self.assertEqual(
            request_frame["ghost_preview"]["runtime"]["developer_diagnostics"]["routing_contract"],
            "ghost_primary",
        )
        self.assertNotIn("request_phase_graph", request_frame["ghost_preview"]["runtime"])
        self.assertIn("request.ghost_preview", ledger_frame["external_snapshots"]["items"])
        self.assertEqual(
            snapshot_preview["runtime"]["request_phase_graph"]["decision_contract"]["evidence"],
            repeated_contract["evidence"],
        )
        self.assertLess(len(ledger_line.encode("utf-8")), raw_frame_size // 10)

    def test_persist_response_frame_compacts_late_fill_heavy_results_without_losing_snapshot_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            raw_image = "iVBORw0KGgo" + ("A" * 240_000)
            raw_image_sha = hashlib.sha256(base64.b64decode(raw_image + "=" * ((-len(raw_image)) % 4))).hexdigest()
            guidance_text = "decision contract guidance " * 18_000
            frame = build_response_frame(
                {
                    "id": "resp_compact_late_fill_truth",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Generated image artifact.",
                    "saved_image_path": "/tmp/generated/compact.png",
                    "artifacts": [{"type": "image", "path": "/tmp/generated/compact.png"}],
                    "late_fill": {
                        "status": "completed",
                        "completed_capabilities": ["image_generation"],
                        "fill_results": [
                            {
                                "branch_id": "branch-image_generation-1",
                                "phase_id": "phase-2",
                                "capability": "image_generation",
                                "status": "fulfilled",
                                "saved_image_path": "/tmp/generated/compact.png",
                                "fill_instance_id": "image-local-1",
                                "result": {
                                    "done": True,
                                    "done_reason": "stop",
                                    "model": "local-image",
                                    "image": raw_image,
                                },
                            }
                        ],
                        "ghost_repair_feedback": {
                            "kind": "ollmo.ghost_repair_feedback",
                            "status": "repair_required",
                            "reason": "binding check",
                            "decision_contract_guidance": {
                                "large_truth": guidance_text,
                            },
                            "items": [
                                {
                                    "branch_id": "branch-image_generation-1",
                                    "repair_action": "rebind_dependency_evidence",
                                    "status": "candidate",
                                }
                            ],
                        },
                    },
                },
                request_payload={"prompt": "generate one image"},
            )
            raw_frame_size = len(json.dumps(frame, ensure_ascii=False).encode("utf-8"))

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_line = target.read_text(encoding="utf-8").splitlines()[0]
            ledger_frame = json.loads(ledger_line)
            compact_late_fill = ledger_frame["late_fill"]
            compact_result = compact_late_fill["fill_results"][0]
            restored_late_fill = _read_snapshot_ref_payload(
                ledger_frame["late_fill_snapshot_ref"],
                frames_dir=frames_dir,
            )
            recovered = load_latest_response_state("resp_compact_late_fill_truth", frames_dir=frames_dir)

        self.assertLess(len(ledger_line.encode("utf-8")), raw_frame_size // 4)
        self.assertNotIn(raw_image, ledger_line)
        self.assertNotIn(guidance_text, ledger_line)
        self.assertIn("late_fill_snapshot_ref", ledger_frame)
        self.assertIn("full_snapshot_ref", compact_late_fill)
        self.assertEqual(compact_late_fill["status"], "completed")
        self.assertEqual(compact_late_fill["fill_result_count"], 1)
        self.assertEqual(compact_result["branch_id"], "branch-image_generation-1")
        self.assertEqual(compact_result["saved_image_path"], "/tmp/generated/compact.png")
        self.assertIn("image", compact_result["backend_result_summary"]["omitted_result_keys"])
        self.assertTrue(compact_result["full_result_in_snapshot"])
        self.assertTrue(compact_late_fill["ghost_repair_feedback"]["decision_contract_guidance_externalized"])
        self.assertEqual(
            restored_late_fill["fill_results"][0]["branch_id"],
            "branch-image_generation-1",
        )
        self.assertEqual(
            restored_late_fill["fill_results"][0]["result"]["image"]["kind"],
            "ollmo.snapshot_stripped_raw_media_payload",
        )
        self.assertEqual(
            restored_late_fill["ghost_repair_feedback"]["decision_contract_guidance"]["large_truth"],
            guidance_text,
        )
        self.assertTrue(recovered["ok"])
        recovered_media_ref = recovered["response_payload"]["late_fill"]["fill_results"][0]["result"]["image"]
        self.assertEqual(recovered_media_ref["kind"], "ollmo.snapshot_stripped_raw_media_payload")
        self.assertEqual(recovered_media_ref["sha256"], raw_image_sha)
        self.assertEqual(recovered_media_ref["truth_preservation"], "raw_media_digest_without_saved_artifact_truth")

    def test_snapshot_externalizes_artifact_backed_raw_media_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / "state" / "response_frames"
            image_path = root / "artifacts" / "images" / "generated.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = b"\x89PNG\r\n\x1a\n" + (b"artifact-backed-image" * 400)
            image_path.write_bytes(image_bytes)
            raw_image = base64.b64encode(image_bytes).decode("ascii")
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            relative_image_path = "artifacts/images/generated.png"
            frame = build_response_frame(
                {
                    "id": "resp_snapshot_media_externalized",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Generated image artifact.",
                    "saved_image_path": relative_image_path,
                    "artifacts": [
                        {
                            "type": "image",
                            "path": relative_image_path,
                            "artifact_ref": "artifact:generated-image",
                        }
                    ],
                    "late_fill": {
                        "status": "completed",
                        "fill_results": [
                            {
                                "branch_id": "branch-image_generation-1",
                                "phase_id": "phase-2",
                                "capability": "image_generation",
                                "status": "fulfilled",
                                "artifact_ref": "artifact:generated-image",
                                "saved_image_path": relative_image_path,
                                "result": {
                                    "done": True,
                                    "model": "local-image",
                                    "image": raw_image,
                                },
                            }
                        ],
                    },
                },
                request_payload={"prompt": "generate one image"},
            )

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_text = target.read_text(encoding="utf-8")
            ledger_frame = json.loads(ledger_text.splitlines()[0])
            snapshot_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted((frames_dir / "snapshots").rglob("*.json"))
            )
            restored_late_fill = _read_snapshot_ref_payload(
                ledger_frame["late_fill_snapshot_ref"],
                frames_dir=frames_dir,
            )
            recovered = load_latest_response_state(
                "resp_snapshot_media_externalized",
                frames_dir=frames_dir,
            )

        self.assertNotIn(raw_image, ledger_text)
        self.assertNotIn(raw_image, snapshot_text)
        media_ref = restored_late_fill["fill_results"][0]["result"]["image"]
        self.assertEqual(media_ref["kind"], "ollmo.snapshot_externalized_media_payload")
        self.assertEqual(media_ref["source_path"], relative_image_path)
        self.assertEqual(media_ref["artifact_ref"], "artifact:generated-image")
        self.assertEqual(media_ref["sha256"], image_sha)
        self.assertEqual(media_ref["size_bytes"], len(image_bytes))
        self.assertTrue(media_ref["raw_payload_externalized"])
        recovered_media_ref = recovered["response_payload"]["late_fill"]["fill_results"][0]["result"]["image"]
        self.assertEqual(recovered_media_ref["sha256"], image_sha)
        self.assertEqual(
            recovered_media_ref["truth_preservation"],
            "artifact_file_sha256_matches_raw_payload",
        )

    def test_snapshot_strips_orphan_raw_media_payload_without_saved_artifact_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            raw_image = base64.b64encode(b"orphan-raw-image" * 400).decode("ascii")
            raw_sha = hashlib.sha256(base64.b64decode(raw_image)).hexdigest()
            frame = build_response_frame(
                {
                    "id": "resp_snapshot_media_orphan_raw",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Generated image bytes.",
                    "late_fill": {
                        "status": "completed",
                        "fill_results": [
                            {
                                "branch_id": "branch-image_generation-1",
                                "phase_id": "phase-2",
                                "capability": "image_generation",
                                "status": "fulfilled",
                                "result": {
                                    "done": True,
                                    "model": "local-image",
                                    "image": raw_image,
                                },
                            }
                        ],
                    },
                },
                request_payload={"prompt": "generate one image"},
            )

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_text = target.read_text(encoding="utf-8")
            ledger_frame = json.loads(ledger_text.splitlines()[0])
            snapshot_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted((frames_dir / "snapshots").rglob("*.json"))
            )
            recovered = load_latest_response_state(
                "resp_snapshot_media_orphan_raw",
                frames_dir=frames_dir,
            )

        self.assertNotIn(raw_image, ledger_text)
        self.assertNotIn(raw_image, snapshot_text)
        self.assertIn("late_fill_snapshot_ref", ledger_frame)
        recovered_media_ref = recovered["response_payload"]["late_fill"]["fill_results"][0]["result"]["image"]
        self.assertEqual(recovered_media_ref["kind"], "ollmo.snapshot_stripped_raw_media_payload")
        self.assertEqual(recovered_media_ref["sha256"], raw_sha)
        self.assertEqual(recovered_media_ref["raw_payload_externalized"], True)
        self.assertEqual(recovered_media_ref["truth_preservation"], "raw_media_digest_without_saved_artifact_truth")

    def test_compact_ledger_refs_context_candidates_request_input_and_work_trees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            context_candidates = [
                {
                    "candidate_id": f"ctx-{index}",
                    "source": "history",
                    "status": "candidate",
                    "text": "candidate evidence " * 90,
                    "updated_at": f"2026-05-12T20:{index:02d}:00Z",
                }
                for index in range(8)
            ]
            work_tree = {
                "kind": "ollmo.work_tree",
                "work_tree_source": "runtime_owned",
                "authoritative": True,
                "updated_at": "2026-05-12T20:00:00Z",
                "root_node_id": "node-request",
                "node_order": ["node-request", "node-output-1"],
                "nodes": [
                    {"node_id": "node-request", "kind": "request", "status": "active"},
                    {
                        "node_id": "node-output-1",
                        "kind": "output",
                        "slot_id": "output-1",
                        "type": "text",
                        "status": "fulfilled",
                        "value": "runtime truth",
                    },
                ],
            }
            request_phase_graph = {
                "kind": "ollmo.request_phase_graph",
                "updated_at": "2026-05-12T20:01:00Z",
                "nodes": [{"id": "phase-1", "summary": "phase truth " * 80}],
            }
            frame = {
                "frame_version": 9,
                "kind": "ollmo.response_frame",
                "response_id": "resp_compact_refs",
                "status": "completed",
                "object": "response",
                "request": {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "long current request " * 700}],
                        }
                    ],
                    "context_candidates": context_candidates,
                },
                "runtime": {
                    "context_strategy": {
                        "kind": "ollmo.context_strategy",
                        "mode": "bounded_file_context",
                        "reason": "selected refs",
                        "context_candidates": context_candidates,
                        "context_gate_review": {
                            "updated_at": "2026-05-12T20:02:00Z",
                            "notes": "gate evidence " * 200,
                        },
                    },
                    "semantic_role_profile": {
                        "kind": "ollmo.semantic_role_profile",
                        "mode": "explorer",
                        "mode_source": "compatibility_alias",
                        "semantic_role_orientation": {
                            "review": "advisory only",
                            "details": "role profile diagnostic " * 500,
                        },
                    },
                },
                "planning": {
                    "request_phase_graph": request_phase_graph,
                    "work_tree": work_tree,
                    "artifact_flow": {
                        "work_tree": work_tree,
                        "output_slots": [{"slot_id": "output-1", "status": "fulfilled", "type": "text"}],
                    },
                },
                "current_state": {
                    "id": "resp_compact_refs",
                    "status": "completed",
                    "work_tree": work_tree,
                },
            }

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            recovered = load_latest_response_state("resp_compact_refs", frames_dir=frames_dir)

        self.assertIn("input_snapshot_ref", ledger_frame["request"])
        self.assertNotIn("input", ledger_frame["request"])
        self.assertIn("context_candidates_snapshot_ref", ledger_frame["request"])
        self.assertNotIn("context_candidates", ledger_frame["request"])
        runtime_context = ledger_frame["runtime"]["context_strategy"]
        self.assertEqual(runtime_context["mode"], "bounded_file_context")
        self.assertIn("context_candidates_snapshot_ref", runtime_context)
        self.assertIn("context_gate_review_snapshot_ref", runtime_context)
        self.assertIn("full_snapshot_ref", runtime_context)
        self.assertNotIn("context_candidates", runtime_context)
        runtime_role = ledger_frame["runtime"]["semantic_role_profile"]
        self.assertEqual(runtime_role["mode"], "explorer")
        self.assertIn("full_snapshot_ref", runtime_role)
        self.assertNotIn("semantic_role_orientation", runtime_role)
        self.assertIn("work_tree_snapshot_ref", ledger_frame["current_state"])
        self.assertNotIn("work_tree", ledger_frame["current_state"])
        self.assertIn("work_tree_snapshot_ref", ledger_frame["planning"])
        self.assertNotIn("work_tree", ledger_frame["planning"])
        self.assertIn("work_tree_snapshot_ref", ledger_frame["planning"]["artifact_flow"])
        self.assertNotIn("work_tree", ledger_frame["planning"]["artifact_flow"])
        self.assertEqual(
            ledger_frame["current_state"]["work_tree_snapshot_ref"]["path"],
            ledger_frame["planning"]["work_tree_snapshot_ref"]["path"],
        )
        self.assertEqual(
            ledger_frame["planning"]["work_tree_snapshot_ref"]["path"],
            ledger_frame["planning"]["artifact_flow"]["work_tree_snapshot_ref"]["path"],
        )
        self.assertTrue(recovered["ok"])
        self.assertEqual(recovered["response_payload"]["work_tree"]["work_tree_source"], "runtime_owned")
        self.assertTrue(recovered["response_payload"]["work_tree"]["authoritative"])
        self.assertEqual(recovered["response_payload"]["work_tree"]["node_order"], ["node-request", "node-output-1"])

    def test_diagnostic_snapshot_hash_ignores_volatile_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            graph_a = {
                "kind": "ollmo.request_phase_graph",
                "updated_at": "2026-05-12T20:00:00Z",
                "nodes": [
                    {
                        "id": "phase-1",
                        "capability": "chat",
                        "started_at": "2026-05-12T20:00:01Z",
                        "summary": "stable graph truth",
                    }
                ],
            }
            graph_b = {
                "kind": "ollmo.request_phase_graph",
                "updated_at": "2026-05-12T20:05:00Z",
                "nodes": [
                    {
                        "id": "phase-1",
                        "capability": "chat",
                        "started_at": "2026-05-12T20:05:01Z",
                        "summary": "stable graph truth",
                    }
                ],
            }

            first = persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": "resp_timestamp_a",
                    "status": "completed",
                    "object": "response",
                    "planning": {"request_phase_graph": graph_a},
                    "current_state": {"id": "resp_timestamp_a", "status": "completed"},
                },
                frames_dir=frames_dir,
            )
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": "resp_timestamp_b",
                    "status": "completed",
                    "object": "response",
                    "planning": {"request_phase_graph": graph_b},
                    "current_state": {"id": "resp_timestamp_b", "status": "completed"},
                },
                frames_dir=frames_dir,
            )
            ledger_frames = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            ref_a = ledger_frames[0]["planning"]["request_phase_graph_snapshot_ref"]
            ref_b = ledger_frames[1]["planning"]["request_phase_graph_snapshot_ref"]
            snapshot_payload = json.loads((frames_dir / ref_a["path"]).read_text(encoding="utf-8"))

        self.assertEqual(ref_a["sha256"], ref_b["sha256"])
        self.assertEqual(ref_a["path"], ref_b["path"])
        self.assertEqual(
            ref_a["content_normalization"]["strategy"],
            "volatile_timestamp_keys_excluded_from_content_hash",
        )
        self.assertEqual(
            ref_a["content_normalization"]["volatile_timestamp_keys"],
            ["started_at", "updated_at"],
        )
        self.assertNotIn("updated_at", snapshot_payload)
        self.assertNotIn("started_at", snapshot_payload["nodes"][0])
        self.assertEqual(snapshot_payload["nodes"][0]["summary"], "stable graph truth")

    def test_successor_frame_delta_logs_inherited_snapshot_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_delta_snapshot_inheritance"
            large_graph = {
                "kind": "ollmo.request_phase_graph",
                "nodes": [{"id": f"phase-{index}", "summary": "stable phase truth " * 100} for index in range(12)],
            }
            initial_frame = {
                "frame_version": 9,
                "kind": "ollmo.response_frame",
                "response_id": response_id,
                "status": "completed",
                "object": "response",
                "runtime": {"request_phase_graph": large_graph},
                "planning": {"request_phase_graph": large_graph},
                "current_state": {"id": response_id, "status": "completed"},
            }
            successor_frame = {
                **initial_frame,
                "late_fill": {
                    "status": "completed",
                    "fill_results": [{"branch_id": "branch-image-1", "status": "fulfilled"}],
                },
            }

            target = persist_response_frame(initial_frame, frames_dir=frames_dir)
            persist_response_frame(successor_frame, frames_dir=frames_dir)
            ledger_frames = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
            ]
            successor = ledger_frames[1]
            successor_items = successor["external_snapshots"].get("items", {})
            inheritance = successor["external_snapshots"]["inheritance"]
            recovered = load_latest_response_state(response_id, frames_dir=frames_dir)
            recovered_items = recovered["response_frame"]["external_snapshots"]["items"]
            recovered_delta_items = recovered["response_frame"]["external_snapshots"]["delta_items"]

        self.assertEqual(successor["frame_relation"]["kind"], "late_fill_successor")
        self.assertNotIn("runtime", successor_items)
        self.assertNotIn("planning.request_phase_graph", successor_items)
        self.assertIn("runtime", inheritance["inherited_json_paths"])
        self.assertIn("planning.request_phase_graph", inheritance["inherited_json_paths"])
        self.assertGreater(inheritance["effective_snapshot_count"], len(successor_items))
        self.assertTrue(recovered["ok"])
        self.assertIn("runtime", recovered_items)
        self.assertIn("planning.request_phase_graph", recovered_items)
        self.assertNotIn("runtime", recovered_delta_items)

    def test_compact_ledger_prunes_dossiers_from_output_projections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            frame = {
                "frame_version": 9,
                "kind": "ollmo.response_frame",
                "response_id": "resp_prune_dossiers",
                "status": "completed",
                "object": "response",
                "artifacts": {
                    "output": [
                        {
                            "type": "image",
                            "path": "/tmp/generated/image.png",
                            "artifact_ref": "artifact:image:one",
                        }
                    ],
                    "dossiers": {
                        "artifact:image:one": {
                            "type": "image",
                            "roles": ["output"],
                            "artifact": {
                                "artifact_ref": "artifact:image:one",
                                "path": "/tmp/generated/image.png",
                            },
                            "metadata": {"seed": 123},
                            "provenance": {"provenance_id": "prov-image-one"},
                            "enrichments": {"image_state": {"subject": "yellow notebook"}},
                        }
                    },
                },
                "planning": {
                    "artifact_flow": {
                        "output_slots": [
                            {
                                "slot_id": "output-image",
                                "type": "image",
                                "status": "fulfilled",
                                "artifact_ref": "artifact:image:one",
                                "artifact_path": "/tmp/generated/image.png",
                                "metadata": {"seed": 123},
                                "provenance": {"provenance_id": "prov-image-one"},
                                "image_state": {"subject": "yellow notebook"},
                            }
                        ]
                    }
                },
                "output": {
                    "outputs": [
                        {
                            "slot_id": "output-image",
                            "type": "image",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:image:one",
                            "artifacts": [
                                {
                                    "type": "image",
                                    "path": "/tmp/generated/image.png",
                                    "artifact_ref": "artifact:image:one",
                                    "metadata": {"seed": 123},
                                }
                            ],
                        }
                    ]
                },
                "current_state": {
                    "id": "resp_prune_dossiers",
                    "status": "completed",
                    "outputs": [
                        {
                            "slot_id": "output-image",
                            "type": "image",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:image:one",
                            "artifacts": [{"path": "/tmp/generated/image.png", "artifact_ref": "artifact:image:one"}],
                        }
                    ],
                    "output_slots": [
                        {
                            "slot_id": "output-image",
                            "type": "image",
                            "status": "fulfilled",
                            "artifact_ref": "artifact:image:one",
                            "artifact_path": "/tmp/generated/image.png",
                        }
                    ],
                },
            }

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            dossier_ref = ledger_frame["artifacts"]["dossiers_snapshot_ref"]
            dossier_payload = json.loads((frames_dir / dossier_ref["path"]).read_text(encoding="utf-8"))

        self.assertNotIn("dossiers", ledger_frame["artifacts"])
        self.assertEqual(dossier_payload["artifact:image:one"]["metadata"]["seed"], 123)
        output = ledger_frame["output"]["outputs"][0]
        self.assertEqual(output["artifact_ref"], "artifact:image:one")
        self.assertNotIn("artifacts", output)
        self.assertNotIn("path", output)
        slot = ledger_frame["planning"]["artifact_flow"]["output_slots"][0]
        self.assertEqual(slot["artifact_ref"], "artifact:image:one")
        self.assertNotIn("artifact_path", slot)
        self.assertNotIn("metadata", slot)
        current_output = ledger_frame["current_state"]["outputs"][0]
        self.assertEqual(current_output["artifact_ref"], "artifact:image:one")
        self.assertNotIn("artifacts", current_output)

    def test_working_frame_snapshot_is_logic_only_with_child_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            graph = build_request_phase_graph("Write a line, then create an image.")
            frame = build_response_frame(
                {
                    "id": "resp_clean_working_frame",
                    "object": "response",
                    "status": "completed",
                    "output_text": "A small lighthouse waits.",
                    "runtime": {"request_phase_graph": graph},
                    "late_fill": {"status": "running"},
                },
                request_payload={"prompt": "Write a line, then create an image."},
            )

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            working_ref = ledger_frame["working_frame_snapshot_ref"]
            working_payload = json.loads((frames_dir / working_ref["path"]).read_text(encoding="utf-8"))

        self.assertEqual(working_payload["kind"], "ollmo.working_frame")
        self.assertIn("status", working_payload)
        self.assertIn("closure", working_payload)
        self.assertIn("loop", working_payload)
        self.assertNotIn("request_phase_graph", working_payload)
        self.assertNotIn("intent_contract", working_payload)
        self.assertNotIn("context_contract", working_payload)
        self.assertNotIn("artifact_flow", working_payload)
        self.assertIn("request_phase_graph_snapshot_ref", working_payload)
        self.assertIn("intent_contract_snapshot_ref", working_payload)
        self.assertIn("artifact_flow_snapshot_ref", working_payload)
        self.assertIn("pending_obligation_ids", working_payload["intent_contract_summary"])
        self.assertNotIn("prompt", working_payload.get("request", {}))

    def test_sidecar_snapshots_split_large_worthwhile_children_and_expand_for_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            graph = {
                "kind": "ollmo.request_phase_graph",
                "nodes": [
                    {
                        "id": f"phase-{index}",
                        "capability": "chat" if index == 0 else "image_generation",
                        "summary": "repeated diagnostic graph truth " * 260,
                        "decision_contract": {
                            "status": "promoted",
                            "evidence": "branch-local evidence " * 180,
                        },
                    }
                    for index in range(9)
                ],
                "edges": [
                    {
                        "from": f"phase-{index}",
                        "to": f"phase-{index + 1}",
                        "evidence": "dependency edge evidence " * 160,
                    }
                    for index in range(8)
                ],
            }
            frame = {
                "frame_version": 9,
                "kind": "ollmo.response_frame",
                "response_id": "resp_recursive_sidecar_split",
                "status": "completed",
                "object": "response",
                "planning": {
                    "request_phase_graph": graph,
                    "artifact_flow": {"request_phase_graph": graph},
                },
                "current_state": {"id": "resp_recursive_sidecar_split", "status": "completed"},
            }

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            graph_ref = ledger_frame["planning"]["request_phase_graph_snapshot_ref"]
            artifact_flow_graph_ref = ledger_frame["planning"]["artifact_flow"]["request_phase_graph_snapshot_ref"]
            raw_graph = json.loads((frames_dir / graph_ref["path"]).read_text(encoding="utf-8"))
            raw_artifact_flow_graph = json.loads(
                (frames_dir / artifact_flow_graph_ref["path"]).read_text(encoding="utf-8")
            )
            expanded_graph = _read_snapshot_ref_payload(graph_ref, frames_dir=frames_dir)

        self.assertIn("sidecar_manifest", graph_ref)
        self.assertGreaterEqual(graph_ref["sidecar_manifest"]["child_ref_count"], 1)
        self.assertIn("nodes_snapshot_ref", raw_graph)
        self.assertNotIn("nodes", raw_graph)
        self.assertIn("nodes_snapshot_ref", raw_artifact_flow_graph)
        self.assertEqual(
            raw_graph["nodes_snapshot_ref"]["sha256"],
            raw_artifact_flow_graph["nodes_snapshot_ref"]["sha256"],
        )
        self.assertEqual(expanded_graph["nodes"][3]["summary"], graph["nodes"][3]["summary"])
        self.assertEqual(expanded_graph["edges"][0]["evidence"], graph["edges"][0]["evidence"])

    def test_recursive_sidecar_authority_restores_two_level_truth_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_two_level_recursive_sidecar"
            candidate_graph = {
                "kind": "ollmo.candidate_graph",
                "nodes": [
                    {
                        "id": f"candidate-{index}",
                        "evidence": f"candidate evidence {index} " * 180,
                    }
                    for index in range(32)
                ],
                "edges": [
                    {"from": f"candidate-{index}", "to": f"candidate-{index + 1}"}
                    for index in range(31)
                ],
            }
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "runtime": {"candidate_graph": candidate_graph},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                    },
                },
                frames_dir=frames_dir,
            )
            runtime_ref = load_response_frame_index(frames_dir=frames_dir)["responses"][
                response_id
            ]["effective_snapshot_manifest"]["runtime"]
            raw_runtime = _read_snapshot_ref_payload(
                runtime_ref,
                frames_dir=frames_dir,
                expand_child_refs=False,
            )
            candidate_ref = raw_runtime["candidate_graph_snapshot_ref"]
            raw_candidate = _read_snapshot_ref_payload(
                candidate_ref,
                frames_dir=frames_dir,
                expand_child_refs=False,
            )
            nodes_ref = raw_candidate["nodes_snapshot_ref"]
            recovered = load_latest_response_state(response_id, frames_dir=frames_dir)

            self.assertTrue(recovered["ok"])
            self.assertEqual(
                recovered["response_payload"]["runtime"]["candidate_graph"],
                candidate_graph,
            )
            self.assertIn("sidecar_manifest", runtime_ref)
            candidate_authority = runtime_ref["sidecar_manifest"]["child_refs"][0]
            self.assertIn("sidecar_manifest", candidate_authority)

            (frames_dir / nodes_ref["path"]).unlink()
            missing_state = load_latest_response_state(response_id, frames_dir=frames_dir)

        self.assertFalse(missing_state["ok"])
        self.assertEqual(missing_state["status_code"], 409)
        self.assertEqual(
            missing_state["error"]["code"],
            "response_frame_snapshot_unavailable",
        )
        self.assertEqual(missing_state["error"]["expected_sha256"], nodes_ref["sha256"])

    def test_recursive_sidecar_split_budget_keeps_more_than_sixty_four_children_exact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_bounded_recursive_sidecars"
            runtime = {
                f"part_{index}_graph": {
                    "kind": "diagnostic_graph",
                    "part_index": index,
                    "payload": f"part-{index}-" + ("g" * (70 * 1024)),
                }
                for index in range(70)
            }
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "runtime": runtime,
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                    },
                },
                frames_dir=frames_dir,
            )
            runtime_ref = load_response_frame_index(frames_dir=frames_dir)["responses"][
                response_id
            ]["effective_snapshot_manifest"]["runtime"]
            raw_runtime = _read_snapshot_ref_payload(
                runtime_ref,
                frames_dir=frames_dir,
                expand_child_refs=False,
            )
            recovered = load_latest_response_state(response_id, frames_dir=frames_dir)

        manifest = runtime_ref["sidecar_manifest"]
        split_keys = [key for key in raw_runtime if key.endswith("_snapshot_ref")]
        inline_keys = [key for key in raw_runtime if key.endswith("_graph")]
        self.assertEqual(manifest["child_ref_count"], 64)
        self.assertEqual(len(manifest["child_refs"]), 64)
        self.assertFalse(manifest["child_refs_truncated"])
        self.assertEqual(manifest["split_ref_limit"], 64)
        self.assertEqual(len(split_keys), 64)
        self.assertEqual(len(inline_keys), 6)
        self.assertTrue(recovered["ok"])
        recovered_runtime = recovered["response_payload"]["runtime"]
        self.assertEqual(recovered_runtime, runtime)
        self.assertFalse(
            any(
                key.endswith("_snapshot_ref")
                for value in recovered_runtime.values()
                if isinstance(value, dict)
                for key in value
            )
        )

    def test_advisory_sidecar_snapshots_dedupe_repeated_semantic_control_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            lens_contract = {
                "kind": "ollmo.semantic_review_lens_contract",
                "semantic_role_id": "quality_reviewer",
                "semantic_role_name": "Quality reviewer",
                "authority": "advisory_read_model_only",
                "semantic_role_orientation": "Verify runtime evidence before any closure decision. " * 12,
                "focus_questions": [
                    f"focus question {index}: inspect the exact runtime evidence before transition"
                    for index in range(12)
                ],
                "failure_modes": [
                    f"failure mode {index}: advisory prose outruns runtime truth"
                    for index in range(10)
                ],
                "evidence_requirements": [
                    f"evidence requirement {index}: cite frame, artifact, or branch id"
                    for index in range(10)
                ],
            }

            def attention_frame(index: int) -> dict[str, object]:
                return {
                    "kind": "ollmo.controlled_attention_frame",
                    "frame_id": f"attention-{index}",
                    "scope": "runtime_state_transition",
                    "priority": "high",
                    "question": "Should this promoted work continue, repair, review, or close?",
                    "allowed_transitions": ["continue", "repair", "review", "waive", "close"],
                    "target_ref": {"branch_id": f"branch-{index % 3}", "phase_id": f"phase-{index % 2}"},
                    "evidence_refs": [f"artifact:{index % 2}", f"frame:{index % 4}"],
                    "semantic_review_lens_contract": dict(lens_contract),
                }

            def semantic_proposal(index: int) -> dict[str, object]:
                return {
                    "kind": "ollmo.semantic_decision_proposal",
                    "proposal_id": f"semantic-proposal-{index}",
                    "action": "review_before_close",
                    "confidence": 0.72,
                    "reason": "Advisory decision must remain traceable without bloating every frame. " * 8,
                    "evidence_refs": [f"check:{index % 3}", f"frame:{index % 5}"],
                    "semantic_review_lens_contract": dict(lens_contract),
                }

            guidance = {
                "kind": "ollmo.decision_contract_guidance",
                "authority": "advisory_read_model_only",
                "controlled_attention_review": {
                    "kind": "ollmo.controlled_attention_review",
                    "frame_count": 18,
                    "authority": "advisory_read_model_only",
                },
                "controlled_attention_frames": [attention_frame(index) for index in range(18)],
                "semantic_decision_review": {
                    "kind": "ollmo.semantic_decision_review",
                    "proposal_count": 12,
                    "authority": "advisory_read_model_only",
                },
                "semantic_decision_proposals": [semantic_proposal(index) for index in range(12)],
            }
            graph_closure_review = {
                "kind": "ollmo.graph_closure_review",
                "status": "pending",
                "checks": [
                    {
                        "check_id": f"check-{index}",
                        "status": "pending",
                        "controlled_attention_frame": attention_frame(index),
                        "decision_contract_controlled_attention_frames": [
                            attention_frame(index),
                            attention_frame(index + 1),
                        ],
                        "semantic_decision_proposal": semantic_proposal(index),
                        "decision_contract_semantic_decision_proposals": [
                            semantic_proposal(index),
                            semantic_proposal(index + 1),
                        ],
                    }
                    for index in range(6)
                ],
            }
            frame = build_response_frame(
                {
                    "id": "resp_advisory_compaction",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Advisory truth preserved by refs.",
                    "runtime": {"graph_closure_review": graph_closure_review},
                    "late_fill": {
                        "status": "running",
                        "ghost_repair_feedback": {
                            "kind": "ollmo.ghost_repair_feedback",
                            "status": "repair_required",
                            "reason": "advisory review pending",
                            "decision_contract_guidance": guidance,
                            "items": [
                                {
                                    "check_id": f"check-{index}",
                                    "status": "candidate",
                                    "repair_action": "review_before_close",
                                    "decision_contract_controlled_attention_frames": [
                                        attention_frame(index)
                                    ],
                                }
                                for index in range(6)
                            ],
                        },
                    },
                },
                request_payload={"prompt": "exercise advisory compaction"},
            )
            raw_guidance_size = len(json.dumps(guidance, ensure_ascii=False, sort_keys=True).encode("utf-8"))

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_text = target.read_text(encoding="utf-8")
            ledger_frame = json.loads(ledger_text.splitlines()[0])

            def raw_snapshot(ref: dict[str, object]):
                return json.loads((frames_dir / str(ref["path"])).read_text(encoding="utf-8"))

            raw_late_fill = raw_snapshot(ledger_frame["late_fill_snapshot_ref"])
            feedback_ref = raw_late_fill["ghost_repair_feedback_snapshot_ref"]
            raw_feedback = raw_snapshot(feedback_ref)
            guidance_ref = raw_feedback["decision_contract_guidance_snapshot_ref"]
            raw_guidance = raw_snapshot(guidance_ref)
            raw_frames = raw_snapshot(raw_guidance["controlled_attention_frames_snapshot_ref"])
            raw_proposals = raw_snapshot(raw_guidance["semantic_decision_proposals_snapshot_ref"])
            recovered = load_latest_response_state("resp_advisory_compaction", frames_dir=frames_dir)

        self.assertIn("ghost_repair_feedback_snapshot_ref", raw_late_fill)
        self.assertNotIn("ghost_repair_feedback", raw_late_fill)
        self.assertIn("decision_contract_guidance_snapshot_ref", raw_feedback)
        self.assertNotIn("decision_contract_guidance", raw_feedback)
        self.assertIn("controlled_attention_frames_snapshot_ref", raw_guidance)
        self.assertIn("semantic_decision_proposals_snapshot_ref", raw_guidance)
        self.assertLess(
            len(json.dumps(raw_guidance, ensure_ascii=False, sort_keys=True).encode("utf-8")),
            raw_guidance_size // 3,
        )
        self.assertIn("semantic_review_lens_contract_snapshot_ref", raw_frames[0])
        self.assertNotIn("semantic_review_lens_contract", raw_frames[0])
        self.assertIn("semantic_review_lens_contract_snapshot_ref", raw_proposals[0])
        lens_shas = {
            item["semantic_review_lens_contract_snapshot_ref"]["sha256"]
            for item in raw_frames[:6]
        } | {
            item["semantic_review_lens_contract_snapshot_ref"]["sha256"]
            for item in raw_proposals[:6]
        }
        self.assertEqual(len(lens_shas), 1)
        self.assertTrue(recovered["ok"])
        recovered_guidance = recovered["response_payload"]["late_fill"]["ghost_repair_feedback"]["decision_contract_guidance"]
        self.assertEqual(
            recovered_guidance["controlled_attention_frames"][0]["semantic_review_lens_contract"]["semantic_role_id"],
            "quality_reviewer",
        )
        self.assertEqual(
            recovered_guidance["semantic_decision_proposals"][0]["semantic_review_lens_contract"]["focus_questions"][0],
            lens_contract["focus_questions"][0],
        )

    def test_advisory_context_splits_generic_frames_proposals_and_surface_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            lens_contract = {
                "kind": "ollmo.semantic_review_lens_contract",
                "semantic_role_id": "transition_committer",
                "authority": "advisory_read_model_only",
                "focus_questions": [
                    f"focus {index}: keep generic advisory children compact but recoverable"
                    for index in range(8)
                ],
                "failure_modes": [
                    f"failure {index}: path context was missed"
                    for index in range(8)
                ],
            }
            attention_frames = [
                {
                    "kind": "ollmo.controlled_attention_frame",
                    "frame_id": f"attention-{index}",
                    "scope": "runtime_state_transition",
                    "question": "Which bounded transition is justified by runtime truth? " * 5,
                    "evidence_refs": [f"evidence:{index}", "artifact:demo"],
                    "semantic_review_lens_contract": dict(lens_contract),
                }
                for index in range(10)
            ]
            semantic_proposals = [
                {
                    "kind": "ollmo.semantic_decision_proposal",
                    "proposal_id": f"proposal-{index}",
                    "action": "review_before_close",
                    "reason": "Generic proposal list should split only in semantic decision context. " * 5,
                    "semantic_review_lens_contract": dict(lens_contract),
                }
                for index in range(8)
            ]
            surface_items = [
                {
                    "category": "controlled_attention_advisory",
                    "label": f"surface-item-{index}",
                    "summary": "Surface state advisory item remains auditable through child refs. " * 8,
                    "semantic_review_lens_contract": dict(lens_contract),
                }
                for index in range(8)
            ]
            frame = build_response_frame(
                {
                    "id": "resp_advisory_context_generic_children",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Contextual advisory split.",
                    "late_fill": {
                        "status": "running",
                        "controlled_attention_review": {
                            "kind": "ollmo.controlled_attention_review",
                            "authority": "advisory_read_model_only",
                            "frame_count": len(attention_frames),
                            "frames": attention_frames,
                        },
                        "semantic_decision_review": {
                            "kind": "ollmo.semantic_decision_review",
                            "authority": "advisory_read_model_only",
                            "proposal_count": len(semantic_proposals),
                            "proposals": semantic_proposals,
                        },
                        "surface_state": {
                            "kind": "ollmo.surface_state",
                            "items": surface_items,
                        },
                    },
                },
                request_payload={"prompt": "exercise contextual advisory compaction"},
            )

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])

            def raw_snapshot(ref: dict[str, object]):
                return json.loads((frames_dir / str(ref["path"])).read_text(encoding="utf-8"))

            def nested_payload(container: dict[str, object], key: str):
                ref_key = f"{key}_snapshot_ref"
                if isinstance(container.get(ref_key), dict):
                    return raw_snapshot(container[ref_key])
                return container[key]

            raw_late_fill = raw_snapshot(ledger_frame["late_fill_snapshot_ref"])
            raw_attention = nested_payload(raw_late_fill, "controlled_attention_review")
            raw_semantic = nested_payload(raw_late_fill, "semantic_decision_review")
            raw_surface = nested_payload(raw_late_fill, "surface_state")
            expanded_late_fill = _read_snapshot_ref_payload(
                ledger_frame["late_fill_snapshot_ref"],
                frames_dir=frames_dir,
            )

        self.assertIn("frames_snapshot_ref", raw_attention)
        self.assertNotIn("frames", raw_attention)
        self.assertIn("proposals_snapshot_ref", raw_semantic)
        self.assertNotIn("proposals", raw_semantic)
        self.assertIn("items_snapshot_ref", raw_surface)
        self.assertNotIn("items", raw_surface)
        self.assertEqual(
            expanded_late_fill["controlled_attention_review"]["frames"][0]["semantic_review_lens_contract"]["semantic_role_id"],
            "transition_committer",
        )
        self.assertEqual(
            expanded_late_fill["semantic_decision_review"]["proposals"][0]["semantic_review_lens_contract"]["semantic_role_id"],
            "transition_committer",
        )
        self.assertEqual(
            expanded_late_fill["surface_state"]["items"][0]["semantic_review_lens_contract"]["semantic_role_id"],
            "transition_committer",
        )

    def test_post_learning_advisory_families_split_to_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)

            def large_record(prefix: str, index: int) -> dict[str, object]:
                return {
                    "id": f"{prefix}-{index}",
                    "status": "candidate",
                    "reason": f"{prefix} advisory evidence should remain recoverable without bloating the ledger. " * 12,
                    "evidence_refs": [f"frame:{index}", f"branch:{index % 3}"],
                    "learning_hint_refs": [f"accepted-learning-{index % 2}"],
                }

            accepted_learning_hints = {
                "kind": "ollmo.accepted_learning_runtime_hints",
                "status": "available",
                "enabled": True,
                "authority": "soft_hint",
                "runtime_effect": "soft_hints_available",
                "hint_count": 6,
                "hints": [large_record("accepted-learning-hint", index) for index in range(6)],
            }
            graph_closure_review = {
                "kind": "ollmo.graph_closure_review",
                "status": "repair_needed",
                "intent_graph_adequacy": {
                    "kind": "ollmo.intent_graph_adequacy_review",
                    "status": "pending",
                    "checks": [large_record("intent-adequacy-check", index) for index in range(10)],
                },
                "graph_repair_proposals": [large_record("graph-repair-proposal", index) for index in range(8)],
                "graph_repair_reviews": [large_record("graph-repair-review", index) for index in range(8)],
                "graph_rebase_lifecycle": [large_record("graph-rebase-lifecycle", index) for index in range(6)],
                "redraw_scope_ladder_review": {
                    "kind": "ollmo.redraw_scope_ladder_review",
                    "status": "selected",
                    "selected_scopes": ["repair_artifact_ref_identity"],
                    "scope_candidates": [large_record("redraw-scope-candidate", index) for index in range(8)],
                },
                "successor_reopen_requests": [
                    {
                        **large_record("successor-reopen", index),
                        "kind": "ollmo.graph_patch_successor_reopen_request",
                    }
                    for index in range(4)
                ],
            }
            frame = build_response_frame(
                {
                    "id": "resp_post_learning_advisory_compaction",
                    "object": "response",
                    "status": "completed",
                    "output_text": "Post-learning advisory truth preserved by refs.",
                    "runtime": {
                        "accepted_learning_hints": accepted_learning_hints,
                        "graph_closure_review": graph_closure_review,
                        "developer_diagnostics": {
                            "runtime_graph_repair_proposals": [
                                large_record("runtime-graph-repair-proposal", index)
                                for index in range(8)
                            ],
                            "runtime_graph_repair_proposal_reviews": [
                                large_record("runtime-graph-repair-review", index)
                                for index in range(8)
                            ],
                            "graph_rebase_enforced_policy_reviews": [
                                large_record("graph-rebase-policy-review", index)
                                for index in range(6)
                            ],
                        },
                    },
                },
                request_payload={"prompt": "exercise post-learning advisory compaction"},
            )

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])

            def raw_snapshot(ref: dict[str, object]):
                return json.loads((frames_dir / str(ref["path"])).read_text(encoding="utf-8"))

            raw_runtime = raw_snapshot(ledger_frame["runtime_snapshot_ref"])
            raw_graph = raw_snapshot(raw_runtime["graph_closure_review_snapshot_ref"])
            raw_learning = raw_snapshot(raw_runtime["accepted_learning_hints_snapshot_ref"])
            raw_intent_adequacy = raw_snapshot(raw_graph["intent_graph_adequacy_snapshot_ref"])
            raw_diagnostics = (
                raw_snapshot(raw_runtime["developer_diagnostics_snapshot_ref"])
                if isinstance(raw_runtime.get("developer_diagnostics_snapshot_ref"), dict)
                else raw_runtime["developer_diagnostics"]
            )
            expanded_runtime = _read_snapshot_ref_payload(
                ledger_frame["runtime_snapshot_ref"],
                frames_dir=frames_dir,
            )
            expanded_graph = _read_snapshot_ref_payload(
                raw_runtime["graph_closure_review_snapshot_ref"],
                frames_dir=frames_dir,
            )

        self.assertIn("accepted_learning_hints_snapshot_ref", ledger_frame["runtime"])
        self.assertNotIn("accepted_learning_hints", ledger_frame["runtime"])
        self.assertEqual(
            ledger_frame["runtime"]["accepted_learning_hints_summary"]["authority"],
            "soft_hint",
        )
        self.assertIn("accepted_learning_hints_snapshot_ref", raw_runtime)
        self.assertNotIn("accepted_learning_hints", raw_runtime)
        self.assertIn("hints_snapshot_ref", raw_learning)
        self.assertNotIn("hints", raw_learning)
        self.assertIn("intent_graph_adequacy_snapshot_ref", raw_graph)
        self.assertIn("graph_repair_proposals_snapshot_ref", raw_graph)
        self.assertIn("graph_repair_reviews_snapshot_ref", raw_graph)
        self.assertIn("graph_rebase_lifecycle_snapshot_ref", raw_graph)
        self.assertIn("redraw_scope_ladder_review_snapshot_ref", raw_graph)
        self.assertIn("successor_reopen_requests_snapshot_ref", raw_graph)
        self.assertIn("checks_snapshot_ref", raw_intent_adequacy)
        self.assertNotIn("checks", raw_intent_adequacy)
        self.assertIn("runtime_graph_repair_proposals_snapshot_ref", raw_diagnostics)
        self.assertIn("runtime_graph_repair_proposal_reviews_snapshot_ref", raw_diagnostics)
        self.assertIn("graph_rebase_enforced_policy_reviews_snapshot_ref", raw_diagnostics)
        self.assertEqual(
            expanded_runtime["accepted_learning_hints"]["hints"][0]["id"],
            "accepted-learning-hint-0",
        )
        self.assertEqual(
            expanded_graph["intent_graph_adequacy"]["checks"][0]["id"],
            "intent-adequacy-check-0",
        )

    def test_persist_response_frame_externalizes_large_contracts_from_ledger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            large_check = {
                "obligation_id": "obligation-large",
                "decision_contract_controlled_attention_frames": [
                    {"frame_id": f"attention-{index}", "text": "contract evidence " * 200}
                    for index in range(12)
                ],
                "decision_contract_semantic_decision_proposals": [
                    {"proposal_id": f"proposal-{index}", "text": "semantic proposal " * 120}
                    for index in range(8)
                ],
            }
            frame = {
                "frame_version": 9,
                "kind": "ollmo.response_frame",
                "response_id": "resp_large_contract",
                "status": "completed",
                "object": "response",
                "planning": {
                    "intent_contract": {
                        "kind": "ollmo.intent_contract",
                        "status": "pending",
                        "counts": {"pending": 1, "fulfilled": 0},
                        "pending_obligation_ids": ["obligation-large"],
                        "checks": [large_check],
                    },
                    "context_contract": {
                        "kind": "ollmo.context_contract",
                        "status": "candidate_only",
                        "candidate_count": 1,
                        "context_gate_review": {
                            "history_scan": {
                                "executed": True,
                                "matched_candidate_count": 1,
                                "raw_notes": "history candidate " * 700,
                            }
                        },
                    },
                },
                "current_state": {"id": "resp_large_contract", "status": "completed"},
            }

            target = persist_response_frame(frame, frames_dir=frames_dir)
            ledger_frame = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            intent_ref = ledger_frame["planning"]["intent_contract_snapshot_ref"]
            context_ref = ledger_frame["planning"]["context_contract_snapshot_ref"]
            intent_snapshot_exists = (frames_dir / intent_ref["path"]).exists()
            context_snapshot_exists = (frames_dir / context_ref["path"]).exists()

        self.assertIn("intent_contract", ledger_frame["planning"])
        self.assertIn("context_contract", ledger_frame["planning"])
        self.assertIn("full_snapshot_ref", ledger_frame["planning"]["intent_contract"])
        self.assertIn("full_snapshot_ref", ledger_frame["planning"]["context_contract"])
        self.assertNotIn("checks", ledger_frame["planning"]["intent_contract"])
        self.assertNotIn("context_gate_review", ledger_frame["planning"]["context_contract"])
        self.assertEqual(ledger_frame["planning"]["intent_contract"]["status"], "pending")
        self.assertEqual(ledger_frame["planning"]["context_contract"]["status"], "candidate_only")
        self.assertTrue(intent_snapshot_exists)
        self.assertTrue(context_snapshot_exists)
        self.assertLess(len(json.dumps(ledger_frame).encode("utf-8")), 12000)

    def test_late_fill_successor_frame_preserves_initial_frozen_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_successor_frame_truth"
            initial_frame = build_response_frame(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "mode": "chat",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "capability": "chat",
                    "output_text": "Initial answer while image continues.",
                },
                request_payload={"prompt": "Describe a cove, then make an image."},
            )
            persist_response_frame(initial_frame, frames_dir=frames_dir)

            successor_frame = build_response_frame(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "mode": "chat",
                    "model": "gemma4:26b",
                    "backend": "ollama",
                    "capability": "chat",
                    "output_text": "Initial answer while image continues.",
                    "saved_image_path": "/tmp/generated/cove.png",
                    "artifacts": [{"type": "image", "path": "/tmp/generated/cove.png"}],
                    "late_fill": {
                        "status": "completed",
                        "completed_capabilities": ["image_generation"],
                        "fill_results": [
                            {
                                "branch_id": "branch-image_generation-1",
                                "phase_id": "phase-2",
                                "capability": "image_generation",
                                "saved_image_path": "/tmp/generated/cove.png",
                            }
                        ],
                    },
                },
                request_payload={"prompt": "Describe a cove, then make an image."},
            )
            persist_response_frame(successor_frame, frames_dir=frames_dir)

            ledger_lines = [
                json.loads(line)
                for line in (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(ledger_lines), 2)
        self.assertEqual(ledger_lines[0]["frame_relation"]["kind"], "initial")
        self.assertNotIn("late_fill", ledger_lines[0])
        self.assertEqual(ledger_lines[0]["output"]["text"], "Initial answer while image continues.")
        self.assertEqual(ledger_lines[1]["frame_relation"]["kind"], "late_fill_successor")
        self.assertEqual(ledger_lines[1]["frame_relation"]["parent_frame_id"], ledger_lines[0]["frame_id"])
        self.assertEqual(ledger_lines[1]["late_fill"]["status"], "completed")

    def test_load_latest_response_state_recovers_current_successor_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_recover_successor"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                    },
                    request_payload={"prompt": "make an image after text"},
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                        "saved_image_path": "/tmp/generated/recovered.png",
                        "artifacts": [{"type": "image", "path": "/tmp/generated/recovered.png"}],
                        "late_fill": {
                            "status": "completed",
                            "completed_capabilities": ["image_generation"],
                        },
                    },
                    request_payload={"prompt": "make an image after text"},
                ),
                frames_dir=frames_dir,
            )

            state = load_latest_response_state(response_id, frames_dir=frames_dir)

        self.assertTrue(state["ok"])
        self.assertEqual(state["frame_count"], 2)
        payload = state["response_payload"]
        self.assertEqual(payload["id"], response_id)
        self.assertEqual(payload["late_fill"]["status"], "completed")
        self.assertEqual(payload["artifacts"][0]["path"], "/tmp/generated/recovered.png")
        self.assertEqual(payload["response_frame"]["frame_relation"]["kind"], "late_fill_successor")

    def test_graph_patch_reopen_successor_relation_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_graph_patch_reopen_successor"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Frozen parent response.",
                    },
                    request_payload={"prompt": "make an image after the parent froze"},
                ),
                frames_dir=frames_dir,
            )
            parent = json.loads(
                (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "late_fill_pending",
                        "output_text": "Frozen parent response.",
                        "frame_relation": {
                            "kind": "graph_patch_reopen_successor",
                            "reason": "graph_patch_reopen",
                            "parent_frame_id": parent["frame_id"],
                            "parent_frame_sequence": parent["frame_sequence"],
                        },
                        "runtime": {
                            "request_phase_graph": {
                                "successor_reopen_requests": [
                                    {
                                        "kind": "ollmo.graph_patch_successor_reopen_request",
                                        "status": "applied_to_successor",
                                        "patch_id": "patch-terminal-successor",
                                    }
                                ]
                            }
                        },
                        "late_fill": {"status": "pending"},
                    },
                    request_payload={"prompt": "make an image after the parent froze"},
                ),
                frames_dir=frames_dir,
            )

            state = load_latest_response_state(response_id, frames_dir=frames_dir)
            index = load_response_frame_index(frames_dir=frames_dir)["responses"][response_id]

        relation = state["response_frame"]["frame_relation"]
        self.assertEqual(relation["kind"], "graph_patch_reopen_successor")
        self.assertEqual(relation["reason"], "graph_patch_reopen")
        self.assertEqual(relation["parent_frame_id"], parent["frame_id"])
        self.assertEqual(index["frame_relation"]["kind"], "graph_patch_reopen_successor")
        self.assertEqual(state["response_payload"]["late_fill"]["status"], "pending")

    def test_response_frame_current_index_tracks_initial_and_successor_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_index_successor"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                    },
                    request_payload={"prompt": "answer then image"},
                ),
                frames_dir=frames_dir,
            )
            first_index = load_response_frame_index(frames_dir=frames_dir)
            first_entry = first_index["responses"][response_id]
            self.assertTrue(first_index["ok"])
            self.assertEqual(first_entry["latest_frame_sequence"], 1)
            self.assertEqual(first_entry["frame_relation"]["kind"], "initial")

            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                        "late_fill": {"status": "completed"},
                        "artifacts": [{"type": "image", "path": "/tmp/generated/indexed.png"}],
                    },
                    request_payload={"prompt": "answer then image"},
                ),
                frames_dir=frames_dir,
            )
            ledger_lines = [
                json.loads(line)
                for line in (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            second_index = load_response_frame_index(frames_dir=frames_dir)
            second_entry = second_index["responses"][response_id]

        self.assertTrue(second_index["ok"])
        self.assertEqual(second_entry["latest_frame_sequence"], 2)
        self.assertEqual(second_entry["latest_frame_id"], ledger_lines[-1]["frame_id"])
        self.assertEqual(second_entry["frame_relation"]["kind"], "late_fill_successor")
        self.assertEqual(second_entry["frame_relation"]["parent_frame_id"], ledger_lines[0]["frame_id"])

    def test_response_frame_append_repairs_legacy_line_count_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            for index in range(4):
                response_id = f"resp_legacy_line_offset_{index}"
                persist_response_frame(
                    build_response_frame(
                        {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "output_text": f"Seed {index}.",
                        },
                        request_payload={"prompt": f"seed {index}"},
                    ),
                    frames_dir=frames_dir,
                )

            index_path = frames_dir / "current_index.json"
            legacy_index = json.loads(index_path.read_text(encoding="utf-8"))
            legacy_index.pop("ledger_line_count_verified_size_bytes", None)
            legacy_index["ledger_line_count"] -= 4
            for entry in legacy_index["responses"].values():
                entry["line_offset"] -= 4
            index_path.write_text(json.dumps(legacy_index), encoding="utf-8")

            response_id = "resp_after_legacy_line_offset_drift"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Appended after legacy drift.",
                    },
                    request_payload={"prompt": "append after legacy drift"},
                ),
                frames_dir=frames_dir,
            )
            current_index = load_response_frame_index(frames_dir=frames_dir)
            current_entry = current_index["responses"][response_id]
            ledger_path = frames_dir / "responses.jsonl"
            ledger_size = ledger_path.stat().st_size

        self.assertEqual(current_entry["line_offset"], 4)
        self.assertEqual(current_index["ledger_line_count"], 5)
        self.assertEqual(
            current_index["ledger_line_count_verified_size_bytes"],
            ledger_size,
        )

    def test_repeated_successor_appends_get_unique_monotonic_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_repeated_successor_identity"
            initial_frame = build_response_frame(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "output_text": "Text while branches continue.",
                },
                request_payload={"prompt": "text then image then audio"},
            )
            persist_response_frame(initial_frame, frames_dir=frames_dir)
            first_ledger_frame = json.loads(
                (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            stale_successor_relation = {
                "kind": "late_fill_successor",
                "response_id": response_id,
                "parent_response_id": response_id,
                "parent_frame_id": first_ledger_frame["frame_id"],
                "parent_frame_sequence": first_ledger_frame["frame_sequence"],
            }

            image_successor = build_response_frame(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "output_text": "Text while branches continue.",
                    "late_fill": {"status": "running"},
                    "artifacts": [{"type": "image", "path": "/tmp/generated/notebook.png"}],
                },
                request_payload={"prompt": "text then image then audio"},
            )
            image_successor["frame_id"] = f"{response_id}:frame-2"
            image_successor["frame_sequence"] = 2
            image_successor["frame_relation"] = dict(stale_successor_relation)
            persist_response_frame(image_successor, frames_dir=frames_dir)

            audio_successor_with_stale_metadata = build_response_frame(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "output_text": "Text while branches continue.",
                    "late_fill": {"status": "completed"},
                    "artifacts": [
                        {"type": "image", "path": "/tmp/generated/notebook.png"},
                        {"type": "audio", "path": "/tmp/generated/notebook.wav"},
                    ],
                },
                request_payload={"prompt": "text then image then audio"},
            )
            audio_successor_with_stale_metadata["frame_id"] = f"{response_id}:frame-2"
            audio_successor_with_stale_metadata["frame_sequence"] = 2
            audio_successor_with_stale_metadata["frame_relation"] = dict(stale_successor_relation)
            persist_response_frame(audio_successor_with_stale_metadata, frames_dir=frames_dir)

            ledger_lines = [
                json.loads(line)
                for line in (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            latest_state = load_latest_response_state(response_id, frames_dir=frames_dir)
            latest_index = load_response_frame_index(frames_dir=frames_dir)["responses"][response_id]

        self.assertEqual([frame["frame_sequence"] for frame in ledger_lines], [1, 2, 3])
        self.assertEqual(
            [frame["frame_id"] for frame in ledger_lines],
            [
                f"{response_id}:frame-1",
                f"{response_id}:frame-2",
                f"{response_id}:frame-3",
            ],
        )
        self.assertEqual(ledger_lines[2]["frame_relation"]["kind"], "late_fill_successor")
        self.assertEqual(ledger_lines[2]["frame_relation"]["parent_frame_id"], ledger_lines[1]["frame_id"])
        self.assertEqual(ledger_lines[2]["frame_relation"]["parent_frame_sequence"], 2)
        self.assertEqual(latest_state["response_frame"]["frame_sequence"], 3)
        self.assertEqual(latest_state["response_frame"]["frame_relation"]["parent_frame_id"], ledger_lines[1]["frame_id"])
        self.assertEqual(latest_state["response_payload"]["late_fill"]["status"], "completed")
        self.assertEqual(latest_index["latest_frame_sequence"], 3)
        self.assertEqual(latest_index["latest_frame_id"], f"{response_id}:frame-3")

    def test_load_latest_response_wire_state_is_index_only_bounded_and_preserves_effective_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_bounded_wire_successor"
            sentinel = "WIRE_SIDECAR_SENTINEL_" + ("x" * 1_500_000)
            artifact = {
                "artifact_id": "image_wire_fixture",
                "artifact_ref": "artifact:image_wire_fixture",
                "kind": "image",
                "mime_type": "image/png",
                "path": "/tmp/artifacts/wire-fixture.png",
                "type": "image",
            }
            output = {
                "artifact_ref": artifact["artifact_ref"],
                "lifecycle": "materialized_output",
                "slot_id": "output-image",
                "status": "fulfilled",
                "type": "image",
            }
            output_slot = {
                "artifact_ref": artifact["artifact_ref"],
                "lifecycle": "materialized_output",
                "slot_id": "output-image",
                "status": "fulfilled",
                "type": "image",
            }
            output_branch = {
                "artifact_ref": artifact["artifact_ref"],
                "branch_id": "branch-image",
                "lifecycle": "materialized_output",
                "slot_id": "output-image",
                "status": "fulfilled",
                "type": "image",
            }
            output_items = [
                {
                    "content": [
                        {"text": "The image is ready.", "type": "output_text"}
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
            input_artifacts = [
                {
                    "artifact_ref": "artifact:wire_input",
                    "path": "/tmp/artifacts/wire-input.txt",
                    "type": "text",
                }
            ]
            reference_artifacts = [
                {
                    "artifact_ref": "artifact:wire_reference",
                    "path": "/tmp/artifacts/wire-reference.png",
                    "type": "image",
                }
            ]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "runtime": {
                        "request_phase_graph": {
                            "kind": "ollmo.request_phase_graph",
                            "nodes": [
                                {
                                    "id": "phase-large-private-truth",
                                    "private_truth": sentinel,
                                }
                            ],
                        },
                    },
                    "output": {"item_count": 1, "outputs": [output]},
                    "artifacts": {"output": [artifact]},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "canonical_status_field": "lifecycle_state",
                        "status_semantics": {"terminal": True},
                        "output_text": "The image is ready.",
                        "output": output_items,
                        "outputs": [output],
                        "output_slots": [output_slot],
                        "output_branches": [output_branch],
                        "artifacts": [artifact],
                        "input_artifacts": input_artifacts,
                        "reference_artifacts": reference_artifacts,
                        "saved_image_path": artifact["path"],
                    },
                },
                frames_dir=frames_dir,
            )
            first_index = load_response_frame_index(frames_dir=frames_dir)
            inherited_runtime_ref = first_index["responses"][response_id][
                "effective_snapshot_manifest"
            ]["runtime"]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "late_fill": {"completed_branch_count": 1, "status": "completed"},
                    "output": {"item_count": 1, "outputs": [output]},
                    "artifacts": {"output": [artifact]},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "canonical_status_field": "lifecycle_state",
                        "status_semantics": {"terminal": True},
                        "output_text": "The image is ready.",
                        "output": output_items,
                        "outputs": [output],
                        "output_slots": [output_slot],
                        "output_branches": [output_branch],
                        "artifacts": [artifact],
                        "input_artifacts": input_artifacts,
                        "reference_artifacts": reference_artifacts,
                        "saved_image_path": artifact["path"],
                    },
                },
                frames_dir=frames_dir,
            )
            current_index = load_response_frame_index(frames_dir=frames_dir)
            current_entry = current_index["responses"][response_id]
            effective_manifest = current_entry["effective_snapshot_manifest"]
            ledger_frames = [
                json.loads(line)
                for line in (frames_dir / "responses.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            before = {
                str(path.relative_to(frames_dir)): (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in frames_dir.rglob("*")
                if path.is_file()
            }

            with (
                patch.object(
                    response_frames_module,
                    "_read_snapshot_ref_payload",
                    side_effect=AssertionError("wire projection must not hydrate sidecars"),
                ),
                patch.object(
                    response_frames_module,
                    "_read_observation_snapshot_payload",
                    side_effect=AssertionError("wire projection must not read sidecars"),
                ),
                patch.object(
                    response_frames_module,
                    "_iter_ledger_frames",
                    side_effect=AssertionError("wire projection must not scan the ledger"),
                ),
            ):
                wire_state = load_latest_response_wire_state(
                    response_id,
                    frames_dir=frames_dir,
                    index_state=current_index,
                )

            after = {
                str(path.relative_to(frames_dir)): (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in frames_dir.rglob("*")
                if path.is_file()
            }
            encoded_wire = json.dumps(
                wire_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            inherited_raw = (
                frames_dir / inherited_runtime_ref["path"]
            ).read_bytes().rstrip(b"\n")

        self.assertTrue(wire_state["ok"])
        self.assertEqual(before, after)
        self.assertLess(len(encoded_wire), 128 * 1024)
        self.assertNotIn("WIRE_SIDECAR_SENTINEL_", encoded_wire.decode("utf-8"))
        payload = wire_state["response_payload"]
        self.assertEqual(payload["lifecycle_state"], "completed")
        self.assertEqual(payload["outputs"], [output])
        self.assertEqual(payload["output_slots"], [output_slot])
        self.assertEqual(payload["output_branches"], [output_branch])
        self.assertEqual(payload["artifacts"], [artifact])
        self.assertEqual(payload["output_text"], "The image is ready.")
        self.assertEqual(payload["output"], output_items)
        self.assertEqual(payload["canonical_status_field"], "lifecycle_state")
        self.assertEqual(payload["status_semantics"], {"terminal": True})
        self.assertEqual(payload["input_artifacts"], input_artifacts)
        self.assertEqual(payload["reference_artifacts"], reference_artifacts)
        self.assertEqual(payload["saved_image_path"], artifact["path"])
        self.assertEqual(payload["response_frame"]["frame_sequence"], 2)
        self.assertEqual(
            payload["response_frame"]["frame_relation"]["parent_frame_id"],
            ledger_frames[0]["frame_id"],
        )
        wire_manifest = payload["response_frame"]["external_snapshots"]["items"]
        self.assertEqual(wire_manifest, effective_manifest)
        self.assertEqual(wire_manifest["runtime"], inherited_runtime_ref)
        self.assertEqual(
            hashlib.sha256(inherited_raw).hexdigest(),
            inherited_runtime_ref["sha256"],
        )
        self.assertEqual(len(inherited_raw), inherited_runtime_ref["size_bytes"])
        self.assertNotIn(
            "runtime",
            payload["response_frame"]["external_snapshots"]["delta_items"],
        )
        self.assertEqual(payload["wire_projection"]["sidecar_reads"], 0)
        self.assertEqual(payload["wire_projection"]["sidecar_hydration"], "none")
        self.assertEqual(
            payload["wire_projection"]["effective_snapshot_manifest_source"],
            "current_index",
        )

    def test_wire_projection_bounds_public_bodies_without_hydrating_cas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_wire_large_public_bodies"
            large_body = "PUBLIC_BODY_START_" + ("x" * 2_000_000) + "_PUBLIC_BODY_END"
            large_media = "data:image/png;base64," + ("a" * 1_500_000) + "_MEDIA_END"
            output = {
                "slot_id": "slot-large-text",
                "status": "fulfilled",
                "type": "text",
                "value": large_body,
            }
            artifact = {
                "artifact_id": "artifact-large-text",
                "artifact_ref": "artifact:large-text",
                "content": large_body,
                "path": "/tmp/artifacts/large.txt",
                "type": "text",
            }
            compatibility_output = [
                {
                    "content": [{"text": large_body, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 1, "outputs": [output]},
                    "artifacts": {"output": [artifact]},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "output_text": large_body,
                        "output": compatibility_output,
                        "outputs": [output],
                        "artifacts": [artifact],
                        "input_artifacts": [
                            {
                                "artifact_ref": "artifact:large-input",
                                "audio_base64": large_media,
                                "type": "audio",
                            }
                        ],
                    },
                },
                frames_dir=frames_dir,
            )
            current_index = load_response_frame_index(frames_dir=frames_dir)

            with (
                patch.object(
                    response_frames_module,
                    "_read_snapshot_ref_payload",
                    side_effect=AssertionError("wire projection must not hydrate CAS"),
                ),
                patch.object(
                    response_frames_module,
                    "_read_observation_snapshot_payload",
                    side_effect=AssertionError("wire projection must not read CAS"),
                ),
                patch.object(
                    response_frames_module,
                    "_iter_ledger_frames",
                    side_effect=AssertionError("wire projection must not scan the ledger"),
                ),
            ):
                wire_state = load_latest_response_wire_state(
                    response_id,
                    frames_dir=frames_dir,
                    index_state=current_index,
                )
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            encoded_wire = json.dumps(
                wire_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            output_text_ref = wire_state["response_payload"][
                "output_text_snapshot_ref"
            ]
            exact_output_text = _read_snapshot_ref_payload(
                output_text_ref,
                frames_dir=frames_dir,
            )

        self.assertTrue(wire_state["ok"])
        self.assertLess(len(encoded_wire), 128 * 1024)
        self.assertNotIn("_PUBLIC_BODY_END", encoded_wire.decode("utf-8"))
        self.assertNotIn("_MEDIA_END", encoded_wire.decode("utf-8"))
        payload = wire_state["response_payload"]
        self.assertEqual(payload["output_text_length_chars"], len(large_body))
        self.assertEqual(
            payload["output_text_sha256"],
            hashlib.sha256(large_body.encode("utf-8")).hexdigest(),
        )
        self.assertTrue(payload["output_text_preview_truncated"])
        self.assertEqual(payload["outputs"][0]["value_length_chars"], len(large_body))
        self.assertEqual(payload["artifacts"][0]["content_length_chars"], len(large_body))
        input_projection = payload["input_artifacts"][0]
        self.assertEqual(input_projection["audio_base64_length_chars"], len(large_media))
        self.assertTrue(input_projection["audio_base64_preview_truncated"])
        self.assertEqual(
            output_text_ref["projection_role"],
            "public_body_exact",
        )
        self.assertEqual(output_text_ref["truth_preservation"], "exact_content_addressed_sidecar")
        self.assertTrue(output_text_ref["content_addressed"])
        self.assertEqual(exact_output_text, large_body)
        self.assertTrue(canonical_state["ok"])
        canonical_frame = canonical_state["response_frame"]["current_state"]
        self.assertEqual(canonical_frame["output_text"], large_body)
        self.assertEqual(canonical_frame["outputs"][0]["value"], large_body)
        self.assertEqual(canonical_frame["artifacts"][0]["content"], large_body)

    def test_canonical_recovery_fails_closed_when_exact_public_body_sidecar_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_missing_exact_public_body"
            output_text = "exact-public-body-" + ("x" * (300 * 1024))
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 0, "text": output_text},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "output_text": output_text,
                    },
                },
                frames_dir=frames_dir,
            )
            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            ref = wire_state["response_payload"]["output_text_snapshot_ref"]
            (frames_dir / ref["path"]).unlink()

            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertFalse(canonical_state["ok"])
        self.assertEqual(canonical_state["status_code"], 409)
        self.assertEqual(
            canonical_state["error"]["code"],
            "response_frame_public_body_snapshot_unavailable",
        )
        self.assertEqual(canonical_state["error"]["expected_sha256"], ref["sha256"])

    def test_public_collection_preview_bounds_one_enormous_mapping_entry_and_restores_exact_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_enormous_public_mapping_key"
            enormous_key = "diagnostic-key-" + ("k" * (1100 * 1024))
            enormous_value = "diagnostic-value-" + ("v" * (128 * 1024))
            exact_details = {enormous_key: enormous_value}
            exact_encoded, exact_sha256 = (
                response_frames_module._response_wire_json_identity(exact_details)
            )
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "failed",
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "failed",
                        "lifecycle_state": "failed",
                        "error_detail": {
                            "details": exact_details,
                            "details_count": -1,
                            "details_length_chars": -1,
                            "details_preview_truncated": False,
                            "details_projection_truncated": False,
                            "details_sha256": "0" * 64,
                            "details_size_bytes": 1,
                            "details_snapshot_ref": {"sha256": "untrusted"},
                        },
                    },
                },
                frames_dir=frames_dir,
            )
            ledger_frame = json.loads(
                (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            compact_error = ledger_frame["current_state"]["error_detail"]
            compact_details = compact_error["details"]
            details_ref = compact_error["details_snapshot_ref"]
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            restored_details = _read_snapshot_ref_payload(
                details_ref,
                frames_dir=frames_dir,
            )

        self.assertLessEqual(
            len(
                json.dumps(
                    compact_details,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            64 * 1024,
        )
        self.assertEqual(len(compact_details), 1)
        preview_key = next(iter(compact_details))
        self.assertLessEqual(len(preview_key.encode("utf-8")), 256)
        self.assertIn("#sha256:", preview_key)
        self.assertEqual(compact_error["details_count"], 1)
        self.assertEqual(compact_error["details_size_bytes"], len(exact_encoded))
        self.assertEqual(compact_error["details_sha256"], exact_sha256)
        self.assertTrue(compact_error["details_projection_truncated"])
        self.assertNotIn("details_length_chars", compact_error)
        self.assertNotIn("details_preview_truncated", compact_error)
        self.assertEqual(details_ref["projection_role"], "public_body_exact")
        self.assertEqual(
            set(details_ref["public_projection_metadata_keys"]),
            {
                "details_count",
                "details_length_chars",
                "details_preview_truncated",
                "details_projection_truncated",
                "details_sha256",
                "details_size_bytes",
            },
        )
        self.assertEqual(restored_details, exact_details)
        self.assertTrue(canonical_state["ok"])
        canonical_error = canonical_state["response_payload"]["error_detail"]
        self.assertEqual(canonical_error["details"], exact_details)
        for suffix in (
            "count",
            "length_chars",
            "preview_truncated",
            "projection_truncated",
            "sha256",
            "size_bytes",
            "snapshot_ref",
        ):
            self.assertNotIn(f"details_{suffix}", canonical_error)

    def test_wire_bounded_mapping_keys_use_unique_deterministic_hash_suffixes(self):
        shared_prefix = "shared-prefix-" + ("x" * 512)
        source = {
            f"{shared_prefix}-alpha": "first",
            f"{shared_prefix}-bravo": "second",
        }
        stats = {}

        first = response_frames_module._response_wire_bounded_value(
            source,
            stats=stats,
        )
        second = response_frames_module._response_wire_bounded_value(source)

        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"first", "second"})
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)
        self.assertTrue(all("#sha256:" in key for key in first))
        self.assertTrue(all(len(key.encode("utf-8")) <= 256 for key in first))
        self.assertEqual(stats["bounded_mapping_key_count"], 2)

    def test_public_string_compaction_overwrites_reserved_metadata_and_hydrates_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_untrusted_public_string_metadata"
            output_text = "authoritative-output-" + ("z" * (300 * 1024))
            encoded = output_text.encode("utf-8")
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 0, "text": output_text},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "output_text": output_text,
                        "output_text_count": -1,
                        "output_text_length_chars": -1,
                        "output_text_preview_truncated": False,
                        "output_text_projection_truncated": False,
                        "output_text_sha256": "0" * 64,
                        "output_text_size_bytes": 1,
                        "output_text_snapshot_ref": {"sha256": "untrusted"},
                    },
                },
                frames_dir=frames_dir,
            )
            ledger_frame = json.loads(
                (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            compact_current = ledger_frame["current_state"]
            output_text_ref = compact_current["output_text_snapshot_ref"]
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertEqual(compact_current["output_text_length_chars"], len(output_text))
        self.assertEqual(compact_current["output_text_size_bytes"], len(encoded))
        self.assertEqual(
            compact_current["output_text_sha256"],
            hashlib.sha256(encoded).hexdigest(),
        )
        self.assertTrue(compact_current["output_text_preview_truncated"])
        self.assertNotIn("output_text_count", compact_current)
        self.assertNotIn("output_text_projection_truncated", compact_current)
        self.assertEqual(output_text_ref["projection_role"], "public_body_exact")
        self.assertEqual(
            set(output_text_ref["public_projection_metadata_keys"]),
            {
                "output_text_count",
                "output_text_length_chars",
                "output_text_preview_truncated",
                "output_text_projection_truncated",
                "output_text_sha256",
                "output_text_size_bytes",
            },
        )
        self.assertTrue(canonical_state["ok"])
        canonical_payload = canonical_state["response_payload"]
        self.assertEqual(canonical_payload["output_text"], output_text)
        for suffix in (
            "count",
            "length_chars",
            "preview_truncated",
            "projection_truncated",
            "sha256",
            "size_bytes",
            "snapshot_ref",
        ):
            self.assertNotIn(f"output_text_{suffix}", canonical_payload)

    def test_aggregate_public_collections_are_cas_compacted_under_ledger_and_wire_budgets(self):
        record = {f"k{index:03d}": "x" * 4000 for index in range(230)}
        collection = [record]
        response_id = "resp_aggregate_bound"
        current_collection_keys = (
            "outputs",
            "output_slots",
            "output_branches",
            "artifacts",
            "input_artifacts",
            "reference_artifacts",
            "output",
            "saved_text_artifacts",
            "text_artifact_requests",
        )
        frame = {
            "frame_version": 9,
            "kind": "ollmo.response_frame",
            "response_id": response_id,
            "status": "completed",
            "output": {"item_count": 1, "outputs": collection},
            "artifacts": {"output": collection},
            "current_state": {
                "id": response_id,
                "status": "completed",
                "lifecycle_state": "completed",
                **{key: collection for key in current_collection_keys},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(frame, frames_dir=frames_dir)
            ledger_path = frames_dir / "responses.jsonl"
            ledger_frame = json.loads(
                ledger_path.read_text(encoding="utf-8").splitlines()[0]
            )
            with patch.object(
                response_frames_module,
                "_read_snapshot_ref_payload",
                side_effect=AssertionError("wire projection must not hydrate CAS"),
            ):
                wire_state = load_latest_response_wire_state(
                    response_id,
                    frames_dir=frames_dir,
                )
            wire_size = len(
                json.dumps(
                    wire_state,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            ledger_size = ledger_path.stat().st_size
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            exact_ref = ledger_frame["current_state"]["outputs_snapshot_ref"]
            exact_snapshot = _read_snapshot_ref_payload(
                exact_ref,
                frames_dir=frames_dir,
            )
            (frames_dir / exact_ref["path"]).unlink()
            missing_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertLessEqual(ledger_size, 8 * 1024 * 1024)
        self.assertTrue(wire_state["ok"])
        self.assertTrue(wire_state["bounded_wire_projection"])
        self.assertLessEqual(wire_size, 8 * 1024 * 1024)
        self.assertEqual(
            wire_state["response_payload"]["wire_projection"]
            ["public_projection_truncation"]
            ["aggregate_externalized_collection_count"],
            11,
        )
        compaction = ledger_frame["public_body_compaction"]
        self.assertEqual(compaction["externalized_collection_count"], 11)
        self.assertEqual(compaction["externalized_string_count"], 0)
        self.assertEqual(exact_ref["projection_role"], "public_body_exact")
        self.assertEqual(exact_snapshot, collection)
        self.assertTrue(canonical_state["ok"])
        canonical_current = canonical_state["response_frame"]["current_state"]
        for key in current_collection_keys:
            self.assertEqual(canonical_current[key], collection)
        self.assertFalse(missing_state["ok"])
        self.assertEqual(missing_state["status_code"], 409)
        self.assertEqual(
            missing_state["error"]["code"],
            "response_frame_public_body_snapshot_unavailable",
        )

    def test_aggregate_many_medium_strings_are_externalized_without_truth_loss(self):
        response_id = "resp_aggregate_medium_strings"
        bodies = {
            f"medium_body_{index}": f"body-{index}-" + ("m" * (192 * 1024))
            for index in range(48)
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        **bodies,
                    },
                },
                frames_dir=frames_dir,
            )
            ledger_path = frames_dir / "responses.jsonl"
            ledger_frame = json.loads(
                ledger_path.read_text(encoding="utf-8").splitlines()[0]
            )
            ledger_size = ledger_path.stat().st_size
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertLessEqual(ledger_size, 8 * 1024 * 1024)
        self.assertGreater(
            ledger_frame["public_body_compaction"]["externalized_string_count"],
            0,
        )
        self.assertTrue(canonical_state["ok"])
        canonical_current = canonical_state["response_frame"]["current_state"]
        for key, body in bodies.items():
            self.assertEqual(canonical_current[key], body)

    def test_ref_shaped_user_json_and_deep_no_ref_payload_remain_ordinary_truth(self):
        response_id = "resp_ref_shaped_user_json"
        ref_shaped_value = {
            "kind": "ollmo.response_frame_snapshot_ref",
            "json_path": "user.claim",
            "path": "snapshots/content_sha256/ff/not-real.json",
            "sha256": "f" * 64,
            "size_bytes": 123,
            "content_addressed": True,
        }
        deep_value = {"leaf": "deep truth"}
        for index in range(135):
            deep_value = {f"level_{index}": deep_value}
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "error_detail": {
                            "model_value": ref_shaped_value,
                            "deep_value": deep_value,
                        },
                    },
                },
                frames_dir=frames_dir,
            )
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(canonical_state["ok"])
        error_detail = canonical_state["response_payload"]["error_detail"]
        self.assertEqual(error_detail["model_value"], ref_shaped_value)
        observed = error_detail["deep_value"]
        for index in reversed(range(135)):
            observed = observed[f"level_{index}"]
        self.assertEqual(observed, {"leaf": "deep truth"})

    def test_cross_response_forged_public_ref_is_preserved_as_data_not_hydrated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            source_id = "resp_public_ref_source"
            target_id = "resp_public_ref_target"
            source_body = "source-secret-" + ("s" * (300 * 1024))
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": source_id,
                    "status": "completed",
                    "output": {"item_count": 0, "text": source_body},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": source_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "output_text": source_body,
                    },
                },
                frames_dir=frames_dir,
            )
            source_wire = load_latest_response_wire_state(
                source_id,
                frames_dir=frames_dir,
            )
            stolen_ref = source_wire["response_payload"]["output_text_snapshot_ref"]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": target_id,
                    "status": "completed",
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": target_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "error_detail": {
                            "note": "keep the ref-shaped value",
                            "stolen_snapshot_ref": stolen_ref,
                        },
                    },
                },
                frames_dir=frames_dir,
            )
            target_state = load_latest_response_state(
                target_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(target_state["ok"])
        target_error = target_state["response_payload"]["error_detail"]
        self.assertEqual(target_error["stolen_snapshot_ref"], stolen_ref)
        self.assertNotIn("stolen", target_error)

    def test_manifest_authorized_canonical_reads_do_not_expand_copied_runtime_work_tree_or_late_fill_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            source_id = "resp_nested_ref_source"
            target_id = "resp_nested_ref_target"
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": source_id,
                    "status": "completed",
                    "runtime": {"private_runtime_marker": "SOURCE_ONLY"},
                    "late_fill": {
                        "status": "completed",
                        "private_late_fill_marker": "SOURCE_ONLY",
                    },
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": source_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "runtime": {"private_runtime_marker": "SOURCE_ONLY"},
                        "work_tree": {"private_work_tree_marker": "SOURCE_ONLY"},
                        "late_fill": {
                            "status": "completed",
                            "private_late_fill_marker": "SOURCE_ONLY",
                        },
                    },
                },
                frames_dir=frames_dir,
            )
            source_index = load_response_frame_index(frames_dir=frames_dir)
            source_manifest = source_index["responses"][source_id][
                "effective_snapshot_manifest"
            ]
            source_runtime_ref = source_manifest["runtime"]
            source_work_tree_ref = source_manifest["current_state.work_tree"]
            source_late_fill_ref = source_manifest["late_fill"]
            source_current_late_fill_ref = source_manifest["current_state.late_fill"]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": target_id,
                    "status": "completed",
                    "runtime": {
                        "owner": "TARGET_ONLY",
                        "developer_diagnostics": {
                            "copied_snapshot_ref": source_runtime_ref,
                        },
                    },
                    "late_fill": {
                        "status": "completed",
                        "owner": "TARGET_ONLY",
                        "copied_snapshot_ref": source_late_fill_ref,
                    },
                    "planning": {
                        "work_tree": {
                            "owner": "TARGET_ONLY",
                            "copied_snapshot_ref": source_work_tree_ref,
                        }
                    },
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": target_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "runtime_snapshot_ref": source_runtime_ref,
                        "work_tree_snapshot_ref": source_work_tree_ref,
                        "late_fill_snapshot_ref": source_current_late_fill_ref,
                    },
                },
                frames_dir=frames_dir,
            )
            target_state = load_latest_response_state(
                target_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(target_state["ok"])
        payload = target_state["response_payload"]
        self.assertEqual(payload["runtime"]["owner"], "TARGET_ONLY")
        self.assertNotIn("private_runtime_marker", payload["runtime"])
        runtime_diagnostics = payload["runtime"]["developer_diagnostics"]
        self.assertIn("copied_snapshot_ref", runtime_diagnostics)
        self.assertNotIn("copied", runtime_diagnostics)
        self.assertEqual(payload["work_tree"]["owner"], "TARGET_ONLY")
        self.assertIn("copied_snapshot_ref", payload["work_tree"])
        self.assertNotIn("copied", payload["work_tree"])
        self.assertEqual(payload["late_fill"]["owner"], "TARGET_ONLY")
        self.assertIn("copied_snapshot_ref", payload["late_fill"])
        self.assertNotIn("copied", payload["late_fill"])
        self.assertEqual(payload["runtime_snapshot_ref"], source_runtime_ref)
        self.assertEqual(payload["work_tree_snapshot_ref"], source_work_tree_ref)
        self.assertEqual(
            payload["late_fill_snapshot_ref"],
            source_current_late_fill_ref,
        )

    def test_corrupt_existing_content_addressed_snapshot_is_atomically_repaired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            body = "shared-authoritative-body-" + ("c" * (300 * 1024))

            def persist_body(response_id):
                persist_response_frame(
                    {
                        "frame_version": 9,
                        "kind": "ollmo.response_frame",
                        "response_id": response_id,
                        "status": "completed",
                        "output": {"item_count": 0, "text": body},
                        "artifacts": {"output": []},
                        "current_state": {
                            "id": response_id,
                            "status": "completed",
                            "lifecycle_state": "completed",
                            "output_text": body,
                        },
                    },
                    frames_dir=frames_dir,
                )

            persist_body("resp_cas_repair_a")
            first_wire = load_latest_response_wire_state(
                "resp_cas_repair_a",
                frames_dir=frames_dir,
            )
            ref = first_wire["response_payload"]["output_text_snapshot_ref"]
            target = frames_dir / ref["path"]
            target.write_bytes(b"corrupt-existing-cas\n")
            persist_body("resp_cas_repair_b")
            repaired_payload = target.read_bytes().rstrip(b"\n")
            first_state = load_latest_response_state(
                "resp_cas_repair_a",
                frames_dir=frames_dir,
            )
            second_state = load_latest_response_state(
                "resp_cas_repair_b",
                frames_dir=frames_dir,
            )
            temp_residue = list(target.parent.glob(f".{target.name}.*.tmp"))

        self.assertEqual(hashlib.sha256(repaired_payload).hexdigest(), ref["sha256"])
        self.assertEqual(len(repaired_payload), ref["size_bytes"])
        self.assertTrue(first_state["ok"])
        self.assertTrue(second_state["ok"])
        self.assertEqual(first_state["response_payload"]["output_text"], body)
        self.assertEqual(second_state["response_payload"]["output_text"], body)
        self.assertEqual(temp_residue, [])

    def test_atomic_index_replace_failure_preserves_prior_verified_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_atomic_index_prior"
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                    },
                },
                frames_dir=frames_dir,
            )
            ledger_path = frames_dir / "responses.jsonl"
            index_path = frames_dir / "current_index.json"
            prior_bytes = index_path.read_bytes()
            prior_index = load_response_frame_index(frames_dir=frames_dir)
            ledger_frame = json.loads(
                ledger_path.read_text(encoding="utf-8").splitlines()[0]
            )
            ledger_size = ledger_path.stat().st_size
            with patch.object(
                response_frames_module.os,
                "replace",
                side_effect=OSError("simulated atomic replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated atomic replace failure"):
                    response_frames_module._write_response_frame_index(
                        ledger_frame,
                        ledger_path=ledger_path,
                        line_offset=0,
                        byte_offset=0,
                        line_length=ledger_size,
                        ledger_size_bytes=ledger_size,
                        frames_dir=frames_dir,
                    )
            after_bytes = index_path.read_bytes()
            after_index = load_response_frame_index(frames_dir=frames_dir)
            temp_residue = list(frames_dir.glob(".current_index.json.*.tmp"))

        self.assertEqual(after_bytes, prior_bytes)
        self.assertTrue(prior_index["ok"])
        self.assertTrue(after_index["ok"])
        self.assertEqual(
            after_index["response_map_digest"],
            prior_index["response_map_digest"],
        )
        self.assertEqual(temp_residue, [])

    def test_sparse_high_batch_index_projects_in_constant_logical_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_sparse_high_batch_index"
            high_index = 10_000_000
            output = {
                "artifact_ref": "artifact:sparse-high",
                "batch_index": high_index,
                "slot_id": "slot-sparse-high",
                "status": "fulfilled",
                "type": "image",
            }
            artifact = {
                "artifact_id": "sparse-high",
                "artifact_ref": "artifact:sparse-high",
                "batch_index": high_index,
                "path": "/tmp/artifacts/sparse-high.png",
                "type": "image",
            }
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "batch": {"count": high_index, "prompts": []},
                    "output": {"item_count": 1, "outputs": [output]},
                    "artifacts": {"output": [artifact]},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "outputs": [output],
                        "artifacts": [artifact],
                    },
                },
                frames_dir=frames_dir,
            )
            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            wire_size = len(
                json.dumps(
                    wire_state,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        self.assertTrue(wire_state["ok"])
        payload = wire_state["response_payload"]
        self.assertEqual(payload["batch_count"], high_index)
        self.assertEqual(payload["batch_results_projected_count"], 1)
        self.assertEqual(payload["batch_result_indices"], [high_index])
        self.assertEqual(payload["results"][0]["index"], high_index)
        self.assertEqual(
            payload["batch_empty_result_handles_omitted"],
            high_index - 1,
        )
        self.assertTrue(payload["batch_results_projection_truncated"])
        self.assertFalse(payload["batch_projection_complete"])
        self.assertEqual(payload["batch_prompts_count"], 0)
        self.assertEqual(payload["batch_prompts_projected_count"], 0)
        self.assertLess(wire_size, 128 * 1024)

    def test_restart_recovery_preserves_route_usage_language_and_format_metadata(self):
        response_id = "resp_public_restart_metadata"
        response = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "lifecycle_state": "completed",
            "message_id": "msg-public-metadata",
            "route_source": "ghost_router",
            "route_reason": "capability and affinity match",
            "route_confidence": 0.91,
            "route_reuse_last_artifact": True,
            "route_artifact_ref": "artifact:route-input",
            "route_artifact_path": "/tmp/artifacts/route-input.png",
            "usage": {"input_tokens": 17, "output_tokens": 23},
            "lang_code": "de",
            "lang_code_source": "runtime_detection",
            "response_format": "json_schema",
            "output_format": "application/json",
            "output_text": "Metadaten bleiben erhalten.",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            frame = build_response_frame(response)
            persist_response_frame(frame, frames_dir=frames_dir)
            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            legacy_id = "resp_legacy_route_hoist"
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": legacy_id,
                    "status": "completed",
                    "route": {
                        "route_source": "legacy_router",
                        "route_reason": "persisted only on frame.route",
                        "route_confidence": 0.77,
                    },
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": legacy_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                    },
                },
                frames_dir=frames_dir,
            )
            legacy_wire = load_latest_response_wire_state(
                legacy_id,
                frames_dir=frames_dir,
            )
            legacy_canonical = load_latest_response_state(
                legacy_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(wire_state["ok"])
        self.assertTrue(canonical_state["ok"])
        for key in (
            "message_id",
            "route_source",
            "route_reason",
            "route_confidence",
            "route_reuse_last_artifact",
            "route_artifact_ref",
            "route_artifact_path",
            "usage",
            "lang_code",
            "lang_code_source",
            "response_format",
            "output_format",
        ):
            self.assertEqual(wire_state["response_payload"][key], response[key])
            self.assertEqual(canonical_state["response_payload"][key], response[key])
        self.assertEqual(
            wire_state["response_frame"]["route"]["route_source"],
            "ghost_router",
        )
        self.assertEqual(
            legacy_wire["response_payload"]["route_source"],
            "legacy_router",
        )
        self.assertEqual(
            legacy_canonical["response_payload"]["route_reason"],
            "persisted only on frame.route",
        )

    def test_canonical_recovery_preserves_large_singular_backend_result_by_ref(self):
        response_id = "resp_public_singular_backend_result"
        result = {
            "transcript": "Deterministic local transcript.",
            "language": "en",
            "segments": [
                {
                    "index": index,
                    "text": f"segment-{index}-" + ("spoken truth " * 9_000),
                }
                for index in range(10)
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "output_text": result["transcript"],
                        "result": result,
                    }
                ),
                frames_dir=frames_dir,
            )
            ledger_frame = json.loads(
                (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        current_state = ledger_frame["current_state"]
        self.assertIn("segments_snapshot_ref", current_state["result"])
        self.assertEqual(
            current_state["result"]["segments_snapshot_ref"]["projection_role"],
            "public_body_exact",
        )
        self.assertNotEqual(current_state["result"], result)
        self.assertTrue(wire_state["ok"])
        self.assertNotIn("result", wire_state["response_payload"])
        self.assertTrue(canonical_state["ok"])
        self.assertEqual(canonical_state["response_payload"]["result"], result)
        self.assertNotIn(
            "segments_snapshot_ref",
            canonical_state["response_payload"]["result"],
        )

    def test_canonical_recovery_validates_recursive_authoritative_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_missing_recursive_runtime_sidecar"
            runtime = {
                f"part_{index}_graph": {
                    "kind": "diagnostic_graph",
                    "payload": "g" * (70 * 1024),
                }
                for index in range(20)
            }
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "runtime": runtime,
                    }
                ),
                frames_dir=frames_dir,
            )
            index_state = load_response_frame_index(frames_dir=frames_dir)
            runtime_ref = index_state["responses"][response_id][
                "effective_snapshot_manifest"
            ]["runtime"]
            raw_runtime = _read_snapshot_ref_payload(
                runtime_ref,
                frames_dir=frames_dir,
                expand_child_refs=False,
            )
            child_ref = next(
                value
                for key, value in raw_runtime.items()
                if key.endswith("_snapshot_ref")
                and isinstance(value, dict)
                and value.get("kind") == "ollmo.response_frame_snapshot_ref"
            )
            (frames_dir / child_ref["path"]).unlink()

            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertFalse(canonical_state["ok"])
        self.assertEqual(canonical_state["status_code"], 409)
        self.assertEqual(
            canonical_state["error"]["code"],
            "response_frame_snapshot_unavailable",
        )
        self.assertEqual(
            canonical_state["error"]["expected_sha256"],
            child_ref["sha256"],
        )

    def test_failed_response_error_survives_wire_and_canonical_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_failed_error_recovery"
            error_message = "backend-exploded-" + ("e" * (300 * 1024))
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "status": "failed",
                        "lifecycle_state": "failed",
                        "error": {
                            "code": "backend_failure",
                            "message": error_message,
                        },
                    }
                ),
                frames_dir=frames_dir,
            )

            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(wire_state["ok"])
        wire_error = wire_state["response_payload"]["error"]
        self.assertEqual(wire_error["code"], "backend_failure")
        self.assertTrue(wire_error["message_preview_truncated"])
        self.assertEqual(
            wire_error["message_snapshot_ref"]["projection_role"],
            "public_body_exact",
        )
        self.assertTrue(canonical_state["ok"])
        self.assertEqual(
            canonical_state["response_payload"]["error"]["message"],
            error_message,
        )

    def test_wire_projection_keeps_five_thousand_character_answer_inline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_wire_normal_five_thousand_chars"
            output_text = "normal-answer-" + ("n" * 4_986)
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 0, "text": output_text},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "output_text": output_text,
                    },
                },
                frames_dir=frames_dir,
            )

            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(wire_state["ok"])
        payload = wire_state["response_payload"]
        self.assertEqual(len(output_text), 5_000)
        self.assertEqual(payload["output_text"], output_text)
        self.assertNotIn("output_text_snapshot_ref", payload)
        self.assertNotIn("output_text_preview_truncated", payload)

    def test_wire_projection_keeps_sixty_five_tiny_public_items_inline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_wire_sixty_five_tiny_items"
            outputs = [
                {
                    "artifact_ref": f"artifact:tiny-{index}",
                    "slot_id": f"slot-tiny-{index}",
                    "status": "fulfilled",
                    "type": "text",
                }
                for index in range(65)
            ]
            artifacts = [
                {
                    "artifact_id": f"tiny-{index}",
                    "artifact_ref": f"artifact:tiny-{index}",
                    "path": f"/tmp/artifacts/tiny-{index}.txt",
                    "type": "text",
                }
                for index in range(65)
            ]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "output": {"item_count": 65, "outputs": outputs},
                    "artifacts": {"output": artifacts},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "completed",
                        "outputs": outputs,
                        "artifacts": artifacts,
                    },
                },
                frames_dir=frames_dir,
            )

            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            encoded = json.dumps(
                wire_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        self.assertTrue(wire_state["ok"])
        payload = wire_state["response_payload"]
        self.assertEqual(len(payload["outputs"]), 65)
        self.assertEqual(len(payload["artifacts"]), 65)
        self.assertEqual(payload["outputs"][-1]["slot_id"], "slot-tiny-64")
        self.assertEqual(
            payload["artifacts"][-1]["artifact_ref"],
            "artifact:tiny-64",
        )
        self.assertLess(len(encoded), 8 * 1024 * 1024)

    def test_wire_projection_restores_batch_handles_and_late_fill_branch_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_wire_batch_late_fill_handles"
            prompts = ["misty forest", "neon diner"]
            outputs = [
                {
                    "artifact_ref": f"artifact:batch-{index}",
                    "batch_index": index,
                    "slot_id": f"output-batch-{index}",
                    "status": "fulfilled",
                    "type": "image",
                }
                for index in (1, 2)
            ]
            artifacts = [
                {
                    "artifact_id": f"batch-image-{index}",
                    "artifact_ref": f"artifact:batch-{index}",
                    "batch_index": index,
                    "path": f"/tmp/artifacts/batch-{index}.png",
                    "prompt": prompts[index - 1],
                    "type": "image",
                }
                for index in (1, 2)
            ]
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "batch": {"count": 2, "prompts": prompts},
                    "late_fill": {
                        "active_branch_count": 1,
                        "active_capabilities": ["vision_analysis"],
                        "active_branches": [
                            {
                                "branch_id": "branch-vision-2",
                                "capability": "vision_analysis",
                                "phase_id": "phase-vision-2",
                                "status": "running",
                            }
                        ],
                        "final_materialization_contract_reason": "one branch remains",
                        "final_materialization_contract_status": "pending",
                        "pending_branch_count": 0,
                        "status": "running",
                    },
                    "output": {"item_count": 2, "outputs": outputs},
                    "artifacts": {"output": artifacts},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "late_fill_running",
                        "outputs": outputs,
                        "artifacts": artifacts,
                    },
                },
                frames_dir=frames_dir,
            )

            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
            )
            observed = load_latest_response_observation_state(
                response_id,
                frames_dir=frames_dir,
            )

        self.assertTrue(wire_state["ok"])
        payload = wire_state["response_payload"]
        self.assertEqual(payload["batch_count"], 2)
        self.assertEqual(payload["batch_prompts"], prompts)
        self.assertEqual(payload["batch_prompts_count"], 2)
        self.assertEqual(payload["batch_prompts_projected_count"], 2)
        self.assertEqual(payload["batch_results_projected_count"], 2)
        self.assertTrue(payload["batch_projection_complete"])
        self.assertEqual([item["index"] for item in payload["results"]], [1, 2])
        self.assertEqual(payload["results"][1]["prompt"], "neon diner")
        self.assertEqual(
            payload["results"][1]["artifacts"][0]["artifact_ref"],
            "artifact:batch-2",
        )
        late_fill = payload["late_fill"]
        self.assertEqual(late_fill["active_count"], 1)
        self.assertEqual(late_fill["pending_count"], 0)
        self.assertEqual(late_fill["pending_branches"], [])
        self.assertEqual(late_fill["completed_branches"], [])
        self.assertEqual(late_fill["failed_branches"], [])
        self.assertEqual(late_fill["cancelled_branches"], [])
        self.assertEqual(late_fill["active_capabilities"], ["vision_analysis"])
        self.assertEqual(late_fill["active_branches"][0]["branch_id"], "branch-vision-2")
        self.assertEqual(
            late_fill["active_branches"][0]["capability"],
            "vision_analysis",
        )
        self.assertEqual(late_fill["final_materialization_contract_status"], "pending")
        self.assertTrue(observed["ok"])
        observed_late_fill = observed["response_payload"]["late_fill"]
        self.assertEqual(observed_late_fill["active_count"], 1)
        self.assertEqual(
            observed_late_fill["active_branches"][0]["phase_id"],
            "phase-vision-2",
        )

    def test_observation_loader_projects_outer_graph_closure_without_child_hydration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_bounded_closure_observation"
            sentinel = "CLOSURE_CHILD_SENTINEL_" + ("z" * 1_000_000)
            closure_review = {
                "kind": "ollmo.graph_closure_review",
                "status": "repair_needed",
                "reason": "one branch needs bounded repair",
                "continuation_required": True,
                "counts": {
                    "blocked": 1,
                    "fulfilled": 2,
                    "pending": 0,
                },
                "repair_action": "repair_dependency_chain",
                "checks": [
                    {
                        "action": "repair_dependency_chain",
                        "check_id": "closure-check-large-private",
                        "private_evidence": sentinel,
                        "status": "blocked",
                    }
                ],
                "surface_state": {
                    "active_categories": ["blocked", "repair_pending"],
                    "category_counts": {"blocked": 1, "repair_pending": 1},
                    "items": [{"private_evidence": sentinel}],
                    "kind": "ollmo.surface_state",
                    "status": "repair_needed",
                },
            }
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "completed",
                    "runtime": {
                        "graph_closure_review": closure_review,
                        "developer_diagnostics": {
                            "graph_closure_review": closure_review,
                        },
                    },
                    "output": {"item_count": 0, "outputs": []},
                    "artifacts": {"output": []},
                    "current_state": {
                        "id": response_id,
                        "status": "completed",
                        "lifecycle_state": "repair_needed",
                        "outputs": [],
                        "artifacts": [],
                    },
                },
                frames_dir=frames_dir,
            )
            current_index = load_response_frame_index(frames_dir=frames_dir)
            manifest = current_index["responses"][response_id][
                "effective_snapshot_manifest"
            ]
            closure_ref = manifest["runtime.graph_closure_review"]
            outer_closure = json.loads(
                (frames_dir / closure_ref["path"]).read_text(encoding="utf-8")
            )
            checks_ref = outer_closure["checks_snapshot_ref"]
            observed_paths = []
            original_read_observation = (
                response_frames_module._read_observation_snapshot_payload
            )

            def tracked_observation_read(ref, *, frames_dir):
                if isinstance(ref, dict):
                    observed_paths.append(ref.get("path"))
                return original_read_observation(ref, frames_dir=frames_dir)

            with (
                patch.object(
                    response_frames_module,
                    "_read_snapshot_ref_payload",
                    side_effect=AssertionError(
                        "bounded observation must not recursively hydrate sidecars"
                    ),
                ),
                patch.object(
                    response_frames_module,
                    "_read_observation_snapshot_payload",
                    side_effect=tracked_observation_read,
                ),
            ):
                observed = load_latest_response_observation_state(
                    response_id,
                    frames_dir=frames_dir,
                    index_state=current_index,
                )
            encoded_observation = json.dumps(
                observed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        self.assertTrue(observed["ok"])
        self.assertLess(len(encoded_observation), 64 * 1024)
        self.assertNotIn("CLOSURE_CHILD_SENTINEL_", encoded_observation.decode("utf-8"))
        projected = observed["response_payload"]["runtime"]["graph_closure_review"]
        self.assertEqual(projected["status"], "repair_needed")
        self.assertEqual(projected["counts"]["blocked"], 1)
        self.assertEqual(projected["repair_action"], "repair_dependency_chain")
        self.assertEqual(projected["checks_snapshot_ref"], checks_ref)
        self.assertEqual(projected["snapshot_ref"], closure_ref)
        self.assertEqual(
            projected["surface_state"]["category_counts"]["repair_pending"],
            1,
        )
        self.assertEqual(
            projected["observation_projection"]["child_sidecar_hydration"],
            "none",
        )
        self.assertIn(closure_ref["path"], observed_paths)
        self.assertNotIn(checks_ref["path"], observed_paths)

    def test_observation_loader_preserves_bounded_rebase_opportunity_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_bounded_rebase_opportunity_evidence"
            sentinel = "OPPORTUNITY_PRIVATE_GRAPH_" + ("q" * 1_000_000)
            persist_response_frame(
                {
                    "frame_version": 9,
                    "kind": "ollmo.response_frame",
                    "response_id": response_id,
                    "status": "repair_needed",
                    "request": {
                        "prompt": "replace only the broken vision subtree",
                        "workload_family": "vision_structural_join",
                    },
                    "runtime": {
                        "graph_closure_review": {
                            "kind": "ollmo.graph_closure_review",
                            "status": "repair_needed",
                            "recommended_transition": "partial_subtree_rebase",
                            "repair_needed": True,
                            "intent_graph_adequacy": {
                                "adequate": False,
                                "candidate_graph": {"private_graph": sentinel},
                                "checks": [
                                    {
                                        "check_id": "adequacy-dependency-edge",
                                        "check_kind": "intent_graph_adequacy",
                                        "evidence": "intent_graph_adequacy_missing_dependency_edge",
                                        "recommended_transition": "partial_subtree_rebase",
                                        "status": "failed",
                                    }
                                ],
                                "reason": "dependency_edge_missing",
                                "recommended_transition": "partial_subtree_rebase",
                                "status": "failed",
                            },
                        },
                        "request_phase_graph": {
                            "kind": "ollmo.request_phase_graph",
                            "graph_rebase_proposals": [
                                {
                                    "proposal_id": "rebase-opportunity-1",
                                    "requested_rebase_class": "partial_subtree_rebase",
                                    "status": "shadow",
                                }
                            ],
                            "graph_repair_proposals": [
                                {
                                    "proposal_id": "additive-repair-1",
                                    "status": "insufficient",
                                }
                            ],
                            "graph_repair_reviews": [
                                {
                                    "proposal_id": "additive-repair-1",
                                    "reason": "dependency_shape_requires_replacement",
                                    "status": "rejected",
                                }
                            ],
                        },
                        "developer_diagnostics": {
                            "graph_patch_autonomy": {
                                "autonomy_level": "apply_safe",
                                "source": "runtime_control",
                            },
                            "graph_rebase_autonomy": {
                                "autonomy_level": "shadow",
                                "source": "runtime_control",
                            },
                            "runtime_graph_repair_proposal_reviews": [
                                {
                                    "proposal_id": "additive-repair-1",
                                    "status": "insufficient",
                                }
                            ],
                            "runtime_graph_repair_proposals": [
                                {
                                    "proposal_id": "additive-repair-1",
                                    "status": "candidate",
                                }
                            ],
                            "surface_repair_actionability": {
                                "action": "partial_subtree_rebase",
                                "reason": "additive_repair_insufficient",
                                "status": "actionable",
                            },
                        },
                    },
                    "current_state": {
                        "id": response_id,
                        "status": "repair_needed",
                        "lifecycle_state": "repair_needed",
                    },
                },
                frames_dir=frames_dir,
            )

            observed = load_latest_response_observation_state(
                response_id,
                frames_dir=frames_dir,
            )
            encoded = json.dumps(
                observed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        self.assertTrue(observed["ok"])
        self.assertLess(len(encoded), 128 * 1024)
        self.assertNotIn("OPPORTUNITY_PRIVATE_GRAPH_", encoded.decode("utf-8"))
        runtime = observed["response_payload"]["runtime"]
        closure = runtime["graph_closure_review"]
        self.assertEqual(closure["recommended_transition"], "partial_subtree_rebase")
        self.assertTrue(closure["repair_needed"])
        adequacy = closure["intent_graph_adequacy"]
        self.assertEqual(adequacy["status"], "failed")
        self.assertEqual(adequacy["reason"], "dependency_edge_missing")
        self.assertEqual(
            adequacy["checks"][0]["evidence"],
            "intent_graph_adequacy_missing_dependency_edge",
        )
        graph = runtime["request_phase_graph"]
        self.assertEqual(graph["graph_repair_proposals"][0]["status"], "insufficient")
        self.assertEqual(graph["graph_repair_reviews"][0]["status"], "rejected")
        diagnostics = runtime["developer_diagnostics"]
        self.assertEqual(
            diagnostics["surface_repair_actionability"]["status"],
            "actionable",
        )
        self.assertEqual(
            diagnostics["graph_patch_autonomy"]["autonomy_level"],
            "apply_safe",
        )
        self.assertEqual(
            diagnostics["graph_rebase_autonomy"]["autonomy_level"],
            "shadow",
        )
        self.assertEqual(
            diagnostics["runtime_graph_repair_proposal_reviews"][0]["status"],
            "insufficient",
        )

    def test_load_latest_response_state_uses_index_and_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_index_recovery"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Indexed recovery.",
                    },
                    request_payload={"prompt": "recover me"},
                ),
                frames_dir=frames_dir,
            )

            indexed_state = load_latest_response_state(response_id, frames_dir=frames_dir)
            (frames_dir / "current_index.json").unlink()
            missing_index_state = load_latest_response_state(response_id, frames_dir=frames_dir)
            (frames_dir / "current_index.json").write_text("{not json}\n", encoding="utf-8")
            corrupt_index_state = load_latest_response_state(response_id, frames_dir=frames_dir)

        self.assertTrue(indexed_state["ok"])
        self.assertTrue(indexed_state["index_used"])
        self.assertFalse(indexed_state["index_stale"])
        self.assertFalse(indexed_state["ledger_fallback_used"])
        self.assertTrue(missing_index_state["ok"])
        self.assertFalse(missing_index_state["index_used"])
        self.assertFalse(missing_index_state["index_stale"])
        self.assertTrue(missing_index_state["ledger_fallback_used"])
        self.assertEqual(missing_index_state["response_payload"]["output_text"], "Indexed recovery.")
        self.assertTrue(corrupt_index_state["ok"])
        self.assertFalse(corrupt_index_state["index_used"])
        self.assertFalse(corrupt_index_state["index_stale"])
        self.assertTrue(corrupt_index_state["ledger_fallback_used"])
        self.assertEqual(corrupt_index_state["index_error"]["code"], "response_frame_index_corrupt")

    def test_load_latest_response_state_uses_global_index_freshness_for_older_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            older_response_id = "resp_index_older_shared_ledger"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": older_response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Older indexed response.",
                    },
                    request_payload={"prompt": "older response"},
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": "resp_index_newer_shared_ledger",
                        "object": "response",
                        "status": "completed",
                        "output_text": "Newer indexed response.",
                    },
                    request_payload={"prompt": "newer response"},
                ),
                frames_dir=frames_dir,
            )
            current_index = load_response_frame_index(frames_dir=frames_dir)
            older_entry = current_index["responses"][older_response_id]
            self.assertLess(older_entry["ledger_size_bytes"], current_index["ledger_size_bytes"])

            with patch(
                "ollmo_services.response_frames._iter_ledger_frames",
                side_effect=AssertionError("fresh indexed lookup must not scan the shared ledger"),
            ):
                state = load_latest_response_state(older_response_id, frames_dir=frames_dir)

        self.assertTrue(state["ok"])
        self.assertTrue(state["index_used"])
        self.assertFalse(state["ledger_fallback_used"])
        self.assertEqual(state["response_payload"]["output_text"], "Older indexed response.")

    def test_load_latest_response_state_proves_absence_from_verified_response_map_without_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            for suffix in ("a", "b"):
                persist_response_frame(
                    build_response_frame(
                        {
                            "id": f"resp_verified_map_{suffix}",
                            "object": "response",
                            "status": "completed",
                            "output_text": f"Indexed response {suffix}.",
                        },
                        request_payload={"prompt": f"response {suffix}"},
                    ),
                    frames_dir=frames_dir,
                )
            current_index = load_response_frame_index(frames_dir=frames_dir)
            self.assertEqual(
                current_index["response_map_verified_size_bytes"],
                current_index["ledger_size_bytes"],
            )

            with patch(
                "ollmo_services.response_frames._iter_ledger_frames",
                side_effect=AssertionError("verified negative lookup must not scan the ledger"),
            ):
                state = load_latest_response_state(
                    "resp_absent_from_verified_map",
                    frames_dir=frames_dir,
                )

        self.assertFalse(state["ok"])
        self.assertEqual(state["status_code"], 404)
        self.assertTrue(state["error"]["index_used"])
        self.assertFalse(state["error"]["ledger_fallback_used"])

    def test_load_latest_response_state_scans_when_response_map_coverage_is_unverified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            hidden_response_id = "resp_unverified_map_hidden"
            for response_id in ("resp_unverified_map_visible", hidden_response_id):
                persist_response_frame(
                    build_response_frame(
                        {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "output_text": f"Payload for {response_id}.",
                        },
                        request_payload={"prompt": response_id},
                    ),
                    frames_dir=frames_dir,
                )
            index_path = frames_dir / "current_index.json"
            incomplete_index = json.loads(index_path.read_text(encoding="utf-8"))
            incomplete_index["responses"].pop(hidden_response_id)
            incomplete_index.pop("response_map_verified_size_bytes", None)
            incomplete_index.pop("response_map_entry_count", None)
            incomplete_index.pop("response_map_digest", None)
            index_path.write_text(json.dumps(incomplete_index), encoding="utf-8")

            with patch(
                "ollmo_services.response_frames._iter_ledger_frames",
                wraps=response_frames_module._iter_ledger_frames,
            ) as iter_ledger:
                state = load_latest_response_state(hidden_response_id, frames_dir=frames_dir)

        self.assertTrue(state["ok"])
        self.assertTrue(state["ledger_fallback_used"])
        self.assertGreaterEqual(iter_ledger.call_count, 1)
        self.assertEqual(state["response_payload"]["id"], hidden_response_id)

    def test_eof_fresh_unverified_index_cannot_hide_newer_response_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_unverified_stale_coordinate"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "first durable value",
                    },
                    request_payload={"prompt": "first"},
                ),
                frames_dir=frames_dir,
            )
            with patch.object(
                response_frames_module,
                "_write_response_frame_index",
                side_effect=OSError("simulated index write failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated index write failure"):
                    persist_response_frame(
                        build_response_frame(
                            {
                                "id": response_id,
                                "object": "response",
                                "status": "completed",
                                "output_text": "second durable value",
                            },
                            request_payload={"prompt": "second"},
                        ),
                        frames_dir=frames_dir,
                    )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": "resp_unrelated_index_advance",
                        "object": "response",
                        "status": "completed",
                        "output_text": "unrelated",
                    },
                    request_payload={"prompt": "unrelated"},
                ),
                frames_dir=frames_dir,
            )
            current_index = load_response_frame_index(frames_dir=frames_dir)
            ledger_path = frames_dir / "responses.jsonl"
            self.assertEqual(
                current_index["ledger_size_bytes"],
                ledger_path.stat().st_size,
            )
            self.assertIsNone(current_index["response_map_verified_size_bytes"])

            canonical_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            wire_state = load_latest_response_wire_state(
                response_id,
                frames_dir=frames_dir,
                index_state=current_index,
            )
            observation_state = load_latest_response_observation_state(
                response_id,
                frames_dir=frames_dir,
                index_state=current_index,
            )

        self.assertTrue(canonical_state["ok"])
        self.assertEqual(
            canonical_state["response_payload"]["output_text"],
            "second durable value",
        )
        self.assertEqual(canonical_state["response_frame"]["frame_sequence"], 2)
        self.assertTrue(canonical_state["ledger_fallback_used"])
        self.assertTrue(canonical_state["index_stale"])
        self.assertFalse(wire_state["ok"])
        self.assertEqual(wire_state["status_code"], 409)
        self.assertEqual(
            wire_state["error"]["code"],
            "response_frame_index_unverified",
        )
        self.assertFalse(observation_state["ok"])
        self.assertEqual(observation_state["status_code"], 409)
        self.assertEqual(
            observation_state["error"]["code"],
            "response_frame_index_unverified",
        )

    def test_attest_response_frame_index_streams_and_preserves_legacy_entries_exactly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            for response_id in ("resp_attest_a", "resp_attest_b"):
                persist_response_frame(
                    build_response_frame(
                        {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "output_text": f"Payload for {response_id}.",
                        },
                        request_payload={"prompt": response_id},
                    ),
                    frames_dir=frames_dir,
                )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": "resp_attest_a",
                        "object": "response",
                        "status": "completed",
                        "output_text": "Latest successor for response A.",
                    },
                    request_payload={"prompt": "resp_attest_a successor"},
                ),
                frames_dir=frames_dir,
            )
            index_path = frames_dir / "current_index.json"
            legacy_index = json.loads(index_path.read_bytes())
            legacy_index["version"] = 1
            legacy_index.pop("response_map_verified_size_bytes", None)
            legacy_index.pop("response_map_entry_count", None)
            legacy_index.pop("response_map_digest", None)
            legacy_index["responses"]["resp_attest_a"]["effective_snapshot_manifest"] = {
                "runtime.synthetic": {
                    "kind": "ollmo.response_frame_snapshot_ref",
                    "sha256": "a" * 64,
                }
            }
            index_path.write_bytes(
                (json.dumps(legacy_index, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
            )
            original_responses = json.loads(json.dumps(legacy_index["responses"]))

            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("attestation must not use Path.read_text"),
            ), patch(
                "ollmo_services.response_frames._iter_ledger_frames",
                side_effect=AssertionError("attestation must not materialize the ledger"),
            ), patch(
                "ollmo_services.response_frames.os.replace",
                wraps=os.replace,
            ) as atomic_replace:
                result = response_frames_module.attest_response_frame_index(
                    frames_dir=frames_dir,
                )

            attested = json.loads(index_path.read_bytes())

        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        self.assertEqual(attested["version"], 2)
        self.assertEqual(attested["responses"], original_responses)
        self.assertEqual(
            attested["response_map_verified_size_bytes"],
            attested["ledger_size_bytes"],
        )
        self.assertEqual(attested["response_map_entry_count"], 2)
        self.assertEqual(
            attested["response_map_digest"],
            response_frames_module._response_map_digest(attested["responses"]),
        )
        atomic_replace.assert_called_once()

    def test_attest_response_frame_index_rejects_incomplete_legacy_map_without_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            for response_id in ("resp_attest_visible", "resp_attest_hidden"):
                persist_response_frame(
                    build_response_frame(
                        {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "output_text": response_id,
                        },
                        request_payload={"prompt": response_id},
                    ),
                    frames_dir=frames_dir,
                )
            index_path = frames_dir / "current_index.json"
            incomplete = json.loads(index_path.read_bytes())
            incomplete["version"] = 1
            incomplete["responses"].pop("resp_attest_hidden")
            for key in (
                "response_map_verified_size_bytes",
                "response_map_entry_count",
                "response_map_digest",
            ):
                incomplete.pop(key, None)
            index_path.write_bytes(json.dumps(incomplete, sort_keys=True).encode("utf-8"))
            before = index_path.read_bytes()

            result = response_frames_module.attest_response_frame_index(
                frames_dir=frames_dir,
            )

            after = index_path.read_bytes()

        self.assertFalse(result["ok"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["error"]["code"], "response_frame_index_response_set_mismatch")
        self.assertEqual(after, before)

    def test_attest_response_frame_index_rejects_latest_coordinate_mismatch_without_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_attest_coordinate_mismatch"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Coordinate evidence.",
                    },
                    request_payload={"prompt": response_id},
                ),
                frames_dir=frames_dir,
            )
            index_path = frames_dir / "current_index.json"
            mismatched = json.loads(index_path.read_bytes())
            mismatched["version"] = 1
            mismatched["responses"][response_id]["line_length"] += 1
            for key in (
                "response_map_verified_size_bytes",
                "response_map_entry_count",
                "response_map_digest",
            ):
                mismatched.pop(key, None)
            index_path.write_bytes(json.dumps(mismatched, sort_keys=True).encode("utf-8"))
            before = index_path.read_bytes()

            result = response_frames_module.attest_response_frame_index(
                frames_dir=frames_dir,
            )
            after = index_path.read_bytes()

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "response_frame_index_latest_entry_mismatch")
        self.assertEqual(after, before)

    def test_attest_response_frame_index_rejects_malformed_ledger_without_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_attest_malformed_ledger"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Before malformed evidence.",
                    },
                    request_payload={"prompt": response_id},
                ),
                frames_dir=frames_dir,
            )
            ledger_path = frames_dir / "responses.jsonl"
            with ledger_path.open("ab") as handle:
                handle.write(b"{malformed json}\n")
            index_path = frames_dir / "current_index.json"
            malformed_epoch = json.loads(index_path.read_bytes())
            malformed_epoch["version"] = 1
            malformed_epoch["ledger_size_bytes"] = ledger_path.stat().st_size
            malformed_epoch["ledger_line_count"] += 1
            malformed_epoch["ledger_line_count_verified_size_bytes"] = ledger_path.stat().st_size
            for key in (
                "response_map_verified_size_bytes",
                "response_map_entry_count",
                "response_map_digest",
            ):
                malformed_epoch.pop(key, None)
            index_path.write_bytes(json.dumps(malformed_epoch, sort_keys=True).encode("utf-8"))
            before = index_path.read_bytes()

            result = response_frames_module.attest_response_frame_index(
                frames_dir=frames_dir,
            )
            after = index_path.read_bytes()

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "response_frame_ledger_malformed")
        self.assertEqual(after, before)

    def test_attest_response_frame_index_rejects_moving_ledger_and_index_evidence(self):
        for moving_target in ("ledger", "index"):
            with self.subTest(moving_target=moving_target), tempfile.TemporaryDirectory() as tmpdir:
                frames_dir = Path(tmpdir)
                response_id = f"resp_attest_moving_{moving_target}"
                persist_response_frame(
                    build_response_frame(
                        {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "output_text": moving_target,
                        },
                        request_payload={"prompt": response_id},
                    ),
                    frames_dir=frames_dir,
                )
                index_path = frames_dir / "current_index.json"
                legacy = json.loads(index_path.read_bytes())
                legacy["version"] = 1
                for key in (
                    "response_map_verified_size_bytes",
                    "response_map_entry_count",
                    "response_map_digest",
                ):
                    legacy.pop(key, None)
                index_path.write_bytes(json.dumps(legacy, sort_keys=True).encode("utf-8"))
                original_scan = response_frames_module._scan_response_frame_ledger_index_truth

                def scan_then_move(*args, **kwargs):
                    scan_result = original_scan(*args, **kwargs)
                    if moving_target == "ledger":
                        with (frames_dir / "responses.jsonl").open("ab") as handle:
                            handle.write(b"\n")
                    else:
                        index_path.write_bytes(index_path.read_bytes() + b"\n")
                    return scan_result

                with patch(
                    "ollmo_services.response_frames._scan_response_frame_ledger_index_truth",
                    side_effect=scan_then_move,
                ):
                    result = response_frames_module.attest_response_frame_index(
                        frames_dir=frames_dir,
                    )

                self.assertFalse(result["ok"])
                self.assertFalse(result["changed"])
                self.assertEqual(result["error"]["code"], f"response_frame_{moving_target}_moved")
                self.assertEqual(json.loads(index_path.read_bytes())["version"], 1)

    def test_response_frame_index_attestation_script_updates_only_exact_temp_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_attest_script"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Script fixture.",
                    },
                    request_payload={"prompt": response_id},
                ),
                frames_dir=frames_dir,
            )
            index_path = frames_dir / "current_index.json"
            legacy = json.loads(index_path.read_bytes())
            legacy["version"] = 1
            for key in (
                "response_map_verified_size_bytes",
                "response_map_entry_count",
                "response_map_digest",
            ):
                legacy.pop(key, None)
            index_path.write_bytes(json.dumps(legacy, sort_keys=True).encode("utf-8"))

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/attest_response_frame_index.py",
                    "--frames-dir",
                    str(frames_dir),
                ],
                cwd=Path(__file__).resolve().parent.parent,
                check=False,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            attested = json.loads(index_path.read_bytes())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(attested["version"], 2)
        self.assertEqual(attested["response_map_entry_count"], 1)

    def test_attest_response_frame_index_supports_temp_index_with_symlinked_source_ledger(self):
        with tempfile.TemporaryDirectory() as source_tmpdir, tempfile.TemporaryDirectory() as index_tmpdir:
            source_frames_dir = Path(source_tmpdir)
            temp_index_dir = Path(index_tmpdir)
            response_id = "resp_attest_symlinked_ledger"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Symlinked ledger evidence.",
                    },
                    request_payload={"prompt": response_id},
                ),
                frames_dir=source_frames_dir,
            )
            source_index_path = source_frames_dir / "current_index.json"
            source_ledger_path = source_frames_dir / "responses.jsonl"
            source_index_before = source_index_path.read_bytes()
            source_ledger_before = source_ledger_path.read_bytes()
            legacy = json.loads(source_index_before)
            legacy["version"] = 1
            for key in (
                "response_map_verified_size_bytes",
                "response_map_entry_count",
                "response_map_digest",
            ):
                legacy.pop(key, None)
            (temp_index_dir / "current_index.json").write_bytes(
                json.dumps(legacy, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            (temp_index_dir / "responses.jsonl").symlink_to(source_ledger_path)

            result = response_frames_module.attest_response_frame_index(
                frames_dir=temp_index_dir,
            )

            temp_index = json.loads((temp_index_dir / "current_index.json").read_bytes())
            source_index_after = source_index_path.read_bytes()
            source_ledger_after = source_ledger_path.read_bytes()

        self.assertTrue(result["ok"])
        self.assertEqual(temp_index["version"], 2)
        self.assertEqual(temp_index["responses"], legacy["responses"])
        self.assertEqual(source_index_after, source_index_before)
        self.assertEqual(source_ledger_after, source_ledger_before)

    def test_load_latest_response_state_rejects_stale_but_valid_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_stale_index_recovery"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial stale-index text.",
                    },
                    request_payload={"prompt": "answer then image"},
                ),
                frames_dir=frames_dir,
            )
            stale_index = (frames_dir / "current_index.json").read_text(encoding="utf-8")
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial stale-index text.",
                        "late_fill": {"status": "completed"},
                        "artifacts": [{"type": "image", "path": "/tmp/generated/stale-index.png"}],
                    },
                    request_payload={"prompt": "answer then image"},
                ),
                frames_dir=frames_dir,
            )
            (frames_dir / "current_index.json").write_text(stale_index, encoding="utf-8")

            state = load_latest_response_state(response_id, frames_dir=frames_dir)

        self.assertTrue(state["ok"])
        self.assertFalse(state["index_used"])
        self.assertTrue(state["index_stale"])
        self.assertTrue(state["ledger_fallback_used"])
        self.assertEqual(state["response_frame"]["frame_sequence"], 2)
        self.assertEqual(state["response_payload"]["artifacts"][0]["path"], "/tmp/generated/stale-index.png")
        self.assertEqual(state["response_frame"]["frame_relation"]["kind"], "late_fill_successor")

    def test_persist_response_frame_rejects_stale_index_parent_before_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_stale_index_append"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                    },
                    request_payload={"prompt": "answer then image then audio"},
                ),
                frames_dir=frames_dir,
            )
            stale_index = (frames_dir / "current_index.json").read_text(encoding="utf-8")
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                        "late_fill": {"status": "running"},
                        "artifacts": [{"type": "image", "path": "/tmp/generated/stale-append.png"}],
                    },
                    request_payload={"prompt": "answer then image then audio"},
                ),
                frames_dir=frames_dir,
            )
            (frames_dir / "current_index.json").write_text(stale_index, encoding="utf-8")
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                        "late_fill": {"status": "completed"},
                        "artifacts": [
                            {"type": "image", "path": "/tmp/generated/stale-append.png"},
                            {"type": "audio", "path": "/tmp/generated/stale-append.wav"},
                        ],
                    },
                    request_payload={"prompt": "answer then image then audio"},
                ),
                frames_dir=frames_dir,
            )
            ledger_lines = [
                json.loads(line)
                for line in (frames_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            latest_state = load_latest_response_state(response_id, frames_dir=frames_dir)

        self.assertEqual([frame["frame_sequence"] for frame in ledger_lines], [1, 2, 3])
        self.assertEqual(ledger_lines[2]["frame_relation"]["parent_frame_id"], ledger_lines[1]["frame_id"])
        self.assertEqual(latest_state["response_frame"]["frame_sequence"], 3)

    def test_load_latest_response_state_falls_back_when_index_points_to_missing_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_missing_index_frame"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                    },
                    request_payload={"prompt": "answer then image"},
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial text.",
                        "late_fill": {"status": "completed"},
                        "artifacts": [{"type": "image", "path": "/tmp/generated/missing-index.png"}],
                    },
                    request_payload={"prompt": "answer then image"},
                ),
                frames_dir=frames_dir,
            )
            index_payload = json.loads((frames_dir / "current_index.json").read_text(encoding="utf-8"))
            index_payload["responses"][response_id]["latest_frame_id"] = "missing-frame-id"
            index_payload["responses"][response_id]["latest_frame_sequence"] = 99
            (frames_dir / "current_index.json").write_text(json.dumps(index_payload), encoding="utf-8")

            state = load_latest_response_state(response_id, frames_dir=frames_dir)

        self.assertTrue(state["ok"])
        self.assertFalse(state["index_used"])
        self.assertTrue(state["index_stale"])
        self.assertTrue(state["ledger_fallback_used"])
        self.assertEqual(state["response_frame"]["frame_sequence"], 2)
        self.assertEqual(state["response_payload"]["artifacts"][0]["path"], "/tmp/generated/missing-index.png")

    def test_load_latest_response_state_tolerates_mixed_corrupt_ledger_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = "resp_mixed_ledger_target"
            unrelated_id = "resp_mixed_ledger_other"
            persist_response_frame(
                build_response_frame(
                    {
                        "id": unrelated_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Other response.",
                    },
                    request_payload={"prompt": "other"},
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial target.",
                    },
                    request_payload={"prompt": "target"},
                ),
                frames_dir=frames_dir,
            )
            persist_response_frame(
                build_response_frame(
                    {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "output_text": "Initial target.",
                        "late_fill": {"status": "completed"},
                        "artifacts": [{"type": "image", "path": "/tmp/generated/target.png"}],
                    },
                    request_payload={"prompt": "target"},
                ),
                frames_dir=frames_dir,
            )
            ledger_path = frames_dir / "responses.jsonl"
            valid_lines = ledger_path.read_text(encoding="utf-8").splitlines()
            ledger_path.write_text(
                "\n".join(
                    [
                        "{bad before",
                        valid_lines[0],
                        valid_lines[1],
                        valid_lines[2],
                        "{bad after",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (frames_dir / "current_index.json").unlink()

            recovered = load_latest_response_state(response_id, frames_dir=frames_dir)
            missing = load_latest_response_state("resp_absent_in_mixed_ledger", frames_dir=frames_dir)

        self.assertTrue(recovered["ok"])
        self.assertFalse(recovered["index_used"])
        self.assertFalse(recovered["index_stale"])
        self.assertTrue(recovered["ledger_fallback_used"])
        self.assertEqual(recovered["response_payload"]["id"], response_id)
        self.assertEqual(recovered["response_payload"]["artifacts"][0]["path"], "/tmp/generated/target.png")
        self.assertEqual(recovered["response_frame"]["frame_relation"]["kind"], "late_fill_successor")
        self.assertIn("parent_frame_id", recovered["response_frame"]["frame_relation"])
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["status_code"], 409)
        self.assertEqual(missing["error"]["code"], "response_frame_ledger_corrupt")

    def test_load_latest_response_state_reports_corrupt_missing_frame_truthfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "responses.jsonl").write_text("{not json}\n", encoding="utf-8")

            state = load_latest_response_state("resp_corrupt_only", frames_dir=frames_dir)

        self.assertFalse(state["ok"])
        self.assertEqual(state["status_code"], 409)
        self.assertEqual(state["error"]["code"], "response_frame_ledger_corrupt")


class ResponseFrameRecoveryCacheInspectionTests(unittest.TestCase):
    @staticmethod
    def _ledger_stat(path):
        stat = path.stat()
        return {
            'size_bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'device': stat.st_dev,
            'inode': stat.st_ino,
        }

    def test_recovery_cache_inspection_is_read_only_and_advances_only_unrelated_tail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            frames_dir.mkdir()
            ledger_path = frames_dir / 'responses.jsonl'
            ledger_path.write_text(
                json.dumps({'response_id': 'other-initial'}) + '\n',
                encoding='utf-8',
            )
            expected = self._ledger_stat(ledger_path)

            unchanged = inspect_response_frame_recovery_cache(
                'resp_target',
                frames_dir=frames_dir,
                expected_ledger_path=ledger_path,
                expected_ledger_state=expected,
            )

            self.assertTrue(unchanged['cache_reusable'])
            self.assertEqual(unchanged['reason'], 'unchanged')
            self.assertNotIn('checkpoint_ledger_state', unchanged)

            with ledger_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'response_id': 'other-appended'}) + '\n')
            bytes_before = ledger_path.read_bytes()
            state_before = self._ledger_stat(ledger_path)

            appended = inspect_response_frame_recovery_cache(
                'resp_target',
                frames_dir=frames_dir,
                expected_ledger_path=ledger_path,
                expected_ledger_state=expected,
            )

            self.assertTrue(appended['cache_reusable'])
            self.assertEqual(appended['reason'], 'unrelated_append_only_tail')
            self.assertEqual(appended['checkpoint_ledger_state'], state_before)
            self.assertEqual(ledger_path.read_bytes(), bytes_before)
            self.assertEqual(self._ledger_stat(ledger_path), state_before)

    def test_recovery_cache_inspection_rejects_target_and_malformed_appended_tail(self):
        tail_cases = (
            (
                json.dumps({'response_id': 'resp_target'}) + '\n',
                'target_response_appended',
            ),
            ('{not-json}\n', 'malformed_appended_ledger_line'),
            ('   \n', 'invalid_appended_ledger_line'),
            ('[]\n', 'non_mapping_appended_ledger_line'),
            (json.dumps({'response_id': 'other'}), 'invalid_appended_ledger_line'),
        )
        for appended_tail, expected_reason in tail_cases:
            with self.subTest(reason=expected_reason):
                with tempfile.TemporaryDirectory() as tmpdir:
                    frames_dir = Path(tmpdir) / 'response_frames'
                    frames_dir.mkdir()
                    ledger_path = frames_dir / 'responses.jsonl'
                    ledger_path.write_text(
                        json.dumps({'response_id': 'other-initial'}) + '\n',
                        encoding='utf-8',
                    )
                    expected = self._ledger_stat(ledger_path)
                    with ledger_path.open('a', encoding='utf-8') as handle:
                        handle.write(appended_tail)

                    inspected = inspect_response_frame_recovery_cache(
                        'resp_target',
                        frames_dir=frames_dir,
                        expected_ledger_path=ledger_path,
                        expected_ledger_state=expected,
                    )

                self.assertFalse(inspected['cache_reusable'])
                self.assertEqual(inspected['reason'], expected_reason)
                self.assertNotIn('checkpoint_ledger_state', inspected)

    def test_recovery_cache_inspection_uses_explicit_frame_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected_frames_dir = root / 'expected_frames'
            active_frames_dir = root / 'active_frames'
            expected_frames_dir.mkdir()
            active_frames_dir.mkdir()
            expected_ledger = expected_frames_dir / 'responses.jsonl'
            expected_ledger.write_text(
                json.dumps({'response_id': 'other'}) + '\n',
                encoding='utf-8',
            )
            (active_frames_dir / 'responses.jsonl').write_text(
                json.dumps({'response_id': 'resp_target'}) + '\n',
                encoding='utf-8',
            )

            inspected = inspect_response_frame_recovery_cache(
                'resp_target',
                frames_dir=active_frames_dir,
                expected_ledger_path=expected_ledger,
                expected_ledger_state=self._ledger_stat(expected_ledger),
            )

        self.assertFalse(inspected['cache_reusable'])
        self.assertEqual(inspected['reason'], 'ledger_path_mismatch')
        self.assertEqual(
            inspected['ledger_path'],
            str(active_frames_dir / 'responses.jsonl'),
        )

    def test_recovery_cache_inspection_rejects_truncation_and_identity_replacement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            truncation_dir = root / 'truncation'
            truncation_dir.mkdir()
            truncated_ledger = truncation_dir / 'responses.jsonl'
            truncated_ledger.write_text(
                json.dumps({'response_id': 'other', 'padding': 'long-value'})
                + '\n',
                encoding='utf-8',
            )
            truncation_expected = self._ledger_stat(truncated_ledger)
            truncated_ledger.write_text('{}\n', encoding='utf-8')

            truncated = inspect_response_frame_recovery_cache(
                'resp_target',
                frames_dir=truncation_dir,
                expected_ledger_path=truncated_ledger,
                expected_ledger_state=truncation_expected,
            )

            replacement_dir = root / 'replacement'
            replacement_dir.mkdir()
            replaced_ledger = replacement_dir / 'responses.jsonl'
            replaced_ledger.write_text(
                json.dumps({'response_id': 'other'}) + '\n',
                encoding='utf-8',
            )
            replacement_expected = self._ledger_stat(replaced_ledger)
            replacement = replacement_dir / 'replacement.jsonl'
            replacement.write_text(
                json.dumps({'response_id': 'other-new'}) + '\n',
                encoding='utf-8',
            )
            replacement.replace(replaced_ledger)

            replaced = inspect_response_frame_recovery_cache(
                'resp_target',
                frames_dir=replacement_dir,
                expected_ledger_path=replaced_ledger,
                expected_ledger_state=replacement_expected,
            )

        self.assertFalse(truncated['cache_reusable'])
        self.assertEqual(truncated['reason'], 'ledger_truncated')
        self.assertFalse(replaced['cache_reusable'])
        self.assertEqual(replaced['reason'], 'ledger_identity_changed')


if __name__ == "__main__":
    unittest.main()
