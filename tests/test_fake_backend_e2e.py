from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import ollmo_webserver
from ollmo_services.graph_repair import (
    build_graph_repair_proposal_from_repair_gap,
    validate_graph_repair_proposal,
)

from tests.fake_backends import FakeBackendHarness
from tests.fake_backends.fixtures import TEXT_ARTIFACT_CONTENT, TRANSCRIPT_TEXT, VISION_RESULT


def _artifact_by_type(payload: dict, artifact_type: str) -> list[dict]:
    return [
        item
        for item in payload.get("artifacts", [])
        if isinstance(item, dict) and item.get("type") == artifact_type
    ]


def _registry_by_path(harness: FakeBackendHarness, path: str) -> dict:
    for record in harness.registry_records():
        artifact = record.get("artifact") if isinstance(record.get("artifact"), dict) else {}
        if artifact.get("path") == path:
            return record
    raise AssertionError(f"missing registry record for {path}")


def _fulfilled_outputs(payload: dict, output_type: str | None = None) -> list[dict]:
    outputs = [
        item
        for item in payload.get("outputs", [])
        if isinstance(item, dict) and item.get("status") == "fulfilled"
    ]
    if output_type:
        outputs = [item for item in outputs if item.get("type") == output_type]
    return outputs


def _frame_slots(payload: dict) -> list[dict]:
    frame = payload.get("response_frame") if isinstance(payload.get("response_frame"), dict) else {}
    planning = frame.get("planning") if isinstance(frame.get("planning"), dict) else {}
    artifact_flow = planning.get("artifact_flow") if isinstance(planning.get("artifact_flow"), dict) else {}
    return artifact_flow.get("output_slots") if isinstance(artifact_flow.get("output_slots"), list) else []


def _runtime_graph(payload: dict) -> dict:
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    graph = runtime.get("request_phase_graph") if isinstance(runtime.get("request_phase_graph"), dict) else {}
    return graph


def _closure_review(payload: dict) -> dict:
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    review = runtime.get("graph_closure_review") if isinstance(runtime.get("graph_closure_review"), dict) else {}
    return review


def _intent_obligations(payload: dict, *, kind: str | None = None) -> list[dict]:
    obligations = [
        item for item in (_runtime_graph(payload).get("intent_obligations") or [])
        if isinstance(item, dict)
    ]
    if kind is None:
        return obligations
    return [item for item in obligations if item.get("kind") == kind]


def _truth_response(harness: FakeBackendHarness, response_id: str) -> dict:
    payload, status = harness.get_response(response_id, view="truth")
    assert status == 200
    assert payload["id"] == response_id
    return payload


def test_text_artifact_truth_uses_saved_file_registry_and_frame_not_prose():
    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_text_truth",
                "capability": "chat",
                "fake_scenario": "text_artifact",
                "prompt": "Create report.md as a saved text artifact.",
                "messages": [{"role": "user", "content": "Create report.md as a saved text artifact."}],
            }
        )

        assert status == 200
        assert harness.calls["chat"] == 1
        assert payload["lifecycle_state"] == "completed"
        text_artifacts = _artifact_by_type(payload, "text")
        assert len(text_artifacts) == 1
        artifact = text_artifacts[0]
        artifact_path = Path(artifact["path"])
        assert artifact_path.exists()
        assert artifact_path.read_text(encoding="utf-8") == TEXT_ARTIFACT_CONTENT
        assert artifact_path.read_text(encoding="utf-8") != payload["output_text"]

        registry_record = _registry_by_path(harness, str(artifact_path))
        assert registry_record["artifact"]["path"] == str(artifact_path)
        assert artifact["artifact_ref"] in registry_record["artifact_alias_refs"]
        assert "resp_fake_text_truth" in registry_record["linked_response_ids"]
        assert any(
            output.get("status") == "fulfilled" and output.get("artifact_ref") == artifact["artifact_ref"]
            for output in payload.get("outputs", [])
        )
        assert any(slot.get("status") == "fulfilled" for slot in payload.get("output_slots", []))

        truth_payload = _truth_response(harness, "resp_fake_text_truth")
        assert truth_payload["artifacts"][0]["path"] == str(artifact_path)
        assert any(slot.get("status") == "fulfilled" for slot in _frame_slots(truth_payload))

        state = harness.response_state("resp_fake_text_truth")
        assert state["ok"] is True
        frame_payload = state["response_payload"]
        assert frame_payload["artifacts"][0]["path"] == str(artifact_path)


def test_image_artifact_truth_records_saved_png_registry_outputs_and_frame():
    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_image_truth",
                "capability": "image_generation",
                "prompt": "Generate one deterministic fixture image.",
            }
        )

        assert status == 200
        assert payload["lifecycle_state"] == "completed"
        image_artifacts = _artifact_by_type(payload, "image")
        assert len(image_artifacts) == 1
        image_artifact = image_artifacts[0]
        image_path = Path(image_artifact["path"])
        assert image_path.suffix == ".png"
        assert image_path.read_bytes().startswith(b"\x89PNG")
        assert image_artifact["artifact_ref"] == "artifact:fake-generated-png"
        assert payload["saved_image_path"] == str(image_path)
        assert payload["image_state"]["summary"] == "deterministic fake image"

        registry_record = _registry_by_path(harness, str(image_path))
        assert registry_record["type"] == "image"
        assert image_artifact["artifact_ref"] in registry_record["artifact_alias_refs"]
        assert registry_record["artifact"]["path"] == str(image_path)
        assert _fulfilled_outputs(payload, "image")
        assert any(
            slot.get("artifact_ref") == image_artifact["artifact_ref"]
            for slot in payload.get("output_slots", [])
        )

        truth_payload = _truth_response(harness, "resp_fake_image_truth")
        assert truth_payload["artifacts"][0]["path"] == str(image_path)
        assert any(
            slot.get("artifact_ref") == image_artifact["artifact_ref"]
            for slot in _frame_slots(truth_payload)
        )


def test_tts_stt_and_vision_truth_stays_in_fake_temp_roots_without_real_ports():
    with FakeBackendHarness() as harness:
        tts_payload, tts_status = harness.post_response(
            {
                "response_id": "resp_fake_tts_truth",
                "capability": "text_to_speech",
                "prompt": "Read this deterministic sentence aloud.",
                "voice": "fake-voice",
            }
        )
        assert tts_status == 200
        audio_artifact = _artifact_by_type(tts_payload, "audio")[0]
        audio_path = Path(audio_artifact["path"])
        assert audio_path.exists()
        assert audio_path.read_bytes().startswith(b"RIFF")
        assert audio_artifact["mime_type"] == "audio/wav"
        integrity = tts_payload["tts_audio_integrity_evidence"]
        assert integrity["status"] == "passed"
        assert integrity["materialization_eligible"] is True
        assert integrity["artifact_path"] == str(audio_path)
        assert integrity["effective_active_seconds"] >= 1.4
        assert _fulfilled_outputs(tts_payload, "audio")
        tts_truth = _truth_response(harness, "resp_fake_tts_truth")
        assert tts_truth["tts_audio_integrity_evidence"]["status"] == "passed"
        assert _closure_review(tts_truth)["status"] == "fulfilled"

        stt_payload, stt_status = harness.post_response(
            {
                "response_id": "resp_fake_stt_truth",
                "capability": "speech_to_text",
                "prompt": "Transcribe the provided fake audio.",
                "file_path": str(audio_path),
            }
        )
        assert stt_status == 200
        assert stt_payload["output_text"] == TRANSCRIPT_TEXT
        assert stt_payload["input_artifacts"][0]["path"] == str(audio_path)
        assert _fulfilled_outputs(stt_payload, "text")
        stt_truth = _truth_response(harness, "resp_fake_stt_truth")
        assert stt_truth["result"]["transcript"] == TRANSCRIPT_TEXT

        image_payload, image_status = harness.post_response(
            {
                "response_id": "resp_fake_vision_source_image",
                "capability": "image_generation",
                "prompt": "Generate a fake image for vision.",
            }
        )
        assert image_status == 200
        image_path = _artifact_by_type(image_payload, "image")[0]["path"]
        vision_payload, vision_status = harness.post_response(
            {
                "response_id": "resp_fake_vision_truth",
                "capability": "vision_analysis",
                "prompt": "Analyze this fake image.",
                "file_path": image_path,
            }
        )
        assert vision_status == 200
        assert vision_payload["output_text"] == VISION_RESULT["description"]
        assert vision_payload["input_artifacts"][0]["path"] == image_path
        vision_truth = _truth_response(harness, "resp_fake_vision_truth")
        assert vision_truth["result"]["labels"] == VISION_RESULT["labels"]
        assert all(record.get("target_port") in (None, 0) for record in harness.call_records)
        assert all(str(Path(record["payload"].get("file_path", harness.root))).startswith(str(harness.root)) for record in harness.call_records if "payload" in record and record["payload"].get("file_path"))


def test_silent_tts_artifact_is_preserved_but_does_not_fulfill_audio():
    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_silent_tts",
                "capability": "text_to_speech",
                "fake_scenario": "silent_tts",
                "prompt": "Fake silent TTS output for a complete spoken sentence.",
                "voice": "fake-voice",
            }
        )

        assert status == 200
        audio_path = Path(payload["saved_audio_path"])
        assert audio_path.exists()
        assert _registry_by_path(harness, str(audio_path))
        assert not _artifact_by_type(payload, "audio")
        integrity = payload["tts_audio_integrity_evidence"]
        assert integrity["status"] == "failed"
        assert integrity["reason_code"] == "TTS_AUDIO_NO_ACTIVE_SIGNAL"
        assert integrity["materialization_eligible"] is False
        assert not _fulfilled_outputs(payload, "audio")
        assert payload["lifecycle_state"] == "repair_needed"
        blocked_output = next(
            item
            for item in payload["outputs"]
            if item.get("type") == "audio"
        )
        assert blocked_output["status"] == "blocked"
        assert blocked_output["lifecycle"] == "diagnostic_artifact"
        assert blocked_output["blocked_reason"] == "TTS_AUDIO_NO_ACTIVE_SIGNAL"

        truth_payload = _truth_response(harness, "resp_fake_silent_tts")
        assert truth_payload["tts_audio_integrity_evidence"]["status"] == "failed"
        assert _closure_review(truth_payload)["counts"]["fulfilled"] == 0
        assert _closure_review(truth_payload)["counts"]["blocked"] == 1
        assert _closure_review(truth_payload)["surface_state"]["status"] == "blocked"
        audio_check = next(
            item
            for item in _closure_review(truth_payload)["checks"]
            if item.get("output_type") == "audio"
        )
        assert audio_check["evidence"] == "tts_audio_integrity_failed"
        assert audio_check["materialization_blocked"] is True


def test_mixed_web_artifact_flow_rebinds_links_to_concrete_saved_artifacts():
    with FakeBackendHarness() as harness:
        image_payload, image_status = harness.post_response(
            {
                "response_id": "resp_fake_web_image",
                "capability": "image_generation",
                "prompt": "Generate the concrete image dependency for a mixed web artifact.",
            }
        )
        assert image_status == 200
        image_artifact = _artifact_by_type(image_payload, "image")[0]
        image_path = Path(image_artifact["path"])

        web_payload, web_status = harness.post_response(
            {
                "response_id": "resp_fake_mixed_web",
                "capability": "chat",
                "fake_scenario": "mixed_web",
                "linked_image_path": str(image_path),
                "derived_from": [image_artifact["artifact_ref"]],
                "prompt": "Create a mixed web artifact with index.html, styles.css, and a generated image.",
                "messages": [{"role": "user", "content": "Create a mixed web artifact with index.html and styles.css."}],
            }
        )

        assert web_status == 200
        text_artifacts = _artifact_by_type(web_payload, "text")
        paths = {Path(item["path"]).name: Path(item["path"]) for item in text_artifacts}
        assert set(paths) == {"index.html", "styles.css"}
        index_html = paths["index.html"].read_text(encoding="utf-8")
        styles_css = paths["styles.css"].read_text(encoding="utf-8")
        image_rel = os.path.relpath(image_path, start=paths["index.html"].parent)
        assert f'src="{image_rel}"' in index_html
        assert f'url("{image_rel}")' in styles_css
        assert "placeholder" not in index_html.lower()
        assert "placeholder" not in styles_css.lower()

        registry_paths = {record["artifact"]["path"] for record in harness.registry_records()}
        assert str(image_path) in registry_paths
        assert str(paths["index.html"]) in registry_paths
        assert str(paths["styles.css"]) in registry_paths
        text_artifact_refs = {item["artifact_ref"] for item in text_artifacts}
        assert text_artifact_refs <= {
            item.get("artifact_ref")
            for item in web_payload.get("outputs", [])
            if item.get("status") == "fulfilled"
        }
        assert text_artifact_refs <= {
            item.get("artifact_ref")
            for item in web_payload.get("output_slots", [])
            if item.get("status") == "fulfilled"
        }

        truth_payload = _truth_response(harness, "resp_fake_mixed_web")
        frame_paths = {
            artifact["path"]
            for artifact in truth_payload["response_frame"]["current_state"]["artifacts"]
            if isinstance(artifact, dict)
        }
        assert str(paths["index.html"]) in frame_paths
        assert str(paths["styles.css"]) in frame_paths
        closure = truth_payload["runtime"]["graph_closure_review"]
        assert closure["status"] in {"completed", "blocked", "pending"}
        if closure["status"] != "completed":
            assert closure.get("repair_gap_code") or closure.get("checks")


def test_original_dedicated_css_prompt_keeps_css_as_a_separate_runtime_artifact():
    prompt = (
        "Build a bold landing page with html and a dedicated css for an eco-friendly clothing line "
        "called 'Pure Thread.' Generate four images: a macro shot of organic cotton texture, a model "
        "wearing a simple white linen shirt, a close-up of sustainable recycled buttons, and a sunny "
        "outdoor scene at a botanical garden."
    )

    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_original_dedicated_css",
                "capability": "chat",
                "fake_scenario": "mixed_web",
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        assert status == 200
        truth_payload = _truth_response(harness, "resp_fake_original_dedicated_css")
        graph = _runtime_graph(truth_payload)
        text_targets = {
            (item.get("target_name"), item.get("target_extension"))
            for item in _intent_obligations(truth_payload, kind="text_artifact")
        }
        assert {("generated-html", "html"), ("generated-css", "css")} <= text_targets
        assert len(
            [
                item for item in _intent_obligations(truth_payload, kind="media_artifact")
                if item.get("capability") == "image_generation"
            ]
        ) == 4
        assert {
            branch.get("text_artifact_extension")
            for branch in graph.get("downstream_branches") or []
            if branch.get("role") == "text_artifact_output"
        } >= {"html", "css"}
        artifact_names = {
            Path(item["path"]).name
            for item in _artifact_by_type(payload, "text")
        }
        assert artifact_names == {"index.html", "styles.css"}


def test_fake_e2e_exposes_intent_obligation_and_adequacy_truth_for_local_web_plan():
    prompt = (
        "Create a two-page website with index.html, suiten.html, shared styles.css, "
        "navigation between both pages, and exactly two generated local images linked from the pages."
    )

    with FakeBackendHarness() as harness:
        image_payload, image_status = harness.post_response(
            {
                "response_id": "resp_fake_obligation_seed_image",
                "capability": "image_generation",
                "prompt": "Generate one concrete dependency image for the fake web plan.",
            }
        )
        assert image_status == 200
        image_artifact = _artifact_by_type(image_payload, "image")[0]

        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_intent_obligation_web",
                "capability": "chat",
                "fake_scenario": "mixed_web",
                "linked_image_path": image_artifact["path"],
                "derived_from": [image_artifact["artifact_ref"]],
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        assert status == 200
        truth_payload = _truth_response(harness, "resp_fake_intent_obligation_web")
        graph = _runtime_graph(truth_payload)
        assert graph["kind"] == "ollmo.request_phase_graph"
        assert graph["prompt_intent"]["requested_visual_output_count"] == 2
        assert graph["prompt_intent"]["intent_obligation_count"] == len(_intent_obligations(truth_payload))
        assert set(graph["prompt_intent"]["intent_obligation_kinds"]) >= {
            "text_artifact",
            "media_artifact",
            "dependency",
            "navigation",
        }

        text_targets = {
            (item.get("target_name"), item.get("target_extension"))
            for item in _intent_obligations(truth_payload, kind="text_artifact")
        }
        assert {("index", "html"), ("suiten", "html"), ("styles", "css")} <= text_targets
        media_obligations = _intent_obligations(truth_payload, kind="media_artifact")
        assert len([item for item in media_obligations if item.get("capability") == "image_generation"]) == 2

        local_dependency_obligations = [
            item for item in _intent_obligations(truth_payload, kind="dependency")
            if item.get("dependency_contract") == "local_visual_asset_binding"
            and item.get("execution_dependency_required") is True
        ]
        assert local_dependency_obligations
        image_phase_ids = {
            item.get("phase_id")
            for item in graph["downstream_branches"]
            if item.get("capability") == "image_generation"
        }
        assert len(image_phase_ids) == 2
        html_branches = [
            item for item in graph["downstream_branches"]
            if item.get("role") == "text_artifact_output"
            and item.get("text_artifact_extension") == "html"
        ]
        assert {item.get("text_artifact_source_name") for item in html_branches} >= {"index", "suiten"}
        assert all(image_phase_ids <= set(item.get("depends_on") or []) for item in html_branches)
        assert all(item.get("dependency_contract") == "local_visual_asset_binding" for item in html_branches)

        navigation_obligations = _intent_obligations(truth_payload, kind="navigation")
        assert {
            (item.get("from_target_name"), item.get("to_target_name"))
            for item in navigation_obligations
        } >= {("index", "suiten"), ("suiten", "index")}

        adequacy = _closure_review(truth_payload)["intent_graph_adequacy"]
        assert adequacy["status"] == "fulfilled"
        assert adequacy["intent_obligation_count"] == len(_intent_obligations(truth_payload))
        assert adequacy["required_intent_obligation_count"] == len(_intent_obligations(truth_payload))
        assert not adequacy.get("checks")


def test_fake_e2e_accepted_learning_hint_stays_soft_and_non_executable():
    accepted_learning_hints = {
        "kind": "ollmo.accepted_learning_runtime_hints",
        "status": "active",
        "enabled": True,
        "authority": "soft_hint",
        "runtime_effect": "soft_hints_available",
        "hint_count": 1,
        "hints": [
            {
                "kind": "ollmo.accepted_learning_runtime_hint",
                "learning_id": "accepted-policy-e2e-graph-repair",
                "target_area": "graph_repair_policy",
                "allowed_use": "soft_hint_only",
                "hint": "Watch for missing graph obligations, but do not patch without runtime evidence.",
            }
        ],
    }

    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_learning_boundary",
                "capability": "chat",
                "fake_scenario": "text_artifact",
                "prompt": "Create report.md as a saved text artifact.",
                "messages": [{"role": "user", "content": "Create report.md as a saved text artifact."}],
                "accepted_learning_hints": accepted_learning_hints,
            }
        )

        assert status == 200
        truth_payload = _truth_response(harness, "resp_fake_learning_boundary")
        graph = _runtime_graph(truth_payload)
        decision_contract = graph["request_ir"]["decision_contract"]
        assert decision_contract["accepted_learning"]["runtime_effect"] == "soft_hints_available"
        assert decision_contract["accepted_learning"]["hints"][0]["allowed_use"] == "soft_hint_only"

        runtime = truth_payload["runtime"]
        diagnostics = runtime.get("developer_diagnostics") or {}
        assert not graph.get("graph_repair_proposals")
        assert not graph.get("graph_repair_reviews")
        assert not diagnostics.get("runtime_graph_repair_proposals")
        assert not diagnostics.get("runtime_graph_repair_proposal_reviews")
        assert "graph_patch_lifecycle" not in graph
        assert "staged_graph_patches" not in graph
        assert "applied_graph_patches" not in graph


def test_fake_e2e_applied_graph_patch_branch_executes_in_same_response_turn():
    phase_graph = {
        "kind": "ollmo.request_phase_graph",
        "graph_id": "graph-fake-same-turn-reseed",
        "current_phase_id": "phase-1",
        "current_phase_capability": "chat",
        "current_phase_resolution": "graph_resolved",
        "mode": "phase_chain",
        "phases": [
            {
                "phase_id": "phase-1",
                "branch_id": "phase-1",
                "capability": "chat",
                "output_type": "text",
                "status": "completed",
            },
            {
                "phase_id": "skipped-context",
                "branch_id": "skipped-context",
                "capability": "chat",
                "output_type": "text",
                "status": "skipped",
            },
        ],
        "downstream_branch_ids": ["skipped-context"],
        "downstream_branches": [
            {
                "phase_id": "skipped-context",
                "branch_id": "skipped-context",
                "capability": "chat",
                "output_type": "text",
                "status": "skipped",
            }
        ],
        "output_obligations": [],
    }
    proposal = build_graph_repair_proposal_from_repair_gap(
        request_phase_graph=phase_graph,
        repair_gap={
            "trigger": "ghost_repair_feedback",
            "ghost_repair_feedback": {"status": "repair_required"},
            "repair_loop": {"status": "promoted"},
            "pending_branches": [
                {
                    "phase_id": "repair-image",
                    "branch_id": "repair-image",
                    "obligation_id": "obligation-repair-image",
                    "capability": "image_generation",
                    "output_type": "image",
                    "artifact_prompt": "A deterministic image materialized by the repaired branch.",
                }
            ],
        },
    )
    review = validate_graph_repair_proposal(
        proposal,
        request_phase_graph=phase_graph,
        closure_review={
            "status": "repair_required",
            "ghost_repair_feedback": {"status": "repair_required"},
        },
        promotion_review={"status": "promoted"},
    )
    assert review["status"] == "accepted"
    phase_graph["graph_repair_reviews"] = [review]

    with FakeBackendHarness() as harness:
        chat_instance = harness.instances["chat"]

        def resolve_with_repair_graph(*args, **kwargs):
            return (
                {
                    "instance_id": chat_instance["instance_id"],
                    "instance": dict(chat_instance),
                    "capability": "chat",
                    "route_source": "ghost_carried",
                    "route_reason": "runtime-owned same-turn graph repair regression",
                    "route_confidence": 1.0,
                    "route_runtime": {"request_phase_graph": phase_graph},
                },
                None,
            )

        def complete_inline(**kwargs):
            ollmo_webserver._complete_response_late_fill(**kwargs)
            return True

        def resolve_fake_late_fill(request_payload, *, expected_capability, **kwargs):
            instance = harness.instances[expected_capability]
            effective_payload = dict(request_payload)
            effective_payload["capability"] = expected_capability
            return (
                effective_payload,
                harness._route_info(expected_capability, instance),
                None,
            )

        with (
            patch.dict(os.environ, {"OLLMO_GRAPH_REPAIR_AUTONOMY": "apply_safe"}, clear=False),
            patch.object(ollmo_webserver, "_resolve_ghost_auto_route", side_effect=resolve_with_repair_graph),
            patch.object(ollmo_webserver, "_resolve_late_fill_route", side_effect=resolve_fake_late_fill),
            patch.object(ollmo_webserver, "_schedule_response_late_fill", side_effect=complete_inline),
        ):
            initial, status = harness.post_response(
                {
                    "response_id": "resp_fake_graph_patch_same_turn_execute",
                    "ghost_route": True,
                    "capability": "chat",
                    "prompt": "Return a concise current response.",
                }
            )

        assert status == 200
        assert initial["late_fill"]["pending_branches"][0]["branch_id"] == "repair-image"
        assert "repair-image" in _runtime_graph(initial)["downstream_branch_ids"]
        assert (
            initial["runtime"]["developer_diagnostics"]
            ["graph_patch_late_fill_reconciliation"]["scheduled_branch_ids"]
            == ["repair-image"]
        )
        recovered, recovered_status = harness.get_response("resp_fake_graph_patch_same_turn_execute")
        assert recovered_status == 200
        assert harness.calls["image_generation"] == 1, {
            key: recovered.get("late_fill", {}).get(key)
            for key in (
                "status",
                "pending_branches",
                "completed_branches",
                "failed_branches",
                "cancelled_branches",
                "fill_results",
                "error",
                "error_message",
            )
        }
        assert recovered["late_fill"]["status"] == "completed"
        assert recovered["late_fill"].get("pending_branches", []) == []
        assert recovered["late_fill"]["completed_branches"][0]["branch_id"] == "repair-image"
        assert _artifact_by_type(recovered, "image")


def test_fake_e2e_terminal_safe_graph_patch_successor_executes_exact_branch_once():
    response_id = "resp_fake_terminal_graph_patch_successor"
    branch_prompt = "A deterministic image created only by the bounded successor branch."
    phase_graph = {
        "kind": "ollmo.request_phase_graph",
        "graph_id": "graph-fake-terminal-successor",
        "current_phase_id": "phase-1",
        "current_phase_capability": "chat",
        "mode": "phase_chain",
        "phases": [
            {
                "phase_id": "phase-1",
                "branch_id": "phase-1",
                "capability": "chat",
                "output_type": "text",
                "status": "completed",
            }
        ],
        "downstream_branch_ids": [],
        "downstream_branches": [],
        "output_obligations": [],
    }
    closure_review = {
        "kind": "ollmo.graph_closure_review",
        "status": "repair_required",
        "checks": [
            {
                "check_kind": "materialization_contract",
                "status": "repair_required",
                "repair_action": "repair_missing_materialization_contract",
                "phase_id": "repair-image",
                "branch_id": "repair-image",
                "evidence": "missing materialization branch",
            }
        ],
    }
    repair_branch = {
        "phase_id": "repair-image",
        "branch_id": "repair-image",
        "obligation_id": "obligation-repair-image",
        "capability": "image_generation",
        "output_type": "image",
        "status": "pending",
        "artifact_prompt": branch_prompt,
    }
    request_payload = {
        "prompt": "This root prompt must not be replayed as the image branch.",
        "response_id": response_id,
    }

    with FakeBackendHarness() as harness:
        parent = harness.freeze_manual_response(
            {
                "id": response_id,
                "status": "completed",
                "lifecycle_state": "completed",
                "mode": "chat",
                "capability": "chat",
                "output_text": "Frozen parent output.",
                "runtime": {
                    "request_phase_graph": phase_graph,
                    "graph_closure_review": closure_review,
                },
                "late_fill": {
                    "status": "partial_failed",
                    "final_materialization_contract_status": "unmet",
                    "materialization_contract_unmet": True,
                    "pending_branches": [repair_branch],
                    "pending_capabilities": ["image_generation"],
                },
            },
            request_payload=request_payload,
        )
        ollmo_webserver._register_response_lookup(
            response_id=response_id,
            message_id="",
            instance_id=harness.instances["chat"]["instance_id"],
            model_name=harness.instances["chat"]["model"],
            backend="fake",
            capability="chat",
            mode="chat",
            route_payload=None,
        )
        ollmo_webserver._touch_response_lookup(
            response_id,
            status="completed",
            output_text=parent["output_text"],
            response_payload=parent,
        )
        ledger_path = harness.response_frames_dir / "responses.jsonl"
        frozen_parent_line = ledger_path.read_bytes().splitlines()[0]

        with patch.dict(
            os.environ,
            {
                "OLLMO_GRAPH_REPAIR_AUTONOMY": "apply_enforced",
                "OLLMO_APPLY_ENFORCED_POLICY": "safe_v1",
            },
            clear=False,
        ):
            prepared = (
                ollmo_webserver._RESPONSES_REQUEST_RUNTIME
                .prepare_terminal_graph_patch_successor(parent)
            )

        assert prepared["status"] == "queued", prepared
        assert [
            item["branch_id"] for item in prepared["artifact_gap"]["pending_branches"]
        ] == ["repair-image"]
        queued_successor = ollmo_webserver._finalize_response_frame_payload(
            prepared["response_payload"],
            request_payload=request_payload,
            persist=True,
        )
        ollmo_webserver._touch_response_lookup(
            response_id,
            status="completed",
            output_text=queued_successor["output_text"],
            response_payload=queued_successor,
        )

        def resolve_fake_late_fill(request_payload, *, expected_capability, **kwargs):
            instance = harness.instances[expected_capability]
            effective_payload = dict(request_payload)
            effective_payload["capability"] = expected_capability
            return (
                effective_payload,
                harness._route_info(expected_capability, instance),
                None,
            )

        with (
            patch.dict(
                os.environ,
                {
                    "OLLMO_GRAPH_REPAIR_AUTONOMY": "apply_enforced",
                    "OLLMO_APPLY_ENFORCED_POLICY": "safe_v1",
                },
                clear=False,
            ),
            patch.object(
                ollmo_webserver,
                "_resolve_late_fill_route",
                side_effect=resolve_fake_late_fill,
            ),
        ):
            ollmo_webserver._complete_response_late_fill(
                response_payload=queued_successor,
                request_payload=request_payload,
                assistant_message=queued_successor["output_text"],
                artifact_gap=prepared["artifact_gap"],
                source_route_payload=None,
            )

        # Force truth recovery from the append-only frame ledger rather than
        # accepting the same-process lookup cache as persistence evidence.
        ollmo_webserver._RESPONSE_LOOKUP.pop(response_id, None)
        recovered, recovered_status = harness.get_response(response_id, view="truth")
        assert recovered_status == 200
        assert harness.calls["image_generation"] == 1
        image_calls = [
            item for item in harness.call_records
            if item.get("capability") == "image_generation"
        ]
        assert len(image_calls) == 1
        assert image_calls[0]["payload"]["prompt"] == branch_prompt
        assert image_calls[0]["payload"]["prompt"] != request_payload["prompt"]
        assert recovered["late_fill"]["status"] == "completed"
        assert recovered["late_fill"]["successor_reopen_execution"]["status"] == "completed"
        recovered_graph = recovered["runtime"]["request_phase_graph"]
        assert recovered_graph["successor_reopen_executions"][0]["status"] == "completed"
        assert (
            recovered_graph["successor_reopen_requests"][0]["execution"]["status"]
            == "completed"
        )
        assert _artifact_by_type(recovered, "image")

        ledger_lines = ledger_path.read_bytes().splitlines()
        assert ledger_lines[0] == frozen_parent_line
        ledger_frames = [json.loads(line) for line in ledger_lines]
        reopen_frames = [
            frame for frame in ledger_frames
            if (frame.get("frame_relation") or {}).get("kind") == "graph_patch_reopen_successor"
        ]
        assert len(reopen_frames) == 1
        assert reopen_frames[0]["frame_relation"]["parent_frame_id"] == parent["response_frame"]["frame_id"]
        assert reopen_frames[0]["response_id"] == response_id


def test_fake_e2e_terminal_late_fill_reviews_rebase_in_shadow_without_executing_candidate():
    phase_graph = {
        "kind": "ollmo.request_phase_graph",
        "graph_version": 3,
        "graph_id": "graph-fake-terminal-rebase-shadow",
        "current_phase_id": "phase-1",
        "mode": "phase_chain",
        "phases": [
            {
                "phase_id": "phase-1",
                "branch_id": "phase-1",
                "obligation_id": "obligation-phase-1",
                "capability": "chat",
                "output_type": "text",
                "status": "completed",
            },
            {
                "phase_id": "phase-image",
                "branch_id": "branch-image",
                "obligation_id": "obligation-image",
                "capability": "image_generation",
                "output_type": "image",
                "status": "pending",
                "depends_on": ["phase-1"],
                "artifact_prompt": "A deterministic terminal shadow review image.",
            },
        ],
        "downstream_branches": [
            {
                "phase_id": "phase-image",
                "branch_id": "branch-image",
                "obligation_id": "obligation-image",
                "capability": "image_generation",
                "output_type": "image",
                "status": "pending",
                "depends_on": ["phase-1"],
                "artifact_prompt": "A deterministic terminal shadow review image.",
            }
        ],
        "intent_obligations": [
            {
                "obligation_id": "intent-image",
                "phase_id": "phase-image",
                "capability": "image_generation",
                "output_type": "image",
                "required": True,
            }
        ],
        "output_obligations": [
            {
                "obligation_id": "obligation-image",
                "phase_id": "phase-image",
                "capability": "image_generation",
                "output_type": "image",
                "required": True,
            }
        ],
    }
    candidate_graph = copy.deepcopy(phase_graph)
    candidate_graph["phases"].append(
        {
            "phase_id": "phase-terminal-review",
            "branch_id": "branch-terminal-review",
            "obligation_id": "obligation-terminal-review",
            "capability": "chat",
            "output_type": "text",
            "kind": "review",
            "status": "pending",
            "depends_on": ["phase-image"],
        }
    )
    candidate_graph["downstream_branches"].append(
        {
            "phase_id": "phase-terminal-review",
            "branch_id": "branch-terminal-review",
            "obligation_id": "obligation-terminal-review",
            "capability": "chat",
            "output_type": "text",
            "kind": "review",
            "status": "pending",
            "depends_on": ["phase-image"],
        }
    )
    candidate_graph["output_obligations"].append(
        {
            "obligation_id": "obligation-terminal-review",
            "phase_id": "phase-terminal-review",
            "capability": "chat",
            "output_type": "text",
            "required": True,
        }
    )
    closure_review = {
        "kind": "ollmo.graph_closure_review",
        "status": "repair_required",
        "checks": [
            {
                "check_kind": "semantic_graph_shape",
                "status": "repair_required",
                "repair_action": "full_successor_rebase",
                "reason": "terminal materialization truth requires successor review",
            }
        ],
    }

    with FakeBackendHarness() as harness:
        chat_instance = harness.instances["chat"]
        terminal_callback_inputs = []
        terminal_callback_outputs = []
        original_terminal_review = (
            ollmo_webserver._RESPONSES_REQUEST_RUNTIME.review_terminal_graph_rebase_after_late_fill
        )

        def resolve_with_phase_graph(*args, **kwargs):
            return (
                {
                    "instance_id": chat_instance["instance_id"],
                    "instance": dict(chat_instance),
                    "capability": "chat",
                    "route_source": "ghost_carried",
                    "route_reason": "runtime-owned terminal graph rebase shadow regression",
                    "route_confidence": 1.0,
                    "route_runtime": {"request_phase_graph": phase_graph},
                },
                None,
            )

        def resolve_fake_late_fill(request_payload, *, expected_capability, **kwargs):
            instance = harness.instances[expected_capability]
            effective_payload = dict(request_payload)
            effective_payload["capability"] = expected_capability
            return (
                effective_payload,
                harness._route_info(expected_capability, instance),
                None,
            )

        def complete_inline(**kwargs):
            ollmo_webserver._complete_response_late_fill(**kwargs)
            return True

        def capture_terminal_review(payload, **kwargs):
            terminal_callback_inputs.append(copy.deepcopy(payload))
            reviewed = original_terminal_review(payload, **kwargs)
            terminal_callback_outputs.append(copy.deepcopy(reviewed))
            return reviewed

        with (
            patch.dict(
                os.environ,
                {"OLLMO_GRAPH_REBASE_AUTONOMY": "shadow"},
                clear=False,
            ),
            patch.object(
                ollmo_webserver,
                "_resolve_ghost_auto_route",
                side_effect=resolve_with_phase_graph,
            ),
            patch.object(
                ollmo_webserver,
                "_resolve_late_fill_route",
                side_effect=resolve_fake_late_fill,
            ),
            patch.object(
                ollmo_webserver,
                "_build_graph_closure_review",
                return_value=closure_review,
            ),
            patch(
                "ollmo_server.responses_request_runtime.build_request_phase_graph",
                return_value=candidate_graph,
            ),
            patch.object(
                ollmo_webserver._RESPONSES_REQUEST_RUNTIME,
                "review_terminal_graph_rebase_after_late_fill",
                side_effect=capture_terminal_review,
            ),
            patch.object(
                ollmo_webserver,
                "_schedule_response_late_fill",
                side_effect=complete_inline,
            ),
        ):
            initial, status = harness.post_response(
                {
                    "response_id": "resp_fake_terminal_rebase_shadow",
                    "ghost_route": True,
                    "capability": "chat",
                    "prompt": "Generate one image and then finish the response.",
                }
            )

        assert status == 200
        assert initial["late_fill"]["status"] == "pending"
        assert terminal_callback_inputs
        assert terminal_callback_outputs
        callback_payload = terminal_callback_inputs[-1]
        assert callback_payload["late_fill"]["status"] == "completed"
        assert callback_payload["late_fill"].get("pending_branches", []) == []
        assert callback_payload["lifecycle_state"] == "late_fill_pending"
        callback_graph = _runtime_graph(terminal_callback_outputs[-1])
        assert callback_graph["graph_rebase_reviews"][0]["status"] == "accepted"

        harness.clear_response_lookup()
        recovered, recovered_status = harness.get_response(
            "resp_fake_terminal_rebase_shadow",
            view="truth",
        )
        assert recovered_status == 200
        graph = _runtime_graph(recovered)
        assert graph.get("graph_rebase_reviews"), {
            "runtime_keys": sorted((recovered.get("runtime") or {}).keys()),
            "graph_keys": sorted(graph.keys()),
        }
        assert graph["graph_rebase_reviews"][0]["status"] == "accepted"
        assert "active_late_fill_must_settle" not in (
            graph["graph_rebase_reviews"][0].get("blocked_reasons") or []
        )
        proposal_id = graph["graph_rebase_reviews"][0]["proposal_id"]
        current_lifecycle = [
            item
            for item in graph.get("graph_rebase_lifecycle") or []
            if item.get("proposal_id") == proposal_id
        ]
        assert len(current_lifecycle) == 1
        assert current_lifecycle[0]["status"] == "validated"
        assert current_lifecycle[0]["outcome"]["runtime_effect"] == "shadow_no_mutation"
        assert graph.get("staged_graph_rebases") in (None, [])
        assert graph.get("successor_rebase_requests") in (None, [])
        assert "phase-terminal-review" not in {
            item.get("phase_id") for item in graph.get("phases") or []
        }
        diagnostics = recovered["runtime"]["developer_diagnostics"]
        assert diagnostics["runtime_graph_rebase_candidate_review"]["status"] == (
            "validated_by_runtime_review"
        )
        assert harness.calls["chat"] == 1
        assert harness.calls["image_generation"] >= 1


def test_failure_truth_keeps_response_failed_without_artifact_completion_claim():
    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_timeout_truth",
                "capability": "image_generation",
                "prompt": "Force a deterministic timeout failure.",
            }
        )

        assert status == 504
        assert payload["status"] == "failed"
        assert payload["lifecycle_state"] == "failed"
        assert payload["status_semantics"]["is_terminal"] is True
        assert payload["status_semantics"]["has_open_continuation"] is False
        assert not payload.get("artifacts")
        assert not _fulfilled_outputs(payload, "image")
        assert harness.registry_records() == []
        frame_state = harness.response_state("resp_fake_timeout_truth")
        assert frame_state["ok"] is True
        assert frame_state["response_payload"]["status"] == "failed"


def test_restart_recovery_reconstructs_from_durable_response_frame_after_lookup_clear():
    with FakeBackendHarness() as harness:
        response_payload = harness.build_response_payload(
            response_id="resp_fake_recovery_truth",
            capability="chat",
            mode="chat",
            output_text="Initial text is complete while audio remains pending.",
            source_payload={"content": "Initial text is complete while audio remains pending."},
        )
        response_payload["status"] = "completed"
        response_payload["lifecycle_state"] = "late_fill_pending"
        response_payload["late_fill"] = {
            "status": "pending",
            "pending_branches": [
                {
                    "branch_id": "branch-text_to_speech-1",
                    "phase_id": "phase-2",
                    "capability": "text_to_speech",
                    "output_type": "audio",
                    "status": "pending",
                }
            ],
        }
        response_payload["runtime"] = {
            **response_payload.get("runtime", {}),
            "graph_closure_review": {
                "kind": "ollmo.graph_closure_review",
                "status": "pending",
                "surface_state": {"state": "open", "reason": "fake audio branch pending"},
            },
        }
        frozen = harness.freeze_manual_response(
            response_payload,
            request_payload={"prompt": "Create text and then read it aloud."},
        )
        assert frozen["response_frame"]["working_frame"]["status"] == "frozen"

        harness.clear_response_lookup()
        recovered, status = harness.get_response("resp_fake_recovery_truth")

        assert status == 200
        assert recovered["id"] == "resp_fake_recovery_truth"
        assert recovered["lifecycle_state"] == "late_fill_pending"
        assert recovered["late_fill"]["status"] == "pending"
        assert recovered["late_fill"]["pending_branches"][0]["capability"] == "text_to_speech"
        assert recovered["ui_compact"] is True
        assert recovered["status_lookup"]["frame_id"] == frozen["response_frame"]["frame_id"]
        assert "response_frame" not in recovered
        assert "runtime" not in recovered


def test_observer_cached_paths_do_not_call_backends_or_write_truth_state():
    with FakeBackendHarness() as harness:
        payload, status = harness.post_response(
            {
                "response_id": "resp_fake_observer_source",
                "capability": "chat",
                "fake_scenario": "text_artifact",
                "prompt": "Create report.md for observer checks.",
                "messages": [{"role": "user", "content": "Create report.md for observer checks."}],
            }
        )
        assert status == 200
        before = harness.snapshot_counts()

        compact, compact_status = harness.get_response("resp_fake_observer_source", view="status")
        assert compact_status == 200
        assert compact["object"] == "response.status"
        assert compact["compact"] is True
        assert "response_frame" not in compact
        assert "artifacts" not in compact

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS", None)
            os.environ.pop("OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE", None)
            preview, preview_status = harness.ghost_preview(
                {
                    "prompt": "Which fake capability would handle this?",
                    "capability_hint": "chat",
                    "compute_semantics": False,
                }
            )
        after_cached = harness.snapshot_counts()

        assert preview_status == 200
        assert preview["runtime_truth"]["truth_mode"] == "cached"
        assert preview["runtime_truth"]["semantic_compute_performed"] is False
        assert after_cached == before

        computed, computed_status = harness.ghost_preview(
            {
                "prompt": "Compute a fake semantic preview.",
                "capability_hint": "chat",
                "compute_semantics": True,
            }
        )

        assert computed_status == 200
        assert computed["runtime_truth"]["truth_mode"] == "computed"
        assert computed["runtime_truth"]["semantic_compute_performed"] is True
        assert harness.calls["embedding"] == 1
