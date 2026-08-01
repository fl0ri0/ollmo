from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import ollmo_webserver
from ollmo_services.response_frames import load_latest_response_state
from ollmo_services.responses import build_canonical_response_payload

from .fixtures import (
    TEXT_ARTIFACT_CONTENT,
    TRANSCRIPT_TEXT,
    VISION_RESULT,
    deterministic_embedding,
    silent_wav_bytes,
    tiny_png_bytes,
    tiny_wav_bytes,
    write_bytes,
    write_text,
)


class FakeBackendHarness:
    """Patch Ollmo's response route to deterministic temp-root fake backends."""

    def __init__(self) -> None:
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._stack: ExitStack | None = None
        self._prior_testing: Any = None
        self._prior_lookup: dict[str, Any] = {}
        self._prior_in_flight: set[str] = set()
        self.calls: Counter[str] = Counter()
        self.call_records: list[dict[str, Any]] = []

    def __enter__(self) -> "FakeBackendHarness":
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.artifacts_dir = self.root / "artifacts"
        self.documents_dir = self.artifacts_dir / "documents"
        self.web_dir = self.artifacts_dir / "web"
        self.images_dir = self.artifacts_dir / "images"
        self.audio_dir = self.artifacts_dir / "audio"
        self.state_dir = self.root / "state"
        self.response_frames_dir = self.state_dir / "response_frames"
        self.chat_history_dir = self.state_dir / "chat_history"
        self.registry_path = self.state_dir / "artifact_registry.jsonl"
        self.logs_dir = self.root / "logs"
        self.runtime_registry_path = self.root / "model_ports.json"
        self.runtime_status_path = self.state_dir / "runtime_status.json"
        self.ghost_preferences_path = self.state_dir / "ghost_preferences.json"
        for directory in (
            self.documents_dir,
            self.web_dir,
            self.images_dir,
            self.audio_dir,
            self.response_frames_dir,
            self.chat_history_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.registry_path.touch()
        self.runtime_registry_path.write_text("[]\n", encoding="utf-8")
        self.runtime_status_path.write_text(
            json.dumps({"schema_version": 1, "updated_at": "fake", "instances": {}}),
            encoding="utf-8",
        )
        self.instances = self._build_instances()

        self._prior_testing = ollmo_webserver.app.config.get("TESTING")
        self._prior_lookup = dict(ollmo_webserver._RESPONSE_LOOKUP)
        self._prior_in_flight = set(ollmo_webserver._RESPONSE_LATE_FILL_IN_FLIGHT)
        ollmo_webserver._RESPONSE_LOOKUP.clear()
        ollmo_webserver._RESPONSE_LATE_FILL_IN_FLIGHT.clear()
        ollmo_webserver.app.config["TESTING"] = False

        self._stack = ExitStack()
        self._stack.enter_context(patch.object(ollmo_webserver, "CONFIG_FILE_NAME", str(self.runtime_registry_path)))
        self._stack.enter_context(patch.object(ollmo_webserver, "RUNTIME_STATUS_PATH", self.runtime_status_path))
        self._stack.enter_context(patch.object(ollmo_webserver, "RESPONSE_FRAMES_DIR", self.response_frames_dir))
        self._stack.enter_context(patch.object(ollmo_webserver, "ARTIFACT_REGISTRY_LEDGER", self.registry_path))
        self._stack.enter_context(patch.object(ollmo_webserver, "CHAT_HISTORY_DIR", self.chat_history_dir))
        self._stack.enter_context(patch.object(ollmo_webserver, "GHOST_PREFERENCES_PATH", self.ghost_preferences_path))
        self._stack.enter_context(patch.object(ollmo_webserver, "load_running_instances", self._load_running_instances))
        self._stack.enter_context(
            patch.object(ollmo_webserver, "merge_instances_with_runtime_status", self._merge_instances_with_runtime_status)
        )
        self._stack.enter_context(
            patch.object(ollmo_webserver, "_resolve_responses_target_instance", self._resolve_responses_target_instance)
        )
        self._stack.enter_context(patch.object(ollmo_webserver, "_resolve_ghost_auto_route", self._resolve_ghost_auto_route))
        self._stack.enter_context(
            patch.object(ollmo_webserver, "_prepare_effective_request_data", self._prepare_effective_request_data)
        )
        self._stack.enter_context(
            patch.object(ollmo_webserver, "_execute_chat_backend_request", self._execute_chat_backend_request)
        )
        self._stack.enter_context(
            patch.object(ollmo_webserver, "_invoke_internal_api_json_route", self._invoke_internal_api_json_route)
        )
        self._stack.enter_context(
            patch.object(ollmo_webserver, "_execute_embedding_backend_request", self._execute_embedding_backend_request)
        )
        self._stack.enter_context(
            patch.object(
                ollmo_webserver,
                "_persist_generated_text_artifact_if_requested",
                self._persist_generated_text_artifact_if_requested,
            )
        )
        self._stack.enter_context(patch.object(ollmo_webserver, "_schedule_response_late_fill", self._schedule_noop))
        self._stack.enter_context(
            patch.object(ollmo_webserver, "_schedule_post_response_substrate_hygiene", self._schedule_noop)
        )
        self._stack.enter_context(patch.object(ollmo_webserver, "_log_unified_event", self._log_noop))
        self._stack.enter_context(patch.object(ollmo_webserver, "read_events", lambda *args, **kwargs: []))
        self._stack.enter_context(
            patch.object(ollmo_webserver, "build_ghost_payload", lambda *args, **kwargs: {"recommendations": [], "issues": []})
        )
        self.client = ollmo_webserver.app.test_client()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stack is not None:
            self._stack.close()
        ollmo_webserver.app.config["TESTING"] = self._prior_testing
        ollmo_webserver._RESPONSE_LOOKUP.clear()
        ollmo_webserver._RESPONSE_LOOKUP.update(self._prior_lookup)
        ollmo_webserver._RESPONSE_LATE_FILL_IN_FLIGHT.clear()
        ollmo_webserver._RESPONSE_LATE_FILL_IN_FLIGHT.update(self._prior_in_flight)
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def post_response(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        response = self.client.post("/api/responses", json=payload)
        return response.get_json(), response.status_code

    def get_response(self, response_id: str, *, view: str | None = None) -> tuple[dict[str, Any], int]:
        query_string = {"view": view} if view else None
        response = self.client.get(f"/api/responses/{response_id}", query_string=query_string)
        return response.get_json(), response.status_code

    def ghost_preview(
        self,
        payload: dict[str, Any],
        *,
        query_string: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        response = self.client.post("/api/ghost_route_preview", json=payload, query_string=query_string)
        return response.get_json(), response.status_code

    def clear_response_lookup(self) -> None:
        ollmo_webserver._RESPONSE_LOOKUP.clear()

    def registry_records(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.registry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    def response_state(self, response_id: str) -> dict[str, Any]:
        return load_latest_response_state(response_id, frames_dir=self.response_frames_dir)

    def snapshot_counts(self) -> dict[str, int | str]:
        frame_ledger = self.response_frames_dir / "responses.jsonl"
        return {
            "artifact_files": len([path for path in self.artifacts_dir.rglob("*") if path.is_file()]),
            "registry_records": len(self.registry_records()),
            "frame_records": len(frame_ledger.read_text(encoding="utf-8").splitlines()) if frame_ledger.exists() else 0,
            "backend_calls": sum(self.calls[key] for key in ("chat", "image_generation", "text_to_speech", "speech_to_text", "vision_analysis")),
            "embedding_calls": self.calls["embedding"],
            "runtime_status_bytes": self.runtime_status_path.read_text(encoding="utf-8"),
        }

    def freeze_manual_response(
        self,
        payload: dict[str, Any],
        *,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response_payload = dict(payload)
        if "object" not in response_payload:
            response_payload["object"] = "response"
        finalized = ollmo_webserver._finalize_response_frame_payload(
            response_payload,
            request_payload=request_payload or {"prompt": "fake frozen response"},
            persist=True,
        )
        return finalized

    def build_response_payload(
        self,
        *,
        response_id: str,
        capability: str,
        mode: str,
        output_text: str,
        source_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self.instances[capability]
        return build_canonical_response_payload(
            instance_id=instance["instance_id"],
            model_name=instance["model"],
            backend=instance["backend"],
            capability=capability,
            mode=mode,
            output_text=output_text,
            source_payload=source_payload or {"content": output_text},
            route_payload=self._route_info(capability, instance),
            response_id=response_id,
        )

    def file_bytes(self, path: str | Path) -> bytes:
        return Path(path).read_bytes()

    def _build_instances(self) -> dict[str, dict[str, Any]]:
        instances: dict[str, dict[str, Any]] = {}
        for capability in (
            "chat",
            "image_generation",
            "vision_analysis",
            "text_to_speech",
            "speech_to_text",
            "embedding",
        ):
            instances[capability] = {
                "instance_id": f"fake-{capability}",
                "model": f"fake-{capability}-model",
                "backend": "fake",
                "capability": capability,
                "port": "0",
                "readiness": "ready",
                "activity": "idle",
                "runtime_status": {
                    "readiness": "ready",
                    "activity": "idle",
                    "process_alive": False,
                    "port_listening": False,
                    "source": "fake_backend_harness",
                },
            }
        return instances

    def _load_running_instances(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.instances.values()]

    def _merge_instances_with_runtime_status(self, instances: Any = None, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if instances:
            return [dict(item) for item in instances]
        return self._load_running_instances()

    def _capability_from_payload(self, data: Any) -> str:
        payload = data if isinstance(data, dict) else {}
        capability = str(
            payload.get("capability")
            or payload.get("capability_hint")
            or payload.get("capabilityHint")
            or "chat"
        ).strip().lower().replace("-", "_")
        if capability in {"embeddings", "embedding_helper"}:
            capability = "embedding"
        return capability

    def _resolve_responses_target_instance(
        self,
        data: Any,
        *,
        forced_instance_id: str | None = None,
        excluded_instance_ids: list[str] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None, str | None, str | None]:
        excluded = set(excluded_instance_ids or [])
        if forced_instance_id:
            for capability, instance in self.instances.items():
                if instance["instance_id"] == forced_instance_id and forced_instance_id not in excluded:
                    return forced_instance_id, dict(instance), capability, None
            return None, None, None, f"Fake instance '{forced_instance_id}' was not found."
        capability = self._capability_from_payload(data)
        instance = self.instances.get(capability)
        if not instance or instance["instance_id"] in excluded:
            return None, None, None, f"No running fake instance for capability '{capability}'."
        return instance["instance_id"], dict(instance), capability, None

    def _route_info(self, capability: str, instance: dict[str, Any], *, preview: bool = False) -> dict[str, Any]:
        return {
            "instance_id": instance["instance_id"],
            "instance": dict(instance),
            "model": instance["model"],
            "backend": instance["backend"],
            "capability": capability,
            "route_source": "fake_backend_harness_preview" if preview else "fake_backend_harness",
            "route_reason": f"deterministic fake {capability} route",
            "route_confidence": 1.0,
            "route_runtime": {
                "truth_source": "fake_backend_harness",
                "request_phase_graph": {
                    "kind": "ollmo.request_phase_graph",
                    "current_phase_id": "phase-1",
                    "phases": [
                        {
                            "phase_id": "phase-1",
                            "capability": capability,
                            "output_type": self._output_type_for_capability(capability),
                            "status": "planned",
                            "obligation_id": "obligation-phase-1",
                        }
                    ],
                    "output_obligations": [
                        {
                            "obligation_id": "obligation-phase-1",
                            "phase_id": "phase-1",
                            "capability": capability,
                            "output_type": self._output_type_for_capability(capability),
                            "required": True,
                            "status": "promoted",
                        }
                    ],
                },
            },
        }

    def _resolve_ghost_auto_route(self, data: Any, *args: Any, **kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
        preview = bool(kwargs.get("preview_mode"))
        compute_semantics = bool(kwargs.get("compute_semantics"))
        capability = self._capability_from_payload(data)
        if capability == "embedding":
            capability = "chat"
        instance = self.instances.get(capability) or self.instances["chat"]
        route_info = self._route_info(capability, instance, preview=preview)
        if compute_semantics:
            self._execute_embedding_backend_request(
                target_port=0,
                model_name=self.instances["embedding"]["model"],
                backend="fake",
                inputs=[str((data or {}).get("prompt") or "")],
                instance=self.instances["embedding"],
            )
            route_runtime = dict(route_info["route_runtime"])
            route_runtime["semantic_compute"] = {
                "requested": True,
                "performed": True,
                "preview": preview,
                "source": "fake_backend_harness",
                "learnable": False,
            }
            route_info["route_runtime"] = route_runtime
        self.calls["route_preview" if preview else "route"] += 1
        return route_info, None

    def _prepare_effective_request_data(
        self,
        data: Any,
        *,
        route_info: dict[str, Any] | None = None,
        instance: dict[str, Any] | None = None,
        compute_semantics: bool = False,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
        effective = dict(data or {}) if isinstance(data, dict) else {}
        updated_route = dict(route_info or {}) if isinstance(route_info, dict) else route_info
        planner_meta = {"attempted": False, "applied": False, "reason": "fake_backend_harness_noop"}
        if compute_semantics and isinstance(updated_route, dict):
            route_runtime = dict(updated_route.get("route_runtime") or {})
            semantic_compute = dict(route_runtime.get("semantic_compute") or {})
            semantic_compute.setdefault("requested", True)
            semantic_compute.setdefault("performed", True)
            semantic_compute.setdefault("preview", True)
            semantic_compute.setdefault("learnable", False)
            route_runtime["semantic_compute"] = semantic_compute
            updated_route["route_runtime"] = route_runtime
            planner_meta = {"attempted": True, "applied": False, "reason": "fake_semantic_preview"}
        return effective, updated_route, planner_meta, {}

    def _execute_chat_backend_request(self, **kwargs: Any) -> str:
        self.calls["chat"] += 1
        self.call_records.append({"capability": "chat", "target_port": kwargs.get("target_port"), "kwargs": kwargs})
        messages = kwargs.get("messages") or []
        prompt = " ".join(str(item.get("content") or "") for item in messages if isinstance(item, dict)).lower()
        if "mixed web" in prompt:
            return "Saved web files are available in runtime artifacts; prose is not the artifact."
        return "Saved report.md is available in runtime artifacts; this prose is not artifact truth."

    def _persist_generated_text_artifact_if_requested(
        self,
        assistant_text: str,
        *,
        prompt: str,
        model_name: str,
        mode: str,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = request_payload if isinstance(request_payload, dict) else {}
        scenario = str(payload.get("fake_scenario") or "").strip()
        if scenario == "mixed_web":
            linked_image_path = Path(str(payload.get("linked_image_path") or ""))
            if not linked_image_path.exists():
                linked_image_path = write_bytes(self.images_dir / "mixed-web.png", tiny_png_bytes())
            image_rel = os.path.relpath(linked_image_path, start=self.web_dir)
            html_path = self.web_dir / "index.html"
            css_path = self.web_dir / "styles.css"
            html = (
                "<!doctype html>\n"
                "<html><head><link rel=\"stylesheet\" href=\"styles.css\"></head>\n"
                "<body><main><h1>Fake Backend Observatory</h1>"
                f"<img src=\"{image_rel}\" alt=\"Generated observatory image\"></main></body></html>\n"
            )
            css = f"main {{ background-image: url(\"{image_rel}\"); color: #111; }}\n"
            write_text(html_path, html)
            write_text(css_path, css)
            return {
                "saved_text_path": str(html_path),
                "saved_text_artifacts": [
                    {
                        "path": str(html_path),
                        "text_artifact_request": {"source_name": "index", "extension": "html"},
                        "mime_type": "text/html",
                    },
                    {
                        "path": str(css_path),
                        "text_artifact_request": {"source_name": "styles", "extension": "css"},
                        "mime_type": "text/css",
                    },
                ],
                "derived_from": payload.get("derived_from") or [],
            }
        if scenario == "text_artifact" or "report" in str(prompt or "").lower():
            report_path = self.documents_dir / "report.md"
            write_text(report_path, TEXT_ARTIFACT_CONTENT)
            return {
                "saved_text_path": str(report_path),
                "saved_text_artifacts": [
                    {
                        "path": str(report_path),
                        "text_artifact_request": {"source_name": "report", "extension": "md"},
                        "mime_type": "text/markdown",
                    }
                ],
            }
        return {}

    def _invoke_internal_api_json_route(
        self,
        path: str | None = None,
        *,
        payload: dict[str, Any] | None = None,
        upload: Any = None,
    ) -> tuple[dict[str, Any], int]:
        request_payload = dict(payload or {})
        capability = self._capability_from_payload(request_payload)
        prompt = str(request_payload.get("prompt") or "")
        if "timeout" in prompt.lower() or "fail" in prompt.lower():
            self.calls[f"{capability}_failure"] += 1
            self.call_records.append({"capability": capability, "status": "timeout", "payload": request_payload})
            return {"error": "Fake backend timeout for deterministic failure truth."}, 504
        if capability == "image_generation":
            self.calls["image_generation"] += 1
            image_path = write_bytes(self.images_dir / "generated.png", tiny_png_bytes())
            result = {
                "mode": "image_generation",
                "content": "Generated deterministic PNG artifact.",
                "saved_image_path": str(image_path),
                "artifact_ref": "artifact:fake-generated-png",
                "artifact_id": "fake-generated-png",
                "provenance_id": "fake-image-provenance",
                "seed": 1234,
                "image_state": {
                    "summary": "deterministic fake image",
                    "labels": ["fixture", "png"],
                },
            }
        elif capability == "text_to_speech":
            self.calls["text_to_speech"] += 1
            audio_bytes = (
                silent_wav_bytes()
                if (
                    request_payload.get("fake_scenario") == "silent_tts"
                    or "fake silent tts" in prompt.lower()
                )
                else tiny_wav_bytes()
            )
            audio_path = write_bytes(self.audio_dir / "speech.wav", audio_bytes)
            result = {
                "mode": "text_to_speech",
                "content": "Generated deterministic WAV artifact.",
                "saved_audio_path": str(audio_path),
                "audio_mimetype": "audio/wav",
                "voice": request_payload.get("voice") or "fake-voice",
            }
        elif capability == "speech_to_text":
            self.calls["speech_to_text"] += 1
            file_path = str(request_payload.get("file_path") or "").strip()
            result = {
                "mode": "speech_to_text",
                "content": TRANSCRIPT_TEXT,
                "input_artifacts": [
                    {
                        "type": "audio",
                        "path": file_path,
                        "origin": "user_input",
                        "mime_type": "audio/wav",
                    }
                ] if file_path else [],
                "result": {"transcript": TRANSCRIPT_TEXT, "language": "en"},
            }
        elif capability == "vision_analysis":
            self.calls["vision_analysis"] += 1
            file_path = str(request_payload.get("file_path") or "").strip()
            result = {
                "mode": "vision_analysis",
                "content": VISION_RESULT["description"],
                "input_artifacts": [
                    {
                        "type": "image",
                        "path": file_path,
                        "origin": "user_input",
                        "mime_type": "image/png",
                    }
                ] if file_path else [],
                "result": dict(VISION_RESULT),
            }
        else:
            self.calls[capability] += 1
            result = {"mode": capability, "content": f"Fake {capability} response."}
        self.call_records.append({"capability": capability, "payload": request_payload, "result": result})
        return result, 200

    def _execute_embedding_backend_request(self, **kwargs: Any) -> list[list[float]]:
        self.calls["embedding"] += 1
        inputs = kwargs.get("inputs") or []
        self.call_records.append({"capability": "embedding", "kwargs": kwargs})
        return [deterministic_embedding(str(item)) for item in inputs]

    def _output_type_for_capability(self, capability: str) -> str:
        return {
            "image_generation": "image",
            "text_to_speech": "audio",
            "speech_to_text": "text",
            "vision_analysis": "text",
            "chat": "text",
            "embedding": "embedding",
        }.get(capability, "text")

    def _schedule_noop(self, *args: Any, **kwargs: Any) -> None:
        self.calls["scheduled_noop"] += 1

    def _log_noop(self, **kwargs: Any) -> None:
        self.calls["log"] += 1
