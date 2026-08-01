"""Generated-image infer post-processing owners for Ollmo."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class GeneratedImagePostprocessOwner:
    runtime_status_path_getter: Callable[[], Any]
    artifact_registry_ledger_getter: Callable[[], Any]
    load_running_instances: Callable[[], list[dict[str, Any]]]
    merge_instances_with_runtime_status: Callable[..., list[dict[str, Any]]]
    build_instance_trait_summary: Callable[[dict[str, Any]], dict[str, Any]]
    normalize_capability: Callable[[Any], Optional[str]]
    normalize_backend: Callable[[Any], Optional[str]]
    capability_vision_analysis: str
    capability_chat: str
    parse_image_state_response: Callable[..., Optional[dict[str, Any]]]
    invoke_internal_api_json_route: Callable[..., tuple[dict[str, Any], int]]
    get_cached_generated_image_state: Callable[[Any], Optional[dict[str, Any]]]
    store_cached_generated_image_state: Callable[[Any, Any], Optional[dict[str, Any]]]
    build_image_state_enrichment_state: Callable[..., dict[str, Any]]
    attach_cached_generated_image_state_to_response_lookups: Callable[[Any, Any], None]
    claim_generated_image_state_enrichment: Callable[[Any], Optional[str]]
    release_generated_image_state_enrichment: Callable[[Any], None]
    extract_semantic_materializer_prompt: Callable[..., Optional[str]]
    compact_request_meta: Callable[[Any], dict[str, Any]]
    extract_request_meta: Callable[[Any], Any]
    build_generated_image_provenance: Callable[..., dict[str, Any]]
    persist_generated_image_provenance: Callable[..., Any]
    persist_artifact_registry_enrichment: Callable[..., Optional[dict[str, Any]]]
    coerce_seed: Callable[[Any], Optional[int]]
    schedule_post_response_substrate_hygiene: Optional[Callable[..., Any]] = None
    helper_cooldown_sec: int = 900
    helper_error_cooldowns: dict[str, float] = field(default_factory=dict)

    def _helper_in_cooldown(self, item: dict[str, Any]) -> bool:
        instance_id = str(item.get('instance_id') or '').strip()
        if not instance_id:
            return False
        cooldown_until = float(self.helper_error_cooldowns.get(instance_id) or 0)
        if cooldown_until > time.monotonic():
            return True
        if cooldown_until:
            self.helper_error_cooldowns.pop(instance_id, None)
        return False

    def _mark_helper_cooldown(self, instance_id: str) -> None:
        token = str(instance_id or '').strip()
        if not token:
            return
        self.helper_error_cooldowns[token] = time.monotonic() + max(0, int(self.helper_cooldown_sec))

    def _list_image_state_helper_candidates(self, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for item in instances:
            if not isinstance(item, dict):
                continue
            capability = self.normalize_capability(item.get('capability'))
            if capability not in {self.capability_vision_analysis, self.capability_chat}:
                continue
            traits = self.build_instance_trait_summary(item)
            if not traits.get('supports_vision'):
                continue
            instance_id = str(item.get('instance_id') or '').strip()
            port = item.get('port')
            if not instance_id or not port:
                continue
            runtime_status = item.get('runtime_status') if isinstance(item.get('runtime_status'), dict) else {}
            readiness = str(item.get('readiness') or runtime_status.get('readiness') or '').strip().lower()
            activity = str(item.get('activity') or runtime_status.get('activity') or '').strip().lower()
            port_listening = item.get('port_listening', runtime_status.get('port_listening'))
            process_alive = item.get('process_alive', runtime_status.get('process_alive'))
            if port_listening is False or process_alive is False:
                continue
            if self._helper_in_cooldown(item):
                continue
            readiness_rank = 2 if readiness == 'ready' else 1 if readiness in {'started', 'idle'} else 0
            if readiness_rank <= 0:
                continue
            activity_rank = 1 if activity in {'idle', 'ready'} else 0
            model_name = str(item.get('model') or '').strip().lower()
            backend_package = str(item.get('backend_package') or '').strip().lower()
            routing_summary = item.get('routing_summary') if isinstance(item.get('routing_summary'), dict) else {}
            session_controls_summary = item.get('session_controls_summary')
            if not isinstance(session_controls_summary, dict):
                session_controls_summary = (
                    routing_summary.get('session_controls')
                    if isinstance(routing_summary.get('session_controls'), dict)
                    else {}
                )
            required_fields = {
                str(field or '').strip().lower()
                for field in (session_controls_summary.get('required_fields') or [])
                if str(field or '').strip()
            }
            ocr_likely = bool(
                re.search(r'(^|[^a-z])ocr([^a-z]|$)', model_name)
                or re.search(r'(^|[^a-z])ocr([^a-z]|$)', backend_package)
                or 'ocr_mode' in required_fields
            )
            backend = self.normalize_backend(item.get('backend'))
            multimodal_chat_rank = (
                2 if capability == self.capability_chat and backend != 'mlx'
                else 1 if capability == self.capability_chat
                else 0
            )
            detail_rank = 1 if traits.get('supports_structured_outputs') else 0
            helper_family_rank = (
                2 if not ocr_likely
                else 1 if backend == 'mlx'
                else 0
            )
            candidates.append(
                {
                    **item,
                    '_score': (
                        readiness_rank,
                        activity_rank,
                        multimodal_chat_rank,
                        helper_family_rank,
                        detail_rank,
                        instance_id,
                    ),
                }
            )
        return sorted(candidates, key=lambda item: item.get('_score'), reverse=True) if candidates else []

    def pick_image_state_helper_instance(self, instances: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        candidates = self._list_image_state_helper_candidates(instances)
        if not candidates:
            return None
        selected = dict(candidates[0])
        selected.pop('_score', None)
        return selected

    def _schedule_image_state_helper_substrate_hygiene(
        self,
        helper_instance: dict[str, Any],
        *,
        image_path: str,
    ) -> None:
        schedule_hygiene = self.schedule_post_response_substrate_hygiene
        if not callable(schedule_hygiene) or not isinstance(helper_instance, dict):
            return
        helper_instance_id = str(helper_instance.get('instance_id') or '').strip()
        if not helper_instance_id:
            return
        helper_capability = (
            self.normalize_capability(helper_instance.get('capability'))
            or self.capability_vision_analysis
        )
        helper_model = str(helper_instance.get('model') or '').strip()
        response_payload = {
            'id': f'generated_image_state_enrichment:{helper_instance_id}',
            'instance_id': helper_instance_id,
            'model': helper_model,
            'capability': helper_capability,
            'mode': 'generated_image_state_enrichment',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'image_path': str(image_path or '').strip(),
        }
        route_payload = {
            'instance_id': helper_instance_id,
            'model': helper_model,
            'capability': helper_capability,
            'route_source': 'generated_image_state_enrichment',
        }
        try:
            schedule_hygiene(
                response_payload,
                route_payload=route_payload,
                reason='generated_image_state_enrichment_helper_terminal',
            )
        except Exception:  # noqa: BLE001
            logging.exception('Could not schedule generated image state helper substrate hygiene.')

    def build_image_state_for_generated_image(self, image_path: str) -> Optional[dict[str, Any]]:
        path = str(image_path or '').strip()
        if not path:
            return None
        instances = self.merge_instances_with_runtime_status(
            self.load_running_instances(),
            path=self.runtime_status_path_getter(),
            refresh=True,
        )
        helper_candidates = self._list_image_state_helper_candidates(instances)
        if not helper_candidates:
            return None
        helper_prompt = (
            'Describe the actual visible generated image, not the original prompt. '
            'Return strict JSON with keys '
            '{"summary": string, "subject": string|null, "scene": string|null, "style": string|null, "key_elements": string[]}. '
            'Keep the summary short and concrete. '
            'Do not OCR or transcribe incidental text unless it is clearly readable and visually central to the image.'
        )
        for helper_instance in helper_candidates[:3]:
            helper_instance_id = str(helper_instance.get('instance_id') or '').strip()
            helper_model = str(helper_instance.get('model') or '').strip()
            helper_capability = self.normalize_capability(helper_instance.get('capability')) or self.capability_vision_analysis
            vision_payload = {
                'instance_id': helper_instance_id,
                'capability': helper_capability,
                'file_path': path,
                'reference_artifacts': [
                    {
                        'type': 'image',
                        'path': path,
                        'artifact_ref': path,
                        'origin': 'generated_output',
                    }
                ],
                'prompt': helper_prompt,
                'infer_timeout_sec': 8,
                'internal_fast_timeout': True,
            }
            try:
                result, status_code = self.invoke_internal_api_json_route(payload=vision_payload)
            except Exception as exc:  # noqa: BLE001
                self._mark_helper_cooldown(helper_instance_id)
                logging.info(
                    'Generated image state helper %s failed for %s: %s',
                    helper_instance_id,
                    path,
                    exc,
                )
                continue
            finally:
                self._schedule_image_state_helper_substrate_hygiene(
                    helper_instance,
                    image_path=path,
                )
            if status_code >= 400:
                self._mark_helper_cooldown(helper_instance_id)
                continue
            content = str(result.get('content') or '').strip()
            if not content:
                continue
            parsed = self.parse_image_state_response(
                content,
                describer_instance_id=helper_instance_id,
                describer_model=helper_model,
            )
            if parsed:
                return parsed
        return None

    def _background_enrich_generated_image_payload(self, image_path: str) -> None:
        try:
            payload = self.enrich_generated_image_payload(
                {
                    'mode': 'image_generation',
                    'saved_image_path': image_path,
                },
                blocking=True,
            )
            image_state = payload.get('image_state')
            if isinstance(image_state, dict) and image_state:
                self.attach_cached_generated_image_state_to_response_lookups(image_path, image_state)
        finally:
            self.release_generated_image_state_enrichment(image_path)

    def _persist_generated_image_state_enrichment(
        self,
        image_path: str,
        image_state: Any,
    ) -> Optional[dict[str, Any]]:
        path = str(image_path or '').strip()
        if not path or not isinstance(image_state, dict) or not image_state:
            return None
        try:
            return self.persist_artifact_registry_enrichment(
                artifact_path=path,
                artifact_type='image',
                enrichments={
                    'image_state': dict(image_state),
                    'image_state_enrichment': self.build_image_state_enrichment_state(status='completed'),
                },
                ledger_path=self.artifact_registry_ledger_getter(),
            )
        except OSError as exc:
            logging.warning('Could not persist generated image enrichment for %s: %s', path, exc)
        except ValueError as exc:
            logging.warning('Could not build generated image enrichment for %s: %s', path, exc)
        return None

    @staticmethod
    def _pop_image_state_enrichment_suppression_reason(payload: dict[str, Any]) -> str:
        suppress_generated = bool(payload.get('suppress_generated_image_enrichment'))
        reason = str(payload.pop('image_state_enrichment_suppression_reason', '') or '').strip()
        payload.pop('suppress_image_state_enrichment', None)
        payload.pop('suppress_generated_image_enrichment', None)
        if reason:
            return reason
        if suppress_generated:
            return 'required_artifact_closure_priority'
        return 'suppressed_by_request'

    def schedule_generated_image_payload_enrichment(self, payload: dict[str, Any]) -> dict[str, Any]:
        updated = dict(payload or {})
        mode = str(updated.get('mode') or '').strip().lower()
        image_path = str(updated.get('saved_image_path') or '').strip()
        if mode not in {'image_generation', 'image_generation_edit'} or not image_path:
            return updated
        if bool(updated.get('suppress_image_state_enrichment') or updated.get('suppress_generated_image_enrichment')):
            reason = self._pop_image_state_enrichment_suppression_reason(updated)
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(
                status='skipped',
                reason=reason,
            )
            return updated
        if isinstance(updated.get('image_state'), dict) and updated.get('image_state'):
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(status='completed')
            return updated
        enrichment_key = self.claim_generated_image_state_enrichment(image_path)
        if not enrichment_key:
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(
                status='pending_existing',
                reason='same_image_path_enrichment_in_flight',
            )
            return updated
        updated['image_state_enrichment'] = self.build_image_state_enrichment_state(status='pending')
        threading.Thread(
            target=self._background_enrich_generated_image_payload,
            args=(enrichment_key,),
            daemon=True,
        ).start()
        return updated

    def enrich_generated_image_payload(self, payload: dict[str, Any], *, blocking: bool = False) -> dict[str, Any]:
        updated = dict(payload or {})
        mode = str(updated.get('mode') or '').strip().lower()
        image_path = str(updated.get('saved_image_path') or '').strip()
        if mode not in {'image_generation', 'image_generation_edit'} or not image_path:
            return updated
        if bool(updated.get('suppress_image_state_enrichment') or updated.get('suppress_generated_image_enrichment')):
            reason = self._pop_image_state_enrichment_suppression_reason(updated)
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(
                status='skipped',
                reason=reason,
            )
            return updated
        cached_image_state = self.get_cached_generated_image_state(image_path)
        if cached_image_state:
            updated['image_state'] = cached_image_state
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(status='completed')
            self._persist_generated_image_state_enrichment(image_path, cached_image_state)
            return updated
        if isinstance(updated.get('image_state'), dict) and updated.get('image_state'):
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(status='completed')
            self._persist_generated_image_state_enrichment(image_path, updated.get('image_state'))
            return updated
        if not blocking:
            return self.schedule_generated_image_payload_enrichment(updated)
        try:
            image_state = self.build_image_state_for_generated_image(image_path)
        except Exception as exc:  # noqa: BLE001
            logging.info('Generated image state enrichment skipped for %s: %s', image_path, exc)
            image_state = None
        if image_state:
            cached_image_state = self.store_cached_generated_image_state(image_path, image_state)
            updated['image_state'] = cached_image_state or image_state
            updated['image_state_enrichment'] = self.build_image_state_enrichment_state(status='completed')
            self._persist_generated_image_state_enrichment(image_path, updated.get('image_state'))
        return updated

    def persist_generated_image_provenance_for_infer_result(
        self,
        payload: Any,
        *,
        request_payload: Optional[dict[str, Any]],
        instance_id: str,
        model_name: str,
        backend: str,
        capability: str,
        user_prompt: str,
        prompt: str,
        raw_file_path: str,
        input_artifacts: list[dict[str, Any]],
        reference_artifacts: Any,
        image_width: Optional[int],
        image_height: Optional[int],
        image_seed: Optional[int],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        mode = str(payload.get('mode') or '').strip().lower()
        image_path = str(payload.get('saved_image_path') or '').strip()
        if mode not in {'image_generation', 'image_generation_edit'} or not image_path:
            return None

        request_data = dict(request_payload or {})
        semantic_prompt = self.extract_semantic_materializer_prompt(
            request_data,
            capability=capability,
        )
        prompt_text = str(
            semantic_prompt
            or prompt
            or user_prompt
            or request_data.get('prompt')
            or request_data.get('input')
            or ''
        ).strip()
        request_meta = self.compact_request_meta(self.extract_request_meta(request_data))
        seed = self.coerce_seed(payload.get('seed'))
        if seed is None:
            seed = image_seed

        record = self.build_generated_image_provenance(
            image_path=image_path,
            prompt_text=prompt_text,
            prompt_preview=prompt_text,
            instance_id=instance_id,
            model=model_name,
            backend=backend,
            capability=capability,
            mode=mode,
            request_origin=str(
                request_data.get('provenance_origin')
                or request_data.get('request_origin')
                or ''
            ).strip() or 'api_infer',
            response_id=str(request_data.get('response_id') or request_data.get('responseId') or '').strip() or None,
            conversation_id=str(request_data.get('conversation_id') or request_data.get('conversationId') or '').strip() or None,
            request_id=str(request_data.get('request_id') or request_data.get('requestId') or '').strip() or None,
            width=image_width,
            height=image_height,
            seed=seed,
            file_path=raw_file_path or request_data.get('file_path'),
            input_artifacts=input_artifacts,
            reference_artifacts=reference_artifacts,
            route_source=str(request_data.get('provenance_route_source') or '').strip() or None,
            route_reason=str(request_data.get('provenance_route_reason') or '').strip() or None,
            request_meta=request_meta if isinstance(request_meta, dict) else None,
        )
        try:
            self.persist_generated_image_provenance(
                record,
                ledger_path=self.artifact_registry_ledger_getter(),
            )
            return record
        except OSError as exc:
            logging.warning('Could not persist generated image provenance for %s: %s', image_path, exc)
        except ValueError as exc:
            logging.warning('Could not build generated image provenance for %s: %s', image_path, exc)
        return None
